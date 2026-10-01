#!/usr/bin/env python3
"""djev front for vLLM — same HTTP API as server.py, the model served by `vllm serve`.

Needs vLLM >= 0.30.1 nightly (structured reads for DiffusionGemma, PRs #57250 / #58216):
per-request `diffusion_seed_canvas` (our scaffold with <mask> answer slots), `diffusion_max_steps: 1`,
`diffusion_read_only` (emit after one denoise pass, return temperature-1 logprobs at every canvas
position), `logprob_token_ids` (exact logprobs of the candidate tokens), and optionally
`diffusion_constrained` (unembed over the candidates only).

    vllm serve google/diffusiongemma-26B-A4B-it --diffusion-config '{"canvas_length": 64}' \\
        --max-logprobs 128 --async-scheduling --enable-prefix-caching --port 8000
    python server_vllm.py --upstream http://127.0.0.1:8000 --port 8765

Two engines:
  --engine inproc (default)  the vLLM engine runs inside this process (AsyncLLM); no second server.
  --engine http              talk to a separate `vllm serve` (as above); ~35 ms slower per request,
                             because the OpenAI layer serializes 64 x N logprobs to JSON.
Every state becomes one vLLM request (vLLM batches them).
"""
import argparse
import asyncio
import json
import math
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import server as S

TIMING = os.environ.get("DJEV_TIMING") == "1"   # [fast] per-read stage timings in each response's execution


class Reader:
    def __init__(self, upstream, model, served_canvas, width, thought, constrained, timeout, max_conn):
        self.upstream = upstream.rstrip("/")
        self.model = model
        self.served_canvas = served_canvas
        self.width = width                  # minimum per-request canvas width (see pick_width)
        self.thought = thought              # "canvas" (chat API, block at canvas head) | "prompt" (token prompt)
        self.constrained = constrained
        self.client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=max_conn,
                                                                             max_keepalive_connections=max_conn))

    def build(self, state, enc):
        images = state.get("images") or []
        text, slots = enc.prompt_text(state, len(images))
        body = enc.scaffold(slots, state["id"])
        use_chat = bool(images) or self.thought == "canvas"
        head = list(enc.thought_prefix_ids) if use_chat else []
        canvas = head + body
        width = self.pick_width(len(canvas))
        if width is None:
            raise ValueError(f"{state['id']}: answer template needs {len(canvas)} canvas tokens; "
                             f"it must fill at most half of the served canvas ({self.served_canvas})")
        canvas += [S.PAD_ID] * (width - len(canvas))
        ids = sorted({i for s in slots for g in s.cand_groups for i in g})
        if len(ids) > 128:
            raise ValueError(f"{state['id']}: {len(ids)} candidate tokens; vLLM allows 128 per request")
        xargs = {"diffusion_seed_canvas": canvas, "diffusion_max_steps": 1, "diffusion_read_only": True}
        if width != self.served_canvas:
            xargs["diffusion_canvas_length"] = width
        if self.constrained:
            xargs["diffusion_constrained"] = True
        common = {"model": self.model, "max_tokens": width, "logprob_token_ids": ids,
                  "return_tokens_as_token_ids": True, "vllm_xargs": xargs}
        if use_chat:
            content = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": text}]
            req = {**common, "messages": [{"role": "user", "content": content}], "logprobs": True,
                   "top_logprobs": len(ids), "chat_template_kwargs": {"enable_thinking": False}}
            url = self.upstream + "/v1/chat/completions"
        else:
            prompt_ids = enc.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                                     add_generation_prompt=True, return_dict=False)
            if isinstance(prompt_ids, dict):
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = list(prompt_ids) + list(enc.thought_prefix_ids)
            req = {**common, "prompt": prompt_ids, "logprobs": len(ids)}
            url = self.upstream + "/v1/completions"
        return url, req, slots, len(head), len(images)

    def pick_width(self, used):
        """Smallest canvas (>= self.width, doubling) whose scaffold fills at most half of it.
        A fuller canvas lets the model copy <mask> and shift the scaffold (probes: width 32 with a
        25-token scaffold read garbage; 64 read 18/18 correct)."""
        w = self.width
        while w <= self.served_canvas:
            if 2 * used <= w:
                return w
            w *= 2
        return self.served_canvas if used <= self.served_canvas // 2 else None

    async def read(self, url, req):
        r = await self.client.post(url, json=req)
        if r.status_code != 200:
            raise RuntimeError(f"vLLM {r.status_code}: {r.text[:500]}")
        return r.json()

    @staticmethod
    def rows(resp):
        """Per canvas position: {token_id: logprob}."""
        ch = resp["choices"][0]
        lp = ch["logprobs"]
        if "content" in lp:                                   # chat API
            return [{int(t["token"].split(":")[1]): t["logprob"] for t in pos["top_logprobs"]} for pos in lp["content"]]
        return [{int(k.split(":")[1]): v for k, v in pos.items()} for pos in lp["top_logprobs"]]  # completions API

    def answers(self, slots, rows, off):
        out = []
        for slot in slots:
            p = off + slot.pos
            d = rows[p] if p < len(rows) else {}
            per_key = [sum(math.exp(d[i]) for i in g if i in d) for g in slot.cand_groups]
            mass = sum(per_key)
            probs = [v / mass if mass > 0 else 1.0 / len(per_key) for v in per_key]
            out.append((probs, mass))
        return out


class InprocReader:
    """Same readout as Reader, but the vLLM engine (AsyncLLM) lives in this process: no HTTP hop and no
    JSON serialization of 64 x N logprobs (that alone cost ~35 ms per request through `vllm serve`)."""


    def __init__(self, engine_args, served_canvas, width, constrained, gpu=None):
        from vllm.v1.engine.async_llm import AsyncLLM
        # The engine core is a child process: pin it to one GPU through the env it inherits at spawn.
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        try:
            self.engine = AsyncLLM.from_engine_args(engine_args)
        finally:
            if gpu is not None:
                if saved is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved
        self.inflight = 0
        self.served_canvas = served_canvas
        self.width = width
        self.thought = "prompt"
        self.constrained = constrained
        self.upstream = "in-process"
        self._ids = __import__("itertools").count()

    pick_width = Reader.pick_width
    answers = Reader.answers

    def build(self, state, enc, images=()):
        """images: already-loaded PIL images (loaded off the event loop by the caller)."""
        from vllm import SamplingParams
        text, slots = enc.prompt_text(state, len(images))
        body = enc.scaffold(slots, state["id"])
        width = self.pick_width(len(body))
        if width is None:
            raise ValueError(f"{state['id']}: answer template needs {len(body)} canvas tokens; "
                             f"it must fill at most half of the served canvas ({self.served_canvas})")
        canvas = body + [S.PAD_ID] * (width - len(body))
        ids = sorted({i for s in slots for g in s.cand_groups for i in g})
        if len(ids) > 128:
            raise ValueError(f"{state['id']}: {len(ids)} candidate tokens; vLLM allows 128 per request")
        if images:
            # The chat template emits one <|image|> per image part; vLLM's Gemma-4 processor expands each into
            # <|image> + N soft tokens + <image|>. Token ids (not text) so no second <bos> gets added.
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            prompt_text = enc.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                      add_generation_prompt=True)
            ids_ = enc.tok.encode(prompt_text, add_special_tokens=False) + list(enc.thought_prefix_ids)
            prompt = {"prompt_token_ids": ids_, "multi_modal_data": {"image": list(images)}}
        else:
            prompt_ids = enc.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                                     add_generation_prompt=True, return_dict=False)
            if isinstance(prompt_ids, dict):
                prompt_ids = prompt_ids["input_ids"]
            prompt = {"prompt_token_ids": list(prompt_ids) + list(enc.thought_prefix_ids)}
        xargs = {"diffusion_seed_canvas": canvas, "diffusion_max_steps": 1, "diffusion_read_only": True}
        if width != self.served_canvas:
            xargs["diffusion_canvas_length"] = width
        if self.constrained:
            xargs["diffusion_constrained"] = True
        params = SamplingParams(max_tokens=width, logprobs=len(ids), logprob_token_ids=ids, extra_args=xargs,
                                detokenize=False)
        return prompt, params, slots, 0, len(images)

    async def read(self, prompt, params):
        final = None
        async for out in self.engine.generate(prompt, params, request_id=f"djev-{next(self._ids)}"):
            final = out
        return final

    @staticmethod
    def rows(out):
        return [{tid: lp.logprob for tid, lp in d.items()} for d in (out.outputs[0].logprobs or [])]

    @staticmethod
    def prompt_tokens(out):
        return len(out.prompt_token_ids or [])

    async def healthy(self):
        try:
            await self.engine.check_health()
            return True
        except Exception:
            return False


class MultiReader:
    """Several independent single-GPU engines behind one front; each request goes to the engine with
    the fewest requests in flight. Unlike vLLM data parallelism (dp>1), the engines never step in
    lockstep, so a lone request is as fast as on a single-engine server (probes: dp=2 cost +4 ms text,
    +10 ms image per request)."""

    def __init__(self, readers):
        self.readers = readers
        r0 = readers[0]
        self.served_canvas, self.width, self.thought, self.constrained = r0.served_canvas, r0.width, r0.thought, r0.constrained
        self.upstream = f"in-process x{len(readers)}"

    def build(self, state, enc, images=()):
        return self.readers[0].build(state, enc, images)

    async def read(self, prompt, params):
        r = min(self.readers, key=lambda x: x.inflight)
        r.inflight += 1
        try:
            return await r.read(prompt, params)
        finally:
            r.inflight -= 1

    rows = staticmethod(InprocReader.rows)
    prompt_tokens = staticmethod(InprocReader.prompt_tokens)

    def answers(self, slots, rows, off):
        return self.readers[0].answers(slots, rows, off)

    async def healthy(self):
        return all([await r.healthy() for r in self.readers])


def _fast_worker(gpu, conn, opts):
    """One GPU: a fastpath.FastReader answering reads sent over `conn` (spawned process)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import traceback
    opts = dict(opts)
    batch_wait = opts.pop("batch_wait_ms", 0.0) / 1000
    try:
        import fastpath
        fr = fastpath.FastReader(**opts)
        conn.send(("ready", {"gpu": gpu, "load_seconds": round(fr.load_seconds, 1),
                             "buckets": [fr.buckets[0], fr.buckets[-1]]}))
    except Exception:
        conn.send(("fatal", traceback.format_exc()))
        return
    def serve_one(rid, reads, t0):
        try:
            res = []
            for r in reads:
                if r[0] == "image":
                    logps, n = fr.read_image(*r[1:])
                else:
                    logps, n = fr.read(*r[1:]), len(r[1])
                res.append((logps, n, fr.last.get("cached", 0)))
            conn.send(("ok", rid, res, (t0, time.time(), fr.last.get("ms")) if TIMING else None))
        except Exception as exc:
            conn.send(("err", rid, f"{type(exc).__name__}: {exc}"))

    busy = False
    while True:
        msg = conn.recv()
        if msg is None:
            return
        # the reads that queued up while the GPU was busy go into one forward (up to batch_max text reads). While
        # busy (the last forward had company), wait up to batch_wait for the reads about to come: bots answered
        # together send again together, a millisecond or two apart, and a forward that leaves without them makes
        # them take turns, each alone. A quiet worker takes each read alone, at once, as before.
        batch, stop = [msg], False
        deadline = time.perf_counter() + (batch_wait if busy else 0.0)
        while fr.batch_max > 1 and sum(len(m[1]) for m in batch) < fr.batch_max:
            if not conn.poll(max(0.0, deadline - time.perf_counter())):
                break
            m = conn.recv()
            if m is None:
                stop = True
                break
            batch.append(m)
        busy = len(batch) > 1
        t0 = time.time()
        text = [(rid, reads) for rid, reads in batch if all(r[0] == "text" for r in reads)]
        if len(text) > 1:
            outs = None
            try:
                got = fr.read_many([r[1:] for _, reads in text for r in reads])
                t1, ms, k, outs = time.time(), fr.last.get("ms"), 0, []
                for rid, reads in text:
                    outs.append((rid, [(logps, len(r[1]), cached) for r, (logps, cached) in zip(reads, got[k:])]))
                    k += len(reads)
            except Exception:
                traceback.print_exc()           # the batch failed: each read alone, so an error stays with its own
                outs = None
            if outs is not None:
                for rid, res in outs:
                    conn.send(("ok", rid, res, (t0, t1, ms) if TIMING else None))
                done = {rid for rid, _ in outs}
                batch = [m for m in batch if m[0] not in done]
        for rid, reads in batch:
            serve_one(rid, reads, t0)
        busy = busy or conn.poll()             # others came while this forward ran
        if stop:
            return


class FastPool:
    """Single-forward reads (fastpath.py) on one worker process per GPU; each call goes to the least busy
    worker. Workers start one after another (concurrent starts race on vLLM's shared caches)."""

    def __init__(self, gpus, opts):
        import multiprocessing as mp
        import threading
        ctx = mp.get_context("spawn")
        self.workers = []
        for g in gpus:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_fast_worker, args=(g, child, opts), daemon=True)
            proc.start()
            msg = parent.recv()
            if msg[0] != "ready":
                raise SystemExit(f"fast worker on GPU {g} failed:\n{msg[1]}")
            print(json.dumps({"fast_worker": g, **msg[1]}), flush=True)
            w = {"gpu": g, "conn": parent, "proc": proc, "inflight": 0, "pending": {}, "lock": threading.Lock()}
            threading.Thread(target=self._collect, args=(w,), daemon=True).start()
            self.workers.append(w)
        self.ids = __import__("itertools").count()
        self.served_canvas = opts.get("canvas", 64)
        self.width = self.served_canvas
        self.thought, self.constrained = "prompt", False
        self.upstream = f"fast x{len(self.workers)}"

    def _collect(self, w):
        while True:
            msg = w["conn"].recv()
            fut, loop, tm = w["pending"].pop(msg[1])
            if tm is not None and msg[0] == "ok" and len(msg) > 3 and msg[3]:
                tm["worker_start"], tm["worker_end"], tm["read_ms"] = msg[3]
                tm["back"] = time.time()
            if msg[0] == "ok":
                loop.call_soon_threadsafe(fut.set_result, msg[2])
            else:
                loop.call_soon_threadsafe(fut.set_exception, RuntimeError(msg[2]))

    async def run(self, reads, timing=None):
        """timing: a dict to fill with wall-clock stamps (sent, worker_start, worker_end, back, resumed) and the
        worker's per-read split (DJEV_TIMING=1 only)."""
        w = min(self.workers, key=lambda x: x["inflight"])
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        rid = next(self.ids)
        w["pending"][rid] = (fut, loop, timing)
        w["inflight"] += 1
        try:
            with w["lock"]:
                if timing is not None:
                    timing["sent"] = time.time()
                w["conn"].send((rid, reads))
            res = await fut
            if timing is not None:
                timing["resumed"] = time.time()
            return res
        finally:
            w["inflight"] -= 1

    async def healthy(self):
        return all(w["proc"].is_alive() for w in self.workers)


def answer_obj(slot, probs, mass):
    dist = dict(zip(slot.keys, probs))
    best = max(dist, key=dist.get)
    if slot.qtype == "choice":
        ans = {"type": "choice", "probabilities": dist, "choice": best, "value": best}
    elif slot.qtype == "boolean":
        ans = {"type": "boolean", "probabilities": dist, "p_true": dist["true"], "value": dist["true"] >= 0.5}
    else:
        score = sum(int(k) * p for k, p in dist.items())
        ans = {"type": "score", "probabilities": dist, "score": score, "level": int(best), "value": score}
    ans["candidate_mass"] = mass
    return ans


def split_questions(state, enc, max_scaffold):
    """Split a state into sub-states whose answer scaffold fits `max_scaffold` canvas tokens (questions are
    independent by contract, so reading them in separate canvases changes nothing). Returns [state] when
    it already fits."""
    items = list(state["questions"].items())
    chunks, cur = [], []
    for item in items:
        trial = cur + [item]
        slots = [S.Slot(q, "boolean", ["false", "true"], [[0], [0]]) for q, _ in trial]
        if len(enc.scaffold(slots)) > max_scaffold and cur:
            chunks.append(cur)
            cur = [item]
        else:
            cur = trial
    chunks.append(cur)
    if len(chunks) == 1:
        return [state]
    return [dict(state, questions=dict(c)) for c in chunks]


def check_key(request, api_key):
    """Accept `Authorization: Bearer <key>` or HTTP basic auth with the key as password."""
    if not api_key:
        return True
    import base64
    import hmac
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return hmac.compare_digest(auth[7:].strip(), api_key)
    if auth.lower().startswith("basic "):
        try:
            _, _, pw = base64.b64decode(auth[6:].strip()).decode().partition(":")
        except Exception:
            return False
        return hmac.compare_digest(pw, api_key)
    return False


def make_app(reader, encoder, limits, model_name, api_key=None):
    app = FastAPI(title="djev (vLLM)", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path.startswith("/api/") and not check_key(request, api_key):
            return JSONResponse(status_code=401, content={"error": "missing or wrong API key"},
                                headers={"WWW-Authenticate": 'Basic realm="djev"'})
        return await call_next(request)
    here = os.path.dirname(os.path.abspath(__file__))
    readme = next((p for p in (os.path.join(here, "docs", "API.md"), os.path.join(here, "README.md")) if os.path.exists(p)), "")
    stats = {"requests": 0, "states": 0, "questions": 0, "images": 0, "errors": 0}

    @app.get("/")
    async def index():
        return {"service": "djev (DiffusionGemma-as-Jev, vLLM backend)", "manual": "GET /manual",
                "health": "GET /api/health", "evaluate": "POST /api/evaluate"}

    @app.get("/manual", response_class=PlainTextResponse)
    async def manual():
        try:
            with open(readme, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return "README.md not found"

    @app.get("/api/health")
    async def health():
        if isinstance(reader, (InprocReader, MultiReader, FastPool)):
            up = await reader.healthy()
        else:
            try:
                r = await reader.client.get(reader.upstream + "/health")
                up = r.status_code == 200
            except Exception:
                up = False
        backend = "vllm-fast" if isinstance(reader, FastPool) else "vllm"
        return {"ready": up, "backend": backend, "upstream": reader.upstream, "model": model_name,
                "readout": {"min_width": reader.width, "served_canvas": reader.served_canvas, "thought": reader.thought,
                            "constrained": reader.constrained, "steps": 1, "slot": "<mask>", "fill": "<pad>"},
                "limits": limits, "stats": stats}

    tk = encoder.tok
    seg_re = __import__("re").compile(r"[^\n]*\n+|[^\n]+$")

    @__import__("functools").lru_cache(maxsize=65536)
    def seg_ids(seg):
        return tuple(tk.encode(seg, add_special_tokens=False))

    def encode_text(text):
        """tk.encode(text) one line (with its run of newlines) at a time, through a cache: agents resend mostly the
        same lines (2.6 -> 0.7 ms per 2k-token prompt). The tokenizer never merges across a newline run: token-identical
        to encoding the whole text on 32k real agent prompts (2026-09-26)."""
        out = []
        for m in seg_re.finditer(text):
            out.extend(seg_ids(m.group(0)))
        return out

    chat_head = [tk.convert_tokens_to_ids("<bos>"), tk.convert_tokens_to_ids("<|turn>")] + tk.encode("user\n", add_special_tokens=False)
    chat_tail = ([tk.convert_tokens_to_ids("<turn|>")] + tk.encode("\n", add_special_tokens=False)
                 + [tk.convert_tokens_to_ids("<|turn>")] + tk.encode("model\n", add_special_tokens=False)
                 + list(encoder.thought_prefix_ids))

    def fast_read_args(sub, image_bytes=None):
        """Worker payload for one read. Text prompts are tokenized here (same ids as the chat template, checked);
        image prompts are built by the worker's processor."""
        text, slots = encoder.prompt_text(sub, len(image_bytes or ()))
        body = encoder.scaffold(slots, sub["id"])
        width = reader.served_canvas
        if 2 * len(body) > width:
            raise ValueError(f"{sub['id']}: answer template needs {len(body)} canvas tokens")
        canvas = body + [S.PAD_ID] * (width - len(body))
        cands = [[i for g in sl.cand_groups for i in g] for sl in slots]
        if image_bytes:
            return ("image", text, image_bytes, canvas, [sl.pos for sl in slots], cands), slots
        pids = chat_head + encode_text(text) + chat_tail
        return ("text", pids, canvas, [sl.pos for sl in slots], cands), slots

    async def evaluate_fast(states, reads, t0):
        try:
            need = [st for st in states if st.get("images")]
            raw = {}
            if need:
                raw = await asyncio.to_thread(lambda: {st["id"]: [S.load_image_bytes(u) for u in st["images"]]
                                                       for st in need})
            prepared = [fast_read_args(sub, raw.get(sub["id"])) for _, sub in reads]
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        tms = [{} if TIMING else None for _ in prepared]
        t_prepared = time.perf_counter()
        try:
            results = await asyncio.gather(*[reader.run([args], tm) for (args, _), tm in zip(prepared, tms)])
        except Exception as exc:
            stats["errors"] += 1
            return JSONResponse(status_code=502, content={"error": "fast engine failed", "detail": str(exc)[:1500]})
        per_state = [{"answers": {}, "prompt_tokens": 0, "cached": 0, "reads": 0} for _ in states]
        n_q = 0
        for (si, sub), (args, slots), res in zip(reads, prepared, results):
            logps_all, ptok, cached = res[0]
            ps = per_state[si]
            ps["prompt_tokens"] = max(ps["prompt_tokens"], ptok)
            ps["cached"] = max(ps["cached"], cached)
            ps["reads"] += 1
            for slot, logps in zip(slots, logps_all):
                n_q += 1
                sizes = [len(g) for g in slot.cand_groups]
                probs_raw, k = [], 0
                for n in sizes:
                    probs_raw.append(sum(math.exp(x) for x in logps[k:k + n]))
                    k += n
                mass = sum(probs_raw)
                probs = [v / mass if mass > 0 else 1.0 / len(probs_raw) for v in probs_raw]
                ps["answers"][slot.qid] = answer_obj(slot, probs, mass)
        out_states = [{"id": st["id"], "state": st["state"], "questions": st["questions"],
                       "answers": {q: ps["answers"][q] for q in st["questions"]}, "prompt_tokens": ps["prompt_tokens"],
                       "cached_prompt_tokens": ps["cached"], "images": len(st.get("images") or ()), "reads": ps["reads"]}
                      for st, ps in zip(states, per_state)]
        stats["requests"] += 1
        stats["states"] += len(states)
        stats["questions"] += n_q
        elapsed = time.perf_counter() - t0
        execution = {"states": len(states), "questions": n_q, "forward_passes": 1,
                     "autoregressive_decode_steps": 0, "network_model_calls": 0, "backend": "vllm-fast",
                     "readout": {"width": reader.served_canvas, "thought": "prompt", "single_pass": True},
                     "server_evaluation_seconds": elapsed}
        if TIMING:   # ms per read: parse+tokenize, pipe to the worker (incl. queueing), worker, pipe back, loop wake-up
            now = time.time()
            ms = lambda a, b: round((b - a) * 1000, 2)
            execution["timing_ms"] = [
                {"prepare": round((t_prepared - t0) * 1000, 2), "to_worker": ms(tm["sent"], tm["worker_start"]),
                 "worker": ms(tm["worker_start"], tm["worker_end"]), "back": ms(tm["worker_end"], tm["back"]),
                 "resume": ms(tm["back"], tm["resumed"]), "finish": ms(tm["resumed"], now), "read": tm.get("read_ms")}
                for tm in tms if "worker_start" in tm]
        return {"model": model_name, "execution": execution, "states": out_states}

    @app.post("/api/evaluate")
    async def evaluate(request: Request):
        t0 = time.perf_counter()
        try:
            payload = await request.json()
            states = S.validate_request(payload)
            if len(states) > limits["max_states_per_request"]:
                raise ValueError(f"at most {limits['max_states_per_request']} states per request")
            # one read per (state, chunk of questions that fits half the served canvas)
            reads = [(si, sub) for si, st in enumerate(states)
                     for sub in split_questions(st, encoder, reader.served_canvas // 2)]
            if isinstance(reader, FastPool):
                return await evaluate_fast(states, reads, t0)
            if isinstance(reader, (InprocReader, MultiReader)):
                need = [s for s in states if s.get("images")]
                loaded = {}
                if need:
                    def load_all():
                        return {s["id"]: [S.load_image(u) for u in s["images"]] for s in need}
                    loaded = await asyncio.to_thread(load_all)
                built = [reader.build(sub, encoder, loaded.get(sub["id"], ())) for _, sub in reads]
            else:
                built = [reader.build(sub, encoder) for _, sub in reads]
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            resps = await asyncio.gather(*[reader.read(a, b) for a, b, *_ in built])
        except Exception as exc:
            stats["errors"] += 1
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=502, content={"error": "vLLM request failed",
                                                          "detail": f"{type(exc).__name__}: {exc}"[:1500]})
        per_state = [{"answers": {}, "prompt_tokens": 0, "images": 0, "reads": 0} for _ in states]
        n_q = 0
        for (si, sub), (url, req, slots, off, n_img), resp in zip(reads, built, resps):
            rows = reader.rows(resp)
            answers = per_state[si]["answers"]
            for slot, (probs, mass) in zip(slots, reader.answers(slots, rows, off)):
                n_q += 1
                answers[slot.qid] = answer_obj(slot, probs, mass)
            if isinstance(reader, (InprocReader, MultiReader)):
                ptok = reader.prompt_tokens(resp)
            else:
                ptok = resp.get("usage", {}).get("prompt_tokens") or 0
            ps = per_state[si]
            ps["prompt_tokens"] = max(ps["prompt_tokens"], ptok)
            ps["images"] = n_img
            ps["reads"] += 1
            stats["images"] += n_img
        out_states = []
        for st, ps in zip(states, per_state):
            ordered = {q: ps["answers"][q] for q in st["questions"]}   # request order
            out_states.append({"id": st["id"], "state": st["state"], "questions": st["questions"], "answers": ordered,
                               "prompt_tokens": ps["prompt_tokens"], "images": ps["images"], "reads": ps["reads"]})
        stats["requests"] += 1
        stats["states"] += len(states)
        stats["questions"] += n_q
        elapsed = time.perf_counter() - t0
        print(json.dumps({"evaluate": {"states": len(states), "questions": n_q, "seconds": round(elapsed, 3)}}), flush=True)
        return {"model": model_name,
                "execution": {"states": len(states), "questions": n_q, "forward_passes": 1,
                              "autoregressive_decode_steps": 0, "network_model_calls": 0, "backend": "vllm",
                              "readout": {"min_width": reader.width, "thought": reader.thought, "constrained": reader.constrained},
                              "server_evaluation_seconds": elapsed},
                "states": out_states}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["inproc", "http", "fast"], default="inproc",
                   help="inproc: run the vLLM engine inside this process (fastest); http: talk to a separate `vllm serve`")
    p.add_argument("--upstream", default="http://127.0.0.1:8000", help="[http] vllm serve base URL")
    p.add_argument("--dp", type=int, default=1, help="[inproc] vLLM data-parallel replicas (lockstep; prefer --gpus)")
    p.add_argument("--gpus", default=None,
                   help="[inproc] comma-separated physical GPU ids: one independent single-GPU engine per id")
    p.add_argument("--ids", choices=["main", "all"], default="main",
                   help="candidate spellings read per answer: main (' A', ' yes', ' no') or all variants summed")
    p.add_argument("--moe-backend", default="triton", help="[inproc] vLLM MoE backend")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="[inproc]")
    p.add_argument("--max-model-len", type=int, default=4096, help="[inproc]")
    p.add_argument("--batch-max", type=int, default=1,
                   help="[fast] text reads per forward when they queue up (1 = one read per forward). More bots per "
                        "GPU under load; a quiet GPU still takes each read alone")
    p.add_argument("--batch-wait-ms", type=float, default=4.0,
                   help="[fast] while busy, how long a forward waits for reads about to arrive (0: only those queued)")
    p.add_argument("--batch-tokens", type=int, default=2048,
                   help="[fast] tokens computed per read that the batched graphs are sized for (longer: run alone)")
    p.add_argument("--prefix-slots", type=int, default=8,
                   help="[fast] prompt-prefix KV slots per GPU (0 = off): a read reuses the longest cached prefix")
    p.add_argument("--triton-attention", action="store_true",
                   help="[fast] vLLM's Triton attention everywhere (default: FlashAttention-2 for text reads)")
    p.add_argument("--quantization", default=None, help="[fast] vLLM online quantization, e.g. fp8 (default: bf16)")
    p.add_argument("--model", default="djev", help="served model name (http: as vllm serve reports it)")
    p.add_argument("--tokenizer", default=S.MODEL_DEFAULT)
    p.add_argument("--served-canvas", type=int, default=64,
                   help="served canvas_length. 64 is ~3 ms/request faster than 256 (vLLM pads logits to it); states "
                        "whose scaffold exceeds half of it are split into several reads")
    p.add_argument("--width", type=int, default=64,
                   help="minimum per-request canvas width; doubled until the scaffold fills at most half (needs --async-scheduling upstream)")
    p.add_argument("--thought", choices=["canvas", "prompt"], default="prompt",
                   help="where the empty thinking block goes: canvas head (chat API; required for images) or prompt (token prompt)")
    p.add_argument("--constrained", action="store_true", help="diffusion_constrained: unembed over candidates only (candidate_mass is then always 1)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-states-per-request", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-connections", type=int, default=256)
    p.add_argument("--api-key-file", help="require this key on /api/* (Bearer token, or basic auth password)")
    a = p.parse_args()
    api_key = None
    if a.api_key_file:
        with open(a.api_key_file) as f:
            api_key = f.read().strip()
        if not api_key:
            raise SystemExit("--api-key-file is empty")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    encoder = S.Encoder(tok, 1 << 30, empty_thought=True)   # no processor: images are handled by vLLM
    if a.ids == "main":   # one spelling per candidate: same answers, ~1 ms less per request (probes/results)
        encoder.label_groups = [g[:1] for g in encoder.label_groups]
        encoder.yes_group, encoder.no_group = encoder.yes_group[:1], encoder.no_group[:1]
    if a.engine == "fast":
        gpus = [g for g in (a.gpus or "0").split(",") if g != ""]
        reader = FastPool(gpus, {"canvas": a.served_canvas, "moe": a.moe_backend,
                                 "gpu_memory_utilization": a.gpu_memory_utilization, "max_model_len": a.max_model_len,
                                 "thought_ids": list(encoder.thought_prefix_ids), "prefix_slots": a.prefix_slots,
                                 "fast_attention": not a.triton_attention, "quantization": a.quantization,
                                 "batch_max": a.batch_max, "batch_tokens": a.batch_tokens,
                                 "batch_wait_ms": a.batch_wait_ms})
    elif a.engine == "inproc":
        from vllm.engine.arg_utils import AsyncEngineArgs
        eargs = AsyncEngineArgs(model=a.tokenizer, served_model_name=[a.model], tokenizer=a.tokenizer,
                                diffusion_config={"canvas_length": a.served_canvas}, max_logprobs=128,
                                async_scheduling=a.width < a.served_canvas, enable_prefix_caching=True,
                                kernel_config={"moe_backend": a.moe_backend},
                                gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                                limit_mm_per_prompt={"image": S.MAX_IMAGES}, data_parallel_size=a.dp)
        if a.gpus:
            gpus = [g for g in a.gpus.split(",") if g != ""]
            readers = []
            for g in gpus:
                readers.append(InprocReader(eargs, a.served_canvas, a.width, a.constrained, gpu=g))
                print(json.dumps({"engine": len(readers) - 1, "gpu": g, "ready": True}), flush=True)
            reader = MultiReader(readers) if len(readers) > 1 else readers[0]
        else:
            reader = InprocReader(eargs, a.served_canvas, a.width, a.constrained)
    else:
        reader = Reader(a.upstream, a.model, a.served_canvas, a.width, a.thought, a.constrained, a.timeout,
                        a.max_connections)
    limits = {"max_states_per_request": a.max_states_per_request, "max_options": len(encoder.labels),
              "max_candidate_tokens": 128, "max_scaffold_tokens": a.served_canvas // 2,
              "max_images_per_state": S.MAX_IMAGES}
    app = make_app(reader, encoder, limits, a.model, api_key)
    import uvicorn
    print(json.dumps({"url": f"http://{a.host}:{a.port}", "engine": a.engine, "upstream": reader.upstream,
                      "auth": bool(api_key), "ready": True}), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
