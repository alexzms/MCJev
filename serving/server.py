#!/usr/bin/env python3
"""DJev — DiffusionGemma as a Jev-style parallel decision model.

A request carries states; each state carries questions (choice / boolean / score).
For every state we build ONE prompt (user turn) and ONE 256-token canvas (model turn).
The canvas is a fixed answer scaffold

    1: ▁?      <- "?" is a noised slot, one per question
    2: ▁?
    <turn|> <pad> <pad> ...

DiffusionGemma's decoder denoises the whole canvas in a single bidirectional pass.
The logits at each slot, restricted to that question's candidate tokens (" A", " B", ...
or " yes"/" no"), are the answer distribution.  No token is ever sampled or decoded.

HTTP API (the NanoJev request/response contract, https://github.com/TianyuCodings/NanoJev, plus optional images):
    GET  /api/health
    POST /api/evaluate   {"states": [{"id", "state", "questions": {qid: {...}}, "images": [url | data-URI]}]}

Images go through DiffusionGemma's vision encoder (~260 soft tokens each) and are referred to in
the prompt as "Image 1", "Image 2", ... in the order given.

Concurrency: the main process tokenizes and batches requests; one worker process per GPU
runs the model.  Requests arriving within `--batch-wait-ms` of each other share a forward.
"""

import argparse
import itertools
import json
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- constants

MODEL_DEFAULT = "google/diffusiongemma-26B-A4B-it"
CANVAS_LENGTH = 256
PAD_ID = 0            # <pad>
EOT_ID = 106          # <turn|>
MASK_ID = 4           # <mask>
MAX_IMAGES = 8        # per state
MAX_IMAGE_BYTES = 20_000_000
# Single-token option labels, tried in this order (verified against the tokenizer at start).
LABEL_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
)

PREAMBLE = (
    "You are a decision model. Read the state, then answer every question below. "
    "Judge each question independently, using only the state and that question's own instructions. "
    "Reply with exactly one line per question in the answer format given at the end, and nothing else."
)


# --------------------------------------------------------------------------- validation

def nonempty_text(v):
    return isinstance(v, str) and v.strip() != ""


def validate_request(payload):
    """The request contract of NanoJev's validate_request, with English messages and optional images.

    Adapted from NanoJev (https://github.com/TianyuCodings/NanoJev, scripts/predict_toy_decisions.py),
    Copyright (c) 2026 OpenJev contributors, MIT License (see THIRD_PARTY_NOTICES.md).
    """
    if not isinstance(payload, dict) or not (set(payload) <= {"states", "options"}) or "states" not in payload:
        raise ValueError('Request must be {"states": [...]} (optional "options")')
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a non-empty list")
    seen = set()
    for s in states:
        if not isinstance(s, dict) or not ({"id", "state", "questions"} <= set(s) <= {"id", "state", "questions", "images"}):
            raise ValueError("each state must have the keys id, state, questions (and optionally images)")
        imgs = s.get("images")
        if imgs is not None and (not isinstance(imgs, list) or not 1 <= len(imgs) <= MAX_IMAGES
                                 or not all(nonempty_text(u) for u in imgs)):
            raise ValueError(f"images must be a list of 1..{MAX_IMAGES} http(s) URLs or data:image/...;base64 URIs")
        if not nonempty_text(s["id"]) or s["id"] in seen:
            raise ValueError("state id must be a unique non-empty string")
        seen.add(s["id"])
        if not isinstance(s["state"], (str, dict, list)) or not s["state"]:
            raise ValueError("state must be a non-empty string, object or array")
        qs = s["questions"]
        if not isinstance(qs, dict) or not qs:
            raise ValueError("questions must be a non-empty object")
        for qid, q in qs.items():
            if not nonempty_text(qid) or not isinstance(q, dict):
                raise ValueError("question id must be a non-empty string and the question an object")
            if set(q) - {"type", "instructions", "criteria"}:
                raise ValueError(f"{s['id']}:{qid} has unsupported fields")
            typ = q.get("type")
            if typ not in {"boolean", "choice", "score"} or not nonempty_text(q.get("instructions")):
                raise ValueError(f"{s['id']}:{qid} needs type in (boolean, choice, score) and instructions")
            if typ == "boolean":
                c = q.get("criteria")
                if c is not None and (not isinstance(c, dict) or set(c) - {"false", "true"}
                                      or not all(nonempty_text(v) for v in c.values())):
                    raise ValueError(f"{s['id']}:{qid} boolean criteria must be an object with true/false strings")
            elif typ == "choice":
                c = q.get("criteria")
                if not isinstance(c, dict) or not 2 <= len(c) <= len(LABEL_POOL):
                    raise ValueError(f"{s['id']}:{qid} choice criteria must be an object with 2..{len(LABEL_POOL)} options")
                if not all(nonempty_text(k) and nonempty_text(v) for k, v in c.items()):
                    raise ValueError(f"{s['id']}:{qid} choice option ids and descriptions must be non-empty strings")
            else:
                c = q.get("criteria")
                if not isinstance(c, list) or not 2 <= len(c) <= 10 or not all(nonempty_text(v) for v in c):
                    raise ValueError(f"{s['id']}:{qid} score criteria must be an ordered list of 2..10 strings")
    return states


# --------------------------------------------------------------------------- encoding

def load_image_bytes(src, timeout=20):
    """Raw bytes of an http(s) URL or a base64 data: URI (decoded later, by the worker)."""
    import base64
    import urllib.request
    if src.startswith("data:"):
        header, _, b64 = src.partition(",")
        if ";base64" not in header or not b64:
            raise ValueError("data: URI must be base64 encoded")
        raw = base64.b64decode(b64)
    elif src.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "djev/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(MAX_IMAGE_BYTES + 1)
        except Exception as exc:
            raise ValueError(f"cannot fetch image {src[:80]}: {exc}") from exc
    else:
        raise ValueError("image must be an http(s) URL or a data:image/...;base64 URI")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image larger than {MAX_IMAGE_BYTES // 1_000_000} MB")
    return raw


def load_image(src, timeout=20):
    """Fetch an http(s) URL or decode a data: URI into an RGB PIL image."""
    import base64
    import io
    import urllib.request
    from PIL import Image

    if src.startswith("data:"):
        header, _, b64 = src.partition(",")
        if ";base64" not in header or not b64:
            raise ValueError("data: URI must be base64 encoded")
        raw = base64.b64decode(b64)
    elif src.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "djev/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(MAX_IMAGE_BYTES + 1)
        except Exception as exc:
            raise ValueError(f"cannot fetch image {src[:80]}: {exc}") from exc
    else:
        raise ValueError("image must be an http(s) URL or a data:image/...;base64 URI")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image larger than {MAX_IMAGE_BYTES // 1_000_000} MB")
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    return im.convert("RGB")


@dataclass
class Slot:
    qid: str
    qtype: str
    keys: list            # output keys, aligned with cand_groups
    cand_groups: list     # per key, the token ids whose probability is summed (" A", "A", ...)
    pos: int = -1         # position inside the canvas


@dataclass
class Example:
    state_id: str
    prompt_ids: list
    canvas_ids: list      # placeholder MASK_ID at slot positions
    slots: list = field(default_factory=list)
    n_images: int = 0
    mm_token_type_ids: list | None = None   # 1 at image soft tokens (only when images are present)
    pixel_values: object = None             # np.float16 (n_images, max_patches, patch_dim)
    image_position_ids: object = None       # np.int32   (n_images, max_patches, 2), -1 = padding

    def payload(self):
        return {"prompt_ids": self.prompt_ids, "canvas_ids": self.canvas_ids,
                "slots": [(s.pos, s.cand_groups) for s in self.slots],
                "mm_token_type_ids": self.mm_token_type_ids,
                "pixel_values": self.pixel_values, "image_position_ids": self.image_position_ids}


class Encoder:
    """Turns a state into (prompt token ids, canvas token ids, slots)."""

    def __init__(self, tokenizer, max_prompt_tokens, empty_thought=True, processor=None):
        self.tok = tokenizer
        self.processor = processor          # needed for images (AutoProcessor); text-only works without
        self.max_prompt_tokens = max_prompt_tokens
        # option labels whose " X" form is a single token; variants ("X", " x") are summed into the same key
        self.labels, self.label_ids, self.label_groups = [], [], []
        for lab in LABEL_POOL:
            ids = tokenizer.encode(" " + lab, add_special_tokens=False)
            if len(ids) == 1:
                self.labels.append(lab)
                self.label_ids.append(ids[0])
                self.label_groups.append(self._variants([" " + lab, lab]))
        self.yes_id = self._single(" yes")
        self.no_id = self._single(" no")
        self.yes_group = self._variants([" yes", "yes", " Yes", "Yes", " YES", " true", " True"])
        self.no_group = self._variants([" no", "no", " No", "No", " NO", " false", " False"])
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        # Gemma 4 replies open with an empty thinking channel when thinking is off; putting it in the
        # prompt keeps the canvas aligned so that "1:" is the first canvas token.
        self.thought_prefix_ids = (tokenizer.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
                                   if empty_thought else [])
        # the scaffold must tokenize identically piecewise and jointly
        joint = tokenizer.encode("1: A\n2: no\n", add_special_tokens=False)
        piece = (tokenizer.encode("1:", add_special_tokens=False) + [self.label_ids[0]] + self.newline_ids
                 + tokenizer.encode("2:", add_special_tokens=False) + [self.no_id] + self.newline_ids)
        if joint != piece:
            raise RuntimeError(f"scaffold tokenization mismatch: {joint} != {piece}")

    def _single(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} is not a single token: {ids}")
        return ids[0]

    def _variants(self, texts):
        """Distinct single-token ids for the given spellings (first one must exist)."""
        out = []
        for t in texts:
            ids = self.tok.encode(t, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        if not out:
            raise RuntimeError(f"no single-token spelling among {texts}")
        return out

    @staticmethod
    def state_text(state):
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

    def prompt_text(self, state, n_images=0):
        """The user-turn text and the answer slots for a state (no tokenization)."""
        questions = list(state["questions"].items())
        lines = [PREAMBLE]
        if n_images:
            lines.append(f"{n_images} image(s) are attached above, in order: "
                         + ", ".join(f"Image {i}" for i in range(1, n_images + 1)) + ".")
        lines += ["", "State:", self.state_text(state["state"]), ""]
        fmt = ["Answer format (one line per question):"]
        slots = []
        for i, (qid, q) in enumerate(questions, start=1):
            typ = q["type"]
            if typ == "choice":
                keys = list(q["criteria"].keys())
                n = len(keys)
                if n > len(self.labels):
                    raise ValueError(f"{state['id']}:{qid} has {n} options; max {len(self.labels)}")
                lines.append(f"Question {i} (choice): {q['instructions']}")
                lines.append("Options:")
                for lab, key in zip(self.labels, keys):
                    lines.append(f"{lab}) {key}: {q['criteria'][key]}")
                fmt.append(f"{i}: <one option letter, {self.labels[0]}-{self.labels[n - 1]}>")
                slots.append(Slot(qid, typ, keys, self.label_groups[:n]))
            elif typ == "boolean":
                lines.append(f"Question {i} (yes/no): {q['instructions']}")
                c = q.get("criteria") or {}
                if "true" in c:
                    lines.append(f'Answer "yes" if: {c["true"]}')
                if "false" in c:
                    lines.append(f'Answer "no" if: {c["false"]}')
                fmt.append(f"{i}: <yes or no>")
                slots.append(Slot(qid, typ, ["false", "true"], [self.no_group, self.yes_group]))
            else:  # score
                levels = q["criteria"]
                n = len(levels)
                lines.append(f"Question {i} (score): {q['instructions']}")
                lines.append("Levels, ordered from lowest to highest:")
                for lab, desc in zip(self.labels, levels):
                    lines.append(f"{lab}) {desc}")
                fmt.append(f"{i}: <one level letter, {self.labels[0]}-{self.labels[n - 1]}>")
                slots.append(Slot(qid, typ, [str(j) for j in range(n)], self.label_groups[:n]))
            lines.append("")
        lines.extend(fmt)
        return "\n".join(lines), slots

    def scaffold(self, slots, state_id="?"):
        """Canvas body: '1: <mask>\n2: <mask>\n...<turn|>' (no padding); sets slot.pos."""
        canvas = []
        for i, slot in enumerate(slots, start=1):
            canvas.extend(self.tok.encode(f"{i}:", add_special_tokens=False))
            slot.pos = len(canvas)
            canvas.append(MASK_ID)
            canvas.extend(self.newline_ids)
        canvas.append(EOT_ID)
        if len(canvas) > CANVAS_LENGTH:
            raise ValueError(f"{state_id}: {len(slots)} questions do not fit in one {CANVAS_LENGTH}-token canvas")
        return canvas

    def encode(self, state) -> Example:
        images = [load_image(u) for u in (state.get("images") or [])]
        if images and self.processor is None:
            raise ValueError("this server was started without image support")
        text, slots = self.prompt_text(state, len(images))
        mm_ids = pixel_values = image_position_ids = None
        if images:
            import numpy as np
            content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
            out = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, return_dict=True,
                add_generation_prompt=True, return_tensors="pt")
            prompt_ids = out["input_ids"][0].tolist() + self.thought_prefix_ids
            mm_ids = out["mm_token_type_ids"][0].tolist() + [0] * len(self.thought_prefix_ids)
            pixel_values = out["pixel_values"].numpy().astype(np.float16)
            image_position_ids = out["image_position_ids"].numpy().astype(np.int32)
        else:
            prompt_ids = self.tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                return_dict=False)
            if isinstance(prompt_ids, dict):
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = list(prompt_ids) + self.thought_prefix_ids
        if len(prompt_ids) > self.max_prompt_tokens:
            raise ValueError(f"{state['id']}: prompt is {len(prompt_ids)} tokens; max {self.max_prompt_tokens}")

        canvas = self.scaffold(slots, state["id"])
        canvas.extend([PAD_ID] * (CANVAS_LENGTH - len(canvas)))
        return Example(state["id"], prompt_ids, canvas, slots, n_images=len(images), mm_token_type_ids=mm_ids,
                       pixel_values=pixel_values, image_position_ids=image_position_ids)


# --------------------------------------------------------------------------- model side (runs in workers)

def load_model(model_path, device, experts_impl="grouped_mm", attn_impl="sdpa"):
    import torch
    from transformers import DiffusionGemmaForBlockDiffusion

    kwargs = dict(dtype=torch.bfloat16, local_files_only=True, attn_implementation=attn_impl)
    if experts_impl:
        kwargs["experts_implementation"] = experts_impl
    try:
        import accelerate  # noqa: F401  (device_map needs it: weights stream straight to the GPU)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_path, device_map={"": device}, **kwargs)
    except ImportError:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    return model


def run_batch(model, payloads, cfg):
    """payloads: list of Example.payload(); returns per example a list of (probs, mass) per slot."""
    import numpy as np
    import torch
    from transformers import DynamicCache

    device, dtype = model.device, model.dtype
    K = cfg["noise_samples"]
    rows = [p for p in payloads for _ in range(K)]        # row r belongs to payload r // K
    R = len(rows)
    L = max(len(p["prompt_ids"]) for p in rows)
    input_ids = torch.full((R, L), PAD_ID, dtype=torch.long)
    attn = torch.zeros((R, L), dtype=torch.bool)
    mm = torch.zeros((R, L), dtype=torch.long)
    pv_list, ipi_list = [], []
    for r, p in enumerate(rows):
        n = len(p["prompt_ids"])
        input_ids[r, L - n:] = torch.tensor(p["prompt_ids"], dtype=torch.long)
        attn[r, L - n:] = True
        if p.get("mm_token_type_ids") is not None:
            mm[r, L - n:] = torch.tensor(p["mm_token_type_ids"], dtype=torch.long)
        if p.get("pixel_values") is not None:
            pv_list.append(p["pixel_values"])
            ipi_list.append(p["image_position_ids"])
    canvas = torch.tensor([p["canvas_ids"] for p in rows], dtype=torch.long)
    input_ids, attn, mm, canvas = input_ids.to(device), attn.to(device), mm.to(device), canvas.to(device)
    has_images = bool(pv_list)
    enc_kwargs = {}
    if has_images:
        enc_kwargs = {
            "pixel_values": torch.from_numpy(np.concatenate(pv_list)).to(device=device, dtype=dtype),
            "image_position_ids": torch.from_numpy(np.concatenate(ipi_list)).to(device=device, dtype=torch.long),
            "mm_token_type_ids": mm,
        }

    slot_rows, slot_cols, cand_sets = [], [], []
    for r, p in enumerate(rows):
        for pos, groups in p["slots"]:
            slot_rows.append(r)
            slot_cols.append(pos)
            cand_sets.append({i for g in groups for i in g})
    rows_t = torch.tensor(slot_rows, device=device)
    cols_t = torch.tensor(slot_cols, device=device)

    def noised():
        c = canvas.clone()
        if cfg["slot_noise"] == "mask":
            c[rows_t, cols_t] = MASK_ID
        else:
            draw = torch.randint(0, model.config.text_config.vocab_size, (len(slot_rows),), device=device).tolist()
            for j, d in enumerate(draw):
                if d in cand_sets[j] or d in (PAD_ID, EOT_ID):
                    draw[j] = MASK_ID
            c[rows_t, cols_t] = torch.tensor(draw, device=device)
        return c

    dec_mask = torch.cat([attn, torch.ones((R, CANVAS_LENGTH), dtype=torch.bool, device=device)], dim=1)
    pos_ids = torch.arange(0, L, device=device).unsqueeze(0)
    dpos_ids = torch.arange(L, L + CANVAS_LENGTH, device=device).unsqueeze(0)

    with torch.inference_mode():
        # 1. encoder: prompt (+ images) -> KV cache.  Same call sequence as generate(): the 4D masks are
        #    built here so image soft tokens attend bidirectionally (mm_token_type_ids).
        cache = DynamicCache(config=model.config.get_text_config(decoder=True))
        dummy = torch.empty((R, L, 0), dtype=dtype, device=device)
        enc_mask = model.model.encoder.create_masks_for_generate(
            config=model.config, inputs_embeds=dummy, attention_mask=attn, past_key_values=cache,
            position_ids=pos_ids, mm_token_type_ids=mm if has_images else None)
        enc_out = model.model.encoder(input_ids=input_ids, attention_mask=enc_mask, position_ids=pos_ids,
                                      past_key_values=cache, **enc_kwargs)
        cache = enc_out.past_key_values
        # 2. decoder: denoise the scaffolded canvas once (optionally more passes with self-conditioning)
        out = model(past_key_values=cache, decoder_input_ids=noised(), decoder_attention_mask=dec_mask,
                    decoder_position_ids=dpos_ids)
        logits = out.logits
        for _ in range(cfg["steps"] - 1):
            sc = (logits / cfg["sc_temperature"]).to(torch.bfloat16)
            out = model(past_key_values=cache, decoder_input_ids=noised(), decoder_attention_mask=dec_mask,
                        decoder_position_ids=dpos_ids, self_conditioning_logits=sc)
            logits = out.logits
        slot_logits = logits[rows_t, cols_t].float()          # (n_slots_total, vocab)
        logp = torch.log_softmax(slot_logits, dim=-1)

    # 3. readout: per payload, average the K noise rows; per slot sum spelling variants, renormalise
    results, j = [], 0
    n_slots = [len(p["slots"]) for p in payloads]
    for b, p in enumerate(payloads):
        per_slot = []
        for si, (pos, groups) in enumerate(p["slots"]):
            idx = torch.tensor([j + k * n_slots[b] + si for k in range(K)], device=device)
            flat = torch.tensor([i for g in groups for i in g], device=device)
            pr = logp[idx][:, flat].exp()                          # (K, n_variants)
            sizes = [len(g) for g in groups]
            per_key = torch.stack([c.sum(-1) for c in torch.split(pr, sizes, dim=-1)], dim=-1)  # (K, n_cand)
            mass = per_key.sum(-1).mean().item()
            probs = (per_key / per_key.sum(-1, keepdim=True).clamp_min(1e-30)).mean(0)
            per_slot.append((probs.tolist(), mass))
        j += K * n_slots[b]
        results.append(per_slot)
    return results


def worker_main(rank, device, model_path, in_q, out_q, cfg):
    import torch
    t0 = time.time()
    try:
        model = load_model(model_path, device, cfg["experts_impl"], cfg["attn_impl"])
        # warm up (lazy kernel init) with a tiny fake example
        warm = {"prompt_ids": [2, 105, 2364, 107, 3112, 106, 107, 105, 4368, 107],
                "canvas_ids": [PAD_ID] * CANVAS_LENGTH, "slots": [(1, [[PAD_ID], [MASK_ID]])]}
        warm["canvas_ids"][0], warm["canvas_ids"][2], warm["canvas_ids"][3] = 236770, 107, EOT_ID
        run_batch(model, [warm], cfg)
        torch.cuda.synchronize(device)
        out_q.put(("ready", rank, {"device": device, "load_seconds": round(time.time() - t0, 1),
                                   "gpu_mem_gb": round(torch.cuda.memory_allocated(device) / 2**30, 1)}))
    except Exception:
        out_q.put(("fatal", rank, traceback.format_exc()))
        return
    while True:
        job = in_q.get()
        if job is None:
            return
        job_id, payloads = job
        t1 = time.time()
        try:
            res = run_batch(model, payloads, cfg)
            torch.cuda.synchronize(device)
            out_q.put(("done", job_id, res, rank, time.time() - t1))
        except Exception:
            out_q.put(("error", job_id, traceback.format_exc(), rank, time.time() - t1))


# --------------------------------------------------------------------------- engine (main process)

class Engine:
    def __init__(self, model_path, gpus, cfg, max_batch, batch_wait_ms):
        import torch.multiprocessing as mp
        self.cfg, self.max_batch, self.batch_wait = cfg, max_batch, batch_wait_ms / 1000.0
        ctx = mp.get_context("spawn")
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=worker_main, args=(r, f"cuda:{g}", model_path, self.in_q, self.out_q, cfg),
                                  daemon=True) for r, g in enumerate(gpus)]
        for p in self.procs:
            p.start()
        self.workers = {}
        for _ in self.procs:
            kind, rank, info = self.out_q.get()
            if kind != "ready":
                raise SystemExit(f"worker {rank} failed to start:\n{info}")
            self.workers[rank] = info
            print(json.dumps({"worker": rank, **info}), flush=True)
        self.pending = queue.Queue()
        self.inflight = {}
        self.job_ids = itertools.count()
        self.stats = {"requests": 0, "states": 0, "questions": 0, "batches": 0, "forward_seconds": 0.0}
        threading.Thread(target=self._dispatch, daemon=True).start()
        threading.Thread(target=self._collect, daemon=True).start()

    def submit(self, examples):
        futs = []
        for ex in examples:
            f = Future()
            self.pending.put((f, ex.payload()))
            futs.append(f)
        return futs

    def _dispatch(self):
        while True:
            first = self.pending.get()
            batch = [first]
            deadline = time.time() + self.batch_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.pending.get(timeout=remaining))
                except queue.Empty:
                    break
            job_id = next(self.job_ids)
            self.inflight[job_id] = [f for f, _ in batch]
            self.in_q.put((job_id, [p for _, p in batch]))

    def _collect(self):
        while True:
            msg = self.out_q.get()
            kind, job_id = msg[0], msg[1]
            futs = self.inflight.pop(job_id, [])
            self.stats["batches"] += 1
            self.stats["forward_seconds"] += msg[4]
            if kind == "done":
                for f, r in zip(futs, msg[2]):
                    f.set_result((r, msg[3], msg[4]))
            else:
                for f in futs:
                    f.set_exception(RuntimeError(msg[2]))


# --------------------------------------------------------------------------- HTTP

def make_app(engine, encoder, model_name, limits):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import asyncio

    from fastapi.responses import PlainTextResponse
    app = FastAPI(title="DJev", docs_url=None, redoc_url=None)
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")

    @app.get("/")
    async def index():
        return {"service": "djev (DiffusionGemma-as-Jev)", "manual": "GET /manual", "health": "GET /api/health",
                "evaluate": "POST /api/evaluate  {\"states\": [{\"id\", \"state\", \"questions\": {...}}]}"}

    @app.get("/manual", response_class=PlainTextResponse)
    async def manual():
        try:
            with open(readme, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return "README.md not found next to server.py"

    @app.get("/api/health")
    async def health():
        return {"ready": True, "model": model_name, "workers": engine.workers, "readout": engine.cfg,
                "limits": limits, "queue_depth": engine.pending.qsize(), "stats": engine.stats}

    @app.post("/api/evaluate")
    async def evaluate(request: Request):
        t0 = time.perf_counter()
        try:
            payload = await request.json()
            states = validate_request(payload)
            if len(states) > limits["max_states_per_request"]:
                raise ValueError(f"at most {limits['max_states_per_request']} states per request")
            examples = await asyncio.to_thread(lambda: [encoder.encode(s) for s in states])
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            results = await asyncio.gather(*[asyncio.wrap_future(f) for f in engine.submit(examples)])
        except Exception as exc:  # worker-side failure
            return JSONResponse(status_code=500, content={"error": "model inference failed", "detail": str(exc)[-2000:]})

        out_states, n_q = [], 0
        for state, ex, (per_slot, rank, fwd) in zip(states, examples, results):
            answers = {}
            for slot, (probs, mass) in zip(ex.slots, per_slot):
                n_q += 1
                dist = {k: p for k, p in zip(slot.keys, probs)}
                best = max(dist, key=dist.get)
                if slot.qtype == "choice":
                    ans = {"type": "choice", "probabilities": dist, "choice": best, "value": best}
                elif slot.qtype == "boolean":
                    ans = {"type": "boolean", "probabilities": dist, "p_true": dist["true"], "value": dist["true"] >= 0.5}
                else:
                    score = sum(int(k) * p for k, p in dist.items())
                    ans = {"type": "score", "probabilities": dist, "score": score, "level": int(best), "value": score}
                ans["candidate_mass"] = mass
                answers[slot.qid] = ans
            out_states.append({"id": state["id"], "state": state["state"], "questions": state["questions"],
                               "answers": answers, "prompt_tokens": len(ex.prompt_ids), "images": ex.n_images,
                               "worker": rank})
        engine.stats["requests"] += 1
        engine.stats["states"] += len(states)
        engine.stats["questions"] += n_q
        elapsed = time.perf_counter() - t0
        print(json.dumps({"evaluate": {"states": len(states), "questions": n_q, "images": sum(e.n_images for e in examples),
                                       "seconds": round(elapsed, 3)}}), flush=True)
        return {"model": model_name,
                "execution": {"states": len(states), "questions": n_q, "forward_passes": engine.cfg["steps"],
                              "autoregressive_decode_steps": 0, "network_model_calls": 0,
                              "readout": engine.cfg, "server_evaluation_seconds": elapsed},
                "states": out_states}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_DEFAULT, help="HF repo id (must already be in the HF cache) or local path")
    p.add_argument("--gpus", default="0", help="comma-separated CUDA device indices, one worker each")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-batch", type=int, default=16, help="states per forward pass")
    p.add_argument("--batch-wait-ms", type=float, default=5.0, help="how long to wait for more states before a forward")
    p.add_argument("--max-states-per-request", type=int, default=64)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1, help="denoising passes over the canvas (>1 uses self-conditioning)")
    p.add_argument("--slot-noise", choices=["random", "mask"], default="mask",
                   help="what the answer slots contain before denoising: a fresh random token (as in generation) or <mask>")
    p.add_argument("--noise-samples", type=int, default=1, help="average the readout over this many noise draws")
    p.add_argument("--sc-temperature", type=float, default=1.0, help="temperature applied to logits fed back as self-conditioning")
    p.add_argument("--experts-impl", default="grouped_mm", help="transformers MoE experts implementation ('' = eager)")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--no-empty-thought", action="store_true",
                   help="do not append the empty <|channel>thought<channel|> block to the prompt")
    a = p.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    encoder = Encoder(processor.tokenizer, a.max_prompt_tokens, empty_thought=not a.no_empty_thought,
                      processor=processor)
    cfg = {"steps": a.steps, "slot_noise": a.slot_noise, "noise_samples": a.noise_samples,
           "sc_temperature": a.sc_temperature, "experts_impl": a.experts_impl, "attn_impl": a.attn_impl}
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    print(json.dumps({"loading": a.model, "gpus": gpus, "readout": cfg}), flush=True)
    engine = Engine(a.model, gpus, cfg, a.max_batch, a.batch_wait_ms)
    limits = {"max_states_per_request": a.max_states_per_request, "max_prompt_tokens": a.max_prompt_tokens,
              "max_options": len(encoder.labels), "max_questions_per_state": (CANVAS_LENGTH - 1) // 4,
              "max_images_per_state": MAX_IMAGES, "max_image_bytes": MAX_IMAGE_BYTES}
    app = make_app(engine, encoder, a.model, limits)
    import uvicorn
    print(json.dumps({"url": f"http://{a.host}:{a.port}", "ready": True}), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")   # returns after SIGINT/SIGTERM
    print(json.dumps({"shutdown": True}), flush=True)
    for proc in engine.procs:
        proc.terminate()
    os._exit(0)  # skip multiprocessing's atexit joins, which can hang on live queues


if __name__ == "__main__":
    main()
