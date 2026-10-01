#!/usr/bin/env python3
"""Prototype: djev read as ONE forward pass driven directly on a vLLM worker (no scheduler).

  tokens  = [prompt (L) ; canvas (C)]            canvas embeddings go through self_conditioning(e, 0)
  batch   = two pseudo-sequences on the same KV blocks:
              A: the prompt, causal, seq_len L
              B: the canvas, non-causal, seq_len L+C  (sees the whole prompt + the whole canvas)
  logits  = lm_head only at the answer slots

Checks the slot distributions against the live djev server (two-step read) and times the forward.
    python probes/fastpath_proto.py [--ref-port 8767]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # engine core in this process: we can reach the worker
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import server as S  # noqa: E402
from client import EXAMPLE  # noqa: E402

CANVAS = 64


class FastPath:
    def __init__(self, model_name=S.MODEL_DEFAULT, moe="triton", graphs=False):
        from vllm import LLM
        self.graphs = graphs
        self.llm = LLM(model=model_name, diffusion_config={"canvas_length": CANVAS}, enforce_eager=not graphs,
                       kernel_config={"moe_backend": moe}, gpu_memory_utilization=0.85, max_model_len=4096,
                       enable_prefix_caching=False, max_logprobs=128, limit_mm_per_prompt={"image": 8})
        core = self.llm.llm_engine.engine_core.engine_core
        worker = core.model_executor.driver_worker.worker
        self.runner = worker.model_runner
        self.model = self.runner.model
        self.vllm_config = self.runner.vllm_config
        self.device = self.runner.device
        kcc = self.runner.kv_cache_config
        self.groups = []
        for gi, g in enumerate(kcc.kv_cache_groups):
            spec = g.kv_cache_spec
            bs = spec.block_size
            kbs = self.runner.kernel_block_sizes[gi]
            self.groups.append((bs, kbs, len(g.layer_names), type(spec).__name__))
        print("kv groups (block_size, kernel_block_size, n_layers, spec):", self.groups, "num_blocks", kcc.num_blocks,
              flush=True)
        self.capture_sizes = sorted(self.vllm_config.compilation_config.cudagraph_capture_sizes or [])
        self.emb_buf = self.runner.model_state._inputs_embeds_buf
        self.pos_buf = self.runner.input_buffers.positions
        print("graphs:", graphs, "capture sizes:", self.capture_sizes[:6], "...", self.capture_sizes[-3:], flush=True)

    def _tables(self, T):
        """Block table (2 identical rows, kernel-block ids) and slot mapping (T,) per KV group, using KV blocks 1..n."""
        bts, sms = [], []
        for bs, kbs, _, _ in self.groups:
            per = bs // kbs
            n_kv = math.ceil(T / bs)
            kv_ids = torch.arange(1, n_kv + 1, device=self.device, dtype=torch.int32)
            kernel_ids = (kv_ids[:, None] * per + torch.arange(per, device=self.device, dtype=torch.int32)[None]).flatten()
            bt = kernel_ids[None].repeat(2, 1).contiguous()
            t = torch.arange(T, device=self.device, dtype=torch.int64)
            sm = kernel_ids.long()[t // kbs] * kbs + t % kbs
            bts.append(bt)
            sms.append(sm)
        return bts, sms

    def _tables3(self, T, P):
        """3 rows: prompt & canvas share KV blocks 1..; the dummy pad row uses blocks after a gap. Slot mapping over
        T real tokens then P pad tokens."""
        bts, sms = [], []
        for bs, kbs, _, _ in self.groups:
            per = bs // kbs
            n_real, n_pad = math.ceil(T / bs), max(1, math.ceil(P / bs))
            width = max(n_real, n_pad) * per
            def kids(first_kv, n):
                kv = torch.arange(first_kv, first_kv + n, device=self.device, dtype=torch.int32)
                return (kv[:, None] * per + torch.arange(per, device=self.device, dtype=torch.int32)[None]).flatten()
            real, pad = kids(1, n_real), kids(1 + n_real + 8, n_pad)
            bt = torch.zeros((3, width), dtype=torch.int32, device=self.device)
            bt[0, :real.numel()] = real
            bt[1, :real.numel()] = real
            bt[2, :pad.numel()] = pad
            t = torch.arange(T, device=self.device, dtype=torch.int64)
            q = torch.arange(P, device=self.device, dtype=torch.int64)
            sm = torch.cat([real.long()[t // kbs] * kbs + t % kbs, pad.long()[q // kbs] * kbs + q % kbs])
            bts.append(bt)
            sms.append(sm)
        return bts, sms

    @torch.inference_mode()
    def read_graph(self, prompt_ids, canvas_ids, slot_positions):
        """Same single pass, run through vLLM's compiled piecewise CUDA graphs (tokens padded to a capture size)."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        L, C = len(prompt_ids), len(canvas_ids)
        T = L + C
        Tp = next((c for c in self.capture_sizes if c >= T + 1), None)
        if Tp is None:
            raise ValueError(f"{T} tokens exceed the largest captured size")
        P = Tp - T
        ids = torch.tensor(prompt_ids + canvas_ids, device=self.device)
        emb = self.model.embed_input_ids(ids)
        cv = emb[L:]
        self.emb_buf[:L].copy_(emb[:L])
        self.emb_buf[L:T].copy_(self.model.self_conditioning(cv, torch.zeros_like(cv)))
        self.emb_buf[T:Tp].zero_()
        self.pos_buf[:T].copy_(torch.arange(T, device=self.device))
        self.pos_buf[T:Tp].copy_(torch.arange(P, device=self.device))
        qsl_cpu = torch.tensor([0, L, T, Tp], dtype=torch.int32)
        bts, sms = self._tables3(T, P)
        md = build_attn_metadata(
            attn_groups=self.runner.attn_groups, num_reqs=3, num_tokens=Tp,
            query_start_loc_gpu=qsl_cpu.to(self.device), query_start_loc_cpu=qsl_cpu, max_query_len=max(L, C, P),
            seq_lens=torch.tensor([L, T, P], dtype=torch.int32, device=self.device), max_seq_len=max(T, P),
            block_tables=bts, slot_mappings=sms, kv_cache_config=self.runner.kv_cache_config,
            causal=torch.tensor([True, False, True], device=self.device))
        by_layer = build_slot_mappings_by_layer(sms, self.runner.kv_cache_config)
        with set_forward_context(md, self.vllm_config, num_tokens=Tp, slot_mapping=by_layer,
                                 cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                                 batch_descriptor=BatchDescriptor(num_tokens=Tp)):
            hidden = self.model(input_ids=None, positions=self.pos_buf[:Tp], inputs_embeds=self.emb_buf[:Tp],
                                intermediate_tensors=None)
        rows = torch.tensor([L + p for p in slot_positions], device=self.device)
        logits = self.model.compute_logits(hidden[rows])
        return torch.log_softmax(logits.float(), -1)

    # ------------------------------------------------------------------ full CUDA graph per size bucket
    def _bucket(self, Tp):
        """Persistent inputs + metadata + captured graph for a padded token count Tp (3 pseudo-sequences)."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        dev = self.device
        b = {"Tp": Tp}
        b["qsl"] = torch.zeros(4, dtype=torch.int32, device=dev)
        b["seq_lens"] = torch.zeros(3, dtype=torch.int32, device=dev)
        b["causal"] = torch.tensor([True, False, True], device=dev)
        b["bts"], b["sms"] = [], []
        for bs, kbs, _, _ in self.groups:
            per = bs // kbs
            width = (math.ceil(Tp / bs) + 1) * per
            b["bts"].append(torch.zeros((3, width), dtype=torch.int32, device=dev))
            b["sms"].append(torch.zeros(Tp, dtype=torch.int64, device=dev))
        # a valid layout for capture: prompt Tp-65, canvas 64, pad 1
        self._fill(b, Tp - 65, 64)
        qsl_cpu = torch.tensor([0, Tp - 65, Tp - 1, Tp], dtype=torch.int32)
        md = build_attn_metadata(
            attn_groups=self.runner.attn_groups, num_reqs=3, num_tokens=Tp, query_start_loc_gpu=b["qsl"],
            query_start_loc_cpu=qsl_cpu, max_query_len=Tp, seq_lens=b["seq_lens"], max_seq_len=Tp,
            block_tables=b["bts"], slot_mappings=b["sms"], kv_cache_config=self.runner.kv_cache_config,
            causal=b["causal"])
        by_layer = build_slot_mappings_by_layer(b["sms"], self.runner.kv_cache_config)
        pos, emb = self.pos_buf[:Tp], self.emb_buf[:Tp]

        def fwd():
            with set_forward_context(md, self.vllm_config, num_tokens=Tp, slot_mapping=by_layer,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE):
                return self.model(input_ids=None, positions=pos, inputs_embeds=emb, intermediate_tensors=None)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fwd()                       # warm up (Triton JIT etc.) outside capture
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            b["out"] = fwd()
        b["graph"] = g
        return b

    def _fill(self, b, L, C):
        """Write the per-request layout (prompt L, canvas C, pad rest) into bucket b's persistent tensors."""
        Tp = b["Tp"]
        T = L + C
        P = Tp - T
        dev = self.device
        b["qsl"].copy_(torch.tensor([0, L, T, Tp], dtype=torch.int32), non_blocking=True)
        b["seq_lens"].copy_(torch.tensor([L, T, P], dtype=torch.int32), non_blocking=True)
        for gi, (bs, kbs, _, _) in enumerate(self.groups):
            per = bs // kbs
            n_real, n_pad = math.ceil(T / bs), max(1, math.ceil(P / bs))
            kv = torch.arange(1, 1 + n_real, device=dev, dtype=torch.int32)
            real = (kv[:, None] * per + torch.arange(per, device=dev, dtype=torch.int32)[None]).flatten()
            kvp = torch.arange(1 + n_real + 8, 1 + n_real + 8 + n_pad, device=dev, dtype=torch.int32)
            pad = (kvp[:, None] * per + torch.arange(per, device=dev, dtype=torch.int32)[None]).flatten()
            bt = b["bts"][gi]
            bt.zero_()
            bt[0, :real.numel()] = real
            bt[1, :real.numel()] = real
            bt[2, :pad.numel()] = pad
            t = torch.arange(T, device=dev, dtype=torch.int64)
            q = torch.arange(P, device=dev, dtype=torch.int64)
            b["sms"][gi].copy_(torch.cat([real.long()[t // kbs] * kbs + t % kbs, pad.long()[q // kbs] * kbs + q % kbs]))

    def setup_full(self, buckets):
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = {}
        with torch.inference_mode():
            for Tp in sorted(buckets):
                self.buckets[Tp] = self._bucket(Tp)
        print("captured full graphs for", sorted(self.buckets), flush=True)

    @torch.inference_mode()
    def read_full(self, prompt_ids, canvas_ids, slot_positions):
        L, C = len(prompt_ids), len(canvas_ids)
        T = L + C
        Tp = next((x for x in sorted(self.buckets) if x >= T + 1), None)
        if Tp is None:
            raise ValueError(f"{T} tokens exceed the largest bucket")
        b = self.buckets[Tp]
        ids = torch.tensor(prompt_ids + canvas_ids, device=self.device)
        emb = self.model.embed_input_ids(ids)
        cv = emb[L:]
        self.emb_buf[:L].copy_(emb[:L])
        self.emb_buf[L:T].copy_(self.model.self_conditioning(cv, torch.zeros_like(cv)))
        self.emb_buf[T:Tp].zero_()
        self.pos_buf[:T].copy_(torch.arange(T, device=self.device))
        self.pos_buf[T:Tp].copy_(torch.arange(Tp - T, device=self.device))
        self._fill(b, L, C)
        b["graph"].replay()
        rows = torch.tensor([L + p for p in slot_positions], device=self.device)
        logits = self.model.compute_logits(b["out"][rows])
        return torch.log_softmax(logits.float(), -1)

    @torch.inference_mode()
    def read(self, prompt_ids, canvas_ids, slot_positions):
        if getattr(self, "buckets", None):
            return self.read_full(prompt_ids, canvas_ids, slot_positions)
        if self.graphs:
            return self.read_graph(prompt_ids, canvas_ids, slot_positions)
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        L, C = len(prompt_ids), len(canvas_ids)
        T = L + C
        ids = torch.tensor(prompt_ids + canvas_ids, device=self.device)
        emb = self.model.embed_input_ids(ids)
        cv = emb[L:]
        emb = torch.cat([emb[:L], self.model.self_conditioning(cv, torch.zeros_like(cv))], dim=0)
        pos = torch.arange(T, device=self.device, dtype=torch.int64)
        qsl_cpu = torch.tensor([0, L, T], dtype=torch.int32)
        bts, sms = self._tables(T)
        md = build_attn_metadata(
            attn_groups=self.runner.attn_groups, num_reqs=2, num_tokens=T,
            query_start_loc_gpu=qsl_cpu.to(self.device), query_start_loc_cpu=qsl_cpu, max_query_len=max(L, C),
            seq_lens=torch.tensor([L, T], dtype=torch.int32, device=self.device), max_seq_len=T,
            block_tables=bts, slot_mappings=sms, kv_cache_config=self.runner.kv_cache_config,
            causal=torch.tensor([True, False], device=self.device))
        by_layer = build_slot_mappings_by_layer(sms, self.runner.kv_cache_config)
        with set_forward_context(md, self.vllm_config, num_tokens=T, slot_mapping=by_layer):
            hidden = self.model(input_ids=None, positions=pos, inputs_embeds=emb, intermediate_tensors=None)
        rows = torch.tensor([L + p for p in slot_positions], device=self.device)
        logits = self.model.compute_logits(hidden[rows])
        return torch.log_softmax(logits.float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-port", type=int, default=8767)
    ap.add_argument("--graphs", action="store_true", help="use vLLM's compiled piecewise CUDA graphs")
    ap.add_argument("--full", action="store_true", help="capture our own full CUDA graph per size bucket")
    ap.add_argument("--profile", action="store_true", help="print the top CUDA kernels of one forward")
    ap.add_argument("--moe", default="triton")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    fp = FastPath(moe=a.moe, graphs=a.graphs or a.full)
    if a.full:
        fp.setup_full([256, 320, 384, 448, 512])

    states = S.validate_request(EXAMPLE) + [
        {"id": "arith", "state": "x = 7, y = 12", "questions": {
            "bigger": {"type": "choice", "instructions": "Which variable is larger?", "criteria": {"x": "the variable x", "y": "the variable y"}},
            "x_odd": {"type": "boolean", "instructions": "Is x an odd number?"},
            "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]}}}]

    # reference: the live two-step server
    import httpx
    key = open(os.path.join(os.path.dirname(HERE), ".djev-api-key")).read().strip()
    ref = httpx.post(f"http://127.0.0.1:{a.ref_port}/api/evaluate", json={"states": states},
                     headers={"Authorization": "Bearer " + key}, timeout=120).json()
    ref = {s["id"]: s["answers"] for s in ref["states"]}

    worst = 0.0
    prepared = []
    for st in states:
        text, slots = enc.prompt_text(st)
        body = enc.scaffold(slots, st["id"])
        canvas = body + [S.PAD_ID] * (CANVAS - len(body))
        pids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                       return_dict=False)
        pids = (pids["input_ids"] if isinstance(pids, dict) else list(pids)) + list(enc.thought_prefix_ids)
        prepared.append((st, slots, pids, canvas))
        lp = fp.read(pids, canvas, [s.pos for s in slots])
        print(f"== {st['id']}  prompt {len(pids)} tok")
        for i, s in enumerate(slots):
            p = [float(lp[i, g].exp().sum()) for g in s.cand_groups]
            mass = sum(p)
            n = [x / mass for x in p]
            r = [ref[st["id"]][s.qid]["probabilities"][k] for k in s.keys]
            d = max(abs(x - y) for x, y in zip(n, r))
            worst = max(worst, d)
            print(f"   {s.qid:11s} fast {[round(x, 3) for x in n]} mass {mass:.3f} | served {[round(x, 3) for x in r]} "
                  f"mass {ref[st['id']][s.qid]['candidate_mass']:.3f} | max|diff| {d:.3f}")
    print(f"WORST max|diff| vs served two-step: {worst:.3f}", flush=True)

    # timing (eager, no CUDA graphs yet)
    st, slots, pids, canvas = prepared[0]
    for _ in range(5):
        fp.read(pids, canvas, [s.pos for s in slots])
    torch.cuda.synchronize()
    xs = []
    for _ in range(30):
        t = time.perf_counter()
        fp.read(pids, canvas, [s.pos for s in slots])
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1000)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(10):
        fp.read(pids, canvas, [s.pos for s in slots])
    ev1.record()
    torch.cuda.synchronize()
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                fp.read(pids, canvas, [s.pos for s in slots])
            torch.cuda.synchronize()
        ev = prof.key_averages()
        tot = sum(e.device_time_total for e in ev) / 5
        print(f"PROFILE per forward: total kernel time {tot/1000:.2f} ms")
        for e in sorted(ev, key=lambda e: -e.device_time_total)[:14]:
            print(f"PROFILE {e.device_time_total/5/1000:6.2f} ms  {e.device_time_total/5/tot*100:5.1f}%  x{e.count//5:4d}  {e.key[:90]}")
    print(f"TIMING {'full-graph' if a.full else 'graphs' if a.graphs else 'eager'} single pass (prompt {len(pids)} + canvas {CANVAS}): wall p50 {statistics.median(xs):.2f} ms, "
          f"min {min(xs):.2f} ms; GPU-event avg {ev0.elapsed_time(ev1) / 10:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
