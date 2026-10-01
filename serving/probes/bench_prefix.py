#!/usr/bin/env python3
"""The fast path's prefix cache on real agent states (a JSON list of {id, state, questions}, in the agent's order):
answers with and without the cache, read time, and a kernel-time profile of one forward.

    python probes/bench_prefix.py [--states ~/tmp/djev/v10_states.json] [--n 120] [--profile]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np  # noqa: E402

import server as S  # noqa: E402


def category(name):
    n = name.lower()
    if "moe" in n or "expert" in n or "groupproblemshape" in n or "expandinputrows" in n or "doactivation" in n:
        return "moe"
    if "attention" in n or "attn" in n or "flash" in n or "reduce_segments" in n:
        return "attention"
    if "gemm" in n or "nvjet" in n or "cublas" in n or "cutlass" in n:
        return "gemm (dense)"
    if "triton_" in n:
        return "inductor (norms, rope, act)"
    return "other"


def profile(fr, g, label):
    import torch
    from torch.profiler import ProfilerActivity, profile as tprofile
    torch.cuda.synchronize()
    with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            g["graph"].replay()
        torch.cuda.synchronize()
    per_name, per_cat = defaultdict(float), defaultdict(float)
    for ev in prof.events():
        if ev.device_type.name != "CUDA":
            continue
        us = ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
        per_name[ev.name] += us / 3
        per_cat[category(ev.name)] += us / 3
    total = sum(per_cat.values())
    print(f"PROFILE {label}: GPU kernel time {total / 1000:.2f} ms per forward", flush=True)
    for c, us in sorted(per_cat.items(), key=lambda x: -x[1]):
        print(f"   {c:28s} {us / 1000:7.2f} ms  {100 * us / total:5.1f}%")
    for n, us in sorted(per_name.items(), key=lambda x: -x[1])[:14]:
        print(f"      {us / 1000:7.3f} ms  {n[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.expanduser("~/tmp/djev/v10_states.json"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (vLLM online quantization)")
    ap.add_argument("--save", help="save the no-cache Triton answers here (.npy)")
    ap.add_argument("--against", help="also compare with answers saved by --save (e.g. a bf16 run)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import fastpath
    tk = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tk, 1 << 30)
    enc.label_groups = [g[:1] for g in enc.label_groups]           # --ids main, as served
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    t0 = time.time()
    fr = fastpath.FastReader(moe=a.moe, thought_ids=enc.thought_prefix_ids, quantization=a.quantization)
    print(f"FastReader ready in {time.time() - t0:.0f}s: {len(fr.graphs)} graphs, {fr.NS} prefix slots, "
          f"align {fr.align}, groups {fr.groups}", flush=True)
    chat_head = fr.chat_head
    chat_tail = fr.chat_tail + list(enc.thought_prefix_ids)
    states = json.load(open(a.states))[:a.n]
    reads = []
    for st in states:
        text, slots = enc.prompt_text(st, 0)
        body = enc.scaffold(slots, st["id"])
        canvas = body + [S.PAD_ID] * (fr.C - len(body))
        cands = [[i for g in s.cand_groups for i in g] for s in slots]
        pids = chat_head + tk.encode(text, add_special_tokens=False) + chat_tail
        reads.append((pids, canvas, [s.pos for s in slots], cands))
    print(f"{len(reads)} reads, prompt tokens p50 {statistics.median(len(r[0]) for r in reads)}", flush=True)

    def probs(lp):
        p = np.exp(np.asarray(lp[0]))
        return p / p.sum()

    def fresh():
        fr.slot_keys = [np.zeros(0, dtype=np.int64) for _ in range(fr.NS)]

    def run(label, fn, rs):
        out, ts = [], []
        for r in rs[:3]:
            fn(r)
        for r in rs:
            t = time.perf_counter()
            lp = fn(r)
            ts.append((time.perf_counter() - t) * 1000)
            out.append(probs(lp))
        print(f"TIMING {label:34s} p50 {statistics.median(ts):6.2f} ms  min {min(ts):6.2f}", flush=True)
        return out

    def diff(label, x, y):
        d = sorted(float(np.abs(u - w).max()) for u, w in zip(x, y))
        agree = sum(int(np.argmax(u) == np.argmax(w)) for u, w in zip(x, y))
        print(f"ANSWERS {label:40s} max|dp| p50 {d[len(d) // 2]:.4f} p90 {d[int(len(d) * .9)]:.4f} max {d[-1]:.4f}; "
              f"argmax agree {agree}/{len(d)}", flush=True)

    def nocache(variant):
        def f(r):
            fresh()
            return fr.read_debug(*r, variant=variant)
        return f

    ref = run("triton, no cache (= production)", nocache("triton"), reads)
    orig = fr.ibuckets

    def up(r):                                   # the same, one bucket up: other GEMM shapes, same math
        T = len(r[0]) + len(r[1])
        need = next(b for b in orig if b >= T + 1)
        fr.ibuckets = [b for b in orig if b > need]
        try:
            return nocache("triton")(r)
        finally:
            fr.ibuckets = orig
    up_ = run("triton, no cache, next bucket up", up, reads[:60])
    fa = run("FA2, no cache", nocache("graph"), reads)
    fresh()
    cached, computed = [], []

    def seq(r):
        lp = fr.read(*r)
        cached.append(fr.last["cached"])
        computed.append(fr.last["computed"])
        return lp
    fac = run("FA2 + prefix cache (agent order)", seq, reads)
    print(f"      cached tokens p50 {statistics.median(cached)}, computed p50 {statistics.median(computed)}", flush=True)
    diff("noise floor: triton next bucket up", ref[:60], up_)
    diff("FA2 vs triton (no cache)", ref, fa)
    diff("FA2 + cache vs triton (no cache)", ref, fac)
    if all("_recorded" in st for st in states):     # answers a server gave for these states (from the agent's log)
        rec = [np.asarray(st["_recorded"]) for st in states]
        diff("recorded (server at the time) vs triton no cache", rec, ref)
        diff("recorded (server at the time) vs FA2 + cache", rec, fac)
        tv = lambda x, y: statistics.median(0.5 * float(np.abs(u - w).sum()) for u, w in zip(x, y))
        print(f"TV distance p50: recorded~triton {tv(rec, ref):.3f}, recorded~FA2+cache {tv(rec, fac):.3f}, "
              f"triton~FA2+cache {tv(ref, fac):.3f}", flush=True)
    if a.save:
        np.save(a.save + "_ref.npy", np.array(ref, dtype=object), allow_pickle=True)
        np.save(a.save + "_fac.npy", np.array(fac, dtype=object), allow_pickle=True)
    if a.against:
        other = list(np.load(a.against + "_ref.npy", allow_pickle=True))
        diff(f"triton no cache vs {os.path.basename(a.against)}", other, ref)
        diff(f"FA2 + cache vs {os.path.basename(a.against)}", other, fac)

    fresh()
    r = reads[len(reads) // 2]
    fr.read(*r)
    xs = []
    for _ in range(20):
        t = time.perf_counter()
        fr.read(*r)
        xs.append((time.perf_counter() - t) * 1000)
    print(f"TIMING same prompt again: p50 {statistics.median(xs):.2f} ms, computed {fr.last['computed']} tokens", flush=True)
    fresh()
    half = len(reads) // 2
    ys = []
    for i in range(half):
        for r in (reads[i], reads[half + i]):
            t = time.perf_counter()
            fr.read(*r)
            ys.append((time.perf_counter() - t) * 1000)
    print(f"TIMING two interleaved streams: p50 {statistics.median(ys):.2f} ms", flush=True)
    short = [(r[0][:40] + r[0][-12:], *r[1:]) for r in reads[:30]]
    run("short prompt (52 tok), FA2", nocache("graph"), short)
    run("short prompt (52 tok), triton", nocache("triton"), short)

    if a.profile:
        fresh()
        r = reads[len(reads) // 2]
        fr.read_debug(*r, variant="triton")
        T = len(r[0]) + len(r[1])
        profile(fr, fr.igraphs[next(b for b in fr.ibuckets if b >= T + 1)], f"triton, no cache (prompt {len(r[0])})")
        fr.read(*r)
        fr.read(*reads[len(reads) // 2 + 1])
        profile(fr, fr.graphs[fr.last["bucket"]], f"FA2 + cache, bucket {fr.last['bucket']} (computed {fr.last['computed']})")
        profile(fr, fr.graphs[min(fr.graphs)], f"FA2 smallest bucket {min(fr.graphs)}")

if __name__ == "__main__":
    main()
