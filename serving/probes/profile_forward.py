#!/usr/bin/env python3
"""Where does one fast-path forward's GPU time go?  One real p_uhc_pro_v10 read (probes/v10.py) with the prefix a bot
has in a match (its previous step cached), alone and in a batch of --batch reads: the CUDA graph's wall time, then
the same forward eager under torch.profiler, its kernels summed by kind (MoE, attention, dense GEMM, other).

    python probes/profile_forward.py [--batch 4]
"""
import argparse
import collections
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import v10  # noqa: E402

KINDS = [("moe", ("moe", "expert", "grouped", "trtllm", "routing", "topk", "permute", "cutlass_fused")),
         ("attention", ("flash", "attn", "attention", "fmha")),
         ("gemm", ("gemm", "nvjet", "cublas", "sm100_xmma", "cutlass", "matmul")),
         ("norm/act/elementwise", ("norm", "elementwise", "vectorized", "gelu", "silu", "act_and_mul", "softmax",
                                   "reduce", "copy", "cat", "index", "gather", "fill", "where", "mul", "add"))]


def kind(name):
    n = name.lower()
    return next((k for k, keys in KINDS if any(x in n for x in keys)), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.85, help="gpu_memory_utilization (big batches need room)")
    a = ap.parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile
    import fastpath
    tok, enc = v10.encoder()
    fr = fastpath.FastReader(canvas=64, moe="flashinfer_cutlass", gpu_memory_utilization=a.mem, max_model_len=4096,
                             thought_ids=list(enc.thought_prefix_ids), prefix_slots=8, fast_attention=True,
                             batch_max=a.batch)
    step = v10.stepper(tok, enc)
    bots = list(range(1, a.batch + 1))
    for b in bots:
        fr.read(*step(b))

    def graph_ms(fn, n=20):
        ts = []
        for _ in range(n + 3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts[3:])

    one = graph_ms(lambda: fr.read(*step(1)))
    print(f"graph, 1 read: {one:.2f} ms (cached {fr.last['cached']}, bucket {fr.last['bucket']})")
    many = graph_ms(lambda: fr.read_many([step(b) for b in bots]))
    print(f"graph, {a.batch} reads in one forward: {many:.2f} ms ({many / a.batch:.2f} ms/read, bucket {fr.last['bucket']})")

    # eager, under the profiler: the text graph's forward for a read with the same shape
    reads = [step(b) for b in bots]
    cu_c1 = fastpath._ATTN["cu_c"]
    for label, rds in (("1 read", reads[:1]), (f"{a.batch} reads", reads)):
        with torch.inference_mode():
            if len(rds) == 1:
                slot, P0 = fr._pick_slot(__import__("numpy").asarray(rds[0][0]))
                new = rds[0][0][P0:]
                Tp = next(b for b in fr.buckets if b >= len(new) + fr.C + 1)
                g = fr.graphs[Tp]
                fr._write(g, slot, P0, new, *rds[0][1:])
                fastpath._ATTN.update(fa=True, r=1, cu_p=g["b32"][7:10], cu_c=cu_c1)
            else:
                items, _, Tp = fr._plan_many(rds)
                g = fr.mgraphs[len(rds), Tp]
                fr._write_many(g, items)
                r = len(rds)
                fastpath._ATTN.update(fa=True, r=r, cu_p=g["b32"][4 * r + 3:5 * r + 5], cu_c=g["cu_c"])
            for _ in range(3):
                g["fwd"]()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g["fwd"]()
                torch.cuda.synchronize()
        fastpath._ATTN["fa"] = False
        tot = collections.Counter()
        per = collections.Counter()
        for e in p.key_averages():
            us = e.self_device_time_total if hasattr(e, "self_device_time_total") else e.self_cuda_time_total
            if us > 0:
                tot[kind(e.key)] += us
                per[e.key] += us
        s = sum(tot.values())
        print(f"== eager kernels, {label} (bucket {Tp}): {s / 1000:.2f} ms of kernel time")
        for k, us in tot.most_common():
            print(f"   {k:22s} {us / 1000:7.2f} ms  {100 * us / s:5.1f}%")
        for name, us in per.most_common(12):
            print(f"      {us / 1000:6.2f} ms  [{kind(name)}] {name[:110]}")


if __name__ == "__main__":
    main()
