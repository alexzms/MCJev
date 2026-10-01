#!/usr/bin/env python3
"""Batched text reads (FastReader.read_many) against one read per forward, on real p_uhc_pro_v10 steps.

    python probes/check_batch.py [--bots 8] [--batch-max 4] [--rounds 20]

Each bot (Jev1..JevN: its own name, so bots share no prompt prefix) sends a step that changes only from the "Now:" line
on, as in a match; its previous step stays in a prefix slot. Per round every bot sends one step:
  * same answers: against the same read with no cached prefix (read_debug), the option probabilities of one read
    per forward with a cached prefix (today's engine: its own rounding noise) and of batches of r reads (max |dp|
    over the options, and whether the chosen option is the same);
  * speed: the whole round one read per forward, then in batches of r = 2..batch_max.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import v10  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default=v10.REQUEST)
    ap.add_argument("--bots", type=int, default=8)
    ap.add_argument("--batch-max", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--no-compare", action="store_true", help="speed only")
    ap.add_argument("--mem", type=float, default=0.85, help="gpu_memory_utilization (big batches need room)")
    ap.add_argument("--fa", type=int, default=2, help="sliding-layer FlashAttention (4: needs --kv-block 128)")
    ap.add_argument("--kv-block", type=int, default=None, help="vLLM KV block size (FA4 needs 128)")
    ap.add_argument("--kv-dtype", default=None, help="KV cache dtype (bfloat16 for an FP8-KV checkpoint)")
    a = ap.parse_args()
    import torch
    import fastpath
    tok, enc = v10.encoder()
    fr = fastpath.FastReader(fa_version=a.fa, kv_block_size=a.kv_block, kv_cache_dtype=a.kv_dtype,
                             canvas=64, moe="flashinfer_cutlass", gpu_memory_utilization=a.mem, max_model_len=4096,
                             thought_ids=list(enc.thought_prefix_ids), prefix_slots=8, fast_attention=True,
                             batch_max=a.batch_max)
    print(json.dumps({"load_seconds": round(fr.load_seconds, 1), "batched graphs": len(fr.mgraphs)}), flush=True)
    step = v10.stepper(tok, enc, a.request)

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = fn()
        torch.cuda.synchronize()
        return res, (time.perf_counter() - t) * 1000

    import math
    agree = {}                                       # mode -> [max |dp| over the options, same choice]

    def compare(mode, ref, got):
        for lr, lg in zip(ref, got):
            pr = [math.exp(x) for x in lr[0]]
            pg = [math.exp(x) for x in lg[0]]
            pr, pg = [x / sum(pr) for x in pr], [x / sum(pg) for x in pg]
            top = lambda p: max(range(len(p)), key=p.__getitem__)
            agree.setdefault(mode, []).append((max(abs(x - y) for x, y in zip(pr, pg)), top(pr) == top(pg)))

    bots = list(range(1, a.bots + 1))
    for b in bots:                                   # every bot's previous step in a prefix slot
        fr.read(*step(b))
    t_single, t_batch = [], {r: [] for r in range(2, a.batch_max + 1)}
    cached = []
    for rnd in range(a.rounds + 2):
        # speed: every mode gets a fresh step from every bot, whose previous step (same rules, other numbers) is
        # in a slot - as in a match
        reads = [step(b) for b in bots]
        got, ms = timed(lambda: [(fr.read(*rd), fr.last["cached"]) for rd in reads])
        if rnd >= 2:
            t_single.append(ms)
            cached += [c for _, c in got]
        for r in range(2, a.batch_max + 1):
            reads = [step(b) for b in bots]
            _, ms = timed(lambda: [x for i in range(0, len(reads), r) for x in fr.read_many(reads[i:i + r])])
            if rnd >= 2:
                t_batch[r].append(ms)
        if a.no_compare:
            continue
        # same answers: identical reads against a reference with no cached prefix; one per forward with a cached
        # prefix (what the engine does today: its own rounding noise) and in batches
        reads = [step(b) for b in bots]
        cold = [fr.read_debug(*rd) for rd in reads]
        warm = [fr.read(*rd) for rd in reads]
        compare("one per forward, cached prefix", cold, warm)
        for r in range(2, a.batch_max + 1):
            got = [x[0] for i in range(0, len(reads), r) for x in fr.read_many(reads[i:i + r])]
            compare(f"batches of {r}", cold, got)
    print("against a read with no cached prefix (option probabilities, normalized over the options):")
    for mode, xs in agree.items():
        dps = [d for d, _ in xs]
        print(f"  {mode:32s} same choice {sum(s for _, s in xs)}/{len(xs)} | max |dp| median {statistics.median(dps):.4f}"
              f" p90 {sorted(dps)[int(0.9 * (len(dps) - 1))]:.4f} max {max(dps):.4f}")
    print(f"cached prompt tokens per read: mean {statistics.mean(cached):.0f} | prompt tokens {len(step(1)[0])}")
    s = statistics.median(t_single)
    print(f"{a.bots} bots, one step each: one read per forward {s:7.1f} ms ({s / a.bots:5.1f} ms/read)")
    for r, ts in t_batch.items():
        m = statistics.median(ts)
        print(f"{a.bots} bots, one step each: batches of {r}       {m:7.1f} ms ({m / a.bots:5.1f} ms/read, "
              f"x{s / m:.2f} throughput)")


if __name__ == "__main__":
    main()
