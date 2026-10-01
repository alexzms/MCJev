#!/usr/bin/env python3
"""Answers of one engine configuration on a fixed set of real p_uhc_pro_v10 steps, to compare configurations that
cannot share a process (another checkpoint, another MoE backend).

    python probes/answers_dump.py --out bf16.json [--model ...] [--moe ...] [--n 200]
    python probes/answers_dump.py --compare bf16.json nvfp4.json

The steps are seeded (probes/v10.py, --seed), so every run reads the same ones. Each is read with no cached prefix
and again with its previous step cached (as in a match); --compare prints, per pair of runs, how often the chosen
option is the same and max |dp| over the options, next to the same numbers between one run's two readings (that
configuration's own rounding noise).
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def probs(logps):
    p = [math.exp(x) for x in logps]
    return [x / sum(p) for x in p]


def compare(a, b):
    same, dps = 0, []
    for x, y in zip(a, b):
        px, py = probs(x), probs(y)
        same += max(range(len(px)), key=px.__getitem__) == max(range(len(py)), key=py.__getitem__)
        dps.append(max(abs(u - v) for u, v in zip(px, py)))
    return f"same choice {same}/{len(dps)} ({100 * same / len(dps):.1f}%), max |dp| median {statistics.median(dps):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fa", type=int, default=2, help="sliding-layer FlashAttention (4: needs --kv-block 128)")
    ap.add_argument("--kv-block", type=int, default=None, help="vLLM KV block size (FA4 needs 128)")
    ap.add_argument("--kv-dtype", default=None, help="KV cache dtype (bfloat16 for an FP8-KV checkpoint)")
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        print(f"{x['config']}: its own noise (cached vs no prefix): {compare(x['cold'], x['warm'])}")
        print(f"{y['config']}: its own noise (cached vs no prefix): {compare(y['cold'], y['warm'])}")
        print(f"{x['config']} vs {y['config']}, no prefix: {compare(x['cold'], y['cold'])}")
        print(f"{x['config']} vs {y['config']}, cached:    {compare(x['warm'], y['warm'])}")
        return
    import fastpath
    import v10
    tok, enc = v10.encoder()
    fr = fastpath.FastReader(fa_version=a.fa, kv_block_size=a.kv_block, kv_cache_dtype=a.kv_dtype,
                             model=a.model or fastpath.MODEL, canvas=64, moe=a.moe, gpu_memory_utilization=0.85,
                             max_model_len=4096, thought_ids=list(enc.thought_prefix_ids), prefix_slots=8,
                             fast_attention=True, quantization=a.quantization)
    step = v10.stepper(tok, enc)
    random.seed(a.seed)
    bots = list(range(1, 9))
    reads = [step(bots[i % len(bots)]) for i in range(a.n + len(bots))]
    cold, warm = [], []
    for i, rd in enumerate(reads):
        if i >= len(bots):                      # the first step of each bot only fills its slot
            cold.append(fr.read_debug(*rd)[0])
            prev = reads[i - len(bots)]
            fr.read(*prev)                      # its previous step, then this one with that prefix cached
            warm.append(fr.read(*rd)[0])
    config = (f"{os.path.basename(a.model or 'bf16')}/{a.moe}" + (f"/{a.quantization}" if a.quantization else "")
              + (f"/fa{a.fa}" if a.fa != 2 else "") + (f"/kv{a.kv_block}" if a.kv_block else ""))
    json.dump({"config": config, "cold": cold, "warm": warm}, open(a.out, "w"))
    print(f"wrote {len(cold)} answers of {config} to {a.out}")


if __name__ == "__main__":
    main()
