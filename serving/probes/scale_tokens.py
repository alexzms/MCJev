#!/usr/bin/env python3
"""How does one fast-path forward's time grow with the tokens it computes?  Text reads with no cached prefix at a
range of prompt lengths (FastReader.read_debug on the FA2 text graphs), in bf16 or with --quantization fp8. If twice
the tokens costs much less than twice the time, several bots' reads in one forward would carry more bots per GPU.

    python probes/scale_tokens.py [--lengths 256 512 1024 1536 2048 2432] [--moe triton] [--quantization fp8]
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import server as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 768, 1024, 1536, 2048, 2432])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    ap.add_argument("--quantization", default=None, help="vLLM online quantization, e.g. fp8 (default: bf16)")
    ap.add_argument("--model", default=None, help="another checkpoint, e.g. nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    ap.add_argument("--fa", type=int, default=2, help="sliding-layer FlashAttention (4: needs --kv-block 128)")
    ap.add_argument("--kv-block", type=int, default=None, help="vLLM KV block size (FA4 needs 128)")
    ap.add_argument("--kv-dtype", default=None, help="KV cache dtype (bfloat16 for an FP8-KV checkpoint)")
    a = ap.parse_args()
    import torch
    import fastpath
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    fr = fastpath.FastReader(fa_version=a.fa, kv_block_size=a.kv_block, kv_cache_dtype=a.kv_dtype,
                             model=a.model or fastpath.MODEL, canvas=64, moe=a.moe, gpu_memory_utilization=0.85,
                             max_model_len=4096, prefix_slots=8, fast_attention=True, quantization=a.quantization)
    print(f"moe {a.moe}, quantization {a.quantization or 'bf16'}, load {fr.load_seconds:.0f} s", flush=True)
    vocab = [i for i in range(1000, 200000) if i < tok.vocab_size]
    canvas = [S.PAD_ID] * 64
    canvas[2] = tok.convert_tokens_to_ids("<mask>")
    cands = [[tok.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCDEFGHIJKLMNOP"]]
    base = None
    print(f"{'prompt tokens':>13s} {'bucket':>6s} {'ms':>7s} {'ms/1k tok':>9s} {'x vs first':>10s}")
    for L in a.lengths:
        ts = []
        for i in range(a.n + 3):
            ids = [random.choice(vocab) for _ in range(L)]
            torch.cuda.synchronize()
            t = time.perf_counter()
            fr.read_debug(ids, canvas, [2], cands, variant="graph")
            ts.append((time.perf_counter() - t) * 1000)
        ms = statistics.median(ts[3:])
        bucket = next(b for b in fr.buckets if b >= L + 64 + 1)
        base = base or ms
        print(f"{L:13d} {bucket:6d} {ms:7.2f} {ms / (L + 64) * 1000:9.2f} {ms / base:10.2f}", flush=True)


if __name__ == "__main__":
    main()
