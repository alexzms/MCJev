#!/usr/bin/env python3
"""vLLM's Triton unified attention on the fast path's layout (prompt causal, canvas non-causal, padding), for
DiffusionGemma's two attention shapes, with the launch config swept. No model load: synthetic q and paged KV.

    python probes/attn_bench.py [--tokens 1600] [--prefix 0]
"""
import argparse
import inspect
import itertools
import math
import os

import torch

import vllm.v1.attention.ops.triton_unified_attention as tua

# ---- a launch-config override injected into vLLM's launcher (the kernel itself is unchanged)
OVERRIDE = {}
_src = inspect.getsource(tua.unified_attention)
_hook = """    _ov = _DJEV_OVERRIDE.get(head_size) if max_seqlen_q > 1 else None
    if _ov:
        BLOCK_M = _ov["BLOCK_M"]
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        grid = (q.shape[0] // BLOCK_Q + num_seqs, num_kv_heads)
        tile_size = _ov["TILE"]
        launch_kwargs = {"num_warps": _ov["warps"], "num_stages": _ov["stages"]}
    kernel_unified_attention[grid](
"""
assert _src.count("    kernel_unified_attention[grid](\n") == 1
_src = _src.replace("    kernel_unified_attention[grid](\n", _hook)
tua._DJEV_OVERRIDE = OVERRIDE
exec(compile(_src, tua.__file__, "exec"), tua.__dict__)

SHAPES = {   # name: (q heads, kv heads, head size, block size, sliding window)
    "sliding": (16, 8, 256, 16, 1024),
    "global": (16, 2, 512, 32, 0),
}


def build(shape, Lnew, P0, C, Tp, dev):
    Hq, Hkv, D, bs, sw = shape
    L = P0 + Lnew
    P = Tp - Lnew - C
    nblk = math.ceil((L + C) / bs) + math.ceil(P / bs) + 4
    kv = torch.randn(nblk, Hkv, bs, 2 * D, device=dev, dtype=torch.bfloat16) * 0.5
    kv = kv.transpose(1, 2)
    k, v = kv.split(D, dim=-1)
    q = torch.randn(Tp, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5
    out = torch.empty_like(q)
    w = math.ceil((L + C) / bs) + 1
    bt = torch.zeros(3, w, dtype=torch.int32, device=dev)
    real = torch.arange(1, 1 + math.ceil((L + C) / bs), dtype=torch.int32, device=dev)
    bt[0, :real.numel()] = real
    bt[1, :real.numel()] = real
    pad = torch.arange(1 + real.numel(), 1 + real.numel() + math.ceil(P / bs), dtype=torch.int32, device=dev)
    bt[2, :pad.numel()] = pad
    cu = torch.tensor([0, Lnew, Lnew + C, Tp], dtype=torch.int32, device=dev)
    seqused = torch.tensor([L, L + C, P], dtype=torch.int32, device=dev)
    causal = torch.tensor([True, False, True], device=dev)
    rng = torch.zeros(3, 8, 2, dtype=torch.int32, device=dev)
    one = torch.ones(1, device=dev, dtype=torch.float32).expand(3, Hkv)
    kw = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=Tp, seqused_k=seqused, max_seqlen_k=L + C,
              softmax_scale=1.0, causal=causal, window_size=(sw - 1, 0) if sw else (-1, -1), block_table=bt,
              softcap=0, q_descale=None, k_descale=one, v_descale=one, mm_prefix_range=rng,
              mm_prefix_clamp_sliding_window=bool(sw))
    return kw


def timed(kw, reps=30):
    for _ in range(3):
        tua.unified_attention(**kw)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        tua.unified_attention(**kw)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1600, help="prompt tokens computed in the forward")
    ap.add_argument("--prefix", type=int, default=0, help="cached prompt tokens before them")
    ap.add_argument("--canvas", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    Tp = a.tokens + a.canvas + 64
    print(f"layout: {a.prefix} cached + {a.tokens} prompt + {a.canvas} canvas, bucket {Tp}", flush=True)
    for name, shape in SHAPES.items():
        D = shape[2]
        kw = build(shape, a.tokens, a.prefix, a.canvas, Tp, dev)
        OVERRIDE.clear()
        base = timed(kw)
        ref = kw["out"].clone()
        print(f"{name} (head {D}, {shape[0]}q/{shape[1]}kv): vLLM default {base * 1000:.0f} us per layer", flush=True)
        res = []
        for bm, tile, warps, stages in itertools.product((16, 32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
            if bm // (shape[0] // shape[1]) < 1:
                continue
            if D == 512 and (bm * tile > 64 * 64 or tile > 64):
                continue
            OVERRIDE.clear()
            OVERRIDE[D] = {"BLOCK_M": bm, "TILE": tile, "warps": warps, "stages": stages}
            try:
                t = timed(kw, reps=10)
                err = (kw["out"].float() - ref.float()).abs().max().item()
            except Exception as exc:   # out of shared memory / registers
                res.append((float("inf"), bm, tile, warps, stages, type(exc).__name__))
                continue
            res.append((t, bm, tile, warps, stages, f"max|diff| {err:.3g}"))
        res.sort()
        for t, bm, tile, warps, stages, note in res[:6]:
            print(f"   {t * 1000:7.0f} us  BLOCK_M {bm:3d} TILE {tile:3d} warps {warps} stages {stages}  {note}", flush=True)
        OVERRIDE.clear()
        # reference: dense causal SDPA over the prompt alone (what a fused FMHA achieves on this GPU)
        Hq, Hkv, D, bs, sw = shape
        qq = torch.randn(1, Hq, a.tokens, D, device=dev, dtype=torch.bfloat16)
        kk = torch.randn(1, Hq, a.prefix + a.tokens, D, device=dev, dtype=torch.bfloat16)
        try:
            f = lambda: torch.nn.functional.scaled_dot_product_attention(qq, kk, kk, is_causal=a.prefix == 0)
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                f()
            e1.record()
            torch.cuda.synchronize()
            print(f"   reference torch SDPA (dense, causal prompt only): {e0.elapsed_time(e1) / 20 * 1000:.0f} us", flush=True)
        except Exception as exc:
            print(f"   reference torch SDPA failed: {type(exc).__name__}: {str(exc)[:100]}")


if __name__ == "__main__":
    main()
