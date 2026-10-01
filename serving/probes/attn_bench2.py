#!/usr/bin/env python3
"""FlashAttention (vLLM's bundled FA2) instead of Triton unified attention for the fast path's read, per layer.
Layout with the canvas first, so the non-causal part is a static slice:
    tokens = [canvas (C) ; prompt after the cached prefix (Lnew) ; padding]
    FA call 1: canvas queries, non-causal, over the sequence's L + C keys (paged)
    FA call 2: prompt + padding queries as two causal sequences (paged; bottom-right aligned, so the cached prefix
               is context)
Checked against Triton on the same layout; timed per layer.

    python probes/attn_bench2.py [--tokens 1600] [--prefix 0]
"""
import argparse
import math

import torch

import vllm.v1.attention.ops.triton_unified_attention as tua

SHAPES = {"sliding": (16, 8, 256, 16, 1024), "global": (16, 2, 512, 32, 0)}


def cuda_time(f, reps=30):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1600)
    ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=64)
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata  # noqa: F401  (import check)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    C, Lnew, P0 = a.canvas, a.tokens, a.prefix
    L = P0 + Lnew
    Tp = Lnew + C + 64
    P = Tp - Lnew - C
    print(f"layout: canvas {C} first, {P0} cached + {Lnew} prompt, pad {P}; bucket {Tp}", flush=True)
    for name, (Hq, Hkv, D, bs, sw) in SHAPES.items():
        nreal, npad = math.ceil((L + C) / bs), math.ceil(P / bs)
        kv = torch.randn(nreal + npad + 4, Hkv, bs, 2 * D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        k, v = kv.split(D, dim=-1)
        q = torch.randn(Tp, Hq, D, device=dev, dtype=torch.bfloat16)
        bt = torch.zeros(3, nreal + 1, dtype=torch.int32, device=dev)
        bt[0, :nreal] = torch.arange(1, 1 + nreal, device=dev)
        bt[1] = bt[0]
        bt[2, :npad] = torch.arange(1 + nreal, 1 + nreal + npad, device=dev)
        cu = torch.tensor([0, C, C + Lnew, Tp], dtype=torch.int32, device=dev)
        seqused = torch.tensor([L + C, L, P], dtype=torch.int32, device=dev)
        causal = torch.tensor([False, True, True], device=dev)
        one = torch.ones(1, device=dev).expand(3, Hkv)
        out_t = torch.empty_like(q)
        win = (sw - 1, 0) if sw else (-1, -1)

        def triton():
            tua.unified_attention(q=q, k=k, v=v, out=out_t, cu_seqlens_q=cu, max_seqlen_q=Tp, seqused_k=seqused,
                                  max_seqlen_k=L + C, softmax_scale=1.0, causal=causal, window_size=win,
                                  block_table=bt, softcap=0, q_descale=None, k_descale=one, v_descale=one,
                                  mm_prefix_range=torch.zeros(3, 8, 2, dtype=torch.int32, device=dev),
                                  mm_prefix_clamp_sliding_window=bool(sw))
        t_tri = cuda_time(triton)
        ref = out_t.clone()
        line = f"{name} (head {D}): triton {t_tri:.0f} us"
        if D > 256:
            print(line + "  (FA2 supports head <= 256)", flush=True)
            continue
        out_f = torch.empty_like(q)
        cu_c = torch.tensor([0, C], dtype=torch.int32, device=dev)
        cu_p = cu[1:] - C
        print(line, flush=True)
        line = "   "
        for ver in (2, 4):   # FA3 is Hopper-only (its launcher aborts the process on Blackwell)
            def fa():
                flash_attn_varlen_func(q=q[:C], k=k, v=v, out=out_f[:C], cu_seqlens_q=cu_c, max_seqlen_q=C,
                                       seqused_k=seqused[:1], max_seqlen_k=L + C, softmax_scale=1.0, causal=False,
                                       window_size=[sw - 1, sw - 1] if sw else [-1, -1], block_table=bt[:1],
                                       fa_version=ver)
                flash_attn_varlen_func(q=q[C:], k=k, v=v, out=out_f[C:], cu_seqlens_q=cu_p, max_seqlen_q=Tp,
                                       seqused_k=seqused[1:], max_seqlen_k=L + C, softmax_scale=1.0, causal=True,
                                       window_size=list(win), block_table=bt[1:], fa_version=ver)
            try:
                t = cuda_time(fa)
                d = (out_f[:C + Lnew].float() - ref[:C + Lnew].float()).abs().max().item()
                print(f"   FA{ver} {t:.0f} us (max|diff| vs triton {d:.3g})", flush=True)
            except Exception as exc:
                print(f"   FA{ver} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)


if __name__ == "__main__":
    main()
