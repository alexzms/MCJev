#!/usr/bin/env python3
"""FlashAttention-2 against FlashAttention-4 (vllm_flash_attn fa_version=4, Blackwell) on the fast path's attention
calls for one p_uhc_pro_v10 read: the prompt (causal, ~1344 new queries over ~1830 keys, 480 of them a cached prefix)
and the canvas (64 queries, non-causal), on a paged KV cache like vLLM's. Sliding layers: 16 query heads, 8 KV heads,
head 256, window 1024; global layers: head 512, 2 KV heads (FA2 has no head 512). Same outputs? How fast?

    python probes/attn_fa4.py [--block 16]
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, nargs="+", default=[16, 32, 128])
    ap.add_argument("--q", type=int, default=1344)
    ap.add_argument("--cached", type=int, default=480)
    ap.add_argument("--canvas", type=int, default=64)
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
    dev, dt = "cuda", torch.bfloat16
    L = a.cached + a.q                              # prompt keys; the canvas sees them plus itself

    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n * 1000        # us

    for hq, hk, d, window, label in ((16, 8, 256, 1023, "sliding"),):   # head 512 (global): neither FA2 nor FA4
        for bs in a.block:
            nblk = (L + a.canvas) // bs + 2
            kc = torch.randn(nblk, bs, hk, d, device=dev, dtype=dt) / 4
            vc = torch.randn(nblk, bs, hk, d, device=dev, dtype=dt)
            bt = torch.arange(nblk, device=dev, dtype=torch.int32)[None]
            q = torch.randn(a.q, hq, d, device=dev, dtype=dt)
            qc = torch.randn(a.canvas, hq, d, device=dev, dtype=dt)
            calls = {
                "prompt": dict(q=q, max_seqlen_q=a.q, cu_seqlens_q=torch.tensor([0, a.q], device=dev, dtype=torch.int32),
                               seqused_k=torch.tensor([L], device=dev, dtype=torch.int32), causal=True,
                               window_size=[window, 0]),
                "canvas": dict(q=qc, max_seqlen_q=a.canvas,
                               cu_seqlens_q=torch.tensor([0, a.canvas], device=dev, dtype=torch.int32),
                               seqused_k=torch.tensor([L + a.canvas], device=dev, dtype=torch.int32), causal=False,
                               window_size=[window, window]),
            }
            for name, kw in calls.items():
                # FA4 (hd256, paged) wants max_seqlen_k == block-table width * page size: seqused_k is the real length
                common = dict(k=kc, v=vc, max_seqlen_k=nblk * bs, softmax_scale=d ** -0.5, block_table=bt, **kw)
                out, t = {}, {}
                for ver in (2, 4):
                    try:
                        out[ver] = fa(fa_version=ver, **common)
                        t[ver] = bench(lambda: fa(fa_version=ver, **common))
                    except Exception as exc:
                        out[ver], t[ver] = None, f"{type(exc).__name__}: {str(exc)[:90]}"
                diff = ((out[2].float() - out[4].float()).abs().max().item()
                        if out[2] is not None and out[4] is not None else float("nan"))
                fmt = lambda x: f"{x:8.1f} us" if isinstance(x, float) else x
                print(f"{label:7s} d={d} kv={hk} block {bs:3d} {name:6s} | FA2 {fmt(t[2])} | FA4 {fmt(t[4])} | "
                      f"max |diff| {diff:.4f}", flush=True)


if __name__ == "__main__":
    main()
