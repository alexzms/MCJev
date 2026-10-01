#!/usr/bin/env python3
"""Global (head 512, 16q/2kv) attention for the fast path's canvas-first layout: Triton (vLLM default and the best
swept launch config) vs torch alternatives over the paged cache gathered to a static key length; plus FA4 for the
sliding layers with the page-table constraints it asks for.

    python probes/attn_bench3.py [--tokens 1600] [--prefix 0]
"""
import argparse
import math

import torch
import torch.nn.functional as F

import vllm.v1.attention.ops.triton_unified_attention as tua
from attn_bench2 import cuda_time

MAXLEN = 4096


def layout(Hq, Hkv, D, bs, C, Lnew, P0, dev):
    L = P0 + Lnew
    Tp = Lnew + C + 64
    P = Tp - Lnew - C
    width = MAXLEN // bs
    nb = 2 * width + 4
    kv = torch.randn(nb, Hkv, bs, 2 * D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    k, v = kv.split(D, dim=-1)
    q = torch.randn(Tp, Hq, D, device=dev, dtype=torch.bfloat16)
    bt = torch.zeros(3, width, dtype=torch.int32, device=dev)
    bt[0] = torch.arange(1, 1 + width, device=dev)
    bt[1] = bt[0]
    bt[2] = torch.arange(1 + width, 1 + 2 * width, device=dev)
    cu = torch.tensor([0, C, C + Lnew, Tp], dtype=torch.int32, device=dev)
    seqused = torch.tensor([L + C, L, P], dtype=torch.int32, device=dev)
    # absolute positions per token and the last key each may see (canvas: all L + C; prompt: causal; pad: key 0)
    pos = torch.cat([torch.arange(L, L + C), torch.arange(P0, L), torch.zeros(P, dtype=torch.long)]).to(dev)
    lim = torch.cat([torch.full((C,), L + C - 1), torch.arange(P0, L), torch.zeros(P, dtype=torch.long)]).to(dev)
    return dict(q=q, k=k, v=v, bt=bt, cu=cu, seqused=seqused, pos=pos, lim=lim, Tp=Tp, L=L, C=C, Lnew=Lnew)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1600)
    ap.add_argument("--prefix", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    C = 64
    Hq, Hkv, D, bs = 16, 2, 512, 32
    x = layout(Hq, Hkv, D, bs, C, a.tokens, a.prefix, dev)
    q, k, v, Tp = x["q"], x["k"], x["v"], x["Tp"]
    print(f"global: canvas {C} first, {a.prefix} cached + {a.tokens} prompt; bucket {Tp}", flush=True)
    out_t = torch.empty_like(q)
    causal = torch.tensor([False, True, True], device=dev)
    one = torch.ones(1, device=dev).expand(3, Hkv)
    rng = torch.zeros(3, 8, 2, dtype=torch.int32, device=dev)

    def triton():
        tua.unified_attention(q=q, k=k, v=v, out=out_t, cu_seqlens_q=x["cu"], max_seqlen_q=Tp, seqused_k=x["seqused"],
                              max_seqlen_k=MAXLEN, softmax_scale=1.0, causal=causal, window_size=(-1, -1),
                              block_table=x["bt"], softcap=0, q_descale=None, k_descale=one, v_descale=one,
                              mm_prefix_range=rng)
    print(f"   triton (vLLM default)         {cuda_time(triton):6.0f} us", flush=True)
    ref = out_t.clone()
    valid = slice(0, C + a.tokens)

    # gather the sequence's keys (static length) from the paged cache: slot index per key position
    kpos = torch.arange(MAXLEN, device=dev)
    blk = x["bt"][0].long()[kpos // bs]
    off = kpos % bs
    mask = kpos[None, :] <= x["lim"][:, None]                   # [Tp, MAXLEN]

    def gathered(nkeys):
        kk = k[blk[:nkeys], off[:nkeys]]                          # [n, Hkv, D]
        vv = v[blk[:nkeys], off[:nkeys]]
        return kk, vv

    for nkeys, label in ((MAXLEN, "static 4096 keys"), (x["L"] + C, "exact keys (not graph-static)")):
        m = mask[:, :nkeys]
        amask = torch.zeros(m.shape, device=dev, dtype=torch.bfloat16).masked_fill_(~m, float("-inf"))

        def mat():   # GQA as one batched GEMM per kv head: [Hkv, 8*Tp, D] x [Hkv, D, n]
            kk, vv = gathered(nkeys)
            qg = q.view(Tp, Hkv, Hq // Hkv, D).permute(1, 2, 0, 3).reshape(Hkv, (Hq // Hkv) * Tp, D)
            s = torch.bmm(qg, kk.permute(1, 2, 0), out_dtype=torch.float32)    # fp32 scores
            s = s.view(Hkv, Hq // Hkv, Tp, nkeys).masked_fill_(~m, float("-inf"))
            p = torch.softmax(s, -1).to(torch.bfloat16).view(Hkv, (Hq // Hkv) * Tp, nkeys)
            o = torch.bmm(p, vv.permute(1, 0, 2))                               # [Hkv, 8*Tp, D]
            return o.view(Hkv, Hq // Hkv, Tp, D).permute(2, 0, 1, 3).reshape(Tp, Hq, D)
        try:
            t = cuda_time(mat)
            d = (mat()[valid].float() - ref[valid].float()).abs().max().item()
            print(f"   bmm+softmax, {label:30s} {t:6.0f} us  max|diff| {d:.3g}", flush=True)
        except Exception as exc:
            print(f"   bmm+softmax, {label}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        for be in ("EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "FLASH_ATTENTION"):
            def sdpa():
                kk, vv = gathered(nkeys)
                qq = q.transpose(0, 1)[None]
                kk = kk.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, 1)
                vv = vv.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, 1)
                with torch.nn.attention.sdpa_kernel(getattr(torch.nn.attention.SDPBackend, be)):
                    o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=amask, scale=1.0)
                return o[0].transpose(0, 1)
            try:
                t = cuda_time(sdpa)
                d = (sdpa()[valid].float() - ref[valid].float()).abs().max().item()
                print(f"   SDPA {be:20s} {label:30s} {t:6.0f} us  max|diff| {d:.3g}", flush=True)
            except Exception as exc:
                print(f"   SDPA {be} {label}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    # FA4 on the sliding shape, with max_seqlen_k = page_table width * page size
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    Hq, Hkv, D, bs, sw = 16, 8, 256, 16, 1024
    y = layout(Hq, Hkv, D, bs, C, a.tokens, a.prefix, dev)
    out = torch.empty_like(y["q"])
    cu_c = torch.tensor([0, C], dtype=torch.int32, device=dev)
    cu_p = y["cu"][1:] - C
    for ver in (2, 4):
        def fa():
            flash_attn_varlen_func(q=y["q"][:C], k=y["k"], v=y["v"], out=out[:C], cu_seqlens_q=cu_c, max_seqlen_q=C,
                                   seqused_k=y["seqused"][:1], max_seqlen_k=MAXLEN, softmax_scale=1.0, causal=False,
                                   window_size=[sw - 1, sw - 1], block_table=y["bt"][:1], fa_version=ver)
            flash_attn_varlen_func(q=y["q"][C:], k=y["k"], v=y["v"], out=out[C:], cu_seqlens_q=cu_p,
                                   max_seqlen_q=y["Tp"], seqused_k=y["seqused"][1:], max_seqlen_k=MAXLEN,
                                   softmax_scale=1.0, causal=True, window_size=[sw - 1, 0], block_table=y["bt"][1:],
                                   fa_version=ver)
        try:
            print(f"   sliding FA{ver} (max_seqlen_k {MAXLEN}): {cuda_time(fa):6.0f} us", flush=True)
            if ver == 2:
                o2 = out.clone()
            else:
                print(f"      FA4 vs FA2 max|diff| {(out[:C + a.tokens].float() - o2[:C + a.tokens].float()).abs().max().item():.3g}")
        except Exception as exc:
            print(f"   sliding FA{ver}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)


if __name__ == "__main__":
    main()
