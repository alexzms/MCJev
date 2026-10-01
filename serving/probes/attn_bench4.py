#!/usr/bin/env python3
"""Per-layer attention time inside CUDA graphs (as served) for the canvas-first layout, at short/medium/long reads:
  sliding (head 256): FA2 canvas + prompt calls, num_splits swept per call
  global  (head 512): Triton one call (per-sequence causal flags: no causal tile skipping) vs two calls (canvas
                      non-causal + prompt/padding plain causal) with the default and the swept launch config

    python probes/attn_bench4.py
"""
import torch

import attn_bench as AB                      # installs the launch-config override hook into vLLM's launcher
import vllm.v1.attention.ops.triton_unified_attention as tua
from attn_bench3 import MAXLEN, layout

CASES = [(400, 0), (1100, 512), (1600, 0)]


def graph_time(f, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000


def main():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    dev = torch.device("cuda")
    torch.manual_seed(0)
    C = 64
    for Lnew, P0 in CASES:
        print(f"== {P0} cached + {Lnew} prompt + canvas {C}", flush=True)
        # sliding
        y = layout(16, 8, 256, 16, C, Lnew, P0, dev)
        out = torch.empty_like(y["q"])
        cu_c = torch.tensor([0, C], dtype=torch.int32, device=dev)
        cu_p = y["cu"][1:] - C
        Tp = y["Tp"]
        res = []
        for sc in (0, 1):
            for sp in (0, 1):
                def fa():
                    flash_attn_varlen_func(q=y["q"][:C], k=y["k"], v=y["v"], out=out[:C], cu_seqlens_q=cu_c,
                                           max_seqlen_q=C, seqused_k=y["seqused"][:1], max_seqlen_k=MAXLEN,
                                           softmax_scale=1.0, causal=False, window_size=[1023, 1023],
                                           block_table=y["bt"][:1], fa_version=2, num_splits=sc)
                    flash_attn_varlen_func(q=y["q"][C:], k=y["k"], v=y["v"], out=out[C:], cu_seqlens_q=cu_p,
                                           max_seqlen_q=Tp - C, seqused_k=y["seqused"][1:], max_seqlen_k=MAXLEN,
                                           softmax_scale=1.0, causal=True, window_size=[1023, 0],
                                           block_table=y["bt"][1:], fa_version=2, num_splits=sp)
                res.append((graph_time(fa), sc, sp))
        res.sort()
        auto = next(t for t, sc, sp in res if sc == 0 and sp == 0)
        print(f"   sliding FA2: auto splits {auto:5.0f} us; best {res[0][0]:5.0f} us (canvas splits {res[0][1]}, "
              f"prompt splits {res[0][2]}); " + ", ".join(f"{t:.0f}({sc},{sp})" for t, sc, sp in res[1:5]), flush=True)
        # global
        x = layout(16, 2, 512, 32, C, Lnew, P0, dev)
        q, k, v = x["q"], x["k"], x["v"]
        o = torch.empty_like(q)
        one = torch.ones(1, device=dev).expand(3, 2)
        rng = torch.zeros(3, 8, 2, dtype=torch.int32, device=dev)
        causal = torch.tensor([False, True, True], device=dev)
        common = dict(k=k, v=v, softmax_scale=1.0, window_size=(-1, -1), softcap=0, q_descale=None)

        def one_call():
            tua.unified_attention(q=q, out=o, cu_seqlens_q=x["cu"], max_seqlen_q=Tp, seqused_k=x["seqused"],
                                  max_seqlen_k=MAXLEN, causal=causal, block_table=x["bt"], k_descale=one,
                                  v_descale=one, mm_prefix_range=rng, **common)
        cu_p2 = x["cu"][1:] - C
        one2 = torch.ones(1, device=dev).expand(2, 2)
        one1 = torch.ones(1, device=dev).expand(1, 2)

        def two_calls():
            tua.unified_attention(q=q[:C], out=o[:C], cu_seqlens_q=cu_c, max_seqlen_q=C, seqused_k=x["seqused"][:1],
                                  max_seqlen_k=MAXLEN, causal=False, block_table=x["bt"][:1], k_descale=one1,
                                  v_descale=one1, **common)
            tua.unified_attention(q=q[C:], out=o[C:], cu_seqlens_q=cu_p2, max_seqlen_q=Tp - C,
                                  seqused_k=x["seqused"][1:], max_seqlen_k=MAXLEN, causal=True,
                                  block_table=x["bt"][1:], k_descale=one2, v_descale=one2, **common)
        line = "   global triton:"
        for label, cfg in (("default", None), ("swept", {"BLOCK_M": 32, "TILE": 32, "warps": 4, "stages": 2})):   # others crash
            AB.OVERRIDE.clear()
            if cfg:
                AB.OVERRIDE[512] = cfg
            try:
                t1 = graph_time(one_call)
                ref = o.clone()
                t2 = graph_time(two_calls)
                d = (o[:C + Lnew].float() - ref[:C + Lnew].float()).abs().max().item()
                line += f"  {label}: one call {t1:.0f} us, two calls {t2:.0f} us (diff {d:.2g});"
            except Exception as exc:
                line += f"  {label}: {type(exc).__name__};"
        AB.OVERRIDE.clear()
        print(line, flush=True)


if __name__ == "__main__":
    main()
