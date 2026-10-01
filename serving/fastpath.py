"""djev fast path: one forward pass per read, replayed from a full CUDA graph, on a vLLM worker (no scheduler).

Math (identical to vLLM's two-step read: encoder prefill, then one decoder denoise of the canvas):
    tokens = [prompt ; canvas];  canvas embeddings -> self_conditioning(e, soft=0) == RMSNorm_noweight(e)
    attention: prompt causal; canvas sees the whole prompt and the whole canvas (bidirectional)
    logits only at the answer slots, log-softmax over the vocabulary, gathered at the candidate token ids.

How it runs on vLLM's kernels without its scheduler:
  * an in-process vLLM worker gives us the loaded model, the paged KV cache and the attention backends;
  * one forward = three pseudo-sequences on the paged cache, in this token order:
        B  the canvas                                  non-causal, seq_len L + C   (same blocks as A)
        A  the prompt tokens after the cached prefix   causal,     seq_len L
        D  padding up to the bucket size               causal,     own blocks      (result ignored)
    (canvas first: its C queries are a static slice, which the FlashAttention path below needs; positions are data,
    so the token order does not change the math)
  * prefix cache: a few slots, each its own KV blocks. A read picks the slot whose last prompt shares the longest
    (block-aligned) prefix with its own, computes only the rest, and so leaves its own prompt's KV in the slot for
    the next read (prompt KV is causal: it never depends on the canvas). Agents that resend a long fixed preamble
    each step pay for it once. The prefix length is data in the index buffers, so the same graphs serve any prefix;
  * per bucket size Tp one CUDA graph contains everything from token ids to candidate log-probs;
    per read we fill two small index buffers (one H2D copy each), replay, and copy back [slots x cands].

KV blocks: vLLM's hybrid allocator overlays the KV tensors of its cache groups (sliding-window and full-attention
layers here), sound because a block id belongs to one group at a time. So every (slot, group) pair and the padding
sequence get disjoint block ids; sharing ids across groups is harmless within one forward but corrupts a kept prefix.

One FastReader per process per GPU (vLLM's in-process engine owns the device).
"""
import math
import os
import time

import numpy as np
import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine core in this process: we need its worker

MODEL = "google/diffusiongemma-26B-A4B-it"
PAD_ID = 0

# ---------------------------------------------------------------------- attention for the text graphs
# vLLM's Triton unified attention handles our layout (per-sequence causal flags, image spans) but is slow at prefill
# sizes on Blackwell: ~520 us per sliding layer at 1.6k tokens where FlashAttention-2 takes ~130 us
# (probes/attn_bench*.py). While a text graph is captured, the sliding layers (head 256) run two FA2 calls instead:
# the canvas (the first C tokens of the forward) non-causal, then prompt + padding causal (bottom-right aligned, so a
# cached prefix is context). FA2 has no head 512, so the global layers stay on Triton with a launch config swept for
# this shape (~25% faster than the default, same results). Image reads need bidirectional image spans: their graphs
# are captured with plain Triton.
_ATTN = {"fa": False}


def _install_attention():
    if _ATTN.get("installed"):
        return
    import inspect
    import vllm.v1.attention.backends.triton_attn as ta
    import vllm.v1.attention.ops.triton_unified_attention as tua
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    src = inspect.getsource(tua.unified_attention)
    hook = ("    _ov = _DJEV_LAUNCH.get(head_size) if max_seqlen_q > 1 else None\n"
            "    if _ov:\n"
            "        BLOCK_M = _ov['BLOCK_M']\n"
            "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
            "        grid = (q.shape[0] // BLOCK_Q + num_seqs, num_kv_heads)\n"
            "        tile_size = _ov['TILE']\n"
            "        launch_kwargs = {'num_warps': _ov['warps'], 'num_stages': _ov['stages']}\n"
            "    kernel_unified_attention[grid](\n")
    assert src.count("    kernel_unified_attention[grid](\n") == 1, "vLLM's unified_attention changed"
    tua._DJEV_LAUNCH = {512: {"BLOCK_M": 32, "TILE": 32, "warps": 4, "stages": 2}}
    exec(compile(src.replace("    kernel_unified_attention[grid](\n", hook), tua.__file__, "exec"), tua.__dict__)
    ta.unified_attention = tua.unified_attention
    orig = ta.TritonAttentionImpl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, *args, **kwargs):
        if not _ATTN["fa"] or attn_metadata is None:
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        # r reads in this forward: their canvases are the first C = r * C1 tokens (pseudo-sequences 0..r-1), then
        # their prompts and the padding (pseudo-sequences r..2r)
        r, C1 = _ATTN["r"], _ATTN["C1"]
        C, n, md = r * C1, attn_metadata.num_actual_tokens, attn_metadata
        k, v = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        left = self.sliding_window[0]
        soft = self.logits_soft_cap or 0.0
        if self.head_size > 256:
            # Triton, as two calls: with per-sequence causal flags the kernel reads every key for every query block;
            # a plain causal call over prompt + padding skips the masked tiles (1.6x faster at 1.6k tokens)
            ones = _ATTN["ones"]
            common = dict(k=k, v=v, max_seqlen_k=_ATTN["max_k"], softmax_scale=self.scale, window_size=(-1, -1),
                          softcap=soft, q_descale=None)
            tua.unified_attention(q=query[:C], out=output[:C], cu_seqlens_q=_ATTN["cu_c"], max_seqlen_q=C1,
                                  seqused_k=md.seq_lens[:r], causal=False, block_table=md.block_table[:r],
                                  k_descale=ones[:r], v_descale=ones[:r], **common)
            tua.unified_attention(q=query[C:n], out=output[C:n], cu_seqlens_q=_ATTN["cu_p"], max_seqlen_q=n - C,
                                  seqused_k=md.seq_lens[r:], causal=True, block_table=md.block_table[r:],
                                  k_descale=ones[r:2 * r + 1], v_descale=ones[r:2 * r + 1], **common)
            return output
        flash_attn_varlen_func(q=query[:C], k=k, v=v, out=output[:C], cu_seqlens_q=_ATTN["cu_c"], max_seqlen_q=C1,
                               seqused_k=md.seq_lens[:r], max_seqlen_k=_ATTN["max_k"], softmax_scale=self.scale,
                               causal=False, window_size=[left, left], softcap=soft, block_table=md.block_table[:r],
                               fa_version=_ATTN["fav"])
        flash_attn_varlen_func(q=query[C:n], k=k, v=v, out=output[C:n], cu_seqlens_q=_ATTN["cu_p"],
                               max_seqlen_q=n - C, seqused_k=md.seq_lens[r:], max_seqlen_k=_ATTN["max_k"],
                               softmax_scale=self.scale, causal=True, window_size=[left, 0], softcap=soft,
                               block_table=md.block_table[r:], fa_version=_ATTN["fav"])
        return output

    ta.TritonAttentionImpl.forward = forward
    _ATTN["installed"] = True


class FastReader:
    def __init__(self, model=MODEL, canvas=64, moe="flashinfer_cutlass", prefix_slots=8, prefix_min=128,
                 bucket_step=32, max_tokens=2560, max_slots=8, max_cands=64, gpu_memory_utilization=0.85,
                 max_model_len=4096, buckets=None, thought_ids=(), max_images=8, compile_vision=True,
                 vision_cache=256, fast_attention=True, quantization=None, batch_max=1, batch_tokens=2048,
                 kv_block_size=None, fa_version=2, kv_cache_dtype=None):
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=model, diffusion_config={"canvas_length": canvas}, enforce_eager=False,
                       kernel_config={"moe_backend": moe}, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=False, max_logprobs=128,
                       limit_mm_per_prompt={"image": 8}, quantization=quantization,
                       **({"block_size": kv_block_size} if kv_block_size else {}),
                       **({"kv_cache_dtype": kv_cache_dtype} if kv_cache_dtype else {}))
        worker = self.llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker
        self.runner = worker.model_runner
        self.model = self.runner.model
        self.cfg = self.runner.vllm_config
        self.dev = self.runner.device
        self.C = canvas
        self.S, self.K = max_slots, max_cands
        self.model_name = model
        self.thought_ids = list(thought_ids)
        self.R = max_images                          # bidirectional image spans per read (kernel mm_prefix ranges)
        self.H = self.cfg.model_config.hf_text_config.hidden_size
        self._proc = None
        self._vcache = __import__("collections").OrderedDict()   # sha1(image bytes) -> [n_tokens, H] on GPU
        self._vcache_max = vision_cache
        vt = self.model.vision_tower
        self.vision_attn = getattr(vt.config, "_attn_implementation", None)
        if self.vision_attn in (None, "eager"):
            vt.config._attn_implementation = "sdpa"
            for m in vt.modules():
                if hasattr(m, "config") and getattr(m.config, "_attn_implementation", None) in (None, "eager"):
                    m.config._attn_implementation = "sdpa"
        if compile_vision:
            vt.encoder = torch.compile(vt.encoder, mode="reduce-overhead", dynamic=False)
        from transformers import AutoTokenizer
        tk = self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.boi, self.img_tok, self.eoi = (tk.convert_tokens_to_ids(t) for t in ("<|image>", "<|image|>", "<image|>"))
        self.chat_head = [tk.convert_tokens_to_ids("<bos>"), tk.convert_tokens_to_ids("<|turn>")] + tk.encode("user\n", add_special_tokens=False)
        self.chat_tail = ([tk.convert_tokens_to_ids("<turn|>")] + tk.encode("\n", add_special_tokens=False)
                          + [tk.convert_tokens_to_ids("<|turn>")] + tk.encode("model\n", add_special_tokens=False))
        kcc = self.runner.kv_cache_config
        self.kcc = kcc
        self.groups = []                              # (block_size, kernel_block_size) per KV group
        for gi, g in enumerate(kcc.kv_cache_groups):
            self.groups.append((g.kv_cache_spec.block_size, self.runner.kernel_block_sizes[gi]))
        self.align = math.lcm(*[bs for bs, _ in self.groups])
        self.max_tokens = max_tokens
        if buckets:
            self.buckets = sorted(buckets)
        else:   # fine steps where text reads live, coarser for long (image) prompts
            self.buckets = (list(range(96, 512, bucket_step)) + list(range(512, 2048, 64))
                            + list(range(2048, max_tokens + 1, 128)))
            self.max_tokens = self.buckets[-1]
        self.max_len = max_model_len                   # prompt + canvas positions per read, cached prefix included
        # block regions (logical block ids, per group): slot s -> region s; the padding sequence -> region NS
        self.NS, self.prefix_min = max(1, prefix_slots), prefix_min
        self.use_prefix = prefix_slots > 0
        self.regions = []                              # [group][region] -> kernel block ids covering the region
        nxt = 1                                        # block 0 is vLLM's null block
        for bs, kbs in self.groups:
            per, rows = bs // kbs, []
            for r in range(self.NS + 1):
                n = math.ceil((self.max_len if r < self.NS else self.max_tokens) / bs)
                ids = np.arange(nxt, nxt + n, dtype=np.int64)
                rows.append((ids[:, None] * per + np.arange(per)[None]).ravel())
                nxt += n
            self.regions.append(rows)
        if nxt > kcc.num_blocks:
            raise ValueError(f"prefix cache needs {nxt} KV blocks, the cache has {kcc.num_blocks}: lower prefix_slots")
        self.slot_keys = [np.zeros(0, dtype=np.int64) for _ in range(self.NS)]   # prompt keys whose KV is in the slot
        self.slot_used = [0] * self.NS
        self._tick = 0
        self.last = {}                                 # slot, cached, computed of the latest read
        # the sc post-norm (weightless RMSNorm) is what self_conditioning(e, 0) reduces to
        self.post_norm = self.model.self_conditioning.post_norm
        self.pool = torch.cuda.graph_pool_handle()
        self.fa = fast_attention
        self.triton_below = 256 if fast_attention else 0     # reads up to this many tokens use the Triton graphs
        # batched text reads: r = 2..batch_max reads in one forward (distinct prefix slots), up to r * batch_tokens
        # tokens computed, on coarser buckets
        self.batch_max = max(1, min(batch_max, self.NS)) if fast_attention else 1
        self.mbuckets = {r: list(range(r * 128, min(2048, r * batch_tokens) + 1, 128))
                         + list(range(2304, r * batch_tokens + 1, 256)) for r in range(2, self.batch_max + 1)}
        if fast_attention:
            _install_attention()
            nkv = max(self.cfg.model_config.hf_text_config.num_key_value_heads, 8)
            _ATTN.update(C1=self.C, r=1, max_k=max(self.max_len, self.max_tokens), fav=fa_version,
                         cu_c=torch.tensor([0, self.C], dtype=torch.int32, device=self.dev),
                         ones=torch.ones(2 * self.batch_max + 1, nkv, dtype=torch.float32, device=self.dev))
        # text reads: FA2 graphs at every bucket; reads with image tokens to compute: Triton graphs, coarser buckets
        self.ibuckets = [b for b in self.buckets if b <= 512 or b % 128 == 0] if fast_attention else self.buckets
        self.graphs, self.igraphs, self.mgraphs = {}, {}, {}
        with torch.inference_mode():
            for Tp in self.buckets:
                self.graphs[Tp] = self._capture(Tp, fa=fast_attention)
            for Tp in self.ibuckets:
                self.igraphs[Tp] = self._capture(Tp, fa=False) if fast_attention else self.graphs[Tp]
            for r, bks in self.mbuckets.items():
                for Tp in bks:
                    self.mgraphs[r, Tp] = self._capture(Tp, fa=True, r=r)
        if compile_vision:
            self._warm_vision()
        self.slot_keys = [np.zeros(0, dtype=np.int64) for _ in range(self.NS)]   # capture wrote into slot 0
        self.load_seconds = time.time() - t0

    def _warm_vision(self):
        """Compile the vision encoder for square and wide inputs before serving (first call compiles)."""
        import io
        from PIL import Image
        for size in ((256, 256), (640, 480), (480, 640)):
            buf = io.BytesIO()
            Image.new("RGB", size, (120, 80, 40)).save(buf, format="PNG")
            for _ in range(3):
                self._vision_embeds([buf.getvalue()], use_cache=False)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ layout helpers (numpy, CPU)
    def _layout(self, Tp, slot, P0, Lnew, C):
        """int32 and int64 host buffers for one read: P0 cached prompt tokens in `slot`, Lnew computed ones."""
        L = P0 + Lnew
        T = Lnew + C                      # tokens in the forward besides padding
        P = Tp - T
        assert P >= 1, (Tp, T)
        assert L + C <= self.max_len, (L, C, self.max_len)
        # token order [canvas; prompt; padding]: query_start_loc, seq_lens, and the prompt+padding query starts
        i32 = [np.asarray([0, C, T, Tp, L + C, L, P, 0, Lnew, Lnew + P], dtype=np.int32)]
        i64 = []
        pos_real = np.concatenate([np.arange(L, L + C), np.arange(P0, L)])
        pos_pad = np.arange(P)
        for (bs, kbs), rows in zip(self.groups, self.regions):
            kreal, kpad = rows[slot], rows[self.NS]
            bt = np.zeros((3, self._bt_width(bs, bs // kbs)), dtype=np.int32)
            bt[0, :kreal.size] = kreal
            bt[1, :kreal.size] = kreal
            bt[2, :kpad.size] = kpad
            i32.append(bt.ravel())
            i64 += [kreal[pos_real // kbs] * kbs + pos_real % kbs, kpad[pos_pad // kbs] * kbs + pos_pad % kbs]
        return np.concatenate(i32), np.concatenate(i64)

    def _bt_width(self, bs, per):
        return math.ceil(max(self.max_len, self.max_tokens) / bs) * per

    # ------------------------------------------------------------------ prefix cache
    def _pick_slot(self, keys, spans=(), exclude=()):
        """(slot, P0): the slot to run in and how many leading prompt tokens its KV already holds. The best match
        is used when it is worth keeping (>= prefix_min tokens); otherwise the least recently used slot is reused.
        P0 is block aligned, leaves at least one prompt token to compute, and never cuts an image span (its tokens
        attend to each other both ways, so it is computed whole or not at all). `exclude`: slots taken by the other
        reads of the same forward."""
        if not self.use_prefix:
            return (min(s for s in range(self.NS) if s not in exclude) if exclude else 0), 0
        m = len(keys) - 1
        lcp = []
        for k in self.slot_keys:
            n = min(len(k), m)
            neq = np.flatnonzero(k[:n] != keys[:n]) if n > 0 else np.zeros(1, dtype=np.int64)
            lcp.append(int(neq[0]) if neq.size else n)
        free = [s for s in range(self.NS) if s not in exclude]
        best = max(free, key=lambda s: (lcp[s], self.slot_used[s]))
        if lcp[best] < self.prefix_min:
            best = min(free, key=self.slot_used.__getitem__)
        P0 = lcp[best]
        while True:
            P0 = (P0 // self.align) * self.align
            cut = [a for a, b in spans if a < P0 <= b]
            if not cut:
                break
            P0 = cut[0]
        return best, P0

    # ------------------------------------------------------------------ graph capture
    def _capture(self, Tp, fa=False, r=1):
        """The CUDA graph for Tp tokens and r reads (r > 1: batched text reads, FA2 only): pseudo-sequences
        [canvas_1..canvas_r, prompt_1..prompt_r, padding]."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        dev, S, K, C = self.dev, self.S, self.K, self.C
        g = {"Tp": Tp, "r": r}
        # per-read inputs: token ids / positions / canvas mask / slot rows / candidate ids  (int64 buffer)
        g["tok"] = torch.zeros(Tp, dtype=torch.int64, device=dev)
        g["pos"] = torch.zeros(Tp, dtype=torch.int64, device=dev)
        g["canvas"] = torch.zeros(Tp, dtype=torch.bool, device=dev)
        g["rows"] = torch.zeros(r * S, dtype=torch.int64, device=dev)
        g["cand"] = torch.zeros((r * S, K), dtype=torch.int64, device=dev)
        g["imask"] = torch.zeros(Tp, dtype=torch.bool, device=dev)            # image soft-token positions
        g["ibuf"] = torch.zeros((Tp, self.H), dtype=self.model_dtype(), device=dev)
        g["rng"] = torch.zeros((2 * r + 1, self.R, 2), dtype=torch.int32, device=dev)  # (start, end) incl.; 0,0 = none
        # attention layout (int32 + int64 flat buffers, views below): query starts (2r+2), seq lens (2r+1), the
        # prompt+padding query starts (r+2), then per KV group the (2r+1)-row block table; int64: slot mappings
        nseq = 2 * r + 1
        n32 = 5 * r + 5 + sum(nseq * self._bt_width(bs, bs // kbs) for bs, kbs in self.groups)
        n64 = len(self.groups) * Tp
        g["b32"] = torch.zeros(n32, dtype=torch.int32, device=dev)
        g["b64"] = torch.zeros(n64, dtype=torch.int64, device=dev)
        qsl = g["b32"][0:2 * r + 2]
        seq_lens = g["b32"][2 * r + 2:4 * r + 3]
        bts, sms, o32, o64 = [], [], 5 * r + 5, 0
        for bs, kbs in self.groups:
            w = self._bt_width(bs, bs // kbs)
            bts.append(g["b32"][o32:o32 + nseq * w].view(nseq, w))
            o32 += nseq * w
            sms.append(g["b64"][o64:o64 + Tp])
            o64 += Tp
        causal = torch.tensor([False] * r + [True] * (r + 1), device=dev)          # canvases, prompts, padding
        # a valid layout to capture with
        if r == 1:
            Lnew = Tp - C - 1
            self._write(g, 0, 0, [PAD_ID] * Lnew, [PAD_ID] * C, [0], [[PAD_ID]])
            qsl_cpu = torch.tensor([0, C, Tp - 1, Tp], dtype=torch.int32)
        else:
            lens = [(Tp - r * C - 1) // r] * r
            lens[0] += Tp - r * C - 1 - sum(lens)
            i32 = self._write_many(g, [(j, 0, [PAD_ID] * n, [PAD_ID] * C, [0], [[PAD_ID]]) for j, n in enumerate(lens)])
            qsl_cpu = torch.from_numpy(i32[:2 * r + 2].copy())
            g["cu_c"] = torch.arange(0, (r + 1) * C, C, dtype=torch.int32, device=dev)
        # max_seq_len only steers the decode (one query per sequence) kernel; reads always take the prefill one
        md = build_attn_metadata(attn_groups=self.runner.attn_groups, num_reqs=nseq, num_tokens=Tp,
                                 query_start_loc_gpu=qsl, query_start_loc_cpu=qsl_cpu, max_query_len=Tp,
                                 seq_lens=seq_lens, max_seq_len=self.max_len, block_tables=bts, slot_mappings=sms,
                                 kv_cache_config=self.kcc, causal=causal)
        by_layer = build_slot_mappings_by_layer(sms, self.kcc)
        for m in {id(v): v for v in md.values()}.values():        # images: bidirectional within each image span
            m.mm_prefix_range = {0: [(0, 0)]}
            m.mm_prefix_range_tensor = g["rng"]

        def fwd():
            emb = self.model.embed_input_ids(g["tok"])
            emb = torch.where(g["imask"][:, None], g["ibuf"], emb)
            emb = torch.where(g["canvas"][:, None], self.post_norm(emb), emb)
            with set_forward_context(md, self.cfg, num_tokens=Tp, slot_mapping=by_layer,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE):
                h = self.model(input_ids=None, positions=g["pos"], inputs_embeds=emb, intermediate_tensors=None)
            logits = self.model.compute_logits(h[g["rows"]])
            return torch.log_softmax(logits.float(), -1).gather(1, g["cand"])      # [S, K]
        g["fwd"] = fwd

        g["fa"] = fa
        # read while capturing: FA2 for the sliding layers or not, and this graph's split into reads
        cu_c1 = _ATTN.get("cu_c")
        _ATTN.update(fa=fa, cu_p=g["b32"][4 * r + 3:5 * r + 5], r=r, cu_c=g.get("cu_c", cu_c1))
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fwd()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                g["out"] = fwd()
        finally:
            _ATTN.update(fa=False, r=1, cu_c=cu_c1)
        g["graph"] = graph
        g["host_out"] = torch.empty((r * S, K), dtype=torch.float32, pin_memory=True)
        return g

    def model_dtype(self):
        return self.cfg.model_config.dtype

    def _write(self, g, slot, P0, new_ids, canvas_ids, slot_positions, cand_ids, image=None):
        """Fill bucket g's input buffers for one read (host-side prep + a few H2D copies). new_ids = the prompt
        after the P0 tokens cached in `slot`. image = (positions int64 np.ndarray, embeddings [n, H] on GPU, spans
        [(start, end)] inclusive), all in prompt coordinates and all at or after P0."""
        Tp, S, K = g["Tp"], self.S, self.K
        Lnew, C = len(new_ids), len(canvas_ids)
        T = Lnew + C
        P = Tp - T
        L = P0 + Lnew
        tok = np.zeros(Tp, dtype=np.int64)          # [canvas; prompt; padding]
        tok[:C] = canvas_ids
        tok[C:T] = new_ids
        pos = np.concatenate([np.arange(L, L + C), np.arange(P0, L), np.arange(P)])
        canvas = np.zeros(Tp, dtype=bool)
        canvas[:C] = True
        rows = np.zeros(S, dtype=np.int64)
        rows[:len(slot_positions)] = slot_positions
        cand = np.zeros((S, K), dtype=np.int64)
        for i, ids in enumerate(cand_ids):
            cand[i, :len(ids)] = ids
        i32, i64 = self._layout(Tp, slot, P0, Lnew, C)
        g["tok"].copy_(torch.from_numpy(tok), non_blocking=True)
        g["pos"].copy_(torch.from_numpy(pos), non_blocking=True)
        g["canvas"].copy_(torch.from_numpy(canvas), non_blocking=True)
        g["rows"].copy_(torch.from_numpy(rows), non_blocking=True)
        g["cand"].copy_(torch.from_numpy(cand), non_blocking=True)
        g["b32"].copy_(torch.from_numpy(i32), non_blocking=True)
        g["b64"].copy_(torch.from_numpy(i64), non_blocking=True)
        imask = np.zeros(Tp, dtype=bool)
        rng = np.zeros((3, self.R, 2), dtype=np.int32)
        if image is not None and len(image[0]):
            positions, embs, spans = image
            at = positions - P0 + C
            imask[at] = True
            g["ibuf"].index_copy_(0, torch.from_numpy(at).to(self.dev, non_blocking=True), embs.to(g["ibuf"].dtype))
            for j, (a, b) in enumerate(spans[:self.R]):
                rng[1, j] = (a, b)      # sequence positions in pseudo-sequence 1 (the prompt)
        g["imask"].copy_(torch.from_numpy(imask), non_blocking=True)
        g["rng"].copy_(torch.from_numpy(rng), non_blocking=True)

    # ------------------------------------------------------------------ several text reads in one forward
    def _layout_many(self, Tp, items):
        """_layout for r reads in one forward, items = [(slot, P0, Lnew)]; token order [canvas_1..canvas_r;
        prompt_1..prompt_r; padding]. With r = 1 it is exactly _layout."""
        r, C = len(items), self.C
        Ls = [P0 + Lnew for _, P0, Lnew in items]
        P = Tp - r * C - sum(Lnew for _, _, Lnew in items)
        assert P >= 1, (Tp, items)
        qsl = [j * C for j in range(r + 1)]
        for _, _, Lnew in items:
            qsl.append(qsl[-1] + Lnew)
        qsl.append(Tp)
        seq_lens = [L + C for L in Ls] + Ls + [P]
        cu_p = [q - r * C for q in qsl[r:]]             # prompt + padding query starts: [0, Lnew_1, .., sum + P]
        i32, i64 = [np.asarray(qsl + seq_lens + cu_p, dtype=np.int32)], []
        for (bs, kbs), rows in zip(self.groups, self.regions):
            kpad = rows[self.NS]
            bt = np.zeros((2 * r + 1, self._bt_width(bs, bs // kbs)), dtype=np.int32)
            canv, prom = [], []
            for j, (slot, P0, _) in enumerate(items):
                kreal = rows[slot]
                assert Ls[j] + C <= self.max_len, (Ls[j], C, self.max_len)
                bt[j, :kreal.size] = kreal
                bt[r + j, :kreal.size] = kreal
                pc, pp = np.arange(Ls[j], Ls[j] + C), np.arange(P0, Ls[j])
                canv.append(kreal[pc // kbs] * kbs + pc % kbs)
                prom.append(kreal[pp // kbs] * kbs + pp % kbs)
            bt[2 * r, :kpad.size] = kpad
            pos_pad = np.arange(P)
            i32.append(bt.ravel())
            i64 += canv + prom + [kpad[pos_pad // kbs] * kbs + pos_pad % kbs]
        return np.concatenate(i32), np.concatenate(i64)

    def _write_many(self, g, items):
        """_write for r text reads in one forward: items = [(slot, P0, new_ids, canvas_ids, slot_positions,
        cand_ids)], each canvas self.C long. Returns the int32 layout (its first 2r+2 entries: the query starts)."""
        Tp, S, K, C, r = g["Tp"], self.S, self.K, self.C, len(items)
        tok = np.zeros(Tp, dtype=np.int64)
        pos = np.zeros(Tp, dtype=np.int64)
        rows = np.zeros(r * S, dtype=np.int64)
        cand = np.zeros((r * S, K), dtype=np.int64)
        o = r * C
        for j, (slot, P0, new_ids, canvas_ids, slot_positions, cand_ids) in enumerate(items):
            assert len(canvas_ids) == C, (len(canvas_ids), C)
            L = P0 + len(new_ids)
            tok[j * C:(j + 1) * C] = canvas_ids
            pos[j * C:(j + 1) * C] = np.arange(L, L + C)
            tok[o:o + len(new_ids)] = new_ids
            pos[o:o + len(new_ids)] = np.arange(P0, L)
            o += len(new_ids)
            rows[j * S:j * S + len(slot_positions)] = np.asarray(slot_positions, dtype=np.int64) + j * C
            for i, ids in enumerate(cand_ids):
                cand[j * S + i, :len(ids)] = ids
        pos[o:] = np.arange(Tp - o)
        canvas = np.zeros(Tp, dtype=bool)
        canvas[:r * C] = True
        i32, i64 = self._layout_many(Tp, [(it[0], it[1], len(it[2])) for it in items])
        g["tok"].copy_(torch.from_numpy(tok), non_blocking=True)
        g["pos"].copy_(torch.from_numpy(pos), non_blocking=True)
        g["canvas"].copy_(torch.from_numpy(canvas), non_blocking=True)
        g["rows"].copy_(torch.from_numpy(rows), non_blocking=True)
        g["cand"].copy_(torch.from_numpy(cand), non_blocking=True)
        g["b32"].copy_(torch.from_numpy(i32), non_blocking=True)
        g["b64"].copy_(torch.from_numpy(i64), non_blocking=True)
        g["imask"].zero_()
        g["rng"].zero_()
        return i32

    def _plan_many(self, reads):
        """For r text reads [(prompt_ids, canvas_ids, slot_positions, cand_ids)]: (items, keys, Tp) to run them in
        one batched forward, each in its own prefix slot, or None when no batched graph fits them."""
        r = len(reads)
        if r not in self.mbuckets:
            return None
        items, keys_all, taken, T = [], [], set(), 0
        for prompt_ids, canvas_ids, slot_positions, cand_ids in reads:
            if len(canvas_ids) != self.C or len(slot_positions) > self.S or any(len(c) > self.K for c in cand_ids):
                return None
            keys = np.asarray(prompt_ids, dtype=np.int64)
            slot, P0 = self._pick_slot(keys, exclude=taken)
            taken.add(slot)
            if len(prompt_ids) + self.C > self.max_len:
                return None
            items.append((slot, P0, prompt_ids[P0:], canvas_ids, slot_positions, cand_ids))
            keys_all.append(keys)
            T += len(prompt_ids) - P0 + self.C
        Tp = next((b for b in self.mbuckets[r] if b >= T + 1), None)
        return None if Tp is None else (items, keys_all, Tp)

    @torch.inference_mode()
    def read_many(self, reads):
        """Text reads [(prompt_ids, canvas_ids, slot_positions, cand_ids)], as many per forward as the batched graphs
        take (up to batch_max; a batch that fits no graph is run one read shorter). Returns [(log-probs
        [n_slots][n_cands], cached prompt tokens)] in order. Each read's answers are what read() gives it (up to
        bf16 rounding: the matrix products see more rows)."""
        out, i = [], 0
        while i < len(reads):
            r, plan = min(self.batch_max, len(reads) - i), None
            while r > 1 and plan is None:
                plan = self._plan_many(reads[i:i + r])
                if plan is None:
                    r -= 1
            if plan is None:
                logps = self.read(*reads[i])
                out.append((logps, self.last.get("cached", 0)))
                i += 1
                continue
            t0 = time.perf_counter()
            items, keys_all, Tp = plan
            g = self.mgraphs[r, Tp]
            for it in items:
                self.slot_keys[it[0]] = np.zeros(0, dtype=np.int64)    # invalid until the forward has run
            self._write_many(g, items)
            t_write = time.perf_counter()
            g["graph"].replay()
            g["host_out"].copy_(g["out"], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            t_gpu = time.perf_counter()
            host = g["host_out"]
            for j, (it, keys) in enumerate(zip(items, keys_all)):
                self._tick += 1
                self.slot_keys[it[0]], self.slot_used[it[0]] = keys, self._tick
                out.append(([host[j * self.S + q, :len(c)].tolist() for q, c in enumerate(it[5])], it[1]))
            ms = lambda a, b: round((b - a) * 1000, 2)
            self.last = {"batch": r, "bucket": Tp, "cached": items[-1][1], "fa": True,
                         "ms": {"pick": 0.0, "write": ms(t0, t_write), "gpu": ms(t_write, t_gpu),
                                "out": ms(t_gpu, time.perf_counter()), "bucket": Tp, "batch": r}}
            i += r
        return out

    @torch.inference_mode()
    def read_debug(self, prompt_ids, canvas_ids, slot_positions, cand_ids, variant="graph"):
        """A read without the prefix cache: "graph" (the text graphs), "triton" (the Triton graphs) or "eager"."""
        T = len(prompt_ids) + len(canvas_ids)
        graphs, buckets = (self.igraphs, self.ibuckets) if variant == "triton" else (self.graphs, self.buckets)
        g = graphs[next(b for b in buckets if b >= T + 1)]
        slot = min(range(self.NS), key=self.slot_used.__getitem__)
        self.slot_keys[slot] = np.zeros(0, dtype=np.int64)
        self._write(g, slot, 0, prompt_ids, canvas_ids, slot_positions, cand_ids)
        if variant == "eager":
            _ATTN.update(fa=g["fa"], cu_p=g["b32"][7:10])
            try:
                out = g["fwd"]()
            finally:
                _ATTN["fa"] = False
        else:
            g["graph"].replay()
            out = g["out"]
        torch.cuda.synchronize()
        return [out[i, :len(c)].tolist() for i, c in enumerate(cand_ids)]

    # ------------------------------------------------------------------ public
    @torch.inference_mode()
    def _vision_embeds_list(self, image_bytes, use_cache=True):
        """Per image [n_tokens, H] embeddings, in order; computed for cache misses only (one encoder call)."""
        import hashlib
        import io
        from PIL import Image
        keys = [hashlib.sha1(b).hexdigest() for b in image_bytes]
        missing = [i for i, k in enumerate(keys) if not (use_cache and k in self._vcache)]
        fresh = {}
        if missing:
            if self._proc is None:
                from transformers import AutoProcessor
                self._proc = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
            t = time.perf_counter()
            ims = [Image.open(io.BytesIO(image_bytes[i])).convert("RGB") for i in missing]
            out = self._proc.image_processor(images=ims, return_tensors="pt")
            if getattr(self, "timing", None) is not None:
                self.timing["preprocess"] = time.perf_counter() - t
            pv = out["pixel_values"].to(self.dev, dtype=self.model_dtype())
            ppi = out["image_position_ids"].to(self.dev)
            got = self.model.embed_multimodal(pixel_values=pv, pixel_position_ids=ppi)
            got = list(got) if not torch.is_tensor(got) else [got.reshape(-1, self.H)]
            for i, e in zip(missing, got):
                fresh[keys[i]] = e.clone()
        res = []
        for k in keys:
            if k in fresh:
                e = fresh[k]
                if use_cache:
                    self._vcache[k] = e
            else:
                e = self._vcache[k]
            if use_cache:
                self._vcache.move_to_end(k)
            res.append(e)
        while len(self._vcache) > self._vcache_max:
            self._vcache.popitem(last=False)
        return res

    def _vision_embeds(self, image_bytes, use_cache=True, out=None):
        return torch.cat(self._vision_embeds_list(image_bytes, use_cache), 0)

    @torch.inference_mode()
    def read_image(self, text, image_bytes, canvas_ids, slot_positions, cand_ids):
        """Like read(), for a prompt with images. Per image: vision embeddings (cached by content hash); the prompt
        is <bos><|turn>user\n + (<|image> <|image|>*n <image|>) per image + text + <turn|>\n<|turn>model\n + thought
        (token-identical to the Gemma-4 processor's chat template, checked). Each image's run of <|image|> tokens
        attends bidirectionally, as in the model's own prefill."""
        tm = self.timing = {}
        t = time.perf_counter()
        per = self._vision_embeds_list(image_bytes)
        torch.cuda.synchronize()
        tm["vision"] = time.perf_counter() - t
        t = time.perf_counter()
        import hashlib
        ids = list(self.chat_head)
        keys = list(ids)
        positions, spans = [], []
        for b, e in zip(image_bytes, per):
            ids.append(self.boi)       # its key names the image, so a prefix match covers the same pixels only
            keys.append(-1 - int(hashlib.sha1(b).hexdigest()[:15], 16))
            a = len(ids)
            ids.extend([self.img_tok] * e.shape[0])
            positions.extend(range(a, len(ids)))
            spans.append((a, len(ids) - 1))
            ids.append(self.eoi)
            keys += ids[len(keys):]
        ids += self.tok.encode(text, add_special_tokens=False) + self.chat_tail + self.thought_ids
        keys += ids[len(keys):]
        embs = torch.cat(per, 0)
        tm["tokenize"] = time.perf_counter() - t
        t = time.perf_counter()
        res = self._read(ids, canvas_ids, slot_positions, cand_ids, keys=keys,
                         image=(np.asarray(positions, dtype=np.int64), embs, spans))
        tm["forward"] = time.perf_counter() - t
        return res, len(ids)

    @torch.inference_mode()
    def read(self, prompt_ids, canvas_ids, slot_positions, cand_ids):
        """Log-probs [n_slots][n_cands] of the candidate ids at the slot positions (canvas-relative)."""
        return self._read(prompt_ids, canvas_ids, slot_positions, cand_ids)

    def _read(self, prompt_ids, canvas_ids, slot_positions, cand_ids, image=None, keys=None):
        t0 = time.perf_counter()
        keys = np.asarray(prompt_ids if keys is None else keys, dtype=np.int64)
        spans = image[2] if image is not None else ()
        slot, P0 = self._pick_slot(keys, spans)
        t_pick = time.perf_counter()
        new_ids = prompt_ids[P0:]
        T = len(new_ids) + len(canvas_ids)
        # image tokens to compute need their bidirectional spans: the Triton graphs; short reads too (Triton is
        # ~0.2 ms faster there, FA2 wins from a few hundred tokens on)
        mm = image is not None and any(a >= P0 for a, _ in image[2])
        tri = mm or T + 1 <= self.triton_below
        graphs, buckets = (self.igraphs, self.ibuckets) if tri else (self.graphs, self.buckets)
        Tp = next((b for b in buckets if b >= T + 1), None)
        if Tp is None:
            raise ValueError(f"read needs {T} tokens after the cached prefix; max {self.max_tokens - 1}")
        if len(slot_positions) > self.S or any(len(c) > self.K for c in cand_ids):
            raise ValueError(f"at most {self.S} slots and {self.K} candidate ids per slot")
        if image is not None:         # only the spans computed in this forward
            positions, embs, spans = image
            keep = positions >= P0
            image = (positions[keep], embs[torch.from_numpy(keep).to(embs.device)] if not keep.all() else embs,
                     [(a, b) for a, b in spans if a >= P0])
        g = graphs[Tp]
        self.slot_keys[slot] = np.zeros(0, dtype=np.int64)     # invalid until the forward has run
        self._write(g, slot, P0, new_ids, canvas_ids, slot_positions, cand_ids, image)
        t_write = time.perf_counter()
        g["graph"].replay()
        g["host_out"].copy_(g["out"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        t_gpu = time.perf_counter()
        self._tick += 1
        self.slot_keys[slot], self.slot_used[slot] = keys, self._tick
        out = g["host_out"]
        res = [out[i, :len(c)].tolist() for i, c in enumerate(cand_ids)]
        t_end = time.perf_counter()
        ms = lambda a, b: round((b - a) * 1000, 2)
        self.last = {"slot": slot, "cached": P0, "computed": len(new_ids), "bucket": Tp, "fa": g["fa"],
                     "ms": {"pick": ms(t0, t_pick), "write": ms(t_pick, t_write), "gpu": ms(t_write, t_gpu),
                            "out": ms(t_gpu, t_end), "bucket": Tp}}
        return res
