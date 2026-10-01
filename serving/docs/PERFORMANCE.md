# DJev-serve performance notes

How one decision went from **174 ms** to **~24 ms**, what the GPU time is spent on, and what we tried. The probes
that produced each number are in [`../probes/`](../probes); the hardware is named for every measurement (NVIDIA GB200
and B200, both Blackwell).

## The workload

A Jev bot sends one request per decision: a text state of ~4.5k characters (~1,760 tokens for the Block UHC harness)
and one 16-way choice question. DiffusionGemma-26B-A4B has 30 layers (25 sliding-window layers with head size 256 and
a 1,024-token window, 5 global layers with head size 512) and a mixture of experts with 128 experts, 8 per token
(3.8B active parameters). The answer canvas is 64 tokens. A decision is **one forward pass**: the prompt is prefilled
causally, the canvas (the answer scaffold with `<mask>` slots) attends bidirectionally to the prompt and to itself,
and the logits at the slot, gathered at the candidate tokens and renormalised, are the answer.

## 1. Hugging Face transformers: 174 ms (`server.py`)

The reference server runs the model in transformers, eager. One forward of a short text request took 174 ms on one
GB200 (encoder 90 ms, decoder 83 ms), and batching 8 requests only raised it to 206 ms. A profile
(`probes/profile-stages.py`) showed why: **the GPU kernels add up to 55 ms, and one forward launches 10,340 kernels**.
The forward was launch-bound: thousands of small elementwise, copy and cast kernels from routing, RMSNorm and rotary
glue across 30 layers, with the MoE expert GEMMs only 23% of GPU time.

## 2. The original vLLM: ~60 ms

vLLM (0.30 nightly) has a DiffusionGemma implementation with CUDA graphs, paged attention and fused MoE, and with
its structured-read options (a per-request seed canvas, one denoising step, logprobs of chosen token ids;
`server_vllm.py --engine inproc` or `--engine http`) a decision took ~60 ms in real rounds. On one B200, a 2.7k-character
text request took **56 ms**. The cost is the generation machinery around the model: a read is an encoder step plus
a decoder step through the scheduler, with per-step bookkeeping and, over HTTP, 64 × N logprobs serialised to JSON.

## 3. DJev-serve's fast path: one forward per decision (`fastpath.py`, `--engine fast`)

DJev-serve keeps vLLM's model code, paged KV cache and kernels, drops the scheduler, and runs exactly one forward per
decision:

- **One CUDA graph from token ids to answer log-probs.** A forward is three pseudo-sequences on the paged KV cache:
  the canvas (non-causal), the uncached prompt tokens (causal), and padding up to a bucket size. Per bucket one CUDA
  graph holds everything from token ids to the candidate log-probs; a read fills two small index buffers, replays the
  graph, and copies back a `[slots × candidates]` array.
- **Prefix cache slots.** A few slots, each with its own KV blocks. A read takes the slot whose last prompt shares the
  longest block-aligned prefix with its own, computes only the rest, and leaves its own prompt's KV in the slot (prompt
  KV is causal, so it never depends on the canvas). The prefix length is data in the index buffers, so the same graphs
  serve any prefix.
- **Faster attention where it counts.** vLLM's Triton unified attention handles this layout but is slow at these
  prefill sizes: ~520 µs per sliding-window layer at 1.6k tokens on a GB200, against ~130 µs for FlashAttention-2
  (`probes/attn_bench*.py`). While a text graph is captured, the 25 sliding-window layers run two FlashAttention-2
  calls instead (the canvas non-causal, then prompt and padding causal and bottom-right aligned, so a cached prefix is
  context). FlashAttention-2 has no head size 512, so the 5 global layers stay on Triton with a launch configuration
  swept for this shape (~25% faster than the default, same results), split into a non-causal and a causal call so the
  causal one skips masked tiles (1.6× faster at 1.6k tokens).
- **Disjoint KV blocks per cache group.** vLLM's hybrid allocator overlays the KV tensors of its sliding-window and
  full-attention groups; every (slot, group) pair and the padding sequence get their own block ids, because sharing
  them corrupts a kept prefix.

Result on one **B200**, the release workload (`probes/stress_text.py`, `DJEV_TIMING=1`): **~26 ms per decision end to
end**, of which 22.8 ms is the GPU forward, 1.0 ms parsing and tokenisation, and under 1 ms inter-process transfer and
input preparation. The same 2.7k-character request that took 56 ms on the original vLLM takes **22 ms**. The
bottleneck is now the GPU.

## 4. Where the GPU time goes (one B200)

Cost of one forward against its token count, no prefix cache (`probes/scale_tokens.py`, bf16, `flashinfer_cutlass` MoE):

| Tokens (incl. the 64-token canvas) | 192 | 576 | 1,088 | 1,600 | 2,112 | 2,496 |
|---|---|---|---|---|---|---|
| Forward | 7.9 ms | 12.2 ms | 17.7 ms | 22.4 ms | 28.3 ms | 32.1 ms |

So a forward costs about **6 ms fixed** (mostly reading the MoE weights once) **plus ~10.4 ms per 1,000 uncached tokens**.

By operator (`probes/profile_forward.py`, eager):

| Category | 1 request (1,408 tokens) | 4 batched (5,632 tokens) |
|---|---|---|
| MoE expert GEMM (CUTLASS grouped GEMM) | 8.8 ms | 17.6 ms |
| Other GEMMs (attention projections, …) | 4.2 ms | 12.3 ms |
| Attention, sliding-window layers (FA2) | 3.8 ms (152 µs/layer) | 9.3 ms |
| Attention, global layers (Triton, head 512) | 2.9 ms (580 µs/layer) | 7.7 ms |
| MoE routing, activation, data movement | 2.1 ms | 8.0 ms |
| Norms and other | 1.7 ms | 5.9 ms |
| **Total** | **23.6 ms** | **61.3 ms (15.3 per request)** |

Replayed from CUDA graphs: 25.0 ms for 1 request, 67.3 ms for 4 batched.

The prefix cache helps less than it could: of the ~1,760 prompt tokens only ~480 hit it, because the per-step content
("Now: …") comes early in the prompt and each bot's name is in the first sentence, so bots share no prefix. That is a
harness change, not a serving one.

## 5. Batching (`--batch-max`, `--batch-wait-ms`)

With one request per forward a GPU saturates at ~43 decisions a second, and every extra bot just queues (each adds
~23 ms). `FastReader.read_many` puts several queued reads into one forward, with one CUDA graph per (bucket, batch
size). In process (`probes/check_batch.py`, 8 bots, 480 cached tokens each):

| Max per batch | Per request | Throughput |
|---|---|---|
| 1 | 23.1 ms | 1.00× |
| 2 | 17.7 ms | 1.30× |
| 3 | 16.2 ms | 1.43× |
| 4 | 15.1 ms | **1.53×** |

End to end over HTTP to one engine, `--batch-max 4`:

| Bots | No batching | Batching | Batching + 4 ms window |
|---|---|---|---|
| 1 | 26.1 ms | 26.4 ms | 26.3 ms |
| 2 | 46.4 ms (43.1 steps/s) | 47.0 ms (42.7) | 43.7 ms (45.9) |
| 4 | 93.4 ms (43.1) | 71.9 ms (55.9) | 68.5 ms (58.4) |
| 8 | 187 ms (43.5) | — | 126 ms (64.4) |

Without a window two bots fall into an alternating rhythm (one computes while the other queues, and the next request
arrives 1–2 ms after the GPU frees up), so they never share a forward; a short wait while the GPU is busy fixes that.
Beyond 4 per batch the gain is small, and with more bots per GPU than prefix slots the cache hit rate drops.

**Answer consistency.** Against an uncached reference, batching changes the top choice no more than the
unbatched cached path does (98–102 of 112 identical vs 101 of 112; median max deviation 0.093–0.098 vs 0.086): bf16
rounding. The model is sensitive to numerical noise to begin with (the same request with a different cache state picks
a different move ~10% of the time), so evaluate any kernel or precision change against that self-noise
(`probes/answers_dump.py --compare`), not against 100% agreement.

## 6. Serving many bots (`gateway.py`)

Each GPU runs its own DJev-serve front; `gateway.py` gives the agents one address, sends each request to the healthy
backend with the fewest requests in flight, checks every backend with a tiny evaluate every 2 s, and retries on the
next one if a backend dies mid-request. Because a bot sends its next request only when the last one has answered,
load is closed-loop. On 8 GPUs with real decisions: 1 client 39 decisions/s at 25 ms; 8 clients 324/s at 25 ms;
24 clients 362/s at a 66 ms median.

## 7. Tried, and not (yet) worth it

| Direction | Result (B200) |
|---|---|
| MoE backend `flashinfer_trtllm` | does not support this model's GELU-tanh activation |
| MoE backend `triton` | ~7% slower than `flashinfer_cutlass` (18.9 vs 17.7 ms at 1,152 tokens) |
| FP8 (vLLM online quantization) | `flashinfer_cutlass` has no per-tensor FP8; on `triton` only 6–7% faster, with accuracy risk |
| Tensor parallel, TP=2 | not supported by the fast path; not faster on the older engine either, and halves the replicas |
| Several replicas or streams per GPU (MPS, green contexts, MIG) | no configuration beat one replica with batch 4 on 70 ms goodput |
| FlashAttention-4 on the sliding layers | attention alone 108 → 44 µs for the prompt part, but needs 128-token KV pages; end to end not yet validated (`FastReader(fa_version=4, kv_block_size=128)`, `probes/answers_dump.py --fa 4 --kv-block 128`) |

Next on the list: NVFP4 weights (GEMMs are more than half of the forward), FA4 end to end, more prefix slots per GPU,
a faster kernel for the head-512 global layers, and prompts designed for the cache.
