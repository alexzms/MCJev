# DJev-serve

**DJev** turns Google's open-weight text-diffusion LLM
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) into a
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style decision model: a state and typed
questions in, complete probability distributions out, every question of a state answered by **one forward pass**,
zero tokens generated. **DJev-serve** is the engine that serves it fast enough for a Minecraft PvP bot: ~24 ms per
decision on one Blackwell GPU, 2.5× faster than the same model on the original vLLM and 7.2× faster than in
Hugging Face transformers.

The HTTP contract (`POST /api/evaluate`) is the one of [NanoJev](https://github.com/TianyuCodings/NanoJev), so
NanoJev clients can be pointed at a DJev-serve front.

## How a decision is read

DiffusionGemma pairs a causal encoder, which prefills the prompt into a KV cache, with a bidirectional decoder that
denoises a "canvas" of answer tokens conditioned on it. Normal generation starts from a noised canvas and denoises it
over many refinement steps. DJev instead writes an **answer scaffold** into the canvas and leaves only the answer
slots noised:

```
prompt (user turn)                    canvas (model turn)
──────────────────────────────        ─────────────────────────────────────────────
State: ...                            1: <mask>     <- one slot per question
Question 1 (choice): ...              2: <mask>
  A) ...  B) ...                      <turn|> <pad> <pad> ...
Question 2 (yes/no): ...
Answer format: 1: <letter> ...
```

One decoder pass denoises every slot at once; each slot sees the whole prompt and the whole canvas. At each slot the
logits are restricted to that question's candidate tokens (` A`, ` B`, … or ` yes`, ` no`, spelling variants summed)
and renormalised: that is the answer distribution. The un-renormalised mass on the candidates is returned as
`candidate_mass` (≈ 1 means the model answered within the options).

Two details matter (both found with `probes/probe-slot-topk.py`):

- Gemma replies open with an empty thinking block when thinking is off. DJev appends it to the **prompt**, so the
  canvas starts directly with `1:`; without it the model writes that block into the canvas, every slot shifts by four
  tokens and reads garbage (mass 0–0.5 instead of ≈ 1).
- The slot holds the `<mask>` token, read in a single pass. A uniform-random noise token or extra self-conditioning
  passes ruin the readout (`probes/probe-readout.py`, log in `probes/probe-readout.log`).

Question types: `choice` (2–52 options), `boolean` (`p_true`) and `score` (2–10 ordered levels, with the expected
level). It is a System-One model: strong at reading, classification and visual judgement, not at arithmetic or
multi-step logic. Full API: [docs/API.md](docs/API.md).

## Three engines, one API

| Engine | File | Per decision | What it is |
|---|---|---|---|
| reference | `server.py` | ~174 ms | the model in Hugging Face transformers (eager); one worker process per GPU |
| vLLM | `server_vllm.py --engine inproc` / `http` | ~60 ms | vLLM's own engine with its DiffusionGemma structured-read options (an encoder step and a decoder step through the scheduler) |
| **fast** | `server_vllm.py --engine fast` + `fastpath.py` | **~24 ms** | one forward per decision on vLLM's model code, paged KV cache and kernels, no scheduler: one CUDA graph per bucket from token ids to answer log-probs, prefix-cache slots, FlashAttention-2 on the sliding-window layers, a tuned Triton launch for the global layers, and batching of queued reads |

The reference number is one forward of a short text request on a GB200; the vLLM and fast numbers are real Block
UHC decisions (~1.7k-token prompts) in live rounds. How the fast path got there, what the GPU time is spent on, and
what did not help: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Install

You need an NVIDIA GPU with ~50 GB of memory for the bf16 weights (we used GB200 and B200) and CUDA 13.

```bash
# 1. vLLM nightly with DiffusionGemma structured reads (vLLM PRs #57250, #58216). fastpath.py patches vLLM
#    internals, so match the version it was built against: 0.30.1rc1.dev192+g5840d9528 (torch 2.13.0+cu130).
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
# 2. the rest
pip install -r serving/requirements.txt
# 3. the weights (~50 GB)
huggingface-cli download google/diffusiongemma-26B-A4B-it
```

## Run

```bash
serving/serve.sh                         # fast engine on GPU 0, http://127.0.0.1:8765 (the release configuration)
GPUS=1 PORT=8766 serving/serve.sh        # another front on GPU 1
python serving/client.py                 # built-in example: prints the distributions
python serving/client.py --concurrency 16 --requests 64   # load test
curl -s http://127.0.0.1:8765/api/health | python -m json.tool
```

`serve.sh` runs `server_vllm.py --engine fast --served-canvas 64 --moe-backend flashinfer_cutlass --batch-max 4
--batch-wait-ms 4 --prefix-slots 8`, the configuration the public release ran with, one front per GPU. The first
start takes ~6 minutes (FlashInfer autotuning, CUDA-graph capture), ~2 minutes with warm caches; the front prints
`{"ready": true}` when it accepts requests. `ENGINE=inproc` or `ENGINE=http` selects the vLLM engines, and
`KEYFILE=path` requires an API key on `/api/*`. The reference server is `python serving/server.py --gpus 0 --port 8765`.

Many fronts behind one address, as the bots use it:

```bash
python serving/gateway.py --listen 127.0.0.1:8780 --backends-file serving/backends.example.txt
export JEV_URL=http://127.0.0.1:8780     # what the harness reads
```

The gateway sends each request to the healthy front with the fewest requests in flight, health-checks every front
with a tiny evaluate every 2 s, and retries on another front if one fails mid-request (`GET /gateway/status`).

## Probes

The experiments behind the design, each runnable on its own (most need a GPU and the weights):

| Probe | What it measures |
|---|---|
| `probe-readout.py`, `probe-slot-topk.py` | readout configurations (mask vs random slot noise, extra passes) and where the slot mass goes |
| `single_pass_check.py`, `debug_fastpath.py`, `fastpath_proto.py` | the fast path's one-forward read against vLLM's two-step read |
| `profile-stages.py`, `profile_steps.py`, `profile_forward.py` | stage, step and per-operator timings |
| `attn_bench*.py`, `attn_fa4.py` | Triton unified attention vs FlashAttention-2 / -4 on this model's shapes |
| `bench_engine.py`, `bench_fastpath.py`, `bench_fastpath_img.py`, `bench_prefix.py`, `workload_prefix.py`, `scale_tokens.py` | engine, prefix-cache and token-count scaling benchmarks (results in `probes/results/`) |
| `check_batch.py`, `answers_dump.py` | batching speed and answer consistency against an uncached reference |
| `stress_text.py`, `v10.py`, `profile_mc_reads.py` | many bots replaying a real Block UHC harness step (`probes/data/uhc_pro_step85.json`) through a front or the gateway |
| `visual-test.py` | image questions with known answers against a running front |

## Files

| File | |
|---|---|
| `server.py` | request validation (the NanoJev contract plus images), prompt and canvas encoding, the reference transformers server |
| `server_vllm.py` | the DJev-serve front (FastAPI) and its three engines; batching of queued reads for the fast engine |
| `fastpath.py` | the fast path: `FastReader`, one CUDA graph per bucket and batch size, prefix-cache slots, attention hooks |
| `client.py` | example requests, image questions, a concurrent load test |
| `gateway.py`, `backends.example.txt` | one address for many fronts (standard library only) |
| `serve.sh` | launcher for one front |
| `docs/API.md`, `docs/PERFORMANCE.md` | the API manual (also served at `GET /manual`) and the performance notes |

## Acknowledgements

DJev builds on [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Google, open weights),
[vLLM](https://github.com/vllm-project/vllm) (model implementation, paged KV cache, kernels),
[FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
and the request contract of [NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT), from which `validate_request`
in `server.py` is adapted. The decision-model idea is TypeSafe AI's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). See the repository README for citations.
