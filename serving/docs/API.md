# DJev-serve API

DJev-serve is a Jev-style **decision model** server. You send a state (text, JSON, or images) and a few typed
questions; one forward pass returns the **complete probability distribution** of every question. No text is generated.
The model is Google's open-weight text-diffusion LLM
[DiffusionGemma-26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it); the engine runs on vLLM's kernels.

The request/response contract is the one of [NanoJev](https://github.com/TianyuCodings/NanoJev)
(`POST /api/evaluate`), extended with optional images. A running front also serves this file at `GET /manual`.

## 1. Endpoints and authentication

| Endpoint | Purpose |
|---|---|
| `POST /api/evaluate` | answer the questions of one or more states |
| `GET /api/health` | readiness, workers, running averages |
| `GET /manual` | this document (no key needed) |

When the front is started with `--api-key-file FILE` (or `KEYFILE=FILE serving/serve.sh`), every `/api/*` call needs
the key in that file, either way:

```bash
curl -H "Authorization: Bearer $DJEV_API_KEY" $DJEV_URL/api/health
curl -u "djev:$DJEV_API_KEY" $DJEV_URL/api/health          # basic auth: any user name, the key as password
```

A missing or wrong key gets `401`. Without `--api-key-file` the front is open: keep it on `127.0.0.1` (the default).

## 2. A minimal example

```bash
export DJEV_URL=http://127.0.0.1:8765

curl -s $DJEV_URL/api/evaluate -H 'Content-Type: application/json' -d '{
  "states": [{
    "id": "ticket-1",
    "state": "Customer: Everything is down and we have a demo at noon.",
    "questions": {
      "urgent":   {"type": "boolean", "instructions": "Does the customer need a reply within the hour?"},
      "category": {"type": "choice", "instructions": "What is the ticket about?",
                   "criteria": {"outage": "service is down", "billing": "payment or invoice", "howto": "usage question"}},
      "severity": {"type": "score", "instructions": "How severe is the problem?",
                   "criteria": ["cosmetic", "minor", "major", "critical"]}
    }
  }]}'
```

Response (abridged):

```json
{"states": [{"id": "ticket-1", "answers": {
   "urgent":   {"type": "boolean", "probabilities": {"false": 0.01, "true": 0.99}, "p_true": 0.99, "value": true,  "candidate_mass": 0.99},
   "category": {"type": "choice",  "probabilities": {"outage": 0.98, "billing": 0.0, "howto": 0.02}, "choice": "outage", "value": "outage", "candidate_mass": 1.0},
   "severity": {"type": "score",   "probabilities": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.9}, "score": 2.9, "level": 3, "value": 2.9, "candidate_mass": 0.97}}}],
 "execution": {"states": 1, "questions": 3, "forward_passes": 1, "backend": "vllm", "server_evaluation_seconds": 0.03}}
```

From Python, `serving/client.py` does the same:

```bash
python serving/client.py --url $DJEV_URL                          # built-in text example
python serving/client.py --url $DJEV_URL --input req.json         # your own request
python serving/client.py --url $DJEV_URL --images photo.jpg       # built-in visual questions
python serving/client.py --url $DJEV_URL --concurrency 16 --requests 64   # load test
```

`client.py` takes the key from `--api-key` or the `DJEV_API_KEY` environment variable.

## 3. Request format

`POST /api/evaluate` with the body `{"states": [...]}`. Each state:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | a string unique within the request, echoed back |
| `state` | yes | a string, or a JSON object/array (serialised compactly for the model) |
| `questions` | yes | `{question id: question}`; the ids are not shown to the model, they only key the answers |
| `images` | no | 1 to 8 images, each an `http(s)://` URL or a `data:image/...;base64,...` URI (at most 20 MB) |

Question types:

| `type` | `criteria` | Answer |
|---|---|---|
| `boolean` | optional, or `{"true": "when it is yes", "false": "when it is no"}` | `probabilities: {false, true}`, `p_true`, `value` (`p_true >= 0.5`) |
| `choice` | `{option id: description}`, 2 to 52 options | a probability per option, `choice` (the most likely id) |
| `score` | an ordered list of 2 to 10 level descriptions, low to high | a probability per level (keys `"0"` … `"n-1"`), `score` (expected level), `level` (most likely) |

Every answer carries `candidate_mass`: the probability the model put on the candidate answers at the answer slot,
before renormalisation. **Trust an answer only when it is close to 1.** Well below 1 (say < 0.5) means the model wanted
to write something else (asked whether a fair coin will land heads, it wants to write "unknown"); treat that
distribution as a hint at most.

All questions of a state are answered by the same forward pass; the states of a request are processed in parallel.

Limits: 64 states per request, about 4,000 prompt tokens per state (an image is about 270), at most 52 options per
choice and 128 candidate tokens per question. A state with more than 7 questions is read in several passes on a
64-token canvas (the `reads` field of the response); the answers do not change.

## 4. Images

Add `images` to a state and refer to them as **Image 1, Image 2, …** (in array order) in `state` and `instructions`:

```bash
IMG=$(base64 < photo.png | tr -d '\n')
curl -s $DJEV_URL/api/evaluate -H 'Content-Type: application/json' -d '{
  "states": [{
    "id": "img-1",
    "state": "Image 1 is a photo from the warehouse camera.",
    "images": ["data:image/png;base64,'"$IMG"'"],
    "questions": {
      "person":  {"type": "boolean", "instructions": "Is there a person in Image 1?"},
      "content": {"type": "choice", "instructions": "What does Image 1 mainly show?",
                  "criteria": {"people": "people", "vehicle": "a vehicle", "boxes": "boxes or shelves", "empty": "an empty area"}}
    }
  }]}'
```

- Prefer **data URIs**. A URL is downloaded by the server first, which adds hundreds of milliseconds to seconds; a
  failed download returns `400`.
- Each worker caches the vision encoding of its last 256 images by content hash, so a repeated image skips the
  vision encoder.
- Images only; no video.

## 5. Writing good questions

- Make every `instructions` self-contained: questions do not see each other's answers, so never write "based on the
  previous question".
- Describe options clearly; the model sees `id: description`, and the id is only a label.
- This is a System-One model, with no reasoning tokens. It is strong at reading, classification and visual judgement;
  **it is not reliable at arithmetic or multi-step logic** (asked whether 7 + 12 is greater than 20, it said no with
  p = 0.94). Split anything that needs reasoning into simple questions and keep the logic in your own code.

## 6. Performance

Fast engine (`--engine fast`), client on the same machine with a reused connection, requests replayed from real
agent logs one at a time, **one NVIDIA GB200**:

| Request | Prompt tokens | Reused prefix | p50 |
|---|---|---|---|
| Block UHC harness `p_uhc_pro_v10` (1 question, 16 options) | ~1,720 | ~860 | **19.9 ms** |
| Sumo harness `p_sumo_v1` (1 question, 14 options) | ~1,480 | 512 | **20.7 ms** |
| `p_uhc_raw_v3` (4 questions, 57 options) | ~2,130 | 384 | **29.6 ms** |
| 1 state, 3 short text questions (the `client.py` example) | short | | **8 ms** |
| 1 image + 3 questions, image seen before | | | **15 ms** |
| 1 image + 3 questions, a new 256×256 image every time | | | ~41 ms (vision encoder ~21 ms) |

| Concurrent clients (text) | p50 | Throughput |
|---|---|---|
| 2 | 12 ms | 163 req/s |
| 4 | 17 ms | 174 req/s |
| 16 | 47 ms | 176 req/s |

On **one NVIDIA B200**, with the Block UHC workload of the public release (each bot its own name, ~1,764 prompt
tokens, ~480 of them cached), a step takes ~26 ms end to end, of which 22.8 ms is the GPU forward; batching up to 4
queued reads per forward (`--batch-max 4`) raises a GPU from ~43 to 58–64 steps a second. See
[PERFORMANCE.md](PERFORMANCE.md) for the breakdown.

**Prefix cache (automatic).** Each GPU keeps a few prefix slots (`--prefix-slots`, default 8). A read finds the slot
whose last prompt shares the longest (block-aligned) beginning with its own and computes only the rest. So **put
whatever is identical every time at the very start of `state`** (rules, techniques, instructions) and the changing
parts (the current situation, counters, time) after it. After the first differing token everything is recomputed,
even if long stretches after it are unchanged. Each state in the response reports `cached_prompt_tokens`; every 100
tokens not recomputed save roughly 0.7–1 ms.

- Reuse connections (`httpx.Client` / `requests.Session` in Python); a new connection per request costs ~10 ms.
- `DJEV_TIMING=1` in the server's environment adds per-read stage timings to every response.
