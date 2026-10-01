#!/usr/bin/env python3
"""Compare vLLM engine configurations for djev: small-batch latency + answer quality.

Runs the same readout as `server_vllm.py --engine inproc` (InprocReader) without HTTP, on one GPU:
  - quality: 4 states / 18 questions with known answers (text + one synthetic image): accuracy,
    mean p(correct), mean candidate_mass; per-question distributions are saved so configs can be
    compared against the bf16 baseline (mean total-variation distance).
  - latency: sequential single requests (text state, image state), then 4 and 16 concurrent.

    python probes/bench_engine.py --tag bf16-triton
    python probes/bench_engine.py --tag nvfp4 --model nvidia/diffusiongemma-26B-A4B-it-NVFP4 --moe auto
    python probes/bench_engine.py --tag fp8 --quantization fp8
Results: probes/results/<tag>.json (+ a one-line summary on stdout).
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import server as S  # noqa: E402
import server_vllm as SV  # noqa: E402
from client import EXAMPLE  # noqa: E402


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def cases(shapes_uri):
    return [
        ({"id": "shapes5", "state": "Image 1 is a simple synthetic picture on a white background.", "images": [shapes_uri],
          "questions": {
              "red_shape": {"type": "choice", "instructions": "Which shape in Image 1 is red?",
                            "criteria": {"triangle": "a triangle", "circle": "a circle", "square": "a square"}},
              "has_circle": {"type": "boolean", "instructions": "Does Image 1 contain a circle?"},
              "has_square": {"type": "boolean", "instructions": "Does Image 1 contain a square?"},
              "count": {"type": "choice", "instructions": "How many shapes are in Image 1?",
                        "criteria": {"one": "exactly one shape", "two": "exactly two shapes", "three": "exactly three shapes"}},
              "circle_side": {"type": "choice", "instructions": "On which side of Image 1 is the blue shape?",
                              "criteria": {"left": "left half", "right": "right half"}}}},
         {"red_shape": "triangle", "has_circle": "true", "has_square": "false", "count": "two", "circle_side": "right"}),
        (EXAMPLE["states"][0], {"device": "unit_1", "execute": "true", "risk": "0"}),
        (EXAMPLE["states"][1], {"north_safe": "false", "west_safe": "true", "move": "south"}),
        ({"id": "arith", "state": "x = 7, y = 12", "questions": {
            "bigger": {"type": "choice", "instructions": "Which variable is larger?", "criteria": {"x": "the variable x", "y": "the variable y"}},
            "sum_gt_15": {"type": "boolean", "instructions": "Is x + y greater than 15?"},
            "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]},
            "x_odd": {"type": "boolean", "instructions": "Is x an odd number?"},
            "y_even": {"type": "boolean", "instructions": "Is y an even number?"},
            "prod": {"type": "choice", "instructions": "What is x times y?", "criteria": {"a": "74", "b": "84", "c": "94"}},
            "diff": {"type": "choice", "instructions": "What is y minus x?", "criteria": {"a": "3", "b": "5", "c": "7"}}}},
         {"bigger": "y", "sum_gt_15": "true", "size": "1", "x_odd": "true", "y_even": "true", "prod": "b", "diff": "b"}),
    ]


TEXT_STATE = {"id": "ticket-1", "state": "Customer: Everything is down and we have a demo at noon.", "questions": {
    "urgent": {"type": "boolean", "instructions": "Does the customer need a reply within the hour?"},
    "category": {"type": "choice", "instructions": "What is the ticket about?",
                 "criteria": {"outage": "service is down", "billing": "payment or invoice", "howto": "usage question"}},
    "severity": {"type": "score", "instructions": "How severe is the problem?",
                 "criteria": ["cosmetic", "minor", "major", "critical"]}}}


def image_state(uri):
    return {"id": "img-1", "state": "Image 1 is attached.", "images": [uri], "questions": {
        "circle": {"type": "boolean", "instructions": "Does Image 1 contain a circle?"},
        "red": {"type": "boolean", "instructions": "Is there anything red in Image 1?"},
        "count": {"type": "choice", "instructions": "How many filled shapes are in Image 1?",
                  "criteria": {"one": "one", "two": "two", "three": "three"}}}}


async def evaluate(reader, enc, state):
    """What the server does per state, minus HTTP: decode images, build, read, read out."""
    imgs = [S.load_image(u) for u in state.get("images") or []]
    prompt, params, slots, off, _ = reader.build(state, enc, imgs)
    out = await reader.read(prompt, params)
    return slots, reader.answers(slots, reader.rows(out), off)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=S.MODEL_DEFAULT)
    ap.add_argument("--tokenizer", default=S.MODEL_DEFAULT)
    ap.add_argument("--moe", default="triton")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--ids", choices=["all", "main"], default="all", help="main: one spelling per candidate")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--served-canvas", type=int, default=256)
    ap.add_argument("--no-async", action="store_true", help="sync scheduling (needs served canvas == width)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--shapes", default=os.path.expanduser("~/tmp/djev/shapes.png"))
    ap.add_argument("--img", default=os.path.expanduser("~/tmp/djev/img256.png"))
    ap.add_argument("--extra", default="{}", help="JSON dict of extra AsyncEngineArgs")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30)
    if a.ids == "main":   # keep only the first spelling of each candidate (" A", " yes", " no")
        enc.label_groups = [g[:1] for g in enc.label_groups]
        enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]

    kw = dict(model=a.model, tokenizer=a.tokenizer, served_model_name=["djev"],
              diffusion_config={"canvas_length": a.served_canvas}, max_logprobs=128, async_scheduling=not a.no_async,
              enable_prefix_caching=True, gpu_memory_utilization=0.85, max_model_len=4096,
              limit_mm_per_prompt={"image": 8}, data_parallel_size=a.dp, kv_cache_dtype=a.kv_cache_dtype)
    if a.moe != "auto":
        kw["kernel_config"] = {"moe_backend": a.moe}
    if a.quantization:
        kw["quantization"] = a.quantization
    kw.update(json.loads(a.extra))
    t0 = time.time()
    reader = SV.InprocReader(AsyncEngineArgs(**kw), a.served_canvas, a.width, a.constrained)
    load_s = time.time() - t0
    print(f"[{a.tag}] engine ready in {load_s:.0f}s", flush=True)

    shapes_uri, img_uri = data_uri(a.shapes), data_uri(a.img)
    res = {"tag": a.tag, "args": vars(a), "load_seconds": round(load_s, 1)}

    # ---- quality
    n = ok = 0
    pc, ms, dists = [], [], {}
    for st, truth in cases(shapes_uri):
        slots, ans = await evaluate(reader, enc, st)
        for slot, (probs, mass) in zip(slots, ans):
            d = dict(zip(slot.keys, probs))
            dists[f"{st['id']}/{slot.qid}"] = d
            t = truth[slot.qid]
            n += 1
            ok += int(max(d, key=d.get) == t)
            pc.append(d[t])
            ms.append(mass)
    res["quality"] = {"accuracy": f"{ok}/{n}", "mean_p_correct": round(statistics.mean(pc), 4),
                      "mean_mass": round(statistics.mean(ms), 4), "min_mass": round(min(ms), 4)}
    res["distributions"] = dists

    # ---- latency
    async def seq(state, reps):
        for _ in range(3):
            await evaluate(reader, enc, state)
        xs = []
        for _ in range(reps):
            t = time.perf_counter()
            await evaluate(reader, enc, state)
            xs.append((time.perf_counter() - t) * 1000)
        return {"p50": round(statistics.median(xs), 1), "p90": round(pct(xs, 0.9), 1), "min": round(min(xs), 1)}

    async def conc(state, c, rounds):
        await asyncio.gather(*[evaluate(reader, enc, state) for _ in range(c)])
        lat = []

        async def one():
            t = time.perf_counter()
            await evaluate(reader, enc, state)
            lat.append((time.perf_counter() - t) * 1000)
        t0 = time.perf_counter()
        for _ in range(rounds):
            await asyncio.gather(*[one() for _ in range(c)])
        wall = time.perf_counter() - t0
        return {"p50": round(statistics.median(lat), 1), "p90": round(pct(lat, 0.9), 1),
                "states_per_s": round(len(lat) / wall, 1)}

    img_st = image_state(img_uri)
    res["latency_ms"] = {
        "text_b1": await seq(TEXT_STATE, a.reps),
        "image_b1": await seq(img_st, a.reps),
        "text_c4": await conc(TEXT_STATE, 4, 8),
        "text_c16": await conc(TEXT_STATE, 16, 4),
        "image_c4": await conc(img_st, 4, 8),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    L, Q = res["latency_ms"], res["quality"]
    print(f"RESULT {a.tag}: text b1 p50 {L['text_b1']['p50']} ms | image b1 p50 {L['image_b1']['p50']} ms | "
          f"text c4 p50 {L['text_c4']['p50']} ms | text c16 {L['text_c16']['states_per_s']} st/s | "
          f"acc {Q['accuracy']} p(correct) {Q['mean_p_correct']} mass {Q['mean_mass']}", flush=True)
    reader.engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
