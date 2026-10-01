#!/usr/bin/env python3
"""Image reads on fastpath.FastReader vs the live djev server (a handful of single requests), plus timing.

    python probes/bench_fastpath_img.py [--ref-port 8767]
"""
import argparse
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import server as S  # noqa: E402
from client import image_source  # noqa: E402

CAT = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-port", type=int, default=8767)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import fastpath
    import httpx
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    shapes = image_source(os.path.expanduser("~/tmp/djev/shapes.png"))
    img256 = image_source(os.path.expanduser("~/tmp/djev/img256.png"))
    states = [
        {"id": "shapes", "state": "Image 1 is a simple synthetic picture on a white background.", "images": [shapes], "questions": {
            "red_shape": {"type": "choice", "instructions": "Which shape in Image 1 is red?", "criteria": {"triangle": "a triangle", "circle": "a circle", "square": "a square"}},
            "has_circle": {"type": "boolean", "instructions": "Does Image 1 contain a circle?"},
            "has_square": {"type": "boolean", "instructions": "Does Image 1 contain a square?"},
            "count": {"type": "choice", "instructions": "How many shapes are in Image 1?", "criteria": {"one": "exactly one shape", "two": "exactly two shapes", "three": "exactly three shapes"}},
            "circle_side": {"type": "choice", "instructions": "On which side of Image 1 is the blue shape?", "criteria": {"left": "left half", "right": "right half"}}}},
        {"id": "cat", "state": "Image 1 is a photo.", "images": [CAT], "questions": {
            "animal": {"type": "choice", "instructions": "Which animal is in Image 1?", "criteria": {"cat": "a cat", "dog": "a dog", "bird": "a bird", "none": "no animal"}},
            "outdoors": {"type": "boolean", "instructions": "Was Image 1 taken outdoors?"},
            "cuteness": {"type": "score", "instructions": "How cute is the subject of Image 1?", "criteria": ["not cute", "somewhat cute", "cute", "very cute"]}}},
        {"id": "two", "state": "Two images are attached.", "images": [shapes, CAT], "questions": {
            "which_photo": {"type": "choice", "instructions": "Which image is a photograph?", "criteria": {"first": "Image 1", "second": "Image 2"}},
            "same": {"type": "boolean", "instructions": "Do Image 1 and Image 2 show the same thing?"}}},
        {"id": "img256", "state": "Image 1 is attached.", "images": [img256], "questions": {
            "circle": {"type": "boolean", "instructions": "Does Image 1 contain a circle?"},
            "red": {"type": "boolean", "instructions": "Is there anything red in Image 1?"},
            "count": {"type": "choice", "instructions": "How many filled shapes are in Image 1?", "criteria": {"one": "one", "two": "two", "three": "three"}}}},
    ]
    key = open(os.path.join(os.path.dirname(HERE), ".djev-api-key")).read().strip()
    ref = httpx.post(f"http://127.0.0.1:{a.ref_port}/api/evaluate", json={"states": states},
                     headers={"Authorization": "Bearer " + key}, timeout=120).json()
    ref = {s["id"]: s["answers"] for s in ref["states"]}

    t0 = time.time()
    fr = fastpath.FastReader(moe=a.moe, thought_ids=enc.thought_prefix_ids)
    print(f"FastReader ready in {time.time()-t0:.0f}s; vision attention was {fr.vision_attn!r}", flush=True)
    worst, prepared = 0.0, []
    for st in states:
        raw = [S.load_image_bytes(u) for u in st["images"]]
        text, slots = enc.prompt_text(st, len(raw))
        body = enc.scaffold(slots, st["id"])
        canvas = body + [S.PAD_ID] * (fr.C - len(body))
        cands = [[i for g in s.cand_groups for i in g] for s in slots]
        args = (text, raw, canvas, [s.pos for s in slots], cands)
        prepared.append((st, args))
        lp, n = fr.read_image(*args)
        print(f"== {st['id']}  prompt {n} tok")
        for s, row in zip(slots, lp):
            p = [math.exp(x) for x in row]
            mass = sum(p)
            nrm = [x / mass for x in p]
            r = [ref[st["id"]][s.qid]["probabilities"][k] for k in s.keys]
            d = max(abs(x - y) for x, y in zip(nrm, r))
            worst = max(worst, d)
            print(f"   {s.qid:11s} fast {[round(x, 3) for x in nrm]} mass {mass:.3f} | served {[round(x, 3) for x in r]} | max|diff| {d:.3f}")
    print(f"WORST max|diff| vs served: {worst:.3f}", flush=True)
    import io
    from PIL import Image
    import numpy as np
    for st, args in (prepared[3], prepared[1]):
        # new images every time (no embedding cache hit): perturb one pixel so the bytes differ
        base = Image.open(io.BytesIO(args[1][0])).convert("RGB")
        ys = []
        for i in range(15):
            im = np.array(base)
            im[0, 0, 0] = i % 255
            b = io.BytesIO()
            Image.fromarray(im).save(b, format="PNG")
            a2 = (args[0], [b.getvalue()], *args[2:])
            t = time.perf_counter()
            fr.read_image(*a2)
            ys.append((time.perf_counter() - t) * 1000)
        ys.sort()
        parts = {k: round(v * 1000, 2) for k, v in fr.timing.items()}
        print(f"TIMING image read {st['id']} NEW image each time: p50 {statistics.median(ys[3:]):.2f} ms  breakdown {parts}", flush=True)
        for _ in range(5):
            fr.read_image(*args)
        xs = []
        for _ in range(30):
            t = time.perf_counter()
            fr.read_image(*args)
            xs.append((time.perf_counter() - t) * 1000)
        xs.sort()
        parts = {k: round(v * 1000, 2) for k, v in fr.timing.items()}
        print(f"TIMING image read {st['id']} REPEATED image (cache hit): p50 {statistics.median(xs):.2f} ms  min {xs[0]:.2f}  last breakdown {parts}", flush=True)


if __name__ == "__main__":
    main()
