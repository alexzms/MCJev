#!/usr/bin/env python3
"""Check fastpath.FastReader against the live djev server and time single reads.

    python probes/bench_fastpath.py [--ref-port 8767] [--moe flashinfer_cutlass] [--no-prefix]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import server as S  # noqa: E402
from client import EXAMPLE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-port", type=int, default=8767)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import fastpath
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]

    def prompt_ids(text):
        p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                    return_dict=False)
        return (p["input_ids"] if isinstance(p, dict) else list(p)) + list(enc.thought_prefix_ids)

    # fixed prefix = the chat-template start + preamble, i.e. what every text prompt begins with
    probe_text, _ = enc.prompt_text({"id": "p", "state": "\u0000", "questions": {"q": {"type": "boolean", "instructions": "x"}}})
    head = probe_text.split("\u0000")[0]
    pref = tok.apply_chat_template([{"role": "user", "content": head}], tokenize=True, add_generation_prompt=False,
                                   return_dict=False)
    pref = pref["input_ids"] if isinstance(pref, dict) else list(pref)
    # keep only the part that is a token-exact prefix of real prompts
    sample = prompt_ids(enc.prompt_text(S.validate_request(EXAMPLE)[0])[0])
    n = 0
    while n < min(len(pref), len(sample)) and pref[n] == sample[n]:
        n += 1
    prefix = sample[:n]
    print(f"common prefix: {n} tokens", flush=True)

    t0 = time.time()
    fr = fastpath.FastReader(moe=a.moe, prefix_ids=None if a.no_prefix else prefix)
    print(f"FastReader ready in {time.time()-t0:.0f}s: prefix cached {fr.P0} tokens, {len(fr.graphs)} graphs "
          f"({fr.buckets[0]}..{fr.buckets[-1]})", flush=True)

    states = S.validate_request(EXAMPLE) + [
        {"id": "arith", "state": "x = 7, y = 12", "questions": {
            "bigger": {"type": "choice", "instructions": "Which variable is larger?", "criteria": {"x": "the variable x", "y": "the variable y"}},
            "x_odd": {"type": "boolean", "instructions": "Is x an odd number?"},
            "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]}}},
        {"id": "ticket", "state": "Customer: Everything is down and we have a demo at noon.", "questions": {
            "urgent": {"type": "boolean", "instructions": "Does the customer need a reply within the hour?"},
            "category": {"type": "choice", "instructions": "What is the ticket about?",
                         "criteria": {"outage": "service is down", "billing": "payment or invoice", "howto": "usage question"}},
            "severity": {"type": "score", "instructions": "How severe is the problem?", "criteria": ["cosmetic", "minor", "major", "critical"]}}}]
    import httpx
    key = open(os.path.join(os.path.dirname(HERE), ".djev-api-key")).read().strip()
    ref = httpx.post(f"http://127.0.0.1:{a.ref_port}/api/evaluate", json={"states": states},
                     headers={"Authorization": "Bearer " + key}, timeout=120).json()
    ref = {s["id"]: s["answers"] for s in ref["states"]}

    worst, prepared = 0.0, []
    for st in states:
        text, slots = enc.prompt_text(st)
        body = enc.scaffold(slots, st["id"])
        canvas = body + [S.PAD_ID] * (fr.C - len(body))
        pids = prompt_ids(text)
        cands = [[i for g in s.cand_groups for i in g] for s in slots]
        prepared.append((st, slots, pids, canvas, cands))
        lp = fr.read(pids, canvas, [s.pos for s in slots], cands)
        print(f"== {st['id']}  prompt {len(pids)} tok ({len(pids) - fr.P0} after prefix)")
        for s, row in zip(slots, lp):
            p = [math.exp(x) for x in row]
            mass = sum(p)
            nrm = [x / mass for x in p]
            r = [ref[st["id"]][s.qid]["probabilities"][k] for k in s.keys]
            d = max(abs(x - y) for x, y in zip(nrm, r))
            worst = max(worst, d)
            print(f"   {s.qid:11s} fast {[round(x, 3) for x in nrm]} mass {mass:.3f} | served {[round(x, 3) for x in r]} | max|diff| {d:.3f}")
    print(f"WORST max|diff| vs served: {worst:.3f}", flush=True)

    for name, idx in (("ticket (3 q)", 3), ("maze (3 q)", 1)):
        st, slots, pids, canvas, cands = prepared[idx]
        for _ in range(10):
            fr.read(pids, canvas, [s.pos for s in slots], cands)
        xs = []
        for _ in range(a.reps):
            t = time.perf_counter()
            fr.read(pids, canvas, [s.pos for s in slots], cands)
            xs.append((time.perf_counter() - t) * 1000)
        xs.sort()
        print(f"TIMING {name}: prompt {len(pids)} tok -> {len(pids) - fr.P0 + fr.C} in forward: "
              f"p50 {statistics.median(xs):.2f} ms  p90 {xs[int(0.9 * (len(xs) - 1))]:.2f}  min {xs[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
