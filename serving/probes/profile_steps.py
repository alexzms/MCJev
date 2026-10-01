#!/usr/bin/env python3
"""Where does one djev request's time go?  Per-iteration engine log + request timings, for
(a) a repeated prompt (prefix cache hit) and (b) a fresh prompt each time (what an agent sends).

    python probes/profile_steps.py [--moe triton] [--model ...]
Engine iteration lines ("Iteration(...)") go to stderr; request timings are printed as REQ lines.
"""
import argparse
import asyncio
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402  (NVTX markers around requests, for nsys)
import server as S  # noqa: E402
import server_vllm as SV  # noqa: E402

BASE = {"id": "t", "state": "Customer: Everything is down and we have a demo at noon.", "questions": {
    "urgent": {"type": "boolean", "instructions": "Does the customer need a reply within the hour?"},
    "category": {"type": "choice", "instructions": "What is the ticket about?",
                 "criteria": {"outage": "service is down", "billing": "payment or invoice", "howto": "usage question"}},
    "severity": {"type": "score", "instructions": "How severe is the problem?",
                 "criteria": ["cosmetic", "minor", "major", "critical"]}}}


def fresh(i):
    s = dict(BASE)
    s["state"] = f"Ticket #{random.randint(10000, 99999)}-{i}: " + random.choice([
        "Everything is down and we have a demo at noon.", "I was charged twice this month.",
        "How do I export my data to CSV?", "The dashboard loads slowly since yesterday.",
        "Our API key stopped working after the rotation.", "Can you add dark mode?"]) + f" (customer {i})"
    return s


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.MODEL_DEFAULT)
    ap.add_argument("--moe", default="triton")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30)
    eargs = AsyncEngineArgs(model=a.model, tokenizer=S.MODEL_DEFAULT, diffusion_config={"canvas_length": 256},
                            max_logprobs=128, async_scheduling=True, enable_prefix_caching=True,
                            kernel_config={"moe_backend": a.moe} if a.moe != "auto" else None,
                            gpu_memory_utilization=0.85, max_model_len=4096, limit_mm_per_prompt={"image": 8},
                            disable_log_stats=False, enable_logging_iteration_details=True)
    reader = SV.InprocReader(eargs, 256, 64, False)

    async def one(state):
        t0 = time.perf_counter()
        prompt, params, slots, off, _ = reader.build(state, enc)
        t1 = time.perf_counter()
        out = await reader.read(prompt, params)
        t2 = time.perf_counter()
        reader.answers(slots, reader.rows(out), off)
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, len(prompt["prompt_token_ids"])

    for i in range(5):
        await one(BASE)
        await one(fresh(1000 + i))
    for name, gen in (("repeated", lambda i: BASE), ("fresh", fresh)):
        print(f"=== {name} ===", file=sys.stderr, flush=True)
        rows = []
        for i in range(a.n):
            await asyncio.sleep(a.sleep)
            print(f"--- REQ {name} {i} start", file=sys.stderr, flush=True)
            torch.cuda.nvtx.range_push(f"REQ {name} {i}")
            r = await one(gen(i))
            torch.cuda.nvtx.range_pop()
            print(f"--- REQ {name} {i} end build {r[0]:.2f} engine {r[1]:.1f} readout {r[2]:.2f} ms, prompt {r[3]} tok",
                  file=sys.stderr, flush=True)
            rows.append(r)
        print(f"REQ {name}: engine p50 {statistics.median(x[1] for x in rows):.1f} ms, build p50 "
              f"{statistics.median(x[0] for x in rows):.2f} ms, readout p50 {statistics.median(x[2] for x in rows):.2f} ms",
              flush=True)
    reader.engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
