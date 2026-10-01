#!/usr/bin/env python3
"""How much of the real agents' prompts a prefix KV cache can reuse (CPU only: tokenizer, no model).

For each run log (jevagent runs/*.jsonl, "step" records hold the state and questions sent to djev), with the prompt
built exactly as the server does:
  * prompt tokens per step
  * reusable now: longest common prefix with any of the last 8 prompts (what fastpath's 8 prefix slots give)
  * fixed prefix: common prefix of all steps in the run
  * upper bound if the state were ordered stable-first: tokens in lines unchanged from the previous step
  * per section ("== Name ==" blocks, plus the question): how often it changes between consecutive steps

    python probes/workload_prefix.py RUN.jsonl [RUN.jsonl ...] [--max 1500]
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import server as S  # noqa: E402

SECTION = re.compile(r"^== (.+?) ==$", re.M)


def sections(state_text):
    """[(name, text)] of the state's '== Name ==' blocks; text before the first header is 'preamble'."""
    out, last, name = [], 0, "preamble"
    for m in SECTION.finditer(state_text):
        out.append((name, state_text[last:m.start()]))
        name, last = m.group(1), m.start()
    out.append((name, state_text[last:]))
    return out


def lcp(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--slots", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tk, 1 << 30, empty_thought=True)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    tok_cache = {}

    def ntok(s):
        if s not in tok_cache:
            tok_cache[s] = len(tk.encode(s, add_special_tokens=False))
        return tok_cache[s]

    for path in a.runs:
        steps = []
        harness = "?"
        for line in open(path):
            d = json.loads(line)
            if d.get("kind") == "step" and d.get("state"):
                harness = d.get("harness", harness)
                steps.append(d)
        steps = steps[-a.max:]
        if len(steps) < 10:
            continue
        ids, texts, states, questions = [], [], [], []
        for d in steps:
            st = S.validate_request({"states": [{"id": "s", "state": d["state"], "questions": d["questions"]}]})[0]
            text, _ = enc.prompt_text(st, 0)
            texts.append(text)
            ids.append(tk.encode(text, add_special_tokens=False))
            states.append(st["state"] if isinstance(st["state"], str) else json.dumps(st["state"]))
            questions.append(json.dumps(d["questions"], sort_keys=True))
        n = [len(x) for x in ids]
        fixed = ids[0]
        for x in ids[1:]:
            fixed = fixed[:lcp(fixed, x)]
        reuse = []
        for i, x in enumerate(ids):
            prev = ids[max(0, i - a.slots):i]
            reuse.append(max((lcp(p, x) for p in prev), default=0) // 32 * 32)
        # line-level: tokens in lines that also appeared in the previous prompt (any position)
        stable = []
        for i in range(1, len(texts)):
            before = set(texts[i - 1].split("\n"))
            stable.append(sum(ntok(ln + "\n") for ln in texts[i].split("\n") if ln in before))
        # section churn
        churn, size = defaultdict(int), defaultdict(list)
        prev_secs = None
        for st, q in zip(states, questions):
            secs = dict(sections(st))
            secs["(question + options)"] = q
            for k, v in secs.items():
                size[k].append(ntok(v) if k != "(question + options)" else 0)
                if prev_secs is not None and prev_secs.get(k) != v:
                    churn[k] += 1
            prev_secs = secs
        qtok = [len(tk.encode(t[t.index("Question 1"):], add_special_tokens=False)) for t in texts[:200]]
        med = statistics.median
        print(f"\n### {os.path.basename(path)}  harness {harness}  ({len(steps)} steps)")
        print(f"prompt tokens p50 {med(n):.0f} (min {min(n)}, max {max(n)}); question+options part p50 {med(qtok):.0f}")
        print(f"fixed prefix of every step: {len(fixed)} tokens ({100 * len(fixed) / med(n):.0f}%) ... "
              f"{tk.decode(fixed[-12:])!r}")
        print(f"reusable now (LCP with last {a.slots} prompts, 32-aligned): p50 {med(reuse):.0f} "
              f"({100 * med(reuse) / med(n):.0f}%), computed p50 {med([x - r for x, r in zip(n, reuse)]):.0f}")
        print(f"upper bound, stable lines first: p50 {med(stable):.0f} tokens unchanged from the previous step "
              f"({100 * med(stable) / med(n):.0f}%)")
        print("sections (tokens p50, changed in % of steps):")
        for k in size:
            print(f"   {k[:40]:40s} {med(size[k]) if size[k] else 0:6.0f} tok   changes {100 * churn[k] / (len(steps) - 1):5.1f}%")


if __name__ == "__main__":
    main()
