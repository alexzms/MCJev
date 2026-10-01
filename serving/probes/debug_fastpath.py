"""Bisect the fastpath correctness bug: graph vs eager vs slice-normalization, per MoE backend."""
import math, os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import server as S
from transformers import AutoTokenizer
import fastpath
moe = sys.argv[1] if len(sys.argv) > 1 else "flashinfer_cutlass"
tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
enc = S.Encoder(tok, 1 << 30)
enc.label_groups = [g[:1] for g in enc.label_groups]; enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
st = {"id": "ticket", "state": "Customer: Everything is down and we have a demo at noon.", "questions": {
    "urgent": {"type": "boolean", "instructions": "Does the customer need a reply within the hour?"},
    "category": {"type": "choice", "instructions": "What is the ticket about?", "criteria": {"outage": "service is down", "billing": "payment or invoice", "howto": "usage question"}},
    "severity": {"type": "score", "instructions": "How severe is the problem?", "criteria": ["cosmetic", "minor", "major", "critical"]}}}
text, slots = enc.prompt_text(st); body = enc.scaffold(slots, "t"); canvas = body + [0] * (64 - len(body))
p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False)
pids = (p["input_ids"] if isinstance(p, dict) else list(p)) + list(enc.thought_prefix_ids)
cands = [[i for g in s.cand_groups for i in g] for s in slots]
bk = sys.argv[2] if len(sys.argv) > 2 else "320,384"
fr = fastpath.FastReader(moe=moe, buckets=None if bk == "all" else [int(x) for x in bk.split(",")])
print("DBG buckets", fr.buckets[:3], "...", fr.buckets[-2:], "tokens in forward", len(pids) + 64, flush=True)
for v in ("graph", "eager", "slice", "graph"):
    lp = fr.read_debug(pids, canvas, [s.pos for s in slots], cands, v)
    row = []
    for s, r in zip(slots, lp):
        pr = [math.exp(x) for x in r]; m = sum(pr)
        row.append(f"{s.qid}: {[round(x/m, 3) for x in pr]} mass {m:.3f}")
    print(f"DBG {bk:8s} {v:6s} | " + " | ".join(row), flush=True)
