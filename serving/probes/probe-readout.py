"""GPU probe: load DiffusionGemma once, compare readout configurations on toy decisions."""
import os, json, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # serving/
import torch
from transformers import AutoTokenizer
import server as S
from client import EXAMPLE

MODEL = S.MODEL_DEFAULT
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
enc = S.Encoder(tok, 4096)
print("labels", len(enc.labels), enc.labels[:5], "...")
states = EXAMPLE["states"] + [
    {"id": "arith", "state": "x = 7, y = 12", "questions": {
        "bigger": {"type": "choice", "instructions": "Which variable is larger?", "criteria": {"x": "the variable x", "y": "the variable y"}},
        "sum_gt_20": {"type": "boolean", "instructions": "Is x + y greater than 20?"},
        "sum_gt_15": {"type": "boolean", "instructions": "Is x + y greater than 15?"},
        "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]}}},
    {"id": "coin", "state": "A fair coin will be flipped once.", "questions": {
        "heads": {"type": "boolean", "instructions": "Will it land heads?"},
        "which": {"type": "choice", "instructions": "Which side will it land on?", "criteria": {"H": "heads", "T": "tails"}}}},
]
examples = [enc.encode(s) for s in states]
print("prompt lens", [len(e.prompt_ids) for e in examples])
print("---- prompt of state 0 ----"); print(tok.decode(examples[0].prompt_ids)); print("---- canvas ----")
print(tok.convert_ids_to_tokens(examples[0].canvas_ids[:12]))

t0 = time.time()
model = S.load_model(MODEL, "cuda:0", "grouped_mm", "sdpa")
torch.cuda.synchronize(); print(f"loaded in {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/2**30:.1f} GB")

def show(tag, cfg, reps=1):
    ts = []
    for r in range(reps):
        torch.cuda.synchronize(); t = time.time()
        res = S.run_batch(model, [e.payload() for e in examples], cfg)
        torch.cuda.synchronize(); ts.append(time.time() - t)
    print(f"\n#### {tag} cfg={cfg} time/batch(5 states)={min(ts)*1000:.0f} ms")
    for e, per_slot in zip(examples, res):
        for slot, (probs, mass) in zip(e.slots, per_slot):
            d = {k: round(p, 3) for k, p in zip(slot.keys, probs)}
            print(f"  {e.state_id:13s} {slot.qid:10s} mass={mass:.3f} {d}")

base = dict(steps=1, slot_noise="random", noise_samples=1, sc_temperature=1.0)
show("random noise, 1 step", base, reps=3)
show("random noise, 1 step (2nd draw)", base)
show("mask slot, 1 step", {**base, "slot_noise": "mask"}, reps=2)
show("random, 4 noise samples", {**base, "noise_samples": 4})
show("random, 2 steps self-cond", {**base, "steps": 2})
show("mask, 2 steps self-cond", {**base, "slot_noise": "mask", "steps": 2})
show("random, 3 steps, sc T=0.8", {**base, "steps": 3, "sc_temperature": 0.8})

# batch scaling timing
for bs in (1, 8, 16, 32):
    pl = [examples[i % len(examples)].payload() for i in range(bs)]
    S.run_batch(model, pl, base); torch.cuda.synchronize(); t = time.time()
    S.run_batch(model, pl, base); torch.cuda.synchronize()
    print(f"batch {bs:2d}: {(time.time()-t)*1000:.0f} ms  -> {bs/(time.time()-t):.1f} states/s")
