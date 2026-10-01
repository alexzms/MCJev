"""Top tokens at each slot for the mask/1-step readout, to see where the leftover mass goes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # serving/
import torch
from transformers import AutoTokenizer
import server as S
from client import EXAMPLE

tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
enc = S.Encoder(tok, 4096)
states = EXAMPLE["states"] + [
    {"id": "arith", "state": "x = 7, y = 12", "questions": {
        "sum_gt_20": {"type": "boolean", "instructions": "Is x + y greater than 20?"},
        "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]}}},
    {"id": "coin", "state": "A fair coin will be flipped once.", "questions": {
        "heads": {"type": "boolean", "instructions": "Will it land heads?"}}},
    {"id": "email", "state": {"from": "billing@shop-example.com", "subject": "Your invoice #4471 is overdue", "body": "Please pay within 3 days to avoid a late fee."},
     "questions": {
        "category": {"type": "choice", "instructions": "What kind of email is this?",
                     "criteria": {"invoice": "a bill or payment reminder", "newsletter": "marketing newsletter", "personal": "personal message from a friend", "phishing": "obvious phishing attempt"}},
        "urgent": {"type": "boolean", "instructions": "Does the email ask for action within a week?"},
        "tone": {"type": "score", "instructions": "How aggressive is the tone?", "criteria": ["polite", "neutral", "firm", "threatening"]}}},
]
examples = [enc.encode(s) for s in states]
model = S.load_model(S.MODEL_DEFAULT, "cuda:0", "grouped_mm", "sdpa")
cfg = dict(steps=1, slot_noise="mask", noise_samples=1, sc_temperature=1.0)

# replicate run_batch up to the logits, then inspect top-k
payloads = [e.payload() for e in examples]
device = model.device
B = len(payloads); L = max(len(p["prompt_ids"]) for p in payloads)
input_ids = torch.full((B, L), S.PAD_ID, dtype=torch.long); attn = torch.zeros((B, L), dtype=torch.bool)
for b, p in enumerate(payloads):
    n = len(p["prompt_ids"]); input_ids[b, L-n:] = torch.tensor(p["prompt_ids"]); attn[b, L-n:] = True
canvas = torch.tensor([p["canvas_ids"] for p in payloads])
input_ids, attn, canvas = input_ids.to(device), attn.to(device), canvas.to(device)
dec_mask = torch.cat([attn, torch.ones((B, S.CANVAS_LENGTH), dtype=torch.bool, device=device)], 1)
with torch.inference_mode():
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=torch.arange(L, device=device)[None],
                decoder_input_ids=canvas, decoder_attention_mask=dec_mask,
                decoder_position_ids=torch.arange(L, L+S.CANVAS_LENGTH, device=device)[None])
    probs = torch.softmax(out.logits.float(), -1)
for b, e in enumerate(examples):
    for slot in e.slots:
        p = probs[b, slot.pos]
        top = torch.topk(p, 8)
        toks = [(tok.convert_ids_to_tokens(int(i)), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
        cand = {k: round(float(p[g].sum()), 3) for k, g in zip(slot.keys, slot.cand_groups)}
        print(f"{e.state_id:13s} {slot.qid:10s} cand={cand}\n{'':25s}top={toks}")
    # what does the model think the whole canvas is? (argmax per position, first 20)
    am = probs[b].argmax(-1)[:20].tolist()
    print(f"{'':25s}argmax canvas: {tok.convert_ids_to_tokens(am)}")
