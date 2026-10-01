#!/usr/bin/env python3
"""Is a single forward over [prompt; canvas] equivalent to djev's two-step read (encoder prefill, then one
decoder denoise pass)?  HF transformers reference, one GPU, bf16.

Two-step (what we serve):   encoder(prompt) -> KV;  decoder(canvas | KV), canvas bidirectional, self-conditioning
                            with zero soft embeddings -> logits at the slots.
Single pass (candidate):    run the *encoder* layers once over [prompt embeds ; self_conditioning(canvas embeds, 0)]
                            with mask = causal over the prompt, and canvas rows attending to the whole prompt and
                            the whole canvas; final norm + lm_head + softcap on the canvas rows.

Prints, per question, the two distributions and the max |diff|; plus both argmax canvases.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server as S  # noqa: E402
from client import EXAMPLE  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402

WIDTH = 64


def two_step(model, prompt_ids, canvas, dev):
    L = len(prompt_ids)
    ids = torch.tensor([prompt_ids], device=dev)
    attn = torch.ones((1, L), dtype=torch.bool, device=dev)
    cache = DynamicCache(config=model.config.get_text_config(decoder=True))
    pos = torch.arange(L, device=dev)[None]
    dummy = torch.empty((1, L, 0), dtype=model.dtype, device=dev)
    m = model.model.encoder.create_masks_for_generate(config=model.config, inputs_embeds=dummy, attention_mask=attn,
                                                      past_key_values=cache, position_ids=pos, mm_token_type_ids=None)
    e = model.model.encoder(input_ids=ids, attention_mask=m, position_ids=pos, past_key_values=cache)
    c = torch.tensor([canvas], device=dev)
    dmask = torch.ones((1, L + len(canvas)), dtype=torch.bool, device=dev)
    out = model(past_key_values=e.past_key_values, decoder_input_ids=c, decoder_attention_mask=dmask,
                decoder_position_ids=torch.arange(L, L + len(canvas), device=dev)[None])
    return torch.log_softmax(out.logits[0].float(), -1)


def single_pass(model, prompt_ids, canvas, dev):
    L, C = len(prompt_ids), len(canvas)
    T = L + C
    enc = model.model.encoder.language_model          # DiffusionGemmaEncoderTextModel
    dec = model.model.decoder
    ids = torch.tensor([prompt_ids + canvas], device=dev)
    emb = enc.embed_tokens(ids)                        # same (tied) embedding as the decoder's
    cv = emb[:, L:]
    emb = torch.cat([emb[:, :L], dec.self_conditioning(cv, torch.zeros_like(cv))], dim=1)
    # mask: rows = queries, cols = keys; True = attend
    allow = torch.ones((T, T), dtype=torch.bool, device=dev).tril()
    allow[L:, :] = True                                 # canvas rows see the whole prompt and the whole canvas
    neg = torch.finfo(model.dtype).min
    mask4 = torch.zeros((1, 1, T, T), dtype=model.dtype, device=dev).masked_fill(~allow, neg)
    mapping = {"full_attention": mask4, "sliding_attention": mask4}   # T << sliding window (1024)
    pos = torch.arange(T, device=dev)[None]
    out = enc(inputs_embeds=emb, attention_mask=mapping, position_ids=pos, past_key_values=None)
    h = out.last_hidden_state[0, L:]
    logits = model.lm_head(h).float()
    cap = model.final_logit_softcapping
    logits = torch.tanh(logits / cap) * cap
    return torch.log_softmax(logits, -1)


def main():
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    encd = S.Encoder(tok, 1 << 30)
    model = S.load_model(S.MODEL_DEFAULT, "cuda:0", "grouped_mm", "eager")
    dev = model.device
    states = S.validate_request(EXAMPLE) + [
        {"id": "arith", "state": "x = 7, y = 12", "questions": {
            "bigger": {"type": "choice", "instructions": "Which variable is larger?", "criteria": {"x": "the variable x", "y": "the variable y"}},
            "x_odd": {"type": "boolean", "instructions": "Is x an odd number?"},
            "size": {"type": "score", "instructions": "How large is x + y?", "criteria": ["below 10", "10 to 19", "20 to 29", "30 or more"]}}}]
    worst = 0.0
    with torch.inference_mode():
        for st in states:
            text, slots = encd.prompt_text(st)
            body = encd.scaffold(slots, st["id"])
            canvas = body + [S.PAD_ID] * (WIDTH - len(body))
            pids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                           return_dict=False)
            pids = (pids["input_ids"] if isinstance(pids, dict) else list(pids)) + list(encd.thought_prefix_ids)
            a = two_step(model, pids, canvas, dev)
            b = single_pass(model, pids, canvas, dev)
            print(f"== {st['id']}  prompt {len(pids)} tok")
            print("   argmax two-step   :", tok.convert_ids_to_tokens(a[:14].argmax(-1).tolist()))
            print("   argmax single-pass:", tok.convert_ids_to_tokens(b[:14].argmax(-1).tolist()))
            for s in slots:
                pa = [float(a[s.pos, g].exp().sum()) for g in s.cand_groups]
                pb = [float(b[s.pos, g].exp().sum()) for g in s.cand_groups]
                na = [x / sum(pa) for x in pa]
                nb = [x / sum(pb) for x in pb]
                d = max(abs(x - y) for x, y in zip(na, nb))
                worst = max(worst, d)
                print(f"   {s.qid:11s} two-step {[round(x, 3) for x in na]} mass {sum(pa):.3f} | single {[round(x, 3) for x in nb]} "
                      f"mass {sum(pb):.3f} | max|diff| {d:.4f}")
    print(f"WORST max|diff| over all questions: {worst:.4f}")


if __name__ == "__main__":
    main()
