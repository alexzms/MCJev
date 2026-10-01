"""Real p_uhc_pro_v10 steps as the fast engine reads them, for the probes: one per bot (Jev1..JevN: its own name, so
bots share no prompt prefix), changing only from the "Now:" line on, as in a match (data/uhc_pro_step85.json)."""
import copy
import json
import os
import random
import re

import server as S
import server_vllm as SV

HERE = os.path.dirname(os.path.abspath(__file__))
REQUEST = os.path.join(HERE, "data", "uhc_pro_step85.json")


def encoder():
    """(tokenizer, encoder) as server_vllm.main builds them with --ids main."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30, empty_thought=True)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    return tok, enc


def stepper(tok, enc, request=REQUEST, canvas=64):
    """step(bot) -> one text read (prompt_ids, canvas_ids, slot_positions, cand_ids) of bot's next step, as
    server_vllm.fast_read_args builds it."""
    base = json.load(open(request))["states"][0]
    head, sep, tail = base["state"].partition("\nNow:")
    assert sep, "the state has no 'Now:' line"
    chat_head = [tok.convert_tokens_to_ids("<bos>"), tok.convert_tokens_to_ids("<|turn>")] + tok.encode("user\n", add_special_tokens=False)
    chat_tail = ([tok.convert_tokens_to_ids("<turn|>")] + tok.encode("\n", add_special_tokens=False)
                 + [tok.convert_tokens_to_ids("<|turn>")] + tok.encode("model\n", add_special_tokens=False)
                 + list(enc.thought_prefix_ids))

    def step(bot):
        st = copy.deepcopy(base)
        st["id"] = "step"
        st["state"] = head.replace("You are JevPro", f"You are Jev{bot}", 1) + re.sub(
            r"\d", lambda m: str(random.randint(0, 9)), sep + tail)
        (sub,) = SV.split_questions(st, enc, canvas // 2)
        text, slots = enc.prompt_text(sub, 0)
        body = enc.scaffold(slots, sub["id"])
        return (chat_head + tok.encode(text, add_special_tokens=False) + chat_tail,
                body + [S.PAD_ID] * (canvas - len(body)),
                [sl.pos for sl in slots], [[i for g in sl.cand_groups for i in g] for sl in slots])
    return step
