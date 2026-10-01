#!/usr/bin/env python3
"""Where does a Minecraft vision read's time go on one GPU?  fastpath.FastReader directly (no HTTP), on a recorded
jevagent request: the `step` state (decision, one screenshot) and the `look` state (five perception questions, the
same screenshot), each with a fresh screenshot (random bytes after the JPEG end: same pixels, new hash), as in a match.

    python probes/profile_mc_reads.py --request minecraft-request.json [--n 20]

Cases: step alone; look alone (what a GPU gets when a gateway splits the two states); look right after its step on
the same GPU (vision cache + prompt-prefix hit); a text-only step. Per case the medians of FastReader.timing
(preprocess, vision, tokenize, forward) and of the whole call.
"""
import argparse
import base64
import copy
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import server as S  # noqa: E402
import server_vllm as SV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--moe", default="flashinfer_cutlass")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    import fastpath
    tok = AutoTokenizer.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
    enc = S.Encoder(tok, 1 << 30, empty_thought=True)
    enc.label_groups = [g[:1] for g in enc.label_groups]
    enc.yes_group, enc.no_group = enc.yes_group[:1], enc.no_group[:1]
    fr = fastpath.FastReader(canvas=a.canvas, moe=a.moe, gpu_memory_utilization=0.85, max_model_len=4096,
                             thought_ids=list(enc.thought_prefix_ids), prefix_slots=8, fast_attention=True)
    print(json.dumps({"load_seconds": round(fr.load_seconds, 1)}), flush=True)

    step, look = json.load(open(a.request))["states"]
    raw = S.load_image_bytes(step["images"][0])
    chat_head = [tok.convert_tokens_to_ids("<bos>"), tok.convert_tokens_to_ids("<|turn>")] + tok.encode("user\n", add_special_tokens=False)
    chat_tail = ([tok.convert_tokens_to_ids("<turn|>")] + tok.encode("\n", add_special_tokens=False)
                 + [tok.convert_tokens_to_ids("<|turn>")] + tok.encode("model\n", add_special_tokens=False)
                 + list(enc.thought_prefix_ids))

    def reads_of(state, image):
        """The fast engine's reads for one state (server_vllm.fast_read_args, per split of its questions)."""
        out = []
        for sub in SV.split_questions(state, enc, a.canvas // 2):
            text, slots = enc.prompt_text(sub, 1 if image else 0)
            body = enc.scaffold(slots, sub["id"])
            canvas = body + [S.PAD_ID] * (a.canvas - len(body))
            cands = [[i for g in sl.cand_groups for i in g] for sl in slots]
            if image:
                out.append(("image", text, [image], canvas, [sl.pos for sl in slots], cands))
            else:
                out.append(("text", chat_head + tok.encode(text, add_special_tokens=False) + chat_tail, canvas,
                            [sl.pos for sl in slots], cands))
        return out

    def vary(state):
        s = copy.deepcopy(state)
        s["state"] = re.sub(r"\d", lambda m: str(random.randint(0, 9)), s["state"])
        return s

    def run(r):
        t = time.perf_counter()
        if r[0] == "image":
            fr.read_image(*r[1:])
            tm = dict(fr.timing)
        else:
            fr.read(*r[1:])
            tm = {}
        torch.cuda.synchronize()
        tm["total"] = time.perf_counter() - t
        tm["cached"] = fr.last.get("cached", 0)
        tm["computed"] = fr.last.get("computed", 0)
        tm["bucket"] = fr.last.get("bucket", 0)
        tm["fa"] = fr.last.get("fa")
        return tm

    cases = {"step alone": [], "look alone": [], "look after its step": [], "text step": []}
    text_step = {k: v for k, v in step.items() if k != "images"}
    for i in range(a.n + 3):
        img = raw + os.urandom(16)
        s_reads = reads_of(vary(step), img)
        l_reads = reads_of(look, img)
        got = {"step alone": [run(r) for r in s_reads], "look after its step": [run(r) for r in l_reads]}
        img2 = raw + os.urandom(16)
        got["look alone"] = [run(r) for r in reads_of(look, img2)]
        got["text step"] = [run(r) for r in reads_of(vary(text_step), None)]
        if i >= 3:
            for k, v in got.items():
                cases[k].append(v)

    print(f"{'case':22s} reads  {'preprocess':>10s} {'vision':>8s} {'tokenize':>8s} {'forward':>8s} {'total':>8s}  cached/computed tokens, bucket, FA2")
    for k, runs in cases.items():
        for j in range(len(runs[0])):
            col = lambda f: statistics.median(r[j].get(f, 0) for r in runs) * 1000
            last = runs[-1][j]
            print(f"{k:22s} {j + 1}/{len(runs[0])}  {col('preprocess'):10.1f} {col('vision'):8.1f} {col('tokenize'):8.1f} "
                  f"{col('forward'):8.1f} {col('total'):8.1f}  {last['cached']}/{last['computed']}, {last['bucket']}, {last['fa']}")


if __name__ == "__main__":
    main()
