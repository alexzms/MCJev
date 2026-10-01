"""Stage-level timing of one djev forward (text batch 1 / 8, one image) + top CUDA kernels."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # serving/
import torch, numpy as np
from transformers import AutoProcessor, DynamicCache
import server as S
from client import EXAMPLE, image_source


def synthetic_shapes(size=256):
    """A red triangle (left) and a blue circle (right) on white, as a PNG data URI (known ground truth)."""
    import base64, io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.polygon([(size * 0.10, size * 0.75), (size * 0.30, size * 0.25), (size * 0.50, size * 0.75)], fill="red")
    d.ellipse([size * 0.58, size * 0.35, size * 0.90, size * 0.67], fill="blue")
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


proc = AutoProcessor.from_pretrained(S.MODEL_DEFAULT, local_files_only=True)
enc = S.Encoder(proc.tokenizer, 4096, processor=proc)
dev = "cuda:0"
model = S.load_model(S.MODEL_DEFAULT, dev, "grouped_mm", "sdpa")
cfg = dict(steps=1, slot_noise="mask", noise_samples=1, sc_temperature=1.0)
text_ex = [enc.encode(s) for s in S.validate_request(EXAMPLE)]
img_state = {"id": "img", "state": "Image 1 attached.", "images": [synthetic_shapes(256)],
             "questions": {"q": {"type": "boolean", "instructions": "Is there a circle in Image 1?"}}}
img_ex = enc.encode(img_state)

def timed(fn, reps=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(dev); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(dev); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

# --- whole run_batch
for name, pl in [("text b1", [text_ex[0].payload()]), ("text b8", [text_ex[i % 2].payload() for i in range(8)]),
                 ("text b16", [text_ex[i % 2].payload() for i in range(16)]), ("image b1", [img_ex.payload()])]:
    print(f"run_batch {name:9s}: {timed(lambda: S.run_batch(model, pl, cfg)):7.1f} ms")

# --- stages for text b1 and b8
def stages(payloads):
    R = len(payloads); L = max(len(p["prompt_ids"]) for p in payloads)
    ids = torch.full((R, L), S.PAD_ID, dtype=torch.long); attn = torch.zeros((R, L), dtype=torch.bool)
    for r, p in enumerate(payloads):
        n = len(p["prompt_ids"]); ids[r, L-n:] = torch.tensor(p["prompt_ids"]); attn[r, L-n:] = True
    canvas = torch.tensor([p["canvas_ids"] for p in payloads]); ids, attn, canvas = ids.to(dev), attn.to(dev), canvas.to(dev)
    dec_mask = torch.cat([attn, torch.ones((R, 256), dtype=torch.bool, device=dev)], 1)
    pos = torch.arange(L, device=dev)[None]; dpos = torch.arange(L, L+256, device=dev)[None]
    out = {}
    with torch.inference_mode():
        def enc_step():
            cache = DynamicCache(config=model.config.get_text_config(decoder=True))
            dummy = torch.empty((R, L, 0), dtype=model.dtype, device=dev)
            m = model.model.encoder.create_masks_for_generate(config=model.config, inputs_embeds=dummy, attention_mask=attn, past_key_values=cache, position_ids=pos, mm_token_type_ids=None)
            return model.model.encoder(input_ids=ids, attention_mask=m, position_ids=pos, past_key_values=cache).past_key_values
        out["encoder (prompt %d tok)" % L] = timed(enc_step)
        cache = enc_step()
        def dec_step():
            return model.model.decoder(decoder_input_ids=canvas, past_key_values=cache, decoder_attention_mask=dec_mask, decoder_position_ids=dpos).last_hidden_state
        out["decoder (256-tok canvas, no head)"] = timed(dec_step)
        h = dec_step()
        def head():
            lg = model.lm_head(h).float(); lg = torch.tanh(lg / 30.0) * 30.0
            return torch.log_softmax(lg[:, :12].float(), -1)
        out["lm_head 256x262k + softcap"] = timed(head)
        def full():
            return model(past_key_values=cache, decoder_input_ids=canvas, decoder_attention_mask=dec_mask, decoder_position_ids=dpos).logits
        out["decoder+lm_head via model()"] = timed(full)
    return out
for name, pl in [("text b1", [text_ex[0].payload()]), ("text b8", [text_ex[i % 2].payload() for i in range(8)])]:
    print(f"--- stages {name}")
    for k, v in stages(pl).items(): print(f"   {k:34s} {v:7.1f} ms")

# --- profiler: kernel count + top kernels for decoder step b1
from torch.profiler import profile, ProfilerActivity


pl = [text_ex[0].payload()]
S.run_batch(model, pl, cfg); torch.cuda.synchronize(dev)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    S.run_batch(model, pl, cfg); torch.cuda.synchronize(dev)
evs = prof.key_averages()
n_kernels = sum(e.count for e in evs if e.device_type.name == "CUDA")
cuda_total = sum(e.device_time_total for e in evs if e.device_type.name == "CUDA") / 1000
print(f"--- profiler text b1: CUDA kernels launched = {n_kernels}, total CUDA time = {cuda_total:.1f} ms (wall ~{timed(lambda: S.run_batch(model, pl, cfg)):.0f} ms)")
print(evs.table(sort_by="device_time_total", row_limit=12, max_name_column_width=60))
