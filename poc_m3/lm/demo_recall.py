"""Demo: plant facts in a long stream of real text, stream it through the model with constant memory, ask a question,
and let the model *generate* the answer.

  python demo_recall.py --run runs_lm8/lm8_gdnhais_d768 --data ~/data/fwe_2b --tokens 131072 \
      --fact "The secret code of Zelith is 482913." --question "What is the secret code of Zelith? The secret code of Zelith is"

The text haystack is FineWeb-Edu validation text (or --text_file). Facts are placed at --depth (fraction of the stream).
Prints the generated continuation, peak GPU memory, and throughput.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", default=None, help="dir with val.bin (FineWeb-Edu tokens) used as haystack")
ap.add_argument("--text_file", default=None, help="use this text file as haystack instead (repeated if short)")
ap.add_argument("--tokens", type=int, default=131072, help="haystack length in tokens")
ap.add_argument("--fact", action="append", required=True)
ap.add_argument("--depth", type=float, default=0.05, help="where the facts go (fraction of the stream)")
ap.add_argument("--question", required=True)
ap.add_argument("--max_new", type=int, default=6)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--arch_tau", type=float, default=None, help="zero archival writes with beta <= tau (inference-time write threshold)")
ap.add_argument("--gdn_half_life", type=float, default=None, help="inference override: cap memory half-life (tokens) of non-archival delta layers")
ap.add_argument("--seed", type=int, default=0, help="picks the haystack window")
ap.add_argument("--reset_arch_at_fact", action="store_true", help="diagnostic: zero the archival heads' state right before the facts")
ap.add_argument("--reset_all_at_fact", action="store_true", help="diagnostic: drop the whole recurrent state right before the facts")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location="cuda", weights_only=False)
model = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], ck["cfg"]).cuda().eval()
model.load_state_dict(ck["model"])
if a.gdn_half_life:
    for b in model.blocks:
        if b.code == "gdn":
            b.mixer.min_decay = max(b.mixer.min_decay, math.log(2) / a.gdn_half_life)
if a.arch_tau is not None:
    for b in model.blocks:
        if getattr(b.mixer, "archival_heads", 0):
            b.mixer.arch_tau = a.arch_tau

if a.text_file:
    hay = enc(open(a.text_file).read())
    hay = (hay * (a.tokens // max(1, len(hay)) + 1))[:a.tokens]
else:
    from lm_data import token_dtype  # noqa: E402
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
    s0 = int(np.random.default_rng(a.seed).integers(0, len(val) - a.tokens - 1))
    hay = val[s0:s0 + a.tokens].astype(np.int64).tolist()
facts = [t for f in a.fact for t in enc(" " + f)]
cut = int(a.depth * len(hay))
stream = hay[:cut] + facts + hay[cut:] + enc(" " + a.question)
print(f"stream: {len(stream):,} tokens, facts at token {cut:,} "
      f"(distance to the question ~{len(stream) - cut:,} tokens)", flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    ids = torch.tensor(stream, device="cuda")[None]
    st, h = None, None
    for s_, e_ in ((0, cut), (cut, ids.shape[1])):
        if s_ == cut and a.reset_all_at_fact:
            st = None
        if s_ == cut and a.reset_arch_at_fact and st is not None:
            for b, x in zip(model.blocks, st):
                if getattr(b.mixer, "archival_heads", 0):
                    x[0][:, :b.mixer.archival_heads] = 0  # S: [B, H, K, V]
        for c0 in range(s_, e_, a.chunk):
            with torch.autocast("cuda", torch.bfloat16):
                h, st = model(ids[:, c0:min(c0 + a.chunk, e_)], st, head=False)
    out = []
    nxt = model.head(h[:, -1]).argmax(-1)  # greedy decoding, state carried token by token
    for _ in range(a.max_new):
        out.append(int(nxt))
        with torch.autocast("cuda", torch.bfloat16):
            h, st = model(nxt[:, None], st, head=False)
        nxt = model.head(h[:, -1]).argmax(-1)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"answer: {tok.decode(out)!r}")
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, {len(stream) / dt:,.0f} tok/s, {dt:.1f}s")
