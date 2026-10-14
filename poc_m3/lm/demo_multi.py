"""Demo: several facts planted at different depths of a long real-text stream, then asked one after another, like a
conversation. The model streams everything with constant memory and *generates* each answer greedily; each question
and the model's own answer stay in the stream, so the state just keeps going.

  python demo_multi.py --run runs_node/gdnhais_d1536 --data ~/data/fwe --tokens 1048576 [--arch_tau 0.9]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa: E402

FACTS = [  # (fact, question prefix, expected answer) -- trained phrasings and two held-out ones
    ("The secret code of Velthu is 731958.", "What is the secret code of Velthu? The secret code of Velthu is", "731958"),
    ("Moraska lives in Lisbon.", "Where does Moraska live? Moraska lives in", "Lisbon"),
    ("The password for Kribrisen is thuzoka.", "What is the password for Kribrisen? The password for Kribrisen is", "thuzoka"),
    ("Galfen's locker combination is 48213.", "What is Galfen's locker combination? Galfen's locker combination is", "48213"),
    ("Sabtor moved to Kyoto last year.", "Where did Sabtor move last year? Sabtor moved to", "Kyoto"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--tokens", type=int, default=1 << 20)
ap.add_argument("--arch_tau", type=float, default=None)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_new", type=int, default=6)
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location="cuda", weights_only=False)
model = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], ck["cfg"]).cuda().eval()
model.load_state_dict(ck["model"])
if a.arch_tau is not None:
    for b in model.blocks:
        if getattr(b.mixer, "archival_heads", 0):
            b.mixer.arch_tau = a.arch_tau
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
s0 = int(np.random.default_rng(a.seed).integers(0, len(val) - a.tokens - 1))
hay = val[s0:s0 + a.tokens].astype(np.int64).tolist()
# facts spread over the first 30% of the stream: every question is asked 0.7-1.0 x tokens later
cuts = sorted(int(f * a.tokens) for f in np.linspace(0.01, 0.3, len(FACTS)))
stream, prev = [], 0
for (fact, _, _), c in zip(FACTS, cuts):
    stream += hay[prev:c] + enc(" " + fact)
    prev = c
stream += hay[prev:]
print(f"{len(stream):,} tokens of FineWeb-Edu with {len(FACTS)} facts at tokens {cuts}", flush=True)


@torch.no_grad()
def feed(ids, st):
    h = None
    for c0 in range(0, len(ids), a.chunk):
        x = torch.tensor(ids[c0:c0 + a.chunk], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            h, st = model(x, st, head=False)
    return h, st


torch.cuda.reset_peak_memory_stats()
t0 = time.time()
h, st = feed(stream, None)
print(f"streamed in {time.time() - t0:.0f}s ({len(stream) / (time.time() - t0):,.0f} tok/s), "
      f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
ok = 0
for (fact, q, want), c in zip(FACTS, cuts):
    h, st = feed(enc(" " + q), st)
    out = []
    with torch.no_grad():
        for _ in range(a.max_new):
            nxt = model.head(h[:, -1]).argmax(-1)
            out.append(int(nxt))
            with torch.autocast("cuda", torch.bfloat16):
                h, st = model(nxt[:, None], st, head=False)
    ans = tok.decode(out)
    hit = ans.strip().startswith(want)
    ok += hit
    print(f"[{'OK ' if hit else 'MISS'}] {len(stream) - c:>9,} tokens after the fact | Q: {q} -> {ans!r}  (truth: {want})",
          flush=True)
print(f"{ok}/{len(FACTS)} correct; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (constant in length)")
