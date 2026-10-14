"""Probe: how strongly do the archival heads write a fact, as a function of how much text came before it?

For each prefix length P: stream P tokens of real text, then the fact; report per archival layer the max write
strength beta over the fact tokens (median over heads, and the fraction of heads above 0.5 / 0.9).

  python probe_write.py --run runs_lm8/lm8_gdnhais_d768 --data ~/data/fwe_2b --prefixes 0,2048,10240,31457,52428
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402
from model import TinyLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--prefixes", default="0,2048,10240,31457,52428,131072")
ap.add_argument("--fact", default="The secret code of Velthu is 731958.")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--arch_tau", type=float, default=None)
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

orig = layers.chunk_gated_delta_rule
REC = []
layers.chunk_gated_delta_rule = lambda q, k, v, g=None, beta=None, **kw: (REC.append(beta.detach()), orig(q, k, v, g=g, beta=beta, **kw))[1]

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location="cuda", weights_only=False)
model = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], ck["cfg"]).cuda().eval()
model.load_state_dict(ck["model"])
mix = [b.mixer for b in model.blocks if isinstance(b.mixer, layers.DeltaMemMixer)]
if a.arch_tau is not None:
    for m in mix:
        if m.archival_heads:
            m.arch_tau = a.arch_tau
from lm_data import token_dtype  # noqa: E402
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
s0 = int(np.random.default_rng(a.seed).integers(0, len(val) - 2_000_000))
fact = tok(" " + a.fact, add_special_tokens=False)["input_ids"]
print("fact tokens:", [tok.decode([t]) for t in fact])


@torch.no_grad()
def run(ids, st):
    REC.clear()
    with torch.autocast("cuda", torch.bfloat16):
        _, st = model(ids, st, head=False)
    return st


for P in map(int, a.prefixes.split(",")):
    ids = torch.from_numpy(val[s0:s0 + P].astype(np.int64)).cuda()[None]
    st = None
    for c0 in range(0, P, a.chunk):
        st = run(ids[:, c0:c0 + a.chunk], st)
    run(torch.tensor(fact).cuda()[None], st)
    rows = []
    for m, b in zip(mix, REC):
        if not m.archival_heads:
            continue
        ba = b[0, :, :m.archival_heads].float()  # [fact tokens, A]
        mx, arg = ba.max(0)
        rows.append(f"med {mx.median():.2f} >0.5 {(mx > 0.5).float().mean():.2f} >0.9 {(mx > 0.9).float().mean():.2f} "
                    f"@{[tok.decode([fact[i]]) for i in arg.tolist()][:3]}")
    print(f"prefix {P:>7}: " + " | ".join(rows), flush=True)
