"""Causal head attribution: zero one memory head's read at a time and re-measure recall vs distance.

  python ablate_heads.py runs/h200/p2_haishx_l1_s0 [--lens 256,4096,65536] [--K 8]
Writes <run>/head_ablation.json: {"none": {L: acc}, "layer1:h0": {L: acc}, ...}
"""
import argparse
import json

import torch

from model import TinyLM
from tasks import Vocab, fact_stream

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lens", default="256,4096,65536")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()

ck = torch.load(f"{a.run}/model.pt", map_location="cuda", weights_only=False)
cfg_args = ck["args"]
cfg = {k: cfg_args[k] for k in ("archival_heads", "beta_bias", "min_decay", "tau", "store_cap", "null_sink", "mem_heads", "write_budget", "write_block", "arch_tau", "v_norm") if cfg_args.get(k) is not None}
V = Vocab()
model = TinyLM(V.size, cfg_args["d_model"], cfg_args["layout"], cfg).cuda().eval()
model.load_state_dict(ck["model"])
mems = [(i, b.mixer) for i, b in enumerate(model.blocks) if hasattr(b.mixer, "b_proj")]


@torch.no_grad()
def recall(L):
    g = torch.Generator(device="cuda").manual_seed(31337 + L)
    B = max(1, min(16, (1 << 21) // L))
    hit = tot = 0
    while tot < a.n:
        ids, labels, _ = fact_stream(B, L, a.K, vocab=V, write_frac=0.05, device="cuda", gen=g)
        st, preds = None, torch.full_like(labels, -1)
        for s in range(0, L, a.chunk):
            with torch.autocast("cuda", torch.bfloat16):
                h, st = model(ids[:, s:s + a.chunk], st, head=False)
            m = labels[:, s:s + a.chunk] != -100
            if m.any():
                preds[:, s:s + a.chunk][m] = model.head(h[m]).argmax(-1)
        m = labels != -100
        hit += (preds[m] == labels[m]).sum().item()
        tot += m.sum().item()
    return round(hit / tot, 4)


lens = [int(x) for x in a.lens.split(",")]
out = {"none": {L: recall(L) for L in lens}}
print("none", out["none"], flush=True)
for i, mix in mems:
    for h in range(mix.h):
        mix.ablate_heads = {h}
        key = f"layer{i}:h{h}" + ("(arch)" if h < getattr(mix, "archival_heads", 0) else "")
        out[key] = {L: recall(L) for L in lens}
        print(key, out[key], flush=True)
        mix.ablate_heads = set()
json.dump(out, open(f"{a.run}/head_ablation.json", "w"), indent=1)
