"""How does a trained delta-memory model protect stored facts from distractor writes?

For every delta-rule mixer (gdn / hais / hx), per head:
  beta_fact   mean write strength at fact-value positions ([FACT k v] -> the v position)
  beta_fill   mean write strength at filler positions
  overlap     mean over filler positions of max |cos(k_filler, k_fact)| within the same sequence
            (keys centred per head first: a shared mean direction would otherwise dominate the cosine)
  lambda_fill per-token decay rate -log(alpha) at filler positions (0 for archival heads)
  c2_fill_fact mean (k_filler . k_fact)^2 over raw unit keys (what the kernel uses)
  efold_tokens 1 / (lambda_fill + beta_fill * c2_fill_fact): predicted e-folding distance of a stored fact
Retention of a stored fact after D distractor tokens ~ exp(-D * (lambda_fill + beta_fill * overlap^2)):
decay and erasure are the two ways a fixed-state memory forgets. Small beta_fill => protection by
gating; small overlap => protection by orthogonal key subspaces; lambda_fill > 0 => forgetting by decay.

  python analyze_keys.py runs/pilot1/gdn
"""
import json
import sys

import torch
import torch.nn.functional as F

from model import TinyLM
from tasks import FACT, Vocab, fact_stream

run = sys.argv[1]
ck = torch.load(f"{run}/model.pt", map_location="cuda", weights_only=False)
a = ck["args"]
cfg = {k: a[k] for k in ("archival_heads", "beta_bias", "min_decay", "tau", "store_cap", "null_sink", "mem_heads", "write_budget", "write_block", "arch_tau", "v_norm") if a.get(k) is not None}
V = Vocab()
model = TinyLM(V.size, a["d_model"], a["layout"], cfg).cuda().eval()
model.load_state_dict(ck["model"])

keys = {}
for i, blk in enumerate(model.blocks):
    if hasattr(blk.mixer, "k_conv"):
        blk.mixer.k_conv.register_forward_hook(lambda m, inp, out, i=i: keys.__setitem__(i, out[0]))

g = torch.Generator(device="cuda").manual_seed(4242)
ids, labels, _ = fact_stream(16, a["L"], 16, vocab=V, device="cuda", gen=g)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    model(ids)
fact_v = torch.zeros_like(ids, dtype=torch.bool)
fact_v[:, 2:] = ids[:, :-2] == FACT            # v position of each [FACT k v]
fill = ids >= V.fill0
out = {}
for i, blk in enumerate(model.blocks):
    if i not in keys:
        continue
    mix = blk.mixer
    k_raw = F.normalize(keys[i].float().view(ids.shape[0], ids.shape[1], mix.h, mix.dk), dim=-1)  # as the kernel sees them
    k = F.normalize(k_raw - k_raw.mean(dim=(0, 1), keepdim=True), dim=-1)  # centred: subspace view
    beta = mix.last_beta.float()               # [B, T, H]
    heads = []
    for h in range(mix.h):
        ov, c2 = [], []
        for b in range(ids.shape[0]):
            kf, kx = k[b, fact_v[b], h], k[b, fill[b], h]
            if len(kf) and len(kx):
                ov.append((kx @ kf.T).abs().max(-1).values.mean().item())
                rf, rx = k_raw[b, fact_v[b], h], k_raw[b, fill[b], h]
                c2.append(((rx @ rf.T) ** 2).mean().item())  # erasure physics uses raw unit keys
        heads.append({
            "head": h, "archival": h < getattr(mix, "archival_heads", 0),
            "beta_fact": round(beta[..., h][fact_v].mean().item(), 4),
            "beta_fill": round(beta[..., h][fill].mean().item(), 4),
            "lambda_fill": round((-mix.last_g.float()[..., h][fill]).mean().item(), 6),  # decay rate per filler token
            "overlap": round(sum(ov) / len(ov), 4),
            "c2_fill_fact": round(sum(c2) / len(c2), 5),
        })
        hd = heads[-1]
        rate = hd["lambda_fill"] + hd["beta_fill"] * hd["c2_fill_fact"]  # predicted loss rate per distractor token
        hd["efold_tokens"] = round(1.0 / rate, 1) if rate > 0 else float("inf")
    out[f"layer{i}:{blk.code}"] = heads
print(json.dumps(out, indent=1))
json.dump(out, open(f"{run}/key_analysis.json", "w"), indent=1)
