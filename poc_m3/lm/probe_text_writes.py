"""Which ordinary-text tokens does a graft's archival memory write? Streams val text through the model and prints the
tokens with the most archival writes (beta > tau after the threshold), with context, plus write counts by token.

  python probe_text_writes.py --ckpt kh1/ckpt/kh1_350m_pilot4.pt --data data_local/fwe_lfm2 --tokens 65536 --tau 0.97
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--tokens", type=int, default=65536)
ap.add_argument("--tau", type=float, default=0.97)
ap.add_argument("--show", type=int, default=25)
ap.add_argument("--text_file", default=None, help="probe this text instead of val.bin")
a = ap.parse_args()

orig = layers.chunk_gated_delta_rule
REC = []
layers.chunk_gated_delta_rule = lambda q, k, v, g=None, beta=None, **kw: (REC.append(beta.detach()), orig(q, k, v, g=g, beta=beta, **kw))[1]
from transformers import AutoTokenizer  # noqa: E402
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
if cfg.get("family") == "gemma4":
    from gemma_hais import GemmaHais as Graft  # noqa: E402
elif cfg.get("family") == "qwen3_5":
    from qwen_hais import QwenHais as Graft  # noqa: E402
else:
    from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
              tau=a.tau, chunk=cfg["chunk"], freeze_base=True).cuda().eval()
model.load_state_dict(ck["model"], strict=False)
for ad in model.adapters.values():
    ad.mixer.arch_tau = a.tau
tok = AutoTokenizer.from_pretrained(cfg["name"])
C = model.chunk
if a.text_file:
    ids = tok(open(a.text_file, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"][:a.tokens]
else:
    from lm_data import token_dtype  # noqa: E402
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
    ids = val[1_000_000:1_000_000 + a.tokens].astype(np.int64).tolist()
ids = ids[:len(ids) // C * C]
writes = torch.zeros(len(ids))  # number of (layer, head) pairs writing at each token
st = None
with torch.no_grad():
    for i in range(0, len(ids), C):
        REC.clear()
        with torch.autocast("cuda", torch.bfloat16):
            _, st = model(torch.tensor(ids[i:i + C], device="cuda")[None], st, head=False)
        writes[i:i + C] = sum((b[0] > 0).float().sum(-1).cpu() for b in REC)
n_heads = sum(len(range(cfg["n_heads"])) for _ in cfg["insert_after"])
print(f"{len(ids):,} tokens; tokens with any archival write: {(writes > 0).float().mean():.4f}; "
      f"mean writing heads per token {writes.mean():.3f} of {n_heads}")
top = writes.argsort(descending=True)[:a.show].tolist()
for t in sorted(top):
    ctx = tok.decode(ids[max(0, t - 12):t]).replace("\n", " ")
    print(f"  {int(writes[t]):3d} heads  ...{ctx[-70:]}[[{tok.decode([ids[t]])}]]{tok.decode(ids[t + 1:t + 4])!r}")
cnt = collections.Counter()
for t in (writes > 0).nonzero().flatten().tolist():
    cnt[tok.decode([ids[t]])] += int(writes[t])
print("most-written token strings:", cnt.most_common(25))
