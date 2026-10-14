"""Check: chunked LFM2 language-modelling loss with and without a BOS token at the start of every chunk.

  python lm/check_bos_chunks.py --name LiquidAI/LFM2-350M --data data_local/fwe_lfm2 --chunk 1024 --n 16
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="LiquidAI/LFM2-350M")
ap.add_argument("--data", required=True)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--n", type=int, default=16, help="chunks")
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_data import token_dtype  # noqa: E402
m = AutoModelForCausalLM.from_pretrained(a.name, dtype=torch.bfloat16).cuda().eval()
bos = m.config.bos_token_id or 1
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
res = {}
with torch.no_grad():
    for use_bos in (True, False):
        losses = []
        for i in range(a.n):
            x = torch.from_numpy(val[i * a.chunk:(i + 1) * a.chunk].astype(np.int64)).cuda()[None]
            inp = torch.cat([torch.tensor([[bos]], device="cuda"), x], 1) if use_bos else x
            lg = m(inp).logits.float()[:, 1:] if use_bos else m(inp).logits.float()
            losses.append(F.cross_entropy(lg[0, :-1], x[0, 1:]).item())
        res["bos" if use_bos else "no_bos"] = round(float(np.mean(losses)), 4)
res.update(name=a.name, chunk=a.chunk, n_chunks=a.n)
print(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
