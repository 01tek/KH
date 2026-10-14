"""Feasibility of the KH-1 graft on a new base model: (1) chunked LM loss with a BOS per chunk, native vs the graft with
its gates at zero (must match); (2) training-step throughput and memory of the graft (adapters only trainable).

  python feasibility_graft.py --name LiquidAI/LFM2.5-8B-A1B --text ../../demo/books/pg1342.txt --insert 11,17,23 --heads 32
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--insert", required=True)
ap.add_argument("--heads", type=int, default=32)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--bs", default="1,2,4")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--experts", default="grouped_mm", help="transformers experts implementation for MoE bases ('' = default)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
from lfm2_hais import Lfm2Hais  # noqa: E402
res = {"name": a.name}
tok = AutoTokenizer.from_pretrained(a.name)
t0 = time.time()
model = Lfm2Hais(a.name, insert_after=[int(i) for i in a.insert.split(",")], n_heads=a.heads, tau=0.5, chunk=a.chunk).cuda()
if a.experts and hasattr(model.base, "set_experts_implementation"):
    try:
        model.base.set_experts_implementation(a.experts); res["experts"] = a.experts
    except Exception as e:  # noqa: BLE001
        res["experts_error"] = str(e)[:200]
res["load_s"] = round(time.time() - t0, 1)
res["trainable_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1)
res["total_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
ids = tok(open(a.text, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"][:8 * a.chunk]
x = torch.tensor(ids, device="cuda").view(8, a.chunk)
bos = model.base.config.bos_token_id
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    nat = model.base(torch.cat([torch.full((8, 1), bos, device="cuda"), x], 1)).logits[:, 1:].float()
    l_nat = F.cross_entropy(nat[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    nob = model.base(x).logits.float()
    l_nobos = F.cross_entropy(nob[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    g, _ = model(x, None)
    l_graft = F.cross_entropy(g.float()[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
res.update(loss_native_bos=round(l_nat, 4), loss_native_nobos=round(l_nobos, 4), loss_graft_gates0=round(l_graft, 4))
print(json.dumps(res), flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True)
V = model.base.config.vocab_size
for bs in map(int, a.bs.split(",")):
    T = 2 * a.chunk
    try:
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.steps + 1):
            if i == 1:
                torch.cuda.synchronize(); t1 = time.time()
            xb = torch.randint(0, V, (bs, T), device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                h, _ = model(xb, None, head=False)
                sel = torch.rand(xb.shape, device="cuda") < 0.1
                loss = F.cross_entropy(model.head(h[sel]).float(), xb[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); dt = (time.time() - t1) / a.steps
        res[f"bs{bs}_tok_s"] = round(bs * T / dt); res[f"bs{bs}_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        print(f"bs {bs}: {bs * T / dt:,.0f} tok/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        res[f"bs{bs}"] = "OOM"; print(f"bs {bs}: OOM", flush=True); break
print(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
