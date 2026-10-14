"""Profile one LM training step: data loading vs forward/backward, and the top CUDA kernels.

  python profile_step.py --data ~/data/fwe_2b --d_model 768 --layout gdn,hais,... [--compile]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa: E402
from lm_data import LaneStream  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--layout", default="gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais")
ap.add_argument("--d_model", type=int, default=768)
ap.add_argument("--mem_heads", type=int, default=12)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--compile", default="", help="'' | blocks | mlp")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
cfg = {"mem_heads": a.mem_heads, "mem_head_dim": 64, "v_norm": True, "archival_heads": a.mem_heads // 2, "min_decay": 0.35}
model = TinyLM(50304, a.d_model, a.layout, cfg).cuda()
if a.compile == "mlp":
    for b in model.blocks:
        b.mlp = torch.compile(b.mlp)
elif a.compile == "blocks":
    for b in model.blocks:
        b.forward = torch.compile(b.forward)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
data = LaneStream(os.path.join(a.data, "train.bin"), tok, a.bs, a.T, seed=0, fact_rate=0.1, query_rate=1.0, drill_frac=0.25,
                  drill_k=8, max_age=2000, age_target="loguniform", max_pending=4096)


def _ce(hc, yc):
    return F.cross_entropy(model.head(hc).float().flatten(0, 1), yc.flatten(), reduction="none").view_as(yc)


states = None
t_data = t_step = 0.0
for i in range(8 + a.steps):
    t0 = time.time()
    x, y, ans = data.next()
    x, y, ans = x.cuda(non_blocking=True), y.cuda(non_blocking=True), ans.cuda(non_blocking=True)
    t1 = time.time()
    with torch.autocast("cuda", torch.bfloat16):
        h, new = model(x, states, head=False)
        ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce, h[:, s:s + 256], y[:, s:s + 256], use_reentrant=False)
                        for s in range(0, h.shape[1], 256)], 1)
        loss = ce.mean() + 5 * ce[ans].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    states = [tuple(z.detach() if torch.is_tensor(z) else z for z in st) for st in new]
    torch.cuda.synchronize()
    t2 = time.time()
    if i >= 8:
        t_data += t1 - t0
        t_step += t2 - t1
n = a.steps
tok_s = a.bs * a.T * n / (t_data + t_step)
print(f"data {1000 * t_data / n:.1f} ms/step, fwd+bwd+opt {1000 * t_step / n:.1f} ms/step, {tok_s:,.0f} tok/s, "
      f"params {model.n_params() / 1e6:.0f}M, ~{6 * model.n_params() * tok_s / 1e12:.0f} TFLOP/s (6N rule)")

from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(3):
        x, y, ans = data.next()
        x, y, ans = x.cuda(), y.cuda(), ans.cuda()
        with torch.autocast("cuda", torch.bfloat16):
            h, new = model(x, states, head=False)
            ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce, h[:, s:s + 256], y[:, s:s + 256], use_reentrant=False)
                            for s in range(0, h.shape[1], 256)], 1)
            loss = ce.mean() + 5 * ce[ans].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        states = [tuple(z.detach() if torch.is_tensor(z) else z for z in st) for st in new]
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=60))
