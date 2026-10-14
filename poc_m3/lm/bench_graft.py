"""Throughput / memory benchmark of a graft training step (synthetic tokens): forward over all chunks, sampled CE
(10% of positions, chunked + checkpointed as in train_lm.py), backward into the adapters, AdamW step.

  python bench_graft.py --base Qwen/Qwen3.5-4B --insert 15,23,31 --heads 40 --chunk 2048 --bs 2,4,8
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
ap.add_argument("--insert", default="15,23,31")
ap.add_argument("--heads", type=int, default=40)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--bs", default="2,4,8")
ap.add_argument("--steps", type=int, default=6)
a = ap.parse_args()
if "gemma" in a.base.lower():
    from gemma_hais import GemmaHais as Graft
elif "qwen" in a.base.lower():
    from qwen_hais import QwenHais as Graft
else:
    from lfm2_hais import Lfm2Hais as Graft
model = Graft(a.base, insert_after=[int(i) for i in a.insert.split(",")], n_heads=a.heads, tau=0.5, chunk=a.chunk).cuda()
V = model.base.config.get_text_config().vocab_size
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True)
print(f"{a.base}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M trainable", flush=True)


def ce1(h, y):
    return F.cross_entropy(model.head(h).float(), y, reduction="none")


for bs in map(int, a.bs.split(",")):
    T = 2 * a.chunk
    try:
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.steps + 2):
            if i == 2:
                torch.cuda.synchronize(); t0 = time.time()
            x = torch.randint(0, V, (bs, T), device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                h, _ = model(x, None, head=False)
                sel = torch.rand(x.shape, device="cuda") < 0.1
                hs, ys = h[sel], x[sel]
                cs = max(512, (1 << 29) // V)
                loss = torch.cat([torch.utils.checkpoint.checkpoint(ce1, hs[j:j + cs], ys[j:j + cs], use_reentrant=False)
                                  for j in range(0, hs.shape[0], cs)]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        print(f"bs {bs} x T {T}: {bs * T / dt:,.0f} tok/s/GPU, {dt:.2f} s/step, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
              flush=True)
    except torch.OutOfMemoryError:
        print(f"bs {bs}: OOM", flush=True)
        break
