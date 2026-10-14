"""Gradient check for the gated delta rule on this GPU/Triton: fla's chunked Triton kernel vs fla's
pure-PyTorch reference (naive_recurrent_gated_delta_rule, fp32 autograd).
Guards against silent backward corruption (fla #640: Triton 3.4-3.7.0 on Hopper)."""
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

torch.manual_seed(0)
B, T, H, D = 2, 512, 4, 64
dev, dt = "cuda", torch.bfloat16


def inputs():
    q = torch.randn(B, T, H, D, device=dev, dtype=dt)
    k = torch.randn(B, T, H, D, device=dev, dtype=dt)
    v = torch.randn(B, T, H, D, device=dev, dtype=dt)
    g = F.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32) + 3)
    beta = torch.rand(B, T, H, device=dev, dtype=dt)
    return [x.requires_grad_() for x in (q, k, v, g, beta)]


base = inputs()
w = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
res = {}
for name in ["chunk", "reference"]:
    xs = [x.detach().clone().requires_grad_() for x in base]
    qn, kn = F.normalize(xs[0], dim=-1), F.normalize(xs[1], dim=-1)
    if name == "chunk":
        o, _ = chunk_gated_delta_rule(qn, kn, xs[2], g=xs[3], beta=xs[4])
    else:
        o, _ = naive_recurrent_gated_delta_rule(qn.float(), kn.float(), xs[2].float(), xs[4].float(), xs[3])
    (o.float() * w).sum().backward()
    res[name] = [o.float()] + [x.grad.float() for x in xs]
names = ["out", "dq", "dk", "dv", "dg", "dbeta"]
worst = 0.0
for n, a, b in zip(names, res["chunk"], res["reference"]):
    rel = ((a - b).norm() / (b.norm() + 1e-6)).item()
    worst = max(worst, rel)
    print(f"{n:6s} rel_err={rel:.2e}")
assert worst < 5e-2, f"GDN chunk vs reference mismatch: {worst}"
print("GDN_GRAD_OK")
