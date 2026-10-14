import torch, torch.nn as nn, torch.nn.functional as F, re, sys
D = 768; H = 12; Dh = 64; DVC = "cpu"
src = open(sys.argv[1]).read()
cls = src[src.index("class HAISArm"):src.index("arm = HAISArm()")]
exec(cls)                                   # patched HAISArm from b/gate_probe.py
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=True)["state"]
arm = HAISArm(); arm.load_state_dict(sd, strict=True); arm.eval()
ly = nn.Module()                            # S1 training forward, verbatim from s1_curriculum.py:41-58
ly.k = nn.Linear(D, D, 0); ly.v = nn.Linear(D, D, 0); ly.mp = nn.Linear(D, D, 0)
ly.g = nn.Linear(D * 2, D, 1); ly.a = nn.Parameter(torch.zeros(1)); ly.rs = nn.Parameter(torch.ones(D))
ly.lb = nn.Parameter(torch.zeros(H)); ly.ld = nn.Parameter(torch.zeros(H)); ly.load_state_dict(sd)
def fwd(h):
    B, L, _ = h.shape
    K = F.normalize(h @ ly.k.weight.t(), p=2, dim=-1).view(B, L, H, Dh)
    V = (h @ ly.v.weight.t()).view(B, L, H, Dh)
    S = torch.zeros(B, H, Dh, Dh, device=DVC, dtype=h.dtype)
    dq = torch.sigmoid(ly.ld).view(1, H, 1, 1); bt = torch.sigmoid(ly.lb).view(1, H, 1, 1); out = []
    for tk in range(L):
        k = K[:, tk]; vh = V[:, tk]
        Sk = torch.matmul(S, k.unsqueeze(-1))
        S = dq * S + bt * (vh.unsqueeze(-1) - Sk) @ k.unsqueeze(-2)
        rd = torch.matmul(S, k.unsqueeze(-1)).squeeze(-1).reshape(B, D)
        mem = rd @ ly.mp.weight.t()
        gv = torch.sigmoid(torch.cat([h[:, tk], mem], -1) @ ly.g.weight.t() + ly.g.bias)
        rms = h[:, tk].pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        out.append((1 - gv) * h[:, tk] + gv * (mem + ly.a * (h[:, tk] * rms * ly.rs)))
    return torch.stack(out, 1)
torch.manual_seed(0); h = torch.randn(96, D)
with torch.no_grad():
    ref = fwd(h[None])[0]; new, _ = arm(h); abl, _ = arm(h, ablate=True)
print(f"max|patched - S1 fwd| = {(new-ref).abs().max():.2e}   max|full - mem-ablated| = {(new-abl).abs().max():.3f}")
