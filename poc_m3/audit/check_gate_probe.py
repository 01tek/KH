"""CPU replica of gate_probe.run_one with the real S1 checkpoint; shows lp_full == lp_abl by construction.
Usage: python check_gate_probe.py CKPT L N_EX
"""
import os, sys, time, random, math
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

torch.set_num_threads(12)
CKPT, L, N_EX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D = 768; DVC = "cpu"
ck = torch.load(CKPT, map_location="cpu", weights_only=True); sd = ck["state"]
print("ckpt step/ctx:", ck.get("step"), ck.get("ctx"), "| keys:", sorted(sd.keys()))
print(f"a(raw)={sd['a'].item():.5f}  sigmoid(a)={torch.sigmoid(sd['a']).item():.5f}  "
      f"|mp|={sd['mp.weight'].norm():.3f} |g.w|={sd['g.weight'].norm():.3f} g.bias mean={sd['g.bias'].mean():.4f} "
      f"sigmoid(ld)={[round(x,4) for x in torch.sigmoid(sd['ld']).tolist()[:3]]}.. sigmoid(lb)={torch.sigmoid(sd['lb']).mean():.4f} "
      f"rs mean={sd['rs'].mean():.4f}")

class HAISArm(nn.Module):  # verbatim from gate_probe.py:30-44
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1)); self.rs = nn.Parameter(torch.ones(D))
        self.k = nn.Linear(D,D,bias=False); self.v = nn.Linear(D,D,bias=False)
        self.mp = nn.Linear(D,D,bias=False); self.g = nn.Linear(D*2,D,bias=True)
    def forward(self, h, mem, ablate=False):
        a = 0.0 if ablate else torch.sigmoid(self.a)
        rms = torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-6)*self.rs
        gv = torch.sigmoid(self.g(torch.cat([h,mem],-1)))
        out = h + a*(h*rms*self.mp(gv*(mem-h)))
        return out, gv

arm = HAISArm().eval()
arm.a.data[:] = sd["a"]; arm.rs.data[:] = sd["rs"]
for n in ["k", "v", "mp", "g"]: getattr(arm, n).weight.data[:] = sd[f"{n}.weight"]
arm.g.bias.data[:] = sd["g.bias"]
print("params in probe arm:", sum(p.numel() for p in arm.parameters()), "(ckpt has lb/ld too; k/v loaded but never used in forward)")

cfg = AutoConfig.from_pretrained("state-spaces/mamba-130m-hf"); cfg.use_cache = False
M = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", config=cfg, torch_dtype=torch.float32,
                                         local_files_only=True).eval()
for p in M.parameters(): p.requires_grad_(False)

def run_one(seq_t, target, ablate=False):  # verbatim logic from gate_probe.py:86-103
    inp = torch.tensor([seq_t])
    with torch.no_grad():
        out = M(inp, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        logits = out.logits[0]
    mem = torch.zeros(D); gv_l, bn_l, lp_l, lpfix_l = [], [], [], []
    with torch.no_grad():
        for t in range(len(seq_t)):
            hx, gv = arm(h[t:t+1], mem.unsqueeze(0), ablate=ablate)
            gv_l.append(gv[0]); bn_l.append(torch.norm(hx[0]-h[t]).item())
            lp_l.append(F.log_softmax(logits[t:t+1], dim=-1)[0, target].item())
            # what the probe presumably intended: logits from the ARM output
            lpfix_l.append(F.log_softmax(M.lm_head(hx).float(), dim=-1)[0, target].item())
            mem = mem*0.99 + h[t]*(1-0.99)
    return torch.stack(gv_l), bn_l, lp_l, lpfix_l, logits.argmax(-1).tolist()

random.seed(0); SAFE = list(range(50, 100))
for i in range(N_EX):
    pos = random.randint(L*3//4, L-3); prefix = 200 + (i % 50); val = 300 + (i % 50)
    seq = [random.choice(SAFE) for _ in range(L)]; seq[pos] = prefix; seq[pos+1] = val
    t0 = time.time()
    gF, bF, lF, xF, pF = run_one(seq, val, False)
    gA, bA, lA, xA, pA = run_one(seq, val, True)
    dl = max(abs(a-b) for a, b in zip(lF, lA)); db = max(abs(a-b) for a, b in zip(bF, bA))
    print(f"[ex{i} L={L} pos={pos}] max|lp_full-lp_abl| over ALL positions = {dl:.3e}; pred_full==pred_abl: {pF==pA}; "
          f"gv_full==gv_abl: {torch.equal(gF,gA)}; max|bn_full-bn_abl| = {db:.3e}; gv mean={gF.mean():.4f}")
    print(f"    logprob(target) at needle pos (backbone logits) = {lF[pos]:.3f}; target rank-1 anywhere in last 300 preds: {val in pF[-300:]}")
    print(f"    if logits were taken from arm output hx: Δlp@needle={xF[pos]-xA[pos]:+.4e}, "
          f"mean|Δlp| over positions={sum(abs(a-b) for a,b in zip(xF,xA))/L:.4e}  ({time.time()-t0:.0f}s)")
