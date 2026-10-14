"""CPU-only forensic replication of the v1.0 S1-curriculum eval (s1_curriculum.py:41-80).
Usage: python replicate_v1.py <ckpt.pt> <ctx_list> <n> <mode_list>
Modes:
  active            - verbatim v1.0 eval: lm_head(fwd(hidden_states[0]))         (log: "active")
  ablated           - verbatim v1.0 ablation: lm_head(hidden_states[0]), adapter skipped (log: "ablated")
  S0                - adapter kept, but S-matrix forced to 0 every step (mem = 0)  (paper's claimed ablation)
  past0             - adapter kept, S reset before each write: readout = beta*v_t only (no memory of earlier tokens)
  raw_embed         - lm_head(token_embedding) -- tied-embedding identity rate, no layer at all
  no_hay_needle     - active, but the needle is NOT inserted in the haystack (random tokens instead)
  shifted_active    - active adapter, labels shifted to proper next-token recall (pos p predicts token p+1)
  shifted_ablated   - same labels, adapter skipped
Eval data generation consumes random.Random(9999) exactly as the original; controls use a separate RNG.
"""
import os, sys, time, random
DEV = os.environ.get("DEVICE", "cpu")          # DEVICE=cuda to run on GPU (needs <1 GB VRAM at L<=2048; untested here)
if DEV == "cpu": os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "12")))
from transformers import AutoModelForCausalLM

CK = sys.argv[1]
CTXS = [int(x) for x in sys.argv[2].split(",")]
N = int(sys.argv[3])
MODES = sys.argv[4].split(",")
FULL_BACKBONE = os.environ.get("FULL_BACKBONE", "0") == "1"

M = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.bfloat16)
M.to(DEV); M.eval(); [p.requires_grad_(False) for p in M.parameters()]
d = M.config.hidden_size; V = M.config.vocab_size; H = 12; Dh = d // 12
print(f"tied lm_head==embeddings: {M.lm_head.weight.data_ptr() == M.backbone.embeddings.weight.data_ptr()}  "
      f"n_hidden_states={M.config.num_hidden_layers}+1", flush=True)

ly = nn.Module()
ly.k = nn.Linear(d, d, 0); ly.v = nn.Linear(d, d, 0); ly.mp = nn.Linear(d, d, 0)
ly.g = nn.Linear(d * 2, d, 1); ly.a = nn.Parameter(torch.zeros(1)); ly.rs = nn.Parameter(torch.ones(d))
ly.lb = nn.Parameter(torch.zeros(H)); ly.ld = nn.Parameter(torch.zeros(H))
ly.load_state_dict(torch.load(CK, map_location="cpu")["state"]); ly.to(DEV); ly.eval()

def gn(r): return [r.randint(5000, 50255) for _ in range(r.randint(3, 5))]

def fwd(h, zeroS=False, past0=False):  # s1_curriculum.py:41-58, DVC->cpu, optional S=0 / no-past-memory
    B, L, _ = h.shape
    K = F.normalize(h @ ly.k.weight.t(), p=2, dim=-1).view(B, L, H, Dh)
    Vv = (h @ ly.v.weight.t()).view(B, L, H, Dh)
    S = torch.zeros(B, H, Dh, Dh, device=h.device, dtype=h.dtype)
    dq = torch.sigmoid(ly.ld).view(1, H, 1, 1)
    bt = torch.sigmoid(ly.lb).view(1, H, 1, 1)
    out = []
    for tk in range(L):
        k = K[:, tk]; vh = Vv[:, tk]
        if past0:
            S = torch.zeros_like(S)   # forget everything written at earlier positions; keep only current write
        if not zeroS:
            Sk = torch.matmul(S, k.unsqueeze(-1))
            S = dq * S + bt * (vh.unsqueeze(-1) - Sk) @ k.unsqueeze(-2)
        rd = torch.matmul(S, k.unsqueeze(-1)).squeeze(-1).reshape(B, d)
        mem = rd @ ly.mp.weight.t()
        gv = torch.sigmoid(torch.cat([h[:, tk], mem], -1) @ ly.g.weight.t() + ly.g.bias)
        rms = h[:, tk].pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        out.append((1 - gv) * h[:, tk] + gv * (mem + ly.a * (h[:, tk] * rms * ly.rs)))
    return torch.stack(out, 1)

@torch.no_grad()
def hs0(x):
    # HF Mamba: hidden_states[0] is the OUTPUT of block 0 (appended after each block), not the raw embedding.
    if FULL_BACKBONE:
        return M(x, output_hidden_states=True).hidden_states[0]
    return M.backbone.layers[0](M.backbone.embeddings(x))

@torch.no_grad()
def ev(ctx, n, mode):  # s1_curriculum.py:65-80 + control modes
    r = random.Random(9999); alt = random.Random(424242); c = t = 0
    PROG = int(os.environ.get("PROGRESS", "0"))
    for i_ex in range(n):
        if PROG and i_ex and i_ex % PROG == 0:
            print(f"   [progress L={ctx} mode={mode}] {i_ex}/{n} examples: acc={c/max(t,1):.4f} ({c}/{t})", flush=True)
        nt = gn(r); nl = len(nt)
        se = [r.randint(5000, 50255) for _ in range(ctx)]
        np_ = r.randint(3, ctx // 3); seq = (se[:np_] + nt + se[np_:])[:ctx]
        if mode == "no_hay_needle":
            seq = (se[:np_] + [alt.randint(5000, 50255) for _ in range(nl)] + se[np_:])[:ctx]
        fu = seq + [50256] + nt; la = [-100] * len(fu)
        if mode.startswith("shifted"):
            for j in range(nl): la[len(fu) - nl + j - 1] = nt[j]   # proper causal target: pos p -> token p+1
        else:
            for j in range(nl): la[len(fu) - nl + j] = nt[j]       # verbatim v1.0 (unshifted)
        pl = ((len(fu) + 16) // 8) * 8
        x = torch.tensor([fu + [50256] * (pl - len(fu))], device=DEV)
        lab = torch.tensor([la + [-100] * (pl - len(la))], device=DEV)
        if mode == "raw_embed":
            h = M.backbone.embeddings(x).float()
        else:
            h = hs0(x).float()
            if mode in ("active", "no_hay_needle", "shifted_active"): h = fwd(h)
            elif mode == "S0": h = fwd(h, zeroS=True)
            elif mode == "past0": h = fwd(h, past0=True)
        msk = lab != -100
        # identical argmax to lm_head(h)[msk]; computed only on scored rows for CPU speed
        ps = M.lm_head(h[msk].to(M.dtype)).argmax(-1)
        c += (ps == lab[msk]).sum().item(); t += msk.sum().item()
    return c / max(t, 1), c, t

if os.environ.get("CHECK_EQUIV", "0") == "1":
    x = torch.randint(5000, 50255, (1, 64), device=DEV)
    with torch.no_grad():
        full = M(x, output_hidden_states=True).hidden_states
        emb = M.backbone.embeddings(x)
        short = M.backbone.layers[0](emb)
    print(f"len(hidden_states)={len(full)}; max|hs[0]-block0(emb)|={(full[0].float()-short.float()).abs().max():.3e}; "
          f"max|hs[0]-emb|={(full[0].float()-emb.float()).abs().max():.3e}; "
          f"rel||hs[0]-emb||/||emb||={((full[0].float()-emb.float()).norm()/emb.float().norm()):.3f}", flush=True)

print(f"ckpt={CK} n={N} full_backbone={FULL_BACKBONE} device={DEV}", flush=True)
for ctx in CTXS:
    for mode in MODES:
        t0 = time.time(); acc, c, t = ev(ctx, N, mode)
        print(f"L={ctx:5d} mode={mode:16s} acc={acc:.4f} ({c}/{t})  [{time.time()-t0:.0f}s]", flush=True)
