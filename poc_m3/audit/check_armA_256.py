import os, sys, random, statistics
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoConfig
torch.set_num_threads(12)
MID = "state-spaces/mamba-130m-hf"; CTX = int(sys.argv[1]); N = 200
cfg = AutoConfig.from_pretrained(MID); cfg.use_cache = False
M = AutoModelForCausalLM.from_pretrained(MID, config=cfg, torch_dtype=torch.float32, local_files_only=True).eval()
E = M.backbone.embeddings.weight.detach(); nrm = E.norm(dim=-1)
M.backbone.layers = M.backbone.layers[:1]
r = random.Random(42); exs = []
for _ in range(N):
    nl = r.randint(3, 5); ndl = [r.randint(5000, 50255) for _ in range(nl)]
    np_ = max(3, int(CTX * 0.9))
    if np_ + nl > CTX - 3: np_ = CTX - nl - 3
    exs.append((np_, ndl))
T = max(a + len(b) for a, b in exs)
X = torch.full((N, T), 50256, dtype=torch.long)
for i, (a, b) in enumerate(exs): X[i, a:a + len(b)] = torch.tensor(b)
with torch.no_grad():
    lg = torch.cat([M.lm_head(M(X[b:b+10], output_hidden_states=True).hidden_states[0]).argmax(-1) for b in range(0, N, 10)])
ok = tot = 0; fails = []; oks = []
for i, (a, b) in enumerate(exs):
    for j, t in enumerate(b):
        p = lg[i, a + j].item(); tot += 1
        if p == t: ok += 1; oks.append(t)
        else: fails.append(t)
print(f"CTX={CTX}: Arm A = {ok}/{tot} = {ok/tot:.4f}")
print(f"median ||E|| failed={statistics.median(nrm[fails].tolist()):.2f} vs correct={statistics.median(nrm[oks].tolist()):.2f}")
