"""Exact replica of s1_scaling_v2.token_eval(n=200, ctx=2048) for Arm A, batched, CPU.
Usage: python check_pilot_armA_full.py fp32|bf16 CTX N BS
"""
import os, sys, time, random
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoConfig

torch.set_num_threads(12)
MID = "state-spaces/mamba-130m-hf"
dt = torch.float32 if sys.argv[1] == "fp32" else torch.bfloat16
CTX, N, BS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cfg = AutoConfig.from_pretrained(MID); cfg.use_cache = False
M = AutoModelForCausalLM.from_pretrained(MID, config=cfg, torch_dtype=dt, local_files_only=True).eval()
for p in M.parameters(): p.requires_grad_(False)
M.backbone.layers = M.backbone.layers[:1]  # hidden_states[0] == output of block 0

r = random.Random(42); exs = []
for _ in range(N):
    nl = r.randint(3, 5); ndl = [r.randint(5000, 50255) for _ in range(nl)]
    np_ = max(3, int(CTX * 0.9))
    if np_ + nl > CTX - 3: np_ = CTX - nl - 3
    exs.append((np_, ndl))
T = max(np_ + len(ndl) for np_, ndl in exs)  # right-truncate: causal model
X = torch.full((N, T), 50256, dtype=torch.long)
for i, (np_, ndl) in enumerate(exs): X[i, np_:np_ + len(ndl)] = torch.tensor(ndl)

ok = tot = 0; t0 = time.time()
with torch.no_grad():
    for b in range(0, N, BS):
        h = M(X[b:b + BS], output_hidden_states=True).hidden_states[0].float()
        lg = M.lm_head(h.to(M.dtype))
        for i in range(h.shape[0]):
            np_, ndl = exs[b + i]
            for j, t in enumerate(ndl):
                ok += lg[i, np_ + j].argmax(-1).item() == t; tot += 1
print(f"dtype={dt} CTX={CTX} N={N}: Arm A acc = {ok}/{tot} = {ok/tot:.4f}  ({time.time()-t0:.0f}s)")
print(f"needle start={exs[0][0]}, total eval tokens={tot}; 1 token = {1/tot:.4f} acc")
