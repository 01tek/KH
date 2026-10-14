import os, sys, random, time
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoConfig
torch.set_num_threads(12); DVC = "cpu"
src = open(sys.argv[1]).read()
exec(src[src.index("SEP = 0"):src.index("def token_eval(")])        # make_ar_batch from patched file
CTX, NB = int(sys.argv[2]), int(sys.argv[3])
cfg = AutoConfig.from_pretrained("state-spaces/mamba-130m-hf"); cfg.use_cache = False
M = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", config=cfg, torch_dtype=torch.float32, local_files_only=True).eval()
r = random.Random(999); ok = tot = in_ctx = 0; t0 = time.time()
for _ in range(NB):
    X, Y = make_ar_batch(CTX, 8, r); m = Y != -100
    assert not (X[m] == Y[m]).any()                                   # answer never visible at scored pos
    with torch.no_grad():
        logits = M(X).logits
    pred = logits[m].argmax(-1); ok += (pred == Y[m]).sum().item(); tot += m.sum().item()
    vals = [set(row[(row >= 5000)].tolist()) for row in X]            # any stored key/value token
    rows = m.nonzero()[:, 0]; in_ctx += sum(p.item() in vals[i] for p, i in zip(pred, rows.tolist()))
print(f"L={CTX}: frozen Mamba-130m AR acc = {ok}/{tot} = {ok/tot:.3f}; preds that are SOME stored k/v token: {in_ctx/tot:.2f}; chance among 16 values = 0.0625 ({time.time()-t0:.0f}s)")
