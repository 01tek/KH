"""Fine-tune KH-1 archival adapters on the BABILong training split (RMT-team/babilong-train-5k-samples): stream each
sample's text through the memory (gradient through every chunk), ask its question as a chat turn, and train only on the
answer tokens. The frozen base never sees the text and the question in the same chunk unless the text is shorter than
one chunk, so long samples can only be answered from the archival memory.

  python lm/babilong_train.py --init ../kh1/ckpt/kh1_350m_pilot6.pt --tasks qa1,qa2,qa3,qa4,qa5 \
      --lengths 0k,1k,2k,4k,8k,16k --samples 6000 --tau 0.97 --out runs_babilong/kh1_350m_bl
The test split (RMT-team/babilong) is never read here.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="KH-1 checkpoint (.pt or release folder) whose adapters we start from")
ap.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5")
ap.add_argument("--lengths", default="0k,1k,2k,4k,8k,16k")
ap.add_argument("--samples", type=int, default=6000)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--tau", type=float, default=0.97)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed)
torch.manual_seed(a.seed)

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from kh1_load import load_adapters  # noqa: E402

cfg, sd = load_adapters(a.init)
fam = cfg.get("family")
if fam == "gemma4":
    from gemma_hais import GemmaHais as Graft  # noqa: E402
elif fam == "qwen3_5":
    from qwen_hais import QwenHais as Graft  # noqa: E402
else:
    from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
              tau=a.tau, chunk=cfg["chunk"], freeze_base=True).cuda()
model.load_state_dict(sd, strict=False)
for ad in model.adapters.values():
    ad.mixer.arch_tau = a.tau
model.train()
tok = AutoTokenizer.from_pretrained(cfg["name"])
C = model.chunk
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
END = next(t for t in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id)
           if isinstance(t, int) and t >= 0)

from huggingface_hub import list_repo_files  # noqa: E402
have = set(list_repo_files("RMT-team/babilong-train-5k-samples", repo_type="dataset"))
pool, used = [], []
for task in a.tasks.split(","):
    for L in a.lengths.split(","):
        f = f"data/{task}/{L}.json"
        if f not in have:  # not every task has every length in the training split
            continue
        rows = json.load(open(hf_hub_download("RMT-team/babilong-train-5k-samples", f, repo_type="dataset")))
        pool += [(task, L, r) for r in rows]
        used.append(f"{task}/{L}")
print("training files:", used, flush=True)
random.shuffle(pool)
pool = pool[:a.samples]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
log = open(os.path.join(a.out, "log.jsonl"), "w")
log.write(json.dumps({"event": "config", **vars(a), "cfg": cfg, "pool": len(pool)}) + "\n")
t0, hits, losses, ntok = time.time(), [], [], 0
for i, (task, L, s) in enumerate(pool):
    kw = {"enable_thinking": False} if fam == "qwen3_5" else {}
    turn = tok.apply_chat_template([{"role": "user", "content": s["question"].strip() + " Answer with one word."}],
                                   tokenize=False, add_generation_prompt=True, **kw)
    q = enc("\n" + turn.replace(tok.bos_token or "\0", ""))
    ans = enc(s["target"]) + [END]
    text = enc(s["input"])
    st = None
    with torch.autocast("cuda", torch.bfloat16):
        for j in range(0, len(text), C):  # gradient flows through the memory across every chunk
            _, st = model(torch.tensor(text[j:j + C], device="cuda")[None], st, head=False)
        x = torch.tensor(q + ans[:-1], device="cuda")[None]
        h, _ = model(x, st, head=False)
        logits = model.head(h[0, len(q) - 1:]).float()
    loss = F.cross_entropy(logits, torch.tensor(ans, device="cuda"))
    (loss / a.accum).backward()
    hits.append(int(logits[0].argmax()) == ans[0])
    losses.append(loss.item())
    ntok += len(text)
    if (i + 1) % a.accum == 0:
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if (i + 1) % 200 == 0:
        rec = {"event": "train", "sample": i + 1, "loss": round(sum(losses) / len(losses), 4),
               "first_token_acc": round(sum(hits) / len(hits), 3), "grad_norm": round(gn, 3), "tok_per_s": round(ntok / (time.time() - t0))}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        hits, losses = [], []
torch.save({"cfg": {**cfg, "tau": a.tau}, "model": model.adapter_state_dict(), "args": vars(a)}, os.path.join(a.out, "ckpt.pt"))
log.write(json.dumps({"event": "done", "train_time_s": round(time.time() - t0)}) + "\n")
print("saved", os.path.join(a.out, "ckpt.pt"))
