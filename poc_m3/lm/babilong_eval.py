"""BABILong (RMT-team/babilong) for KH-1 grafts: stream each sample's long text through the archival memory, then ask
its question in chat. Baselines on the same samples: the frozen base with the memory off (it sees only the question),
and RAG (BM25 top-k sentences of the text put in the prompt, memory off).

  python lm/babilong_eval.py --ckpt KH-1-350M --tau 0.99 --tasks qa1,qa2,qa3,qa4,qa5 --lengths 0k,4k,16k,64k,128k --n 50
  python lm/babilong_eval.py ... --mode rag      # or --mode nomem
Scoring follows BABILong: correct if the target word appears among the first words of the reply (lower-cased).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tau", type=float, default=0.99)
ap.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5")
ap.add_argument("--lengths", default="0k,4k,16k,64k,128k")
ap.add_argument("--n", type=int, default=50, help="samples per task and length (the dataset has 100)")
ap.add_argument("--mode", default="memory", choices=["memory", "nomem", "rag"])
ap.add_argument("--k", type=int, default=10, help="RAG: sentences retrieved")
ap.add_argument("--quant", default=None, choices=[None, "nf4", "int8"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from kh1_load import load_adapters  # noqa: E402

cfg, adapter_sd = load_adapters(a.ckpt)
fam = cfg.get("family")
if fam == "gemma4":
    from gemma_hais import GemmaHais as Graft  # noqa: E402
elif fam == "qwen3_5":
    from qwen_hais import QwenHais as Graft  # noqa: E402
else:
    from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
              tau=a.tau, chunk=cfg["chunk"], freeze_base=True, **({"quant": a.quant} if a.quant else {}))
(model.adapters.cuda() if a.quant else model.cuda()).eval()
model.load_state_dict(adapter_sd, strict=False)
for ad in model.adapters.values():
    ad.mixer.arch_tau = a.tau
    if a.mode != "memory":
        ad.alpha.data.zero_()
tok = AutoTokenizer.from_pretrained(cfg["name"])
C = model.chunk
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
END = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<end_of_turn>"))
       if isinstance(t, int) and t >= 0}


@torch.no_grad()
def feed(ids, st):
    for i in range(0, len(ids), C):
        x = torch.tensor(ids[i:i + C], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            _, st = model(x, st, head=False)
    return st


@torch.no_grad()
def reply(user, st, max_new=12):
    kw = {"enable_thinking": False} if fam == "qwen3_5" else {}
    turn = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)
    ids = enc("\n" + turn.replace(tok.bos_token or "\0", ""))
    out = []
    for _ in range(max_new):
        x = torch.tensor(ids[-C:], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            h, _ = model(x, st, head=False)
        nxt = int(model.head(h[:, -1]).argmax(-1))
        if nxt in END:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out).strip()


WORD = re.compile(r"[a-z]+")


def bm25_top(text, question, k):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    docs = [WORD.findall(s.lower()) for s in sents]
    q = [w for w in WORD.findall(question.lower()) if w not in {"where", "is", "the", "was", "what", "who", "did", "a", "to", "before", "of", "in"}]
    N, avg = len(docs), sum(map(len, docs)) / max(1, len(docs))
    df = Counter(w for d in docs for w in set(d))
    def score(d):
        tf = Counter(d)
        return sum(math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
                   for w in q if w in tf)
    top = sorted(range(N), key=lambda i: -score(docs[i]))[:k]
    return " ".join(sents[i] for i in sorted(top))  # keep the story order


def correct(reply_text, target):
    return target.lower() in WORD.findall(reply_text.lower())[:6]


res = {"ckpt": a.ckpt, "base": cfg["name"], "tau": a.tau, "mode": a.mode, "k": a.k, "n": a.n, "scores": {}, "samples": []}
for task in a.tasks.split(","):
    for L in a.lengths.split(","):
        data = json.load(open(hf_hub_download("RMT-team/babilong", f"data/{task}/{L}.json", repo_type="dataset")))[:a.n]
        hits, t0 = 0, time.time()
        for i, s in enumerate(data):
            q = s["question"].strip() + " Answer with one word."
            if a.mode == "memory":
                st = feed(enc(s["input"]), None)
                r = reply(q, st)
            elif a.mode == "rag":
                r = reply("Context: " + bm25_top(s["input"], s["question"], a.k) + "\nQuestion: " + q, None)
            else:
                r = reply(q, None)
            ok = correct(r, s["target"])
            hits += ok
            res["samples"].append({"task": task, "len": L, "i": i, "target": s["target"], "reply": r, "ok": ok})
        res["scores"][f"{task}/{L}"] = round(hits / len(data), 3)
        print(f"{a.mode} {task} {L}: {hits}/{len(data)} = {hits / len(data):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res["scores"]))
