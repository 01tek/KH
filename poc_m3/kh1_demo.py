"""Scripted, verifiable KH-1 demo: chat, then read >1M tokens of books with facts planted near the start, then ask
about those facts in chat. Logs answers, GPU memory and speed; writes a JSON transcript.

  python kh1_demo.py --ckpt runs_kh1/kh1_1p2b/ckpt.pt --books demo/books/pg2600.txt demo/books/pg2701.txt demo/books/pg1342.txt
  python kh1_demo.py ... --no_memory     # control: same base model with the archival memory switched off
Works for LFM2 (KH-1) and Qwen3.5 (KH-1-Q) checkpoints. Scoring is exact: the reply's first word must equal the value.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (sentence planted in the text, chat question, answer prefix the reply should start with, expected value, phrasing seen in training?)
FACTS = [
    ("Velthu's locker number is 482.", "What is Velthu's locker number?", "Velthu's locker number is", "482", True),
    ("The wifi password at Kribrisen is thuzoka.", "What is the wifi password at Kribrisen?",
     "The wifi password at Kribrisen is", "thuzoka", True),
    ("Moraska told me that their safe combination is 739561.", "What did Moraska say their safe combination is?",
     "Their safe combination is", "739561", True),
    ("The birth month of Galfen is September.", "What is the birth month of Galfen?",
     "The birth month of Galfen is", "September", True),
    ("Officially, Sabtor's radio frequency reads 287.", "Officially, what does Sabtor's radio frequency read?",
     "It reads", "287", False),
]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--books", nargs="+", required=True)
ap.add_argument("--no_think", action="store_true", help="Qwen3.5-4B: chat in non-thinking mode (match --no_think training)")
ap.add_argument("--quant", default=None, choices=["nf4", "int8"], help="Qwen grafts: quantise the frozen base (small GPUs)")
ap.add_argument("--tau", type=float, default=0.9)
ap.add_argument("--tau_hi", type=float, default=None, help="mixed thresholds: the last --hi_frac of the heads use this")
ap.add_argument("--hi_frac", type=float, default=0.5)
ap.add_argument("--no_memory", action="store_true", help="control: zero the archival gates (plain chunked LFM2)")
ap.add_argument("--plant_at", type=float, default=0.02, help="facts go in at this fraction of the first book")
ap.add_argument("--out", default=None)
a = ap.parse_args()

from kh1_load import load_adapters  # noqa: E402
cfg, adapter_sd = load_adapters(a.ckpt)
if cfg.get("family") == "qwen3_5":
    from qwen_hais import QwenHais as Graft  # noqa: E402
else:
    from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
              tau=a.tau, chunk=cfg["chunk"], freeze_base=True, **({"quant": a.quant} if a.quant else {}))
if a.quant:
    model.adapters.cuda()  # a bitsandbytes base is already on the GPU
else:
    model.cuda()
model.eval()
model.load_state_dict(adapter_sd, strict=False)
for ad in model.adapters.values():
    A = ad.mixer.archival_heads
    n_hi = int(round(a.hi_frac * A)) if a.tau_hi is not None else 0
    ad.mixer.arch_tau = a.tau if not n_hi else torch.tensor([a.tau] * (A - n_hi) + [a.tau_hi] * n_hi, device="cuda")
    if a.no_memory:
        ad.alpha.data.zero_()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(cfg["name"])
C = model.chunk
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
log = {"ckpt": a.ckpt, "base": cfg["name"], "tau": a.tau, "tau_hi": a.tau_hi, "hi_frac": a.hi_frac, "no_memory": a.no_memory,
       "turns": []}


@torch.no_grad()
def feed(ids, st):
    for i in range(0, len(ids), C):
        x = torch.tensor(ids[i:i + C], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            _, st = model(x, st, head=False)
    return st


@torch.no_grad()
def reply(user, prefix, st, max_new=24):
    turn = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   **({"enable_thinking": False} if a.no_think else {}))
    ids = enc("\n" + turn.replace(tok.bos_token or "\0", "") + prefix)
    out = []
    for _ in range(max_new):
        x = torch.tensor(ids[-C:], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            h, _ = model(x, st, head=False)
        nxt = int(model.head(h[:, -1]).argmax(-1))
        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out).strip(), feed(ids, st)  # commit the turn once


torch.cuda.reset_peak_memory_stats()
st = None
r, st = reply("Hi! Who are you?", "", st, max_new=40)
print("user> Hi! Who are you?\nkh1>", r)
log["turns"].append({"user": "Hi! Who are you?", "kh1": r})

texts = [open(p, encoding="utf-8", errors="ignore").read() for p in a.books]
cut = int(a.plant_at * len(texts[0]))
cut = texts[0].rfind("\n\n", 0, cut) + 2  # plant at a paragraph break
texts[0] = texts[0][:cut] + " ".join(f[0] for f in FACTS) + "\n\n" + texts[0][cut:]
ids = [t for txt in texts for t in enc(txt)]
planted_at = len(enc(texts[0][:cut]))
t0 = time.time()
step = 64 * C
for i in range(0, len(ids), step):
    st = feed(ids[i:i + step], st)
    print(f"\r  reading: {min(i + step, len(ids)):,}/{len(ids):,} tokens  {min(i + step, len(ids)) / (time.time() - t0):,.0f} tok/s"
          f"  GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB (peak {torch.cuda.max_memory_allocated() / 1e9:.2f})",
          end="", flush=True)
read_s = time.time() - t0
print(f"\n  read {len(ids):,} tokens in {read_s:.0f}s; facts were planted at token {planted_at:,}")
log.update(tokens_read=len(ids), read_seconds=round(read_s, 1), planted_at=planted_at,
           peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))

score = 0
for fact, q, prefix, want, seen in FACTS:
    r, st = reply(q, prefix, st)
    got = r.split(".")[0].strip().split()
    ok = bool(got) and got[0].strip(",;:!?'\"").lower() == want.lower()  # exact: the first word of the reply is the value
    score += ok
    dist = len(ids) - planted_at
    print(f"user> {q}\nkh1> {prefix} {r}    [{'OK' if ok else 'MISS'}; truth {want}; ~{dist:,} tokens after the fact; "
          f"{'trained' if seen else 'UNSEEN'} phrasing]")
    log["turns"].append({"user": q, "kh1": f"{prefix} {r}", "truth": want, "correct": bool(ok), "distance": dist,
                         "trained_phrasing": seen})
print(f"\n{score}/{len(FACTS)} correct after {len(ids):,} tokens | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
log["score"] = f"{score}/{len(FACTS)}"
if a.out:
    json.dump(log, open(a.out, "w"), indent=1)
