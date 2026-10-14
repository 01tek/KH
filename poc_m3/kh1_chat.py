"""KH-1 chat: LFM2 + archival memory (lfm2_hais.py), constant memory at any context length.

  python kh1_chat.py --ckpt runs_lfm2/lfm2_350m_pilot/ckpt.pt [--tau 0.9]

Commands:
  <text>            chat turn (the model answers; the turn is then committed to memory)
  /load FILE        stream a whole text file into memory (any length), with live tokens/s and GPU memory
  /plant TEXT       insert a sentence into memory as plain text (for planting facts inside a long document)
  /ask Q || PREFIX  ask Q and let the answer start with PREFIX (e.g. "Pierre's locker number is")
  /mem              show tokens seen, GPU memory, archival state size
  /reset            forget everything
  /quit

LFM2's own layers only ever see the current chunk (<= C tokens), so the KV/conv memory is bounded; everything older lives
in the fixed-size archival state. Generation re-runs the current turn's chunk from a snapshot of the archival state,
and the state is committed once per turn, so memory is written exactly once per token.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--no_think", action="store_true", help="Qwen3.5-4B: chat in non-thinking mode (match --no_think training)")
ap.add_argument("--quant", default=None, choices=["nf4", "int8"], help="Qwen grafts: quantise the frozen base (small GPUs)")
ap.add_argument("--tau", type=float, default=0.9, help="archival write threshold at inference")
ap.add_argument("--max_new", type=int, default=96)
ap.add_argument("--temperature", type=float, default=0.3, help="free chat sampling (LFM2's recommended 0.3); /ask is greedy")
ap.add_argument("--min_p", type=float, default=0.15)
ap.add_argument("--rep_penalty", type=float, default=1.05)
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
    ad.mixer.arch_tau = a.tau
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(cfg["name"])
C = model.chunk
state, seen = None, 0


def gpu_gb():
    return torch.cuda.memory_allocated() / 1e9


@torch.no_grad()
def feed(ids, st):
    """Stream token ids through the model in chunks of C, carrying the archival state. Returns (last hidden, state)."""
    h = None
    for i in range(0, len(ids), C):
        x = torch.tensor(ids[i:i + C], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            h, st = model(x, st, head=False)
    return h, st


@torch.no_grad()
def generate(prompt_ids, st, max_new, greedy=True):
    """Answer to a prompt that fits in one chunk (greedy, or LFM2's recommended sampling). The archival state st is
    read but not committed here."""
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new):
        x = torch.tensor(ids[-C:], device="cuda")[None]
        with torch.autocast("cuda", torch.bfloat16):
            h, _ = model(x, st, head=False)
        logits = model.head(h[:, -1]).float()[0]
        if greedy:
            nxt = int(logits.argmax())
        else:
            if out and a.rep_penalty != 1.0:
                seen_ids = torch.tensor(sorted(set(out)), device=logits.device)
                lv = logits[seen_ids]
                logits[seen_ids] = torch.where(lv > 0, lv / a.rep_penalty, lv * a.rep_penalty)
            probs = torch.softmax(logits / max(a.temperature, 1e-5), -1)
            probs = torch.where(probs >= a.min_p * probs.max(), probs, torch.zeros_like(probs))
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def chat_ids(user, prefix=""):
    msgs = [{"role": "user", "content": user}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  **({"enable_thinking": False} if a.no_think else {})) + prefix
    txt = "\n" + txt.replace(tok.bos_token or "\0", "")  # the model adds a BOS to every chunk itself
    return tok(txt, add_special_tokens=False)["input_ids"]


print(f"KH-1 ({cfg['name']} + {len(model.adapters)} archival layers, chunk {C}, write threshold {a.tau}). "
      f"GPU {gpu_gb():.2f} GB. /help for commands.")
while True:
    try:
        line = input("\nyou> ").strip()
    except EOFError:
        break
    if not line:
        continue
    if line in ("/quit", "/exit"):
        break
    if line == "/help":
        print(__doc__)
        continue
    if line == "/reset":
        state, seen = None, 0
        print("memory cleared")
        continue
    if line == "/mem":
        n = sum(s[0].numel() for s in state) if state else 0
        print(f"tokens seen {seen:,} | GPU allocated {gpu_gb():.2f} GB (peak {torch.cuda.max_memory_allocated() / 1e9:.2f}) "
              f"| archival state {n:,} floats ({n * 4 / 1e6:.1f} MB), fixed")
        continue
    if line.startswith("/load "):
        path = line[6:].strip()
        text = open(path, encoding="utf-8", errors="ignore").read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        t0 = time.time()
        step = 64 * C
        for i in range(0, len(ids), step):
            _, state = feed(ids[i:i + step], state)
            seen += len(ids[i:i + step])
            el = time.time() - t0
            print(f"\r  loaded {min(i + step, len(ids)):,}/{len(ids):,} tokens  {min(i + step, len(ids)) / el:,.0f} tok/s  "
                  f"GPU {gpu_gb():.2f} GB", end="", flush=True)
        print(f"\n  done: {len(ids):,} tokens in {time.time() - t0:.0f}s")
        continue
    if line.startswith("/plant "):
        ids = tok(" " + line[7:].strip(), add_special_tokens=False)["input_ids"]
        _, state = feed(ids, state)
        seen += len(ids)
        print(f"  planted ({len(ids)} tokens)")
        continue
    if line.startswith("/ask "):
        q, _, prefix = line[5:].partition("||")
        prompt = chat_ids(q.strip(), prefix.strip())
        shown = prefix.strip()
    else:
        prompt = chat_ids(line)
        shown = ""
    t0 = time.time()
    out = generate(prompt, state, a.max_new, greedy=line.startswith("/ask "))
    answer = (shown + " " if shown else "") + tok.decode(out).strip()
    print(f"kh1> {answer}   [{len(out) / max(1e-6, time.time() - t0):.0f} tok/s]")
    _, state = feed(prompt + out, state)  # commit the turn to memory once
    seen += len(prompt) + len(out)
