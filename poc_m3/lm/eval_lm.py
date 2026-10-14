"""Phase-3 evaluation, streamed in chunks with carried state (constant memory at any length).

  python eval_lm.py --run runs_lm/hais8 --data /data/fwe --lens 2048,16384,131072,1048576
Tasks:
  passkey   classic passkey retrieval (Mohtashami & Jaggi 2023 prompt), depths 0.1/0.5/0.9
  facts     K facts ("The secret code of <name> is <6 digits>.") spliced into real FineWeb-Edu text in the
            first 10% of the context, queried at the end; a fact counts only if every code token is argmax-correct
  ppl       validation perplexity with warm carried state
Teacher-forced exact match on answer tokens == greedy decoding success for the answer span.
"""
import argparse
import math
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa: E402
from lm_data import FactFactory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--lens", default="2048,16384,131072,1048576")
ap.add_argument("--n", type=int, default=20, help="samples per cell")
ap.add_argument("--K", default="1,8")
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--warm", type=int, default=0, help="stream this many val tokens first (training never starts from an empty state)")
ap.add_argument("--tag", default="")
ap.add_argument("--tasks", default="passkey,facts")
ap.add_argument("--templates", default="secret", choices=["secret", "multi", "xl", "heldout", "comp", "comp_heldout", "mix"], help="fact phrasings for the facts task")
ap.add_argument("--no_ppl", action="store_true")
ap.add_argument("--no_think", action="store_true", help="chat queries in non-thinking mode (Qwen3.5-4B template)")
ap.add_argument("--natural_names", type=float, default=0.0, help="share of eval facts about real-looking names")
ap.add_argument("--chat_frac", type=float, default=0.0, help="share of eval queries asked as chat turns")
ap.add_argument("--batch", type=int, default=1, help="facts task: stream this many samples at once (same samples, faster)")
ap.add_argument("--arch_tau", type=float, default=None, help="override: zero archival writes with beta <= tau at inference")
ap.add_argument("--arch_tau_hi", type=float, default=None,
                help="mixed thresholds: the last --hi_frac of each layer's archival heads use this (higher) threshold")
ap.add_argument("--hi_frac", type=float, default=0.5)
ap.add_argument("--eval_seed", type=int, default=1234, help="sampler seed for the eval streams (1234 in all reported tables)")
ap.add_argument("--ppl_T", type=int, default=2048, help="perplexity: tokens per forward call")
ap.add_argument("--ppl_reset", action="store_true",
                help="perplexity control: adapters on, but the memory state is cleared before every forward call (use --ppl_T = chunk)")
ap.add_argument("--no_memory", action="store_true", help="control: zero the archival adapters' ReZero gates (plain chunked base model)")
ap.add_argument("--gdn_half_life", type=float, default=None, help="inference override: cap memory half-life (tokens) of non-archival delta layers")
a = ap.parse_args()
dev = "cuda"

from transformers import AutoTokenizer  # noqa: E402
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location=dev, weights_only=False)
args, cfg = ck["args"], ck["cfg"]
if "name" in cfg:  # pretrained base + archival adapters: base from the hub, adapters from the checkpoint
    if cfg.get("family") == "gemma4":
        from gemma_hais import GemmaHais as Graft  # noqa: E402
    elif cfg.get("family") == "qwen3_5":
        from qwen_hais import QwenHais as Graft  # noqa: E402
    else:
        from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
    model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
                  tau=cfg["tau"], chunk=cfg["chunk"], freeze_base=True).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    a.chunk = cfg["chunk"]  # LFM2 layers see one chunk at a time; eval lengths are multiples of it
    tok = AutoTokenizer.from_pretrained(cfg["name"])
else:
    model = TinyLM(50304, args["d_model"], args["layout"], cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
if a.no_memory:
    for ad in model.adapters.values():
        ad.alpha.data.zero_()
if a.gdn_half_life:
    for b in model.blocks:
        if b.code == "gdn":
            b.mixer.min_decay = max(b.mixer.min_decay, math.log(2) / a.gdn_half_life)
if a.arch_tau is not None:
    for b in model.blocks:
        if getattr(b.mixer, "archival_heads", 0):
            A = b.mixer.archival_heads
            if a.arch_tau_hi is None:
                b.mixer.arch_tau = a.arch_tau
            else:  # per-head thresholds: a few precise heads keep facts long, the rest keep recall high
                n_hi = int(round(a.hi_frac * A))
                b.mixer.arch_tau = torch.tensor([a.arch_tau] * (A - n_hi) + [a.arch_tau_hi] * n_hi, device=dev)
from lm_data import token_dtype  # noqa: E402
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
rng = np.random.default_rng(a.eval_seed)  # 1234 = the streams every reported number uses; change it for fresh streams


@torch.no_grad()
def score(seq, answers):
    """seq: list[int]; answers: list of (start, length) of answer tokens in seq. Returns per-answer exact match."""
    ids = torch.tensor(seq, device=dev)[None]
    tgt_pos = {s + j - 1 for s, n in answers for j in range(n)}
    preds, st = {}, warm_state()
    for c0 in range(0, ids.shape[1], a.chunk):
        with torch.autocast("cuda", torch.bfloat16):
            h, st = model(ids[:, c0:c0 + a.chunk], st, head=False)
        want = [p for p in tgt_pos if c0 <= p < c0 + h.shape[1]]
        if want:
            idx = torch.tensor([p - c0 for p in want], device=dev)
            out = model.head(h[0, idx]).argmax(-1).tolist()
            preds.update(dict(zip(want, out)))
    return [all(preds[s + j - 1] == seq[s + j] for j in range(n)) for s, n in answers]


@torch.no_grad()
def score_batch(seqs, answers_list):
    """Equal-length sequences streamed together (no warm state). Returns per-sample lists of per-answer exact match."""
    ids = torch.tensor(seqs, device=dev)
    tgt = [sorted({s + j - 1 for s, n in ans for j in range(n)}) for ans in answers_list]
    preds, st = [dict() for _ in seqs], None
    for c0 in range(0, ids.shape[1], a.chunk):
        with torch.autocast("cuda", torch.bfloat16):
            h, st = model(ids[:, c0:c0 + a.chunk], st, head=False)
        for b in range(len(seqs)):
            want = [p for p in tgt[b] if c0 <= p < c0 + h.shape[1]]
            if want:
                out = model.head(h[b, torch.tensor([p - c0 for p in want], device=dev)]).argmax(-1).tolist()
                preds[b].update(zip(want, out))
    return [[all(preds[b][s + j - 1] == seqs[b][s + j] for j in range(n)) for s, n in answers_list[b]]
            for b in range(len(seqs))]


@torch.no_grad()
def warm_state():
    if not a.warm:
        return None
    s0 = int(rng.integers(len(val) // 2, len(val) - a.warm - 1))  # second half of val, away from fact haystacks
    x = torch.from_numpy(val[s0:s0 + a.warm].astype(np.int64)).to(dev)[None]
    st = None
    for c0 in range(0, x.shape[1], a.chunk):
        with torch.autocast("cuda", torch.bfloat16):
            _, st = model(x[:, c0:c0 + a.chunk], st, head=False)
    return st


FILL = enc(" The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.")
INTRO = enc("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. "
            "I will quiz you about the important information there.")


def passkey(L, depth):
    key = str(rng.integers(10000, 100000))
    needle = enc(f" The pass key is {key}. Remember it. {key} is the pass key.")
    q, ans = enc(" What is the pass key? The pass key is"), enc(f" {key}")
    body_len = L - len(INTRO) - len(needle) - len(q) - len(ans)
    body = (FILL * (body_len // len(FILL) + 1))[:body_len]
    cut = int(depth * body_len)
    seq = INTRO + body[:cut] + needle + body[cut:] + q + ans
    return score(seq, [(len(seq) - len(ans), len(ans))])[0]


def facts(L, K, build_only=False):
    ff = FactFactory(tok, rng, a.templates, a.chat_frac, a.natural_names, a.no_think)
    fs = [ff.make() for _ in range(K)]
    tail = sum(len(f["query"]) for f in fs)
    start = int(rng.integers(0, len(val) - L - 1))
    hay = val[start:start + L - tail - sum(len(f["fact"]) for f in fs)].astype(np.int64).tolist()
    region = max(1, int(0.1 * len(hay)))
    cuts = sorted(rng.integers(0, region, size=K).tolist())
    seq, prev = [], 0
    for f, c in zip(fs, cuts):
        seq += hay[prev:c] + f["fact"]
        prev = c
    seq += hay[prev:]
    answers = []
    for f in fs:
        answers.append((len(seq) + f["ans_start"], f["ans_len"]))
        seq += f["query"]
    if build_only:
        return seq, answers
    return sum(score(seq, answers)) / K


def facts_batched(L, K, n):
    """Same samples (same RNG draws, in order) as n calls of facts(L, K), scored a.batch at a time."""
    out = []
    for i in range(0, n, a.batch):
        built = [facts(L, K, build_only=True) for _ in range(min(a.batch, n - i))]
        out += [sum(r) / K for r in score_batch([b[0] for b in built], [b[1] for b in built])]
    return out


@torch.no_grad()
def ppl(n_tokens=2_000_000, T=None):
    T = T or a.ppl_T
    st, losses = None, []
    start = 0
    for s in range(0, n_tokens, T):
        x = torch.from_numpy(val[start + s:start + s + T + 1].astype(np.int64)).to(dev)[None]
        with torch.autocast("cuda", torch.bfloat16):
            lg, st = model(x[:, :-1], None if a.ppl_reset else st)
        if s >= n_tokens // 2:  # warm state for the first half
            losses.append(F.cross_entropy(lg.float()[0], x[0, 1:]).item())
    return float(np.exp(np.mean(losses)))


res = {"ppl_val": round(ppl(), 3)} if not (a.warm or a.no_ppl) else {}
print(json.dumps(res), flush=True)
for L in map(int, a.lens.split(",")):
    for d in ((0.1, 0.5, 0.9) if "passkey" in a.tasks else ()):
        r = np.mean([passkey(L, d) for _ in range(a.n)])
        res[f"passkey/L{L}/d{d}"] = round(float(r), 3)
        print(f"passkey L={L} depth={d}: {r:.3f}", flush=True)
    for K in (map(int, a.K.split(",")) if "facts" in a.tasks else ()):
        n_s = max(1, a.n // max(1, K // 4))
        r = np.mean(facts_batched(L, K, n_s) if a.batch > 1 and not a.warm else [facts(L, K) for _ in range(n_s)])
        res[f"facts/L{L}/K{K}"] = round(float(r), 3)
        print(f"facts L={L} K={K}: {r:.3f}", flush=True)
json.dump(res, open(os.path.join(a.run, f"eval_lm{a.tag}.json"), "w"), indent=1)
