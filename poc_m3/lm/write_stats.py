"""Diagnostic: how fast does ordinary text erase a stored fact from the archival heads?

Streams real text with one planted fact (training format), then tracks, for every archival head, the exact retention
of the fact's key under the model's own subsequent delta-rule writes:

    r_T = k_f^T  prod_t (I - beta_t k_t k_t^T)  k_f        (archival heads: decay pinned to 1)

computed with the fla kernel itself on v = 0 from initial state k_f e_1^T, i.e. the model's real state dynamics
(k_f = unit key of the fact token the head wrote hardest). r = 1 means untouched, r -> 0 means overwritten.
Also reports archival write statistics on plain text (beta quantiles, fraction above thresholds).

  python write_stats.py --run runs_lm8/lm8_gdnhais_d768 --data ~/data/fwe_2b --tokens 1048576 --n 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402
from model import TinyLM  # noqa: E402
from lm_data import FactFactory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--tokens", type=int, default=1 << 20, help="text after the fact")
ap.add_argument("--prefix", type=int, default=8192, help="text before the fact (warm state)")
ap.add_argument("--n", type=int, default=4, help="independent streams")
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--taus", default="0,0.1,0.3,0.5", help="simulated archival write thresholds for the retention curves")
ap.add_argument("--templates", default="secret")
ap.add_argument("--out", default=None, help="json path (default: <run>/write_stats.json)")
ap.add_argument("--arch_tau", type=float, default=None, help="inference write threshold for the model itself")
ap.add_argument("--arch_tau_hi", type=float, default=None, help="two-tier: the last --hi_frac of each layer's archival heads use this threshold")
ap.add_argument("--hi_frac", type=float, default=0.5)
a = ap.parse_args()
dev = "cuda"

orig = layers.chunk_gated_delta_rule
REC = []


def rec(q, k, v, g=None, beta=None, **kw):
    REC.append((k.detach(), beta.detach()))
    return orig(q, k, v, g=g, beta=beta, **kw)


layers.chunk_gated_delta_rule = rec

from transformers import AutoTokenizer  # noqa: E402
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location=dev, weights_only=False)
cfg = ck["cfg"]
if "name" in cfg:  # pretrained base + archival adapters (KH-1 / Qwen graft): only the adapters call the recorded kernel
    if cfg.get("family") == "qwen3_5":
        from qwen_hais import QwenHais as Graft  # noqa: E402
    else:
        from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
    model = Graft(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
                  tau=cfg["tau"], chunk=cfg["chunk"], freeze_base=True).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    a.chunk = cfg["chunk"]
    tok = AutoTokenizer.from_pretrained(cfg["name"])
else:
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
if a.arch_tau is not None:
    for b in model.blocks:
        A = getattr(b.mixer, "archival_heads", 0)
        if A:
            n_hi = int(round(a.hi_frac * A)) if a.arch_tau_hi is not None else 0
            b.mixer.arch_tau = a.arch_tau if not n_hi else torch.tensor([a.arch_tau] * (A - n_hi) + [a.arch_tau_hi] * n_hi, device=dev)
delta = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, layers.DeltaMemMixer)]
arch = {j: model.blocks[i].mixer.archival_heads for j, i in enumerate(delta) if model.blocks[i].mixer.archival_heads}
from lm_data import token_dtype  # noqa: E402
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
rng = np.random.default_rng(7)
marks = [m for m in (4096, 16384, 65536, 131072, 262144, 524288, 1048576, 4194304) if m <= a.tokens]
TH = (0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.97, 0.99)
TAUS = tuple(float(t) for t in a.taus.split(","))  # retention if archival writes with beta <= tau were dropped


@torch.no_grad()
def fwd(ids, st):
    REC.clear()
    with torch.autocast("cuda", torch.bfloat16):
        _, st = model(ids, st, head=False)
    return st, list(REC)


ret_all = {m: [] for m in marks}
written_all = []
head_text = {j: torch.zeros(A, device=dev) for j, A in arch.items()}  # per head: text tokens actually written
n_text = 0
frac = {t: [] for t in TH}
mean_beta, fact_beta = [], []
for s in range(a.n):
    fact = FactFactory(tok, rng, a.templates).make()["fact"]
    s0 = int(rng.integers(0, len(val) - a.prefix - a.tokens - 1))
    # the fact is the last tokens of the prefix, so it is read in context and ends on a chunk boundary
    nf = len(fact)
    pre = torch.cat([torch.from_numpy(val[s0:s0 + a.prefix - nf].astype(np.int64)), torch.tensor(fact)]).to(dev)[None]
    post = torch.from_numpy(val[s0 + a.prefix:s0 + a.prefix + a.tokens].astype(np.int64)).to(dev)[None]
    st = None
    for c0 in range(0, pre.shape[1], a.chunk):
        st, recs = fwd(pre[:, c0:c0 + a.chunk], st)
    recs = [(k[:, -nf:], beta[:, -nf:]) for k, beta in recs]  # the fact tokens' keys and writes
    # per archival layer: key of the fact token each archival head wrote hardest
    kf, S = {}, {}
    for j, A in arch.items():
        k, beta = recs[j]
        k, beta = k[0, :, :A].float(), beta[0, :, :A].float()  # [t, A, dk], [t, A]
        top = beta.argmax(0)                                      # [A]
        kf[j] = torch.nn.functional.normalize(k[top, torch.arange(A)], dim=-1)  # [A, dk]
        fact_beta.append(beta.max(0).values.cpu())
        S[j] = torch.zeros(len(TAUS), A, k.shape[-1], 64, device=dev)  # one retention state per simulated tau
        S[j][:, :, :, 0] = kf[j]
    written_all.append(torch.cat(fact_beta[-len(arch):]) > 0.5)  # [layers*A] heads that actually stored the fact
    done = 0
    for c0 in range(0, post.shape[1], a.chunk):
        st, recs = fwd(post[:, c0:c0 + a.chunk], st)
        for j, A in arch.items():
            k, beta = recs[j]
            kA, bA = k[:, :, :A].expand(len(TAUS), -1, -1, -1).contiguous(), beta[:, :, :A]
            bT = torch.cat([bA * (bA > t) for t in TAUS]).contiguous()
            z = torch.zeros(*kA.shape[:3], 64, device=dev, dtype=kA.dtype)
            _, S[j] = orig(kA, kA, z, g=torch.zeros_like(bT, dtype=torch.float32), beta=bT,
                           initial_state=S[j], output_final_state=True, use_qk_l2norm_in_kernel=True)
            bf = bA.float()
            head_text[j] += (bf[0] > 0.5).float().sum(0)
            for t in TH:
                frac[t].append((bf > t).float().mean().item())
            mean_beta.append(bf.mean().item())
        done += kA.shape[1]
        n_text += kA.shape[1]
        if done in marks:
            r = torch.stack([(kf[j] * S[j][:, :, :, 0]).sum(-1) for j in arch], 1).cpu()  # [taus, layers, A]
            ret_all[done].append(r)
    print(f"stream {s}: " + "  ".join(f"{m // 1024}K r={ret_all[m][-1][0].median():.3f}/best {ret_all[m][-1][0].max():.3f}"
                                      for m in marks), flush=True)

out = {"run": a.run, "archival_layers": len(arch), "heads_per_layer": list(arch.values()),
       "fact_beta_max_median": round(torch.cat(fact_beta).median().item(), 3),
       "text_beta_mean": round(float(np.mean(mean_beta)), 4),
       "text_frac_beta_gt": {str(t): round(float(np.mean(frac[t])), 5) for t in TH}}
for m in marks:
    R = torch.stack(ret_all[m])  # [n, taus, layers, A]
    W = torch.stack(written_all)  # [n, layers*A]
    for i, tau in enumerate(TAUS):
        r = R[:, i]
        rw = r.flatten(1)[W]  # retention over heads that actually stored the fact
        best = r.flatten(1).max(1)  # best-surviving archival head per stream
        out[f"retention/{m}/tau{tau}"] = {"median_all_heads": round(r.median().item(), 4),
                                          "best_head_median": round(best.values.median().item(), 4),
                                          "frac_heads_gt_0.5": round((r > 0.5).float().mean().item(), 3),
                                          "written_heads_median": round(rw.median().item(), 4) if rw.numel() else None,
                                          "written_heads_frac_gt_0.9": round((rw > 0.9).float().mean().item(), 3) if rw.numel() else None,
                                          "written_heads_per_stream": round(W.float().sum(1).mean().item(), 1),
                                          "best_head_ids": best.indices.tolist()}
# per head: how often it writes ordinary text, how often it stores the fact, how much of the fact it keeps at the last mark
tw = torch.cat([head_text[j] for j in arch]).cpu() / n_text          # [layers*A] text-write rate (model's own threshold)
fs = torch.stack(written_all).float().mean(0)                        # [layers*A] share of streams in which the head stored the fact
rl = torch.stack(ret_all[marks[-1]])[:, 0].flatten(1)                # [n, layers*A] retention at the last mark, first simulated tau
out["per_head"] = {"text_write_rate": [round(x, 6) for x in tw.tolist()], "fact_store_rate": [round(x, 3) for x in fs.tolist()],
                   f"retention_{marks[-1]}_median": [round(x, 4) for x in rl.median(0).values.tolist()]}
out["fact_heads_text_write_rate"] = round(float((tw * fs).sum() / fs.sum().clamp_min(1e-9)), 6)  # text writes seen by fact-storing heads
out["all_heads_text_write_rate"] = round(float(tw.mean()), 6)
for q in (1e-4, 5e-4, 1e-3):
    out[f"fact_heads_quiet_lt_{q:g}"] = int(((fs >= 0.5) & (tw < q)).sum())   # heads that usually store facts and rarely write text
print(json.dumps(out, indent=1))
json.dump(out, open(a.out or os.path.join(a.run, "write_stats.json"), "w"), indent=1)
