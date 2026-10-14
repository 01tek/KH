"""Probe: does the text *before* a fact change where (which key) the archival heads store it, and how long it survives?

For each prefix length P: stream text A[0:P] (fixed window A), then the fact, then the SAME continuation text B[0:D]
(fixed window B, identical for every P). Per archival head we record the key of the fact token it wrote hardest and
track its exact retention r = k_f^T prod (I - beta_t k_t k_t^T) k_f under the model's own writes on B, with archival
writes thresholded at --arch_tau (as at inference). Reports retention and key similarity to the P = 0 key.

  python probe_keys.py --run runs_lm8/lm8_gdnhais_d768 --data ~/data/fwe_2b --prefixes 0,10240,31457,52428 --arch_tau 0.9
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402
from model import TinyLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--prefixes", default="0,10240,31457,52428")
ap.add_argument("--D", default="16384,131072,1048576", help="distances (multiples of --chunk) at which to report")
ap.add_argument("--fact", default="The secret code of Velthu is 731958.")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--arch_tau", type=float, default=0.9)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--contiguous", action="store_true", help="continuation = the text right after the prefix (as in demo_recall)")
a = ap.parse_args()

orig = layers.chunk_gated_delta_rule
REC = []


def rec(q, k, v, g=None, beta=None, **kw):
    REC.append((k.detach(), beta.detach()))
    return orig(q, k, v, g=g, beta=beta, **kw)


layers.chunk_gated_delta_rule = rec
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location="cuda", weights_only=False)
model = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], ck["cfg"]).cuda().eval()
model.load_state_dict(ck["model"])
mix = [b.mixer for b in model.blocks if isinstance(b.mixer, layers.DeltaMemMixer)]
arch = {j: m.archival_heads for j, m in enumerate(mix) if m.archival_heads}
for m in mix:
    if m.archival_heads:
        m.arch_tau = a.arch_tau
from lm_data import token_dtype  # noqa: E402
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=token_dtype(a.data), mode="r")
rng = np.random.default_rng(a.seed)
Ds = list(map(int, a.D.split(",")))
sA = int(rng.integers(0, len(val) // 2 - 200_000))
if a.contiguous:  # same window as demo_recall.py --seed: s0 = rng(seed).integers(0, len(val) - tokens - 1)
    sA = int(np.random.default_rng(a.seed).integers(0, len(val) - (1 << 20) - 1))
sB = int(rng.integers(len(val) // 2, len(val) - max(Ds) - 1))
fact = torch.tensor(tok(" " + a.fact, add_special_tokens=False)["input_ids"]).cuda()[None]


@torch.no_grad()
def fwd(ids, st):
    REC.clear()
    with torch.autocast("cuda", torch.bfloat16):
        _, st = model(ids, st, head=False)
    return st, list(REC)


key0 = None
for P in map(int, a.prefixes.split(",")):
    A = torch.from_numpy(val[sA:sA + P].astype(np.int64)).cuda()[None]
    st = None
    for c0 in range(0, P, a.chunk):
        st, _ = fwd(A[:, c0:c0 + a.chunk], st)
    st, recs = fwd(fact, st)
    kf, S, bmax = {}, {}, []
    for j, H in arch.items():
        k, beta = recs[j][0][0, :, :H].float(), recs[j][1][0, :, :H].float()
        top = beta.argmax(0)
        kf[j] = F.normalize(k[top, torch.arange(H)], dim=-1)
        bmax.append(beta.max(0).values)
        S[j] = torch.zeros(1, H, k.shape[-1], 64, device="cuda")
        S[j][0, :, :, 0] = kf[j]
    keys = torch.cat([kf[j] for j in arch])  # [heads, dk]
    bmax = torch.cat(bmax)
    if key0 is None:
        key0 = keys
    cos = (keys * key0).sum(-1)
    b0 = sA + P if a.contiguous else sB
    B = torch.from_numpy(val[b0:b0 + max(Ds)].astype(np.int64)).cuda()[None]
    out, done = [], 0
    for c0 in range(0, B.shape[1], a.chunk):
        st, recs = fwd(B[:, c0:c0 + a.chunk], st)
        for j, H in arch.items():
            k, beta = recs[j]
            kA, bA = k[:, :, :H].contiguous(), beta[:, :, :H].contiguous()  # beta already thresholded by the mixer
            z = torch.zeros(*kA.shape[:3], 64, device="cuda", dtype=kA.dtype)
            _, S[j] = orig(kA, kA, z, g=torch.zeros_like(bA, dtype=torch.float32), beta=bA,
                           initial_state=S[j], output_final_state=True, use_qk_l2norm_in_kernel=True)
        done += a.chunk
        if done in Ds:
            r = torch.cat([(kf[j] * S[j][0, :, :, 0]).sum(-1) for j in arch])
            w = bmax > a.arch_tau  # heads that actually stored the fact
            out.append(f"D={done // 1024}K: written-heads r med {r[w].median():.3f} best {r[w].max():.3f} "
                       f"n>0.9 {(r[w] > 0.9).sum().item()}/{w.sum().item()}")
    print(f"prefix {P:>6}: key cos vs P=0 med {cos.median():.3f} min {cos.min():.3f} | " + " | ".join(out), flush=True)
