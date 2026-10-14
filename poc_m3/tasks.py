"""Synthetic long-context recall task: "fact stream".

A sequence is mostly filler (Zipf-distributed tokens, text-like). K facts are
written at random positions as [FACT k v]. At the end, after a QUERY marker,
Q of the facts are queried as [k v] pairs; the only supervised targets are the
v tokens (predicted at the k position). Keys are distinct within a sequence and
every sequence is freshly sampled, so the answer cannot be memorised in weights.

decoy_rate > 0 sprinkles key/value tokens into the filler WITHOUT the FACT
marker, so salience must be read from context, not from token identity.
"""
from dataclasses import dataclass

import torch

PAD, BOS, FACT, QUERY = 0, 1, 2, 3


@dataclass
class Vocab:
    n_keys: int = 512
    n_vals: int = 512
    n_filler: int = 2048
    off: int = 8

    @property
    def key0(self):
        return self.off

    @property
    def val0(self):
        return self.off + self.n_keys

    @property
    def fill0(self):
        return self.off + self.n_keys + self.n_vals

    @property
    def size(self):
        return self.fill0 + self.n_filler


def _zipf_probs(n, a=1.1, device="cpu"):
    r = torch.arange(1, n + 1, device=device, dtype=torch.float64)
    p = 1.0 / (r + 2.7) ** a
    return (p / p.sum()).float()


_ZIPF = {}


def fact_stream(B, L, K, Q=None, vocab=Vocab(), write_frac=1.0, decoy_rate=0.0, device="cuda", gen=None):
    """Returns ids [B, L], labels [B, L] (-100 except answers), dist [B, Q] (query pos - write pos)."""
    Q = K if Q is None else min(Q, K)
    q_len = 1 + 2 * Q
    full = L - 1 - q_len
    region = min(full, max(int(full * write_frac), 3 * K))  # "early" mode still needs 3 tokens per fact
    slot = region // K
    assert slot >= 3, f"L={L} too short for K={K}"

    key = (device, vocab.n_filler)
    if key not in _ZIPF:
        _ZIPF[key] = _zipf_probs(vocab.n_filler, device=device)
    ids = torch.multinomial(_ZIPF[key], B * L, replacement=True, generator=gen).view(B, L) + vocab.fill0

    if decoy_rate > 0:
        m = torch.rand(B, L, device=device, generator=gen) < decoy_rate
        kv = torch.randint(0, vocab.n_keys + vocab.n_vals, (B, L), device=device, generator=gen) + vocab.key0
        ids = torch.where(m, kv, ids)
    ids[:, 0] = BOS

    keys = torch.rand(B, vocab.n_keys, device=device, generator=gen).argsort(1)[:, :K] + vocab.key0
    vals = torch.randint(0, vocab.n_vals, (B, K), device=device, generator=gen) + vocab.val0
    # stratified write positions: one [FACT k v] per slot, random offset inside the slot
    pos = 1 + torch.arange(K, device=device) * slot + torch.randint(0, slot - 2, (B, K), device=device, generator=gen)
    rows = torch.arange(B, device=device)[:, None]
    ids[rows, pos] = FACT
    ids[rows, pos + 1] = keys
    ids[rows, pos + 2] = vals

    qs = L - 2 * Q
    ids[:, qs - 1] = QUERY
    pick = torch.rand(B, K, device=device, generator=gen).argsort(1)[:, :Q]
    qk, qv = keys.gather(1, pick), vals.gather(1, pick)
    qpos = qs + 2 * torch.arange(Q, device=device)
    ids[:, qpos] = qk
    ids[:, qpos + 1] = qv

    labels = torch.full((B, L), -100, device=device, dtype=torch.long)
    labels[:, qpos] = qv
    dist = qpos[None, :] - (pos.gather(1, pick) + 1)
    return ids, labels, dist
