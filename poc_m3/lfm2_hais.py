"""LFM2 + HAIS: a pretrained Liquid LFM2 model given constant-memory, distance-invariant recall.

* LFM2's own layers (short convs + attention) only ever see one chunk of C tokens, prefixed with BOS: attention and
  convs are block-local and positions restart in every chunk, so their memory is bounded and RoPE stays in range.
* Archival delta-rule layers (decay pinned to 1, write threshold tau, v_norm) are inserted after some LFM2 layers as
  residual adapters. They run over the whole sequence and carry their state across chunks and calls: they are the
  model's only long-range memory. A ReZero gate (alpha = 0) makes the model start as exactly LFM2.

forward(ids [B, T], states) with T <= C (one chunk) or T % C == 0 (T // C chunks, archival layers span all of them,
so a query in chunk 2 trains the write in chunk 1). Returns (logits or hidden, new_states).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from layers import DeltaMemMixer, RMSNorm


class ArchivalAdapter(nn.Module):
    def __init__(self, d, n_heads=16, head_dim=64, tau=0.5):
        super().__init__()
        self.norm = RMSNorm(d, eps=1e-5)
        self.mixer = DeltaMemMixer(d, n_heads=n_heads, head_dim=head_dim, archival_heads=n_heads, beta_bias=-2.0,
                                   v_norm=True, arch_tau=tau)
        # ReZero gate: exact identity at init (alpha = 0) while the branch itself keeps a normal init, so gradients
        # still flow (alpha first, then the branch). A zero output projection instead blocks every gradient into the
        # memory (grad norm 0.0), and a tiny one let the branch swamp LFM2's residual stream (lm loss 3 -> 10).
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h, state=None):
        y, state = self.mixer(self.norm(h), state)
        return h + (self.alpha * y).to(h.dtype), state


class Lfm2Hais(nn.Module):
    def __init__(self, name="LiquidAI/LFM2-350M", insert_after=(7, 11, 15), n_heads=16, head_dim=64, tau=0.5,
                 chunk=2048, freeze_base=True, dtype=torch.bfloat16):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        if hasattr(self.base, "set_experts_implementation"):  # MoE bases (LFM2.5-8B-A1B, 24B-A2B): grouped-GEMM experts
            try:
                self.base.set_experts_implementation("grouped_mm")
            except Exception:  # dense bases have no experts
                pass
        self.chunk, self.insert_after = chunk, tuple(insert_after)
        d = self.base.config.hidden_size
        self.adapters = nn.ModuleDict({str(i): ArchivalAdapter(d, n_heads, head_dim, tau) for i in insert_after})
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.bos_id = self.base.config.bos_token_id if self.base.config.bos_token_id is not None else 1
        self.cfg = {"name": name, "insert_after": list(insert_after), "n_heads": n_heads, "head_dim": head_dim,
                    "tau": tau, "chunk": chunk, "freeze_base": freeze_base, "bos_per_chunk": True}

    @property
    def blocks(self):  # same diagnostic interface as TinyLM (write_rate, text-write penalty, inference tau)
        return [type("B", (), {"mixer": a.mixer})() for a in self.adapters.values()]

    def write_rate(self):
        betas = [a.mixer.last_beta.mean() for a in self.adapters.values() if hasattr(a.mixer, "last_beta")]
        return torch.stack(betas).mean() if betas else None

    def archival_write_rate(self):
        return self.write_rate()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def head(self, h):
        return self.base.lm_head(h)

    def forward(self, ids, states=None, head=True):
        B, T = ids.shape
        C = T if T <= self.chunk else self.chunk
        assert T % C == 0, f"sequence length {T} must be <= {self.chunk} or a multiple of it"
        n = T // C
        m = self.base.model
        # every chunk is processed as [BOS] + chunk, like LFM2's training sequences: without the BOS at position 0
        # (its attention sink) LFM2 degrades badly, e.g. a chat reply "This is a way to to to to ...".
        bos = torch.full((B * n, 1), self.bos_id, dtype=ids.dtype, device=ids.device)
        h = m.embed_tokens(torch.cat([bos, ids.view(B * n, C)], 1))  # [B*n, C+1, D]
        pos = torch.arange(C + 1, device=ids.device)
        pos_ids = pos[None].expand(B * n, C + 1)
        rot = m.pos_emb if hasattr(m, "pos_emb") else m.rotary_emb  # LFM2-MoE / LFM2 on transformers 4.x: pos_emb; LFM2 on 5.x: rotary_emb
        pos_emb = rot(h, pos_ids)
        states = list(states) if states else [None] * len(self.adapters)  # one state per adapter, in layer order
        new_states, j = [], 0
        for i, layer in enumerate(m.layers):
            h = layer(h, attention_mask=None, position_ids=pos_ids, past_key_values=None, cache_position=pos,
                      position_embeddings=pos_emb)
            if str(i) in self.adapters:  # the archival memory sees only real tokens (never the per-chunk BOS)
                hs, st = self.adapters[str(i)](h[:, 1:].reshape(B, T, -1), states[j])
                new_states.append(st)
                j += 1
                h = torch.cat([h[:, :1], hs.reshape(B * n, C, -1)], 1)
        h = m.embedding_norm(h[:, 1:]).reshape(B, T, -1)
        return (self.base.lm_head(h) if head else h), new_states

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("adapters.")}
