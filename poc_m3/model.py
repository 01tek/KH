"""Tiny LM built from a layout string, e.g. "m3,m3,m3,m3" or "m3,hais,m3,m3".

Mixer codes:
  m3   official Mamba-3 SISO
  gdn  gated delta-rule memory, GDN defaults (all heads decay, beta bias 0)
  hais gated delta-rule memory, HAIS variant (archival heads + sparse writes)
"""
import torch
import torch.nn as nn

from layers import AttnMixer, DeltaMemMixer, EpisodicDeltaMixer, Mamba3Mixer, RMSNorm, SwiGLU


def make_mixer(code, d_model, cfg):
    if code == "m3":
        return Mamba3Mixer(d_model, d_state=cfg.get("d_state", 64), headdim=cfg.get("headdim", 64),
                           expand=cfg.get("expand", 2))
    if code == "fgdn":  # official fla GatedDeltaNet layer (reference; not streamable here)
        from fla.layers import GatedDeltaNet

        class _Official(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = GatedDeltaNet(hidden_size=d_model, expand_v=1, head_dim=cfg.get("mem_head_dim", 64),
                                           num_heads=cfg.get("mem_heads", 4), mode="chunk", use_short_conv=True)

            def forward(self, x, state=None):
                if state is not None:
                    raise NotImplementedError
                return self.layer(x)[0], None
        return _Official()
    if code == "attn":
        return AttnMixer(d_model, n_heads=cfg.get("mem_heads", 4), head_dim=cfg.get("mem_head_dim", 64))
    if code == "gdn":
        return DeltaMemMixer(d_model, n_heads=cfg.get("mem_heads", 4), head_dim=cfg.get("mem_head_dim", 64),
                             expand_v=cfg.get("mem_expand_v", 1.0), v_norm=cfg.get("v_norm", False),
                             min_decay=cfg.get("gdn_min_decay", 0.0))  # > 0 caps GDN memory: only archival heads keep long state
    if code == "hais":
        h = cfg.get("mem_heads", 4)
        return DeltaMemMixer(d_model, n_heads=h, head_dim=cfg.get("mem_head_dim", 64),
                             expand_v=cfg.get("mem_expand_v", 1.0),
                             archival_heads=cfg.get("archival_heads", h // 2),
                             beta_bias=cfg.get("beta_bias", -4.0), min_decay=cfg.get("min_decay", 0.0),
                             v_norm=cfg.get("v_norm", False), write_budget=cfg.get("write_budget", 0),
                             write_block=cfg.get("write_block", 64), arch_tau=cfg.get("arch_tau", 0.0))
    if code == "hx":  # HAIS-X: archival delta memory + episodic store, one write gate
        h = cfg.get("mem_heads", 4)
        return EpisodicDeltaMixer(d_model, n_heads=h, head_dim=cfg.get("mem_head_dim", 64),
                                  expand_v=cfg.get("mem_expand_v", 1.0),
                                  archival_heads=cfg.get("archival_heads", h // 2),
                                  beta_bias=cfg.get("beta_bias", -4.0), tau=cfg.get("tau", 0.05),
                                  min_decay=cfg.get("min_decay", 0.0), null_sink=cfg.get("null_sink", False), v_norm=cfg.get("v_norm", False),
                                  store_cap=cfg.get("store_cap", 16384))
    raise ValueError(code)


class Block(nn.Module):
    def __init__(self, code, d_model, cfg):
        super().__init__()
        self.code = code
        self.n1 = RMSNorm(d_model, eps=1e-5)
        self.mixer = make_mixer(code, d_model, cfg)
        self.n2 = RMSNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, cfg.get("mlp_mult", 2))

    def forward(self, x, state=None):
        # the residual stream stays fp32; normed inputs are cast to the autocast dtype once (not once per Linear)
        cast = (lambda h: h.to(torch.get_autocast_dtype("cuda"))) if torch.is_autocast_enabled() else (lambda h: h)
        y, state = self.mixer(cast(self.n1(x)), state)
        x = x + y
        return x + self.mlp(cast(self.n2(x))), state


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, layout, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.layout = layout.split(",")
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(c, d_model, cfg) for c in self.layout])
        self.norm = RMSNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids, states=None, head=True):
        """head=False returns the final normed hidden states (apply self.head yourself
        at the few positions you need -- saves a [B, T, V] tensor on long streams)."""
        x = self.emb(ids)
        states = states or [None] * len(self.blocks)
        new_states = []
        for blk, st in zip(self.blocks, states):
            x, st = blk(x, st)
            new_states.append(st)
        x = self.norm(x)
        return (self.head(x) if head else x), new_states

    def write_rate(self):
        """Mean write strength (beta) of delta-rule mixers in the last forward."""
        betas = [b.mixer.last_beta.mean() for b in self.blocks if hasattr(b.mixer, "last_beta")]
        return torch.stack(betas).mean() if betas else None

    def archival_write_rate(self):
        """Mean write strength of archival heads only (None if the model has none)."""
        rates = [b.mixer.last_beta[..., :b.mixer.archival_heads].mean() for b in self.blocks
                 if getattr(b.mixer, "archival_heads", 0) and hasattr(b.mixer, "last_beta")]
        return torch.stack(rates).mean() if rates else None

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
