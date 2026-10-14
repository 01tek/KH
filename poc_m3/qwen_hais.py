"""Qwen3.5 + HAIS: constant-memory, distance-invariant fact recall grafted onto a pretrained Qwen3.5 model.

Same design as lfm2_hais.py (KH-1):
* Qwen3.5's own layers see one chunk of C tokens at a time (no sink token: Qwen3.5-0.8B chunked LM loss 3.165 without,
  3.273 with <|endoftext|>; identical to the native model on the same chunks). Its 6 full-attention
  layers have no KV cache across chunks, and its 18 Gated-DeltaNet layers start every chunk from an empty state. So
  Qwen's own memory is bounded and cannot carry facts across chunks: the routing principle holds by construction.
* Archival delta-rule adapters (decay 1, write threshold, v_norm, ReZero gate) are the only cross-chunk memory.
Interface matches Lfm2Hais (forward(ids, states, head), blocks, write_rate, head, adapter_state_dict, cfg).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from lfm2_hais import ArchivalAdapter


def _fast_conv():
    """Qwen3.5's GDN short conv falls back to F.conv1d without the causal_conv1d package; use fla's Triton kernel."""
    try:
        from fla.modules.convolution import causal_conv1d
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    except ImportError:
        return

    def conv_fn(hidden_states, weight, bias=None, activation=None, **kw):  # [B, D, T] in and out, like causal_conv1d_fn
        y, _ = causal_conv1d(hidden_states.transpose(1, 2), weight=weight.to(hidden_states.dtype),
                             bias=None if bias is None else bias.to(hidden_states.dtype), activation=activation)
        return y.transpose(1, 2)
    mq.causal_conv1d_fn = conv_fn


_fast_conv()


class QwenHais(nn.Module):
    def __init__(self, name="Qwen/Qwen3.5-4B", insert_after=(11, 17, 23), n_heads=32, head_dim=64, tau=0.9,
                 chunk=2048, freeze_base=True, sink=False, dtype=torch.bfloat16, quant=None):
        super().__init__()
        kw = {}
        if quant in ("nf4", "int8"):  # inference on small GPUs: quantised frozen base (bitsandbytes), adapters stay as trained
            from transformers import BitsAndBytesConfig
            q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                    bnb_4bit_use_double_quant=True) if quant == "nf4" else BitsAndBytesConfig(load_in_8bit=True))
            kw = dict(quantization_config=q, device_map={"": 0})
        self.base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
        self.quant = quant
        self.chunk, self.insert_after, self.sink = chunk, tuple(insert_after), sink
        d = self.base.config.hidden_size
        self.adapters = nn.ModuleDict({str(i): ArchivalAdapter(d, n_heads, head_dim, tau) for i in insert_after})
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        tok_cfg = self.base.config
        # sink token for position 0 of every chunk: BOS if the model has one, else <|endoftext|>-style eos/pad
        self.sink_id = next(t for t in (getattr(tok_cfg, "bos_token_id", None), getattr(tok_cfg, "eos_token_id", None),
                                        getattr(tok_cfg, "pad_token_id", None)) if isinstance(t, int))
        self.cfg = {"name": name, "insert_after": list(insert_after), "n_heads": n_heads, "head_dim": head_dim,
                    "tau": tau, "chunk": chunk, "freeze_base": freeze_base, "sink": sink, "family": "qwen3_5"}

    @property
    def blocks(self):
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
        n, s = T // C, int(self.sink)
        m = self.base.model
        x = ids.view(B * n, C)
        if s:
            x = torch.cat([torch.full((B * n, 1), self.sink_id, dtype=ids.dtype, device=ids.device), x], 1)
        h = m.embed_tokens(x)
        L = C + s
        pos = torch.arange(L, device=ids.device)[None].expand(B * n, L)
        pos_emb = m.rotary_emb(h, pos[None])  # text-only M-RoPE: the same positions on all 3 axes
        states = list(states) if states else [None] * len(self.adapters)
        new_states, j = [], 0
        for i, layer in enumerate(m.layers):
            h = layer(h, position_embeddings=pos_emb, attention_mask=None, position_ids=pos, past_key_values=None)
            if str(i) in self.adapters:
                hs, st = self.adapters[str(i)](h[:, s:].reshape(B, T, -1), states[j])
                new_states.append(st)
                j += 1
                h = torch.cat([h[:, :s], hs.reshape(B * n, C, -1)], 1) if s else hs.reshape(B * n, C, -1)
        h = m.norm(h[:, s:]).reshape(B, T, -1)
        return (self.base.lm_head(h) if head else h), new_states

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("adapters.")}
