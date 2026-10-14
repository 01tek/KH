"""Gemma 4 + HAIS: the KH-1 archival memory grafted onto a frozen Gemma 4 text model (E2B / E4B).

Same design as lfm2_hais.py / qwen_hais.py:
* Gemma's own layers see one chunk at a time, as [<bos>] + chunk, with no KV cache across chunks, so its sliding-window
  and global attention cannot carry anything from one chunk to the next.
* Archival delta-rule adapters (decay 1, write threshold, v_norm, ReZero gate) after chosen decoder layers are the
  only cross-chunk memory.
Gemma 4 has per-layer embeddings, local/global attention with separate rotary bases, and KV sharing across its last
layers, so instead of re-implementing its layer loop the adapters run as forward hooks inside Gemma's own forward.
Interface matches Lfm2Hais (forward(ids, states, head), blocks, write_rate, head, adapter_state_dict, cfg).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from lfm2_hais import ArchivalAdapter


class GemmaHais(nn.Module):
    def __init__(self, name="google/gemma-4-E2B-it", insert_after=(16, 25, 34), n_heads=24, head_dim=64, tau=0.9,
                 chunk=2048, freeze_base=True, dtype=torch.bfloat16, quant=None):
        super().__init__()
        kw = {}
        if quant in ("nf4", "int8"):  # inference on small GPUs: quantised frozen base, adapters stay as trained
            from transformers import BitsAndBytesConfig
            q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                    bnb_4bit_use_double_quant=True) if quant == "nf4" else BitsAndBytesConfig(load_in_8bit=True))
            kw = dict(quantization_config=q, device_map={"": 0})
        self.base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
        self.quant = quant
        m = self.base.model
        self.text = getattr(m, "language_model", m)  # text decoder (a multimodal checkpoint nests it)
        tcfg = getattr(self.base.config, "text_config", self.base.config)
        self.softcap = getattr(tcfg, "final_logit_softcapping", None)
        self.chunk, self.insert_after = chunk, tuple(insert_after)
        d = tcfg.hidden_size
        self.adapters = nn.ModuleDict({str(i): ArchivalAdapter(d, n_heads, head_dim, tau) for i in insert_after})
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.bos_id = tcfg.bos_token_id if isinstance(getattr(tcfg, "bos_token_id", None), int) else 2
        self._ctx = None
        for i in insert_after:
            self.text.layers[i].register_forward_hook(self._hook(str(i)))
        self.cfg = {"name": name, "insert_after": list(insert_after), "n_heads": n_heads, "head_dim": head_dim,
                    "tau": tau, "chunk": chunk, "freeze_base": freeze_base, "family": "gemma4"}

    def _hook(self, key):
        def fn(module, inputs, out):
            c = self._ctx
            if c is None:  # plain base-model call (e.g. the native-loss check): leave it untouched
                return out
            h = out[0] if isinstance(out, tuple) else out
            hs, st = self.adapters[key](h[:, 1:].reshape(c["B"], c["T"], -1), c["states"][c["j"]])
            c["new"].append(st)
            c["j"] += 1
            h = torch.cat([h[:, :1], hs.reshape(c["B"] * c["n"], c["C"], -1)], 1)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn

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
        logits = self.base.lm_head(h)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def forward(self, ids, states=None, head=True):
        B, T = ids.shape
        C = T if T <= self.chunk else self.chunk
        assert T % C == 0, f"sequence length {T} must be <= {self.chunk} or a multiple of it"
        n = T // C
        x = torch.cat([torch.full((B * n, 1), self.bos_id, dtype=ids.dtype, device=ids.device), ids.view(B * n, C)], 1)
        self._ctx = {"B": B, "T": T, "n": n, "C": C, "j": 0, "new": [],
                     "states": list(states) if states else [None] * len(self.adapters)}
        try:
            h = self.text(input_ids=x, use_cache=False).last_hidden_state  # final norm applied
            new_states = self._ctx["new"]
        finally:
            self._ctx = None
        h = h[:, 1:].reshape(B, T, -1)
        return (self.head(h) if head else h), new_states

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("adapters.")}
