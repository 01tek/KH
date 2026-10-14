"""Mixer layers for the tiny-model PoC. Every mixer supports streaming:
forward(x, state=None) -> (y, new_state), so a model can be run over an
arbitrarily long sequence in fixed-size chunks with O(1) memory.

- Mamba3Mixer:   the official Mamba-3 SISO layer (mamba_ssm 2.3.2), with the
                 kernel's Input_States / final-state passing exposed.
- DeltaMemMixer: gated delta-rule memory (Gated DeltaNet style, fla kernel).
                 With default flags it is the GDN baseline; with
                 `archival_heads > 0` and/or `beta_bias < 0` it is the HAIS
                 memory variant (some heads never decay, writes are sparse).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
try:
    from mamba_ssm.modules.mamba3 import Mamba3
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
except ImportError:  # e.g. the Qwen3.5 env (transformers 5): only the delta-rule layers are needed there
    Mamba3, mamba3_siso_combined = nn.Module, None


class Mamba3Mixer(Mamba3):
    """Official Mamba-3 (SISO) with explicit state in/out.

    State is the kernel's own tuple (angle, ssm, k, v); it is passed back
    unchanged, so chunked execution matches a single full-length pass.
    """

    def __init__(self, d_model, d_state=64, headdim=64, expand=2, **kw):
        super().__init__(d_model, d_state=d_state, headdim=headdim, expand=expand, is_mimo=False, **kw)

    def forward(self, u, state=None):
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads, self.d_state * self.num_bc_heads,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")
        _A = torch.clamp(-F.softplus(dd_A.to(torch.float32)), max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = rearrange(_A * DT, "b l n -> b n l")
        DT = rearrange(DT, "b l n -> b n l")
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        B, C = self.B_norm(B), self.C_norm(C)
        out = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D, Z=z, chunk_size=self.chunk_size,
            Input_States=state, return_final_states=True,
        )
        y, new_state = out[0], tuple(out[1:5])
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y.to(x.dtype)), new_state


class RMSNorm(nn.RMSNorm):
    """nn.RMSNorm that casts its fp32 weight to the input dtype, so bf16 inputs hit the fused kernel (mixed dtypes fall
    back to a slow unfused path). Same parameters and state-dict keys as nn.RMSNorm."""

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class ShortConv(nn.Module):
    """Causal depthwise conv (kernel k) + SiLU with a carried state for streaming.

    On CUDA this is fla's Triton causal_conv1d (state [B, D, k]; 3.8x faster fwd+bwd than conv1d on [32, 1024, 768],
    same math: fp32 max diff 1e-6, chunked == full pass exactly). The PyTorch path (state = last k-1 inputs) is kept
    for CPU. Both use the same weight [D, 1, k], so checkpoints are interchangeable.
    """

    def __init__(self, dim, k=4):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(dim, 1, k) / math.sqrt(k))

    def forward(self, x, tail=None):  # x: [B, T, D]
        if x.is_cuda and (tail is None or tail.dim() == 3 and tail.shape[-1] == self.k):
            y, state = causal_conv1d(x, weight=self.weight[:, 0].to(x.dtype), activation="silu",
                                     initial_state=tail, output_final_state=True)
            return y, state.detach()
        b, t, d = x.shape
        if tail is None:
            tail = x.new_zeros(b, self.k - 1, d)
        xx = torch.cat([tail, x], dim=1)
        y = F.conv1d(xx.transpose(1, 2), self.weight.to(x.dtype), groups=d).transpose(1, 2)
        return F.silu(y), xx[:, -(self.k - 1):].detach()


class DeltaMemMixer(nn.Module):
    """Gated delta-rule memory: S_t = a_t S_{t-1} + b_t (v_t - S_{t-1} k_t) k_t^T, read o_t = S_t q_t.

    archival_heads: number of heads whose decay is pinned to 1 (log-decay 0):
        state never fades; it changes only by delta-rule overwrites.
    beta_bias: init of the write-strength bias. Negative => writes start sparse
        (sigmoid(-4) ~ 0.02) and training opens them where useful.
    """

    def __init__(self, d_model, n_heads=4, head_dim=64, expand_v=1.0,
                 archival_heads=0, beta_bias=0.0, conv_k=4, min_decay=0.0, v_norm=False, write_budget=0, write_block=64, arch_tau=0.0):
        super().__init__()
        self.h, self.dk = n_heads, head_dim
        # v_norm: L2-normalise values per head (x learnable scale). With unit keys and beta <= 1 the delta-rule state
        # is then bounded (||S||_F <= sqrt(d_k) * scale), so it cannot drift on unbounded streams.
        self.v_norm = v_norm
        # write_budget > 0: each archival head may write only its top-`write_budget` tokens per 64-token block
        # (hard zero elsewhere). Distractor leakage then cannot accumulate with stream length: erasure scales with
        # the number of writes, not the number of tokens.
        self.write_budget, self.write_block = write_budget, write_block
        # arch_tau > 0: archival writes with beta <= arch_tau are exactly zero (forward), straight-through gradient
        # (backward), so sub-threshold distractor leakage cannot accumulate over unbounded streams.
        self.arch_tau = arch_tau
        if v_norm:
            self.v_scale = nn.Parameter(torch.ones(n_heads))
        self.dv = int(head_dim * expand_v)
        self.archival_heads = archival_heads
        # min_decay > 0: non-archival heads must forget at least this fast (log-decay <= -min_decay per token),
        # so only archival heads can carry facts over long distances
        self.min_decay = min_decay
        self.q_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.g_proj = nn.Linear(d_model, self.h * self.dv, bias=False)  # output gate
        self.a_proj = nn.Linear(d_model, self.h, bias=False)  # decay input
        self.b_proj = nn.Linear(d_model, self.h, bias=True)   # write strength
        self.o_proj = nn.Linear(self.h * self.dv, d_model, bias=False)
        self.q_conv = ShortConv(self.h * self.dk, conv_k)
        self.k_conv = ShortConv(self.h * self.dk, conv_k)
        self.v_conv = ShortConv(self.h * self.dv, conv_k)
        self.o_norm = RMSNorm(self.dv, eps=1e-5)
        # GDN-style decay parameterisation: log a_t = -exp(A_log) * softplus(a_proj(x) + dt_bias)
        A = torch.empty(self.h).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        nn.init.constant_(self.b_proj.bias, beta_bias)

    def _budget(self, beta):
        A, m = self.archival_heads, self.write_budget
        if A and (torch.is_tensor(self.arch_tau) or self.arch_tau):  # float, or one threshold per archival head
            ba = beta[..., :A]
            hard = ba * (ba > self.arch_tau)
            beta = torch.cat([ba + (hard - ba).detach(), beta[..., A:]], -1)
        if not (A and m):
            return beta
        B, T, H = beta.shape
        blk = self.write_block
        pad = (-T) % blk
        a = F.pad(beta[..., :A], (0, 0, 0, pad), value=-1.0).view(B, -1, blk, A)
        thr = a.topk(min(m, blk), dim=2).values[:, :, -1:, :]          # m-th largest beta per (block, head)
        keep = (a >= thr).view(B, -1, A)[:, :T]
        return torch.cat([beta[..., :A] * keep, beta[..., A:]], -1)

    def _in_proj(self, x):
        """q, k, v, output-gate, decay and write projections as ONE GEMM over the concatenated weights (same parameters
        and state-dict keys as the six separate Linears; fewer launches and casts)."""
        W = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.g_proj.weight,
                       self.a_proj.weight, self.b_proj.weight])
        q, k, v, gt, a, b = F.linear(x, W).split([self.h * self.dk, self.h * self.dk, self.h * self.dv, self.h * self.dv,
                                                   self.h, self.h], -1)
        return q, k, v, gt, a, b + self.b_proj.bias

    def forward(self, x, state=None):
        b, t, _ = x.shape
        S, tq, tk, tv = state if state is not None else (None, None, None, None)
        q, k, v, gt, a_in, b_in = self._in_proj(x)
        q, tq = self.q_conv(q, tq)
        k, tk = self.k_conv(k, tk)
        v, tv = self.v_conv(v, tv)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.h)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.h)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.h)
        if self.v_norm:
            v = F.normalize(v, dim=-1) * self.v_scale.to(v.dtype)[:, None]
        g = -self.A_log.float().exp() * F.softplus(a_in.float() + self.dt_bias)
        if self.min_decay:
            g = torch.clamp(g, max=-self.min_decay)
        if self.archival_heads:
            g = torch.cat([torch.zeros_like(g[..., :self.archival_heads]), g[..., self.archival_heads:]], -1)
        beta = self._budget(torch.sigmoid(b_in.float()))
        self.last_blogit = b_in  # [B, T, H] pre-sigmoid write logits, for the text-write BCE penalty
        self.last_g = g  # [B, T, H] log-decay per step, for diagnostics
        o, S = chunk_gated_delta_rule(
            q, k, v.to(q.dtype), g=g, beta=beta.to(q.dtype), initial_state=S,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        o = self.o_norm(o) * F.silu(rearrange(gt, "b t (h d) -> b t h d", h=self.h))
        if getattr(self, "ablate_heads", None):
            o = o.clone()
            o[:, :, sorted(self.ablate_heads)] = 0
        y = self.o_proj(rearrange(o, "b t h d -> b t (h d)").to(x.dtype))
        self.last_beta = beta  # [B, T, H], for write-sparsity stats / penalty
        return y, (S, tq, tk, tv)


class AttnMixer(nn.Module):
    """Positive control: causal softmax attention (short conv on q/k/v, no RoPE).
    Not streamable (state must be None) -- use only with eval lengths <= --chunk."""

    def __init__(self, d_model, n_heads=4, head_dim=64, conv_k=4):
        super().__init__()
        self.h, self.dk = n_heads, head_dim
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.conv = ShortConv(3 * n_heads * head_dim, conv_k)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, state=None):
        if state is not None:
            raise NotImplementedError("AttnMixer has no streaming state")
        qkv, _ = self.conv(self.qkv(x))
        q, k, v = rearrange(qkv, "b t (s h d) -> s b h t d", s=3, h=self.h)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(rearrange(o, "b h t d -> b t (h d)")), None


class EpisodicDeltaMixer(DeltaMemMixer):
    """HAIS-X: delta-rule working memory + an episodic store, both driven by ONE write gate beta.

    Episodic read = causal softmax attention over past tokens with an additive log(beta) bias, so
    tokens the gate doesn't write are (softly) invisible. Training uses the exact soft form. Streaming
    keeps only entries with beta > tau in an append-only per-(batch, head) store, so memory grows
    with what is written, not with tokens. Per-token read cost is O(store size).
    """

    def __init__(self, *a, tau=0.05, store_cap=1 << 16, null_sink=False, **kw):
        super().__init__(*a, **kw)
        self.tau, self.store_cap = tau, store_cap
        self.e_norm = RMSNorm(self.dv, eps=1e-5)
        # null_sink: an always-present slot with a fixed logit (0) and zero value. Without it the softmax is
        # invariant to a uniform shift of log(beta), so the gate can collapse to "tiny everywhere" for free and
        # the thresholded store ends up empty. With it, an entry only matters if cos*scale + log(beta) beats 0.
        self.null_sink = null_sink
        # per-head log temperature for the cosine-similarity read, init sqrt(d)
        self.e_logscale = nn.Parameter(torch.full((self.h,), math.log(self.dk ** 0.5)))

    def _qkvb(self, x, tails):
        tq, tk, tv = tails
        q, tq = self.q_conv(self.q_proj(x), tq)
        k, tk = self.k_conv(self.k_proj(x), tk)
        v, tv = self.v_conv(self.v_proj(x), tv)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.h)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.h)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.h)
        if self.v_norm:
            v = F.normalize(v, dim=-1) * self.v_scale.to(v.dtype)[:, None]
        beta = self._budget(torch.sigmoid(self.b_proj(x).float()))
        return q, k, v, beta, (tq, tk, tv)

    def forward(self, x, state=None):
        b, t, _ = x.shape
        S, tq, tk, tv, store = state if state is not None else (None, None, None, None, None)
        q, k, v, beta, (tq, tk, tv) = self._qkvb(x, (tq, tk, tv))
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        if self.min_decay:
            g = torch.clamp(g, max=-self.min_decay)
        if self.archival_heads:
            g = torch.cat([torch.zeros_like(g[..., :self.archival_heads]), g[..., self.archival_heads:]], -1)
        self.last_g = g
        o_d, S = chunk_gated_delta_rule(q, k, v.to(q.dtype), g=g, beta=beta.to(q.dtype), initial_state=S,
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
        # episodic read over (store from earlier chunks) + (this chunk, causal), biased by log(beta)
        qn, kn = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        qh, kh, vh = (z.transpose(1, 2) for z in (qn, kn, v.float()))  # [B, H, T, D]
        logb = torch.log(beta.clamp_min(1e-6)).transpose(1, 2)        # [B, H, T]
        scale = self.e_logscale.float().exp()[None, :, None, None]
        s_loc = (qh @ kh.transpose(-1, -2)) * scale + logb[:, :, None, :]
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        s_loc = s_loc.masked_fill(~causal, float("-inf"))
        parts = []
        if store is not None and store[0].shape[2] > 0:
            Ks, Vs, Ls = store
            parts.append((qh @ Ks.transpose(-1, -2)) * scale + Ls[:, :, None, :])
        parts.append(s_loc)
        if self.null_sink:
            parts.append(s_loc.new_zeros(*s_loc.shape[:-1], 1))
        w = torch.softmax(torch.cat(parts, -1), -1)
        n_st = parts[0].shape[-1] if len(parts) > (2 if self.null_sink else 1) else 0
        o_e = w[..., n_st:n_st + t] @ vh
        if n_st:
            o_e = o_e + w[..., :n_st] @ Vs
        o_e = o_e.transpose(1, 2).to(o_d.dtype)  # [B, T, H, Dv]
        gate = F.silu(rearrange(self.g_proj(x), "b t (h d) -> b t h d", h=self.h))
        o = (self.o_norm(o_d) + self.e_norm(o_e)) * gate
        if getattr(self, "ablate_heads", None):
            o = o.clone()
            o[:, :, sorted(self.ablate_heads)] = 0
        y = self.o_proj(rearrange(o, "b t h d -> b t (h d)").to(x.dtype))
        self.last_beta = beta
        if not self.training:  # streaming eval: append written entries; training keeps no store
            store = self._append(store, kh, vh, logb)
        return y, (S, tq, tk, tv, store)

    @torch.no_grad()
    def _append(self, store, kh, vh, logb):
        """Keep entries with beta > tau. Pads per-(b, h) rows to a common length with log-weight -inf."""
        B, H = logb.shape[:2]
        keep = logb > (math.log(self.tau) if self.tau > 0 else float("-inf"))
        rows_k, rows_v, rows_l = [], [], []
        n_max = 0
        for bi in range(B):
            for hi in range(H):
                idx = keep[bi, hi].nonzero().squeeze(-1)
                if store is not None:
                    Ks, Vs, Ls = store
                    valid = torch.isfinite(Ls[bi, hi])
                    k_ = torch.cat([Ks[bi, hi][valid], kh[bi, hi, idx]])
                    v_ = torch.cat([Vs[bi, hi][valid], vh[bi, hi, idx]])
                    l_ = torch.cat([Ls[bi, hi][valid], logb[bi, hi, idx]])
                else:
                    k_, v_, l_ = kh[bi, hi, idx], vh[bi, hi, idx], logb[bi, hi, idx]
                k_, v_, l_ = k_[-self.store_cap:], v_[-self.store_cap:], l_[-self.store_cap:]
                rows_k.append(k_), rows_v.append(v_), rows_l.append(l_)
                n_max = max(n_max, k_.shape[0])
        D, Dv = kh.shape[-1], vh.shape[-1]
        Ks = kh.new_zeros(B, H, n_max, D)
        Vs = vh.new_zeros(B, H, n_max, Dv)
        Ls = logb.new_full((B, H, n_max), float("-inf"))
        for i, (k_, v_, l_) in enumerate(zip(rows_k, rows_v, rows_l)):
            bi, hi = divmod(i, H)
            n = k_.shape[0]
            Ks[bi, hi, :n], Vs[bi, hi, :n], Ls[bi, hi, :n] = k_, v_, l_
        return (Ks, Vs, Ls)

    def store_size(self, state):
        store = state[4] if state is not None else None
        return 0 if store is None else int(torch.isfinite(store[2]).sum())


class SwiGLU(nn.Module):
    def __init__(self, d_model, mult=2):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * mult * d_model, bias=False)
        self.w3 = nn.Linear(mult * d_model, d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)
