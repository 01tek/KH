# HAIS-M3 → Tiny-Model PoC Plan (Mamba-3 era)

**Date:** 2026-10-11 · **Hardware:** local RTX 3070 Ti (8 GB, sm_86) · **Supersedes:** the "ground truth" section of `MAMBA3_HAIS_MASTERPLAN.md`
**Code:** `poc_m3/` · **Env:** `m3poc_env/` (torch 2.9.1+cu128, triton 3.5.1, mamba-ssm 2.3.2.post1, fla 0.5.2)

---

## 0. TL;DR

1. **Audit result.** Most HAIS "recall" results in this repo did not measure recall. The S1/v1.0 curriculum, the H8C4 4-arm pilot, the gate probe and the GPT-2 Phase-1 runs all scored *unshifted* labels: the target at each scored position was the input token at that same position (an echo task). Remove the needle and v1.0 still scores 99.6%. Its true next-token recall is **0/800**. The checkpoints are genuine (sha256 38/38 OK, numbers reproduce). Neither side of the agent argument caught the label bug.
2. **The only valid recall task** in the repo is the Mamba-1 130m *delta-needle* line (`train_delta_needle.py`, shifted labels, real query). Its control (bare backbone vs HAIS vs HAIS with writes disabled) is queued; results go in §8.2.
3. **Mamba-3 today.**
   - Official code: `mamba-ssm` 2.3.2, with Triton SISO kernels.
   - Official checkpoints: 187M–1.5B, released 2026-07-28.
   - It is *weak at retrieval beyond its 2K training length*; its own authors say so.
   - The same lab's **Raven** (sparse routed slot memory, Jul 2026) is the nearest competitor and does not compare against Mamba-3.
4. **Verified today.** Mamba-3 SISO **trains on the 3070 Ti**: a 4.1M-param model runs at 384K tok/s and 2.0 GB peak. The new `poc_m3/` harness streams any length at O(1) memory, with chunked output equal to a full pass (fp32 Δ ≈ 1e-3).
5. **The bet** has two pre-registered claims:
   - **(A)** A fixed-size HAIS memory gives *unbounded distance, bounded #facts*.
   - **(B)** HAIS-X, a salient-write episodic store, gives *unbounded #facts* with roughly constant per-token compute. This is the only honest route to "unlimited context".
6. **First gate** (G1, §5): about 3.5 h on the cloud box (`poc_m3/cloud_bootstrap.sh`), or ~1 day locally.
7. **Pilot-1 result (§8.4, 1 seed).**
   - Every fixed-state model (GDN, Mamba-3, hybrids) recalls perfectly to ~16× its training length and is dead by 256×.
   - HAIS-v3 as designed collapses *earlier*. Its archival heads learned dense writes (β = 1.0 on filler), so distractors erase facts.
   - The archival idea therefore needs a pressure for sparse writes. Long-chain state passing is the next test (queued).

---

## 1. Audit: what is real

| Line of work | Claimed | What it actually measured | Verdict |
|---|---|---|---|
| **v1.0 curriculum** (`s1_curriculum.py`, tag `v1.0-curriculum-complete`) | 100% needle recall to L=2048; "93% collapse" when S is ablated | Labels unshifted (`:72/:96/:108`), so each target is the current input token. Needle removed: 0.9962, unchanged. Shifted next-token recall: 0/800. "93%" is the *ablated accuracy* (collapse = 7%). The "ablation" removes the whole adapter, not S. | Genuine run; **not a recall result** |
| "Backbone bypassed" (`README_HAIS_NOTE.md`) | Adapter sat on input embeddings | It reads `hidden_states[0]` = **block-1 output** and skips 23/24 blocks + `norm_f`. Right in effect, wrong in detail, and it misses the real flaw. | Partly right |
| v2 retrain on `hidden_states[-1]` (Jul 19) | — | Still the echo task (unshifted, padding scored). 0.17–0.26 at L=2048, killed mid-stage. | Not a fair test |
| **H8C4 pilot** (`s1_scaling_v2.py`) | A=0.92, B=0.994, C=0.995, D=0.995 | Same echo task. Frozen block 0 already decodes the current token (Arm A). Every arm hits the ceiling. Arms trained on *different* data slices. L=256 never ran. D's "matched" param count includes frozen params. | **Zero power** |
| **Gate-liveness v2** (`gate_probe.py`) | Δlogprob = 0 | Log-probs come from the **backbone's** logits, never the arm's, so Δ=0 by construction. The probe arm ≠ the trained arm (EMA instead of S·k read, `lb/ld` not loaded). Wrong input layer. Crashed (`NameError lp_128`). | **Bug** |
| GPT-2 Phase 1/1b + paper (`paper/hais_m3_paper.tex`) | 78%→11% etc. | Also unshifted echo labels. The paper says GPT-2 where S1 used Mamba-130m, and describes the ablation as S=0 (it wasn't). | Not recall; paper needs rewrite |
| **Mamba-1 130m delta-needle** (`train_delta_needle.py`, Jul 5–11) | Recall 1.00 to 3200 tok; k=2 0.80, k=3 0.67, k=5 wall | Proper task: shifted labels, real query "The secret code for X is". | **Valid task.** Control in §8 |
| 1.4B delta-needle | Masterplan: A=1.00, k=2 0.80, k=3 0.67 | Only local log (`train_needle_1_4b.log`): recall 0.00, loss 5–7. The masterplan's k-numbers are identical to the 130m report. | **Unverified** |
| "Fake 84%" Mamba-3 180M (H100) | — | Weight memorization, self-identified and retracted. | Correctly retracted |

**Who was right?** You were right that the checkpoints and logs are real. The critics were right that v1.0 never tested backbone+adapter. But **everyone, humans and agents, kept believing the S-Matrix stored needles.** The actual bug was the unshifted labels, and nobody traced it until this audit. Evidence and fix patches are in `poc_m3/audit/`: `replicate_v1.py`, `gate_probe_fix.patch`, `s1_scaling_v2_fix.patch`, plus `check_*.py`.

**Design flaw carried into every HAIS variant.** `hais_m3_delta.py:138` reads with the same key it just wrote (`q = k`, same position). The read therefore always contains the current token's own value. That is why echo tasks looked like memory. New designs must use a separate query projection.

---

## 2. Mamba-3 as of today (what changed since the July masterplan)

- **Paper.** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, *Mamba-3*, arXiv **2603.15569** (2026-03-16), **ICLR 2026 Oral**.
  - Exponential-trapezoidal discretization.
  - Complex state via data-dependent RoPE (gives parity/mod-arith state tracking).
  - MIMO rank 4.
  - **No short conv**, replaced by trapezoid + B/C biases and BCNorm.
- **Recall is the known weakness.**
  - 1.5B, trained at 2K. NIAH at 4K (2× train): SISO S1/S2/S3 = 88.2 / 50.6 / 34.2. GDN = 99.8 / 49.8 / 34.2.
  - SWDE/FDA trail Transformers by ~20–35 pts.
  - The authors expect pure Mamba-3 to need hybrids for retrieval.
  - No MQAR in the paper. They note MQAR is *harder* without the conv.
- **Code.**
  - `mamba-ssm` 2.3.2.post1 (2026-05-09) ships `Mamba3`. SISO uses Triton; MIMO uses TileLang; `step()` decode uses CuTe.
  - `main` is newer: heavy-tail A activation (2026-06-01), `MambaLMHeadModel` support, Blackwell fix.
  - On **sm_86**: SISO train/prefill works (**verified today**). MIMO backward needs >99 KB shared memory, so it won't run here. `step()`/`generate()` are buggy and Hopper-tested only, so **evaluate with chunked forward passes, not `generate()`**. That is what `poc_m3` does.
- **Checkpoints exist now** (contradicts the masterplan, which predates them).
  - `state-spaces/mamba3-{siso,mimo}-{187m,443m,893m,1.5b}`, released 2026-07-28.
  - 2K context, 100B FineWeb-Edu tokens.
  - **Llama-3.1 tokenizer (gated on HF).**
  - In the 187M model, ~98.5M of the params are embeddings.
- **Competition from the same lab.** **Raven** (arXiv 2607.25357, 2026-07-28), sparse routed slot memory: 424M, train 2K, passkey 100 / 99.8 / 91.4 at 2K / 8K / 32K vs GDN 22.1 at 32K. No Mamba-3 comparison.

---

## 3. What "unlimited context" can honestly mean

- **Theorem-level facts.**
  - A fixed-size state cannot copy, or exactly recall, an unbounded *amount* of information: Jelassi et al. 2024; Arora et al. 2024, "Ω(N) bits for MQAR"; Wen et al. 2024.
  - Delta-rule capacity is about d_k associations per head. Measured: ~0.55 bits per state element (RWKV-7 MQAR).
- **What a fixed state can do:** recall a bounded number of salient facts across **unbounded distance**, if nothing overwrites or decays them. That is **Claim A**.
- **Truly unlimited recall needs memory that grows with *content*,** not tokens: store only salient writes and read them sparsely. Per-token compute stays roughly constant, and memory grows with #facts. That is **Claim B (HAIS-X)**. Precedents exist at ≤370M: RAMba/HSA (passkey to 64M), DRT (to 16M). The contribution must therefore be a *specific mechanism with ablations*, not the general idea.

---

## 4. The PoC design

### 4.1 Testbed: from-scratch tiny models, synthetic "fact stream" (`poc_m3/tasks.py`)
- Zipf-distributed filler, with K facts written as `[FACT k v]` at random positions.
- A final `QUERY` block asks Q of the keys. The **only** supervised tokens are the values, predicted at the key position.
- Keys are distinct and sampled fresh per sequence, so weight memorization is impossible. Eval seeds are disjoint from training.
- `decoy_rate` sprinkles key/value tokens without the FACT marker. Salience must then come from *context*, not token identity.
- `write_frac=0.05` ("early" mode) plants all facts in the first 5%. That maximizes distance.
- Why from scratch: no frozen-backbone confound, full control, and ~5 minutes per run, so many seeds and LRs are affordable.

### 4.2 Arms (≈4M params each, d=256, 4 blocks of mixer + SwiGLU; `poc_m3/model.py`)
| Arm | Layout | Role |
|---|---|---|
| `m3` | Mamba-3 ×4 | backbone of interest |
| `gdn` | Gated DeltaNet ×4 | strongest fixed-state recall baseline |
| `m3+gdn` | m3, **gdn**, m3, m3 | hybrid control (isolates "just add a delta layer") |
| `m3+hais` | m3, **hais**, m3, m3 | the HAIS memory |

**HAIS memory v3** (`DeltaMemMixer`, `poc_m3/layers.py`) differs from GDN in exactly two places, each of which can be ablated:
1. **Archival heads** (`archival_heads`, default h/2): decay pinned to 1. The state never fades; only delta-rule overwrites change it.
2. **Sparse writes** (`beta_bias=-4`): the write gate starts closed (σ(−4)≈0.02) and training opens it where useful. An optional L1 on write rate is available.

Also fixed relative to the old HAIS: a **separate q/k projection with a short conv**, so no same-position self-read.
Training option: **state passing** (Buitrago & Gu 2025): start batches from the previous batch's final state, so the model sees "lived-in" states.

### 4.3 HAIS-X (Claim B), built only after G1
- **Write.** When the HAIS write gate β_t > τ, append (k̂_t, v_t) to an episodic store. It lives on GPU and spills to host RAM.
- **Read.** q_t → top-m inner-product search over the store → softmax-weighted values → gated into the output next to S·q_t.
- **Training at 1–2K.** The store is just "attention restricted to the write set", so the values are differentiable. Scale values by β so the gate gets gradient.
- **The novelty bet.** *One* learned write gate decides both what enters working memory (S) and what gets archived, so storage grows with information, not tokens. It has to beat HSA-lite, Raven and SWA hybrids at matched compute to be more than a demo.

### 4.4 Metrics
- Recall accuracy at answer tokens vs **context length** (1K → 1M → 16M), **distance**, and **#facts K**.
- Write rate (fraction of tokens with β > τ). Throughput (tok/s) vs L. Peak memory.
- Params **and state bytes**.
- 3 seeds minimum, with 95% CIs.

---

## 5. Phases and gates

| Phase | What | Time / compute | Gate |
|---|---|---|---|
| **0. Clean slate** | ✅ env, ✅ Mamba-3 smoke on sm_86, ✅ streaming harness, ✅ audit, ✅ Pilot-0, recipe calibration and Pilot-1 (§8). ☐ Delta-needle control (queued: `poc_m3/queue_verify.sh`). ☐ Strike the false claims in the paper/masterplan. | done + ½ day | — |
| **1. Claim A** (synthetic) | `poc_m3/cloud_grid.sh`: 5 arms {`hais,hais`, `gdn,gdn`, `m3,hais`, `m3,gdn`, `m3,m3`} × LR {3e-3, 1e-3} × 3 seeds × state_passing {0.5, 0} = 60 runs. Add 15 mechanism ablations on `hais,hais` (archival_heads {0, 4}, beta_bias 0, decoy 0.1) and 3 attention controls: 78 runs at 20K steps (~20 min each). With 8 in parallel on the 96 GB RTX PRO 6000 that is ≈ 3.5 h; locally ≈ 26 h. **Primary comparison: `hais,hais` vs `gdn,gdn`.** Pilot-1 showed pure GDN far ahead of the Mamba-3 hybrids at the training length (§8.4), so the HAIS mechanism must be tested on the strong baseline. Then a capacity sweep: K ∈ {1…256} × mem heads/head_dim gives recall vs state bytes. | 1–3 days, ~25 GPU-h | **G1** |
| **2. Claim B** (HAIS-X) | Implement the store. Eval K up to 4096 facts, L up to 16M (host-RAM store). Baselines: m3+SWA hybrid, Raven layer, HSA-lite chunk retrieval, oracle. | 1–2 weeks, ~30 GPU-h | **G2** |
| **3. Language** | 30–60M LM on FineWeb-Edu (1–2B tok) plus 10% synthetic recall. Open 32K tokenizer, avoiding the gated Llama one. Evals: RULER S/MK/MV-NIAH at 1–256× train length, BABILong QA1–3 (fine-tune protocol), passkey to 1M, PPL vs position. Same-token baselines: Mamba-3, GDN, GDN-2/KDA, Mamba-3 + 1 NoPE attention. | 2–4 weeks local (1–3 days per run) or ~$200–600 rented H100 | **G3** |
| **4. Bolt-on** (optional) | The original HAIS vision on the frozen official `mamba3-siso-187m`. Secondary, because retrieval circuits form during pretraining (MiniMax M2 lesson). | 1 week | — |
| **5. Write-up** | Pre-registered predictions vs outcomes; every number linked to a log and commit; release the harness. | 1 week | — |

**G1 passes only if all three hold:**
- (i) The best HAIS arm (`hais,hais` or `m3,hais`) reaches ≥ 90% recall at **L = 1M, early mode, K ≤ 8**, on 3/3 seeds.
- (ii) It beats the best of {`gdn,gdn`, `m3,gdn`, `m3,m3`}, **each also trained with state passing and its best LR**, by ≥ 20 pts at L ≥ 256K.
- (iii) `archival_heads=0` loses the advantage, so the mechanism is isolated.

If (ii) fails because GDN + state passing is also flat, Claim A is a rediscovery. Report it and go straight to Claim B.

**G2:**
- ≥ 95% recall at K = 1024 facts, L = 1M.
- ≥ 90% at L = 16M.
- Tokens/s flat in L (±10%).
- Store size ∝ facts (write rate ≤ 1% of tokens).
- Beats HSA-lite and Raven at matched compute.

**G3:**
- Beats every same-token baseline on MK-NIAH and BABILong QA2/QA3 beyond 16× train length.
- Costs ≤ 5% perplexity.

### Pre-registered predictions for Phase 1 (written 2026-10-11; P1 revised the same day after recipe calibration, *before* any Pilot-1 or grid result was seen)
- **P1.** Without state passing, `m3` and `gdn` reach ≥ 90% at the training length (L=256, K ≤ 8) but fall below 50% at L ≥ 64K in early mode.
- **P2.** With state passing, `gdn` improves at long L more than `m3` does. Delta-rule erasure handles reused states better than pure decay.
- **P3.** `m3+hais` stays ≥ 90% to L = 1M for K ≤ 8, and its write rate drops below 5% of tokens.
- **P4.** Setting `archival_heads=0` in `m3+hais` reproduces the `m3+gdn` long-L curve within ±10 pts.
- **P5.** `m3+hais` recall vs K falls off at roughly archival_heads × d_k / 4 (≈ 32 facts at 2 heads × 64).

---

## 6. Experiment hygiene: the anti-echo checklist (mandatory)

1. **Assert in the data generator** that every scored target is *not* visible at the scored input position: `labels[t] ≠ ids[t]`, predicted from the prefix ≤ t. Add a **needle-removed control** that must drop to chance.
2. **Preflight, both directions:**
   - The no-memory baseline must score **< 20%** before any arm comparison counts.
   - A **positive control** (softmax attention) must solve the task (≥ 95%) with the same data and recipe. Otherwise a failing arm says nothing.
3. **Ablations** are computed from the **arm's own logits**, paired on identical examples, and reported as Δ ± 95% CI.
4. **Identical data per arm** (per-arm seeded generator). Eval seeds are disjoint from training.
5. **≥ 3 seeds and an LR sweep per arm.** Recurrent models solve MQAR only in narrow LR windows.
6. **Match params *and* state bytes**; report both.
7. **Every number in a doc cites a log line and a commit.** Agents may not write results sections without the log.
8. **Two-agent rule:** one agent proposes a claim, another must reproduce it from raw logs before it enters a doc.

---

## 7. Risks and kill criteria

- **Kernels.** Mamba-3 SISO fwd/bwd is verified on sm_86. First compile takes about 15 min; it is cached in `~/.triton`. `generate()` is broken, so stick to chunked forward.
- **Claim A may be a rediscovery** (GDN + state passing already flat). That is acceptable: report it and pivot to B.
- **Claim B novelty** vs HSA/RAMba, Raven and Memorizing Transformers. If HAIS-X doesn't beat them at matched compute, it is a nice demo, not a paper. Decide at G2.
- **Shared GPU.** The local TTS server (`01videos/tools/local_tts`) competes for VRAM and RAM. **It was OOM-killed today at 15:36** during these experiments. Restart it with `npm run tts:local` (from `01videos`) and don't run heavy jobs concurrently.
- **Kernel correctness traps found today. Check these on every new GPU before trusting a number:**
  1. **Blackwell (sm_100/103/120):** PyPI `mamba-ssm` 2.3.2.post1 silently corrupts the Mamba-3 SISO *forward* (Triton `num_stages` 2/3). Fixed on `main` @ e9594ce (PR #997). We overlay that file.
  2. **Hopper (H100/H200):** fla's gated chunk backward is wrong with Triton 3.4–3.7.0 (fla #640); fla raises an error. Fix: `triton==3.7.1`.
  3. **Verification:** official `test_mamba3_siso.py` passes (8 passed, 1 skipped) and `poc_m3/test_gdn_grad.py` (fla chunk kernel vs fp32 PyTorch reference, rel. err ≤ 0.5%) on the H200 with Triton 3.7.1.
- **Compute in use (2026-10-11):** non-preemptible **H200** `<node>` with `~/hais_poc/env` (torch 2.9.1, Triton 3.7.1, fla 0.5.2, mamba-ssm 2.3.2.post1 + #997 overlay). CUDA MPS runs 14 jobs in parallel at ~15 s per 100 steps each. The RTX PRO 6000 box `<node>` is preemptible and has been down since ~16:35.
- **H100 box `<node>` is unreachable.** The `sync_h100.sh` cron has failed every 30 min (50/50 logs). Disable it, or fix it if that box still holds data.

---

## 8. Results so far (2026-10-11)

### 8.1 Smoke test (`poc_m3/smoke.json`), 3070 Ti, d=256, 4 blocks, B=16, L=1024
| Arm | Params | Stream Δ (bf16) | Train tok/s | Peak GB |
|---|---|---|---|---|
| m3 | 4.11M | 0.017 | 383,909 | 1.97 |
| gdn | 3.70M | 0.085 (fp32: 0.0018) | 288,783 | 2.10 |
| m3+gdn | 4.01M | 0.017 | 350,010 | 2.01 |
| m3+hais | 4.01M | 0.016 | 350,646 | 2.01 |

### 8.2 Delta-needle control (`poc_m3/verify_delta_needle.json`; Mamba-1 130m; repo's own eval protocol; n = 30 per cell; identical prompts across conditions)
| Condition | 100 / 400 / 800 / 1600 tok | 3200 tok | needle depth 0 / 0.5 / 0.95 (800 tok) |
|---|---|---|---|
| bare frozen backbone (no HAIS) | 1.00 / 1.00 / 1.00 / 1.00 | 0.67 | 1.00 / 1.00 / 1.00 |
| HAIS checkpoint (`hais_m3_delta_needle_best.pt`) | 1.00 / 1.00 / 1.00 / 1.00 | **0.97** | 1.00 / 1.00 / 1.00 |
| same checkpoint, β forced to 0 (S-Matrix never written) | 1.00 / 1.00 / 1.00 / 1.00 | 0.67 | 1.00 / 1.00 / 1.00 |

**Verdict on the "first checkpoints".**
- The O1 report's "recall 1.00 at 100–3200 tokens" is mostly the **backbone**: bare Mamba-130m already scores 1.00 to 1600 tokens on this task.
- HAIS adds real recall only beyond the backbone's native range: **+30 pts at 3200 tokens** (29/30 vs 20/30).
- Disabling memory writes removes exactly that gain, so it comes from the S-Matrix content.
- This is a genuine but narrow effect: one seed, one length beyond the backbone's range.

### 8.3 Pilot-0 and recipe calibration (`poc_m3/runs/pilot0/`, `runs/recipe/`)
**Pilot-0:** 4 blocks, train L=1024, K ≤ 8, batch 32, LR 1e-3, 2,000 steps. It was a **calibration failure**, not a result:
- `m3` stayed at chance at every length (train acc ≤ 0.8%).
- `gdn` reached only 9–12% at K=1.
- Cause: far too little signal (≈144 supervised tokens per step) for recurrent models.

I stopped the run and calibrated on the GDN control:

| Run (L=256, K ≤ 32, bs 128) | Steps | Eval L=256 K=1 / 8 / 32 | Note |
|---|---|---|---|
| `attn,attn` (conv + softmax attention, positive control) | 1,500 | **1.00 / 1.00 / 1.00** (also 1.00 at L=1024) | solved by step 300, so the task and harness are sound |
| `gdn ×4`, my DeltaMem, LR 3e-3 | 1,500 | 0.01 / 0.00 / 0.01 | plateau |
| `gdn ×4`, **official `fla.layers.GatedDeltaNet`**, LR 3e-3 | 1,500 | 0.03 / 0.03 / 0.01 | same as mine, so not an implementation bug |
| `gdn ×2`, LR 3e-3 | 3,000 | **0.43 / 0.26 / 0.04** | "clicks" at ~2.4K steps |
| `gdn ×2`, LR 1e-2 | 3,000 | 0.05 / 0.00 / 0.01 | outside the LR window |

**Takeaways:**
- Recurrent models need about 10× more steps than attention on this task.
- They work only in a narrow LR window, as in Okpekpe & Orvieto 2025.
- **Phase 1 therefore uses** 2-layer models, train L=256, K ≤ 32, batch 128, LR {1e-3, 3e-3}, 12–20K steps. Every grid includes an `attn` positive control.

### 8.4 Pilot-1 (`poc_m3/run_pilot1.sh`; 1 seed, LR 3e-3, 12K steps, state passing 0.5, train L=256)
Recall in "early" mode (facts in the first 5% of the stream, so distance ≈ L). From `python report.py runs/pilot1`:

| Arm | Final train acc | K=1: L 256 / 4K / 64K / 1M | K=8: 256 / 4K / 64K / 1M | K=32: 256 / 4K / 64K / 1M |
|---|---|---|---|---|
| `gdn,gdn` | 1.00 | 1.00 / 1.00 / 0.04 / 0.00 | 1.00 / 1.00 / 0.10 / 0.00 | 1.00 / 1.00 / 0.08 / 0.02 |
| `m3,m3` | 0.17 | 1.00 / 1.00 / 0.02 / 0.00 | 0.39 / 0.37 / 0.01 / 0.00 | 0.13 / 0.11 / 0.00 / 0.00 |
| `m3,gdn` | 0.18 | 1.00 / 0.30 / 0.01 / 0.00 | 0.27 / 0.00 / 0.00 / 0.00 | 0.12 / 0.00 / 0.01 / 0.00 |
| `m3,hais` | 0.30 | 1.00 / 0.02 / 0.00 / 0.00 | 0.49 / 0.01 / 0.00 / 0.00 | 0.21 / 0.01 / 0.01 / 0.00 |
| `hais,hais` | 1.00 | 1.00 / 1.00 / 0.01 / 0.00 | 1.00 / 1.00 / 0.02 / 0.00 | 0.93 / 0.99 / 0.01 / 0.00 |

`hais,hais` learns as well as GDN (≈100% by step 1.2K, faster than GDN) but hits **the same cliff** between 4K and 64K.
- Mechanism (`runs/pilot1/haishais/key_analysis.json`): no archival head learned "write facts, skip filler".
  - Layer-0 archival heads write *only* filler (β 1.0 on filler, 0.0 at facts), acting as a running summary.
  - The layer-1 archival head that does store facts writes filler too (β 0.998), so filler keeps erasing the stored facts.
- Fact-selective gating appears only in decaying heads, as it did in GDN.

**Readout (1 seed, so directional only):**
1. **Every fixed-state model works up to ~16× its training length and is dead by 256×.** Pure GDN is perfect at all K up to 4K tokens, then 4–10% at 64K. Pure Mamba-3 holds 1 fact to 4K, then 2%. This is the gap HAIS has to close.
2. **Mamba-3 in the stack hurts this task.** Both Mamba-3 hybrids learn far less than pure GDN at the training length. Mamba-3 (no short conv) is known to be weak at MQAR-style recall.
3. **HAIS-v3 (`m3,hais`) is a negative result as designed.** It is the best hybrid at L=256 but collapses *first* (0.02 at 4K).
   - Mechanism (`analyze_keys.py`): **both archival heads write every filler token at β = 1.000.** Centered key overlap with fact keys is 0.33–0.57.
   - With no decay, each distractor write erases part of the stored facts and nothing ever cleans up, so error grows with stream length.
   - The β = −4 init only delayed opening the gate. At a 256-token training length, dense writes cost nothing, so there is no pressure to be sparse.
   - The GDN baseline, by contrast, learned specialized heads: one head writes only at facts (β 0.26 vs 0.00 on filler). Selective gating can emerge; decay is what kills GDN past 16×.
4. **Retention model (open item; do not cite yet).**
   - First-order picture: a stored fact loses ≈ λ + β·c² per distractor token (λ = decay, c = key overlap), so Δ\* ∝ 1/(λ̄ + β̄_fill·c̄²).
   - `analyze_keys.py` now reports these terms per head (`lambda_fill`, `c2_fill_fact`, `efold_tokens`).
   - As measured, it does **not** predict the cliff. Predicted e-folds of 1–100 tokens vs. observed perfect recall at 4K.
   - Two reasons: (a) raw keys share a dominant direction (c² 0.7–0.99), so the fact-identity components must be isolated; (b) "fact = v-position write" may not be where each head binds.
   - Next: causal **head ablations** (zero one head's read, re-measure recall vs distance) to find the fact-storing heads, then fit retention on those heads only.
5. **Implication:** archival memory helps only together with sparse writes. The next tests target the missing pressure directly:
   - `hais,hais` (HAIS on the strong baseline).
   - **Long-chain state passing** (p = 0.95 ⇒ ~20 chained batches ≈ 5K-token effective streams while training at 256), for both `hais,hais` and `gdn,gdn`. Queued: `poc_m3/queue_sp95.sh`.
   - An explicit **write-sparsity penalty on archival heads only** (`--write_l1 0.05 --l1_scope archival`), combined with long-chain state passing. Queued: `poc_m3/queue_l1.sh`.
   - If neither produces fact-selective archival writes, move to HAIS-X (§4.3). Its episodic store keeps facts verbatim, so retention does not depend on the state surviving distractor writes.

---

### 8.5 Pilot 2 (H200, 3 seeds; train L=256, K ≤ 32, long-chain state passing p = 0.95, 20K steps)
Early mode ("facts in first 5%", so distance ≈ L). Per seed, from `python report.py runs/h200`:

| Run | 64K: K=1 / 8 / 32 | 1M: K=1 / 8 / 32 |
|---|---|---|
| `gdn,gdn` sp0.95 (3 seeds) | ≤ 0.13 all seeds | 0.00 |
| `hais,hais` sp0.95 (3 seeds) | 0.05–0.45 | ≤ 0.02 |
| `hais,hais` sp0.95 + archival-only write L1 0.05, **seed 2** | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 0.98** (n = 16 / 16 / 64; confirmation queued) |
| same, seeds 0 and 1 | 0.07–0.12 | 0.00 |
| `gdn,hx` + write L1 0.05 (2 seeds) | **1.00 / 1.00 / 1.00** (re-eval n = 512: 0.998) | 0.56 / 0.38 / 0.55 and 0.12 / 0.12 / 0.06 |
| `hais,hx` + write L1 0.05 (3 seeds) | 0.00 (dead at 4K) | 0.00 |

**What the diagnostics show** (`analyze_keys.py`, `runs/reeval/`):
1. **Distance-invariant recall with constant memory exists.** The successful seed has a layer-0 archival head with **β = 0.999 at facts and 0.0000 on filler, no decay**. It is a perfectly gated, never-forgetting slot memory. Its other heads decay within a few tokens.
   - In the failed seeds the write penalty **killed the archival heads** (β = 0 everywhere). The model routed facts through "decaying" heads that learned λ ≈ 0 with dense, non-selective writes, which is erasure-limited.
2. **`gdn,hx` succeeds for the same reason, not because of the episodic store.** The store is **empty** (0 entries per head at τ = 0.05 and at 1e-3). A layer-0 GDN head learned λ = 0 with fact-biased small writes (β 0.022 facts vs 0.0055 filler).
3. **HAIS-X v1 has a design flaw: gate collapse.** The episodic read is a softmax, so shifting log β uniformly changes nothing. With the write penalty, the model drove β → ~0 *everywhere* at no cost and used the episodic path as plain attention.
   - At eval the absolute threshold τ discards everything, so recall dies at the first chunk boundary (`hais,hx`: 1.00 at 256 tokens, 0.00 at 4K).
   - Fix: a **null sink** (fixed-logit empty slot) so the gate's absolute scale matters (`--null_sink`).

**Working hypothesis (pre-registered for Pilot 3).** Recall far beyond the training length needs a memory head that (a) does not decay and (b) writes only salient tokens. Pressure for (b) comes from a write penalty. Making (a)+(b) the *only* way to solve training, by forcing non-archival heads to forget (`--min_decay 0.0217`, half-life 32 tokens), should make success reliable across seeds.

### 8.6 Pilot 3 (running: `runs/p3`, 5 configs × 3 seeds)
| Config | Purpose |
|---|---|
| `hais,hais` + archival L1 0.05 + min_decay | the fix for seed-dependence |
| `hais,hais` + archival L1 0.02 + min_decay | penalty strength |
| `hais,hais` all-archival (4 heads) + archival L1 0.05 | no non-archival route at all |
| `gdn,hx` + null sink + L1 0.05 | HAIS-X v2: does the store fill with facts now? |
| `hais,hx` + null sink + L1 0.05 + min_decay | HAIS-X v2 with the forced route |

**Predictions (P3).**
- (i) The `min_decay` configs succeed at 1M in ≥ 2/3 seeds, with an archival head at β_fact > 0.5 and β_fill < 0.01.
- (ii) With the null sink, the HAIS-X store holds O(K) entries per head (≈ facts, not tokens) and recall holds at 1M.

**Pilot-3 results** (first 7 runs; the 5 OOM retries are pending). Early mode, recall at K = 1 / 8 / 32:

| Config | Seed | 64K | **1M** |
|---|---|---|---|
| archival L1 0.05 + min_decay | s0 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** |
| | s1 | 1.00 / 1.00 / 0.99 | **1.00 / 1.00 / 1.00** |
| archival L1 0.02 + min_decay | s0 | 1.00 / 1.00 / 1.00 | 0.81 / 0.69 / 0.64 |
| | s1 | 1.00 / 1.00 / 0.99 | **1.00 / 1.00 / 0.98** |
| | s2 | 1.00 / 1.00 / 0.97 | **1.00 / 1.00 / 0.97** |
| all-archival + L1 0.05 | s1 | 1.00 / 1.00 / 1.00 | **0.94 / 1.00 / 0.97** |
| | s2 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** |
| HAIS-X v2 `hais,hx` sink + md | s0 | dead at 4K | store empty |

**Pilot-3 final (all 15 runs).** Early mode, mean ± 95% CI over 3 seeds (`python report.py runs/p3 --group`):

| Config | 64K: K = 1 / 8 / 32 | **1M: K = 1 / 8 / 32** | Seeds ≥ 0.88 at 1M |
|---|---|---|---|
| **HAIS recipe** (archival L1 0.05 + min_decay) | 1.00 / 1.00 / 0.99 | **1.00 / 1.00 / 0.96 ± 0.08** | **3/3** |
| **all-archival** + archival L1 0.05 | 1.00 / 1.00 / 1.00 | **0.98 ± 0.04 / 1.00 / 0.99 ± 0.02** | **3/3** |
| archival L1 0.02 + min_decay | 1.00 / 1.00 / 0.99 | 0.94 / 0.90 / 0.86 (± ≤ 0.22) | 2/3 |
| `gdn,hx` + null sink + L1 0.05 | 1.00 / 1.00 / 1.00 | 0.48 / 0.60 / 0.44 (± ≤ 0.60) | 1/3 |
| `hais,hx` + null sink + min_decay | dead at 4K | 0.00 | 0/3 |

- **Prediction (i): confirmed** for the two main configs: **6/6 seeds** recall 1–32 facts at 1M tokens and all 6 are perfect at 64K, vs 1/3 seeds in Pilot 2.
- **Mechanism** (`p3_arch_l105_md_s0`): one head (layer 0, archival h1) writes facts at **β = 0.997** and filler at **β = 0.0002**, with λ = 0. Every other head is a short-lived scratchpad (forced decay) or unused.
- **Prediction (ii): falsified so far.** The HAIS-X store is still empty (learned β stays below τ = 0.05 even with the sink).
- **Headline candidate:** a 2-layer, ~3.7M-param recurrent model with **constant memory**, trained only on 256-token sequences, recalls 32 facts at 1M tokens (4096× its training length).
- Confirmation evals running (`run_confirm3.sh`): n = 128 at 1M/4M, a 16M-token stream, capacity K = 64/128/256.

### 8.7 Phase 3 pipeline (built 2026-10-11, `poc_m3/lm/`): the "working unlimited-context LM"
- **Data** (`prep_data.py`): FineWeb-Edu `sample-10BT`, GPT-2 BPE (MIT, ungated), uint16 token files. Val = the first documents, train = the rest, so they are disjoint.
- **Training stream** (`lm_data.py`): B contiguous *lanes*. Every step continues each lane and **the recurrent state is carried across all steps** (truncated BPTT), so the training context is unbounded.
  - Synthetic facts ("The secret code of Zelith is 482913.") are spliced into the real text.
  - They are queried 1–200 segments later (1K–200K tokens), with extra loss weight on the code tokens.
- **Model** (`train_lm.py`): `TinyLM` (vocab 50304), e.g. d = 512 with 8 blocks mixing Mamba-3 and HAIS layers. It uses the Pilot-3 recipe (archival heads, min_decay, archival-only write L1) and checkpoints/resumes preemption-safely.
- **Eval** (`eval_lm.py`), streamed in constant memory:
  - classic **passkey** at depths 0.1 / 0.5 / 0.9;
  - **K facts in a real-text haystack** (first 10% of context, queried at the end, exact match on every code token);
  - **val perplexity** with warm state.
  - Lengths 2K → 1M.
- **G3 (pre-registered).** The HAIS LM must beat a same-size, same-token GDN / Mamba-3 LM on facts and passkey at ≥ 128K by ≥ 30 pts, with val perplexity within 5%. Report 3 seeds once the single-seed signal is positive.

**LM v1 → v2 (2026-10-11, ~23:30 UTC).**
- v1 used Bernoulli p_fact = p_query = 0.1 per lane-segment, which gave ~3 answer tokens per step. All three LMs had **0% answer accuracy at ~6.5K steps**, even for queries only 3–5 segments after the fact. The signal was ~1000× sparser than in the synthetic task. Partial runs are kept in `lm/runs_lm_v1/`.
- At that point, LM loss was 4.91 (HAIS), 4.70 (GDN) and 5.51 (Mamba-3, at more steps).
- **v2 curriculum:**
  - 25% "drill" lanes: 8 facts plus their 8 queries in every segment.
  - Poisson(1) facts and Poisson(1) queries per segment on the other lanes.
  - Query-age curriculum: max age grows 1 → 200 segments over 7K steps.
  - 5× answer weight.
  - Result: ~260 answer tokens per step. Same data, tokens and seed for all three LMs. Running now (`runs_lm/`).

### 8.8 Pilot 4 (ablations, 3 seeds) and LM v2 results (2026-10-12)
**Pilot 4.** Early mode, recall at 1M tokens, K = 1 / 8 / 32, mean of 3 seeds:

| Config | Recall at 1M | Seeds that work |
|---|---|---|
| recipe without the write penalty | 0.33 / 0.33 / 0.20 | 1/3 |
| recipe without state passing | 0.31 / 0.33 / 0.33 | 1/3 |
| recipe + decoys (0.1) | 0.27 / 0.21 / 0.18 | ~1/3 |
| Mamba-3 at the matched recipe (sp 0.95) | 0.33 / 0.10 / 0.03 | train acc only 0.57 |

- The write penalty and long-chain state passing are **both necessary**: each drops reliability from 3/3 to 1/3.
- Decoys (salience from context rather than token identity) are the weak point.

**LM v2** (500M tokens, 50M params, v2 curriculum). Val ppl, then facts-in-text recall for K = 1 / 8:

| LM | Val ppl | 2K | 16K | 131K | 1M |
|---|---|---|---|---|---|
| HAIS (archival L1 0.05) | **89.4** | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| GDN | 111.8 | **0.80 / 0.88** | **0.60 / 0.89** | 0 / 0 | 0 / 0 |
| Mamba-3 | 1311 (diverged, grad norm 68) | 0 | 0 | 0 | 0 |

- Passkey is 0 everywhere: that prompt format is out of distribution for these models.
- **The synthetic recipe did not transfer as-is.** HAIS got the best perplexity but learned no recall (train answer acc 0.06 vs GDN 0.73). The likely cause is the 0.05 archival penalty starving the memory in the LM setting.
- LM v3 (running, H200 #1) varies the penalty and min_decay, and adds a second GDN seed and Mamba-3 at LR 1e-3.
- Pilot 5 (running, H200 #2): decoys at 40K steps with L1 0.05 / 0.02; capacity scaling (mem_heads 4 / 8 / 16, train L = 1024, K ≤ 128, eval K up to 512); recipe seeds 3–5.

### 8.9 Correction: the recipe is 4/6 seeds, not 3/3. Why, and the fix (2026-10-12 ~03:30 UTC)
- **Replication fails.** Recipe seeds 3–5 (Pilot 5, same config as Pilot 3): s4 is perfect at 1M, but **s3 and s5 collapse by 4K**. Over all 6 seeds the recipe works in **4/6**. The Pilot-3 "3/3" was a lucky draw.
- **Mechanism of the failures** (`analyze_keys.py`): the failing seeds store facts in a *non-archival* head running exactly at the floor λ = min_decay = 0.0217 (half-life 32 tokens), with fact-selective writes (s3 L0 h2: β_fact 0.995, β_fill 0.007).
  - That should forget within the 256-token window, **but the read goes through RMSNorm, so uniform decay is scale-invariant**. Shrinking the whole state loses no information until it underflows.
  - 2^-(4096/32) = 2^-128 hits the fp32 floor, which is exactly where these seeds die (~4K tokens). The "forced forgetting" escape route was never closed.
- **Fix (Pilot 6, running):** min_decay = 0.35, so non-archival state shrinks by e^-90 within the 256-token window and underflows during training. Facts can then only survive in archival heads.
  - 6 seeds on H200 #2; 6 seeds plus 6 decoy seeds on H200 #1.
- **Capacity (Pilot 5, L = 1024, K ≤ 128, 2 seeds; noisy):**
  - mem_heads 16 holds more facts at 4K (K = 128: 0.84, K = 256: 0.48) than mem_heads 4 (0.27 / 0.08).
  - Recall still degrades with distance, likely the same escape.
  - mem_heads 8 failed to train (acc 0.03). Rerun after the fix.
- **LM v3** (penalty variants):
  - All HAIS variants: 0 facts recall at eval, even though train answer acc reached 0.40–0.47 without the 0.05 penalty. Val ppl 94–115.
  - GDN seed 1: train answer acc 0.51 but eval ppl 19,600, so its carried state blows up on long streams. GDN seed 0's 0.80–0.89 recall at 2K–16K did **not** replicate.
  - Mamba-3 at LR 1e-3: ppl 231, no recall.
  - LM v4 (running) applies min_decay 0.35 to the HAIS LM.

### 8.10 Pilot 6: min_decay 0.35 closes the escape (12 seeds)
Early mode, recall K = 1 / 8 / 32. Mean ± 95% CI over 12 seeds (`runs/p6`):

| Context | Recall |
|---|---|
| 256 | 1.00 / 0.96 / 0.88 |
| 64K | 0.98 / 0.93 / 0.86 |
| **1M** | **0.93 / 0.91 / 0.81** |

- **8/12 seeds are perfect at 1M** (≥ 0.98 at every K). s6 reaches 0.81 / 1.00 / 0.56.
- **Recall is distance-invariant in 11/12 seeds.** The three weak seeds (s3, s5, s8) are *undertrained* (final train acc 0.50–0.65), not forgetting: s3 and s8 score the same at 1M as at 256. Only s5 still decays with distance.
- **Decoys (6 seeds):** hold to 64K (0.80 / 0.79 / 0.67) but collapse at 1M (0.15 / 0.10 / 0.07). One seed did not train.
- **LM v4** (HAIS, min_decay 0.35), at step ~8.2K of 15.3K:
  - without the write penalty: train answer acc **0.41**;
  - with penalty 0.01: 0.05 (the penalty still starves it in the LM).
- **Pilot 7 (running):**
  - Undertrained seeds 3/5/8 at 40K steps and at LR 1e-3.
  - Decoys at 40K steps (6 seeds).
  - Capacity scaling with min_decay 0.35 (mem_heads 4 / 8 / 16, 2 seeds).

### 8.11 Pilot 7: the synthetic recipe is reliable; the LM still doesn't transfer (2026-10-12 ~06:30 UTC)
- **Undertrained seeds rescued.** Seeds 3 / 5 / 8 (which failed in Pilot 6) with min_decay 0.35:
  - 40K steps: **1.00 / 1.00 / 1.00 at 1M, 3/3 seeds**.
  - LR 1e-3: 1.00 / 1.00 / 0.99, 3/3.
- **Final recipe:** `hais,hais`, archival-only write L1 0.05, min_decay 0.35, state passing 0.95, **40K steps**. Six fresh seeds (12–17) are running in Pilot 8 for the headline statistic.
- **Decoys at 40K steps (6 seeds):** ≥ 0.94 to 64K, but 0.18 / 0.18 / 0.11 at 1M (early mode).
  - Pilot 8 tests state passing 0.99 and L1 0.1 (3 seeds each, plus the combination).
- **LM v4** (`m3,hais`, min_decay 0.35, 500M tokens):
  - No penalty: train answer acc 0.51, but eval facts ≤ 0.08 at every length; val ppl 142.
  - L1 0.01: 0 recall.
  - Warm-starting the eval state (16K tokens of text first) doesn't fix it: HAIS 0.10–0.14 vs GDN 0.60–0.90 to 16K, and 0 at 131K for both.
- **LM v5 (running):** `gdn,hais` ×4. GDN gives local memory and HAIS gives archival memory, with no Mamba-3, since Mamba-3 hybrids hurt recall in Pilot 1 and GDN is the only LM that learned in-text recall.
- **Capacity (Pilot 7 cap):** relaunched on H200 #1. The first launch failed because that box had an old `train.py` without `--mem_heads`.

### 8.12 Headline result (Pilot 8) and open problems (2026-10-12 ~08:00 UTC)
- **Final recipe**: `hais,hais`, archival-only write L1 0.05, min_decay 0.35, state passing 0.95, LR 3e-3, 40K steps, trained at L = 256.
  - **6/6 fresh seeds (12–17): recall 1.00 / 1.00 / 0.99 ± 0.01 at 1M tokens** for K = 1 / 8 / 32 (early mode).
  - With the 3 rescued seeds from Pilot 7, that is **9/9 seeds** at 40K steps.
- **Decoys at 1M** (3 seeds each):
  - L1 0.1: 0.54 / 0.44 / 0.43.
  - State passing 0.99: 0.52 / 0.42 / 0.38. Both are perfect to 64K.
  - Combining them hurts training (acc 0.55).
  - Pilot 9 tries 80K steps.
- **Capacity at train L = 1024, K ≤ 128, min_decay 0.35, 20K steps: did not train** (acc 0.00 / 0.47 / 0.39 for mem_heads 4 / 8 / 16). A training failure, not a memory result. Pilot 9 retries with a length curriculum (256 → 1024) and 40K steps.
- **LM v5** (`gdn,hais`) is at step ~9.5K, train answer acc 0.51–0.58. Eval pending.

### 8.13 LM v5: first in-text recall from a HAIS LM, and the state-drift problem (2026-10-12 ~08:40 UTC)
**`gdn,hais` ×4**, min_decay 0.35, 500M tokens. Facts-in-text recall, K = 1 / 8:

| Variant | Val ppl (carried 2M) | 2K | 16K | 131K |
|---|---|---|---|---|
| write L1 0.005 | 1403 | **0.95 / 0.56** | **0.45 / 0.38** | 0.05 / 0.03 |
| no penalty | 66.9 | ≈ 0 | ≈ 0 | ≈ 0 |

- **Drift diagnosis** (`lm/stream_drift.py`): along a 2M-token carried stream, some delta-rule state norms grow without bound (≈ 200 → 9,800). The L1-0.005 model's CE rises 4.27 → 7.3. GDN's states also grow, but its CE stays at 4.6–5.0.
- **Cause:** values are unnormalised, so the delta-rule state is unbounded.
- **Fix (`--v_norm`):** L2-normalise values per head, times a learnable scale. That bounds the state by √d_k · scale. Streaming equivalence verified (fp32 Δ 9e-4).
- **LM v6 (running, H200 #1):** `gdn,hais` + L1 0.005 + v_norm; `gdn,hais` + v_norm; `gdn` ×8 + v_norm.

### 8.14 LM v6: v_norm fixes the drift, and HAIS beats GDN on in-text recall to 131K (2026-10-12 ~10:00 UTC)
500M tokens, 50M params, n = 20 per cell, facts-in-text recall K = 1 / 8:

| LM | Val ppl | 2K | 16K | **131K** | 1M |
|---|---|---|---|---|---|
| **`gdn,hais` + v_norm (no penalty)** | 69.0 | 1.00 / 0.65 | 1.00 / 0.89 | **1.00 / 0.14** | 0.15 / 0 |
| `gdn,hais` + v_norm + L1 0.005 | 68.1 | 1.00 / 0.93 | 1.00 / 0.88 | 0.05 / 0 | 0 / 0 |
| `gdn` ×8 + v_norm | **59.8** | 0.25 / 0.25 | 0 / 0.10 | 0 | 0 |

- Perplexity is stable on carried 2M streams now, vs 1403 before v_norm.
- HAIS costs ~15% perplexity vs pure GDN but gives the only long-range in-text recall.
- Single seed, small n. **LM v7 (running)**: seeds 1 and 2 of the winner, max query age 1000 segments (aiming at 1M), mem_heads 16.
- **Launcher bug found and fixed:** v3–v6 passed per-run args *before* the common args, and argparse keeps the last value. So "lm3_gdn_s1" actually ran with **seed 0**; its ppl-19,600 divergence vs lm_gdn's 112 came from GPU nondeterminism plus unbounded state, not a different seed. LR/layout flags were unaffected (not in the common args). v7 passes common args first, and the configs were verified from the logs.

### 8.15 Capacity law (Pilot 9) and LM scaling launched (2026-10-12 ~11:30 UTC)
**Capacity** (`hais,hais` recipe + length curriculum 256 → 1024, K ≤ 128, 40K steps, 2 seeds). Recall at **64K**, K = 128 / 256 / 512:

| mem_heads (archival = half) | K = 128 | K = 256 | K = 512 |
|---|---|---|---|
| 4 | 0.38 (1 seed failed) | 0.18 | 0.07 |
| 8 | 0.94 | 0.67 | 0.27 |
| 16 | **0.99** | **0.83** | 0.44 |

- **Capacity grows with archival key dimensions:** ~350 facts at 50% recall with 8 heads (4 archival × 64-d), ~470 with 16.
- At 1M tokens, one seed per config holds and one degrades. Extra seeds 2–3 for h8/h16 are running.

**Decoys at 80K steps** (3 seeds):
- L1 0.1: 0.35 / 0.31 / 0.35 at 1M.
- State passing 0.99: 0.48 / 0.35 / 0.32.
- Both are ≥ 0.98 to 64K. Still open.

**LM v8 (H200 #2):** d = 768 × 12 blocks (117M params), 1B tokens, max query age 1000 segments, n = 40 eval. Compares `gdn,hais` ×6 (+ v_norm, min_decay 0.35, archival 6/12 heads) vs `gdn` ×12 (+ v_norm). Data: `~/data/fwe_2b` (2B tokens).

### 8.16 LM v7: 131K in-text recall is seed-dependent (1/3), and LM v9 launched (2026-10-12 ~12:40 UTC)
`gdn,hais` + v_norm, 500M tokens, facts-in-text recall, K = 1 / 8:

| Run | Val ppl | 2K | 16K | 131K |
|---|---|---|---|---|
| seed 0 (v6) | 69.0 | 1.00 / 0.65 | 1.00 / 0.89 | **1.00** / 0.14 |
| seed 1 | 68.1 | 0.90 / 0.55 | 0.50 / 0.04 | 0 |
| seed 2 | 65.8 | 0.40 / 0.21 | 0 / 0.04 | 0 |
| seed 0, max_age 1000 | 69.2 | 1.00 / 0.89 | 1.00 / 0.89 | 0.50 / 0.03 |
| seed 0, mem_heads 16 | **55.0** | 1.00 / 0.49 | 0.45 / 0.34 | 0 |

- The LM-level result is not yet reliable, which mirrors the synthetic task before training length was doubled (20K → 40K steps took it from 4/6 to 9/9 seeds).
- **LM v9 (H200 #1):** 3 seeds at 1B tokens (30.5K steps), max_age 1000, plus a matched `gdn` ×8 baseline.
- **LM v8 (H200 #2, 117M params)** at step ~4.7K/30.5K: train answer acc `gdn,hais` 0.69 vs `gdn` 0.39.

### 8.17 Capacity with 4 seeds (Pilot 9 + 10, 2026-10-12 ~13:20 UTC)
Recall at **64K**, mean ± 95% CI over 4 seeds:

| Archival capacity | K = 128 | K = 256 | K = 512 |
|---|---|---|---|
| mem_heads 8 (4 archival × 64-d) | 0.96 ± 0.05 | 0.74 ± 0.12 | 0.34 ± 0.13 |
| mem_heads 16 (8 archival × 64-d) | **0.99 ± 0.01** | **0.87 ± 0.04** | 0.49 ± 0.06 |

- Capacity at 50% recall: **~400 facts (h8) → ~500 (h16)**.
- At **1M**, large-K recall partly decays (h16, K = 128: 0.56 ± 0.31). Only 1/4 seeds is fully distance-invariant (s1: 1.00 / 1.00 / 1.00 / 0.83 at K = 32 / 64 / 128 / 256).
- **Pilot 11 (running):** h16 at 80K steps, 3 seeds. 2× training fixed distance-invariance at K ≤ 32 before.
- LMs at ~6.5K / 30.5K steps. Train answer acc: v9 `gdn,hais` 0.76–0.79 (3 seeds) vs `gdn` 0.45; v8 117M about equal (0.61 vs 0.63).

### 8.18 LM v9 (1B tokens, 3 seeds): reliable to 16K, HAIS at no perplexity cost (2026-10-12 ~16:45 UTC)
Facts-in-text recall, K = 1 / 8 (n = 20 per cell):

| LM (50M) | Val ppl | 2K | 16K | 131K | 1M |
|---|---|---|---|---|---|
| `gdn,hais` s0 | 49.1 | 1.00 / 0.90 | 0.85 / 0.59 | 0.25 / 0.03 | 0 |
| `gdn,hais` s1 | 48.3 | 1.00 / 0.98 | 1.00 / 0.90 | **0.55 / 0.26** | 0 |
| `gdn,hais` s2 | 51.0 | 1.00 / 0.91 | 1.00 / 0.40 | 0.20 / 0.01 | 0 |
| `gdn` ×8 | 48.7 | 0.55 / 0.79 | 0 / 0.29 | 0 | 0 |

- 2× training made LM recall reliable to 16K (3/3 seeds). HAIS no longer costs perplexity.
- 131K is partial and 1M is zero. That is the same shape as the synthetic *decoy* failure: real text is full of fact-like tokens whose small write leaks accumulate.
- **Next lever: budgeted archival writes** (`--write_budget m`): each archival head may write only its top-m tokens per 64-token block, with hard zeros elsewhere, so erasure scales with writes, not tokens. Streaming equivalence verified.
  - Pilot 12 (H200 #1): decoys with m ∈ {1, 2, 4} × 3 seeds, plus clean m = 2.
- **Multi-GPU ready:** `lm/train_lm.py` supports `torchrun` (DDP; per-rank lanes and state, rank-0 logging, per-rank data pointers). The single-GPU path is regression-tested; multi-GPU is untested until a node exists.
  - `node_setup.sh`: env, 3 correctness gates, 20B-token data prep.
  - `lm/launch_node.sh`: `gdn,hais` ×8, d = 1024, ~300M params, 10B tokens, plus an optional matched GDN baseline.

### 8.19 Capacity at 2× training (Pilot 11): hundreds of facts at 1M tokens (2026-10-12 ~18:00 UTC)
`hais,hais` recipe, mem_heads 16, length curriculum 256 → 1024, K ≤ 128 in training, **80K steps**. Recall:

| Seed | 64K: K = 256 / 512 | **1M: K = 128 / 256 / 512** |
|---|---|---|
| s0 | 0.85 / 0.45 | 1.00 / 0.84 / 0.44 (flat with distance) |
| s1 | 0.98 / 0.68 | 0.82 / 0.77 / 0.53 (partial decay) |
| s2 | **1.00 / 0.88** | **0.98 / 0.97 / 0.81** (flat) |

- 2× training again buys distance-invariance (2/3 seeds) and more capacity: s2 holds ~500+ facts at 1M with 8 archival 64-d heads.
- **LM v10 (H200 #2, running):** write_budget 2 and 4 in the 50M LM (1B tokens, seed 0), to compare with `lm9_gdnhais_s0`.

### 8.20 Pilot 12: write budgets: a design flaw, then a resized version (2026-10-12 ~18:10 UTC)
Top-m archival writes per **64-token** block, decoy 0.1, 3 seeds. Mean recall at 1M (early), K = 1 / 8 / 32:

| Budget | Final train acc | Recall at 1M |
|---|---|---|
| m = 1 | 0.64 | 0.69 / 0.54 / 0.31 |
| m = 2 | 0.68 | 0.67 / 0.69 / 0.47 |
| m = 4 | 0.77 | 0.46 / 0.48 / 0.28 |
| clean m = 2 (no decoys) | 0.71 | collapses beyond 4K |

- **Flaw:** training packs up to 32 facts into 256 tokens, up to ~8 per 64-token block, so a budget of 1–4 starves real facts. Training accuracy drops from 1.00 to 0.64–0.77, and even the clean task fails.
- **Positive signal:** with decoys, two m = 2 seeds reach 1.00 at 1M (K = 1), vs 0.18–0.5 without a budget.
- **Pilot 13 (H200 #1, running):** budget sized to the data. 256-token blocks, m ∈ {40, 48} (above the max training facts per block), decoys plus clean, 3 seeds.
- LM v10 (H200 #2) uses 64-token blocks with m = 2 / 4. LM fact density is lower (drills: 8 facts per 512 tokens), but it may still starve multi-token facts. Watch its train answer acc vs lm9.

### 8.21 Pilot 13: data-sized budget doesn't fix decoys; Pilot 14 tries a hard write threshold (2026-10-12 ~19:20 UTC)
Recall at 1M (early), K = 1 / 8 / 32, mean over 3 seeds:

| Config | Train acc | Recall at 1M |
|---|---|---|
| clean, block 256, m = 40 | 0.97 | 1.00 / 0.92 / 0.83 |
| decoy 0.1, m = 40 | 0.73 | 0.04 / 0.15 / 0.11 |
| decoy 0.1, m = 48 | 0.95 | 0.52 / 0.52 / 0.29 |

- A budget loose enough for real facts (~19% of tokens) is looser than the decoy rate (10%), so it does not constrain leakage.
- **Pilot 14 (H200 #1):** `--arch_tau τ` makes archival writes with β ≤ τ exactly zero in the forward pass, with a straight-through gradient, so sub-threshold decoy leakage cannot accumulate at all. Decoy τ ∈ {0.1, 0.3} plus clean τ = 0.1, 3 seeds.
- LM v8 (117M) at step ~28K/30.5K. LM v10 (budget m = 2 / 4, 64-token blocks) at 8.6K: train answer acc 0.66 / 0.79 vs lm9 at the same step.

### 8.22 Pilot 14: hard threshold makes the clean recipe bulletproof; decoys need pressure (2026-10-12 ~20:20 UTC)
`--arch_tau`, 3 seeds, recall at 1M (early), K = 1 / 8 / 32:

| Config | Recall at 1M |
|---|---|
| **clean τ = 0.1** | **1.00 / 1.00 / 1.00 (3/3 seeds, train acc 1.00)** |
| decoy τ = 0.1 | 0.35 / 0.40 / 0.28 (one seed at 0.94 for K = 1) |
| decoy τ = 0.3 | ≈ 0 |

- **Revised decoy diagnosis:** decoys are not faint leakage. A mis-gated decoy is a *full-strength* write that overwrites the real fact for that key.
  - At 1M each key gets ~200 decoy hits (100K decoys over 512 keys); at 64K only ~12. That is why 64K survives and 1M does not.
  - Training at 256 tokens gives ~1.5 key collisions per sequence, too little pressure to make the gate context-perfect.
- **Pilot 15 (running, H200 #1):** train at decoy rate 0.3 and evaluate at 0.1 (`--eval_decoy`), with and without τ = 0.1, plus an 80K-step τ run. 3 seeds each.
- LM v8 (117M) finished training; eval running.

### 8.23 LM v8 (117M) and the real reason LM recall stops at 16K–131K (2026-10-12 ~21:40 UTC)
**LM v8**: d = 768 × 12 blocks, 117M params, 1B tokens, n = 40 per cell. Facts-in-text recall, K = 1 / 8:

| LM | Val ppl | 2K | 16K | 131K | 1M |
|---|---|---|---|---|---|
| **`gdn,hais` ×6** | **41.5** | 0.98 / 0.93 | **0.95 / 0.90** | 0.13 / 0 | 0 |
| `gdn` ×12 | 42.9 | 0.48 / 0.06 | 0.05 / 0.04 | 0 | 0 |

At 117M, HAIS wins on recall *and* perplexity.

**Diagnosis: a data gap, not an architecture gap.**
- Query ages in lm9 training after warmup: **median 19 segments** (~19K tokens), p90 79, p99 180.
- Only **3.9% of queries were ≥ 128 segments** (131K) and **0% ≥ 512**. The model never practised the distances where it fails, and short-range GDN heads could serve almost every training query, so the archival heads had no pressure to work.
- Cause: queries popped a *random* pending fact, which keeps the pool young.

**Fix (`--age_target loguniform --max_pending 4096`, plus fact_rate 0.1):**
- Facts persist.
- Each query targets a log-uniform age (1–2000 segments) and *refreshes* the fact (the query re-states the answer).
- Simulated over 5K steps: median 42, p90 880, **35% ≥ 128 segments, 17% ≥ 512**, max ~1990.

**LM v11 (H200 #2, running):** 50M, 1B tokens. `gdn,hais` (age-targeted); `gdn,hais` + arch_tau 0.1; `gdn` ×8 baseline on the same data.

### 8.24 What erases a fact on real text, and a training-free fix that reaches 1M (2026-10-12 ~23:30 UTC)
**LM v10 (write budget, 50M, seed 0)** does not help. Facts K = 1 / 8 (n = 20):

| Run | Val ppl | 16K | 131K | 1M |
|---|---|---|---|---|
| m = 2 | 53.8 | 1.00 / 0.86 | 0.35 / 0.08 | 0 |
| m = 4 | 52.3 | 0.90 / 0.13 | 0.05 / 0 | 0 |

That is within lm9's seed spread and costs ~4 ppl. Dropped.

**Erasure diagnostic (`lm/write_stats.py`).**
- Method: stream real text after one planted fact. For every archival head, track the exact retention of the fact's key under the model's own later writes, r = k_fᵀ ∏(I − β_t k_t k_tᵀ) k_f. It is computed with the fla kernel on v = 0, so it is the real state dynamics.
- lm8 (117M):
  - The fact is written at full strength (median max β 0.996).
  - Plain text is written with β > 0.1 at 7% of tokens.
  - Median-head retention falls roughly as a power law: 0.45 at 16K, 0.12 at 131K, 0.013 at 1M.
  - The best head per stream keeps 0.87 at 1M. A few heads hold the fact in a key subspace that text barely touches.
- Simulated threshold: drop writes with β ≤ τ in the retention calculation only.
  - lm8: the best head goes to 1.000 at 1M for every τ ≥ 0.1. The median head goes 0.12 → 0.56 at 131K with τ = 0.5.
  - lm9 s1: the median head is 1.00 at 1M with τ = 0.5.
- **Conclusion: most erasure comes from the many weak writes (β ≤ 0.1–0.5, ~90% of tokens).** The rare strong text writes are mostly orthogonal to fact keys.

**Training-free inference threshold (`eval_lm.py --arch_tau τ`, lm8 unchanged).** Facts K = 1 / 8, n = 10 (n = 20 for τ ≥ 0.95):

| τ | 16K | 131K | 1M | Val ppl |
|---|---|---|---|---|
| 0 (as trained) | 1.00 / 0.88 | 0.20 / 0 | 0 / 0 | 41.5 |
| 0.1 | 1.00 / 0.88 | 0.80 / 0.23 | 0 / 0 | |
| 0.5 | 1.00 / 0.90 | 0.90 / 0.45 | 0.30 / 0.03 | 41.8 |
| 0.7 | 1.00 / 0.90 | 0.90 / 0.48 | 0.60 / 0.05 | |
| **0.9** | 1.00 / 0.90 | **0.90 / 0.55** | **0.70 / 0.10** | 42.5 |
| 0.95 | | 0.70 / 0.44 | 0.50 / 0.09 | |
| 0.98 | | 0.50 / 0.39 | 0.30 / 0.06 | |

- At τ = 0.9 with n = 40, 1M K = 1 is **0.30**. Pooled over the two runs it is 19/50 = **0.38**.
- **First non-zero 1M in-text recall from an LM, with no retraining, at +2.4% perplexity.** That is still better than the matched GDN's 42.9.
- The 50M lm9 s1 improves at 131K (0.60 → 0.80) but stays 0 at 1M.

**Generative demo** (`demo_recall.py --arch_tau 0.9`, 1.05M tokens, 2.7 GB constant memory, ~65K tok/s):
- A fact written in the first 10K tokens is recalled exactly from ~1.04M tokens away ("731958", 2/2 haystacks).
- The same fact written after a 31K–52K-token prefix fails, even at 79K distance.
- `lm/probe_write.py`: write strength at the fact is almost unchanged by the prefix, except that one archival layer loses a head. The fact write is context-dependent, and **missing one robust head is enough to lose the fact**.
- Next lever: training. The model should learn to always write facts into the protected subspace and never write text there.

**LM v13 (H200 #1, running):** fine-tune lm8 for 300M tokens on age-targeted long-range queries, with and without train τ = 0.5. Each is evaluated at inference τ ∈ {0, 0.5, 0.9}.

### 8.25 Why late-written facts are lost, decoys, and what the node will run (2026-10-13 ~00:30 UTC)
**Late facts fail at the read, not at storage.** lm8 at τ = 0.9, one fact, same haystack window (`probe_keys.py --contiguous`, `demo_recall.py`):
- **Storage is fine.** A fact written after a 52K-token prefix still has 15/27 written archival heads at r > 0.9 at 80K (median 0.996), and 13/27 at 1M. Yet greedy recall fails at 79K.
- **Zeroing the archival state** right before the fact changes nothing: prefix content in the archival memory is not the problem.
- **Dropping the *whole* recurrent state** right before the fact makes recall exact at 79K *and* at ~1M ("731958"). So the non-archival state at write time corrupts how the fact is encoded.
- **Culprit candidates:**
  - The GDN layers learned near-archival heads: median half-lives of 13.9K, 47.8K, 348K and ∞ tokens in block 8, and 9K, 11.8K and ∞ in block 10.
  - Fact keys shift with the prefix: median cosine to the fresh-state key is 0.93, and some heads are ~0 because the argmax token changed.
  - These heads inject slowly varying context into the residual stream that the archival keys are computed from. Write and read then happen under different contexts.
- **Capping GDN memory at inference** (`--gdn_half_life`, no retraining). lm8, τ = 0.9, n = 20, K = 1 / 8:

  | GDN half-life cap | 16K | 131K | **1M** | Val ppl |
  |---|---|---|---|---|
  | none | 1.00 / 0.89 | 0.80 / 0.43 | 0.35 / 0.11 | 42.5 |
  | 32K | 1.00 / 0.88 | 0.70 / 0.13 | 0.40 / 0.14 | 42.6 |
  | **8K** | 0.85 / 0.53 | 0.70 / 0.13 | **0.65 / 0.16** | 42.8 |
  | 2K | 0.60 / 0.23 | 0.65 / 0.11 | **0.65 / 0.13** | 42.9 |

  - **1M recall nearly doubles** (0.35 → 0.65). It also fixes the late-fact demo.
  - It hurts 16K–131K, because the trained model relies on those long GDN heads at mid range.
  - So timescale separation must be *trained in*: archival heads are the only long memory, and everything else has a bounded half-life. It is the LM analogue of the synthetic min_decay result.

**Pilot 15 (decoys, train rate 0.3, eval 0.1, 3 seeds).** Recall at 1M, early mode, K = 1 / 8 / 32:

| Config | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| 40K steps | 0.88 / 0.88 / 0.88 | 0.81 / 0.38 / 0.30 | 0.62 / 0.62 / 0.50 | 0.77 / 0.63 / 0.56 |
| + τ = 0.1 | 0.44 / 0.44 / 0.27 | 0.56 / 0.69 / 0.58 | **1.00 / 1.00 / 1.00** | 0.67 / 0.71 / 0.62 |
| + τ = 0.1, 80K steps | 0.44 / 0.50 / 0.30 | 0.19 / 0.12 / 0.11 | **1.00 / 1.00 / 0.92** | 0.54 / 0.54 / 0.44 |

- Decoys remain a seed lottery. Some seeds solve them perfectly at 1M, and more training does not move the mean.
- Parked: the LM's problems are elsewhere for now.

**Data: 10 fact phrasings plus 3 held-out ones** (`lm_data.TEMPLATES` / `HELDOUT`, `--templates multi|heldout`).
- lm8 scores 0 on the classic passkey prompt even at 2K. Its archival memory is template-specific, which a usable model cannot be.
- Values cover digits (3–7), pseudo-words, cities, years, colours and weekdays.
- `secret` mode reproduces the v1–v13 fact stream exactly (verified over 200 facts).
- Training now logs exact-match answer accuracy by query age (`age_acc`, buckets 0–4 / 4–32 / 32–256 / 256+ segments).

**8-GPU node plan** (scripts ready: `node_bootstrap.sh <ip>`, `lm/launch_node_sweep.sh`, `lm/launch_node.sh`):
1. **Sweep:** 8 × 117M, 1B tokens each, multi-template, age-targeted. A factorial over train τ (0 / 0.5) × GDN half-life cap (none / 8K), plus a 2K cap, a seed, 24 heads, and a GDN baseline. Evaluated on trained and held-out phrasings.
2. **Scale-up:** ~300M (d = 1024, 16 blocks) with the winning flags on all 8 GPUs, about 10B tokens.

### 8.26 A data flaw invalidated LM v11–v14 and node sweep 1; fix and relaunch (2026-10-13)
**The flaw (`--age_target loguniform`, introduced in §8.23).**
- Facts persisted, and every query re-stated the answer to "refresh" the fact.
- So the same (name, code) pair recurred in a lane for thousands of steps, and the models memorised the pairs **in their weights**.
- **Evidence:**
  - Node sweep 1 training logs: the pure-GDN baseline answered **80% of queries 256+ segments old** (≥ 262K tokens). That is impossible for its state, which forgets beyond ~16K.
  - Every run of that data family scored near 0 on fresh facts at eval, even at 2K: LM v11 (3 runs), LM v13 (3 fine-tunes, which also lost lm8's own recall and went from ppl 41.5 to 50.8), LM v14, and node sweep 1 (8 runs).
  - Held-out phrasings scored exactly 0.
- In-context recall never had to form, because a weight lookup answered almost every training query.

**Fix (`--age_target loguniform_once`).**
- Log-uniform target age, nearest pending fact, **asked exactly once** (popped). No refresh, and no two live facts in a lane share a name.
- To keep old facts available, facts arrive faster than queries: fact_rate 0.5 and query_rate 0.25 per 1K-token segment. Half the facts are never asked, so they act as real distractors.
- Simulated (4 lanes, 3,500 segments): median query age 44 segments, p90 918, **36% ≥ 128, 17% ≥ 512**, ~580 live facts per lane.
- With fact_rate = query_rate the pool drains: at most 7% of queries reach ≥ 512.
- The old mode is kept only for reproducibility, documented as flawed.

**Sweep 1 still contains one usable signal.** s_tau (train τ = 0.5) evaluated at τ = 0.9 reached 16K 0.78 / 0.43, 131K 0.55 / 0.35 and **1M 0.40 / 0.26** on fresh multi-template facts, despite training that rewarded memorisation. Held-out phrasings were 0 for every run.

**Engineering (all verified against the previous implementation):**

| Change | Speedup | Check |
|---|---|---|
| fla Triton causal conv | 3.8× per conv | fp32 Δ 5e-7, chunked == full |
| fused-dtype RMSNorm, bf16 v_norm, one GEMM for the six mixer projections, single input cast | 1.5× together | lm8 streamed CE 3.5236 vs 3.5231 |
| `--compile` per block | 1.3× | fp32 grads match eager to 8e-4 median; bf16 differences are below the bf16-vs-fp32 noise |

- Net speedup: **1.9×** on a contended H200. One **B200 trains the 117M model at ~320K tok/s**.
- Background batch prefetch.
- Batched facts eval (`--batch 8`): identical samples and scores, ~3× faster.
- DDP path tested with 2 ranks (gloo on one GPU), including checkpoint and resume.

**Node (8× B200, <node>):**
- Gate results:
  - Mamba-3 SISO test: 1 failed, 8 passed. Mamba-3 is not used in the LMs.
  - GDN_GRAD_OK.
  - HX_STREAM_OK.
- `prep_data.py` hung at pool exit after train.bin was complete. meta.json was written by hand, and the 10B-token prep still has to be run.

**Running:**
- Node sweep 2: the same 8 arms on loguniform_once data.
- LM v15 (H200 #2): lm8's original random-pop recipe from scratch, plus train τ 0.5, with and without the GDN cap.
- LM v16 (H200 #1): fine-tune lm8 on loguniform_once data at LR 3e-4, with and without the cap.

### 8.27 Node sweep 2: distance-invariant in-text recall to 1M tokens in a 117M LM (2026-10-13 ~06:00 UTC)
**Setup.** 8 × B200, one run per GPU. 117M `gdn,hais` ×6, 1B tokens, loguniform_once data (fact 0.5 / query 0.25 per segment, ages to 2000 segments), 10 phrasings. Fresh-fact eval: n = 40 for K = 1, n = 20 samples × 8 facts for K = 8.

Recall K = 1 / 8 on the trained phrasings:

| Run (train-time levers) | inference τ | 16K | 131K | **1M** | Val ppl |
|---|---|---|---|---|---|
| `gdn` ×12 baseline | – | 0.05 / 0.03 | 0.08 / 0.05 | 0 / 0.03 | 56.8 |
| base (no levers) | 0.9 | 0.93 / 0.82 | 0.40 / 0.32 | 0.50 / 0.24 | 52.2 |
| τ 0.5 | as trained | 0.83 / 0.54 | 0.40 / 0.30 | 0 / 0.01 | 60.2 |
| hl8k | 0.9 | 0.48 / 0.21 | 0.40 / 0.16 | 0.38 / 0.13 | 58.0 |
| τ 0.5 + hl8k (s0 / s1) | 0.9 | 0.08 / 0.31 · 0.40 / 0.58 | ≤ 0.05 | ≤ 0.08 | 59.4 / 59.3 |
| **τ 0.5 + hl8k + text-write L1 0.05, s0** | **0.9** | 0.98 / 0.69 | **0.88 / 0.66** | **0.90 / 0.64** | 54.8 |
| **same, s1** | **0.9** | 1.00 / 0.93 | **0.98 / 0.89** | **0.98 / 0.74** | 60.6 |
| same recipe, s0 / s1 | as trained (τ 0.5) | 0.98 / 0.80 · 1.00 / 0.98 | 0.68 / 0.53 · 0.93 / 0.77 | 0.20 / 0.08 · 0.08 / 0.04 | |

- **The winning recipe is flat from 16K to 1M in both seeds**: 0.90–0.98 for one fact and 0.64–0.74 for eight facts, at constant memory.
- It needs all three train-time levers *and* the inference threshold raised to 0.9. τ + hl8k without the text-write penalty fails in both seeds.
- The penalty acts only on archival writes at ordinary text tokens; inserted facts and queries are free. It is the LM analogue of the synthetic archival-only L1.
- **Held-out phrasings: 0 for every run.** Archival writes are phrasing-specific. Generalisation is the next problem.
- **Training-time per-age accuracy is ~0.03 at every age** for every run.
  - With ~580 live facts per lane, the long-range training queries are overloaded.
  - The recall mechanism is learned from the short-range drills. Distance invariance comes from the architecture and penalties, not from long-range supervision, as in the synthetic task (trained at 256, works at 1M+).

**Mechanism check (`write_stats.py`, retention only over the heads that stored the fact, 4 streams of 1M tokens):**

| Model | Archival heads storing a fact | Text tokens written (β > 0.1) | Written-head retention at 1M, τ 0.9: median / share > 0.9 | 1M recall (K = 1) |
|---|---|---|---|---|
| lm8 (old recipe) | 30 of 36 | 7.1% | 0.55 / 36% (~11 heads) | ~0.35 |
| s_tau_hl8k_twl s1 | 10 of 36 | 1.2% | 0.09 / 12.5% (~1 head) | 0.98 |

- The new model keeps *less* of the fact, yet recalls far better.
- Retention is not the bottleneck. **A single surviving head is enough once the read is clean.**
  - The GDN cap removes the context drift of §8.25.
  - The text-write penalty cuts text writes 6×, so far less unrelated content sits in the archival state to corrupt the read.

**Launched:**
- **Phase 2 on the node:** 494M `gdn,hais` ×8 (d = 1536, 16 blocks, 24 heads, 12 archival), 10B tokens of sample-10BT. Same levers plus **40 phrasings** (`--templates xl`) for generalisation, then a matched 494M `gdn` baseline. Both evaluated on trained and held-out phrasings at τ as trained and 0.9.
- **H200 ablations** (`launch_ablate_twl.sh`): text-write L1 alone; + τ; + hl8k; and the full recipe with seed 2.

### 8.28 Ablations: what each lever does (H200s, 117M, 1B tokens, sweep-2 data, 2026-10-13 ~09:00 UTC)
Recall K = 1 / 8 at inference τ = 0.9; n = 40.

| Train-time levers | 16K | 131K | 1M | Val ppl |
|---|---|---|---|---|
| none (sweep 2) | 0.93 / 0.82 | 0.40 / 0.32 | 0.50 / 0.24 | 52.2 |
| τ + cap, no text-write L1 (2 seeds, sweep 2) | ≤ 0.40 / ≤ 0.58 | ≤ 0.05 | ≤ 0.08 | 59.3 |
| text-write L1 only | 0.88 / 0.56 | 0.53 / 0.41 | 0.50 / 0.29 | 60.9 |
| text-write L1 + τ | 0.68 / 0.65 | 0.38 / 0.31 | 0.38 / 0.35 | 63.4 |
| text-write L1 + cap | 0.85 / 0.27 | 0.93 / 0.31 | **0.93 / 0.56** | 55.5 |
| **all three, s0 / s1 / s2** | 0.98–1.00 / 0.69–0.93 | 0.65–0.98 / 0.66–0.89 | **0.83–0.98 / 0.54–0.74** | 54.8–60.6 |

- **The GDN half-life cap is what makes recall distance-invariant.** Without it, recall halves between 16K and 131K and stays there. With it, 131K ≈ 1M.
- **The text-write penalty is necessary for everything.** τ + cap without it fails completely.
- **Train-time τ mostly adds multi-fact robustness** (K = 8 at 16K: 0.27 → 0.69–0.93).
- **The inference threshold of 0.9 is needed in all variants** except seed 2, which also does well as trained (1M 0.73 / 0.44).
- 3 of 3 seeds of the full recipe hold 1M recall.

### 8.29 The 494M model, v1: better LM, recall degrades late in training; v2 relaunched (2026-10-13 ~10:00 UTC)
**v1** (`runs_node/gdnhais_d1536_v1`): d = 1536, 16 blocks, 494M params, 10B tokens, ~1M tok/s on 8 × B200 (2.8 h). Full recipe, 40 phrasings, fact 1.0 / query 0.5 per 2K segment.
- **LM:** val ppl **24.2**, vs 52–61 for the 117M models.
- **Recall at inference τ 0.9**, K = 1 / 8:

  | Checkpoint | 16K | 131K | 1M | 2K |
  |---|---|---|---|---|
  | step 12K (31%) | 1.00 / 0.93 | 0.35 / 0.26 | 0.25 / 0.15 | |
  | final | 0.53 / 0.38 | 0.08 / 0.08 | 0.05 / 0.07 | 0.25 |

- **Generative 1M demo: 0/5.** Answers have the right *format* (6-digit codes, a city, a pseudo-word) but the wrong values, which looks like a learned prior.
- **Diagnosis:**
  - ~565 live facts per lane overload the archival memory. Training per-age accuracy is ≈ 0.07 at *every* age, even 1–4 segments, so non-drill queries are essentially unanswerable.
  - The drills (8 facts + queries inside one 2K segment) can be served by the 8K-half-life GDN layers alone, so nothing rewards archival reads.
  - The larger model learns to guess instead. The 117M models did not, but their margin was thin.
- **v2:**
  - fact 0.2 / query 0.1 per segment. Simulated: **~110 live facts**, median query age 31 segments, 40% ≥ 131K tokens, 20% ≥ 524K, 90% beyond the GDN half-life.
  - drill_frac 0.15, age warmup 10K, 8B tokens.
  - Per-age training accuracy is the early-warning signal: it should now rise for 4+ / 32+ / 256+ segments.
- The 0.5B GDN baseline is postponed until the HAIS 0.5B works.

### 8.30 v2/v3 at 494M, compositional facts, and a diagnostic sweep (2026-10-13 ~11:00 UTC)
**v2** (low fact load) diverged at step 9.4K.
- Grad norm reached 1.1e5 and lm loss jumped 3.9 → 12.6 within 50 steps. The checkpoint (saved every 500 steps) was already post-spike.
- The model is not the trigger: on held-out text with fresh or carried states (2M tokens), grad norm is 0.6–0.7 and ‖S‖ stays ≤ 21.
- **Fixes:**
  - `--spike_guard 20`: skip the update and reset the states if grad norm > 20 × the running median.
  - The previous checkpoint is now kept as `ckpt.pt.prev`.

**v3** (v2 + guard, LR 6e-4):
- The guard fired 32 times by step 8.6K. These were mostly grad-norm-only bursts (gn 120–1200 with normal loss) lasting up to ~16 consecutive steps.
- Recall at step 9.5K, τ 0.9: 16K 0.55 / 0.46, 131K 0.05 / 0.04, **1M 0 / 0.03**. Stopped.
- **Two 494M runs now fail at long range where every multi-template 117M run succeeded.**

**Compositional facts** (117M, 6 frames × 53 attributes = 318 phrasings), recall K = 1 / 8:

| Run | 16K | 131K | 1M | Unseen frames × attributes, 16K |
|---|---|---|---|---|
| text-write L1 0.02 | 1.00 / 0.81 | 0.90 / 0.53 | 0.90 / 0.55 | 0.18 / 0.07 |
| text-write L1 0.05, s0 | 0.83 / 0.66 | 0.08 / 0.04 | 0.05 / 0.06 | 0.03 / 0.02 |
| text-write L1 0.05, s1 | 0.55 / 0.44 | 0.53 / 0.44 | 0.43 / 0.31 | 0.03 / 0.01 |
| text-write L1 0.05 + drill_k 16 | 0.93 / 0.81 | 0 / 0.04 | 0.05 / 0.03 | 0 |

- With more varied facts, long-range success becomes seed- and setting-dependent.
- Generalisation to unseen phrasings stays ≤ 0.18.

**Phase 2b diagnostic sweep** (node, `launch_node_diag.sh`): one single-GPU run per factor that differs between the 117M success and the 494M failure, everything else equal to the 117M recipe. The factors are a 4th seed, 40 phrasings, 2K segments, low fact load, 0.15 drills, 250M, 494M on a single GPU, and LR 6e-4.

### 8.31 The training signal, not the architecture: diagnostic sweep, sweep 3, and the cap on query age (2026-10-13 evening)
**Diagnostic sweep** (node, single-GPU, 117M recipe ± one factor, with `--spike_guard 20`). Most runs failed at 131K+, including the exact-recipe seed 3.
- **The spike guard was the main confounder:** it tripped 5,013 times in seed 3, and each trip wiped every lane's memory. It is dropped.
- **Final training per-age accuracy (non-drill queries):**
  - ≈ 0.03 at every age with the dense stream (~580 live facts). There is no learning signal; the earlier successes learned from drills alone and were partly luck.
  - Low load (~55 live facts): **1.00 at 0–4 segments and 0.82 at 4–32**, then ~0.
  - Eval of the low-load run: 16K 1.00 / 0.90, 131K 0.28 / 0.22, 1M 0.35 / 0.24.

**Sweep 3** (low load, no guard, train τ 0.9 / 0.5, several variants, max query age 2000 segments):
- At step 10K: 0.90–1.00 at 0–4 segments and 0.67–0.86 at 4–32, in 6/8 runs including 250M.
- At step 20K, when the age warmup reaches 2000 segments: **every run collapsed to ~0 at every age**, and stayed there to the end.
- **Mechanism:** a flood of unanswerable old queries, with answer loss weighted ×5, teaches the model to guess plausible values instead of reading memory. Earlier late-training recall losses look like the same thing: 494M v1 and v3, and the comp runs.

**Sweep 4** (node, chained after sweep 3) and **H200 h4 runs:**
- Training queries are capped at 32 / 64 / 128 segments, i.e. within what is learnable. Distance beyond that is left to the architecture, as in the synthetic task.
- Arms: τ 0.9 / 0.5, a seed replicate, ans_weight 1, 40 phrasings, 250M, and compositional phrasings (H200).

### 8.32 Capped query ages do not extrapolate; the routing hypothesis; sweep 5 (2026-10-13 night)
**Capping training query ages (sweep 4, H200 h4: low load, train τ 0.9):**
- 16K is excellent: up to 1.00 / 0.98.
- Beyond the trained range there is nothing. Max age 32: 131K ≤ 0.45, 1M ≤ 0.08. Max age 128: 131K 0 / 0.03, 1M 0.08 / 0.02.
- **The LM does not extrapolate past the distances its training queries covered**, unlike the synthetic models.

**Routing hypothesis (explains the seed lottery).**
- With truncated BPTT, a query in a later segment sends no gradient to the write that stored its fact.
- Write behaviour is therefore learned only from same-segment queries, i.e. the drills.
- Drills (gap ~500 tokens) can be served by the GDN layers (half-life 8K) *or* by the archival heads.
  - Runs that route drill recall through the archival heads extrapolate: sweep 2 s0/s1 and ablation s2.
  - Runs that route it through the GDN layers stop at the GDN horizon: everything else.

**Sweep 5 (node GPUs 0–6, chained after sweep 4):**
- Make drills *require* archival memory: GDN half-life 256 / 512 / 1024 plus **far drills** (`--drill_gap far`: facts in the first 15% of the segment, queries in the last 15%, gap ≥ 712 tokens).
- 2 seeds at 256 and at 512.
- Controls: sweep-2 recipe seed 5, and far drills with the 8K cap.

### 8.33 Sweep 5: forcing drills through archival memory makes extrapolation reliable (2026-10-13 ~23:00 UTC)
117M, sweep-2 data and levers (τ 0.5, text-write L1 0.05), 1B tokens. Recall K = 1 / 8 at inference τ = 0.9, n = 40:

| Run | 16K | 131K | **1M** |
|---|---|---|---|
| control: sweep-2 recipe (half drills, GDN cap 8K), seed 5 | 1.00 / 0.99 | 0.08 / 0.04 | 0.03 / 0.03 |
| far drills, GDN cap 8K | 0.68 / 0.72 | 0.48 / 0.43 | **0.40 / 0.44** |
| far drills, GDN cap 1024 | 0.53 / 0.41 | 0.43 / 0.36 | **0.38 / 0.21** |
| far drills, GDN cap 512, s0 | 0.45 / 0.32 | 0.45 / 0.31 | **0.55 / 0.36** |
| far drills, GDN cap 512, s1 | 0.60 / 0.61 | 0.73 / 0.56 | **0.65 / 0.55** |
| far drills, GDN cap 256, s0 | 0.50 / 0.39 | 0.50 / 0.36 | **0.63 / 0.36** |
| **far drills, GDN cap 256, s1** | **0.98 / 0.73** | **0.95 / 0.79** | **0.93 / 0.75** |

- **6/6 far-drill runs are distance-invariant:** 131K ≈ 1M ≈ the 16K level. The control fails again, as 4 of the 7 half-drill runs did; this makes the routing hypothesis (§8.32) the working explanation.
- **The extrapolation lottery is gone; the level lottery remains** (0.40–0.98 for K = 1). Raising it is the next problem.
- **The principle:** a fixed-state LM extrapolates recall only if its in-segment training signal cannot be satisfied by its bounded-memory layers. This is the LM version of the synthetic min_decay lesson.

**Launched:**
- **494M v4** (node, DDP): far drills, GDN cap 256, τ 0.5, text-write L1 0.05, 10 phrasings, LR 8e-4, 6B tokens.
- **H200 h5:** cap 512 + far drills, seeds 2 and 3, and the compositional version (2 seeds).

### 8.34 Far drills on more seeds and on compositional facts (H200 h5, 2026-10-14 ~00:00 UTC)
GDN cap 512 + far drills (sweep-2 data and levers otherwise). Recall K = 1 / 8 at τ 0.9:

| Run | 16K | 131K | 1M | Held-out sentences, 16K |
|---|---|---|---|---|
| 10 phrasings, seed 2 | 0.33 / 0.21 | 0.28 / 0.19 | 0.28 / 0.16 | |
| 10 phrasings, seed 3 | 0.73 / 0.40 | 0.70 / 0.46 | **0.88 / 0.53** | |
| **318 compositional phrasings, s0** | 1.00 / 0.84 | 0.90 / 0.83 | **0.98 / 0.73** | 0.13 / 0.03 |
| **318 compositional phrasings, s1** | 0.93 / 0.58 | 0.90 / 0.73 | **0.90 / 0.68** | 0.23 / 0.10 |

- **Far-drill runs are now 10/10 distance-invariant** (sweep 5 plus h5).
- On compositional facts, both seeds are strong to 1M; without far drills it was 1 of 4 runs.
- The level still varies by seed (0.28–0.98 for K = 1).
- Unseen phrasings stay low (≤ 0.23), so the LFM2 graft is the bet for generalisation.

**LFM2 graft (KH-1)** — `lfm2_hais.py`, trained with `train_lm.py --lfm2 LiquidAI/LFM2-350M`.
- **Design:** LFM2 is frozen; its attention and convs are limited to 1024-token chunks with positions restarting per chunk (constant memory). Archival adapters sit after layers 7 / 11 / 15, 16 heads × 64, all archival.
- **Two init traps:**
  - A zero output projection blocks every gradient into the memory (grad norm 0.0).
  - A tiny one lets the branch swamp LFM2 (lm loss 3 → 10).
  - The fix is a ReZero gate α = 0.
- **Training:** segments are 2 chunks. Drill facts land in chunk 1 and their queries in chunk 2, so only the archival memory can answer and its writes get the gradient, by construction.
- **Pilot:** 350M, 300M tokens, compositional facts, sampled LM loss (10%), running on H200 #1.
- **Local chat demo:** `kh1_chat.py` (`/load` a book, `/plant`, `/ask`, `/mem`).

### 8.35 The recipe scales: a 494M LM recalls flat to 1M (v4), and KH-1 (2026-10-14 ~00:20 UTC)
**494M v4** (`runs_node/gdnhais_d1536_v4`): d = 1536, 16 blocks, far drills + GDN cap 256 + τ 0.5 + text-write L1 0.05, 10 phrasings, 6B tokens on 8 × B200 (~1M tok/s).

| Eval (K = 1 / 8) | 16K | 131K | **1M** |
|---|---|---|---|
| inference τ 0.9 | 0.85 / 0.79 | 0.88 / 0.74 | **0.88 / 0.74** |
| as trained (τ 0.5) | 0.85 / 0.84 | 0.85 / 0.64 | 0.60 / 0.28 |
| held-out sentences | 0 | ≤ 0.03 | 0 |

- Val ppl **27.0** (v1, without the GDN cap at 256: 24.2).
- The two earlier 494M attempts (standard drills) failed beyond 16K. **With far drills the 494M model is distance-invariant.**

**KH-1 (LFM2 graft):**
- **BOS trap:** chunk-local LFM2 needs its BOS at position 0 of *every* chunk. Without it, LFM2's loss on 1K chunks is 5.42 instead of 3.08 and chat degenerates ("to to to ..."). Fixed in `lfm2_hais.py` (BOS per chunk; the archival memory never sees it).
- **Pilot 1** (no BOS) was stopped. **Pilot 2** (BOS, 358 phrasings, half of the queries as chat turns) is running on H200 #1.
- **KH-1-1.2B:** LFM2-1.2B frozen, 2048-token chunks, 3 archival adapters × 32 heads, 3B tokens. Running on 8 × B200 since 00:19 UTC.
- **Local RTX 3070 Ti (8 GB):** 350M streams 14.2K tok/s at 1.10 GB; 1.2B streams 13.5K tok/s at 3.04 GB. That is **1M tokens in ~75 s at constant memory**.

### 8.36 KH-1 pilot 2: the graft works at 16K; writes must be decisive (2026-10-14 ~00:40 UTC)
LFM2-350M + archival adapters, BOS per chunk, 358 phrasings, half the queries as chat turns, 300M tokens, train τ 0.5. Recall K = 1 / 8:

| Eval | 16K | 131K | 1M |
|---|---|---|---|
| compositional, inference τ 0.5 (as trained) | **1.00 / 0.74** | 0.15 / 0.16 | – |
| compositional, inference τ 0.9 | 0.23 / 0.31 | 0.28 / 0.30 | 0.20 / 0.19 |
| unseen frames × attributes, τ 0.9 | 0.20 / 0.19 | 0.15 / 0.17 | 0.05 / 0.06 |

- At the training threshold the graft recalls perfectly at 16K. Many fact writes have β in (0.5, 0.9), so at τ 0.9 they are dropped (flat but low), and at τ 0.5 the weaker writes are erased by 131K.
- **Unlike every from-scratch model, unseen phrasings score about as well as trained ones at τ 0.9** (0.20 vs 0.23). The pretrained base may solve the generalisation problem once writes are decisive.
- **Pilot 3:** pilot 2 fine-tuned with train τ 0.9 (100M tokens). Starting τ 0.9 from scratch is a dead start, because no β clears 0.9 at init, so nothing is read.
- **KH-1-1.2B** v1 (τ 0.5, 3B tokens) is training. `after_kh1v1.sh` evaluates it, then fine-tunes v2 with τ 0.9 (1B tokens).

### 8.37 KH-1 works: 1.3M-token book demo on an RTX 3070 Ti (2026-10-14 ~01:50 UTC)
**Write decisiveness and the text-write penalty decide long-range recall in the graft.** LFM2-350M + archival adapters, fine-tunes of pilot 2. Compositional phrasings, K = 1 / 8:

| Pilot | train τ / text-write L1 | inference τ | 16K | 131K | 1M | unseen frames × attributes (16K / 131K / 1M, K = 1) |
|---|---|---|---|---|---|---|
| 2 | 0.5 / 0.05 | 0.9 | 0.23 / 0.31 | 0.28 / 0.30 | 0.20 / 0.19 | 0.20 / 0.15 / 0.05 |
| 3 | 0.9 / 0.05 | 0.9 | 0.98 / 0.86 | 0.43 / 0.23 | 0.05 / 0.04 | **0.63** / 0.20 / 0 |
| 4 | 0.9 / 0.2 | 0.9 | 1.00 / 0.89 | 0.70 / 0.59 | 0.05 / 0.08 | 0.75 / 0.25 / 0 |
| **4** | 0.9 / 0.2 | **0.97** | | **1.00 / 0.80** | **0.75 / 0.70** | 0.65 / 0.55 / **0.30** |
| 5 | 0.9 / 0.6 | 0.9 | 0.98 / 0.91 | 0.98 / 0.86 | 0.38 / 0.41 | |

- **Unseen phrasings generalise with a pretrained base:** 0.63–0.75 at 16K and 0.30 at 1M. Every from-scratch model was ≤ 0.25 at 16K and ≈ 0 at 1M.

**KH-1-1.2B** (LFM2-1.2B, 2048-token chunks, 3 × 32-head adapters):
- **v1:** train τ 0.5, 3B tokens. Val ppl **13.6**. At τ 0.9:
  - 16K: 0.80 / 0.86;
  - 1M: 0.43 / 0.35, and 0.48 / 0.41 with chat-format questions.
- **v2:** v1 + τ 0.9 + text-write L1 0.2, 1B tokens. At τ 0.97:
  - plain questions: 16K 0.90 / 0.92, 131K 0.95 / 0.89, 1M 0.50 / 0.40;
  - **chat questions: 1M 0.58 / 0.51**;
  - unseen frames × attributes: 16K 0.73 / 0.55, 131K 0.70 / 0.44, 1M 0.18 / 0.18;
  - 10 sentence templates: 1M 0.43 / 0.31.
- **v3** (v2 + text-write L1 0.6) is running.

**Book demo** (`kh1_demo.py`; *War and Peace* + *Moby Dick* + *Pride and Prejudice* = **1,318,920 tokens**, 5 facts planted at token 16,814, questions asked in chat; pilot 4 = KH-1-350M at τ 0.97):
- **3/5 exact after ~1.30M tokens:** "482", "739561", "September".
- The 4th answer was "thuzo" for "thuzoka". The unseen-phrasing question missed.
- Chat greeting coherent.
- **RTX 3070 Ti (8 GB): read in 40 s (33K tok/s), peak 0.84 GB.** The H200 run gives identical answers.
- Transcripts: `kh1/demo_transcripts/`. Adapters: `kh1/ckpt/`.

### 8.38 A stronger text-write penalty does not help at τ 0.97; KH-1-1.2B v2 is the best checkpoint; Qwen3.5 graft ready (2026-10-14 ~02:20 UTC)
**Text-write L1 0.6 vs 0.2 at inference τ 0.97** (compositional phrasings, K = 1 / 8):

| Model | text-write L1 | Phrasings | 16K | 131K | 1M |
|---|---|---|---|---|---|
| 350M pilot 4 | 0.2 | comp | | 1.00 / 0.80 | **0.75 / 0.70** |
| 350M pilot 5 | 0.6 | comp | 0.48 / 0.68 | 0.65 / 0.57 | 0.30 / 0.31 |
| 350M pilot 5 | 0.6 | comp, chat | 0.40 / 0.71 | 0.53 / 0.51 | 0.48 / 0.34 |
| 350M pilot 4 | 0.2 | unseen frames × attributes | 0.65 / 0.44 | 0.55 / 0.36 | 0.30 / 0.30 |
| 350M pilot 5 | 0.6 | unseen frames × attributes | 0.28 / 0.34 | 0.30 / 0.19 | 0.15 / 0.14 |
| 1.2B v2 | 0.2 | comp | 0.90 / 0.92 | 0.95 / 0.89 | **0.50** / 0.40 |
| 1.2B v2 | 0.2 | comp, chat | 0.95 / 0.94 | 0.98 / 0.86 | **0.58 / 0.51** |
| 1.2B v3 | 0.6 | comp | 0.65 / 0.81 | 0.70 / 0.83 | 0.45 / 0.44 |
| 1.2B v3 | 0.6 | comp, chat | 0.65 / 0.89 | 0.73 / 0.77 | 0.50 / 0.48 |
| 1.2B v3 | 0.6 | unseen frames × attributes | 0.50 / 0.45 | 0.30 / 0.43 | 0.13 / 0.16 |
| 1.2B v3 | 0.6 | 10 sentence templates | 0.40 / 0.62 | 0.55 / 0.58 | 0.30 / 0.24 |

Other measurements:
- **The penalty-0.6 models do better at τ 0.9:**
  - pilot 5 comp: 0.98 / 0.91, 0.98 / 0.86, 0.38 / 0.41;
  - v3 comp: 1.00 / 0.96, 0.95 / 0.86, 0.38 / 0.40.
  - Both still fall to ~0.4 at 1M.
- **v1** (train τ 0.5) at τ 0.97: ≤ 0.36 everywhere. A τ 0.5 model needs τ ≈ 0.9 at inference.
- **v3 val ppl:** 13.75. v1 was 13.6.

Conclusions:
- **Recipe:** keep text-write L1 at 0.2 and use τ 0.97 at inference.
- **Why 0.6 fails (hypothesis):** a penalty of 0.6 pushes fact writes below 0.97 too, so the high threshold drops facts.
- **Best checkpoints:** KH-1-1.2B **v2** (`kh1/ckpt/kh1_1p2b_v2.pt`) and KH-1-350M pilot 4.

**Book demo with KH-1-1.2B v2** (RTX 3070 Ti, 1,318,920 tokens; 13.6K tok/s, peak 2.9 GB):
- **2/5 at both τ 0.97 and τ 0.95:** "thuzoka" and "September". 482, 739561 and the unseen-phrasing fact were missed.
- The 350M pilot 4 gets 3/5 on the same demo, so it stays the demo model for now.

**Qwen3.5 graft (QwenHais), ready for the 8×B200 day:**
- **Model:** Qwen3.5-4B has 32 layers (24 GDN + 8 full attention, pattern lllF), d = 2560, tied embeddings, vocab 248,320.
- **Adapters:** after layers 15/23/31, 40 heads each, **99.0M trainable parameters**.
- **Speed and memory** (one B200, bs 8 × T 4096):

  | Conv kernel | tok/s per GPU | peak memory |
  |---|---|---|
  | transformers fallback (`F.conv1d`) | 32.9K | 122 GB |
  | fla Triton `causal_conv1d` (patched in `qwen_hais.py`) | **40.0K** | **116 GB** |

  The fallback is used because the `causal_conv1d` package is not installed.
- **Patch check** on Qwen3.5-0.8B, 16 × 2048 val tokens: LM loss 2.84877 vs 2.84836, argmax agreement 98.6% (bf16 noise).
- **Expected throughput:** ~320K tok/s on 8 GPUs, i.e. 1B tokens ≈ 52 min.
- **Data:** `~/data/fwe_qwen35` (2B train + 20M val tokens, uint32) on H200 #1.
- **Node:** `env_qwen` (torch 2.13, transformers 5.19, fla 0.5.2) and the 4B / 0.8B weights are on the B200 node, if its disk survives the stop.
- **Overnight pilot on H200 #1** (`chain_qwen08.sh`): Qwen3.5-0.8B, 1024-token chunks, 16 heads after 11/17/23. Stage 1: τ 0.5, 300M tokens. Stage 2: τ 0.9, text-write L1 0.2, 100M tokens. Evals at τ 0.97.
- **Training fix:** sampled-CE chunk = 2^29 / vocab positions. With 8192 positions, Qwen's 248K vocab made 7 GB fp32 logit chunks and OOM retries.
- **Chat format:** Qwen3.5's template appends `<think>\n\n</think>\n\n` to the assistant turn by default. Training chat queries include it, which matches native non-thinking chat.

### 8.39 Controls: no-memory recall is 0, the memory lowers LM perplexity by 9% (2026-10-14 ~02:30 UTC, RTX 3070 Ti)
- **Memory off** = `--no_memory` (ReZero gates zeroed, i.e. plain chunked LFM2):
  - KH-1-350M 16K recall 0.000 / 0.000 (n = 40);
  - book demo 0/5, with invented answers ("12", "Kribrisen@wifi", "MORA", "February 13th", "96.5 MHz");
  - transcript `kh1/demo_transcripts/book_demo_pilot4_350m_NO_MEMORY_control_rtx3070ti.json`.
- **Val perplexity** (`eval_lm.py` ppl: 2M-token stream, loss on the second 1M, carried state):

  | Model | with memory | memory off |
  |---|---|---|
  | KH-1-1.2B v2 | **13.64** | 15.00 |
  | KH-1-350M pilot 4 | **20.52** | 22.64 |

  - The memory recovers cross-chunk context: −9%.
  - The local 13.64 matches the node's 13.639 (reproducibility check).
- **Exact scoring:** `kh1_demo.py` now requires the reply's first word to equal the value. The old substring test would have counted "7395614" as correct. The 3/5 and 2/5 transcripts were rechecked and are exact.

### 8.40 Why grafts fade at 1M: saturated text writes erase facts; fact and text writes are not separated (2026-10-14 ~02:45 UTC)
Measurement: `write_stats.py` (now graft-aware), model at inference τ 0.97, 3 streams × 1M FineWeb-Edu tokens. The fact is read in context at the end of an 8K prefix.

| Measure | KH-1-350M (pilot 4) | KH-1-1.2B (v2) |
|---|---|---|
| Text tokens written per head at τ 0.97 | 0.131% | 0.119% |
| Text tokens written per head at β > 0.99 | 0.009% | 0.014% |
| Heads that wrote the fact (β > 0.5) | 29.7 / 48 | 44 / 96 |
| Fact write strength (median over heads of max β) | 0.977, barely above τ | 0, i.e. most heads do not write it |
| Retention at 1M, model τ 0.97 (median over written heads) | 0.069; 1% of heads > 0.9 | 0.017; 0% > 0.9 |
| Retention at 1M, simulated τ 0.99 | **0.46**; 24% of heads > 0.9 | 0.15; 8% > 0.9 |

- **What erases facts:** almost every surviving text write is saturated (β ∈ (0.97, 0.99]). That is ~1,300 writes per head per 1M tokens, against a 64-dimensional key space.
- **Why τ 0.99 at inference is not the fix:** it would cut text writes 9–14× and leave 6–7× more of the fact, but it would also drop most fact writes, which sit at β ≈ 0.97–0.98.
- **Why recall survives at all:** recall at 1M (0.5–0.75) is far above per-head retention, so the readout pools partial traces across heads.
- **Lever:** separate fact writes (β → 1) from text writes (β < τ) by training at the inference threshold: τ curriculum 0.5 → 0.9 → 0.97 → 0.99.
  - Pilot 6 (τ 0.97 from pilot 4) is running.
  - Pilot 7 (τ 0.99) follows.
  - This also explains §8.38: a stronger text-write L1 lowers *all* β, fact writes included, so the high threshold drops facts.

### 8.41 Training at the inference threshold shifts *both* fact and text writes up (pilot 6, 2026-10-14 ~03:00 UTC)
Pilot 6 = pilot 4 + 100M tokens trained at τ 0.97 (text-write L1 0.2). `write_stats.py`, 3 × 1M tokens:

| Measure | pilot 4, model τ 0.97 | pilot 6, model τ 0.97 | pilot 6, model τ 0.99 |
|---|---|---|---|
| Fact write strength (median over heads of max β) | 0.977 | **0.992** | 0.992 |
| Text tokens written per head | 0.131% | **2.29%** | 0.177% |
| Text writes per head with β > 0.99 | 0.009% | 0.141% | 0.164% |
| Retention at 1M (median, written heads) | 0.069 | 0.002 | 0.077 |
| Retention at 1M, simulated τ 0.995 | — | 1.0 (56% of heads > 0.9) | 0.72 (43% > 0.9) |

- **The threshold moves both distributions up.** Fact writes move above the new threshold, but so do text writes: 17× more text writes at τ 0.97.
- **Retention is a separation problem, not a threshold problem.** Whatever τ is used in training, the facts sit ~1 logit above it and the text writes just below or around it.
- **Recall at 16K improved** (τ 0.97: comp 1.00 / 0.91, chat 1.00 / 0.96, unseen frames × attributes 0.80 / 0.48). 131K / 1M results pending.
- **Why the L1 cannot fix this:** its β-gradient vanishes on saturated writes, β(1 − β) ≈ 0.02.
- **Next:** a logit-space penalty on text writes (`--text_write_bce`, pilot 8) that pushes saturated text logits down.
- **Ops note:** pilots 7/8 first crashed with OOM, because the Qwen-0.8B pilot was using 85 GB on the shared H200. The Qwen pilot is paused at its step-1500 checkpoint and resumes after pilots 7/8.

### 8.42 What the graft's memory writes in ordinary text: rare names and numbers (shortcut), 2026-10-14 ~03:15 UTC
`probe_text_writes.py`, KH-1-350M pilot 4 at τ 0.97, 131K tokens:

| Text | Tokens where any head writes | Writing heads per token (of 48) |
|---|---|---|
| FineWeb-Edu val | 6.6% | 0.096 |
| *War and Peace* | 6.9% | 0.123 |

Most-written tokens:
- **FineWeb-Edu:** rare proper nouns (Wolbachia, Benzer, Pinkas Hakehillot Polin, Horowicz), years and digits, and the period or comma closing such phrases.
- ***War and Peace*:** fragments of Russian names (ó, sh, ov, ú, ís, kh, hov, ski).

**Diagnosis:**
- Training facts use made-up syllable names (Velthu, Kribrisen) and mostly numeric values. The write gate learned "unusual name / number → write", not "attribute–value statement → write".
- Natural text full of exotic names, such as a Russian novel, therefore triggers hundreds of writes per head per 100K tokens. These erase the planted facts (§8.40).

**Fixes under test (H200 #1):**
- **Pilot 8:** τ 0.97 + `--text_write_bce 0.1` (logit-space penalty), L1 off.
- **Pilot 9:** pilot 8 + `--natural_names 0.5`. Half of the facts are about real-looking people ("Youssef Hernandez told me that their gym day is Monday."), so name rarity stops predicting a fact.

### 8.43 Threshold curriculum, text-write BCE and per-head thresholds do not beat pilot 4 on the book demo (2026-10-14 ~03:30 UTC)
**Book demo** (5 facts, 1.3M tokens, RTX 3070 Ti, exact match; n = 5, so it only flags large differences):

| Model | Inference τ | Score |
|---|---|---|
| pilot 4 (train τ 0.9, L1 0.2) | 0.97 | **3/5** |
| pilot 4 | 0.97, half the heads at 0.995 | 0/5 |
| pilot 6 (+ train τ 0.97) | 0.97 | 0/5 |
| pilot 6 | 0.99 | **3/5** |
| pilot 7 (+ train τ 0.99) | 0.99 | 1/5 |
| pilot 7 | 0.995 | **3/5** |
| pilot 8 (train τ 0.97 + text-write BCE 0.1, no L1) | 0.97 | 2/5 |
| pilot 8 | 0.99 | 0/5 |
| pilot 8 | 0.97, half the heads at 0.995 | 1/5 |

- Every 3/5 is the same three facts. The wifi password always comes back as "thuzo" or similar, and the unseen phrasing always misses.

**Write statistics** at the training threshold (`write_stats.py`, 3 × 1M tokens):

| Model | Text tokens written per head | Fact write strength (median max β) | Retention at 1M (median, written heads) |
|---|---|---|---|
| pilot 6 | 2.29% | 0.992 | 0.002 |
| pilot 7 | 0.444% | 0.996 | 0.090 |
| pilot 8 | 0.446% | 0.984 | 0.027 |
| pilot 4 | 0.131% | 0.977 | 0.069 |

- Pilot 8 at simulated τ 0.995: 83% of written heads keep > 0.9 at 1M, but the fact writes are below that threshold.

Conclusions:
- **The threshold curriculum moves the operating point; it does not separate fact writes from text writes.**
- **The logit-space BCE cuts text writes 5×** relative to plain τ 0.97 training, but not below pilot 4.
- **Per-head mixed thresholds at inference break the readout** (3/5 → 0/5). The readout is tuned to the state statistics of uniform thresholds; mixed thresholds would have to be trained.
- **Remaining lever: what the gate learns to detect** (§8.42). Pilot 9 (natural names) is training. FineWeb 1M evals (n = 40) of pilots 6–8 are pending.

### 8.44 Training at τ 0.97 and inferring at τ 0.99 makes KH-1-350M nearly flat to 1M: 0.925–0.95 (pilot 6), 2026-10-14 ~04:00 UTC
Pilot 6 = pilot 4 + 100M tokens at train τ 0.97 (L1 0.2). FineWeb-Edu, n = 40, K = 1 / 8:

| Model | Inference τ | Questions | 16K | 131K | 1M |
|---|---|---|---|---|---|
| pilot 4 | 0.97 | plain | | 1.00 / 0.80 | 0.75 / 0.70 |
| **pilot 6** | **0.99** | plain | 0.95 / 0.84 | 1.00 / 0.91 | **0.925 / 0.77** |
| **pilot 6** | **0.99** | chat | 0.975 / 0.91 | 0.975 / 0.81 | **0.95 / 0.78** |
| pilot 4 | 0.97 | unseen frames × attributes | 0.65 / 0.44 | 0.55 / 0.36 | 0.30 / 0.30 |
| **pilot 6** | **0.99** | unseen frames × attributes | 0.80 / 0.45 | 0.80 / 0.41 | **0.575 / 0.325** |
| pilot 6 | 0.97 (its training τ) | plain | 1.00 / 0.91 | 0.975 / 0.88 | 0.175 / 0.125 |

- **Rule: infer one step above the training threshold.**
  - At its own training τ, pilot 6 collapses at 1M: 2.3% of text tokens write (§8.41).
  - At τ 0.99, K = 1 recall is nearly flat from 16K to 1M (0.95 → 1.00 → 0.925), i.e. close to the distance invariance of the from-scratch models.
- **Unseen phrasings at 1M nearly double:** 0.30 → 0.575.
- **The 5-fact book demo does not separate them:** 3/5 for both (§8.43).
- **Book-haystack eval:** facts spliced at random into 1M-token windows of the three novels, chat questions, n = 40, K = 1. Pilot 4 at τ 0.97 scores **0.775**; pilot 6 at τ 0.99 scores **0.825**. The novels are harder than FineWeb (0.95 for pilot 6) because they are dense with names. Pilot 6 is now the KH-1-350M demo checkpoint (`kh1/ckpt/kh1_350m_pilot6.pt`, τ 0.99).
- **Applied to KH-1-Q:** Qwen3.5-4B stage 2 (τ 0.9, 0.5B tokens) → stage 3 (τ 0.97, 0.5B tokens) → evals at τ 0.99 / 0.995, via `go_qwen4b_s3.sh`.

### 8.45 One threshold step is enough; the BCE penalty does not help (pilots 7–9, 2026-10-14 ~04:15 UTC)
FineWeb-Edu at 1M tokens, K = 1 / 8 (n = 40):

| Model | Inference τ | Plain | Chat | Unseen frames × attributes |
|---|---|---|---|---|
| **pilot 6** (train 0.97) | 0.99 | **0.925 / 0.77** | **0.95 / 0.78** | **0.575 / 0.325** |
| pilot 7 (+ train 0.99) | 0.995 | 0.925 / 0.73 | 0.825 / 0.74 | 0.375 / 0.29 |
| pilot 7 | 0.99 | 0.30 / 0.28 | 0.275 / 0.29 | 0.10 / 0.15 |
| pilot 8 (train 0.97 + BCE 0.1, no L1) | 0.97 | 0.625 / 0.49 | 0.70 / 0.49 | 0.20 / 0.19 |
| pilot 8 | 0.99 | 0.325 / 0.375 | 0.425 / 0.37 | 0.175 / 0.18 |
| pilot 9 (pilot 8 + natural names 0.5) | 0.97 | 0.55 / 0.45 | 0.70 / 0.51 | 0.275 / 0.21 |
| pilot 9 | 0.99 | 0.425 / 0.31 | 0.50 / 0.36 | 0.125 / 0.14 |

- **A second step (0.97 → 0.99) adds nothing.**
- **The BCE penalty (L1 off) lowers fact writes along with text writes,** so no threshold recovers pilot 6's level.
- **Natural-name facts (pilot 9) do not change made-up-name recall** relative to pilot 8. The book demo dropped to 0–1/5 (n = 5). Facts about real names were not evaluated separately.
- **Recipe of record:** train τ 0.5 → 0.9 → 0.97 with text-write L1 0.2, infer at 0.99.
- **Applied to:** KH-1-Q stage 3 and KH-1-1.2B v4 (both on the node).

### 8.46 KH-1-Q (Qwen3.5-4B) first look; the Qwen3.5-4B chat template defaults to *thinking* (2026-10-14 ~04:45 UTC)
**Training run** (`go_qwen4b.sh` → `go_qwen4b_s3.sh` on the 8×B200 node; adapters after 15/23/31, 40 heads, 99M trainable):

| Stage | Train τ | Text-write L1 | Tokens | Recall accuracy (train) |
|---|---|---|---|---|
| 1 | 0.5 | 0.05 | 1.5B | 0.96 by 330M tokens; 0.98 at the end |
| 2 | 0.9 | 0.2 | 0.5B | — |
| 3 | 0.97 | 0.2 | 0.5B | (running) |

- Throughput: 280–295K tok/s.

**Local RTX 3070 Ti, Qwen3.5-4B in nf4** (bitsandbytes, `--quant nf4`, `env_qwen_local`):
- Memory: 3.53 GB allocated, peak 3.86–4.07 GB.
- Speed: ~4.2K tok/s after Triton autotuning. 1.28M tokens are read in 320 s.
- Stage-2 adapters at τ 0.97 on the book demo: **1/5** (September). Near misses: 412 for 482, 7395610 for 739561, "thunevoro" for "thuzoka".
- [bf16 demo on the node pending, to separate the nf4 effect.]

**Template trap:**
- `apply_chat_template(add_generation_prompt=True)` ends with:
  - Qwen3.5-4B: `<think>\n`, an open thinking block;
  - Qwen3.5-0.8B: the empty block `<think>\n\n</think>\n\n`.
- KH-1-Q chat queries were therefore trained with the answer prefix *inside an open think block*.
- Recall evals and `kh1_demo.py` use the same format, so they are consistent.
- Free chat thinks out loud ("Okay, the user is asking…"). The native nf4 model does the same.
- **Next run:** pass `enable_thinking=False` in `FactFactory` and the demos for Qwen, so recall works in non-thinking chat.

### 8.47 KH-1-Q (Qwen3.5-4B + archival memory): 0.925 at 1M after stage 1; the τ 0.97 stage collapses it (2026-10-14 ~05:35 UTC)
FineWeb-Edu, n = 40, K = 1 / 8. "Train τ → inference τ" is given per stage:

| Stage (train τ → inference τ) | Questions | 16K | 131K | 1M |
|---|---|---|---|---|
| **s1** (0.5 → 0.9) | plain | 0.95 / 0.70 | 0.975 / 0.79 | **0.925 / 0.69** |
| **s2** (0.9 → 0.97) | plain | 1.00 / 0.89 | 1.00 / 0.83 | 0.85 / 0.66 |
| **s2** (0.9 → 0.97) | chat | 1.00 / 0.91 | 1.00 / 0.75 | **0.925 / 0.66** |
| **s2** (0.9 → 0.97) | unseen frames × attributes | 0.80 / 0.57 | 0.875 / 0.47 | **0.575 / 0.45** |
| s3 (0.97 → 0.99) | plain | 1.00 / 0.92 | 0.825 / 0.61 | 0.25 / 0.24 |
| s3 (0.97 → 0.995) | plain | 0.925 / 0.78 | 0.80 / 0.57 | 0.40 / 0.33 |
| s3 (0.97 → 0.99) | chat | 0.975 / 0.96 | 0.90 / 0.57 | 0.375 / 0.27 |
| s3 (0.97 → 0.99) | wholly unseen sentence templates | **0.675** / 0.275 | 0.50 / 0.22 | 0.10 / 0.10 |
| s3 (0.97 → 0.99) | 10 sentence templates | 0.95 / 0.66 | 0.75 / 0.50 | 0.40 / 0.24 |
| s3, memory off | plain | 0.00 / 0.006 | 0.00 / 0.025 | 0.00 / 0.006 |

- **The 4B base helps a lot.**
  - Stage 1 alone (τ 0.5 → 0.9) reaches 0.925 at 1M. LFM2-1.2B v1 reached 0.43 at the same stage.
  - Wholly unseen sentence templates reach 0.675 at 16K. LFM2 is ≤ 0.08.
- **The τ 0.97 stage hurts the 4B,** the opposite of pilot 6.
  - 1M falls to 0.25–0.45, and τ 0.995 does not rescue it.
  - Hypothesis: the 4B's text writes shift further with the training threshold, so one inference step is not enough. A sweep of s1 / s2 at more thresholds (`go_q4b_sweep.sh`) is running.
  - The threshold step must be tuned per base model, and checked at 1M before it is used.
- **Val ppl:** 9.81 with memory, 10.29 off (−4.7%).
- **Book demo** (1.28M Qwen tokens; peak 9.3 GB in bf16 on a B200):

  | Stage | Inference τ | Score | Correct facts |
  |---|---|---|---|
  | s2 | 0.97 | 2/5 | 739561, September |
  | s3 | 0.99 | 0/5 | |
  | s3 | 0.995 | 2/5 | |
  | s3, memory off | 0.99 | 0/5 | answers "12345", "not publicly available…" |

  - Near misses: 412 / 432 for 482; "thuzo…" prefixes; the unseen-phrasing answer mixes digits of 739561.
- **Local RTX 3070 Ti, nf4:** s2 at τ 0.97 scores 1/5 (§8.46).

### 8.48 The threshold step works for KH-1-1.2B too (v4); Qwen3.5-0.8B pilot stage 2 (2026-10-14 ~06:00 UTC)
**KH-1-1.2B v4** = v2 + 300M tokens at train τ 0.97 (L1 0.2), ~6 min on 8×B200. FineWeb-Edu, n = 40, K = 1 / 8:

| Model | Inference τ | Questions | 16K | 131K | 1M |
|---|---|---|---|---|---|
| v2 | 0.97 | chat | 0.95 / 0.94 | 0.98 / 0.86 | 0.58 / 0.51 |
| **v4** | 0.99 | chat | 0.95 / 0.91 | 0.95 / 0.86 | **0.85 / 0.71** |
| v4 | 0.99 | plain | 0.875 / 0.88 | 0.95 / 0.90 | 0.80 / 0.66 |
| v4 | 0.995 | plain | 0.525 / 0.69 | 0.65 / 0.79 | 0.90 / 0.725 |
| v4 | 0.995 | chat | 0.60 / 0.78 | 0.725 / 0.72 | 0.85 / 0.81 |
| v2 | 0.97 | unseen frames × attributes | 0.73 / 0.55 | 0.70 / 0.44 | 0.18 / 0.18 |
| **v4** | 0.99 | unseen frames × attributes | 0.875 / 0.59 | 0.825 / 0.475 | **0.45 / 0.39** |
| v4 | 0.99 | 10 sentence templates | 0.825 / 0.675 | 0.925 / 0.74 | 0.675 / 0.425 |

- v4 val ppl: 13.66. v2 was 13.64.
- The step from §8.44 transfers from 350M to 1.2B within LFM2: 1M chat recall 0.58 → 0.85, unseen phrasings 0.18 → 0.45.
- It does not transfer to Qwen3.5-4B (§8.47).

**Qwen3.5-0.8B pilot (H200 #1), stage 2** (τ 0.9 → 0.97, 1024-token chunks, 16 heads):

| Questions | 16K | 131K | 1M |
|---|---|---|---|
| plain | 0.775 / 0.64 | 0.75 / 0.69 | 0.825 / 0.59 |
| chat | 0.80 / 0.725 | 0.825 / 0.66 | 0.775 / 0.61 |
| unseen frames × attributes | 0.50 / 0.41 | 0.525 / 0.375 | 0.60 / 0.37 |
| **wholly unseen sentence templates** | 0.325 / 0.14 | 0.30 / 0.18 | **0.375 / 0.14** |

- Recall is flat across distance but at a lower level.
- Unseen sentence templates survive to 1M, which no LFM2 model does.
- Stage 3 (τ 0.97) is running.

### 8.49 KH-1-Q stage 2 at τ 0.98: 0.975 at 1M (chat) and 3/5 on the book demo, the best of the night (2026-10-14 ~06:00 UTC)
Inference-threshold sweep, 1M tokens, chat questions, n = 40, K = 1 / 8:

| Stage | τ 0.9 | τ 0.95 | τ 0.97 | τ 0.98 | τ 0.99 |
|---|---|---|---|---|---|
| s1 (train 0.5) | **0.925 / 0.76** | 0.875 / 0.68 | 0.70 / 0.46 | | |
| s2 (train 0.9) | | | 0.925 / 0.66 | **0.975 / 0.70** | 0.85 / 0.575 |

Other measurements:
- s1, unseen frames × attributes at τ 0.95: 0.45 / 0.375.
- **Book demo** (bf16, B200):

  | Stage | τ | Score |
  |---|---|---|
  | **s2** | **0.98** | **3/5** |
  | s2 | 0.97 | 2/5 |
  | s1 | 0.9 | 0/5 |
  | s1 | 0.95 | 1/5 |
  | s1 | 0.97 | 0/5 |

- **The 4B's best inference threshold for a τ 0.9-trained model is 0.98**, not 0.97. The LFM2 models are best one full step up.
- **Training at τ 0.97 (s3) is not needed for the 4B** and hurts it (§8.47).
- **Best checkpoints:**
  - `kh1/ckpt/kh1q_4b_s2.pt` (Qwen3.5-4B, use at τ 0.98);
  - `kh1/ckpt/kh1_1p2b_v4.pt` (use at τ 0.99);
  - `kh1/ckpt/kh1_350m_pilot6.pt` (use at τ 0.99).
- **Copied and verified locally** (`results/b200_node/`, all finite):
  - checkpoints: s1 / s2 / s3 and v4;
  - all eval JSONs and logs;
  - book-demo transcripts.

### 8.50 KH-1-1.2B v4 answers 4/5 on the book demo, the best demo result (2026-10-14 ~06:15 UTC, RTX 3070 Ti)
- **KH-1-1.2B v4 at τ 0.99:** 4/5 exact — 482, **thuzoka** (first full recall of the 3-token made-up word), 739561, September.
  - The unseen phrasing gets "2879" for 287.
  - Peak 2.81 GB; 1,318,920 tokens read in 96 s.
  - Transcript: `kh1/demo_transcripts/book_demo_kh1_1p2b_v4_tau0.99_rtx3070ti.json`.
- **KH-1-Q s2 in nf4 at τ 0.98 on the 3070 Ti:** 2/5, against 3/5 in bf16 (482 → 432). nf4 costs a little recall.
- **v4 is now the recommended demo model** (`kh1/DEMO_VIDEO.md`). Its book-haystack n = 40 eval is running.

### 8.51 Book-haystack recall for KH-1-1.2B v4: 0.90 (2026-10-14 ~07:15 UTC, RTX 3070 Ti)
- **Setup:** the same facts as the FineWeb-Edu evals, spliced into random 1M-token windows of the three novels (`data_local/books_lfm2`). Chat questions, K = 1, n = 40, inference τ 0.99.
- **Result:** KH-1-1.2B v4 scores **0.900** (`results/b200_node/runs_kh1/kh1_1p2b_v4/eval_lm_books_comp_chat1_tau0.99.json`).
- **Comparison:** KH-1-350M scores 0.825 (pilot 6, τ 0.99) and 0.775 (pilot 4, τ 0.97).

### 8.52 Paper audit and new evidence (2026-10-14 ~07:35 UTC)
**Audit.** Three independent audits rebuilt every number in the paper from the raw files under `results/`. Their findings, applied to `paper_v2/arxiv/main.tex`:

| Claim | Problem | Correction |
|---|---|---|
| "8/8 far-drill runs flat" | Overstated | Only the 6 runs with GDN cap ≤ 512 are flat (one-sided Fisher on K = 1, all p ≥ 0.3). Cap 8K drops 27/40 → 16/40 (p = 0.012). |
| Synthetic 16M-token result | No evidence anywhere | Cut |
| Synthetic recipe counts | Mixed criteria | Restated under one criterion (≥ 0.98 at 1M for every K): 1/3, 3/6, 8/12, 9/9, 3/3. |
| Far-drill gap ≥ 712 tokens | Ignores the inserted fact tokens | ≥ ~530 tokens (median ~780) |
| Trap 3 | Conflated two runs; "capping ages avoids the collapse" was wrong | Rewritten; cause marked unresolved (a guarded run did not collapse) |
| Trap 1 "every model ≈ 0" | Not all | "Most" |
| From-scratch unseen phrasings at 16K | Wrong evaluation cited | ≤ 0.18, not ≤ 0.25 |
| Qwen "0.675 where LFM2 ≈ 0" | LFM2 is 0.43–0.55 at 16K | Contrast restated |
| Write-ratio "17×" | Wrong pair of numbers | ~13× (2.3% vs 0.18%) |
| 0.85 GB / 33K tok/s | Rounded up | 0.84 GB / 32.5K tok/s |
| "n = 40 per cell" | True only for K = 1 | K = 8 uses 20 streams × 8 facts |

- **Also corrected:** several model and threshold attributions (perplexity source models, nf4 at τ 0.98, the memory-off control model), decoy-rate conditions, the v_norm bound wording, App. A/B qualifiers, and the 117M throughput (260–300K tok/s).
- **Files pulled from H200 #1** to verify log-only rows: `results/h200_92/lm/runs_abl`, `runs_comp`, `runs_lm*`, and the LM v5 stream drift (207 → 9,772 state norm, CE 4.27 → 7.32).

**New evidence saved:**
- **Same-model memory-off control for KH-1-1.2B v4:** 0/5 ("12", "KRIBRISEN-WIFI", "123.456 MHz"); `kh1/demo_transcripts/book_demo_kh1_1p2b_v4_NO_MEMORY_control_rtx3070ti.json`.
- **Interactive transcripts:**
  - scripted order: 4/5, peak 2.81 GB, state 1.6 MB;
  - other order: 4826 and 4785619 for 482 and thuzoka;
  - both in `results/demo_logs/`.
- **`probe_text_writes` outputs:** 6.59% (FineWeb-Edu) and 6.90% (*War and Peace*) of tokens trigger an archival write; `results/probe_text_writes/`.
- **BOS-per-chunk check:** LFM2-350M loss 3.065 with BOS vs 5.163 without (1,024-token chunks); `results/checks/lfm2_350m_bos_per_chunk.json`.
- **Running:** fresh-stream confirmation (eval seed 4321) of the 1M headline numbers for Qwen s2 τ 0.98, pilot 6 τ 0.99 and v4 τ 0.99 (`runs_confirm/` on H200 #1), and seeds 1–2 of the KH-1-350M recipe (`seed_350m.sh`).

### 8.53 Seed replicas and fresh-stream checks of the released grafts (2026-10-14 ~09:40 UTC)
**Fresh streams.** The inference thresholds of the released models were chosen on the evaluation streams, so the 1M headline cells were re-run on 40 new streams (`eval_lm --eval_seed 4321`, `poc_m3/lm/confirm_fresh.sh`). 1M, chat questions, K = 1 / K = 8:

| Model | Evaluation streams | Fresh streams |
|---|---|---|
| KH-1-350M (pilot 6, τ 0.99) | 0.95 / 0.781 | **0.90 / 0.75** |
| KH-1-1.2B v4 (τ 0.99) | 0.85 / 0.706 | **0.75 / 0.70** |
| KH-1-Q-4B s2 (τ 0.98, picked from 0.97 / 0.98 / 0.99) | 0.975 / 0.70 | **0.90 / 0.70** |

K = 1 drops by 0.05–0.10; K = 8 by at most 0.03. Files: `results/h200_92/runs_confirm/{v4,q4b_s2}/`, `results/h200_92/runs_lfm2/lfm2_350m_pilot6/eval_lm_fresh4321_comp_chat1_tau0.99.json`.

**Seeds.** The full four-stage KH-1-350M recipe (`poc_m3/lm/seed_350m.sh`: τ 0.5 / L1 0.05 / 300M → 0.9 / 0.05 / 100M → 0.9 / 0.2 / 100M → 0.97 / 0.2 / 100M, run at 0.99) was retrained with seeds 1 and 2; seed 0 is the pilot 2 → 3 → 4 → 6 lineage. K = 1, 40 streams (`python paper_v2/seed_table.py`):

| Cell | Seed 0 | Seed 1 | Seed 2 | Mean ± s.d. |
|---|---|---|---|---|
| chat, 16K | 0.975 | 0.975 | 0.95 | 0.967 ± 0.014 |
| chat, 131K | 0.975 | 0.90 | 0.925 | 0.933 ± 0.038 |
| **chat, 1M** | **0.95** | **0.675** | **0.575** | **0.733 ± 0.194** |
| chat, 1M, K = 8 | 0.781 | 0.644 | 0.581 | 0.669 ± 0.102 |
| plain, 1M | 0.925 | 0.825 | 0.70 | 0.817 ± 0.113 |
| unseen, 16K | 0.80 | 0.70 | 0.55 | 0.683 ± 0.126 |
| unseen, 1M | 0.575 | 0.375 | 0.45 | 0.467 ± 0.101 |
| books, 1M | 0.825 | 0.65 | 0.80 | 0.758 ± 0.095 |
| one stage earlier (0.9 → 0.97), chat, 1M | 0.875 | 0.55 | 0.40 | 0.608 ± 0.243 |

- **Reading:** the τ 0.97 step raises 1M recall in every seed (+0.075 / +0.125 / +0.175) and short range is stable, but the 1M level depends on the seed. The released 350M model is the best of three.
- **Perplexity** (second 1M of a 2M-token stream): seed 1 20.692, seed 2 20.459 (seed 0 / pilot 4: 20.52).
- **Paper:** Table 7 (`tab:seeds`), abstract, contribution 4, §6.3 and Limitations now quote the fresh numbers and the seed spread.

### 8.54 LFM2.5-8B-A1B stage-1 pilot: the memory works on the MoE base but needs more tokens (2026-10-14 ~10:45 UTC, H200 #1)
- **Setup:** `poc_m3/lm/pilot_lfm25.sh`. Frozen LFM2.5-8B-A1B (MoE, 24 layers, 128K vocab; experts via `grouped_mm`), adapters after layers 11/17/23 (32 heads), 2,048-token chunks, τ 0.5, text-write L1 0.05, 150M tokens, bs 8 × T 4096 on one H200 at ~46K tok/s (55 min). Data: `~/data/fwe_lfm25` (LFM2.5 tokenizer, 300M train / 20M val tokens).
- **Training curve:** ans_acc flat at ~0.33 to 92M tokens, then 0.41 (105M), 0.46 (144M), 0.50 (150M). Qwen3.5-4B stage 1 looked the same (0.33 at 98M, 0.60 at 190M, 1.00 at 374M), so the onset is normal for a larger base.
- **Evaluation** (chat questions, 40 streams, K = 1 / K = 8): at the training threshold 0.5, 16K recall is **0.35 / 0.256** (identical at 0.7); at 0.9 it is 0.075 / 0.044 (16K) and 0.05 / 0.094 (131K); memory off 0.00 / 0.006. The memory carries facts across chunks on LFM2.5, but writes are not yet decisive enough for a 0.9 threshold. Files: `results/h200_92/runs_lfm25/`.
- **LFM2.5 line-up checked on Hugging Face:** text models at 230M, 350M, 1.2B (65K vocab, like LFM2), 2.6B (dense `lfm2`, 30 layers, 128K vocab), 8B-A1B and 24B-A2B (MoE, 128K vocab). There is no 4B. The 2.6B, 8B-A1B and 24B-A2B share one tokenizer, so `fwe_lfm25` serves all three.
- **Next:** `poc_m3/lm/go_lfm25.sh` on 8×B200: 8B-A1B, then 2.6B, then 24B-A2B; stage 1 1B tokens, stage 2 0.4B, stage 3 0.2B; evals on development and fresh streams.

### 8.55 Back on one H200: LFM2.5-8B-A1B full curriculum; LFM2.5-350M is a poor base (2026-10-14 ~12:15 UTC)
- **Compute:** the B200 and an 8×H100 node were not available; an RTX PRO 6000 box was tried briefly (fla gated-delta-rule gradient check passes on sm_120 with Triton 3.7.1: `GDN_GRAD_OK`) and stopped. Work continues on H200 #1 (restarted, disk intact).
- **Running:** `poc_m3/lm/go_l25_8b_h200.sh`. LFM2.5-8B-A1B graft, continuing the 150M-token pilot: stage 1b τ 0.5 / L1 0.05 / 300M tokens → stage 2 τ 0.9 / 0.2 / 200M → stage 3 τ 0.97 / 0.2 / 100M; evals at 0.99 (stage 3) and 0.98 (stage 2) on development and fresh streams, memory off, perplexity. Stage 1b starts at ans_acc ~0.50 (where the pilot stopped), 48K tok/s.
- **transformers 5 fix:** dense `Lfm2Model` names its rotary embedding `rotary_emb` in transformers 5.x (`Lfm2MoeModel` keeps `pos_emb`); `lfm2_hais.py` now takes either. Check: with the gates at zero the graft reproduces the native loss exactly (LFM2.5-350M 6.4781 = 6.4781; LFM2-350M 3.6367 under transformers 5.19 and 3.6363 under 4.57).
- **LFM2.5-350M is badly calibrated on plain text.** Same architecture and identical token ids as LFM2-350M (checked on 14,781 tokens of *Pride and Prejudice*), and it chats normally ("Paris"), but its loss on book text is **6.29** against **3.40** for LFM2-350M (4,096 tokens, BOS, both transformers versions, tied embeddings verified). LFM2.5-1.2B-Instruct is healthy (**2.88** vs 2.98 for LFM2-1.2B). We do not use LFM2.5-350M as a memory base: adapters could lower its loss by recalibrating rather than remembering. Files: `results/h200_92/runs_lfm25/feasibility_350m.json`, `feasibility_lfm2_350m_*.json`.

### 8.56 Why seeds differ at 1M: quiet fact heads (2026-10-14 ~12:50 UTC, RTX 3070 Ti)
> **Correction (§8.60):** the cross-seed ranking below (3 models) did not hold over 10 models (Spearman ρ = 0.02 / 0.11). Only the within-model relation (text-write rate vs retention over heads) stands.

`write_stats.py` (now with per-head outputs), the three KH-1-350M seeds at τ 0.99, the same 6 streams of 1M FineWeb-Edu tokens, one planted fact each (`results/seed_analysis/write_stats_n6_seed*.json`):

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1M recall, chat (n = 40) | **0.95** | 0.675 | 0.575 |
| Heads storing the fact, per stream (of 48) | 32.5 | 26.7 | 28.2 |
| Quiet fact heads: store the fact in ≥ 50% of streams, write < 0.01% of text tokens | **10** | 6 | 4 |
| Fact heads still holding the fact at 1M (retention > 0.5) | **5** | 1 | 0 |
| Median retention at 1M, written heads | **0.063** | 0.006 | 0.014 |
| Text-write rate seen by fact heads | 0.21% | 0.57% | 0.19% |
| Correlation of a fact head's text-write rate with its 1M retention | −0.51 | −0.41 | −0.47 |

- **Reading:** a fact survives 1M tokens only in heads that almost never write ordinary text (median ~13 text writes per 1M tokens among the heads that keep it). How many such heads a run grows is left to chance, and recall ranks exactly with their number. The overall text-write rate does not explain it (seed 2 writes less text than seed 0); where the text writes land does.
- **KH-2 candidates** (`poc_m3/lm/kh2_stage4.sh`, queued after the LFM2.5-8B chain): redo only stage 4 of seeds 1 and 2 with (a) two-tier thresholds (half of each layer's heads train at 0.995 and run at 0.998) or (b) quiet heads (5× text-write L1 on half of each layer's heads). Local smoke tests at batch 1: quiet heads keep baseline gradient norms (4.5–9.7 vs 6.7–8.2); two-tier starts with large gradient norms (190–821), presumably because half the heads are suddenly silenced; clipping bounds the update.

### 8.57 LFM2.5-8B-A1B learns next-chunk recall but does not keep facts: 2 quiet fact heads of 96 (2026-10-14 ~16:00 UTC, H200 #1)
> **Correction (§8.60):** "across four models, 1M recall tracks the number of quiet fact heads" is withdrawn; see §8.60.

- **Chain** (`go_l25_8b_h200.sh`): stage 1b done 13:49 (training ans_acc 0.50 → 0.95–0.97), stage 2 (τ 0.9, L1 0.2, 200M) done 14:58, stage 3 (τ 0.97, L1 0.2, 100M) done 15:57; 450M + 200M + 100M = 750M tokens in total (the 1.2B and Qwen grafts used 4.3B and 2B), batch 32K tokens per step (one GPU; the B200 runs used 131K–262K).
- **Stage 1b, 16K, chat (K = 1 / K = 8):** 0.50 / 0.68 at τ 0.5; 0.375 / 0.44 at τ 0.9 (the 350M's stage 1: 0.225 / 0.31 at τ 0.9).
- **Stage 2 at τ 0.98:** 0.425 / 0.55 (16K), 0.375 / 0.325 (131K), 0.275 / 0.33 (1M); at τ 0.97, 1M: 0.375 / 0.33.
- **Training accuracy hides it.** By fact age, stage 2 answers next-chunk facts (1.00) but 0.58 at 4–31 segments (16K–127K tokens) and 0.22 at 32+ segments; the overall ans_acc (0.97) is dominated by short ages. The 350M (pilot 6): 0.86 at 4+.
- **Mechanism (`write_stats.py`, stage 2 at τ 0.98, 3 × 1M tokens):** only **2** quiet fact heads of 96 (< 0.01% text writes); fact heads write 0.97% of text tokens (350M seeds: 0.19–0.57%); median retention at 1M 0.0017. Across four models, 1M recall tracks the number of quiet fact heads: 10 → 0.95, 6 → 0.675, 4 → 0.575, 2 → 0.275–0.375 (§8.56).
- **Next:** stage-3 evals (0.99, dev and fresh); then KH-2 on the 350M seeds (`kh2_stage4.sh`) and the winning variant on the 8B (`kh2_8b.sh`).

### 8.58 How much of the perplexity gain is the memory? About half (2026-10-14 ~17:30 UTC)
New control `eval_lm.py --ppl_reset --ppl_T <chunk>`: adapters on, memory state cleared before every chunk, so "carried vs cleared" isolates the cross-chunk memory. FineWeb-Edu val, second 1M tokens of a 2M-token stream:

| Graft | carried | cleared | adapters off | adapters vs off | carried vs cleared |
|---|---|---|---|---|---|
| 350M (pilot 4) | 20.52 | 21.60 | 22.64 | −9.4% | **−5.0%** |
| 1.2B v2 | 13.64 | 14.34 | 15.00 | −9.1% | **−4.9%** |
| 1.2B v4 (released) | 13.66 | 14.22 | 15.00 | −8.9% | **−3.9%** |
| Qwen3.5-4B stage 3 | 9.81 | 10.00 | 10.29 | −4.7% | **−1.9%** |
| LFM2.5-8B-A1B stage 3 | 12.71 | 17.70 | 42.51 | −70% | −28% |

- **Correction (paper v2.2, README, blog, deck):** v2.1 attributed the whole 5–9% to the memory. Carrying the memory across chunks accounts for 2–5% (about half for the LFM2 grafts); the rest is the adapters' processing within a chunk.
- **LFM2.5-8B:** its plain-text calibration is poor (3.73 on *Pride and Prejudice* with BOS, like LFM2.5-350M at 6.29), so the adapters, trained with a small LM loss, recalibrate it; its perplexity gains are not comparable with the other grafts.
- Checks: both 1.2B models give 15.00 with the adapters off (same plain base); every v2.1 number reproduces (20.52, 13.64, 13.66, 9.81, 10.29).

### 8.59 KH-2 candidates against a rerun control: small gains, and a noise floor (2026-10-14 ~18:40 UTC)
Stage 4 of the 350M recipe redone from the stage-3 checkpoints (same seed, data and settings; `kh2_stage4.sh`, `kh2_verify.sh`). Chat questions, 1M, K = 1 / K = 8, development streams (fresh streams):

| Seed | KH-1 stage 4 (original) | KH-1 stage 4 rerun (control) | Two-tier (0.97 / 0.995 → 0.99 / 0.998) | Quiet heads (5× text-write L1 on half the heads) |
|---|---|---|---|---|
| 1 | 0.675 / 0.644 | 0.625 / 0.581 (0.825 / 0.562) | 0.725 / **0.675** (0.70 / **0.644**) | 0.575 / 0.475 (0.50 / 0.531) |
| 2 | 0.575 / 0.581 | 0.75 / 0.606 (0.60 / 0.512) | 0.775 / **0.662** (0.675 / **0.575**) | 0.775 / 0.70 (0.65 / 0.662) |
| 0 | 0.95 / 0.781 (0.90 / 0.75) | — | **0.975 / 0.806** (0.925 / 0.787) | — |

- **Noise floor.** Rerunning only stage 4 with the same seed moved seed 2 from 0.575 to 0.75 (K = 1), and the control's development and fresh streams differ by up to 0.20. With 40 streams a K = 1 recall near 0.7 has a 95% interval of about ±0.14. Single-run K = 1 differences below ~0.15 are noise; K = 8 cells (160 facts) are tighter.
- **Two-tier:** beats the rerun control at K = 8 in all four comparisons (+0.06 to +0.09; sign test p = 0.06), does not hurt seed 0 (0.975 / 0.806, the best 350M result), and keeps short range (131K: 0.925–1.00). K = 1 differences are within noise. A small, consistent gain, not a fix.
- **Quiet-head penalty:** mixed (seed 1 worse, seed 2 better) and lowers 16K recall (0.85–0.875).
- **Mechanism (`write_stats.py`, same 6 streams):** two-tier grew quiet fact heads for seed 1 (9 vs 6; median 1M retention 0.027 vs 0.006) but not for seed 2 (4 vs 4; fewer heads store each fact, 21 vs 28). The 0.998 tier writes 4–7× less text (median 0.011–0.014% vs 0.057–0.075%) but also stores facts less often. The quiet-head penalty raised seed 1's retention (0.034) yet lowered its recall (22 vs 27 heads store each fact). Both interventions trade fact writes for quiet: the gate cannot yet tell a fact from a rare name or number, which is the problem KH-2 has to solve.
- **Next:** two-tier stage 3 on LFM2.5-8B-A1B is running (`kh2_8b.sh`).

### 8.60 The quiet-head ranking does not generalise: 10 models (2026-10-14 ~19:15 UTC, RTX 3070 Ti)
`paper_v2/quiet_heads_corr.py`: all ten 350M-recipe models (seeds 0–2, stage-4 reruns of seeds 1–2, two-tier and quiet-head variants of seeds 1–2, two-tier seed 0), write statistics over the same 6 streams, 1M recall = mean of development and fresh streams where both exist.

| Model | 1M K = 1 | 1M K = 8 | quiet fact heads | median retention at 1M |
|---|---|---|---|---|
| seed 0 | 0.925 | 0.766 | 10 | 0.063 |
| seed 0, two-tier | 0.95 | 0.796 | 6 | 0.049 |
| seed 1 | 0.675 | 0.644 | 6 | 0.006 |
| seed 1, rerun | 0.725 | 0.572 | 5 | 0.015 |
| seed 1, two-tier | 0.712 | 0.659 | 9 | 0.027 |
| seed 1, quiet-head penalty | 0.537 | 0.503 | 8 | 0.034 |
| seed 2 | 0.575 | 0.581 | 4 | 0.014 |
| seed 2, rerun | 0.675 | 0.559 | 6 | 0.015 |
| seed 2, two-tier | 0.725 | 0.619 | 4 | 0.011 |
| seed 2, quiet-head penalty | 0.713 | 0.681 | 2 | 0.019 |

- **Spearman ρ with 1M recall (K = 1 / K = 8):** quiet fact heads +0.02 / +0.11; fact heads keeping the fact +0.29 / +0.27; heads storing the fact +0.06 / +0.22; median retention +0.33 / +0.39; fact-head text-write rate +0.07 / −0.20. With n = 10, none is significant (|ρ| ≳ 0.65 needed).
- **Conclusion:** the three-seed ranking of §8.56 was a coincidence of three points (a 1-in-6 chance). What holds: within a model, heads that write more text keep facts less (r −0.41 to −0.51); the seed-0 lineage has the highest retention and recall. What makes seed 0 better is open; recall noise (±0.14 at K = 1, n = 40; reruns moving by 0.175) limits any such analysis.
- **Corrected:** paper v2.2 draft (Table 9 now lists all ten models; "Why seeds differ" → "What differs between seeds"; LFM2.5 and Limitations text softened).

### 8.61 Two-tier thresholds on LFM2.5-8B-A1B: a large gain, pending a control (2026-10-14 ~19:40 UTC, H200 #1)
`kh2_8b.sh` (VARIANT=tier): stage 3 redone from the stage-2 checkpoint (τ 0.97, L1 0.2, 100M tokens, seed 0) with half of each adapter's 32 heads at 0.995; run at 0.99 / 0.998. Chat questions, K = 1 / K = 8:

| LFM2.5-8B stage 3 | 16K | 131K | 1M | 1M, fresh streams |
|---|---|---|---|---|
| KH-1 (standard) | 0.675 / 0.875 | 0.55 / 0.569 | 0.375 / 0.356 | 0.375 / 0.331 |
| **Two-tier** | **0.95 / 0.894** | **0.85 / 0.769** | **0.675 / 0.588** | **0.525 / 0.456** |

- Gains at every distance (+0.275 at 16K, +0.30 at 131K and 1M on development streams; +0.15 / +0.125 on fresh streams), about twice the stage-4 rerun noise seen on the 350M (§8.59). Single run: **not a claim until** the control (standard stage 3 rerun, seed 0) and a replicate (two-tier, seed 1) are in (`kh2_8b_verify.sh`, running).

### 8.62 The 8B two-tier gain does not replicate (2026-10-14 ~21:35 UTC, H200 #1)
`kh2_8b_verify.sh`, chat questions, 1M, K = 1 / K = 8, development (fresh) streams:

| LFM2.5-8B stage 3 | 16K | 131K | 1M | 1M fresh |
|---|---|---|---|---|
| standard, seed 0 (original) | 0.675 / 0.875 | 0.55 / 0.569 | 0.375 / 0.356 | 0.375 / 0.331 |
| standard, seed 0 rerun (control) | 0.50 / 0.725 | 0.45 / 0.481 | 0.35 / 0.388 | 0.35 / 0.338 |
| two-tier, seed 0 | 0.95 / 0.894 | 0.85 / 0.769 | 0.675 / 0.588 | 0.525 / 0.456 |
| two-tier, seed 1 (replicate) | 0.475 / 0.713 | 0.425 / 0.50 | 0.275 / 0.331 | 0.225 / 0.344 |

- The control reproduces the standard result at 1M (0.35 vs 0.375); the replicate shows no gain. §8.61's gain was one run, not the method. Two-tier thresholds are not confirmed on either base.
- Retracts the provisional §8.61 reading.

### 8.63 Next: KH-1 on Gemma 4 E2B / E4B (planned, not started; 2026-10-14)
- **Bases:** `google/gemma-4-E2B-it` (35 layers, hidden 1536) and `google/gemma-4-E4B-it` (42 layers, hidden 2560): pure attention, sliding window 512 with a global layer every 5–6 layers, 262K vocabulary. Run chunk-local like Qwen3.5 (no KV cache across chunks), so the archival adapters are the only long-range memory.
- **Work:** a `gemma_hais.py` wrapper modelled on `qwen_hais.py`; the gates-at-zero equivalence check (graft = native chunked loss); FineWeb-Edu tokenised with the Gemma 4 tokenizer (uint32).
- **Budget (lesson of §8.57–8.62):** give it the Qwen budget, ~2B tokens with ≥131K-token batches (2–3 h on 8×B200 or 8×H100; about a day on one H200). Do not judge it on a short single-GPU run.
- **Demo:** E2B with a 4-bit base should fit the 8 GB RTX 3070 Ti.

### 8.64 BABILong: zero-shot 0; a short fine-tune beats RAG from 4K to 128K (2026-10-15 ~02:40 UTC, H200 #1)
`lm/babilong_eval.py` on the test split (RMT-team/babilong), chat question + "Answer with one word.", correct if the target is among the first six words; 50 samples per cell (20 at 1M). Fine-tune: `lm/babilong_train.py`, KH-1-350M adapters (pilot 6) on RMT-team/babilong-train-5k-samples (qa1–qa5, 0K–16K), 6,000 samples, τ 0.97, answer-token loss only. Mean over qa1–qa5:

| | 0K | 4K | 16K | 64K | 128K | 1M |
|---|---|---|---|---|---|---|
| KH-1 zero-shot | 0.03 | 0.00 | 0.00 | | | |
| fine-tuned, τ 0.97 | 0.41 | 0.43 | 0.40 | 0.34 | 0.32 | |
| fine-tuned, τ 0.99 | 0.04 | 0.39 | 0.44 | 0.39 | 0.35 | 0.21 |
| RAG (BM25 top-10, same base) | 0.36 | 0.24 | 0.23 | 0.20 | 0.21 | 0.25 |
| memory off | 0.02 | 0.02 | 0.02 | | | |

- KH-1 never learned bAbI sentences, so zero-shot it answers nothing. After a short fine-tune it beats RAG from 4K to 128K, mainly on the multi-hop tasks (qa2/qa3: 0.2–0.36 vs RAG ≈ 0). It fades with length (0.21 at 1M; RAG 0.25, n = 20). Far below fine-tuned RMT/ARMT (~0.9 on qa1 at 1M). Round 2 (12K more samples, up to 32K) running: `go_babilong2.sh`.
- Files: `results/babilong/`.

### 8.65 BABILong round 2: about twice RAG from 4K to 128K (2026-10-15 ~07:30 UTC)
Round 1 checkpoint + 12,000 more training samples (qa1–qa5, 0K–32K, τ 0.97, seed 1; first-token accuracy 0.43 → 0.54). Mean over qa1–qa5 (50 per cell, 20 at 1M):

| | 0K | 4K | 16K | 64K | 128K | 1M |
|---|---|---|---|---|---|---|
| round 2, τ 0.97 | 0.55 | 0.54 | 0.48 | 0.47 | 0.42 | 0.26 |
| round 2, τ 0.99 | 0.47 | 0.55 | 0.50 | 0.46 | 0.42 | |
| RAG | 0.36 | 0.24 | 0.23 | 0.20 | 0.21 | 0.25 |

- qa5 0.84–0.90 to 128K, qa4 0.48–0.68; qa1–qa3 0.24–0.46. 1M ties RAG (0.26 vs 0.25, n = 20). Round 3 (4,000 samples at 16K–64K only, `go_babilong3.sh`) running.

### 8.66 KH-1 on Gemma 4 E2B: flat to 1M at 0.4–0.575 after a single-GPU run (2026-10-15 ~11:50 UTC, H200 #1)
`gemma_hais.py` (adapters as forward hooks inside Gemma's own forward; gates-at-zero loss = native 5.034), adapters after layers 16/25/34 (24 heads, 35.7M trainable), 2,048-token chunks, batch 4 × 4096 (16K tokens per step), 25K tok/s. Stage 1 τ 0.5 / L1 0.05 / 600M → stage 2 τ 0.9 / 0.2 / 250M → stage 3 τ 0.97 / 0.2 / 150M (1B tokens, `go_gemma_h200.sh`). Chat questions, K = 1 / K = 8:

| | 16K | 131K | 1M |
|---|---|---|---|
| stage 1, τ 0.9 | 0.00 / 0.04 | | |
| stage 2 (mid-run), τ 0.9 | 0.55 / 0.51 | | |
| stage 2 (end), τ 0.98 | 0.325 / 0.36 | 0.45 / 0.32 | 0.525 / 0.34 |
| **stage 3, τ 0.99** | **0.575 / 0.49** | **0.55 / 0.40** | **0.575 / 0.42** |
| stage 3, fresh streams | | | 0.40 / 0.44 |
| stage 3, unseen frames × attributes | 0.20 / 0.24 | | 0.225 / 0.20 |
| memory off | 0.00 / 0.006 | | |

- Stage 1 learned slowly (training ans_acc 0.27 → 0.45 over 600M tokens; age 4+ accuracy 0.17); stage 2 switched the memory on. Recall is flat from 16K to 1M, at a level below the dense LFM2/Qwen grafts (0.75–0.90 fresh) and above the LFM2.5-8B single-GPU run (0.35–0.375). Same budget caveat: 1B tokens and 16K-token batches against Qwen's 2B and 131K.
- Files: `results/gemma/`.

### 8.67 BABILong round 3: flat from 0K to 1M at about twice RAG (2026-10-15 ~14:07 UTC, H200 #1)
Round 2 checkpoint + 4,000 training samples at 16K–64K only (`go_babilong3.sh`, τ 0.97, seed 2). Test split, mean over qa1–qa5 (50 per cell, 20 at 1M):

| | 0K | 4K | 16K | 64K | 128K | 1M |
|---|---|---|---|---|---|---|
| **round 3, τ 0.97** | **0.60** | **0.55** | **0.58** | **0.61** | **0.57** | **0.50** |
| round 3, τ 0.99 | 0.42 | 0.56 | 0.57 | 0.59 | 0.54 | |
| RAG (BM25 top-10, same base) | 0.36 | 0.24 | 0.23 | 0.20 | 0.21 | 0.25 |

- Per task (τ 0.97): qa1 0.48–0.56 (1M 0.45), qa2 0.28–0.34 (1M 0.45), qa3 0.36–0.42 (1M 0.25), qa4 0.64–0.84 (1M 0.40), qa5 0.96–0.98 (1M 0.95).
- Training on long samples removed the fade with length. Total BABILong fine-tuning: 22,000 samples, about 4 h on a shared H200. Fine-tuned RMT/ARMT remain stronger on qa1 (~0.9 at 1M).
- Checkpoint: `results/babilong/kh1_350m_bl3/ckpt.pt` (63 MB, adapters only).

## 9. How to run

**Your workstation is not used for experiments any more.** Everything runs on the H200 (`<user>@<node>`, `~/hais_poc`).

```bash
# one-time per new GPU box: env + kernel checks (see §7 for the two traps)
#   torch 2.9.1+cu128, triton==3.7.1, flash-linear-attention 0.5.2, mamba-ssm 2.3.2.post1 wheel + PR #997 overlay
env/bin/python -m pytest -q -p no:warnings test_mamba3_siso.py   # expect 8 passed, 1 skipped
cd poc_m3 && ../env/bin/python test_gdn_grad.py                  # expect GDN_GRAD_OK
../env/bin/python test_hx_stream.py                               # expect HX_STREAM_OK

# launch / resume the whole queue (pilot 2 + Phase-1 grid), 14 runs in parallel under CUDA MPS
PAR=14 setsid nohup ./h200_launch.sh > launch.out 2>&1 < /dev/null &

# from the workstation: pull logs (not weights), then tables / figures / diagnostics
./pull_results.sh <user>@<node> runs/h200 'p2_*' 15
python report.py runs/h200 --group
python plot_results.py runs/h200 fig.png "gdn_lr3e-3_sp0.5=GDN" "p2_haishx_l1=HAIS-X" ...
python analyze_keys.py runs/h200/<run>     # per-head write gates, decay, key overlap
python ablate_heads.py runs/h200/<run>     # causal: recall when each memory head is zeroed
```

**Other scripts.**
- `train.py --eval_only --out <run>`: re-run the streaming eval on a saved model.
- `reeval.sh`: re-run evals that crashed.
- `verify_delta_needle.py`: Mamba-1 control (`venv_torch27`).

**Files.**
- `poc_m3/layers.py`: Mamba-3 (official kernel, state in/out), DeltaMem (GDN / HAIS).
- `model.py`: TinyLM.
- `tasks.py`: fact stream.
- `train.py`: train + streaming eval.
- `audit/`: forensic scripts and fix patches.

## References (all checked 2026-10-11)
Mamba-3 arXiv 2603.15569 · mamba-ssm 2.3.2.post1 (PyPI, 2026-05-09) · HF `state-spaces/mamba3-siso-187m` (2026-07-28) · Raven arXiv 2607.25357 · Gated DeltaNet arXiv 2412.06464 · GDN-2 arXiv 2605.22791 · Buitrago & Gu arXiv 2507.02782 · Stuffed Mamba arXiv 2410.07145 · Jelassi et al. arXiv 2402.01032 · Based arXiv 2402.18668 · RAMba/HSA arXiv 2504.16795 · DRT arXiv 2410.01651 · HSA-LG arXiv 2510.17196 · ARMT arXiv 2407.04841 · BABILong arXiv 2406.10149 · RULER arXiv 2404.06654 · Okpekpe & Orvieto arXiv 2508.19029 · fla 0.5.2 (2026-07-27).
