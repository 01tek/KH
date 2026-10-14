# KH-1: constant-memory recall over 1M+ tokens for Liquid LFM2

KH-1 adds a small **archival memory** to [LiquidAI LFM2](https://huggingface.co/LiquidAI/LFM2-1.2B):
- 16M (350M) or 63M (1.2B) trainable parameters;
- LFM2's own 350M / 1.2B weights stay frozen.

The result reads documents of any length with **constant GPU memory**. On an 8 GB RTX 3070 Ti:

| Model | GPU memory (any length) | Reading speed | 1M tokens in |
|---|---|---|---|
| KH-1-350M | 0.84 GB | 32.5K tok/s | ~32 s |
| KH-1-1.2B | 2.8 GB | 13.7K tok/s | ~75 s |
| KH-1-Q-4B (Qwen3.5-4B, nf4 base) | 3.9 GB | 4.1K tok/s | ~4 min |

These are peak GPU memory and speed while reading the 1.3M-token book demo.

It answers questions about facts it read up to a million tokens earlier. LFM2's own configuration stops at 128K positions, and its KV cache grows with every token.

*Status: research preview. Every number below comes from a logged run under `results/`; see the paper `paper_v2/arxiv/main.tex` (Section 6) and the research log `MAMBA3_HAIS_POC_PLAN.md` §8.36–8.53.*

## How it works
- **Bounded local context.** LFM2's attention and convolutions see one chunk at a time: 1–2K tokens, each chunk prefixed with LFM2's BOS. Their memory is bounded, and positions never leave LFM2's trained range.
- **Archival memory.** Gated delta-rule layers are inserted after LFM2 blocks 7, 11 and 15:
  - decay pinned to 1, so they never forget on their own;
  - a write threshold τ: curriculum 0.5 → 0.9 → 0.97 in training; 0.99 at inference;
  - normalised values, so the state stays bounded;
  - a ReZero gate, so training starts from exactly LFM2.

  They carry everything older than the current chunk, in a fixed-size state (under 2 MB).
- **Training that routes facts into the archival memory.** Drill facts appear in one chunk and are asked in the next chunk, so LFM2's local layers cannot answer them. Only the archival memory can, and it receives the gradient.
  - An L1 penalty on archival writes at ordinary text tokens keeps the memory for facts.
  - This is the routing principle from the paper. With it (and a short half-life on the non-archival layers), all six 117M runs and a 494M model trained from scratch keep recall flat from 16K to 1M tokens.

## Results
All numbers below come from logged runs.
- **Task:** facts spliced into FineWeb-Edu validation text are asked after 16K / 131K / 1M tokens.
- **Scores:** exact-match recall for one fact / eight facts per stream (40 streams for one fact; 20 streams of 8 facts).

| Model | Question format | 16K | 131K | 1M |
|---|---|---|---|---|
| **KH-1-350M** (τ 0.99) | trained phrasings | 0.95 / 0.84 | 1.00 / 0.91 | **0.925 / 0.77** |
| **KH-1-350M** (τ 0.99) | trained phrasings, chat turns | 0.975 / 0.91 | 0.975 / 0.81 | **0.95 / 0.78** |
| **KH-1-350M** (τ 0.99) | unseen phrasings (new frame × new attribute) | 0.80 / 0.45 | 0.80 / 0.41 | 0.575 / 0.325 |
| KH-1-1.2B v4 (τ 0.99) | trained phrasings, chat turns | 0.95 / 0.91 | 0.95 / 0.86 | **0.85 / 0.71** |
| KH-1-1.2B v4 (τ 0.99) | unseen phrasings | 0.875 / 0.59 | 0.825 / 0.475 | 0.45 / 0.39 |

- **Book haystack:** the same facts in 1M-token windows of three novels, chat turns, one fact: **0.90** for KH-1-1.2B v4, 0.825 for KH-1-350M.
- **KH-1-350M recipe:** adapters trained at τ 0.5 → 0.9 → 0.97 (text-write L1 0.2), run at τ 0.99.
- **KH-1-1.2B v4** uses the same curriculum. Before the τ 0.97 step (v2, run at τ 0.97) it scored 0.575 / 0.51 at 1M.

**Fresh streams and seeds** (paper Table 7, research log §8.53):
- The write thresholds were chosen on the streams above. On 40 fresh streams, 1M recall with chat questions is **0.90 / 0.75** (KH-1-350M), **0.75 / 0.70** (KH-1-1.2B v4) and **0.90 / 0.70** (KH-1-Q-4B, τ 0.98).
- The KH-1-350M recipe retrained with seeds 1 and 2 (`poc_m3/lm/seed_350m.sh`) reaches 0.675 and 0.575 at 1M (chat, one fact) against 0.95 for seed 0. 16K is stable (0.95–0.975), and the τ 0.97 step helped every seed. The released model is the best of three; `python paper_v2/seed_table.py` rebuilds the table.

**Book demo** (`kh1_demo.py`, RTX 3070 Ti 8 GB):
- **Stream:** *War and Peace* + *Moby Dick* + *Pride and Prejudice*, 1,318,920 tokens.
- **Facts:** five planted near the start; the questions come ~1.30M tokens later, as chat turns.

| Model | Exact answers | GPU memory (constant) | Reading speed |
|---|---|---|---|
| KH-1-350M (τ 0.99) | 3/5 (482, 739561, September) | 0.84 GB | 32.5K tok/s |
| **KH-1-1.2B v4** (τ 0.99) | **4/5** (482, thuzoka, 739561, September; unseen phrasing "2879" for 287) | 2.8 GB | 13.7K tok/s |
| KH-1-1.2B v2 (τ 0.97) | 2/5 (thuzoka, September) | 2.9 GB | 12.7K tok/s |

- No model answers the unseen-phrasing fact exactly.
- KH-1-350M's near miss is "thuzo" for "thuzoka".
- Transcripts: `kh1/demo_transcripts/` (scripted) and `results/demo_logs/` (interactive).

**Control (memory switched off: ReZero gates zeroed, i.e. plain chunked LFM2):**
- 16K recall is 0.00 / 0.00 (KH-1-350M).
- The book demo gets 0/5 for both models, and they invent answers:
  - KH-1-1.2B: "12", "KRIBRISEN-WIFI", "123.456 MHz";
  - KH-1-350M: "12", "Kribrisen@wifi", "February 13th".

**Language modelling.** FineWeb-Edu val perplexity, measured over the second 1M of a 2M-token stream (1K / 2K-token chunks, state carried):

| Model | memory carried | memory cleared every chunk | adapters off | adapters vs off | carried vs cleared |
|---|---|---|---|---|---|
| KH-1-350M (pilot 4) | 20.52 | 21.60 | 22.64 | −9.4% | −5.0% |
| KH-1-1.2B (v2) | 13.64 | 14.34 | 15.00 | −9.1% | −4.9% |
| KH-1-1.2B (v4, released) | 13.66 | 14.22 | 15.00 | −8.9% | −3.9% |
| KH-1-Q-4B (stage 3) | 9.81 | 10.00 | 10.29 | −4.7% | −1.9% |

- **No LM cost:** the adapters lower perplexity by 5–9% against zeroing them.
- **The memory's share:** "memory cleared" keeps the adapters on but clears their state before every chunk (`eval_lm.py --ppl_reset`). Carrying the memory across chunks accounts for 2–5%, about half of the total for the LFM2 grafts. The rest comes from the adapters' processing within a chunk.

## KH-1-Q: the same memory on Qwen3.5-4B
**Setup:**
- **Base model:** Qwen3.5-4B (Apache-2.0), frozen. It runs in 2,048-token chunks: no KV cache and no Gated-DeltaNet state carried across chunks.
- **Adapters:** 99M trainable parameters after layers 15/23/31. Checkpoint `kh1/ckpt/kh1q_4b_s2.pt`.
- **Training:** τ 0.5 → 0.9 (2B tokens on 8×B200, ~2 h). Run at τ **0.98**.

**Recall** (FineWeb-Edu; 40 streams for one fact, 20 streams of 8 facts):

| Distance | 16K | 131K | 1M |
|---|---|---|---|
| chat questions, τ 0.98 | | | **0.975 / 0.70** |
| chat questions, τ 0.97 | 1.00 / 0.91 | 1.00 / 0.75 | 0.925 / 0.66 |
| unseen phrasings, plain questions, τ 0.97 | 0.80 / 0.57 | 0.875 / 0.47 | 0.575 / 0.45 |

τ 0.98 was the best of three inference thresholds (0.97 / 0.98 / 0.99: 0.925 / 0.975 / 0.85) on these streams; on 40 fresh streams it scores **0.90 / 0.70**.

**Book demo** (1.28M tokens with Qwen's tokenizer):

| Setting | Score | Peak GPU memory |
|---|---|---|
| τ 0.98, bf16 | **3/5** (482, 739561, September; "thuzoquo" for thuzoka; unseen phrasing missed) | 9.3 GB |
| memory off | 0/5 | |
| nf4 base on an RTX 3070 Ti, τ 0.98 | 2/5 (739561, September) | 3.9 GB, 4.1K tok/s |

- **Perplexity** (stage-3 checkpoint): 9.81 with memory, 10.29 without.
- **Chat format:** Qwen3.5-4B's chat template opens a `<think>` block by default, and KH-1-Q was trained with that format. Free chat therefore "thinks out loud". Recall questions answer directly when given an answer prefix (`/ask Q || prefix`).

## Limitations (read before using)
- **Facts, not whole documents.** KH-1 recalls specific attribute–value facts it stored, not the full text. It cannot summarise a book it read 500K tokens ago: only the current chunk is visible to LFM2 itself.
- **Trained phrasings.** The memory was trained on 358 fact phrasings (a name's attribute is a value).
  - New combinations of frames and attributes: 0.80–0.875 at 16K and 0.45–0.575 at 1M for the released models.
  - Wholly new sentence templates: 0.43–0.55 at 16K for most LFM2 grafts, but ≈ 0 (≤ 0.05) at 1M.
- **Long range is good, not perfect.** At 1M on FineWeb-Edu, KH-1-350M recalls 0.925–0.95 and KH-1-1.2B 0.80–0.85 (0.90 and 0.75 on fresh streams). Inside novels, where dense proper names compete for the memory, they recall 0.825 and 0.90.
- **Seeds matter.** Two retrained seeds of the 350M recipe reach only 0.575–0.675 at 1M; the 1.2B and 4B grafts were trained once.
- **Capacity is finite.** The state has a fixed size; we evaluated at most 8 planted facts per stream.
- **The conversation is stored too.** Every chat turn is written to memory, so the order of questions can change later answers.

## Next
- **Gemma 4 E2B / E4B:** the same graft on Google's on-device models (pure attention, 262K vocabulary), with the full Qwen-sized training budget. Planned; see research log §8.63.
- **LFM2.5-8B-A1B:** the mixture-of-experts graft works but recalls only 0.35–0.375 at 1M after a short single-GPU run; it needs a matched budget.
- **KH-2:** teach the write gate to tell a fact from a rare name or number (paper §6, "Making fact heads quieter by design").

## Usage
```bash
hf download 01TEK/KH-1-1.2B --local-dir KH-1-1.2B      # adapters + config (a local .pt checkpoint also works)
python poc_m3/kh1_chat.py --ckpt KH-1-1.2B --tau 0.99         # interactive: /load book.txt, /ask ..., /mem
python poc_m3/kh1_demo.py --ckpt KH-1-1.2B --tau 0.99 --books demo/books/pg2600.txt demo/books/pg2701.txt demo/books/pg1342.txt
```

## License
- The KH-1 checkpoints contain only the archival-memory adapter weights.
- KH-1 requires the LiquidAI LFM2 base weights, which are released under the **LFM Open License v1.0**; read it before any commercial use.
- KH-1-Q requires Qwen3.5-4B (Apache-2.0).
- KH-1 is an independent research project and is not affiliated with Liquid AI or the Qwen team.
