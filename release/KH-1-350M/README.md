---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE
base_model: LiquidAI/LFM2-350M
library_name: pytorch
tags: [long-context, memory, recurrent, delta-rule, adapters, kh-1]
---

# KH-1-350M: constant-memory fact recall over 1M+ tokens for LFM2-350M

KH-1-350M adds a small **archival memory** (16M trainable parameters, three delta-rule adapter layers) to the frozen [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M).

**How it reads:**
- The base model reads in 1,024-token chunks.
- Its own weights are unchanged.
- The adapters carry facts across chunks in a fixed-size state.

GPU memory therefore stays constant at any length: 0.84 GB on an 8 GB RTX 3070 Ti, reading at 32.5K tokens/s.

*Research preview. Paper: "Write Where Short Memory Cannot Reach: Distance-Invariant Fact Recall in Fixed-State Language Models" (Youssef Khouidi, 01TEK, 2026). Code: https://github.com/01TEK/KH (tag `v2.2`).*

## Results
- **Task:** facts spliced into FineWeb-Edu validation text, asked 16K / 131K / 1M tokens later.
- **Scores:** exact-match recall for one fact / eight facts per stream (40 streams for one fact; 20 streams of 8 facts).
- **Streams:** the table uses the evaluation streams on which the write thresholds were chosen; fresh-stream results are below it.
- **Setting:** inference write threshold τ = 0.99 unless noted.

| Questions | 16K | 131K | 1M |
|---|---|---|---|
| trained phrasings | 0.95 / 0.84 | 1.00 / 0.91 | **0.925 / 0.77** |
| trained phrasings, chat | 0.975 / 0.91 | 0.975 / 0.81 | **0.95 / 0.78** |
| unseen phrasings | 0.80 / 0.45 | 0.80 / 0.41 | 0.575 / 0.325 |

- Inside novels (1M-token windows of three Project Gutenberg books, chat questions, one fact): 0.825.
- Book demo (5 facts, 1.3M tokens): 3/5.
- **Fresh streams:** on 40 new streams that played no part in any choice, 1M recall with chat questions is **0.90 / 0.75** (one / eight facts), against 0.95 / 0.78 on the evaluation streams.
- **Seeds:** retraining this recipe from scratch with two more seeds gave 0.675 and 0.575 at 1M (chat questions, one fact, evaluation streams); this checkpoint (seed 0) is the best of three. The threshold curriculum helped in all three.

With the memory switched off (adapter gates zeroed), recall is 0 and the book demo scores 0/5.

## Usage
```bash
git clone https://github.com/01TEK/KH && cd KH && git checkout v2.2
hf download 01TEK/KH-1-350M --local-dir KH-1-350M      # config.json + adapters.safetensors
python poc_m3/kh1_chat.py --ckpt ./KH-1-350M --tau 0.99          # interactive: /load book.txt, /ask Q || answer prefix, /mem
python poc_m3/kh1_demo.py --ckpt ./KH-1-350M --tau 0.99 --books pg2600.txt pg2701.txt pg1342.txt
```
Requirements:
- PyTorch with CUDA and `flash-linear-attention` 0.5.2;
- Transformers: 4.57 for LFM2, 5.19 for Qwen3.5;
- `safetensors`.

## Training
Four stages on one H200 (68 min in total): write threshold 0.5 (300M tokens) → 0.9 (100M) → 0.9 with text-write penalty 0.2 (100M) → 0.97 (100M).
- Data: FineWeb-Edu text with synthetic facts in 358 phrasings, half of the questions as chat turns.
- Only the adapters are trained.

## Limitations
- It recalls stored facts (a name's attribute is a value), not whole documents.
- Facts phrased very differently from training are recalled less often (see "unseen phrasings").
- Text dense with names competes for the fixed memory.
- Every chat turn is written to memory, so the question order can change later answers.
- Training seeds matter: two retrained seeds of this recipe reach 0.575–0.675 at 1M.

## License
- The adapter weights are released by 01TEK.
- Using them requires the base model, under its own license: the LFM Open License v1.0.
- This is an independent research project, not affiliated with Liquid AI.
