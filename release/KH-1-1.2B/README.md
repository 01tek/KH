---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/LICENSE
base_model: LiquidAI/LFM2-1.2B
library_name: pytorch
tags: [long-context, memory, recurrent, delta-rule, adapters, kh-1]
---

# KH-1-1.2B: constant-memory fact recall over 1M+ tokens for LFM2-1.2B

KH-1-1.2B adds a small **archival memory** (63M trainable parameters, three delta-rule adapter layers) to the frozen [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B).

**How it reads:**
- The base model reads in 2,048-token chunks.
- Its own weights are unchanged.
- The adapters carry facts across chunks in a fixed-size state.

GPU memory therefore stays constant at any length: 2.8 GB on an 8 GB RTX 3070 Ti, reading at 13.7K tokens/s.

*Research preview. Paper: "Write Where Short Memory Cannot Reach: Distance-Invariant Fact Recall in Fixed-State Language Models" (Youssef Khouidi, 01TEK, 2026). Code: https://github.com/01TEK/KH (tag `v2.1`).*

## Results
- **Task:** facts spliced into FineWeb-Edu validation text, asked 16K / 131K / 1M tokens later.
- **Scores:** exact-match recall for one fact / eight facts per stream (40 streams for one fact; 20 streams of 8 facts).
- **Streams:** the table uses the evaluation streams on which the write thresholds were chosen; fresh-stream results are below it.
- **Setting:** inference write threshold τ = 0.99 unless noted.

| Questions | 16K | 131K | 1M |
|---|---|---|---|
| trained phrasings | 0.875 / 0.88 | 0.95 / 0.90 | 0.80 / 0.66 |
| trained phrasings, chat | 0.95 / 0.91 | 0.95 / 0.86 | **0.85 / 0.71** |
| unseen phrasings | 0.875 / 0.59 | 0.825 / 0.475 | 0.45 / 0.39 |

- Inside novels (1M-token windows of three Project Gutenberg books, chat questions, one fact): **0.90**.
- Book demo (5 facts, 1.3M tokens): **4/5**.
- **Fresh streams:** on 40 new streams that played no part in any choice, 1M recall with chat questions is **0.75 / 0.70** (one / eight facts), against 0.85 / 0.71 on the evaluation streams.

With the memory switched off (adapter gates zeroed), recall is 0 and the book demo scores 0/5.

## Usage
```bash
git clone https://github.com/01TEK/KH && cd KH && git checkout v2.1
hf download 01TEK/KH-1-1.2B --local-dir KH-1-1.2B      # config.json + adapters.safetensors
python poc_m3/kh1_chat.py --ckpt ./KH-1-1.2B --tau 0.99          # interactive: /load book.txt, /ask Q || answer prefix, /mem
python poc_m3/kh1_demo.py --ckpt ./KH-1-1.2B --tau 0.99 --books pg2600.txt pg2701.txt pg1342.txt
```
Requirements:
- PyTorch with CUDA and `flash-linear-attention` 0.5.2;
- Transformers: 4.57 for LFM2, 5.19 for Qwen3.5;
- `safetensors`.

## Training
Three stages on 8×B200 (77 min in total): write threshold 0.5 (3B tokens) → 0.9 with text-write penalty 0.2 (1B) → 0.97 (0.3B).
- Data: FineWeb-Edu text with synthetic facts in 358 phrasings, half of the questions as chat turns.
- Only the adapters are trained.

## Limitations
- It recalls stored facts (a name's attribute is a value), not whole documents.
- Facts phrased very differently from training are recalled less often (see "unseen phrasings").
- Text dense with names competes for the fixed memory.
- Every chat turn is written to memory, so the question order can change later answers.
- Trained once (one seed); the 350M recipe's 1M recall varied from 0.575 to 0.95 across three seeds.

## License
- The adapter weights are released by 01TEK.
- Using them requires the base model, under its own license: the LFM Open License v1.0.
- This is an independent research project, not affiliated with Liquid AI.
