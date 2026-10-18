---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE
base_model: LiquidAI/LFM2-350M
library_name: pytorch
tags: [long-context, memory, babilong, recurrent, delta-rule, adapters, kh-1]
---

# KH-1-350M-BABILong

KH-1-350M archival-memory adapters (16M parameters, frozen LiquidAI/LFM2-350M) fine-tuned on the BABILong training split
(RMT-team/babilong-train-5k-samples, qa1–qa5, 22K samples up to 64K tokens). Constant GPU memory at any length.

**BABILong test split** (mean over qa1–qa5; 50 samples per cell, 20 at 1M; inference threshold 0.97):

| | 0K | 4K | 16K | 64K | 128K | 1M |
|---|---|---|---|---|---|---|
| **KH-1-350M-BABILong** | 0.60 | 0.55 | 0.58 | 0.61 | 0.57 | 0.50 |
| RAG (BM25 top-10, same base) | 0.36 | 0.24 | 0.23 | 0.20 | 0.21 | 0.25 |

Per task: qa5 0.96–0.98 (0.95 at 1M), qa4 0.64–0.84, qa1 0.48–0.56, qa2 0.28–0.34, qa3 0.36–0.42.

```bash
git clone https://github.com/01tek/KH && cd KH
hf download 01TEK/KH-1-350M-BABILong --local-dir KH-1-350M-BABILong
python poc_m3/lm/babilong_eval.py --ckpt KH-1-350M-BABILong --tau 0.97 --tasks qa1 --lengths 128k --n 50 --out qa1_128k.json
```

Research preview from 01TEK R&D. Paper: https://doi.org/10.5281/zenodo.23447665. Code and logs: https://github.com/01tek/KH. Not affiliated with Liquid AI.
