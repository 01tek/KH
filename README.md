# KH: constant-memory fact recall over 1M+ tokens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23447665.svg)](https://doi.org/10.5281/zenodo.23447665)

KH is 01TEK's long-memory model line. **KH-1** is the first version; later versions will be added here.

This release has the code, evaluation logs and paper for **"Write Where Short Memory Cannot Reach: Distance-Invariant Fact Recall in Fixed-State Language Models"** (Youssef Khouidi, 01TEK, 2026), and for the **KH-1** memory adapters.

KH-1 grafts a small **archival memory** (three gated delta-rule layers) onto a frozen open model:
- the base model reads in 1–2K-token chunks and its weights are unchanged;
- the adapters carry facts across chunks in a fixed-size state;
- so GPU memory stays the same at any length. On an 8 GB RTX 3070 Ti, KH-1-1.2B read 1.3M tokens (three novels) in 2.6 GB and answered 4 of 5 questions about facts planted near the start.

**Watch:** [the 3-minute demo video](https://youtu.be/mCGc3_Mh0HI): three novels, 1.3M tokens, an 8 GB gaming GPU.

**Try it:**
- [KH-1 Live](https://huggingface.co/spaces/01TEK/KH-1-live): upload a book (up to 2M tokens), plant facts anywhere, read it into a fixed 1.9 MB memory on a GPU (1.3M tokens in about 75 s), then ask. Switch the memory off to see the same model fail.
- [KH-1 Demo](https://huggingface.co/spaces/01TEK/KH-1-demo): CPU, a precomputed 1.87 MB memory of three novels (1.3M tokens).
- [Colab notebook](https://colab.research.google.com/github/01tek/KH/blob/main/space/KH1_demo_colab.ipynb): the same CPU demo on a free runtime.

## Results
Recall of a fact read 1M tokens earlier: FineWeb-Edu text, chat questions, one fact per stream, 40 fresh streams per model.

| Model | Frozen base | Adapters | 1M recall | GPU memory | Weights |
|---|---|---|---|---|---|
| KH-1-350M | LFM2-350M | 16M | **0.90** | 0.84 GB | [01TEK/KH-1-350M](https://huggingface.co/01TEK/KH-1-350M) |
| KH-1-1.2B | LFM2-1.2B | 63M | **0.75** | 2.8 GB | [01TEK/KH-1-1.2B](https://huggingface.co/01TEK/KH-1-1.2B) |
| KH-1-Q-4B | Qwen3.5-4B | 99M | **0.90** | 3.9 GB (4-bit base) | [01TEK/KH-1-Q-4B](https://huggingface.co/01TEK/KH-1-Q-4B) |
| same models, memory off | | | 0.00 | | |

Read these with the caveats:
- The write thresholds were chosen on our development streams, where the same models score 0.95 / 0.85 / 0.975.
- Retrained with two more seeds, the 350M recipe reaches 0.675 and 0.575 at 1M; the released model is the best of three.
- KH-1 recalls stored facts, not whole documents. Fact phrasings unseen in training reach 0.375–0.575 at 1M.

**BABILong** (standard benchmark, test split, mean over qa1–qa5; 50 samples per cell, 20 at 1M). KH-1-350M with only its adapters fine-tuned on the BABILong training split, against retrieval with the same base:

| | 4K | 16K | 64K | 128K | 1M |
|---|---|---|---|---|---|
| **KH-1-350M (BABILong-tuned)** | **0.55** | **0.58** | **0.61** | **0.57** | **0.50** |
| RAG (BM25 top-10, same base) | 0.24 | 0.23 | 0.20 | 0.21 | 0.25 |

Out of the box KH-1 scores 0 here (it never saw bAbI-style sentences); weights: [01TEK/KH-1-350M-BABILong](https://huggingface.co/01TEK/KH-1-350M-BABILong). Scripts: `poc_m3/lm/babilong_train.py`, `babilong_eval.py`.

**Other bases** (short single-GPU runs): Gemma 4 E2B recalls 0.40–0.575 at 1M, flat from 16K; LFM2.5-8B-A1B (mixture of experts) 0.35–0.375. See the paper.

Details: [`kh1/README.md`](kh1/README.md) and the paper ([doi:10.5281/zenodo.23447665](https://doi.org/10.5281/zenodo.23447665); PDF also in [`paper_v2/arxiv/main.pdf`](paper_v2/arxiv/main.pdf)).

## Quick start (LFM2 grafts)
Needs an NVIDIA GPU with CUDA. Versions we tested: PyTorch 2.9.1, Transformers 4.57.6, flash-linear-attention 0.5.2.
```bash
git clone https://github.com/01TEK/KH && cd KH
pip install torch transformers==4.57.6 flash-linear-attention==0.5.2 safetensors huggingface_hub
hf download 01TEK/KH-1-1.2B --local-dir KH-1-1.2B

# the book demo: three novels (1.32M tokens), five planted facts, questions at the end
python poc_m3/kh1_demo.py --ckpt KH-1-1.2B --tau 0.99 --books demo/books/pg2600.txt demo/books/pg2701.txt demo/books/pg1342.txt

# interactive: /load FILE, /ask Q || answer prefix, /mem, /reset
python poc_m3/kh1_chat.py --ckpt KH-1-1.2B --tau 0.99
```
- **KH-1-350M:** the same commands with `--ckpt KH-1-350M --tau 0.99`.
- **KH-1-Q-4B:** Transformers 5.19 (Qwen3.5) and `--tau 0.98`; add `--quant nf4` (bitsandbytes) to fit an 8 GB GPU.

## Reproducing the paper
- **Numbers:** every evaluation output and training log is under [`results/`](results/) (checkpoints are not included). `paper_v2/seed_table.py`, `paper_v2/plot_kh1.py` and the other `paper_v2/*.py` scripts rebuild the tables and figures from them.
- **Training commands:** [`paper_v2/repro/commands.md`](paper_v2/repro/commands.md), regenerated from each run's logged configuration by `paper_v2/make_repro.py`. Appendix D of the paper lists hardware, software and compute.
- **Kernels:** training on Hopper needs Triton 3.7.1 or later with flash-linear-attention 0.5.2 (Appendix A).
- **Research log:** [`MAMBA3_HAIS_POC_PLAN.md`](MAMBA3_HAIS_POC_PLAN.md); Section 8 records every experiment in order, including the failures.

## Layout
- `poc_m3/`: from-scratch models and synthetic tasks (`model.py`, `layers.py`, `tasks.py`, `train.py`), the language-model pipeline (`lm/`), the grafts (`lfm2_hais.py`, `qwen_hais.py`) and the KH-1 chat, demo and loader.
- `results/`: raw evaluation JSONs and training logs behind every number in the paper.
- `paper_v2/`: the paper's LaTeX source and figures, and the scripts that rebuild them.
- `kh1/`: KH-1 notes, demo transcripts and the demo video script.
- `release/`: the Hugging Face model cards.
- `demo/books/`: three Project Gutenberg novels and the demo stream with the planted facts.

## Citation
```bibtex
@misc{khouidi2026kh1,
  title  = {Write Where Short Memory Cannot Reach: Distance-Invariant Fact Recall in Fixed-State Language Models},
  author = {Khouidi, Youssef},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.23447665},
  url    = {https://doi.org/10.5281/zenodo.23447665}
}
```

## License
- **Code:** Apache-2.0 ([`LICENSE`](LICENSE)).
- **Adapter weights:** each follows its base model's license: the LFM Open License v1.0 for the LFM2 grafts and Apache-2.0 for KH-1-Q-4B.
- **Books:** the Project Gutenberg texts are in the public domain in the United States.
- KH-1 is an independent research project, not affiliated with Liquid AI or the Qwen team.

## Acknowledgements
We thank Nebius, our compute partner through the NVIDIA Inception program, for the GPU credits that paid for the H200 and B200 compute.
