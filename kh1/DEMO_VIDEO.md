# KH-1 demo video runbook (RTX 3070 Ti, 8 GB)

These steps were checked end to end on NINJA-471 on 2026-10-14 (06:20 and 07:28 UTC). The results below are what the checkpoint actually does. The full transcripts are saved in `results/demo_logs/` and `kh1/demo_transcripts/`.

**Recommended model:** KH-1-1.2B v4 (`kh1_1p2b_v4.pt`, 253 MB of adapters on frozen LFM2-1.2B), run at write threshold τ 0.99. It is the best on the book demo (4/5).

The alternative is KH-1-350M (`kh1_350m_pilot6.pt`, τ 0.99). It is faster (32.5K tok/s, 0.84 GB) and better on FineWeb-Edu at 1M (0.90 vs 0.75 on fresh streams), but gets 3/5 on this book.

## Interactive version (for the video)
```bash
cd ~/projects/hais_engine
m3poc_env/bin/python poc_m3/kh1_chat.py --ckpt kh1/ckpt/kh1_1p2b_v4.pt --tau 0.99
```
Type these lines **in this order**:
```
Hi! Who are you?
/load demo/books/kh1_demo_book.txt
/mem
/ask What is Velthu's locker number? || Velthu's locker number is
/ask What is the wifi password at Kribrisen? || The wifi password at Kribrisen is
/ask What did Moraska say their safe combination is? || Their safe combination is
/ask What is the birth month of Galfen? || The birth month of Galfen is
/ask Officially, what does Sabtor's radio frequency read? || It reads
/mem
```
- **The book file:** `demo/books/kh1_demo_book.txt` is *War and Peace* + *Moby Dick* + *Pride and Prejudice*.
  - It is 1,318,920 tokens long.
  - Five facts are planted 2% of the way into *War and Peace* (token ~16.8K).
  - The questions are therefore ~1.30M tokens after the facts.
- **`/load`** reads the file in ~96 s (~13.7K tok/s). GPU memory stays at **2.60 GB** (peak 2.81) the whole time.
- **`/mem`** shows the archival state: **393,216 floats (1.6 MB), fixed**, before and after the questions.
- **Expected answers** (verified twice: interactive and scripted):

  | Question | Answer | Correct? |
  |---|---|---|
  | Velthu's locker number | 482 | yes |
  | Kribrisen's wifi password | thuzoka | yes |
  | Moraska's safe combination | 739561 | yes |
  | Galfen's birth month | September | yes |
  | Sabtor's radio frequency (phrasing never seen in training) | 2879 (truth: 287) | no |

- **Keep the order and the `|| prefix` parts.** Every chat turn is also written into the memory, so the model's own answers can disturb it. A run that asked Galfen's month first as free chat, and Velthu without a prefix, changed later answers (4826, 4785619): see `results/demo_logs/v4_interactive_other_order.txt`.
- **Generation speed:** about 25–47 tok/s in our runs. Generation re-runs the current chunk for each new token; reading is the fast part.
- **Tip:** run `nvidia-smi -l 1` in a second window during `/load` to show that memory does not grow.

## Scripted version (writes a JSON transcript)
```bash
m3poc_env/bin/python poc_m3/kh1_demo.py --ckpt kh1/ckpt/kh1_1p2b_v4.pt --tau 0.99 \
  --books demo/books/pg2600.txt demo/books/pg2701.txt demo/books/pg1342.txt --out /tmp/kh1_demo.json
```
- Expected: **4/5 exact**, peak 2.81 GB.
- Control: add `--no_memory`. The same model with the memory switched off gets **0/5** and invents answers ("12", "KRIBRISEN-WIFI", "123.456 MHz"). Transcript: `kh1/demo_transcripts/book_demo_kh1_1p2b_v4_NO_MEMORY_control_rtx3070ti.json`.

## Honest framing for the video
- **What KH-1 does:** it remembers specific facts it read (name → attribute → value) at any distance, in fixed memory.
- **What it does not do:** it does not re-read or summarise the whole book. LFM2 itself only sees the current 2K-token chunk.
- **Bigger model:** the Qwen3.5-4B version (KH-1-Q) recalls 0.90 at 1M on fresh streams and runs on this GPU with a 4-bit base (`--quant nf4`, 3.9 GB, about 4.1K tok/s). It gets 2/5 on this demo in 4-bit, and its free chat "thinks out loud", so the 1.2B model is the better live demo.
