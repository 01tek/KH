---
title: KH-1 Demo
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: A 1.9 MB memory of three novels (1.3M tokens), on CPU
---

# KH-1 demo

LFM2-1.2B (frozen) with the KH-1 archival memory. The memory was computed once by streaming 1,318,920 tokens
(*War and Peace*, *Moby Dick*, *Pride and Prejudice*, with five made-up facts planted 2% into the first book) through
the model; this Space loads that fixed 1.87 MB state and answers from it on CPU.

- Paper and code: https://github.com/01tek/KH
- Adapters: https://huggingface.co/01TEK/KH-1-1.2B
- Base model: LiquidAI/LFM2-1.2B (LFM Open License v1.0). Not affiliated with Liquid AI.
- Research preview: KH-1 recalls stored facts ("a name's attribute is a value"), not whole documents.
