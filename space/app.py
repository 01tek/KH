"""KH-1 demo Space: LFM2-1.2B with a 1.9 MB archival memory that has read 1.3 million tokens (three novels).

Runs on CPU. The memory was computed once by streaming War and Peace + Moby Dick + Pride and Prejudice (with five
made-up facts planted 2% into War and Peace) through KH-1; this Space loads that fixed-size state and answers from it.
Each visitor gets a private copy; every question and answer is written into it, as in a real chat.
"""
import copy
import os
import sys
import time

import gradio as gr
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kh1"))
from kh1_load import load_adapters  # noqa: E402
from lfm2_hais import Lfm2Hais  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
TAU = 0.99
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
if not os.path.exists(os.path.join(MODEL_DIR, "adapters.safetensors")):  # not bundled (GitHub / Colab): fetch from the Hub
    from huggingface_hub import snapshot_download
    MODEL_DIR = snapshot_download("01TEK/KH-1-1.2B", allow_patterns=["config.json", "adapters.safetensors"])
cfg, sd = load_adapters(MODEL_DIR)
model = Lfm2Hais(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], tau=TAU,
                 chunk=cfg["chunk"], freeze_base=True, dtype=torch.float32).eval()
model.load_state_dict(sd, strict=False)
for ad in model.adapters.values():
    ad.mixer.arch_tau = TAU
tok = AutoTokenizer.from_pretrained(cfg["name"])
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
END = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
BOOK = torch.load(os.path.join(HERE, "memory_three_novels_1p2b.pt"))
BOOK_MB = sum(x.numel() * 4 for s in BOOK["state"] for x in s if x is not None) / 1e6
MAX_TEXT_TOKENS = 6000


@torch.no_grad()
def feed(ids, st):
    for i in range(0, len(ids), model.chunk):
        _, st = model(torch.tensor(ids[i:i + model.chunk])[None], st, head=False)
    return st


@torch.no_grad()
def answer(question, prefix, st, max_new=24):
    turn = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    ids = enc("\n" + turn.replace(tok.bos_token or "\0", "") + prefix)
    out = []
    for _ in range(max_new):
        h, _ = model(torch.tensor(ids[-model.chunk:])[None], st, head=False)
        nxt = int(model.head(h[:, -1]).argmax(-1))
        if nxt in END:
            break
        out.append(nxt)
        ids.append(nxt)
    return " ".join((prefix + " " + tok.decode(out)).split()), feed(ids, st)  # the turn is written into memory, as in a chat


def ask_book(question, prefix, st):
    if not question.strip():
        return "Type a question first.", st
    st = st if st is not None else copy.deepcopy(BOOK["state"])
    t0 = time.time()
    reply, st = answer(question.strip(), prefix.strip(), st)
    return f"{reply}\n\n({time.time() - t0:.1f} s on CPU; memory still {BOOK_MB:.2f} MB)", st


def reset_book():
    return None, "Memory reset to the state right after reading the three novels."


def read_text(text):
    ids = enc(text or "")
    if not ids:
        return None, "Paste some text first."
    if len(ids) > MAX_TEXT_TOKENS:
        ids = ids[:MAX_TEXT_TOKENS]
    t0 = time.time()
    st = feed(ids, None)
    return st, f"Read {len(ids):,} tokens in {time.time() - t0:.0f} s. The model now sees none of it directly; only the fixed-size memory has it."


def ask_text(question, prefix, st):
    if st is None:
        return "Read a text first.", st
    reply, st = answer(question.strip(), prefix.strip(), st)
    return reply, st


EXAMPLES = [
    ["What is Velthu's locker number?", "Velthu's locker number is"],
    ["What is the wifi password at Kribrisen?", "The wifi password at Kribrisen is"],
    ["What did Moraska say their safe combination is?", "Their safe combination is"],
    ["What is the birth month of Galfen?", "The birth month of Galfen is"],
    ["Officially, what does Sabtor's radio frequency read?", "It reads"],
]

with gr.Blocks(title="KH-1: a 1.9 MB memory of three novels") as demo:
    gr.Markdown(f"""# KH-1: a {BOOK_MB:.1f} MB memory of three novels
LFM2-1.2B (frozen) + KH-1 archival memory read **{BOOK['tokens']:,} tokens**: *War and Peace*, *Moby Dick* and *Pride and Prejudice*,
with five made-up facts hidden 2% into the first book. Everything it read now lives in a fixed **{BOOK_MB:.2f} MB** memory.
The base model only ever sees the current 2K-token chunk, so it can answer only through that memory.

Ask the five questions **in the listed order** (each answer is also written into the memory, as in a chat; other orders can change later answers).
On our 8 GB GPU this exact model answers 4 of 5; the fifth fact uses a phrasing never seen in training. With the memory switched off it answers 0 of 5.
Research preview: it recalls facts, not whole books. Paper and code: [github.com/01tek/KH](https://github.com/01tek/KH).""")
    with gr.Tab("Three novels, 1.3M tokens"):
        book_state = gr.State(None)
        with gr.Row():
            q = gr.Textbox(label="Question", scale=3)
            p = gr.Textbox(label="Answer starts with (optional)", scale=2)
        go = gr.Button("Ask the memory", variant="primary")
        out = gr.Textbox(label="Answer", lines=3)
        gr.Examples(EXAMPLES, inputs=[q, p])
        reset = gr.Button("Reset memory")
        go.click(ask_book, [q, p, book_state], [out, book_state])
        reset.click(reset_book, None, [book_state, out])
    with gr.Tab("Your own text"):
        gr.Markdown(f"Paste a text (up to {MAX_TEXT_TOKENS:,} tokens on this free CPU), plant a fact like *The code for the blue door is 4417.*, then ask about it. "
                    "KH-1 was trained on facts of the form \"name's attribute is value\"; other kinds of questions are outside what it learned.")
        text_state = gr.State(None)
        txt = gr.Textbox(label="Text", lines=10)
        read = gr.Button("Read into memory")
        status = gr.Markdown()
        q2 = gr.Textbox(label="Question")
        p2 = gr.Textbox(label="Answer starts with (optional)")
        go2 = gr.Button("Ask", variant="primary")
        out2 = gr.Textbox(label="Answer", lines=2)
        read.click(read_text, [txt], [text_state, status])
        go2.click(ask_text, [q2, p2, text_state], [out2, text_state])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(share=os.environ.get("KH1_SHARE") == "1")  # KH1_SHARE=1: public link from Colab
