"""KH-1 live: read a whole book (up to ~2M tokens) into a fixed 1.9 MB memory on a ZeroGPU H200 slice, then ask about facts
you planted anywhere in it. Memory-off control included. LFM2-1.2B (frozen) + KH-1 archival adapters.
"""
import os
import sys
import time

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kh1"))
from kh1_load import load_adapters  # noqa: E402
from lfm2_hais import Lfm2Hais  # noqa: E402

TAU = 0.99
cfg, sd = load_adapters(os.path.join(HERE, "model"))
model = Lfm2Hais(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], tau=TAU,
                 chunk=cfg["chunk"], freeze_base=True, dtype=torch.bfloat16).eval()
model.load_state_dict(sd, strict=False)
for ad in model.adapters.values():
    ad.mixer.arch_tau = TAU
model.to("cuda")  # ZeroGPU attaches a GPU only inside @spaces.GPU calls
tok = AutoTokenizer.from_pretrained(cfg["name"])
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
END = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
C = model.chunk
MAX_TOKENS = 2_000_000
BOOKS = [os.path.join(HERE, "books", f) for f in ("pg2600.txt", "pg2701.txt", "pg1342.txt")]
DEFAULT_FACTS = ("Velthu's locker number is 482. The wifi password at Kribrisen is thuzoka. "
                 "Moraska told me that their safe combination is 739561. The birth month of Galfen is September.")


def to_cpu(st):
    return [tuple(None if x is None else x.detach().float().cpu() for x in s) for s in st]


def to_gpu(st):
    return [tuple(None if x is None else x.to("cuda") for x in s) for s in st]


ALPHA = {k: ad.alpha.data.clone() for k, ad in model.adapters.items()}


def memory_on(on):
    for k, ad in model.adapters.items():
        ad.alpha.data.copy_(ALPHA[k] if on else torch.zeros_like(ALPHA[k]))


@torch.no_grad()
def feed(ids, st):
    for i in range(0, len(ids), C):
        with torch.autocast("cuda", torch.bfloat16):
            _, st = model(torch.tensor(ids[i:i + C], device="cuda")[None], st, head=False)
    return st


@spaces.GPU(duration=240)
def read(upload, facts, where):
    if upload is not None:
        text = open(upload, encoding="utf-8", errors="ignore").read()
        source = os.path.basename(upload)
    else:
        text = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in BOOKS)
        source = "War and Peace + Moby Dick + Pride and Prejudice"
    cut = int(len(text) * where / 100)
    cut = max(0, text.rfind("\n\n", 0, cut) + 2) if cut else 0
    planted = (facts or "").strip()
    if planted:
        text = text[:cut] + planted + "\n\n" + text[cut:]
    ids = enc(text)[:MAX_TOKENS]
    pos = len(enc(text[:cut])) if planted else None
    memory_on(True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    st = feed(ids, None)
    torch.cuda.synchronize(); dt = time.time() - t0
    mb = sum(x.numel() * 4 for s in st for x in s if x is not None) / 1e6
    report = (f"**Read {len(ids):,} tokens** of *{source}* in **{dt:.0f} s** ({len(ids) / dt:,.0f} tokens/s) on one GPU slice.  \n"
              f"Peak GPU memory **{torch.cuda.max_memory_allocated() / 1e9:.2f} GB** (model included), the same at token 1 and at the end.  \n"
              f"Everything it read is now in a fixed **{mb:.2f} MB** memory. "
              + (f"Your facts sit at token ~{pos:,}, so questions about them come ~{len(ids) - pos:,} tokens later." if planted else ""))
    return to_cpu(st), report


@spaces.GPU(duration=60)
def ask(question, prefix, st, use_memory):
    if st is None:
        return "Read a book first.", st
    memory_on(use_memory)
    g = to_gpu(st) if use_memory else None
    turn = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    ids = enc("\n" + turn.replace(tok.bos_token or "\0", "") + (prefix or "").strip())
    out = []
    with torch.no_grad():
        for _ in range(24):
            with torch.autocast("cuda", torch.bfloat16):
                h, _ = model(torch.tensor(ids[-C:], device="cuda")[None], g, head=False)
            nxt = int(model.head(h[:, -1]).argmax(-1))
            if nxt in END:
                break
            out.append(nxt)
            ids.append(nxt)
        if use_memory:
            g = feed(ids, g)  # the turn is written into memory, as in a chat
    memory_on(True)
    reply = " ".join(((prefix or "") + " " + tok.decode(out)).split())
    return (reply if use_memory else reply + "\n\n(memory switched off: the model sees only your question)"), (to_cpu(g) if use_memory else st)


EXAMPLES = [["What is Velthu's locker number?", "Velthu's locker number is"],
            ["What is the wifi password at Kribrisen?", "The wifi password at Kribrisen is"],
            ["What did Moraska say their safe combination is?", "Their safe combination is"],
            ["What is the birth month of Galfen?", "The birth month of Galfen is"]]

with gr.Blocks(title="KH-1 live: read a whole book into 1.9 MB") as demo:
    gr.Markdown("""# KH-1 live: read a whole book into a 1.9 MB memory
A frozen 1.2B model (LiquidAI LFM2-1.2B) reads text in 2,048-token chunks and **never sees earlier chunks again**. Our KH-1 archival
memory, three small adapter layers with a fixed 1.9 MB state, carries facts across them. Read a million tokens, then ask about a fact
from the first pages. Switch the memory off to see the same model fail.

1. Keep the default (three novels, 1.3M tokens) or upload your own `.txt` (up to 2M tokens). Edit the facts to plant, and where.
2. **Read** (about 1–2 minutes for 1.3M tokens).
3. Ask about the planted facts. KH-1 learned facts of the form *"a name's attribute is a value"*; it recalls facts, not summaries.

Research preview from 01TEK R&D · paper, code and every eval log: [github.com/01tek/KH](https://github.com/01tek/KH) · CPU demo: [01TEK/KH-1-demo](https://huggingface.co/spaces/01TEK/KH-1-demo)""")
    st = gr.State(None)
    with gr.Row():
        up = gr.File(label="Your book (.txt), optional", file_types=[".txt"], type="filepath")
        with gr.Column():
            facts = gr.Textbox(label="Facts to plant", value=DEFAULT_FACTS, lines=3)
            where = gr.Slider(0, 100, value=2, step=1, label="Plant them at (% into the text)")
    read_btn = gr.Button("Read into memory", variant="primary")
    report = gr.Markdown()
    with gr.Row():
        q = gr.Textbox(label="Question", scale=3)
        p = gr.Textbox(label="Answer starts with (optional)", scale=2)
    use_mem = gr.Checkbox(value=True, label="Memory on")
    ask_btn = gr.Button("Ask")
    out = gr.Textbox(label="Answer", lines=3)
    gr.Examples(EXAMPLES, inputs=[q, p])
    read_btn.click(read, [up, facts, where], [st, report])
    ask_btn.click(ask, [q, p, st, use_mem], [out, st])

demo.queue(max_size=20).launch()
