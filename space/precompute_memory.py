"""Read the 1.3M-token demo book through KH-1 once (GPU) and save the fixed-size memory; then check the five demo
questions on GPU and, if --cpu_check, on the CPU path from the saved file."""
import argparse, json, sys, time
import torch
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kh1"))
from kh1_load import load_adapters
from lfm2_hais import Lfm2Hais
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--book", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tau", type=float, default=0.99); ap.add_argument("--cpu_check", action="store_true")
a = ap.parse_args()
Q = [("What is Velthu's locker number?", "Velthu's locker number is", "482"),
     ("What is the wifi password at Kribrisen?", "The wifi password at Kribrisen is", "thuzoka"),
     ("What did Moraska say their safe combination is?", "Their safe combination is", "739561"),
     ("What is the birth month of Galfen?", "The birth month of Galfen is", "September"),
     ("Officially, what does Sabtor's radio frequency read?", "It reads", "287")]
cfg, sd = load_adapters(a.model)
tok = AutoTokenizer.from_pretrained(cfg["name"])
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
def build(dev, dtype):
    m = Lfm2Hais(cfg["name"], insert_after=cfg["insert_after"], n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], tau=a.tau,
                 chunk=cfg["chunk"], freeze_base=True, dtype=dtype).to(dev).eval()
    m.load_state_dict(sd, strict=False)
    for ad in m.adapters.values(): ad.mixer.arch_tau = a.tau
    return m
@torch.no_grad()
def ask(m, st, q, prefix, dev, max_new=12):
    turn = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = enc("\n" + turn.replace(tok.bos_token or "\0", "") + prefix); out = []
    for _ in range(max_new):
        with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
            h, _ = m(torch.tensor(ids[-m.chunk:], device=dev)[None], st, head=False)
        nxt = int(m.head(h[:, -1]).argmax(-1))
        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
        out.append(nxt); ids.append(nxt)
    return tok.decode(out).strip()
m = build("cuda", torch.bfloat16)
# same sequence as the verified video demo (kh1_demo.py): one chat turn, committed to memory, then the book
st = None
hi = tok.apply_chat_template([{"role": "user", "content": "Hi! Who are you?"}], tokenize=False, add_generation_prompt=True)
hi_ids = enc("\n" + hi.replace(tok.bos_token or "\0", ""))
r = ask(m, None, "Hi! Who are you?", "", "cuda", max_new=40); print("hi>", r)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    _, st = m(torch.tensor((hi_ids + enc(r))[-m.chunk:], device="cuda")[None], st, head=False)
ids = enc(open(a.book, encoding="utf-8", errors="ignore").read()); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(ids), m.chunk):
        with torch.autocast("cuda", torch.bfloat16):
            _, st = m(torch.tensor(ids[i:i + m.chunk], device="cuda")[None], st, head=False)
cpu_st = [tuple(None if x is None else x.detach().float().cpu() for x in s) for s in st]
nbytes = sum(x.numel() * 4 for s in cpu_st for x in s if x is not None)
torch.save({"state": cpu_st, "tokens": len(ids), "model": a.model, "tau": a.tau}, a.out)
print(f"read {len(ids):,} tokens in {time.time()-t0:.0f}s; memory file {nbytes/1e6:.2f} MB -> {a.out}")
gpu = [ask(m, st, q, p, "cuda") for q, p, _ in Q]
print("GPU:", [(g, t) for g, (_, _, t) in zip(gpu, Q)])
if a.cpu_check:
    del m; torch.cuda.empty_cache()
    mc = build("cpu", torch.float32); stc = torch.load(a.out)["state"]; t0 = time.time()
    cpu = [ask(mc, stc, q, p, "cpu") for q, p, _ in Q]
    print("CPU:", [(c, t) for c, (_, _, t) in zip(cpu, Q)], f"({time.time()-t0:.0f}s for 5 questions)")
