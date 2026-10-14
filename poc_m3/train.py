"""Train a tiny LM on the fact-stream task at a short length, then evaluate recall
by streaming far longer sequences in fixed-size chunks (O(1) memory).

Example:
  python train.py --layout m3,hais,m3,m3 --steps 2000 --L 1024 --kmax 8 \
      --eval_lens 1024,4096,16384,65536,262144,1048576 --out runs/m3hais_s0
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model import TinyLM
from tasks import Vocab, fact_stream

p = argparse.ArgumentParser()
p.add_argument("--layout", default="m3,m3,m3,m3")
p.add_argument("--d_model", type=int, default=256)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--L", type=int, default=1024)
p.add_argument("--L_start", type=int, default=None,
               help="length curriculum: start here and double up to --L over the first --curriculum_frac of steps")
p.add_argument("--curriculum_frac", type=float, default=0.5)
p.add_argument("--kmax", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--decoy", type=float, default=0.0)
p.add_argument("--eval_decoy", type=float, default=None, help="decoy rate at eval (default: same as --decoy)")
p.add_argument("--state_passing", type=float, default=0.0,
               help="prob. of starting a batch from the previous batch's final state (Buitrago & Gu 2025)")
p.add_argument("--write_l1", type=float, default=0.0, help="penalty on mean write strength of delta mixers")
p.add_argument("--l1_scope", default="all", choices=["all", "archival"],
               help="apply --write_l1 to all delta heads, or only to archival (no-decay) heads")
p.add_argument("--archival_heads", type=int, default=None)
p.add_argument("--write_budget", type=int, default=None, help="archival writes allowed per 64-token block per head")
p.add_argument("--v_norm", action="store_true")
p.add_argument("--write_block", type=int, default=None)
p.add_argument("--arch_tau", type=float, default=None)
p.add_argument("--mem_heads", type=int, default=None, help="delta-memory heads per layer (archival = half unless --archival_heads)")
p.add_argument("--tau", type=float, default=None, help="HAIS-X store threshold on beta (default 0.05)")
p.add_argument("--null_sink", action="store_true", help="HAIS-X: add a fixed-logit null slot to the episodic softmax")
p.add_argument("--min_decay", type=float, default=None,
               help="force non-archival heads to forget at least this fast (log-decay per token), e.g. 0.0217 = half-life 32")
p.add_argument("--store_cap", type=int, default=None, help="HAIS-X FIFO cap per (sequence, head) (default 16384)")
p.add_argument("--beta_bias", type=float, default=None)
p.add_argument("--eval_lens", default="1024,4096,16384,65536")
p.add_argument("--eval_K", default="1,4,8")
p.add_argument("--eval_n", type=int, default=128, help="queries per (L, K) cell, approx.")
p.add_argument("--eval_modes", default="uniform,early")
p.add_argument("--chunk", type=int, default=4096)
p.add_argument("--out", required=True)
p.add_argument("--eval_only", action="store_true", help="load <out>/model.pt and only run the eval")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
dev = "cuda"
V = Vocab()
cfg = {}
if args.archival_heads is not None:
    cfg["archival_heads"] = args.archival_heads
if args.beta_bias is not None:
    cfg["beta_bias"] = args.beta_bias
if args.write_budget is not None:
    cfg["write_budget"] = args.write_budget
if args.v_norm:
    cfg["v_norm"] = True
if args.arch_tau is not None:
    cfg["arch_tau"] = args.arch_tau
if args.write_block is not None:
    cfg["write_block"] = args.write_block
if args.mem_heads is not None:
    cfg["mem_heads"] = args.mem_heads
if args.tau is not None:
    cfg["tau"] = args.tau
if args.min_decay is not None:
    cfg["min_decay"] = args.min_decay
if args.null_sink:
    cfg["null_sink"] = True
if args.store_cap is not None:
    cfg["store_cap"] = args.store_cap
model = TinyLM(V.size, args.d_model, args.layout, cfg).to(dev)
if args.eval_only:
    model.load_state_dict(torch.load(os.path.join(args.out, "model.pt"), map_location=dev)["model"])
    args.steps = 0
log = open(os.path.join(args.out, "log.jsonl"), "a" if args.eval_only else "w")


def emit(**kw):
    print(json.dumps(kw), flush=True)
    log.write(json.dumps(kw) + "\n")
    log.flush()


emit(event="config", n_params=model.n_params(), **vars(args))

decay, no_decay = [], []
for n, p_ in model.named_parameters():
    (no_decay if p_.dim() < 2 or "bias" in n or "A_log" in n or n.endswith(".D") else decay).append(p_)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95))
warm = max(1, args.steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps)))))
gen = torch.Generator(device=dev).manual_seed(1000 + args.seed)  # training stream
stages = []
if args.L_start and args.L_start < args.L:
    s = args.L_start
    while s < args.L:
        stages.append(s)
        s *= 2
n_cur = int(args.steps * args.curriculum_frac)


def train_len(step):
    if not stages or step >= n_cur:
        return args.L
    return stages[step * len(stages) // max(1, n_cur)]


carry = None
t0 = time.time()
for step in range(args.steps):
    model.train()
    K = int(torch.randint(1, args.kmax + 1, (1,), generator=gen, device=dev))
    L_now = train_len(step)
    K = min(K, (L_now - 2) // 5)  # curriculum: early short sequences can't hold kmax facts
    ids, labels, _ = fact_stream(args.bs, L_now, K, vocab=V, decoy_rate=args.decoy, device=dev, gen=gen)
    init = carry if (carry is not None and torch.rand(1).item() < args.state_passing) else None
    with torch.autocast("cuda", torch.bfloat16):
        logits, states = model(ids, init)
        loss = F.cross_entropy(logits.float().view(-1, V.size), labels.view(-1), ignore_index=-100)
        wr = model.write_rate()
        wr_pen = model.archival_write_rate() if args.l1_scope == "archival" else wr
        total = loss + (args.write_l1 * wr_pen if (wr_pen is not None and args.write_l1) else 0.0)
    opt.zero_grad(set_to_none=True)
    total.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if args.state_passing:
        carry = [tuple(x.detach() if torch.is_tensor(x) else x for x in st) for st in states]
    if step % 100 == 0 or step == args.steps - 1:
        with torch.no_grad():
            m = labels != -100
            acc = (logits.argmax(-1)[m] == labels[m]).float().mean().item()
        emit(event="train", step=step, L=L_now, K=K, loss=round(loss.item(), 4), acc=round(acc, 4),
             write_rate=None if wr is None else round(wr.item(), 4), grad_norm=round(gn.item(), 3),
             elapsed=round(time.time() - t0, 1))

train_time = time.time() - t0
if not args.eval_only:
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "model.pt"))


@torch.no_grad()
def eval_cell(L, K, mode, n_queries, seed):
    """Stream B sequences of length L through the model in chunks; score answers only."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)  # disjoint from the training stream
    B = max(1, min(16, (1 << 21) // L))  # cap tokens per eval batch at 2M
    correct, total, by_dist, store = 0, 0, {}, []
    wf = 1.0 if mode == "uniform" else 0.05
    while total < n_queries:
        ids, labels, dist = fact_stream(B, L, K, vocab=V, write_frac=wf, decoy_rate=args.decoy if args.eval_decoy is None else args.eval_decoy, device=dev, gen=g)
        st = None
        preds = torch.full_like(labels, -1)
        for s in range(0, L, args.chunk):
            with torch.autocast("cuda", torch.bfloat16):
                h, st = model(ids[:, s:s + args.chunk], st, head=False)
            m = labels[:, s:s + args.chunk] != -100
            if m.any():
                preds[:, s:s + args.chunk][m] = model.head(h[m]).argmax(-1)
        for blk, stt in zip(model.blocks, st):  # episodic store size per sequence at the end of the stream
            if hasattr(blk.mixer, "store_size"):
                store.append(blk.mixer.store_size(stt) / (B * blk.mixer.h))
        m = labels != -100
        hit = (preds[m] == labels[m]).view(B, -1)  # [B, Q], same order as dist
        correct += hit.sum().item()
        total += hit.numel()
        for d, h_ in zip(dist.flatten().tolist(), hit.flatten().tolist()):
            b = 1 << max(0, int(math.log2(max(d, 1))))
            c = by_dist.setdefault(b, [0, 0])
            c[0] += h_
            c[1] += 1
    by_d = {k: round(v[0] / v[1], 4) for k, v in sorted(by_dist.items())}
    return correct / total, total, by_d, (round(sum(store) / len(store), 1) if store else None)


t1 = time.time()
summary = {}
for mode in args.eval_modes.split(","):
    for L in map(int, args.eval_lens.split(",")):
        for K in map(int, args.eval_K.split(",")):
            n_q = max(16, int(args.eval_n * min(1.0, 65536 / L)))  # fewer queries at very long L
            acc, n, by_d, store_per_head = eval_cell(L, K, mode, n_q, seed=777 + L + K)
            summary[f"{mode}/L{L}/K{K}"] = acc
            emit(event="eval", mode=mode, L=L, K=K, acc=round(acc, 4), n=n, by_dist=by_d,
                 store_per_head=store_per_head)  # HAIS-X: mean episodic entries per (sequence, head) at stream end
emit(event="done", train_time_s=round(train_time, 1), eval_time_s=round(time.time() - t1, 1),
     peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
json.dump({"args": vars(args), "n_params": model.n_params(), "summary": summary},
          open(os.path.join(args.out, "summary.json"), "w"), indent=2)
