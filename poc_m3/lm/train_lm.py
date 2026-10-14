"""Phase-3: train a tiny LM with carried recurrent state on FineWeb-Edu + spliced long-range facts.

  python train_lm.py --data /data/fwe --layout m3,hais,m3,hais,m3,hais,m3,hais --d_model 512 \
      --tokens 1_000_000_000 --out runs_lm/hais8 [--archival_heads 4 --min_decay 0.0217 --write_l1 0.05 --l1_scope archival]

Preemption-safe: checkpoints every --ckpt_every steps (model, optimizer, data pointers) and resumes.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa: E402
from lm_data import LaneStream, Prefetcher  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--layout", default=None, help="TinyLM block codes, e.g. gdn,hais,... (not used with --lfm2)")
p.add_argument("--d_model", type=int, default=512)
p.add_argument("--mem_heads", type=int, default=8)
p.add_argument("--archival_heads", type=int, default=None)
p.add_argument("--min_decay", type=float, default=None)
p.add_argument("--beta_bias", type=float, default=None)
p.add_argument("--gdn_half_life", type=float, default=None, help="cap the memory half-life (tokens) of gdn layers")
p.add_argument("--v_norm", action="store_true")
p.add_argument("--write_budget", type=int, default=None)
p.add_argument("--arch_tau", type=float, default=None)
p.add_argument("--write_l1", type=float, default=0.0)
p.add_argument("--l1_scope", default="archival", choices=["all", "archival"])
p.add_argument("--text_write_l1", type=float, default=0.0,
               help="L1 on archival write strength at ordinary-text tokens only (inserted facts/queries are free)")
p.add_argument("--bs", type=int, default=32)
p.add_argument("--T", type=int, default=1024)
p.add_argument("--tokens", type=float, default=1e9)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--ans_weight", type=float, default=1.0, help="extra loss weight on answer tokens")
p.add_argument("--p_fact", type=float, default=0.1)
p.add_argument("--p_query", type=float, default=0.1)
p.add_argument("--reset_every", type=int, default=0, help="reset carried state every N steps (0 = never)")
p.add_argument("--fact_rate", type=float, default=None, help="Poisson facts per lane-segment (v2 curriculum)")
p.add_argument("--query_rate", type=float, default=None, help="Poisson queries per lane-segment (v2 curriculum)")
p.add_argument("--drill_frac", type=float, default=0.0, help="fraction of lanes that are dense fact/query drills")
p.add_argument("--drill_k", type=int, default=8)
p.add_argument("--no_think", action="store_true", help="chat queries in non-thinking mode (Qwen3.5-4B template)")
p.add_argument("--natural_names", type=float, default=0.0, help="share of facts about real-looking names instead of made-up ones")
p.add_argument("--chat_frac", type=float, default=0.0, help="share of queries asked as chat turns (chat-template tokenizers)")
p.add_argument("--drill_gap", default="half", choices=["half", "far"],
               help="far: drill facts in the first 15%% of the segment, queries in the last 15%%")
p.add_argument("--max_age", type=int, default=200, help="max query age in segments")
p.add_argument("--age_warmup", type=int, default=0, help="steps over which max query age grows 1 -> max_age")
p.add_argument("--age_target", default="random", choices=["random", "loguniform", "loguniform_once"])
p.add_argument("--max_pending", type=int, default=64)
p.add_argument("--templates", default="secret", choices=["secret", "multi", "xl", "comp", "mix"], help="fact phrasings (lm_data.TEMPLATES / TEMPLATES_XL)")
p.add_argument("--ckpt_every", type=int, default=1000)
p.add_argument("--spike_guard", type=float, default=0.0,
               help="skip the update AND reset the carried states when grad norm > this x the running median (0 = off)")
p.add_argument("--compile", action="store_true", help="torch.compile each block's forward (~1.3x; fla kernels stay eager)")
p.add_argument("--init_from", default=None, help="fine-tune: start from these model weights (fresh optimizer/schedule/data)")
p.add_argument("--lfm2", default=None, help="graft archival memory onto this pretrained LFM2 model, e.g. LiquidAI/LFM2-350M")
p.add_argument("--lfm2_insert", default="7,11,15", help="LFM2 layer indices after which archival adapters are inserted")
p.add_argument("--lfm2_chunk", type=int, default=2048, help="LFM2 attention/conv span (constant memory); T must be a multiple")
p.add_argument("--lfm2_unfreeze", action="store_true", help="also train the LFM2 weights (default: only the adapters)")
p.add_argument("--text_write_bce", type=float, default=0.0,
               help="weight of a BCE-to-zero penalty on archival write logits at non-fact tokens (targets saturated text writes)")
p.add_argument("--lm_loss_frac", type=float, default=1.0,
               help="LM loss on this random fraction of positions (answer positions always): cheap when the base is frozen")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
assert a.lfm2 or a.layout, "--layout is required unless --lfm2 is given"
os.makedirs(a.out, exist_ok=True)
# multi-GPU: launch with `torchrun --nproc_per_node=8 train_lm.py ...`. --bs is per GPU; --tokens is the global budget.
# Every rank reads its own lanes (seed + 1000*rank) and carries its own recurrent state; rank 0 logs/checkpoints.
WORLD = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
if WORLD > 1:
    import torch.distributed as dist
    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"))  # gloo: lets 2 ranks share one GPU for tests
    LOCAL = int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()  # modulo: lets a 1-GPU box run a 2-rank test
    torch.cuda.set_device(LOCAL)
torch.manual_seed(a.seed)
dev = "cuda"

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.lfm2 or "openai-community/gpt2")
V = len(tok)  # ~ logits width, sizes the CE chunks
cfg = {"mem_heads": a.mem_heads, "mem_head_dim": 64}
if a.v_norm:
    cfg["v_norm"] = True
if a.write_budget:
    cfg["write_budget"] = a.write_budget
if a.arch_tau:
    cfg["arch_tau"] = a.arch_tau
if a.gdn_half_life:
    cfg["gdn_min_decay"] = math.log(2) / a.gdn_half_life
for k in ("archival_heads", "min_decay", "beta_bias"):
    if getattr(a, k) is not None:
        cfg[k] = getattr(a, k)
if a.lfm2:  # pretrained-model graft: LiquidAI LFM2 (lfm2_hais.py) or Qwen3.5 (qwen_hais.py)
    if "qwen" in a.lfm2.lower():
        from qwen_hais import QwenHais as Graft  # noqa: E402
    else:
        from lfm2_hais import Lfm2Hais as Graft  # noqa: E402
    model = Graft(a.lfm2, insert_after=[int(i) for i in a.lfm2_insert.split(",")], n_heads=a.mem_heads, head_dim=64,
                  tau=a.arch_tau or 0.5, chunk=a.lfm2_chunk, freeze_base=not a.lfm2_unfreeze).to(dev)
    cfg = model.cfg
else:
    model = TinyLM(50304, a.d_model, a.layout, cfg).to(dev)  # vocab padded to a multiple of 64
if a.compile:  # per-block, so checkpoints/state dicts are unchanged (no _orig_mod prefixes)
    for blk in model.blocks:
        blk.forward = torch.compile(blk.forward)
net = model
if WORLD > 1:
    from torch.nn.parallel import DistributedDataParallel as DDP
    net = DDP(model, device_ids=[LOCAL], find_unused_parameters=True)
steps = int(a.tokens // (a.bs * a.T * WORLD))
decay, no_decay = [], []
for n, p_ in model.named_parameters():
    if not p_.requires_grad:  # frozen LFM2 base
        continue
    (no_decay if p_.dim() < 2 or "bias" in n or "A_log" in n or n.endswith(".D") else decay).append(p_)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
warm = max(1, steps // 50)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
data = LaneStream(os.path.join(a.data, "train.bin"), tok, a.bs, a.T, seed=a.seed + 1000 * RANK, p_fact=a.p_fact, p_query=a.p_query,
                  max_age=a.max_age, fact_rate=a.fact_rate, query_rate=a.query_rate, drill_frac=a.drill_frac,
                  drill_k=a.drill_k, age_warmup=a.age_warmup, age_target=a.age_target, max_pending=a.max_pending,
                  templates=a.templates, drill_gap=a.drill_gap, chat_frac=a.chat_frac,
                  natural_names=a.natural_names, no_think=a.no_think)
log = open(os.path.join(a.out, "log.jsonl"), "a") if RANK == 0 else None
ck = os.path.join(a.out, "ckpt.pt")
ck_data = os.path.join(a.out, f"data_rank{RANK}.pt")
step = 0
if os.path.exists(ck):
    s = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(s["model"], strict=not a.lfm2); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    step, data.ptr, data.step = s["step"], s["ptr"], s["data_step"]
    if os.path.exists(ck_data):
        d = torch.load(ck_data, weights_only=False)
        data.ptr, data.step = d["ptr"], d["data_step"]
    print("resumed at step", step, flush=True)
elif a.init_from:
    model.load_state_dict(torch.load(a.init_from, map_location=dev, weights_only=False)["model"], strict=not a.lfm2)
    print("initialised from", a.init_from, flush=True)


def emit(**kw):
    if RANK != 0:
        return
    log.write(json.dumps(kw) + "\n"); log.flush()
    print(json.dumps(kw), flush=True)


emit(event="config", n_params=model.n_params(), steps=steps, world=WORLD, **vars(a))
def _ce(hc, yc):
    return F.cross_entropy(model.head(hc).float().flatten(0, 1), yc.flatten(), reduction="none").view_as(yc)


def _ce1(hc, yc):  # flat [N, D] positions
    return F.cross_entropy(model.head(hc).float(), yc, reduction="none")


states, t0, feed = None, time.time(), None
from collections import deque  # noqa: E402
gn_hist, n_skip = deque(maxlen=200), 0
while step < steps:
    model.train()
    if feed is None:  # started lazily, after a resume has restored the stream position
        feed = Prefetcher(data)
    x, y, ans, mem_b, queries_b, ages_b, ptr_b, dstep_b = feed.next()
    x, y, ans = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), ans.to(dev, non_blocking=True)
    if a.reset_every and step % a.reset_every == 0:
        states = None
    with torch.autocast("cuda", torch.bfloat16):
        h, new_states = net(x, states, head=False)
        # chunked + checkpointed CE: never materialises the full [B, T, V] fp32 logits (same values as one shot)
        if a.lm_loss_frac < 1:  # sampled LM loss: random positions + every answer position
            sel = (torch.rand(y.shape, device=dev) < a.lm_loss_frac) | ans
            hs, ys, an = h[sel], y[sel], ans[sel]
            cs = max(512, (1 << 29) // V)  # positions per CE chunk: <= 2 GB of fp32 logits (8192 for LFM2, 2162 for Qwen3.5)
            ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce1, hs[i:i + cs], ys[i:i + cs], use_reentrant=False)
                            for i in range(0, hs.shape[0], cs)])
            lm_loss = ce[~an].mean() if (~an).any() else ce.new_zeros(())
            ans_loss = ce[an].mean() if an.any() else ce.new_zeros(())
        else:
            ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce, h[:, t0:t0 + 256], y[:, t0:t0 + 256], use_reentrant=False)
                            for t0 in range(0, h.shape[1], 256)], 1)
            lm_loss = ce.mean()
            ans_loss = ce[ans].mean() if ans.any() else ce.new_zeros(())
        loss = lm_loss + a.ans_weight * ans_loss
        pen = model.archival_write_rate() if a.l1_scope == "archival" else model.write_rate()
        if a.write_l1 and pen is not None:
            loss = loss + a.write_l1 * pen
        if a.text_write_bce:  # BCE toward "no write" at non-fact tokens, on the logits: softplus(l) = -log(1 - beta).
            # Its logit-gradient is beta, so it hits the saturated text writes (beta ~ 0.98) that erase facts; the L1 on
            # beta has gradient beta(1 - beta) there and barely moves them (MAMBA3_HAIS_POC_PLAN.md 8.40).
            txt = (~mem_b).to(dev, non_blocking=True)[..., None]
            pens = [(F.softplus(b.mixer.last_blogit[..., :b.mixer.archival_heads].float()) * txt).sum() /
                    (txt.sum() * b.mixer.archival_heads).clamp_min(1)
                    for b in model.blocks if getattr(b.mixer, "archival_heads", 0) and hasattr(b.mixer, "last_blogit")]
            loss = loss + a.text_write_bce * torch.stack(pens).mean()
        if a.text_write_l1:
            txt = (~mem_b).to(dev, non_blocking=True)[..., None]  # [B, T, 1]
            pens = [(b.mixer.last_beta[..., :b.mixer.archival_heads] * txt).sum() /
                    (txt.sum() * b.mixer.archival_heads).clamp_min(1)
                    for b in model.blocks if getattr(b.mixer, "archival_heads", 0) and hasattr(b.mixer, "last_beta")]
            loss = loss + a.text_write_l1 * torch.stack(pens).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
    g = gn.item()
    if a.spike_guard and len(gn_hist) >= 20 and not (g <= a.spike_guard * sorted(gn_hist)[len(gn_hist) // 2]):
        # a rare runaway (the 494M v2 run hit grad norm 1.1e5, loss 3.9 -> 12.6, from its carried states): drop this
        # update and restart every lane's recurrent state, so neither the weights nor the states inherit it
        opt.zero_grad(set_to_none=True)
        states = None
        n_skip += 1
        emit(event="spike", step=step, grad_norm=round(g, 1), lm_loss=round(lm_loss.item(), 4), skipped=n_skip)
    else:
        opt.step()
        gn_hist.append(g)
        states = [tuple(z.detach() if torch.is_tensor(z) else z for z in st) for st in new_states]
    sched.step()
    step += 1
    if step % 50 == 0 or step == steps:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ans_acc = (model.head(h.detach()[ans]).argmax(-1) == y[ans]).float().mean().item() if ans.any() else None
        wr = model.write_rate()
        # exact-match accuracy of whole answers, bucketed by query age (segments since the fact was written / refreshed)
        age_acc = {}
        if queries_b:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for lo, hi in ((0, 4), (4, 32), (32, 256), (256, 10 ** 9)):
                    qs = [q for q in queries_b if lo <= q[2] < hi]
                    if qs:
                        bi = torch.tensor([b for b, pos, _ in qs for _ in pos], device=dev)
                        pi = torch.tensor([p_ for _, pos, _ in qs for p_ in pos], device=dev)
                        ok = (model.head(h.detach()[bi, pi]).argmax(-1) == y[bi, pi]).tolist()
                        k, good = 0, 0
                        for _, pos, _ in qs:
                            good += all(ok[k:k + len(pos)]); k += len(pos)
                        age_acc[f"{lo}+"] = [round(good / len(qs), 3), len(qs)]
        emit(event="train", step=step, lm_loss=round(lm_loss.item(), 4), ans_loss=round(ans_loss.item(), 4),
             ans_acc=None if ans_acc is None else round(ans_acc, 4), write_rate=None if wr is None else round(wr.item(), 4),
             grad_norm=round(gn.item(), 3), query_ages=ages_b[:4], age_acc=age_acc, lr=sched.get_last_lr()[0],
             tok_per_s=round(step * a.bs * a.T * WORLD / (time.time() - t0)), elapsed=round(time.time() - t0, 1))
    if step % a.ckpt_every == 0 or step == steps:
        torch.save({"ptr": ptr_b, "data_step": dstep_b}, ck_data)  # position right after the consumed batch
        if RANK != 0:
            continue
        torch.save({"model": model.adapter_state_dict() if a.lfm2 and not a.lfm2_unfreeze else model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "ptr": ptr_b, "data_step": dstep_b, "args": vars(a), "cfg": cfg}, ck + ".tmp")
        if os.path.exists(ck):
            os.replace(ck, ck + ".prev")  # keep the previous checkpoint: lets a run be rolled back past a bad stretch
        os.replace(ck + ".tmp", ck)
emit(event="done", train_time_s=round(time.time() - t0, 1))
if WORLD > 1:
    dist.destroy_process_group()
