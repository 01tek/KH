"""CPU replica of s1_scaling_v2.token_eval for Arm A (no adapter).
hidden_states[0] only depends on block 0, so the model is truncated to 1 layer.
Usage: python check_pilot_armA.py fp32|bf16 N_EVAL
"""
import os, sys, time, random, statistics
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoConfig

torch.set_num_threads(8)
MID = "state-spaces/mamba-130m-hf"
dt = torch.float32 if sys.argv[1] == "fp32" else torch.bfloat16
N = int(sys.argv[2])
cfg = AutoConfig.from_pretrained(MID); cfg.use_cache = False
M = AutoModelForCausalLM.from_pretrained(MID, config=cfg, torch_dtype=dt, local_files_only=True).eval()
for p in M.parameters(): p.requires_grad_(False)
E = M.backbone.embeddings.weight
print(f"dtype={dt} n_layers={len(M.backbone.layers)} V={M.config.vocab_size} "
      f"lm_head tied to embeddings: {M.lm_head.weight.data_ptr() == E.data_ptr()}")

# 1) Pure embedding identity over the whole needle pool [5000, 50255]
pool = torch.arange(5000, 50256)
ok = 0
with torch.no_grad():
    for ch in pool.split(4096):
        ok += (M.lm_head(E[ch]).argmax(-1) == ch).sum().item()
print(f"[pool] argmax(lm_head(E[x])) == x for {ok}/{len(pool)} = {ok/len(pool):.4f} of tokens 5000..50255")

# 2) Truncate to block 0 -> hidden_states[0] identical to full model's hidden_states[0]
M.backbone.layers = M.backbone.layers[:1]

def token_eval_armA(ctx, n):
    r = random.Random(42); ok = tot = ok_emb = ok_shift = 0; fails = []
    t0 = time.time()
    for i in range(n):
        nl = r.randint(3, 5)
        ndl = [r.randint(5000, 50255) for _ in range(nl)]
        np_ = max(3, int(ctx * 0.9))
        if np_ + nl > ctx - 3: np_ = ctx - nl - 3
        inp = torch.full((1, ctx), 50256, dtype=torch.long)
        inp[0, np_:np_ + nl] = torch.tensor(ndl)
        inp = inp[:, :np_ + nl]  # causal: positions <= np+nl-1 do not depend on later tokens
        with torch.no_grad():
            h = M(inp, output_hidden_states=True).hidden_states[0].float()
            lg = M.lm_head(h.to(M.dtype))
            lg_emb = M.lm_head(E[inp[0]])  # embedding only, no block-0 mixer
        for j in range(nl):
            pr = lg[0, np_ + j].argmax(-1).item()
            ok += pr == ndl[j]; tot += 1
            ok_emb += lg_emb[np_ + j].argmax(-1).item() == ndl[j]
            # proper next-token alignment: predict ndl[j] from position np+j-1
            ok_shift += lg[0, np_ + j - 1].argmax(-1).item() == ndl[j]
            if pr != ndl[j]: fails.append((ndl[j], pr))
        if i == 0: print(f"  first example took {time.time()-t0:.2f}s")
    return ok, tot, ok_emb, ok_shift, fails

with torch.no_grad():
    nrm = E.float().norm(dim=-1)
med = nrm[5000:50256].median().item()
for ctx in [2048, 256]:
    ok, tot, ok_emb, ok_shift, fails = token_eval_armA(ctx, N)
    print(f"[ctx={ctx}] Arm A (lm_head(hidden_states[0]) at needle pos, label=SAME-position input token): "
          f"{ok}/{tot} = {ok/tot:.4f}")
    print(f"[ctx={ctx}]   embedding-only (no block 0)         : {ok_emb}/{tot} = {ok_emb/tot:.4f}")
    print(f"[ctx={ctx}]   properly shifted next-token target   : {ok_shift}/{tot} = {ok_shift/tot:.4f}")
    fn = [nrm[t].item() for t, _ in fails]
    if fn:
        print(f"[ctx={ctx}]   failures={len(fails)}; median ||E|| of failed targets={statistics.median(fn):.3f} "
              f"vs pool median={med:.3f}; first 8 (target->pred): {fails[:8]}")
