"""Phase-0 gate: does each layout run fwd+bwd on this GPU, is chunked streaming
identical to a full pass, and how fast is training? Writes smoke.json."""
import json
import time

import torch

from model import TinyLM
from tasks import Vocab, fact_stream

torch.manual_seed(0)
dev = "cuda"
V = Vocab()
LAYOUTS = {"m3": "m3,m3,m3,m3", "gdn": "gdn,gdn,gdn,gdn", "m3+gdn": "m3,gdn,m3,m3", "m3+hais": "m3,hais,m3,m3"}
results = {}

for name, layout in LAYOUTS.items():
    r = {}
    model = TinyLM(V.size, 256, layout).to(dev)
    r["params"] = model.n_params()

    # 1) streaming equivalence: full pass vs 4 chunks with carried state
    model.eval()
    ids, _, _ = fact_stream(2, 1024, 4, vocab=V, device=dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        full, _ = model(ids)
        st, outs = None, []
        for s in range(0, 1024, 256):
            o, st = model(ids[:, s:s + 256], st)
            outs.append(o)
        chunked = torch.cat(outs, 1)
    r["stream_max_abs_diff"] = (full.float() - chunked.float()).abs().max().item()
    r["logit_scale"] = full.float().abs().max().item()

    # 2) training throughput + finite grads
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, L = 16, 1024
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(12):
        ids, labels, _ = fact_stream(B, L, 8, vocab=V, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(logits.float().view(-1, V.size), labels.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    r["loss_finite"] = bool(torch.isfinite(loss).item())
    r["grads_finite"] = bool(ok)
    r["train_tok_per_s"] = round(B * L / (sum(times[4:]) / len(times[4:])))
    r["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    results[name] = r
    print(name, r, flush=True)
    del model, opt
    torch.cuda.empty_cache()

json.dump(results, open("smoke.json", "w"), indent=2)
