"""Does 1M recall track quiet fact heads across all 350M-recipe models (3 seeds, 2 stage-4 reruns, 4 KH-2 variants, seed-0 two-tier)?
Recall from the raw eval JSONs (chat, 1M; mean of development and fresh streams where both exist); mechanism metrics from
write_stats.py (same 6 streams of 1M tokens for every model).   python paper_v2/quiet_heads_corr.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, K2, WS = (os.path.join(ROOT, "results", *p) for p in (("h200_92", "runs_lfm2"), ("kh2",), ("seed_analysis",)))
M = {  # name: (eval JSONs (dev, fresh or None), write_stats JSON)
    "seed0": ((f"{R}/lfm2_350m_pilot6/eval_lm_comp_chat1_tau0.99.json", f"{R}/lfm2_350m_pilot6/eval_lm_fresh4321_comp_chat1_tau0.99.json"), "write_stats_n6_seed0.json"),
    "seed1": ((f"{R}/kh1_350m_seed1_s4/eval_lm_comp_chat1_tau0.99.json", None), "write_stats_n6_seed1.json"),
    "seed2": ((f"{R}/kh1_350m_seed2_s4/eval_lm_comp_chat1_tau0.99.json", None), "write_stats_n6_seed2.json"),
}
for s in (1, 2):
    for v, tag in (("ctrl", "_tau0.99"), ("tier", "_tier"), ("quiet", "_tau0.99")):
        d = f"{K2}/kh2_350m_seed{s}_{v}"
        M[f"seed{s}_{v}"] = ((f"{d}/eval_lm_comp_chat1{tag}_dev.json", f"{d}/eval_lm_comp_chat1{tag}_fresh4321.json"), f"write_stats_n6_kh2_seed{s}_{v}.json")
d = f"{K2}/kh2_350m_seed0_tier"
M["seed0_tier"] = ((f"{d}/eval_lm_comp_chat1_tier_dev.json", f"{d}/eval_lm_comp_chat1_tier_fresh4321.json"), "write_stats_n6_kh2_seed0_tier.json")


def rank(x):
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in np.unique(x):  # average ranks for ties
        o[x == v] = o[x == v].mean()
    return o


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x, float)), rank(np.asarray(y, float)))[0, 1])


rows = []
for name, ((dev, fresh), ws) in M.items():
    p = os.path.join(WS, ws)
    if not os.path.exists(p):
        print("missing", name); continue
    e = [json.load(open(f)) for f in (dev, fresh) if f]
    k1 = np.mean([x["facts/L1048576/K1"] for x in e]); k8 = np.mean([x["facts/L1048576/K8"] for x in e])
    w = json.load(open(p)); r = w["retention/1048576/tau0.97"]
    ph = w["per_head"]; fs = np.array(ph["fact_store_rate"]); rl = np.array(ph["retention_1048576_median"])
    rows.append(dict(name=name, k1=k1, k8=k8, quiet=w["fact_heads_quiet_lt_0.0001"], heads=r["written_heads_per_stream"],
                     ret=r["written_heads_median"], keep=int(((fs >= 0.5) & (rl > 0.5)).sum()), tw=w["fact_heads_text_write_rate"]))
print(f"{'model':12s} {'1M K1':>6s} {'1M K8':>6s} {'quiet':>5s} {'keep':>4s} {'heads':>5s} {'ret':>6s} {'fact-tw':>7s}")
for x in rows:
    print(f"{x['name']:12s} {x['k1']:6.3f} {x['k8']:6.3f} {x['quiet']:5d} {x['keep']:4d} {x['heads']:5.1f} {x['ret']:6.3f} {x['tw']:7.4f}")
print(f"\nSpearman rank correlation with 1M recall across {len(rows)} models:")
for m, lab in (("quiet", "quiet fact heads"), ("keep", "fact heads keeping the fact at 1M"), ("heads", "heads storing the fact"),
               ("ret", "median 1M retention"), ("tw", "fact-head text-write rate")):
    print(f"  {lab:36s} K1 {spearman([x[m] for x in rows], [x['k1'] for x in rows]):+.2f}   K8 {spearman([x[m] for x in rows], [x['k8'] for x in rows]):+.2f}")
