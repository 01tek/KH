"""Seed robustness of the KH-1-350M recipe: seed 0 (pilot lineage) plus the seed replicas, from the raw eval JSONs.

  python paper_v2/seed_table.py
"""
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "h200_92", "runs_lfm2")
RUNS = {0: ("lfm2_350m_pilot6", "lfm2_350m_pilot4"), 1: ("kh1_350m_seed1_s4", "kh1_350m_seed1_s3"), 2: ("kh1_350m_seed2_s4", "kh1_350m_seed2_s3")}
CELLS = [  # (label, which stage, eval tag, key)
    ("chat, 16K, K=1", 0, "comp_chat1_tau0.99", "facts/L16384/K1"),
    ("chat, 131K, K=1", 0, "comp_chat1_tau0.99", "facts/L131072/K1"),
    ("chat, 1M, K=1", 0, "comp_chat1_tau0.99", "facts/L1048576/K1"),
    ("chat, 1M, K=8", 0, "comp_chat1_tau0.99", "facts/L1048576/K8"),
    ("plain, 1M, K=1", 0, "comp_chat0_tau0.99", "facts/L1048576/K1"),
    ("unseen, 16K, K=1", 0, "comp_heldout_chat0_tau0.99", "facts/L16384/K1"),
    ("unseen, 1M, K=1", 0, "comp_heldout_chat0_tau0.99", "facts/L1048576/K1"),
    ("books, 1M, K=1", 0, "books_comp_chat1_tau0.99", "facts/L1048576/K1"),
    ("one step earlier (0.9 -> 0.97), chat, 1M, K=1", 1, "comp_chat1_tau0.97", "facts/L1048576/K1"),
]


def get(run, tag, key):
    alt = tag[len("books_"):] + "_books" if tag.startswith("books_") else tag  # seed replicas name the books eval "<tag>_books"
    for p in (os.path.join(R, run, f"eval_lm_{tag}.json"), os.path.join(R, run, f"eval_lm_{alt}.json")):
        if os.path.exists(p):
            return json.load(open(p)).get(key)
    return None


print(f"{'cell':48s} " + " ".join(f"seed{s:>2d}" for s in RUNS) + "   mean ± sd")
for label, stage, tag, key in CELLS:
    vals = [get(RUNS[s][stage], tag, key) for s in RUNS]
    ok = [v for v in vals if v is not None]
    ms = f"{np.mean(ok):.3f} ± {np.std(ok, ddof=1):.3f}" if len(ok) > 1 else ""
    print(f"{label:48s} " + " ".join(f"{v:7.3f}" if v is not None else "      -" for v in vals) + f"   {ms}")
