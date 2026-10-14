"""Figure: KH-1 (frozen LFM2 + archival adapters) recall vs distance, with the memory-off control.

  python plot_kh1.py fig_kh1_recall.png
Reads eval_lm*.json files from the result folders listed in RUNS (paths relative to the repo root).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENS = [16384, 131072, 1048576]
RUNS = [  # (eval json, label, style)
    ("results/h200_92/runs_lfm2/lfm2_350m_pilot6/eval_lm_comp_chat1_tau0.99.json", "KH-1-350M (τ 0.97→0.99), chat",
     dict(color="tab:red", marker="s")),
    ("results/h200_92/runs_lfm2/lfm2_350m_pilot6/eval_lm_comp_heldout_chat0_tau0.99.json", "KH-1-350M (τ 0.97→0.99), unseen phrasings",
     dict(color="tab:red", marker="^", ls="--")),
    ("results/h200_92/runs_lfm2/lfm2_350m_pilot4/eval_lm_comp_tau0.97.json", "KH-1-350M (τ 0.9→0.97), plain",
     dict(color="tab:orange", marker="s", ls=":")),
    ("results/b200_node/runs_kh1/kh1_1p2b_v4/eval_lm_comp_chat1_tau0.99.json", "KH-1-1.2B (τ 0.97→0.99), chat",
     dict(color="tab:blue", marker="o")),
    ("results/b200_node/runs_kh1/kh1_1p2b_v4/eval_lm_comp_heldout_chat0_tau0.99.json", "KH-1-1.2B (τ 0.97→0.99), unseen phrasings",
     dict(color="tab:blue", marker="^", ls="--")),
    ("results/b200_node/runs_kh1/kh1_1p2b_t09/eval_lm_comp_chat1_tau0.97.json", "KH-1-1.2B (τ 0.9→0.97), chat",
     dict(color="tab:cyan", marker="o", ls=":")),
    ("results/b200_node/runs_qwen/q4b_s2/eval_lm_comp_chat1_tau0.97.json", "KH-1-Q-4B (τ 0.9→0.97), chat",
     dict(color="tab:green", marker="D")),
    ("results/b200_node/runs_qwen/q4b_s2/eval_lm_comp_chat1_tau0.98.json", "KH-1-Q-4B (τ 0.9→0.98), chat",
     dict(color="tab:green", marker="*", ls="none", markersize=13)),
    ("results/local_ppl/p4/eval_lm_comp_nomem.json", "memory off (plain chunked LFM2)", dict(color="0.4", marker="x", ls="--")),
]
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper_v2", "fig_kh1_recall.png")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
for ax, K in zip(axes, (1, 8)):
    for path, label, style in RUNS:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print("missing", path)
            continue
        r = json.load(open(p))
        pts = [(L, r[f"facts/L{L}/K{K}"]) for L in LENS if f"facts/L{L}/K{K}" in r]
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], label=label, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENS, ["16K", "131K", "1M"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("distance from the facts to the questions (tokens)")
    ax.set_title(f"{K} fact{'s' if K > 1 else ''} per stream")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("exact recall")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.suptitle("Archival-memory grafts on frozen LFM2 and Qwen3.5 (constant GPU memory); τ = training → inference write threshold", fontsize=10)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(out, dpi=150)
print("wrote", out)
