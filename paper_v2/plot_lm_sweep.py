"""Figure: in-text recall vs context length for the node sweep-2 LMs (117M, 1B tokens).

  python plot_lm_sweep.py data/sweep2 fig_lm_recall_vs_length.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else "sweep2"
LENS = [16384, 131072, 1048576]
RUNS_SWEEP5 = [
    ("s5_ctrl_s5", "_multi_tau0.9", "control: half drills, GDN cap 8K", dict(color="0.4", ls="--", marker="x")),
    ("s5_hl8k_far", "_multi_tau0.9", "far drills, GDN cap 8K", dict(color="tab:purple", marker="o")),
    ("s5_hl1024_far", "_multi_tau0.9", "far drills, GDN cap 1K", dict(color="tab:blue", marker="o")),
    ("s5_hl512_far", "_multi_tau0.9", "far drills, GDN cap 512 (s0)", dict(color="tab:orange", marker="s")),
    ("s5_hl512_far_s1", "_multi_tau0.9", "far drills, GDN cap 512 (s1)", dict(color="tab:orange", marker="s", ls=":")),
    ("s5_hl256_far", "_multi_tau0.9", "far drills, GDN cap 256 (s0)", dict(color="tab:green", marker="D")),
    ("s5_hl256_far_s1", "_multi_tau0.9", "far drills, GDN cap 256 (s1)", dict(color="tab:green", marker="D", ls=":")),
]
RUNS = [  # (dir, eval tag, label, style)
    ("s_gdn", "_multi_tau0.9", "GDN baseline", dict(color="0.5", ls="--", marker="x")),
    ("s_base", "_multi_tau0.9", "HAIS, no levers", dict(color="tab:blue", marker="o")),
    ("s_tau_hl8k", "_multi_tau0.9", "HAIS + τ + GDN cap", dict(color="tab:orange", marker="s")),
    ("s_tau_hl8k_twl", "_multi_tau0.9", "HAIS + τ + GDN cap + text-write L1 (s0)", dict(color="tab:green", marker="D")),
    ("s_tau_hl8k_twl_s1", "_multi_tau0.9", "same, seed 1", dict(color="tab:green", marker="D", ls=":")),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, K in zip(axes, (1, 8)):
    for d, tag, label, style in (RUNS_SWEEP5 if which == "sweep5" else RUNS):
        p = os.path.join(root, d, f"eval_lm{tag}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        ys = [r.get(f"facts/L{L}/K{K}") for L in LENS]
        ax.plot(LENS, ys, label=label, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENS, ["16K", "131K", "1M"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("context length (tokens, constant memory)")
    ax.set_title(f"{K} fact{'s' if K > 1 else ''} in real text, fresh each sample")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("exact recall")
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("117M LMs, 1B FineWeb-Edu tokens, inference write threshold 0.9" + (" — sweep 5: drills that need archival memory" if which == "sweep5" else ""), fontsize=10)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(out, dpi=160)
print("wrote", out)
