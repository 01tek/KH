"""Figure: exact retention of a stored fact's key in the archival heads of a trained LM, along real text after the fact,
r = k_f^T prod_t (I - beta_t k_t k_t^T) k_f, with archival writes at or below a threshold tau dropped (simulated).

  python plot_retention.py data/retention/lm8_write_stats.json fig_retention.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
lens = sorted({int(k.split("/")[1]) for k in d if k.startswith("retention/")})
taus = sorted({float(k.split("/tau")[1]) for k in d if k.startswith("retention/")})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
cols = plt.cm.viridis([i / max(1, len(taus) - 1) for i in range(len(taus))])
for ax, key, title in ((axes[0], "median_all_heads", "median archival head"),
                       (axes[1], "best_head_median", "best archival head (per stream)")):
    for c, t in zip(cols, taus):
        ys = [d[f"retention/{L}/tau{t}"][key] for L in lens]
        ax.plot(lens, ys, marker="o", color=c, label="as trained" if t == 0 else f"drop writes β ≤ {t}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens, [f"{L // 1024}K" if L < 1 << 20 else f"{L >> 20}M" for L in lens])
    ax.set_title(title)
    ax.set_xlabel("tokens of real text after the fact")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("retention of the fact's key")
axes[0].legend(fontsize=8)
fig.suptitle(f"117M HAIS LM (lm8): text writes β>0.1 at {100 * d['text_frac_beta_gt']['0.1']:.0f}% of tokens; "
             f"fact written at median β {d['fact_beta_max_median']:.3f}", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
print("wrote", sys.argv[2])
