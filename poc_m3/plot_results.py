"""Paper figure: recall vs context length, one panel per K, one line per arm (mean over seeds, 95% CI band).

  python plot_results.py runs/h200 out.png "gdn_lr3e-3_sp0.5=GDN" "p2_haishx_l1=HAIS-X" ...
Each spec is <run-prefix>=<label>; runs named <prefix>_s<seed> are averaged. Mode: early (distance ~ L).
Palette: reference categorical slots in fixed order (validated: scripts/validate_palette.js, light mode);
three slots sit below 3:1 contrast, so every line also gets a marker shape and a direct end label.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

root, out = Path(sys.argv[1]), sys.argv[2]
specs = [s.split("=", 1) for s in sys.argv[3:]]
mode = "early"


def load(run_dir):
    ev = {}
    for line in open(run_dir / "log.jsonl"):
        d = json.loads(line)
        if d["event"] == "eval" and d["mode"] == mode:
            ev[(d["L"], d["K"])] = d["acc"]
    return ev


series = {}
for prefix, label in specs:
    base = root
    if "/" in prefix:  # "<dir>/<prefix>" overrides the root for this series
        d, prefix = prefix.rsplit("/", 1)
        base = Path(d)
    runs = [p.parent for p in base.glob("*/summary.json")
            if re.fullmatch(re.escape(prefix) + r"(_s\d+)?", p.parent.name)]
    if not runs:
        print("no runs for", prefix)
        continue
    agg = defaultdict(list)
    for r in runs:
        for k, v in load(r).items():
            agg[k].append(v)
    series[label] = (agg, len(runs))

Ks = sorted({k for agg, _ in series.values() for (_, k) in agg})
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, len(Ks), figsize=(3.2 * len(Ks) + 1.4, 3.1), sharey=True, facecolor=SURFACE)
axes = [axes] if len(Ks) == 1 else list(axes)
end_labels = []
for ax, K in zip(axes, Ks):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvline(256, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    for i, (label, (agg, n)) in enumerate(series.items()):
        pts = sorted((L, v) for (L, k), v in agg.items() if k == K)
        if not pts:
            continue
        n_series = len(series)
        dodge = 2 ** (0.18 * (i - (n_series - 1) / 2))  # small horizontal offset so coincident series stay visible
        xs = [L * dodge for L, _ in pts]
        ms = [sum(v) / len(v) for _, v in pts]
        ci = [1.96 * math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v)) if len(v) > 1 else 0
              for (_, v), m in zip(pts, ms)]
        c = SLOTS[i % len(SLOTS)]
        ax.fill_between(xs, [max(0, m - e) for m, e in zip(ms, ci)], [min(1, m + e) for m, e in zip(ms, ci)],
                        color=c, alpha=0.15, linewidth=0)
        ax.plot(xs, ms, color=c, linewidth=2, marker=MARKERS[i % len(MARKERS)], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} (n={n})")
        if ax is axes[-1]:
            end_labels.append([ms[-1], xs[-1], label])
    if ax is axes[-1] and end_labels:
        end_labels.sort()
        gap, prev = 0.085, -1.0
        for e in end_labels:  # push labels up until they clear the previous one
            e.append(max(e[0], prev + gap))
            prev = e[3]
        for y_true, x, label, y_lab in end_labels:
            ax.annotate(label, (x, y_true), xytext=(x * 1.35, y_lab), textcoords="data", va="center",
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 4096, 65536, 1048576], ["256", "4K", "64K", "1M"])
    ax.set_xlim(256 / 2.2, 1048576 * (4.5 if ax is axes[-1] else 2.2))
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"K = {K} facts", color=INK, fontsize=10)
    ax.set_xlabel("context length (tokens, log scale)")
axes[0].set_ylabel("recall (exact match)")
axes[0].annotate("train length", (256, 0.5), xytext=(-9, 0), textcoords="offset points", fontsize=7,
                 color=INK2, rotation=90, va="center", ha="center")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 1.02), labelcolor=INK)
fig.suptitle("Recall vs. distance: facts in the first 5% of the stream, models trained at 256 tokens",
             y=1.10, color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
