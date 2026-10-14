"""Slide chart: recall vs distance for the released KH-1 LFM2 grafts, all three 350M seeds, fresh-stream checks and memory off.

  python kh1/deck/plot_deck_recall.py kh1/deck/deck_recall.png
Every point is read from a raw eval_lm*.json under results/ (K = 1, chat questions, 40 streams).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LENS = [16384, 131072, 1048576]
L = "results/h200_92/runs_lfm2/"
LINES = [  # (eval json, label, style)
    (L + "kh1_350m_seed1_s4/eval_lm_comp_chat1_tau0.99.json", "KH-1-350M, seeds 1 and 2", dict(color="#E8A0A0", lw=2.5, marker="s", ms=8)),
    (L + "kh1_350m_seed2_s4/eval_lm_comp_chat1_tau0.99.json", None, dict(color="#E8A0A0", lw=2.5, marker="s", ms=8)),
    ("results/b200_node/runs_kh1/kh1_1p2b_v4/eval_lm_comp_chat1_tau0.99.json", "KH-1-1.2B (released)", dict(color="#1F77B4", lw=4, marker="o", ms=11)),
    (L + "lfm2_350m_pilot6/eval_lm_comp_chat1_tau0.99.json", "KH-1-350M, seed 0 (released)", dict(color="#D62728", lw=4, marker="s", ms=11)),
]
FRESH = [  # 1M on 40 fresh streams
    (L + "lfm2_350m_pilot6/eval_lm_fresh4321_comp_chat1_tau0.99.json", "#D62728"),
    ("results/h200_92/runs_confirm/v4/eval_lm_fresh4321_comp_chat1_tau0.99.json", "#1F77B4"),
]
NOMEM = "results/local_ppl/p4/eval_lm_comp_nomem.json"


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "kh1", "deck", "deck_recall.png")
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(figsize=(16.64, 6.4))
for path, label, style in LINES:
    r = load(path)
    ax.plot(range(3), [r[f"facts/L{n}/K1"] for n in LENS], label=label, **style)
for i, (path, color) in enumerate(FRESH):
    v = load(path)["facts/L1048576/K1"]
    ax.plot([2], [v], ls="none", marker="o", ms=16, mfc="white", mec=color, mew=3,
            label="1M on 40 fresh streams" if i == 0 else None)
nm = load(NOMEM)["facts/L16384/K1"]
ax.plot([0], [nm], ls="none", marker="x", ms=16, mew=3, color="0.35", label=f"memory off ({nm:.2f})")
ax.set_xticks(range(3), ["16K tokens", "131K tokens", "1M tokens"])
ax.set_xlim(-0.15, 2.15)
ax.set_ylim(-0.04, 1.04)
ax.set_ylabel("exact recall")
ax.grid(alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h, lab = ax.get_legend_handles_labels()
order = [lab.index(x) for x in ("KH-1-350M, seed 0 (released)", "KH-1-350M, seeds 1 and 2", "KH-1-1.2B (released)")] + [i for i, x in enumerate(lab) if x.startswith(("1M", "memory"))]
ax.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", bbox_to_anchor=(0.55, 0.03), fontsize=17, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)
