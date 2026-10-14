"""Summarise runs: recall per (mode, L, K) for each run group, mean ± 95% CI over seeds.

  python report.py runs/pilot1            # one row per run
  python report.py runs/phase1 --group    # groups runs that differ only by _s<seed>
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def load(run_dir):
    evals, train = {}, None
    for line in open(run_dir / "log.jsonl"):
        d = json.loads(line)
        if d["event"] == "eval":
            evals[(d["mode"], d["L"], d["K"])] = d["acc"]
        elif d["event"] == "train":
            train = d
    return evals, train


def main():
    root = Path(sys.argv[1])
    group = "--group" in sys.argv
    runs = sorted(p.parent for p in root.glob("*/summary.json"))
    if not runs:
        print("no finished runs under", root)
        return
    groups = defaultdict(list)
    for r in runs:
        key = re.sub(r"_s\d+$", "", r.name) if group else r.name
        groups[key].append(load(r))
    cells = sorted({c for g in groups.values() for ev, _ in g for c in ev})
    modes = sorted({c[0] for c in cells})
    for mode in modes:
        cols = [c for c in cells if c[0] == mode]
        print(f"\n## mode={mode}   (cell = recall; mean±CI95 over n seeds when grouped)")
        print("| run | n | final train acc | " + " | ".join(f"L{c[1]} K{c[2]}" for c in cols) + " |")
        print("|---|---|---|" + "---|" * len(cols))
        for key, items in sorted(groups.items()):
            row = []
            for c in cols:
                xs = [ev[c] for ev, _ in items if c in ev]
                if not xs:
                    row.append("–")
                elif len(xs) == 1:
                    row.append(f"{xs[0]:.2f}")
                else:
                    m = sum(xs) / len(xs)
                    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                    row.append(f"{m:.2f}±{1.96 * sd / math.sqrt(len(xs)):.2f}")
            tr = [t["acc"] for _, t in items if t]
            print(f"| {key} | {len(items)} | {sum(tr) / max(1, len(tr)):.2f} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
