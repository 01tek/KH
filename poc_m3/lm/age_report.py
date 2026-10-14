"""Aggregate the per-age answer accuracy logged by train_lm.py over the last N log lines.
  python age_report.py runs_lm14/lm14_cand_50m [--last 40]"""
import json
import sys

last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 40
for run in [x for x in sys.argv[1:] if not x.startswith("--") and not x.isdigit()]:
    rows = [json.loads(l) for l in open(f"{run}/log.jsonl") if '"train"' in l][-last:]
    agg = {}
    for r in rows:
        for k, (acc, n) in (r.get("age_acc") or {}).items():
            g, t = agg.get(k, (0.0, 0))
            agg[k] = (g + acc * n, t + n)
    lo = {"0+": 0, "4+": 1, "32+": 2, "256+": 3}
    cells = "  ".join(f"age {k:>4} seg: {g / t:.2f} (n={t})" for k, (g, t) in sorted(agg.items(), key=lambda kv: lo[kv[0]]))
    print(f"{run} @step {rows[-1]['step'] if rows else '?'}: {cells}")
