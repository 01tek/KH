#!/usr/bin/env bash
# Re-run only the eval for finished trainings whose eval crashed (model.pt present, summary.json missing).
# Waits until no train.py is running so evals don't contend for the GPU.
# Usage: ./reeval.sh runs/pilot1
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../m3poc_env/bin/python}
ROOT=${1:-runs/pilot1}
while ps -eo cmd | grep -q "[t]rain.py --layout"; do sleep 20; done
for d in "$ROOT"/*/; do
  d=${d%/}
  [ -f "$d/summary.json" ] && continue
  [ -f "$d/model.pt" ] || continue
  args=$($PY - "$d" <<'EOF'
import sys, torch
a = torch.load(sys.argv[1] + "/model.pt", map_location="cpu", weights_only=False)["args"]
keep = ("layout", "d_model", "L", "kmax", "decoy", "archival_heads", "beta_bias", "min_decay", "tau", "store_cap", "null_sink", "mem_heads", "write_budget", "write_block", "arch_tau", "v_norm",
        "eval_lens", "eval_K", "eval_n", "eval_modes", "chunk", "seed")
print(" ".join((f"--{k}" if a[k] is True else f"--{k} {a[k]}") for k in keep if a.get(k) not in (None, False)))
EOF
)
  echo "[reeval] $d :: $args"
  $PY train.py $args --out "$d" --eval_only > "$d.reeval.log" 2>&1 || echo "REEVAL_FAILED $d"
done
echo REEVAL_FINISHED
