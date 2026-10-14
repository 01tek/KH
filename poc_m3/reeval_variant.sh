#!/usr/bin/env bash
# Re-evaluate a saved model under different eval/store settings without touching the original run.
# Usage: ./reeval_variant.sh <src_run_dir> <dst_name> [extra train.py args that override...]
# Output: runs/reeval/<dst_name>/{model.pt (copy), log.jsonl, summary.json}
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../env/bin/python}
SRC=$1; DST=runs/reeval/$2; shift 2
mkdir -p "$DST" && cp "$SRC/model.pt" "$DST/model.pt"
base=$($PY - "$SRC" <<'EOF'
import sys, torch
a = torch.load(sys.argv[1] + "/model.pt", map_location="cpu", weights_only=False)["args"]
keep = ("layout", "d_model", "L", "kmax", "decoy", "archival_heads", "beta_bias", "min_decay", "tau", "store_cap", "null_sink", "mem_heads", "write_budget", "write_block", "arch_tau", "v_norm", "seed")
print(" ".join((f"--{k}" if a[k] is True else f"--{k} {a[k]}") for k in keep if a.get(k) not in (None, False)))
EOF
)
echo "[reeval] $SRC -> $DST :: $base $*"
$PY train.py $base "$@" --out "$DST" --eval_only > "$DST.log" 2>&1 || echo "REEVAL_FAILED $DST"
