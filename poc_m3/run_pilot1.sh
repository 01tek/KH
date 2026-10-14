#!/usr/bin/env bash
# Pilot-1: calibrated recipe (2-layer models, train L=256, up to 32 facts, LR 3e-3, state passing on).
# Recipe chosen from runs/recipe/: attention solves the task in <300 steps; GDN clicks ~2.4K steps at LR 3e-3
# (fails at 1e-2), so recurrent arms get 12K steps. 1 seed: calibration, not a claim.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../m3poc_env/bin/python}
OUT=${OUT:-runs/pilot1}
COMMON="--L 256 --kmax 32 --bs 128 --lr ${LR:-3e-3} --steps ${STEPS:-12000} --state_passing 0.5 --seed ${SEED:-0} \
  --eval_lens 256,4096,65536,1048576 --eval_K 1,8,32 --eval_modes early,uniform"
mkdir -p $OUT
for spec in "m3hais:m3,hais" "gdn:gdn,gdn" "m3gdn:m3,gdn" "m3:m3,m3"; do
  name=${spec%%:*}; layout=${spec#*:}
  [ -f $OUT/$name/summary.json ] && continue
  $PY train.py --layout $layout $COMMON --out $OUT/$name > $OUT/$name.log 2>&1 || echo "FAILED $name" >> $OUT/failures.txt
done
echo PILOT1_FINISHED
