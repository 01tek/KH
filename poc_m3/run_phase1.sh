#!/usr/bin/env bash
# Phase 1 (Claim A) grid: 4 arms x 3 LRs x 3 seeds x state_passing {0, 0.5} = 72 runs (~6 min each, ~7 GPU-h),
# then the m3+hais mechanism ablations. Resumable: finished runs (summary.json present) are skipped.
# Don't run alongside other GPU jobs on the 8 GB card (see MAMBA3_HAIS_POC_PLAN.md section 7).
set -uo pipefail
cd "$(dirname "$0")"
PY=../m3poc_env/bin/python
EVAL="--eval_lens 1024,16384,262144,1048576 --eval_K 1,4,8 --eval_modes early"
OUT=runs/phase1
mkdir -p $OUT

run() {  # name, extra args...
  local name=$1; shift
  [ -f $OUT/$name/summary.json ] && return 0
  $PY train.py "$@" --steps ${STEPS:-3000} --bs 32 --L 1024 --kmax 8 $EVAL --out $OUT/$name > $OUT/$name.log 2>&1 \
    || echo "FAILED $name" >> $OUT/failures.txt
}

declare -A ARMS=([m3]="m3,m3,m3,m3" [gdn]="gdn,gdn,gdn,gdn" [m3gdn]="m3,gdn,m3,m3" [m3hais]="m3,hais,m3,m3")
for sp in 0 0.5; do
  for lr in 3e-4 1e-3 3e-3; do
    for seed in 0 1 2; do
      for arm in m3 gdn m3gdn m3hais; do
        run ${arm}_lr${lr}_sp${sp}_s${seed} --layout ${ARMS[$arm]} --lr $lr --seed $seed --state_passing $sp
      done
    done
  done
done

# Mechanism ablations on m3+hais (best LR is picked by hand from the grid; default 1e-3)
LR=${ABL_LR:-1e-3}
for seed in 0 1 2; do
  run abl_arch0_s$seed     --layout m3,hais,m3,m3 --lr $LR --seed $seed --state_passing 0.5 --archival_heads 0
  run abl_arch4_s$seed     --layout m3,hais,m3,m3 --lr $LR --seed $seed --state_passing 0.5 --archival_heads 4
  run abl_beta0_s$seed     --layout m3,hais,m3,m3 --lr $LR --seed $seed --state_passing 0.5 --beta_bias 0
  run abl_decoy_s$seed     --layout m3,hais,m3,m3 --lr $LR --seed $seed --state_passing 0.5 --decoy 0.1
  run base_gdn_decoy_s$seed --layout m3,gdn,m3,m3 --lr $LR --seed $seed --state_passing 0.5 --decoy 0.1
done
echo PHASE1_DONE
