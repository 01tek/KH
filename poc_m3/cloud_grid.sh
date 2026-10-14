#!/usr/bin/env bash
# Parallel, resumable Phase-1 grid for a big single GPU (e.g. the 96 GB RTX PRO 6000 Blackwell box).
# Calibrated recipe (see runs/recipe, run_pilot1.sh): 2-layer models, train L=256, up to 32 facts, bs 128.
# Preemption-safe: each run writes summary.json when done; finished runs are skipped on relaunch.
# Usage (on the box):  cd ~/hais_poc/poc_m3 && PAR=8 nohup ./cloud_grid.sh > grid.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../env/bin/python}
PAR=${PAR:-8}                  # concurrent runs (tiny models: ~4 GB VRAM each)
STEPS=${STEPS:-20000}
OUT=${OUT:-runs/phase1}
COMMON="--steps $STEPS --bs 128 --L 256 --kmax 32 --eval_K 1,8,32 --eval_modes early,uniform"
LONG="--eval_lens 256,4096,65536,1048576"
SHORT="--eval_lens 256,4096"   # attention control cannot stream past one chunk
mkdir -p $OUT

declare -A ARMS=([haishais]="hais,hais" [gdn]="gdn,gdn" [m3hais]="m3,hais" [m3gdn]="m3,gdn" [m3]="m3,m3")
jobs_file=$OUT/jobs.txt
: > $jobs_file
for seed in 0 1 2; do
  echo "attn_s$seed --layout attn,attn --lr 1e-3 --seed $seed $SHORT" >> $jobs_file   # task-is-learnable control
done
for sp in 0.5 0; do
  for lr in ${LRS:-3e-3 1e-3}; do
    for seed in 0 1 2; do
      for arm in haishais gdn m3hais m3gdn m3; do
        echo "${arm}_lr${lr}_sp${sp}_s${seed} --layout ${ARMS[$arm]} --lr $lr --seed $seed --state_passing $sp $LONG" >> $jobs_file
      done
    done
  done
done
for seed in 0 1 2; do  # mechanism ablations on hais,hais (primary comparison: hais,hais vs gdn,gdn) (state passing on, LR 3e-3)
  echo "abl_arch0_s$seed --layout hais,hais --lr 3e-3 --seed $seed --state_passing 0.5 --archival_heads 0 $LONG" >> $jobs_file
  echo "abl_arch4_s$seed --layout hais,hais --lr 3e-3 --seed $seed --state_passing 0.5 --archival_heads 4 $LONG" >> $jobs_file
  echo "abl_beta0_s$seed --layout hais,hais --lr 3e-3 --seed $seed --state_passing 0.5 --beta_bias 0 $LONG" >> $jobs_file
  echo "abl_decoy_s$seed --layout hais,hais --lr 3e-3 --seed $seed --state_passing 0.5 --decoy 0.1 $LONG" >> $jobs_file
  echo "gdn_decoy_s$seed --layout gdn,gdn --lr 3e-3 --seed $seed --state_passing 0.5 --decoy 0.1 $LONG" >> $jobs_file
done

run_one() {
  local name=$1; shift
  [ -f $OUT/$name/summary.json ] && return 0
  $PY train.py "$@" $COMMON --out $OUT/$name > $OUT/$name.log 2>&1 || echo "FAILED $name" >> $OUT/failures.txt
}
export -f run_one
export PY OUT COMMON
xargs -P $PAR -L 1 bash -c 'run_one "$@"' _ < $jobs_file
echo GRID_DONE
