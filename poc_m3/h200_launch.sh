#!/usr/bin/env bash
# H200 launcher: CUDA MPS on, then priority follow-ups (pilot 2) + the Phase-1 grid, all in parallel.
# Resumable (runs with summary.json are skipped). Usage on the box:
#   cd ~/hais_poc/poc_m3 && PAR=16 nohup ./h200_launch.sh > launch.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
pgrep -x nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d || echo "MPS unavailable, continuing without"
PY=${PY:-../env/bin/python}
PAR=${PAR:-16}
STEPS=${STEPS:-20000}
OUT=${OUT:-runs/h200}
COMMON="--steps $STEPS --bs 128 --L 256 --kmax 32 --eval_K 1,8,32 --eval_modes early,uniform"
LONG="--eval_lens 256,4096,65536,1048576"
mkdir -p $OUT
J=$OUT/jobs.txt
: > $J

# --- pilot 2 (priority): what makes archival memory survive distractors? 3 seeds each ---
for s in 0 1 2; do
  echo "p2_haishais_sp95_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.95 $LONG" >> $J
  echo "p2_gdn_sp95_s$s --layout gdn,gdn --lr 3e-3 --seed $s --state_passing 0.95 $LONG" >> $J
  echo "p2_haishais_sp95_l1arch_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.95 --write_l1 0.05 --l1_scope archival $LONG" >> $J
  echo "p2_haishx_l1_s$s --layout hais,hx --lr 3e-3 --seed $s --state_passing 0.95 --write_l1 0.05 --chunk 1024 $LONG" >> $J
  echo "p2_gdnhx_l1_s$s --layout gdn,hx --lr 3e-3 --seed $s --state_passing 0.95 --write_l1 0.05 --chunk 1024 $LONG" >> $J
done

# --- Phase-1 grid (see cloud_grid.sh for rationale) ---
declare -A ARMS=([haishais]="hais,hais" [gdn]="gdn,gdn" [m3hais]="m3,hais" [m3gdn]="m3,gdn" [m3]="m3,m3")
for s in 0 1 2; do echo "attn_s$s --layout attn,attn --lr 1e-3 --seed $s --eval_lens 256,4096" >> $J; done
for sp in 0.5 0; do
  for lr in 3e-3 1e-3; do
    for s in 0 1 2; do
      for arm in haishais gdn m3hais m3gdn m3; do
        echo "${arm}_lr${lr}_sp${sp}_s$s --layout ${ARMS[$arm]} --lr $lr --seed $s --state_passing $sp $LONG" >> $J
      done
    done
  done
done
for s in 0 1 2; do
  echo "abl_arch0_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.5 --archival_heads 0 $LONG" >> $J
  echo "abl_arch4_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.5 --archival_heads 4 $LONG" >> $J
  echo "abl_beta0_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.5 --beta_bias 0 $LONG" >> $J
  echo "abl_decoy_s$s --layout hais,hais --lr 3e-3 --seed $s --state_passing 0.5 --decoy 0.1 $LONG" >> $J
  echo "gdn_decoy_s$s --layout gdn,gdn --lr 3e-3 --seed $s --state_passing 0.5 --decoy 0.1 $LONG" >> $J
done

run_one() {
  local name=$1; shift
  [ -f $OUT/$name/summary.json ] && return 0
  $PY train.py "$@" $COMMON --out $OUT/$name > $OUT/$name.log 2>&1 || echo "FAILED $name" >> $OUT/failures.txt
}
export -f run_one
export PY OUT COMMON
echo "[$(date)] $(wc -l < $J) jobs, PAR=$PAR"
xargs -P $PAR -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] H200_GRID_DONE"
