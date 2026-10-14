#!/usr/bin/env bash
# Pilot 6: close the scale-invariant-decay escape: min_decay 0.35 (state underflows within the 256-token window).
# Usage: SEEDS="0 1 2 3 4 5" bash p6_launch.sh
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
PY=../env/bin/python; OUT=runs/p6; mkdir -p $OUT; J=$OUT/jobs_$$.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
C="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --lr 3e-3 --steps 20000 --bs 128 --L 256 --kmax 32"
for s in $SEEDS; do
  echo "p6_md035_s$s $C --seed $s $EV" >> $J
  [ -n "$DECOY" ] && echo "p6_md035_decoy_s$s $C --decoy 0.1 --seed $s $EV" >> $J
done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P ${PAR:-12} -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] P6_DONE"
