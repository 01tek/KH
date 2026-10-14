#!/usr/bin/env bash
# Pilot 12: budgeted archival writes (top-m per 64-token block) vs decoys at 1M. Recipe: md 0.35, L1 0.05, sp 0.95, 40K steps.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p12; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
R="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --lr 3e-3 --steps 40000 --bs 128 --L 256 --kmax 32"
for s in 0 1 2; do for m in 1 2 4; do echo "p12_decoy_wb${m}_s$s $R --decoy 0.1 --write_budget $m --seed $s $EV" >> $J; done; done
for s in 0 1; do echo "p12_clean_wb2_s$s $R --write_budget 2 --seed $s $EV" >> $J; done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P 11 -L 1 bash -c 'run_one "$@"' _ < $J
echo P12_DONE
