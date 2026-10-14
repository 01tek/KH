#!/usr/bin/env bash
# Pilot 13: budget sized to the data -- 256-token blocks, budget 40/48 (> max training facts per block) vs decoys at 1M.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p13; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
R="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --lr 3e-3 --steps 40000 --bs 128 --L 256 --kmax 32 --write_block 256"
for s in 0 1 2; do for m in 40 48; do echo "p13_decoy_b256m${m}_s$s $R --decoy 0.1 --write_budget $m --seed $s $EV" >> $J; done
  echo "p13_clean_b256m40_s$s $R --write_budget 40 --seed $s $EV" >> $J; done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P 9 -L 1 bash -c 'run_one "$@"' _ < $J
echo P13_DONE
