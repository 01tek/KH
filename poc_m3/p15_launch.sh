#!/usr/bin/env bash
# Pilot 15: decoy pressure -- train at decoy 0.3, evaluate at decoy 0.1 (hard threshold tau 0.1 on/off).
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p15; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
R="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --lr 3e-3 --steps 40000 --bs 128 --L 256 --kmax 32"
for s in 0 1 2; do echo "p15_dtrain03_tau0.1_s$s $R --decoy 0.3 --eval_decoy 0.1 --arch_tau 0.1 --seed $s $EV" >> $J
  echo "p15_dtrain03_s$s $R --decoy 0.3 --eval_decoy 0.1 --seed $s $EV" >> $J
  echo "p15_dtrain03_tau0.1_80k_s$s $R --decoy 0.3 --eval_decoy 0.1 --arch_tau 0.1 --steps 80000 --seed $s $EV" >> $J; done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P 9 -L 1 bash -c 'run_one "$@"' _ < $J
echo P15_DONE
