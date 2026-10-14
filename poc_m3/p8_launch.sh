#!/usr/bin/env bash
# Pilot 8 (H200 #2): decoy robustness at 1M (longer state-passing chains, stronger penalty) + clean seeds of the final recipe.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p8; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
B="--layout hais,hais --l1_scope archival --min_decay 0.35 --bs 128 --L 256 --kmax 32 --lr 3e-3 --steps 40000"
for s in 0 1 2; do
  echo "p8_decoy_sp99_s$s $B --write_l1 0.05 --state_passing 0.99 --decoy 0.1 --seed $s $EV" >> $J
  echo "p8_decoy_l11_s$s $B --write_l1 0.1 --state_passing 0.95 --decoy 0.1 --seed $s $EV" >> $J
  echo "p8_decoy_sp99_l11_s$s $B --write_l1 0.1 --state_passing 0.99 --decoy 0.1 --seed $s $EV" >> $J
done
for s in 12 13 14 15 16 17; do echo "p8_final_s$s $B --write_l1 0.05 --state_passing 0.95 --seed $s $EV" >> $J; done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P ${PAR:-15} -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] P8_DONE"
