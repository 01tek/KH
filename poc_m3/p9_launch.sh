#!/usr/bin/env bash
# Pilot 9 (H200 #2): capacity scaling with a length curriculum + decoys at 80K steps.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p9; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
B="--layout hais,hais --l1_scope archival --min_decay 0.35 --lr 3e-3"
for s in 0 1; do for mh in 4 8 16; do
  echo "p9_cap_h${mh}_s$s $B --write_l1 0.05 --state_passing 0.95 --mem_heads $mh --steps 40000 --bs 64 --L 1024 --L_start 256 --curriculum_frac 0.5 --kmax 128 --eval_K 32,64,128,256,512 --eval_lens 4096,65536,1048576 --eval_modes early --chunk 4096 --seed $s" >> $J
done; done
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
for s in 0 1 2; do
  echo "p9_decoy_l11_80k_s$s $B --write_l1 0.1 --state_passing 0.95 --decoy 0.1 --steps 80000 --bs 128 --L 256 --kmax 32 --seed $s $EV" >> $J
  echo "p9_decoy_sp99_80k_s$s $B --write_l1 0.05 --state_passing 0.99 --decoy 0.1 --steps 80000 --bs 128 --L 256 --kmax 32 --seed $s $EV" >> $J
done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P ${PAR:-12} -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] P9_DONE"
