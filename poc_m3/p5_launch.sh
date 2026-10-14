#!/usr/bin/env bash
# Pilot 5 (H200 #2): decoy robustness, capacity scaling, extra seeds of the Pilot-3 recipe.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -x nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
PY=../env/bin/python; OUT=runs/p5; mkdir -p $OUT; J=$OUT/jobs.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform"
R="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.0217 --state_passing 0.95 --lr 3e-3"
for s in 0 1 2; do
  echo "p5_decoy_l105_40k_s$s $R --decoy 0.1 --steps 40000 --bs 128 --L 256 --kmax 32 --eval_K 1,8,32 --seed $s $EV" >> $J
  echo "p5_decoy_l102_40k_s$s --layout hais,hais --write_l1 0.02 --l1_scope archival --min_decay 0.0217 --state_passing 0.95 --lr 3e-3 --decoy 0.1 --steps 40000 --bs 128 --L 256 --kmax 32 --eval_K 1,8,32 --seed $s $EV" >> $J
done
for s in 3 4 5; do
  echo "p5_recipe_s$s $R --steps 20000 --bs 128 --L 256 --kmax 32 --eval_K 1,8,32 --seed $s $EV" >> $J
done
# capacity: train with up to 128 facts at L=1024, vary archival key dims (mem_heads/2 archival heads x 64)
for s in 0 1; do
  for mh in 4 8 16; do
    echo "p5_cap_h${mh}_s$s $R --mem_heads $mh --steps 20000 --bs 64 --L 1024 --kmax 128 --eval_K 32,64,128,256,512 --eval_lens 4096,65536,1048576 --eval_modes early --chunk 4096 --seed $s" >> $J
  done
done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
echo "[$(date)] $(wc -l < $J) jobs"
xargs -P ${PAR:-15} -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] P5_DONE"
