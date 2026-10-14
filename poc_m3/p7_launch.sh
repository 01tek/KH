#!/usr/bin/env bash
# Pilot 7: min_decay 0.35 recipe. PART=fix: rescue undertrained seeds + decoys at 40K steps. PART=cap: capacity scaling.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
PY=../env/bin/python; OUT=runs/p7; mkdir -p $OUT; J=$OUT/jobs_$PART.txt; : > $J
EV="--eval_lens 256,4096,65536,1048576 --eval_modes early,uniform --eval_K 1,8,32"
R="--layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --bs 128 --L 256 --kmax 32"
if [ "$PART" = fix ]; then
  for s in 3 5 8; do
    echo "p7_md035_40k_s$s $R --lr 3e-3 --steps 40000 --seed $s $EV" >> $J
    echo "p7_md035_lr1e3_s$s $R --lr 1e-3 --steps 40000 --seed $s $EV" >> $J
  done
  for s in 6 7 8 9 10 11; do echo "p7_md035_decoy_40k_s$s $R --lr 3e-3 --steps 40000 --decoy 0.1 --seed $s $EV" >> $J; done
else
  for s in 0 1; do for mh in 4 8 16; do
    echo "p7_cap_h${mh}_s$s --layout hais,hais --write_l1 0.05 --l1_scope archival --min_decay 0.35 --state_passing 0.95 --mem_heads $mh --lr 3e-3 --steps 20000 --bs 64 --L 1024 --kmax 128 --eval_K 32,64,128,256,512 --eval_lens 4096,65536,1048576 --eval_modes early --chunk 4096 --seed $s" >> $J
  done; done
fi
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P ${PAR:-12} -L 1 bash -c 'run_one "$@"' _ < $J
echo "[$(date)] P7_${PART}_DONE"
