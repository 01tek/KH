#!/usr/bin/env bash
# Pilot-1 extra arm: hais,hais (HAIS mechanism on the strong pure-delta baseline). Waits for re-evals.
cd "$(dirname "$0")"
while ps -eo cmd | grep -q "[r]eeval.sh"; do sleep 30; done
../m3poc_env/bin/python train.py --layout hais,hais --L 256 --kmax 32 --bs 128 --lr 3e-3 --steps 12000 \
  --state_passing 0.5 --seed 0 --eval_lens 256,4096,65536,1048576 --eval_K 1,8,32 --eval_modes early,uniform \
  --out runs/pilot1/haishais > runs/pilot1/haishais.log 2>&1
echo QUEUED_ARM_DONE
