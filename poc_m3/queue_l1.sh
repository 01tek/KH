#!/usr/bin/env bash
# hais,hais with long-chain state passing + write-sparsity penalty on archival heads only.
cd "$(dirname "$0")"
while ps -eo cmd | grep -q "[q]ueue_sp95.sh\|[q]ueue_verify.sh\|[q]ueue_haishais.sh"; do sleep 30; done
../m3poc_env/bin/python train.py --layout hais,hais --state_passing 0.95 --write_l1 0.05 --l1_scope archival \
  --L 256 --kmax 32 --bs 128 --lr 3e-3 --steps 12000 --seed 0 --eval_lens 256,4096,65536,1048576 \
  --eval_K 1,8,32 --eval_modes early,uniform --out runs/pilot1/haishais_sp95_l1arch > runs/pilot1/haishais_sp95_l1arch.log 2>&1
echo L1_DONE
