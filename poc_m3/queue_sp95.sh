#!/usr/bin/env bash
# Long-chain state passing (p=0.95 => ~20 chained batches ~ 5K-token effective streams) to create pressure
# for sparse writes in archival heads. Control: same for gdn,gdn. Runs after the queued control finishes.
cd "$(dirname "$0")"
while ps -eo cmd | grep -q "[q]ueue_verify.sh\|[q]ueue_haishais.sh\|[r]eeval.sh"; do sleep 30; done
PY=../m3poc_env/bin/python
C="--L 256 --kmax 32 --bs 128 --lr 3e-3 --steps 12000 --seed 0 --eval_lens 256,4096,65536,1048576 --eval_K 1,8,32 --eval_modes early,uniform"
$PY train.py --layout hais,hais --state_passing 0.95 $C --out runs/pilot1/haishais_sp95 > runs/pilot1/haishais_sp95.log 2>&1
$PY train.py --layout gdn,gdn   --state_passing 0.95 $C --out runs/pilot1/gdn_sp95 > runs/pilot1/gdn_sp95.log 2>&1
echo SP95_DONE
