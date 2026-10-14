#!/usr/bin/env bash
# Pilot-0: 4 arms, 1 seed, train at L=1024 (K<=8 facts), evaluate by streaming 1K -> 1M tokens.
# Purpose: calibrate the harness and get first numbers, NOT a claim (1 seed, 1 LR).
set -euo pipefail
cd "$(dirname "$0")"
PY=../m3poc_env/bin/python
COMMON="--steps ${STEPS:-2000} --bs 32 --L 1024 --kmax 8 --lr ${LR:-1e-3} --seed ${SEED:-0} \
  --eval_lens 1024,4096,16384,65536,262144,1048576 --eval_K 1,4,8 --eval_modes uniform,early"
OUT=runs/pilot0
mkdir -p $OUT
$PY train.py --layout m3,m3,m3,m3     $COMMON --out $OUT/m3      > $OUT/m3.log 2>&1
$PY train.py --layout gdn,gdn,gdn,gdn $COMMON --out $OUT/gdn     > $OUT/gdn.log 2>&1
$PY train.py --layout m3,gdn,m3,m3    $COMMON --out $OUT/m3_gdn  > $OUT/m3_gdn.log 2>&1
$PY train.py --layout m3,hais,m3,m3   $COMMON --out $OUT/m3_hais > $OUT/m3_hais.log 2>&1
echo PILOT0_DONE
