#!/usr/bin/env bash
# Phase-3 LM runs: same data, tokens, seed and block structure; only the memory layer differs.
# Usage (on the GPU box): cd ~/hais_poc/poc_m3/lm && TOKENS=5e8 nohup ./launch_lm.sh > lm.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python
DATA=${DATA:-$HOME/data/fwe_1b}
TOKENS=${TOKENS:-5e8}
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens $TOKENS --lr 2e-3 --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"  # v2 recall curriculum
HAIS_RECIPE=${HAIS_RECIPE:-"--archival_heads 4 --min_decay 0.0217 --write_l1 0.05 --l1_scope archival"}
run() {  # name, args...
  local n=$1; shift
  $PY train_lm.py "$@" $C --out runs_lm/$n > runs_lm/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm/failures.txt
}
mkdir -p runs_lm
run lm_hais --layout m3,hais,m3,hais,m3,hais,m3,hais $HAIS_RECIPE &
run lm_gdn  --layout m3,gdn,m3,gdn,m3,gdn,m3,gdn &
run lm_m3   --layout m3,m3,m3,m3,m3,m3,m3,m3 &
wait
echo LM_DONE
