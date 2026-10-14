#!/usr/bin/env bash
# LM v11: age-targeted long-range queries (log-uniform ages up to 2000 segments, facts persist + refresh on query).
# v2-v10 trained with median query age 19 segments and 0% >= 512 -- the model never practised 131K+.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_2b; mkdir -p runs_lm11
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens 1e9 --seed 0 --v_norm --lr 2e-3 \
  --fact_rate 0.1 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform --max_pending 4096"
H="--layout gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 4 --min_decay 0.35"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm11/$n > runs_lm11/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm11/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm11/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm11/failures.txt; }
run lm11_gdnhais_age $H &
run lm11_gdnhais_age_tau $H --arch_tau 0.1 &
run lm11_gdn_age --layout gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn &
wait; echo LM11_DONE
