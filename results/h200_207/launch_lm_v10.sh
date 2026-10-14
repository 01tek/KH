#!/usr/bin/env bash
# LM v10: budgeted archival writes in the LM (top-m per 64-token block), 50M, 1B tokens, seed 0. Compare with lm9_gdnhais_s0.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_2b; mkdir -p runs_lm10
C="--data $DATA --d_model 512 --mem_heads 8 --archival_heads 4 --bs 32 --T 1024 --tokens 1e9 --seed 0 --v_norm --min_decay 0.35 --lr 2e-3 \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 1000 --age_warmup 20000 --ans_weight 5 --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm10/$n > runs_lm10/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm10/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm10/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm10/failures.txt; }
run lm10_wb2 --write_budget 2 &
run lm10_wb4 --write_budget 4 &
wait; echo LM10_DONE
