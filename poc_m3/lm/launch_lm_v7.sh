#!/usr/bin/env bash
# LM v7: replicate lm6_gdnhais (131K in-text recall) on 2 more seeds; push max query age to 1000 segments; 2x memory heads.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm7
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens ${TOKENS:-5e8} --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"
H="--layout gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 4 --min_decay 0.35"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm7/$n > runs_lm7/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm7/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm7/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm7/failures.txt; }
run lm7_gdnhais_s1 $H --v_norm --lr 2e-3 --seed 1 &
run lm7_gdnhais_s2 $H --v_norm --lr 2e-3 --seed 2 &
run lm7_gdnhais_age1k $H --v_norm --lr 2e-3 --max_age 1000 --age_warmup 10000 &
run lm7_gdnhais_mh16 $H --v_norm --lr 2e-3 --mem_heads 16 &
wait; echo LM7_DONE
