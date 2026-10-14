#!/usr/bin/env bash
# LM v9: reliability -- 3 seeds of gdn,hais + v_norm at 2x training (1B tokens, 30.5K steps), long-query curriculum (max_age 1000).
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm9
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens ${TOKENS:-5e8} --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"
H="--layout gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 4 --min_decay 0.35"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm9/$n > runs_lm9/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm9/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm9/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm9/failures.txt; }
run lm9_gdnhais_s0 $H --v_norm --lr 2e-3 --seed 0 --max_age 1000 --age_warmup 20000 --tokens 1e9 &
run lm9_gdnhais_s1 $H --v_norm --lr 2e-3 --seed 1 --max_age 1000 --age_warmup 20000 --tokens 1e9 &
run lm9_gdnhais_s2 $H --v_norm --lr 2e-3 --seed 2 --max_age 1000 --age_warmup 20000 --tokens 1e9 &
wait; echo LM9_DONE
