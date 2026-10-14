#!/usr/bin/env bash
# LM v6: v_norm bounds the delta-rule state (v5 state norms grew without bound on long streams; ppl 1403).
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm6
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens ${TOKENS:-5e8} --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"
H="--layout gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 4 --min_decay 0.35"
run() { local n=$1; shift
  $PY train_lm.py "$@" $C --out runs_lm6/$n > runs_lm6/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm6/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm6/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm6/failures.txt; }
run lm6_gdnhais_l1005 $H --write_l1 0.005 --l1_scope archival --v_norm --lr 2e-3 &
run lm6_gdnhais $H --v_norm --lr 2e-3 &
run lm6_gdn --layout gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn --v_norm --lr 2e-3 &
wait; echo LM6_DONE
