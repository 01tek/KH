#!/usr/bin/env bash
# LM v4: min_decay 0.35 closes the scale-invariant decay escape (RMSNorm read makes uniform decay lossless until fp32 underflow).
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm4
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens ${TOKENS:-5e8} --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"
H="--layout m3,hais,m3,hais,m3,hais,m3,hais --archival_heads 4"
run() { local n=$1; shift
  $PY train_lm.py "$@" $C --out runs_lm4/$n > runs_lm4/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm4/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm4/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm4/failures.txt; }
run lm4_hais_md035 $H --min_decay 0.35 --lr 2e-3 &
run lm4_hais_md035_l101 $H --min_decay 0.35 --write_l1 0.01 --l1_scope archival --lr 2e-3 &
wait; echo LM4_DONE
