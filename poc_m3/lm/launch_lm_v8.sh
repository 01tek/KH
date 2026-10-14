#!/usr/bin/env bash
# LM v8 (H200 #2): scale the winning LM (gdn,hais + v_norm, no penalty) to d=768 x 12 blocks, 1B tokens, vs a pure-GDN baseline.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
PY=../../env/bin/python; DATA=$HOME/data/fwe_2b; mkdir -p runs_lm8
C="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens ${TOKENS:-1e9} --lr 1.5e-3 --seed 0 --v_norm \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 1000 --age_warmup 15000 --ans_weight 5"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm8/$n > runs_lm8/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm8/$n --data $DATA --lens 2048,16384,131072,1048576 --n 40 --K 1,8 > runs_lm8/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm8/failures.txt; }
run lm8_gdnhais_d768 --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 &
run lm8_gdn_d768 --layout gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn,gdn &
wait; echo LM8_DONE
