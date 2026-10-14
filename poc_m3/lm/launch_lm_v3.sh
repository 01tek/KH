#!/usr/bin/env bash
# LM v3: why did the HAIS LM learn no recall (v2: ans_acc 0.06 vs GDN 0.73)? Vary the write penalty / min_decay.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm3
C="--data $DATA --d_model 512 --mem_heads 8 --bs 32 --T 1024 --tokens ${TOKENS:-5e8} --seed ${SEED:-0} \
  --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 200 --age_warmup 7000 --ans_weight 5"
H="--layout m3,hais,m3,hais,m3,hais,m3,hais --archival_heads 4"
run() { local n=$1; shift
  $PY train_lm.py "$@" $C --out runs_lm3/$n > runs_lm3/$n.train.log 2>&1 &&
  $PY eval_lm.py --run runs_lm3/$n --data $DATA --lens 2048,16384,131072,1048576 --n 20 --K 1,8 > runs_lm3/$n.eval.log 2>&1 ||
  echo "FAILED $n" >> runs_lm3/failures.txt; }
run lm3_hais_l101 $H --min_decay 0.0217 --write_l1 0.01 --l1_scope archival --lr 2e-3 &
run lm3_hais_l10  $H --min_decay 0.0217 --lr 2e-3 &
run lm3_hais_nomd $H --write_l1 0.01 --l1_scope archival --lr 2e-3 &
run lm3_gdn_s1 --layout m3,gdn,m3,gdn,m3,gdn,m3,gdn --lr 2e-3 --seed 1 &
run lm3_m3_lr1e3 --layout m3,m3,m3,m3,m3,m3,m3,m3 --lr 1e-3 &
wait; echo LM3_DONE
