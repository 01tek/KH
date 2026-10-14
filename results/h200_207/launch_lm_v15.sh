#!/usr/bin/env bash
# LM v15 (H200 #2): lm8's exact recipe (random-pop queries, each fact asked once) + the new levers, from scratch, 1B tokens:
# train-time write threshold tau 0.5, with and without a GDN half-life cap of 8K tokens.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_2b; mkdir -p runs_lm15
C="--data $DATA --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --d_model 768 --mem_heads 12 --archival_heads 6 \
  --min_decay 0.35 --v_norm --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --seed 0 --fact_rate 1.0 --query_rate 1.0 --drill_frac 0.25 \
  --drill_k 8 --max_age 1000 --age_warmup 15000 --ans_weight 5 --compile"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm15/$n > runs_lm15/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_lm15/failures.txt; return; }
  for t in "" 0.9; do
    $PY eval_lm.py --run runs_lm15/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      ${t:+--arch_tau $t} --tag _tau${t:-train} > runs_lm15/$n.tau${t:-train}.log 2>&1
  done
  $PY eval_lm.py --run runs_lm15/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_lm15/$n.ppl.log 2>&1; }
run lm15_tau_hl8k --arch_tau 0.5 --gdn_half_life 8192 &
run lm15_tau --arch_tau 0.5 &
wait; echo LM15_DONE
