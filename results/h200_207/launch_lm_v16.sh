#!/usr/bin/env bash
# LM v16 (H200 #1): fine-tune lm8 (117M) for 300M tokens on CORRECTED long-range data (loguniform_once: log-uniform query
# ages up to 2000 segments, each fact asked once, so nothing can be memorised in weights), lower LR than v13.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm16
C="--data $DATA --init_from lm8_gdnhais_model.pt --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --d_model 768 \
  --mem_heads 12 --archival_heads 6 --min_decay 0.35 --v_norm --bs 32 --T 1024 --tokens 3e8 --lr 3e-4 --seed 0 \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 0 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --compile"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm16/$n > runs_lm16/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_lm16/failures.txt; return; }
  for t in "" 0.9; do
    $PY eval_lm.py --run runs_lm16/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      ${t:+--arch_tau $t} --tag _tau${t:-train} > runs_lm16/$n.tau${t:-train}.log 2>&1
  done
  $PY eval_lm.py --run runs_lm16/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_lm16/$n.ppl.log 2>&1; }
run lm16_ft_tau_hl8k --arch_tau 0.5 --gdn_half_life 8192 &
run lm16_ft_tau --arch_tau 0.5 &
wait; echo LM16_DONE
