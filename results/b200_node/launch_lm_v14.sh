#!/usr/bin/env bash
# LM v14 (H200 #1): (a) the lm13 fine-tune + capped GDN memory (half-life 8K): trains timescale separation into lm8;
# (b) a 50M from-scratch preview of the node candidate recipe: age-targeted, train tau 0.5, GDN half-life 8K, 10 phrasings.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm13 runs_lm14
A="--fact_rate 0.1 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 2000 --ans_weight 5 --age_target loguniform --max_pending 4096"
ev() { local r=$1 tpl=$2; shift 2
  for t in "" 0.9; do
    $PY eval_lm.py --run $r --data $DATA --lens 16384,131072,1048576 --n 20 --K 1,8 --tasks facts --no_ppl --templates $tpl \
      ${t:+--arch_tau $t} --tag _${tpl}_tau${t:-train} > $r.$tpl.tau${t:-train}.log 2>&1
  done; }
( r=runs_lm13/lm13_ft_age_tau0.5_hl8k
  $PY train_lm.py --data $DATA --init_from lm8_gdnhais_model.pt --d_model 768 --mem_heads 12 --archival_heads 6 --min_decay 0.35 \
    --v_norm --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --bs 16 --T 1024 --tokens 3e8 --lr 5e-4 --seed 0 \
    $A --age_warmup 0 --arch_tau 0.5 --gdn_half_life 8192 --out $r > $r.train.log 2>&1 && ev $r secret
  $PY eval_lm.py --run $r --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > $r.ppl.log 2>&1 ) &
( r=runs_lm14/lm14_cand_50m
  $PY train_lm.py --data $DATA --d_model 512 --mem_heads 8 --archival_heads 4 --min_decay 0.35 --v_norm \
    --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais --bs 32 --T 1024 --tokens 1e9 --lr 2e-3 --seed 0 \
    $A --age_warmup 20000 --arch_tau 0.5 --gdn_half_life 8192 --templates multi --out $r > $r.train.log 2>&1 &&
  ev $r multi && ev $r heldout
  $PY eval_lm.py --run $r --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > $r.ppl.log 2>&1 ) &
wait; echo LM14_DONE
