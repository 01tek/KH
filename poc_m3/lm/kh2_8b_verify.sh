#!/usr/bin/env bash
# Verify the LFM2.5-8B two-tier gain: (1) control = rerun the standard stage 3 (tau 0.97) from stage 2 with the same seed;
# (2) replicate = two-tier stage 3 with seed 1. Evals of (1) overlap the training of (2).
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; D=$HOME/data/fwe_lfm25
COMMON="--lfm2 LiquidAI/LFM2.5-8B-A1B --lfm2_insert 11,17,23 --lfm2_chunk 2048 --mem_heads 32 --bs 8 --T 4096 --lr 3e-4
  --wd 0.0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000
  --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 250"
train() {  # name seed extra
  local P=runs_lfm25/l25_8b_s3_$1
  grep -q '"event": "done"' $P.train.log 2>/dev/null && return 0
  echo "start $1 $(date -u +%H:%M)" >> runs_lfm25/kh2_8b_verify.progress
  $PY train_lm.py --data $D $COMMON --seed $2 --init_from runs_lfm25/l25_8b_s2/ckpt.pt --arch_tau 0.97 --text_write_l1 0.2 --tokens 1e8 \
    $3 --out $P > $P.train.log 2>&1 || { echo "FAILED $1" >> runs_lfm25/kh2_8b_verify.progress; return 1; }
  echo "done $1 $(date -u +%H:%M)" >> runs_lfm25/kh2_8b_verify.progress
}
evals() {  # name eval-flags
  local P=runs_lfm25/l25_8b_s3_$1
  $PY eval_lm.py --run $P --data $D --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp \
    --chat_frac 1 --eval_seed 1234 $2 --tag _comp_chat1_dev > $P.eval_dev.log 2>&1 &
  $PY eval_lm.py --run $P --data $D --lens 1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp \
    --chat_frac 1 --eval_seed 4321 $2 --tag _comp_chat1_fresh4321 > $P.eval_fresh.log 2>&1 &
}
train ctrl 0 ""
evals ctrl "--arch_tau 0.99"
train tier_seed1 1 "--arch_tau_hi 0.995 --hi_frac 0.5"
wait
evals tier_seed1 "--arch_tau 0.99 --arch_tau_hi 0.998 --hi_frac 0.5"
wait
echo "ALL DONE $(date -u +%H:%M)" >> runs_lfm25/kh2_8b_verify.progress; echo DONE > runs_lfm25/kh2_8b_verify.done
