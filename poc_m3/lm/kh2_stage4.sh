#!/usr/bin/env bash
# KH-2 candidates: redo only the last stage (S4, 100M tokens) of the KH-1-350M recipe from the stage-3 checkpoints of seeds
# 1 and 2 (the seeds whose 1M recall faded: 0.675 / 0.575), with the exact S4 settings plus one change:
#   tier : two-tier thresholds, the last half of each layer's archival heads train at 0.995 (run at 0.998), the rest at 0.97 (run at 0.99)
#   quiet: quiet heads, the last half of each layer's archival heads get 5x the text-write L1
# Starts after the LFM2.5-8B chain; evals on development (1234) and fresh (4321) streams, chat questions.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
while [ ! -f runs_lfm25/l25_8b.done ] && [ -z "$NOWAIT" ]; do sleep 60; done
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; mkdir -p runs_kh2
COMMON="--data $DATA --lfm2 LiquidAI/LFM2-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 --bs 32 --T 2048 --lr 3e-4
  --wd 0.0 --lm_loss_frac 0.1 --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 500"
run() {  # seed variant extra-train-flags
  local P=runs_kh2/kh2_350m_seed$1_$2
  grep -q '"event": "done"' $P.train.log 2>/dev/null && return 0
  echo "start seed$1 $2 $(date -u +%H:%M)" >> runs_kh2/progress
  $PY train_lm.py $COMMON --seed $1 --init_from runs_lfm2/kh1_350m_seed$1_s3/ckpt.pt --arch_tau 0.97 --text_write_l1 0.2 \
    --tokens 1e8 $3 --out $P > $P.train.log 2>&1 || { echo "FAILED seed$1 $2" >> runs_kh2/progress; return 1; }
  echo "done seed$1 $2 $(date -u +%H:%M)" >> runs_kh2/progress
}
ev() {  # seed variant eval-flags tag eval-seed lens
  local P=runs_kh2/kh2_350m_seed$1_$2
  $PY eval_lm.py --run $P --data $DATA --lens $6 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 \
    --eval_seed $5 $3 --tag _comp_chat1$4 > $P.eval$4.log 2>&1
}
TIER_T="--arch_tau_hi 0.995 --hi_frac 0.5"; TIER_E="--arch_tau 0.99 --arch_tau_hi 0.998 --hi_frac 0.5"
QUIET_T="--quiet_frac 0.5 --quiet_mult 5"; QUIET_E="--arch_tau 0.99"
for s in 1 2; do
  run $s tier "$TIER_T"; run $s quiet "$QUIET_T"
done
for s in 1 2; do
  ev $s tier "$TIER_E" _tier_dev 1234 16384,131072,1048576 & ev $s quiet "$QUIET_E" _tau0.99_dev 1234 16384,131072,1048576 &
  ev $s tier "$TIER_E" _tier_fresh4321 4321 1048576 & ev $s quiet "$QUIET_E" _tau0.99_fresh4321 4321 1048576 &
  wait
done
echo "ALL DONE $(date -u +%H:%M)" >> runs_kh2/progress; echo DONE > runs_kh2/kh2.done
