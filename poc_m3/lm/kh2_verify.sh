#!/usr/bin/env bash
# Verify the two-tier result: (1) rerun the standard stage 4 of seeds 1 and 2 (run-to-run control: same init, data and seed),
# (2) two-tier stage 4 on seed 0 (does it hurt the good seed?), then (3) two-tier stage 3 on LFM2.5-8B-A1B (kh2_8b.sh).
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2
COMMON="--data $DATA --lfm2 LiquidAI/LFM2-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 --bs 32 --T 2048 --lr 3e-4
  --wd 0.0 --lm_loss_frac 0.1 --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 500"
run() {  # name seed init extra
  local P=runs_kh2/kh2_350m_$1
  grep -q '"event": "done"' $P.train.log 2>/dev/null && return 0
  echo "start $1 $(date -u +%H:%M)" >> runs_kh2/progress
  $PY train_lm.py $COMMON --seed $2 --init_from $3 --arch_tau 0.97 --text_write_l1 0.2 --tokens 1e8 $4 --out $P > $P.train.log 2>&1 \
    || { echo "FAILED $1" >> runs_kh2/progress; return 1; }
  echo "done $1 $(date -u +%H:%M)" >> runs_kh2/progress
}
ev() {  # name eval-flags tag eval-seed lens
  $PY eval_lm.py --run runs_kh2/kh2_350m_$1 --data $DATA --lens $5 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp \
    --chat_frac 1 --eval_seed $4 $2 --tag _comp_chat1$3 > runs_kh2/kh2_350m_$1.eval$3.log 2>&1
}
TIER_T="--arch_tau_hi 0.995 --hi_frac 0.5"; TIER_E="--arch_tau 0.99 --arch_tau_hi 0.998 --hi_frac 0.5"; CTRL_E="--arch_tau 0.99"
run seed1_ctrl 1 runs_lfm2/kh1_350m_seed1_s3/ckpt.pt ""
run seed2_ctrl 2 runs_lfm2/kh1_350m_seed2_s3/ckpt.pt ""
run seed0_tier 0 runs_lfm2/lfm2_350m_pilot4/ckpt.pt "$TIER_T"
ev seed1_ctrl "$CTRL_E" _tau0.99_dev 1234 16384,131072,1048576 & ev seed2_ctrl "$CTRL_E" _tau0.99_dev 1234 16384,131072,1048576 &
ev seed0_tier "$TIER_E" _tier_dev 1234 16384,131072,1048576 & wait
ev seed1_ctrl "$CTRL_E" _tau0.99_fresh4321 4321 1048576 & ev seed2_ctrl "$CTRL_E" _tau0.99_fresh4321 4321 1048576 &
ev seed0_tier "$TIER_E" _tier_fresh4321 4321 1048576 & wait
echo "VERIFY DONE $(date -u +%H:%M)" >> runs_kh2/progress; echo DONE > runs_kh2/verify.done
NOWAIT=1 VARIANT=tier bash kh2_8b.sh
