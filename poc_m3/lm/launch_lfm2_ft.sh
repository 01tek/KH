#!/usr/bin/env bash
# LFM2-350M + archival memory fine-tune (H200 #1) from an earlier pilot, then evals at several inference thresholds.
#   INIT=runs_lfm2/lfm2_350m_pilot4/ckpt.pt N=lfm2_350m_pilot6 TAU=0.97 TW=0.2 EV_TAUS="0.97 0.99" bash launch_lfm2_ft.sh
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; mkdir -p runs_lfm2
$PY train_lm.py --data $DATA --init_from $INIT --lfm2 LiquidAI/LFM2-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 \
  --bs ${BS:-32} --T 2048 --tokens ${TOKENS:-1e8} --lr ${LR:-3e-4} --wd 0.0 --seed 0 --lm_loss_frac 0.1 \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age ${MAXAGE:-64} --age_warmup 2000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau $TAU --text_write_l1 $TW --ckpt_every 500 --out runs_lfm2/$N > runs_lfm2/$N.train.log 2>&1 || { echo "FAILED $N"; exit 1; }
for t in ${EV_TAUS:-0.97}; do
  for spec in "comp 0" "comp 1" "comp_heldout 0"; do
    set -- $spec
    $PY eval_lm.py --run runs_lfm2/$N --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates $1 --chat_frac $2 --arch_tau $t --tag _$1_chat$2_tau$t > runs_lfm2/$N.$1.chat$2.tau$t.log 2>&1 &
  done
done
wait; echo LFM2_FT_DONE $N
