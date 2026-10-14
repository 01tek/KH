#!/usr/bin/env bash
# LFM2-350M + archival memory pilot (H200 #1). LFM2 frozen, attention/convs limited to 1024-token chunks (constant
# memory), archival adapters after layers 7/11/15 are the only long-range memory. Segments = 2 chunks: drill facts land in
# chunk 1 and their queries in chunk 2, so only the archival memory can answer and the write gets the gradient
# (the sweep-5 routing rule by construction). 318 compositional phrasings; eval on trained + unseen phrasings.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; mkdir -p runs_lfm2
N=${N:-lfm2_350m_pilot5}
$PY train_lm.py --data $DATA ${INIT:+--init_from $INIT} --lfm2 LiquidAI/LFM2-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 \
  --bs ${BS:-32} --T 2048 --tokens ${TOKENS:-3e8} --lr 3e-4 --wd 0.0 --seed 0 --lm_loss_frac 0.1 \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau ${TAU:-0.9} --text_write_l1 ${TW:-0.2} --ckpt_every 500 --out runs_lfm2/$N > runs_lfm2/$N.train.log 2>&1 || { echo "FAILED $N"; exit 1; }
for tpl in comp comp_heldout heldout multi; do
  $PY eval_lm.py --run runs_lfm2/$N --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
    --templates $tpl --arch_tau 0.9 --tag _${tpl}_tau0.9 > runs_lfm2/$N.$tpl.tau0.9.log 2>&1
done
$PY eval_lm.py --run runs_lfm2/$N --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_lfm2/$N.ppl.log 2>&1
echo LFM2_PILOT_DONE
