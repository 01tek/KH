#!/bin/bash
# KH-1-1.2B v4 = v2 + the pilot-6 step (train tau 0.97, text-write L1 0.2; 350M: 1M recall 0.75 -> 0.93 at inference tau
# 0.99), after the Qwen stage-3 chain. ~300M tokens (~6 min on 8x B200), evals at tau 0.99 / 0.995, harvest.
cd ~/hais_poc/poc_m3/lm
until [ -f q4b_s3.done ] || [ -f q4b_s3.failed ] || [ -f q4b.failed ]; do sleep 30; done
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; n=kh1_1p2b_v4
../../env/bin/torchrun --standalone --nproc_per_node=8 train_lm.py --data $DATA --init_from runs_kh1/kh1_1p2b_t09/ckpt.pt \
  --lfm2 LiquidAI/LFM2-1.2B --lfm2_insert 7,11,15 --lfm2_chunk 2048 --mem_heads 32 --bs 8 --T 4096 --tokens 3e8 --lr 3e-4 \
  --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 \
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau 0.97 --text_write_l1 0.2 --ckpt_every 250 --out runs_kh1/$n > runs_kh1/$n.train.log 2>&1 || { echo FAILED > kh1v4.failed; exit 1; }
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
         --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 --arch_tau $4 --tag _$2_chat$3_tau$4 \
         > runs_kh1/$n.$2.chat$3.tau$4.log 2>&1; }
ev 0 comp 0 0.99 & ev 1 comp 1 0.99 & ev 2 comp_heldout 0 0.99 & ev 3 comp 0 0.995 & ev 4 comp 1 0.995 & ev 5 multi 0 0.99 &
CUDA_VISIBLE_DEVICES=6 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_kh1/$n.ppl.log 2>&1 &
wait
~/hais_poc/env/bin/python ~/harvest.py runs_kh1/$n > ~/harvest_kh1v4.log 2>&1
echo DONE > kh1v4.done
