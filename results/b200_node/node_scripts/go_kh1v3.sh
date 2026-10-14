#!/bin/bash
# KH-1-1.2B v3: v2 + text-write penalty 0.6 (pilot 5 trend), 1B tokens; evals at inference tau 0.97; then harvest
cd $HOME/hais_poc/poc_m3/lm
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; n=kh1_1p2b_v3
until [ -f after_kh1v2.done ]; do sleep 15; done
../../env/bin/torchrun --standalone --nproc_per_node=8 train_lm.py --data $DATA --init_from runs_kh1/kh1_1p2b_t09/ckpt.pt \
  --lfm2 LiquidAI/LFM2-1.2B --lfm2_insert 7,11,15 --lfm2_chunk 2048 --mem_heads 32 --bs 8 --T 4096 --tokens 1e9 --lr 2e-4 \
  --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 \
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau 0.9 --text_write_l1 0.6 --ckpt_every 500 --out runs_kh1/$n > runs_kh1/$n.train.log 2>&1 || { echo FAILED > kh1v3.failed; exit 1; }
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
         --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 --arch_tau $4 --tag _$2_chat$3_tau$4 \
         > runs_kh1/$n.$2.chat$3.tau$4.log 2>&1; }
ev 0 comp 0 0.97 & ev 1 comp 1 0.97 & ev 2 comp_heldout 0 0.97 & ev 3 multi 0 0.97 & ev 4 comp 0 0.9 & ev 5 heldout 0 0.97 &
CUDA_VISIBLE_DEVICES=6 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_kh1/$n.ppl.log 2>&1 &
wait
~/hais_poc/env/bin/python ~/harvest.py runs_kh1/kh1_1p2b runs_kh1/kh1_1p2b_t09 runs_kh1/kh1_1p2b_v3 > ~/harvest_kh1v3.log 2>&1
echo DONE > kh1v3.done
