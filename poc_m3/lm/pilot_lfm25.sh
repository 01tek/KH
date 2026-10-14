#!/bin/bash
# De-risk the LFM2.5 B200 run on H200 #1: LFM2.5 data (128K vocab, uint32) + a short stage-1 graft pilot on LFM2.5-8B-A1B.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python
[ -f $HOME/data/fwe_lfm25/meta.json ] || nice -n 5 $PY prep_data.py --out $HOME/data/fwe_lfm25 --shards 1 --train_tokens 300000000 \
  --val_tokens 20000000 --workers 14 --tokenizer LiquidAI/LFM2.5-8B-A1B > runs_lfm25/prep.log 2>&1
$PY train_lm.py --data $HOME/data/fwe_lfm25 --lfm2 LiquidAI/LFM2.5-8B-A1B --lfm2_insert 11,17,23 --lfm2_chunk 2048 --mem_heads 32 \
  --bs 8 --T 4096 --tokens 1.5e8 --lr 3e-4 --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 \
  --drill_k 8 --max_age 64 --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix \
  --chat_frac 0.5 --arch_tau 0.5 --text_write_l1 0.05 --ckpt_every 250 --out runs_lfm25/lfm25_8b_s1_pilot > runs_lfm25/lfm25_8b_s1_pilot.train.log 2>&1
echo DONE > runs_lfm25/pilot.done
