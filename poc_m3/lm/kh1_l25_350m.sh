#!/usr/bin/env bash
# The KH-1-350M recipe on LFM2.5-350M (same architecture and tokenizer ids as LFM2-350M), one H200 under MPS:
#   S1 tau 0.5 / text-write L1 0.05 / 300M -> S2 0.9 / 0.05 / 100M -> S3 0.9 / 0.2 / 100M -> S4 0.97 / 0.2 / 100M, run at 0.99.
# LFM2.5's tokenizer needs transformers 5, so this uses env_qwen. Data: fwe_lfm2 (identical ids for plain text).
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; DATA=$HOME/data/fwe_lfm2; S=${SEED:-0}; P=runs_lfm25/kh1_l25_350m_seed$S; mkdir -p runs_lfm25
COMMON="--data $DATA --lfm2 LiquidAI/LFM2.5-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 --bs 32 --T 2048 --lr 3e-4
  --wd 0.0 --seed $S --lm_loss_frac 0.1 --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 500"
stage() {  # name init tau tw tokens
  grep -q '"event": "done"' ${P}_$1.train.log 2>/dev/null && return 0
  echo "start $1 $(date -u +%H:%M)" >> $P.progress
  $PY train_lm.py $COMMON ${2:+--init_from ${P}_$2/ckpt.pt} --arch_tau $3 --text_write_l1 $4 --tokens $5 --out ${P}_$1 \
    > ${P}_$1.train.log 2>&1 || { echo "FAILED $1 $(date -u +%H:%M)" >> $P.progress; exit 1; }
  echo "done $1 $(date -u +%H:%M)" >> $P.progress
}
stage s1 "" 0.5 0.05 3e8
stage s2 s1 0.9 0.05 1e8
stage s3 s2 0.9 0.2 1e8
stage s4 s3 0.97 0.2 1e8
ev() {  # stage templates chat tau [data] [lens] [K] [tag-suffix] [eval-seed]
  $PY eval_lm.py --run ${P}_$1 --data ${5:-$DATA} --lens ${6:-16384,131072,1048576} --n 40 --K ${7:-1,8} --tasks facts --no_ppl \
    --batch 8 --templates $2 --chat_frac $3 --arch_tau $4 --eval_seed ${9:-1234} --tag _$2_chat$3_tau$4$8 > ${P}_$1.$2.chat$3.tau$4$8.log 2>&1
}
ev s4 comp 0 0.99 & ev s4 comp 1 0.99 & ev s4 comp_heldout 0 0.99 & ev s3 comp 1 0.97 &
ev s4 comp 1 0.99 $HOME/data/books_lfm2 1048576 1 _books &
ev s4 comp 1 0.99 $DATA 1048576 1,8 _fresh4321 4321 &
$PY eval_lm.py --run ${P}_s4 --data $DATA --lens 16384 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 \
  --no_memory --tag _comp_chat1_nomem > ${P}_s4.nomem.log 2>&1 &
$PY eval_lm.py --run ${P}_s4 --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > ${P}_s4.ppl.log 2>&1 &
wait
echo "ALL DONE $(date -u +%H:%M)" >> $P.progress; echo DONE > ${P}.done
