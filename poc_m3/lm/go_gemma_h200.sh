#!/bin/bash
# KH-1 on Gemma 4 E2B (google/gemma-4-E2B-it), one H200 under MPS. Stage 1 (tau 0.5, L1 0.05, 600M) -> stage 2 (tau 0.9,
# L1 0.2, 250M) -> stage 3 (tau 0.97, L1 0.2, 150M); evals: stage 3 at 0.99 and stage 2 at 0.98, development and fresh streams,
# memory off, perplexity. Batch 4 x 4096 (batch 8 does not fit next to the 262K-vocabulary head).
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
while [ ! -f $HOME/data/fwe_gemma4/meta.json ]; do sleep 30; done
PY=../../env_qwen/bin/python; D=$HOME/data/fwe_gemma4; P=runs_gemma/g4e2b
COMMON="--data $D --lfm2 google/gemma-4-E2B-it --lfm2_insert 16,25,34 --lfm2_chunk 2048 --mem_heads 24 --bs 4 --T 4096 --lr 3e-4
  --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000
  --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 250"
stage() {  # name init tau tw tokens
  grep -q '"event": "done"' ${P}_$1.train.log 2>/dev/null && return 0
  echo "start $1 $(date -u +%H:%M)" >> $P.progress
  $PY train_lm.py $COMMON ${2:+--init_from $2} --arch_tau $3 --text_write_l1 $4 --tokens $5 --out ${P}_$1 > ${P}_$1.train.log 2>&1 \
    || { echo "FAILED $1 $(date -u +%H:%M)" >> $P.progress; exit 1; }
  echo "done $1 $(date -u +%H:%M)" >> $P.progress
}
ev() {  # stage templates chat tau eval-seed lens tag-suffix [extra]
  $PY eval_lm.py --run ${P}_$1 --data $D --lens $6 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 \
    --arch_tau $4 --eval_seed $5 --tag _$2_chat$3_tau$4$7 $8 > ${P}_$1.$2.chat$3.tau$4$7.log 2>&1
}
stage s1 "" 0.5 0.05 6e8
ev s1 comp 1 0.9 1234 16384 _16k &
stage s2 ${P}_s1/ckpt.pt 0.9 0.2 2.5e8
ev s2 comp 1 0.98 1234 16384,131072,1048576 &
stage s3 ${P}_s2/ckpt.pt 0.97 0.2 1.5e8
wait
ev s3 comp 1 0.99 1234 16384,131072,1048576 & ev s3 comp_heldout 0 0.99 1234 16384,1048576 &
ev s3 comp 1 0.99 4321 1048576 _fresh4321 & wait
$PY eval_lm.py --run ${P}_s3 --data $D --lens 16384 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 \
  --no_memory --tag _comp_chat1_nomem > ${P}_s3.nomem.log 2>&1
echo "ALL DONE $(date -u +%H:%M)" >> $P.progress; echo DONE > $P.done
