#!/bin/bash
# KH-1-Q stage 3: tau 0.97 from stage 2 (the pilot-6 step: 350M 1M-recall 0.75 -> 0.93 at inference tau 0.99), then one
# parallel batch of evals (stage 3 at tau 0.99 / 0.995, stage 2 at 0.97, stage 1 at 0.9, memory-off control), the
# 1.3M-token book demo, and a harvest. Late settings (S3_TOKENS, S3_TAU, S3_TW, S3_EXTRA) go in qwen_s3.env.
cd ~/hais_poc/poc_m3/lm
until [ -f q4b.done ] || [ -f q4b.failed ]; do sleep 30; done
[ -f q4b.failed ] && exit 1
: > qwen_overrides.env  # drop the stage-2 overrides (NO_EVAL, S2_TOKENS)
[ -f qwen_s3.env ] && source qwen_s3.env
export TOKENIZERS_PARALLELISM=false
TW="${S3_TW:-0.2} ${S3_EXTRA}" NAME=q4b_s3 INIT=runs_qwen/q4b_s2/ckpt.pt TAU=${S3_TAU:-0.97} TOKENS=${S3_TOKENS:-5e8} NO_EVAL=1 \
  bash launch_qwen.sh || { echo FAILED_S3 > q4b_s3.failed; exit 1; }
PY=../../env_qwen/bin/python; DATA=$HOME/data/fwe_qwen35; B="../../demo/books/pg2600.txt ../../demo/books/pg2701.txt ../../demo/books/pg1342.txt"
ev() {  # gpu run templates chat tau [extra]
  CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_qwen/$2 --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts \
    --no_ppl --batch 8 --templates $3 --chat_frac $4 --arch_tau $5 $6 --tag _$3_chat$4_tau$5$7 > runs_qwen/$2.$3.chat$4.tau$5$7.log 2>&1; }
demo() { CUDA_VISIBLE_DEVICES=$1 $PY ../kh1_demo.py --ckpt runs_qwen/$2/ckpt.pt --books $B --tau $3 $4 \
           --out runs_qwen/$2.book_demo_tau$3$5.json > runs_qwen/$2.book_demo_tau$3$5.log 2>&1; }
ev 0 q4b_s3 comp 0 0.99 & ev 0 q4b_s3 comp 0 0.995 &
ev 1 q4b_s3 comp 1 0.99 & ev 1 q4b_s3 comp 1 0.995 &
ev 2 q4b_s3 comp_heldout 0 0.99 & ev 2 q4b_s3 heldout 0 0.99 &
ev 3 q4b_s3 multi 0 0.99 & ev 3 q4b_s3 comp_heldout 1 0.99 &
ev 4 q4b_s2 comp 0 0.97 & ev 4 q4b_s2 comp 1 0.97 &
ev 5 q4b_s2 comp_heldout 0 0.97 & ev 5 q4b_s1 comp 0 0.9 &
(CUDA_VISIBLE_DEVICES=5 $PY eval_lm.py --run runs_qwen/q4b_s3 --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_qwen/q4b_s3.ppl.log 2>&1;
 CUDA_VISIBLE_DEVICES=5 $PY eval_lm.py --run runs_qwen/q4b_s3 --data $DATA --lens 2048 --n 4 --K 1 --tasks facts --no_memory --tag _nomem \
   > runs_qwen/q4b_s3.ppl_nomem.log 2>&1) &
ev 6 q4b_s3 comp 0 0.99 --no_memory _nomem &
(demo 6 q4b_s3 0.99; demo 6 q4b_s3 0.995) &
(demo 7 q4b_s2 0.97; demo 7 q4b_s3 0.99 --no_memory _nomem) &
wait
~/hais_poc/env/bin/python ~/harvest.py runs_qwen/q4b_s1 runs_qwen/q4b_s2 runs_qwen/q4b_s3 > ~/harvest_q4b_s3.log 2>&1
echo DONE > q4b_s3.done
