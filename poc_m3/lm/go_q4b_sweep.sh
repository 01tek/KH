#!/bin/bash
# KH-1-Q inference-threshold sweep (stage 3 at train tau 0.97 collapsed at 1M; stages 1-2 are the good ones):
# 1M recall (chat questions, n 40) and the book demo for stage 1 / stage 2 at more thresholds, then harvest.
cd ~/hais_poc/poc_m3/lm
until [ -f kh1v4.done ] || [ -f kh1v4.failed ]; do sleep 30; done
export TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; DATA=$HOME/data/fwe_qwen35; B="../../demo/books/pg2600.txt ../../demo/books/pg2701.txt ../../demo/books/pg1342.txt"
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_qwen/$2 --data $DATA --lens 1048576 --n 40 --K 1,8 --tasks facts \
         --no_ppl --batch 8 --templates $3 --chat_frac 1 --arch_tau $4 --tag _$3_chat1_tau$4 > runs_qwen/$2.$3.chat1.tau$4.log 2>&1; }
demo() { CUDA_VISIBLE_DEVICES=$1 $PY ../kh1_demo.py --ckpt runs_qwen/$2/ckpt.pt --books $B --tau $3 \
           --out runs_qwen/$2.book_demo_tau$3.json > runs_qwen/$2.book_demo_tau$3.log 2>&1; }
ev 0 q4b_s1 comp 0.9 & ev 1 q4b_s1 comp 0.95 & ev 2 q4b_s1 comp 0.97 & ev 3 q4b_s2 comp 0.98 & ev 4 q4b_s2 comp 0.99 &
ev 5 q4b_s1 comp_heldout 0.95 & (demo 6 q4b_s1 0.9; demo 6 q4b_s1 0.95) & (demo 7 q4b_s2 0.98; demo 7 q4b_s1 0.97) &
wait
~/hais_poc/env/bin/python ~/harvest.py runs_qwen/q4b_s1 runs_qwen/q4b_s2 runs_qwen/q4b_s3 > ~/harvest_q4b_sweep.log 2>&1
echo DONE > q4b_sweep.done
