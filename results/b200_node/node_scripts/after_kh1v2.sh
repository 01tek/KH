#!/bin/bash
# after KH-1 v2 training: extra evals at inference tau 0.97 (v2 was trained at tau 0.9) on the GPUs the launcher leaves free,
# then a model-only copy of the adapters for the harvest
cd $HOME/hais_poc/poc_m3/lm
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; n=kh1_1p2b_t09
until grep -q '"done"' runs_kh1/$n/log.jsonl 2>/dev/null; do sleep 20; done
sleep 30
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
         --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 --arch_tau 0.97 --tag _$2_chat$3_tau0.97 \
         > runs_kh1/$n.$2.chat$3.tau0.97.log 2>&1; }
(ev 6 comp 0; ev 6 multi 0) & (ev 7 comp 1; ev 7 comp_heldout 0) &
wait
~/hais_poc/env/bin/python ~/harvest.py runs_kh1/kh1_1p2b runs_kh1/kh1_1p2b_t09 > ~/harvest_kh1.log 2>&1
echo DONE > after_kh1v2.done
