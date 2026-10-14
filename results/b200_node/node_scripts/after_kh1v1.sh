#!/bin/bash
# KH-1 v1 training keeps running (torchrun); this script replaces the edited launcher's tail:
# wait for v1's "done", evaluate v1, then fine-tune v2 (write threshold 0.9) and evaluate it.
cd $HOME/hais_poc/poc_m3/lm
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2
until grep -q '"done"' runs_kh1/kh1_1p2b/log.jsonl 2>/dev/null; do sleep 20; done
sleep 30  # let rank 0 finish writing the final checkpoint
ev() { local n=$1 gpu=$2 tpl=$3 chat=$4 tau=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
    --tasks facts --no_ppl --batch 8 --templates $tpl --chat_frac $chat --arch_tau $tau --tag _${tpl}_chat${chat}_tau$tau \
    > runs_kh1/$n.$tpl.chat$chat.tau$tau.log 2>&1; }
n=kh1_1p2b
ev $n 0 comp 0 0.9 & ev $n 1 comp 1 0.9 & ev $n 2 comp 0 0.5 & ev $n 3 comp_heldout 0 0.5 &
ev $n 4 heldout 0 0.5 & ev $n 5 multi 0 0.9 &
CUDA_VISIBLE_DEVICES=6 $PY eval_lm.py --run runs_kh1/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_kh1/$n.ppl.log 2>&1 &
wait
echo KH1_V1_EVAL_DONE > kh1_v1_eval.done
BASE=LiquidAI/LFM2-1.2B NAME=kh1_1p2b_t09 HEADS=32 TOKENS=1e9 LR=2e-4 TAU=0.9 INIT=runs_kh1/kh1_1p2b/ckpt.pt bash launch_kh1.sh > kh1_v2.log 2>&1
