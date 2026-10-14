#!/bin/bash
# after KH-1 v1 (train tau 0.5) and its evals: fine-tune it with the write threshold at 0.9 (writes must be decisive)
cd $HOME/hais_poc/poc_m3/lm
until grep -q KH1_DONE kh1.log 2>/dev/null; do sleep 20; done
BASE=LiquidAI/LFM2-1.2B NAME=kh1_1p2b_t09 HEADS=32 TOKENS=1e9 LR=2e-4 TAU=0.9 INIT=runs_kh1/kh1_1p2b/ckpt.pt bash launch_kh1.sh > kh1_v2.log 2>&1
