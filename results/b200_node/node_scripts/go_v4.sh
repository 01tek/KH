#!/bin/bash
# 494M v4: the far-drill recipe that made 6/6 117M runs distance-invariant (sweep 5), on all 8 GPUs (DDP)
cd $HOME/hais_poc/poc_m3/lm
NAME=gdnhais_d1536_v4 D=1536 NB=16 MH=24 TAU=0.5 HL=256 TW=0.05 TOKENS=6e9 FR=1.0 QR=0.5 DF=0.25 MA=1000 AW=20000 LR=8e-4 \
  TPL=multi EXTRA="--drill_gap far" bash launch_node.sh > node_phase2_v4.log 2>&1
