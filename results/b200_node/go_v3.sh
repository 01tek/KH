#!/bin/bash
# 494M v3: v2's low-load data + spike guard + LR 6e-4
cd $HOME/hais_poc/poc_m3/lm
NAME=gdnhais_d1536_v3 D=1536 NB=16 MH=24 TAU=0.5 HL=8192 TW=0.05 TOKENS=8e9 FR=0.2 QR=0.1 DF=0.15 MA=1000 AW=10000 LR=6e-4 \
  EXTRA="--spike_guard 20" bash launch_node.sh > node_phase2_v3.log 2>&1
