#!/bin/bash
# when the 494M v4 run (training + its evals) is fully done: harvest it, then launch KH-1-1.2B on all 8 GPUs
cd $HOME/hais_poc/poc_m3/lm
until grep -q NODE_LM_DONE node_phase2_v4.log 2>/dev/null; do sleep 20; done
~/hais_poc/env/bin/python ~/harvest.py runs_node/gdnhais_d1536_v4 runs_node/gdnhais_d1536_v1 runs_s5/s5_hl256_far_s1 runs_s5/s5_hl512_far_s1 runs_sweep2/s_tau_hl8k_twl_s1 > ~/harvest_v4.log 2>&1
BASE=LiquidAI/LFM2-1.2B NAME=kh1_1p2b HEADS=32 TOKENS=3e9 bash launch_kh1.sh > kh1.log 2>&1
