#!/bin/bash
# stop the 494M v3 run (no long-range recall at step 9.5K) and start the phase-2b diagnostic sweep
cd $HOME/hais_poc/poc_m3/lm
for p in $(ps -eo pid,args | grep -E "[g]o_v3.sh|[l]aunch_node.sh" | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -eo pid,args | grep -E "[t]orchrun|[t]rain_lm.py|[e]val_lm.py" | awk '{print $1}'); do kill $p; done
sleep 15
bash launch_node_diag.sh > diag.log 2>&1
