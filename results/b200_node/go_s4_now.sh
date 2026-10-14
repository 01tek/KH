#!/bin/bash
# GPUs 0-6 are idle while sweep 3's last eval runs on GPU 7: start sweep 4 now instead of waiting for the chain
cd $HOME/hais_poc/poc_m3/lm
for p in $(ps -eo pid,args | grep -E "[c]hain_s4.sh" | awk '{print $1}'); do kill $p; done
bash launch_node_sweep4.sh > sweep4.log 2>&1
