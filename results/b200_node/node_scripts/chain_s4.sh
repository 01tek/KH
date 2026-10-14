#!/bin/bash
# start sweep 4 the moment sweep 3 has finished (no idle GPUs)
cd $HOME/hais_poc/poc_m3/lm
until grep -q SWEEP3_DONE sweep3.log 2>/dev/null; do sleep 30; done
bash launch_node_sweep4.sh > sweep4.log 2>&1
