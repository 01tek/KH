#!/bin/bash
# start the phase-1 sweep as soon as the 2B-token data is ready
until [ -f $HOME/data/fwe_2b/meta.json ]; do sleep 30; done
cd $HOME/hais_poc/poc_m3/lm && bash launch_node_sweep.sh > sweep.log 2>&1
