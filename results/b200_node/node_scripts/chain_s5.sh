#!/bin/bash
# start sweep 5 on GPUs 0-6 as soon as sweep 4's seven 117M runs have finished their evals (the 250M keeps GPU 7)
cd $HOME/hais_poc/poc_m3/lm
until [ "$(ls runs_s4/*.ppl.log 2>/dev/null | grep -v 250m | xargs -r grep -l ppl_val 2>/dev/null | wc -l)" -ge 7 ]; do sleep 30; done
bash launch_node_sweep5.sh > sweep5.log 2>&1
