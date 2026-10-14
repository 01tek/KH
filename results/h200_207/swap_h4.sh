#!/bin/bash
# replace the collapsing max_age-2000 H200 runs with the capped (max_age 32) versions
cd $HOME/hais_poc/poc_m3/lm
for p in $(ps -eo pid,args | grep -E "[l]aunch_h200_s3.sh" | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -eo pid,args | grep -E "[t]rain_lm.py|[e]val_lm.py" | awk '{print $1}'); do kill $p; done
sleep 10
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
echo quit | nvidia-cuda-mps-control; sleep 3; nvidia-cuda-mps-control -d; sleep 2   # killed clients can wedge MPS
DATA=${DATA:-$HOME/data/fwe_2b} bash launch_h200_s4.sh $1 > h4_$1.log 2>&1
