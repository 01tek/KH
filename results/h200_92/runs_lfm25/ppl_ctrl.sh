#!/bin/bash
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; O=runs_lfm25/ppl_ctrl.txt
r() { echo "$1 $3 $($PY eval_lm.py --run $2 --data $4 --lens 2048 --n 1 --K 1 --tasks facts --ppl_T 2048 $5 --tag _ppl_ctrl_$3 2>&1 | grep ppl_val)" >> $O; }
r l25_8b_s3 runs_lfm25/l25_8b_s3 reset $HOME/data/fwe_lfm25 --ppl_reset
r q4b_s3 runs_confirm/q4b_s3 carried $HOME/data/fwe_qwen35 ""
r q4b_s3 runs_confirm/q4b_s3 reset $HOME/data/fwe_qwen35 --ppl_reset
r q4b_s3 runs_confirm/q4b_s3 off $HOME/data/fwe_qwen35 --no_memory
echo DONE >> $O
