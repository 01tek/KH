#!/bin/bash
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log TOKENIZERS_PARALLELISM=false
R=runs_lfm25/lfm25_8b_s1_pilot
for t in 0.5 0.7; do
  ../../env_qwen/bin/python eval_lm.py --run $R --data $HOME/data/fwe_lfm25 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
    --templates comp --chat_frac 1 --lens 16384 --arch_tau $t --tag _comp_chat1_tau${t}_16k > $R.eval_tau$t.log 2>&1
done
echo DONE > runs_lfm25/lowtau.done
