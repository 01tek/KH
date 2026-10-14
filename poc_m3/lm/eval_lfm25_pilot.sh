#!/bin/bash
# Evaluate the LFM2.5-8B-A1B stage-1 graft pilot (trained at tau 0.5, run at 0.9) once pilot_lfm25.sh finishes.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log TOKENIZERS_PARALLELISM=false
while [ ! -f runs_lfm25/pilot.done ]; do sleep 30; done
PY=../../env_qwen/bin/python; R=runs_lfm25/lfm25_8b_s1_pilot; D=$HOME/data/fwe_lfm25
E="--n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 --arch_tau 0.9"
$PY eval_lm.py --run $R --data $D --lens 16384,131072 $E --tag _comp_chat1_tau0.9 > $R.eval_16k_131k.log 2>&1
$PY eval_lm.py --run $R --data $D --lens 16384 $E --no_memory --tag _comp_chat1_nomem > $R.eval_nomem.log 2>&1
$PY eval_lm.py --run $R --data $D --lens 1048576 $E --tag _comp_chat1_tau0.9_1M > $R.eval_1m.log 2>&1
echo DONE > runs_lfm25/eval.done
