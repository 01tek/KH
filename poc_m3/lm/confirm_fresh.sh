#!/bin/bash
# Fresh-stream confirmation of the headline 1M numbers (eval sampler seed 4321 instead of 1234): the inference
# thresholds were picked on the 1234 streams, so re-measure on streams no choice was made on.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log TOKENIZERS_PARALLELISM=false
E="--lens 1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 --eval_seed 4321"
../../env_qwen/bin/python eval_lm.py --run runs_confirm/q4b_s2 --data $HOME/data/fwe_qwen35 $E --arch_tau 0.98 --tag _fresh4321_comp_chat1_tau0.98 > runs_confirm/q4b_s2.fresh.log 2>&1
../../env/bin/python eval_lm.py --run runs_lfm2/lfm2_350m_pilot6 --data $HOME/data/fwe_lfm2 $E --arch_tau 0.99 --tag _fresh4321_comp_chat1_tau0.99 > runs_confirm/pilot6.fresh.log 2>&1
../../env/bin/python eval_lm.py --run runs_confirm/v4 --data $HOME/data/fwe_lfm2 $E --arch_tau 0.99 --tag _fresh4321_comp_chat1_tau0.99 > runs_confirm/v4.fresh.log 2>&1
echo DONE > runs_confirm/fresh.done
