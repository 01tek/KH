#!/bin/bash
# KH-1-Q day on the 8xB200 node: Qwen env/weights -> data -> stage 1 (tau 0.5) -> stage 2 (tau 0.9 + text-write L1)
# -> evals at tau 0.97 -> harvest. Late decisions go in qwen_overrides.env (S1_TOKENS, S2_TAU, S2_TW, S2_TOKENS, BS, ...).
cd ~/hais_poc/poc_m3/lm
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
[ -x ../../env_qwen/bin/python ] || bash ../node_qwen_env.sh > ~/qwen_env.log 2>&1
if [ ! -f ~/data/fwe_qwen35/meta.json ]; then
  ../../env_qwen/bin/python prep_data.py --out ~/data/fwe_qwen35 --shards 3 --train_tokens 2000000000 --val_tokens 20000000 \
    --workers 64 --tokenizer Qwen/Qwen3.5-4B > ~/prep_qwen.log 2>&1 || { echo FAILED_PREP > q4b.failed; exit 1; }
fi
[ -f qwen_overrides.env ] && source qwen_overrides.env
if ! grep -q '"event": "done"' runs_qwen/q4b_s1.train.log 2>/dev/null; then
  NAME=q4b_s1 TAU=0.5 TW=${S1_TW:-0.05} TOKENS=${S1_TOKENS:-1.5e9} NO_EVAL=1 bash launch_qwen.sh || { echo FAILED_S1 > q4b.failed; exit 1; }
fi
[ -f qwen_overrides.env ] && source qwen_overrides.env
NAME=${S2_NAME:-q4b_s2} INIT=runs_qwen/q4b_s1/ckpt.pt TAU=${S2_TAU:-0.9} TW=${S2_TW:-0.2} TOKENS=${S2_TOKENS:-1e9} \
  bash launch_qwen.sh || { echo FAILED_S2 > q4b.failed; exit 1; }
~/hais_poc/env/bin/python ~/harvest.py runs_qwen/q4b_s1 runs_qwen/${S2_NAME:-q4b_s2} > ~/harvest_q4b.log 2>&1
echo DONE > q4b.done
