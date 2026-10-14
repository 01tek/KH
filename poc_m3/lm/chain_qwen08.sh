#!/bin/bash
# Qwen3.5-0.8B pilot on H200 #1: stage 1 (tau 0.5) -> stage 2 (tau 0.9 + text-write penalty) -> evals at tau 0.97.
# Stage-2 settings come from qwen_overrides.env (read when stage 2 starts), so they can change while this runs.
cd ~/hais_poc/poc_m3/lm
[ -f runs_qwen/qwen08_s1/ckpt.pt ] && grep -q '"event": "done"' runs_qwen/qwen08_s1.train.log 2>/dev/null || \
  STAGE=1 N=qwen08_s1 NO_EVAL=1 bash launch_qwen_pilot.sh || { echo FAILED > qwen08.failed; exit 1; }
[ -f qwen_overrides.env ] && source qwen_overrides.env
STAGE=2 INIT=runs_qwen/qwen08_s1/ckpt.pt N=${S2_NAME:-qwen08_s2} TOKENS=${S2_TOKENS:-1e8} TW=${S2_TW:-0.2} \
  bash launch_qwen_pilot.sh || { echo FAILED > qwen08.failed; exit 1; }
# stage 3: the pilot-6 step (train tau 0.97), evals at tau 0.99
STAGE=2 TAU=0.97 TW=0.2 ETAU=0.99 INIT=runs_qwen/${S2_NAME:-qwen08_s2}/ckpt.pt N=qwen08_s3 TOKENS=1e8 \
  bash launch_qwen_pilot.sh || { echo FAILED > qwen08.failed; exit 1; }
echo DONE > qwen08.done
