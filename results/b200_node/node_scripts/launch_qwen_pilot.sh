#!/usr/bin/env bash
# Qwen3.5 + archival memory pilot (H200 #1), same recipe as KH-1 (LFM2): Qwen frozen, its attention + Gated-DeltaNet
# layers chunk-local (no KV cache / GDN state across chunks), archival adapters are the only cross-chunk memory.
# Stage 1: tau 0.5 (a tau-0.9 start is dead); stage 2: tau 0.9 + text-write penalty via --init_from. Evals at tau 0.97.
#   STAGE=1 N=qwen08_s1 bash launch_qwen_pilot.sh ;  STAGE=2 INIT=runs_qwen/qwen08_s1/ckpt.pt N=qwen08_s2 bash launch_qwen_pilot.sh
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; DATA=$HOME/data/fwe_qwen35; mkdir -p runs_qwen
BASE=${BASE:-Qwen/Qwen3.5-0.8B}; N=${N:-qwen08_s1}; STAGE=${STAGE:-1}
if [ "$STAGE" = 1 ]; then TAU=${TAU:-0.5}; TW=${TW:-0.05}; else TAU=${TAU:-0.9}; TW=${TW:-0.2}; fi
$PY train_lm.py --data $DATA ${INIT:+--init_from $INIT} --lfm2 $BASE --lfm2_insert ${INSERT:-11,17,23} --lfm2_chunk ${CHUNK:-1024} \
  --mem_heads ${HEADS:-16} --bs ${BS:-32} --T ${T:-2048} --tokens ${TOKENS:-3e8} --lr ${LR:-3e-4} --wd 0.0 --seed 0 --lm_loss_frac 0.1 \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau $TAU --text_write_l1 $TW --ckpt_every 500 --out runs_qwen/$N > runs_qwen/$N.train.log 2>&1 || { echo "FAILED $N"; exit 1; }
[ -n "$NO_EVAL" ] && exit 0
ETAU=$([ "$STAGE" = 1 ] && echo 0.9 || echo 0.97)
for spec in "comp 0" "comp 1" "comp_heldout 0" "heldout 0" "multi 0"; do
  set -- $spec
  $PY eval_lm.py --run runs_qwen/$N --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
    --templates $1 --chat_frac $2 --arch_tau $ETAU --tag _$1_chat$2_tau$ETAU > runs_qwen/$N.$1.chat$2.tau$ETAU.log 2>&1 &
done
$PY eval_lm.py --run runs_qwen/$N --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_qwen/$N.ppl.log 2>&1 &
wait
echo QWEN_PILOT_DONE $N
