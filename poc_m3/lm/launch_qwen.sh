#!/usr/bin/env bash
# KH-1-Q: Qwen3.5 + archival memory on all GPUs (DDP). Same recipe as KH-1 (LFM2): Qwen frozen; its 8 full-attention
# layers and 24 Gated-DeltaNet layers see one chunk at a time (no KV cache, GDN state reset per chunk = constant memory);
# ReZero-gated archival adapters carry everything older. Segments = 2 chunks (cross-chunk drills route through the memory).
#   stage 1: NAME=q4b_s1 TAU=0.5 TW=0.05 TOKENS=1.5e9 bash launch_qwen.sh
#   stage 2: NAME=q4b_s2 INIT=runs_qwen/q4b_s1/ckpt.pt TAU=0.9 TW=0.6 TOKENS=1e9 bash launch_qwen.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
[ -f qwen_overrides.env ] && source qwen_overrides.env  # late decisions without editing a running chain
NG=${NG:-$(nvidia-smi -L | wc -l)}
PY=../../env_qwen/bin/python; DATA=${DATA:-$HOME/data/fwe_qwen35}; mkdir -p runs_qwen
N=${NAME:-q4b_s1}; CH=${CHUNK:-2048}; TAU=${TAU:-0.5}; ETAU=${ETAU:-$([ "$TAU" = 0.5 ] && echo 0.9 || echo 0.97)}
../../env_qwen/bin/torchrun --standalone --nproc_per_node=$NG train_lm.py --data $DATA ${INIT:+--init_from $INIT} \
  --lfm2 ${BASE:-Qwen/Qwen3.5-4B} --lfm2_insert ${INSERT:-15,23,31} --lfm2_chunk $CH --mem_heads ${HEADS:-40} \
  --bs ${BS:-4} --T $((2 * CH)) --tokens ${TOKENS:-1.5e9} --lr ${LR:-3e-4} --wd 0.0 --seed 0 --lm_loss_frac 0.1 \
  --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau $TAU --text_write_l1 ${TW:-0.05} --ckpt_every 250 --out runs_qwen/$N > runs_qwen/$N.train.log 2>&1 || { echo "FAILED $N"; exit 1; }
[ -n "$NO_EVAL" ] && exit 0
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_qwen/$N --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
         --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 --arch_tau $ETAU --tag _$2_chat$3_tau$ETAU \
         > runs_qwen/$N.$2.chat$3.tau$ETAU.log 2>&1; }
ev 0 comp 0 & ev 1 comp 1 & ev 2 comp_heldout 0 & ev 3 heldout 0 & ev 4 multi 0 & ev 5 comp_heldout 1 &
CUDA_VISIBLE_DEVICES=6 $PY eval_lm.py --run runs_qwen/$N --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_qwen/$N.ppl.log 2>&1 &
wait; echo QWEN_DONE $N
