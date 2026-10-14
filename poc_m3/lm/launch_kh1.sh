#!/usr/bin/env bash
# KH-1: LiquidAI LFM2 + archival memory, trained on all GPUs (DDP). LFM2 frozen; its attention/convs see one chunk at a
# time (constant memory); archival adapters (ReZero-gated) carry everything older. Segments = 2 chunks so drill queries
# in chunk 2 can only be answered from facts in chunk 1 via the archival memory (writes get gradient by construction).
# 358 fact phrasings (compositional + sentences), half the queries as chat turns.
#   BASE=LiquidAI/LFM2-1.2B NAME=kh1_1p2b HEADS=32 TOKENS=3e9 bash launch_kh1.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
[ -f kh1_overrides.env ] && source kh1_overrides.env  # late decisions (e.g. TW) without editing a running chain
NG=$(nvidia-smi -L | wc -l)
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_lfm2}; mkdir -p runs_kh1
N=${NAME:-kh1_1p2b}; CH=${CHUNK:-2048}
../../env/bin/torchrun --standalone --nproc_per_node=$NG train_lm.py --data $DATA ${INIT:+--init_from $INIT} --lfm2 ${BASE:-LiquidAI/LFM2-1.2B} \
  --lfm2_insert ${INSERT:-7,11,15} --lfm2_chunk $CH --mem_heads ${HEADS:-32} \
  --bs ${BS:-8} --T $((2 * CH)) --tokens ${TOKENS:-3e9} --lr ${LR:-3e-4} --wd 0.0 --seed 0 --lm_loss_frac 0.1 \
  --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 \
  --arch_tau ${TAU:-0.5} --text_write_l1 ${TW:-0.05} --ckpt_every 500 --out runs_kh1/$N > runs_kh1/$N.train.log 2>&1 || { echo "FAILED $N"; exit 1; }
ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_kh1/$N --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 \
         --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 --arch_tau 0.9 --tag _$2_chat$3 > runs_kh1/$N.$2.chat$3.log 2>&1; }
ev 0 comp 0 & ev 1 comp 1 & ev 2 comp_heldout 0 & ev 3 heldout 0 & ev 4 multi 0 &
CUDA_VISIBLE_DEVICES=5 $PY eval_lm.py --run runs_kh1/$N --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_kh1/$N.ppl.log 2>&1 &
wait; echo KH1_DONE
