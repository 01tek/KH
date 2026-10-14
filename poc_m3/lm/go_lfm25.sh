#!/bin/bash
# KH-1 on LFM2.5 (8xB200): data -> 8B-A1B -> 2.6B -> 24B-A2B. Each base: stage 1 (tau 0.5, text-write L1 0.05) ->
# stage 2 (tau 0.9, L1 0.2) -> stage 3 (tau 0.97, L1 0.2), then evals: stage 3 at 0.99 and stage 2 at 0.98 (the Qwen lesson),
# development streams (1234) and fresh streams (4321). Late changes go in lfm25_overrides.env (S1_TOKENS, S2_TOKENS, S3_TOKENS, ONLY).
cd ~/hais_poc/poc_m3/lm
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
[ -x ../../env_qwen/bin/python ] || bash ../node_qwen_env.sh > ~/qwen_env.log 2>&1
PY=../../env_qwen/bin/python; D=$HOME/data/fwe_lfm25; mkdir -p runs_lfm25
if [ ! -f $D/meta.json ]; then
  $PY prep_data.py --out $D --shards 3 --train_tokens 2000000000 --val_tokens 20000000 --workers 64 \
    --tokenizer LiquidAI/LFM2.5-8B-A1B > runs_lfm25/prep_b200.log 2>&1 || { echo FAILED_PREP > runs_lfm25/b200.failed; exit 1; }
fi
[ -f lfm25_overrides.env ] && source lfm25_overrides.env
graft() {  # tag base insert per-gpu-batch
  local T=$1 B=$2 I=$3 BS=$4; local C="BASE=$B INSERT=$I HEADS=32 BS=$BS DATA=$D RUNS=runs_lfm25"
  grep -q '"event": "done"' runs_lfm25/${T}_s1.train.log 2>/dev/null || env $C NAME=${T}_s1 TAU=0.5 TW=0.05 TOKENS=${S1_TOKENS:-1e9} NO_EVAL=1 bash launch_qwen.sh || return 1
  grep -q '"event": "done"' runs_lfm25/${T}_s2.train.log 2>/dev/null || env $C NAME=${T}_s2 INIT=runs_lfm25/${T}_s1/ckpt.pt TAU=0.9 TW=0.2 TOKENS=${S2_TOKENS:-4e8} NO_EVAL=1 bash launch_qwen.sh || return 1
  grep -q '"event": "done"' runs_lfm25/${T}_s3.train.log 2>/dev/null || env $C NAME=${T}_s3 INIT=runs_lfm25/${T}_s2/ckpt.pt TAU=0.97 TW=0.2 TOKENS=${S3_TOKENS:-2e8} NO_EVAL=1 bash launch_qwen.sh || return 1
  ev() { CUDA_VISIBLE_DEVICES=$1 $PY eval_lm.py --run runs_lfm25/$2 --data $D --lens $3 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
           --templates $4 --chat_frac $5 --arch_tau $6 --eval_seed $7 --tag _$4_chat$5_tau$6${8} > runs_lfm25/$2.$4.chat$5.tau$6$8.log 2>&1; }
  ev 0 ${T}_s3 16384,131072,1048576 comp 1 0.99 1234 & ev 1 ${T}_s3 16384,131072,1048576 comp 0 0.99 1234 &
  ev 2 ${T}_s3 16384,131072,1048576 comp_heldout 0 0.99 1234 & ev 3 ${T}_s2 16384,131072,1048576 comp 1 0.98 1234 &
  ev 4 ${T}_s3 1048576 comp 1 0.99 4321 _fresh4321 & ev 5 ${T}_s2 1048576 comp 1 0.98 4321 _fresh4321 &
  CUDA_VISIBLE_DEVICES=6 $PY eval_lm.py --run runs_lfm25/${T}_s3 --data $D --lens 16384 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
     --templates comp --chat_frac 1 --no_memory --tag _comp_chat1_nomem > runs_lfm25/${T}_s3.nomem.log 2>&1 &
  CUDA_VISIBLE_DEVICES=7 $PY eval_lm.py --run runs_lfm25/${T}_s3 --data $D --lens 2048 --n 4 --K 1 --tasks facts > runs_lfm25/${T}_s3.ppl.log 2>&1 &
  wait; echo "DONE $T" >> runs_lfm25/b200.progress
}
for job in ${ONLY:-l25_8b l25_2p6b l25_24b}; do
  case $job in
    l25_8b)   graft l25_8b   LiquidAI/LFM2.5-8B-A1B  11,17,23 8 ;;
    l25_2p6b) graft l25_2p6b LiquidAI/LFM2.5-2.6B    14,21,29 8 ;;
    l25_24b)  graft l25_24b  LiquidAI/LFM2.5-24B-A2B 19,29,39 4 ;;
  esac || echo "FAILED $job" >> runs_lfm25/b200.progress
done
echo DONE > runs_lfm25/b200.done
