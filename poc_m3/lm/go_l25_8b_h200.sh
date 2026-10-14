#!/bin/bash
# KH-1 on LFM2.5-8B-A1B, one H200 under MPS. Continues the 150M-token stage-1 pilot (tau 0.5) for 300M more tokens, then
# stage 2 (tau 0.9, text-write L1 0.2, 200M tokens) and stage 3 (tau 0.97, L1 0.2, 100M). Evals: stage 3 at 0.99 and
# stage 2 at 0.98 (the Qwen lesson), on development (1234) and fresh (4321) streams; memory off; perplexity.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env_qwen/bin/python; D=$HOME/data/fwe_lfm25; P=runs_lfm25/l25_8b
DT() { [ -f $HOME/data/fwe_lfm25_2b/meta.json ] && echo $HOME/data/fwe_lfm25_2b || echo $D; }   # bigger train split once prepped
COMMON="--lfm2 LiquidAI/LFM2.5-8B-A1B --lfm2_insert 11,17,23 --lfm2_chunk 2048 --mem_heads 32 --bs 8 --T 4096 --lr 3e-4
  --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000
  --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 250"
stage() {  # name init tau tw tokens
  grep -q '"event": "done"' ${P}_$1.train.log 2>/dev/null && return 0
  echo "start $1 $(date -u +%H:%M) data $(DT)" >> $P.progress
  $PY train_lm.py --data $(DT) $COMMON --init_from $2 --arch_tau $3 --text_write_l1 $4 --tokens $5 --out ${P}_$1 > ${P}_$1.train.log 2>&1 \
    || { echo "FAILED $1 $(date -u +%H:%M)" >> $P.progress; exit 1; }
  echo "done $1 $(date -u +%H:%M)" >> $P.progress
}
ev() {  # stage templates chat tau eval-seed lens tag-suffix [extra flags]
  $PY eval_lm.py --run ${P}_$1 --data $D --lens $6 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates $2 --chat_frac $3 \
    --arch_tau $4 --eval_seed $5 --tag _$2_chat$3_tau$4$7 $8 > ${P}_$1.$2.chat$3.tau$4$7.log 2>&1
}
stage s1b runs_lfm25/lfm25_8b_s1_pilot/ckpt.pt 0.5 0.05 ${S1B:-3e8}
ev s1b comp 1 0.9 1234 16384 _16k &
ev s1b comp 1 0.5 1234 16384 _16k &
stage s2 ${P}_s1b/ckpt.pt 0.9 0.2 ${S2:-2e8}
ev s2 comp 1 0.98 1234 16384,131072,1048576 &
ev s2 comp 1 0.97 1234 1048576 &
stage s3 ${P}_s2/ckpt.pt 0.97 0.2 ${S3:-1e8}
ev s3 comp 1 0.99 1234 16384,131072,1048576 &
ev s3 comp 0 0.99 1234 1048576 &
ev s3 comp_heldout 0 0.99 1234 16384,1048576 &
wait
ev s3 comp 1 0.99 4321 1048576 _fresh4321 &
ev s2 comp 1 0.98 4321 1048576 _fresh4321 &
$PY eval_lm.py --run ${P}_s3 --data $D --lens 16384 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 \
  --no_memory --tag _comp_chat1_nomem > ${P}_s3.nomem.log 2>&1 &
$PY eval_lm.py --run ${P}_s3 --data $D --lens 2048 --n 4 --K 1 --tasks facts > ${P}_s3.ppl.log 2>&1 &
wait
echo "ALL DONE $(date -u +%H:%M)" >> $P.progress; echo DONE > $P.done
