#!/usr/bin/env bash
# KH-2 on LFM2.5-8B-A1B: redo stage 3 (tau 0.97, text-write L1 0.2, 100M tokens) from the stage-2 checkpoint with one
# KH-2 change, then evaluate like the standard stage 3 (0.99; tier: 0.99 / 0.998). Usage: VARIANT=quiet|tier bash kh2_8b.sh
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
while [ ! -f runs_kh2/kh2.done ] && [ -z "$NOWAIT" ]; do sleep 60; done
PY=../../env_qwen/bin/python; D=$HOME/data/fwe_lfm25; V=${VARIANT:?quiet or tier}; P=runs_lfm25/l25_8b_s3_kh2$V
case $V in quiet) T="--quiet_frac 0.5 --quiet_mult 5"; E="--arch_tau 0.99";; tier) T="--arch_tau_hi 0.995 --hi_frac 0.5"; E="--arch_tau 0.99 --arch_tau_hi 0.998 --hi_frac 0.5";; esac
COMMON="--lfm2 LiquidAI/LFM2.5-8B-A1B --lfm2_insert 11,17,23 --lfm2_chunk 2048 --mem_heads 32 --bs 8 --T 4096 --lr 3e-4
  --wd 0.0 --seed 0 --lm_loss_frac 0.1 --fact_rate 0.4 --query_rate 0.2 --drill_frac 0.5 --drill_k 8 --max_age 64 --age_warmup 2000
  --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 250"
grep -q '"event": "done"' $P.train.log 2>/dev/null || $PY train_lm.py --data $D $COMMON --init_from runs_lfm25/l25_8b_s2/ckpt.pt \
  --arch_tau 0.97 --text_write_l1 0.2 --tokens 1e8 $T --out $P > $P.train.log 2>&1 || { echo "FAILED $V" >> $P.progress; exit 1; }
ev() { $PY eval_lm.py --run $P --data $D --lens $1 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates comp --chat_frac 1 \
         --eval_seed $2 $E --tag _comp_chat1_kh2$V$3 > $P.eval$3.log 2>&1; }
ev 16384,131072,1048576 1234 _dev & ev 1048576 4321 _fresh4321 & wait
echo "DONE $(date -u +%H:%M)" >> $P.progress
