#!/usr/bin/env bash
# H200 companions to node sweep 5 (drills that require archival memory: GDN half-life 512 + far drills). Usage: A|B
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_h5
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --arch_tau 0.5 --text_write_l1 0.05 --gdn_half_life 512 --drill_gap far"
run() { local n=$1 tpl=$2; shift 2
  $PY train_lm.py $BASE --templates $tpl "$@" --out runs_h5/$n > runs_h5/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_h5/failures.txt; return; }
  for e in $tpl heldout $( [ $tpl = comp ] && echo comp_heldout ); do
    $PY eval_lm.py --run runs_h5/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates $e --arch_tau 0.9 --tag _${e}_tau0.9 > runs_h5/$n.$e.tau0.9.log 2>&1
  done
  $PY eval_lm.py --run runs_h5/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_h5/$n.ppl.log 2>&1; }
if [ "$1" = A ]; then
  run h5_hl512_far_s2 multi --seed 2 &
  run h5_hl512_far_s3 multi --seed 3 &
else
  run h5_comp_hl512_far    comp --seed 0 &
  run h5_comp_hl512_far_s1 comp --seed 1 &
fi
wait; echo H5_DONE
