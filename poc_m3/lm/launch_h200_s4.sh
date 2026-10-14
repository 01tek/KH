#!/usr/bin/env bash
# H200 companions to node sweep 4: low fact load AND training queries capped at 32 segments (learnable). Usage: A|B
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_h4
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.25 --drill_k 8 --max_age 32 --age_warmup 5000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --gdn_half_life 8192"
run() { local n=$1 tpl=$2; shift 2
  $PY train_lm.py $BASE --templates $tpl "$@" --out runs_h4/$n > runs_h4/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_h4/failures.txt; return; }
  for e in $tpl heldout $( [ $tpl = comp ] && echo comp_heldout ); do
    $PY eval_lm.py --run runs_h4/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates $e --arch_tau 0.9 --tag _${e}_tau0.9 > runs_h4/$n.$e.tau0.9.log 2>&1
  done
  $PY eval_lm.py --run runs_h4/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_h4/$n.ppl.log 2>&1; }
if [ "$1" = A ]; then
  run h4_comp_tau09      comp  --seed 0 --arch_tau 0.9 --text_write_l1 0.05 &
  run h4_comp_tau09_s1   comp  --seed 1 --arch_tau 0.9 --text_write_l1 0.05 &
else
  run h4_tau09_twl02     multi --seed 0 --arch_tau 0.9 --text_write_l1 0.02 &
  run h4_notau_twl05     multi --seed 0 --text_write_l1 0.05 &
fi
wait; echo H4_DONE
