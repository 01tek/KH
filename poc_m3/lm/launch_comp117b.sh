#!/usr/bin/env bash
# Generalisation test at 117M, round 2 (lighter text-write penalty, which generalised best in round 1): the winning recipe (tau 0.5 + GDN cap 8K + text-write L1) trained on COMPOSITIONAL facts
# (6 frames x 53 attributes = 318 phrasings), evaluated on unseen frames x unseen attributes (comp_heldout) and on the
# old held-out sentences. Two runs per H200. Usage: bash launch_comp117.sh A|B
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_comp
C="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates comp --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --arch_tau 0.5 --gdn_half_life 8192"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_comp/$n > runs_comp/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_comp/failures.txt; return; }
  for tpl in comp comp_heldout heldout; do
    $PY eval_lm.py --run runs_comp/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates $tpl --arch_tau 0.9 --tag _${tpl}_tau0.9 > runs_comp/$n.$tpl.tau0.9.log 2>&1
  done
  $PY eval_lm.py --run runs_comp/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_comp/$n.ppl.log 2>&1; }
if [ "$1" = A ]; then
  run c_twl02_s1 --seed 1 --text_write_l1 0.02 &
  run c_twl02_s2 --seed 2 --text_write_l1 0.02 &
else
  run c_twl01 --seed 0 --text_write_l1 0.01 &
  run c_twl02_notau --seed 0 --text_write_l1 0.02 --arch_tau 0 &
fi
wait; echo COMP_DONE
