#!/usr/bin/env bash
# Which parts of the winning sweep-2 recipe (train tau 0.5 + GDN half-life cap 8K + text-only write penalty 0.05) matter?
# 117M, sweep-2 data (loguniform_once, 10 phrasings), 1B tokens; two runs per H200. Usage: bash launch_ablate_twl.sh A|B
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_abl
C="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_abl/$n > runs_abl/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_abl/failures.txt; return; }
  for t in "" 0.9; do
    $PY eval_lm.py --run runs_abl/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates multi ${t:+--arch_tau $t} --tag _multi_tau${t:-train} > runs_abl/$n.multi.tau${t:-train}.log 2>&1
  done
  $PY eval_lm.py --run runs_abl/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_abl/$n.ppl.log 2>&1; }
TW="--text_write_l1 0.05"
if [ "$1" = A ]; then
  run a_twl        --seed 0 $TW &
  run a_twl_tau    --seed 0 $TW --arch_tau 0.5 &
else
  run a_twl_hl8k   --seed 0 $TW --gdn_half_life 8192 &
  run a_full_s2    --seed 2 $TW --arch_tau 0.5 --gdn_half_life 8192 &
fi
wait; echo ABL_DONE
