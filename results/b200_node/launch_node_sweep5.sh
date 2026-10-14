#!/usr/bin/env bash
# Node sweep 5: make the drills (the only place writes get gradient) require ARCHIVAL memory, so the model cannot route
# them through the GDN layers (which never extrapolate). Short GDN half-life (256-1024) + "far" drills (facts in the
# first 15% of a segment, queries in the last 15%: gap >= 712 tokens). Data/LR/levers = sweep 2's successful recipe.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_s5
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --arch_tau 0.5 --text_write_l1 0.05"
run() { local gpu=$1 n=$2; shift 2
  ( export CUDA_VISIBLE_DEVICES=$gpu
    $PY train_lm.py $BASE "$@" --out runs_s5/$n > runs_s5/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_s5/failures.txt; exit 1; }
    for t in "" 0.9; do
      $PY eval_lm.py --run runs_s5/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
        --templates multi ${t:+--arch_tau $t} --tag _multi_tau${t:-train} > runs_s5/$n.multi.tau${t:-train}.log 2>&1
    done
    $PY eval_lm.py --run runs_s5/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_s5/$n.ppl.log 2>&1 ) &
}
run 0 s5_ctrl_s5           --seed 5 --gdn_half_life 8192
run 1 s5_hl512_far         --seed 0 --gdn_half_life 512 --drill_gap far
run 2 s5_hl512_far_s1      --seed 1 --gdn_half_life 512 --drill_gap far
run 3 s5_hl256_far         --seed 0 --gdn_half_life 256 --drill_gap far
run 4 s5_hl256_far_s1      --seed 1 --gdn_half_life 256 --drill_gap far
run 5 s5_hl1024_far        --seed 0 --gdn_half_life 1024 --drill_gap far
run 6 s5_hl8k_far          --seed 0 --gdn_half_life 8192 --drill_gap far
wait; echo SWEEP5_DONE
