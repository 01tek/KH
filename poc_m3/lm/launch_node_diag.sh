#!/usr/bin/env bash
# Node phase 2b: why does the proven 117M recipe (sweep 2: distance-invariant to 1M) lose recall at 494M with DDP?
# One single-GPU run per factor that differs between the two setups; everything else = the 117M recipe.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_diag
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --arch_tau 0.5 --gdn_half_life 8192 --text_write_l1 0.05 --spike_guard 20"
L16=$(python3 -c "print(','.join(['gdn','hais'] * 8))")
run() { local gpu=$1 n=$2; shift 2
  ( export CUDA_VISIBLE_DEVICES=$gpu
    $PY train_lm.py $BASE "$@" --out runs_diag/$n > runs_diag/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_diag/failures.txt; exit 1; }
    $PY eval_lm.py --run runs_diag/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates multi --arch_tau 0.9 --tag _multi_tau0.9 > runs_diag/$n.multi.tau0.9.log 2>&1
    $PY eval_lm.py --run runs_diag/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_diag/$n.ppl.log 2>&1 ) &
}
run 0 d_base_s3   --seed 3
run 1 d_xl        --seed 0 --templates xl
run 2 d_T2048     --seed 0 --bs 16 --T 2048 --fact_rate 1.0 --query_rate 0.5 --max_age 1000
run 3 d_lowload   --seed 0 --fact_rate 0.1 --query_rate 0.05
run 4 d_drill015  --seed 0 --drill_frac 0.15
run 5 d_250m      --seed 0 --d_model 1024 --mem_heads 16 --archival_heads 8 --layout $L16 --lr 1e-3
run 6 d_494m      --seed 0 --d_model 1536 --mem_heads 24 --archival_heads 12 --layout $L16 --lr 6e-4
run 7 d_lr6e-4    --seed 0 --lr 6e-4
wait; echo DIAG_DONE
