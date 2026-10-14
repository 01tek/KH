#!/usr/bin/env bash
# Node sweep 3: make long-range training queries answerable so archival recall gets a learning signal.
# Diagnostic sweep: with the dense fact stream (~580 live facts) training accuracy is ~0.03 at every age (no signal);
# with low load (~55 live facts) it is 1.00 / 0.82 up to 32 segments, then ~0. The spike guard backfired (state resets).
# Here: low load everywhere, no spike guard, and archival writes thresholded at 0.9 already in training.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_s3
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --gdn_half_life 8192 --text_write_l1 0.05"
L16=$(python3 -c "print(','.join(['gdn','hais'] * 8))")
run() { local gpu=$1 n=$2; shift 2
  ( export CUDA_VISIBLE_DEVICES=$gpu
    $PY train_lm.py $BASE "$@" --out runs_s3/$n > runs_s3/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_s3/failures.txt; exit 1; }
    for t in "" 0.9; do
      $PY eval_lm.py --run runs_s3/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
        --templates multi ${t:+--arch_tau $t} --tag _multi_tau${t:-train} > runs_s3/$n.multi.tau${t:-train}.log 2>&1
    done
    $PY eval_lm.py --run runs_s3/$n --data $DATA --lens 16384,131072 --n 20 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates heldout --tag _heldout > runs_s3/$n.heldout.log 2>&1
    $PY eval_lm.py --run runs_s3/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_s3/$n.ppl.log 2>&1 ) &
}
run 0 s3_tau05         --seed 0 --arch_tau 0.5
run 1 s3_tau09         --seed 0 --arch_tau 0.9
run 2 s3_tau09_s1      --seed 1 --arch_tau 0.9
run 3 s3_tau09_twl01   --seed 0 --arch_tau 0.9 --text_write_l1 0.1
run 4 s3_tau09_hl2k    --seed 0 --arch_tau 0.9 --gdn_half_life 2048
run 5 s3_tau09_vlow    --seed 0 --arch_tau 0.9 --fact_rate 0.05 --query_rate 0.03
run 6 s3_tau09_xl      --seed 0 --arch_tau 0.9 --templates xl
run 7 s3_tau09_250m    --seed 0 --arch_tau 0.9 --d_model 1024 --mem_heads 16 --archival_heads 8 --layout $L16 --lr 1e-3
wait; echo SWEEP3_DONE
