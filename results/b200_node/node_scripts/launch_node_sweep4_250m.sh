#!/usr/bin/env bash
# Node sweep 4: sweep 3 showed long-range training recall (0.67-0.86 at 4-32 segments by step 10K) COLLAPSING to ~0
# at every age once the age warmup reached 2000 segments (step 20K): a flood of unanswerable old queries (ans_weight 5)
# teaches the model to guess. Here training queries stay within what is learnable (max_age 32 / 64 / 128 segments);
# distance beyond that is left to the architecture, as in the synthetic task (trained at 256 tokens, recalls at 1M+).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_s4
BASE="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens 1e9 --lr 1.5e-3 --v_norm \
  --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.25 --drill_k 8 --max_age 32 --age_warmup 5000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --archival_heads 6 --min_decay 0.35 \
  --gdn_half_life 8192 --text_write_l1 0.05"
L16=$(python3 -c "print(','.join(['gdn','hais'] * 8))")
run() { local gpu=$1 n=$2; shift 2
  ( export CUDA_VISIBLE_DEVICES=$gpu
    $PY train_lm.py $BASE "$@" --out runs_s4/$n > runs_s4/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_s4/failures.txt; exit 1; }
    for t in "" 0.9; do
      $PY eval_lm.py --run runs_s4/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
        --templates multi ${t:+--arch_tau $t} --tag _multi_tau${t:-train} > runs_s4/$n.multi.tau${t:-train}.log 2>&1
    done
    $PY eval_lm.py --run runs_s4/$n --data $DATA --lens 16384,131072 --n 20 --K 1,8 --tasks facts --no_ppl --batch 8 \
      --templates heldout --tag _heldout > runs_s4/$n.heldout.log 2>&1
    $PY eval_lm.py --run runs_s4/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_s4/$n.ppl.log 2>&1 ) &
}
run 7 s4_ma32_tau09_250m --seed 0 --arch_tau 0.9 --d_model 1024 --mem_heads 16 --archival_heads 8 --layout $L16 --lr 1e-3
wait; echo SWEEP4_250M_DONE
