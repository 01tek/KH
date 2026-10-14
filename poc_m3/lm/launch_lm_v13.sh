#!/usr/bin/env bash
# LM v13: fine-tune the 117M lm8 gdn,hais model (1B tokens) for 300M more tokens on age-targeted long-range queries,
# with and without a hard archival write threshold (train tau 0.5). Tests whether the training-free inference-tau
# gain (131K K=1 0.2 -> 0.9) grows into 1M recall once the model adapts. bs 16 to fit next to LM v12.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../../env/bin/python; DATA=$HOME/data/fwe_1b; mkdir -p runs_lm13
C="--data $DATA --init_from lm8_gdnhais_model.pt --d_model 768 --mem_heads 12 --archival_heads 6 --min_decay 0.35 --v_norm \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais --bs 16 --T 1024 --tokens 3e8 --lr 5e-4 --seed 0 \
  --fact_rate 0.1 --query_rate 1.0 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 0 --ans_weight 5 \
  --age_target loguniform --max_pending 4096"
run() { local n=$1; shift
  $PY train_lm.py $C "$@" --out runs_lm13/$n > runs_lm13/$n.train.log 2>&1 &&
  for t in 0.0 0.5 0.9; do
    $PY eval_lm.py --run runs_lm13/$n --data $DATA --lens 16384,131072,1048576 --n 20 --K 1,8 --tasks facts --no_ppl \
      --arch_tau $t --tag _inftau$t > runs_lm13/$n.inftau$t.log 2>&1 &
  done && $PY eval_lm.py --run runs_lm13/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_lm13/$n.ppl.log 2>&1
  wait || echo "FAILED $n" >> runs_lm13/failures.txt; }
run lm13_ft_age_tau0.5 --arch_tau 0.5 &
run lm13_ft_age &
wait; echo LM13_DONE
