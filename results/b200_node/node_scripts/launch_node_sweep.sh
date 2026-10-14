#!/usr/bin/env bash
# Node phase 1: 8 single-GPU 117M runs (d768 x 12 blocks, 1B tokens, age-targeted long-range queries) that decide the
# recipe for 1M-token in-text recall. Facts use 10 phrasings (--templates multi); each run is evaluated on trained and
# held-out phrasings, as trained and with inference tau 0.9. One run per GPU.
#   bash launch_node_sweep.sh            (expects ~/data/fwe_2b from node_setup.sh)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false  # 8 GPU processes share 160 cores with data prep
PY=../../env/bin/python; DATA=${DATA:-$HOME/data/fwe_2b}; mkdir -p runs_sweep2
L6="gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais"
C="--data $DATA --d_model 768 --mem_heads 12 --bs 32 --T 1024 --tokens ${TOKENS:-1e9} --lr 1.5e-3 --v_norm \
  --fact_rate 0.5 --query_rate 0.25 --drill_frac 0.25 --drill_k 8 --max_age 2000 --age_warmup 20000 --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates multi --compile"
H="--layout $L6 --archival_heads 6 --min_decay 0.35"
run() { local gpu=$1 n=$2; shift 2
  ( export CUDA_VISIBLE_DEVICES=$gpu
    $PY train_lm.py $C "$@" --out runs_sweep2/$n > runs_sweep2/$n.train.log 2>&1 || { echo "FAILED $n" >> runs_sweep2/failures.txt; exit 1; }
    for tpl in multi heldout; do for t in "" 0.9; do  # "" = as trained
      $PY eval_lm.py --run runs_sweep2/$n --data $DATA --lens 16384,131072,1048576 --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 \
        --templates $tpl ${t:+--arch_tau $t} --tag _${tpl}_tau${t:-train} > runs_sweep2/$n.$tpl.tau${t:-train}.log 2>&1
    done; done
    $PY eval_lm.py --run runs_sweep2/$n --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_sweep2/$n.ppl.log 2>&1 ) &
}
# factorial over write threshold (tau), capped GDN memory (hl8k), and a text-only archival write penalty (twl),
# with seed replicates of the two candidate recipes and the GDN baseline
T5="--arch_tau 0.5"; HL="--gdn_half_life 8192"; TW="--text_write_l1 0.05"
run 0 s_base             $H --seed 0
run 1 s_tau              $H --seed 0 $T5
run 2 s_hl8k             $H --seed 0 $HL
run 3 s_tau_hl8k         $H --seed 0 $T5 $HL
run 4 s_tau_hl8k_twl     $H --seed 0 $T5 $HL $TW
run 5 s_tau_hl8k_twl_s1  $H --seed 1 $T5 $HL $TW
run 6 s_tau_hl8k_s1      $H --seed 1 $T5 $HL
run 7 s_gdn              --layout ${L6//hais/gdn} --seed 0
wait; echo SWEEP_DONE
