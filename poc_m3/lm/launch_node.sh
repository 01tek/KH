#!/usr/bin/env bash
# Node phase 2: the scaled LM on all GPUs (DDP): alternating gdn,hais blocks, age-targeted long-range queries, 10 fact
# phrasings. Size and recipe via env, e.g.  D=1536 NB=16 MH=24 TAU=0.5 HL=8192 TW=0.05 TOKENS=1e10 bash launch_node.sh
# (D=1536, NB=16, MH=24 is ~0.5B params; BASELINE=1 adds the matched gdn-only model afterwards)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false  # 8 GPU processes share 160 cores with data prep
NG=$(nvidia-smi -L | wc -l)
DATA=${DATA:-$HOME/data/fwe}
X=""; [ -n "$TAU" ] && X="$X --arch_tau $TAU"; [ -n "$HL" ] && X="$X --gdn_half_life $HL"; [ -n "$TW" ] && X="$X --text_write_l1 $TW"
D=${D:-1024}; NB=${NB:-16}; MH=${MH:-16}; BS=${BS:-16}; N=${NAME:-gdnhais_d${D}}; X="$X ${EXTRA}"
C="--data $DATA --d_model $D --mem_heads $MH --bs $BS --T 2048 --tokens ${TOKENS:-1e10} --lr ${LR:-8e-4} --seed 0 --v_norm \
  --fact_rate ${FR:-1.0} --query_rate ${QR:-0.5} --drill_frac ${DF:-0.25} --drill_k 8 --max_age ${MA:-1000} --age_warmup ${AW:-20000} --ans_weight 5 \
  --age_target loguniform_once --max_pending 4096 --templates ${TPL:-xl} --ckpt_every 500 --compile"
L=$(python3 -c "print(','.join(['gdn','hais'] * ($NB // 2)))")
mkdir -p runs_node
ev() { for tpl in multi heldout; do ../../env/bin/python eval_lm.py --run runs_node/$1 --data $DATA --lens 16384,131072,1048576 \
         --n 40 --K 1,8 --tasks facts --no_ppl --batch 8 --templates $tpl $2 --tag "_$tpl$3" > runs_node/$1.eval_$tpl$3.log 2>&1; done; }
../../env/bin/torchrun --standalone --nproc_per_node=$NG train_lm.py $C --layout $L --archival_heads $((MH / 2)) --min_decay 0.35 $X \
  --out runs_node/${N} > runs_node/${N}.train.log 2>&1
for t in "" 0.9; do (export CUDA_VISIBLE_DEVICES=$((${#t} % NG)); ev ${N} "${t:+--arch_tau $t}" "_tau${t:-train}") & done
../../env/bin/python eval_lm.py --run runs_node/${N} --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_node/${N}.ppl.log 2>&1
wait
if [ -n "$BASELINE" ]; then
  ../../env/bin/torchrun --standalone --nproc_per_node=$NG train_lm.py $C --layout ${L//hais/gdn} --out runs_node/gdn_d${D} > runs_node/gdn_d${D}.train.log 2>&1
  ev gdn_d${D} "" "" &
  ../../env/bin/python eval_lm.py --run runs_node/gdn_d${D} --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > runs_node/gdn_d${D}.ppl.log 2>&1
  wait
fi
echo NODE_LM_DONE
