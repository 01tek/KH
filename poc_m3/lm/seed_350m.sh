#!/usr/bin/env bash
# Independent re-run of the KH-1-350M recipe with another seed (paper: "the result is not luck").
# Exactly the pilot 2 -> 3 -> 4 -> 6 lineage, only --seed changes (adapter init, data order, fact stream):
#   S1 tau 0.5 / text-write L1 0.05 / 300M tokens -> S2 tau 0.9 / 0.05 / 100M -> S3 tau 0.9 / 0.2 / 100M -> S4 tau 0.97 / 0.2 / 100M
# Evals: S4 at inference tau 0.99 (FineWeb-Edu comp chat0/chat1, unseen frames x attributes, book haystack), S3 at 0.97.
#   SEED=1 bash seed_350m.sh
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; DATA=$HOME/data/fwe_lfm2; S=${SEED:?set SEED}; P=runs_lfm2/kh1_350m_seed$S; mkdir -p runs_lfm2
COMMON="--data $DATA --lfm2 LiquidAI/LFM2-350M --lfm2_insert 7,11,15 --lfm2_chunk 1024 --mem_heads 16 --bs 32 --T 2048 --lr 3e-4
  --wd 0.0 --seed $S --lm_loss_frac 0.1 --fact_rate 0.1 --query_rate 0.05 --drill_frac 0.5 --drill_k 8 --max_age 64
  --age_warmup 2000 --ans_weight 5 --age_target loguniform_once --max_pending 4096 --templates mix --chat_frac 0.5 --ckpt_every 500"
stage() {  # name init tau tw tokens
  grep -q '"event": "done"' ${P}_$1.train.log 2>/dev/null && return 0
  $PY train_lm.py $COMMON ${2:+--init_from ${P}_$2/ckpt.pt} --arch_tau $3 --text_write_l1 $4 --tokens $5 --out ${P}_$1 \
    > ${P}_$1.train.log 2>&1 || { echo "FAILED ${P}_$1" > ${P}.failed; exit 1; }
}
stage s1 "" 0.5 0.05 3e8
stage s2 s1 0.9 0.05 1e8
stage s3 s2 0.9 0.2 1e8
stage s4 s3 0.97 0.2 1e8
ev() {  # stage templates chat tau [data] [lens] [K] [tag-suffix]
  $PY eval_lm.py --run ${P}_$1 --data ${5:-$DATA} --lens ${6:-16384,131072,1048576} --n 40 --K ${7:-1,8} --tasks facts --no_ppl \
    --batch 8 --templates $2 --chat_frac $3 --arch_tau $4 --tag _$2_chat$3_tau$4$8 > ${P}_$1.$2.chat$3.tau$4$8.log 2>&1
}
ev s4 comp 0 0.99 & ev s4 comp 1 0.99 & ev s4 comp_heldout 0 0.99 & ev s3 comp 1 0.97 &
ev s4 comp 1 0.99 $HOME/data/books_lfm2 1048576 1 _books &
$PY eval_lm.py --run ${P}_s4 --data $DATA --lens 2048 --n 4 --K 1 --tasks facts > ${P}_s4.ppl.log 2>&1 &
wait
echo DONE > ${P}.done
