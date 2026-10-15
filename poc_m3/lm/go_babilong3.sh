#!/bin/bash
# BABILong round 3: long samples only (4K samples, 16K-64K), then the same test-split eval.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; O=runs_babilong; FT=$O/kh1_350m_bl3
[ -f $FT/ckpt.pt ] || $PY babilong_train.py --init $O/kh1_350m_bl2/ckpt.pt --tasks qa1,qa2,qa3,qa4,qa5 --lengths 16k,32k,64k \
  --samples 4000 --accum 8 --tau 0.97 --seed 2 --out $FT > $FT.train.log 2>&1 || { echo FAILED > $O/failed3; exit 1; }
E="--tasks qa1,qa2,qa3,qa4,qa5"
for t in 0.97 0.99; do
  $PY babilong_eval.py --ckpt $FT/ckpt.pt --tau $t $E --lengths 0k,4k,16k,64k,128k --n 50 --mode memory --out $O/ft3_memory_tau$t.json > $O/ft3_memory_tau$t.log 2>&1 &
done; wait
$PY babilong_eval.py --ckpt $FT/ckpt.pt --tau 0.97 $E --lengths 1M --n 20 --mode memory --out $O/ft3_memory_tau0.97_1M.json > $O/ft3_1M.log 2>&1
echo DONE > $O/done3
