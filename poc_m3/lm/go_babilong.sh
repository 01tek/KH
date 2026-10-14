#!/bin/bash
# BABILong for KH-1-350M: fine-tune the adapters on the training split, then evaluate on the test split against RAG and memory-off.
cd ~/hais_poc/poc_m3/lm
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=../../env/bin/python; O=runs_babilong; mkdir -p $O; INIT=runs_lfm2/lfm2_350m_pilot6/ckpt.pt; FT=$O/kh1_350m_bl
[ -f $FT/ckpt.pt ] || $PY babilong_train.py --init $INIT --tasks qa1,qa2,qa3,qa4,qa5 --lengths 0k,1k,2k,4k,8k,16k --samples 6000 \
  --accum 8 --tau 0.97 --out $FT > $FT.train.log 2>&1 || { echo FAILED_TRAIN > $O/failed; exit 1; }
E="--tasks qa1,qa2,qa3,qa4,qa5"
$PY babilong_eval.py --ckpt $FT/ckpt.pt --tau 0.99 $E --lengths 0k,4k,16k,64k,128k --n 50 --mode memory --out $O/ft_memory_tau0.99.json > $O/ft_memory_tau0.99.log 2>&1 &
$PY babilong_eval.py --ckpt $FT/ckpt.pt --tau 0.97 $E --lengths 0k,4k,16k,64k,128k --n 50 --mode memory --out $O/ft_memory_tau0.97.json > $O/ft_memory_tau0.97.log 2>&1 &
$PY babilong_eval.py --ckpt $INIT --tau 0.99 $E --lengths 0k,4k,16k,64k,128k --n 50 --mode rag --out $O/rag.json > $O/rag.log 2>&1 &
$PY babilong_eval.py --ckpt $INIT --tau 0.99 $E --lengths 0k,4k,16k --n 50 --mode nomem --out $O/nomem.json > $O/nomem.log 2>&1 &
$PY babilong_eval.py --ckpt $INIT --tau 0.99 $E --lengths 0k,4k,16k --n 50 --mode memory --out $O/kh1_zeroshot.json > $O/kh1_zeroshot.log 2>&1 &
wait
$PY babilong_eval.py --ckpt $FT/ckpt.pt --tau 0.99 $E --lengths 1M --n 20 --mode memory --out $O/ft_memory_tau0.99_1M.json > $O/ft_memory_1M.log 2>&1 &
$PY babilong_eval.py --ckpt $INIT --tau 0.99 $E --lengths 1M --n 20 --mode rag --out $O/rag_1M.json > $O/rag_1M.log 2>&1 &
wait
echo DONE > $O/done
