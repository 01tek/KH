#!/usr/bin/env bash
# HAIS-X (episodic store + archival delta memory, one write gate): correctness test, then a pilot run.
# Write penalty on all heads pushes the shared gate to be selective; the store keeps beta>tau entries
# verbatim (FIFO cap 16384 per head), so retention of facts doesn't depend on surviving distractor writes.
cd "$(dirname "$0")"
while ps -eo cmd | grep -q "[q]ueue_l1.sh\|[q]ueue_sp95.sh\|[q]ueue_verify.sh"; do sleep 30; done
PY=../m3poc_env/bin/python
$PY test_hx_stream.py > runs/pilot1/test_hx_stream.log 2>&1 || { echo HX_TEST_FAILED; exit 1; }
$PY train.py --layout hais,hx --state_passing 0.95 --write_l1 0.05 --l1_scope all \
  --L 256 --kmax 32 --bs 128 --lr 3e-3 --steps 12000 --seed 0 --chunk 1024 \
  --eval_lens 256,4096,65536,1048576 --eval_K 1,8,32 --eval_modes early,uniform \
  --out runs/pilot1/haishx_l1 > runs/pilot1/haishx_l1.log 2>&1
echo HX_DONE
