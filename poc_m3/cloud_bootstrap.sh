#!/usr/bin/env bash
# One-shot cloud launcher: verify kernels on this GPU, smoke-test, then run the parallel grid.
# Launch detached:  cd ~/hais_poc/poc_m3 && nohup ./cloud_bootstrap.sh > bootstrap.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=env/bin/python
echo "[$(date)] kernel tests"
# Official Mamba-3 SISO tests @ e9594ce, incl. the Blackwell num_stages regression test.
timeout 3600 $PY -m pytest -x -q test_mamba3_siso.py > poc_m3/kernel_test.log 2>&1 \
  || { echo "KERNEL_TEST_FAILED (see poc_m3/kernel_test.log)"; exit 1; }
cd poc_m3
echo "[$(date)] smoke"
$PY smoke.py > smoke.log 2>&1 || { echo "SMOKE_FAILED"; exit 1; }
echo "[$(date)] grid PAR=${PAR:-8} STEPS=${STEPS:-8000}"
PY=../env/bin/python ./cloud_grid.sh > grid.out 2>&1
echo "[$(date)] BOOTSTRAP_DONE"
