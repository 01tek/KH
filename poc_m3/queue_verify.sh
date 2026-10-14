#!/usr/bin/env bash
# Run the Mamba-1 delta-needle control alone on the GPU, after the queued pilot arm finishes.
cd "$(dirname "$0")/.."
while ps -eo cmd | grep -q "[q]ueue_haishais.sh\|[r]eeval.sh"; do sleep 30; done
TRIALS=${TRIALS:-30} venv_torch27/bin/python poc_m3/verify_delta_needle.py > poc_m3/verify_delta_needle.log 2>&1
echo VERIFY_DONE
