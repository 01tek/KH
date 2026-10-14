#!/usr/bin/env bash
# Pilot 10: extra capacity seeds (mem_heads 8, 16) for the 1M-token capacity curve.
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
PY=../env/bin/python; OUT=runs/p9; J=$OUT/jobs_p10.txt
grep -E "^p9_cap_h(8|16)_s0 " $OUT/jobs.txt | sed -e 's/_s0 /_sX /' > /tmp/p10tmpl_$$
: > $J; for s in 2 3; do sed -e "s/_sX /_s$s /" -e "s/--seed 0/--seed $s/" /tmp/p10tmpl_$$ >> $J; done
run_one() { local n=$1; shift; [ -f $OUT/$n/summary.json ] && return 0; $PY train.py "$@" --out $OUT/$n > $OUT/$n.log 2>&1 || echo "FAILED $n" >> $OUT/failures.txt; }
export -f run_one; export PY OUT
xargs -P 4 -L 1 bash -c 'run_one "$@"' _ < $J
echo P10_DONE
