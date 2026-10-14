#!/bin/bash
# Resume the Qwen3.5-0.8B pilot chain (paused for GPU memory) once pilots 7/8/9 and their evals are done.
cd ~/hais_poc/poc_m3/lm
until [ -f p78.done ] && [ -f p9.done ]; do sleep 30; done
bash chain_qwen08.sh > chain_qwen08.log 2>&1
