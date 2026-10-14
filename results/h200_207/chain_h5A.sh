#!/bin/bash
# start the H200 sweep-5 companions (A) once this box's h4 evals are done
cd $HOME/hais_poc/poc_m3/lm
until ! ps -eo args | grep -qE "[t]rain_lm|[e]val_lm"; do sleep 30; done
bash launch_h200_s5.sh A > h5_A.log 2>&1
