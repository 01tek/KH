#!/bin/bash
# After pilot 6 trains: pilot 7 (tau 0.99 from pilot 6) and pilot 8 (tau 0.97 + text-write BCE 0.1 from pilot 4), concurrently.
cd ~/hais_poc/poc_m3/lm
until grep -q '"event": "done"' runs_lfm2/lfm2_350m_pilot6.train.log 2>/dev/null; do sleep 15; done
INIT=runs_lfm2/lfm2_350m_pilot6/ckpt.pt N=lfm2_350m_pilot7 TAU=0.99 TW=0.2 EV_TAUS="0.99 0.995" bash launch_lfm2_ft2.sh > pilot7.log 2>&1 &
INIT=runs_lfm2/lfm2_350m_pilot4/ckpt.pt N=lfm2_350m_pilot8 TAU=0.97 TW=0 EXTRA="--text_write_bce 0.1" EV_TAUS="0.97 0.99" bash launch_lfm2_ft2.sh > pilot8.log 2>&1 &
wait
echo P78_DONE > p78.done
