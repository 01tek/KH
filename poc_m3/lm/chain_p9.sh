#!/bin/bash
# Pilot 9 = pilot 8 recipe (tau 0.97 + text-write BCE 0.1, no L1) + half the facts about real-looking names, so the
# write gate cannot key on made-up names (probe_text_writes.py: pilot 4 writes rare proper nouns and numbers in text).
cd ~/hais_poc/poc_m3/lm
until grep -q '"event": "done"' runs_lfm2/lfm2_350m_pilot7.train.log 2>/dev/null && grep -q '"event": "done"' runs_lfm2/lfm2_350m_pilot8.train.log 2>/dev/null; do sleep 20; done
INIT=runs_lfm2/lfm2_350m_pilot4/ckpt.pt N=lfm2_350m_pilot9 TAU=0.97 TW=0 EXTRA="--text_write_bce 0.1 --natural_names 0.5" \
  EV_TAUS="0.97 0.99" bash launch_lfm2_ft2.sh > pilot9.log 2>&1
echo P9_DONE > p9.done
