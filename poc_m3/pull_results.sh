#!/usr/bin/env bash
# Pull run logs/summaries (not weights) from a cloud box until N runs matching PATTERN have summary.json.
# Usage: ./pull_results.sh <user>@<h200-node> runs/h200 'p2_*' 14
set -uo pipefail
cd "$(dirname "$0")"
HOST=$1; DIR=$2; PAT=${3:-'*'}; N=${4:-1}; EVERY=${EVERY:-300}
mkdir -p "$DIR"
while true; do
  rsync -az --include='*/' --include='log.jsonl' --include='summary.json' --include='key_analysis.json' \
    --include='*.log' --include='failures.txt' --include='jobs.txt' --exclude='*' \
    "$HOST:~/hais_poc/poc_m3/$DIR/" "$DIR/" 2>/dev/null
  done_n=$(ls -d $DIR/$PAT/ 2>/dev/null | while read d; do [ -f "$d/summary.json" ] && echo x; done | wc -l)
  echo "[$(date +%H:%M)] $done_n/$N finished ($PAT)"
  [ "$done_n" -ge "$N" ] && break
  sleep $EVERY
done
echo PULL_DONE
