#!/usr/bin/env bash
# Bootstrap a fresh GPU node you can already ssh into: push poc_m3 and start node_setup.sh (env, gates, data) detached.
#   bash node_bootstrap.sh <ip> [user] [shards]
set -euo pipefail
IP=$1; U=${2:-$USER}; SHARDS=${3:-14}
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 60); do ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$U@$IP" true 2>/dev/null && break; sleep 10; done
ssh "$U@$IP" 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sort | uniq -c; echo "cpus $(nproc)"; free -g | sed -n 2p; df -h ~ | tail -1'
ssh "$U@$IP" 'mkdir -p ~/hais_poc'
rsync -az --exclude runs --exclude 'runs_*' --exclude '__pycache__' --exclude '*.pt' --exclude '.node_ip' "$HERE/" "$U@$IP:hais_poc/poc_m3/"
ssh "$U@$IP" "cd ~/hais_poc && setsid nohup bash poc_m3/node_setup.sh $SHARDS > setup.log 2>&1 < /dev/null &"
echo "$U@$IP" > "$HERE/.node_ip"
echo "setup started: ssh $U@$IP tail -f hais_poc/setup.log"
