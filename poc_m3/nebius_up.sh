#!/usr/bin/env bash
# Provision one 8-GPU Nebius VM, push poc_m3 to it and start node_setup.sh (env + correctness gates + data prep).
#   bash nebius_up.sh                    # tries 8x B200, then 8x H200
#   PLATFORMS="gpu-h200-sxm:8gpu-128vcpu-1600gb" bash nebius_up.sh
#   bash nebius_up.sh down               # delete the VM (and its boot disk) -- stops billing
# Needs a working `nebius` profile (see `nebius profile list`).
set -euo pipefail
NB=${NB:-$HOME/.local/bin/nebius}
PROJECT=${PROJECT:?set PROJECT to your Nebius project id}
NAME=${NAME:-hais-node}
IMAGE=${IMAGE:-ubuntu24.04-cuda13.0}   # driver 580.x; recommended by Nebius for both B200 and H200
DISK_GB=${DISK_GB:-1024}
PLATFORMS=${PLATFORMS:-"gpu-b200-sxm:8gpu-160vcpu-1792gb gpu-h200-sxm:8gpu-128vcpu-1600gb"}
HERE=$(cd "$(dirname "$0")" && pwd)

vm_id() { $NB compute instance get-by-name --parent-id "$PROJECT" --name "$NAME" --format json 2>/dev/null | jq -r '.metadata.id // empty'; }

if [ "${1:-}" = "down" ]; then
  id=$(vm_id); [ -n "$id" ] && $NB compute instance delete --id "$id"
  disk=$($NB compute disk get-by-name --parent-id "$PROJECT" --name "$NAME-boot" --format json 2>/dev/null | jq -r '.metadata.id // empty')
  [ -n "$disk" ] && $NB compute disk delete --id "$disk"
  exit 0
fi

if [ -z "$(vm_id)" ]; then
  SUBNET=$($NB vpc subnet list --parent-id "$PROJECT" --format json | jq -r '.items[0].metadata.id')
  echo "subnet: $SUBNET"
  USER_DATA=$(cat <<EOF
#cloud-config
users:
  - name: $USER
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_rsa.pub)
EOF
)
  ok=""
  for pp in $PLATFORMS; do
    plat=${pp%%:*}; preset=${pp##*:}
    echo "trying $plat / $preset ..."
    if $NB compute instance create --parent-id "$PROJECT" --name "$NAME" \
        --resources-platform "$plat" --resources-preset "$preset" \
        --boot-disk-managed-disk-name "$NAME-boot" --boot-disk-managed-disk-type network_ssd \
        --boot-disk-managed-disk-size-gibibytes "$DISK_GB" \
        --boot-disk-managed-disk-source-image-family-image-family "$IMAGE" \
        --boot-disk-attach-mode READ_WRITE \
        --cloud-init-user-data "$USER_DATA" \
        --network-interfaces "[{\"name\": \"eth0\", \"ip_address\": {}, \"public_ip_address\": {}, \"subnet_id\": \"$SUBNET\"}]" \
        --timeout 30m; then
      ok=$plat; break
    fi
    echo "  $plat failed (quota/capacity?), cleaning up"
    id=$(vm_id); [ -n "$id" ] && $NB compute instance delete --id "$id" || true
  done
  [ -n "$ok" ] || { echo "no 8-GPU platform could be created"; exit 1; }
fi

IP=$($NB compute instance get-by-name --parent-id "$PROJECT" --name "$NAME" --format json \
     | jq -r '.status.network_interfaces[0].public_ip_address.address' | cut -d/ -f1)
echo "VM up at $IP"
for i in $(seq 60); do ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$USER@$IP" true 2>/dev/null && break; sleep 10; done
ssh "$USER@$IP" 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2; df -h / | tail -1'
ssh "$USER@$IP" 'mkdir -p ~/hais_poc'
rsync -az --exclude runs --exclude 'runs_*' --exclude '__pycache__' --exclude '*.pt' "$HERE/" "$USER@$IP:hais_poc/poc_m3/"
ssh "$USER@$IP" 'cd ~/hais_poc && setsid nohup bash poc_m3/node_setup.sh 14 > setup.log 2>&1 < /dev/null &'
echo "setup running: ssh $USER@$IP tail -f hais_poc/setup.log"
echo "$IP" > "$HERE/.node_ip"
