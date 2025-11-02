#!/usr/bin/env bash
set -euo pipefail

# Deploy balanced low-vol rollout (soft-TP lead + hazard tuned) to the VM.
#
# Prereqs:
#   - gcloud authenticated with access to the project/VM
#   - OS Login key registered if using IAP (-t) and --ssh-key-file
#   - scripts/vm.sh available in this repo
#
# Usage examples:
#   scripts/deploy_balanced_lowvol.sh \
#     -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t \
#     -k ~/.ssh/gcp_oslogin_quantrabbit
#
# Options:
#   -p <PROJECT>   GCP project
#   -z <ZONE>      GCE zone
#   -m <INSTANCE>  VM instance name
#   -k <KEYFILE>   SSH private key for OS Login (optional)
#   -t             Use IAP tunnel (no external IP)
#   -d <DIR>       Remote repo dir (default: /home/tossaki/QuantRabbit)
#   -A <ACCOUNT>   gcloud account (optional)

PROJECT=""
ZONE=""
INSTANCE=""
KEYFILE=""
USE_IAP=""
REMOTE_DIR="/home/tossaki/QuantRabbit"
ACCOUNT=""

die() { echo "[deploy_balanced_lowvol] $*" >&2; exit 1; }

while getopts ":p:z:m:k:td:A:" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    t) USE_IAP=1 ;;
    d) REMOTE_DIR="$OPTARG" ;;
    A) ACCOUNT="$OPTARG" ;;
    :) die "Option -$OPTARG requires an argument" ;;
    \?) die "Unknown option: -$OPTARG" ;;
  esac
done

[[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p, -z, -m are required"

VM_SH="$(dirname "$0")/vm.sh"
[[ -x "$VM_SH" ]] || die "scripts/vm.sh not found or not executable"

VM_BASE=("$VM_SH" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE")
if [[ -n "$KEYFILE" ]]; then VM_BASE+=( -k "$KEYFILE" ); fi
if [[ -n "$USE_IAP" ]]; then VM_BASE+=( -t ); fi
if [[ -n "$ACCOUNT" ]]; then VM_BASE+=( -A "$ACCOUNT" ); fi

echo "[deploy_balanced_lowvol] Copying execution/exit_manager.py to remote /tmp ..."
"${VM_BASE[@]}" scp --to-remote execution/exit_manager.py /tmp/exit_manager.py

echo "[deploy_balanced_lowvol] Installing exit_manager.py into $REMOTE_DIR/execution ..."
"${VM_BASE[@]}" exec -- "sudo -u tossaki -H bash -lc 'install -m 664 /tmp/exit_manager.py ${REMOTE_DIR}/execution/exit_manager.py'"

ENV_PATH="${REMOTE_DIR}/config/.service.env"
echo "[deploy_balanced_lowvol] Writing balanced env flags into ${ENV_PATH} ..."
read -r -d '' REMOTE_ENV <<'EOF' || true
set -euo pipefail
FILE="$1"
mkdir -p "$(dirname "$FILE")"
touch "$FILE"

upsert() {
  local key="$1"; shift
  local val="$1"; shift
  if grep -qE "^${key}=" "$FILE"; then
    sed -i.bak "s|^${key}=.*|${key}=${val}|" "$FILE"
  else
    printf '%s\n' "${key}=${val}" >>"$FILE"
  fi
}

# Feature flags
upsert LOWVOL_ENABLE true
upsert HAZARD_EXIT_ENABLE true
upsert LOWVOL_CANAIRY_SYMBOLS USD_JPY,EUR_USD
upsert LOWVOL_CANAIRY_HOURS 07:00-10:00,14:00-17:00

# Exit control (balanced)
upsert EVENT_BUDGET_TICKS 16
upsert GRACE_MS 200
upsert HAZARD_DEBOUNCE_TICKS 1
upsert UPPER_BOUND_MIN_SEC 1.6
upsert UPPER_BOUND_MAX_SEC 3.2

# Hazard cost
upsert HAZARD_COST_SPREAD_BASE 0.22
upsert HAZARD_COST_LATENCY_BASE_MS 220

# Soft TP before timeout
upsert SOFT_TP_PIPS 0.30
upsert TIMEOUT_SOFT_TP_FRAC 0.70

echo "[remote] Updated $FILE:" && tail -n +1 "$FILE" | sed -n '1,200p'
EOF

"${VM_BASE[@]}" exec -- "bash -lc '$(printf "%q" "$REMOTE_ENV") ${ENV_PATH}'"

echo "[deploy_balanced_lowvol] Restarting systemd service ..."
"${VM_BASE[@]}" exec -- "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"

echo "[deploy_balanced_lowvol] Tail recent logs (20 lines) ..."
"${VM_BASE[@]}" exec -- "journalctl -u quantrabbit.service -n 20 --output=short-iso"

echo "[deploy_balanced_lowvol] Done."

