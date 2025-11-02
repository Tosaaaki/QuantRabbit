#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit – MacroState deployment helper
#
# This script configures MacroState gating, event-window risk controls, and
# USD exposure clamp on a remote VM via gcloud (OS Login/IAP). It also deploys
# the current repo revision to the VM and restarts the systemd service.
#
# Requirements (local):
#   - gcloud CLI with OS Login/IAP access
#   - permissions: roles/compute.osAdminLogin (+ iap.tunnelResourceAccessor if IAP)
#
# Example:
#   scripts/deploy_macro_state.sh \
#     -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
#     -d /home/tossaki/QuantRabbit -s quantrabbit.service \
#     --snapshot /home/tossaki/QuantRabbit/fixtures/macro_snapshots/latest.json \
#     --gate --deadzone 0.25 --event --event-before 2 --event-after 1 --event-mul 0.3 \
#     --usd-long-cap 2.5 --restart --iap

PROJECT=""
ZONE=""
INSTANCE=""
REPO_DIR="/home/tossaki/QuantRabbit"
SERVICE="quantrabbit.service"
BRANCH="main"
USE_IAP=false
SSH_KEY=""

SNAPSHOT=""
GATE=false
DEADZONE="0.25"
STALE_WARN_SEC="900"
EVENT=false
EVENT_BEFORE="2"
EVENT_AFTER="1"
EVENT_MUL="0.3"
USD_LONG_CAP="2.5"
DRY_RUN=false
DO_RESTART=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2;;
    -z|--zone) ZONE="$2"; shift 2;;
    -m|--instance) INSTANCE="$2"; shift 2;;
    -d|--repo-dir) REPO_DIR="$2"; shift 2;;
    -s|--service) SERVICE="$2"; shift 2;;
    -b|--branch) BRANCH="$2"; shift 2;;
    -k|--ssh-key) SSH_KEY="$2"; shift 2;;
    --iap) USE_IAP=true; shift;;
    --snapshot) SNAPSHOT="$2"; shift 2;;
    --gate) GATE=true; shift;;
    --deadzone) DEADZONE="$2"; shift 2;;
    --stale-warn) STALE_WARN_SEC="$2"; shift 2;;
    --event) EVENT=true; shift;;
    --event-before) EVENT_BEFORE="$2"; shift 2;;
    --event-after) EVENT_AFTER="$2"; shift 2;;
    --event-mul) EVENT_MUL="$2"; shift 2;;
    --usd-long-cap) USD_LONG_CAP="$2"; shift 2;;
    --restart) DO_RESTART=true; shift;;
    --dry-run) DRY_RUN=true; shift;;
    -h|--help)
      echo "Usage: $0 -p <PROJECT> -z <ZONE> -m <INSTANCE> [options]";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "$PROJECT" || -z "$ZONE" || -z "$INSTANCE" ]]; then
  echo "ERROR: --project/--zone/--instance are required." >&2
  exit 2
fi

GCLOUD_ARGS=("compute" "ssh" "$INSTANCE" "--project=$PROJECT" "--zone=$ZONE")
if $USE_IAP; then GCLOUD_ARGS+=("--tunnel-through-iap"); fi
if [[ -n "$SSH_KEY" ]]; then GCLOUD_ARGS+=("--ssh-key-file" "$SSH_KEY"); fi

run_ssh() {
  local cmd="$1"
  if $DRY_RUN; then
    echo "gcloud ${GCLOUD_ARGS[*]} --command $cmd"
  else
    gcloud "${GCLOUD_ARGS[@]}" --command "$cmd"
  fi
}

echo "[INFO] Applying systemd override envs on $INSTANCE ..."
MACRO_ENABLED=$($GATE && echo true || echo false)
EVENT_ENABLED=$($EVENT && echo true || echo false)

read -r -d '' REMOTE_OVERRIDES <<EOF || true
sudo bash -lc '
  set -euo pipefail
  mkdir -p /etc/systemd/system/$SERVICE.d
  cat > /etc/systemd/system/$SERVICE.d/override.conf <<EOC
[Service]
Environment=MACRO_STATE_SNAPSHOT_PATH=$SNAPSHOT
Environment=MACRO_STATE_GATE_ENABLED=$MACRO_ENABLED
Environment=MACRO_STATE_DEADZONE=$DEADZONE
Environment=MACRO_STATE_STALE_WARN_SEC=$STALE_WARN_SEC
Environment=EVENT_WINDOW_ENABLED=$EVENT_ENABLED
Environment=EVENT_WINDOW_BEFORE_HOURS=$EVENT_BEFORE
Environment=EVENT_WINDOW_AFTER_HOURS=$EVENT_AFTER
Environment=EVENT_WINDOW_RISK_MULTIPLIER=$EVENT_MUL
Environment=EXPOSURE_USD_LONG_MAX_LOT=$USD_LONG_CAP
Environment=LLM_MODE=
EOC
  systemctl daemon-reload
'
EOF

run_ssh "$REMOTE_OVERRIDES"

echo "[INFO] Deploying repo to $REPO_DIR (branch=$BRANCH) ..."
read -r -d '' REMOTE_DEPLOY <<EOF || true
sudo -u tossaki -H bash -lc '
  set -euo pipefail
  cd $REPO_DIR
  git fetch --all -q || true
  git checkout -q $BRANCH || git checkout -b $BRANCH origin/$BRANCH || true
  git pull --ff-only
  if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi
'
EOF

run_ssh "$REMOTE_DEPLOY"

if $DO_RESTART; then
  echo "[INFO] Restarting $SERVICE ..."
  run_ssh "sudo systemctl restart $SERVICE && sudo systemctl status --no-pager -l $SERVICE || true"
fi

echo "[OK] Deployment script finished. Use scripts/vm.sh tail -s $SERVICE -t to follow logs."
