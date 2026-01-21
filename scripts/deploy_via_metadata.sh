#!/usr/bin/env bash
set -euo pipefail

# Deploy fallback: use instance metadata startup-script + reset (no SSH/IAP needed).

usage() {
  cat <<'USAGE'
Usage: scripts/deploy_via_metadata.sh [options]
  -p <PROJECT>       GCP project (default: gcloud config)
  -z <ZONE>          GCE zone (default: asia-northeast1-a)
  -m <INSTANCE>      VM instance (default: fx-trader-vm)
  -b <BRANCH>        Git branch (default: current local branch or main)
  -d <REPO_DIR>      Remote repo dir (default: /home/tossaki/QuantRabbit)
  -s <SERVICE>       systemd service (default: quantrabbit.service)
  -i                Install requirements in remote .venv (if exists)
  -r                Run health report after restart (serial output)
  -K <SA_KEYFILE>    Service Account JSON key (optional)
  -A <SA_ACCOUNT>    Service Account email to impersonate (optional)
USAGE
}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "[deploy-meta] gcloud not found; run scripts/install_gcloud.sh first." >&2
  exit 2
fi

PROJECT="$(gcloud config get-value project 2>/dev/null || echo "")"
ZONE="asia-northeast1-a"
INSTANCE="fx-trader-vm"
BRANCH=""
REPO_DIR="/home/tossaki/QuantRabbit"
SERVICE="quantrabbit.service"
INSTALL_DEPS=0
RUN_REPORT=0
SA_KEYFILE=""
SA_IMPERSONATE=""

while getopts ":p:z:m:b:d:s:irK:A:" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    b) BRANCH="$OPTARG" ;;
    d) REPO_DIR="$OPTARG" ;;
    s) SERVICE="$OPTARG" ;;
    i) INSTALL_DEPS=1 ;;
    r) RUN_REPORT=1 ;;
    K) SA_KEYFILE="$OPTARG" ;;
    A) SA_IMPERSONATE="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "[deploy-meta] GCP project is not set; pass -p." >&2
  exit 2
fi

if [[ -z "$BRANCH" ]]; then
  if command -v git >/dev/null 2>&1; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
  else
    BRANCH="main"
  fi
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' || true)
if [[ -z "$ACTIVE_ACCOUNT" && -n "$SA_KEYFILE" ]]; then
  echo "[deploy-meta] No active account; activating service account from key: $SA_KEYFILE"
  gcloud auth activate-service-account --key-file "$SA_KEYFILE"
fi

IMPERSONATE_ARG=""
if [[ -n "$SA_IMPERSONATE" ]]; then
  IMPERSONATE_ARG="--impersonate-service-account=$SA_IMPERSONATE"
fi

DEPLOY_ID="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat > "$TMP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ID="$DEPLOY_ID"
REPO_DIR="$REPO_DIR"
BRANCH="$BRANCH"
SERVICE="$SERVICE"
INSTALL_DEPS="$INSTALL_DEPS"
RUN_REPORT="$RUN_REPORT"
REPO_OWNER="\$(basename "\$(dirname "\$REPO_DIR")")"
STAMP_DIR="/var/lib/quantrabbit"
STAMP_FILE="\$STAMP_DIR/deploy_id"

mkdir -p "\$STAMP_DIR"
if [[ -f "\$STAMP_FILE" ]] && [[ "\$(cat "\$STAMP_FILE")" == "\$DEPLOY_ID" ]]; then
  echo "[startup] deploy_id already applied: \$DEPLOY_ID"
  exit 0
fi
echo "\$DEPLOY_ID" > "\$STAMP_FILE"

if [[ ! -d "\$REPO_DIR/.git" ]]; then
  sudo -u "\$REPO_OWNER" -H bash -lc "git clone https://github.com/Tosaaaki/QuantRabbit.git \"\$REPO_DIR\""
fi

sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git fetch --all -q || true && git checkout -q \"\$BRANCH\" || git checkout -b \"\$BRANCH\" \"origin/\$BRANCH\" || true && git pull --ff-only -q || true"

if [[ -f "\$REPO_DIR/scripts/ssh_watchdog.sh" ]]; then
  bash "\$REPO_DIR/scripts/install_trading_services.sh" --repo "\$REPO_DIR" --units "quant-ssh-watchdog.service quant-ssh-watchdog.timer"
fi

if [[ "\$INSTALL_DEPS" == "1" ]]; then
  sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && if [ -d .venv ]; then . .venv/bin/activate && pip install -r requirements.txt; else echo '[startup] venv not found, skipping pip install'; fi"
fi

systemctl restart "\$SERVICE"
systemctl is-active "\$SERVICE" || systemctl status --no-pager -l "\$SERVICE" || true

if [[ "\$RUN_REPORT" == "1" && -f "\$REPO_DIR/scripts/report_vm_health.sh" ]]; then
  bash "\$REPO_DIR/scripts/report_vm_health.sh" || true
fi
EOF

echo "[deploy-meta] Applying startup-script metadata (deploy_id=$DEPLOY_ID)"
gcloud compute instances add-metadata "$INSTANCE" \
  --project "$PROJECT" --zone "$ZONE" ${IMPERSONATE_ARG} \
  --metadata-from-file startup-script="$TMP_SCRIPT"

echo "[deploy-meta] Resetting instance to run startup-script"
gcloud compute instances reset "$INSTANCE" --project "$PROJECT" --zone "$ZONE" ${IMPERSONATE_ARG}

echo "[deploy-meta] Done. When SSH/IAP recovers, tail logs via journalctl."
