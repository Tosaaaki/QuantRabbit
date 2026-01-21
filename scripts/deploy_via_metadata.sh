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
  -e <ENV_FILE>      Local env overrides file (default: local/vm_env_overrides.env)
  -g <GCS_BUNDLE>    GCS bundle path (e.g. gs://bucket/deploy/qr_bundle.tar.gz)
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
ENV_FILE="local/vm_env_overrides.env"
BUNDLE_GCS=""
SA_KEYFILE=""
SA_IMPERSONATE=""

while getopts ":p:z:m:b:d:s:ire:g:K:A:" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    b) BRANCH="$OPTARG" ;;
    d) REPO_DIR="$OPTARG" ;;
    s) SERVICE="$OPTARG" ;;
    i) INSTALL_DEPS=1 ;;
    r) RUN_REPORT=1 ;;
    e) ENV_FILE="$OPTARG" ;;
    g) BUNDLE_GCS="$OPTARG" ;;
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
ENV_OVERRIDES_CONTENT=""
if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  ENV_OVERRIDES_CONTENT="$(sed -e 's/\\/\\\\/g' -e 's/`/\\`/g' -e 's/\\$/\\$/g' "${ENV_FILE}")"
fi

cat > "$TMP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
LOG_FILE="/var/log/quantrabbit-startup.log"
if [[ -w /dev/ttyS0 ]]; then
  exec > >(tee -a "\$LOG_FILE" /dev/ttyS0) 2>&1
else
  exec > >(tee -a "\$LOG_FILE") 2>&1
fi
DEPLOY_ID="$DEPLOY_ID"
REPO_DIR="$REPO_DIR"
BRANCH="$BRANCH"
SERVICE="$SERVICE"
INSTALL_DEPS="$INSTALL_DEPS"
RUN_REPORT="$RUN_REPORT"
BUNDLE_GCS="$BUNDLE_GCS"
REPO_OWNER="\$(basename "\$(dirname "\$REPO_DIR")")"
STAMP_DIR="/var/lib/quantrabbit"
STAMP_FILE="\$STAMP_DIR/deploy_id"

echo "[startup] deploy_id=\$DEPLOY_ID branch=\$BRANCH repo=\$REPO_DIR service=\$SERVICE"
MARKER_BUCKET=""
if [[ -f /etc/quantrabbit.env ]]; then
  MARKER_BUCKET="\$(grep -E '^(GCS_UI_BUCKET|ui_bucket_name)=' /etc/quantrabbit.env | tail -n 1 | cut -d= -f2-)"
  if [[ -z "\$MARKER_BUCKET" ]]; then
    MARKER_BUCKET="\$(grep -E '^GCS_BACKUP_BUCKET=' /etc/quantrabbit.env | tail -n 1 | cut -d= -f2-)"
  fi
fi
if [[ -n "\$MARKER_BUCKET" ]] && command -v python3 >/dev/null 2>&1; then
  MARKER_BUCKET="\$MARKER_BUCKET" DEPLOY_ID="\$DEPLOY_ID" python3 - <<'PY'
import json
import os
import socket
import urllib.parse
import urllib.request
from datetime import datetime, timezone

bucket = os.environ.get("MARKER_BUCKET")
deploy_id = os.environ.get("DEPLOY_ID")
if not bucket or not deploy_id:
    raise SystemExit(0)
payload = json.dumps(
    {
        "deploy_id": deploy_id,
        "hostname": socket.gethostname(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    },
    ensure_ascii=True,
    separators=(",", ":"),
).encode("utf-8")
token_req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"},
)
try:
    with urllib.request.urlopen(token_req, timeout=2.0) as resp:
        token = json.loads(resp.read().decode("utf-8")).get("access_token")
except Exception:
    token = None
if not token:
    raise SystemExit(0)
obj = f"realtime/startup_{socket.gethostname()}_{deploy_id}.json"
obj_enc = urllib.parse.quote(obj, safe="/")
url = (
    "https://storage.googleapis.com/upload/storage/v1/b/"
    f"{bucket}/o?uploadType=media&name={obj_enc}"
)
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Content-Length": str(len(payload)),
}
req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
try:
    with urllib.request.urlopen(req, timeout=5.0):
        pass
except Exception:
    pass
PY
fi
mkdir -p "\$STAMP_DIR"
if [[ -f "\$STAMP_FILE" ]] && [[ "\$(cat "\$STAMP_FILE")" == "\$DEPLOY_ID" ]]; then
  echo "[startup] deploy_id already applied: \$DEPLOY_ID"
  exit 0
fi
echo "\$DEPLOY_ID" > "\$STAMP_FILE"

if [[ -n "${ENV_OVERRIDES_CONTENT}" ]]; then
  echo "[startup] applying env overrides from ${ENV_FILE}"
  cat > /tmp/qr_env_overrides <<EOF_OVR
${ENV_OVERRIDES_CONTENT}
EOF_OVR
  touch /etc/quantrabbit.env
  cp /etc/quantrabbit.env "/etc/quantrabbit.env.bak.\$DEPLOY_ID" || true
  while IFS= read -r line; do
    trimmed="\${line%%#*}"
    trimmed="\$(echo "\$trimmed" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "\$trimmed" ]] && continue
    key="\${trimmed%%=*}"
    if grep -q "^\\\${key}=" /etc/quantrabbit.env; then
      sed -i "s|^\\\${key}=.*|\\\${trimmed}|" /etc/quantrabbit.env
    else
      echo "\$trimmed" >> /etc/quantrabbit.env
    fi
  done < /tmp/qr_env_overrides
fi

if [[ ! -d "\$REPO_DIR/.git" ]]; then
  echo "[startup] cloning repo"
  sudo -u "\$REPO_OWNER" -H bash -lc "git clone https://github.com/Tosaaaki/QuantRabbit.git \"\$REPO_DIR\""
fi

git_ok=1
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git fetch --all -q"; then
  echo "[startup] git fetch failed"
  git_ok=0
fi
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git checkout -q \"\$BRANCH\" || git checkout -b \"\$BRANCH\" \"origin/\$BRANCH\""; then
  echo "[startup] git checkout failed"
  git_ok=0
fi
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git pull --ff-only -q"; then
  echo "[startup] git pull failed"
  git_ok=0
fi
sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && echo \"[startup] git_rev=\$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\""
if [[ "\$git_ok" -ne 1 ]]; then
  echo "[startup] git update failed; continuing with existing code"
fi

if [[ -n "\$BUNDLE_GCS" ]]; then
  echo "[startup] applying bundle deploy from \$BUNDLE_GCS"
  export BUNDLE_GCS
  tmp_bundle="/tmp/qr_bundle.tar.gz"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "\$BUNDLE_GCS" "\$tmp_bundle" || true
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil cp "\$BUNDLE_GCS" "\$tmp_bundle" || true
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import os
import sys
import json
import urllib.parse
import urllib.request

def _metadata_token():
    url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("access_token")
    except Exception:
        return None

def _download_via_metadata(bucket, obj, dest):
    token = _metadata_token()
    if not token:
        return False
    obj_enc = urllib.parse.quote(obj, safe="/")
    url = (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{bucket}/o/{obj_enc}?alt=media"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = resp.read()
    except Exception:
        return False
    try:
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False

try:
    from google.cloud import storage
except Exception:
    storage = None
path = os.environ.get("BUNDLE_GCS", "")
if not path.startswith("gs://"):
    sys.exit(0)
bucket, _, obj = path[5:].partition("/")
if not bucket or not obj:
    sys.exit(0)
dest = "/tmp/qr_bundle.tar.gz"
if storage is not None:
    try:
        client = storage.Client()
        blob = client.bucket(bucket).blob(obj)
        blob.download_to_filename(dest)
        sys.exit(0)
    except Exception:
        pass
if _download_via_metadata(bucket, obj, dest):
    sys.exit(0)
PY
  fi
  if [[ -f "\$tmp_bundle" ]]; then
    mkdir -p "\$REPO_DIR"
    tar -xzf "\$tmp_bundle" -C "\$REPO_DIR"
    echo "[startup] bundle extracted"
  else
    echo "[startup] bundle download failed"
  fi
fi

if [[ -f "\$REPO_DIR/scripts/ssh_watchdog.sh" ]]; then
  bash "\$REPO_DIR/scripts/install_trading_services.sh" --repo "\$REPO_DIR" --units "quant-ssh-watchdog.service quant-ssh-watchdog.timer quant-health-snapshot.service quant-health-snapshot.timer quant-ui-snapshot.service quant-ui-snapshot.timer quant-bq-sync.service"
fi

if [[ "\$INSTALL_DEPS" == "1" ]]; then
  sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && if [ -d .venv ]; then . .venv/bin/activate && pip install -r requirements.txt; else echo '[startup] venv not found, skipping pip install'; fi"
fi

systemctl restart "\$SERVICE"
systemctl is-active "\$SERVICE" || systemctl status --no-pager -l "\$SERVICE" || true

if ! systemctl list-unit-files --type=service | grep -q '^ssh\\.service\\|^sshd\\.service'; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[startup] installing openssh-server"
    apt-get update -y || true
    apt-get install -y openssh-server || true
  fi
fi
if [[ -d /etc/ssh/sshd_config.d ]]; then
  cat > /etc/ssh/sshd_config.d/99-qr.conf <<'EOF_QR_SSH'
Port 22
AddressFamily any
ListenAddress 0.0.0.0
EOF_QR_SSH
elif [[ -f /etc/ssh/sshd_config ]]; then
  grep -q '^Port 22' /etc/ssh/sshd_config || echo 'Port 22' >> /etc/ssh/sshd_config
  grep -q '^AddressFamily' /etc/ssh/sshd_config || echo 'AddressFamily any' >> /etc/ssh/sshd_config
  grep -q '^ListenAddress 0.0.0.0' /etc/ssh/sshd_config || echo 'ListenAddress 0.0.0.0' >> /etc/ssh/sshd_config
fi
if command -v ssh-keygen >/dev/null 2>&1; then
  ssh-keygen -A || true
fi
if systemctl list-unit-files --type=service | grep -q '^ssh\\.service'; then
  systemctl unmask ssh || true
  systemctl enable ssh || true
  systemctl restart ssh || true
elif systemctl list-unit-files --type=service | grep -q '^sshd\\.service'; then
  systemctl unmask sshd || true
  systemctl enable sshd || true
  systemctl restart sshd || true
fi
if systemctl list-unit-files --type=service | grep -q '^google-guest-agent\\.service'; then
  systemctl unmask google-guest-agent || true
  systemctl restart google-guest-agent || true
fi
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
  ufw allow 22/tcp || true
fi

if systemctl list-unit-files --type=service | grep -q '^quant-health-snapshot\\.service'; then
  systemctl start quant-health-snapshot.service || true
fi
if systemctl list-unit-files --type=service | grep -q '^quant-ui-snapshot\\.service'; then
  systemctl start quant-ui-snapshot.service || true
fi
if systemctl list-unit-files --type=service | grep -q '^quant-bq-sync\\.service'; then
  systemctl restart quant-bq-sync.service || true
fi

if [[ -f "\$REPO_DIR/scripts/run_health_snapshot.sh" ]]; then
  bash "\$REPO_DIR/scripts/run_health_snapshot.sh" || true
fi
if [[ -f "\$REPO_DIR/scripts/run_ui_snapshot.sh" ]]; then
  bash "\$REPO_DIR/scripts/run_ui_snapshot.sh" || true
fi

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
