#!/usr/bin/env bash
set -euo pipefail

# Deploy fallback: use instance metadata startup-script + reset (no SSH/IAP needed).
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
SINGLE_VM_PREFIX="fx-trader"
SINGLE_VM_GUARD="$SCRIPT_DIR/ensure_single_trading_vm.sh"

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
DIAG_FILE="/var/log/qr-startup-guard.log"
if [[ -w /dev/ttyS0 ]]; then
  exec > >(tee -a "\$LOG_FILE" /dev/ttyS0) 2>&1
else
  exec > >(tee -a "\$LOG_FILE") 2>&1
fi
DEPLOY_ID="$DEPLOY_ID"
REPO_DIR="$REPO_DIR"
BRANCH="$BRANCH"
SERVICE="$SERVICE"
RUNTIME_ENV_FILE="\$REPO_DIR/ops/env/quant-v2-runtime.env"
INSTALL_DEPS="$INSTALL_DEPS"
RUN_REPORT="$RUN_REPORT"
BUNDLE_GCS="$BUNDLE_GCS"
REPO_OWNER="\$(basename "\$(dirname "\$REPO_DIR")")"
STAMP_DIR="/var/lib/quantrabbit"
STAMP_FILE="\$STAMP_DIR/deploy_id"
if [[ -d "\$REPO_DIR" ]]; then
  mkdir -p "\$REPO_DIR/logs" "\$REPO_DIR/config"
  chown -R "\$REPO_OWNER":"\$REPO_OWNER" "\$REPO_DIR/logs" "\$REPO_DIR/config" || true
  chmod -R u+rwX "\$REPO_DIR/logs" "\$REPO_DIR/config" || true
fi
if [[ -d "\$REPO_DIR" ]]; then
  REPO_OWNER="\$(stat -c '%U' "\$REPO_DIR" 2>/dev/null || true)"
fi
if [[ -z "\$REPO_OWNER" ]]; then
  REPO_OWNER="\$(basename "\$(dirname "\$REPO_DIR")")"
fi
if ! id -u "\$REPO_OWNER" >/dev/null 2>&1; then
  echo "[startup] repo owner missing or invalid: \$REPO_OWNER, fallback root"
  REPO_OWNER="root"
fi
log_startup() {
  local msg="\$1"
  local ts
  ts="\$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '%s [startup] %s\n' "\$ts" "\$msg" | tee -a "\$LOG_FILE" "\$DIAG_FILE"
  logger -t quantrabbit-startup "\$msg" || true
}
log_startup "bootstrap-start deploy_id=\$DEPLOY_ID branch=\$BRANCH repo=\$REPO_DIR service=\$SERVICE"
log_startup "tmp_repo_owner=\$REPO_OWNER"

sshd_config() {
  local cfg="\$1"
  local backup="\$2"

  if [[ ! -f "\$cfg" ]] && [[ -f "\$backup" ]]; then
    cp -f "\$backup" "\$cfg"
  elif [[ ! -f "\$cfg" ]]; then
    mkdir -p "/etc/ssh"
    cat > "\$cfg" <<'EOF_MIN_CFG'
Port 22
AddressFamily any
ListenAddress 0.0.0.0
PermitRootLogin prohibit-password
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding no
PrintMotd no
Subsystem sftp /usr/lib/openssh/sftp-server
EOF_MIN_CFG
    if ! grep -q '^Subsystem sftp ' "\$cfg" && [[ -x /usr/libexec/openssh/sftp-server ]]; then
      sed -i 's#/usr/lib/openssh/sftp-server#/usr/libexec/openssh/sftp-server#' "\$cfg"
    fi
  fi

  if command -v sshd >/dev/null 2>&1 && ! sshd -t -f "\$cfg" >/dev/null 2>&1; then
    log_startup "sshd config invalid: \$cfg"
    if [[ -f "\$backup" ]] && sshd -t -f "\$backup" >/dev/null 2>&1; then
      cp -f "\$backup" "\$cfg"
      log_startup "sshd_config restored from backup: \$cfg"
    fi
  fi
}

ensure_ssh_service() {
  local ssh_unit=""
  local port_ok=0
  if systemctl list-unit-files --type=service | grep -q '^ssh\.service'; then
    ssh_unit="ssh"
  elif systemctl list-unit-files --type=service | grep -q '^sshd\.service'; then
    ssh_unit="sshd"
  fi

  if [[ -z "\$ssh_unit" ]]; then
    log_startup "ssh unit file not found"
    return 0
  fi

  log_startup "ensuring ssh unit: \$ssh_unit"
  systemctl unmask "\$ssh_unit" || true
  systemctl enable "\$ssh_unit" || true
  if ! systemctl restart "\$ssh_unit"; then
    log_startup "failed to restart \$ssh_unit"
    return 0
  fi

  if ! systemctl is-active "\$ssh_unit" >/dev/null 2>&1; then
    log_startup "\$ssh_unit is not active"
    return 0
  fi

  if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print \$4}' | grep -Eq '(:22)$'; then
    port_ok=1
  fi
  if (( port_ok == 0 )); then
    log_startup "ssh port 22 not listening; retry restart"
    systemctl restart "\$ssh_unit" || true
    local retry_count=0
    while (( retry_count < 5 )); do
      sleep 1
      if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print \$4}' | grep -Eq '(:22)$'; then
        port_ok=1
        break
      fi
      systemctl restart "\$ssh_unit" || true
      retry_count=\$(( retry_count + 1 ))
    done
  fi

  if (( port_ok == 0 )); then
    if [[ -f /etc/ssh/sshd_config ]]; then
      sshd -t -f /etc/ssh/sshd_config || true
    fi
    log_startup "ssh still not listening on 22 after retry"
  else
    log_startup "ssh service \$ssh_unit listening on port 22"
  fi
}

ssh_port_snapshot() {
  local tag="\$1"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn | awk '{print \$4}' | grep -Eq '(:22)$'; then
      log_startup "port-check[\$tag]: port22=LISTEN"
    else
      log_startup "port-check[\$tag]: port22=not-listening"
    fi
  else
    log_startup "port-check[\$tag]: ss unavailable"
  fi
}

disable_ssh_guard_units() {
  local line unit
  local legacy_units=(
    ssh-guard.service
    ssh-guard.timer
    sshguard.service
    sshguard.timer
    quant-ssh-watchdog.service
    quant-ssh-watchdog.timer
    quant-boot-sync.service
    quant-boot-sync.timer
  )
  while IFS= read -r line; do
    unit="\${line%% *}"
    for pattern in "\${legacy_units[@]}"; do
      if [[ "\$unit" == "\$pattern" ]]; then
        log_startup "disabling legacy unit \$unit"
        systemctl stop "\$unit" || true
        systemctl disable "\$unit" || true
        systemctl mask "\$unit" || true
      fi
    done
  done < <(systemctl list-unit-files --type=service --no-legend)
  while IFS= read -r line; do
    unit="\${line%% *}"
    for pattern in "\${legacy_units[@]}"; do
      if [[ "\$unit" == "\$pattern" ]]; then
        log_startup "disabling legacy unit \$unit"
        systemctl stop "\$unit" || true
        systemctl disable "\$unit" || true
        systemctl mask "\$unit" || true
      fi
    done
  done < <(systemctl list-unit-files --type=timer --no-legend)

  # Some AMIs/snapshots can re-enable the legacy guard from stale unit files after boot.
  rm -f \
    /etc/systemd/system/ssh-guard.service /etc/systemd/system/ssh-guard.timer \
    /etc/systemd/system/sshguard.service /etc/systemd/system/sshguard.timer \
    /etc/systemd/system/quant-ssh-watchdog.service /etc/systemd/system/quant-ssh-watchdog.timer \
    /etc/systemd/system/quant-boot-sync.service /etc/systemd/system/quant-boot-sync.timer \
    /usr/lib/systemd/system/ssh-guard.service /usr/lib/systemd/system/ssh-guard.timer \
    /usr/lib/systemd/system/sshguard.service /usr/lib/systemd/system/sshguard.timer \
    /usr/lib/systemd/system/quant-ssh-watchdog.service /usr/lib/systemd/system/quant-ssh-watchdog.timer \
    /usr/lib/systemd/system/quant-boot-sync.service /usr/lib/systemd/system/quant-boot-sync.timer \
    /lib/systemd/system/ssh-guard.service /lib/systemd/system/ssh-guard.timer \
    /lib/systemd/system/sshguard.service /lib/systemd/system/sshguard.timer \
    /lib/systemd/system/quant-ssh-watchdog.service /lib/systemd/system/quant-ssh-watchdog.timer \
    /lib/systemd/system/quant-boot-sync.service /lib/systemd/system/quant-boot-sync.timer || true
  systemctl daemon-reload || true
}

ensure_repo_venv() {
  local venv_dir="\$REPO_DIR/.venv"
  if [[ -x "\$venv_dir/bin/python" ]]; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    log_startup "python3 not available; skip .venv bootstrap"
    return 0
  fi
  log_startup "creating missing virtualenv at \$venv_dir"
  rm -rf "\$venv_dir" || true
  if ! python3 -m venv "\$venv_dir"; then
    log_startup "python3 -m venv failed; skipping .venv bootstrap"
    return 0
  fi
  chown -R "\$REPO_OWNER":"\$REPO_OWNER" "\$venv_dir" || true
}


log_startup "deploy_id=\$DEPLOY_ID branch=\$BRANCH repo=\$REPO_DIR service=\$SERVICE"
MARKER_BUCKET=""
if [[ -f "\$RUNTIME_ENV_FILE" ]]; then
  MARKER_BUCKET="\$(grep -E '^(GCS_UI_BUCKET|ui_bucket_name)=' "\$RUNTIME_ENV_FILE" | tail -n 1 | cut -d= -f2-)"
  if [[ -z "\$MARKER_BUCKET" ]]; then
    MARKER_BUCKET="\$(grep -E '^GCS_BACKUP_BUCKET=' "\$RUNTIME_ENV_FILE" | tail -n 1 | cut -d= -f2-)"
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
  log_startup "deploy_id already applied: \$DEPLOY_ID"
  exit 0
fi
echo "\$DEPLOY_ID" > "\$STAMP_FILE"

if [[ -n "${ENV_OVERRIDES_CONTENT}" ]]; then
  log_startup "applying env overrides from ${ENV_FILE}"
  if mkdir -p "\$REPO_DIR/ops/env"; then
  cat > /tmp/qr_env_overrides <<EOF_OVR
${ENV_OVERRIDES_CONTENT}
EOF_OVR
  : > "\$RUNTIME_ENV_FILE" || true
  cp "\$RUNTIME_ENV_FILE" "\$RUNTIME_ENV_FILE.bak.\$DEPLOY_ID" || true
  while IFS= read -r line; do
    trimmed="\${line%%#*}"
    trimmed="\$(echo "\$trimmed" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "\$trimmed" ]] && continue
    key="\${trimmed%%=*}"
    if grep -q "^\\\${key}=" "\$RUNTIME_ENV_FILE"; then
      sed -i "s|^\\\${key}=.*|\\\${trimmed}|" "\$RUNTIME_ENV_FILE"
    else
      echo "\$trimmed" >> "\$RUNTIME_ENV_FILE"
    fi
  done < /tmp/qr_env_overrides
  rm -f /tmp/qr_env_overrides || true
  else
    log_startup "failed to prepare ops/env; skip env override apply"
  fi
fi

if [[ ! -d "\$REPO_DIR/.git" ]]; then
  if [[ -d "\$REPO_DIR" ]]; then
    log_startup "stale repo dir without .git, removing: \$REPO_DIR"
    rm -rf "\$REPO_DIR"
  fi
  mkdir -p "\$REPO_DIR"
  chown "\$REPO_OWNER":"\$REPO_OWNER" "\$REPO_DIR" || true
  log_startup "cloning repo"
  sudo -u "\$REPO_OWNER" -H bash -lc "git clone https://github.com/Tosaaaki/QuantRabbit.git \"\$REPO_DIR\""
fi

git_ok=1
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git fetch --all -q"; then
  log_startup "git fetch failed"
  git_ok=0
fi
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git checkout -q \"\$BRANCH\" || git checkout -b \"\$BRANCH\" \"origin/\$BRANCH\""; then
  log_startup "git checkout failed"
  git_ok=0
fi
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git pull --ff-only -q"; then
  log_startup "git pull failed"
  git_ok=0
fi
if ! sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && git rev-parse --short HEAD 2>/dev/null || echo unknown"; then
  log_startup "git_rev lookup failed"
fi
if [[ "\$git_ok" -ne 1 ]]; then
  log_startup "git update failed; continuing with existing code"
fi
ensure_repo_venv || true

if [[ -n "\$BUNDLE_GCS" ]]; then
  log_startup "applying bundle deploy from \$BUNDLE_GCS"
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
    log_startup "bundle extracted"
  else
    log_startup "bundle download failed"
  fi
fi

disable_ssh_guard_units
if [[ -f "\$REPO_DIR/scripts/install_core_backup_service.sh" ]]; then
  log_startup "installing quant-core-backup executable and service bundle"
  bash "\$REPO_DIR/scripts/install_core_backup_service.sh" --repo "\$REPO_DIR" || true
fi
if [[ -f "\$REPO_DIR/scripts/install_trading_services.sh" ]]; then
  log_startup "running install_trading_services.sh for core services"
  bash "\$REPO_DIR/scripts/install_trading_services.sh" --repo "\$REPO_DIR" --units "quant-market-data-feed.service quant-strategy-control.service quant-order-manager.service quant-position-manager.service quant-core-backup.service quant-core-backup.timer cleanup-qr-logs.service cleanup-qr-logs.timer"
fi
if [[ -f "\$REPO_DIR/scripts/ssh_watchdog.sh" ]]; then
  log_startup "running install_trading_services.sh with guard pre-clear for observability units"
  disable_ssh_guard_units
  bash "\$REPO_DIR/scripts/install_trading_services.sh" --repo "\$REPO_DIR" --units "quant-forecast-watchdog.service quant-forecast-watchdog.timer quant-health-snapshot.service quant-health-snapshot.timer quant-ui-snapshot.service quant-ui-snapshot.timer quant-bq-sync.service"
fi
disable_ssh_guard_units
log_startup "post-service setup guard re-check"

if [[ "\$INSTALL_DEPS" == "1" ]]; then
  ensure_repo_venv || true
  if [[ -f "\$REPO_DIR/.venv/bin/pip" ]]; then
    sudo -u "\$REPO_OWNER" -H bash -lc "cd \"\$REPO_DIR\" && . .venv/bin/activate && pip install -r requirements.txt"
  else
    log_startup ".venv pip missing; skip requirements install"
  fi
else
  ensure_repo_venv || true
fi

if systemctl list-unit-files --type=service | grep -F -q "^\$SERVICE"; then
  if ! systemctl restart "\$SERVICE"; then
    log_startup "restart failed: \$SERVICE"
  fi
  log_startup "service restart attempted: \$SERVICE"
  if ! systemctl is-active "\$SERVICE"; then
    log_startup "service not active after restart: \$SERVICE"
    systemctl status --no-pager -l "\$SERVICE" || true
  fi
else
  log_startup "target service not found, skip restart: \$SERVICE"
fi

if ! systemctl list-unit-files --type=service | grep -q '^ssh\\.service\\|^sshd\\.service'; then
  if command -v apt-get >/dev/null 2>&1; then
    log_startup "installing openssh-server"
    apt-get update -y || true
    apt-get install -y openssh-server || true
  fi
fi
if command -v sshd >/dev/null 2>&1; then
  sshd_config /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
else
  if [[ -f /etc/ssh/sshd_config ]]; then
    if ! grep -q 'Port 22' /etc/ssh/sshd_config; then
      echo "Port 22" >> /etc/ssh/sshd_config
    fi
    if ! grep -q 'AddressFamily' /etc/ssh/sshd_config; then
      echo "AddressFamily any" >> /etc/ssh/sshd_config
    fi
    if ! grep -q 'ListenAddress 0.0.0.0' /etc/ssh/sshd_config; then
      echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config
    fi
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

disable_ssh_guard_units
ssh_port_snapshot "before-final-ensure"
ensure_ssh_service
ssh_port_snapshot "after-final-ensure"
if systemctl list-unit-files --type=service | grep -q '^google-guest-agent\\.service'; then
  systemctl unmask google-guest-agent || true
  systemctl restart google-guest-agent || true
fi

disable_ssh_guard_units
ssh_port_snapshot "after-final-guard"

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
"$SINGLE_VM_GUARD" \
  -p "$PROJECT" \
  -m "$INSTANCE" \
  -P "$SINGLE_VM_PREFIX" \
  -A "$SA_IMPERSONATE" \
  --strict
gcloud compute instances add-metadata "$INSTANCE" \
  --project "$PROJECT" --zone "$ZONE" ${IMPERSONATE_ARG} \
  --metadata-from-file startup-script="$TMP_SCRIPT"

echo "[deploy-meta] Resetting instance to run startup-script"
gcloud compute instances reset "$INSTANCE" --project "$PROJECT" --zone "$ZONE" ${IMPERSONATE_ARG}

echo "[deploy-meta] Done. When SSH/IAP recovers, tail logs via journalctl."
