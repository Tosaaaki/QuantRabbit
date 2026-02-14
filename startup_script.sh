#!/bin/bash
# Bootstrap script for new GCE VM (Terraform metadata or manual run).
# Installs dependencies, configures systemd service, and starts the trading loop.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "[startup] Please run as root" >&2
  exit 1
fi

QR_USER="${QR_USER:-tossaki}"
QR_HOME="/home/${QR_USER}"
RUNTIME_ENV_FILE="${QR_HOME}/QuantRabbit/ops/env/quant-v2-runtime.env"
export RUNTIME_ENV_FILE

if ! id -u "$QR_USER" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "$QR_USER"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
dpkg --configure -a || true
apt-get install -y python3 python3-venv python3-pip git ca-certificates curl

# Provide Secret Manager client for the bootstrap helper below.
python3 -m pip install --no-cache-dir --upgrade --break-system-packages \
  google-cloud-secret-manager google-cloud-logging

sudo -u "$QR_USER" -H bash -lc "
  cd \"$QR_HOME\"
  if [ ! -d QuantRabbit ]; then
    git clone https://github.com/Tosaaaki/QuantRabbit.git
  else
    cd QuantRabbit && git pull --rebase
  fi
  cd \"$QR_HOME/QuantRabbit\"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

# Write environment variables from Google Secret Manager into the V2 runtime env file.
python3 - <<'PY'
import os
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
project = (
    os.environ.get("GCP_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER")
    or "quantrabbit"
)

def read_secret(name: str) -> str:
    try:
        resource = f"projects/{project}/secrets/{name}/versions/latest"
        response = client.access_secret_version(name=resource)
        return response.payload.data.decode("utf-8")
    except Exception:
        return ""

pairs = {
    "OANDA_TOKEN": read_secret("oanda_token"),
    "OANDA_ACCOUNT": read_secret("oanda_account_id"),
    "OANDA_PRACTICE": read_secret("oanda_practice"),
}

os.makedirs(os.path.dirname(os.environ["RUNTIME_ENV_FILE"]), exist_ok=True)
with open(os.environ["RUNTIME_ENV_FILE"], "w") as fh:
    for key, value in pairs.items():
        if value:
            fh.write(f"{key}={value}\n")
    if not (pairs["OANDA_TOKEN"] and pairs["OANDA_ACCOUNT"]):
        fh.write("MOCK_TICK_STREAM=1\n")
    fh.write("WORKER_ONLY_MODE=true\n")
    fh.write("MAIN_TRADING_ENABLED=0\n")
    fh.write("SIGNAL_GATE_ENABLED=0\n")
    fh.write("ORDER_FORWARD_TO_SIGNAL_GATE=0\n")
    fh.write("EXIT_MANAGER_DISABLED=1\n")
    fh.write("ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0\n")
    fh.write("ORDER_MANAGER_SERVICE_ENABLED=1\n")
    fh.write("ORDER_MANAGER_SERVICE_URL=http://127.0.0.1:8300\n")
    fh.write("ORDER_MANAGER_SERVICE_TIMEOUT=5.0\n")
    fh.write("ORDER_MANAGER_SERVICE_FALLBACK_LOCAL=0\n")
    fh.write("POSITION_MANAGER_SERVICE_ENABLED=1\n")
    fh.write("POSITION_MANAGER_SERVICE_URL=http://127.0.0.1:8301\n")
    fh.write("POSITION_MANAGER_SERVICE_TIMEOUT=5.0\n")
    fh.write("POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0\n")
    fh.write("BRAIN_ENABLED=0\n")
    fh.write("ORDER_MANAGER_BRAIN_GATE_ENABLED=0\n")
    fh.write("ORDER_MANAGER_FORECAST_GATE_ENABLED=0\n")
    fh.write("POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0\n")
    fh.write("ENTRY_GUARD_ENABLED=0\n")
    fh.write("ENTRY_TECH_ENABLED=0\n")
    fh.write("TUNER_ENABLE=1\n")
    fh.write("TUNER_SHADOW_MODE=true\n")
    fh.write("TUNER_INTERVAL_SEC=600\n")
    fh.write("TUNER_WINDOW_MINUTES=15\n")
    fh.write("TUNER_LOGS_GLOB=tmp/exit_eval_*_v2.csv\n")
PY

sudo -u "$QR_USER" -H bash -lc "
  cd \"$QR_HOME/QuantRabbit\"
  if [ -f \"${RUNTIME_ENV_FILE}\" ]; then
    set -a
    . \"${RUNTIME_ENV_FILE}\"
    set +a
  fi
  if [ -x .venv/bin/python ]; then
    .venv/bin/python scripts/refresh_env_from_gcp.py || true
  else
    python3 scripts/refresh_env_from_gcp.py || true
  fi
"

cat >/etc/google-cloud-ops-agent/config.yaml <<'CFG'
logging:
  receivers:
    journald_receiver:
      type: systemd_journald
      units: ["quantrabbit.service"]
  processors:
    parse_level:
      type: parse_json
  service:
    pipelines:
      journald_pipeline:
        receivers: [journald_receiver]
        processors: []
metrics:
  receivers:
    hostmetrics:
      type: hostmetrics
  service:
    pipelines:
      default_pipeline:
        receivers: [hostmetrics]
CFG

install -m 0644 ops/systemd/quantrabbit.service /etc/systemd/system/quantrabbit.service

if [[ "$QR_USER" != "tossaki" ]]; then
  install -d /etc/systemd/system/quantrabbit.service.d
  cat >/etc/systemd/system/quantrabbit.service.d/override.conf <<EOF
[Service]
User=${QR_USER}
WorkingDirectory=${QR_HOME}/QuantRabbit
Environment=HOME=${QR_HOME}
ExecStart=
ExecStart=${QR_HOME}/QuantRabbit/.venv/bin/python ${QR_HOME}/QuantRabbit/main.py
EOF
fi

systemctl daemon-reload
systemctl enable --now quantrabbit.service

echo "[startup] quantrabbit.service started"
