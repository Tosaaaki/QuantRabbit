#!/bin/bash
# Bootstrap script for bare GCE VM when Terraform metadata script is not available.
# Installs dependencies, configures systemd service, and starts the trading loop.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "[startup] Please run as root" >&2
  exit 1
fi

useradd -m -s /bin/bash quantrabbit || true

export DEBIAN_FRONTEND=noninteractive
apt-get update
dpkg --configure -a || true
apt-get install -y python3 python3-venv python3-pip git

# Provide Secret Manager client for the bootstrap helper below.
python3 -m pip install --no-cache-dir --upgrade --break-system-packages \
  google-cloud-secret-manager google-cloud-logging

sudo -u quantrabbit bash -lc "
  cd \$HOME
  if [ ! -d QuantRabbit ]; then
    git clone https://github.com/Tosaaaki/QuantRabbit.git
  else
    cd QuantRabbit && git pull --rebase
  fi
  cd \$HOME/QuantRabbit
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

# Write environment variables from Google Secret Manager into /etc/quantrabbit.env
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
    "OPENAI_API_KEY": read_secret("openai_api_key"),
    "OANDA_TOKEN": read_secret("oanda_token"),
    "OANDA_ACCOUNT": read_secret("oanda_account_id"),
    "OANDA_PRACTICE": read_secret("oanda_practice"),
}

with open("/etc/quantrabbit.env", "w") as fh:
    for key, value in pairs.items():
        if value:
            fh.write(f"{key}={value}\n")
    if not (pairs["OANDA_TOKEN"] and pairs["OANDA_ACCOUNT"]):
        fh.write("MOCK_TICK_STREAM=1\n")
    fh.write("LOOP_SEC=10\n")
    fh.write("TUNER_ENABLE=1\n")
    fh.write("TUNER_SHADOW_MODE=false\n")
    fh.write("TUNER_INTERVAL_SEC=600\n")
    fh.write("TUNER_WINDOW_MINUTES=15\n")
    fh.write("TUNER_LOGS_GLOB=tmp/exit_eval_*_v2.csv\n")
PY

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

systemctl daemon-reload
systemctl enable --now quantrabbit.service

echo "[startup] quantrabbit.service started"
