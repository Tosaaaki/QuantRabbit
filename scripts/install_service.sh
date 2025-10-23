#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   sudo bash scripts/install_service.sh /opt/quantrabbit <linux-user>
#   (repo must exist at the given path)
#
# This installs the systemd service+timer to run auto-tuning periodically.

REPO_DIR="${1:-/opt/quantrabbit}"
LINUX_USER="${2:-$SUDO_USER}"
WITH_UI=0
if [[ "${3:-}" == "--with-ui" ]]; then
  WITH_UI=1
fi

if [[ -z "${LINUX_USER}" ]]; then
  echo "Please provide linux user as 2nd arg"; exit 1
fi

SERVICE_NAME="quant-autotune"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
TIMER_FILE="/etc/systemd/system/${SERVICE_NAME}.timer"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=QuantRabbit Continuous Backtest + AutoTune
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=/usr/bin/env python3 scripts/continuous_backtest.py --profile all --write-best
Restart=on-failure
RestartSec=10
User=${LINUX_USER}
Environment=PYTHONUNBUFFERED=1
Environment=AUTOTUNE_BQ_TABLE=${AUTOTUNE_BQ_TABLE:-quantrabbit.autotune_runs}
Environment=AUTOTUNE_BQ_SETTINGS_TABLE=${AUTOTUNE_BQ_SETTINGS_TABLE:-quantrabbit.autotune_settings}

[Install]
WantedBy=default.target
EOF

cat > "${TIMER_FILE}" <<EOF
[Unit]
Description=Run quant-autotune.service every hour

[Timer]
OnBootSec=2min
OnUnitActiveSec=1h
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME}.timer

echo "Installed and started ${SERVICE_NAME}.timer"
systemctl status ${SERVICE_NAME}.timer --no-pager

if [[ $WITH_UI -eq 1 ]]; then
  UI_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}-ui.service"
  cat > "${UI_SERVICE_FILE}" <<EOF
[Unit]
Description=QuantRabbit Autotune Review UI
After=network.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/.venv/bin/uvicorn apps.autotune_ui:app --host 0.0.0.0 --port 8088
Restart=on-failure
RestartSec=5
User=${LINUX_USER}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF
  systemctl daemon-reload
  systemctl enable --now ${SERVICE_NAME}-ui.service
  echo "Installed and started ${SERVICE_NAME}-ui.service (listening on port 8088)"
  systemctl status ${SERVICE_NAME}-ui.service --no-pager
fi
