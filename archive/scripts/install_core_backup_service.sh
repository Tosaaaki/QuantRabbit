#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/tossaki/QuantRabbit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: sudo bash scripts/install_core_backup_service.sh [--repo /home/tossaki/QuantRabbit]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

SCRIPT_SRC="$REPO_DIR/scripts/qr_gcs_backup_core.sh"
SERVICE_SRC="$REPO_DIR/systemd/quant-core-backup.service"
TIMER_SRC="$REPO_DIR/systemd/quant-core-backup.timer"
ENV_SRC="$REPO_DIR/ops/env/quant-core-backup.env"

for path in "$SCRIPT_SRC" "$SERVICE_SRC" "$TIMER_SRC" "$ENV_SRC"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

install -m 0755 "$SCRIPT_SRC" /usr/local/bin/qr-gcs-backup-core
install -m 0644 "$SERVICE_SRC" /etc/systemd/system/quant-core-backup.service
install -m 0644 "$TIMER_SRC" /etc/systemd/system/quant-core-backup.timer

ENV_DEST="/home/tossaki/QuantRabbit/ops/env/quant-core-backup.env"
if [[ "$(realpath "$ENV_SRC")" != "$(realpath "$ENV_DEST" 2>/dev/null || echo "$ENV_DEST")" ]]; then
  install -m 0644 "$ENV_SRC" "$ENV_DEST"
fi

if [[ -f /etc/cron.hourly/qr-gcs-backup-core ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  mv /etc/cron.hourly/qr-gcs-backup-core "/etc/cron.hourly/qr-gcs-backup-core.disabled.${ts}"
  echo "Disabled legacy cron.hourly backup script."
fi

# Stop legacy backup/tar jobs if they are still running.
pkill -f "/usr/local/bin/qr-gcs-backup-core|tar --ignore-failed-read -cf /tmp/qr_logs_core_" 2>/dev/null || true

systemctl daemon-reload
systemctl enable --now quant-core-backup.timer

# Run once now for smoke check. Guard logic may skip on busy VM and still be healthy.
if ! systemctl start quant-core-backup.service; then
  echo "quant-core-backup.service returned non-zero (check journal)." >&2
fi

systemctl --no-pager -l status quant-core-backup.timer
systemctl --no-pager -l status quant-core-backup.service || true

echo "Installed guarded core backup service/timer."
