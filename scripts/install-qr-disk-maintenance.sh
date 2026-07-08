#!/bin/bash
# Install the QuantRabbit disk maintenance launchd agent.
# It compresses local replay history only; it never places or modifies orders.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install-qr-disk-maintenance.sh [--check]

Environment:
  QR_DISK_MAINTENANCE_LIVE_ROOT   default: /Users/tossaki/App/QuantRabbit-live
  QR_DISK_MAINTENANCE_INTERVAL    default: 1800
  QR_PYTHON                       default: /opt/homebrew/bin/python3 or /usr/bin/python3
USAGE
}

CHECK=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install-qr-disk-maintenance] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

LIVE_ROOT="${QR_DISK_MAINTENANCE_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
SCRIPT="$LIVE_ROOT/scripts/qr_disk_maintenance.py"
LABEL="com.quantrabbit.disk-maintenance"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL="${QR_DISK_MAINTENANCE_INTERVAL:-1800}"

if [[ -z "${QR_PYTHON:-}" ]]; then
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    QR_PYTHON="/opt/homebrew/bin/python3"
  else
    QR_PYTHON="/usr/bin/python3"
  fi
fi

if [[ ! -x "$QR_PYTHON" ]]; then
  echo "[install-qr-disk-maintenance] QR_PYTHON is not executable: $QR_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[install-qr-disk-maintenance] missing maintenance script: $SCRIPT" >&2
  exit 2
fi

"$QR_PYTHON" "$SCRIPT" --help >/dev/null

if [[ "$CHECK" -eq 1 ]]; then
  echo "[install-qr-disk-maintenance] preflight OK: $SCRIPT"
  exit 0
fi

chmod +x "$SCRIPT"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$QR_PYTHON</string>
    <string>$SCRIPT</string>
    <string>--root</string>
    <string>$LIVE_ROOT</string>
    <string>--apply</string>
    <string>--min-age-minutes</string>
    <string>30</string>
    <string>--prune-temp-days</string>
    <string>2</string>
  </array>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/tmp/$LABEL.out.log</string>
  <key>StandardErrorPath</key><string>/tmp/$LABEL.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed: $PLIST (every ${INTERVAL}s)"
