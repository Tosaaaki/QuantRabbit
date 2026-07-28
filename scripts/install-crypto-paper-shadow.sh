#!/bin/bash
# Install isolated bitbank Spot/Margin Paper Shadow and reporting launchd agents.

set -euo pipefail

ACTION="${1:-install}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${QR_CRYPTO_SHADOW_PYTHON:-$(command -v python3)}"
RUNTIME_ROOT="$ROOT/data/crypto/paper-shadow"
LOG_ROOT="$ROOT/logs/crypto-paper-shadow"
DOMAIN="gui/$(id -u)"
LABELS=(
  "com.quantrabbit.crypto-paper-shadow-spot"
  "com.quantrabbit.crypto-paper-shadow-margin"
  "com.quantrabbit.crypto-paper-shadow-reporter"
)

write_plist() {
  local target="$1"
  local label="$2"
  local mode="$3"
  "$PYTHON" - "$target" "$label" "$mode" "$PYTHON" "$ROOT" \
    "$RUNTIME_ROOT" "$LOG_ROOT" <<'PY'
import os
import plistlib
import sys

target, label, mode, python, root, runtime_root, log_root = sys.argv[1:]
if mode == "reporter":
    arguments = [
        python,
        "-m",
        "quant_rabbit.crypto.cli",
        "shadow-report",
        "--runtime-root",
        runtime_root,
    ]
    schedule = {"StartInterval": 300, "RunAtLoad": True}
else:
    arguments = [
        python,
        "-m",
        "quant_rabbit.crypto.cli",
        "shadow-service",
        "--mode",
        mode,
        "--runtime-root",
        runtime_root,
        "--initial-cash-jpy",
        "10000",
    ]
    schedule = {"KeepAlive": True, "RunAtLoad": True}
payload = {
    "Label": label,
    "ProgramArguments": arguments,
    "WorkingDirectory": root,
    "EnvironmentVariables": {
        "PYTHONPATH": os.path.join(root, "src"),
        "NO_EXECUTE": "true",
        "CRYPTO_LIVE_READY": "false",
        "WITHDRAWAL_ENABLED": "false",
        "CRYPTO_ORDER_AUTHORITY": "NONE",
        "QR_CRYPTO_FAST_TELEMETRY_EVERY_EVENTS": "25",
    },
    **schedule,
    "ProcessType": "Background",
    "StandardOutPath": os.path.join(log_root, f"{label}.out.log"),
    "StandardErrorPath": os.path.join(log_root, f"{label}.err.log"),
}
with open(target, "wb") as handle:
    plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=False)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

plist_for() {
  echo "$HOME/Library/LaunchAgents/$1.plist"
}

case "$ACTION" in
  --check|check)
    for index in 0 1 2; do
      mode=("spot" "margin" "reporter")
      temporary="$(mktemp "${TMPDIR:-/tmp}/crypto-shadow.XXXXXX")"
      write_plist "$temporary" "${LABELS[$index]}" "${mode[$index]}"
      plutil -lint "$temporary" >/dev/null
      rm -f "$temporary"
    done
    echo "[crypto-paper-shadow] preflight OK root=$ROOT python=$PYTHON"
    ;;
  status)
    for label in "${LABELS[@]}"; do
      launchctl print "$DOMAIN/$label" 2>/dev/null \
        | sed -n '1,16p' || echo "$label not-loaded"
    done
    ;;
  uninstall|stop)
    for label in "${LABELS[@]}"; do
      plist="$(plist_for "$label")"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      if [[ "$ACTION" == "uninstall" ]]; then
        rm -f "$plist"
      fi
    done
    echo "[crypto-paper-shadow] $ACTION complete"
    ;;
  install|start)
    mkdir -p "$HOME/Library/LaunchAgents" "$RUNTIME_ROOT" "$LOG_ROOT"
    modes=("spot" "margin" "reporter")
    for index in 0 1 2; do
      label="${LABELS[$index]}"
      plist="$(plist_for "$label")"
      temporary="$(mktemp "$HOME/Library/LaunchAgents/.$label.XXXXXX")"
      write_plist "$temporary" "$label" "${modes[$index]}"
      plutil -lint "$temporary" >/dev/null
      chmod 600 "$temporary"
      mv -f "$temporary" "$plist"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      launchctl bootstrap "$DOMAIN" "$plist"
    done
    echo "[crypto-paper-shadow] started Spot, Margin, reporter"
    ;;
  *)
    echo "usage: $0 [install|start|stop|status|uninstall|--check]" >&2
    exit 2
    ;;
esac
