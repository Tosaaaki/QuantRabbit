#!/bin/bash
# Install the QuantRabbit disk maintenance launchd agent.
# It reclaims only strict runtime/temp allowlists; it never places or modifies orders.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install-qr-disk-maintenance.sh [--check]

Environment:
  QR_DISK_MAINTENANCE_LIVE_ROOT   default: /Users/tossaki/App/QuantRabbit-live
  QR_DISK_MAINTENANCE_INTERVAL    default: 1800
  QR_DISK_ATOMIC_HOURS            default: 48
  QR_DISK_DIAGNOSTIC_HOURS        default: 6
  QR_DISK_GUARDIAN_CACHE_HOURS    default: 168
  QR_DISK_GUARDIAN_TEMP_HOURS     default: 24
  QR_DISK_LAUNCHD_LOG_MAX_MB      default: 8
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
ATOMIC_HOURS="${QR_DISK_ATOMIC_HOURS:-48}"
DIAGNOSTIC_HOURS="${QR_DISK_DIAGNOSTIC_HOURS:-6}"
GUARDIAN_CACHE_HOURS="${QR_DISK_GUARDIAN_CACHE_HOURS:-168}"
GUARDIAN_TEMP_HOURS="${QR_DISK_GUARDIAN_TEMP_HOURS:-24}"
LAUNCHD_LOG_MAX_MB="${QR_DISK_LAUNCHD_LOG_MAX_MB:-8}"

if [[ ! "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[install-qr-disk-maintenance] interval must be a positive integer: $INTERVAL" >&2
  exit 2
fi
for value in "$ATOMIC_HOURS" "$DIAGNOSTIC_HOURS" "$GUARDIAN_CACHE_HOURS" "$GUARDIAN_TEMP_HOURS"; do
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[install-qr-disk-maintenance] retention values must be non-negative numbers: $value" >&2
    exit 2
  fi
done
if [[ ! "$LAUNCHD_LOG_MAX_MB" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$LAUNCHD_LOG_MAX_MB" =~ ^0+([.]0+)?$ ]]; then
  echo "[install-qr-disk-maintenance] log max must be a positive number: $LAUNCHD_LOG_MAX_MB" >&2
  exit 2
fi

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

HELP_OUTPUT="$("$QR_PYTHON" "$SCRIPT" --help)"
for required_flag in \
  --root --apply --min-age-minutes --prune-temp-days \
  --prune-atomic-hours --prune-external-diagnostics \
  --prune-diagnostic-hours --guardian-cache-hours \
  --guardian-temp-hours --log-max-mb; do
  if [[ "$HELP_OUTPUT" != *"$required_flag"* ]]; then
    echo "[install-qr-disk-maintenance] live script is missing required option: $required_flag" >&2
    exit 2
  fi
done

write_plist() {
  local target="$1"
  "$QR_PYTHON" - \
    "$target" "$LABEL" "$QR_PYTHON" "$SCRIPT" "$LIVE_ROOT" \
    "$ATOMIC_HOURS" "$DIAGNOSTIC_HOURS" "$GUARDIAN_CACHE_HOURS" \
    "$GUARDIAN_TEMP_HOURS" "$LAUNCHD_LOG_MAX_MB" "$INTERVAL" <<'PY'
import plistlib
import os
import sys

(
    target,
    label,
    python,
    script,
    live_root,
    atomic_hours,
    diagnostic_hours,
    guardian_cache_hours,
    guardian_temp_hours,
    launchd_log_max_mb,
    interval,
) = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [
        python,
        script,
        "--root",
        live_root,
        "--apply",
        "--min-age-minutes",
        "30",
        "--prune-temp-days",
        "2",
        "--prune-atomic-hours",
        atomic_hours,
        "--prune-external-diagnostics",
        "--prune-diagnostic-hours",
        diagnostic_hours,
        "--guardian-cache-hours",
        guardian_cache_hours,
        "--guardian-temp-hours",
        guardian_temp_hours,
        "--log-max-mb",
        launchd_log_max_mb,
    ],
    "StartInterval": int(interval),
    "RunAtLoad": True,
    "StandardOutPath": f"/tmp/{label}.out.log",
    "StandardErrorPath": f"/tmp/{label}.err.log",
}
with open(target, "wb") as handle:
    plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=False)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

if [[ "$CHECK" -eq 1 ]]; then
  CHECK_PLIST="$(mktemp "${TMPDIR:-/tmp}/$LABEL.check.XXXXXX")"
  trap 'rm -f "$CHECK_PLIST"' EXIT
  write_plist "$CHECK_PLIST"
  /usr/bin/plutil -lint "$CHECK_PLIST" >/dev/null
  echo "[install-qr-disk-maintenance] preflight OK: $SCRIPT"
  exit 0
fi

chmod +x "$SCRIPT"
mkdir -p "$HOME/Library/LaunchAgents"
PLIST_TMP="$(mktemp "$HOME/Library/LaunchAgents/.$LABEL.XXXXXX")"
trap 'rm -f "$PLIST_TMP"' EXIT
write_plist "$PLIST_TMP"
/usr/bin/plutil -lint "$PLIST_TMP" >/dev/null
chmod 600 "$PLIST_TMP"
mv -f "$PLIST_TMP" "$PLIST"
trap - EXIT

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed: $PLIST (every ${INTERVAL}s)"
