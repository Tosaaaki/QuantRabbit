#!/usr/bin/env bash
set -euo pipefail

LABEL="com.quantrabbit.local-v2-autorecover"

usage() {
  cat <<'USAGE'
Usage:
  scripts/status_local_v2_launchd.sh [--label <label>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
echo "label=${LABEL}"
echo "plist=${PLIST_PATH}"
echo "---"
if [[ ! -f "${PLIST_PATH}" ]]; then
  echo "[missing] plist not installed"
  exit 1
fi

launchctl print "gui/${UID}/${LABEL}" 2>/dev/null | sed -n '1,120p' || {
  echo "[warn] launchctl print failed; trying list"
  launchctl list | grep -F "${LABEL}" || true
}
