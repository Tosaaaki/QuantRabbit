#!/usr/bin/env bash
set -euo pipefail

LABEL="com.quantrabbit.local-v2-autorecover"
ROOT_DIR="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

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

if grep -Fq "/Users/tossaki/Documents/App/QuantRabbit/" "${PLIST_PATH}"; then
  echo "[warn] plist still references the Documents symlink path"
  echo "[hint] reinstall launchd from the physical repo root: ${ROOT_DIR}/scripts/install_local_v2_launchd.sh"
fi
