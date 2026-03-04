#!/usr/bin/env bash
set -euo pipefail

LABEL="com.quantrabbit.local-v2-autorecover"

usage() {
  cat <<'USAGE'
Usage:
  scripts/uninstall_local_v2_launchd.sh [--label <label>]
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
launchctl bootout "gui/${UID}" "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl disable "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true
launchctl remove "${LABEL}" >/dev/null 2>&1 || true
launchctl unload -w "${PLIST_PATH}" >/dev/null 2>&1 || true

if [[ -f "${PLIST_PATH}" ]]; then
  rm -f "${PLIST_PATH}"
fi

echo "[ok] removed launchd agent: ${LABEL}"
echo "[ok] plist removed: ${PLIST_PATH}"
