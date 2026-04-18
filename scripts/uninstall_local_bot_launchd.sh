#!/usr/bin/env bash
set -euo pipefail

LABEL="com.quantrabbit.local-bot-cycle"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/${UID}" "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl disable "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true
launchctl remove "${LABEL}" >/dev/null 2>&1 || true
launchctl unload -w "${PLIST_PATH}" >/dev/null 2>&1 || true
rm -f "${PLIST_PATH}"

echo "[ok] removed launchd agent: ${LABEL}"
