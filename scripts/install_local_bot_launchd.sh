#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LABEL="com.quantrabbit.local-bot-cycle"
INTERVAL_SEC=60
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
RUNNER="${ROOT_DIR}/tools/local_bot_cycle.sh"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[error] runner missing or not executable: ${RUNNER}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${PLIST_DIR}" "${ROOT_DIR}/logs"

cat >"${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUNNER}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT_DIR}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>${INTERVAL_SEC}</integer>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${ROOT_DIR}/logs/local_bot_cycle.launchd.out</string>
  <key>StandardErrorPath</key>
  <string>${ROOT_DIR}/logs/local_bot_cycle.launchd.err</string>
</dict>
</plist>
PLIST

"${PYTHON_BIN}" "${ROOT_DIR}/tools/render_bot_inventory_policy.py" --write-default --ttl-min 45 >/dev/null

launchctl bootout "gui/${UID}" "${PLIST_PATH}" >/dev/null 2>&1 || true
if ! launchctl bootstrap "gui/${UID}" "${PLIST_PATH}"; then
  launchctl load -w "${PLIST_PATH}"
fi
launchctl enable "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true
launchctl kickstart -k "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true

echo "[ok] installed launchd agent: ${LABEL}"
echo "[ok] plist: ${PLIST_PATH}"
