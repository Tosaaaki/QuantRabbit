#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTORECOVER_SCRIPT="${ROOT_DIR}/scripts/local_v2_autorecover_once.sh"

LABEL="com.quantrabbit.local-v2-autorecover"
PROFILE="trade_min"
ENV_FILE="${ROOT_DIR}/ops/env/local-v2-stack.env"
SERVICES=""
INTERVAL_SEC=20
NET_HOST="api-fxtrade.oanda.com"
NET_PORT="443"
NET_TIMEOUT_SEC="2.0"

usage() {
  cat <<'USAGE'
Usage:
  scripts/install_local_v2_launchd.sh [options]

Options:
  --label <label>         launchd label (default: com.quantrabbit.local-v2-autorecover)
  --profile <profile>     local_v2_stack profile (default: trade_min)
  --env <file>            override env path (default: ops/env/local-v2-stack.env)
  --services <csv>        optional explicit service list
  --interval-sec <sec>    health/recovery check interval (default: 20)
  --net-host <host>       network reachability host (default: api-fxtrade.oanda.com)
  --net-port <port>       network reachability port (default: 443)
  --net-timeout <sec>     network check timeout sec (default: 2.0)
  -h, --help              show help
USAGE
}

xml_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "${s}"
}

shell_single_quote() {
  local s="$1"
  s="${s//\'/\'\"\'\"\'}"
  printf "'%s'" "${s}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --services)
      SERVICES="$2"
      shift 2
      ;;
    --interval-sec)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --net-host)
      NET_HOST="$2"
      shift 2
      ;;
    --net-port)
      NET_PORT="$2"
      shift 2
      ;;
    --net-timeout)
      NET_TIMEOUT_SEC="$2"
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

case "${ENV_FILE}" in
  /*) ;;
  *) ENV_FILE="${ROOT_DIR}/${ENV_FILE}" ;;
esac

if [[ ! -x "${AUTORECOVER_SCRIPT}" ]]; then
  echo "[error] autorecover script not executable: ${AUTORECOVER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[error] env file not found: ${ENV_FILE}" >&2
  exit 1
fi
if ! [[ "${INTERVAL_SEC}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_SEC}" -lt 5 ]]; then
  echo "[error] --interval-sec must be integer >= 5" >&2
  exit 1
fi
if ! [[ "${NET_PORT}" =~ ^[0-9]+$ ]]; then
  echo "[error] --net-port must be numeric" >&2
  exit 1
fi

PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
mkdir -p "${PLIST_DIR}" "${ROOT_DIR}/logs"

SERVICES_XML=""
AUTORECOVER_CMD="cd / && exec $(shell_single_quote "${AUTORECOVER_SCRIPT}") --profile $(shell_single_quote "${PROFILE}") --env $(shell_single_quote "${ENV_FILE}")"
if [[ -n "${SERVICES}" ]]; then
  AUTORECOVER_CMD="${AUTORECOVER_CMD} --services $(shell_single_quote "${SERVICES}")"
fi

cat >"${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "${LABEL}")</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>$(xml_escape "${AUTORECOVER_CMD}")</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>QR_LOCAL_V2_NET_CHECK_HOST</key>
    <string>$(xml_escape "${NET_HOST}")</string>
    <key>QR_LOCAL_V2_NET_CHECK_PORT</key>
    <string>$(xml_escape "${NET_PORT}")</string>
    <key>QR_LOCAL_V2_NET_TIMEOUT_SEC</key>
    <string>$(xml_escape "${NET_TIMEOUT_SEC}")</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>
  <key>AbandonProcessGroup</key>
  <true/>
  <key>StartInterval</key>
  <integer>${INTERVAL_SEC}</integer>
  <key>KeepAlive</key>
  <dict>
    <key>NetworkState</key>
    <true/>
  </dict>

  <key>StandardOutPath</key>
  <string>$(xml_escape "${ROOT_DIR}/logs/local_v2_autorecover.launchd.out")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "${ROOT_DIR}/logs/local_v2_autorecover.launchd.err")</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/${UID}" "${PLIST_PATH}" >/dev/null 2>&1 || true
if ! launchctl bootstrap "gui/${UID}" "${PLIST_PATH}"; then
  # Fallback for older launchctl behavior.
  launchctl load -w "${PLIST_PATH}"
fi
launchctl enable "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true
launchctl kickstart -k "gui/${UID}/${LABEL}" >/dev/null 2>&1 || true

echo "[ok] installed launchd agent: ${LABEL}"
echo "[ok] plist: ${PLIST_PATH}"
echo "[ok] log: ${ROOT_DIR}/logs/local_v2_autorecover.log"
