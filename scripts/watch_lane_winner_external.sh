#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRIDGE_ENV="${BRIDGE_ENV:-${ROOT_DIR}/ops/env/local-lane-bridge.env}"

if [[ ! -f "${BRIDGE_ENV}" ]]; then
  echo "[error] bridge env not found: ${BRIDGE_ENV}" >&2
  echo "Create it from: ${ROOT_DIR}/ops/env/local-lane-bridge.env.example" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "${BRIDGE_ENV}"
set +a

LOCAL_LANE_REPO="${LOCAL_LANE_REPO:-${HOME}/Documents/App/QuantRabbitLocalLane}"
LOCAL_LOG="${LOCAL_LOG:-${LOCAL_LANE_REPO}/state/codex_long_autotrade.log}"
VM_TRADES_DB="${VM_TRADES_DB:-${ROOT_DIR}/logs/trades.db}"
HOURS="${HOURS:-24}"
MIN_TRADES="${MIN_TRADES:-5}"
VM_PREFIX="${VM_PREFIX:-qr-}"
LOCAL_PREFIX="${LOCAL_PREFIX:-codexlhf_}"

export LOCAL_LOG VM_TRADES_DB HOURS MIN_TRADES VM_PREFIX LOCAL_PREFIX

exec "${ROOT_DIR}/scripts/watch_lane_winner.sh"
