#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${ROOT_DIR}/logs/lane_winner_latest.json"
OUT_JSONL="${ROOT_DIR}/logs/lane_winner_history.jsonl"

LOCAL_LOG="${LOCAL_LOG:-${ROOT_DIR}/logs/codex_long_autotrade.log}"
VM_TRADES_DB="${VM_TRADES_DB:-${ROOT_DIR}/logs/trades.db}"
HOURS="${HOURS:-24}"
MIN_TRADES="${MIN_TRADES:-5}"
VM_PREFIX="${VM_PREFIX:-qr-}"
LOCAL_PREFIX="${LOCAL_PREFIX:-codexlhf_}"

mkdir -p "${ROOT_DIR}/logs"

RESULT="$("${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/compare_live_lanes.py" \
  --local-log "${LOCAL_LOG}" \
  --vm-trades-db "${VM_TRADES_DB}" \
  --hours "${HOURS}" \
  --min-trades "${MIN_TRADES}" \
  --vm-prefix "${VM_PREFIX}" \
  --local-prefix "${LOCAL_PREFIX}")"

printf '%s\n' "${RESULT}" > "${OUT_JSON}"
printf '%s\n' "${RESULT}" >> "${OUT_JSONL}"
printf '%s\n' "${RESULT}"
