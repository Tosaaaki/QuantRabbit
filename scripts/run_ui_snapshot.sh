#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/home/tossaki/QuantRabbit}"
MAX_RUNTIME_SEC="${UI_SNAPSHOT_MAX_RUNTIME_SEC:-90}"
EXTRA_ARGS=()
if [[ -n "${UI_SNAPSHOT_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${UI_SNAPSHOT_ARGS}"
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[ui-snapshot] repo not found: $REPO_DIR" >&2
  exit 1
fi

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[ui-snapshot] python3 not found" >&2
  exit 1
fi

if ! [[ "$MAX_RUNTIME_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ui-snapshot] invalid UI_SNAPSHOT_MAX_RUNTIME_SEC=$MAX_RUNTIME_SEC; fallback=90" >&2
  MAX_RUNTIME_SEC="90"
fi

SNAPSHOT_CMD=("$PYTHON_BIN" -u "$REPO_DIR/scripts/publish_ui_snapshot.py")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  SNAPSHOT_CMD+=("${EXTRA_ARGS[@]}")
fi

if command -v timeout >/dev/null 2>&1; then
  if timeout --signal=TERM --kill-after=10s "$MAX_RUNTIME_SEC" "${SNAPSHOT_CMD[@]}"; then
    exit 0
  else
    rc=$?
  fi
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    echo "[ui-snapshot] timed out after ${MAX_RUNTIME_SEC}s; skip this cycle and continue next timer run"
    exit 0
  fi
  exit "$rc"
fi

exec "${SNAPSHOT_CMD[@]}"
