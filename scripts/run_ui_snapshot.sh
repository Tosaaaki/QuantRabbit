#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/home/tossaki/QuantRabbit}"
EXTRA_ARGS=()
if [[ -n "${UI_SNAPSHOT_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${UI_SNAPSHOT_ARGS}"
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[ui-snapshot] repo not found: $REPO_DIR" >&2
  exit 1
fi

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  exec "$REPO_DIR/.venv/bin/python" -u "$REPO_DIR/scripts/publish_ui_snapshot.py" "${EXTRA_ARGS[@]}"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -u "$REPO_DIR/scripts/publish_ui_snapshot.py" "${EXTRA_ARGS[@]}"
fi

echo "[ui-snapshot] python3 not found" >&2
exit 1
