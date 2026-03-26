#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${1:-${HEALTH_SNAPSHOT_REPO:-$DEFAULT_REPO_DIR}}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[health-snapshot] repo not found: $REPO_DIR" >&2
  exit 1
fi

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  exec "$REPO_DIR/.venv/bin/python" -u "$REPO_DIR/scripts/publish_health_snapshot.py"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -u "$REPO_DIR/scripts/publish_health_snapshot.py"
fi

echo "[health-snapshot] python3 not found" >&2
exit 1
