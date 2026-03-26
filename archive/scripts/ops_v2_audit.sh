#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${OPS_AUDIT_REPO:-$DEFAULT_REPO_DIR}"
PYTHON_BIN="${OPS_AUDIT_PYTHON:-$REPO_DIR/.venv/bin/python}"

# Ensure ops_v2_audit.py resolves the same repo path when launched via subprocess.
export OPS_AUDIT_REPO="$REPO_DIR"
export OPS_AUDIT_PYTHON="$PYTHON_BIN"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[ops-v2-audit] repo not found: $REPO_DIR" >&2
  exit 1
fi

if [[ -x "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" -u "$REPO_DIR/scripts/ops_v2_audit.py"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -u "$REPO_DIR/scripts/ops_v2_audit.py"
fi

echo "[ops-v2-audit] python3 not found" >&2
exit 1
