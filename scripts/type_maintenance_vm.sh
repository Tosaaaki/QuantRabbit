#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${TYPE_MAINTENANCE_REPO:-/home/tossaki/QuantRabbit}"
MODE="${TYPE_MAINTENANCE_MODE:-check}" # check | apply | fix | optimize
PYTHON="${TYPE_MAINTENANCE_PYTHON:-$REPO_DIR/.venv/bin/python}"
DEPS_FILE="$REPO_DIR/requirements-dev.txt"
LOG_ROOT="$REPO_DIR/logs"

if [[ ! -x "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "[type-maintenance] python3 not found" >&2
    exit 1
  fi
fi

case "$MODE" in
  check) APPLY=0 ;;
  apply|fix|optimize) APPLY=1 ;;
  *)
    echo "[type-maintenance] invalid TYPE_MAINTENANCE_MODE=$MODE (must be check|apply|fix|optimize)" >&2
    exit 1
    ;;
esac

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"

if ! "$PYTHON" -m ruff --version >/dev/null 2>&1 || ! "$PYTHON" -m mypy --version >/dev/null 2>&1; then
  echo "[type-maintenance] missing type tools. install from $DEPS_FILE"
  "$PYTHON" -m pip install -q -r "$DEPS_FILE"
fi

if [[ "$APPLY" -eq 1 ]]; then
  echo "[type-maintenance] mode=apply (add annotations automatically)"
  "$PYTHON" scripts/type_maintenance.py --apply
else
  echo "[type-maintenance] mode=check"
  "$PYTHON" scripts/type_maintenance.py
fi
STATUS=$?

if [[ "$APPLY" -eq 1 ]]; then
  if ! git diff --quiet; then
    TS="$(date +%Y%m%d_%H%M%S)"
    git diff > "$LOG_ROOT/type_maintenance_${TS}.patch"
    git status --short > "$LOG_ROOT/type_maintenance_${TS}_status.txt"
    echo "[type-maintenance] changes: $LOG_ROOT/type_maintenance_${TS}.patch"
  else
    echo "[type-maintenance] no changes were generated"
  fi
fi

exit "$STATUS"
