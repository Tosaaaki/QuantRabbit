#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${TYPE_MAINTENANCE_REPO:-/home/tossaki/QuantRabbit}"
MODE="${TYPE_MAINTENANCE_MODE:-check}" # check | apply | fix | optimize
ALLOW_FAILURE="${TYPE_MAINTENANCE_ALLOW_FAILURE:-1}" # 1: do not fail timer on type check failures
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

if [[ "$MODE" == "optimize" ]]; then
  ALLOW_FAILURE=1
fi

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"

if ! "$PYTHON" -m ruff --version >/dev/null 2>&1 || ! "$PYTHON" -m mypy --version >/dev/null 2>&1; then
  echo "[type-maintenance] missing type tools. install from $DEPS_FILE"
  "$PYTHON" -m pip install -q -r "$DEPS_FILE"
fi

if [[ "$APPLY" -eq 1 ]]; then
  echo "[type-maintenance] mode=apply (add annotations automatically)"
  set +e
  "$PYTHON" scripts/type_maintenance.py --apply
  STATUS=$?
  set -e
else
  echo "[type-maintenance] mode=check"
  set +e
  "$PYTHON" scripts/type_maintenance.py
  STATUS=$?
  set -e
fi

if [[ "$STATUS" -ne 0 && "$MODE" == "optimize" && "$ALLOW_FAILURE" == "1" ]]; then
  echo "[type-maintenance] optimize mode has issues; recorded in logs/type_audit_report.json. Non-zero status will be treated as non-fatal."
  exit 0
fi

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
