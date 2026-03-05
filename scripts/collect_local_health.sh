#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${HEALTH_SNAPSHOT_REPO:-$DEFAULT_REPO_DIR}"
if [[ $# -gt 0 ]] && [[ -d "$1" ]]; then
  REPO_DIR="$1"
fi
HEALTH_JSON_PATH="$REPO_DIR/logs/health_snapshot.json"
HEALTH_SNAPSHOT_STALE_SEC="${HEALTH_SNAPSHOT_STALE_SEC:-300}"
before_mtime="$(stat -f %m "$HEALTH_JSON_PATH" 2>/dev/null || echo 0)"

export HEALTH_UPLOAD_DISABLE="${HEALTH_UPLOAD_DISABLE:-1}"
bash "$SCRIPT_DIR/run_health_snapshot.sh" "$@"
after_mtime="$(stat -f %m "$HEALTH_JSON_PATH" 2>/dev/null || echo 0)"
updated="no"
if [[ "$after_mtime" != "$before_mtime" ]]; then
  updated="yes"
fi
now_sec="$(date +%s)"
age_sec=$((now_sec - after_mtime))
if (( age_sec < 0 )); then
  age_sec=0
fi
stale_warn="no"
if (( age_sec > HEALTH_SNAPSHOT_STALE_SEC )); then
  stale_warn="yes"
fi

echo "[collect-local-health] ok: ${HEALTH_JSON_PATH} updated=${updated}"
echo "[collect-local-health] snapshot_age_sec=${age_sec} stale_warn=${stale_warn} threshold_sec=${HEALTH_SNAPSHOT_STALE_SEC}"
