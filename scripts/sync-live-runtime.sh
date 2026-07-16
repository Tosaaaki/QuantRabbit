#!/usr/bin/env bash
set -euo pipefail

# Git hooks export repository-local GIT_* variables. This script creates and
# tests temporary repositories, so keep all git invocations path-scoped.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime-drift-allowlist.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/sync-live-runtime.sh [--live-only] [--skip-tests]

Promote committed QuantRabbit code into the runtime path:
  source branch -> main -> codex/live-trader-runtime worktree

Defaults:
  source branch: current branch in /Users/tossaki/App/QuantRabbit
  main branch:   main
  live branch:   codex/live-trader-runtime
  live worktree: /Users/tossaki/App/QuantRabbit-live

Environment overrides:
  QR_SYNC_DEV_ROOT
  QR_SYNC_LIVE_ROOT
  QR_SYNC_SOURCE_BRANCH
  QR_SYNC_MAIN_BRANCH
  QR_SYNC_LIVE_BRANCH
  QR_SYNC_AUTOMATION_FILE
  QR_WEEKEND_TASK_STATE_FILE
  QR_SYNC_SKIP_AUTOMATION_CHECK=1
USAGE
}

readonly DEFAULT_DEV_ROOT="/Users/tossaki/App/QuantRabbit"
readonly DEFAULT_LIVE_ROOT="/Users/tossaki/App/QuantRabbit-live"
readonly DEFAULT_MAIN_BRANCH="main"
readonly DEFAULT_LIVE_BRANCH="codex/live-trader-runtime"
if [[ -z "${QR_PYTHON:-}" ]]; then
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    QR_PYTHON="/opt/homebrew/bin/python3"
  else
    QR_PYTHON="/usr/bin/python3"
  fi
fi
readonly QR_PYTHON
readonly DEFAULT_AUTOMATION_FILE="/Users/tossaki/.codex/automations/qr-trader/automation.toml"
readonly DEFAULT_WEEKEND_TASK_STATE_FILE="/Users/tossaki/.codex/quant_rabbit_weekend_task_state.json"

LIVE_ONLY=0
SKIP_TESTS=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --live-only)
      LIVE_ONLY=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[sync-live-runtime] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DEV_ROOT="${QR_SYNC_DEV_ROOT:-$DEFAULT_DEV_ROOT}"
LIVE_ROOT="${QR_SYNC_LIVE_ROOT:-$DEFAULT_LIVE_ROOT}"
MAIN_BRANCH="${QR_SYNC_MAIN_BRANCH:-$DEFAULT_MAIN_BRANCH}"
LIVE_BRANCH="${QR_SYNC_LIVE_BRANCH:-$DEFAULT_LIVE_BRANCH}"
AUTOMATION_FILE="${QR_SYNC_AUTOMATION_FILE:-$DEFAULT_AUTOMATION_FILE}"
WEEKEND_TASK_STATE_FILE="${QR_WEEKEND_TASK_STATE_FILE:-$DEFAULT_WEEKEND_TASK_STATE_FILE}"

resolve_git_common_dir() {
  local root="$1"
  local label="$2"
  local common_dir inside_work_tree

  if [[ ! -d "$root" ]]; then
    echo "[sync-live-runtime] missing $label: $root" >&2
    return 2
  fi
  if ! inside_work_tree="$(git -C "$root" rev-parse --is-inside-work-tree 2>/dev/null)"; then
    echo "[sync-live-runtime] missing $label: $root" >&2
    return 2
  fi
  if [[ "$inside_work_tree" != "true" ]]; then
    echo "[sync-live-runtime] missing $label: $root" >&2
    return 2
  fi
  if ! common_dir="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    echo "[sync-live-runtime] cannot resolve git common directory for $label: $root" >&2
    return 2
  fi
  if [[ ! -d "$common_dir" ]]; then
    echo "[sync-live-runtime] missing git common directory for $label: $common_dir" >&2
    return 2
  fi
  (cd "$common_dir" && pwd -P)
}

if [[ ! -x "$QR_PYTHON" ]]; then
  echo "[sync-live-runtime] QR_PYTHON is not executable: $QR_PYTHON" >&2
  exit 2
fi

DEV_GIT_COMMON_DIR="$(resolve_git_common_dir "$DEV_ROOT" "development git repo")" || exit "$?"
LIVE_GIT_COMMON_DIR="$(resolve_git_common_dir "$LIVE_ROOT" "live worktree")" || exit "$?"
readonly DEV_GIT_COMMON_DIR LIVE_GIT_COMMON_DIR
if [[ "$DEV_GIT_COMMON_DIR" != "$LIVE_GIT_COMMON_DIR" ]]; then
  echo "[sync-live-runtime] development and live worktrees do not share the same git common directory." >&2
  echo "[sync-live-runtime] development common dir: $DEV_GIT_COMMON_DIR" >&2
  echo "[sync-live-runtime] live common dir: $LIVE_GIT_COMMON_DIR" >&2
  exit 2
fi

SOURCE_BRANCH="${QR_SYNC_SOURCE_BRANCH:-}"
if [[ -z "$SOURCE_BRANCH" ]]; then
  SOURCE_BRANCH="$(git -C "$DEV_ROOT" branch --show-current)"
fi

if [[ "$LIVE_ONLY" -eq 0 && -z "$SOURCE_BRANCH" ]]; then
  echo "[sync-live-runtime] source branch is detached; set QR_SYNC_SOURCE_BRANCH explicitly." >&2
  exit 2
fi

status_lines() {
  git -C "$1" status --short --untracked-files=all
}

status_path() {
  local line="$1"
  printf '%s' "${line:3}"
}

clear_runtime_verdict_markers() {
  local root="$1"
  local label="$2"
  local path file
  for path in EXTEND HOLD REVIEW_CLOSE RECOMMEND_CLOSE STILL_VALID WEAKENED BROKEN; do
    file="$root/$path"
    if [[ -f "$file" && ! -s "$file" ]]; then
      rm -f "$file"
      echo "[sync-live-runtime] removed empty $label verdict marker: $path" >&2
    fi
  done
}

assert_only_report_drift() {
  local root="$1"
  local label="$2"
  local dirty=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local path
    path="$(status_path "$line")"
    if ! qr_is_runtime_drift_path "$path"; then
      echo "[sync-live-runtime] blocking dirty $label path: $line" >&2
      dirty=1
    fi
  done < <(status_lines "$root")
  if [[ "$dirty" -ne 0 ]]; then
    echo "[sync-live-runtime] $label must be clean except report/action-review/guardian-contract/receipt/proof-evidence runtime drift." >&2
    exit 3
  fi
}

clear_report_drift() {
  local root="$1"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local status path
    status="${line:0:2}"
    path="$(status_path "$line")"
    if ! qr_is_runtime_drift_path "$path"; then
      continue
    fi
    if [[ "$status" == "??" ]]; then
      rm -f "$root/$path"
    else
      git -C "$root" restore -- "$path"
    fi
  done < <(status_lines "$root")
}

backup_report_drift() {
  local root="$1"
  local backup_root="$2"
  local paths_file="${backup_root}/.paths"
  : > "$paths_file"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local path
    path="$(status_path "$line")"
    if ! qr_is_runtime_drift_path "$path"; then
      continue
    fi
    if [[ -f "$root/$path" ]]; then
      mkdir -p "$backup_root/$(dirname "$path")"
      cp -p "$root/$path" "$backup_root/$path"
      printf '%s\n' "$path" >> "$paths_file"
    fi
  done < <(status_lines "$root")
}

restore_report_drift() {
  local root="$1"
  local backup_root="$2"
  local paths_file="${backup_root}/.paths"
  [[ -f "$paths_file" ]] || return 0
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    [[ -f "$backup_root/$path" ]] || continue
    mkdir -p "$root/$(dirname "$path")"
    cp -p "$backup_root/$path" "$root/$path"
  done < "$paths_file"
}

require_branch() {
  local branch="$1"
  if ! git -C "$DEV_ROOT" rev-parse --verify --quiet "$branch" >/dev/null; then
    echo "[sync-live-runtime] missing branch: $branch" >&2
    exit 2
  fi
}

fast_forward_ref() {
  local branch="$1"
  local target="$2"
  local current target_sha
  current="$(git -C "$DEV_ROOT" rev-parse "$branch")"
  target_sha="$(git -C "$DEV_ROOT" rev-parse "$target")"
  if [[ "$current" == "$target_sha" ]]; then
    echo "[sync-live-runtime] $branch already at $target_sha"
    return 0
  fi
  if ! git -C "$DEV_ROOT" merge-base --is-ancestor "$branch" "$target"; then
    echo "[sync-live-runtime] refusing non-fast-forward: $branch -> $target" >&2
    exit 4
  fi
  git -C "$DEV_ROOT" update-ref "refs/heads/$branch" "$target_sha" "$current"
  echo "[sync-live-runtime] fast-forwarded $branch to $target_sha"
}

sync_live_worktree() {
  local target_branch="$1"
  local live_current target_sha
  local live_checked_out
  live_checked_out="$(git -C "$LIVE_ROOT" branch --show-current)"
  if [[ "$live_checked_out" != "$LIVE_BRANCH" ]]; then
    echo "[sync-live-runtime] live worktree is on $live_checked_out, expected $LIVE_BRANCH" >&2
    exit 5
  fi

  assert_only_report_drift "$LIVE_ROOT" "live"

  live_current="$(git -C "$LIVE_ROOT" rev-parse HEAD)"
  target_sha="$(git -C "$DEV_ROOT" rev-parse "$target_branch")"
  if [[ "$live_current" == "$target_sha" ]]; then
    echo "[sync-live-runtime] live worktree already at $target_sha; preserving runtime report drift"
    assert_only_report_drift "$LIVE_ROOT" "live"
    return 0
  fi
  if ! git -C "$DEV_ROOT" merge-base --is-ancestor "$LIVE_BRANCH" "$target_branch"; then
    echo "[sync-live-runtime] refusing non-fast-forward: $LIVE_BRANCH -> $target_branch" >&2
    exit 4
  fi

  local report_backup
  report_backup="$(mktemp -d)"
  backup_report_drift "$LIVE_ROOT" "$report_backup"
  clear_report_drift "$LIVE_ROOT"
  if ! git -C "$LIVE_ROOT" merge --ff-only "$target_branch"; then
    local merge_status="$?"
    restore_report_drift "$LIVE_ROOT" "$report_backup"
    rm -rf "$report_backup"
    exit "$merge_status"
  fi
  restore_report_drift "$LIVE_ROOT" "$report_backup"
  rm -rf "$report_backup"
  assert_only_report_drift "$LIVE_ROOT" "live"
}

assert_live_target() {
  local expected_sha="$1"
  local live_head live_branch_head
  live_head="$(git -C "$LIVE_ROOT" rev-parse HEAD)"
  live_branch_head="$(git -C "$LIVE_ROOT" rev-parse "refs/heads/$LIVE_BRANCH")"
  if [[ "$live_head" != "$expected_sha" || "$live_branch_head" != "$expected_sha" ]]; then
    echo "[sync-live-runtime] live sync mismatch: expected=$expected_sha HEAD=$live_head $LIVE_BRANCH=$live_branch_head" >&2
    exit 7
  fi
}

verify_automation() {
  if [[ "${QR_SYNC_SKIP_AUTOMATION_CHECK:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$AUTOMATION_FILE" ]]; then
    echo "[sync-live-runtime] automation file not found: $AUTOMATION_FILE" >&2
    exit 6
  fi
  if ! grep -Fq "cwds = [\"$LIVE_ROOT\"]" "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR AI Supervisor automation does not point at $LIVE_ROOT." >&2
    exit 6
  fi
  if ! grep -Fq 'id = "qr-trader"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR AI Supervisor must retain the qr-trader compatibility id." >&2
    exit 6
  fi
  if ! grep -Fq 'name = "QR AI Supervisor"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] qr-trader compatibility automation name must be QR AI Supervisor." >&2
    exit 6
  fi
  if ! grep -Fq 'rrule = "FREQ=MINUTELY;INTERVAL=360;BYDAY=SU,MO,TU,WE,TH,FR,SA"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR AI Supervisor cadence must be 360 minutes; deterministic market monitoring remains continuous outside the AI schedule." >&2
    exit 6
  fi
  if ! grep -Fq 'model = "gpt-5.5"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR AI Supervisor automation model must be gpt-5.5." >&2
    exit 6
  fi
  if ! grep -Fq 'reasoning_effort = "high"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR AI Supervisor automation reasoning_effort must be high." >&2
    exit 6
  fi
  for required in \
    'AI_ORDER_AUTHORITY=NONE' \
    'REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY' \
    'data/ai_regime_supervision.json' \
    'tools/ai_regime_supervision.py' \
    'GO CAUTION STOP' \
    'data/guardian_tuning_work_order.json'
  do
    if ! grep -Fq "$required" "$AUTOMATION_FILE"; then
      echo "[sync-live-runtime] QR AI Supervisor prompt is stale; missing: $required" >&2
      exit 6
    fi
  done
  for forbidden in \
    'run-autotrade-live.sh' \
    '--send' \
    'QR_LIVE_ENABLED=1' \
    'QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ=1'
  do
    if grep -Fq -- "$forbidden" "$AUTOMATION_FILE"; then
      echo "[sync-live-runtime] QR AI Supervisor prompt contains forbidden live-order token: $forbidden" >&2
      exit 6
    fi
  done
  if grep -Fq 'status = "ACTIVE"' "$AUTOMATION_FILE"; then
    return 0
  fi
  if grep -Fq 'status = "PAUSED"' "$AUTOMATION_FILE" && weekend_guard_paused; then
    echo "[sync-live-runtime] QR AI Supervisor automation is PAUSED by weekend task guard." >&2
    return 0
  fi
  echo "[sync-live-runtime] QR AI Supervisor automation is not ACTIVE." >&2
  exit 6
}

weekend_guard_paused() {
  [[ -f "$WEEKEND_TASK_STATE_FILE" ]] || return 1
  "$QR_PYTHON" - "$WEEKEND_TASK_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    sys.exit(1)
tasks = payload.get("tasks")
if payload.get("mode") == "paused" and isinstance(tasks, dict) and "codex:qr-trader" in tasks:
    sys.exit(0)
sys.exit(1)
PY
}

copy_local_runtime_files() {
  if [[ ! -f "$LIVE_ROOT/.env.local" && -f "$DEV_ROOT/.env.local" ]]; then
    cp -p "$DEV_ROOT/.env.local" "$LIVE_ROOT/.env.local"
    echo "[sync-live-runtime] copied .env.local to live worktree without printing values."
  fi
  mkdir -p "$LIVE_ROOT/data" "$LIVE_ROOT/logs"
}

require_branch "$MAIN_BRANCH"
require_branch "$LIVE_BRANCH"
clear_runtime_verdict_markers "$LIVE_ROOT" "live"
assert_only_report_drift "$LIVE_ROOT" "live"
if [[ "$LIVE_ONLY" -eq 0 ]]; then
  require_branch "$SOURCE_BRANCH"
  assert_only_report_drift "$DEV_ROOT" "development"
  if [[ "$SKIP_TESTS" -eq 0 ]]; then
    echo "[sync-live-runtime] running unit tests before promotion."
    (cd "$DEV_ROOT" && PYTHONPATH=src "$QR_PYTHON" -m unittest discover -s tests -v)
  fi
  fast_forward_ref "$MAIN_BRANCH" "$SOURCE_BRANCH"
else
  echo "[sync-live-runtime] live-only mode: syncing runtime branch from $MAIN_BRANCH."
fi

copy_local_runtime_files
sync_live_worktree "$MAIN_BRANCH"
verify_automation
assert_live_target "$(git -C "$DEV_ROOT" rev-parse "$MAIN_BRANCH")"

echo "[sync-live-runtime] OK: source=${SOURCE_BRANCH:-<live-only>} main=$(git -C "$DEV_ROOT" rev-parse --short "$MAIN_BRANCH") live=$(git -C "$LIVE_ROOT" rev-parse --short HEAD)"
