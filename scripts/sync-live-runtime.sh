#!/usr/bin/env bash
set -euo pipefail

# Git hooks export repository-local GIT_* variables. This script creates and
# tests temporary repositories, so keep all git invocations path-scoped.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX

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

if [[ ! -x "$QR_PYTHON" ]]; then
  echo "[sync-live-runtime] QR_PYTHON is not executable: $QR_PYTHON" >&2
  exit 2
fi

if [[ ! -d "$DEV_ROOT/.git" ]]; then
  echo "[sync-live-runtime] missing development git repo: $DEV_ROOT" >&2
  exit 2
fi
if [[ ! -d "$LIVE_ROOT" ]]; then
  echo "[sync-live-runtime] missing live worktree: $LIVE_ROOT" >&2
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

is_report_path() {
  local path="$1"
  case "$path" in
    docs/*_report.md|\
docs/*_report.close_reentry.md|\
docs/guardian_action_review.md|\
docs/as_lane_candidate_board.md|\
docs/as_proof_pack_queue.md|\
docs/audjpy_short_breakout_failure_limit_proof_pack.md|\
docs/audjpy_short_breakout_failure_repair_proof.md|\
docs/active_trader_contract.md|\
docs/active_opportunity_board.md|\
docs/entry_frequency_recovery_report.md|\
docs/forecast_pattern_refresh_report.md|\
docs/range_rail_geometry_repair_report.md|\
docs/eurusd_short_breakout_failure_evidence_acquisition_plan.md|\
docs/eurusd_short_breakout_failure_legacy_sample_search.md|\
docs/eurusd_short_breakout_failure_limit_s5_bidask_replay.md|\
docs/eurusd_short_breakout_failure_limit_sample_mining.md|\
docs/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md|\
docs/eurusd_short_breakout_failure_proof_floor_update.md|\
docs/eurusd_short_breakout_failure_scout_plan.md|\
docs/eurusd_short_breakout_failure_spread_slippage_proof.md|\
docs/eurusd_short_breakout_failure_stop_harvest_replay.md|\
docs/eurusd_short_breakout_failure_vehicle_split_diagnosis.md|\
docs/historical_only_to_fresh_proof_replay.md|\
docs/manual_eurusd_tp_replacement_provenance.md|\
docs/non_eurusd_live_grade_frontier.md|\
docs/non_eurusd_proof_lane_mapper.md|\
docs/operator_review_report.md|\
docs/portfolio_4x_path_planner.md|\
docs/post_gate_capture_economics_decomposition.md|\
docs/post_gate_expectancy_gap_trace.md|\
docs/post_gate_gap_family_repair_table.md|\
docs/profitability_acceptance_blocker_reconciliation.md|\
docs/remaining_profitability_p0_decomposition.md|\
docs/rolling_30d_4x_firepower_board.md|\
data/guardian_trigger_contract.json|\
data/guardian_receipt_consumption.json|\
data/guardian_receipt_operator_review.json|\
data/as_lane_candidate_board.json|\
data/as_proof_pack_queue.json|\
data/audjpy_short_breakout_failure_limit_proof_pack.json|\
data/audjpy_short_breakout_failure_repair_proof.json|\
data/active_trader_contract.json|\
data/active_opportunity_board.json|\
data/entry_frequency_recovery.json|\
data/forecast_pattern_refresh.json|\
data/range_rail_geometry_repair.json|\
data/eurusd_short_breakout_failure_evidence_acquisition_plan.json|\
data/eurusd_short_breakout_failure_legacy_sample_search.json|\
data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json|\
data/eurusd_short_breakout_failure_limit_sample_mining.json|\
data/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json|\
data/eurusd_short_breakout_failure_proof_floor_update.json|\
data/eurusd_short_breakout_failure_scout_plan.json|\
data/eurusd_short_breakout_failure_spread_slippage_proof.json|\
data/eurusd_short_breakout_failure_stop_harvest_replay.json|\
data/eurusd_short_breakout_failure_vehicle_split_diagnosis.json|\
data/harvest_live_grade_path.json|\
data/historical_only_to_fresh_proof_replay.json|\
data/manual_eurusd_tp_replacement_provenance.json|\
data/non_eurusd_live_grade_frontier.json|\
data/non_eurusd_proof_lane_mapper.json|\
data/operator_review_report.json|\
data/portfolio_4x_path_planner.json|\
data/post_gate_capture_economics_decomposition.json|\
data/post_gate_expectancy_gap_trace.json|\
data/post_gate_gap_family_repair_table.json|\
data/payoff_shape_diagnosis.json|\
data/profitability_acceptance_blocker_reconciliation.json|\
data/remaining_profitability_p0_decomposition.json|\
data/rolling_30d_4x_firepower_board.json|\
data/trader_goal_loop_orchestrator.json)
      return 0
      ;;
  esac
  return 1
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
    if ! is_report_path "$path"; then
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
    if ! is_report_path "$path"; then
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
    if ! is_report_path "$path"; then
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

verify_automation() {
  if [[ "${QR_SYNC_SKIP_AUTOMATION_CHECK:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$AUTOMATION_FILE" ]]; then
    echo "[sync-live-runtime] automation file not found: $AUTOMATION_FILE" >&2
    exit 6
  fi
  if ! grep -Fq "cwds = [\"$LIVE_ROOT\"]" "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation does not point at $LIVE_ROOT." >&2
    exit 6
  fi
  if ! grep -Fq 'rrule = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation cadence must be 60 minutes; frequent risk monitoring belongs to guardian probe/router." >&2
    exit 6
  fi
  if ! grep -Fq 'model = "gpt-5.5"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation model must be gpt-5.5." >&2
    exit 6
  fi
  if ! grep -Fq 'reasoning_effort = "high"' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation reasoning_effort must be high." >&2
    exit 6
  fi
  if grep -Fq 'After the receipt is ACCEPTED by `gpt-trader-decision`' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation has stale ACCEPTED-only gateway handoff text." >&2
    exit 6
  fi
  if grep -Fq '`data/codex_trader_decision_response.json` was written very recently by another cycle' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation has stale recent-receipt STOP text." >&2
    exit 6
  fi
  if ! grep -Fq 'Run exactly one gateway cycle after every completed `gpt-trader-decision` verification result, including REJECTED' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation is missing verifier-result gateway handoff text." >&2
    exit 6
  fi
  if ! grep -Fq 'Do **not** stop solely because `data/codex_trader_decision_response.json` was written recently' "$AUTOMATION_FILE"; then
    echo "[sync-live-runtime] QR vNext Trader automation is missing recent-receipt router handoff text." >&2
    exit 6
  fi
  for required in \
    'data/guardian_receipt_consumption.json' \
    'data/guardian_receipt_operator_review.json' \
    'named proof/acceptance evidence' \
    'data/trader_goal_loop_orchestrator.json' \
    'data/active_trader_contract.json' \
    'data/active_opportunity_board.json' \
    'docs/active_opportunity_board.md' \
    'eurusd_short_breakout_failure_*' \
    'runtime drift and **do not** block the run'
  do
    if ! grep -Fq "$required" "$AUTOMATION_FILE"; then
      echo "[sync-live-runtime] QR vNext Trader automation clean-tree runtime drift allow-list is stale; missing: $required" >&2
      exit 6
    fi
  done
  if grep -Fq 'status = "ACTIVE"' "$AUTOMATION_FILE"; then
    return 0
  fi
  if grep -Fq 'status = "PAUSED"' "$AUTOMATION_FILE" && weekend_guard_paused; then
    echo "[sync-live-runtime] QR vNext Trader automation is PAUSED by weekend task guard." >&2
    return 0
  fi
  echo "[sync-live-runtime] QR vNext Trader automation is not ACTIVE." >&2
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

echo "[sync-live-runtime] OK: source=${SOURCE_BRANCH:-<live-only>} main=$(git -C "$DEV_ROOT" rev-parse --short "$MAIN_BRANCH") live=$(git -C "$LIVE_ROOT" rev-parse --short HEAD)"
