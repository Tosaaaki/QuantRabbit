#!/usr/bin/env bash
# Install (or reinstall) the QuantRabbit position guardian as a launchd agent.
# It runs the live position-only guard frequently so open trades can be managed
# without waiting for a full new-entry trader cycle.

set -euo pipefail

LIVE_ROOT="${QR_SYNC_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
DEV_ROOT="${QR_SYNC_DEV_ROOT:-/Users/tossaki/App/QuantRabbit}"
MAIN_BRANCH="${QR_SYNC_MAIN_BRANCH:-main}"
SCRIPT="$LIVE_ROOT/scripts/run-position-guardian-live.sh"
LABEL="com.quantrabbit.position-guardian"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
STDOUT_LOG="$LIVE_ROOT/logs/position_guardian.launchd.log"
STDERR_LOG="$LIVE_ROOT/logs/position_guardian.launchd.err"
INTERVAL="${QR_POSITION_GUARDIAN_INTERVAL:-30}"
ENV_FILE="${QR_OANDA_ENV_FILE:-.env.local}"
HEARTBEAT_MAX_AGE_SECONDS="${QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS:-$((INTERVAL * 4))}"
REQUIRE_HEARTBEAT="${QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT:-1}"
CHECK_ONLY=0
STATUS_ONLY=0
REQUIRE_LOADED=0

usage() {
  cat <<'USAGE'
Usage: scripts/install-position-guardian.sh [--check] [--status] [--require-loaded]

Install the QuantRabbit position guardian launchd agent. --check runs the same
activation preflight without writing a plist or loading launchd.

--status prints the current plist/launchd state without writing anything.
--require-loaded exits non-zero unless the LaunchAgent plist exists and the
launchd label is currently loaded and a recent guardian heartbeat exists.
Combine it with --check when activation preflight should also be enforced.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --status)
      STATUS_ONLY=1
      shift
      ;;
    --require-loaded)
      REQUIRE_LOADED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install-position-guardian] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

die() {
  echo "[install-position-guardian] $1" >&2
  exit "${2:-2}"
}

status_path() {
  local line="$1"
  printf '%s' "${line:3}"
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
data/predictive_scout_forward_proof.json|\
data/trader_goal_loop_orchestrator.json)
      return 0
      ;;
  esac
  return 1
}

assert_only_report_drift() {
  local dirty=0 line path
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="$(status_path "$line")"
    if ! is_report_path "$path"; then
      echo "[install-position-guardian] blocking dirty live path: $line" >&2
      dirty=1
    fi
  done < <(git -C "$LIVE_ROOT" status --short --untracked-files=all)
  if [[ "$dirty" -ne 0 ]]; then
    die "live worktree must be clean except report/action-review/guardian-contract/receipt/proof-evidence runtime drift." 3
  fi
}

env_file_path() {
  if [[ "$ENV_FILE" = /* ]]; then
    printf '%s\n' "$ENV_FILE"
  else
    printf '%s/%s\n' "$LIVE_ROOT" "$ENV_FILE"
  fi
}

validate_env_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing OANDA env file: $path" 2
  local required_key line value
  for required_key in QR_OANDA_ACCOUNT_ID QR_OANDA_TOKEN QR_OANDA_BASE_URL; do
    if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${required_key}[[:space:]]*=" "$path"; then
      die "missing ${required_key} in $path" 2
    fi
  done
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?QR_LIVE_ENABLED[[:space:]]*=" "$path" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    value="${line#*=}"
    value="${value%%#*}"
    value="$(printf '%s' "$value" | tr -d "[:space:]\"'")"
    case "$value" in
      0|1) ;;
      *) die "invalid QR_LIVE_ENABLED in $path; expected 0 or 1." 2 ;;
    esac
  fi
}

launchd_label_loaded() {
  command -v launchctl >/dev/null 2>&1 || return 2
  launchctl list "$LABEL" >/dev/null 2>&1 && return 0
  launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && return 0
  return 1
}

guardian_loaded_status() {
  if ! command -v launchctl >/dev/null 2>&1; then
    printf 'launchctl_unavailable'
    return 0
  fi
  if launchd_label_loaded; then
    printf 'loaded'
  else
    printf 'not_loaded'
  fi
}

print_guardian_status() {
  local plist_state loaded_state
  plist_state="missing"
  [[ -f "$PLIST" ]] && plist_state="present"
  loaded_state="$(guardian_loaded_status)"
  echo "[install-position-guardian] status: label=$LABEL plist=$plist_state launchd=$loaded_state live_root=$LIVE_ROOT interval=${INTERVAL}s heartbeat_max_age=${HEARTBEAT_MAX_AGE_SECONDS}s plist_path=$PLIST"
}

require_recent_guardian_heartbeat() {
  case "$REQUIRE_HEARTBEAT" in
    0|false|FALSE|no|NO)
      echo "[install-position-guardian] heartbeat check skipped by QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT=${REQUIRE_HEARTBEAT}"
      return 0
      ;;
    1|true|TRUE|yes|YES) ;;
    *)
      die "invalid QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT=${REQUIRE_HEARTBEAT}; expected 0 or 1." 2
      ;;
  esac
  [[ "$HEARTBEAT_MAX_AGE_SECONDS" =~ ^[0-9]+$ ]] || die "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS must be an integer >= ${INTERVAL} seconds." 2
  [[ "$HEARTBEAT_MAX_AGE_SECONDS" -ge "$INTERVAL" ]] || die "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS must be an integer >= ${INTERVAL} seconds." 2
  local py
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    py="/opt/homebrew/bin/python3"
  else
    py="$(command -v python3 || true)"
  fi
  [[ -n "$py" ]] || die "python3 is required to validate position guardian heartbeat freshness." 2
  "$py" - "$HEARTBEAT_MAX_AGE_SECONDS" \
    "$LIVE_ROOT/data/position_guardian_execution.json" \
    "$LIVE_ROOT/data/position_guardian.json" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

max_age = int(sys.argv[1])
paths = [Path(item) for item in sys.argv[2:]]
now = datetime.now(timezone.utc)
seen = []

def parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

for path in paths:
    if not path.exists():
        seen.append(f"{path}: missing")
        continue
    generated = None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        generated = parse_ts(payload.get("generated_at_utc"))
    if generated is not None:
        age = (now - generated).total_seconds()
        source = "generated_at_utc"
    else:
        age = time.time() - path.stat().st_mtime
        source = "mtime"
    seen.append(f"{path}: age={age:.1f}s source={source}")
    if -60.0 <= age <= max_age:
        print(f"[install-position-guardian] heartbeat OK: path={path} age={age:.1f}s source={source} max_age={max_age}s")
        raise SystemExit(0)

print(
    "[install-position-guardian] position guardian heartbeat is missing or stale; "
    "fresh entry sends remain blocked so TP-progress profit capture is not blind. "
    + "; ".join(seen),
    file=sys.stderr,
)
raise SystemExit(6)
PY
}

require_guardian_loaded() {
  [[ -f "$PLIST" ]] || die "position guardian plist is missing: $PLIST" 6
  command -v launchctl >/dev/null 2>&1 || die "launchctl is not available; position guardian cannot be proven active." 6
  if ! launchd_label_loaded; then
    die "position guardian launchd label is not loaded: $LABEL" 6
  fi
  require_recent_guardian_heartbeat
  echo "[install-position-guardian] active OK: label=$LABEL plist=$PLIST"
}

preflight() {
  [[ "$INTERVAL" =~ ^[0-9]+$ ]] || die "QR_POSITION_GUARDIAN_INTERVAL must be an integer >= 15 seconds." 2
  [[ "$INTERVAL" -ge 15 ]] || die "QR_POSITION_GUARDIAN_INTERVAL must be an integer >= 15 seconds." 2
  [[ -x "$SCRIPT" ]] || {
    echo "[install-position-guardian] missing executable live guardian script: $SCRIPT" >&2
    die "run scripts/sync-live-runtime.sh after committing the guardian change." 2
  }
  git -C "$LIVE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "missing live git worktree: $LIVE_ROOT" 2
  git -C "$DEV_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "missing development git repo: $DEV_ROOT" 2

  local live_head main_head
  live_head="$(git -C "$LIVE_ROOT" rev-parse HEAD)"
  main_head="$(git -C "$DEV_ROOT" rev-parse "$MAIN_BRANCH")"
  if [[ "$live_head" != "$main_head" ]]; then
    die "live HEAD $live_head does not match $MAIN_BRANCH $main_head; run scripts/sync-live-runtime.sh first." 4
  fi

  assert_only_report_drift
  validate_env_file "$(env_file_path)"
}

if [[ "$CHECK_ONLY" -eq 1 || ( "$CHECK_ONLY" -eq 0 && "$STATUS_ONLY" -eq 0 && "$REQUIRE_LOADED" -eq 0 ) ]]; then
  preflight
fi

if [[ "$STATUS_ONLY" -eq 1 ]]; then
  print_guardian_status
fi

if [[ "$REQUIRE_LOADED" -eq 1 ]]; then
  require_guardian_loaded
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "[install-position-guardian] preflight OK: live_root=$LIVE_ROOT interval=${INTERVAL}s plist=$PLIST"
  exit 0
fi

if [[ "$STATUS_ONLY" -eq 1 || "$REQUIRE_LOADED" -eq 1 ]]; then
  exit 0
fi

command -v launchctl >/dev/null 2>&1 || die "launchctl is not available." 2
mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$LIVE_ROOT/logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$STDOUT_LOG</string>
  <key>StandardErrorPath</key><string>$STDERR_LOG</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed: $PLIST (every ${INTERVAL}s)"
