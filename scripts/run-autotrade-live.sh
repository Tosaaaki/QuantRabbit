#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="${QR_TRADER_ROOT_DIR:-/Users/tossaki/App/QuantRabbit-live}"
cd "$ROOT_DIR"

export PYTHONPATH="src"
export QR_OANDA_ENV_FILE="${QR_OANDA_ENV_FILE:-.env.local}"
load_live_enabled_from_env_file() {
  if [[ -n "${QR_LIVE_ENABLED:-}" || ! -f "$QR_OANDA_ENV_FILE" ]]; then
    return 0
  fi

  local line value
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?QR_LIVE_ENABLED[[:space:]]*=" "$QR_OANDA_ENV_FILE" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return 0
  fi

  value="${line#*=}"
  value="${value%%#*}"
  value="$(printf '%s' "$value" | tr -d "[:space:]\"'")"
  case "$value" in
    0|1)
      export QR_LIVE_ENABLED="$value"
      ;;
    *)
      echo "[run-autotrade-live] invalid QR_LIVE_ENABLED in ${QR_OANDA_ENV_FILE}; expected 0 or 1." >&2
      exit 2
      ;;
  esac
}

load_live_enabled_from_env_file
export QR_LIVE_ENABLED="${QR_LIVE_ENABLED:-0}"

# SL-free strategy knobs (`feedback_no_tight_sl_thin_market.md`,
# `project_sl_free_strategy.md`, 2026-05-07 user directive 「SLいらない」).
# Defaults widen SL beyond M5 noise so routine wicks cannot stop us out, and
# disable position-manager SL-repair on trader-owned positions whose SL was
# deliberately removed. Override per shell if you need to revert.
export QR_GEOMETRY_ATR_MULT="${QR_GEOMETRY_ATR_MULT:-5.0}"
export QR_GEOMETRY_SPREAD_FLOOR_MULT="${QR_GEOMETRY_SPREAD_FLOOR_MULT:-12.0}"
export QR_TRADER_DISABLE_SL_REPAIR="${QR_TRADER_DISABLE_SL_REPAIR:-1}"
# Concurrent trader-owned positions cap. Default 4 in code; live runs a
# wider portfolio so 3-pair-simultaneous attack (`feedback_attack_mode_sizing.md`)
# fits comfortably with margin headroom.
export QR_MAX_PORTFOLIO_POSITIONS="${QR_MAX_PORTFOLIO_POSITIONS:-10}"
# NAV-pct sizing: each new position locks % of current NAV as margin, so
# unit count auto-scales with equity (user 2026-05-08「BaseUnitを決めると、
# 資産が増えたときに追従できないよ。％で決めないといけなくない？」). 30%
# per position lands ≈10000u for EUR_USD at NAV 227k — three concurrent
# positions reach ~90% margin utilization, just inside the 92% cap.
# Override per shell to dial conservative (15-20) or all-in (40-50).
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
# Legacy fixed-units fallback used only when QR_TRADER_POSITION_NAV_PCT
# is unset. Kept for backward compat with smoke scripts that pin units.
export QR_TRADER_BASE_UNITS="${QR_TRADER_BASE_UNITS:-3000}"
# Deterministic REVIEW_EXIT is advisory by default in SL-free live mode.
# Legacy/no-ledger closes still need gpt_trader Gate A/B; next-generation
# trader entries with an entry thesis ledger may execute only hard structural
# loss-cut exits.
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
# Fresh live entries require a current executable pair forecast. Campaign
# pressure cannot turn a stale/no-forecast lane into a broker-fillable order.
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
# The forecast must also be auditable: current forecast_history row, projection
# ledger calibration target, and OANDA execution ledger sync all gate LIVE_READY.
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"

readonly QR_AUTOTRADE_LOCK_DIR="${QR_AUTOTRADE_LOCK_DIR:-${ROOT_DIR}/.quant_rabbit_live.lock}"
readonly QR_LIVE_SYNC_ENABLED="${QR_LIVE_SYNC_ENABLED:-1}"
readonly DEFAULT_SYNC_DEV_ROOT="/Users/tossaki/App/QuantRabbit"
readonly DEFAULT_SYNC_MAIN_BRANCH="main"

acquire_lock() {
  if mkdir "$QR_AUTOTRADE_LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "${QR_AUTOTRADE_LOCK_DIR}/pid"
    export QR_AUTOTRADE_LOCK_HELD=1
    trap 'rm -rf "$QR_AUTOTRADE_LOCK_DIR"' EXIT INT TERM
    return 0
  fi

  local existing_pid=""
  if [[ -f "${QR_AUTOTRADE_LOCK_DIR}/pid" ]]; then
    existing_pid="$(cat "${QR_AUTOTRADE_LOCK_DIR}/pid" 2>/dev/null || true)"
  fi
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[run-autotrade-live] another autotrade cycle is already running pid=${existing_pid}; refusing overlap." >&2
    exit 75
  fi

  echo "[run-autotrade-live] removing stale lock: ${QR_AUTOTRADE_LOCK_DIR}" >&2
  rm -rf "$QR_AUTOTRADE_LOCK_DIR"
  if mkdir "$QR_AUTOTRADE_LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "${QR_AUTOTRADE_LOCK_DIR}/pid"
    export QR_AUTOTRADE_LOCK_HELD=1
    trap 'rm -rf "$QR_AUTOTRADE_LOCK_DIR"' EXIT INT TERM
    return 0
  fi

  echo "[run-autotrade-live] failed to acquire autotrade lock: ${QR_AUTOTRADE_LOCK_DIR}" >&2
  exit 75
}

is_report_path() {
  local path="$1"
  [[ "$path" == docs/*_report.md || "$path" == docs/*_report.close_reentry.md ]]
}

clear_runtime_verdict_markers() {
  local path file
  for path in EXTEND HOLD REVIEW_CLOSE RECOMMEND_CLOSE STILL_VALID WEAKENED BROKEN; do
    file="${ROOT_DIR}/${path}"
    if [[ -f "$file" && ! -s "$file" ]]; then
      rm -f "$file"
      echo "[run-autotrade-live] removed empty verdict marker: ${path}" >&2
    fi
  done
}

can_continue_after_sync_failure() {
  if [[ "${QR_LIVE_SYNC_CONTINUE_IF_CURRENT:-1}" != "1" ]]; then
    return 1
  fi

  local dev_root main_branch live_head main_head line path lock_rel
  dev_root="${QR_SYNC_DEV_ROOT:-$DEFAULT_SYNC_DEV_ROOT}"
  main_branch="${QR_SYNC_MAIN_BRANCH:-$DEFAULT_SYNC_MAIN_BRANCH}"
  lock_rel="${QR_AUTOTRADE_LOCK_DIR#$ROOT_DIR/}"
  live_head="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null)" || return 1
  main_head="$(git -C "$dev_root" rev-parse "$main_branch" 2>/dev/null)" || return 1
  if [[ "$live_head" != "$main_head" ]]; then
    return 1
  fi

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line:3}"
    if [[ "$lock_rel" != "$QR_AUTOTRADE_LOCK_DIR" && ( "$path" == "$lock_rel" || "$path" == "$lock_rel/"* ) ]]; then
      continue
    fi
    if ! is_report_path "$path"; then
      echo "[run-autotrade-live] live sync failed and runtime has non-report drift: ${line}" >&2
      return 1
    fi
  done < <(git -C "$ROOT_DIR" status --short --untracked-files=all 2>/dev/null) || return 1
  return 0
}

acquire_lock
clear_runtime_verdict_markers

if [[ "$QR_LIVE_SYNC_ENABLED" == "1" && -x "${ROOT_DIR}/scripts/sync-live-runtime.sh" ]]; then
  set +e
  "${ROOT_DIR}/scripts/sync-live-runtime.sh" --live-only --skip-tests
  sync_status="$?"
  set -e
  if [[ "$sync_status" -ne 0 ]]; then
    if can_continue_after_sync_failure; then
      echo "[run-autotrade-live] live sync failed with status=${sync_status}, but runtime HEAD matches main and only docs/*_report.md drift is present; continuing this trader cycle." >&2
    else
      exit "$sync_status"
    fi
  fi
fi

if [[ ! -f "$QR_OANDA_ENV_FILE" ]]; then
  echo "[run-autotrade-live] missing OANDA env file: $QR_OANDA_ENV_FILE" >&2
  exit 2
fi

if [[ "$QR_LIVE_ENABLED" != "1" ]]; then
  echo "[run-autotrade-live] QR_LIVE_ENABLED=$QR_LIVE_ENABLED; forcing dry-run mode." >&2
fi

missing_keys=0
for required_key in QR_OANDA_ACCOUNT_ID QR_OANDA_TOKEN QR_OANDA_BASE_URL; do
  if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${required_key}[[:space:]]*=" "$QR_OANDA_ENV_FILE"; then
    echo "[run-autotrade-live] missing ${required_key} in ${QR_OANDA_ENV_FILE}" >&2
    missing_keys=1
  fi
done

if [[ "$missing_keys" -ne 0 ]]; then
  echo "[run-autotrade-live] OANDA env file is invalid for live operations." >&2
  exit 2
fi

readonly LIVE_ARG="$*"
echo "[run-autotrade-live] running from ${ROOT_DIR} with env_file=${QR_OANDA_ENV_FILE}, QR_LIVE_ENABLED=${QR_LIVE_ENABLED}, args=${LIVE_ARG:-<none>}" >&2

declare -a args
args=()
if [[ "$#" -gt 0 ]]; then
  args=("$@")
fi
arg_count="$#"
has_arg() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" || "$item" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$arg_count" -gt 0 ]] && has_arg "--send" "${args[@]}" && ! has_arg "--use-gpt-trader" "${args[@]}"; then
  gpt_args=("--use-gpt-trader")
  if ! has_arg "--reuse-market-artifacts" "${args[@]}"; then
    gpt_args=("--reuse-market-artifacts" "${gpt_args[@]}")
  fi
  if ! has_arg "--gpt-decision-response" "${args[@]}"; then
    gpt_args=("${gpt_args[@]}" "--gpt-decision-response" "data/codex_trader_decision_response.json")
  fi
  args=("${gpt_args[@]}" "${args[@]}")
  arg_count="${#args[@]}"
  echo "[run-autotrade-live] live send requires trader decision handoff; using args=${args[*]}" >&2
fi

if [[ "$QR_LIVE_ENABLED" == "1" && "$arg_count" -gt 0 ]] \
  && has_arg "--use-gpt-trader" "${args[@]}" \
  && has_arg "--gpt-decision-response" "${args[@]}" \
  && ! has_arg "--send" "${args[@]}"; then
  if [[ "${QR_ALLOW_LIVE_STAGE_ONLY:-0}" == "1" ]]; then
    echo "[run-autotrade-live] QR_ALLOW_LIVE_STAGE_ONLY=1; keeping GPT handoff stage-only." >&2
  else
    args=("${args[@]}" "--send")
    arg_count="${#args[@]}"
    echo "[run-autotrade-live] QR_LIVE_ENABLED=1 GPT handoff omitted --send; adding --send to avoid a stage-only live trader cycle." >&2
  fi
fi

set +e
if [[ "$arg_count" -gt 0 ]]; then
  python3 -m quant_rabbit.cli autotrade-cycle "${args[@]}"
else
  python3 -m quant_rabbit.cli autotrade-cycle
fi
cycle_exit="$?"
set -e

# Slack notifications are opt-in. User directive 2026-05-30:
# 「Slackに送らないで」. Each notifier is still idempotent if explicitly
# enabled, but the live trader must not post to Slack by default.
if [[ "${QR_SLACK_NOTIFY_ENABLE:-0}" == "1" && "${QR_SLACK_NOTIFY_DISABLE:-0}" != "1" ]]; then
  if [[ -x "$(command -v python3)" && -f "${ROOT_DIR}/tools/slack_fill_notify.py" ]]; then
    python3 "${ROOT_DIR}/tools/slack_fill_notify.py" 2>&1 | sed 's/^/[slack-fill] /' || true
  fi
  if [[ -f "${ROOT_DIR}/tools/slack_target_milestone.py" ]]; then
    python3 "${ROOT_DIR}/tools/slack_target_milestone.py" 2>&1 | sed 's/^/[slack-target] /' || true
  fi
  if [[ -f "${ROOT_DIR}/tools/slack_cycle_alert.py" ]]; then
    python3 "${ROOT_DIR}/tools/slack_cycle_alert.py" 2>&1 | sed 's/^/[slack-cycle] /' || true
  fi
fi

exit "$cycle_exit"
