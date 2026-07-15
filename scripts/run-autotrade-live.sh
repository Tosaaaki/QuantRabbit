#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="${QR_TRADER_ROOT_DIR:-/Users/tossaki/App/QuantRabbit-live}"
cd "$ROOT_DIR"

export PYTHONPATH="src"
if [[ -z "${QR_PYTHON:-}" ]]; then
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    QR_PYTHON="/opt/homebrew/bin/python3"
  else
    QR_PYTHON="/usr/bin/python3"
  fi
fi
readonly QR_PYTHON
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
# positions reach ~90% margin utilization, just inside the 95% cap.
# Override per shell to dial conservative (15-20) or all-in (40-50).
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
# Deterministic REVIEW_EXIT is advisory by default in SL-free live mode.
# Loss-side closes still need gpt_trader Gate A/B unless the operator
# explicitly opts into structural deterministic auto-close with
# QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1.
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
# Fresh live entries require a current executable pair forecast. Campaign
# pressure cannot turn a stale/no-forecast lane into a broker-fillable order.
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
# The forecast must also be auditable: current forecast_history row, projection
# ledger calibration target, and OANDA execution ledger sync all gate LIVE_READY.
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"
# Bounded forward evidence for reproducible forecast-failure contrarian rules.
# The environment flag is only the second half of a double gate: the canonical
# rule digest and config/predictive_scout_policy.json must also pass, and the
# gateway still enforces LIMIT/GTD/attached TP+SL, current-NAV risk sizing to
# positive integer units (including 1-999u), max-two active, thirty atomic
# broker-POST reservations/day, post-loss cooldown, and
# cumulative-negative quarantine.
export QR_PREDICTIVE_SCOUT_LIVE_ENABLED="${QR_PREDICTIVE_SCOUT_LIVE_ENABLED:-1}"
case "$QR_PREDICTIVE_SCOUT_LIVE_ENABLED" in
  0|1) ;;
  *)
    echo "[run-autotrade-live] invalid QR_PREDICTIVE_SCOUT_LIVE_ENABLED=${QR_PREDICTIVE_SCOUT_LIVE_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
# The independently locked causal-technical candidate is collected as a
# read-only forward shadow after each successful full cycle. It writes no
# order intent and the candidate contract hard-codes live_order_enabled=false.
export QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED="${QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED:-1}"
case "$QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED" in
  0|1) ;;
  *)
    echo "[run-autotrade-live] invalid QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED=${QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
# Mature shadow signals are resolved from read-only OANDA S5 bid/ask truth.
# The scorecard remains non-live even when every preregistered gate passes.
export QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED="${QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED:-1}"
case "$QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED" in
  0|1) ;;
  *)
    echo "[run-autotrade-live] invalid QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED=${QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
# Entry sends need the fast position guardian alive so TP-progress profit can be
# converted between full trader cycles. The gateway uses the status exported by
# this wrapper to block new risk while still allowing position-management closes.
export QR_REQUIRE_POSITION_GUARDIAN_ACTIVE="${QR_REQUIRE_POSITION_GUARDIAN_ACTIVE:-1}"
case "$QR_REQUIRE_POSITION_GUARDIAN_ACTIVE" in
  0|1) ;;
  *)
    echo "[run-autotrade-live] invalid QR_REQUIRE_POSITION_GUARDIAN_ACTIVE=${QR_REQUIRE_POSITION_GUARDIAN_ACTIVE}; expected 0 or 1." >&2
    exit 2
    ;;
esac
# The scheduled Codex trader authors only the bounded market-read overlay.
# When enabled, this wrapper publishes that overlay, verifies the merged
# decision, and reaches the gateway while retaining one live-runtime lock.
export QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ="${QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ:-0}"
case "$QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ" in
  0|1) ;;
  *)
    echo "[run-autotrade-live] invalid QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ=${QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ}; expected 0 or 1." >&2
    exit 2
    ;;
esac

readonly QR_AUTOTRADE_LOCK_DIR="${QR_AUTOTRADE_LOCK_DIR:-${ROOT_DIR}/.quant_rabbit_live.lock}"
readonly QR_AUTOTRADE_LOCK_WAIT_SECONDS="${QR_AUTOTRADE_LOCK_WAIT_SECONDS:-180}"
readonly QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN="${QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN:-run-position-guardian-live}"
readonly QR_AUTOTRADE_LOCK_POLL_SECONDS="${QR_AUTOTRADE_LOCK_POLL_SECONDS:-2}"
readonly QR_LIVE_SYNC_ENABLED="${QR_LIVE_SYNC_ENABLED:-1}"
readonly DEFAULT_SYNC_DEV_ROOT="/Users/tossaki/App/QuantRabbit"
readonly DEFAULT_SYNC_MAIN_BRANCH="main"

source "${SCRIPT_DIR}/qr-live-lock.sh"
source "${SCRIPT_DIR}/runtime-drift-allowlist.sh"

acquire_lock() {
  qr_live_lock_acquire \
    "$QR_AUTOTRADE_LOCK_DIR" \
    "run-autotrade-live" \
    "$QR_AUTOTRADE_LOCK_WAIT_SECONDS" \
    "$QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN" \
    "$QR_AUTOTRADE_LOCK_POLL_SECONDS"
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

  local dev_root main_branch live_head main_head line path lock_rel lock_guard_rel
  dev_root="${QR_SYNC_DEV_ROOT:-$DEFAULT_SYNC_DEV_ROOT}"
  main_branch="${QR_SYNC_MAIN_BRANCH:-$DEFAULT_SYNC_MAIN_BRANCH}"
  lock_rel="${QR_AUTOTRADE_LOCK_DIR#$ROOT_DIR/}"
  lock_guard_rel="${QR_AUTOTRADE_LOCK_DIR}.acquire.guard"
  lock_guard_rel="${lock_guard_rel#$ROOT_DIR/}"
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
    if [[ "$lock_guard_rel" != "${QR_AUTOTRADE_LOCK_DIR}.acquire.guard" && "$path" == "$lock_guard_rel" ]]; then
      continue
    fi
    if ! qr_is_runtime_drift_path "$path"; then
      echo "[run-autotrade-live] live sync failed and runtime has non-evidence drift: ${line}" >&2
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
      echo "[run-autotrade-live] live sync failed with status=${sync_status}, but runtime HEAD matches main and only report/action-review/guardian-contract/receipt/proof-evidence drift is present; continuing this trader cycle." >&2
    else
      exit "$sync_status"
    fi
  fi
fi

if [[ ! -f "$QR_OANDA_ENV_FILE" ]]; then
  echo "[run-autotrade-live] missing OANDA env file: $QR_OANDA_ENV_FILE" >&2
  exit 2
fi

if [[ ! -x "$QR_PYTHON" ]]; then
  echo "[run-autotrade-live] QR_PYTHON is not executable: $QR_PYTHON" >&2
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

arg_value() {
  local needle="$1"
  shift
  local item next_is_value=0
  for item in "$@"; do
    if [[ "$next_is_value" == "1" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
    if [[ "$item" == "$needle" ]]; then
      next_is_value=1
      continue
    fi
    if [[ "$item" == "${needle}="* ]]; then
      printf '%s\n' "${item#*=}"
      return 0
    fi
  done
  return 1
}

initialize_consolidated_cycle_id() {
  local explicit_id inherited_id generated_id response_path
  explicit_id="$({
    "$QR_PYTHON" - "${QR_CONSOLIDATED_CYCLE_ID:-}" <<'PY'
import re
import sys

value = str(sys.argv[1] or "").strip()
if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
    print(value)
PY
  } 2>/dev/null || true)"
  if [[ -n "$explicit_id" ]]; then
    export QR_CONSOLIDATED_CYCLE_ID="$explicit_id"
    export QR_CONSOLIDATED_CYCLE_LINEAGE_STATUS="CALLER_EXPLICIT"
    return 0
  fi

  inherited_id=""
  if has_arg "--reuse-market-artifacts" "${args[@]}"; then
    response_path="$(arg_value "--gpt-decision-response" "${args[@]}" || true)"
    response_path="${response_path:-data/codex_trader_decision_response.json}"
    inherited_id="$({
      "$QR_PYTHON" - \
        data/pair_charts.json \
        data/market_read_evidence_packet.json \
        "$response_path" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

pair_charts_path = Path(sys.argv[1])
packet_path = Path(sys.argv[2])
response_path = Path(sys.argv[3])
try:
    pair_bytes = pair_charts_path.read_bytes()
    pair_payload = json.loads(pair_bytes)
    packet_payload = json.loads(packet_path.read_text())
    response_payload = json.loads(response_path.read_text())
except (OSError, json.JSONDecodeError, TypeError, ValueError):
    raise SystemExit(0)
if not all(isinstance(item, dict) for item in (pair_payload, packet_payload, response_payload)):
    raise SystemExit(0)

source_metadata = packet_payload.get("source_metadata")
pair_source = (
    source_metadata.get("pair_charts")
    if isinstance(source_metadata, dict)
    else None
)
material_sources = packet_payload.get("sources")
material_pair_source = (
    material_sources.get("pair_charts")
    if isinstance(material_sources, dict)
    else None
)
expected_sha = (
    str(pair_source.get("sha256") or "").strip().lower()
    if isinstance(pair_source, dict)
    else ""
)
expected_size = pair_source.get("size_bytes") if isinstance(pair_source, dict) else None
source_path = pair_source.get("path") if isinstance(pair_source, dict) else None
actual_sha = hashlib.sha256(pair_bytes).hexdigest()
cycle_id = str(pair_payload.get("cycle_id") or "").strip()
packet_sha = str(packet_payload.get("evidence_packet_sha256") or "").strip().lower()
packet_material = {
    "schema_version": packet_payload.get("schema_version"),
    "baseline_sha256": packet_payload.get("baseline_sha256"),
    "sources": material_sources,
    "capital_allocation_board_sha256": packet_payload.get(
        "capital_allocation_board_sha256"
    ),
    "projection_calibration_evidence_sha256": packet_payload.get(
        "projection_calibration_evidence_sha256"
    ),
}
try:
    canonical_packet_bytes = json.dumps(
        packet_material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
except (TypeError, ValueError):
    raise SystemExit(0)
rebuilt_packet_sha = hashlib.sha256(canonical_packet_bytes).hexdigest()
decision = response_payload.get("decision")
if not isinstance(decision, dict):
    decision = response_payload
provenance = decision.get("decision_provenance")
bound_packet_sha = (
    str(provenance.get("evidence_packet_sha256") or "").strip().lower()
    if isinstance(provenance, dict)
    else ""
)
valid_source_path = False
if isinstance(source_path, str) and source_path.strip():
    try:
        valid_source_path = Path(source_path).resolve() == pair_charts_path.resolve()
    except OSError:
        valid_source_path = False
if (
    re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", cycle_id)
    and re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    and expected_sha == actual_sha
    and isinstance(expected_size, int)
    and not isinstance(expected_size, bool)
    and expected_size == len(pair_bytes)
    and valid_source_path
    and isinstance(material_pair_source, dict)
    and {
        key: material_pair_source.get(key)
        for key in ("path", "sha256", "size_bytes")
    }
    == {
        key: pair_source.get(key)
        for key in ("path", "sha256", "size_bytes")
    }
    and re.fullmatch(r"[0-9a-f]{64}", packet_sha)
    and rebuilt_packet_sha == packet_sha
    and bound_packet_sha == packet_sha
):
    print(cycle_id)
PY
    } 2>/dev/null || true)"
  fi
  if [[ -n "$inherited_id" ]]; then
    export QR_CONSOLIDATED_CYCLE_ID="$inherited_id"
    export QR_CONSOLIDATED_CYCLE_LINEAGE_STATUS="MARKET_READ_SHA_BOUND"
    echo "[run-autotrade-live] inherited consolidated cycle identity from the exact market-read pair-chart evidence." >&2
    return 0
  fi

  generated_id="$({
    "$QR_PYTHON" - <<'PY'
import uuid

print(uuid.uuid4().hex)
PY
  } 2>/dev/null || true)"
  if [[ -z "$generated_id" ]]; then
    echo "[run-autotrade-live] failed to initialize consolidated cycle identity." >&2
    exit 2
  fi
  export QR_CONSOLIDATED_CYCLE_ID="$generated_id"
  export QR_CONSOLIDATED_CYCLE_LINEAGE_STATUS="UNBOUND_WRAPPER"
}

json_string_value() {
  local path="$1"
  local key="$2"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  "$QR_PYTHON" - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError, ValueError):
    raise SystemExit(1)

key = sys.argv[2]
decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else payload
if key == "action":
    value = decision.get("action")
elif key == "decision_author_kind":
    provenance = decision.get("decision_provenance")
    value = provenance.get("author_kind") if isinstance(provenance, dict) else None
else:
    value = payload.get(key)
if not isinstance(value, str):
    raise SystemExit(1)
print(value)
PY
}

gpt_handoff_needs_refresh() {
  local response_path="$1"
  local dep status action author_kind autotrade_report_path
  if [[ ! -f "$response_path" ]]; then
    echo "[run-autotrade-live] GPT handoff response missing; composing a fresh receipt: ${response_path}" >&2
    return 0
  fi
  author_kind="$(json_string_value "$response_path" decision_author_kind || true)"
  if [[ "$author_kind" == "CODEX_MARKET_READ" ]]; then
    echo "[run-autotrade-live] preserving Codex-authored market-read receipt; wrapper auto-draft must not replace AI provenance, even when stale/rejected/consumed." >&2
    return 1
  fi
  for dep in data/broker_snapshot.json data/order_intents.json data/ai_attack_advice.json data/active_trader_contract.json data/active_opportunity_board.json data/non_eurusd_live_grade_frontier.json data/range_rail_geometry_repair.json; do
    if [[ -f "$dep" && "$dep" -nt "$response_path" ]]; then
      echo "[run-autotrade-live] GPT handoff response predates ${dep}; composing a fresh receipt." >&2
      return 0
    fi
  done
  if [[ -f data/gpt_trader_decision.json && data/gpt_trader_decision.json -nt "$response_path" ]]; then
    status="$(json_string_value data/gpt_trader_decision.json status || true)"
    action="$(json_string_value data/gpt_trader_decision.json action || true)"
    if [[ "$status" != "ACCEPTED" ]]; then
      echo "[run-autotrade-live] GPT handoff response was already verified as ${status:-UNKNOWN} ${action:-NO_ACTION}; composing a fresh receipt." >&2
      return 0
    fi
    case "$action" in
      TRADE)
        ;;
      WAIT|REQUEST_EVIDENCE|PROTECT|TIGHTEN_SL|CLOSE|CANCEL_PENDING)
        autotrade_report_path="$(arg_value "--report" "${args[@]}" || true)"
        autotrade_report_path="${autotrade_report_path:-docs/autotrade_cycle_report.md}"
        if [[ -f "$autotrade_report_path" && "$autotrade_report_path" -nt data/gpt_trader_decision.json ]]; then
          echo "[run-autotrade-live] GPT handoff response was already consumed as ACCEPTED ${action}; composing a fresh receipt." >&2
          return 0
        fi
        echo "[run-autotrade-live] preserving unconsumed ACCEPTED ${action} for one gateway-maintenance cycle." >&2
        ;;
      *)
        echo "[run-autotrade-live] GPT handoff response has unsupported ACCEPTED action ${action:-NO_ACTION}; composing a fresh receipt." >&2
        return 0
        ;;
    esac
  fi
  return 1
}

refresh_gpt_handoff_if_needed() {
  if [[ "${QR_LIVE_WRAPPER_AUTO_DRAFT_DECISION:-1}" != "1" ]]; then
    return 0
  fi
  if [[ "$arg_count" -le 0 ]] || ! has_arg "--use-gpt-trader" "${args[@]}" || ! has_arg "--gpt-decision-response" "${args[@]}"; then
    return 0
  fi
  local response_path
  response_path="$(arg_value "--gpt-decision-response" "${args[@]}")" || return 0
  if ! gpt_handoff_needs_refresh "$response_path"; then
    return 0
  fi
  local draft_status verify_status
  set +e
  "$QR_PYTHON" -m quant_rabbit.cli trader-draft-decision \
    --snapshot data/broker_snapshot.json \
    --output "$response_path"
  draft_status="$?"
  set -e
  if [[ "$draft_status" -ne 0 ]]; then
    echo "[run-autotrade-live] trader-draft-decision failed with status=${draft_status}; aborting before gateway handoff." >&2
    exit "$draft_status"
  fi
  set +e
  "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
    --snapshot data/broker_snapshot.json \
    --decision-response "$response_path"
  verify_status="$?"
  set -e
  if [[ "$verify_status" -ne 0 ]]; then
    echo "[run-autotrade-live] gpt-trader-decision returned status=${verify_status}; continuing to autotrade-cycle so stale/rejected receipts cannot skip position maintenance." >&2
  fi
}

finalize_codex_market_read_handoff() {
  if [[ "$QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ" != "1" ]]; then
    return 0
  fi
  if [[ "$arg_count" -le 0 ]] \
    || ! has_arg "--use-gpt-trader" "${args[@]}" \
    || ! has_arg "--gpt-decision-response" "${args[@]}"; then
    echo "[run-autotrade-live] Codex market-read finalization requires --use-gpt-trader and --gpt-decision-response." >&2
    exit 2
  fi

  local response_path apply_status verify_status
  response_path="$(arg_value "--gpt-decision-response" "${args[@]}" || true)"
  if [[ -z "$response_path" ]]; then
    echo "[run-autotrade-live] Codex market-read finalization requires a non-empty decision response path." >&2
    exit 2
  fi

  echo "[run-autotrade-live] finalizing the scheduled Codex market read under the live lock." >&2
  set +e
  "$QR_PYTHON" -m quant_rabbit.cli trader-apply-market-read \
    --baseline data/trader_decision_baseline.json \
    --packet data/market_read_evidence_packet.json \
    --overlay data/codex_market_read_overlay.json \
    --output "$response_path"
  apply_status="$?"
  set -e
  if [[ "$apply_status" -ne 0 ]]; then
    echo "[run-autotrade-live] trader-apply-market-read failed with status=${apply_status}; aborting before verifier and gateway so an older receipt cannot be reused." >&2
    exit "$apply_status"
  fi

  set +e
  "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
    --snapshot data/broker_snapshot.json \
    --decision-response "$response_path"
  verify_status="$?"
  set -e
  if [[ "$verify_status" -ne 0 ]]; then
    echo "[run-autotrade-live] gpt-trader-decision returned status=${verify_status}; continuing to autotrade-cycle under the same lock so rejection blocks new risk without skipping position maintenance." >&2
  fi
}

refresh_position_guardian_send_status() {
  if [[ "$QR_LIVE_ENABLED" != "1" || "$arg_count" -le 0 || "$QR_REQUIRE_POSITION_GUARDIAN_ACTIVE" != "1" ]] \
    || ! has_arg "--send" "${args[@]}"; then
    return 0
  fi

  local guardian_check
  guardian_check="${ROOT_DIR}/scripts/install-position-guardian.sh"
  if [[ ! -x "$guardian_check" ]]; then
    export QR_POSITION_GUARDIAN_ACTIVE=0
    echo "[run-autotrade-live] position guardian checker missing: ${guardian_check}; LiveOrderGateway will block fresh entry sends." >&2
    return 0
  fi

  if QR_SYNC_LIVE_ROOT="$ROOT_DIR" QR_OANDA_ENV_FILE="$QR_OANDA_ENV_FILE" "$guardian_check" --require-loaded >&2; then
    export QR_POSITION_GUARDIAN_ACTIVE=1
    echo "[run-autotrade-live] position guardian active; fresh entry sends may proceed through gateway validation." >&2
  else
    export QR_POSITION_GUARDIAN_ACTIVE=0
    echo "[run-autotrade-live] position guardian is not active; full cycle will still run, but LiveOrderGateway will block fresh entry sends so TP-progress profit capture is not blind." >&2
  fi
}

emit_technical_forecast_forward_shadow() {
  if [[ "$QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED" != "1" ]]; then
    return 0
  fi
  local emitter shadow_status
  emitter="${ROOT_DIR}/scripts/emit_technical_forecast_forward_shadow.py"
  if [[ ! -f "$emitter" ]]; then
    echo "[run-autotrade-live] technical forecast forward-shadow emitter missing: ${emitter}" >&2
    return 0
  fi
  set +e
  "$QR_PYTHON" "$emitter" >&2
  shadow_status="$?"
  set -e
  if [[ "$shadow_status" -ne 0 ]]; then
    echo "[run-autotrade-live] technical forecast forward-shadow refresh failed status=${shadow_status}; no shadow signal was admitted." >&2
  fi
}

resolve_technical_forecast_forward_outcomes() {
  if [[ "$QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED" != "1" ]]; then
    return 0
  fi
  local resolver outcome_status
  resolver="${ROOT_DIR}/scripts/resolve_technical_forecast_forward_outcomes.py"
  if [[ ! -f "$resolver" ]]; then
    echo "[run-autotrade-live] technical forecast forward resolver missing: ${resolver}" >&2
    return 0
  fi
  set +e
  "$QR_PYTHON" "$resolver" >&2
  outcome_status="$?"
  set -e
  if [[ "$outcome_status" -ne 0 ]]; then
    echo "[run-autotrade-live] technical forecast forward resolution failed status=${outcome_status}; live permission remains false." >&2
  fi
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

refresh_position_guardian_send_status
finalize_codex_market_read_handoff
refresh_gpt_handoff_if_needed
initialize_consolidated_cycle_id

set +e
if [[ "$arg_count" -gt 0 ]]; then
  "$QR_PYTHON" -m quant_rabbit.cli autotrade-cycle "${args[@]}"
else
  "$QR_PYTHON" -m quant_rabbit.cli autotrade-cycle
fi
cycle_exit="$?"
set -e

if [[ "${QR_RUN_POST_GATEWAY_SIDECARS:-1}" == "1" ]]; then
  if [[ "$cycle_exit" -eq 0 ]]; then
    echo "[run-autotrade-live] refreshing post-gateway sidecars under live lock." >&2
    "$QR_PYTHON" -m quant_rabbit.cli cycle-sidecars
  else
    echo "[run-autotrade-live] autotrade-cycle exited status=${cycle_exit}; refreshing failure-repair sidecars under live lock." >&2
    set +e
    "$QR_PYTHON" -m quant_rabbit.cli post-autotrade-failure-sidecars
    post_failure_sidecars_exit="$?"
    set -e
    if [[ "$post_failure_sidecars_exit" -ne 0 ]]; then
      echo "[run-autotrade-live] failure-repair sidecar refresh incomplete: post-autotrade-failure-sidecars=${post_failure_sidecars_exit}" >&2
    fi
  fi
fi

if [[ "$cycle_exit" -eq 0 ]]; then
  emit_technical_forecast_forward_shadow
  resolve_technical_forecast_forward_outcomes
fi

# Slack notifications are opt-in. User directive 2026-05-30:
# 「Slackに送らないで」. Each notifier is still idempotent if explicitly
# enabled, but the live trader must not post to Slack by default.
if [[ "${QR_SLACK_NOTIFY_ENABLE:-0}" == "1" && "${QR_SLACK_NOTIFY_DISABLE:-0}" != "1" ]]; then
  if [[ -f "${ROOT_DIR}/tools/slack_fill_notify.py" ]]; then
    "$QR_PYTHON" "${ROOT_DIR}/tools/slack_fill_notify.py" 2>&1 | sed 's/^/[slack-fill] /' || true
  fi
  if [[ -f "${ROOT_DIR}/tools/slack_target_milestone.py" ]]; then
    "$QR_PYTHON" "${ROOT_DIR}/tools/slack_target_milestone.py" 2>&1 | sed 's/^/[slack-target] /' || true
  fi
  if [[ -f "${ROOT_DIR}/tools/slack_cycle_alert.py" ]]; then
    "$QR_PYTHON" "${ROOT_DIR}/tools/slack_cycle_alert.py" 2>&1 | sed 's/^/[slack-cycle] /' || true
  fi
fi

exit "$cycle_exit"
