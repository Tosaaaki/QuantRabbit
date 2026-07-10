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
export QR_LIVE_ENABLED="${QR_LIVE_ENABLED:-0}"

load_live_enabled_from_env_file() {
  if [[ -n "${QR_LIVE_ENABLED:-}" && "${QR_LIVE_ENABLED}" != "0" ]]; then
    return 0
  fi
  if [[ ! -f "$QR_OANDA_ENV_FILE" ]]; then
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
    0|1) export QR_LIVE_ENABLED="$value" ;;
    *)
      echo "[run-position-guardian-live] invalid QR_LIVE_ENABLED in ${QR_OANDA_ENV_FILE}; expected 0 or 1." >&2
      exit 2
      ;;
  esac
}

readonly QR_AUTOTRADE_LOCK_DIR="${QR_AUTOTRADE_LOCK_DIR:-${ROOT_DIR}/.quant_rabbit_live.lock}"
readonly QR_AUTOTRADE_LOCK_WAIT_SECONDS="${QR_AUTOTRADE_LOCK_WAIT_SECONDS:-0}"
readonly QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN="${QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN:-}"
readonly QR_AUTOTRADE_LOCK_POLL_SECONDS="${QR_AUTOTRADE_LOCK_POLL_SECONDS:-2}"
# launchd can fire this guardian every 30s. When the full trader cycle is
# already holding the live lock, the guardian must yield instead of turning
# profit protection into a cycle-blocking error stream.
readonly QR_POSITION_GUARDIAN_LOCK_BUSY_MODE="${QR_POSITION_GUARDIAN_LOCK_BUSY_MODE:-skip}"

source "${SCRIPT_DIR}/qr-live-lock.sh"

acquire_lock() {
  qr_live_lock_acquire \
    "$QR_AUTOTRADE_LOCK_DIR" \
    "run-position-guardian-live" \
    "$QR_AUTOTRADE_LOCK_WAIT_SECONDS" \
    "$QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN" \
    "$QR_AUTOTRADE_LOCK_POLL_SECONDS"
}

skip_if_live_lock_busy() {
  local existing_pid existing_command existing_label
  existing_pid="$(qr_live_lock_pid "$QR_AUTOTRADE_LOCK_DIR")"
  if ! qr_live_lock_pid_is_running "$existing_pid"; then
    return 0
  fi
  existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
  existing_label=""
  if [[ -f "${QR_AUTOTRADE_LOCK_DIR}/command" ]]; then
    existing_label="$(cat "${QR_AUTOTRADE_LOCK_DIR}/command" 2>/dev/null || true)"
  fi
  if [[ "$QR_POSITION_GUARDIAN_LOCK_BUSY_MODE" == "skip" ]]; then
    echo "[run-position-guardian-live] live runtime lock busy pid=${existing_pid} label=${existing_label:-unknown}; skipped guardian cycle." >&2
    if [[ -n "$existing_command" ]]; then
      echo "[run-position-guardian-live] lock owner command=${existing_command}" >&2
    fi
    exit 0
  fi
}

position_guardian_pair_scope() {
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" <<'PY'
import json
import re
import sys
from pathlib import Path

snapshot_path = Path(sys.argv[1])
trigger_contract_path = Path(sys.argv[2])
order_intents_path = Path(sys.argv[3])
try:
    max_candidate_pairs = min(6, max(0, int(sys.argv[4])))
except (TypeError, ValueError):
    max_candidate_pairs = 6


def load(path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def pair_of(item):
    if not isinstance(item, dict):
        return ""
    value = str(item.get("pair") or item.get("instrument") or "").strip().upper()
    return value if re.fullmatch(r"[A-Z0-9]{2,12}_[A-Z0-9]{2,12}", value) else ""


snapshot = load(snapshot_path)
positions = [item for item in snapshot.get("positions", []) or [] if isinstance(item, dict)]
open_pairs = sorted({pair_of(item) for item in positions if pair_of(item)})
trader_pairs = sorted(
    {
        pair_of(item)
        for item in positions
        if pair_of(item) and str(item.get("owner") or "").strip().lower() == "trader"
    }
)

# The hourly trader publishes the trigger contract in priority order. Keep
# every open-position pair, then add only a small candidate frontier so this
# 30-second loop never degenerates into all-pairs/all-timeframes polling.
candidate_rows = []
for index, item in enumerate(snapshot.get("orders", []) or []):
    if not isinstance(item, dict):
        continue
    state = str(item.get("state") or "PENDING").strip().upper()
    pair = pair_of(item)
    if pair and state in {"PENDING", "LIVE", "OPEN"}:
        candidate_rows.append((-2, 0, index, pair))
contract = load(trigger_contract_path)
for index, item in enumerate(contract.get("entries", []) or []):
    pair = pair_of(item)
    if not pair or not isinstance(item, dict) or not item.get("lane_id"):
        continue
    status = str(item.get("status") or "").strip().upper()
    candidate_rows.append(
        (
            0 if status == "LIVE_READY" else 1 if status in {"RISK_ALLOWED", "TRIGGER_READY"} else 2,
            1 if bool(item.get("watch_only")) else 0,
            index,
            pair,
        )
    )

# A newly generated LIVE_READY intent may precede its next trigger-contract
# refresh. Parse the much larger intent packet only in that narrow newer-file
# window; the trigger contract is the steady-state candidate source.
try:
    intents_newer_than_contract = order_intents_path.stat().st_mtime_ns > trigger_contract_path.stat().st_mtime_ns
except OSError:
    intents_newer_than_contract = order_intents_path.exists()
intents = load(order_intents_path) if intents_newer_than_contract else {}
for index, item in enumerate(intents.get("results", []) or []):
    if not isinstance(item, dict):
        continue
    status = str(item.get("status") or "").strip().upper()
    if status not in {"LIVE_READY", "RISK_ALLOWED", "TRIGGER_READY"}:
        continue
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    pair = pair_of(intent) or pair_of(item)
    if pair:
        candidate_rows.append((-1, 0, index, pair))

candidates = []
seen = set(open_pairs)
if max_candidate_pairs > 0:
    for _status_rank, _watch_rank, _index, pair in sorted(candidate_rows):
        if pair in seen:
            continue
        seen.add(pair)
        candidates.append(pair)
        if len(candidates) >= max_candidate_pairs:
            break

monitor_pairs = [*open_pairs, *candidates]
print(
    "|".join(
        (
            ",".join(open_pairs),
            ",".join(trader_pairs),
            ",".join(candidates),
            ",".join(monitor_pairs),
        )
    )
)
PY
}

write_trader_only_snapshot() {
  "$QR_PYTHON" - "$1" "$2" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
payload = json.loads(source.read_text())
positions = payload.get("positions", []) or []
payload["positions"] = [
    item
    for item in positions
    if isinstance(item, dict) and str(item.get("owner") or "").strip().lower() == "trader"
]
orders = payload.get("orders", []) or []
payload["orders"] = [
    item
    for item in orders
    if isinstance(item, dict) and str(item.get("owner") or "").strip().lower() == "trader"
]
payload["position_guardian_scope"] = {
    "management_owner": "trader",
    "non_trader_positions_monitor_only": True,
    "non_trader_orders_monitor_only": True,
    "source_snapshot": str(source),
}
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
}

chart_refresh_due() {
  "$QR_PYTHON" - "$1" "$2" "$3" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

charts_path = Path(sys.argv[1])
freshness_path = Path(sys.argv[2])
expected_pairs = [item for item in sys.argv[3].split(",") if item]
try:
    charts = json.loads(charts_path.read_text())
    freshness = json.loads(freshness_path.read_text())
except Exception:
    print("1")
    raise SystemExit(0)

if charts.get("generated_at_utc") != freshness.get("source_generated_at_utc"):
    print("1")
    raise SystemExit(0)
if freshness.get("monitor_pairs") != expected_pairs:
    print("1")
    raise SystemExit(0)
if freshness.get("timeframes") != ["M1", "M5", "M15"]:
    print("1")
    raise SystemExit(0)
try:
    next_due = datetime.fromisoformat(str(freshness.get("next_refresh_after_utc") or "").replace("Z", "+00:00"))
except ValueError:
    print("1")
    raise SystemExit(0)
if next_due.tzinfo is None:
    next_due = next_due.replace(tzinfo=timezone.utc)
print("1" if datetime.now(timezone.utc) >= next_due else "0")
PY
}

write_chart_freshness() {
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" <<'PY'
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

charts_path = Path(sys.argv[1])
freshness_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
open_pairs = [item for item in sys.argv[4].split(",") if item]
trader_pairs = [item for item in sys.argv[5].split(",") if item]
candidate_pairs = [item for item in sys.argv[6].split(",") if item]
try:
    close_grace_seconds = max(0, int(sys.argv[7]))
except (TypeError, ValueError):
    close_grace_seconds = 5
snapshot_path = Path(sys.argv[8])
trigger_contract_path = Path(sys.argv[9])
order_intents_path = Path(sys.argv[10])

payload = json.loads(charts_path.read_text())
now = datetime.now(timezone.utc)
timeframe_seconds = {"M1": 60, "M5": 300, "M15": 900}
monitor_pairs = [*open_pairs, *candidate_pairs]
try:
    snapshot = json.loads(snapshot_path.read_text())
except Exception:
    snapshot = {}
owners_by_pair = {}
for position in snapshot.get("positions", []) or []:
    if not isinstance(position, dict):
        continue
    pair = str(position.get("pair") or position.get("instrument") or "").strip().upper()
    if pair not in monitor_pairs:
        continue
    owner = str(position.get("owner") or "unknown").strip().lower() or "unknown"
    owners_by_pair.setdefault(pair, set()).add(owner)
pending_order_pairs = set()
pending_order_owners_by_pair = {}
for order in snapshot.get("orders", []) or []:
    if not isinstance(order, dict):
        continue
    pair = str(order.get("pair") or order.get("instrument") or "").strip().upper()
    state = str(order.get("state") or "PENDING").strip().upper()
    if pair not in monitor_pairs or state not in {"PENDING", "LIVE", "OPEN"}:
        continue
    pending_order_pairs.add(pair)
    owner = str(order.get("owner") or "unknown").strip().lower() or "unknown"
    pending_order_owners_by_pair.setdefault(pair, set()).add(owner)


def load_dict(path):
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def item_pair(item):
    if not isinstance(item, dict):
        return ""
    return str(item.get("pair") or item.get("instrument") or "").strip().upper()


trigger_pairs = {
    item_pair(item)
    for item in load_dict(trigger_contract_path).get("entries", []) or []
    if isinstance(item, dict) and item.get("lane_id") and item_pair(item)
}
try:
    intents_newer_than_contract = order_intents_path.stat().st_mtime_ns > trigger_contract_path.stat().st_mtime_ns
except OSError:
    intents_newer_than_contract = order_intents_path.exists()
active_intents = load_dict(order_intents_path) if intents_newer_than_contract else {}
active_intent_pairs = set()
for item in active_intents.get("results", []) or []:
    if not isinstance(item, dict):
        continue
    if str(item.get("status") or "").strip().upper() not in {"LIVE_READY", "RISK_ALLOWED", "TRIGGER_READY"}:
        continue
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    pair = item_pair(intent) or item_pair(item)
    if pair:
        active_intent_pairs.add(pair)
monitor_scope = {}
for pair in monitor_pairs:
    owners = sorted(owners_by_pair.get(pair, set()))
    order_owners = sorted(pending_order_owners_by_pair.get(pair, set()))
    reasons = []
    if owners:
        reasons.append("open_position")
    reasons.extend(f"{owner}_open_position" for owner in owners)
    if pair in pending_order_pairs:
        reasons.append("active_pending_order")
    if pair in trigger_pairs:
        reasons.append("active_trigger_candidate")
    if pair in active_intent_pairs:
        reasons.append("active_intent_candidate")
    monitor_scope[pair] = {
        "reasons": reasons,
        "position_owners": owners,
        "pending_order_owners": order_owners,
        "management_write_eligible": "trader" in owners,
        "management_write_eligible_owners": ["trader"] if "trader" in owners else [],
        "non_trader_monitor_only": any(owner != "trader" for owner in owners),
    }
# The guardian event router receives this bounded chart artifact directly, so
# its technical-state surface cannot accidentally expand back to the hourly
# all-28-pair packet. These top-level fields also make the scope auditable.
payload["guardian_monitor_pairs"] = monitor_pairs
payload["guardian_monitor_scope"] = monitor_scope
charts_temporary = charts_path.with_name(f".{charts_path.name}.{os.getpid()}.tmp")
charts_temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(charts_temporary, charts_path)
chart_by_pair = {
    str(item.get("pair") or "").upper(): item
    for item in payload.get("charts", []) or []
    if isinstance(item, dict)
}
rows = []
missing = []
stale = []
for pair in monitor_pairs:
    chart = chart_by_pair.get(pair)
    views = {
        str(item.get("granularity") or "").upper(): item
        for item in (chart or {}).get("views", []) or []
        if isinstance(item, dict)
    }
    for timeframe, seconds in timeframe_seconds.items():
        view = views.get(timeframe)
        candles = (view or {}).get("recent_candles", []) or []
        complete = [item for item in candles if isinstance(item, dict) and item.get("complete") is True]
        if not complete:
            missing.append(f"{pair}:{timeframe}")
            rows.append({"pair": pair, "timeframe": timeframe, "status": "MISSING_COMPLETE_CANDLE"})
            continue
        latest = complete[-1]
        try:
            started_at = datetime.fromisoformat(str(latest.get("t") or "").replace("Z", "+00:00"))
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
        except ValueError:
            missing.append(f"{pair}:{timeframe}")
            rows.append({"pair": pair, "timeframe": timeframe, "status": "INVALID_COMPLETE_CANDLE_TIME"})
            continue
        closed_at = started_at + timedelta(seconds=seconds)
        age_seconds = max(0.0, (now - closed_at).total_seconds())
        max_age_seconds = float(seconds * 2)
        status = "FRESH" if age_seconds <= max_age_seconds else "STALE"
        if status == "STALE":
            stale.append(f"{pair}:{timeframe}")
        rows.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "status": status,
                "latest_complete_candle_started_at_utc": started_at.isoformat(),
                "latest_complete_candle_closed_at_utc": closed_at.isoformat(),
                "age_seconds": round(age_seconds, 3),
                "max_age_seconds": max_age_seconds,
            }
        )

next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=close_grace_seconds)
status = "FRESH" if not missing and not stale and not bool(payload.get("partial")) else "PARTIAL" if missing else "STALE"
freshness = {
    "checked_at_utc": now.isoformat(),
    "status": status,
    "refresh_mode": "CLOSED_CANDLE_CADENCE",
    "freshness_basis": "latest_complete_candle",
    "analysis_packet_may_include_current_open_candle": False,
    "analysis_packet_complete_only": True,
    "source_generated_at_utc": payload.get("generated_at_utc"),
    "next_refresh_after_utc": next_minute.isoformat(),
    "timeframes": list(timeframe_seconds),
    "open_position_pairs": open_pairs,
    "trader_management_pairs": trader_pairs,
    "candidate_pairs": candidate_pairs,
    "monitor_pairs": monitor_pairs,
    "guardian_monitor_pairs": monitor_pairs,
    "guardian_monitor_scope": monitor_scope,
    "candidate_pair_limit_applied": True,
    "missing_complete_candles": missing,
    "stale_complete_candles": stale,
    "rows": rows,
}
freshness_path.parent.mkdir(parents=True, exist_ok=True)
temporary = freshness_path.with_name(f".{freshness_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(freshness, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(temporary, freshness_path)

if report_path.exists():
    report = report_path.read_text()
    marker = "\n## Closed-candle freshness\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report += marker
    report += f"\n- Status: `{status}`\n"
    report += f"- Checked at UTC: `{now.isoformat()}`\n"
    report += f"- Monitor pairs: `{','.join(monitor_pairs) or 'none'}`\n"
    report += "- Timeframes: `M1,M5,M15` (refresh no faster than the next M1 close)\n"
    report += f"- Fresh/missing/stale rows: `{len(rows) - len(missing) - len(stale)}/{len(missing)}/{len(stale)}`\n"
    report += f"- Detail JSON: `{freshness_path}`\n"
    report_path.write_text(report)
print(f"{status}|{len(rows)}|{len(missing)}|{len(stale)}|{next_minute.isoformat()}")
PY
}

write_empty_monitor_charts() {
  "$QR_PYTHON" - "$1" "$2" "$3" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

charts_path = Path(sys.argv[1])
freshness_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
now = datetime.now(timezone.utc).isoformat()
charts = {
    "generated_at_utc": now,
    "timeframes": ["M1", "M5", "M15"],
    "pairs_requested": 0,
    "pairs_succeeded": 0,
    "pairs_failed": 0,
    "partial": False,
    "failures": [],
    "charts": [],
    "guardian_monitor_pairs": [],
    "guardian_monitor_scope": {},
}
freshness = {
    "checked_at_utc": now,
    "status": "NO_MONITOR_SCOPE",
    "refresh_mode": "CLOSED_CANDLE_CADENCE",
    "freshness_basis": "latest_complete_candle",
    "source_generated_at_utc": now,
    "next_refresh_after_utc": now,
    "timeframes": ["M1", "M5", "M15"],
    "open_position_pairs": [],
    "trader_management_pairs": [],
    "candidate_pairs": [],
    "monitor_pairs": [],
    "guardian_monitor_pairs": [],
    "guardian_monitor_scope": {},
    "rows": [],
}
for path, payload in ((charts_path, charts), (freshness_path, freshness)):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "# Position Guardian Pair Charts Report\n\n"
    f"- Generated at UTC: `{now}`\n"
    "- Status: `NO_MONITOR_SCOPE`\n"
    "- Guardian monitor pairs: `none`\n"
)
PY
}

write_monitor_only_artifacts() {
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

snapshot_path = Path(sys.argv[1])
management_path = Path(sys.argv[2])
open_pairs = [item for item in sys.argv[3].split(",") if item]
candidate_pairs = [item for item in sys.argv[4].split(",") if item]
monitor_pairs = [item for item in sys.argv[5].split(",") if item]
now = datetime.now(timezone.utc).isoformat()
try:
    snapshot = json.loads(snapshot_path.read_text())
except Exception:
    snapshot = {}
management = {
    "generated_at_utc": now,
    "snapshot_fetched_at_utc": snapshot.get("fetched_at_utc"),
    "action": "MONITOR_ONLY_NO_TRADER_POSITION",
    "positions": [],
    "monitor_scope": {
        "open_position_pairs": open_pairs,
        "candidate_pairs": candidate_pairs,
        "monitor_pairs": monitor_pairs,
        "write_owner": "trader",
        "non_trader_positions_monitor_only": True,
    },
}
management_path.parent.mkdir(parents=True, exist_ok=True)
management_tmp = management_path.with_name(f".{management_path.name}.{os.getpid()}.tmp")
management_tmp.write_text(json.dumps(management, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(management_tmp, management_path)

heartbeat = {
    "generated_at_utc": now,
    "status": "MONITOR_ONLY_NO_TRADER_POSITION",
    "sent": False,
    "reason": "non-trader open positions and bounded candidates are read-only; no trader-owned position execution ran",
    "open_position_pairs": open_pairs,
    "candidate_pairs": candidate_pairs,
    "monitor_pairs": monitor_pairs,
    "management_write_owner": "trader",
}
path = Path("data/position_guardian.json")
path.parent.mkdir(parents=True, exist_ok=True)
heartbeat_tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
heartbeat_tmp.write_text(json.dumps(heartbeat, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(heartbeat_tmp, path)
PY
}

load_live_enabled_from_env_file
skip_if_live_lock_busy
acquire_lock

# Mirror the live trader risk/protection defaults so direct CLI calls and the
# full autotrade wrapper read protected SL-free positions the same way.
export QR_GEOMETRY_ATR_MULT="${QR_GEOMETRY_ATR_MULT:-5.0}"
export QR_GEOMETRY_SPREAD_FLOOR_MULT="${QR_GEOMETRY_SPREAD_FLOOR_MULT:-12.0}"
export QR_TRADER_DISABLE_SL_REPAIR="${QR_TRADER_DISABLE_SL_REPAIR:-1}"
export QR_MAX_PORTFOLIO_POSITIONS="${QR_MAX_PORTFOLIO_POSITIONS:-10}"
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
export QR_TRADER_BASE_UNITS="${QR_TRADER_BASE_UNITS:-3000}"
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"

guardian_snapshot="${QR_POSITION_GUARDIAN_SNAPSHOT:-data/position_guardian_broker_snapshot.json}"
guardian_management="${QR_POSITION_GUARDIAN_MANAGEMENT:-data/position_guardian_management.json}"
guardian_management_report="${QR_POSITION_GUARDIAN_MANAGEMENT_REPORT:-docs/position_guardian_management_report.md}"
guardian_execution="${QR_POSITION_GUARDIAN_EXECUTION:-data/position_guardian_execution.json}"
guardian_execution_report="${QR_POSITION_GUARDIAN_EXECUTION_REPORT:-docs/position_guardian_execution_report.md}"
guardian_charts="${QR_POSITION_GUARDIAN_PAIR_CHARTS:-data/position_guardian_pair_charts.json}"
guardian_report="${QR_POSITION_GUARDIAN_PAIR_CHARTS_REPORT:-docs/position_guardian_pair_charts_report.md}"
guardian_chart_freshness="${QR_POSITION_GUARDIAN_CHART_FRESHNESS:-data/position_guardian_chart_freshness.json}"
guardian_trader_snapshot="${QR_POSITION_GUARDIAN_TRADER_SNAPSHOT:-data/position_guardian_trader_snapshot.json}"
guardian_trigger_contract="${QR_POSITION_GUARDIAN_TRIGGER_CONTRACT:-data/guardian_trigger_contract.json}"
guardian_order_intents="${QR_POSITION_GUARDIAN_ORDER_INTENTS:-data/order_intents.json}"
guardian_count="${QR_POSITION_GUARDIAN_CANDLE_COUNT:-120}"
guardian_candidate_limit="${QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS:-6}"
guardian_candle_close_grace_seconds="${QR_POSITION_GUARDIAN_CANDLE_CLOSE_GRACE_SECONDS:-5}"

run_guardian_event_router() {
  "$QR_PYTHON" -m quant_rabbit.cli guardian-event-router \
    --snapshot "$guardian_snapshot" \
    --pair-charts "$guardian_charts" \
    --order-intents "$guardian_order_intents" \
    --position-management "$guardian_management" \
    --thesis-evolution "${QR_POSITION_GUARDIAN_THESIS_EVOLUTION:-data/thesis_evolution_report.json}" \
    --forecast-persistence "${QR_POSITION_GUARDIAN_FORECAST_PERSISTENCE:-data/forecast_persistence_report.json}" \
    --market-context-matrix "${QR_POSITION_GUARDIAN_MARKET_CONTEXT_MATRIX:-data/market_context_matrix.json}" \
    --output "${QR_POSITION_GUARDIAN_EVENTS:-data/guardian_events.json}" \
    --escalation-output "${QR_POSITION_GUARDIAN_ESCALATION:-data/guardian_escalation.json}" \
    --report "${QR_POSITION_GUARDIAN_EVENT_REPORT:-docs/guardian_event_report.md}" \
    --action-review-report "${QR_POSITION_GUARDIAN_ACTION_REVIEW:-docs/guardian_action_review.md}"
}

"$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output "$guardian_snapshot"
pair_scope="$(position_guardian_pair_scope \
  "$guardian_snapshot" \
  "$guardian_trigger_contract" \
  "$guardian_order_intents" \
  "$guardian_candidate_limit")"
IFS='|' read -r open_pairs trader_pairs candidate_pairs monitor_pairs <<< "$pair_scope"

if [[ -n "$monitor_pairs" ]]; then
  refresh_due="$(chart_refresh_due "$guardian_charts" "$guardian_chart_freshness" "$monitor_pairs")"
  if [[ "$refresh_due" == "1" ]]; then
    "$QR_PYTHON" -m quant_rabbit.cli pair-charts \
      --pairs "$monitor_pairs" \
      --timeframes M1,M5,M15 \
      --count "$guardian_count" \
      --output "$guardian_charts" \
      --report "$guardian_report"
    freshness_summary="$(write_chart_freshness \
      "$guardian_charts" \
      "$guardian_chart_freshness" \
      "$guardian_report" \
      "$open_pairs" \
      "$trader_pairs" \
      "$candidate_pairs" \
      "$guardian_candle_close_grace_seconds" \
      "$guardian_snapshot" \
      "$guardian_trigger_contract" \
      "$guardian_order_intents")"
    echo "[run-position-guardian-live] closed-candle chart refresh ${freshness_summary} pairs=${monitor_pairs}." >&2
  else
    echo "[run-position-guardian-live] closed-candle charts remain within M1 cadence; reused ${guardian_charts}." >&2
  fi
else
  write_empty_monitor_charts "$guardian_charts" "$guardian_chart_freshness" "$guardian_report"
fi

if [[ -z "$trader_pairs" ]]; then
  write_monitor_only_artifacts \
    "$guardian_snapshot" \
    "$guardian_management" \
    "$open_pairs" \
    "$candidate_pairs" \
    "$monitor_pairs"
  run_guardian_event_router
  echo "[run-position-guardian-live] no trader-owned open positions; completed read-only monitor scope pairs=${monitor_pairs:-none}." >&2
  exit 0
fi

write_trader_only_snapshot "$guardian_snapshot" "$guardian_trader_snapshot"

"$QR_PYTHON" -m quant_rabbit.cli position-management \
  --snapshot "$guardian_trader_snapshot" \
  --pair-charts "$guardian_charts" \
  --output "$guardian_management" \
  --report "$guardian_management_report"

pexec_args=(
  position-execution
  --snapshot "$guardian_trader_snapshot"
  --position-management "$guardian_management"
  --output "$guardian_execution"
  --report "$guardian_execution_report"
)
if [[ "$QR_LIVE_ENABLED" == "1" ]]; then
  pexec_args+=(--send --confirm-live)
fi

"$QR_PYTHON" -m quant_rabbit.cli "${pexec_args[@]}"
run_guardian_event_router
