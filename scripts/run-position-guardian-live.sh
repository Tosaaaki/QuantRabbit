#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="${QR_TRADER_ROOT_DIR:-/Users/tossaki/App/QuantRabbit-live}"
cd "$ROOT_DIR"

# Resolve from the wrapper itself so test/runtime roots and live worktree
# indirection cannot make helper imports depend on the current directory.
export PYTHONPATH="${SCRIPT_DIR}/../src"
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
    # `mkdir` becomes visible just before the owner PID is persisted. The
    # guardian must yield during that initialization window, not race the full
    # trader into stale-lock recovery and accidentally delete its new lock.
    if [[ -d "$QR_AUTOTRADE_LOCK_DIR" && ! "$existing_pid" =~ ^[0-9]+$ \
      && "$QR_POSITION_GUARDIAN_LOCK_BUSY_MODE" == "skip" ]]; then
      echo "[run-position-guardian-live] live runtime lock owner metadata is initializing; skipped guardian cycle." >&2
      exit 0
    fi
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
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

snapshot_path = Path(sys.argv[1])
trigger_contract_path = Path(sys.argv[2])
order_intents_path = Path(sys.argv[3])
active_board_path = Path(sys.argv[5])
frontier_path = Path(sys.argv[6])
freshness_path = Path(sys.argv[7])
try:
    max_candidate_pairs = max(0, int(sys.argv[4]))
except (TypeError, ValueError):
    max_candidate_pairs = 6
configured_universe = tuple(DEFAULT_TRADER_PAIRS)
configured_pair_set = set(configured_universe)
if max_candidate_pairs > len(configured_universe):
    raise SystemExit(
        "QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS must be in "
        f"0..{len(configured_universe)}; "
        f"received {max_candidate_pairs}"
    )


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
prior_freshness = load(freshness_path)
now_utc = datetime.now(timezone.utc)


def parse_utc(value):
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


cursor_override = os.environ.get("QR_POSITION_GUARDIAN_ROTATION_CURSOR")
cursor_override_value = None
if cursor_override is not None:
    try:
        cursor_override_value = int(cursor_override)
    except (TypeError, ValueError):
        raise SystemExit(
            "QR_POSITION_GUARDIAN_ROTATION_CURSOR must be a non-negative integer"
        )
    if cursor_override_value < 0:
        raise SystemExit(
            "QR_POSITION_GUARDIAN_ROTATION_CURSOR must be a non-negative integer"
        )
    cursor_override_value %= len(configured_universe)
scope_frozen_before_next_close = False
scope_retry_required = False
coverage_next_cursor_hint = None
prior_scope_available = all(
    isinstance(prior_freshness.get(key), list)
    for key in ("candidate_pairs", "monitor_pairs")
)
prior_retry_required = bool(prior_freshness.get("coverage_retry_required")) or str(
    prior_freshness.get("status") or ""
).upper() in {"PARTIAL", "STALE"}
if cursor_override_value is not None and not prior_freshness:
    # An environment variable is persistent across launchd cycles, while a
    # coverage reset must be one-shot.  Treat the override as a bootstrap seed
    # only; after the first artifact exists, its successful absolute cursor is
    # authoritative.  Otherwise a configured override would pin every closed
    # M1 cycle to the same pair window and silently stop 28-pair coverage.
    coverage_cursor = cursor_override_value
    coverage_cursor_source = "BOOTSTRAP_EXPLICIT_OVERRIDE"
else:
    cursor_semantics = str(
        prior_freshness.get("coverage_cursor_semantics") or ""
    ).upper()
    if cursor_semantics == "NEXT_CONFIGURED_PAIR_INDEX":
        prior_success_value = prior_freshness.get("coverage_cursor")
    elif prior_freshness:
        # Legacy cursors counted refresh attempts and multiplied that value by
        # the current number of free slots.  Because priority churn changed the
        # multiplier, no safe next-pair position can be reconstructed. Restart
        # once from the first configured pair; duplicate reads are preferable
        # to carrying a hidden coverage hole into the new cursor contract.
        prior_success_value = 0
    else:
        prior_success_value = -1
    try:
        last_success_cursor = int(prior_success_value)
    except (TypeError, ValueError):
        last_success_cursor = -1
    try:
        active_scope_cursor = int(
            prior_freshness.get("active_scope_cursor", last_success_cursor)
        )
    except (TypeError, ValueError):
        active_scope_cursor = last_success_cursor
    try:
        prior_proposed_next_cursor = int(
            prior_freshness.get(
                "proposed_coverage_next_cursor",
                last_success_cursor,
            )
        )
    except (TypeError, ValueError):
        prior_proposed_next_cursor = last_success_cursor
    retry_required = prior_retry_required
    next_refresh_after = parse_utc(prior_freshness.get("next_refresh_after_utc"))
    if (
        next_refresh_after is not None
        and now_utc < next_refresh_after
        and active_scope_cursor >= 0
    ):
        coverage_cursor = active_scope_cursor
        coverage_cursor_source = "PERSISTED_ACTIVE_SCOPE_BEFORE_NEXT_CLOSED_M1"
        coverage_next_cursor_hint = prior_proposed_next_cursor
        scope_frozen_before_next_close = prior_scope_available
    elif retry_required and active_scope_cursor >= 0 and prior_scope_available:
        coverage_cursor = active_scope_cursor
        coverage_cursor_source = "RETRY_ACTIVE_SCOPE_AFTER_PARTIAL_OR_STALE"
        coverage_next_cursor_hint = prior_proposed_next_cursor
        scope_retry_required = True
    elif retry_required:
        coverage_cursor = 0
        coverage_cursor_source = "RETRY_SCOPE_MISSING_RESET_TO_FIRST_CONFIGURED_PAIR"
    elif last_success_cursor < 0:
        coverage_cursor = 0
        coverage_cursor_source = "INITIAL_PROPOSED_CURSOR"
    else:
        coverage_cursor = last_success_cursor
        coverage_cursor_source = (
            "NEXT_UNCOVERED_PAIR_AFTER_CLOSED_M1"
            if next_refresh_after is not None and now_utc >= next_refresh_after
            else "LAST_SUCCESS_NEXT_PAIR_CURSOR"
        )
    coverage_cursor %= len(configured_universe)

positions = [item for item in snapshot.get("positions", []) or [] if isinstance(item, dict)]
open_pairs = sorted({pair_of(item) for item in positions if pair_of(item)})
trader_pairs = sorted(
    {
        pair_of(item)
        for item in positions
        if pair_of(item) and str(item.get("owner") or "").strip().lower() == "trader"
    }
)

# Keep every open-position pair. Pending/LIVE/TRIGGER-ready pairs are hard
# priority, the hourly board/frontier is pinned next, and the remaining G8
# universe rotates once per M1 window. This preserves the six-pair steady-state
# request budget while removing the old permanent first-six blind spot.
hard_rows = []
for index, item in enumerate(snapshot.get("orders", []) or []):
    if not isinstance(item, dict):
        continue
    state = str(item.get("state") or "PENDING").strip().upper()
    pair = pair_of(item)
    if pair in configured_pair_set and state in {"PENDING", "LIVE", "OPEN"}:
        hard_rows.append((-3, index, pair, "active_pending_order"))
contract = load(trigger_contract_path)
for index, item in enumerate(contract.get("entries", []) or []):
    pair = pair_of(item)
    if pair not in configured_pair_set or not isinstance(item, dict) or not item.get("lane_id"):
        continue
    status = str(item.get("status") or "").strip().upper()
    row = (
        0 if status == "LIVE_READY" else 1,
        index,
        pair,
        f"trigger_contract:{status or 'UNKNOWN'}",
    )
    if status in {"LIVE_READY", "RISK_ALLOWED", "TRIGGER_READY"}:
        hard_rows.append(row)

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
    if pair in configured_pair_set:
        hard_rows.append((-1, index, pair, f"order_intent:{status}"))


def fresh_hourly_artifact(payload):
    generated_at = parse_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        return False
    age_seconds = (now_utc - generated_at).total_seconds()
    return -300 <= age_seconds <= 90 * 60 and not str(
        payload.get("status") or ""
    ).upper().startswith(("ERROR", "FAILED", "MISSING"))


def lane_pair(payload, *keys):
    if not fresh_hourly_artifact(payload):
        return ""
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    if not isinstance(current, dict) or not str(current.get("lane_id") or "").strip():
        return ""
    pair = pair_of(current)
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    return pair if pair in configured_pair_set and pair in quotes else ""


priority_rows = []
board = load(active_board_path)
frontier = load(frontier_path)
for index, pair in enumerate(
    (
        lane_pair(board, "top_lane"),
        lane_pair(frontier, "next_evidence_lane"),
        lane_pair(frontier, "top_non_eurusd_lane"),
        lane_pair(frontier, "top_lane"),
    )
):
    if pair:
        priority_rows.append((index, pair, "hourly_active_frontier"))

candidates = []
hard_candidates = []
priority_candidates = []
seen = set(open_pairs)
for _status_rank, _index, pair, _source in sorted(hard_rows):
    if pair in seen:
        continue
    seen.add(pair)
    candidates.append(pair)
    hard_candidates.append(pair)

# Hard execution-adjacent pairs must not be evicted even when the operator set
# the ordinary rotation budget to zero. Filtering to the configured G8 set
# above guarantees this expansion can never exceed 28 candidate pairs.
effective_limit = max(max_candidate_pairs, len(candidates))
if effective_limit > len(configured_universe):
    raise SystemExit("hard-priority candidate scope exceeds configured G8 universe")
for _index, pair, _source in sorted(priority_rows):
    if len(candidates) >= effective_limit or pair in seen:
        continue
    seen.add(pair)
    candidates.append(pair)
    priority_candidates.append(pair)

remaining = max(0, effective_limit - len(candidates))
coverage_next_cursor = coverage_cursor
scanned = 0
while remaining and scanned < len(configured_universe):
    pair_index = (coverage_cursor + scanned) % len(configured_universe)
    pair = configured_universe[pair_index]
    scanned += 1
    coverage_next_cursor = (pair_index + 1) % len(configured_universe)
    if pair in seen:
        # Open/hard/hourly-priority pairs are already in this same fetch, so
        # advancing past them preserves complete coverage without duplication.
        continue
    seen.add(pair)
    candidates.append(pair)
    remaining -= 1
coverage_scanned_count = scanned

monitor_pairs = [*open_pairs, *candidates]
if scope_frozen_before_next_close:
    # The current packet remains the authoritative technical surface until the
    # next closed M1 candle plus grace.  Keep current ownership separately, but
    # do not claim a newly proposed pair is being monitored before its charts
    # have actually been fetched.
    candidate_pairs = prior_freshness.get("candidate_pairs", [])
    monitor_pairs = prior_freshness.get("monitor_pairs", [])
    hard_candidates = prior_freshness.get("hard_priority_candidate_pairs", [])
    priority_candidates = prior_freshness.get("hourly_priority_candidate_pairs", [])
    if not isinstance(candidate_pairs, list):
        candidate_pairs = []
    if not isinstance(monitor_pairs, list):
        monitor_pairs = []
    if not isinstance(hard_candidates, list):
        hard_candidates = []
    if not isinstance(priority_candidates, list):
        priority_candidates = []
    try:
        effective_limit = max(
            0,
            int(
                prior_freshness.get(
                    "effective_candidate_pair_limit",
                    len(candidate_pairs),
                )
            ),
        )
    except (TypeError, ValueError):
        effective_limit = len(candidate_pairs)
    candidates = [str(item) for item in candidate_pairs if str(item)]
    monitor_pairs = [str(item) for item in monitor_pairs if str(item)]
    hard_candidates = [str(item) for item in hard_candidates if str(item)]
    priority_candidates = [str(item) for item in priority_candidates if str(item)]
    if coverage_next_cursor_hint is not None:
        coverage_next_cursor = coverage_next_cursor_hint % len(configured_universe)
    try:
        coverage_scanned_count = max(
            0,
            int(prior_freshness.get("proposed_coverage_scanned_count", 0)),
        )
    except (TypeError, ValueError):
        coverage_scanned_count = 0
elif scope_retry_required:
    # Retry the exact failed candidate surface even if the hourly board moves.
    # New open/hard execution-adjacent pairs may be added for safety, but they
    # can never replace a pair whose prior refresh was partial or stale.
    prior_candidates = prior_freshness.get("candidate_pairs", [])
    prior_monitor = prior_freshness.get("monitor_pairs", [])
    prior_hard = prior_freshness.get("hard_priority_candidate_pairs", [])
    prior_hourly = prior_freshness.get("hourly_priority_candidate_pairs", [])
    if not isinstance(prior_candidates, list):
        prior_candidates = []
    if not isinstance(prior_monitor, list):
        prior_monitor = []
    if not isinstance(prior_hard, list):
        prior_hard = []
    if not isinstance(prior_hourly, list):
        prior_hourly = []
    current_hard = list(hard_candidates)
    open_set = set(open_pairs)
    candidates = []
    prior_retry_pairs = [
        str(pair)
        for pair in prior_monitor
        if str(pair) in configured_pair_set
    ]
    for pair in [*prior_retry_pairs, *prior_candidates, *current_hard]:
        pair = str(pair)
        if pair and pair not in open_set and pair not in candidates:
            candidates.append(pair)
    monitor_pairs = [*open_pairs, *candidates]
    hard_candidates = []
    for pair in [*prior_hard, *current_hard]:
        pair = str(pair)
        if pair and pair not in hard_candidates:
            hard_candidates.append(pair)
    priority_candidates = [str(item) for item in prior_hourly if str(item)]
    try:
        prior_effective_limit = max(
            0,
            int(prior_freshness.get("effective_candidate_pair_limit", 0)),
        )
    except (TypeError, ValueError):
        prior_effective_limit = 0
    effective_limit = max(prior_effective_limit, len(candidates))
    if coverage_next_cursor_hint is not None:
        coverage_next_cursor = coverage_next_cursor_hint % len(configured_universe)
    try:
        coverage_scanned_count = max(
            0,
            int(prior_freshness.get("proposed_coverage_scanned_count", 0)),
        )
    except (TypeError, ValueError):
        coverage_scanned_count = 0
print(
    "|".join(
        (
            ",".join(open_pairs),
            ",".join(trader_pairs),
            ",".join(candidates),
            ",".join(monitor_pairs),
            ",".join(hard_candidates),
            ",".join(priority_candidates),
            str(coverage_cursor),
            str(coverage_next_cursor),
            str(coverage_scanned_count),
            coverage_cursor_source,
            str(effective_limit),
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
now = datetime.now(timezone.utc)
# A newly prioritized/rotated scope is not a new closed candle. Reuse the
# current bounded packet until the persisted close-plus-grace deadline, then
# refresh the proposed next scope once. This prevents priority churn from
# multiplying M1/M5/M15 broker reads inside one candle.
if now < next_due:
    print("0")
    raise SystemExit(0)
if freshness.get("monitor_pairs") != expected_pairs:
    print("1")
    raise SystemExit(0)
print("1")
PY
}

write_chart_freshness() {
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.strategy.directional_forecaster import (
    validate_mba_integrity_receipt,
)

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
hard_priority_pairs = [item for item in sys.argv[11].split(",") if item]
hourly_priority_pairs = [item for item in sys.argv[12].split(",") if item]
try:
    coverage_cursor = int(sys.argv[13])
except (TypeError, ValueError):
    coverage_cursor = None
try:
    coverage_next_cursor = int(sys.argv[14])
except (TypeError, ValueError):
    coverage_next_cursor = coverage_cursor
try:
    coverage_scanned_count = max(0, int(sys.argv[15]))
except (TypeError, ValueError):
    coverage_scanned_count = 0
coverage_cursor_source = str(sys.argv[16] or "UNKNOWN")
try:
    configured_candidate_limit = max(0, int(sys.argv[17]))
except (TypeError, ValueError):
    configured_candidate_limit = 6
try:
    effective_candidate_limit = max(0, int(sys.argv[18]))
except (TypeError, ValueError):
    effective_candidate_limit = configured_candidate_limit

charts_bytes = charts_path.read_bytes()
payload = json.loads(charts_bytes.decode("utf-8"))
try:
    prior_freshness = json.loads(freshness_path.read_text())
except Exception:
    prior_freshness = {}
if not isinstance(prior_freshness, dict):
    prior_freshness = {}
now = datetime.now(timezone.utc)
timeframe_seconds = {"M1": 60, "M5": 300, "M15": 900}
monitor_pairs = [*open_pairs, *candidate_pairs]


def parse_aware_utc(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


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
source_pair_charts_sha256 = hashlib.sha256(charts_path.read_bytes()).hexdigest()
chart_by_pair = {
    str(item.get("pair") or "").upper(): item
    for item in payload.get("charts", []) or []
    if isinstance(item, dict)
}
rows = []
missing = []
stale = []
blocked = []
invalid_integrity = []
incomplete_integrity = []
chart_generated_at = parse_aware_utc(payload.get("generated_at_utc"))
chart_generated_at_valid = bool(
    chart_generated_at is not None
    and (chart_generated_at - now).total_seconds() <= 5
)
for pair in monitor_pairs:
    chart = chart_by_pair.get(pair)
    views = {
        str(item.get("granularity") or "").upper(): item
        for item in (chart or {}).get("views", []) or []
        if isinstance(item, dict)
    }
    for timeframe, seconds in timeframe_seconds.items():
        view = views.get(timeframe)
        integrity = (
            view.get("candle_integrity")
            if isinstance(view, dict)
            and isinstance(view.get("candle_integrity"), dict)
            else {}
        )
        integrity_present = bool(integrity)
        blocking_codes = integrity.get("blocking_codes")
        integrity_scope_bound = bool(
            integrity.get("pair") == pair
            and integrity.get("granularity") == timeframe
            and isinstance(view, dict)
            and view.get("granularity") == timeframe
        )
        integrity_receipt_valid = bool(
            integrity_present
            and chart_generated_at_valid
            and integrity_scope_bound
            and isinstance(view, dict)
            and validate_mba_integrity_receipt(
                integrity,
                chart_generated_at=chart_generated_at,
                view=view,
                now_utc=None,
            )
        )
        integrity_blocked = bool(
            integrity_receipt_valid
            and integrity.get("forecast_blocking") is True
            and str(integrity.get("evaluation_status") or "").upper() == "BLOCKED"
            and isinstance(blocking_codes, list)
            and any(isinstance(code, str) and code for code in blocking_codes)
        )
        integrity_acquisition_complete = bool(
            integrity_receipt_valid
            and integrity.get("coverage_complete") is True
            and integrity.get("provenance_complete") is True
            and type(integrity.get("malformed_count")) is int
            and integrity.get("malformed_count") == 0
            and type(integrity.get("complete_entry_count")) is int
            and integrity.get("complete_entry_count") > 0
        )
        if not chart_generated_at_valid:
            missing.append(f"{pair}:{timeframe}")
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "status": "INVALID_CHART_GENERATED_AT_UTC",
                    "source_generated_at_utc": payload.get("generated_at_utc"),
                }
            )
            continue
        if not integrity_receipt_valid:
            missing.append(f"{pair}:{timeframe}")
            invalid_integrity.append(f"{pair}:{timeframe}")
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "status": "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
                    "blocking_codes": blocking_codes,
                }
            )
            continue
        if integrity_blocked and not integrity_acquisition_complete:
            missing.append(f"{pair}:{timeframe}")
            incomplete_integrity.append(f"{pair}:{timeframe}")
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "status": "INCOMPLETE_TECHNICAL_CANDLE_INTEGRITY_ACQUISITION",
                    "blocking_codes": blocking_codes,
                    "coverage_complete": integrity.get("coverage_complete"),
                    "provenance_complete": integrity.get("provenance_complete"),
                    "malformed_count": integrity.get("malformed_count"),
                    "complete_entry_count": integrity.get("complete_entry_count"),
                }
            )
            continue
        candles = (view or {}).get("recent_candles", []) or []
        complete = [item for item in candles if isinstance(item, dict) and item.get("complete") is True]
        if not complete:
            # The chart reader intentionally withholds an execution-timeframe
            # series when the most recent broker MBA candle is spread- or
            # provenance-contaminated.  That is current, fail-closed market
            # evidence, not a transport/shape failure.  Preserve the block for
            # the event router and entry gates, but do not let one quarantined
            # pair pin the entire 28-pair rotation until its next clean M5.
            # A malformed/missing integrity clock still remains PARTIAL.
            if integrity_acquisition_complete:
                started_at = parse_aware_utc(
                    integrity.get("latest_complete_timestamp_utc")
                )
                if started_at is None:
                    missing.append(f"{pair}:{timeframe}")
                    rows.append(
                        {
                            "pair": pair,
                            "timeframe": timeframe,
                            "status": "BLOCKED_INTEGRITY_CLOCK_MISSING",
                            "blocking_codes": blocking_codes,
                        }
                    )
                    continue
            else:
                missing.append(f"{pair}:{timeframe}")
                rows.append({"pair": pair, "timeframe": timeframe, "status": "MISSING_COMPLETE_CANDLE"})
                continue
        else:
            latest = complete[-1]
            started_at = parse_aware_utc(latest.get("t"))
            if started_at is None:
                missing.append(f"{pair}:{timeframe}")
                rows.append({"pair": pair, "timeframe": timeframe, "status": "INVALID_COMPLETE_CANDLE_TIME"})
                continue
        closed_at = started_at + timedelta(seconds=seconds)
        if closed_at > now:
            missing.append(f"{pair}:{timeframe}")
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "status": "FUTURE_COMPLETE_CANDLE_TIME",
                    "latest_complete_candle_started_at_utc": started_at.isoformat(),
                    "latest_complete_candle_closed_at_utc": closed_at.isoformat(),
                    "blocking_codes": blocking_codes if integrity_blocked else [],
                }
            )
            continue
        age_seconds = max(0.0, (now - closed_at).total_seconds())
        max_age_seconds = float(seconds * 2)
        row_status = (
            "TECHNICAL_INPUT_BLOCKED_CURRENT"
            if integrity_blocked and age_seconds <= max_age_seconds
            else "FRESH"
            if age_seconds <= max_age_seconds
            else "STALE"
        )
        if row_status == "STALE":
            stale.append(f"{pair}:{timeframe}")
        elif row_status == "TECHNICAL_INPUT_BLOCKED_CURRENT":
            blocked.append(f"{pair}:{timeframe}")
        rows.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "status": row_status,
                "latest_complete_candle_started_at_utc": started_at.isoformat(),
                "latest_complete_candle_closed_at_utc": closed_at.isoformat(),
                "age_seconds": round(age_seconds, 3),
                "max_age_seconds": max_age_seconds,
                "blocking_codes": blocking_codes if integrity_blocked else [],
            }
        )

next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=close_grace_seconds)
status = (
    "FRESH"
    if not missing and not stale and not bool(payload.get("partial"))
    else "PARTIAL"
    if missing or bool(payload.get("partial"))
    else "STALE"
)
prior_success_value = (
    prior_freshness.get("coverage_cursor")
    if str(prior_freshness.get("coverage_cursor_semantics") or "").upper()
    == "NEXT_CONFIGURED_PAIR_INDEX"
    else None
)
try:
    prior_success_cursor = int(prior_success_value)
except (TypeError, ValueError):
    prior_success_cursor = None
proposed_coverage_cursor = coverage_cursor
proposed_coverage_next_cursor = coverage_next_cursor
if status == "FRESH":
    committed_coverage_cursor = proposed_coverage_next_cursor
    committed_coverage_cursor_source = coverage_cursor_source
    coverage_retry_required = False
    coverage_retry_status = None
else:
    committed_coverage_cursor = prior_success_cursor
    committed_coverage_cursor_source = (
        f"PRIOR_SUCCESS_RETAINED_AFTER_{status}"
        if prior_success_cursor is not None
        else f"NO_SUCCESS_CURSOR_AFTER_{status}"
    )
    coverage_retry_required = True
    coverage_retry_status = status
coverage_cursor_advanced = bool(
    status == "FRESH" and coverage_scanned_count > 0
)
candidate_limit_expanded = bool(
    effective_candidate_limit > configured_candidate_limit
)
candidate_limit_expansion_reasons = []
if candidate_limit_expanded and hard_priority_pairs:
    candidate_limit_expansion_reasons.append("HARD_PRIORITY_SCOPE")
if candidate_limit_expanded and coverage_cursor_source.startswith("RETRY_"):
    candidate_limit_expansion_reasons.append("RETRY_SCOPE_RETENTION")
if candidate_limit_expanded and not candidate_limit_expansion_reasons:
    candidate_limit_expansion_reasons.append("SAFETY_SCOPE_RETENTION")
freshness = {
    "checked_at_utc": now.isoformat(),
    "status": status,
    "refresh_mode": "CLOSED_CANDLE_CADENCE",
    "freshness_basis": "latest_complete_candle",
    "analysis_packet_may_include_current_open_candle": False,
    "analysis_packet_complete_only": True,
    "source_generated_at_utc": payload.get("generated_at_utc"),
    "source_pair_charts_sha256": source_pair_charts_sha256,
    "next_refresh_after_utc": next_minute.isoformat(),
    "timeframes": list(timeframe_seconds),
    "open_position_pairs": open_pairs,
    "trader_management_pairs": trader_pairs,
    "candidate_pairs": candidate_pairs,
    "hard_priority_candidate_pairs": hard_priority_pairs,
    "hourly_priority_candidate_pairs": hourly_priority_pairs,
    "coverage_cursor": committed_coverage_cursor,
    "coverage_cursor_semantics": "NEXT_CONFIGURED_PAIR_INDEX",
    "coverage_cursor_source": committed_coverage_cursor_source,
    "proposed_coverage_cursor": proposed_coverage_cursor,
    "proposed_coverage_next_cursor": proposed_coverage_next_cursor,
    "proposed_coverage_scanned_count": coverage_scanned_count,
    "proposed_coverage_cursor_source": coverage_cursor_source,
    "active_scope_cursor": proposed_coverage_cursor,
    "coverage_cursor_advanced": coverage_cursor_advanced,
    "coverage_cursor_wrapped": bool(
        status == "FRESH"
        and coverage_cursor_advanced
        and proposed_coverage_next_cursor <= proposed_coverage_cursor
    ),
    "coverage_retry_required": coverage_retry_required,
    "coverage_retry_status": coverage_retry_status,
    "coverage_cursor_advance_policy": "AFTER_SUCCESSFUL_CLOSED_M1_REFRESH_ONLY",
    "rotation_period_seconds": 60,
    "coverage_policy": "HARD_PRIORITY_THEN_HOURLY_FRONTIER_THEN_G8_ROUND_ROBIN",
    "monitor_pairs": monitor_pairs,
    "guardian_monitor_pairs": monitor_pairs,
    "guardian_monitor_scope": monitor_scope,
    "configured_candidate_pair_limit": configured_candidate_limit,
    "effective_candidate_pair_limit": effective_candidate_limit,
    "candidate_pair_limit_expanded": candidate_limit_expanded,
    "candidate_pair_limit_expansion_reasons": candidate_limit_expansion_reasons,
    "candidate_pair_limit_expanded_for_hard_priority": (
        candidate_limit_expanded and bool(hard_priority_pairs)
    ),
    "requested_candidate_pair_count": len(candidate_pairs),
    "candidate_pair_limit_applied": len(candidate_pairs) <= effective_candidate_limit,
    "missing_complete_candles": missing,
    "stale_complete_candles": stale,
    "blocked_technical_inputs": blocked,
    "invalid_technical_integrity_receipts": invalid_integrity,
    "incomplete_technical_integrity_acquisitions": incomplete_integrity,
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
    report += f"- Current technical-integrity blocks: `{','.join(blocked) or 'none'}`\n"
    report += f"- Invalid technical-integrity receipts: `{','.join(invalid_integrity) or 'none'}`\n"
    report += f"- Incomplete technical-integrity acquisitions: `{','.join(incomplete_integrity) or 'none'}`\n"
    report += f"- Detail JSON: `{freshness_path}`\n"
    report_path.write_text(report)
print(f"{status}|{len(rows)}|{len(missing)}|{len(stale)}|{next_minute.isoformat()}")
PY
}

write_empty_monitor_charts() {
  "$QR_PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

charts_path = Path(sys.argv[1])
freshness_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
try:
    proposed_cursor = max(0, int(sys.argv[4]))
except (TypeError, ValueError):
    proposed_cursor = 0
try:
    proposed_next_cursor = max(0, int(sys.argv[5]))
except (TypeError, ValueError):
    proposed_next_cursor = proposed_cursor
proposed_cursor_source = str(sys.argv[6] or "NO_MONITOR_SCOPE")
try:
    configured_candidate_limit = max(0, int(sys.argv[7]))
except (TypeError, ValueError):
    configured_candidate_limit = 0
try:
    effective_candidate_limit = max(0, int(sys.argv[8]))
except (TypeError, ValueError):
    effective_candidate_limit = configured_candidate_limit
try:
    prior_freshness = json.loads(freshness_path.read_text())
except Exception:
    prior_freshness = {}
if not isinstance(prior_freshness, dict):
    prior_freshness = {}
prior_cursor_is_absolute = (
    str(prior_freshness.get("coverage_cursor_semantics") or "").upper()
    == "NEXT_CONFIGURED_PAIR_INDEX"
)
try:
    committed_cursor = (
        int(prior_freshness.get("coverage_cursor"))
        if prior_cursor_is_absolute
        else proposed_cursor
    )
except (TypeError, ValueError):
    committed_cursor = proposed_cursor
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
    "hard_priority_candidate_pairs": [],
    "hourly_priority_candidate_pairs": [],
    "coverage_cursor": committed_cursor,
    "coverage_cursor_semantics": "NEXT_CONFIGURED_PAIR_INDEX",
    "coverage_cursor_source": str(
        prior_freshness.get("coverage_cursor_source")
        if prior_cursor_is_absolute
        else proposed_cursor_source
    )
    or "NO_MONITOR_SCOPE_RETAINED",
    "proposed_coverage_cursor": proposed_cursor,
    "proposed_coverage_next_cursor": proposed_next_cursor,
    "proposed_coverage_scanned_count": 0,
    "proposed_coverage_cursor_source": proposed_cursor_source,
    "active_scope_cursor": proposed_cursor,
    "coverage_cursor_advanced": False,
    "coverage_cursor_wrapped": False,
    "coverage_retry_required": False,
    "coverage_retry_status": None,
    "coverage_cursor_advance_policy": "AFTER_SUCCESSFUL_CLOSED_M1_REFRESH_ONLY",
    "rotation_period_seconds": 60,
    "coverage_policy": "HARD_PRIORITY_THEN_HOURLY_FRONTIER_THEN_G8_ROUND_ROBIN",
    "configured_candidate_pair_limit": configured_candidate_limit,
    "effective_candidate_pair_limit": effective_candidate_limit,
    "candidate_pair_limit_expanded": False,
    "candidate_pair_limit_expansion_reasons": [],
    "candidate_pair_limit_expanded_for_hard_priority": False,
    "requested_candidate_pair_count": 0,
    "candidate_pair_limit_applied": True,
    "blocked_technical_inputs": [],
    "invalid_technical_integrity_receipts": [],
    "incomplete_technical_integrity_acquisitions": [],
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
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"
export QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED="${QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED:-1}"
case "$QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED=${QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED="${QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED:-1}"
case "$QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED=${QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED="${QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED:-1}"
case "$QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED=${QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED="${QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED:-1}"
case "$QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED=${QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_FAST_BOT_SHADOW_ENABLED="${QR_FAST_BOT_SHADOW_ENABLED:-1}"
case "$QR_FAST_BOT_SHADOW_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_FAST_BOT_SHADOW_ENABLED=${QR_FAST_BOT_SHADOW_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_FAST_BOT_OUTCOME_ENABLED="${QR_FAST_BOT_OUTCOME_ENABLED:-1}"
case "$QR_FAST_BOT_OUTCOME_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_FAST_BOT_OUTCOME_ENABLED=${QR_FAST_BOT_OUTCOME_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac
export QR_FAST_BOT_LEARNING_ENABLED="${QR_FAST_BOT_LEARNING_ENABLED:-1}"
case "$QR_FAST_BOT_LEARNING_ENABLED" in
  0|1) ;;
  *)
    echo "[run-position-guardian-live] invalid QR_FAST_BOT_LEARNING_ENABLED=${QR_FAST_BOT_LEARNING_ENABLED}; expected 0 or 1." >&2
    exit 2
    ;;
esac

guardian_snapshot="${QR_POSITION_GUARDIAN_SNAPSHOT:-data/position_guardian_broker_snapshot.json}"
guardian_management="${QR_POSITION_GUARDIAN_MANAGEMENT:-data/position_guardian_management.json}"
guardian_management_report="${QR_POSITION_GUARDIAN_MANAGEMENT_REPORT:-docs/position_guardian_management_report.md}"
guardian_execution="${QR_POSITION_GUARDIAN_EXECUTION:-data/position_guardian_execution.json}"
guardian_execution_report="${QR_POSITION_GUARDIAN_EXECUTION_REPORT:-docs/position_guardian_execution_report.md}"
guardian_charts="${QR_POSITION_GUARDIAN_PAIR_CHARTS:-data/position_guardian_pair_charts.json}"
guardian_report="${QR_POSITION_GUARDIAN_PAIR_CHARTS_REPORT:-docs/position_guardian_pair_charts_report.md}"
guardian_chart_freshness="${QR_POSITION_GUARDIAN_CHART_FRESHNESS:-data/position_guardian_chart_freshness.json}"
guardian_all_pair_m1="${QR_POSITION_GUARDIAN_ALL_PAIR_M1:-data/position_guardian_all_pair_m1.json}"
guardian_all_pair_m1_report="${QR_POSITION_GUARDIAN_ALL_PAIR_M1_REPORT:-docs/position_guardian_all_pair_m1_report.md}"
guardian_slow_retention="${QR_POSITION_GUARDIAN_SLOW_RETENTION:-data/position_guardian_slow_retention.json}"
guardian_trader_snapshot="${QR_POSITION_GUARDIAN_TRADER_SNAPSHOT:-data/position_guardian_trader_snapshot.json}"
guardian_trigger_contract="${QR_POSITION_GUARDIAN_TRIGGER_CONTRACT:-data/guardian_trigger_contract.json}"
guardian_order_intents="${QR_POSITION_GUARDIAN_ORDER_INTENTS:-data/order_intents.json}"
guardian_active_board="${QR_POSITION_GUARDIAN_ACTIVE_BOARD:-data/active_opportunity_board.json}"
guardian_non_eurusd_frontier="${QR_POSITION_GUARDIAN_NON_EURUSD_FRONTIER:-data/non_eurusd_live_grade_frontier.json}"
guardian_count="${QR_POSITION_GUARDIAN_CANDLE_COUNT:-120}"
guardian_candidate_limit="${QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS:-6}"
guardian_candle_close_grace_seconds="${QR_POSITION_GUARDIAN_CANDLE_CLOSE_GRACE_SECONDS:-5}"
guardian_all_pair_observation_enabled="${QR_ALL_PAIR_OBSERVATION_ENABLED:-1}"
guardian_observation_module="quant_rabbit.guardian_observation"
guardian_all_pairs="EUR_USD,GBP_USD,AUD_USD,NZD_USD,USD_JPY,USD_CAD,USD_CHF,EUR_GBP,EUR_JPY,EUR_AUD,EUR_CAD,EUR_CHF,EUR_NZD,GBP_JPY,GBP_AUD,GBP_CAD,GBP_CHF,GBP_NZD,AUD_JPY,AUD_CAD,AUD_CHF,AUD_NZD,CAD_JPY,CAD_CHF,CHF_JPY,NZD_JPY,NZD_CAD,NZD_CHF"
guardian_active_chart_wall_seconds=0
guardian_active_chart_pair_count=0

if [[ "$guardian_all_pair_observation_enabled" != "0" \
  && "$guardian_all_pair_observation_enabled" != "1" ]]; then
  echo "[run-position-guardian-live] QR_ALL_PAIR_OBSERVATION_ENABLED must be 0 or 1." >&2
  exit 2
fi

if [[ ! "$guardian_candidate_limit" =~ ^[0-9]+$ ]] \
  || (( guardian_candidate_limit > 28 )); then
  echo "[run-position-guardian-live] QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS must be an integer in 0..28; received ${guardian_candidate_limit}." >&2
  exit 2
fi

run_guardian_event_router() {
  "$QR_PYTHON" -m quant_rabbit.cli guardian-event-router \
    --snapshot "$guardian_snapshot" \
    --pair-charts "$guardian_charts" \
    --chart-freshness "$guardian_chart_freshness" \
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

emit_technical_forecast_forward_shadow() {
  if [[ "$QR_TECHNICAL_FORECAST_FORWARD_SHADOW_ENABLED" != "1" ]]; then
    return 0
  fi
  local emitter shadow_status
  emitter="${ROOT_DIR}/scripts/emit_technical_forecast_forward_shadow.py"
  if [[ ! -f "$emitter" ]]; then
    return 0
  fi
  set +e
  "$QR_PYTHON" "$emitter" --fresh-m5 >&2
  shadow_status="$?"
  set -e
  if [[ "$shadow_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] technical forecast forward-shadow refresh failed status=${shadow_status}; no shadow signal was admitted." >&2
  fi
}

resolve_technical_forecast_forward_outcomes() {
  if [[ "$QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_ENABLED" != "1" ]]; then
    return 0
  fi
  local resolver outcome_status
  resolver="${ROOT_DIR}/scripts/resolve_technical_forecast_forward_outcomes.py"
  if [[ ! -f "$resolver" ]]; then
    return 0
  fi
  set +e
  "$QR_PYTHON" "$resolver" >&2
  outcome_status="$?"
  set -e
  if [[ "$outcome_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] technical forecast forward resolution failed status=${outcome_status}; live permission remains false." >&2
  fi
}

emit_contextual_technical_forward_shadow() {
  if [[ "$QR_CONTEXTUAL_TECHNICAL_FORWARD_SHADOW_ENABLED" != "1" ]]; then
    return 0
  fi
  local emitter shadow_status
  emitter="${ROOT_DIR}/scripts/emit_contextual_technical_240m_forward_shadow.py"
  if [[ ! -f "$emitter" ]]; then
    return 0
  fi
  set +e
  "$QR_PYTHON" "$emitter" >&2
  shadow_status="$?"
  set -e
  if [[ "$shadow_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] contextual technical forward shadow failed status=${shadow_status}; no signal was admitted." >&2
  fi
}

resolve_contextual_technical_forward_outcomes() {
  if [[ "$QR_CONTEXTUAL_TECHNICAL_FORWARD_OUTCOME_ENABLED" != "1" ]]; then
    return 0
  fi
  local resolver outcome_status
  resolver="${ROOT_DIR}/scripts/resolve_contextual_technical_240m_forward_outcomes.py"
  if [[ ! -f "$resolver" ]]; then
    return 0
  fi
  set +e
  "$QR_PYTHON" "$resolver" >&2
  outcome_status="$?"
  set -e
  if [[ "$outcome_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] contextual technical forward resolution failed status=${outcome_status}; live permission remains false." >&2
  fi
}

run_fast_bot_shadow() {
  if [[ "$QR_FAST_BOT_SHADOW_ENABLED" != "1" ]]; then
    return 0
  fi
  local runner bot_status fast_charts slow_charts
  runner="${ROOT_DIR}/scripts/run-fast-bot-shadow.py"
  if [[ ! -f "$runner" ]]; then
    echo "[run-position-guardian-live] fast bot shadow runner is missing; no signal was admitted." >&2
    return 0
  fi
  fast_charts="$guardian_charts"
  slow_charts="${QR_FAST_BOT_SLOW_PAIR_CHARTS:-$guardian_charts}"
  if [[ "$guardian_all_pair_observation_enabled" == "1" ]]; then
    fast_charts="$guardian_all_pair_m1"
    slow_charts="${QR_FAST_BOT_SLOW_PAIR_CHARTS:-$guardian_slow_retention}"
  fi
  set +e
  "$QR_PYTHON" "$runner" \
    --fast-pair-charts "$fast_charts" \
    --slow-pair-charts "$slow_charts" \
    --broker-snapshot "$guardian_snapshot" \
    --guardian-events "${QR_POSITION_GUARDIAN_EVENTS:-data/guardian_events.json}" \
    --ai-supervision "${QR_FAST_BOT_AI_SUPERVISION:-data/ai_regime_supervision.json}" \
    --regime-output "${QR_FAST_BOT_REGIME_OUTPUT:-data/hierarchical_bot_regime.json}" \
    --output "${QR_FAST_BOT_SHADOW_OUTPUT:-data/fast_bot_shadow.json}" \
    --ledger "${QR_FAST_BOT_SHADOW_LEDGER:-data/fast_bot_shadow_ledger.jsonl}" \
    --report "${QR_FAST_BOT_SHADOW_REPORT:-docs/fast_bot_shadow_report.md}" >&2
  bot_status="$?"
  set -e
  if [[ "$bot_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] fast bot shadow failed status=${bot_status}; live permission remains false." >&2
  fi
}

resolve_fast_bot_shadow_outcomes() {
  if [[ "$QR_FAST_BOT_OUTCOME_ENABLED" != "1" ]]; then
    return 0
  fi
  local resolver outcome_status
  resolver="${ROOT_DIR}/scripts/resolve-fast-bot-shadow-outcomes.py"
  if [[ ! -f "$resolver" ]]; then
    echo "[run-position-guardian-live] fast bot outcome resolver is missing; promotion remains blocked." >&2
    return 0
  fi
  set +e
  "$QR_PYTHON" "$resolver" \
    --shadow-ledger "${QR_FAST_BOT_SHADOW_LEDGER:-data/fast_bot_shadow_ledger.jsonl}" \
    --outcome-ledger "${QR_FAST_BOT_OUTCOME_LEDGER:-data/fast_bot_outcome_ledger.jsonl}" \
    --scorecard "${QR_FAST_BOT_SCORECARD:-data/fast_bot_scorecard.json}" >&2
  outcome_status="$?"
  set -e
  if [[ "$outcome_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] fast bot outcome resolution failed status=${outcome_status}; promotion remains blocked." >&2
  fi
}

run_fast_bot_learning_shadow() {
  if [[ "$QR_FAST_BOT_LEARNING_ENABLED" != "1" ]]; then
    return 0
  fi
  local emitter learning_status
  emitter="${ROOT_DIR}/tools/fast_bot_learning_shadow.py"
  if [[ ! -f "$emitter" ]]; then
    echo "[run-position-guardian-live] fast bot learning emitter is missing; counterfactual collection remains unavailable." >&2
    return 0
  fi
  set +e
  "$QR_PYTHON" "$emitter" \
    --regime-contract "${QR_FAST_BOT_REGIME_OUTPUT:-data/hierarchical_bot_regime.json}" \
    --broker-snapshot "$guardian_snapshot" \
    --output "${QR_FAST_BOT_LEARNING_OUTPUT:-data/fast_bot_learning_shadow.json}" \
    --ledger "${QR_FAST_BOT_LEARNING_LEDGER:-data/fast_bot_learning_seat_ledger.jsonl}" >&2
  learning_status="$?"
  set -e
  if [[ "$learning_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] fast bot learning shadow failed status=${learning_status}; primary and live permissions remain unchanged." >&2
  fi
}

resolve_fast_bot_learning_outcomes() {
  if [[ "$QR_FAST_BOT_LEARNING_ENABLED" != "1" ]]; then
    return 0
  fi
  local resolver outcome_status
  resolver="${ROOT_DIR}/tools/resolve-fast-bot-learning-outcomes.py"
  if [[ ! -f "$resolver" ]]; then
    echo "[run-position-guardian-live] fast bot learning outcome resolver is missing; counterfactual promotion remains blocked." >&2
    return 0
  fi
  set +e
  "$QR_PYTHON" "$resolver" \
    --shadow-ledger "${QR_FAST_BOT_LEARNING_LEDGER:-data/fast_bot_learning_seat_ledger.jsonl}" \
    --outcome-ledger "${QR_FAST_BOT_LEARNING_OUTCOME_LEDGER:-data/fast_bot_learning_outcome_ledger.jsonl}" \
    --scorecard "${QR_FAST_BOT_LEARNING_SCORECARD:-data/fast_bot_learning_scorecard.json}" >&2
  outcome_status="$?"
  set -e
  if [[ "$outcome_status" -ne 0 ]]; then
    echo "[run-position-guardian-live] fast bot learning outcome resolution failed status=${outcome_status}; primary and live permissions remain unchanged." >&2
  fi
}

"$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output "$guardian_snapshot"
pair_scope="$(position_guardian_pair_scope \
  "$guardian_snapshot" \
  "$guardian_trigger_contract" \
  "$guardian_order_intents" \
  "$guardian_candidate_limit" \
  "$guardian_active_board" \
  "$guardian_non_eurusd_frontier" \
  "$guardian_chart_freshness")"
IFS='|' read -r open_pairs trader_pairs candidate_pairs monitor_pairs hard_priority_pairs hourly_priority_pairs coverage_cursor coverage_next_cursor coverage_scanned_count coverage_cursor_source effective_candidate_limit <<< "$pair_scope"

if [[ -n "$monitor_pairs" ]]; then
  refresh_due="$(chart_refresh_due "$guardian_charts" "$guardian_chart_freshness" "$monitor_pairs")"
  if [[ "$refresh_due" == "1" ]]; then
    guardian_active_chart_started="$SECONDS"
    "$QR_PYTHON" -m quant_rabbit.cli pair-charts \
      --pairs "$monitor_pairs" \
      --timeframes M1,M5,M15,M30,H1,H4,D \
      --count "$guardian_count" \
      --output "$guardian_charts" \
      --report "$guardian_report"
    guardian_active_chart_wall_seconds="$((SECONDS - guardian_active_chart_started))"
    IFS=',' read -r -a guardian_active_chart_pairs <<< "$monitor_pairs"
    guardian_active_chart_pair_count="${#guardian_active_chart_pairs[@]}"
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
      "$guardian_order_intents" \
      "$hard_priority_pairs" \
      "$hourly_priority_pairs" \
      "$coverage_cursor" \
      "$coverage_next_cursor" \
      "$coverage_scanned_count" \
      "$coverage_cursor_source" \
      "$guardian_candidate_limit" \
      "$effective_candidate_limit")"
    echo "[run-position-guardian-live] closed-candle chart refresh ${freshness_summary} pairs=${monitor_pairs}." >&2
    if [[ "$guardian_all_pair_observation_enabled" == "1" ]]; then
      set +e
      "$QR_PYTHON" -m "$guardian_observation_module" retain \
        --source "$guardian_charts" \
        --output "$guardian_slow_retention" \
        --source-pairs "$monitor_pairs" >&2
      guardian_retention_status="$?"
      set -e
      if [[ "$guardian_retention_status" -ne 0 ]]; then
        echo "[run-position-guardian-live] slow-view retention rejected the active packet; previous sealed retention remains unchanged." >&2
      fi
    fi
  else
    echo "[run-position-guardian-live] closed-candle charts remain within M1 cadence; reused ${guardian_charts}." >&2
  fi
else
  write_empty_monitor_charts \
    "$guardian_charts" \
    "$guardian_chart_freshness" \
    "$guardian_report" \
    "$coverage_cursor" \
    "$coverage_next_cursor" \
    "$coverage_cursor_source" \
    "$guardian_candidate_limit" \
    "$effective_candidate_limit"
fi

if [[ "$guardian_all_pair_observation_enabled" == "1" ]]; then
  guardian_all_pair_due="$(
    "$QR_PYTHON" -m "$guardian_observation_module" due \
      --current "$guardian_all_pair_m1"
  )"
  if [[ "$guardian_all_pair_due" == "1" ]]; then
    guardian_all_pair_raw="$(mktemp "${TMPDIR:-/tmp}/qr-guardian-all-pair-m1.XXXXXX.json")"
    guardian_all_pair_started="$SECONDS"
    set +e
    "$QR_PYTHON" -m quant_rabbit.cli pair-charts \
      --pairs "$guardian_all_pairs" \
      --timeframes M1 \
      --count "$guardian_count" \
      --output "$guardian_all_pair_raw" \
      --report "$guardian_all_pair_m1_report" \
      --require-complete
    guardian_all_pair_fetch_status="$?"
    set -e
    guardian_all_pair_wall_seconds="$((SECONDS - guardian_all_pair_started))"

    guardian_post_snapshot_started="$SECONDS"
    set +e
    "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output "$guardian_snapshot"
    guardian_post_snapshot_status="$?"
    set -e
    guardian_post_snapshot_wall_seconds="$((SECONDS - guardian_post_snapshot_started))"

    guardian_publish_status=1
    if [[ "$guardian_all_pair_fetch_status" -eq 0 \
      && "$guardian_post_snapshot_status" -eq 0 ]]; then
      set +e
      "$QR_PYTHON" -m "$guardian_observation_module" publish-current \
        --source "$guardian_all_pair_raw" \
        --snapshot "$guardian_snapshot" \
        --output "$guardian_all_pair_m1" \
        --active-pair-count "$guardian_active_chart_pair_count" \
        --active-chart-wall-seconds "$guardian_active_chart_wall_seconds" \
        --all-pair-m1-wall-seconds "$guardian_all_pair_wall_seconds" \
        --post-chart-snapshot-wall-seconds "$guardian_post_snapshot_wall_seconds" \
        --candle-close-grace-seconds "$guardian_candle_close_grace_seconds" >&2
      guardian_publish_status="$?"
      set -e
    fi
    if [[ "$guardian_publish_status" -ne 0 ]]; then
      "$QR_PYTHON" -m "$guardian_observation_module" publish-blocked \
        --output "$guardian_all_pair_m1" \
        --reason "ALL_PAIR_M1_OR_POST_CHART_QUOTES_INVALID" \
        --active-pair-count "$guardian_active_chart_pair_count" \
        --active-chart-wall-seconds "$guardian_active_chart_wall_seconds" \
        --all-pair-m1-wall-seconds "$guardian_all_pair_wall_seconds" \
        --post-chart-snapshot-wall-seconds "$guardian_post_snapshot_wall_seconds" \
        --candle-close-grace-seconds "$guardian_candle_close_grace_seconds" >&2
      echo "[run-position-guardian-live] exact-28 current-M1 observation blocked; slow retention was not erased." >&2
    else
      echo "[run-position-guardian-live] sealed exact-28 current-M1 observation with post-chart quotes." >&2
    fi
    rm -f "$guardian_all_pair_raw"
  fi
fi

if [[ -z "$trader_pairs" ]]; then
  write_monitor_only_artifacts \
    "$guardian_snapshot" \
    "$guardian_management" \
    "$open_pairs" \
    "$candidate_pairs" \
    "$monitor_pairs"
  run_guardian_event_router
  run_fast_bot_shadow
  resolve_fast_bot_shadow_outcomes
  run_fast_bot_learning_shadow
  resolve_fast_bot_learning_outcomes
  emit_technical_forecast_forward_shadow
  resolve_technical_forecast_forward_outcomes
  emit_contextual_technical_forward_shadow
  resolve_contextual_technical_forward_outcomes
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
run_fast_bot_shadow
resolve_fast_bot_shadow_outcomes
run_fast_bot_learning_shadow
resolve_fast_bot_learning_outcomes
emit_technical_forecast_forward_shadow
resolve_technical_forecast_forward_outcomes
emit_contextual_technical_forward_shadow
resolve_contextual_technical_forward_outcomes
