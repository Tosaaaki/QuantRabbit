#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PY_BIN:-}" ]]; then
  PYTHON_BIN="${PY_BIN}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

LOG_DIR="${BRAIN_AUTOPDCA_LOG_DIR:-${ROOT_DIR}/logs}"
BENCHMARK_SCRIPT="${BRAIN_AUTOPDCA_BENCHMARK_SCRIPT:-${ROOT_DIR}/scripts/benchmark_brain_local_llm.py}"
APPLY_SCRIPT="${BRAIN_AUTOPDCA_APPLY_SCRIPT:-${ROOT_DIR}/scripts/apply_brain_model_selection.py}"
STACK_SCRIPT="${BRAIN_AUTOPDCA_STACK_SCRIPT:-${ROOT_DIR}/scripts/local_v2_stack.sh}"
STACK_ENV="${BRAIN_AUTOPDCA_STACK_ENV:-${ROOT_DIR}/ops/env/local-v2-stack.env}"
ENV_PROFILE="${BRAIN_AUTOPDCA_ENV_PROFILE:-${ROOT_DIR}/ops/env/profiles/brain-ollama.env}"

BENCHMARK_OUTPUT="${BRAIN_AUTOPDCA_BENCHMARK_OUTPUT:-${LOG_DIR}/brain_local_llm_benchmark_latest.json}"
SELECTION_OUTPUT="${BRAIN_AUTOPDCA_SELECTION_OUTPUT:-${LOG_DIR}/brain_model_selection_latest.json}"
CYCLE_OUTPUT="${BRAIN_AUTOPDCA_CYCLE_OUTPUT:-${LOG_DIR}/brain_autopdca_cycle_latest.json}"
HISTORY_OUTPUT="${BRAIN_AUTOPDCA_HISTORY_OUTPUT:-${LOG_DIR}/brain_autopdca_cycle_history.jsonl}"
RESTART_SERVICES="${BRAIN_AUTOPDCA_RESTART_SERVICES:-quant-order-manager,quant-strategy-control}"

BENCHMARK_LOG="${BRAIN_AUTOPDCA_BENCHMARK_LOG:-${LOG_DIR}/brain_autopdca_benchmark_latest.log}"
APPLY_LOG="${BRAIN_AUTOPDCA_APPLY_LOG:-${LOG_DIR}/brain_autopdca_apply_latest.log}"
RESTART_LOG="${BRAIN_AUTOPDCA_RESTART_LOG:-${LOG_DIR}/brain_autopdca_restart_latest.log}"

LOCK_DIR="${BRAIN_AUTOPDCA_LOCK_DIR:-${LOG_DIR}/brain_autopdca.lock}"
LOCK_PID_FILE="${LOCK_DIR}/pid"
STATE_FILE="${BRAIN_AUTOPDCA_STATE_FILE:-${LOG_DIR}/brain_autopdca.state}"
MARKET_JSON_TMP=""
APPLY_STDOUT_TMP=""

DEFAULT_INTERVAL_SEC="${BRAIN_AUTOPDCA_INTERVAL_SEC:-1800}"
INTERVAL_SEC="${DEFAULT_INTERVAL_SEC}"
FORCE_RUN=0
DRY_RUN=0
BENCHMARK_EXTRA_ARGS=()
APPLY_EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_brain_autopdca_cycle.sh [options]

Options:
  --interval-sec <sec>        Minimum cycle interval in seconds (default: env or 1800).
  --force                     Ignore interval guard and market sanity guard.
  --dry-run                   Run benchmark/apply but do not restart services.
  --benchmark-arg <arg>       Extra arg for benchmark_brain_local_llm.py (repeatable).
  --apply-arg <arg>           Extra arg for apply_brain_model_selection.py (repeatable).
  -h, --help                  Show help.
USAGE
}

as_positive_int() {
  local value="${1:-}"
  local fallback="${2:-1}"
  if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    printf '%s\n' "${value}"
    return 0
  fi
  printf '%s\n' "${fallback}"
}

acquire_lock() {
  local owner_pid=""

  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" >"${LOCK_PID_FILE}"
    return 0
  fi

  if [[ -f "${LOCK_PID_FILE}" ]]; then
    owner_pid="$(cat "${LOCK_PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ -n "${owner_pid}" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    echo "[brain-autopdca] lock held pid=${owner_pid}; skip overlapping run"
    return 1
  fi

  rm -rf "${LOCK_DIR}" >/dev/null 2>&1 || true
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" >"${LOCK_PID_FILE}"
    echo "[brain-autopdca] removed stale lock owner_pid=${owner_pid:-unknown}"
    return 0
  fi
  return 1
}

release_lock() {
  local owner_pid=""
  if [[ -f "${LOCK_PID_FILE}" ]]; then
    owner_pid="$(cat "${LOCK_PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ "${owner_pid}" == "$$" ]]; then
    rm -rf "${LOCK_DIR}" >/dev/null 2>&1 || true
  fi
}

cleanup_tmp() {
  rm -f "${APPLY_STDOUT_TMP}" "${MARKET_JSON_TMP}" >/dev/null 2>&1 || true
}

load_last_run_epoch() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    return 0
  fi
  awk -F= '$1=="last_run_epoch"{print $2}' "${STATE_FILE}" | tail -n1
}

persist_last_run_epoch() {
  local now_epoch="$1"
  local tmp="${STATE_FILE}.tmp.$$"
  mkdir -p "$(dirname "${STATE_FILE}")"
  printf 'last_run_epoch=%s\n' "${now_epoch}" >"${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

collect_market_snapshot() {
  "${PYTHON_BIN}" - <<'PY'
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from utils.secrets import get_secret


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def tr(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(tr_values: list[float], period: int = 14) -> float | None:
    if len(tr_values) < period:
        return None
    seed = sum(tr_values[:period]) / period
    value = seed
    for item in tr_values[period:]:
        value = ((period - 1) * value + item) / period
    return value


try:
    instrument = "USD_JPY"
    now_utc = datetime.now(timezone.utc)
    practice = str(get_secret("oanda_practice")).strip().lower() in {"1", "true", "yes"}
    api_base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
    account_id = get_secret("oanda_account_id")
    token = get_secret("oanda_token")
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=15.0) as client:
        pricing = client.get(
            f"{api_base}/v3/accounts/{account_id}/pricing",
            headers=headers,
            params={"instruments": instrument},
        )
        pricing.raise_for_status()
        price = pricing.json()["prices"][0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        mid = (bid + ask) / 2.0
        spread_pips = (ask - bid) * 100.0

        from_m1 = (now_utc - timedelta(minutes=180)).isoformat()
        candles_m1_resp = client.get(
            f"{api_base}/v3/instruments/{instrument}/candles",
            headers=headers,
            params={"granularity": "M1", "price": "M", "from": from_m1, "count": 180},
        )
        candles_m1_resp.raise_for_status()
        candles_m1 = [row for row in candles_m1_resp.json().get("candles", []) if row.get("complete")]

        from_m5 = (now_utc - timedelta(minutes=600)).isoformat()
        candles_m5_resp = client.get(
            f"{api_base}/v3/instruments/{instrument}/candles",
            headers=headers,
            params={"granularity": "M5", "price": "M", "from": from_m5, "count": 120},
        )
        candles_m5_resp.raise_for_status()
        candles_m5 = [row for row in candles_m5_resp.json().get("candles", []) if row.get("complete")]

    m1_closes: list[float] = []
    m1_tr: list[float] = []
    prev_close: float | None = None
    for row in candles_m1:
        high = float(row["mid"]["h"])
        low = float(row["mid"]["l"])
        close = float(row["mid"]["c"])
        m1_closes.append(close)
        m1_tr.append(tr(high, low, prev_close))
        prev_close = close

    m5_tr: list[float] = []
    prev_close = None
    for row in candles_m5:
        high = float(row["mid"]["h"])
        low = float(row["mid"]["l"])
        close = float(row["mid"]["c"])
        m5_tr.append(tr(high, low, prev_close))
        prev_close = close

    m1_atr = atr(m1_tr, 14)
    m5_atr = atr(m5_tr, 14)
    recent_close = m1_closes[-60:] if len(m1_closes) >= 60 else m1_closes
    range_60m_pips = (max(recent_close) - min(recent_close)) * 100.0 if recent_close else None

    order_total = 0
    order_reject_like = 0
    order_status_counts: list[list[object]] = []
    orders_path = Path("logs/orders.db")
    if orders_path.exists():
        conn = sqlite3.connect(str(orders_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT status, COUNT(*)
            FROM orders
            WHERE instrument = ? AND ts >= datetime('now','-60 minutes')
            GROUP BY status
            ORDER BY COUNT(*) DESC
            """,
            (instrument,),
        )
        rows = cur.fetchall()
        conn.close()
        reject_like_exact = {
            "rejected",
            "skipped",
            "entry_probability_reject",
            "strategy_control_entry_disabled",
            "order_margin_block",
            "pattern_block",
            "brain_block",
            "spread_block",
        }
        order_status_counts = [[str(status), int(count)] for status, count in rows]
        order_total = sum(int(count) for _, count in rows)
        order_reject_like = sum(
            int(count)
            for status, count in rows
            if status in reject_like_exact or str(status).endswith("_reject") or str(status).endswith("_block")
        )
    reject_like_rate_pct = (float(order_reject_like) / float(order_total) * 100.0) if order_total > 0 else 0.0

    emit(
        {
            "status": "ok",
            "utc_now": now_utc.isoformat(),
            "instrument": instrument,
            "price": {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pips": round(spread_pips, 3),
            },
            "m1": {
                "atr14_pips": round(m1_atr * 100.0, 3) if m1_atr is not None else None,
                "range_60m_pips": round(range_60m_pips, 3) if range_60m_pips is not None else None,
            },
            "m5": {
                "atr14_pips": round(m5_atr * 100.0, 3) if m5_atr is not None else None,
            },
            "orders_60m_total": int(order_total),
            "orders_60m_reject_like": int(order_reject_like),
            "orders_60m_reject_like_rate_pct": round(reject_like_rate_pct, 2),
            "orders_60m_status_counts": order_status_counts[:12],
            "oanda_api_base": api_base,
        }
    )
except Exception as exc:
    emit({"status": "error", "error": str(exc)})
PY
}

parse_env_changed() {
  local selection_path="$1"
  local apply_stdout_path="$2"
  "${PYTHON_BIN}" - "${selection_path}" "${apply_stdout_path}" <<'PY'
import json
import pathlib
import sys


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return None


def parse_file_json(path: pathlib.Path):
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and "env_changed" in payload:
        return parse_bool(payload.get("env_changed"))
    return None


def parse_stdout_json(path: pathlib.Path):
    if not path.exists() or path.stat().st_size <= 0:
        return None
    raw = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict) and "env_changed" in payload:
            return parse_bool(payload.get("env_changed"))
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if isinstance(payload, dict) and "env_changed" in payload:
        return parse_bool(payload.get("env_changed"))
    return None


selection = pathlib.Path(sys.argv[1])
stdout_path = pathlib.Path(sys.argv[2])
parsed = parse_file_json(selection)
if parsed is None:
    parsed = parse_stdout_json(stdout_path)
print("1" if parsed else "0")
PY
}

write_cycle_payload() {
  STARTED_AT="${STARTED_AT}" \
  FINISHED_AT="${FINISHED_AT}" \
  RUN_ID="${RUN_ID}" \
  STATUS="${STATUS}" \
  REASON="${REASON}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT}" \
  APPLY_SCRIPT="${APPLY_SCRIPT}" \
  STACK_SCRIPT="${STACK_SCRIPT}" \
  STACK_ENV="${STACK_ENV}" \
  ENV_PROFILE="${ENV_PROFILE}" \
  BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT}" \
  SELECTION_OUTPUT="${SELECTION_OUTPUT}" \
  RESTART_SERVICES="${RESTART_SERVICES}" \
  DRY_RUN="${DRY_RUN}" \
  FORCE_RUN="${FORCE_RUN}" \
  INTERVAL_SEC="${INTERVAL_SEC}" \
  ENV_CHANGED="${ENV_CHANGED}" \
  RESTART_TRIGGERED="${RESTART_TRIGGERED}" \
  RESTART_PERFORMED="${RESTART_PERFORMED}" \
  BENCHMARK_LOG="${BENCHMARK_LOG}" \
  APPLY_LOG="${APPLY_LOG}" \
  RESTART_LOG="${RESTART_LOG}" \
  CYCLE_OUTPUT="${CYCLE_OUTPUT}" \
  HISTORY_OUTPUT="${HISTORY_OUTPUT}" \
  MARKET_JSON_TMP="${MARKET_JSON_TMP}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def tail_text(path_value: str, max_chars: int = 2000) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    text = text.rstrip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_market(path_value: str):
    path = Path(path_value)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


payload = {
    "generated_at": os.environ["FINISHED_AT"],
    "run_id": os.environ["RUN_ID"],
    "status": os.environ["STATUS"],
    "reason": os.environ["REASON"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "inputs": {
        "python_bin": os.environ["PYTHON_BIN"],
        "benchmark_script": os.environ["BENCHMARK_SCRIPT"],
        "apply_script": os.environ["APPLY_SCRIPT"],
        "stack_script": os.environ["STACK_SCRIPT"],
        "stack_env": os.environ["STACK_ENV"],
        "env_profile": os.environ["ENV_PROFILE"],
        "benchmark_output": os.environ["BENCHMARK_OUTPUT"],
        "selection_output": os.environ["SELECTION_OUTPUT"],
        "restart_services": os.environ["RESTART_SERVICES"],
        "dry_run": as_bool(os.environ["DRY_RUN"]),
        "force_run": as_bool(os.environ["FORCE_RUN"]),
        "interval_sec": int(os.environ["INTERVAL_SEC"]),
    },
    "market_snapshot": load_market(os.environ["MARKET_JSON_TMP"]),
    "apply_result": {
        "env_changed": as_bool(os.environ["ENV_CHANGED"]),
        "restart_triggered": as_bool(os.environ["RESTART_TRIGGERED"]),
        "restart_performed": as_bool(os.environ["RESTART_PERFORMED"]),
    },
    "logs": {
        "benchmark_tail": tail_text(os.environ["BENCHMARK_LOG"]),
        "apply_tail": tail_text(os.environ["APPLY_LOG"]),
        "restart_tail": tail_text(os.environ["RESTART_LOG"]),
    },
}

latest_path = Path(os.environ["CYCLE_OUTPUT"])
latest_path.parent.mkdir(parents=True, exist_ok=True)
latest_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

history_path = Path(os.environ["HISTORY_OUTPUT"])
history_path.parent.mkdir(parents=True, exist_ok=True)
with history_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY
}

while (($# > 0)); do
  case "$1" in
    --interval-sec)
      if (($# < 2)); then
        echo "[brain-autopdca] missing value for --interval-sec" >&2
        exit 2
      fi
      INTERVAL_SEC="$(as_positive_int "$2" "${DEFAULT_INTERVAL_SEC}")"
      shift 2
      ;;
    --force)
      FORCE_RUN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --benchmark-arg)
      if (($# < 2)); then
        echo "[brain-autopdca] missing value for --benchmark-arg" >&2
        exit 2
      fi
      BENCHMARK_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --apply-arg)
      if (($# < 2)); then
        echo "[brain-autopdca] missing value for --apply-arg" >&2
        exit 2
      fi
      APPLY_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[brain-autopdca] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"

for required in "${BENCHMARK_SCRIPT}" "${APPLY_SCRIPT}" "${STACK_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[brain-autopdca] missing required file: ${required}" >&2
    exit 1
  fi
done

if ! acquire_lock; then
  exit 0
fi

APPLY_STDOUT_TMP="$(mktemp "${TMPDIR:-/tmp}/brain_autopdca_apply_stdout.XXXXXX")"
MARKET_JSON_TMP="$(mktemp "${TMPDIR:-/tmp}/brain_autopdca_market.XXXXXX")"
trap 'release_lock; cleanup_tmp' EXIT INT TERM

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")"
STATUS="ok"
REASON="completed"
ENV_CHANGED=0
RESTART_TRIGGERED=0
RESTART_PERFORMED=0
now_epoch="$(date +%s)"

INTERVAL_SEC="$(as_positive_int "${INTERVAL_SEC}" "${DEFAULT_INTERVAL_SEC}")"
if (( FORCE_RUN == 0 )); then
  last_epoch="$(load_last_run_epoch)"
  if [[ "${last_epoch}" =~ ^[0-9]+$ ]] && (( now_epoch > last_epoch )) && (( now_epoch - last_epoch < INTERVAL_SEC )); then
    STATUS="skipped"
    REASON="interval_not_elapsed"
    printf '{"status":"skipped","reason":"interval_not_elapsed","last_run_epoch":%s,"now_epoch":%s,"interval_sec":%s}\n' \
      "${last_epoch}" "${now_epoch}" "${INTERVAL_SEC}" >"${MARKET_JSON_TMP}"
    FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    write_cycle_payload >/dev/null
    echo "[brain-autopdca] skipped interval_not_elapsed interval_sec=${INTERVAL_SEC}"
    exit 0
  fi
fi

MAX_SPREAD_PIPS="${BRAIN_AUTOPDCA_MAX_SPREAD_PIPS:-2.2}"
MAX_REJECT_RATE_PCT="${BRAIN_AUTOPDCA_MAX_REJECT_LIKE_RATE_PCT:-45.0}"
MIN_ORDERS_60M="${BRAIN_AUTOPDCA_MIN_ORDERS_60M:-50}"

MARKET_STDOUT="$(collect_market_snapshot 2>&1 || true)"
printf '%s\n' "${MARKET_STDOUT}" >"${MARKET_JSON_TMP}"
if (( FORCE_RUN == 0 )); then
  MARKET_OK="$(
    "${PYTHON_BIN}" - "${MARKET_JSON_TMP}" "${MAX_SPREAD_PIPS}" "${MAX_REJECT_RATE_PCT}" "${MIN_ORDERS_60M}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
max_spread = float(sys.argv[2])
max_reject = float(sys.argv[3])
min_orders = int(float(sys.argv[4]))

if not path.exists() or path.stat().st_size <= 0:
    print("0")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)
if not isinstance(payload, dict) or payload.get("status") != "ok":
    print("0")
    raise SystemExit(0)
spread = payload.get("price", {}).get("spread_pips")
reject_rate = payload.get("orders_60m_reject_like_rate_pct")
orders = payload.get("orders_60m_total")
if spread is None:
    print("0")
    raise SystemExit(0)
if float(spread) > max_spread:
    print("0")
    raise SystemExit(0)
if orders is not None and int(orders) >= min_orders and reject_rate is not None and float(reject_rate) > max_reject:
    print("0")
    raise SystemExit(0)
print("1")
PY
  )"
  if [[ "${MARKET_OK}" != "1" ]]; then
    STATUS="skipped"
    REASON="market_sanity_guard"
    FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    write_cycle_payload >/dev/null
    echo "[brain-autopdca] skipped market_sanity_guard"
    exit 0
  fi
fi

BENCHMARK_CMD=(
  "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}"
  "--output" "${BENCHMARK_OUTPUT}"
)
if ((${#BENCHMARK_EXTRA_ARGS[@]} > 0)); then
  BENCHMARK_CMD+=("${BENCHMARK_EXTRA_ARGS[@]}")
fi

BENCHMARK_STDOUT=""
if ! BENCHMARK_STDOUT="$("${BENCHMARK_CMD[@]}" 2>&1)"; then
  STATUS="benchmark_failed"
  REASON="benchmark_brain_local_llm_failed"
fi
printf '%s\n' "${BENCHMARK_STDOUT}" >"${BENCHMARK_LOG}"

APPLY_STDOUT=""
if [[ "${STATUS}" == "ok" ]]; then
  APPLY_CMD=(
    "${PYTHON_BIN}" "${APPLY_SCRIPT}"
    "--benchmark" "${BENCHMARK_OUTPUT}"
    "--env-profile" "${ENV_PROFILE}"
    "--output" "${SELECTION_OUTPUT}"
  )
  if ((DRY_RUN == 1)); then
    APPLY_CMD+=("--dry-run")
  fi
  if ((${#APPLY_EXTRA_ARGS[@]} > 0)); then
    APPLY_CMD+=("${APPLY_EXTRA_ARGS[@]}")
  fi
  if ! APPLY_STDOUT="$("${APPLY_CMD[@]}" 2>&1)"; then
    STATUS="apply_failed"
    REASON="apply_brain_model_selection_failed"
  fi
fi
printf '%s\n' "${APPLY_STDOUT}" >"${APPLY_LOG}"
printf '%s' "${APPLY_STDOUT}" >"${APPLY_STDOUT_TMP}"

if [[ "${STATUS}" == "ok" ]]; then
  ENV_CHANGED="$(parse_env_changed "${SELECTION_OUTPUT}" "${APPLY_STDOUT_TMP}")"
fi

RESTART_STDOUT=""
if [[ "${STATUS}" == "ok" ]] && ((DRY_RUN == 0)) && [[ "${ENV_CHANGED}" == "1" ]]; then
  RESTART_TRIGGERED=1
  RESTART_CMD=(
    "${STACK_SCRIPT}" restart
    "--env" "${STACK_ENV}"
    "--services" "${RESTART_SERVICES}"
  )
  if ! RESTART_STDOUT="$("${RESTART_CMD[@]}" 2>&1)"; then
    STATUS="restart_failed"
    REASON="local_v2_stack_restart_failed"
  else
    RESTART_PERFORMED=1
  fi
fi
printf '%s\n' "${RESTART_STDOUT}" >"${RESTART_LOG}"

if [[ "${STATUS}" == "ok" ]]; then
  persist_last_run_epoch "${now_epoch}"
fi

FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
write_cycle_payload >/dev/null

if [[ "${STATUS}" == "benchmark_failed" || "${STATUS}" == "apply_failed" || "${STATUS}" == "restart_failed" ]]; then
  echo "[brain-autopdca] failed status=${STATUS} reason=${REASON}" >&2
  exit 1
fi

echo "[brain-autopdca] completed report=${CYCLE_OUTPUT} status=${STATUS}"
exit 0
