from __future__ import annotations
import glob
import json
import os
import logging
import time
import copy
import sqlite3
import pathlib
import fcntl
import math
from bisect import bisect_left
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.secrets import get_secret


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}

# --- config ---
# env / secret から OANDA 設定を取得
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACT = get_secret("oanda_practice").lower() == "true"
except KeyError:
    PRACT = False
REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

_DB = pathlib.Path("logs/trades.db")
_ORDERS_DB = pathlib.Path("logs/orders.db")
_METRICS_DB = pathlib.Path("logs/metrics.db")
_CHUNK_SIZE = 100
_MAX_FETCH = int(os.getenv("POSITION_MANAGER_MAX_FETCH", "1000"))
_REQUEST_TIMEOUT = float(os.getenv("POSITION_MANAGER_HTTP_TIMEOUT", "7.0"))
_OPEN_TRADES_REQUEST_TIMEOUT = max(
    0.5,
    _env_float(
        "POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT",
        min(_REQUEST_TIMEOUT, 3.5),
    ),
)
_ORDERS_DB_READ_TIMEOUT_SEC = max(
    0.05,
    _env_float("POSITION_MANAGER_ORDERS_DB_READ_TIMEOUT_SEC", 0.35),
)
_RETRY_STATUS_CODES = tuple(
    int(code.strip())
    for code in os.getenv(
        "POSITION_MANAGER_HTTP_RETRY_CODES", "408,409,425,429,500,502,503,504,520,522"
    ).split(",")
    if code.strip().isdigit()
)
_OPEN_TRADES_CACHE_TTL = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_CACHE_TTL", "5.0")
)
_OPEN_TRADES_FAIL_BACKOFF_BASE = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_BACKOFF_BASE", "2.5")
)
_OPEN_TRADES_FAIL_BACKOFF_MAX = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_BACKOFF_MAX", "60.0")
)
_DB_REOPEN_MIN_INTERVAL_SEC = float(
    os.getenv("POSITION_MANAGER_DB_REOPEN_MIN_INTERVAL_SEC", "1.0")
)
_DB_REOPEN_MAX_INTERVAL_SEC = float(
    os.getenv("POSITION_MANAGER_DB_REOPEN_MAX_INTERVAL_SEC", "30.0")
)
_DB_REOPEN_ATTEMPTS = int(os.getenv("POSITION_MANAGER_DB_REOPEN_ATTEMPTS", "4"))
_MANUAL_POCKET_NAME = os.getenv("POSITION_MANAGER_MANUAL_POCKET", "manual")
_KNOWN_POCKETS = {"micro", "macro", "scalp", "scalp_fast"}
_JST = timezone(timedelta(hours=9))
_WORKER_TAG_MAP = {
    "OnePipMakerS1": "scalp",
    "ImpulseRetrace": "scalp",
    "ImpulseRetraceScalp": "scalp",
    "RangeFader": "scalp",
    "PulseBreak": "scalp",
    "MicroMA": "micro",
    "TrendMA": "macro",
    "DonchianM1": "macro",
    "H1Momentum": "macro",
    "BB_RSI": "micro",
    "BB_RSI_Fast": "micro",
    "MicroLevelReactor": "micro",
    "MicroMomentumBurst": "micro",
    "MicroMomentumStack": "micro",
    "MicroPullbackEMA": "micro",
    "MicroRangeBreak": "micro",
    "MicroVWAPBound": "micro",
    "MicroVWAPRevert": "micro",
    "TrendMomentumMicro": "micro",
    "MomentumBurst": "micro",
    "MomentumPulse": "micro",
    "VolCompressionBreak": "micro",
    "VolSpikeRider": "scalp",
    "TechFusion": "macro",
    "MacroTechFusion": "macro",
    "MicroPullbackFib": "micro",
    "ScalpReversalNWave": "scalp",
    "RangeCompressionBreak": "micro",
    "fast_scalp": "scalp",
    "pullback_s5": "scalp",
    "pullback_runner_s5": "scalp",
    "impulse_break_s5": "scalp",
    "impulse_retest_s5": "scalp",
    "impulse_momentum_s5": "scalp",
    "vwap_magnet_s5": "scalp",
    "mm_lite": "scalp",
}


def _normalize_worker_tag(tag: object | str | None) -> str | None:
    if tag is None:
        return None
    tag_str = str(tag).strip()
    if not tag_str:
        return None

    mapped = _WORKER_TAG_MAP.get(tag_str) or _WORKER_TAG_MAP.get(tag_str.lower())
    if mapped:
        return mapped

    lower = tag_str.lower()
    if "micro" in lower:
        return "micro"
    if (
        lower.startswith("scalp")
        or "_scalp" in lower
        or ("s5" in lower and "scalp" in lower)
    ):
        return "scalp"
    return tag_str
# SQLite locking周り
# ロック頻発に備えてデフォルトを広めに取る
_DB_BUSY_TIMEOUT_MS = int(os.getenv("POSITION_MANAGER_DB_BUSY_TIMEOUT_MS", "20000"))
_DB_LOCK_RETRY = int(os.getenv("POSITION_MANAGER_DB_LOCK_RETRY", "6"))
_DB_LOCK_RETRY_SLEEP = float(os.getenv("POSITION_MANAGER_DB_LOCK_RETRY_SLEEP", "0.5"))
# PRAGMA values
_DB_JOURNAL_MODE = os.getenv("POSITION_MANAGER_DB_JOURNAL_MODE", "WAL")
_DB_SYNCHRONOUS = os.getenv("POSITION_MANAGER_DB_SYNCHRONOUS", "NORMAL")
_DB_TEMP_STORE = os.getenv("POSITION_MANAGER_DB_TEMP_STORE", "MEMORY")
_DB_LOCK_PATH = pathlib.Path(os.getenv("POSITION_MANAGER_DB_LOCK_PATH", "logs/trades.db.lock"))
_DB_FILE_LOCK_TIMEOUT = float(os.getenv("POSITION_MANAGER_DB_FILE_LOCK_TIMEOUT", "30.0"))
_METRICS_READ_TIMEOUT_SEC = float(
    os.getenv("POSITION_MANAGER_METRICS_TIMEOUT_SEC", "0.2")
)
_CANDLE_DIR = pathlib.Path(os.getenv("UI_CANDLE_DIR", "logs/oanda"))
_CHARTS_ENABLED = os.getenv("UI_CHARTS_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_CHART_MAX_POINTS = int(os.getenv("UI_CHART_MAX_POINTS", "260"))
_CHART_PRICE_MAX_POINTS = int(os.getenv("UI_CHART_PRICE_MAX_POINTS", "320"))
_CHART_MARKERS_MAX = int(os.getenv("UI_CHART_MARKERS_MAX", "140"))
_HOURLY_LOOKBACK_HOURS = max(
    6, int(os.getenv("POSITION_MANAGER_HOURLY_LOOKBACK_HOURS", "24"))
)
_BACKFILL_ENABLED = os.getenv("POSITION_MANAGER_BACKFILL_ATTR", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_BACKFILL_TTL_SEC = float(os.getenv("POSITION_MANAGER_BACKFILL_TTL_SEC", "86400"))
_BACKFILL_MAX_ROWS = int(os.getenv("POSITION_MANAGER_BACKFILL_MAX_ROWS", "5000"))
_BACKFILL_MARKER = pathlib.Path(
    os.getenv("POSITION_MANAGER_BACKFILL_MARKER", "logs/position_manager_backfill.json")
)
_CASHFLOW_TYPES = {"TRANSFER_FUNDS", "CASH_TRANSFER"}
_CASHFLOW_BACKFILL_ENABLED = os.getenv(
    "POSITION_MANAGER_CASHFLOW_BACKFILL", "1"
).strip().lower() not in {"", "0", "false", "no"}
_CASHFLOW_BACKFILL_OANDA_ENABLED = os.getenv(
    "POSITION_MANAGER_CASHFLOW_BACKFILL_OANDA", "1"
).strip().lower() not in {"", "0", "false", "no"}
_CASHFLOW_OANDA_QUERY_TYPES = tuple(
    t.strip()
    for t in os.getenv(
        "POSITION_MANAGER_CASHFLOW_BACKFILL_OANDA_TYPES",
        # OANDA production accounts accept TRANSFER_FUNDS; some older names return 400.
        "TRANSFER_FUNDS",
    ).split(",")
    if t.strip()
)
_CASHFLOW_BACKFILL_TTL_SEC = float(
    os.getenv("POSITION_MANAGER_CASHFLOW_BACKFILL_TTL_SEC", "86400")
)
_CASHFLOW_BACKFILL_MARKER_VERSION = 1
_CASHFLOW_BACKFILL_MARKER = pathlib.Path(
    os.getenv(
        "POSITION_MANAGER_CASHFLOW_BACKFILL_MARKER",
        "logs/position_manager_cashflow_backfill.json",
    )
)
_CASHFLOW_BACKFILL_GLOB = os.getenv(
    "POSITION_MANAGER_CASHFLOW_BACKFILL_GLOB", "logs/oanda/transactions_*.jsonl"
)
_POSITION_MANAGER_SERVICE_ENABLED = _env_bool("POSITION_MANAGER_SERVICE_ENABLED", False)
_POSITION_MANAGER_SERVICE_URL = os.getenv(
    "POSITION_MANAGER_SERVICE_URL", "http://127.0.0.1:8301"
)
_POSITION_MANAGER_SERVICE_TIMEOUT = _env_float(
    "POSITION_MANAGER_SERVICE_TIMEOUT", max(_REQUEST_TIMEOUT + 1.0, 8.0)
)
if _POSITION_MANAGER_SERVICE_TIMEOUT < _REQUEST_TIMEOUT + 0.5:
    logging.warning(
        "[PositionManager] POSITION_MANAGER_SERVICE_TIMEOUT=%s is too low for POSITION_MANAGER_HTTP_TIMEOUT=%s; "
        "using fallback %s",
        _POSITION_MANAGER_SERVICE_TIMEOUT,
        _REQUEST_TIMEOUT,
        _REQUEST_TIMEOUT + 1.0,
    )
    _POSITION_MANAGER_SERVICE_TIMEOUT = _REQUEST_TIMEOUT + 1.0
_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT = max(
    0.2,
    _env_float(
        "POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT",
        min(_POSITION_MANAGER_SERVICE_TIMEOUT, 4.5),
    ),
)
if _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT > _POSITION_MANAGER_SERVICE_TIMEOUT:
    _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT = _POSITION_MANAGER_SERVICE_TIMEOUT
_POSITION_MANAGER_SERVICE_FALLBACK_LOCAL = _env_bool(
    "POSITION_MANAGER_SERVICE_FALLBACK_LOCAL", False
)
_POSITION_MANAGER_SERVICE_FAIL_BACKOFF_SEC = max(
    0.5, _env_float("POSITION_MANAGER_SERVICE_FAIL_BACKOFF_SEC", 6.0)
)
_POSITION_MANAGER_SERVICE_FAIL_BACKOFF_MAX_SEC = max(
    _POSITION_MANAGER_SERVICE_FAIL_BACKOFF_SEC,
    _env_float("POSITION_MANAGER_SERVICE_FAIL_BACKOFF_MAX_SEC", 45.0),
)
_POSITION_MANAGER_SERVICE_NEXT_RETRY = 0.0
_POSITION_MANAGER_SERVICE_ERROR_COUNT = 0
_POSITION_MANAGER_SERVICE_LAST_ERROR = None
_POSITION_MANAGER_SERVICE_LAST_SUPPRESS_LOG_TS = 0.0
_POSITION_MANAGER_SERVICE_POOL_CONNECTIONS = max(
    8,
    int(os.getenv("POSITION_MANAGER_SERVICE_POOL_CONNECTIONS", "64")),
)
_POSITION_MANAGER_SERVICE_POOL_MAXSIZE = max(
    _POSITION_MANAGER_SERVICE_POOL_CONNECTIONS,
    int(os.getenv("POSITION_MANAGER_SERVICE_POOL_MAXSIZE", "256")),
)
_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC = max(
    0.0,
    _env_float("POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC", 0.35),
)
_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC = max(
    _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC,
    _env_float("POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC", 2.0),
)
_POSITION_MANAGER_SERVICE_SESSION: requests.Session | None = None
_POSITION_MANAGER_SERVICE_SESSION_LOCK = threading.Lock()
_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE: dict[bool, tuple[float, object]] = {}
_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_LOCK = threading.Lock()


def _position_manager_service_enabled() -> bool:
    return bool(
        _POSITION_MANAGER_SERVICE_ENABLED
        and _POSITION_MANAGER_SERVICE_URL
        and not str(_POSITION_MANAGER_SERVICE_URL).lower().startswith("disabled")
    )


def _position_manager_service_url(path: str) -> str:
    base = str(_POSITION_MANAGER_SERVICE_URL).rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _normalize_service_path(path: str) -> str:
    normalized_path = path.strip()
    if normalized_path.endswith("/") and normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")
    return normalized_path


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _position_manager_service_session() -> requests.Session:
    global _POSITION_MANAGER_SERVICE_SESSION
    session = _POSITION_MANAGER_SERVICE_SESSION
    if session is not None:
        return session
    with _POSITION_MANAGER_SERVICE_SESSION_LOCK:
        session = _POSITION_MANAGER_SERVICE_SESSION
        if session is not None:
            return session
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=_POSITION_MANAGER_SERVICE_POOL_CONNECTIONS,
            pool_maxsize=_POSITION_MANAGER_SERVICE_POOL_MAXSIZE,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _POSITION_MANAGER_SERVICE_SESSION = session
    return session


def _open_positions_cache_key(payload: dict) -> bool:
    return _as_bool(payload.get("include_unknown"), default=False)


def _get_cached_open_positions(
    payload: dict,
    *,
    max_age_sec: float,
) -> object | None:
    if max_age_sec <= 0.0:
        return None
    key = _open_positions_cache_key(payload)
    now = time.monotonic()
    with _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_LOCK:
        cached = _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE.get(key)
        if cached is None:
            return None
        ts, value = cached
        age = now - ts
        if age > max_age_sec:
            if max_age_sec <= _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC:
                _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE.pop(key, None)
            return None
    return copy.deepcopy(value)


def _set_cached_open_positions(payload: dict, value: object) -> None:
    if _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC <= 0.0:
        return
    key = _open_positions_cache_key(payload)
    with _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_LOCK:
        _POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE[key] = (
            time.monotonic(),
            copy.deepcopy(value),
        )


def _normalize_for_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_service_payload(path: str, payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    if "ok" not in payload:
        return payload.get("result", payload)
    if not bool(payload.get("ok")):
        msg = str(
            payload.get("error")
            or payload.get("detail")
            or payload.get("message")
            or payload.get("reason")
            or "service returned ok=false"
        )
        raise RuntimeError(f"position_manager service error for {path}: {msg}")
    if "result" in payload:
        return payload["result"]
    return payload


def _position_manager_service_call(path: str, payload: dict) -> object:
    normalized_path = _normalize_service_path(path)
    url = _position_manager_service_url(normalized_path)
    normalized_payload = _normalize_for_json(payload)
    session = _position_manager_service_session()
    if normalized_path == "/position/open_positions":
        response = session.get(
            url,
            params=normalized_payload,
            timeout=float(_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT),
        )
    else:
        response = session.post(
            url,
            json=normalized_payload,
            timeout=float(_POSITION_MANAGER_SERVICE_TIMEOUT),
            headers={"Content-Type": "application/json"},
        )
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict):
        return body
    raise RuntimeError(
        f"Unexpected position_manager service response type: {type(body).__name__}"
    )


def _position_manager_service_request(path: str, payload: dict) -> object | None:
    if not _position_manager_service_enabled():
        return None
    normalized_path = _normalize_service_path(path)
    if normalized_path == "/position/open_positions":
        cached = _get_cached_open_positions(
            payload,
            max_age_sec=_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC,
        )
        if cached is not None:
            return cached
    global _POSITION_MANAGER_SERVICE_NEXT_RETRY
    global _POSITION_MANAGER_SERVICE_ERROR_COUNT
    global _POSITION_MANAGER_SERVICE_LAST_ERROR
    global _POSITION_MANAGER_SERVICE_LAST_SUPPRESS_LOG_TS
    now_ts = time.monotonic()
    if now_ts < _POSITION_MANAGER_SERVICE_NEXT_RETRY:
        if normalized_path == "/position/open_positions":
            stale_cached = _get_cached_open_positions(
                payload,
                max_age_sec=_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC,
            )
            if stale_cached is not None:
                return stale_cached
        if _POSITION_MANAGER_SERVICE_FALLBACK_LOCAL:
            if now_ts - _POSITION_MANAGER_SERVICE_LAST_SUPPRESS_LOG_TS > 30.0:
                remain = max(0.0, _POSITION_MANAGER_SERVICE_NEXT_RETRY - now_ts)
                logging.debug(
                    "[POSITION] position_manager service retry cool-down remain=%.2fs errors=%d",
                    remain,
                    _POSITION_MANAGER_SERVICE_ERROR_COUNT,
                )
                _POSITION_MANAGER_SERVICE_LAST_SUPPRESS_LOG_TS = now_ts
            return None
        raise RuntimeError(
            "position_manager service temporarily unavailable"
            f" (retry in {max(0.0, _POSITION_MANAGER_SERVICE_NEXT_RETRY - now_ts):.1f}s)"
        )
    try:
        result = _extract_service_payload(path, _position_manager_service_call(path, payload))
        _POSITION_MANAGER_SERVICE_ERROR_COUNT = 0
        _POSITION_MANAGER_SERVICE_NEXT_RETRY = 0.0
        _POSITION_MANAGER_SERVICE_LAST_ERROR = None
        if normalized_path == "/position/open_positions":
            _set_cached_open_positions(payload, result)
        return result
    except Exception as exc:
        _POSITION_MANAGER_SERVICE_ERROR_COUNT += 1
        backoff = min(
            _POSITION_MANAGER_SERVICE_FAIL_BACKOFF_SEC
            * (2 ** (_POSITION_MANAGER_SERVICE_ERROR_COUNT - 1)),
            _POSITION_MANAGER_SERVICE_FAIL_BACKOFF_MAX_SEC,
        )
        _POSITION_MANAGER_SERVICE_NEXT_RETRY = now_ts + backoff
        _POSITION_MANAGER_SERVICE_LAST_ERROR = str(exc)
        logging.warning(
            "[POSITION] position_manager service call failed path=%s payload=%s err=%s",
            path,
            payload,
            exc,
        )
        if normalized_path == "/position/open_positions":
            stale_cached = _get_cached_open_positions(
                payload,
                max_age_sec=_POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC,
            )
            if stale_cached is not None:
                return stale_cached
        if not _POSITION_MANAGER_SERVICE_FALLBACK_LOCAL:
            raise
        return None


def _is_closed_db_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "closed database" in message
        or "connection is closed" in message
        or "closed" in message and "database" in message
    )

# Agent-generated client order ID prefixes (qr-...), used to classify pockets.
agent_client_prefixes = tuple(
    os.getenv("AGENT_CLIENT_PREFIXES", "qr-,qs-").split(",")
)
agent_client_prefixes = tuple(p for p in agent_client_prefixes if p)
if not agent_client_prefixes:
    agent_client_prefixes = ("qr-",)
# pockets that belong to this agent
agent_pockets = {"micro", "macro", "scalp", "scalp_fast"}

# Strategy tag normalization: map suffix/abbrev tags to canonical bases for EXIT filtering.
_CANONICAL_STRATEGY_TAGS = {
    # Core strategies
    "TrendMA",
    "Donchian55",
    "H1Momentum",
    "ImpulseRetrace",
    "RangeFader",
    "PulseBreak",
    "BB_RSI",
    "BB_RSI_Fast",
    "MomentumBurst",
    "MomentumPulse",
    "VolCompressionBreak",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MicroRangeBreak",
    "MicroVWAPBound",
    "MicroVWAPRevert",
    "TrendMomentumMicro",
    # Worker-only tags
    "trend_h1",
    "pullback_s5",
    "pullback_runner_s5",
    "impulse_break_s5",
    "impulse_retest_s5",
    "impulse_momentum_s5",
    "vwap_magnet_s5",
    "VolSpikeRider",
    "fast_scalp",
    "mm_lite",
    "manual_swing",
    "OnePipMakerS1",
    "LondonMomentum",
    "TechFusion",
    "MacroTechFusion",
    "MicroPullbackFib",
    "ScalpReversalNWave",
    "RangeCompressionBreak",
    "M1Scalper",
    "M1Scalper-M1",
}
_CANONICAL_TAGS_LOWER = {tag.lower(): tag for tag in _CANONICAL_STRATEGY_TAGS}
_TAG_ALIAS_PREFIXES = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "impulsere": "ImpulseRetrace",
    "impulser": "ImpulseRetrace",
    "rangefad": "RangeFader",
    "mmlite": "mm_lite",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
    "m1scalpe": "M1Scalper-M1",
    "m1scalper": "M1Scalper-M1",
    "m1scalp": "M1Scalper-M1",
}

_STRATEGY_POCKET_MAP = {
    # Macro
    "TrendMA": "macro",
    "Donchian55": "macro",
    "H1Momentum": "macro",
    "LondonMomentum": "macro",
    "trend_h1": "macro",
    "manual_swing": "macro",
    "OnePipMakerS1": "scalp",
    # Micro
    "MomentumBurst": "micro",
    "MomentumPulse": "micro",
    "VolCompressionBreak": "micro",
    "MicroMomentumStack": "micro",
    "MicroPullbackEMA": "micro",
    "MicroLevelReactor": "micro",
    "MicroRangeBreak": "micro",
    "MicroVWAPBound": "micro",
    "MicroVWAPRevert": "micro",
    "TrendMomentumMicro": "micro",
    "BB_RSI": "micro",
    "BB_RSI_Fast": "micro",
    # Scalp
    "ImpulseRetrace": "scalp",
    "RangeFader": "scalp",
    "PulseBreak": "scalp",
    "pullback_s5": "scalp",
    "pullback_runner_s5": "scalp",
    "impulse_break_s5": "scalp",
    "impulse_retest_s5": "scalp",
    "impulse_momentum_s5": "scalp",
    "vwap_magnet_s5": "scalp",
    "mm_lite": "scalp",
    "VolSpikeRider": "scalp",
    "fast_scalp": "scalp",
    "MacroTechFusion": "macro",
    "MicroPullbackFib": "micro",
    "ScalpReversalNWave": "scalp",
    "RangeCompressionBreak": "micro",
    "M1Scalper": "scalp",
    "M1Scalper-M1": "scalp",
}


def _normalize_strategy_tag(tag: object | None) -> str | None:
    if tag is None:
        return None
    tag_str = str(tag).strip()
    if not tag_str:
        return None
    lower = tag_str.lower()
    if lower in _CANONICAL_TAGS_LOWER:
        return _CANONICAL_TAGS_LOWER[lower]
    base = tag_str.split("-", 1)[0].strip()
    if base:
        base_lower = base.lower()
        if base_lower in _CANONICAL_TAGS_LOWER:
            return _CANONICAL_TAGS_LOWER[base_lower]
    for prefix, canonical in _TAG_ALIAS_PREFIXES.items():
        if lower.startswith(prefix):
            return canonical
    alnum = "".join(ch for ch in lower if ch.isalnum())
    if len(alnum) >= 4:
        best = None
        best_len = 0
        for base_lower, canonical in _CANONICAL_TAGS_LOWER.items():
            base_alnum = "".join(ch for ch in base_lower if ch.isalnum())
            if not base_alnum:
                continue
            if alnum.startswith(base_alnum) or base_alnum.startswith(alnum):
                if len(base_alnum) > best_len:
                    best = canonical
                    best_len = len(base_alnum)
        if best:
            return best
    return tag_str


def _normalize_regime(value: object | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


def _extract_regime(thesis: dict | None) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(thesis, dict):
        return None, None
    reg = thesis.get("regime")
    macro = None
    micro = None
    if isinstance(reg, dict):
        macro = reg.get("macro") or reg.get("macro_regime") or reg.get("reg_macro")
        micro = reg.get("micro") or reg.get("micro_regime") or reg.get("reg_micro")
    macro = macro or thesis.get("macro_regime") or thesis.get("reg_macro")
    micro = micro or thesis.get("micro_regime") or thesis.get("reg_micro")
    return _normalize_regime(macro), _normalize_regime(micro)


def _apply_strategy_tag_normalization(thesis: dict | None, raw_tag: object | None) -> tuple[dict | None, str | None]:
    """Ensure thesis carries normalized strategy_tag while preserving raw tag."""
    norm = _normalize_strategy_tag(raw_tag)
    if not norm and thesis is None:
        return thesis, None
    if thesis is None or not isinstance(thesis, dict):
        thesis = {}
    if raw_tag and norm and str(raw_tag) != norm:
        thesis.setdefault("strategy_tag_raw", str(raw_tag))
    if norm:
        thesis["strategy_tag"] = norm
    return thesis, norm


def _infer_pocket_from_client_id(client_id: object | None) -> str | None:
    if not client_id:
        return None
    cid = str(client_id).strip()
    if not cid:
        return None
    if cid.startswith(("qr-fast-", "qs-fast-")):
        return "scalp"
    try:
        import re
        match = re.match(r"^(?:qr|qs)-\d+-(micro|macro|scalp|scalp_fast)-", cid)
        if match:
            return match.group(1)
        match = re.match(r"^(?:qr|qs)-(micro|macro|scalp|scalp_fast)-\d+-", cid)
        if match:
            return match.group(1)
    except Exception:
        return None
    return None


def _infer_pocket_from_tag(tag: object | None) -> str | None:
    if tag is None:
        return None
    norm = _normalize_strategy_tag(tag)
    if norm and norm in _STRATEGY_POCKET_MAP:
        return _STRATEGY_POCKET_MAP[norm]
    raw = str(tag).strip().lower()
    if not raw:
        return None
    if raw.startswith(("mmlite", "mm_lite")):
        return "scalp"
    return None


def _normalize_pocket(pocket: str | None) -> str:
    """Map pocket values to a canonical set."""
    allowed = {"macro", "micro", "scalp", "scalp_fast", "manual"}
    if not pocket:
        return "manual"
    p = str(pocket).strip().lower()
    if p in ("", "unknown"):
        return "manual"
    return p if p in allowed else "manual"


def _resolve_pocket(
    pocket_raw: object | None,
    strategy_tag: object | None,
    client_id: object | None,
) -> str:
    pocket = _normalize_pocket(str(pocket_raw) if pocket_raw is not None else None)
    if pocket != "manual":
        return pocket
    inferred = _infer_pocket_from_client_id(client_id)
    if inferred:
        return inferred
    inferred = _infer_pocket_from_tag(strategy_tag)
    if inferred:
        return inferred
    return pocket


def _configure_sqlite(con: sqlite3.Connection) -> sqlite3.Connection:
    """Apply SQLite PRAGMAs to reduce lock contention."""
    try:
        con.execute(f"PRAGMA journal_mode={_DB_JOURNAL_MODE}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA busy_timeout={_DB_BUSY_TIMEOUT_MS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA synchronous={_DB_SYNCHRONOUS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA temp_store={_DB_TEMP_STORE}")
    except sqlite3.Error:
        pass
    return con


def _open_trades_db() -> sqlite3.Connection:
    con = sqlite3.connect(
        _DB, timeout=_DB_BUSY_TIMEOUT_MS / 1000, check_same_thread=False
    )
    con.row_factory = sqlite3.Row
    return _configure_sqlite(con)


@contextmanager
def _file_lock(path: pathlib.Path, timeout: float = _DB_FILE_LOCK_TIMEOUT):
    """
    Inter-process advisory lock to serialize schema/migration/writes.

    Prevents multiple worker processes from ALTER/INSERT at the same time,
    which is the main source of sqlite 'database is locked' at boot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Timed out acquiring file lock: {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _build_http_session() -> requests.Session:
    total = int(os.getenv("POSITION_MANAGER_HTTP_RETRY_TOTAL", "3"))
    backoff = float(os.getenv("POSITION_MANAGER_HTTP_BACKOFF", "0.6"))
    retry = Retry(
        total=total,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=frozenset({"GET", "POST"}),
        backoff_factor=backoff,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=8,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _open_orders_db_readonly(
    timeout_sec: float = _ORDERS_DB_READ_TIMEOUT_SEC,
) -> sqlite3.Connection | None:
    if not _ORDERS_DB.exists():
        return None
    timeout_sec = max(0.05, float(timeout_sec))
    con = sqlite3.connect(
        f"file:{_ORDERS_DB}?mode=ro",
        uri=True,
        timeout=timeout_sec,
    )
    con.row_factory = sqlite3.Row
    try:
        con.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except sqlite3.Error:
        pass
    return con


def _ensure_orders_db() -> sqlite3.Connection:
    """orders.db が存在しない/テーブル欠損時に安全に初期化する。"""
    _ORDERS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_ORDERS_DB, timeout=_DB_BUSY_TIMEOUT_MS / 1000)
    con.row_factory = sqlite3.Row
    _configure_sqlite(con)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT,
          pocket TEXT,
          instrument TEXT,
          side TEXT,
          units INTEGER,
          sl_price REAL,
          tp_price REAL,
          client_order_id TEXT,
          status TEXT,
          attempt INTEGER,
          stage_index INTEGER,
          ticket_id TEXT,
          executed_price REAL,
          error_code TEXT,
          error_message TEXT,
          request_json TEXT,
          response_json TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(client_order_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
    con.commit()
    return con


def _tx_sort_key(tx: dict) -> int:
    try:
        return int(tx.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(ts: str) -> datetime:
    """OANDAのISO文字列（ナノ秒精度）をPythonのdatetimeに変換する。"""
    if not ts:
        return datetime.now(timezone.utc)

    ts = ts.replace("Z", "+00:00")
    if "." not in ts:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    head, tail = ts.split(".", 1)
    tz = ""
    if "+" in tail:
        frac, tz = tail.split("+", 1)
        tz = "+" + tz
    elif "-" in tail[6:]:
        frac, tz = tail.split("-", 1)
        tz = "-" + tz
    else:
        frac, tz = tail, ""

    frac = frac[:6].ljust(6, "0")
    dt = datetime.fromisoformat(f"{head}.{frac}{tz}")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _extract_cashflow_tx(
    tx: dict,
) -> tuple[int, str, float, float | None, str, str | None] | None:
    tx_type = str(tx.get("type") or "")
    if tx_type not in _CASHFLOW_TYPES:
        return None
    try:
        tx_id = int(tx.get("id") or 0)
    except (TypeError, ValueError):
        return None
    if tx_id <= 0:
        return None
    amount = _coerce_float(
        tx.get("amount") or tx.get("transferAmount") or tx.get("amountTransferred")
    )
    if amount is None:
        return None
    balance = _coerce_float(tx.get("accountBalance"))
    ts = tx.get("time") or tx.get("timestamp")
    if not ts:
        return None
    funding_reason = tx.get("fundingReason") or tx.get("reason")
    return (
        tx_id,
        str(ts),
        float(amount),
        float(balance) if balance is not None else None,
        tx_type,
        str(funding_reason) if funding_reason is not None else None,
    )


def _load_latest_metric(metric: str) -> float | None:
    if not _METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(_METRICS_DB, timeout=_METRICS_READ_TIMEOUT_SEC)
        cur = con.execute(
            "SELECT value FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return float(row[0])
    except Exception:
        return None


def _load_first_metric_since(metric: str, since_ts: str) -> float | None:
    if not _METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(_METRICS_DB, timeout=_METRICS_READ_TIMEOUT_SEC)
        cur = con.execute(
            "SELECT value FROM metrics WHERE metric = ? AND ts >= ? ORDER BY ts ASC LIMIT 1",
            (metric, since_ts),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return float(row[0])
    except Exception:
        return None


def _load_metric_series(metric: str, since_ts: str) -> list[tuple[datetime, float]]:
    if not _METRICS_DB.exists():
        return []
    try:
        con = sqlite3.connect(_METRICS_DB, timeout=_METRICS_READ_TIMEOUT_SEC)
        cur = con.execute(
            "SELECT ts, value FROM metrics WHERE metric = ? AND ts >= ? ORDER BY ts ASC",
            (metric, since_ts),
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return []
    series: list[tuple[datetime, float]] = []
    for ts_raw, val_raw in rows:
        if ts_raw is None:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_raw))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            val = float(val_raw)
        except (TypeError, ValueError):
            continue
        series.append((dt, val))
    return series


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _downsample_rows(rows: list[tuple], max_points: int) -> list[tuple]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    step = max(1, int(math.ceil(len(rows) / float(max_points))))
    sampled = rows[::step]
    if sampled and sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled


def _load_latest_candles(tf: str) -> list[tuple[datetime, float, float, float, float]]:
    path = _CANDLE_DIR / f"candles_{tf}_latest.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = payload.get("candles") or []
    candles: list[tuple[datetime, float, float, float, float]] = []
    for item in raw:
        ts = item.get("time") or item.get("timestamp")
        if not ts:
            continue
        try:
            dt = _parse_timestamp(ts)
        except Exception:
            continue
        if not isinstance(dt, datetime):
            continue
        open_v = _coerce_float(item.get("open"))
        high_v = _coerce_float(item.get("high"))
        low_v = _coerce_float(item.get("low"))
        close_v = _coerce_float(item.get("close"))
        if open_v is None or high_v is None or low_v is None or close_v is None:
            mid = item.get("mid") or {}
            if isinstance(mid, dict):
                open_v = open_v if open_v is not None else _coerce_float(mid.get("o"))
                high_v = high_v if high_v is not None else _coerce_float(mid.get("h"))
                low_v = low_v if low_v is not None else _coerce_float(mid.get("l"))
                close_v = close_v if close_v is not None else _coerce_float(mid.get("c"))
        if open_v is None or high_v is None or low_v is None or close_v is None:
            continue
        candles.append((dt, open_v, high_v, low_v, close_v))
    candles.sort(key=lambda row: row[0])
    return candles


def _build_chart_data(
    *,
    now: datetime,
    year_start: datetime,
    closed_events: list[tuple[datetime, float, float]],
    con: sqlite3.Connection,
) -> dict:
    if not _CHARTS_ENABLED:
        return {}

    def _to_ms(dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    perf_specs = {
        "5m": {"delta": timedelta(minutes=5), "max_points": min(_CHART_MAX_POINTS, 120)},
        "1h": {"delta": timedelta(hours=1), "max_points": min(_CHART_MAX_POINTS, 200)},
        "1d": {"delta": timedelta(days=1), "max_points": _CHART_MAX_POINTS},
        "1w": {"delta": timedelta(days=7), "max_points": _CHART_MAX_POINTS},
        "1m": {"delta": timedelta(days=30), "max_points": _CHART_MAX_POINTS},
        "1y": {"start": year_start, "max_points": max(_CHART_MAX_POINTS, 320)},
    }

    price_specs = {
        "5m": {"delta": timedelta(minutes=5), "tf": "M1", "max_points": min(_CHART_PRICE_MAX_POINTS, 120)},
        "1h": {"delta": timedelta(hours=1), "tf": "M1", "max_points": min(_CHART_PRICE_MAX_POINTS, 200)},
        "1d": {"delta": timedelta(days=1), "tf": "M1", "max_points": _CHART_PRICE_MAX_POINTS},
        "1w": {"delta": timedelta(days=7), "tf": "H1", "max_points": _CHART_PRICE_MAX_POINTS},
        "1m": {"delta": timedelta(days=30), "tf": "H4", "max_points": _CHART_PRICE_MAX_POINTS},
        "1y": {"start": year_start, "tf": "H4", "max_points": _CHART_PRICE_MAX_POINTS},
    }

    closed_events.sort(key=lambda row: row[0])
    event_times = [row[0] for row in closed_events]
    cum_jpy: list[float] = []
    cum_pips: list[float] = []
    total_jpy = 0.0
    total_pips = 0.0
    for _, jpy, pips in closed_events:
        total_jpy += float(jpy)
        total_pips += float(pips)
        cum_jpy.append(total_jpy)
        cum_pips.append(total_pips)

    def _series_since(start_dt: datetime) -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]], int]:
        if not event_times:
            return [], [], 0
        idx = bisect_left(event_times, start_dt)
        base_jpy = cum_jpy[idx - 1] if idx > 0 else 0.0
        base_pips = cum_pips[idx - 1] if idx > 0 else 0.0
        jpy_points: list[tuple[datetime, float]] = []
        pips_points: list[tuple[datetime, float]] = []
        if idx < len(event_times):
            jpy_points.append((start_dt, 0.0))
            pips_points.append((start_dt, 0.0))
        for i in range(idx, len(event_times)):
            jpy_points.append((event_times[i], cum_jpy[i] - base_jpy))
            pips_points.append((event_times[i], cum_pips[i] - base_pips))
        if jpy_points:
            last_val_jpy = jpy_points[-1][1]
            last_val_pips = pips_points[-1][1]
            if event_times[-1] < now:
                jpy_points.append((now, last_val_jpy))
                pips_points.append((now, last_val_pips))
        return jpy_points, pips_points, max(0, len(event_times) - idx)

    perf_ranges: dict[str, dict] = {}
    for key, spec in perf_specs.items():
        start_dt = spec.get("start") or (now - spec["delta"])
        jpy_points, pips_points, trade_count = _series_since(start_dt)
        if not jpy_points:
            perf_ranges[key] = {
                "label": key,
                "start_ts": _to_ms(start_dt),
                "end_ts": _to_ms(now),
                "pnl_jpy": [],
                "pnl_pips": [],
                "trades": trade_count,
            }
            continue
        jpy_points = _downsample_rows(jpy_points, spec["max_points"])
        pips_points = _downsample_rows(pips_points, spec["max_points"])
        perf_ranges[key] = {
            "label": key,
            "start_ts": _to_ms(start_dt),
            "end_ts": _to_ms(now),
            "pnl_jpy": [[_to_ms(dt), round(val, 2)] for dt, val in jpy_points],
            "pnl_pips": [[_to_ms(dt), round(val, 2)] for dt, val in pips_points],
            "trades": trade_count,
        }

    def _marker_label(
        *,
        kind: str,
        side: str,
        pocket: str,
        worker: str | None,
        strategy: str | None,
        ts: datetime,
        price: float,
        units_abs: int | None,
    ) -> str:
        ts_label = ts.astimezone(_JST).strftime("%Y-%m-%d %H:%M JST")
        parts = [kind.capitalize(), side.capitalize(), pocket]
        if worker:
            parts.append(worker)
        if strategy:
            parts.append(strategy)
        parts.append(ts_label)
        parts.append(f"@ {price:.3f}")
        if units_abs:
            parts.append(f"{int(units_abs)}u")
        return " ".join(p for p in parts if p)

    price_ranges: dict[str, dict] = {}
    earliest_candidates: list[datetime] = []
    for spec in price_specs.values():
        start_dt = spec.get("start")
        if isinstance(start_dt, datetime):
            earliest_candidates.append(start_dt)
        else:
            earliest_candidates.append(now - spec["delta"])
    earliest_price_start = min(earliest_candidates) if earliest_candidates else now - timedelta(days=1)
    markers: list[dict] = []
    agent_prefixes = ("qr-", "qs-")

    def _is_agent_trade(pocket: str, client_id: str | None) -> bool:
        if pocket in _KNOWN_POCKETS:
            return True
        if client_id:
            return str(client_id).startswith(agent_prefixes)
        return False

    try:
        cur = con.execute(
            "SELECT entry_time, close_time, entry_price, close_price, fill_price, "
            "units, pocket, strategy_tag, strategy, client_order_id, entry_thesis "
            "FROM trades WHERE entry_time IS NOT NULL OR close_time IS NOT NULL"
        )
        rows = cur.fetchall()
    except Exception:
        rows = []

    for row in rows:
        pocket = (row["pocket"] or "unknown").lower()
        client_id = row["client_order_id"]
        if pocket == _MANUAL_POCKET_NAME or not _is_agent_trade(pocket, client_id):
            continue
        units = _coerce_float(row["units"]) or 0.0
        side = "long" if units > 0 else "short" if units < 0 else "flat"
        if side == "flat":
            continue
        strategy = row["strategy_tag"] or row["strategy"]
        strategy_label = None
        if strategy is not None:
            strategy_label = str(strategy).strip() or None
        thesis = row["entry_thesis"]
        if isinstance(thesis, str) and thesis:
            try:
                thesis = json.loads(thesis)
            except Exception:
                thesis = None
        if not isinstance(thesis, dict):
            thesis = None
        worker = None
        if isinstance(thesis, dict):
            worker = thesis.get("worker_id") or thesis.get("worker")
            if worker:
                worker = str(worker)
            else:
                worker = None
            if not strategy_label:
                thesis_tag = thesis.get("strategy_tag") or thesis.get("strategy")
                if thesis_tag is not None:
                    strategy_label = str(thesis_tag).strip() or None
        if not worker and strategy_label:
            worker = _normalize_worker_tag(strategy_label) or strategy_label
        entry_time = row["entry_time"]
        close_time = row["close_time"]
        entry_price = _coerce_float(row["entry_price"]) or _coerce_float(row["fill_price"])
        close_price = _coerce_float(row["close_price"])
        units_abs = int(abs(units)) if units else None

        if entry_time and entry_price is not None:
            try:
                entry_dt = _parse_timestamp(entry_time)
            except Exception:
                entry_dt = None
            if entry_dt and entry_dt >= earliest_price_start:
                markers.append(
                    {
                        "ts_ms": _to_ms(entry_dt),
                        "price": round(entry_price, 5),
                        "side": side,
                        "kind": "entry",
                        "pocket": pocket,
                        "worker": worker,
                        "strategy": strategy_label,
                        "units_abs": units_abs,
                        "label": _marker_label(
                            kind="entry",
                            side=side,
                            pocket=pocket,
                            worker=worker,
                            strategy=strategy_label,
                            ts=entry_dt,
                            price=entry_price,
                            units_abs=units_abs,
                        ),
                    }
                )
        if close_time and close_price is not None:
            try:
                close_dt = _parse_timestamp(close_time)
            except Exception:
                close_dt = None
            if close_dt and close_dt >= earliest_price_start:
                markers.append(
                    {
                        "ts_ms": _to_ms(close_dt),
                        "price": round(close_price, 5),
                        "side": side,
                        "kind": "exit",
                        "pocket": pocket,
                        "worker": worker,
                        "strategy": strategy_label,
                        "units_abs": units_abs,
                        "label": _marker_label(
                            kind="exit",
                            side=side,
                            pocket=pocket,
                            worker=worker,
                            strategy=strategy_label,
                            ts=close_dt,
                            price=close_price,
                            units_abs=units_abs,
                        ),
                    }
                )

    markers.sort(key=lambda m: m["ts_ms"])
    if _CHART_MARKERS_MAX > 0 and len(markers) > _CHART_MARKERS_MAX:
        markers = markers[-_CHART_MARKERS_MAX :]

    for key, spec in price_specs.items():
        start_dt = spec.get("start") or (now - spec["delta"])
        candles = _load_latest_candles(spec["tf"])
        if candles:
            candles = [row for row in candles if row[0] >= start_dt]
        candles = _downsample_rows(candles, spec["max_points"])
        range_start_ms = _to_ms(start_dt)
        range_end_ms = _to_ms(now)
        if candles:
            range_start_ms = _to_ms(candles[0][0])
            range_end_ms = _to_ms(candles[-1][0])
            start_label = candles[0][0].astimezone(_JST).strftime("%m/%d %H:%M")
            end_label = candles[-1][0].astimezone(_JST).strftime("%m/%d %H:%M")
            range_label = f"{start_label} - {end_label} JST"
        else:
            range_label = None
        range_markers = [
            m for m in markers if range_start_ms <= m["ts_ms"] <= range_end_ms
        ]
        if _CHART_MARKERS_MAX > 0 and len(range_markers) > _CHART_MARKERS_MAX:
            range_markers = range_markers[-_CHART_MARKERS_MAX :]
        price_ranges[key] = {
            "label": key,
            "tf": spec["tf"],
            "start_ts": range_start_ms,
            "end_ts": range_end_ms,
            "candles": [
                [_to_ms(dt), round(o, 5), round(h, 5), round(l, 5), round(c, 5)]
                for dt, o, h, l, c in candles
            ],
            "markers": range_markers,
            "range_label": range_label,
        }

    has_perf = any(r.get("pnl_jpy") for r in perf_ranges.values())
    has_price = any(r.get("candles") for r in price_ranges.values())
    return {
        "available": bool(has_perf or has_price),
        "generated_at": now.isoformat(),
        "performance": {
            "default_range": "1d",
            "ranges": perf_ranges,
        },
        "price": {
            "default_range": "1d",
            "ranges": price_ranges,
        },
    }


class PositionManager:
    def __init__(self):
        self._db_reopen_failures: int = 0
        self._db_reopen_block_until: float = 0.0
        self.con = _open_trades_db()
        needs_bootstrap = not _DB.exists()
        try:
            with _file_lock(_DB_LOCK_PATH):
                self._ensure_schema_with_retry()
        except TimeoutError:
            if needs_bootstrap:
                raise
            logging.warning(
                "[PositionManager] schema lock busy; skipping schema check to avoid startup timeout"
            )
        # Must be ready before any backfill that calls OANDA endpoints.
        self._http = _build_http_session()
        try:
            self._maybe_backfill_attribution()
        except Exception as exc:
            logging.warning("[PositionManager] attribution backfill skipped: %s", exc)
        try:
            self._maybe_backfill_cashflows()
        except Exception as exc:
            logging.warning("[PositionManager] cashflow backfill skipped: %s", exc)
        self._last_tx_id = self._get_last_transaction_id_with_retry()
        self._pocket_cache: dict[str, str] = {}
        self._client_cache: dict[str, str] = {}
        self._last_positions: dict[str, dict] = {}
        self._last_positions_ts: float = 0.0
        self._open_trade_failures: int = 0
        self._next_open_fetch_after: float = 0.0
        self._open_positions_lock = threading.Lock()
        self._sync_trades_lock = threading.Lock()
        self._last_sync_cache: list[dict] = []
        self._last_sync_cache_ts: float = 0.0
        self._last_sync_cache_window_sec: float = 1.2
        self._last_positions_meta: dict | None = None
        self._entry_meta_cache: dict[str, dict] = {}
        self._last_sync_fetch_meta: dict[str, float | int] = {}
        self._last_sync_parse_meta: dict[str, float | int] = {}
        self._last_sync_breakdown: dict[str, float | int] = {}

    def _reopen_trades_connection(self) -> None:
        now = time.monotonic()
        db_reopen_block_until = getattr(self, "_db_reopen_block_until", 0.0)
        db_reopen_failures = getattr(self, "_db_reopen_failures", 0)
        if now < db_reopen_block_until:
            raise sqlite3.OperationalError(
                "trades DB reconnect temporarily blocked due to repeated failures"
            )

        backoff = min(
            _DB_REOPEN_MAX_INTERVAL_SEC,
            _DB_REOPEN_MIN_INTERVAL_SEC * (2**db_reopen_failures),
        )

        for attempt in range(_DB_REOPEN_ATTEMPTS):
            previous = getattr(self, "con", None)
            if previous is not None:
                try:
                    previous.close()
                except Exception:
                    pass
            self.con = None

            try:
                self.con = _open_trades_db()
                with _file_lock(_DB_LOCK_PATH):
                    self._ensure_schema_with_retry()
                self._db_reopen_failures = 0
                self._db_reopen_block_until = 0.0
                return
            except Exception as exc:
                db_reopen_failures += 1
                self._db_reopen_failures = db_reopen_failures
                self._db_reopen_block_until = time.monotonic() + backoff
                if self.con is not None:
                    try:
                        self.con.close()
                    except Exception:
                        pass
                    self.con = None

                if attempt >= _DB_REOPEN_ATTEMPTS - 1:
                    raise

                logging.warning(
                    "[PositionManager] failed to reopen trades DB (attempt=%s): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _DB_REOPEN_MAX_INTERVAL_SEC)

    def _ensure_connection_open(self) -> None:
        db_reopen_block_until = getattr(self, "_db_reopen_block_until", 0.0)
        if time.monotonic() < db_reopen_block_until:
            raise sqlite3.ProgrammingError(
                "trades DB temporarily unavailable after repeated reconnect failures"
            )

        if self.con is None:
            self._reopen_trades_connection()
            return

        try:
            self.con.execute("SELECT 1")
            return
        except (sqlite3.ProgrammingError, sqlite3.OperationalError) as exc:
            if not _is_closed_db_error(exc):
                raise
            logging.warning("[PositionManager] detected closed trades DB, reopening")
            self._reopen_trades_connection()

    def get_last_sync_breakdown(self) -> dict[str, float | int]:
        return dict(self._last_sync_breakdown or {})

    @staticmethod
    def _infer_strategy_tag(thesis: dict | None, client_id: str | None, pocket: str | None) -> str | None:
        """
        推定ルール:
        - thesis.strategy_tag / strategy を最優先
        - client_id から既知プレフィックスや qr-<ts>-<pocket>-<tag> / qr-<pocket>-<ts>-<tag> を抽出
        """
        if thesis and isinstance(thesis, dict):
            tag = thesis.get("strategy_tag") or thesis.get("strategy")
            if tag:
                return _normalize_strategy_tag(tag)
        if not client_id:
            return None
        cid = str(client_id).strip()
        try:
            import re

            pocket_tokens = {"micro", "macro", "scalp", "scalp_fast", "event", "hybrid", "mar"}
            suffix_token_re = re.compile(r"^[sb]?[a-f0-9]{6,16}$")

            # Known prefixes
            if cid.startswith("qr-fast-"):
                return _normalize_strategy_tag("fast_scalp")
            if cid.startswith("qr-pullback-s5-"):
                return _normalize_strategy_tag("pullback_s5")

            if cid.startswith("qr-"):
                body = cid[3:]
                if body and body[0].isdigit():
                    tokens = body.split("-")
                    tokens = tokens[1:]  # remove ts
                    if not tokens:
                        return None
                    if suffix_token_re.match(tokens[-1]):
                        tokens = tokens[:-1]
                    if not tokens:
                        return None
                    # qr-<ts>-<pocket>-...  (old convention)
                    if tokens[0] in pocket_tokens:
                        tokens = tokens[1:]
                        if not tokens:
                            return None
                        # qr-<ts>-<pocket>-<ts>-<tag> (legacy form)
                        if tokens[0].isdigit():
                            tokens = tokens[1:]
                        if not tokens:
                            return None
                    inferred = "-".join(tokens)
                    if inferred:
                        return _normalize_strategy_tag(inferred)

            if cid.startswith("qs-"):
                body = cid[3:]
                if body and body[0].isdigit():
                    tokens = body.split("-")
                    tokens = tokens[1:]  # remove ts
                    if not tokens:
                        return None
                    if suffix_token_re.match(tokens[-1]):
                        tokens = tokens[:-1]
                    if not tokens:
                        return None
                    if tokens[0] in pocket_tokens:
                        tokens = tokens[1:]
                        if not tokens:
                            return None
                        if tokens[0].isdigit():
                            tokens = tokens[1:]
                        if not tokens:
                            return None
                    inferred = "-".join(tokens)
                    if inferred:
                        return _normalize_strategy_tag(inferred)

            # qr-<pocket>-<ts>-<tag...>
            if cid.startswith("qr-") and cid.count("-") >= 4:
                m2 = re.match(r"^qr-(micro|macro|scalp|scalp_fast|event|hybrid|mar)-\d+-([^-]+)", cid)
                if m2:
                    return _normalize_strategy_tag(m2.group(2))

            # qr-<ts>-<pocket>-<tag...>
            if cid.startswith("qr-") and cid.count("-") >= 3:
                m = re.match(r"^qr-\d+-(micro|macro|scalp|scalp_fast|event|hybrid|mar)-(.+)$", cid)
                if m:
                    return _normalize_strategy_tag(m.group(2))
            # fallback: qr-<word>-<rest>
            m3 = re.match(r"^qr-([a-zA-Z0-9_]+)-(.+)$", cid)
            if m3 and m3.group(1) not in {"micro", "macro", "scalp", "scalp_fast", "event", "hybrid", "mar", "lm"}:
                return _normalize_strategy_tag(m3.group(1))
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_pocket(pocket: str | None) -> str:
        """Map pocket values to a canonical set."""
        return _normalize_pocket(pocket)

    def _ensure_schema_with_retry(self):
        self._ensure_connection_open()
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self._ensure_schema()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP)
        # 最後のリトライで例外を伝播
        self._ensure_schema()

    def _ensure_schema(self):
        self._ensure_connection_open()
        # trades テーブルが存在しない場合のベース定義
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              transaction_id INTEGER,
              ticket_id TEXT,
              pocket TEXT,
              instrument TEXT,
              units INTEGER,
              closed_units INTEGER,
              entry_price REAL,
              close_price REAL,
              fill_price REAL,
              pl_pips REAL,
              realized_pl REAL,
              commission REAL,
              financing REAL,
              entry_time TEXT,
              open_time TEXT,
              close_time TEXT,
              close_reason TEXT,
              state TEXT,
              updated_at TEXT,
              version TEXT DEFAULT 'v1',
              unrealized_pl REAL
            )
            """
        )

        # 欠損カラムを追加（既存データを保持する）
        existing = {row[1] for row in self.con.execute("PRAGMA table_info(trades)")}
        columns: dict[str, str] = {
            "transaction_id": "INTEGER",
            "ticket_id": "TEXT",
            "pocket": "TEXT",
            "instrument": "TEXT",
            "units": "INTEGER",
            "closed_units": "INTEGER",
            "entry_price": "REAL",
            "close_price": "REAL",
            "fill_price": "REAL",
            "pl_pips": "REAL",
            "realized_pl": "REAL",
            "commission": "REAL",
            "financing": "REAL",
            "entry_time": "TEXT",
            "open_time": "TEXT",
            "close_time": "TEXT",
            "close_reason": "TEXT",
            "state": "TEXT",
            "updated_at": "TEXT",
            "version": "TEXT DEFAULT 'v1'",
            "unrealized_pl": "REAL",
            # strategy attribution
            "strategy": "TEXT",
            "client_order_id": "TEXT",
            "strategy_tag": "TEXT",
            "entry_thesis": "TEXT",
            "macro_regime": "TEXT",
            "micro_regime": "TEXT",
        }
        added_cols: set[str] = set()
        for name, ddl in columns.items():
            if name not in existing:
                self.con.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")
                added_cols.add(name)
        # 既存データで strategy が空の場合は strategy_tag をコピー
        if "strategy" in added_cols or "strategy_tag" in added_cols:
            try:
                self.con.execute(
                    "UPDATE trades SET strategy = strategy_tag WHERE (strategy IS NULL OR strategy = '') AND strategy_tag IS NOT NULL"
                )
            except sqlite3.Error:
                pass

        if self._has_ticket_unique_constraint():
            self._migrate_remove_ticket_unique()

        # 旧ユニークインデックス（ticket_id 単独）は部分決済を上書きしてしまうため削除し、
        # (transaction_id, ticket_id) の複合ユニークへ移行、ticket_id は非ユニーク索引に変更
        try:
            self.con.execute("DROP INDEX IF EXISTS idx_trades_ticket")
        except Exception:
            pass
        self.con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uniq_trades_tx_trade ON trades(transaction_id, ticket_id)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time)"
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS cashflows (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              transaction_id INTEGER UNIQUE,
              time TEXT,
              amount REAL,
              balance REAL,
              type TEXT,
              funding_reason TEXT
            )
            """
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_cashflows_time ON cashflows(time)"
        )
        self._commit_with_retry()

    def _maybe_backfill_attribution(self) -> None:
        if not _BACKFILL_ENABLED:
            return
        now = time.time()
        if _BACKFILL_TTL_SEC > 0 and _BACKFILL_MARKER.exists():
            try:
                data = json.loads(_BACKFILL_MARKER.read_text(encoding="utf-8"))
                last_ts = float(data.get("ts") or 0.0)
                if last_ts > 0 and now - last_ts < _BACKFILL_TTL_SEC:
                    return
            except Exception:
                pass
        try:
            with _file_lock(_DB_LOCK_PATH):
                updated = self._backfill_missing_attribution()
        except TimeoutError:
            logging.warning("[PositionManager] attribution backfill lock busy; skip")
            return
        try:
            _BACKFILL_MARKER.parent.mkdir(parents=True, exist_ok=True)
            _BACKFILL_MARKER.write_text(
                json.dumps({"ts": now, "updated": updated}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _maybe_backfill_cashflows(self) -> None:
        if not _CASHFLOW_BACKFILL_ENABLED:
            return
        now_ts = time.time()
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat().replace("+00:00", "Z")
        now_jst = now_dt.astimezone(_JST)
        year_start_jst = now_jst.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        year_start_utc = year_start_jst.astimezone(timezone.utc)
        year_start_iso = year_start_utc.isoformat().replace("+00:00", "Z")

        # Skip if we already ran recently for this YTD window.
        if _CASHFLOW_BACKFILL_TTL_SEC > 0 and _CASHFLOW_BACKFILL_MARKER.exists():
            try:
                data = json.loads(
                    _CASHFLOW_BACKFILL_MARKER.read_text(encoding="utf-8")
                )
                last_ts = float(data.get("ts") or 0.0)
                marker_ver = int(data.get("version") or 0)
                marker_from = str(data.get("from") or "")
                if (
                    last_ts > 0
                    and now_ts - last_ts < _CASHFLOW_BACKFILL_TTL_SEC
                    and marker_ver == _CASHFLOW_BACKFILL_MARKER_VERSION
                    and marker_from == year_start_iso
                ):
                    return
            except Exception:
                pass

        def _flush_batch(batch: list[tuple]) -> int:
            if not batch:
                return 0
            try:
                with _file_lock(_DB_LOCK_PATH):
                    self._executemany_with_retry(
                        """
                        INSERT OR IGNORE INTO cashflows (
                            transaction_id, time, amount, balance, type, funding_reason
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    self._commit_with_retry()
                inserted = len(batch)
                batch.clear()
                return inserted
            except TimeoutError:
                logging.warning("[PositionManager] cashflow backfill lock busy; skip batch")
                batch.clear()
                return 0

        inserted_file = 0
        inserted_oanda = 0
        batch: list[tuple] = []
        paths = sorted(glob.glob(_CASHFLOW_BACKFILL_GLOB))
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if "TRANSFER_FUNDS" not in line and "CASH_TRANSFER" not in line:
                            continue
                        try:
                            tx = json.loads(line)
                        except Exception:
                            continue
                        record = _extract_cashflow_tx(tx)
                        if not record:
                            continue
                        batch.append(record)
                        if len(batch) >= 500:
                            inserted_file += _flush_batch(batch)
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.warning(
                    "[PositionManager] cashflow backfill read failed: %s", exc
                )
        if batch:
            inserted_file += _flush_batch(batch)

        # Backfill directly from OANDA using the transaction type filter (YTD in JST). This keeps
        # cashflow metrics and "balance growth ex cashflow" accurate without requiring local jsonl archives.
        if _CASHFLOW_BACKFILL_OANDA_ENABLED and _CASHFLOW_OANDA_QUERY_TYPES:
            url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
            for tx_type in _CASHFLOW_OANDA_QUERY_TYPES:
                try:
                    summary = self._request_json(
                        url,
                        params={
                            "from": year_start_iso,
                            "to": now_iso,
                            "pageSize": 1000,
                            "type": tx_type,
                        },
                    ) or {}
                except requests.RequestException as exc:
                    logging.warning(
                        "[PositionManager] cashflow backfill summary failed type=%s: %s",
                        tx_type,
                        exc,
                    )
                    self._reset_http_if_needed(exc)
                    continue

                # Some OANDA responses may include transactions directly.
                txs = summary.get("transactions") or []
                if isinstance(txs, list):
                    for tx in txs:
                        if not isinstance(tx, dict):
                            continue
                        record = _extract_cashflow_tx(tx)
                        if record:
                            batch.append(record)
                            if len(batch) >= 200:
                                inserted_oanda += _flush_batch(batch)

                pages = summary.get("pages") or []
                if isinstance(pages, list):
                    for page_url in pages:
                        if not isinstance(page_url, str) or not page_url:
                            continue
                        try:
                            payload = self._request_json(page_url) or {}
                        except requests.RequestException as exc:
                            logging.warning(
                                "[PositionManager] cashflow backfill page failed: %s",
                                exc,
                            )
                            self._reset_http_if_needed(exc)
                            continue
                        for tx in payload.get("transactions") or []:
                            if not isinstance(tx, dict):
                                continue
                            record = _extract_cashflow_tx(tx)
                            if record:
                                batch.append(record)
                                if len(batch) >= 200:
                                    inserted_oanda += _flush_batch(batch)
            if batch:
                inserted_oanda += _flush_batch(batch)
        try:
            _CASHFLOW_BACKFILL_MARKER.parent.mkdir(parents=True, exist_ok=True)
            _CASHFLOW_BACKFILL_MARKER.write_text(
                json.dumps(
                    {
                        "ts": now_ts,
                        "version": _CASHFLOW_BACKFILL_MARKER_VERSION,
                        "inserted_file": inserted_file,
                        "inserted_oanda": inserted_oanda,
                        "from": year_start_iso,
                        "to": now_iso,
                        "paths": len(paths),
                    }
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _backfill_missing_attribution(self) -> int:
        try:
            con = sqlite3.connect(_DB, timeout=_DB_BUSY_TIMEOUT_MS / 1000.0)
            con.row_factory = sqlite3.Row
            _configure_sqlite(con)
        except sqlite3.Error as exc:
            logging.warning("[PositionManager] backfill open failed: %s", exc)
            return 0
        rows = con.execute(
            """
            SELECT id, pocket, client_order_id, strategy_tag, strategy, entry_thesis, macro_regime, micro_regime
            FROM trades
            WHERE (strategy_tag IS NULL OR strategy_tag = '')
               OR (pocket IS NULL OR pocket = '' OR pocket = 'unknown')
               OR (macro_regime IS NULL OR macro_regime = '')
               OR (micro_regime IS NULL OR micro_regime = '')
            ORDER BY id DESC
            LIMIT ?
            """,
            (_BACKFILL_MAX_ROWS,),
        ).fetchall()
        updates = []
        for row in rows:
            current_pocket = row["pocket"] or ""
            current_tag = row["strategy_tag"] or row["strategy"]
            entry_thesis_obj = None
            if row["entry_thesis"]:
                try:
                    payload = json.loads(row["entry_thesis"])
                    if isinstance(payload, dict):
                        entry_thesis_obj = payload
                        if not current_tag:
                            current_tag = payload.get("strategy_tag") or payload.get("strategy")
                except Exception:
                    entry_thesis_obj = None
            norm_tag = _normalize_strategy_tag(current_tag)
            final_tag = norm_tag or current_tag
            if final_tag:
                final_tag = str(final_tag).strip() or None
            final_strategy = row["strategy"] or final_tag
            if final_strategy:
                final_strategy = str(final_strategy).strip() or None
            final_pocket = _resolve_pocket(
                current_pocket,
                final_tag,
                row["client_order_id"],
            )
            macro_regime = row["macro_regime"]
            micro_regime = row["micro_regime"]
            reg_macro, reg_micro = _extract_regime(entry_thesis_obj)
            if not macro_regime:
                macro_regime = reg_macro
            if not micro_regime:
                micro_regime = reg_micro
            if (
                str(current_pocket or "") == str(final_pocket or "")
                and (row["strategy_tag"] or "") == (final_tag or "")
                and (row["strategy"] or "") == (final_strategy or "")
                and (row["macro_regime"] or "") == (macro_regime or "")
                and (row["micro_regime"] or "") == (micro_regime or "")
            ):
                continue
            updates.append(
                (final_pocket, final_tag, final_strategy, macro_regime, micro_regime, row["id"])
            )
        if updates:
            con.executemany(
                "UPDATE trades SET pocket = ?, strategy_tag = ?, strategy = ?, macro_regime = ?, micro_regime = ? WHERE id = ?",
                updates,
            )
            con.commit()
        con.close()
        return len(updates)

    def _get_last_transaction_id_with_retry(self) -> int:
        self._ensure_connection_open()
        for attempt in range(_DB_LOCK_RETRY):
            try:
                return self._get_last_transaction_id()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP)
        return self._get_last_transaction_id()

    def _get_last_transaction_id(self) -> int:
        """DBに記録済みの最新トランザクションIDを取得"""
        self._ensure_connection_open()
        # 旧スキーマ互換のため、transaction_id 優先で取得する
        cursor = self.con.cursor()
        max_tx = 0
        try:
            row = cursor.execute("SELECT MAX(transaction_id) FROM trades").fetchone()
            if row and row[0]:
                max_tx = max(max_tx, int(row[0]))
        except sqlite3.OperationalError:
            pass
        try:
            row = cursor.execute("SELECT MAX(transaction_id) FROM cashflows").fetchone()
            if row and row[0]:
                max_tx = max(max_tx, int(row[0]))
        except sqlite3.OperationalError:
            pass
        if max_tx > 0:
            return max_tx

        row = cursor.execute("SELECT MAX(id) FROM trades").fetchone()
        return int(row[0]) if row and row[0] else 0

    def _has_ticket_unique_constraint(self) -> bool:
        self._ensure_connection_open()
        try:
            indexes = list(self.con.execute("PRAGMA index_list(trades)"))
        except sqlite3.Error:
            return False
        for idx in indexes:
            name = idx[1]
            unique = idx[2]
            if not unique:
                continue
            if name.startswith("sqlite_autoindex_trades"):
                try:
                    cols = list(self.con.execute(f"PRAGMA index_info('{name}')"))
                except sqlite3.Error:
                    continue
                if len(cols) == 1 and cols[0][2] == "ticket_id":
                    return True
        return False

    def _migrate_remove_ticket_unique(self) -> None:
        self._ensure_connection_open()
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS trades_migrated (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  transaction_id INTEGER,
                  ticket_id TEXT,
                  pocket TEXT,
                  instrument TEXT,
                  units INTEGER,
                  closed_units INTEGER,
                  entry_price REAL,
                  close_price REAL,
                  fill_price REAL,
                  pl_pips REAL,
                  realized_pl REAL,
                  commission REAL,
                  financing REAL,
                  entry_time TEXT,
                  open_time TEXT,
                  close_time TEXT,
                  close_reason TEXT,
                  state TEXT,
                  updated_at TEXT,
                  version TEXT DEFAULT 'v1',
                  unrealized_pl REAL
                )
                """
            )
            columns = (
                "id, transaction_id, ticket_id, pocket, instrument, units, closed_units, "
                "entry_price, close_price, fill_price, pl_pips, realized_pl, commission, "
                "financing, entry_time, open_time, close_time, close_reason, state, "
                "updated_at, version, unrealized_pl"
            )
            self.con.execute(
                f"INSERT OR IGNORE INTO trades_migrated ({columns}) "
                f"SELECT {columns} FROM trades"
            )
            self.con.execute("DROP TABLE trades")
            self.con.execute("ALTER TABLE trades_migrated RENAME TO trades")
            self.con.commit()
        except sqlite3.Error as exc:
            self.con.rollback()
            print(f"[PositionManager] Failed to migrate trades table: {exc}")
        finally:
            self._commit_with_retry()

    def _fetch_closed_trades(self):
        """OANDAから決済済みトランザクションを取得"""
        fetch_start = time.monotonic()
        summary_start = time.monotonic()
        summary_url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
        try:
            summary = self._request_json(
                summary_url,
                params={"sinceID": self._last_tx_id},
            ) or {}
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching transactions summary: {e}")
            self._reset_http_if_needed(e)
            summary_ms = max(0.0, (time.monotonic() - summary_start) * 1000.0)
            self._last_sync_fetch_meta = {
                "fetch_ms": summary_ms,
                "summary_ms": summary_ms,
                "chunk_ms": 0.0,
                "chunk_count": 0,
                "transactions": 0,
            }
            return []
        summary_ms = max(0.0, (time.monotonic() - summary_start) * 1000.0)

        try:
            last_tx_id = int(summary.get("lastTransactionID") or 0)
        except (TypeError, ValueError):
            last_tx_id = 0

        if last_tx_id <= self._last_tx_id:
            fetch_ms = max(0.0, (time.monotonic() - fetch_start) * 1000.0)
            self._last_sync_fetch_meta = {
                "fetch_ms": fetch_ms,
                "summary_ms": summary_ms,
                "chunk_ms": 0.0,
                "chunk_count": 0,
                "transactions": 0,
            }
            return []

        fetch_from = self._last_tx_id + 1
        min_allowed = max(1, last_tx_id - _MAX_FETCH + 1)
        if fetch_from < min_allowed:
            fetch_from = min_allowed

        transactions: list[dict] = []
        chunk_from = fetch_from
        chunk_url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions/idrange"
        chunk_ms = 0.0
        chunk_count = 0

        while chunk_from <= last_tx_id:
            chunk_to = min(chunk_from + _CHUNK_SIZE - 1, last_tx_id)
            params = {"from": chunk_from, "to": chunk_to}
            chunk_start = time.monotonic()
            try:
                data = self._request_json(chunk_url, params=params) or {}
            except requests.RequestException as e:
                print(
                    "[PositionManager] Error fetching transaction chunk "
                    f"{chunk_from}-{chunk_to}: {e}"
                )
                self._reset_http_if_needed(e)
                chunk_ms += max(0.0, (time.monotonic() - chunk_start) * 1000.0)
                break
            chunk_ms += max(0.0, (time.monotonic() - chunk_start) * 1000.0)
            chunk_count += 1
            for tx in data.get("transactions") or []:
                try:
                    tx_id = int(tx.get("id"))
                except (TypeError, ValueError):
                    continue
                if tx_id <= self._last_tx_id:
                    continue
                transactions.append(tx)

            chunk_from = chunk_to + 1

        transactions.sort(key=_tx_sort_key)
        fetch_ms = max(0.0, (time.monotonic() - fetch_start) * 1000.0)
        self._last_sync_fetch_meta = {
            "fetch_ms": fetch_ms,
            "summary_ms": summary_ms,
            "chunk_ms": chunk_ms,
            "chunk_count": chunk_count,
            "transactions": len(transactions),
        }
        return transactions

    def _get_trade_details(self, trade_id: str) -> dict | None:
        """tradeIDを使ってOANDAから取引詳細を取得する"""
        local = self._resolve_entry_meta(trade_id)
        if local:
            if not local.get("details_source"):
                local["details_source"] = "orders"
            return local
        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}"
        try:
            payload = self._request_json(url) or {}
            trade = payload.get("trade", {})
            client_ext = trade.get("clientExtensions", {}) or {}
            pocket_tag = client_ext.get("tag", "pocket=unknown")
            pocket = pocket_tag.split("=")[1] if "=" in pocket_tag else "unknown"
            client_id = client_ext.get("id")

            details = {
                "entry_price": float(trade.get("price", 0.0)),
                "entry_time": _parse_timestamp(trade.get("openTime")),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
                "client_order_id": client_id,
                "strategy_tag": None,
                "entry_thesis": None,
                "details_source": "oanda",
            }
            # Heuristic mapping from client id prefix
            try:
                if client_id:
                    strat = self._infer_strategy_tag(None, client_id, pocket)
                    if strat:
                        details["strategy_tag"] = strat
            except Exception:
                pass
            # Augment with local orders log for strategy_tag/thesis if possible
            try:
                from_orders = self._get_trade_details_from_orders(trade_id)
                if from_orders:
                    for key in ("client_order_id", "strategy_tag", "entry_thesis"):
                        if from_orders.get(key) and not details.get(key):
                            details[key] = from_orders.get(key)
            except Exception:
                pass
            thesis_obj, norm_tag = _apply_strategy_tag_normalization(
                details.get("entry_thesis"),
                details.get("strategy_tag"),
            )
            if thesis_obj is not None:
                details["entry_thesis"] = thesis_obj
            if norm_tag:
                details["strategy_tag"] = norm_tag
            details["pocket"] = _resolve_pocket(
                details.get("pocket"),
                details.get("strategy_tag"),
                client_id,
            )
            macro_regime, micro_regime = _extract_regime(details.get("entry_thesis"))
            details["macro_regime"] = macro_regime
            details["micro_regime"] = micro_regime
            return details
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                fallback = self._get_trade_details_from_orders(trade_id)
                if fallback:
                    if not fallback.get("details_source"):
                        fallback["details_source"] = "orders"
                    return fallback
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {exc}")
            fallback = self._get_trade_details_from_orders(trade_id)
            if fallback and not fallback.get("details_source"):
                fallback["details_source"] = "orders"
            return fallback
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {e}")
            self._reset_http_if_needed(e)
            fallback = self._get_trade_details_from_orders(trade_id)
            if fallback and not fallback.get("details_source"):
                fallback["details_source"] = "orders"
            return fallback

    def _get_trade_details_from_orders(self, trade_id: str) -> dict | None:
        con = None
        try:
            con = _open_orders_db_readonly()
            if con is None:
                return None
            row = con.execute(
                """
                SELECT pocket, instrument, units, executed_price, ts, client_order_id
                FROM orders
                WHERE ticket_id = ?
                  AND status = 'filled'
                ORDER BY id ASC
                LIMIT 1
                """,
                (trade_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            print(f"[PositionManager] orders.db lookup failed for trade {trade_id}: {exc}")
            return None
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        if not row:
            return None
        ts = row["ts"]
        try:
            entry_time = datetime.fromisoformat(ts)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            else:
                entry_time = entry_time.astimezone(timezone.utc)
        except Exception:
            entry_time = datetime.now(timezone.utc)
        client_id = row["client_order_id"]
        strategy_tag = None
        thesis_obj = None
        if client_id:
            con2 = None
            try:
                con2 = _open_orders_db_readonly()
                if con2 is None:
                    raise sqlite3.OperationalError("orders.db unavailable")
                att = con2.execute(
                    """
                    SELECT request_json FROM orders
                    WHERE client_order_id = ? AND status = 'submit_attempt'
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (client_id,),
                ).fetchone()
                if att and att["request_json"]:
                    try:
                        payload = json.loads(att["request_json"]) or {}
                        thesis_obj = payload.get("entry_thesis") or {}
                        if isinstance(thesis_obj, dict):
                            strategy_tag = thesis_obj.get("strategy_tag")
                    except Exception:
                        thesis_obj = None
            except sqlite3.Error:
                pass
            finally:
                if con2 is not None:
                    try:
                        con2.close()
                    except Exception:
                        pass
        if strategy_tag is None:
            strategy_tag = self._infer_strategy_tag(thesis_obj, client_id, row["pocket"])
        raw_tag = None
        if isinstance(thesis_obj, dict):
            raw_tag = thesis_obj.get("strategy_tag") or thesis_obj.get("strategy")
        if not raw_tag:
            raw_tag = strategy_tag
        thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
        if norm_tag:
            strategy_tag = norm_tag
        pocket = _resolve_pocket(row["pocket"], strategy_tag, client_id)
        macro_regime, micro_regime = _extract_regime(thesis_obj)
        return {
            "entry_price": float(row["executed_price"] or 0.0),
            "entry_time": entry_time,
            "units": int(row["units"] or 0),
            "pocket": pocket,
            "client_order_id": client_id,
            "strategy_tag": strategy_tag,
            "entry_thesis": thesis_obj,
            "macro_regime": macro_regime,
            "micro_regime": micro_regime,
            "details_source": "orders",
        }

    def _resolve_entry_meta(self, trade_id: str) -> dict | None:
        if not trade_id:
            return None
        cached = self._entry_meta_cache.get(trade_id)
        if cached:
            return cached
        details = self._get_trade_details_from_orders(trade_id)
        if details:
            if not details.get("details_source"):
                details["details_source"] = "orders"
            self._entry_meta_cache[trade_id] = details
        return details

    def _parse_and_save_trades(self, transactions: list[dict]):
        """トランザクションを解析し、DBに保存"""
        parse_start = time.monotonic()
        details_ms = 0.0
        details_calls = 0
        db_ms = 0.0
        trades_to_save = []
        cashflows_to_save: list[tuple] = []
        saved_records: list[dict] = []
        processed_tx_ids = set()

        for tx in transactions:
            tx_id_raw = tx.get("id")
            try:
                tx_id = int(tx_id_raw)
            except (TypeError, ValueError):
                continue
            cashflow_record = _extract_cashflow_tx(tx)
            if cashflow_record:
                cashflows_to_save.append(cashflow_record)
            # ORDER_FILLのみ処理（クローズ/部分約定両対応）
            if tx.get("type") != "ORDER_FILL":
                processed_tx_ids.add(tx_id)
                continue

            closures: list[tuple[str, dict]] = []
            for closed_trade in tx.get("tradesClosed") or []:
                closures.append(("CLOSED", closed_trade))
            trade_reduced = tx.get("tradeReduced")
            if trade_reduced:
                closures.append(("PARTIAL", trade_reduced))
            for reduced in tx.get("tradesReduced") or []:
                closures.append(("PARTIAL", reduced))

            if not closures:
                processed_tx_ids.add(tx_id)
                continue

            for state_label, closed_trade in closures:
                trade_id = closed_trade.get("tradeID")
                if not trade_id:
                    continue

                details_start = time.monotonic()
                details = self._get_trade_details(trade_id)
                details_ms += max(0.0, (time.monotonic() - details_start) * 1000.0)
                details_calls += 1
                if not details:
                    continue
                inferred_tag = self._infer_strategy_tag(
                    details.get("entry_thesis"),
                    details.get("client_order_id"),
                    details.get("pocket"),
                )
                if inferred_tag and not details.get("strategy_tag"):
                    details["strategy_tag"] = inferred_tag
                client_trade_id = closed_trade.get("clientTradeID") or closed_trade.get("client_trade_id")
                if client_trade_id:
                    if not details.get("client_order_id") or details.get("details_source") != "oanda":
                        details["client_order_id"] = client_trade_id
                    inferred_from_trade = self._infer_strategy_tag(
                        details.get("entry_thesis"),
                        client_trade_id,
                        details.get("pocket"),
                    )
                    if inferred_from_trade and (
                        not details.get("strategy_tag") or details.get("details_source") != "oanda"
                    ):
                        details["strategy_tag"] = inferred_from_trade

                close_price = float(tx.get("price", 0.0))
                close_time = _parse_timestamp(tx.get("time"))

                # USD/JPY の pips を計算 (1 pip = 0.01 JPY)
                # OANDAのPLは通貨額なので、価格差からpipsを計算する
                entry_price = details["entry_price"]
                units = details["units"]
                # 実際にクローズされたユニット（部分決済対応）
                try:
                    closed_units_raw = closed_trade.get("units")
                    # OANDAの tradesClosed[].units は方向に応じて符号が付く場合があるため、絶対値で保存
                    closed_units = abs(int(float(closed_units_raw))) if closed_units_raw is not None else 0
                except Exception:
                    closed_units = 0

                realized_pl = float(closed_trade.get("realizedPL", 0.0) or 0.0)
                details_source = details.get("details_source") or "unknown"
                pip_units = abs(closed_units or units)
                pip_value = 0.01
                if (
                    details_source == "oanda"
                    and units != 0
                    and entry_price > 0.0
                    and close_price > 0.0
                ):
                    if units > 0:  # Buy
                        pl_pips = (close_price - entry_price) * 100
                    else:  # Sell
                        pl_pips = (entry_price - close_price) * 100
                    if pip_units > 0 and abs(pl_pips) < 0.2 and abs(realized_pl) >= 100:
                        pl_pips = realized_pl / (pip_units * pip_value)
                else:
                    pl_pips = realized_pl / (pip_units * pip_value) if pip_units else 0.0
                # 取引コスト類（存在すれば保存）
                try:
                    commission = float(tx.get("commission", 0.0) or 0.0)
                except Exception:
                    commission = 0.0
                try:
                    financing = float(tx.get("financing", 0.0) or 0.0)
                except Exception:
                    financing = 0.0
                transaction_id = int(tx.get("id", 0) or 0)
                updated_at = datetime.now(timezone.utc).isoformat()
                close_reason = tx.get("reason") or tx.get("type") or "UNKNOWN"
                details["pocket"] = _resolve_pocket(
                    details.get("pocket"),
                    details.get("strategy_tag"),
                    details.get("client_order_id"),
                )
                macro_regime = details.get("macro_regime")
                micro_regime = details.get("micro_regime")
                if not macro_regime or not micro_regime:
                    macro_fallback, micro_fallback = _extract_regime(
                        details.get("entry_thesis")
                    )
                    if not macro_regime:
                        macro_regime = macro_fallback
                    if not micro_regime:
                        micro_regime = micro_fallback

                record_tuple = (
                    transaction_id,
                    trade_id,
                    details["pocket"],
                    tx.get("instrument"),
                    units,
                    closed_units,
                    entry_price,
                    close_price,
                    close_price,
                    pl_pips,
                    realized_pl,
                    commission,
                    financing,
                    details["entry_time"].isoformat(),
                    details["entry_time"].isoformat(),
                    close_time.isoformat(),
                    close_reason,
                    "CLOSED" if state_label == "CLOSED" else "PARTIAL",
                    updated_at,
                    "v3",
                    0.0,
                    # attribution
                    details.get("client_order_id"),
                    details.get("strategy_tag"),
                    details.get("strategy_tag"),
                    json.dumps(details.get("entry_thesis"), ensure_ascii=False),
                    macro_regime,
                    micro_regime,
                )
                trades_to_save.append(record_tuple)
                saved_records.append(
                    {
                        "transaction_id": transaction_id,
                        "ticket_id": trade_id,
                        "pocket": details["pocket"],
                        "instrument": tx.get("instrument"),
                        "units": units,
                        "closed_units": closed_units,
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "fill_price": close_price,
                        "pl_pips": pl_pips,
                        "realized_pl": realized_pl,
                        "commission": commission,
                        "financing": financing,
                        "entry_time": details["entry_time"].isoformat(),
                        "close_time": close_time.isoformat(),
                        "close_reason": close_reason,
                        "state": "CLOSED",
                        "updated_at": updated_at,
                        "version": "v3",
                        "unrealized_pl": 0.0,
                        "client_order_id": details.get("client_order_id"),
                        "strategy": details.get("strategy_tag"),
                        "strategy_tag": details.get("strategy_tag"),
                        "macro_regime": macro_regime,
                        "micro_regime": micro_regime,
                    }
                )
                if details["pocket"]:
                    self._pocket_cache[trade_id] = details["pocket"]

                processed_tx_ids.add(tx_id)

        if trades_to_save:
            # ticket_id (OANDA tradeID) が重複しないように挿入
            db_start = time.monotonic()
            try:
                with _file_lock(_DB_LOCK_PATH):
                    self._executemany_with_retry(
                        """
                        INSERT OR REPLACE INTO trades (
                            transaction_id,
                            ticket_id,
                            pocket,
                            instrument,
                            units,
                            closed_units,
                            entry_price,
                            close_price,
                            fill_price,
                            pl_pips,
                            realized_pl,
                            commission,
                            financing,
                            entry_time,
                            open_time,
                            close_time,
                            close_reason,
                            state,
                            updated_at,
                            version,
                            unrealized_pl,
                            client_order_id,
                            strategy,
                            strategy_tag,
                            entry_thesis,
                            macro_regime,
                            micro_regime
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        trades_to_save,
                    )
                    self._commit_with_retry()
                db_ms = max(0.0, (time.monotonic() - db_start) * 1000.0)
            except TimeoutError:
                logging.warning(
                    "[PositionManager] file lock busy; defer saving %d trades",
                    len(trades_to_save),
                )
                db_ms = max(0.0, (time.monotonic() - db_start) * 1000.0)
                parse_ms = max(0.0, (time.monotonic() - parse_start) * 1000.0)
                self._last_sync_parse_meta = {
                    "parse_ms": parse_ms,
                    "details_ms": details_ms,
                    "details_calls": details_calls,
                    "db_ms": db_ms,
                    "trades_saved": len(trades_to_save),
                }
                return []
            print(f"[PositionManager] Saved {len(trades_to_save)} new trades.")

        if cashflows_to_save:
            try:
                with _file_lock(_DB_LOCK_PATH):
                    self._executemany_with_retry(
                        """
                        INSERT OR IGNORE INTO cashflows (
                            transaction_id, time, amount, balance, type, funding_reason
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        cashflows_to_save,
                    )
                    self._commit_with_retry()
            except TimeoutError:
                logging.warning(
                    "[PositionManager] file lock busy; defer saving %d cashflows",
                    len(cashflows_to_save),
                )

        if processed_tx_ids:
            self._last_tx_id = max(processed_tx_ids)
        parse_ms = max(0.0, (time.monotonic() - parse_start) * 1000.0)
        self._last_sync_parse_meta = {
            "parse_ms": parse_ms,
            "details_ms": details_ms,
            "details_calls": details_calls,
            "db_ms": db_ms,
            "trades_saved": len(trades_to_save),
        }
        return saved_records

    def sync_trades(self, max_fetch: int | None = None):
        """定期的に呼び出し、決済済みトレードを同期する"""
        self._ensure_connection_open()
        if max_fetch is None:
            max_fetch = _MAX_FETCH
        try:
            max_fetch_int = int(max_fetch)
        except (TypeError, ValueError):
            max_fetch_int = _MAX_FETCH

        service_result = _position_manager_service_request(
            "/position/sync_trades", {"max_fetch": max_fetch_int}
        )
        if service_result is not None:
            if isinstance(service_result, list):
                if max_fetch_int > 0:
                    return service_result[-max_fetch_int:]
                return service_result
            if isinstance(service_result, dict):
                return list(service_result.values()) if service_result else []
            return []

        if not self._sync_trades_lock.acquire(blocking=False):
            now = time.monotonic()
            if self._last_sync_cache and (
                now - self._last_sync_cache_ts
            ) <= self._last_sync_cache_window_sec:
                if max_fetch_int > 0:
                    return self._last_sync_cache[-max_fetch_int:]
                return list(self._last_sync_cache)
            return []

        try:
            sync_start = time.monotonic()
            transactions = self._fetch_closed_trades()
            fetch_meta = dict(self._last_sync_fetch_meta or {})
            if not transactions:
                total_ms = max(0.0, (time.monotonic() - sync_start) * 1000.0)
                fetch_meta.setdefault("transactions", 0)
                fetch_meta["total_ms"] = total_ms
                self._last_sync_breakdown = fetch_meta
                self._last_sync_cache_ts = time.monotonic()
                self._last_sync_cache = []
                return []
            saved_records = self._parse_and_save_trades(transactions)
            parse_meta = dict(self._last_sync_parse_meta or {})
            total_ms = max(0.0, (time.monotonic() - sync_start) * 1000.0)
            breakdown = {}
            breakdown.update(fetch_meta)
            breakdown.update(parse_meta)
            breakdown["total_ms"] = total_ms
            breakdown.setdefault("transactions", len(transactions))
            self._last_sync_breakdown = breakdown
            self._last_sync_cache = list(saved_records)
            self._last_sync_cache_ts = time.monotonic()
            if max_fetch_int > 0:
                return saved_records[-max_fetch_int:]
            return saved_records
        finally:
            self._sync_trades_lock.release()

    def _request_json(self, url: str, params: dict | None = None) -> dict:
        return self._request_json_with_timeout(url, params=params, timeout_sec=_REQUEST_TIMEOUT)

    def _request_json_with_timeout(
        self,
        url: str,
        *,
        params: dict | None = None,
        timeout_sec: float,
    ) -> dict:
        resp = self._http.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=max(0.5, float(timeout_sec)),
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            logging.warning("[PositionManager] Non-JSON response from %s", url)
            return {}

    def _reset_http_if_needed(self, exc: requests.RequestException) -> None:
        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
            ),
        ):
            try:
                self._http.close()
            except Exception:
                pass
            self._http = _build_http_session()

    def register_open_trade(self, trade_id: str, pocket: str, client_id: str | None = None):
        if trade_id and pocket:
            self._pocket_cache[str(trade_id)] = pocket
        if client_id and trade_id:
            self._client_cache[client_id] = trade_id

    def get_open_positions(self, include_unknown: bool = False) -> dict[str, dict]:
        """現在の保有ポジションを pocket 単位で集計して返す"""
        service_result = _position_manager_service_request(
            "/position/open_positions",
            {"include_unknown": include_unknown},
        )
        if service_result is not None:
            if isinstance(service_result, dict):
                return service_result
            return {}

        now_mono = time.monotonic()
        if not self._open_positions_lock.acquire(blocking=False):
            if self._last_positions:
                age = max(0.0, now_mono - self._last_positions_ts)
                return self._package_positions(
                    self._last_positions,
                    stale=True,
                    age_sec=age,
                    extra_meta=self._last_positions_meta,
                )
            return {}

        try:
            if self._last_positions and now_mono < self._next_open_fetch_after:
                age = max(0.0, now_mono - self._last_positions_ts)
                stale = self._open_trade_failures > 0
                return self._package_positions(
                    self._last_positions,
                    stale=stale,
                    age_sec=age,
                    extra_meta=self._last_positions_meta,
                )

            url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/openTrades"
            try:
                payload = self._request_json_with_timeout(
                    url,
                    timeout_sec=_OPEN_TRADES_REQUEST_TIMEOUT,
                ) or {}
                trades = payload.get("trades", [])
                self._open_trade_failures = 0
                self._next_open_fetch_after = now_mono + _OPEN_TRADES_CACHE_TTL
            except requests.RequestException as e:
                logging.warning("[PositionManager] Error fetching open trades: %s", e)
                self._reset_http_if_needed(e)
                self._open_trade_failures += 1
                backoff = min(
                    _OPEN_TRADES_FAIL_BACKOFF_BASE * (2 ** (self._open_trade_failures - 1)),
                    _OPEN_TRADES_FAIL_BACKOFF_MAX,
                )
                self._next_open_fetch_after = now_mono + backoff
                if self._last_positions:
                    age = max(0.0, now_mono - self._last_positions_ts)
                    return self._package_positions(
                        self._last_positions,
                        stale=True,
                        age_sec=age,
                        extra_meta=self._last_positions_meta,
                    )
                return {}

            pockets: dict[str, dict] = {}
            net_units = 0
            manual_trades = 0
            manual_units = 0
            manual_unrealized = 0.0
            client_ids: set[str] = set()
            for tr in trades:
                client_ext = tr.get("clientExtensions", {}) or {}
                client_id_raw = client_ext.get("id")
                client_id = str(client_id_raw or "")
                tag_raw = client_ext.get("tag") or ""
                tag = str(tag_raw)
                trade_id = tr.get("id") or tr.get("tradeID")
                cached_pocket = self._pocket_cache.get(str(trade_id), "") if trade_id else ""

                pocket: str
                is_agent_client = client_id.startswith(agent_client_prefixes)
                if is_agent_client:
                    if tag.startswith("pocket="):
                        pocket = tag.split("=", 1)[1]
                    elif cached_pocket in agent_pockets:
                        pocket = cached_pocket
                    else:
                        pocket = "unknown"
                else:
                    # Non-agent trades are always treated as manual, even if tags hint a pocket.
                    # This prevents manual orders from being managed/closed by bot workers.
                    trade_id = tr.get("id") or tr.get("tradeID")
                    pocket = self._pocket_cache.get(str(trade_id), _MANUAL_POCKET_NAME)
                    if pocket in agent_pockets:
                        pocket = _MANUAL_POCKET_NAME
                if pocket not in _KNOWN_POCKETS:
                    if not (include_unknown and pocket == "unknown"):
                        pocket = _MANUAL_POCKET_NAME
                units = int(tr.get("currentUnits", 0))
                if units == 0:
                    continue
                trade_id_raw = tr.get("id") or tr.get("tradeID")
                trade_id = str(trade_id_raw)
                price = float(tr.get("price", 0.0))
                if client_id:
                    client_ids.add(client_id)
                open_time_raw = tr.get("openTime")
                open_time_iso: str | None = None
                if open_time_raw:
                    try:
                        opened_dt = _parse_timestamp(open_time_raw).astimezone(timezone.utc)
                        open_time_iso = opened_dt.isoformat()
                    except Exception:
                        open_time_iso = open_time_raw
                info = pockets.setdefault(
                    pocket,
                    {
                        "units": 0,
                        "avg_price": 0.0,
                        "trades": 0,
                        "long_units": 0,
                        "long_avg_price": 0.0,
                        "short_units": 0,
                        "short_avg_price": 0.0,
                        "open_trades": [],
                        "unrealized_pl": 0.0,
                        "unrealized_pl_pips": 0.0,
                    },
                )
                if pocket == "manual":
                    info["manual"] = True
                try:
                    unrealized_pl = float(tr.get("unrealizedPL", 0.0) or 0.0)
                except Exception:
                    unrealized_pl = 0.0
                abs_units = abs(units)
                pip_value = abs_units * 0.01
                unrealized_pl_pips = unrealized_pl / pip_value if pip_value else 0.0
                client_id = client_ext.get("id")
                trade_entry = {
                    "trade_id": trade_id,
                    "units": units,
                    "price": price,
                    "client_id": client_id,
                    "client_order_id": client_id,
                    "side": "long" if units > 0 else "short",
                    "unrealized_pl": unrealized_pl,
                    "unrealized_pl_pips": unrealized_pl_pips,
                    "open_time": open_time_iso or open_time_raw,
                }
                tp_order = tr.get("takeProfitOrder") or {}
                if isinstance(tp_order, dict):
                    tp_price_raw = tp_order.get("price")
                    tp_order_id = tp_order.get("id")
                    try:
                        tp_price = float(tp_price_raw) if tp_price_raw is not None else None
                    except (TypeError, ValueError):
                        tp_price = None
                    if tp_price is not None and tp_price > 0:
                        take_profit = {"price": tp_price}
                        if tp_order_id:
                            take_profit["order_id"] = str(tp_order_id)
                        trade_entry["take_profit"] = take_profit

                sl_order = tr.get("stopLossOrder") or {}
                if isinstance(sl_order, dict):
                    sl_price_raw = sl_order.get("price")
                    sl_order_id = sl_order.get("id")
                    try:
                        sl_price = float(sl_price_raw) if sl_price_raw is not None else None
                    except (TypeError, ValueError):
                        sl_price = None
                    if sl_price is not None and sl_price > 0:
                        stop_loss = {"price": sl_price}
                        if sl_order_id:
                            stop_loss["order_id"] = str(sl_order_id)
                        trade_entry["stop_loss"] = stop_loss
                # Fallback: decode clientExtensions.comment to recover entry meta for EXIT判定
                thesis_from_comment = None
                try:
                    comment_raw = client_ext.get("comment")
                    if (
                        isinstance(comment_raw, str)
                        and comment_raw.startswith("{")
                        and len(comment_raw) <= 256
                    ):
                        parsed = json.loads(comment_raw)
                        if isinstance(parsed, dict):
                            thesis_from_comment = parsed
                except Exception:
                    thesis_from_comment = None
                # Keep open_positions hot path lightweight:
                # orders.db lookup per trade can block under write contention and fan out into
                # request timeouts across strategy workers. Resolve from orders.db only when we
                # cannot infer a strategy tag from client metadata.
                meta = None
                if is_agent_client:
                    inferred_fast = self._infer_strategy_tag(
                        thesis_from_comment,
                        client_id,
                        pocket,
                    )
                    needs_orders_lookup = (not client_id) or (inferred_fast is None)
                    if needs_orders_lookup:
                        meta = self._resolve_entry_meta(trade_id)
                if meta:
                    thesis = meta.get("entry_thesis")
                    if isinstance(thesis, str):
                        try:
                            thesis = json.loads(thesis)
                        except Exception:
                            thesis = None
                    if isinstance(thesis, dict):
                        trade_entry["entry_thesis"] = thesis
                    if meta.get("client_order_id"):
                        trade_entry["client_order_id"] = meta.get("client_order_id")
                    strategy_tag = meta.get("strategy_tag")
                    if strategy_tag:
                        trade_entry["strategy_tag"] = strategy_tag
                if thesis_from_comment:
                    if not trade_entry.get("entry_thesis"):
                        trade_entry["entry_thesis"] = thesis_from_comment
                    if not trade_entry.get("strategy_tag"):
                        tag_val = thesis_from_comment.get("strategy_tag") or thesis_from_comment.get(
                            "tag"
                        )
                        if tag_val:
                            trade_entry["strategy_tag"] = tag_val
                if not trade_entry.get("strategy_tag"):
                    inferred = self._infer_strategy_tag(trade_entry.get("entry_thesis"), client_id, pocket)
                    if inferred:
                        trade_entry["strategy_tag"] = inferred
                raw_tag = None
                thesis_obj = trade_entry.get("entry_thesis")
                if isinstance(thesis_obj, dict):
                    raw_tag = thesis_obj.get("strategy_tag") or thesis_obj.get("strategy")
                if not raw_tag:
                    raw_tag = trade_entry.get("strategy_tag")
                thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
                if thesis_obj is not None:
                    trade_entry["entry_thesis"] = thesis_obj
                if norm_tag:
                    trade_entry["strategy_tag"] = norm_tag
                # EXIT誤爆防止: エージェント管理ポケットかつ strategy_tag 不明なら除外
                if pocket in agent_pockets and not trade_entry.get("strategy_tag"):
                    if include_unknown:
                        trade_entry["strategy_tag"] = "unknown"
                        trade_entry["missing_strategy_tag"] = True
                    else:
                        logging.warning(
                            "[PositionManager] skip open trade without strategy_tag pocket=%s trade_id=%s client_id=%s",
                            pocket,
                            trade_id,
                            trade_entry.get("client_id"),
                        )
                        continue
                info["open_trades"].append(trade_entry)
                prev_total_units = info["units"]
                new_total_units = prev_total_units + units
                if new_total_units != 0:
                    info["avg_price"] = (
                        info["avg_price"] * prev_total_units + price * units
                    ) / new_total_units
                else:
                    info["avg_price"] = price
                info["units"] = new_total_units
                info["trades"] += 1

                if units > 0:
                    prev_units = info["long_units"]
                    new_units = prev_units + units
                    if new_units > 0:
                        if prev_units == 0 or info["long_avg_price"] == 0.0:
                            info["long_avg_price"] = price
                        else:
                            info["long_avg_price"] = (
                                info["long_avg_price"] * prev_units + price * units
                            ) / new_units
                    info["long_units"] = new_units
                elif units < 0:
                    abs_units = abs(units)
                    prev_units = info["short_units"]
                    new_units = prev_units + abs_units
                    if new_units > 0:
                        if prev_units == 0 or info["short_avg_price"] == 0.0:
                            info["short_avg_price"] = price
                        else:
                            info["short_avg_price"] = (
                                info["short_avg_price"] * prev_units + price * abs_units
                            ) / new_units
                    info["short_units"] = new_units

                info["unrealized_pl"] = info.get("unrealized_pl", 0.0) + unrealized_pl
                info["unrealized_pl_pips"] = info.get("unrealized_pl_pips", 0.0) + unrealized_pl_pips
                net_units += units
                if pocket == _MANUAL_POCKET_NAME:
                    manual_trades += 1
                    manual_units += units
                    manual_unrealized += unrealized_pl

            if client_ids:
                entry_map = self._load_entry_thesis(client_ids)
                for pocket_info in pockets.values():
                    trades_list = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                    if not trades_list:
                        continue
                    for trade in trades_list:
                        cid = trade.get("client_id")
                        if cid and cid in entry_map:
                            thesis_obj = entry_map[cid]
                            raw_tag = None
                            if isinstance(thesis_obj, dict):
                                raw_tag = thesis_obj.get("strategy_tag_raw") or thesis_obj.get("strategy_tag")
                                if not raw_tag:
                                    raw_tag = thesis_obj.get("strategy")
                            thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
                            if thesis_obj is not None:
                                trade["entry_thesis"] = thesis_obj
                            if norm_tag:
                                trade["strategy_tag"] = norm_tag

            pockets["__net__"] = {"units": net_units}
            self._last_positions = copy.deepcopy(pockets)
            self._last_positions_ts = now_mono

            extra_meta = {}
            if manual_trades:
                extra_meta = {
                    "manual_trades": manual_trades,
                    "manual_units": manual_units,
                    "manual_unrealized_pl": manual_unrealized,
                }
            self._last_positions_meta = dict(extra_meta)
            return self._package_positions(pockets, stale=False, age_sec=0.0, extra_meta=extra_meta)
        finally:
            self._open_positions_lock.release()


    def _commit_with_retry(self) -> None:
        """Commit with retry to survive short lock bursts."""
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self._ensure_connection_open()
                self.con.commit()
                return
            except (sqlite3.OperationalError, sqlite3.ProgrammingError) as exc:
                if _is_closed_db_error(exc):
                    self._reopen_trades_connection()
                    continue
                if "locked" not in str(exc).lower() or attempt == _DB_LOCK_RETRY - 1:
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP * (attempt + 1))

    def _executemany_with_retry(self, sql: str, params) -> None:
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self._ensure_connection_open()
                self.con.executemany(sql, params)
                return
            except (sqlite3.OperationalError, sqlite3.ProgrammingError) as exc:
                if _is_closed_db_error(exc):
                    self._reopen_trades_connection()
                    continue
                if "locked" not in str(exc).lower() or attempt == _DB_LOCK_RETRY - 1:
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP * (attempt + 1))

    def _package_positions(
        self,
        pockets: dict[str, dict],
        *,
        stale: bool,
        age_sec: float,
        extra_meta: dict | None = None,
    ) -> dict[str, dict]:
        snapshot = copy.deepcopy(pockets)
        meta = snapshot.setdefault("__meta__", {})
        meta.update(
            {
                "stale": bool(stale),
                "age_sec": round(max(age_sec, 0.0), 2),
                "consecutive_failures": self._open_trade_failures if stale else 0,
            }
        )
        if extra_meta:
            meta.update(extra_meta)
        return snapshot

    def manual_exposure(
        self,
        positions: Dict[str, Dict] | None = None,
    ) -> Dict[str, float]:
        """
        Return a lightweight snapshot of manual/unknown exposure.
        """
        snapshot = positions or self.get_open_positions()
        units = 0
        unrealized = 0.0

        def _sum(info: Dict) -> Tuple[int, float]:
            if not info:
                return 0, 0.0
            trades = info.get("open_trades") or []
            total_units = 0
            unreal = 0.0
            if trades:
                for tr in trades:
                    try:
                        total_units += abs(int(tr.get("units", 0) or 0))
                    except (TypeError, ValueError):
                        continue
                    try:
                        unreal += float(tr.get("unrealized_pl", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                return total_units, unreal
            try:
                total_units = abs(int(info.get("units", 0) or 0))
            except (TypeError, ValueError):
                total_units = 0
            try:
                unreal = float(info.get("unrealized_pl", 0.0) or 0.0)
            except (TypeError, ValueError):
                unreal = 0.0
            return total_units, unreal

        for pocket in (_MANUAL_POCKET_NAME, "unknown"):
            info = snapshot.get(pocket) or {}
            pocket_units, pocket_unreal = _sum(info)
            units += pocket_units
            unrealized += pocket_unreal

        return {
            "units": float(units),
            "lots": units / 100000.0,
            "unrealized_pl": float(unrealized),
        }

    def _load_entry_thesis(self, client_ids: list[str]) -> Dict[str, dict]:
        unique_ids = tuple(dict.fromkeys(cid for cid in client_ids if cid))
        if not unique_ids:
            return {}
        con = None
        try:
            con = _open_orders_db_readonly()
            if con is None:
                return {}
        except sqlite3.Error as exc:
            print(f"[PositionManager] Failed to open orders.db: {exc}")
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        try:
            rows = con.execute(
                f"""
                SELECT client_order_id, request_json
                FROM orders
                WHERE client_order_id IN ({placeholders})
                  AND status='submit_attempt'
                ORDER BY id DESC
                """,
                unique_ids,
            ).fetchall()
        except sqlite3.Error as exc:
            print(f"[PositionManager] orders.db query failed: {exc}")
            return {}
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        result: Dict[str, dict] = {}
        for row in rows:
            cid = row["client_order_id"]
            if cid in result:
                continue
            payload_raw = row["request_json"]
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            thesis = payload.get("entry_thesis") or (payload.get("meta") or {}).get("entry_thesis") or {}
            result[cid] = thesis
        return result

    def _load_cashflows(self) -> list[dict]:
        self._ensure_connection_open()
        try:
            cur = self.con.execute(
                "SELECT transaction_id, time, amount, balance, type, funding_reason "
                "FROM cashflows ORDER BY time ASC"
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def get_performance_summary(self, now: datetime | None = None) -> dict:
        self._ensure_connection_open()
        now = now or datetime.now(timezone.utc)
        service_result = _position_manager_service_request(
            "/position/performance_summary",
            {"now": now.isoformat() if isinstance(now, datetime) else None},
        )
        if service_result is not None:
            if isinstance(service_result, dict):
                return service_result
            return {}

        jst = timezone(timedelta(hours=9))
        now_jst = now.astimezone(jst)
        today_start = now_jst.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        week_start = today_start - timedelta(days=6)

        buckets = {
            "daily": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
            "yesterday": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
            "weekly": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
            "total": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
        }
        latest_close: datetime | None = None

        rows = self.con.execute(
            "SELECT pl_pips, realized_pl, close_time, pocket, client_order_id "
            "FROM trades WHERE close_time IS NOT NULL"
        ).fetchall()
        year_start = now_jst.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        year_start_utc = year_start.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
        agent_prefixes = tuple(p for p in ("qr-", "qs-") if p)
        daily_table: dict[str, dict] = {}
        weekly_table: dict[str, dict] = {}
        monthly_table: dict[str, dict] = {}
        ytd_bot = {
            "pips": 0.0,
            "jpy": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
        }
        hourly_buckets: dict[datetime, dict[str, float | int]] = {}
        now_hour = now_jst.replace(minute=0, second=0, microsecond=0)
        start_hour = now_hour - timedelta(hours=_HOURLY_LOOKBACK_HOURS - 1)
        for i in range(_HOURLY_LOOKBACK_HOURS):
            hour = now_hour - timedelta(hours=i)
            hourly_buckets[hour] = {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            }
        closed_events: list[tuple[datetime, float, float]] = []
        cashflow_rows = self._load_cashflows()
        cashflow_buckets = {
            "daily": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "yesterday": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "weekly": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "total": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "ytd": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
        }

        def _apply_cashflow(bucket: dict, amount: float) -> None:
            bucket["net"] += amount
            bucket["count"] += 1
            if amount >= 0:
                bucket["in"] += amount
            else:
                bucket["out"] += abs(amount)

        for row in cashflow_rows:
            ts = row.get("time")
            if not ts:
                continue
            try:
                flow_dt = _parse_timestamp(str(ts))
            except Exception:
                continue
            amount = float(row.get("amount") or 0.0)
            flow_dt_jst = flow_dt.astimezone(jst)
            _apply_cashflow(cashflow_buckets["total"], amount)
            if flow_dt_jst >= week_start:
                _apply_cashflow(cashflow_buckets["weekly"], amount)
            if flow_dt_jst >= today_start:
                _apply_cashflow(cashflow_buckets["daily"], amount)
            elif flow_dt_jst >= yesterday_start:
                _apply_cashflow(cashflow_buckets["yesterday"], amount)
            if flow_dt_jst >= year_start:
                _apply_cashflow(cashflow_buckets["ytd"], amount)

        def _is_agent_trade(pocket: str, client_id: str | None) -> bool:
            if pocket in _KNOWN_POCKETS:
                return True
            if not client_id:
                return False
            cid = str(client_id)
            return cid.startswith(agent_prefixes)

        def _bucket(table: dict[str, dict], key: str) -> dict:
            if key not in table:
                table[key] = {
                    "pips": 0.0,
                    "jpy": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_pips": 0.0,
                    "loss_pips": 0.0,
                }
            return table[key]
        pocket_raw: dict[str, dict[str, float | int]] = {}
        for row in rows:
            try:
                close_dt = _parse_timestamp(row["close_time"])
            except Exception:
                continue
            if latest_close is None or close_dt > latest_close:
                latest_close = close_dt
            close_dt_jst = close_dt.astimezone(jst)
            pl_pips = float(row["pl_pips"] or 0.0)
            pl_jpy = float(row["realized_pl"] or 0.0)
            pocket = (row["pocket"] or "unknown").lower()
            client_id = row["client_order_id"]
            pkt = pocket_raw.setdefault(
                pocket,
                {"pips": 0.0, "jpy": 0.0, "trades": 0, "wins": 0, "losses": 0},
            )

            def _apply(bucket: dict) -> None:
                bucket["pips"] += pl_pips
                bucket["jpy"] += pl_jpy
                bucket["trades"] += 1
                if pl_pips > 0:
                    bucket["wins"] += 1
                elif pl_pips < 0:
                    bucket["losses"] += 1

            _apply(buckets["total"])
            _apply(pkt)
            if close_dt_jst >= week_start:
                _apply(buckets["weekly"])
            if close_dt_jst >= today_start:
                _apply(buckets["daily"])
            elif close_dt_jst >= yesterday_start:
                _apply(buckets["yesterday"])

            if close_dt_jst >= year_start and pocket != _MANUAL_POCKET_NAME:
                if _is_agent_trade(pocket, client_id):
                    day_key = close_dt_jst.date().isoformat()
                    week_start_date = close_dt_jst.date() - timedelta(days=close_dt_jst.weekday())
                    week_key = week_start_date.isoformat()
                    month_key = close_dt_jst.strftime("%Y-%m")
                    for key, table in (
                        (day_key, daily_table),
                        (week_key, weekly_table),
                        (month_key, monthly_table),
                    ):
                        bkt = _bucket(table, key)
                        bkt["pips"] += pl_pips
                        bkt["jpy"] += pl_jpy
                        bkt["trades"] += 1
                        if pl_pips > 0:
                            bkt["wins"] += 1
                            bkt["win_pips"] += pl_pips
                        elif pl_pips < 0:
                            bkt["losses"] += 1
                            bkt["loss_pips"] += abs(pl_pips)
                    ytd_bot["pips"] += pl_pips
                    ytd_bot["jpy"] += pl_jpy
                    ytd_bot["trades"] += 1
                    if pl_pips > 0:
                        ytd_bot["wins"] += 1
                    elif pl_pips < 0:
                        ytd_bot["losses"] += 1
            if pocket != _MANUAL_POCKET_NAME and _is_agent_trade(pocket, client_id):
                closed_events.append((close_dt, pl_jpy, pl_pips))

            if close_dt_jst >= start_hour and pocket != _MANUAL_POCKET_NAME:
                if _is_agent_trade(pocket, client_id):
                    hour_key = close_dt_jst.replace(minute=0, second=0, microsecond=0)
                    bucket = hourly_buckets.get(hour_key)
                    if bucket is not None:
                        bucket["pips"] += pl_pips
                        bucket["jpy"] += pl_jpy
                        bucket["trades"] += 1
                        if pl_pips > 0:
                            bucket["wins"] += 1
                        elif pl_pips < 0:
                            bucket["losses"] += 1

        def _finalise(data: dict) -> dict:
            trades = data["trades"]
            win_rate = (data["wins"] / trades) if trades else 0.0
            pf = None
            if data["losses"] > 0:
                loss_sum = abs(data["pips"] - max(0.0, data["pips"]))
                win_sum = max(0.0, data["pips"])
                if loss_sum > 0:
                    pf = win_sum / loss_sum if loss_sum else None
            return {
                "pips": round(data["pips"], 2),
                "jpy": round(data["jpy"], 2),
                "trades": trades,
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": win_rate,
                "pf": pf,
            }

        def _finalise_cashflow(data: dict) -> dict:
            return {
                "net": round(data["net"], 2),
                "in": round(data["in"], 2),
                "out": round(data["out"], 2),
                "count": int(data["count"]),
            }

        pockets_final: dict[str, dict] = {}
        for name, raw in pocket_raw.items():
            pockets_final[name] = _finalise(raw)

        daily = _finalise(buckets["daily"])
        weekly = _finalise(buckets["weekly"])
        total = _finalise(buckets["total"])
        yesterday = _finalise(buckets["yesterday"])
        cashflow_summary = {
            "daily": _finalise_cashflow(cashflow_buckets["daily"]),
            "yesterday": _finalise_cashflow(cashflow_buckets["yesterday"]),
            "weekly": _finalise_cashflow(cashflow_buckets["weekly"]),
            "total": _finalise_cashflow(cashflow_buckets["total"]),
            "ytd": _finalise_cashflow(cashflow_buckets["ytd"]),
        }

        def _growth_rate_equity(delta: float, equity: float | None) -> float | None:
            if equity is None or equity <= 0:
                return None
            return round(delta * 100.0 / equity, 2)

        equity_nav = _load_latest_metric("account.nav")
        equity_source = "nav"
        if equity_nav is None or equity_nav <= 0:
            equity_nav = _load_latest_metric("account.balance")
            equity_source = "balance"
        if equity_nav is None or equity_nav <= 0:
            equity_source = "unknown"
            equity_nav = None

        start_balance = _load_first_metric_since("account.balance", year_start_utc)
        start_balance_source = "metric"
        if start_balance is None or start_balance <= 0:
            start_balance = None
            start_balance_source = "missing"
        current_balance = _load_latest_metric("account.balance")
        if current_balance is not None and current_balance <= 0:
            current_balance = None

        daily_change = {
            "pips": round(daily["pips"] - yesterday["pips"], 2),
            "jpy": round(daily["jpy"] - yesterday["jpy"], 2),
            "pips_pct": None,
            "jpy_pct": _growth_rate_equity(
                round(daily["jpy"] - yesterday["jpy"], 2), equity_nav
            ),
            "equity_nav": equity_nav,
            "equity_source": equity_source,
        }

        def _finalise_table(table: dict[str, dict], *, label_fn) -> list[dict]:
            rows: list[dict] = []
            for key, data in table.items():
                trades = data["trades"]
                win_rate = (data["wins"] / trades) if trades else 0.0
                pf = None
                if data["loss_pips"] > 0:
                    pf = data["win_pips"] / data["loss_pips"]
                rows.append(
                    {
                        "key": key,
                        "label": label_fn(key),
                        "pips": round(data["pips"], 2),
                        "jpy": round(data["jpy"], 2),
                        "trades": trades,
                        "wins": data["wins"],
                        "losses": data["losses"],
                        "win_rate": win_rate,
                        "pf": pf,
                    }
                )
            return rows

        profit_tables = {
            "timezone": "JST",
            "year_start": year_start.date().isoformat(),
            "exclude_manual": True,
            "daily": _finalise_table(daily_table, label_fn=lambda key: key),
            "weekly": _finalise_table(
                weekly_table,
                label_fn=lambda key: f"{key} (Mon)",
            ),
            "monthly": _finalise_table(monthly_table, label_fn=lambda key: key),
        }
        profit_tables["daily"].sort(key=lambda r: r["label"], reverse=True)
        profit_tables["weekly"].sort(key=lambda r: r["label"], reverse=True)
        profit_tables["monthly"].sort(key=lambda r: r["label"], reverse=True)

        balance_series = _load_metric_series("account.balance", year_start_utc)

        def _period_bounds_for_key(key: str, kind: str) -> tuple[datetime, datetime]:
            if kind == "daily":
                start_jst = datetime.fromisoformat(key).replace(tzinfo=jst)
                end_jst = start_jst + timedelta(days=1)
            elif kind == "weekly":
                start_jst = datetime.fromisoformat(key).replace(tzinfo=jst)
                end_jst = start_jst + timedelta(days=7)
            else:
                year, month = key.split("-", 1)
                start_jst = datetime(int(year), int(month), 1, tzinfo=jst)
                if int(month) == 12:
                    end_jst = datetime(int(year) + 1, 1, 1, tzinfo=jst)
                else:
                    end_jst = datetime(int(year), int(month) + 1, 1, tzinfo=jst)
            return (
                start_jst.astimezone(timezone.utc),
                end_jst.astimezone(timezone.utc),
            )

        def _start_balance_map(keys: list[str], kind: str) -> dict[str, float | None]:
            if not balance_series:
                return {key: None for key in keys}
            intervals: list[tuple[str, datetime, datetime]] = []
            for key in keys:
                start_utc, end_utc = _period_bounds_for_key(key, kind)
                intervals.append((key, start_utc, end_utc))
            intervals.sort(key=lambda item: item[1])
            idx = 0
            out: dict[str, float | None] = {}
            for key, start_utc, end_utc in intervals:
                while idx < len(balance_series) and balance_series[idx][0] < start_utc:
                    idx += 1
                if idx < len(balance_series) and balance_series[idx][0] < end_utc:
                    out[key] = balance_series[idx][1]
                else:
                    out[key] = None
            return out

        daily_start = _start_balance_map(list(daily_table.keys()), "daily")
        weekly_start = _start_balance_map(list(weekly_table.keys()), "weekly")
        monthly_start = _start_balance_map(list(monthly_table.keys()), "monthly")

        def _apply_start_balance(rows: list[dict], start_map: dict[str, float | None]) -> None:
            for row in rows:
                key = row.get("key")
                start_balance = start_map.get(key)
                row["start_balance"] = start_balance
                if start_balance:
                    row["return_pct"] = round(row["jpy"] * 100.0 / start_balance, 2)
                else:
                    row["return_pct"] = None

        ytd_summary = {
            "year_start": year_start.date().isoformat(),
            "timezone": "JST",
            "exclude_manual": True,
            "start_balance": start_balance,
            "start_balance_source": start_balance_source,
            "current_balance": current_balance,
            "balance_growth_pct": None,
            "balance_growth_ex_cashflow_jpy": None,
            "balance_growth_ex_cashflow_pct": None,
            "bot_profit_jpy": round(ytd_bot["jpy"], 2),
            "bot_profit_pips": round(ytd_bot["pips"], 2),
            "bot_return_pct": None,
            "trades": ytd_bot["trades"],
            "wins": ytd_bot["wins"],
            "losses": ytd_bot["losses"],
            "cashflow_net": cashflow_summary["ytd"]["net"],
            "cashflow_in": cashflow_summary["ytd"]["in"],
            "cashflow_out": cashflow_summary["ytd"]["out"],
        }
        if start_balance and current_balance:
            net_growth = current_balance - start_balance
            ytd_summary["balance_growth_pct"] = round(
                net_growth * 100.0 / start_balance, 2
            )
            adj_growth = net_growth - cashflow_summary["ytd"]["net"]
            ytd_summary["balance_growth_ex_cashflow_jpy"] = round(adj_growth, 2)
            ytd_summary["balance_growth_ex_cashflow_pct"] = round(
                adj_growth * 100.0 / start_balance, 2
            )
        if start_balance:
            ytd_summary["bot_return_pct"] = round(
                ytd_summary["bot_profit_jpy"] * 100.0 / start_balance, 2
            )

        _apply_start_balance(profit_tables["daily"], daily_start)
        _apply_start_balance(profit_tables["weekly"], weekly_start)
        _apply_start_balance(profit_tables["monthly"], monthly_start)

        hourly_rows: list[dict] = []
        for hour in sorted(hourly_buckets.keys(), reverse=True):
            data = hourly_buckets[hour]
            trades = int(data["trades"])
            win_rate = (data["wins"] / trades) if trades else 0.0
            hourly_rows.append(
                {
                    "key": hour.isoformat(),
                    "label": hour.strftime("%m/%d %H:00"),
                    "pips": round(float(data["pips"]), 2),
                    "jpy": round(float(data["jpy"]), 2),
                    "trades": trades,
                    "wins": int(data["wins"]),
                    "losses": int(data["losses"]),
                    "win_rate": win_rate,
                }
            )

        hourly_trades = {
            "timezone": "JST",
            "lookback_hours": _HOURLY_LOOKBACK_HOURS,
            "exclude_manual": True,
            "hours": hourly_rows,
        }

        chart_data = _build_chart_data(
            now=now,
            year_start=year_start,
            closed_events=closed_events,
            con=self.con,
        )

        return {
            "daily": daily,
            "yesterday": yesterday,
            "daily_change": daily_change,
            "weekly": weekly,
            "total": total,
            "last_trade_at": latest_close.isoformat() if latest_close else None,
            "pockets": pockets_final,
            "profit_tables": profit_tables,
            "ytd_summary": ytd_summary,
            "hourly_trades": hourly_trades,
            "chart_data": chart_data,
            "cashflow": cashflow_summary,
        }

    def fetch_recent_trades(self, limit: int = 50) -> list[dict]:
        """UI 表示用に最新のトレードを取得"""
        service_result = _position_manager_service_request(
            "/position/fetch_recent_trades", {"limit": int(limit)}
        )
        if service_result is not None:
            if isinstance(service_result, list):
                return service_result
            return []

        self._ensure_connection_open()
        cursor = self.con.execute(
            """
            SELECT ticket_id, pocket, instrument, units, closed_units, entry_price, close_price,
                   fill_price, pl_pips, realized_pl, commission, financing,
                   entry_time, close_time, close_reason,
                   state, updated_at,
                   strategy_tag, strategy, client_order_id, entry_thesis
            FROM trades
            ORDER BY datetime(updated_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "ticket_id": row["ticket_id"],
                "pocket": row["pocket"],
                "instrument": row["instrument"],
                "units": row["units"],
                "closed_units": row["closed_units"],
                "entry_price": row["entry_price"],
                "close_price": row["close_price"],
                "fill_price": row["fill_price"],
                "pl_pips": row["pl_pips"],
                "realized_pl": row["realized_pl"],
                "commission": row["commission"],
                "financing": row["financing"],
                "entry_time": row["entry_time"],
                "close_time": row["close_time"],
                "close_reason": row["close_reason"],
                "state": row["state"],
                "updated_at": row["updated_at"],
                "strategy_tag": row["strategy_tag"],
                "strategy": row["strategy"],
                "client_order_id": row["client_order_id"],
                "entry_thesis": row["entry_thesis"],
            }
            for row in rows
            ]

    def close(self):
        # Never propagate close() to the shared position-manager service.
        # In runtime deployments we often run with SERVICE_ENABLED=1 and
        # FALLBACK_LOCAL=1; forwarding /position/close from each client would
        # close the singleton DB connection for everyone.
        if _position_manager_service_enabled():
            if self.con is None:
                return None
            try:
                self.con.close()
            except sqlite3.ProgrammingError:
                pass
            self.con = None
            logging.debug(
                "[PositionManager] close() cleaned local fallback connection only; "
                "skipped shared service /position/close call."
            )
            return None
        service_result = _position_manager_service_request("/position/close", {})
        if service_result is not None:
            return None
        if self.con is None:
            return None
        try:
            self.con.close()
        except sqlite3.ProgrammingError:
            pass
        self.con = None
