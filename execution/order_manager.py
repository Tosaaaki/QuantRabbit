"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import sqlite3
import pathlib
import re
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import os
import math
import threading
from typing import Any, Literal, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCancel, OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, TradeClose, TradeDetails

from execution.order_ids import build_client_order_id
from execution.stop_loss_policy import (
    fixed_sl_mode,
    stop_loss_disabled_for_pocket,
    trailing_sl_allowed,
)
from execution.section_axis import attach_section_axis

from analysis import policy_bus
from utils.secrets import get_secret
from utils.market_hours import is_market_open
from indicators.factor_cache import all_factors, get_candles_snapshot
from analysis.range_guard import detect_range_mode
from utils.metrics_logger import log_metric
from execution import strategy_guard, reentry_gate
from execution.position_manager import PositionManager, agent_client_prefixes
from execution.risk_guard import POCKET_MAX_RATIOS, MAX_LEVERAGE
from workers.common import (
    perf_guard,
    profit_guard,
    brain,
    forecast_gate,
    pattern_gate,
    strategy_control,
)
from workers.common.quality_gate import current_regime
from indicators.factor_cache import get_last_regime
from utils import signal_bus
from utils.oanda_account import get_account_snapshot
try:
    from market_data import tick_window
except Exception:  # pragma: no cover - optional
    tick_window = None

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACTICE_FLAG = get_secret("oanda_practice").lower() == "true"
except KeyError:
    PRACTICE_FLAG = False  # デフォルトは本番環境

try:
    HEDGING_ENABLED = get_secret("oanda_hedging_enabled").lower() == "true"
except KeyError:
    HEDGING_ENABLED = False

ENVIRONMENT = "practice" if PRACTICE_FLAG else "live"
POSITION_FILL = "OPEN_ONLY" if HEDGING_ENABLED else "DEFAULT"

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


# --- OANDA client runtime ---
_ORDER_OANDA_REQUEST_TIMEOUT_SEC = max(
    1.0,
    _env_float("ORDER_OANDA_REQUEST_TIMEOUT_SEC", 8.0),
)
_ORDER_OANDA_POOL_CONNECTIONS = max(
    4,
    _env_int("ORDER_OANDA_REQUEST_POOL_CONNECTIONS", 16),
)
_ORDER_OANDA_POOL_MAXSIZE = max(
    _ORDER_OANDA_POOL_CONNECTIONS,
    _env_int("ORDER_OANDA_REQUEST_POOL_MAXSIZE", 32),
)

api = API(
    access_token=TOKEN,
    environment=ENVIRONMENT,
    request_params={"timeout": _ORDER_OANDA_REQUEST_TIMEOUT_SEC},
)
try:
    _oanda_http_adapter = HTTPAdapter(
        pool_connections=_ORDER_OANDA_POOL_CONNECTIONS,
        pool_maxsize=_ORDER_OANDA_POOL_MAXSIZE,
        max_retries=0,
    )
    api.client.mount("https://", _oanda_http_adapter)
    api.client.mount("http://", _oanda_http_adapter)
except Exception:
    pass

if HEDGING_ENABLED:
    logging.info("[ORDER] Hedging mode enabled (positionFill=OPEN_ONLY).")
logging.info(
    "[ORDER] OANDA client timeout=%.1fs pool=%d/%d env=%s",
    _ORDER_OANDA_REQUEST_TIMEOUT_SEC,
    _ORDER_OANDA_POOL_CONNECTIONS,
    _ORDER_OANDA_POOL_MAXSIZE,
    ENVIRONMENT,
)


# --- Service mode (quant-order-manager externalization) ---
_ORDER_MANAGER_SERVICE_ENABLED = _env_bool("ORDER_MANAGER_SERVICE_ENABLED", False)
_ORDER_MANAGER_SERVICE_URL = os.getenv("ORDER_MANAGER_SERVICE_URL", "")
_ORDER_MANAGER_SERVICE_TIMEOUT = max(
    0.5,
    float(_env_float("ORDER_MANAGER_SERVICE_TIMEOUT", 5.0)),
)
_ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT = max(
    0.2,
    float(
        _env_float(
            "ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT",
            min(1.5, _ORDER_MANAGER_SERVICE_TIMEOUT),
        )
    ),
)
if _ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT > _ORDER_MANAGER_SERVICE_TIMEOUT:
    _ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT = _ORDER_MANAGER_SERVICE_TIMEOUT
_ORDER_MANAGER_SERVICE_POOL_CONNECTIONS = max(
    2,
    _env_int("ORDER_MANAGER_SERVICE_POOL_CONNECTIONS", 8),
)
_ORDER_MANAGER_SERVICE_POOL_MAXSIZE = max(
    _ORDER_MANAGER_SERVICE_POOL_CONNECTIONS,
    _env_int("ORDER_MANAGER_SERVICE_POOL_MAXSIZE", 32),
)
_ORDER_MANAGER_SERVICE_FALLBACK_LOCAL = _env_bool(
    "ORDER_MANAGER_SERVICE_FALLBACK_LOCAL", False
)
_ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE = _env_bool(
    "ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE", False
)
_ORDER_MANAGER_SERVICE_SESSION: requests.Session | None = None
_ORDER_MANAGER_SERVICE_SESSION_LOCK = threading.Lock()


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _env_float_int(name: str, default: float | int) -> float | int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw) if isinstance(default, float) else int(float(raw))
    except (TypeError, ValueError):
        return default


def _order_manager_service_enabled() -> bool:
    return bool(
        _ORDER_MANAGER_SERVICE_ENABLED
        and _ORDER_MANAGER_SERVICE_URL
        and not _ORDER_MANAGER_SERVICE_URL.lower().startswith("disabled")
    )


def _order_manager_service_url(path: str) -> str:
    base = _ORDER_MANAGER_SERVICE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _order_manager_service_session() -> requests.Session:
    global _ORDER_MANAGER_SERVICE_SESSION
    session = _ORDER_MANAGER_SERVICE_SESSION
    if session is not None:
        return session
    with _ORDER_MANAGER_SERVICE_SESSION_LOCK:
        session = _ORDER_MANAGER_SERVICE_SESSION
        if session is not None:
            return session
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=_ORDER_MANAGER_SERVICE_POOL_CONNECTIONS,
            pool_maxsize=_ORDER_MANAGER_SERVICE_POOL_MAXSIZE,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _ORDER_MANAGER_SERVICE_SESSION = session
    return session


def _order_manager_service_reset_session() -> None:
    global _ORDER_MANAGER_SERVICE_SESSION
    with _ORDER_MANAGER_SERVICE_SESSION_LOCK:
        session = _ORDER_MANAGER_SERVICE_SESSION
        _ORDER_MANAGER_SERVICE_SESSION = None
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


def _order_manager_service_payload_summary(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    keys = (
        "instrument",
        "pocket",
        "strategy_tag",
        "side",
        "units",
        "raw_units",
        "entry_probability",
        "client_order_id",
        "trade_id",
        "order_id",
        "min_units",
    )
    summary = {k: payload.get(k) for k in keys if k in payload}
    if "entry_thesis" in payload and isinstance(payload.get("entry_thesis"), dict):
        thesis = payload.get("entry_thesis") or {}
        summary["entry_thesis_keys"] = sorted(list(thesis.keys()))[:20]
    if "meta" in payload and isinstance(payload.get("meta"), dict):
        meta = payload.get("meta") or {}
        summary["meta_keys"] = sorted(list(meta.keys()))[:20]
    return summary


def _order_manager_service_call(path: str, payload: dict) -> dict:
    url = _order_manager_service_url(path)
    normalized_payload = _normalize_for_json(payload)
    session = _order_manager_service_session()
    try:
        response = session.post(
            url,
            json=normalized_payload,
            timeout=(
                float(_ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT),
                float(_ORDER_MANAGER_SERVICE_TIMEOUT),
            ),
            headers={"Content-Type": "application/json"},
        )
    except requests.RequestException:
        _order_manager_service_reset_session()
        raise
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict):
        return body
    raise RuntimeError(f"Unexpected order_manager service response type: {type(body).__name__}")


def _extract_service_payload(path: str, payload: Any) -> Any:
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
        raise RuntimeError(f"order_manager service error for {path}: {msg}")
    if "result" in payload:
        return payload["result"]
    return payload


def _order_manager_service_request(
    path: str,
    payload: dict,
) -> Any:
    if not _order_manager_service_enabled():
        return None
    try:
        return _extract_service_payload(path, _order_manager_service_call(path, payload))
    except Exception as exc:
        logging.warning(
            "[ORDER] order_manager service call failed path=%s payload=%s err=%s",
            path,
            _order_manager_service_payload_summary(payload),
            exc,
        )
        if not _ORDER_MANAGER_SERVICE_FALLBACK_LOCAL:
            raise
        return None


async def _order_manager_service_request_async(
    path: str,
    payload: dict,
) -> Any:
    if not _order_manager_service_enabled():
        return None
    try:
        return _extract_service_payload(
            path, await asyncio.to_thread(_order_manager_service_call, path, payload)
        )
    except Exception as exc:
        logging.warning(
            "[ORDER] order_manager service call failed path=%s payload=%s err=%s",
            path,
            _order_manager_service_payload_summary(payload),
            exc,
        )
        if not _ORDER_MANAGER_SERVICE_FALLBACK_LOCAL:
            raise
        return None


def _should_persist_preservice_order_log() -> bool:
    if _ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE:
        return True
    return not _order_manager_service_enabled()


def _forecast_service_enabled() -> bool:
    return bool(
        _FORECAST_SERVICE_ENABLED
        and _FORECAST_SERVICE_URL
        and not _FORECAST_SERVICE_URL.lower().startswith("disabled")
    )


def _forecast_service_url(path: str) -> str:
    base = _FORECAST_SERVICE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _forecast_service_request(path: str, payload: dict) -> Any:
    if not _forecast_service_enabled():
        return None
    try:
        normalized_payload = _normalize_for_json(payload)
        response = requests.post(
            _forecast_service_url(path),
            json=normalized_payload,
            timeout=float(_FORECAST_SERVICE_TIMEOUT),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"forecast service returned non-dict body: {type(body).__name__}")
        if "ok" not in body:
            return body.get("result", body)
        if not bool(body.get("ok")):
            message = str(
                body.get("error")
                or body.get("detail")
                or body.get("message")
                or body.get("reason")
                or "forecast service returned ok=false"
            )
            raise RuntimeError(f"forecast service error for {path}: {message}")
        if "result" in body:
            return body["result"]
        return body
    except Exception as exc:
        raise RuntimeError(f"forecast service request failed path={path} err={exc}") from exc


def _forecast_service_decision_from_payload(
    payload: Any,
) -> forecast_gate.ForecastDecision | None:
    if not isinstance(payload, dict):
        return None
    if "allowed" not in payload:
        return None
    return forecast_gate.ForecastDecision(
        allowed=bool(payload.get("allowed")),
        scale=_as_float(payload.get("scale"), 0.0),
        reason=str(payload.get("reason", "")),
        horizon=str(payload.get("horizon") or ""),
        edge=_as_float(payload.get("edge"), 0.0),
        p_up=_as_float(payload.get("p_up"), 0.0),
        rebound_probability=_as_float(payload.get("rebound_probability"), None),
        expected_pips=_as_float(payload.get("expected_pips"), None),
        anchor_price=_as_float(payload.get("anchor_price"), None),
        target_price=_as_float(payload.get("target_price"), None),
        range_low_pips=_as_float(payload.get("range_low_pips"), None),
        range_high_pips=_as_float(payload.get("range_high_pips"), None),
        range_sigma_pips=_as_float(payload.get("range_sigma_pips"), None),
        range_low_price=_as_float(payload.get("range_low_price"), None),
        range_high_price=_as_float(payload.get("range_high_price"), None),
        tp_pips_hint=_as_float(payload.get("tp_pips_hint"), None),
        target_reach_prob=_as_float(payload.get("target_reach_prob"), None),
        sl_pips_cap=_as_float(payload.get("sl_pips_cap"), None),
        rr_floor=_as_float(payload.get("rr_floor"), None),
        feature_ts=payload.get("feature_ts") if payload.get("feature_ts") is not None else None,
        source=str(payload.get("source")) if payload.get("source") is not None else None,
        style=str(payload.get("style")) if payload.get("style") is not None else None,
        trend_strength=_as_float(payload.get("trend_strength"), None),
        range_pressure=_as_float(payload.get("range_pressure"), None),
        future_flow=payload.get("future_flow"),
        volatility_state=(
            str(payload.get("volatility_state"))
            if payload.get("volatility_state") is not None
            else None
        ),
        trend_state=(
            str(payload.get("trend_state")) if payload.get("trend_state") is not None else None
        ),
        range_state=(
            str(payload.get("range_state")) if payload.get("range_state") is not None else None
        ),
        volatility_rank=_as_float(payload.get("volatility_rank"), None),
        regime_score=_as_float(payload.get("regime_score"), None),
        leading_indicator=(
            str(payload.get("leading_indicator"))
            if payload.get("leading_indicator") is not None
            else None
        ),
        leading_indicator_strength=_as_float(payload.get("leading_indicator_strength"), None),
        tf_confluence_score=_as_float(payload.get("tf_confluence_score"), None),
        tf_confluence_count=_as_int(payload.get("tf_confluence_count"), None),
        tf_confluence_horizons=(
            str(payload.get("tf_confluence_horizons"))
            if payload.get("tf_confluence_horizons") is not None
            else None
        ),
    )


def _forecast_decide_with_service(
    *,
    strategy_tag: Optional[str],
    pocket: str,
    side: str,
    units: int,
    entry_thesis: Optional[dict],
    meta: Optional[dict],
) -> forecast_gate.ForecastDecision | None:
    if _forecast_service_enabled():
        try:
            result = _forecast_service_request(
                "/forecast/decide",
                {
                    "strategy_tag": strategy_tag,
                    "pocket": pocket,
                    "side": side,
                    "units": units,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
        except Exception as exc:
            logging.warning(
                "[FORECAST] service call failed path=%s err=%s",
                "/forecast/decide",
                exc,
            )
            if not _FORECAST_SERVICE_FALLBACK_LOCAL:
                raise
            return forecast_gate.decide(
                strategy_tag=strategy_tag,
                pocket=pocket,
                side=side,
                units=units,
                entry_thesis=entry_thesis,
                meta=meta,
            )
        if result is None:
            return None
        decision = _forecast_service_decision_from_payload(result)
        if decision is not None:
            return decision
        raise RuntimeError("forecast service returned invalid decision payload")
    return forecast_gate.decide(
        strategy_tag=strategy_tag,
        pocket=pocket,
        side=side,
        units=units,
        entry_thesis=entry_thesis,
        meta=meta,
    )


async def _invoke_order_manager_service(path: str, payload: dict) -> dict:
    return await asyncio.to_thread(_order_manager_service_call, path, payload)


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    if pct == 100.0:
        return sorted_vals[-1]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_env_prefix(value: object) -> Optional[str]:
    if not isinstance(value, str):
        if value is None:
            return None
        value = str(value)
    value = value.strip().upper()
    return value or None


def _infer_env_prefix_from_strategy_tag(strategy_tag: Optional[str]) -> Optional[str]:
    tag = str(strategy_tag or "").strip().upper()
    if not tag:
        return None
    if tag.startswith("SCALP_PING_5S_FLOW"):
        return "SCALP_PING_5S_FLOW"
    if tag.startswith("SCALP_PING_5S_B"):
        return "SCALP_PING_5S_B"
    if tag.startswith("SCALP_PING_5S"):
        return "SCALP_PING_5S"
    return None


def _resolve_env_prefix_for_order(
    entry_env_prefix: Optional[str],
    meta_env_prefix: Optional[str],
    strategy_tag: Optional[str],
) -> Optional[str]:
    candidates: list[str] = []
    for prefix in (entry_env_prefix, meta_env_prefix):
        if not prefix:
            continue
        if prefix not in candidates:
            candidates.append(prefix)
    inferred_env_prefix = _infer_env_prefix_from_strategy_tag(strategy_tag)
    if inferred_env_prefix is not None:
        return inferred_env_prefix
    if candidates:
        return candidates[0]
    return None


def _entry_execution_deadline_sec(pocket: Optional[str]) -> float:
    """Return entry preflight deadline seconds (0 = disabled).

    NOTE: This is opt-in to avoid changing existing trading behavior.
    """

    p = (pocket or "").strip().lower()
    suffix = p.upper() if p else ""
    raw = None
    if suffix:
        raw = os.getenv(f"ENTRY_EXECUTION_DEADLINE_SEC_{suffix}")
    if raw is None:
        raw = os.getenv("ENTRY_EXECUTION_DEADLINE_SEC")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


_ENTRY_QUALITY_RANGE_TAG_HINTS = (
    "range",
    "revert",
    "reversal",
    "fader",
    "magnet",
    "mean",
    "vwap",
)
_ENTRY_QUALITY_TREND_TAG_HINTS = (
    "trend",
    "momentum",
    "break",
    "burst",
    "impulse",
    "runner",
    "pullback",
    "spike",
    "rider",
)


def _normalize_regime_label(label: object) -> str | None:
    if label in (None, "", 0, False):
        return None
    text = str(label).strip()
    if not text:
        return None
    key = text.strip().lower()
    if key == "trend":
        return "Trend"
    if key == "range":
        return "Range"
    if key == "breakout":
        return "Breakout"
    if key == "mixed":
        return "Mixed"
    if key == "event":
        return "Event"
    return None


def _entry_quality_regime_from_thesis(entry_thesis: Optional[dict], pocket: str) -> str | None:
    if not isinstance(entry_thesis, dict):
        return None
    reg = entry_thesis.get("regime")
    macro = None
    micro = None
    if isinstance(reg, dict):
        macro = reg.get("macro") or reg.get("macro_regime") or reg.get("reg_macro")
        micro = reg.get("micro") or reg.get("micro_regime") or reg.get("reg_micro")
    macro = macro or entry_thesis.get("macro_regime") or entry_thesis.get("reg_macro")
    micro = micro or entry_thesis.get("micro_regime") or entry_thesis.get("reg_micro")
    if (pocket or "").strip().lower() == "macro":
        return _normalize_regime_label(macro)
    return _normalize_regime_label(micro)


def _infer_entry_quality_style(
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
) -> Literal["trend", "range"] | None:
    range_votes = 0
    trend_votes = 0
    if isinstance(entry_thesis, dict):
        if entry_thesis.get("range_active") not in (None, False, "", 0):
            range_votes += 1
        profile = entry_thesis.get("profile")
        if profile:
            p = str(profile).strip().lower()
            if any(key in p for key in ("range", "revert", "mean", "mr")):
                range_votes += 1
            if any(key in p for key in ("trend", "momentum", "break", "impulse")):
                trend_votes += 1
        for key in ("mr_guard", "mr_overlay", "entry_mean", "reversion_failure"):
            if entry_thesis.get(key) not in (None, False, "", 0):
                range_votes += 1
                break
        for key in ("trend_bias", "trend_score"):
            if entry_thesis.get(key) not in (None, False, "", 0):
                trend_votes += 1
                break
    tag = str(strategy_tag or "").strip().lower()
    if tag:
        if any(key in tag for key in _ENTRY_QUALITY_RANGE_TAG_HINTS):
            range_votes += 1
        if any(key in tag for key in _ENTRY_QUALITY_TREND_TAG_HINTS):
            trend_votes += 1
    if range_votes and not trend_votes:
        return "range"
    if trend_votes and not range_votes:
        return "trend"
    return None


def _entry_quality_confidence_value(confidence: Optional[int], entry_thesis: Optional[dict]) -> int | None:
    raw = confidence
    if raw is None and isinstance(entry_thesis, dict):
        raw = entry_thesis.get("confidence")  # type: ignore[assignment]
    if raw is None:
        return None
    try:
        val = int(raw)
    except Exception:
        return None
    return max(0, min(100, val))


def _entry_quality_regime_gate_decision(
    *,
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
    confidence: Optional[int],
) -> tuple[bool, str | None, dict]:
    """
    Opt-in gate: when strategy style and current regime disagree, require higher confidence.
    """
    if not _env_bool("ORDER_ENTRY_QUALITY_REGIME_PENALTY_ENABLED", False):
        return True, None, {}
    pocket_key = (pocket or "").strip().lower()
    if not pocket_key or pocket_key == "manual":
        return True, None, {}
    required = _env_int(
        f"ORDER_ENTRY_QUALITY_REGIME_MISMATCH_MIN_CONF_{pocket_key.upper()}",
        0,
    )
    if required <= 0:
        return True, None, {}
    style = _infer_entry_quality_style(strategy_tag=strategy_tag, entry_thesis=entry_thesis)
    regime = _entry_quality_regime_from_thesis(entry_thesis=entry_thesis, pocket=pocket_key)
    details: dict[str, object] = {
        "style": style,
        "regime": regime,
        "required_conf": required,
    }
    if style is None or regime in (None, "Mixed", "Event"):
        return True, None, details

    mismatch_reason = None
    if style == "trend" and regime == "Range":
        mismatch_reason = "trend_in_range"
    elif style == "range" and regime == "Trend":
        mismatch_reason = "range_in_trend"
    elif style == "range" and regime == "Breakout":
        mismatch_reason = "range_in_breakout"
    if mismatch_reason is None:
        details["mismatch"] = False
        return True, None, details

    conf_val = _entry_quality_confidence_value(confidence, entry_thesis)
    details["mismatch"] = True
    details["mismatch_reason"] = mismatch_reason
    details["confidence"] = conf_val
    if conf_val is None:
        # If we don't have a confidence score, do not block (safe default).
        return True, None, details
    if conf_val < required:
        return False, "entry_quality_regime_confidence", details
    return True, None, details


def _entry_quality_microstructure_gate_decision(
    *,
    pocket: str,
) -> tuple[bool, str | None, dict]:
    """
    Opt-in gate: require healthy tick window (fresh + dense enough) for scalp-style entries.
    """
    if not _env_bool("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_ENABLED", False):
        return True, None, {}
    pocket_key = (pocket or "").strip().lower()
    if not pocket_key or pocket_key == "manual":
        return True, None, {}
    min_density = _env_float(
        f"ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY_{pocket_key.upper()}",
        0.0,
    )
    if min_density <= 0.0:
        return True, None, {}
    window_sec = max(1.0, _env_float("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_WINDOW_SEC", 60.0))
    max_age_ms = max(0.0, _env_float("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MAX_AGE_MS", 2500.0))
    min_span_ratio = _env_float("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_SPAN_RATIO", 0.7)
    min_span_ratio = max(0.0, min(1.0, float(min_span_ratio)))
    details: dict[str, object] = {
        "window_sec": round(window_sec, 3),
        "min_tick_density": round(float(min_density), 4),
        "max_age_ms": round(float(max_age_ms), 1),
        "min_span_ratio": round(float(min_span_ratio), 3),
    }
    if tick_window is None:
        return False, "entry_quality_microstructure_missing", details
    try:
        ticks = tick_window.recent_ticks(seconds=window_sec)
    except Exception as exc:
        details["error"] = str(exc)
        return False, "entry_quality_microstructure_error", details
    if not ticks:
        details["tick_count"] = 0
        return False, "entry_quality_microstructure_empty", details
    details["tick_count"] = len(ticks)
    last_epoch = _as_float(ticks[-1].get("epoch"))
    if last_epoch is None:
        return False, "entry_quality_microstructure_bad_epoch", details
    now = time.time()
    age_ms = max(0.0, (now - float(last_epoch)) * 1000.0)
    details["age_ms"] = round(age_ms, 1)
    if max_age_ms > 0.0 and age_ms > max_age_ms:
        return False, "entry_quality_microstructure_stale", details
    first_epoch = _as_float(ticks[0].get("epoch")) or last_epoch
    span_sec = max(0.0, float(last_epoch) - float(first_epoch))
    span_ratio = span_sec / window_sec if window_sec > 0.0 else 0.0
    details["span_sec"] = round(span_sec, 3)
    details["span_ratio"] = round(span_ratio, 3)
    if min_span_ratio > 0.0 and span_ratio < min_span_ratio:
        return False, "entry_quality_microstructure_span", details
    tick_density = len(ticks) / window_sec if window_sec > 0.0 else 0.0
    details["tick_density"] = round(tick_density, 4)
    if tick_density < min_density:
        return False, "entry_quality_microstructure_density", details
    return True, None, details


# Apply policy overlay gates even when signal gate is disabled.
_POLICY_GATE_ENABLED = _env_bool(
    "ORDER_POLICY_GATE_ENABLED",
    _env_bool("SIGNAL_GATE_POLICY_ENABLED", False),
)

# Block entries when factor cache is stale (configurable by env).
_ENTRY_FACTOR_MAX_AGE_SEC = float(os.getenv("ENTRY_FACTOR_MAX_AGE_SEC", "600"))
_ENTRY_FACTOR_STALE_ALLOW_POCKETS = _env_csv_set(
    "ENTRY_FACTOR_STALE_ALLOW_POCKETS",
    "micro,scalp,scalp_fast",
)
_EXIT_CONTEXT_ENABLED = _env_bool("ORDER_EXIT_CONTEXT_ENABLED", True)
_EXIT_END_REVERSAL_ENABLED = _env_bool("EXIT_END_REVERSAL_ENABLED", True)
_EXIT_END_REVERSAL_SCORE_MIN = max(0.0, min(1.0, _env_float("EXIT_END_REVERSAL_SCORE_MIN", 0.58)))
_EXIT_END_REVERSAL_MIN_PROFIT_PIPS = max(0.0, _env_float("EXIT_END_REVERSAL_MIN_PROFIT_PIPS", 0.2))
_EXIT_END_REVERSAL_MAX_FACTOR_AGE_SEC = max(0.0, _env_float("EXIT_END_REVERSAL_MAX_FACTOR_AGE_SEC", 150.0))
_EXIT_END_REVERSAL_DEFAULT_REASONS = (
    "take_profit",
    "take_profit_zone",
    "rsi_take",
    "lock_floor",
    "trail_take",
    "trail_lock",
    "profit_lock",
    "lock_trail",
    "near_be",
    "range_timeout",
    "candle_*",
)

# Strategy-level BE/TP protection overrides (YAML).
try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

_STRATEGY_PROTECTION_ENABLED = _env_bool("STRATEGY_PROTECTION_ENABLED", True)
_STRATEGY_PROTECTION_PATH = pathlib.Path(
    os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml")
)
_STRATEGY_PROTECTION_TTL_SEC = _env_float("STRATEGY_PROTECTION_TTL_SEC", 12.0)
_STRATEGY_PROTECTION_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_STRATEGY_ALIAS_BASE = {
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "donchian55": "Donchian55",
    "h1momentum": "H1Momentum",
    "microlevelreactor": "MicroLevelReactor",
    "microrangebreak": "MicroRangeBreak",
    "microvwapbound": "MicroVWAPBound",
    "momentumburst": "MomentumBurst",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}


def _base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip() or text
    alias = _STRATEGY_ALIAS_BASE.get(base.lower())
    return alias or base


def _load_strategy_protection_config() -> dict:
    if not _STRATEGY_PROTECTION_ENABLED:
        return {"defaults": {}, "strategies": {}}
    now = time.monotonic()
    cached_ts = float(_STRATEGY_PROTECTION_CACHE.get("ts") or 0.0)
    if (now - cached_ts) < _STRATEGY_PROTECTION_TTL_SEC and isinstance(
        _STRATEGY_PROTECTION_CACHE.get("data"), dict
    ):
        return _STRATEGY_PROTECTION_CACHE["data"]  # type: ignore[return-value]
    # Keep last-known-good config on transient read/parse errors (deploy writes, partial files, etc).
    payload: dict[str, Any] = {"defaults": {}, "strategies": {}}
    if isinstance(_STRATEGY_PROTECTION_CACHE.get("data"), dict):
        payload = _STRATEGY_PROTECTION_CACHE.get("data")  # type: ignore[assignment]
    if yaml is not None and _STRATEGY_PROTECTION_PATH.exists():
        try:
            loaded = yaml.safe_load(_STRATEGY_PROTECTION_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                payload = loaded
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "[ORDER] Strategy protection config load failed (using cached): %s",
                exc,
            )
    _STRATEGY_PROTECTION_CACHE["ts"] = now
    _STRATEGY_PROTECTION_CACHE["data"] = payload
    return payload


def _strategy_override(config: dict, strategy_tag: Optional[str]) -> dict:
    if not isinstance(config, dict):
        return {}
    strategies = config.get("strategies")
    if not isinstance(strategies, dict) or not strategy_tag:
        return {}
    base = _base_strategy_tag(strategy_tag)
    candidates = [
        strategy_tag,
        base,
        strategy_tag.lower(),
        base.lower(),
    ]
    for key in candidates:
        if not key:
            continue
        override = strategies.get(key)
        if isinstance(override, dict):
            return override
    return {}


def _strategy_tag_from_client_id(client_order_id: Optional[str]) -> str:
    if not client_order_id:
        return ""
    text = str(client_order_id).strip()
    if not text.startswith("qr-"):
        return ""
    parts = text.split("-", 3)
    if len(parts) < 4:
        return ""
    return parts[3].strip()


def _reject_entry_by_control(
    strategy_tag: Optional[str],
    *,
    pocket: str,
) -> bool:
    if str(pocket).strip().lower() == "manual":
        return False
    if not strategy_tag:
        return False
    return not strategy_control.can_enter(strategy_tag, default=True)


def _reject_exit_by_control(
    strategy_tag: Optional[str],
    *,
    pocket: str,
) -> bool:
    if str(pocket).strip().lower() == "manual":
        return False
    if not strategy_tag:
        return False
    return not strategy_control.can_exit(strategy_tag, default=True)


def _reason_matches_tokens(exit_reason: Optional[str], tokens: list[str]) -> bool:
    if not exit_reason:
        return False
    reason_key = str(exit_reason).strip().lower()
    if not reason_key:
        return False
    for token in tokens:
        t = str(token).strip().lower()
        if not t:
            continue
        if t.endswith("*"):
            if reason_key.startswith(t[:-1]):
                return True
        elif reason_key == t:
            return True
    return False


def _allow_negative_near_be(exit_reason: Optional[str], est_pips: Optional[float]) -> bool:
    if _EXIT_ALLOW_NEGATIVE_NEAR_BE_PIPS <= 0:
        return False
    if est_pips is None or est_pips >= 0:
        return False
    if est_pips < -_EXIT_ALLOW_NEGATIVE_NEAR_BE_PIPS:
        return False
    tokens = list(_EXIT_ALLOW_NEGATIVE_NEAR_BE_REASONS)
    if not tokens:
        return False
    return _reason_matches_tokens(exit_reason, tokens)


def _strategy_neg_exit_policy(strategy_tag: Optional[str]) -> dict:
    cfg = _load_strategy_protection_config()
    defaults = {}
    if isinstance(cfg, dict):
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
    neg_defaults = defaults.get("neg_exit") if isinstance(defaults, dict) else None
    merged: dict[str, Any] = dict(neg_defaults) if isinstance(neg_defaults, dict) else {}
    override = _strategy_override(cfg, strategy_tag)
    neg_override = override.get("neg_exit") if isinstance(override, dict) else None
    if isinstance(neg_override, dict):
        merged.update(neg_override)
    return merged


def _min_profit_pips(pocket: Optional[str], strategy_tag: Optional[str]) -> Optional[float]:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}

    def _pick(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict) and pocket:
            for key in (pocket, str(pocket).lower(), str(pocket).upper()):
                if key in value:
                    return _as_float(value.get(key))
            return None
        return _as_float(value)

    selected = _pick(defaults.get("min_profit_pips"))
    override = _strategy_override(cfg, strategy_tag)
    if isinstance(override, dict):
        picked = _pick(override.get("min_profit_pips"))
        if picked is not None:
            selected = picked

    if selected is None and pocket:
        selected = _as_float(os.getenv(f"EXIT_MIN_PROFIT_PIPS_{str(pocket).upper()}"))
    if selected is None:
        selected = _as_float(os.getenv("EXIT_MIN_PROFIT_PIPS"))
    if selected is None or selected <= 0:
        return None
    return float(selected)


def _min_profit_ratio(pocket: Optional[str], strategy_tag: Optional[str]) -> Optional[float]:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}

    def _pick(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict) and pocket:
            for key in (pocket, str(pocket).lower(), str(pocket).upper()):
                if key in value:
                    return _as_float(value.get(key))
            return None
        return _as_float(value)

    selected = _pick(defaults.get("min_profit_ratio"))
    override = _strategy_override(cfg, strategy_tag)
    if isinstance(override, dict):
        picked = _pick(override.get("min_profit_ratio"))
        if picked is not None:
            selected = picked

    if selected is None and pocket:
        selected = _as_float(os.getenv(f"EXIT_MIN_PROFIT_RATIO_{str(pocket).upper()}"))
    if selected is None:
        selected = _as_float(os.getenv("EXIT_MIN_PROFIT_RATIO"))
    if selected is None or selected <= 0:
        return None
    return float(selected)


def _min_profit_ratio_reasons(strategy_tag: Optional[str]) -> set[str]:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    override = _strategy_override(cfg, strategy_tag)
    raw = None
    if isinstance(override, dict) and override.get("min_profit_ratio_reasons") is not None:
        raw = override.get("min_profit_ratio_reasons")
    if raw is None:
        raw = defaults.get("min_profit_ratio_reasons")
    if raw is None:
        raw = os.getenv("EXIT_MIN_PROFIT_RATIO_REASONS", "")
    if isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        items = str(raw).split(",") if raw else []
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _min_profit_ratio_min_tp_pips(strategy_tag: Optional[str]) -> float:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    override = _strategy_override(cfg, strategy_tag)
    value = None
    if isinstance(override, dict) and override.get("min_profit_ratio_min_tp_pips") is not None:
        value = _as_float(override.get("min_profit_ratio_min_tp_pips"))
    if value is None:
        value = _as_float(defaults.get("min_profit_ratio_min_tp_pips"))
    if value is None:
        value = _as_float(os.getenv("EXIT_MIN_PROFIT_RATIO_MIN_TP_PIPS"))
    if value is None or value < 0:
        return 0.0
    return float(value)


def _bounded01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _first_float(*values: object) -> Optional[float]:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _reason_tokens(raw: object, fallback: tuple[str, ...]) -> list[str]:
    if raw is None:
        return [str(token) for token in fallback]
    if isinstance(raw, (list, tuple, set)):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return items or [str(token) for token in fallback]


def _end_reversal_policy(strategy_tag: Optional[str]) -> dict[str, Any]:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    merged: dict[str, Any] = {}
    defaults_cfg = defaults.get("end_reversal_exit") if isinstance(defaults, dict) else None
    if isinstance(defaults_cfg, dict):
        merged.update(defaults_cfg)
    override = _strategy_override(cfg, strategy_tag)
    override_cfg = override.get("end_reversal_exit") if isinstance(override, dict) else None
    if isinstance(override_cfg, dict):
        merged.update(override_cfg)

    score_min = _as_float(merged.get("score_min"), _EXIT_END_REVERSAL_SCORE_MIN)
    min_profit_pips = _as_float(merged.get("min_profit_pips"), _EXIT_END_REVERSAL_MIN_PROFIT_PIPS)
    max_factor_age_sec = _as_float(
        merged.get("max_factor_age_sec"),
        _EXIT_END_REVERSAL_MAX_FACTOR_AGE_SEC,
    )
    rsi_center = _as_float(merged.get("rsi_center"), 35.0)
    rsi_band = _as_float(merged.get("rsi_band"), 12.0)
    adx_center = _as_float(merged.get("adx_center"), 24.0)
    adx_band = _as_float(merged.get("adx_band"), 12.0)
    gap_ref_pips = _as_float(merged.get("gap_ref_pips"), 8.0)
    gap_band_pips = _as_float(merged.get("gap_band_pips"), 10.0)
    vwap_ref = _as_float(merged.get("vwap_ref"), 16.0)
    slope_ref_pips = _as_float(merged.get("slope_ref_pips"), 6.0)

    return {
        "enabled": _coerce_bool(merged.get("enabled"), _EXIT_END_REVERSAL_ENABLED),
        "score_min": _bounded01(score_min if score_min is not None else _EXIT_END_REVERSAL_SCORE_MIN),
        "min_profit_pips": max(0.0, min_profit_pips if min_profit_pips is not None else _EXIT_END_REVERSAL_MIN_PROFIT_PIPS),
        "max_factor_age_sec": max(0.0, max_factor_age_sec if max_factor_age_sec is not None else _EXIT_END_REVERSAL_MAX_FACTOR_AGE_SEC),
        "reasons": _reason_tokens(merged.get("reasons"), _EXIT_END_REVERSAL_DEFAULT_REASONS),
        "rsi_center": max(1.0, rsi_center if rsi_center is not None else 35.0),
        "rsi_band": max(0.5, rsi_band if rsi_band is not None else 12.0),
        "adx_center": max(1.0, adx_center if adx_center is not None else 24.0),
        "adx_band": max(0.5, adx_band if adx_band is not None else 12.0),
        "gap_ref_pips": max(0.1, gap_ref_pips if gap_ref_pips is not None else 8.0),
        "gap_band_pips": max(0.1, gap_band_pips if gap_band_pips is not None else 10.0),
        "vwap_ref": max(0.1, vwap_ref if vwap_ref is not None else 16.0),
        "slope_ref_pips": max(0.1, slope_ref_pips if slope_ref_pips is not None else 6.0),
    }


def _exit_end_reversal_eval(
    *,
    exit_reason: Optional[str],
    strategy_tag: Optional[str],
    units_ctx: int,
    est_pips: Optional[float],
    exit_context: Optional[dict],
    instrument: Optional[str],
    policy: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = dict(policy) if isinstance(policy, dict) else _end_reversal_policy(strategy_tag)
    enabled = bool(cfg.get("enabled"))
    reason_tokens = list(cfg.get("reasons") or [])
    reason_match = _reason_matches_tokens(exit_reason, reason_tokens)
    min_profit_pips = float(cfg.get("min_profit_pips") or 0.0)
    score_min = _bounded01(float(cfg.get("score_min") or 0.0))
    factor_age = _as_float((exit_context or {}).get("factor_age_m1_sec"))
    max_age = max(0.0, float(cfg.get("max_factor_age_sec") or 0.0))
    side = "long" if units_ctx > 0 else ("short" if units_ctx < 0 else "flat")
    pip = 0.01 if str(instrument or "").upper().endswith("_JPY") else 0.0001

    result: dict[str, Any] = {
        "enabled": enabled,
        "triggered": False,
        "score": 0.0,
        "score_min": score_min,
        "side": side,
        "reason": exit_reason,
        "reason_match": reason_match,
        "min_profit_pips": min_profit_pips,
        "est_pips": _as_float(est_pips),
        "components": {},
    }
    if not enabled:
        result["state"] = "disabled"
        return result
    if side == "flat":
        result["state"] = "no_units"
        return result
    if not reason_match:
        result["state"] = "reason_mismatch"
        return result
    if est_pips is None or est_pips < min_profit_pips:
        result["state"] = "insufficient_profit"
        return result
    if max_age > 0 and factor_age is not None and factor_age > max_age:
        result["state"] = "stale_factors"
        result["factor_age_m1_sec"] = factor_age
        return result

    factors = (exit_context or {}).get("factors") if isinstance(exit_context, dict) else None
    if not isinstance(factors, dict):
        result["state"] = "missing_factors"
        return result

    def _f(tf: str, key: str) -> Optional[float]:
        bucket = factors.get(tf)
        if not isinstance(bucket, dict):
            return None
        return _as_float(bucket.get(key))

    rsi = _first_float(_f("M1", "rsi"), _f("M5", "rsi"), _f("H1", "rsi"))
    adx = _first_float(_f("M1", "adx"), _f("M5", "adx"), _f("H1", "adx"))
    vwap_gap = _first_float(_f("M1", "vwap_gap"), _f("M5", "vwap_gap"))
    ma10_m1 = _f("M1", "ma10")
    ma20_m1 = _f("M1", "ma20")
    ma10_m5 = _f("M5", "ma10")
    ma20_m5 = _f("M5", "ma20")
    gap_m1 = None if ma10_m1 is None or ma20_m1 is None else (ma10_m1 - ma20_m1) / pip
    gap_m5 = None if ma10_m5 is None or ma20_m5 is None else (ma10_m5 - ma20_m5) / pip

    rsi_center = float(cfg.get("rsi_center") or 35.0)
    rsi_band = max(0.5, float(cfg.get("rsi_band") or 12.0))
    if rsi is not None:
        if side == "short":
            rsi_component = _bounded01((rsi_center - rsi) / rsi_band)
        else:
            rsi_component = _bounded01((rsi - (100.0 - rsi_center)) / rsi_band)
    else:
        rsi_component = None

    gap_ref = max(0.1, float(cfg.get("gap_ref_pips") or 8.0))
    gap_band = max(0.1, float(cfg.get("gap_band_pips") or 10.0))
    if gap_m1 is not None:
        if side == "short":
            gap_component_m1 = _bounded01((gap_m1 + gap_ref) / gap_band)
        else:
            gap_component_m1 = _bounded01(((-gap_m1) + gap_ref) / gap_band)
    else:
        gap_component_m1 = None
    if gap_m5 is not None:
        if side == "short":
            gap_component_m5 = _bounded01((gap_m5 + gap_ref) / gap_band)
        else:
            gap_component_m5 = _bounded01(((-gap_m5) + gap_ref) / gap_band)
    else:
        gap_component_m5 = None
    if gap_component_m1 is not None and gap_component_m5 is not None:
        gap_component = 0.6 * gap_component_m1 + 0.4 * gap_component_m5
    elif gap_component_m1 is not None:
        gap_component = gap_component_m1
    else:
        gap_component = gap_component_m5

    adx_center = max(1.0, float(cfg.get("adx_center") or 24.0))
    adx_band = max(0.5, float(cfg.get("adx_band") or 12.0))
    adx_component = None if adx is None else _bounded01((adx_center - adx) / adx_band)

    vwap_ref = max(0.1, float(cfg.get("vwap_ref") or 16.0))
    vwap_component = None if vwap_gap is None else _bounded01(1.0 - abs(vwap_gap) / vwap_ref)

    slope_ref = max(0.1, float(cfg.get("slope_ref_pips") or 6.0))
    slope_component = None
    if gap_m1 is not None and gap_m5 is not None:
        if side == "short":
            slope_component = _bounded01((gap_m1 - gap_m5) / slope_ref)
        else:
            slope_component = _bounded01((gap_m5 - gap_m1) / slope_ref)

    weighted_components = {
        "rsi": (rsi_component, 0.32),
        "gap": (gap_component, 0.30),
        "adx": (adx_component, 0.18),
        "vwap": (vwap_component, 0.10),
        "slope": (slope_component, 0.10),
    }
    score_num = 0.0
    score_den = 0.0
    components: dict[str, Any] = {
        "rsi": rsi,
        "adx": adx,
        "vwap_gap": vwap_gap,
        "gap_pips_m1": gap_m1,
        "gap_pips_m5": gap_m5,
    }
    for key, (value, weight) in weighted_components.items():
        if value is None:
            continue
        score_num += float(value) * weight
        score_den += weight
        components[f"{key}_score"] = round(float(value), 6)
    if score_den <= 0.0:
        result["state"] = "insufficient_signals"
        result["components"] = components
        return result

    score = _bounded01(score_num / score_den)
    triggered = bool(score >= score_min)
    result["score"] = round(score, 6)
    result["triggered"] = triggered
    result["state"] = "triggered" if triggered else "below_threshold"
    result["components"] = components
    if factor_age is not None:
        result["factor_age_m1_sec"] = round(float(factor_age), 3)
    return result


def _hold_until_profit_config() -> dict:
    cfg = _load_strategy_protection_config()
    hold = cfg.get("hold_until_profit") if isinstance(cfg, dict) else None
    return hold if isinstance(hold, dict) else {}


def _normalize_hold_ids(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    results: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text:
            results.add(text.lower())
    return results


def _hold_until_profit_targets() -> tuple[set[str], set[str], float, bool]:
    client_ids = set(_EXIT_HOLD_UNTIL_PROFIT_CLIENT_IDS)
    trade_ids = set(_EXIT_HOLD_UNTIL_PROFIT_TRADE_IDS)
    min_pips = _EXIT_HOLD_UNTIL_PROFIT_MIN_PIPS
    strict = _EXIT_HOLD_UNTIL_PROFIT_STRICT
    cfg = _hold_until_profit_config()
    if cfg:
        client_ids |= _normalize_hold_ids(cfg.get("client_order_ids") or cfg.get("client_ids"))
        trade_ids |= _normalize_hold_ids(cfg.get("trade_ids"))
        cfg_min = _as_float(cfg.get("min_profit_pips"))
        if cfg_min is not None:
            min_pips = max(0.0, cfg_min)
        if "strict" in cfg:
            strict = _coerce_bool(cfg.get("strict"), strict)
    return client_ids, trade_ids, min_pips, strict


def _hold_until_profit_match(
    trade_id: str,
    client_order_id: Optional[str],
) -> tuple[bool, float, bool]:
    if not _EXIT_HOLD_UNTIL_PROFIT_ENABLED:
        return False, _EXIT_HOLD_UNTIL_PROFIT_MIN_PIPS, _EXIT_HOLD_UNTIL_PROFIT_STRICT
    if _EXIT_HOLD_UNTIL_PROFIT_ALL:
        return True, _EXIT_HOLD_UNTIL_PROFIT_MIN_PIPS, _EXIT_HOLD_UNTIL_PROFIT_STRICT
    client_ids, trade_ids, min_pips, strict = _hold_until_profit_targets()
    if not client_ids and not trade_ids:
        return False, min_pips, strict
    if client_order_id:
        if str(client_order_id).strip().lower() in client_ids:
            return True, min_pips, strict
    if trade_id:
        if str(trade_id).strip().lower() in trade_ids:
            return True, min_pips, strict
    return False, min_pips, strict


def _factor_age_seconds(tf: str = "M1") -> float | None:
    try:
        fac = (all_factors().get(tf.upper()) or {})
    except Exception:
        return None
    ts_raw = fac.get("timestamp")
    if not ts_raw:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            ts_dt = datetime.utcfromtimestamp(float(ts_raw)).replace(tzinfo=timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - ts_dt).total_seconds())
    except Exception:
        return None


def _range_active_for_entry() -> bool:
    try:
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        if not fac_m1 or not fac_h4:
            return False
        return bool(detect_range_mode(fac_m1, fac_h4).active)
    except Exception:
        return False
# エントリーが詰まらないようデフォルトの最小ユニットを下げる。
_DEFAULT_MIN_UNITS = _env_int("ORDER_MIN_UNITS_DEFAULT", 1_000)
# Pattern gate で scale 適用後にポケット最小未満になった場合、最小値へフォールバックする。
_PATTERN_GATE_SCALE_TO_MIN_UNITS = _env_bool(
    "ORDER_PATTERN_GATE_SCALE_TO_MIN_UNITS", False
)
# Macro も環境変数で可変（デフォルト 2,000 units、最低でも DEFAULT_MIN を確保）
_MACRO_MIN_UNITS_DEFAULT = max(_env_int("ORDER_MIN_UNITS_MACRO", 2_000), _DEFAULT_MIN_UNITS)
_MIN_UNITS_BY_POCKET: dict[str, int] = {
    "micro": _env_int("ORDER_MIN_UNITS_MICRO", _DEFAULT_MIN_UNITS),
    "macro": _env_int("ORDER_MIN_UNITS_MACRO", _MACRO_MIN_UNITS_DEFAULT),
    # scalp 系も同じ下限を使う（環境変数で上書き可）
    "scalp_fast": _env_int("ORDER_MIN_UNITS_SCALP_FAST", _DEFAULT_MIN_UNITS),
    "scalp": _env_int("ORDER_MIN_UNITS_SCALP", _DEFAULT_MIN_UNITS),
}
# Default MTF hints for workers without explicit entry_thesis TFs.
_DEFAULT_ENTRY_THESIS_TFS: dict[str, tuple[str, str]] = {
    "macro": ("H4", "H1"),
    "micro": ("H1", "M5"),
    "scalp": ("M5", "M1"),
    "scalp_fast": ("M5", "M1"),
}
# Entry stop-loss is controlled by execution.stop_loss_policy (global + per-pocket overrides).
TRAILING_SL_ALLOWED = trailing_sl_allowed()

_ENTRY_HARD_STOP_PIPS_DEFAULT = max(0.0, _env_float("ORDER_ENTRY_HARD_STOP_PIPS", 0.0))
_ENTRY_MAX_SL_PIPS_DEFAULT = max(0.0, _env_float("ORDER_ENTRY_MAX_SL_PIPS", 0.0))
_ENTRY_LOSS_CAP_JPY_DEFAULT = max(0.0, _env_float("ORDER_ENTRY_LOSS_CAP_JPY", 0.0))
_ENTRY_LOSS_CAP_BUFFER_PIPS_DEFAULT = max(
    0.0, _env_float("ORDER_ENTRY_LOSS_CAP_BUFFER_PIPS", 0.0)
)
_ENTRY_LOSS_CAP_NAV_BPS_DEFAULT = max(0.0, _env_float("ORDER_ENTRY_LOSS_CAP_NAV_BPS", 0.0))
_ENTRY_LOSS_CAP_NAV_MIN_JPY_DEFAULT = max(
    0.0, _env_float("ORDER_ENTRY_LOSS_CAP_NAV_MIN_JPY", 0.0)
)
_ENTRY_LOSS_CAP_NAV_MAX_JPY_DEFAULT = max(
    0.0, _env_float("ORDER_ENTRY_LOSS_CAP_NAV_MAX_JPY", 0.0)
)
_ENTRY_LOSS_CAP_NAV_SNAPSHOT_TTL_SEC = max(
    0.0, _env_float("ORDER_ENTRY_LOSS_CAP_NAV_SNAPSHOT_TTL_SEC", 1.0)
)
_ENTRY_POLICY_GENERATION = os.getenv("ORDER_ENTRY_POLICY_GENERATION", "").strip()
_JPY_PER_PIP_PER_UNIT_USDJPY = 0.01


def _strategy_env_key(strategy_tag: Optional[str]) -> Optional[str]:
    raw = str(strategy_tag or "").strip()
    if not raw:
        return None
    key = re.sub(r"[^0-9A-Za-z]+", "_", raw).upper().strip("_")
    return key or None


def _strategy_env_key_signature(strategy_key: str) -> str:
    """Normalize a strategy-key token for fuzzy env-var matching."""
    return re.sub(r"[^0-9A-Za-z]+", "", strategy_key).upper()


def _strategy_env_key_candidates(
    env_name: str,
    strategy_tag: Optional[str],
) -> tuple[str, ...]:
    raw = str(strategy_tag or "").strip()
    if not raw:
        return tuple()
    key = _strategy_env_key(raw)
    if not key:
        return tuple()

    seen: set[str] = set()
    candidates: list[str] = []

    def _add(candidate: Optional[str]) -> None:
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    _add(key)
    base_key = _strategy_env_key(_base_strategy_tag(raw))
    _add(base_key)

    if len(key) >= 8:
        prefix = f"{env_name}_STRATEGY_"
        key_signature = _strategy_env_key_signature(key)
        extended: list[str] = []
        for env_key in os.environ.keys():
            if not env_key.startswith(prefix):
                continue
            suffix = env_key[len(prefix) :].strip()
            if not suffix:
                continue
            suffix_key = _strategy_env_key(suffix)
            if not suffix_key:
                continue
            suffix_signature = _strategy_env_key_signature(suffix_key)
            if not suffix_signature or not key_signature:
                continue
            if not (
                suffix_signature.startswith(key_signature)
                or key_signature.startswith(suffix_signature)
            ):
                continue
            extended.append(suffix_key)
        for suffix_key in sorted(extended, key=len, reverse=True):
            _add(suffix_key)

    return tuple(candidates)


def _strategy_env_float(name: str, strategy_tag: Optional[str]) -> Optional[float]:
    for key in _strategy_env_key_candidates(name, strategy_tag):
        raw = os.getenv(f"{name}_STRATEGY_{key}")
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return None


def _strategy_env_int(name: str, strategy_tag: Optional[str]) -> Optional[int]:
    for key in _strategy_env_key_candidates(name, strategy_tag):
        raw = os.getenv(f"{name}_STRATEGY_{key}")
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def _entry_hard_stop_pips(pocket: Optional[str], *, strategy_tag: Optional[str] = None) -> float:
    """Return the entry hard SL distance in pips (0 = disabled)."""

    strategy_key = _strategy_env_key(strategy_tag)
    if strategy_key:
        raw = os.getenv(f"ORDER_ENTRY_HARD_STOP_PIPS_STRATEGY_{strategy_key}")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
    if not pocket:
        return _ENTRY_HARD_STOP_PIPS_DEFAULT
    pocket_key = str(pocket).strip().upper()
    if not pocket_key:
        return _ENTRY_HARD_STOP_PIPS_DEFAULT
    return max(
        0.0,
        _env_float(
            f"ORDER_ENTRY_HARD_STOP_PIPS_{pocket_key}",
            _ENTRY_HARD_STOP_PIPS_DEFAULT,
        ),
    )


def _entry_max_sl_pips(pocket: Optional[str], *, strategy_tag: Optional[str] = None) -> float:
    """Return the maximum entry SL distance in pips (0 = disabled)."""

    strategy_key = _strategy_env_key(strategy_tag)
    if strategy_key:
        raw = os.getenv(f"ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_{strategy_key}")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
    if not pocket:
        return _ENTRY_MAX_SL_PIPS_DEFAULT
    pocket_key = str(pocket).strip().upper()
    if not pocket_key:
        return _ENTRY_MAX_SL_PIPS_DEFAULT
    return max(
        0.0,
        _env_float(
            f"ORDER_ENTRY_MAX_SL_PIPS_{pocket_key}",
            _ENTRY_MAX_SL_PIPS_DEFAULT,
        ),
    )


def _entry_loss_cap_jpy(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
    nav_hint: Optional[float] = None,
) -> float:
    """Return per-trade loss cap in JPY (0 = disabled)."""

    if isinstance(entry_thesis, dict):
        thesis_cap = _as_float(entry_thesis.get("loss_cap_jpy"))
        if thesis_cap is not None and thesis_cap > 0.0:
            return float(thesis_cap)
    strategy_static = _strategy_env_float("ORDER_ENTRY_LOSS_CAP_JPY", strategy_tag)
    if strategy_static is not None:
        return max(0.0, float(strategy_static))
    static_cap = max(
        0.0,
        _pocket_env_float("ORDER_ENTRY_LOSS_CAP_JPY", pocket, _ENTRY_LOSS_CAP_JPY_DEFAULT),
    )
    if static_cap > 0.0:
        return static_cap

    thesis_nav_bps = None
    thesis_nav_min = None
    thesis_nav_max = None
    if isinstance(entry_thesis, dict):
        thesis_nav_bps = _as_float(entry_thesis.get("loss_cap_nav_bps"))
        thesis_nav_min = _as_float(entry_thesis.get("loss_cap_nav_min_jpy"))
        thesis_nav_max = _as_float(entry_thesis.get("loss_cap_nav_max_jpy"))

    nav_bps = thesis_nav_bps
    if nav_bps is None:
        nav_bps = _strategy_env_float("ORDER_ENTRY_LOSS_CAP_NAV_BPS", strategy_tag)
    if nav_bps is None:
        nav_bps = _pocket_env_float(
            "ORDER_ENTRY_LOSS_CAP_NAV_BPS",
            pocket,
            _ENTRY_LOSS_CAP_NAV_BPS_DEFAULT,
        )
    nav_bps = max(0.0, float(nav_bps or 0.0))
    if nav_bps <= 0.0:
        return 0.0

    nav_min = thesis_nav_min
    if nav_min is None:
        nav_min = _strategy_env_float("ORDER_ENTRY_LOSS_CAP_NAV_MIN_JPY", strategy_tag)
    if nav_min is None:
        nav_min = _pocket_env_float(
            "ORDER_ENTRY_LOSS_CAP_NAV_MIN_JPY",
            pocket,
            _ENTRY_LOSS_CAP_NAV_MIN_JPY_DEFAULT,
        )
    nav_min = max(0.0, float(nav_min or 0.0))

    nav_max = thesis_nav_max
    if nav_max is None:
        nav_max = _strategy_env_float("ORDER_ENTRY_LOSS_CAP_NAV_MAX_JPY", strategy_tag)
    if nav_max is None:
        nav_max = _pocket_env_float(
            "ORDER_ENTRY_LOSS_CAP_NAV_MAX_JPY",
            pocket,
            _ENTRY_LOSS_CAP_NAV_MAX_JPY_DEFAULT,
        )
    nav_max = max(0.0, float(nav_max or 0.0))
    if nav_max > 0.0 and nav_min > nav_max:
        nav_max = nav_min

    nav_value = _as_float(nav_hint)
    if nav_value is None or nav_value <= 0.0:
        try:
            snap = get_account_snapshot(cache_ttl_sec=_ENTRY_LOSS_CAP_NAV_SNAPSHOT_TTL_SEC)
            nav_value = float(snap.nav or snap.balance or 0.0)
        except Exception:
            nav_value = 0.0
    if nav_value <= 0.0:
        return 0.0

    cap = float(nav_value) * (nav_bps / 10_000.0)
    if nav_min > 0.0:
        cap = max(cap, nav_min)
    if nav_max > 0.0:
        cap = min(cap, nav_max)
    return max(0.0, cap)


def _entry_loss_cap_buffer_pips(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
) -> float:
    """Return pips buffer for loss-cap sizing (spread/slippage allowance)."""

    if isinstance(entry_thesis, dict):
        thesis_buf = _as_float(entry_thesis.get("loss_cap_buffer_pips"))
        if thesis_buf is not None and thesis_buf >= 0.0:
            return float(thesis_buf)
    strategy_key = _strategy_env_key(strategy_tag)
    if strategy_key:
        raw = os.getenv(f"ORDER_ENTRY_LOSS_CAP_BUFFER_PIPS_STRATEGY_{strategy_key}")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
    return max(
        0.0,
        _pocket_env_float(
            "ORDER_ENTRY_LOSS_CAP_BUFFER_PIPS",
            pocket,
            _ENTRY_LOSS_CAP_BUFFER_PIPS_DEFAULT,
        ),
    )


def _loss_cap_units_from_sl(*, loss_cap_jpy: float, sl_pips: float) -> int:
    if loss_cap_jpy <= 0.0 or sl_pips <= 0.0:
        return 0
    try:
        return max(0, int(math.floor(loss_cap_jpy / (sl_pips * _JPY_PER_PIP_PER_UNIT_USDJPY))))
    except Exception:
        return 0


def _order_spread_block_pips(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
) -> float:
    if isinstance(entry_thesis, dict):
        thesis_val = _as_float(entry_thesis.get("spread_block_pips"))
        if thesis_val is None:
            thesis_val = _as_float(entry_thesis.get("max_spread_pips"))
        if thesis_val is not None and thesis_val > 0.0:
            return float(thesis_val)
    strategy_key = _strategy_env_key(strategy_tag)
    if strategy_key:
        raw = os.getenv(f"ORDER_SPREAD_BLOCK_PIPS_STRATEGY_{strategy_key}")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
    if pocket:
        pocket_key = str(pocket).strip().upper()
        if pocket_key:
            raw = os.getenv(f"ORDER_SPREAD_BLOCK_PIPS_{pocket_key}")
            if raw is not None:
                try:
                    return max(0.0, float(raw))
                except Exception:
                    pass
    return max(0.0, float(_ORDER_SPREAD_BLOCK_PIPS))


def _is_isolated_spread_spike(
    *,
    spread_pips: float,
    threshold_pips: float,
) -> tuple[bool, dict[str, float | int | bool | str | None]]:
    if not _ORDER_SPREAD_SPIKE_TOLERANCE_ENABLED:
        return False, {"enabled": False}
    if spread_pips <= 0.0 or threshold_pips <= 0.0 or tick_window is None:
        return False, {"enabled": True, "reason": "invalid_input"}
    if (
        _ORDER_SPREAD_SPIKE_HARD_MAX_PIPS > 0.0
        and spread_pips > _ORDER_SPREAD_SPIKE_HARD_MAX_PIPS
    ):
        return False, {
            "enabled": True,
            "reason": "hard_max_exceeded",
            "hard_max_pips": _ORDER_SPREAD_SPIKE_HARD_MAX_PIPS,
            "spread_pips": spread_pips,
        }

    try:
        rows = tick_window.recent_ticks(
            seconds=_ORDER_SPREAD_SPIKE_WINDOW_SEC, limit=_ORDER_SPREAD_SPIKE_MIN_TICKS * 6
        )
    except Exception as exc:
        return False, {"enabled": True, "reason": "tick_window_error", "error": str(exc)}

    spreads: list[float] = []
    now_sec = time.time()
    for row in rows:
        epoch = _as_float(row.get("epoch"))
        if (
            _ORDER_SPREAD_SPIKE_TICK_MAX_AGE_SEC > 0.0
            and epoch is not None
            and now_sec - epoch > _ORDER_SPREAD_SPIKE_TICK_MAX_AGE_SEC
        ):
            continue
        bid = _as_float(row.get("bid"))
        ask = _as_float(row.get("ask"))
        if bid is None or ask is None:
            continue
        spread = max(0.0, ask - bid) / 0.01
        spreads.append(spread)

    if len(spreads) < _ORDER_SPREAD_SPIKE_MIN_TICKS:
        return False, {
            "enabled": True,
            "reason": "insufficient_ticks",
            "sample_count": len(spreads),
        }

    window_median = _percentile(spreads, 50.0)
    if _ORDER_SPREAD_SPIKE_MEDIAN_PIPS > 0.0:
        median_cap = _ORDER_SPREAD_SPIKE_MEDIAN_PIPS
    else:
        median_cap = threshold_pips * _ORDER_SPREAD_SPIKE_MEDIAN_RATIO

    high_count = sum(1 for val in spreads if val >= threshold_pips)
    if high_count <= _ORDER_SPREAD_SPIKE_ALLOW_MAX_COUNT and window_median <= median_cap:
        return True, {
            "enabled": True,
            "reason": "isolated",
            "sample_count": len(spreads),
            "high_count": high_count,
            "window_median": window_median,
            "median_cap": median_cap,
        }

    return False, {
        "enabled": True,
        "reason": "sustained_wide",
        "sample_count": len(spreads),
        "high_count": high_count,
        "window_median": window_median,
        "median_cap": median_cap,
    }


def _augment_entry_thesis_policy_generation(
    entry_thesis: Optional[dict],
    *,
    reduce_only: bool,
) -> Optional[dict]:
    if reduce_only or not isinstance(entry_thesis, dict):
        return entry_thesis
    if not _ENTRY_POLICY_GENERATION:
        return entry_thesis
    merged = dict(entry_thesis)
    merged.setdefault("policy_generation", _ENTRY_POLICY_GENERATION)
    merged.setdefault("policy_scope", "new_entries_only")
    return merged


def _sl_price_from_pips(entry_price: float, units: int, sl_pips: float) -> Optional[float]:
    if entry_price <= 0 or units == 0 or sl_pips <= 0:
        return None
    offset = round(sl_pips * 0.01, 3)
    if units > 0:
        return round(entry_price - offset, 3)
    return round(entry_price + offset, 3)


def min_units_for_pocket(pocket: Optional[str]) -> int:
    if not pocket:
        return _DEFAULT_MIN_UNITS
    return int(_MIN_UNITS_BY_POCKET.get(pocket, _DEFAULT_MIN_UNITS))



def min_units_for_strategy(
    strategy_tag: Optional[str],
    pocket: Optional[str] = None,
) -> int:
    strategy_min_units = _strategy_env_int("ORDER_MIN_UNITS", strategy_tag)
    if strategy_min_units is not None:
        return strategy_min_units
    base_tag = _base_strategy_tag(strategy_tag)
    strategy_min_units = _strategy_env_int("ORDER_MIN_UNITS", base_tag)
    if strategy_min_units is not None:
        return strategy_min_units
    return min_units_for_pocket(pocket)






def _policy_gate_allows_entry(
    pocket: str,
    side_label: str,
    strategy_tag: Optional[str],
    *,
    reduce_only: bool,
) -> tuple[bool, Optional[str], dict[str, object]]:
    if reduce_only or not _POLICY_GATE_ENABLED:
        return True, None, {}
    policy = policy_bus.latest()
    if policy is None:
        return True, None, {}
    pockets = policy.pockets if isinstance(policy.pockets, dict) else {}
    pocket_policy = pockets.get(pocket)
    if not isinstance(pocket_policy, dict):
        return True, None, {}

    enabled = pocket_policy.get("enabled")
    if enabled is not None and not _coerce_bool(enabled, True):
        return False, "policy_disabled", {"enabled": enabled}

    entry_gates = pocket_policy.get("entry_gates")
    if isinstance(entry_gates, dict):
        allow_new = entry_gates.get("allow_new")
        if allow_new is not None and not _coerce_bool(allow_new, True):
            return False, "policy_allow_new_false", {"allow_new": allow_new}

    bias = str(pocket_policy.get("bias") or "").strip().lower()
    if bias == "long" and side_label != "buy":
        return False, "policy_bias_long", {"bias": bias}
    if bias == "short" and side_label != "sell":
        return False, "policy_bias_short", {"bias": bias}

    strategies_raw = pocket_policy.get("strategies")
    strategy_list: list[str] = []
    if isinstance(strategies_raw, str):
        strategy_list = [strategies_raw]
    elif isinstance(strategies_raw, (list, tuple, set)):
        strategy_list = [str(item) for item in strategies_raw if item]
    if strategy_list:
        if not strategy_tag:
            return False, "policy_missing_strategy", {}
        tag_lower = str(strategy_tag).strip().lower()
        base = tag_lower.split("-", 1)[0]
        allow = {str(item).strip().lower() for item in strategy_list if str(item).strip()}
        if tag_lower not in allow and base not in allow:
            return False, "policy_strategy_block", {"allow": sorted(allow)}

    return True, None, {}




def _min_rr_for(pocket: Optional[str]) -> float:
    if not _MIN_RR_ENABLED or not pocket:
        return 0.0
    try:
        value = float(_MIN_RR_BY_POCKET.get(str(pocket).lower(), 0.0))
    except Exception:
        return 0.0
    if value <= 0.0:
        return 0.0
    return min(value, 5.0)


def _tp_cap_env_float(name: str, pocket: Optional[str], default: float) -> float:
    pocket_upper = str(pocket).upper() if pocket else ""
    if pocket_upper:
        raw = os.getenv(f"{name}_{pocket_upper}")
        if raw is not None:
            try:
                return float(raw)
            except Exception:
                pass
    return _env_float(name, default)


def _tp_cap_env_str(name: str, pocket: Optional[str], default: str) -> str:
    pocket_upper = str(pocket).upper() if pocket else ""
    if pocket_upper:
        raw = os.getenv(f"{name}_{pocket_upper}")
        if raw:
            return raw.strip()
    raw = os.getenv(name)
    if raw:
        return raw.strip()
    return default


def _tp_cap_factors(
    pocket: Optional[str],
    entry_thesis: Optional[dict],
) -> tuple[str, dict]:
    default_tf = _TP_CAP_TF_DEFAULTS.get(str(pocket).lower() if pocket else "", "M1")
    tf = _tp_cap_env_str("ORDER_TP_CAP_TF", pocket, default_tf).upper()
    fac: dict = {}
    try:
        fac = (all_factors().get(tf) or {}) if tf else {}
    except Exception:
        fac = {}
    if fac:
        return tf, fac
    if isinstance(entry_thesis, dict):
        thesis_factors = entry_thesis.get("factors")
        if isinstance(thesis_factors, dict):
            fallback = thesis_factors.get(tf.lower()) or thesis_factors.get(tf)
            if isinstance(fallback, dict):
                return tf, fallback
    return tf, {}


def _tp_cap_normalize(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return None
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _tp_cap_dynamic_multiplier(
    pocket: Optional[str],
    entry_thesis: Optional[dict],
) -> tuple[float, Optional[dict]]:
    if not _TP_CAP_DYNAMIC_ENABLED:
        return 1.0, None
    tf, fac = _tp_cap_factors(pocket, entry_thesis)
    if not fac:
        return 1.0, {"tf": tf, "reason": "no_factors"}
    adx = _as_float(fac.get("adx"))
    atr_pips = _as_float(fac.get("atr_pips"))
    if atr_pips is None:
        atr = _as_float(fac.get("atr"))
        if atr is not None:
            atr_pips = atr * 100.0
    adx_low = _tp_cap_env_float("ORDER_TP_CAP_ADX_LOW", pocket, 18.0)
    adx_high = _tp_cap_env_float("ORDER_TP_CAP_ADX_HIGH", pocket, 35.0)
    atr_low = _tp_cap_env_float("ORDER_TP_CAP_ATR_LOW", pocket, 6.0)
    atr_high = _tp_cap_env_float("ORDER_TP_CAP_ATR_HIGH", pocket, 14.0)
    trend_score = _tp_cap_normalize(adx, adx_low, adx_high)
    vol_score = _tp_cap_normalize(atr_pips, atr_low, atr_high)
    weight_trend = _tp_cap_env_float("ORDER_TP_CAP_WEIGHT_TREND", pocket, 0.6)
    weight_vol = _tp_cap_env_float("ORDER_TP_CAP_WEIGHT_VOL", pocket, 0.4)
    score_sum = 0.0
    weight_sum = 0.0
    if trend_score is not None:
        score_sum += trend_score * weight_trend
        weight_sum += weight_trend
    if vol_score is not None:
        score_sum += vol_score * weight_vol
        weight_sum += weight_vol
    if weight_sum <= 0.0:
        return 1.0, {
            "tf": tf,
            "adx": adx,
            "atr_pips": atr_pips,
            "reason": "no_scores",
        }
    score = score_sum / weight_sum
    min_mult = _tp_cap_env_float("ORDER_TP_CAP_MULT_MIN", pocket, 0.8)
    max_mult = _tp_cap_env_float("ORDER_TP_CAP_MULT_MAX", pocket, 1.6)
    if max_mult < min_mult:
        min_mult, max_mult = max_mult, min_mult
    mult = min_mult + (max_mult - min_mult) * score
    meta = {
        "tf": tf,
        "adx": adx,
        "atr_pips": atr_pips,
        "trend_score": trend_score,
        "vol_score": vol_score,
        "score": score,
        "min_mult": min_mult,
        "max_mult": max_mult,
        "weight_trend": weight_trend,
        "weight_vol": weight_vol,
    }
    return mult, meta


def _tp_cap_for(
    pocket: Optional[str],
    entry_thesis: Optional[dict] = None,
) -> tuple[float, Optional[dict]]:
    if not _TP_CAP_ENABLED or not pocket:
        return 0.0, None
    try:
        base = float(_TP_CAP_BY_POCKET.get(str(pocket).lower(), 0.0))
    except Exception:
        return 0.0, None
    if base <= 0.0:
        return 0.0, None
    mult, meta = _tp_cap_dynamic_multiplier(pocket, entry_thesis)
    if mult <= 0.0:
        mult = 0.1
    cap = min(base * mult, 200.0)
    if meta:
        meta = dict(meta)
        meta.update({"cap_base": base, "cap_mult": mult, "cap_final": cap})
    return cap, meta


_EXIT_NO_NEGATIVE_CLOSE = os.getenv("EXIT_NO_NEGATIVE_CLOSE", "1").strip().lower() not in {"", "0", "false", "no"}
_EXIT_ALLOW_NEGATIVE_BY_WORKER = os.getenv("EXIT_ALLOW_NEGATIVE_BY_WORKER", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
_EXIT_ALLOW_NEGATIVE_REASONS = {
    token.strip().lower()
    for token in os.getenv(
        "EXIT_ALLOW_NEGATIVE_REASONS",
        "hard_stop,tech_hard_stop,drawdown,max_drawdown,health_exit,hazard_exit,"
        "margin_health,free_margin_low,margin_usage_high",
    ).split(",")
    if token.strip()
}
_EXIT_ALLOW_NEGATIVE_NEAR_BE_PIPS = max(0.0, _env_float("EXIT_ALLOW_NEGATIVE_NEAR_BE_PIPS", 0.6))
_EXIT_ALLOW_NEGATIVE_NEAR_BE_REASONS = {
    token.strip().lower()
    for token in os.getenv(
        "EXIT_ALLOW_NEGATIVE_NEAR_BE_REASONS",
        "lock_floor,trail_lock,profit_lock,trail_backoff,lock_trail,near_be",
    ).split(",")
    if token.strip()
}
_EXIT_HOLD_UNTIL_PROFIT_ENABLED = _env_bool("EXIT_HOLD_UNTIL_PROFIT_ENABLED", True)
_EXIT_HOLD_UNTIL_PROFIT_CLIENT_IDS = _env_csv_set("EXIT_HOLD_UNTIL_PROFIT_CLIENT_IDS", "")
_EXIT_HOLD_UNTIL_PROFIT_TRADE_IDS = _env_csv_set("EXIT_HOLD_UNTIL_PROFIT_TRADE_IDS", "")
_EXIT_HOLD_UNTIL_PROFIT_MIN_PIPS = max(0.0, _env_float("EXIT_HOLD_UNTIL_PROFIT_MIN_PIPS", 0.0))
_EXIT_HOLD_UNTIL_PROFIT_STRICT = _env_bool("EXIT_HOLD_UNTIL_PROFIT_STRICT", False)
_EXIT_HOLD_UNTIL_PROFIT_ALL = _env_bool("EXIT_HOLD_UNTIL_PROFIT_ALL", False)
_EXIT_FORCE_ALLOW_REASONS = {
    token.strip().lower()
    for token in os.getenv(
        "EXIT_FORCE_ALLOW_REASONS",
        "drawdown,max_drawdown,health_exit,hazard_exit,margin_health,free_margin_low,margin_usage_high",
    ).split(",")
    if token.strip()
}
_EXIT_EMERGENCY_ALLOW_NEGATIVE = os.getenv("EXIT_EMERGENCY_ALLOW_NEGATIVE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}


def _allow_stop_loss_on_fill(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
) -> bool:
    """Return whether broker stopLossOnFill can be attached for entries.

    V1 behavior: only ORDER_FIXED_SL_MODE decides attachment:
    - 1: always attach
    - 0: never attach
    - unset: default OFF
    """
    tag = str(strategy_tag or "").strip().lower()
    strategy_override = _strategy_env_int(
        "ORDER_ALLOW_STOP_LOSS_ON_FILL",
        strategy_tag,
    )
    if strategy_override is not None:
        return strategy_override > 0

    def _ping_5s_variant_default_allow(tag_text: str) -> Optional[bool]:
        if tag_text.startswith("scalp_ping_5s_b"):
            return _env_bool("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B", True)
        if tag_text.startswith("scalp_ping_5s_c"):
            return _env_bool("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_C", False)
        if tag_text.startswith("scalp_ping_5s_d"):
            return _env_bool("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_D", False)
        return None

    mode = fixed_sl_mode()
    if mode is not None:
        if bool(mode):
            return True
        # Keep strategy-specific 5s variants configurable even when global
        # fixed-mode is OFF unless explicitly disabled per strategy.
        family_override = _ping_5s_variant_default_allow(tag)
        if family_override is not None:
            return family_override
        return False
    family_override = _ping_5s_variant_default_allow(tag)
    if family_override is not None:
        return family_override
    return False


def _disable_hard_stop_by_strategy(
    strategy_tag: Optional[str],
    pocket: Optional[str],
    entry_thesis: Optional[dict],
) -> bool:
    base_tag = (strategy_tag or "").strip().lower()
    if not base_tag and isinstance(entry_thesis, dict):
        base_tag = str(
            entry_thesis.get("strategy_tag") or entry_thesis.get("strategy") or ""
        ).strip().lower()
    if not base_tag:
        base_tag = (_strategy_tag_from_thesis(entry_thesis) or "").strip().lower()

    if isinstance(entry_thesis, dict):
        if "disable_entry_hard_stop" in entry_thesis:
            return _coerce_bool(entry_thesis.get("disable_entry_hard_stop"), False)
    # 5s variants: keep per-variant defaults while allowing explicit overrides.
    if base_tag.startswith("scalp_ping_5s_b"):
        return _env_bool("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_B", False)
    if base_tag.startswith("scalp_ping_5s_c"):
        return _env_bool("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_C", True)
    if base_tag.startswith("scalp_ping_5s_d"):
        return _env_bool("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_D", True)
    # Legacy 5s ping family keeps previous default (disabled) unless overridden.
    if base_tag.startswith("scalp_ping_5"):
        return _env_bool("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5", True)
    if base_tag in {
        "scalp_ping_5",
        "scalp_ping_5s",
    }:
        if (pocket or "").strip().lower() == "scalp_fast":
            return True
    return base_tag in _env_csv_set(
        "ORDER_DISABLE_ENTRY_HARD_STOP_TAGS",
        "scalp_ping_5,scalp_ping_5s",
    )


_PROFIT_GUARD_BYPASS_RANGE = os.getenv("PROFIT_GUARD_BYPASS_RANGE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_EXIT_EMERGENCY_HEALTH_BUFFER = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_HEALTH_BUFFER", "0.07"))
)
_EXIT_EMERGENCY_MARGIN_USAGE_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_MARGIN_USAGE_RATIO", "0.92"))
)
_EXIT_EMERGENCY_FREE_MARGIN_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_FREE_MARGIN_RATIO", "0.12"))
)
_EXIT_EMERGENCY_UNREALIZED_DD_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_UNREALIZED_DD_RATIO", "0.06"))
)
_EXIT_EMERGENCY_CACHE_TTL_SEC = max(
    0.5, float(os.getenv("EXIT_EMERGENCY_CACHE_TTL_SEC", "2.0"))
)
_LAST_EMERGENCY_LOG_TS: float = 0.0

_DYNAMIC_SL_ENABLE = os.getenv("ORDER_DYNAMIC_SL_ENABLE", "true").lower() in {
    "1",
    "true",
    "yes",
}
_DYNAMIC_SL_POCKETS = {
    token.strip().lower()
    for token in os.getenv("ORDER_DYNAMIC_SL_POCKETS", "micro,macro").split(",")
    if token.strip()
}
_DYNAMIC_SL_RATIO = float(os.getenv("ORDER_DYNAMIC_SL_RATIO", "1.2"))
_DYNAMIC_SL_MAX_PIPS = float(os.getenv("ORDER_DYNAMIC_SL_MAX_PIPS", "8.0"))
_ENTRY_QUALITY_GATE_ENABLED = _env_bool("ORDER_ENTRY_QUALITY_GATE_ENABLED", True)
_ORDER_MANAGER_BRAIN_GATE_ENABLED = _env_bool("ORDER_MANAGER_BRAIN_GATE_ENABLED", False)
_ORDER_MANAGER_FORECAST_GATE_ENABLED = _env_bool("ORDER_MANAGER_FORECAST_GATE_ENABLED", False)
_FORECAST_SERVICE_ENABLED = _env_bool("FORECAST_SERVICE_ENABLED", False)
_FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL", "http://127.0.0.1:8302").strip()
_FORECAST_SERVICE_TIMEOUT = max(0.5, float(_env_float("FORECAST_SERVICE_TIMEOUT", 5.0)))
_FORECAST_SERVICE_FALLBACK_LOCAL = _env_bool("FORECAST_SERVICE_FALLBACK_LOCAL", True)
_ORDER_MANAGER_PATTERN_GATE_ENABLED = _env_bool("ORDER_MANAGER_PATTERN_GATE_ENABLED", False)
_ORDER_MANAGER_PRESERVE_STRATEGY_INTENT = _env_bool(
    "ORDER_MANAGER_PRESERVE_STRATEGY_INTENT", True
)
_ORDER_MANAGER_PRESERVE_INTENT_PROBABILITY_MIN_SCALE = _env_float(
    "ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE", 0.35
)
_ORDER_MANAGER_PRESERVE_INTENT_PROBABILITY_REJECT_UNDER = _env_float(
    "ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER", 0.0
)
_ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE = _env_float(
    "ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE", 1.25
)
_ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY = _env_float(
    "ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY", 0.80
)


def _order_manager_preserve_intent_min_scale(
    strategy_tag: Optional[str],
) -> float:
    strategy_override = _strategy_env_float(
        "ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE",
        strategy_tag,
    )
    if strategy_override is not None:
        return max(0.0, min(1.0, float(strategy_override)))
    return max(
        0.0,
        min(1.0, float(_ORDER_MANAGER_PRESERVE_INTENT_PROBABILITY_MIN_SCALE)),
    )


def _order_manager_preserve_intent_max_scale(
    strategy_tag: Optional[str],
) -> float:
    strategy_override = _strategy_env_float(
        "ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE",
        strategy_tag,
    )
    if strategy_override is not None:
        return max(1.0, min(5.0, float(strategy_override)))
    return max(1.0, float(_ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE))


def _order_manager_preserve_intent_boost_probability(
    strategy_tag: Optional[str],
) -> float:
    strategy_override = _strategy_env_float(
        "ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY",
        strategy_tag,
    )
    if strategy_override is not None:
        return max(0.0, min(1.0, float(strategy_override)))
    return max(
        0.0,
        min(1.0, float(_ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY)),
    )


def _order_manager_preserve_intent_reject_under(
    strategy_tag: Optional[str],
) -> float:
    strategy_override = _strategy_env_float(
        "ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER",
        strategy_tag,
    )
    if strategy_override is not None:
        return max(0.0, min(1.0, float(strategy_override)))
    return max(
        0.0,
        min(1.0, float(_ORDER_MANAGER_PRESERVE_INTENT_PROBABILITY_REJECT_UNDER)),
    )


def _order_manager_coordination_rejection_dominance(
    strategy_tag: Optional[str],
) -> float:
    strategy_override = _strategy_env_float(
        "ORDER_INTENT_COORDINATION_REJECTION_DOMINANCE",
        strategy_tag,
    )
    if strategy_override is not None:
        return max(1.0, float(strategy_override))
    return _ORDER_INTENT_COORDINATION_REJECTION_DOMINANCE


_ORDER_INTENT_COORDINATION_ENABLED = _env_bool("ORDER_INTENT_COORDINATION_ENABLED", True)
_ORDER_INTENT_COORDINATION_WINDOW_SEC = max(
    0.25,
    float(os.getenv("ORDER_INTENT_COORDINATION_WINDOW_SEC", "2.0")),
)
_ORDER_INTENT_COORDINATION_REJECTION_DOMINANCE = max(
    1.0,
    float(os.getenv("ORDER_INTENT_COORDINATION_REJECTION_DOMINANCE", "1.12")),
)
_ORDER_INTENT_COORDINATION_MIN_SCALE = max(
    0.0,
    min(1.0, float(os.getenv("ORDER_INTENT_COORDINATION_MIN_SCALE", "0.2"))),
)
_ENTRY_QUALITY_POCKETS = _env_csv_set(
    "ORDER_ENTRY_QUALITY_POCKETS",
    "micro,macro,scalp,scalp_fast",
)
_ENTRY_QUALITY_BASE_MIN_CONF = {
    "macro": 54.0,
    "micro": 58.0,
    "scalp": 62.0,
    "scalp_fast": 65.0,
}
_ENTRY_QUALITY_SPREAD_SL_MAX_RATIO = {
    "macro": 0.26,
    "micro": 0.34,
    "scalp": 0.46,
    "scalp_fast": 0.52,
}
_ENTRY_QUALITY_SPREAD_TP_MAX_RATIO = {
    "macro": 0.14,
    "micro": 0.20,
    "scalp": 0.28,
    "scalp_fast": 0.34,
}
_ENTRY_QUALITY_SPREAD_TP_SOFTEN_ENABLED = (
    os.getenv("ORDER_ENTRY_QUALITY_SPREAD_TP_SOFTEN", "1").strip().lower()
    not in {"", "0", "false", "no"}
)
_ENTRY_QUALITY_SPREAD_TP_SOFTEN_MIN_SCALE = max(
    0.05,
    _env_float("ORDER_ENTRY_QUALITY_SPREAD_TP_SOFTEN_MIN_SCALE", 0.30),
)
_ENTRY_QUALITY_ATR_BANDS = {
    "macro": (9.0, 24.0),
    "micro": (1.2, 4.0),
    "scalp": (0.9, 3.0),
    "scalp_fast": (0.8, 2.6),
}
_ENTRY_QUALITY_VOL_BANDS = {
    "macro": (0.85, 1.7),
    "micro": (0.8, 1.6),
    "scalp": (0.75, 1.55),
    "scalp_fast": (0.75, 1.5),
}
_ENTRY_QUALITY_STRAT_ENABLED = _env_bool("ORDER_ENTRY_QUALITY_STRAT_ENABLED", True)
_ENTRY_QUALITY_STRAT_LOOKBACK_DAYS = max(
    1, _env_int("ORDER_ENTRY_QUALITY_STRAT_LOOKBACK_DAYS", 5)
)
_ENTRY_QUALITY_STRAT_MIN_TRADES = max(
    5, _env_int("ORDER_ENTRY_QUALITY_STRAT_MIN_TRADES", 12)
)
_ENTRY_QUALITY_STRAT_TTL_SEC = max(
    15.0, _env_float("ORDER_ENTRY_QUALITY_STRAT_TTL_SEC", 120.0)
)
_ENTRY_QUALITY_STRAT_DB_PATH = pathlib.Path(
    os.getenv("TRADES_DB_PATH", "logs/trades.db")
)
_ENTRY_QUALITY_STRAT_CACHE: dict[
    tuple[str, str], tuple[float, dict[str, float]]
] = {}

_LAST_PROTECTIONS: dict[str, dict[str, float | None]] = {}
_LAST_ROLLOVER_SL_STRIP_TS: dict[str, float] = {}
_JST = timezone(timedelta(hours=9))
_ROLLOVER_SL_STRIP_ENABLED = _env_bool("ORDER_ROLLOVER_SL_STRIP_ENABLED", False)
_ROLLOVER_SL_STRIP_JST_HOUR = max(
    0, min(23, _env_int("ORDER_ROLLOVER_SL_STRIP_JST_HOUR", 7))
)
_ROLLOVER_SL_STRIP_WINDOW_MIN = max(
    1, _env_int("ORDER_ROLLOVER_SL_STRIP_WINDOW_MIN", 90)
)
_ROLLOVER_SL_STRIP_REQUIRE_CARRYOVER = _env_bool(
    "ORDER_ROLLOVER_SL_STRIP_REQUIRE_CARRYOVER", True
)
_ROLLOVER_SL_STRIP_INCLUDE_MANUAL = _env_bool(
    "ORDER_ROLLOVER_SL_STRIP_INCLUDE_MANUAL", False
)
_ROLLOVER_SL_STRIP_COOLDOWN_SEC = max(
    10.0, _env_float("ORDER_ROLLOVER_SL_STRIP_COOLDOWN_SEC", 120.0)
)
_ROLLOVER_SL_STRIP_MAX_ACTIONS = max(
    1, _env_int("ORDER_ROLLOVER_SL_STRIP_MAX_ACTIONS", 6)
)
MACRO_BE_GRACE_SECONDS = 45
_MARGIN_REJECT_UNTIL: dict[str, float] = {}
_PROTECTION_MIN_BUFFER = max(0.0005, float(os.getenv("ORDER_PROTECTION_MIN_BUFFER", "0.003")))
_PROTECTION_MIN_SEPARATION = max(0.001, float(os.getenv("ORDER_PROTECTION_MIN_SEPARATION", "0.006")))
_PROTECTION_FALLBACK_PIPS = max(0.02, float(os.getenv("ORDER_PROTECTION_FALLBACK_PIPS", "0.12")))
_PROTECTION_RETRY_REASONS = {
    "STOP_LOSS_ON_FILL_LOSS",
    "STOP_LOSS_ON_FILL_INVALID",
    "STOP_LOSS_LOSS",
    "TAKE_PROFIT_ON_FILL_LOSS",
    # Observed from OANDA as a take-profit protection reject when price moved past TP.
    "LOSING_TAKE_PROFIT",
}
_PARTIAL_CLOSE_RETRY_CODES = {
    "CLOSE_TRADE_UNITS_EXCEED_TRADE_SIZE",
    "POSITION_TO_REDUCE_TOO_SMALL",
}
_ORDER_SPREAD_BLOCK_PIPS = float(os.getenv("ORDER_SPREAD_BLOCK_PIPS", "1.6"))
_ORDER_SPREAD_SPIKE_TOLERANCE_ENABLED = _env_bool(
    "ORDER_SPREAD_SPIKE_TOLERANCE_ENABLED", True
)
_ORDER_SPREAD_SPIKE_WINDOW_SEC = max(
    0.5, _env_float("ORDER_SPREAD_SPIKE_WINDOW_SEC", 1.5)
)
_ORDER_SPREAD_SPIKE_MIN_TICKS = max(2, _env_int("ORDER_SPREAD_SPIKE_MIN_TICKS", 3))
_ORDER_SPREAD_SPIKE_ALLOW_MAX_COUNT = max(
    1, _env_int("ORDER_SPREAD_SPIKE_ALLOW_MAX_COUNT", 2)
)
_ORDER_SPREAD_SPIKE_MEDIAN_PIPS = max(
    0.0, _env_float("ORDER_SPREAD_SPIKE_MEDIAN_PIPS", 0.0)
)
_ORDER_SPREAD_SPIKE_MEDIAN_RATIO = max(
    0.4, min(3.0, _env_float("ORDER_SPREAD_SPIKE_MEDIAN_RATIO", 1.25))
)
_ORDER_SPREAD_SPIKE_HARD_MAX_PIPS = max(
    0.0, _env_float("ORDER_SPREAD_SPIKE_HARD_MAX_PIPS", 0.0)
)
_ORDER_SPREAD_SPIKE_TICK_MAX_AGE_SEC = max(
    0.0, _env_float("ORDER_SPREAD_SPIKE_TICK_MAX_AGE_SEC", 5.0)
)
# Minimum reward/risk ratio for new entries.
_MIN_RR_ENABLED = os.getenv("ORDER_MIN_RR_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_MIN_RR_BY_POCKET = {
    "macro": float(os.getenv("ORDER_MIN_RR_MACRO", "1.5")),
    "micro": float(os.getenv("ORDER_MIN_RR_MICRO", "1.35")),
    "scalp": float(os.getenv("ORDER_MIN_RR_SCALP", "1.2")),
    "scalp_fast": float(os.getenv("ORDER_MIN_RR_SCALP_FAST", "1.15")),
    "manual": float(os.getenv("ORDER_MIN_RR_MANUAL", "1.0")),
}
_MIN_RR_ADJUST_MODE = os.getenv("ORDER_MIN_RR_ADJUST_MODE", "tp").strip().lower()
_MIN_RR_ADJUST_MODES = {"tp", "sl", "sl_first", "both"}


def _min_rr_adjust_mode_for(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
) -> str:
    strategy_key = _strategy_env_key(strategy_tag)
    if strategy_key:
        raw = os.getenv(f"ORDER_MIN_RR_ADJUST_MODE_STRATEGY_{strategy_key}")
        if raw is not None:
            mode = str(raw).strip().lower()
            if mode in _MIN_RR_ADJUST_MODES:
                return mode
    pocket_key = str(pocket or "").strip().upper()
    if pocket_key:
        raw = os.getenv(f"ORDER_MIN_RR_ADJUST_MODE_{pocket_key}")
        if raw is not None:
            mode = str(raw).strip().lower()
            if mode in _MIN_RR_ADJUST_MODES:
                return mode
    if _MIN_RR_ADJUST_MODE in _MIN_RR_ADJUST_MODES:
        return _MIN_RR_ADJUST_MODE
    return "tp"


def _protection_fallback_gap_price(
    pocket: Optional[str],
    *,
    strategy_tag: Optional[str] = None,
) -> float:
    strategy_override = _strategy_env_float("ORDER_PROTECTION_FALLBACK_PIPS", strategy_tag)
    if strategy_override is not None:
        return max(0.0005, float(strategy_override))
    pocket_val = _pocket_env_float(
        "ORDER_PROTECTION_FALLBACK_PIPS",
        pocket,
        _PROTECTION_FALLBACK_PIPS,
    )
    return max(0.0005, float(pocket_val))

# Cap extreme TP distances (pips) to keep targets realistic.
_TP_CAP_ENABLED = os.getenv("ORDER_TP_CAP_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_TP_CAP_BY_POCKET = {
    "macro": float(os.getenv("ORDER_TP_CAP_MACRO", "24.0")),
    "micro": float(os.getenv("ORDER_TP_CAP_MICRO", "12.0")),
    "scalp": float(os.getenv("ORDER_TP_CAP_SCALP", "6.0")),
    "scalp_fast": float(os.getenv("ORDER_TP_CAP_SCALP_FAST", "3.0")),
    "manual": float(os.getenv("ORDER_TP_CAP_MANUAL", "0.0")),
}
# Momentum-aware TP cap scaling (opt-in).
_TP_CAP_DYNAMIC_ENABLED = os.getenv("ORDER_TP_CAP_DYNAMIC_ENABLED", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_TP_CAP_TF_DEFAULTS = {
    "macro": "H1",
    "micro": "M5",
    "scalp": "M1",
    "scalp_fast": "M1",
    "manual": "H1",
}
# ワーカーのオーダーをメインの関所に転送するフラグ（reduce_only は除外）
_FORWARD_TO_SIGNAL_GATE = (
    os.getenv("ORDER_FORWARD_TO_SIGNAL_GATE", "0").strip().lower()
    not in {"", "0", "false", "no"}
)
_WORKER_ONLY_MODE = os.getenv("WORKER_ONLY_MODE", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
if _WORKER_ONLY_MODE:
    _FORWARD_TO_SIGNAL_GATE = False
# コメントを付けるとリジェクトが発生する場合に強制オフにするトグル（デフォルトで無効化）
_DISABLE_CLIENT_COMMENT = os.getenv("ORDER_DISABLE_CLIENT_COMMENT", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}

# Max units per new entry order (reduce_only orders are exempt)
try:
    _MAX_ORDER_UNITS = int(os.getenv("MAX_ORDER_UNITS", "40000"))
except Exception:
    _MAX_ORDER_UNITS = 40000
# Hard safety cap even after dynamic boosts
try:
    _MAX_ORDER_UNITS_HARD = int(os.getenv("MAX_ORDER_UNITS_HARD", "60000"))
except Exception:
    _MAX_ORDER_UNITS_HARD = 60000

# 最低発注単位（AGENT.me 6.1 に準拠）。
# リスク計算・ステージ係数適用後の“最終 units”に対して適用する最終ゲート。
# reduce_only（決済）では適用しない。
_MIN_ORDER_UNITS = 500

# Directional exposure cap (dynamic; scales units instead of rejecting)
_DIR_CAP_ENABLE = os.getenv("DIR_CAP_ENABLE", "1").strip().lower() not in {"", "0", "false", "no"}
_DIR_CAP_RATIO = float(os.getenv("DIR_CAP_RATIO", "0.70"))
_DIR_CAP_WARN_RATIO = float(os.getenv("DIR_CAP_WARN_RATIO", "0.98"))
# Floor multiplier to avoid crushing frequency when shrinking; 0.0 to disable
_DIR_CAP_MIN_FRACTION = float(os.getenv("DIR_CAP_MIN_FRACTION", "0.15"))
_DIR_CAP_CACHE: Optional[PositionManager] = None
_BLOCK_MANUAL_NETTING = _env_bool("BLOCK_MANUAL_NETTING", True)
_DIR_CAP_ADVERSE_ENABLE = _env_bool("DIR_CAP_ADVERSE_ENABLE", True)
_DIR_CAP_ADVERSE_LOOKBACK_MIN = max(5, min(120, _env_int("DIR_CAP_ADVERSE_LOOKBACK_MIN", 20)))
_DIR_CAP_ADVERSE_START_PIPS = max(0.0, _env_float("DIR_CAP_ADVERSE_START_PIPS", 6.0))
_DIR_CAP_ADVERSE_FULL_PIPS = max(
    _DIR_CAP_ADVERSE_START_PIPS + 1e-6,
    _env_float("DIR_CAP_ADVERSE_FULL_PIPS", 20.0),
)
_DIR_CAP_ADVERSE_MIN_SCALE = max(0.05, min(1.0, _env_float("DIR_CAP_ADVERSE_MIN_SCALE", 0.35)))
_DIR_CAP_ADVERSE_MAX_AGE_SEC = max(30.0, _env_float("DIR_CAP_ADVERSE_MAX_AGE_SEC", 180.0))
_DIR_CAP_ADVERSE_TTL_SEC = max(1.0, _env_float("DIR_CAP_ADVERSE_TTL_SEC", 5.0))
_DIR_CAP_ADVERSE_CACHE: dict[str, object] = {
    "ts": 0.0,
    "side": "",
    "lookback": 0,
    "adverse": 0.0,
    "details": {},
}


def _parse_iso_ts(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "+" not in text:
            text = text + "+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except Exception:
        return None


def _dir_cap_adverse_pips(side_label: str, pocket: str) -> tuple[float, dict[str, float]]:
    if not _DIR_CAP_ADVERSE_ENABLE:
        return 0.0, {}
    lookback = _DIR_CAP_ADVERSE_LOOKBACK_MIN
    now_mono = time.monotonic()
    cached_ts = float(_DIR_CAP_ADVERSE_CACHE.get("ts") or 0.0)
    cached_side = str(_DIR_CAP_ADVERSE_CACHE.get("side") or "")
    cached_lookback = int(_DIR_CAP_ADVERSE_CACHE.get("lookback") or 0)
    if (
        now_mono - cached_ts <= _DIR_CAP_ADVERSE_TTL_SEC
        and cached_side == side_label
        and cached_lookback == lookback
    ):
        cached_adverse = float(_DIR_CAP_ADVERSE_CACHE.get("adverse") or 0.0)
        cached_details = _DIR_CAP_ADVERSE_CACHE.get("details")
        if isinstance(cached_details, dict):
            return cached_adverse, cached_details  # type: ignore[return-value]
        return cached_adverse, {}

    candles = get_candles_snapshot("M1", limit=lookback + 1)
    if len(candles) < lookback + 1:
        return 0.0, {}
    try:
        last = candles[-1]
        ref = candles[-(lookback + 1)]
        last_close = float(last.get("close") or 0.0)
        ref_close = float(ref.get("close") or 0.0)
    except Exception:
        return 0.0, {}
    if last_close <= 0.0 or ref_close <= 0.0:
        return 0.0, {}
    drift_pips = (last_close - ref_close) / 0.01
    if side_label == "buy":
        adverse_pips = max(0.0, -drift_pips)
    else:
        adverse_pips = max(0.0, drift_pips)

    age_sec = 0.0
    last_ts = _parse_iso_ts(last.get("timestamp"))
    if last_ts is not None:
        age_sec = max(0.0, (datetime.now(timezone.utc) - last_ts).total_seconds())
        if age_sec > _DIR_CAP_ADVERSE_MAX_AGE_SEC:
            adverse_pips = 0.0

    details = {
        "lookback_min": float(lookback),
        "drift_pips": float(drift_pips),
        "adverse_pips": float(adverse_pips),
        "last_close": float(last_close),
        "ref_close": float(ref_close),
        "age_sec": float(age_sec),
    }
    _DIR_CAP_ADVERSE_CACHE.update(
        {
            "ts": now_mono,
            "side": side_label,
            "lookback": lookback,
            "adverse": adverse_pips,
            "details": details,
        }
    )
    try:
        log_metric(
            "dir_cap_adverse_pips",
            adverse_pips,
            tags={"pocket": pocket, "side": side_label, "lookback": lookback},
        )
    except Exception:
        pass
    return adverse_pips, details


def _manual_net_units(positions: Optional[dict] = None) -> tuple[int, int]:
    """Return (net_units, trade_count) for manual/unknown pockets."""
    if positions is None:
        try:
            global _DIR_CAP_CACHE
            if _DIR_CAP_CACHE is None:
                _DIR_CAP_CACHE = PositionManager()
            positions = _DIR_CAP_CACHE.get_open_positions()
        except Exception:
            return 0, 0
    if not isinstance(positions, dict):
        return 0, 0
    net_units = 0
    trade_count = 0
    for pocket in ("manual", "unknown"):
        info = positions.get(pocket) or {}
        trades = info.get("open_trades") or []
        if trades:
            trade_count += len(trades)
            for tr in trades:
                try:
                    net_units += int(tr.get("units") or 0)
                except (TypeError, ValueError):
                    continue
            continue
        try:
            net_units += int(info.get("units") or 0)
        except (TypeError, ValueError):
            pass
    meta = positions.get("__meta__") or {}
    try:
        manual_trades = int(meta.get("manual_trades") or 0)
        if manual_trades > trade_count:
            trade_count = manual_trades
    except (TypeError, ValueError):
        pass
    return net_units, trade_count

# ---------- orders logger (logs/orders.db) ----------
_ORDERS_DB_PATH = pathlib.Path("logs/orders.db")
_ORDER_DB_JOURNAL_MODE = os.getenv("ORDER_DB_JOURNAL_MODE", "WAL")
_ORDER_DB_SYNCHRONOUS = os.getenv("ORDER_DB_SYNCHRONOUS", "NORMAL")
# Keep order logging responsive while tolerating short write contention bursts.
_ORDER_DB_BUSY_TIMEOUT_MS = int(os.getenv("ORDER_DB_BUSY_TIMEOUT_MS", "1500"))
_ORDER_DB_WAL_AUTOCHECKPOINT_PAGES = int(
    os.getenv("ORDER_DB_WAL_AUTOCHECKPOINT_PAGES", "500")
)
_ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES = int(
    os.getenv("ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES", "67108864")
)
_ORDER_DB_CHECKPOINT_ENABLE = (
    os.getenv("ORDER_DB_CHECKPOINT_ENABLE", "1").strip().lower()
    not in {"", "0", "false", "no"}
)
_ORDER_DB_CHECKPOINT_INTERVAL_SEC = float(
    os.getenv("ORDER_DB_CHECKPOINT_INTERVAL_SEC", "60")
)
_ORDER_DB_CHECKPOINT_MIN_WAL_BYTES = int(
    os.getenv("ORDER_DB_CHECKPOINT_MIN_WAL_BYTES", "33554432")
)
_ORDER_DB_LOG_RETRY_ATTEMPTS = max(
    1, int(os.getenv("ORDER_DB_LOG_RETRY_ATTEMPTS", "3"))
)
_ORDER_DB_LOG_RETRY_SLEEP_SEC = max(
    0.0, float(os.getenv("ORDER_DB_LOG_RETRY_SLEEP_SEC", "0.03"))
)
_ORDER_DB_LOG_RETRY_BACKOFF = max(
    1.0, float(os.getenv("ORDER_DB_LOG_RETRY_BACKOFF", "2.0"))
)
_ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC = max(
    0.0, float(os.getenv("ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC", "0.20"))
)
_ORDER_DB_LOG_FAST_RETRY_ATTEMPTS = max(
    1, int(os.getenv("ORDER_DB_LOG_FAST_RETRY_ATTEMPTS", "1"))
)
_ORDER_DB_LOG_FAST_RETRY_SLEEP_SEC = max(
    0.0, float(os.getenv("ORDER_DB_LOG_FAST_RETRY_SLEEP_SEC", "0.0"))
)
_ORDER_DB_LOG_FAST_RETRY_BACKOFF = max(
    1.0, float(os.getenv("ORDER_DB_LOG_FAST_RETRY_BACKOFF", "1.0"))
)
_ORDER_DB_LOG_FAST_RETRY_MAX_SLEEP_SEC = max(
    0.0, float(os.getenv("ORDER_DB_LOG_FAST_RETRY_MAX_SLEEP_SEC", "0.0"))
)
_ORDERS_DB_WAL_PATH = _ORDERS_DB_PATH.with_suffix(_ORDERS_DB_PATH.suffix + "-wal")
_ORDERS_DB_LOCK_PATH = _ORDERS_DB_PATH.with_suffix(_ORDERS_DB_PATH.suffix + ".lock")
_ORDER_DB_FILE_LOCK_ENABLED = _env_bool("ORDER_DB_FILE_LOCK_ENABLED", True)
_ORDER_DB_FILE_LOCK_TIMEOUT_SEC = max(
    0.0, float(os.getenv("ORDER_DB_FILE_LOCK_TIMEOUT_SEC", "0.25"))
)
_ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC = max(
    0.0, float(os.getenv("ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC", "0.03"))
)
_ORDER_STATUS_CACHE_TTL_SEC = max(
    5.0, float(os.getenv("ORDER_STATUS_CACHE_TTL_SEC", "180.0"))
)
_LAST_ORDER_DB_CHECKPOINT = 0.0
_ORDER_STATUS_CACHE_LOCK = threading.Lock()
_ORDER_STATUS_CACHE: dict[str, tuple[float, dict[str, object]]] = {}
_ORDER_DB_LOCAL_WRITE_LOCK = threading.RLock()

_DEFAULT_MIN_HOLD_SEC = {
    "macro": 360.0,
    "micro": 150.0,
    "scalp": 75.0,
}


@contextmanager
def _order_db_file_lock(*, fast_fail: bool = False):
    """
    Cross-process write lock for orders.db access.

    WAL mode tolerates concurrent readers, but many writers from strategy/service
    processes can still thrash on SQLITE_BUSY. A short flock window serializes
    write attempts and avoids hot lock retry loops.
    """
    if not _ORDER_DB_FILE_LOCK_ENABLED or fcntl is None:
        yield
        return
    timeout_sec = (
        _ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC
        if fast_fail
        else _ORDER_DB_FILE_LOCK_TIMEOUT_SEC
    )
    _ORDERS_DB_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = open(_ORDERS_DB_LOCK_PATH, "a+")
    locked = False
    start = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if timeout_sec <= 0.0:
                    raise sqlite3.OperationalError(
                        "database is locked (orders.db file lock busy)"
                    )
                if time.monotonic() - start >= timeout_sec:
                    raise sqlite3.OperationalError(
                        "database is locked (orders.db file lock timeout)"
                    )
                time.sleep(0.005)
        yield
    finally:
        try:
            if locked:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_fp.close()
        except Exception:
            pass


def _configure_orders_sqlite(con: sqlite3.Connection) -> sqlite3.Connection:
    """Apply SQLite PRAGMAs for WAL size control."""
    try:
        con.execute(f"PRAGMA journal_mode={_ORDER_DB_JOURNAL_MODE}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA synchronous={_ORDER_DB_SYNCHRONOUS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA busy_timeout={_ORDER_DB_BUSY_TIMEOUT_MS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(
            f"PRAGMA wal_autocheckpoint={_ORDER_DB_WAL_AUTOCHECKPOINT_PAGES}"
        )
    except sqlite3.Error:
        pass
    try:
        con.execute(
            f"PRAGMA journal_size_limit={_ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES}"
        )
    except sqlite3.Error:
        pass
    return con


def _maybe_checkpoint_orders_db(con: sqlite3.Connection) -> None:
    """Best-effort WAL checkpoint to avoid runaway orders.db-wal growth."""
    if not _ORDER_DB_CHECKPOINT_ENABLE:
        return
    global _LAST_ORDER_DB_CHECKPOINT
    now = time.monotonic()
    if now - _LAST_ORDER_DB_CHECKPOINT < _ORDER_DB_CHECKPOINT_INTERVAL_SEC:
        return
    try:
        wal_size = _ORDERS_DB_WAL_PATH.stat().st_size
    except FileNotFoundError:
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("[ORDER][DB] WAL size check failed: %s", exc)
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    if wal_size < _ORDER_DB_CHECKPOINT_MIN_WAL_BYTES:
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    try:
        row = con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        busy = row[0] if row else 0
        if busy == 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        logging.info("[ORDER][DB] wal checkpoint failed: %s", exc)
    _LAST_ORDER_DB_CHECKPOINT = now


def _ensure_orders_schema() -> sqlite3.Connection:
    _ORDERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_ORDERS_DB_PATH)
    _configure_orders_sqlite(con)
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
    cols = {row[1] for row in con.execute("PRAGMA table_info(orders)")}
    if "stage_index" not in cols:
        con.execute("ALTER TABLE orders ADD COLUMN stage_index INTEGER")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entry_intent_board (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT,
          pocket TEXT,
          instrument TEXT,
          strategy_tag TEXT,
          side INTEGER,
          raw_units INTEGER,
          final_units INTEGER,
          entry_probability REAL,
          client_order_id TEXT,
          status TEXT,
          reason TEXT,
          request_json TEXT,
          ts_epoch REAL,
          expires_at REAL
        )
        """
    )
    # Useful indexes
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(client_order_id)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_entry_intent_board_scope ON entry_intent_board (strategy_tag, instrument, side, expires_at)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_entry_intent_board_status_scope ON entry_intent_board (strategy_tag, instrument, status, expires_at)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_entry_intent_board_client ON entry_intent_board (client_order_id, ts_epoch)"
    )
    con.commit()
    return con


def _orders_con() -> sqlite3.Connection:
    # Singleton-ish connection
    global _ORDERS_CON
    try:
        return _ORDERS_CON
    except NameError:
        _ORDERS_CON = _ensure_orders_schema()
        return _ORDERS_CON


def _reset_orders_con() -> None:
    con = globals().get("_ORDERS_CON")
    if con is None:
        return
    try:
        con.close()
    except Exception:
        pass
    try:
        del globals()["_ORDERS_CON"]
    except Exception:
        pass


def _rollback_orders_con(con: Optional[sqlite3.Connection]) -> None:
    if con is None:
        return
    try:
        con.rollback()
    except Exception:
        pass


def _is_sqlite_locked_error(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).strip().lower()
    return "locked" in msg or "busy" in msg


def _cache_order_status(
    *,
    ts: str,
    client_order_id: Optional[str],
    status: Optional[str],
    attempt: Optional[int],
    side: Optional[str] = None,
    units: Optional[int] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    request_payload: Optional[dict] = None,
    response_payload: Optional[dict] = None,
) -> None:
    if not client_order_id:
        return
    now_mono = time.monotonic()
    payload: dict[str, object] = {
        "ts": ts,
        "status": status,
        "attempt": attempt,
        "side": side,
        "units": units,
        "error_code": error_code,
        "error_message": error_message,
        "request_json": _safe_json(request_payload),
        "response_json": _safe_json(response_payload),
    }
    with _ORDER_STATUS_CACHE_LOCK:
        _ORDER_STATUS_CACHE[str(client_order_id)] = (now_mono, payload)
        if len(_ORDER_STATUS_CACHE) <= 2048:
            return
        stale_keys = [
            key
            for key, (updated, _) in _ORDER_STATUS_CACHE.items()
            if now_mono - updated > _ORDER_STATUS_CACHE_TTL_SEC
        ]
        for key in stale_keys:
            _ORDER_STATUS_CACHE.pop(key, None)
        if len(_ORDER_STATUS_CACHE) > 2048:
            oldest = sorted(
                _ORDER_STATUS_CACHE.items(),
                key=lambda item: item[1][0],
            )
            drop_n = len(_ORDER_STATUS_CACHE) - 2048
            for key, _ in oldest[:drop_n]:
                _ORDER_STATUS_CACHE.pop(key, None)


def _cached_order_status(client_order_id: Optional[str]) -> Optional[dict[str, object]]:
    if not client_order_id:
        return None
    now_mono = time.monotonic()
    with _ORDER_STATUS_CACHE_LOCK:
        entry = _ORDER_STATUS_CACHE.get(str(client_order_id))
        if entry is None:
            return None
        updated, payload = entry
        if now_mono - updated > _ORDER_STATUS_CACHE_TTL_SEC:
            _ORDER_STATUS_CACHE.pop(str(client_order_id), None)
            return None
        return dict(payload)


def _ensure_utc(candidate: Optional[datetime]) -> datetime:
    if candidate is None:
        return datetime.now(timezone.utc)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _estimate_price(meta: Optional[dict]) -> Optional[float]:
    if not meta:
        return None
    for key in ("entry_price", "price", "mid_price"):
        try:
            val = float(meta.get(key))
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _latest_mid_price() -> Optional[float]:
    if tick_window is None:
        return None
    try:
        ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
    except Exception:
        return None
    if not ticks:
        return None
    tick = ticks[-1]
    try:
        if tick.get("mid") is not None:
            return float(tick.get("mid"))
    except Exception:
        pass
    bid = tick.get("bid")
    ask = tick.get("ask")
    try:
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        return None
    try:
        if bid is not None:
            return float(bid)
    except Exception:
        pass
    try:
        if ask is not None:
            return float(ask)
    except Exception:
        pass
    return None


def _latest_bid_ask() -> tuple[Optional[float], Optional[float]]:
    if tick_window is None:
        return None, None
    try:
        ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
    except Exception:
        return None, None
    if not ticks:
        return None, None
    tick = ticks[-1]
    bid = _as_float(tick.get("bid"))
    ask = _as_float(tick.get("ask"))
    return bid, ask


def _entry_price_hint(entry_thesis: Optional[dict], meta: Optional[dict]) -> Optional[float]:
    if isinstance(entry_thesis, dict):
        for key in ("entry_price", "price", "entry_ref"):
            try:
                val = float(entry_thesis.get(key))
                if val > 0:
                    return val
            except Exception:
                continue
    est = _estimate_price(meta)
    if est is not None:
        return est
    return _latest_mid_price()


def _entry_nav_hint(meta: Optional[dict], entry_thesis: Optional[dict]) -> Optional[float]:
    for src in (meta, entry_thesis):
        if not isinstance(src, dict):
            continue
        for key in ("nav", "account_nav", "equity", "ref_equity", "balance"):
            value = _as_float(src.get(key))
            if value is not None and value > 0.0:
                return float(value)
    return None


def _pocket_env_float(name: str, pocket: Optional[str], default: float) -> float:
    pocket_upper = str(pocket).strip().upper() if pocket else ""
    if pocket_upper:
        raw = os.getenv(f"{name}_{pocket_upper}")
        if raw is not None:
            try:
                return float(raw)
            except Exception:
                pass
    return _env_float(name, default)


def _entry_confidence_score(
    confidence: Optional[float],
    entry_thesis: Optional[dict],
    *,
    default: float = 50.0,
) -> float:
    raw: object = confidence
    if raw is None and isinstance(entry_thesis, dict):
        raw = entry_thesis.get("confidence")
    conf = _as_float(raw, default)
    if conf is None:
        conf = default
    return max(0.0, min(100.0, float(conf)))


def _entry_probability_value(
    confidence: Optional[float],
    entry_thesis: Optional[dict],
) -> Optional[float]:
    """Return normalized entry probability in [0.0, 1.0]."""
    candidates: list[object] = []
    if isinstance(entry_thesis, dict):
        if "entry_probability" in entry_thesis:
            candidates.append(entry_thesis.get("entry_probability"))
        if "confidence" in entry_thesis:
            candidates.append(entry_thesis.get("confidence"))
    candidates.append(confidence)

    for raw in candidates:
        if raw is None:
            continue
        try:
            prob = float(raw)
        except Exception:
            continue
        if math.isnan(prob) or math.isinf(prob):
            continue
        if prob < 0:
            return 0.0
        if prob <= 1.0:
            return min(1.0, max(0.0, prob))
    if prob <= 100.0:
        return min(1.0, max(0.0, prob / 100.0))
    return None


def _probability_scaled_units(
    units: int,
    *,
    pocket: Optional[str],
    strategy_tag: Optional[str],
    entry_probability: Optional[float],
) -> tuple[int, Optional[str]]:
    """
    Return size scaled by entry_probability.
    Returns `(scaled_units, reject_reason)` where reject_reason is not None when
    the order should be skipped under preserve-intent mode.
    """
    strategy_min_scale = _order_manager_preserve_intent_min_scale(strategy_tag)
    strategy_max_scale = _order_manager_preserve_intent_max_scale(strategy_tag)
    strategy_boost_probability = _order_manager_preserve_intent_boost_probability(
        strategy_tag
    )
    strategy_reject_under = _order_manager_preserve_intent_reject_under(strategy_tag)
    if strategy_boost_probability <= strategy_reject_under:
        strategy_boost_probability = min(
            1.0,
            strategy_reject_under + 0.0001,
        )

    if units == 0:
        return 0, None
    if entry_probability is None:
        return units, None
    if entry_probability <= strategy_reject_under:
        return 0, "entry_probability_reject_threshold"

    probability = max(0.0, min(1.0, float(entry_probability)))
    if probability <= strategy_boost_probability:
        scale = max(float(strategy_min_scale), probability)
    elif strategy_max_scale <= 1.0:
        scale = 1.0
    else:
        boost_span = 1.0 - strategy_boost_probability
        if boost_span <= 0.0:
            scale = 1.0
        else:
            scale = 1.0 + (
                (probability - strategy_boost_probability) / boost_span
            ) * (strategy_max_scale - 1.0)

    if scale <= 0:
        return 0, "entry_probability_scale_to_zero"
    scale = max(0.0, min(strategy_max_scale, scale))
    scaled_abs = int(round(abs(units) * scale))
    if scaled_abs <= 0:
        return 0, "entry_probability_scale_to_zero"
    min_units = min_units_for_strategy(strategy_tag, pocket=pocket)
    if min_units > 0 and scaled_abs < min_units:
        return 0, "entry_probability_below_min_units"
    return (scaled_abs if units > 0 else -scaled_abs), None


def _normalize_intent_probability(value: Optional[float]) -> float:
    """Normalize intent probability to [0,1] weight."""
    if value is None:
        return 1.0
    try:
        prob = float(value)
    except Exception:
        return 1.0
    if math.isnan(prob) or math.isinf(prob):
        return 1.0
    if prob <= 1.0:
        return max(0.0, min(1.0, prob))
    return max(0.0, min(1.0, prob / 100.0))


def _entry_intent_board_purge(now_epoch: float | None = None) -> None:
    """Purge expired entries from the intent board."""
    if not _ORDER_INTENT_COORDINATION_ENABLED:
        return
    if now_epoch is None:
        now_epoch = time.time()
    con: Optional[sqlite3.Connection] = None
    try:
        with _order_db_file_lock(fast_fail=True):
            con = _orders_con()
            con.execute(
                "DELETE FROM entry_intent_board WHERE expires_at < ?",
                (float(now_epoch),),
            )
            con.commit()
    except Exception as exc:
        _rollback_orders_con(con)
        if _is_sqlite_locked_error(exc):
            _reset_orders_con()
        logging.debug("[ORDER][INTENT] purge failed: %s", exc)


def _entry_intent_board_record(
    *,
    pocket: str,
    instrument: str,
    strategy_tag: Optional[str],
    side: int,
    raw_units: int,
    final_units: int,
    entry_probability: Optional[float],
    client_order_id: Optional[str],
    status: str,
    reason: Optional[str] = None,
    request_payload: Optional[dict] = None,
) -> None:
    """Persist one intent record for downstream coordination."""
    now_epoch = time.time()
    ts = datetime.fromtimestamp(now_epoch, tz=timezone.utc).isoformat()
    payload = request_payload if isinstance(request_payload, dict) else {}
    con: Optional[sqlite3.Connection] = None
    try:
        with _order_db_file_lock(fast_fail=True):
            con = _orders_con()
            con.execute(
                """
                INSERT INTO entry_intent_board (
                  ts, pocket, instrument, strategy_tag, side, raw_units, final_units,
                  entry_probability, client_order_id, status, reason, request_json,
                  ts_epoch, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    pocket,
                    instrument,
                    strategy_tag,
                    int(side),
                    int(abs(raw_units)) if raw_units is not None else 0,
                    int(abs(final_units)) if final_units is not None else 0,
                    _normalize_intent_probability(entry_probability),
                    client_order_id,
                    status,
                    str(reason) if reason else None,
                    _safe_json(payload),
                    float(now_epoch),
                    float(now_epoch + _ORDER_INTENT_COORDINATION_WINDOW_SEC),
                ),
            )
            con.commit()
    except Exception as exc:
        _rollback_orders_con(con)
        if _is_sqlite_locked_error(exc):
            _reset_orders_con()
        logging.debug("[ORDER][INTENT] record failed: %s", exc)


def _coordinate_entry_intent(
    *,
    instrument: str,
    pocket: str,
    strategy_tag: Optional[str],
    side: int,
    raw_units: int,
    entry_probability: Optional[float],
    client_order_id: Optional[str],
    min_units: int,
    forecast_context: Optional[dict[str, object]] = None,
) -> tuple[int, Optional[str], dict[str, float | str | int | object]]:
    """
    Coordinate one intent with recent intents in the same strategy+instrument scope.

    Returns (final_units, reason, details).
    reason is None when accepted, "reject" for hard reject.
    """
    if not _ORDER_INTENT_COORDINATION_ENABLED:
        return raw_units, None, {"coordination_enabled": 0.0}
    if not (pocket and pocket.lower() != "manual"):
        return raw_units, None, {"coordination_enabled": 0.0}
    if not instrument:
        return raw_units, "reject", {"coordination_error": "missing_instrument"}
    if not strategy_tag:
        return raw_units, "reject", {"coordination_error": "missing_strategy_tag"}
    if raw_units == 0:
        return raw_units, None, {"coordination_enabled": 1.0}

    try:
        requested_side = 1 if raw_units > 0 else -1
        if side != 0 and requested_side != side:
            requested_side = 1 if side > 0 else -1
        side = requested_side
    except Exception:
        side = 1 if raw_units > 0 else -1

    now_epoch = time.time()
    _entry_intent_board_purge(now_epoch=now_epoch)

    try:
        con = _orders_con()
        window_start = now_epoch - _ORDER_INTENT_COORDINATION_WINDOW_SEC
        rows = con.execute(
            """
            SELECT side, final_units, entry_probability, strategy_tag
            FROM entry_intent_board
            WHERE strategy_tag = ?
              AND instrument = ?
              AND status IN ('intent_accepted', 'intent_scaled')
              AND ts_epoch >= ?
              AND (client_order_id IS NULL OR client_order_id != ?)
            """,
            (
                str(strategy_tag),
                instrument,
                window_start,
                str(client_order_id) if client_order_id else "",
            ),
        ).fetchall()
    except Exception as exc:
        logging.debug("[ORDER][INTENT] load failed: %s", exc)
        _entry_intent_board_record(
            pocket=pocket,
            instrument=instrument,
            strategy_tag=strategy_tag,
            side=side,
            raw_units=raw_units,
            final_units=raw_units,
            entry_probability=entry_probability,
            client_order_id=client_order_id,
            status="intent_accepted",
            reason="coordination_load_error",
            request_payload={"error": str(exc)},
        )
        return raw_units, None, {"coordination_enabled": 0.0, "coordination_error": str(exc)}

    own_prob = _normalize_intent_probability(entry_probability)
    own_units = abs(int(raw_units))
    own_score = float(max(1, own_units)) * max(0.0, own_prob)

    opposite_score = 0.0
    same_score = 0.0
    opposite_count = 0
    same_count = 0
    for row in rows:
        try:
            row_side = int(row[0])
            raw = int(row[1] or 0)
            row_prob = _normalize_intent_probability(_as_float(row[2]))
        except Exception:
            continue
        if raw <= 0 or abs(row_side) not in (1, -1):
            continue
        row_score = float(raw) * max(0.0, row_prob)
        if row_side == side:
            same_score += row_score
            same_count += 1
        else:
            opposite_score += row_score
            opposite_count += 1

    final_abs = abs(raw_units)
    forecast_context = (
        dict(forecast_context)
        if isinstance(forecast_context, dict)
        else None
    )
    details = {
        "coordination_enabled": 1.0,
        "raw_units": own_units,
        "entry_probability": own_prob,
        "same_count": int(same_count),
        "opposite_count": int(opposite_count),
        "same_score": round(same_score, 6),
        "opposite_score": round(opposite_score, 6),
        "scale": 1.0,
    }
    if forecast_context:
        details["forecast_context"] = forecast_context
    if final_abs <= 0:
        details["decision"] = "reject"
        _entry_intent_board_record(
            pocket=pocket,
            instrument=instrument,
            strategy_tag=strategy_tag,
            side=side,
            raw_units=raw_units,
            final_units=0,
            entry_probability=entry_probability,
            client_order_id=client_order_id,
            status="intent_rejected",
            reason="scale_to_zero",
            request_payload=details,
        )
        return 0, "reject", details
    if min_units > 0 and final_abs < min_units:
        details["decision"] = "reject"
        details["min_units"] = int(min_units)
        _entry_intent_board_record(
            pocket=pocket,
            instrument=instrument,
            strategy_tag=strategy_tag,
            side=side,
            raw_units=raw_units,
            final_units=0,
            entry_probability=entry_probability,
            client_order_id=client_order_id,
            status="intent_rejected",
            reason="below_min_units_after_scale",
            request_payload=details,
        )
        return 0, "reject", details
    final_units = raw_units
    if opposite_score <= 0.0:
        final_units = raw_units
        details["decision"] = "accepted"
        _entry_intent_board_record(
            pocket=pocket,
            instrument=instrument,
            strategy_tag=strategy_tag,
            side=side,
            raw_units=raw_units,
            final_units=final_units,
            entry_probability=entry_probability,
            client_order_id=client_order_id,
            status="intent_accepted",
            request_payload=details,
        )
        return final_units, None, details
    dominance = opposite_score / max(own_score, 1.0)
    details["opposite_ratio_to_own"] = round(dominance, 6)
    details["dominance_threshold"] = max(
        _order_manager_coordination_rejection_dominance(strategy_tag),
        1.0,
    )
    details["decision"] = "accepted"
    _entry_intent_board_record(
        pocket=pocket,
        instrument=instrument,
        strategy_tag=strategy_tag,
        side=side,
        raw_units=raw_units,
        final_units=final_units,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        status="intent_accepted",
        reason=details["decision"],
        request_payload=details,
    )
    return final_units, None, details


async def coordinate_entry_intent(
    *,
    instrument: str,
    pocket: str,
    strategy_tag: Optional[str],
    side: int,
    raw_units: int,
    entry_probability: Optional[float],
    client_order_id: Optional[str],
    min_units: int,
    forecast_context: Optional[dict[str, object]] = None,
) -> tuple[int, Optional[str], dict[str, float | str | int | object]]:
    """
    Coordinate one strategy intent.

    This public entry is used by strategy side before dispatching to market/limit
    so that the strategy can inject intent context first and avoid duplicate
    coordination on order-manager.
    """
    payload = {
        "instrument": instrument,
        "pocket": pocket,
        "strategy_tag": strategy_tag,
        "side": side,
        "raw_units": raw_units,
        "entry_probability": entry_probability,
        "client_order_id": client_order_id,
        "min_units": min_units,
        "forecast_context": forecast_context,
    }

    service_result = await _order_manager_service_request_async(
        "/order/coordinate_entry_intent",
        payload,
    )
    if service_result is not None:
        if isinstance(service_result, dict):
            final_units = service_result.get("final_units")
            reason = service_result.get("reason")
            details = service_result.get("details")
            if (
                isinstance(final_units, int)
                and isinstance(details, dict)
            ):
                return final_units, reason, details
        raise RuntimeError(
            f"invalid coordinate_entry_intent service payload: {service_result!r}"
        )

    return _coordinate_entry_intent(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=side,
        raw_units=raw_units,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        min_units=min_units,
        forecast_context=forecast_context,
    )


def _ensure_entry_intent_payload(
    units: int,
    confidence: Optional[float],
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
) -> Optional[dict]:
    """Attach mandatory intent fields to entry_thesis for order-manager compatibility."""
    if not isinstance(entry_thesis, dict):
        if entry_thesis is None:
            entry_thesis = {}
        else:
            return None

    thesis = dict(entry_thesis)
    if strategy_tag:
        if isinstance(strategy_tag, str):
            strategy_tag = strategy_tag.strip()
            if strategy_tag and "strategy_tag" not in thesis:
                thesis["strategy_tag"] = strategy_tag

    if "entry_units_intent" not in thesis:
        thesis["entry_units_intent"] = abs(int(units))

    prob = _entry_probability_value(confidence, thesis)
    if prob is not None:
        if thesis.get("entry_probability") != prob:
            thesis["entry_probability"] = prob

    return thesis


def _entry_market_snapshot(
    pocket: Optional[str],
    entry_thesis: Optional[dict],
) -> dict[str, float | str | None]:
    tf, fac = _tp_cap_factors(pocket, entry_thesis)
    atr_pips = _as_float(fac.get("atr_pips"))
    if atr_pips is None:
        atr_raw = _as_float(fac.get("atr"))
        if atr_raw is not None:
            atr_pips = atr_raw * 100.0
    vol_5m = _as_float(fac.get("vol_5m"))
    adx = _as_float(fac.get("adx"))

    if isinstance(entry_thesis, dict):
        if atr_pips is None:
            atr_pips = _as_float(entry_thesis.get("atr_pips"))
            if atr_pips is None:
                atr_raw = _as_float(entry_thesis.get("atr"))
                if atr_raw is not None:
                    atr_pips = atr_raw * 100.0
        if vol_5m is None:
            vol_5m = _as_float(entry_thesis.get("vol_5m"))

    return {
        "tf": tf,
        "atr_pips": atr_pips,
        "vol_5m": vol_5m,
        "adx": adx,
    }


def _entry_quality_tag_variants(strategy_tag: Optional[str]) -> tuple[str, ...]:
    raw = str(strategy_tag or "").strip()
    if not raw:
        return tuple()
    base = _base_strategy_tag(raw)
    root = raw.split("-", 1)[0].strip()
    variants = {raw.lower()}
    if base:
        variants.add(str(base).strip().lower())
    if root:
        variants.add(root.lower())
    return tuple(sorted(v for v in variants if v))


def _entry_strategy_quality_snapshot(
    strategy_tag: Optional[str],
    pocket: Optional[str],
) -> Optional[dict[str, float]]:
    if not _ENTRY_QUALITY_STRAT_ENABLED:
        return None
    pocket_key = str(pocket or "").strip().lower()
    if not pocket_key:
        return None
    variants = _entry_quality_tag_variants(strategy_tag)
    if not variants:
        return None
    db_path = _ENTRY_QUALITY_STRAT_DB_PATH
    if not db_path.exists():
        return None

    cache_key = (pocket_key, str(strategy_tag or "").strip().lower())
    now_mono = time.monotonic()
    cached = _ENTRY_QUALITY_STRAT_CACHE.get(cache_key)
    if cached and now_mono - cached[0] <= _ENTRY_QUALITY_STRAT_TTL_SEC:
        return dict(cached[1])

    placeholders = ",".join("?" for _ in variants)
    params: list[Any] = list(variants)
    params.extend([pocket_key, f"-{_ENTRY_QUALITY_STRAT_LOOKBACK_DAYS} day"])

    row = None
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        row = con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win_n,
              SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS loss_n,
              SUM(pl_pips) AS sum_pips
            FROM trades
            WHERE LOWER(COALESCE(NULLIF(strategy_tag, ''), strategy)) IN ({placeholders})
              AND pocket = ?
              AND close_time >= datetime('now', ?)
            """,
            params,
        ).fetchone()
    except Exception:
        row = None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    if not row:
        return None

    sample = int(row["n"] or 0)
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win_n = int(row["win_n"] or 0)
    loss_n = int(row["loss_n"] or 0)
    sum_pips = float(row["sum_pips"] or 0.0)

    pf = profit / loss if loss > 0.0 else float("inf")
    win_rate = (float(win_n) / float(sample)) if sample > 0 else 0.0
    avg_pips = (sum_pips / float(sample)) if sample > 0 else 0.0
    avg_win = (profit / float(win_n)) if win_n > 0 else 0.0
    avg_loss = (loss / float(loss_n)) if loss_n > 0 else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else float("inf")

    stats = {
        "sample": float(sample),
        "pf": float(pf),
        "win_rate": float(win_rate),
        "avg_pips": float(avg_pips),
        "avg_win_pips": float(avg_win),
        "avg_loss_pips": float(avg_loss),
        "payoff": float(payoff),
    }
    _ENTRY_QUALITY_STRAT_CACHE[cache_key] = (now_mono, dict(stats))
    return stats


def _dynamic_entry_sl_target_pips(
    pocket: Optional[str],
    *,
    entry_thesis: Optional[dict],
    quote: Optional[dict],
    sl_hint_pips: Optional[float],
    loss_guard_pips: Optional[float],
) -> tuple[Optional[float], dict[str, float | str | None]]:
    pocket_key = str(pocket or "").strip().lower()
    if not _DYNAMIC_SL_ENABLE or pocket_key not in _DYNAMIC_SL_POCKETS:
        return None, {"enabled": 0.0}

    base_sl = 0.0
    for cand in (sl_hint_pips, loss_guard_pips):
        if cand is not None and cand > base_sl:
            base_sl = float(cand)
    if base_sl <= 0.0:
        return None, {"enabled": 1.0, "reason": "no_sl_hint"}

    snap = _entry_market_snapshot(pocket, entry_thesis)
    atr_pips = _as_float(snap.get("atr_pips"))
    vol_5m = _as_float(snap.get("vol_5m"))
    spread_pips = _as_float((quote or {}).get("spread_pips")) if isinstance(quote, dict) else None

    ratio = max(1.0, _pocket_env_float("ORDER_DYNAMIC_SL_RATIO", pocket, _DYNAMIC_SL_RATIO))
    target = base_sl * ratio

    atr_mult = max(0.0, _pocket_env_float("ORDER_DYNAMIC_SL_ATR_MULT", pocket, 0.72))
    if atr_pips is not None and atr_pips > 0.0 and atr_mult > 0.0:
        target = max(target, atr_pips * atr_mult)

    spread_mult = max(0.0, _pocket_env_float("ORDER_DYNAMIC_SL_SPREAD_MULT", pocket, 2.8))
    if spread_pips is not None and spread_pips > 0.0 and spread_mult > 0.0:
        target = max(target, spread_pips * spread_mult)

    vol_low = _pocket_env_float("ORDER_DYNAMIC_SL_VOL5M_LOW", pocket, 0.8)
    vol_high = max(vol_low + 1e-6, _pocket_env_float("ORDER_DYNAMIC_SL_VOL5M_HIGH", pocket, 1.6))
    if vol_5m is not None:
        if vol_5m > vol_high:
            target *= min(1.35, 1.0 + (vol_5m - vol_high) * 0.14)
        elif vol_5m < vol_low:
            target *= max(0.90, 1.0 - (vol_low - vol_5m) * 0.06)

    max_pips = max(0.0, _pocket_env_float("ORDER_DYNAMIC_SL_MAX_PIPS", pocket, _DYNAMIC_SL_MAX_PIPS))
    if max_pips > 0.0:
        target = min(target, max_pips)
    min_pips = max(0.0, _pocket_env_float("ORDER_DYNAMIC_SL_MIN_PIPS", pocket, 0.0))
    target = max(base_sl, target, min_pips)
    target = round(float(target), 4)

    return target, {
        "enabled": 1.0,
        "base_sl_pips": base_sl,
        "target_sl_pips": target,
        "atr_pips": atr_pips,
        "vol_5m": vol_5m,
        "spread_pips": spread_pips,
        "ratio": ratio,
        "atr_mult": atr_mult,
        "spread_mult": spread_mult,
        "tf": str(snap.get("tf") or ""),
    }


def _entry_quality_gate(
    pocket: Optional[str],
    *,
    confidence: Optional[float],
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
    quote: Optional[dict],
    sl_pips: Optional[float],
    tp_pips: Optional[float],
) -> tuple[bool, Optional[str], dict[str, float | str | None]]:
    pocket_key = str(pocket or "").strip().lower()
    if not _ENTRY_QUALITY_GATE_ENABLED or pocket_key not in _ENTRY_QUALITY_POCKETS:
        return True, None, {"enabled": 0.0}

    conf = _entry_confidence_score(confidence, entry_thesis)
    base_default = _ENTRY_QUALITY_BASE_MIN_CONF.get(pocket_key, 56.0)
    required = _pocket_env_float("ORDER_ENTRY_QUALITY_MIN_CONF", pocket, base_default)
    max_conf = max(required, _pocket_env_float("ORDER_ENTRY_QUALITY_MAX_CONF", pocket, 95.0))
    bypass_conf = _pocket_env_float("ORDER_ENTRY_QUALITY_BYPASS_CONF", pocket, 92.0)

    snap = _entry_market_snapshot(pocket, entry_thesis)
    atr_pips = _as_float(snap.get("atr_pips"))
    vol_5m = _as_float(snap.get("vol_5m"))
    spread_pips = _as_float((quote or {}).get("spread_pips")) if isinstance(quote, dict) else None

    atr_low_d, atr_high_d = _ENTRY_QUALITY_ATR_BANDS.get(pocket_key, (1.2, 4.0))
    atr_low = _pocket_env_float("ORDER_ENTRY_QUALITY_ATR_LOW", pocket, atr_low_d)
    atr_high = max(atr_low + 1e-6, _pocket_env_float("ORDER_ENTRY_QUALITY_ATR_HIGH", pocket, atr_high_d))
    if atr_pips is not None:
        if atr_pips >= atr_high:
            required += _pocket_env_float("ORDER_ENTRY_QUALITY_HIGH_ATR_BONUS", pocket, 4.0)
        elif atr_pips <= atr_low:
            required += _pocket_env_float("ORDER_ENTRY_QUALITY_LOW_ATR_BONUS", pocket, 2.0)

    vol_low_d, vol_high_d = _ENTRY_QUALITY_VOL_BANDS.get(pocket_key, (0.8, 1.6))
    vol_low = _pocket_env_float("ORDER_ENTRY_QUALITY_VOL5M_LOW", pocket, vol_low_d)
    vol_high = max(vol_low + 1e-6, _pocket_env_float("ORDER_ENTRY_QUALITY_VOL5M_HIGH", pocket, vol_high_d))
    if vol_5m is not None:
        if vol_5m >= vol_high:
            required += _pocket_env_float("ORDER_ENTRY_QUALITY_HIGH_VOL_BONUS", pocket, 3.0)
        elif vol_5m <= vol_low:
            required += _pocket_env_float("ORDER_ENTRY_QUALITY_LOW_VOL_BONUS", pocket, 1.0)

    strategy_sample = None
    strategy_pf = None
    strategy_win_rate = None
    strategy_avg_pips = None
    strategy_payoff = None
    strategy_penalty = 0.0
    if _ENTRY_QUALITY_STRAT_ENABLED and strategy_tag:
        stats = _entry_strategy_quality_snapshot(strategy_tag, pocket)
        if stats:
            strategy_sample = int(stats.get("sample", 0.0) or 0.0)
            strategy_pf = _as_float(stats.get("pf"))
            strategy_win_rate = _as_float(stats.get("win_rate"))
            strategy_avg_pips = _as_float(stats.get("avg_pips"))
            strategy_payoff = _as_float(stats.get("payoff"))

            strat_min_trades = int(
                max(
                    5.0,
                    _pocket_env_float(
                        "ORDER_ENTRY_QUALITY_STRAT_MIN_TRADES",
                        pocket,
                        float(_ENTRY_QUALITY_STRAT_MIN_TRADES),
                    ),
                )
            )
            if strategy_sample >= strat_min_trades:
                pf_floor = max(
                    0.0,
                    _pocket_env_float("ORDER_ENTRY_QUALITY_STRAT_PF_MIN", pocket, 0.90),
                )
                payoff_floor = max(
                    0.0,
                    _pocket_env_float("ORDER_ENTRY_QUALITY_STRAT_PAYOFF_MIN", pocket, 0.30),
                )
                avg_pips_floor = _pocket_env_float(
                    "ORDER_ENTRY_QUALITY_STRAT_AVG_PIPS_MIN",
                    pocket,
                    0.0,
                )
                bonus_gain = max(
                    0.0,
                    _pocket_env_float("ORDER_ENTRY_QUALITY_STRAT_BONUS_GAIN", pocket, 7.0),
                )
                bonus_max = max(
                    0.0,
                    _pocket_env_float("ORDER_ENTRY_QUALITY_STRAT_BONUS_MAX", pocket, 10.0),
                )

                pf_gap = 0.0
                if (
                    pf_floor > 0.0
                    and strategy_pf is not None
                    and math.isfinite(strategy_pf)
                    and strategy_pf < pf_floor
                ):
                    pf_gap = (pf_floor - strategy_pf) / pf_floor

                payoff_gap = 0.0
                if (
                    payoff_floor > 0.0
                    and strategy_payoff is not None
                    and math.isfinite(strategy_payoff)
                    and strategy_payoff < payoff_floor
                ):
                    payoff_gap = (payoff_floor - strategy_payoff) / payoff_floor

                avg_gap = 0.0
                if (
                    strategy_avg_pips is not None
                    and math.isfinite(strategy_avg_pips)
                    and strategy_avg_pips < avg_pips_floor
                ):
                    denom = max(0.6, abs(avg_pips_floor) + 0.6)
                    avg_gap = (avg_pips_floor - strategy_avg_pips) / denom

                severity = max(0.0, pf_gap, payoff_gap, avg_gap)
                if severity > 0.0 and bonus_gain > 0.0 and bonus_max > 0.0:
                    strategy_penalty = min(bonus_max, severity * bonus_gain)
                    required += strategy_penalty

    spread_bonus = 0.0
    if spread_pips is not None and spread_pips > 0.0:
        spread_ref_default = max(0.25, _ORDER_SPREAD_BLOCK_PIPS * 0.55)
        spread_ref = max(
            0.05,
            _pocket_env_float("ORDER_ENTRY_QUALITY_SPREAD_REF_PIPS", pocket, spread_ref_default),
        )
        spread_gain = max(
            0.0,
            _pocket_env_float("ORDER_ENTRY_QUALITY_SPREAD_BONUS_GAIN", pocket, 6.0),
        )
        spread_max_bonus = max(
            0.0,
            _pocket_env_float("ORDER_ENTRY_QUALITY_SPREAD_MAX_BONUS", pocket, 10.0),
        )
        spread_pressure = spread_pips / spread_ref
        if spread_pressure > 1.0:
            spread_bonus = min(spread_max_bonus, (spread_pressure - 1.0) * spread_gain)
            required += spread_bonus

    spread_sl_ratio = None
    spread_tp_ratio = None
    if spread_pips is not None and spread_pips > 0.0:
        if sl_pips is not None and sl_pips > 0.0:
            spread_sl_ratio = spread_pips / sl_pips
        if tp_pips is not None and tp_pips > 0.0:
            spread_tp_ratio = spread_pips / tp_pips

    max_spread_sl_ratio = _pocket_env_float(
        "ORDER_ENTRY_QUALITY_SPREAD_SL_MAX_RATIO",
        pocket,
        _ENTRY_QUALITY_SPREAD_SL_MAX_RATIO.get(pocket_key, 0.36),
    )
    max_spread_tp_ratio = _pocket_env_float(
        "ORDER_ENTRY_QUALITY_SPREAD_TP_MAX_RATIO",
        pocket,
        _ENTRY_QUALITY_SPREAD_TP_MAX_RATIO.get(pocket_key, 0.24),
    )
    if spread_sl_ratio is not None and spread_sl_ratio > max_spread_sl_ratio and conf < bypass_conf:
        return False, "entry_quality_spread_sl", {
            "enabled": 1.0,
            "confidence": conf,
            "required_conf": required,
            "spread_pips": spread_pips,
            "sl_pips": sl_pips,
            "spread_sl_ratio": spread_sl_ratio,
            "max_spread_sl_ratio": max_spread_sl_ratio,
            "bypass_conf": bypass_conf,
            "atr_pips": atr_pips,
            "vol_5m": vol_5m,
            "strategy_tag": str(strategy_tag or ""),
            "strategy_sample": strategy_sample,
            "strategy_pf": strategy_pf,
            "strategy_win_rate": strategy_win_rate,
            "strategy_avg_pips": strategy_avg_pips,
            "strategy_payoff": strategy_payoff,
            "strategy_penalty": strategy_penalty,
            "tf": str(snap.get("tf") or ""),
        }
    if spread_tp_ratio is not None and spread_tp_ratio > max_spread_tp_ratio and conf < bypass_conf:
        spread_tp_soften_target_ratio = _pocket_env_float(
            "ORDER_ENTRY_QUALITY_SPREAD_TP_SOFTEN_TARGET_RATIO",
            pocket,
            max_spread_tp_ratio,
        )
        spread_tp_soften_target_ratio = max(
            1e-9, min(max_spread_tp_ratio, spread_tp_soften_target_ratio)
        )
        spread_tp_soften_min_scale = _pocket_env_float(
            "ORDER_ENTRY_QUALITY_SPREAD_TP_SOFTEN_MIN_SCALE",
            pocket,
            _ENTRY_QUALITY_SPREAD_TP_SOFTEN_MIN_SCALE,
        )
        spread_tp_soften_min_scale = max(0.01, min(1.0, spread_tp_soften_min_scale))
        soften_scale = min(
            1.0,
            max(0.0, spread_tp_soften_target_ratio / spread_tp_ratio),
        )
        if (
            _ENTRY_QUALITY_SPREAD_TP_SOFTEN_ENABLED
            and 0.0 < soften_scale < 1.0
            and math.isfinite(soften_scale)
            and soften_scale >= spread_tp_soften_min_scale
        ):
            return False, "entry_quality_spread_tp_softened", {
                "enabled": 1.0,
                "confidence": conf,
                "required_conf": required,
                "spread_pips": spread_pips,
                "tp_pips": tp_pips,
                "spread_tp_ratio": spread_tp_ratio,
                "max_spread_tp_ratio": max_spread_tp_ratio,
                "spread_tp_soften_target_ratio": spread_tp_soften_target_ratio,
                "spread_tp_soften_min_scale": spread_tp_soften_min_scale,
                "recommended_units_scale": soften_scale,
                "bypass_conf": bypass_conf,
                "atr_pips": atr_pips,
                "vol_5m": vol_5m,
                "strategy_tag": str(strategy_tag or ""),
                "strategy_sample": strategy_sample,
                "strategy_pf": strategy_pf,
                "strategy_win_rate": strategy_win_rate,
                "strategy_avg_pips": strategy_avg_pips,
                "strategy_payoff": strategy_payoff,
                "strategy_penalty": strategy_penalty,
                "tf": str(snap.get("tf") or ""),
            }
        return False, "entry_quality_spread_tp", {
            "enabled": 1.0,
            "confidence": conf,
            "required_conf": required,
            "spread_pips": spread_pips,
            "tp_pips": tp_pips,
            "spread_tp_ratio": spread_tp_ratio,
            "max_spread_tp_ratio": max_spread_tp_ratio,
            "bypass_conf": bypass_conf,
            "atr_pips": atr_pips,
            "vol_5m": vol_5m,
            "strategy_tag": str(strategy_tag or ""),
            "strategy_sample": strategy_sample,
            "strategy_pf": strategy_pf,
            "strategy_win_rate": strategy_win_rate,
            "strategy_avg_pips": strategy_avg_pips,
            "strategy_payoff": strategy_payoff,
            "strategy_penalty": strategy_penalty,
            "tf": str(snap.get("tf") or ""),
        }

    required = max(0.0, min(max_conf, required))
    if conf + 1e-6 < required:
        reason = "entry_quality_strategy_confidence" if strategy_penalty > 0.0 else "entry_quality_confidence"
        return False, reason, {
            "enabled": 1.0,
            "confidence": conf,
            "required_conf": required,
            "spread_pips": spread_pips,
            "spread_bonus": spread_bonus,
            "atr_pips": atr_pips,
            "vol_5m": vol_5m,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "spread_sl_ratio": spread_sl_ratio,
            "spread_tp_ratio": spread_tp_ratio,
            "max_spread_sl_ratio": max_spread_sl_ratio,
            "max_spread_tp_ratio": max_spread_tp_ratio,
            "bypass_conf": bypass_conf,
            "strategy_tag": str(strategy_tag or ""),
            "strategy_sample": strategy_sample,
            "strategy_pf": strategy_pf,
            "strategy_win_rate": strategy_win_rate,
            "strategy_avg_pips": strategy_avg_pips,
            "strategy_payoff": strategy_payoff,
            "strategy_penalty": strategy_penalty,
            "tf": str(snap.get("tf") or ""),
        }

    return True, None, {
        "enabled": 1.0,
        "confidence": conf,
        "required_conf": required,
        "spread_pips": spread_pips,
        "spread_bonus": spread_bonus,
        "atr_pips": atr_pips,
        "vol_5m": vol_5m,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "spread_sl_ratio": spread_sl_ratio,
        "spread_tp_ratio": spread_tp_ratio,
        "max_spread_sl_ratio": max_spread_sl_ratio,
        "max_spread_tp_ratio": max_spread_tp_ratio,
        "bypass_conf": bypass_conf,
        "strategy_tag": str(strategy_tag or ""),
        "strategy_sample": strategy_sample,
        "strategy_pf": strategy_pf,
        "strategy_win_rate": strategy_win_rate,
        "strategy_avg_pips": strategy_avg_pips,
        "strategy_payoff": strategy_payoff,
        "strategy_penalty": strategy_penalty,
        "tf": str(snap.get("tf") or ""),
    }


def _strategy_tag_from_thesis(entry_thesis: Optional[dict]) -> Optional[str]:
    if not isinstance(entry_thesis, dict):
        return None
    raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
    if raw_tag:
        return str(raw_tag)
    return None


def _strategy_tag_from_client_id(client_order_id: Optional[str]) -> Optional[str]:
    if not client_order_id:
        return None
    cid = str(client_order_id)
    if not cid:
        return None
    if cid.startswith("qr-fast-"):
        return "fast_scalp"
    if cid.startswith("qr-pullback-s5-"):
        return "pullback_s5"
    parts = cid.split("-")
    if len(parts) < 3:
        return None

    # Newer IDs: qr-<ts>-<pocket>-<tag>-<digest>
    if parts[1].isdigit() and len(parts) >= 4:
        tag = "-".join(parts[3:-1]) or "-".join(parts[3:])
        return tag or None

    # Legacy / fallback IDs: qr-<focus>-<tag>-<digest>
    tag = "-".join(parts[2:-1]) or "-".join(parts[2:])
    return tag or None


def _apply_default_entry_thesis_tfs(
    entry_thesis: Optional[dict],
    pocket: Optional[str],
) -> Optional[dict]:
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    if not pocket or pocket == "manual":
        return entry_thesis
    defaults = _DEFAULT_ENTRY_THESIS_TFS.get(pocket)
    if not defaults:
        return entry_thesis
    env_tf_default, struct_tf_default = defaults

    def _missing(key: str) -> bool:
        val = entry_thesis.get(key)
        if val is None:
            return True
        return isinstance(val, str) and not val.strip()

    updates: dict[str, str] = {}
    struct_tf_value = entry_thesis.get("struct_tf")
    if _missing("env_tf"):
        updates["env_tf"] = env_tf_default
    if _missing("struct_tf"):
        updates["struct_tf"] = struct_tf_default
        struct_tf_value = struct_tf_default
    if _missing("entry_tf"):
        entry_tf_value = struct_tf_value or struct_tf_default
        updates["entry_tf"] = entry_tf_value
    if not updates:
        return entry_thesis
    merged = dict(entry_thesis)
    merged.update(updates)
    return merged


_ENTRY_THESIS_FLAG_KEYS = (
    "trend_bias",
    "trend_score",
    "size_factor_hint",
    "range_snapshot",
    "entry_mean",
    "reversion_failure",
    "mr_guard",
    "mr_overlay",
    "tp_mode",
    "tp_target",
    "section_axis",
    "pattern_tag",
    "pattern_meta",
    "profile",
)

_ORDER_THESIS_REGIME_ENABLED = os.getenv("ORDER_THESIS_REGIME_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}


def _augment_entry_thesis_flags(entry_thesis: Optional[dict]) -> Optional[dict]:
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    flags: set[str] = set()
    existing = entry_thesis.get("flags")
    if isinstance(existing, (list, tuple, set)):
        for item in existing:
            if item:
                flags.add(str(item))
    for key in _ENTRY_THESIS_FLAG_KEYS:
        val = entry_thesis.get(key)
        if val in (None, False, "", 0):
            continue
        flags.add(key)
    if not flags:
        return entry_thesis
    merged = dict(entry_thesis)
    merged["flags"] = sorted(flags)
    return merged


def _augment_entry_thesis_regime(entry_thesis: Optional[dict], pocket: str) -> Optional[dict]:
    if not _ORDER_THESIS_REGIME_ENABLED or not isinstance(entry_thesis, dict):
        return entry_thesis
    reg = entry_thesis.get("regime")
    macro = None
    micro = None
    if isinstance(reg, dict):
        macro = reg.get("macro") or reg.get("macro_regime") or reg.get("reg_macro")
        micro = reg.get("micro") or reg.get("micro_regime") or reg.get("reg_micro")
    macro = macro or entry_thesis.get("macro_regime") or entry_thesis.get("reg_macro")
    micro = micro or entry_thesis.get("micro_regime") or entry_thesis.get("reg_micro")
    if macro and micro:
        return entry_thesis

    macro_live = macro
    micro_live = micro
    if not macro_live:
        macro_live = current_regime("H4", event_mode=False) or current_regime("H1", event_mode=False)
    if not micro_live:
        micro_live = current_regime("M1", event_mode=False)
    if not macro_live:
        macro_live, _ = get_last_regime("H4")
        if not macro_live:
            macro_live, _ = get_last_regime("H1")
    if not micro_live:
        micro_live, _ = get_last_regime("M1")
    fallback_used = False
    if not macro_live:
        macro_live = "Mixed"
        fallback_used = True
    if not micro_live:
        micro_live = "Mixed"
        fallback_used = True

    merged = dict(entry_thesis)
    reg_dict = reg if isinstance(reg, dict) else {}
    if macro_live:
        reg_dict = dict(reg_dict)
        reg_dict.setdefault("macro", macro_live)
        reg_dict.setdefault("macro_regime", macro_live)
        reg_dict.setdefault("reg_macro", macro_live)
        merged.setdefault("macro_regime", macro_live)
        merged.setdefault("reg_macro", macro_live)
    if micro_live:
        reg_dict = dict(reg_dict)
        reg_dict.setdefault("micro", micro_live)
        reg_dict.setdefault("micro_regime", micro_live)
        reg_dict.setdefault("reg_micro", micro_live)
        merged.setdefault("micro_regime", micro_live)
        merged.setdefault("reg_micro", micro_live)
    if reg_dict:
        merged["regime"] = reg_dict
    if fallback_used:
        merged.setdefault("regime_source", "fallback")
    return merged


def _scaled_thresholds(
    pocket: str,
    base: tuple[float, ...],
    atr_m1: float,
    vol_5m: float,
) -> tuple[float, ...]:
    """
    Scale partial/lock thresholds by short-term volatility/ATR.
    Bound the scale to avoid extreme shrink/expansion.
    """
    scale = 1.0
    if pocket in {"micro", "scalp", "scalp_fast"}:
        if vol_5m < 0.8:
            scale *= 0.9  # 早めに利確
        elif vol_5m > 1.6:
            scale *= 1.2  # 伸ばす
        if atr_m1 > 3.0:
            scale *= 1.15
        elif atr_m1 < 1.2:
            scale *= 0.92
    elif pocket == "macro":
        if vol_5m > 1.6 or atr_m1 > 3.0:
            scale *= 1.12
    scale = max(0.7, min(1.4, scale))
    return tuple(max(1.0, t * scale) for t in base)


def _apply_directional_cap(
    units: int,
    pocket: str,
    side_label: str,
    meta: Optional[dict],
) -> int:
    """
    Dynamically scale units to keep same-direction exposure within a cap derived
    from NAV and pocket ratio. Never rejects outright; scales down to remaining.
    """
    if not _DIR_CAP_ENABLE or units == 0 or pocket is None:
        return units
    try:
        from utils.oanda_account import get_account_snapshot
    except Exception:
        return units
    try:
        snap = get_account_snapshot(cache_ttl_sec=3.0)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[DIR_CAP] snapshot fetch failed: %s", exc)
        return units
    price = _estimate_price(meta) or 0.0
    if price <= 0:
        return units
    if snap.margin_rate <= 0 or snap.nav <= 0:
        return units
    try:
        pocket_ratio = POCKET_MAX_RATIOS.get(pocket, 1.0)
    except Exception:
        pocket_ratio = 1.0
    # cap derived from notional limit (NAV * leverage) and pocket share
    notional_cap_units = (snap.nav / price) * MAX_LEVERAGE * _DIR_CAP_RATIO * pocket_ratio
    if notional_cap_units <= 0:
        return units
    adverse_scale = 1.0
    adverse_pips = 0.0
    adverse_details: dict[str, float] = {}
    if _DIR_CAP_ADVERSE_ENABLE:
        adverse_pips, adverse_details = _dir_cap_adverse_pips(side_label, pocket)
        start = _DIR_CAP_ADVERSE_START_PIPS
        full = _DIR_CAP_ADVERSE_FULL_PIPS
        if full > start and adverse_pips > start:
            frac = min(1.0, max(0.0, (adverse_pips - start) / max(1e-6, full - start)))
            adverse_scale = 1.0 - frac * (1.0 - _DIR_CAP_ADVERSE_MIN_SCALE)
            notional_cap_units *= adverse_scale
            try:
                log_metric(
                    "dir_cap_adverse_scale",
                    adverse_scale,
                    tags={
                        "pocket": pocket,
                        "side": side_label,
                        "adverse_pips": round(adverse_pips, 1),
                        "lookback": int(adverse_details.get("lookback_min") or _DIR_CAP_ADVERSE_LOOKBACK_MIN),
                    },
                )
            except Exception:
                pass
            logging.info(
                "[DIR_CAP] adverse shrink pocket=%s side=%s adverse=%.1f scale=%.2f cap=%.0f",
                pocket,
                side_label,
                adverse_pips,
                adverse_scale,
                notional_cap_units,
            )
    # fetch cached PositionManager to avoid repeated instantiation
    global _DIR_CAP_CACHE
    if _DIR_CAP_CACHE is None:
        _DIR_CAP_CACHE = PositionManager()
    positions = _DIR_CAP_CACHE.get_open_positions()
    info = positions.get(pocket) or {}
    current_same_dir = 0
    try:
        if side_label == "buy":
            current_same_dir = int(info.get("long_units", 0) or 0)
        else:
            current_same_dir = int(info.get("short_units", 0) or 0)
    except Exception:
        current_same_dir = 0
    remaining = max(0.0, notional_cap_units - current_same_dir)
    if remaining <= 0:
        logging.warning(
            "[DIR_CAP] pocket=%s side=%s blocked at cap cap_units=%.0f current=%s",
            pocket,
            side_label,
            notional_cap_units,
            current_same_dir,
        )
        return 0
    target = abs(units)
    if remaining < target:
        scaled = max(remaining, target * _DIR_CAP_MIN_FRACTION)
        new_units = int(max(0, round(scaled)))
        logging.info(
            "[DIR_CAP] scale pocket=%s side=%s units=%d -> %d (current=%s cap=%.0f)",
            pocket,
            side_label,
            units,
            new_units if units > 0 else -new_units,
            current_same_dir,
            notional_cap_units,
        )
        return new_units if units > 0 else -new_units
    # near-cap warning
    if current_same_dir + target >= _DIR_CAP_WARN_RATIO * notional_cap_units:
        logging.warning(
            "[DIR_CAP] near cap pocket=%s side=%s current=%s pending=%d cap=%.0f",
            pocket,
            side_label,
            current_same_dir,
            target,
            notional_cap_units,
        )
    return units


def _fetch_quote(instrument: str) -> dict[str, float] | None:
    """Fetch a single pricing snapshot to derive bid/ask/spread."""
    # Prefer tick_window (shared disk cache) to avoid per-order REST calls.
    if tick_window is not None:
        try:
            max_age = _env_float("ORDER_TICK_QUOTE_MAX_AGE_SEC", 3.0)
            now = time.time()
            ticks = tick_window.recent_ticks(seconds=max(0.5, max_age), limit=1)
            if ticks:
                tick = ticks[-1]
                epoch = _as_float(tick.get("epoch"))
                bid = _as_float(tick.get("bid"))
                ask = _as_float(tick.get("ask"))
                mid = _as_float(tick.get("mid"))
                if bid is not None and ask is not None:
                    if epoch is not None and max_age > 0 and abs(now - float(epoch)) > max_age:
                        # Stale tick (stream stalled) – fall back to REST snapshot.
                        pass
                    else:
                        spread_pips = (ask - bid) / 0.01
                        ts_iso = datetime.fromtimestamp(
                            float(epoch) if epoch is not None else now, tz=timezone.utc
                        ).isoformat()
                        return {
                            "bid": bid,
                            "ask": ask,
                            "mid": mid if mid is not None else (bid + ask) / 2.0,
                            "spread_pips": spread_pips,
                            "ts": ts_iso,
                        }
        except Exception:
            # tick_window is best-effort; ignore and fall back to REST.
            pass
    try:
        base = "https://api-fxpractice.oanda.com" if PRACTICE_FLAG else "https://api-fxtrade.oanda.com"
        url = f"{base}/v3/accounts/{ACCOUNT}/pricing"
        headers = {"Authorization": f"Bearer {TOKEN}"}
        resp = requests.get(
            url,
            params={"instruments": instrument},
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        price = (data.get("prices") or [{}])[0]
        bid = _as_float(price.get("bids", [{}])[0].get("price"))
        ask = _as_float(price.get("asks", [{}])[0].get("price"))
        if bid is None or ask is None:
            return None
        spread_pips = (ask - bid) / 0.01
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "spread_pips": spread_pips,
            "ts": price.get("time") or datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


def _safe_json(payload: Optional[dict]) -> str:
    """
    Serialize payload safely; never raises. None -> "{}".
    Coerces non-serializable objects to string to avoid dropping the payload.
    """
    def _coerce(obj: object):
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            try:
                v = float(obj)
                # SQLite JSON1 expects strict JSON; avoid NaN/Infinity.
                return v if math.isfinite(v) else None
            except Exception:
                return None
        if isinstance(obj, dict):
            return {str(k): _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_coerce(v) for v in obj]
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    if payload is None:
        return "{}"
    try:
        coerced = _coerce(payload)
        return json.dumps(coerced, ensure_ascii=False, allow_nan=False)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[ORDER][LOG] failed to serialize payload: %s", exc)
        return "{}"


def _exit_context_snapshot(exit_reason: Optional[str]) -> Optional[dict]:
    if not _EXIT_CONTEXT_ENABLED:
        return None
    now = datetime.now(timezone.utc).isoformat()
    factors = {}
    try:
        fac = all_factors()
    except Exception:
        fac = {}

    def _snap(tf: str) -> dict:
        data = fac.get(tf) or {}
        return {
            "timestamp": data.get("timestamp"),
            "close": _as_float(data.get("close")),
            "rsi": _as_float(data.get("rsi")),
            "adx": _as_float(data.get("adx")),
            "bbw": _as_float(data.get("bbw")),
            "atr_pips": _as_float(data.get("atr_pips")),
            "ma10": _as_float(data.get("ma10")),
            "ma20": _as_float(data.get("ma20")),
            "vwap_gap": _as_float(data.get("vwap_gap")),
        }

    for tf in ("M1", "M5", "H1", "H4"):
        if fac.get(tf):
            factors[tf] = _snap(tf)

    mid = None
    if tick_window is not None:
        try:
            tick = tick_window.recent_ticks(seconds=2.0, limit=1)
            if tick:
                mid = _as_float(tick[-1].get("mid"))
        except Exception:
            mid = None
    if mid is None:
        try:
            mid = _as_float((fac.get("M1") or {}).get("close"))
        except Exception:
            mid = None

    return {
        "ts": now,
        "reason": exit_reason,
        "mid": mid,
        "factor_age_m1_sec": _factor_age_seconds("M1"),
        "factors": factors,
    }


def _log_order(
    *,
    pocket: Optional[str],
    instrument: Optional[str],
    side: Optional[str],
    units: Optional[int],
    sl_price: Optional[float],
    tp_price: Optional[float],
    client_order_id: Optional[str],
    status: str,
    attempt: int,
    stage_index: Optional[int] = None,
    ticket_id: Optional[str] = None,
    executed_price: Optional[float] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    request_payload: Optional[dict] = None,
    response_payload: Optional[dict] = None,
    fast_fail: bool = False,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    _cache_order_status(
        ts=ts,
        client_order_id=client_order_id,
        status=status,
        attempt=attempt,
        side=side,
        units=units,
        error_code=error_code,
        error_message=error_message,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    row_values = (
        ts,
        pocket,
        instrument,
        side,
        int(units) if units is not None else None,
        float(sl_price) if sl_price is not None else None,
        float(tp_price) if tp_price is not None else None,
        client_order_id,
        status,
        int(attempt),
        int(stage_index) if stage_index is not None else None,
        ticket_id,
        float(executed_price) if executed_price is not None else None,
        error_code,
        error_message,
        _safe_json(request_payload),
        _safe_json(response_payload),
    )

    retry_attempts = (
        _ORDER_DB_LOG_FAST_RETRY_ATTEMPTS
        if fast_fail
        else _ORDER_DB_LOG_RETRY_ATTEMPTS
    )
    retry_sleep_sec = (
        _ORDER_DB_LOG_FAST_RETRY_SLEEP_SEC
        if fast_fail
        else _ORDER_DB_LOG_RETRY_SLEEP_SEC
    )
    retry_backoff = (
        _ORDER_DB_LOG_FAST_RETRY_BACKOFF
        if fast_fail
        else _ORDER_DB_LOG_RETRY_BACKOFF
    )
    retry_max_sleep_sec = (
        _ORDER_DB_LOG_FAST_RETRY_MAX_SLEEP_SEC
        if fast_fail
        else _ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC
    )

    sleep_sec = retry_sleep_sec
    for attempt_idx in range(retry_attempts):
        con: Optional[sqlite3.Connection] = None
        try:
            # Serialize in-process writes first, then hold a short cross-process lock.
            # This avoids self-contention on flock() when many async requests log at once.
            with _ORDER_DB_LOCAL_WRITE_LOCK:
                with _order_db_file_lock(fast_fail=fast_fail):
                    con = _orders_con()
                    con.execute(
                        """
                        INSERT INTO orders (
                          ts, pocket, instrument, side, units, sl_price, tp_price,
                          client_order_id, status, attempt, stage_index, ticket_id, executed_price,
                          error_code, error_message, request_json, response_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row_values,
                    )
                    con.commit()
                    _maybe_checkpoint_orders_db(con)
            return
        except sqlite3.OperationalError as exc:
            if (
                _is_sqlite_locked_error(exc)
                and attempt_idx + 1 < retry_attempts
            ):
                _rollback_orders_con(con)
                _reset_orders_con()
                delay = min(sleep_sec, retry_max_sleep_sec)
                if delay > 0:
                    time.sleep(delay)
                sleep_sec = min(
                    retry_max_sleep_sec,
                    sleep_sec * retry_backoff,
                )
                continue
            _rollback_orders_con(con)
            if _is_sqlite_locked_error(exc):
                _reset_orders_con()
            if fast_fail and _is_sqlite_locked_error(exc):
                logging.info("[ORDER][LOG] fast-fail dropped by lock: %s", exc)
            else:
                logging.warning("[ORDER][LOG] failed to persist orders log: %s", exc)
            return
        except Exception as exc:  # noqa: BLE001
            _rollback_orders_con(con)
            _reset_orders_con()
            logging.warning("[ORDER][LOG] failed to persist orders log: %s", exc)
            return


def get_last_order_status_by_client_id(
    client_order_id: Optional[str],
) -> Optional[dict[str, object]]:
    """
    Fetch the latest orders.db entry for a client order id.
    Returns a compact reason payload for upper-layer diagnostics.
    """
    if not client_order_id:
        return None
    try:
        con = _orders_con()
    except Exception:
        return _cached_order_status(client_order_id)
    try:
        row = con.execute(
            """
            SELECT
              ts,
              status,
              attempt,
              side,
              units,
              error_code,
              error_message,
              request_json,
              response_json
            FROM orders
            WHERE client_order_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (str(client_order_id),),
        ).fetchone()
    except Exception:
        return _cached_order_status(client_order_id)
    if not row:
        return _cached_order_status(client_order_id)
    try:
        ts, status, attempt, side, units, error_code, error_message, request_json, response_json = row
        req = None
        res = None
        if request_json:
            try:
                req = json.loads(request_json)
            except Exception:
                req = None
        if response_json:
            try:
                res = json.loads(response_json)
            except Exception:
                res = None
        return {
            "ts": ts,
            "status": status,
            "attempt": attempt,
            "side": side,
            "units": units,
            "error_code": error_code,
            "error_message": error_message,
            "request": req,
            "response": res,
        }
    except Exception:
        return _cached_order_status(client_order_id)


def _console_order_log(
    event: str,
    *,
    pocket: Optional[str],
    strategy_tag: Optional[str],
    side: Optional[str],
    units: Optional[int],
    sl_price: Optional[float],
    tp_price: Optional[float],
    client_order_id: Optional[str],
    ticket_id: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    logging.info(
        "[ORDER][%s] pocket=%s strategy=%s side=%s units=%s sl=%s tp=%s client=%s ticket=%s note=%s",
        event,
        pocket or "-",
        strategy_tag or "-",
        side or "-",
        units or "-",
        _fmt(sl_price),
        _fmt(tp_price),
        client_order_id or "-",
        ticket_id or "-",
        note or "-",
    )


def _normalize_protections(
    estimated_price: Optional[float],
    sl_price: Optional[float],
    tp_price: Optional[float],
    is_buy: bool,
) -> tuple[Optional[float], Optional[float], bool]:
    """Ensure SL/TP are on the correct side of the entry with a minimal buffer."""

    if estimated_price is None:
        return sl_price, tp_price, False
    changed = False
    price = float(estimated_price)
    buffer = max(_PROTECTION_MIN_BUFFER, 0.0005)
    separation = max(_PROTECTION_MIN_SEPARATION, buffer * 2)
    if sl_price is not None:
        if is_buy and sl_price >= price - buffer:
            sl_price = round(price - buffer, 3)
            changed = True
        elif (not is_buy) and sl_price <= price + buffer:
            sl_price = round(price + buffer, 3)
            changed = True
    if tp_price is not None:
        if is_buy and tp_price <= price + buffer:
            tp_price = round(price + buffer, 3)
            changed = True
        elif (not is_buy) and tp_price >= price - buffer:
            tp_price = round(price - buffer, 3)
            changed = True
    if sl_price is not None and tp_price is not None:
        gap = tp_price - sl_price if is_buy else sl_price - tp_price
        if gap < separation:
            sl_delta = separation / 2.0
            if is_buy:
                sl_price = round(price - sl_delta, 3)
                tp_price = round(price + sl_delta, 3)
            else:
                sl_price = round(price + sl_delta, 3)
                tp_price = round(price - sl_delta, 3)
            changed = True
    return sl_price, tp_price, changed


def _fallback_protections(
    baseline_price: Optional[float],
    *,
    is_buy: bool,
    has_sl: bool,
    has_tp: bool,
    reason_key: Optional[str],
    sl_gap_pips: Optional[float],
    tp_gap_pips: Optional[float],
    fallback_gap_price: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return retry SL/TP values after protection rejects."""

    if baseline_price is None:
        return None, None
    gap_fallback = max(0.0005, float(fallback_gap_price))
    gap_sl = gap_fallback
    gap_tp = gap_fallback
    try:
        parsed_sl = float(sl_gap_pips) if sl_gap_pips is not None else None
    except (TypeError, ValueError):
        parsed_sl = None
    try:
        parsed_tp = float(tp_gap_pips) if tp_gap_pips is not None else None
    except (TypeError, ValueError):
        parsed_tp = None
    if parsed_sl is not None and parsed_sl > 0.0:
        gap_sl = max(0.0005, parsed_sl * 0.01)
    if parsed_tp is not None and parsed_tp > 0.0:
        gap_tp = max(0.0005, parsed_tp * 0.01)
    if is_buy:
        sl_price = round(baseline_price - gap_sl, 3) if has_sl else None
        tp_price = round(baseline_price + gap_tp, 3) if has_tp else None
    else:
        sl_price = round(baseline_price + gap_sl, 3) if has_sl else None
        tp_price = round(baseline_price - gap_tp, 3) if has_tp else None

    needs_sl = False
    needs_tp = False
    reason_upper = str(reason_key or "").upper()
    if has_sl:
        needs_sl = "STOP_LOSS" in reason_upper or not reason_upper
    if has_tp:
        needs_tp = "TAKE_PROFIT" in reason_upper or not reason_upper
    if (has_sl or has_tp) and not (needs_sl or needs_tp):
        needs_sl = bool(has_sl)
        needs_tp = bool(has_tp)

    sl_price, tp_price, _ = _normalize_protections(
        baseline_price,
        sl_price,
        tp_price,
        is_buy,
    )
    if has_sl and needs_sl and sl_price is not None:
        distance = abs(float(baseline_price) - float(sl_price))
        if distance + 1e-9 < gap_fallback:
            sl_price = round(
                float(baseline_price) - gap_fallback if is_buy else float(baseline_price) + gap_fallback,
                3,
            )
    if has_tp and needs_tp and tp_price is not None:
        distance = abs(float(tp_price) - float(baseline_price))
        if distance + 1e-9 < gap_fallback:
            tp_price = round(
                float(baseline_price) + gap_fallback if is_buy else float(baseline_price) - gap_fallback,
                3,
            )
    return sl_price, tp_price


def _derive_fallback_basis(
    estimated_price: Optional[float],
    sl_price: Optional[float],
    tp_price: Optional[float],
    is_buy: bool,
    *,
    fallback_gap_price: float,
) -> Optional[float]:
    if estimated_price is not None:
        return estimated_price
    if sl_price is not None and tp_price is not None:
        return float((sl_price + tp_price) / 2.0)
    gap = max(0.0005, float(fallback_gap_price))
    if sl_price is not None:
        return float(sl_price + (gap if is_buy else -gap))
    if tp_price is not None:
        return float(tp_price - (gap if is_buy else -gap))
    return None


def _current_trade_units(trade_id: str) -> Optional[int]:
    try:
        req = TradeDetails(accountID=ACCOUNT, tradeID=trade_id)
        api.request(req)
        trade = req.response.get("trade") or {}
        units_raw = trade.get("currentUnits")
        if units_raw is None:
            return None
        return abs(int(float(units_raw)))
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Failed to fetch trade units trade=%s err=%s", trade_id, exc)
        return None


def _current_trade_unrealized_pl(trade_id: str) -> Optional[float]:
    try:
        req = TradeDetails(accountID=ACCOUNT, tradeID=trade_id)
        api.request(req)
        trade = req.response.get("trade") or {}
        pl = trade.get("unrealizedPL")
        if pl is None:
            return None
        return float(pl)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Failed to fetch unrealized PL trade=%s err=%s", trade_id, exc)
        return None


def _should_allow_negative_close(client_order_id: Optional[str]) -> bool:
    if not _EXIT_EMERGENCY_ALLOW_NEGATIVE:
        return False
    if not client_order_id or not client_order_id.startswith(agent_client_prefixes):
        return False
    try:
        snapshot = get_account_snapshot(cache_ttl_sec=_EXIT_EMERGENCY_CACHE_TTL_SEC)
    except Exception as exc:  # noqa: BLE001
        logging.debug("[ORDER] emergency health check failed: %s", exc)
        return False
    hb = snapshot.health_buffer
    free_ratio = snapshot.free_margin_ratio
    nav = snapshot.nav or 0.0
    margin_used = snapshot.margin_used
    unrealized = snapshot.unrealized_pl
    allow = False
    reasons = {}
    if hb is not None and _EXIT_EMERGENCY_HEALTH_BUFFER > 0 and hb <= _EXIT_EMERGENCY_HEALTH_BUFFER:
        allow = True
        reasons["health_buffer"] = hb
    if (
        free_ratio is not None
        and _EXIT_EMERGENCY_FREE_MARGIN_RATIO > 0
        and free_ratio <= _EXIT_EMERGENCY_FREE_MARGIN_RATIO
    ):
        allow = True
        reasons["free_margin_ratio"] = free_ratio
    if nav > 0 and _EXIT_EMERGENCY_MARGIN_USAGE_RATIO > 0:
        usage_ratio = (margin_used or 0.0) / nav
        if usage_ratio >= _EXIT_EMERGENCY_MARGIN_USAGE_RATIO:
            allow = True
            reasons["margin_usage_ratio"] = usage_ratio
    if nav > 0 and _EXIT_EMERGENCY_UNREALIZED_DD_RATIO > 0 and unrealized < 0:
        dd_ratio = abs(unrealized) / nav
        if dd_ratio >= _EXIT_EMERGENCY_UNREALIZED_DD_RATIO:
            allow = True
            reasons["unrealized_dd_ratio"] = dd_ratio
    if allow:
        global _LAST_EMERGENCY_LOG_TS
        now = time.time()
        if now - _LAST_EMERGENCY_LOG_TS >= 30.0:
            log_metric(
                "close_emergency_allow_negative",
                float(hb) if hb is not None else -1.0,
                tags={
                    "threshold": _EXIT_EMERGENCY_HEALTH_BUFFER,
                    "reason": next(iter(reasons.keys()), "unknown"),
                },
            )
            _LAST_EMERGENCY_LOG_TS = now
        return True
    return False


def _reason_allows_negative(exit_reason: Optional[str]) -> bool:
    if not exit_reason:
        return False
    reason_key = str(exit_reason).strip().lower()
    if not reason_key:
        return False
    for token in _EXIT_ALLOW_NEGATIVE_REASONS:
        if token.endswith("*"):
            if reason_key.startswith(token[:-1]):
                return True
        elif reason_key == token:
            return True
    return False


def _reason_force_allow(exit_reason: Optional[str]) -> bool:
    if not exit_reason:
        return False
    reason_key = str(exit_reason).strip().lower()
    if not reason_key:
        return False
    for token in _EXIT_FORCE_ALLOW_REASONS:
        if token.endswith("*"):
            if reason_key.startswith(token[:-1]):
                return True
        elif reason_key == token:
            return True
    return False


def _neg_exit_decision(
    *,
    exit_reason: Optional[str],
    est_pips: Optional[float],
    emergency_allow: bool,
    reason_allow: bool,
    worker_allow: bool,
    neg_policy: Optional[dict],
) -> tuple[bool, bool]:
    """Return (neg_allowed, near_be_allow) for negative close policy.

    Behavior:
    - Emergency/near-BE allowances are always honored.
    - Strategy allow_reasons are treated as explicit permissions, even when
      global EXIT_ALLOW_NEGATIVE_REASONS does not include the reason.
    - If no strategy allow_reasons are defined, global/worker gates are used.
    - strict_no_negative=true blocks all negative closes for that strategy.
    """
    neg_policy = neg_policy if isinstance(neg_policy, dict) else {}
    if _coerce_bool(neg_policy.get("strict_no_negative"), False):
        strict_allow_tokens = neg_policy.get("strict_allow_reasons")
        strict_allow = False
        if strict_allow_tokens is not None:
            if isinstance(strict_allow_tokens, (list, tuple, set)):
                strict_list = [str(token) for token in strict_allow_tokens]
            else:
                strict_list = [str(strict_allow_tokens)]
            strict_allow = _reason_matches_tokens(exit_reason, strict_list)
        if not strict_allow:
            return False, False
    policy_enabled = _coerce_bool(neg_policy.get("enabled"), True)
    allow_tokens = neg_policy.get("allow_reasons")
    deny_tokens = neg_policy.get("deny_reasons")

    policy_reason_match = False
    if allow_tokens is None:
        policy_allow = policy_enabled
    else:
        if isinstance(allow_tokens, (list, tuple, set)):
            allow_list = [str(token) for token in allow_tokens]
        else:
            allow_list = [str(allow_tokens)]
        policy_reason_match = _reason_matches_tokens(exit_reason, allow_list)
        policy_allow = policy_enabled and policy_reason_match

    if deny_tokens is not None:
        if isinstance(deny_tokens, (list, tuple, set)):
            deny_list = [str(token) for token in deny_tokens]
        else:
            deny_list = [str(deny_tokens)]
        if _reason_matches_tokens(exit_reason, deny_list):
            policy_allow = False
            policy_reason_match = False

    near_be_allow = policy_allow and _allow_negative_near_be(exit_reason, est_pips)
    policy_explicit_allow = policy_allow and policy_reason_match
    neg_allowed = bool(
        emergency_allow
        or near_be_allow
        or (policy_allow and (reason_allow or worker_allow or policy_explicit_allow))
    )
    return neg_allowed, near_be_allow


def _latest_exit_price() -> Optional[float]:
    price = _latest_mid_price()
    if price is not None:
        return price
    try:
        return _as_float((all_factors().get("M1") or {}).get("close"))
    except Exception:
        return None


def _estimate_trade_pnl_pips(
    *,
    entry_price: float,
    units: int,
    bid: Optional[float],
    ask: Optional[float],
    mid: Optional[float],
) -> Optional[float]:
    if entry_price <= 0 or units == 0:
        return None
    if units > 0:
        price = bid if bid is not None else mid
        if price is None:
            return None
        return (price - entry_price) / 0.01
    price = ask if ask is not None else mid
    if price is None:
        return None
    return (entry_price - price) / 0.01


def _load_exit_trade_context(
    trade_id: str,
    client_order_id: Optional[str],
) -> Optional[dict]:
    try:
        con = _orders_con()
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT pocket, units, executed_price, client_order_id, instrument, tp_price
            FROM orders
            WHERE ticket_id = ?
              AND status = 'filled'
            ORDER BY id ASC
            LIMIT 1
            """,
            (trade_id,),
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logging.debug("[ORDER] exit context lookup failed trade=%s err=%s", trade_id, exc)
        return None
    if not row:
        return None
    client_id = row["client_order_id"] or client_order_id
    entry_thesis = None
    strategy_tag = None
    if client_id:
        try:
            att = con.execute(
                """
                SELECT request_json
                FROM orders
                WHERE client_order_id = ?
                  AND status = 'submit_attempt'
                ORDER BY id ASC
                LIMIT 1
                """,
                (client_id,),
            ).fetchone()
            if att and att["request_json"]:
                payload = json.loads(att["request_json"]) or {}
                entry_thesis = _coerce_entry_thesis(
                    payload.get("entry_thesis") or (payload.get("meta") or {}).get("entry_thesis")
                )
                if entry_thesis:
                    strategy_tag = _strategy_tag_from_thesis(entry_thesis)
        except Exception:
            entry_thesis = None
            strategy_tag = None
    if not strategy_tag and client_id and client_id.startswith("qr-"):
        parts = client_id.split("-", 3)
        if len(parts) == 4 and parts[3]:
            strategy_tag = parts[3]
    tp_price = _as_float(row["tp_price"])
    if tp_price is None:
        cached = _LAST_PROTECTIONS.get(str(trade_id))
        if isinstance(cached, dict):
            tp_price = _as_float(cached.get("tp"))
    return {
        "entry_price": _as_float(row["executed_price"]) or 0.0,
        "units": int(row["units"] or 0),
        "pocket": row["pocket"] or "unknown",
        "instrument": row["instrument"] if row["instrument"] else None,
        "tp_price": tp_price,
        "strategy_tag": strategy_tag,
        "entry_thesis": entry_thesis if isinstance(entry_thesis, dict) else None,
    }


def _retry_close_with_actual_units(
    trade_id: str,
    requested_units: Optional[int],
) -> bool:
    actual_units = _current_trade_units(trade_id)
    if actual_units is None or actual_units <= 0:
        logging.info(
            "[ORDER] Trade %s already flat or missing when retrying close; treat as success.",
            trade_id,
        )
        return True
    target_units = actual_units
    if requested_units is not None:
        try:
            target_units = min(actual_units, abs(int(requested_units)))
        except Exception:
            target_units = actual_units
    if target_units <= 0:
        return True
    data = {"units": str(target_units)}
    try:
        req = TradeClose(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        response = req.response if isinstance(getattr(req, "response", None), dict) else {}
        reject = response.get("orderRejectTransaction") or response.get("orderCancelTransaction")
        if reject:
            reason = reject.get("rejectReason") or reject.get("reason")
            reason_key = str(reason or "").upper() or "rejected"
            _log_order(
                pocket=None,
                instrument=None,
                side=None,
                units=target_units,
                sl_price=None,
                tp_price=None,
                client_order_id=None,
                status="close_retry_failed",
                attempt=2,
                ticket_id=str(trade_id),
                executed_price=None,
                error_code=str(reject.get("errorCode") or reason_key),
                error_message=reject.get("errorMessage") or str(reason_key),
                response_payload=response,
                request_payload={"retry": True, "data": data, "reason": reason_key},
            )
            _console_order_log(
                "CLOSE_FAIL",
                pocket=None,
                strategy_tag=None,
                side=None,
                units=target_units,
                sl_price=None,
                tp_price=None,
                client_order_id=None,
                ticket_id=str(trade_id),
                note=f"retry:{reason_key}",
            )
            return False
        if not (response.get("orderFillTransaction") or {}):
            _log_order(
                pocket=None,
                instrument=None,
                side=None,
                units=target_units,
                sl_price=None,
                tp_price=None,
                client_order_id=None,
                status="close_retry_failed",
                attempt=2,
                ticket_id=str(trade_id),
                executed_price=None,
                error_code="missing_fill",
                error_message="missing orderFillTransaction in TradeClose retry response",
                response_payload=response,
                request_payload={"retry": True, "data": data},
            )
            return False
        _LAST_PROTECTIONS.pop(trade_id, None)
        _PARTIAL_STAGE.pop(trade_id, None)
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=target_units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_retry_ok",
            attempt=2,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload={"retry": True, "data": data},
        )
        _console_order_log(
            "CLOSE_OK",
            pocket=None,
            strategy_tag=None,
            side=None,
            units=target_units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            ticket_id=str(trade_id),
            note="retry",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Retry close failed trade=%s units=%s err=%s",
            trade_id,
            target_units,
            exc,
        )
        return False

def _estimate_entry_price(
    *, units: int, sl_price: Optional[float], tp_price: Optional[float], meta: Optional[dict]
) -> Optional[float]:
    """Estimate entry price for margin preflight.

    Prefer caller-provided meta['entry_price']. As a fallback, try a naive
    inference from SL/TP if片方のみ与えられている場合は精度が落ちるため None を返す。
    """
    try:
        if meta and isinstance(meta, dict) and meta.get("entry_price"):
            return float(meta["entry_price"])
    except Exception:
        pass
    # Heuristic fallback (avoid guessing if insufficient data)
    if sl_price is not None and tp_price is not None:
        # Mid between SL and TP as a last resort
        return float((sl_price + tp_price) / 2.0)
    return None


def _preflight_units(
    *,
    estimated_price: float,
    requested_units: int,
    margin_buffer: float = 0.92,
) -> Tuple[int, float]:
    """Return (allowed_units, required_margin_estimate).

    allowed_units may be 0 if margin is insufficient even for minimum size.
    Netting-aware: if the requested units shrink current net exposure, allow them
    even when free margin is low.
    """
    try:
        from utils.oanda_account import (  # lazy import
            get_account_snapshot,
            get_position_summary,
        )

        snap = get_account_snapshot()
        margin_avail = float(getattr(snap, "margin_available", 0.0) or 0.0)
        margin_used = float(getattr(snap, "margin_used", 0.0) or 0.0)
        margin_rate = float(getattr(snap, "margin_rate", 0.0) or 0.0)
    except Exception as exc:
        logging.warning("[ORDER] preflight snapshot failed: %s", exc)
        try:
            log_metric(
                "order_margin_block",
                1.0,
                tags={"reason": "preflight_snapshot_failed"},
            )
        except Exception:
            pass
        return (0, 0.0)

    if margin_rate <= 0.0 or estimated_price <= 0.0:
        logging.warning(
            "[ORDER] preflight missing margin data rate=%.4f price=%.4f",
            margin_rate,
            estimated_price,
        )
        try:
            log_metric(
                "order_margin_block",
                1.0,
                tags={"reason": "preflight_missing_margin_data"},
            )
        except Exception:
            pass
        return (0, 0.0)

    per_unit_margin = estimated_price * margin_rate
    if per_unit_margin <= 0.0:
        return (0, 0.0)

    long_units: float | None = None
    short_units: float | None = None
    try:
        long_units, short_units = get_position_summary(timeout=3.0)
        long_units = max(0.0, float(long_units or 0.0))
        short_units = max(0.0, float(short_units or 0.0))
    except Exception as exc:
        logging.debug("[ORDER] preflight position summary unavailable: %s", exc)
        long_units = short_units = None

    if long_units is not None and short_units is not None:
        net_before = long_units - short_units
        net_after = net_before + requested_units
        margin_after = abs(net_after) * per_unit_margin
        # If the order reduces or keeps net margin, allow it regardless of free margin.
        if margin_after <= margin_used * 1.0005:
            return (requested_units, margin_after)

        budget = margin_used + margin_avail * margin_buffer
        if margin_after <= budget:
            return (requested_units, margin_after)

        # Scale down to stay within budget while preserving direction.
        target_net = budget / per_unit_margin
        clamped_net_after = max(-target_net, min(target_net, net_after))
        allowed_units = clamped_net_after - net_before
        if requested_units > 0:
            allowed_units = min(max(0.0, allowed_units), float(requested_units))
        else:
            allowed_units = -min(max(0.0, -allowed_units), float(abs(requested_units)))
        allowed_units_int = int(round(allowed_units))
        margin_allowed = abs(net_before + allowed_units_int) * per_unit_margin
        return (allowed_units_int, margin_allowed)

    # Fallback: use free margin only (no position breakdown available)
    req = abs(requested_units) * per_unit_margin
    if req * 1.0 <= margin_avail * margin_buffer:
        return (requested_units, req)

    max_units = int((margin_avail * margin_buffer) / per_unit_margin)
    if max_units <= 0:
        return (0, req)
    # Keep sign
    allowed = max_units if requested_units > 0 else -max_units
    # Round to nearest 100 units (OANDA accepts 1, but we prefer coarse steps)
    if allowed > 0:
        allowed = int((allowed // 100) * 100)
    else:
        allowed = -int((abs(allowed) // 100) * 100)
    return (allowed, req)


def _projected_usage_with_netting(
    nav: float,
    margin_rate: float,
    side_label: str,
    units: int,
    margin_used: float | None = None,
    meta: Optional[dict] = None,
) -> Optional[float]:
    """
    Estimate margin usage after applying `units`, accounting for netting.
    Returns None if estimation is not possible.
    """
    if nav <= 0 or margin_rate <= 0 or units == 0:
        return None
    try:
        from utils.oanda_account import get_position_summary

        long_u, short_u = get_position_summary()
    except Exception:
        return None

    price_hint = _estimate_price(meta) or 0.0
    net_before = float(long_u) - float(short_u)
    # If price is missing, infer from current margin usage as a last resort.
    if price_hint <= 0 and margin_used and net_before:
        try:
            price_hint = (margin_used / abs(net_before)) / margin_rate
        except Exception:
            price_hint = 0.0
    if price_hint <= 0:
        return None

    new_long = float(long_u)
    new_short = float(short_u)
    if side_label.lower() == "buy":
        new_long += abs(units)
    else:
        new_short += abs(units)
    projected_net_units = abs(new_long - new_short)
    projected_used = projected_net_units * price_hint * margin_rate
    return projected_used / nav


def _is_passive_price(
    *,
    units: int,
    price: float,
    current_bid: Optional[float],
    current_ask: Optional[float],
    min_buffer: float = 0.0001,
) -> bool:
    if units == 0:
        return False
    if units > 0:
        if current_ask is None:
            return False
        return price <= (current_ask - min_buffer)
    if current_bid is None:
        return False
    return price >= (current_bid + min_buffer)
_PARTIAL_STAGE: dict[str, int] = {}

_PARTIAL_THRESHOLDS = {
    # トレンド場面（通常時）の段階利確トリガー（pip）
    # 直近の実測では micro/scalp は数 pip の含み益が頻出する一方、macro は伸びが限定的。
    # 小さめのしきい値で部分利確を優先し、ランナーは trailing に委ねる。
    "macro": (5.0, 10.0),
    "micro": (2.0, 4.2),
    # 早めに小利を拾いつつ、2段目は少し先にしてランナーを残す
    "scalp": (1.6, 3.6),
    # 超短期（fast scalp）は利幅が小さいため閾値も縮小
    # まずは小刻みにヘッジしてランナーのみを残す方針
    "scalp_fast": (1.0, 1.8),
}
_PARTIAL_THRESHOLDS_RANGE = {
    # AGENT.me 仕様（3.5.1）に合わせ、レンジ時は段階利確を引き延ばしすぎず早期ヘッジ。
    # macro 16/22, micro 10/16, scalp 6/10 pips
    "macro": (16.0, 22.0),
    "micro": (10.0, 16.0),
    "scalp": (6.0, 10.0),
    # fast scalp はさらに近い利確で早めに在庫を薄くする
    "scalp_fast": (0.9, 1.4),
}
_PARTIAL_FRACTIONS = (0.4, 0.3)
# micro の平均建玉（~160u）でも段階利確が動作するよう下限を緩和
_PARTIAL_MIN_UNITS = 20
_PARTIAL_RANGE_MACRO_MIN_AGE_MIN = 6.0

# Rate limit market-closed close attempts. When the market is halted (weekend/daily halt),
# OANDA returns HTTP 200 with an orderCancelTransaction (reason=MARKET_HALTED). Treat that
# as a failed close and back off to avoid flooding the broker / orders.db.
_CLOSE_MARKET_CLOSED_LOG_TS: dict[str, float] = {}
_CLOSE_MARKET_CLOSED_LOG_INTERVAL_SEC = 900.0
_CLOSE_MARKET_CLOSED_BACKOFF_SEC = 60.0


def _extract_trade_id(response: dict) -> Optional[str]:
    """
    OANDA の orderFillTransaction から tradeID を抽出する。
    tradeOpened が無い場合でも tradeReduced / tradesClosed を拾って決済扱いにする。
    """
    fill = response.get("orderFillTransaction") or {}

    opened = fill.get("tradeOpened")
    if opened and opened.get("tradeID"):
        return str(opened["tradeID"])

    reduced = fill.get("tradeReduced")
    if reduced and reduced.get("tradeID"):
        return str(reduced["tradeID"])

    for closed in fill.get("tradesClosed") or []:
        trade_id = closed.get("tradeID")
        if trade_id:
            return str(trade_id)

    # tradeReduced が複数のケース（現状 API 仕様では単一辞書）に備えて念のため走査
    for reduced_item in fill.get("tradesReduced") or []:
        trade_id = reduced_item.get("tradeID")
        if trade_id:
            return str(trade_id)

    return None


def _cancel_order_sync(
    *,
    order_id: str,
    pocket: Optional[str] = None,
    client_order_id: Optional[str] = None,
    reason: str = "user_cancel",
) -> bool:
    service_result = _order_manager_service_request(
        "/order/cancel_order",
        {
            "order_id": order_id,
            "pocket": pocket,
            "client_order_id": client_order_id,
            "reason": reason,
        },
    )
    if service_result is not None:
        return bool(service_result)

    try:
        endpoint = OrderCancel(accountID=ACCOUNT, orderID=order_id)
        api.request(endpoint)
        _log_order(
            pocket=pocket,
            instrument=None,
            side=None,
            units=None,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status=reason,
            attempt=0,
            request_payload={"order_id": order_id},
            response_payload=endpoint.response,
        )
        logging.info("[ORDER] Cancelled order %s (%s).", order_id, reason)
        return True
    except V20Error as exc:
        logging.warning("[ORDER] Cancel failed for %s: %s", order_id, exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Cancel exception for %s: %s", order_id, exc)
    return False


async def cancel_order(
    *,
    order_id: str,
    pocket: Optional[str] = None,
    client_order_id: Optional[str] = None,
    reason: str = "user_cancel",
) -> bool:
    service_result = await _order_manager_service_request_async(
        "/order/cancel_order",
        {
            "order_id": order_id,
            "pocket": pocket,
            "client_order_id": client_order_id,
            "reason": reason,
        },
    )
    if service_result is not None:
        return bool(service_result)

    return _cancel_order_sync(
        order_id=order_id,
        pocket=pocket,
        client_order_id=client_order_id,
        reason=reason,
    )


def _parse_trade_open_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    candidate = value.strip()
    try:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        if "." in candidate:
            head, frac = candidate.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            if len(frac_digits) > 6:
                frac_digits = frac_digits[:6]
            tz_part = ""
            if "+" in candidate:
                tz_part = candidate[candidate.rfind("+") :]
            if not tz_part:
                tz_part = "+00:00"
            candidate = f"{head}.{frac_digits}{tz_part}"
        elif "+" not in candidate:
            candidate = f"{candidate}+00:00"
        dt = datetime.fromisoformat(candidate)
        return dt.astimezone(timezone.utc)
    except ValueError:
        try:
            trimmed = candidate.split(".", 1)[0]
            if not trimmed.endswith("+00:00"):
                trimmed = trimmed.rstrip("Z") + "+00:00"
            dt = datetime.fromisoformat(trimmed)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None


def _rollover_sl_strip_context(now_utc: Optional[datetime] = None) -> dict[str, Any]:
    if not _ROLLOVER_SL_STRIP_ENABLED:
        return {"active": False}
    now_utc_val = now_utc or datetime.now(timezone.utc)
    now_jst = now_utc_val.astimezone(_JST)
    cutoff_jst = now_jst.replace(
        hour=_ROLLOVER_SL_STRIP_JST_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    end_jst = cutoff_jst + timedelta(minutes=float(_ROLLOVER_SL_STRIP_WINDOW_MIN))
    active = cutoff_jst <= now_jst < end_jst
    return {
        "active": bool(active),
        "now_utc": now_utc_val,
        "now_jst": now_jst,
        "cutoff_jst": cutoff_jst,
    }


def _trade_matches_rollover_sl_strip(
    trade: dict,
    pocket: str,
    ctx: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(ctx, dict) or not bool(ctx.get("active")):
        return False
    if not isinstance(trade, dict):
        return False
    if pocket == "__net__":
        return False
    client_id = str(trade.get("client_id") or "")
    if not _ROLLOVER_SL_STRIP_INCLUDE_MANUAL and not client_id.startswith("qr-"):
        return False
    if not _ROLLOVER_SL_STRIP_REQUIRE_CARRYOVER:
        return True
    opened_at = _parse_trade_open_time(trade.get("open_time"))
    if opened_at is None:
        return False
    cutoff_jst = ctx.get("cutoff_jst")
    if not isinstance(cutoff_jst, datetime):
        return False
    return opened_at.astimezone(_JST) < cutoff_jst


def _strip_rollover_stop_losses(open_positions: dict, ctx: Optional[dict[str, Any]]) -> int:
    if not isinstance(open_positions, dict):
        return 0
    if not isinstance(ctx, dict) or not bool(ctx.get("active")):
        return 0
    now_ts = time.time()
    actions = 0
    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        trades = info.get("open_trades") if isinstance(info, dict) else None
        if not isinstance(trades, list):
            continue
        for tr in trades:
            if actions >= _ROLLOVER_SL_STRIP_MAX_ACTIONS:
                return actions
            if not _trade_matches_rollover_sl_strip(tr, str(pocket), ctx):
                continue
            trade_id = str((tr or {}).get("trade_id") or "")
            if not trade_id:
                continue
            last_ts = float(_LAST_ROLLOVER_SL_STRIP_TS.get(trade_id) or 0.0)
            if (now_ts - last_ts) < _ROLLOVER_SL_STRIP_COOLDOWN_SEC:
                continue
            sl_info = tr.get("stop_loss") if isinstance(tr, dict) else None
            if not isinstance(sl_info, dict):
                continue
            order_id = str(sl_info.get("order_id") or "").strip()
            if not order_id:
                continue
            client_id = str(tr.get("client_id") or "")
            ok = _cancel_order_sync(
                order_id=order_id,
                pocket=str(pocket),
                client_order_id=client_id or None,
                reason="rollover_sl_strip",
            )
            _LAST_ROLLOVER_SL_STRIP_TS[trade_id] = now_ts
            if ok:
                actions += 1
                _LAST_PROTECTIONS.pop(trade_id, None)
                logging.warning(
                    "[PROTECT] rollover SL stripped trade=%s pocket=%s client=%s order=%s",
                    trade_id,
                    pocket,
                    client_id or "-",
                    order_id,
                )
    return actions


def _coerce_entry_thesis(meta: Any) -> dict:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _soft_tp_mode(thesis: Optional[dict]) -> bool:
    if not isinstance(thesis, dict):
        return False
    mode = thesis.get("tp_mode")
    if not mode and isinstance(thesis.get("execution"), dict):
        mode = thesis.get("execution", {}).get("tp_mode")
    return str(mode or "").lower() in {"soft_zone", "soft"}


def _encode_thesis_comment(entry_thesis: Optional[dict]) -> Optional[str]:
    """
    Serialize a minimal subset of the thesis into OANDA clientExtensions.comment
    so exit側がリスタート後も fast_cut/kill メタを参照できるようにする。
    """
    if _DISABLE_CLIENT_COMMENT:
        return None
    if not isinstance(entry_thesis, dict):
        return None
    keys = (
        "strategy_tag",
        "profile",
        "tag",
        "fast_cut_pips",
        "fast_cut_time_sec",
        "fast_cut_hard_mult",
        "kill_switch",
        "loss_guard_pips",
        "min_hold_sec",
        "target_tp_pips",
    )
    compact: dict[str, object] = {}
    for key in keys:
        val = entry_thesis.get(key)
        if val in (None, "", False):
            continue
        compact[key] = val
    if not compact:
        return None
    try:
        text = json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return None
    # OANDA comment max 256 chars
    return text[:255]


def _trade_min_hold_seconds(trade: dict, pocket: str) -> float:
    thesis = _coerce_entry_thesis(trade.get("entry_thesis"))
    hold = thesis.get("min_hold_sec") or thesis.get("min_hold_seconds")
    if hold is None:
        try:
            hold = float(thesis.get("min_hold_min") or thesis.get("min_hold_minutes")) * 60.0
        except Exception:
            hold = None
    try:
        hold_val = float(hold)
    except (TypeError, ValueError):
        hold_val = _DEFAULT_MIN_HOLD_SEC.get(pocket, 60.0)
    if hold_val <= 0.0:
        return _DEFAULT_MIN_HOLD_SEC.get(pocket, 60.0)
    return hold_val


def _trade_age_seconds(trade: dict, now: datetime) -> Optional[float]:
    opened_at = _parse_trade_open_time(trade.get("open_time"))
    if not opened_at:
        return None
    delta = (now - opened_at).total_seconds()
    return max(0.0, delta)


def _maybe_update_protections(
    trade_id: str,
    sl_price: Optional[float],
    tp_price: Optional[float],
    *,
    context: str = "auto",
    ref_price: Optional[float] = None,
) -> None:
    if not TRAILING_SL_ALLOWED:
        sl_price = None
    if not trade_id or (sl_price is None and tp_price is None):
        return

    data: dict[str, dict[str, str]] = {}
    if sl_price is not None:
        data["stopLoss"] = {
            "price": f"{sl_price:.3f}",
            "timeInForce": "GTC",
        }
    if tp_price is not None:
        data["takeProfit"] = {
            "price": f"{tp_price:.3f}",
            "timeInForce": "GTC",
        }

    if not data:
        return

    previous = _LAST_PROTECTIONS.get(trade_id) or {}
    prev_sl = previous.get("sl")
    prev_tp = previous.get("tp")
    current_sl = round(sl_price, 3) if sl_price is not None else None
    current_tp = round(tp_price, 3) if tp_price is not None else None
    if prev_sl == current_sl and prev_tp == current_tp:
        return

    def _fmt(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.3f}"

    logging.info(
        "[PROTECT][%s] trade=%s sl=%s->%s tp=%s->%s ref=%s",
        context,
        trade_id,
        _fmt(prev_sl),
        _fmt(current_sl),
        _fmt(prev_tp),
        _fmt(current_tp),
        _fmt(ref_price),
    )

    try:
        req = TradeCRCDO(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        _LAST_PROTECTIONS[trade_id] = {
            "sl": current_sl,
            "tp": current_tp,
            "ts": time.time(),
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Failed to update protections trade=%s sl=%s tp=%s ctx=%s: %s",
            trade_id,
            sl_price,
            tp_price,
            context,
            exc,
        )


async def close_trade(
    trade_id: str,
    units: Optional[int] = None,
    client_order_id: Optional[str] = None,
    allow_negative: bool = False,
    exit_reason: Optional[str] = None,
) -> bool:
    service_result = await _order_manager_service_request_async(
        "/order/close_trade",
        {
            "trade_id": trade_id,
            "units": units,
            "client_order_id": client_order_id,
            "allow_negative": allow_negative,
            "exit_reason": exit_reason,
        },
    )
    if service_result is not None:
        return bool(service_result)

    data: Optional[dict[str, str]] = None
    exit_reason = str(exit_reason).strip() if exit_reason else None
    exit_context = _exit_context_snapshot(exit_reason)

    def _with_exit_reason(payload: Optional[dict]) -> dict:
        base = dict(payload) if payload else {}
        if exit_reason:
            base["exit_reason"] = exit_reason
        if exit_context:
            base["exit_context"] = exit_context
        return base

    def _log_close_order(**kwargs: Any) -> None:
        # Close path must prioritize broker round-trip over local audit logging.
        # Use fast-fail logging to avoid blocking /order/close_trade on orders.db lock contention.
        _log_order(fast_fail=True, **kwargs)

    # close 側も client_order_id を必須化。欠損かつ agent 管理外の建玉はスキップして無駄打ちを防ぐ。
    if not client_order_id:
        log_metric("close_skip_missing_client_id", 1.0, tags={"trade_id": str(trade_id)})
        logging.info("[ORDER] skip close trade=%s missing client_id (likely manual/external)", trade_id)
        return False
    ctx = _load_exit_trade_context(trade_id, client_order_id)
    strategy_tag = None
    pocket = None
    if isinstance(ctx, dict):
        pocket = ctx.get("pocket")
        strategy_tag = _strategy_tag_from_thesis(ctx.get("entry_thesis"))
        if not strategy_tag:
            strategy_tag = ctx.get("strategy_tag")
    if not strategy_tag:
        strategy_tag = _strategy_tag_from_client_id(client_order_id)

    if _reject_exit_by_control(strategy_tag, pocket=str(pocket or "")):
        note = "strategy_control_exit_disabled"
        _console_order_log(
            "CLOSE_REJECT",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
            note=note,
        )
        _log_close_order(
            pocket=pocket,
            instrument=(ctx or {}).get("instrument") if isinstance(ctx, dict) else None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status=note,
            attempt=0,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload={"trade_id": str(trade_id), "strategy_tag": strategy_tag},
        )
        log_metric(
            "close_blocked_by_strategy_control",
            1.0,
            tags={"pocket": str(pocket or "unknown"), "strategy": str(strategy_tag or "unknown"), "action": "exit"},
        )
        return False

    emergency_allow: Optional[bool] = None
    entry_price = _as_float((ctx or {}).get("entry_price")) or 0.0
    units_ctx = int((ctx or {}).get("units") or 0)
    instrument = (ctx or {}).get("instrument") if isinstance(ctx, dict) else None
    min_profit_pips = _min_profit_pips(pocket, strategy_tag)
    bid, ask = _latest_bid_ask()
    mid = None
    if bid is None or ask is None:
        quote = _fetch_quote(instrument) if instrument else None
        if quote:
            bid = quote.get("bid")
            ask = quote.get("ask")
            mid = quote.get("mid")
    if bid is not None and ask is not None and mid is None:
        mid = (bid + ask) / 2.0
    if (bid is None or ask is None) and (_EXIT_NO_NEGATIVE_CLOSE or min_profit_pips is not None):
        emergency_allow = _should_allow_negative_close(client_order_id)
        if not emergency_allow and not _reason_force_allow(exit_reason):
            log_metric(
                "close_blocked_missing_quote",
                1.0,
                tags={"trade_id": str(trade_id), "instrument": str(instrument or "unknown")},
            )
            _console_order_log(
                "CLOSE_REJECT",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                ticket_id=str(trade_id),
                note="missing_quote",
            )
            _log_close_order(
                pocket=pocket,
                instrument=instrument,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                status="close_reject_missing_quote",
                attempt=0,
                ticket_id=str(trade_id),
                executed_price=None,
                request_payload=_with_exit_reason({"trade_id": trade_id}),
            )
            return False
    est_pips = _estimate_trade_pnl_pips(
        entry_price=entry_price,
        units=units_ctx,
        bid=bid,
        ask=ask,
        mid=mid,
    )
    pl: Optional[float] = None
    end_reversal = _exit_end_reversal_eval(
        exit_reason=exit_reason,
        strategy_tag=strategy_tag,
        units_ctx=units_ctx,
        est_pips=est_pips,
        exit_context=exit_context,
        instrument=instrument,
    )
    if isinstance(exit_context, dict):
        exit_context["end_reversal"] = end_reversal
    bypass_gates: set[str] = set()

    def _log_end_reversal_bypass(gate: str, payload: dict) -> None:
        if gate in bypass_gates:
            return
        bypass_gates.add(gate)
        score = _as_float(end_reversal.get("score"), 0.0) or 0.0
        log_metric(
            "close_guard_bypass_end_reversal",
            float(est_pips if est_pips is not None else 0.0),
            tags={
                "trade_id": str(trade_id),
                "pocket": str(pocket or "unknown"),
                "strategy": str(strategy_tag or "unknown"),
                "gate": gate,
                "exit_reason": str(exit_reason or "unknown"),
                "score": f"{score:.3f}",
            },
        )
        _console_order_log(
            "CLOSE_BYPASS",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
            note=f"{gate}:end_reversal",
        )
        _log_close_order(
            pocket=pocket,
            instrument=instrument,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status="close_guard_bypass_end_reversal",
            attempt=0,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload=_with_exit_reason(payload),
        )
    if min_profit_pips is not None:
        if est_pips is not None and est_pips >= 0 and est_pips < min_profit_pips:
            emergency_allow = _should_allow_negative_close(client_order_id)
            bypass_by_end_reversal = bool(end_reversal.get("triggered"))
            if not emergency_allow and not _reason_force_allow(exit_reason) and not bypass_by_end_reversal:
                log_metric(
                    "close_blocked_profit_buffer",
                    float(est_pips),
                    tags={
                        "trade_id": str(trade_id),
                        "pocket": str(pocket or "unknown"),
                        "strategy": str(strategy_tag or "unknown"),
                        "min_pips": str(min_profit_pips),
                    },
                )
                _console_order_log(
                    "CLOSE_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    ticket_id=str(trade_id),
                    note="profit_buffer",
                )
                _log_close_order(
                    pocket=pocket,
                    instrument=None,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    status="close_reject_profit_buffer",
                    attempt=0,
                    ticket_id=str(trade_id),
                    executed_price=None,
                    request_payload=_with_exit_reason(
                        {
                            "trade_id": trade_id,
                            "data": {"min_profit_pips": min_profit_pips, "est_pips": est_pips},
                        }
                    ),
                )
                return False
            if bypass_by_end_reversal and not emergency_allow and not _reason_force_allow(exit_reason):
                _log_end_reversal_bypass(
                    "profit_buffer",
                    {
                        "trade_id": trade_id,
                        "data": {
                            "gate": "profit_buffer",
                            "min_profit_pips": min_profit_pips,
                            "est_pips": est_pips,
                            "end_reversal": end_reversal,
                        },
                    },
                )
    ratio = _min_profit_ratio(pocket, strategy_tag)
    if ratio is not None and ratio > 0:
        ratio_reasons = _min_profit_ratio_reasons(strategy_tag)
        if exit_reason and _reason_matches_tokens(exit_reason, list(ratio_reasons)):
            tp_price = _as_float((ctx or {}).get("tp_price"))
            tp_min = _min_profit_ratio_min_tp_pips(strategy_tag)
            if tp_price is not None and entry_price > 0:
                tp_pips = abs(tp_price - entry_price) / 0.01
                if tp_pips >= tp_min and est_pips is not None and est_pips >= 0:
                    min_ratio_pips = tp_pips * ratio
                    if est_pips < min_ratio_pips:
                        emergency_allow = _should_allow_negative_close(client_order_id)
                        bypass_by_end_reversal = bool(end_reversal.get("triggered"))
                        if not emergency_allow and not _reason_force_allow(exit_reason) and not bypass_by_end_reversal:
                            log_metric(
                                "close_blocked_profit_ratio",
                                float(est_pips),
                                tags={
                                    "trade_id": str(trade_id),
                                    "pocket": str(pocket or "unknown"),
                                    "strategy": str(strategy_tag or "unknown"),
                                    "min_ratio": str(ratio),
                                    "tp_pips": str(round(tp_pips, 3)),
                                },
                            )
                            _console_order_log(
                                "CLOSE_REJECT",
                                pocket=pocket,
                                strategy_tag=strategy_tag,
                                side=None,
                                units=units,
                                sl_price=None,
                                tp_price=None,
                                client_order_id=client_order_id,
                                ticket_id=str(trade_id),
                                note="profit_ratio",
                            )
                            _log_close_order(
                                pocket=pocket,
                                instrument=instrument,
                                side=None,
                                units=units,
                                sl_price=None,
                                tp_price=None,
                                client_order_id=client_order_id,
                                status="close_reject_profit_ratio",
                                attempt=0,
                                ticket_id=str(trade_id),
                                executed_price=None,
                                request_payload=_with_exit_reason(
                                    {
                                        "trade_id": trade_id,
                                        "data": {
                                            "min_ratio": ratio,
                                            "tp_pips": round(tp_pips, 3),
                                            "min_ratio_pips": round(min_ratio_pips, 3),
                                            "est_pips": est_pips,
                                        },
                                    }
                                ),
                            )
                            return False
                        if bypass_by_end_reversal and not emergency_allow and not _reason_force_allow(exit_reason):
                            _log_end_reversal_bypass(
                                "profit_ratio",
                                {
                                    "trade_id": trade_id,
                                    "data": {
                                        "gate": "profit_ratio",
                                        "min_ratio": ratio,
                                        "tp_pips": round(tp_pips, 3),
                                        "min_ratio_pips": round(min_ratio_pips, 3),
                                        "est_pips": est_pips,
                                        "end_reversal": end_reversal,
                                    },
                                },
                            )
    hold_match, hold_min_pips, hold_strict = _hold_until_profit_match(trade_id, client_order_id)
    if hold_match:
        profit_ok = False
        if est_pips is not None:
            profit_ok = est_pips > hold_min_pips
        else:
            pl = _current_trade_unrealized_pl(trade_id)
            profit_ok = pl is not None and pl > 0.0
        if not profit_ok:
            if emergency_allow is None:
                emergency_allow = _should_allow_negative_close(client_order_id)
            # "Hold until profit" must not trap a trade indefinitely. Even when strict,
            # honor risk-style exits that are explicitly requested by the worker.
            # Note: this is intentionally conservative; we only bypass for emergency/force
            # reasons and a few hard safety exits used by fast scalps.
            hold_bypass = bool(
                emergency_allow
                or _reason_force_allow(exit_reason)
                or _reason_matches_tokens(
                    exit_reason,
                    ["time_stop", "no_recovery", "max_floating_loss"],
                )
            )
            if hold_bypass:
                profit_ok = True
            if not profit_ok:
                log_metric(
                    "close_blocked_hold_profit",
                    float(est_pips if est_pips is not None else (pl or 0.0)),
                    tags={
                        "trade_id": str(trade_id),
                        "min_pips": str(hold_min_pips),
                        "strict": str(hold_strict),
                    },
                )
                _console_order_log(
                    "CLOSE_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    ticket_id=str(trade_id),
                    note="hold_until_profit",
                )
                _log_close_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    status="close_reject_hold_profit",
                    attempt=0,
                    ticket_id=str(trade_id),
                    executed_price=None,
                    request_payload=_with_exit_reason(
                        {
                            "trade_id": trade_id,
                            "data": {
                                "min_profit_pips": hold_min_pips,
                                "est_pips": est_pips,
                                "unrealized_pl": pl,
                            },
                        }
                    ),
                )
                return False
    if _EXIT_NO_NEGATIVE_CLOSE:
        if pl is None:
            pl = _current_trade_unrealized_pl(trade_id)
        pl_pips: Optional[float] = None
        if pl is not None and units_ctx:
            # Fallback pips estimate based on OANDA unrealizedPL. This makes near-BE exits
            # (lock_floor/trail_lock/etc) robust even when quote-based est_pips is missing
            # or rounds to 0 around the spread.
            try:
                pip = 0.01 if str(instrument or "").upper().endswith("_JPY") else 0.0001
                denom = abs(int(units_ctx)) * pip
                if denom > 0:
                    pl_pips = float(pl) / denom
            except Exception:
                pl_pips = None
        if pl_pips is not None:
            if est_pips is None:
                est_pips = pl_pips
            elif est_pips >= 0 and pl_pips < 0 and abs(est_pips) <= 0.2:
                est_pips = pl_pips
        negative_by_pips = est_pips is not None and est_pips <= 0
        negative_by_pl = pl is not None and pl <= 0
        if negative_by_pips or (est_pips is None and negative_by_pl):
            if emergency_allow is None:
                emergency_allow = _should_allow_negative_close(client_order_id)
            reason_allow = _reason_allows_negative(exit_reason)
            worker_allow = _EXIT_ALLOW_NEGATIVE_BY_WORKER and allow_negative
            neg_policy = _strategy_neg_exit_policy(strategy_tag)
            neg_allowed, near_be_allow = _neg_exit_decision(
                exit_reason=exit_reason,
                est_pips=est_pips,
                emergency_allow=bool(emergency_allow),
                reason_allow=reason_allow,
                worker_allow=worker_allow,
                neg_policy=neg_policy,
            )
            allow_negative = bool(neg_allowed)
            if not neg_allowed:
                log_metric(
                    "close_blocked_negative",
                    float(est_pips if est_pips is not None else (pl or 0.0)),
                    tags={
                        "trade_id": str(trade_id),
                        "reason": str(exit_reason or "unknown"),
                    },
                )
                _console_order_log(
                    "CLOSE_REJECT",
                    pocket=None,
                    strategy_tag=None,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    ticket_id=str(trade_id),
                    note="no_negative_close",
                )
                _log_close_order(
                    pocket=None,
                    instrument=None,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    status="close_reject_no_negative",
                    attempt=0,
                    ticket_id=str(trade_id),
                    executed_price=None,
                    request_payload=_with_exit_reason(
                        {"trade_id": trade_id, "data": {"unrealized_pl": pl}}
                    ),
                )
                return False
    if units is None:
        data = {"units": "ALL"}
    else:
        rounded_units = int(units)
        if rounded_units == 0:
            return True
        target_units = abs(rounded_units)
        actual_units = _current_trade_units(trade_id)
        if actual_units is not None:
            if actual_units <= 0:
                logging.info("[ORDER] Close skipped trade=%s already flat.", trade_id)
                return True
            if target_units > actual_units:
                logging.info(
                    "[ORDER] Clamping close units trade=%s requested=%s available=%s",
                    trade_id,
                    target_units,
                    actual_units,
                )
                target_units = actual_units
        # OANDA expects the absolute size; the trade side is derived from trade_id.
        data = {"units": str(target_units)}

    if not is_market_open():
        # OANDA returns HTTP 200 with orderCancelTransaction(reason=MARKET_HALTED) during weekend/daily halt.
        # Don't spam the broker nor orders.db with repeated close attempts.
        now_mono = time.monotonic()
        key = str(trade_id)
        last_log = _CLOSE_MARKET_CLOSED_LOG_TS.get(key, 0.0)
        if now_mono - last_log >= _CLOSE_MARKET_CLOSED_LOG_INTERVAL_SEC:
            _CLOSE_MARKET_CLOSED_LOG_TS[key] = now_mono
            logging.info(
                "[ORDER] Market closed. Skip TradeClose trade=%s units=%s client=%s",
                trade_id,
                units if units is not None else "ALL",
                client_order_id or "-",
            )
            _console_order_log(
                "CLOSE_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                ticket_id=str(trade_id),
                note="market_closed",
            )
        await asyncio.sleep(_CLOSE_MARKET_CLOSED_BACKOFF_SEC)
        return False
    req = TradeClose(accountID=ACCOUNT, tradeID=trade_id, data=data)
    if not client_order_id:
        log_metric("close_missing_client_id", 1.0, tags={"trade_id": str(trade_id)})
        client_order_id = None
    _console_order_log(
        "CLOSE_REQ",
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=None,
        units=units,
        sl_price=None,
        tp_price=None,
        client_order_id=client_order_id,
        ticket_id=str(trade_id),
    )
    try:
        _log_close_order(
            pocket=pocket,
            instrument=instrument,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status="close_request",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload=_with_exit_reason({"trade_id": trade_id, "data": data or {}}),
        )
        api.request(req)
        response = req.response if isinstance(getattr(req, "response", None), dict) else {}
        reject = response.get("orderRejectTransaction") or response.get("orderCancelTransaction")
        if reject:
            reason = reject.get("rejectReason") or reject.get("reason")
            reason_key = str(reason or "").upper() or "rejected"
            err_code = str(reject.get("errorCode") or reason_key)
            _log_close_order(
                pocket=pocket,
                instrument=instrument,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                status="close_failed",
                attempt=1,
                ticket_id=str(trade_id),
                executed_price=None,
                error_code=err_code,
                error_message=reject.get("errorMessage") or str(reason_key),
                response_payload=response,
                request_payload=_with_exit_reason({"trade_id": trade_id, "data": data or {}}),
            )
            _console_order_log(
                "CLOSE_FAIL",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                ticket_id=str(trade_id),
                note=f"cancel:{reason_key}",
            )
            if reason_key == "MARKET_HALTED":
                await asyncio.sleep(_CLOSE_MARKET_CLOSED_BACKOFF_SEC)
            return False
        fill = response.get("orderFillTransaction") or {}
        if not fill:
            _log_close_order(
                pocket=pocket,
                instrument=instrument,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                status="close_failed",
                attempt=1,
                ticket_id=str(trade_id),
                executed_price=None,
                error_code="missing_fill",
                error_message="missing orderFillTransaction in TradeClose response",
                response_payload=response,
                request_payload=_with_exit_reason({"trade_id": trade_id, "data": data or {}}),
            )
            _console_order_log(
                "CLOSE_FAIL",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=None,
                units=units,
                sl_price=None,
                tp_price=None,
                client_order_id=client_order_id,
                ticket_id=str(trade_id),
                note="missing_fill",
            )
            return False
        executed_price = None
        if fill.get("price"):
            try:
                executed_price = float(fill.get("price"))
            except Exception:
                executed_price = None
        _LAST_PROTECTIONS.pop(trade_id, None)
        _PARTIAL_STAGE.pop(trade_id, None)
        _log_close_order(
            pocket=pocket,
            instrument=instrument,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status="close_ok",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=executed_price,
            request_payload=_with_exit_reason({"trade_id": trade_id}),
        )
        _console_order_log(
            "CLOSE_OK",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
        )
        return True
    except V20Error as exc:
        error_payload = {}
        try:
            error_payload = json.loads(exc.msg or "{}")
        except json.JSONDecodeError:
            error_payload = {"errorMessage": exc.msg}
        error_code = error_payload.get("errorCode")
        logging.warning(
            "[ORDER] TradeClose rejected trade=%s units=%s code=%s",
            trade_id,
            units,
            error_code or exc.code,
        )
        log_error_code = str(error_code) if error_code is not None else str(exc.code)
        _log_close_order(
            pocket=pocket,
            instrument=instrument,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status="close_failed",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            error_code=log_error_code,
            error_message=error_payload.get("errorMessage") or str(exc),
            response_payload=error_payload if error_payload else None,
            request_payload=_with_exit_reason({"trade_id": trade_id, "data": data or {}}),
        )
        _console_order_log(
            "CLOSE_FAIL",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
            note=f"code={log_error_code}",
        )
        if (log_error_code or "").upper() in _PARTIAL_CLOSE_RETRY_CODES:
            if _retry_close_with_actual_units(trade_id, units):
                return True
        # OANDA returns a few variants for missing/closed trades; treat as idempotent success
        benign_missing = {
            "TRADE_DOES_NOT_EXIST",
            "TRADE_DOESNT_EXIST",
            "NOT_FOUND",
        }
        benign_reduce_only = {"NO_POSITION_TO_REDUCE"}
        if error_code in benign_missing or error_code in benign_reduce_only:
            logging.info(
                "[ORDER] Close benign rejection for trade %s (code=%s) – treating as success.",
                trade_id,
                error_code,
            )
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Failed to close trade %s units=%s: %s", trade_id, units, exc
        )
        _log_close_order(
            pocket=pocket,
            instrument=instrument,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status="close_failed",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            error_message=str(exc),
        )
        _console_order_log(
            "CLOSE_FAIL",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
            note="exception",
        )
        return False


def update_dynamic_protections(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    if not open_positions:
        return
    rollover_ctx = _rollover_sl_strip_context()
    try:
        _strip_rollover_stop_losses(open_positions, rollover_ctx)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[PROTECT] rollover SL strip failed: %s", exc)
    if not TRAILING_SL_ALLOWED:
        return
    _apply_dynamic_protections_v2(open_positions, fac_m1, fac_h4, rollover_ctx=rollover_ctx)
    return
    atr_m1 = fac_m1.get("atr_pips")
    if atr_m1 is None:
        atr_m1 = (fac_m1.get("atr") or 0.0) * 100
    atr_h4 = fac_h4.get("atr_pips")
    if atr_h4 is None:
        atr_h4 = (fac_h4.get("atr") or atr_m1 or 0.0)
    current_price = fac_m1.get("close")
    defaults = {
        "macro": (max(25.0, (atr_h4 or atr_m1 or 0.0) * 1.1), 2.2),
        "micro": (max(8.0, (atr_m1 or 0.0) * 0.9), 1.9),
        # Scalpは原則として建玉時のSL/TPを尊重する。
        # 必要最小限の安全網のみ。mirror-s5 等のスキャル戦略では広げない。
        "scalp": (max(6.0, (atr_m1 or 0.0) * 1.2), 1.0),
    }
    # ポケット別の BE/トレーリング開始閾値（pip）
    # micro/scalp は早めに建値超えへ移行し、利確を積み上げる方針
    per_pocket_triggers = {
        "macro": max(8.0, (atr_h4 or atr_m1 or 0.0) * 1.5),
        "micro": max(3.0, (atr_m1 or 0.0) * 0.8),
        # スキャルも一定の含み益で建値超えに移行（戻り負けの抑制）
        "scalp": max(2.4, (atr_m1 or 0.0) * 1.2),
    }
    lock_ratio = 0.6
    per_pocket_min_lock = {"macro": 3.0, "micro": 2.0, "scalp": 0.6}
    pip = 0.01
    now_ts = datetime.now(timezone.utc)
    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        base = defaults.get(pocket)
        if not base:
            continue
        base_sl, tp_ratio = base
        trail_trigger = per_pocket_triggers.get(pocket, per_pocket_triggers["macro"])
        min_lock = per_pocket_min_lock.get(pocket, 3.0)
        trades = info.get("open_trades") or []
        for tr in trades:
            trade_id = tr.get("trade_id")
            price = tr.get("price")
            side = tr.get("side")
            if not trade_id or price is None or not side:
                continue
            client_id = str(tr.get("client_id") or "")
            # Safety: skip manual/unknown trades unless explicitly managed by the bot
            if not client_id.startswith("qr-"):
                continue
            # mirror-s5（client_id が 'qr-mirror-s5-' プレフィクス）については
            # ここでの動的SL更新をスキップし、エントリー時のSL/TPを維持する。
            if pocket == "scalp":
                client_id = str(tr.get("client_id") or "")
                if client_id.startswith("qr-mirror-s5-"):
                    continue
            entry = float(price)
            sl_pips = max(1.0, base_sl)
            tp_pips = max(sl_pips * tp_ratio, sl_pips + 5.0) if pocket != "scalp" else None
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            hold_seconds = 0.0
            if opened_at:
                hold_seconds = max(0.0, (now_ts - opened_at).total_seconds())
            gain_pips = 0.0
            if side == "long":
                gain_pips = (current_price or entry) - entry
                gain_pips *= 100
                sl_price = round(entry - sl_pips * pip, 3)
                tp_price = round(entry + tp_pips * pip, 3) if tp_pips is not None else None
                allow_lock = True
                if (
                    pocket == "macro"
                    and hold_seconds < MACRO_BE_GRACE_SECONDS
                ):
                    momentum = (fac_m1.get("close") or 0.0) - (
                        fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0
                    )
                    if momentum >= 0.0:
                        allow_lock = False
                if gain_pips > trail_trigger and allow_lock and pocket != "scalp":
                    lock_pips = max(min_lock, gain_pips * lock_ratio)
                    be_price = entry + lock_pips * pip
                    sl_price = max(sl_price, round(be_price, 3))
                    tp_price = round(entry + max(gain_pips + sl_pips, tp_pips) * pip, 3)
                    if tp_price <= sl_price + 0.002:
                        tp_price = round(sl_price + max(0.004, tp_pips * pip), 3)
                if current_price is not None and sl_price >= current_price:
                    sl_price = round(current_price - 0.003, 3)
            else:
                gain_pips = entry - (current_price or entry)
                gain_pips *= 100
                sl_price = round(entry + sl_pips * pip, 3)
                tp_price = round(entry - tp_pips * pip, 3) if tp_pips is not None else None
                allow_lock = True
                if (
                    pocket == "macro"
                    and hold_seconds < MACRO_BE_GRACE_SECONDS
                ):
                    momentum = (fac_m1.get("close") or 0.0) - (
                        fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0
                    )
                    if momentum <= 0.0:
                        allow_lock = False
                if gain_pips > trail_trigger and allow_lock and pocket != "scalp":
                    lock_pips = max(min_lock, gain_pips * lock_ratio)
                    be_price = entry - lock_pips * pip
                    sl_price = min(sl_price, round(be_price, 3))
                    tp_price = round(entry - max(gain_pips + sl_pips, tp_pips) * pip, 3)
                    if tp_price >= sl_price - 0.002:
                        tp_price = round(sl_price - max(0.004, tp_pips * pip), 3)
                if current_price is not None and sl_price <= current_price:
                    sl_price = round(current_price + 0.003, 3)
            _maybe_update_protections(
                trade_id,
                sl_price,
                tp_price,
                context="dynamic_protection_v1",
                ref_price=current_price,
            )


def _apply_dynamic_protections_v2(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
    *,
    rollover_ctx: Optional[dict[str, Any]] = None,
) -> None:
    policy = policy_bus.latest()
    pockets_policy = policy.pockets if policy else {}
    try:
        atr_m1 = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
    except Exception:
        atr_m1 = 0.0
    try:
        vol_5m = float(fac_m1.get("vol_5m") or 1.0)
    except Exception:
        vol_5m = 1.0

    now_ts = time.time()
    current_price = fac_m1.get("close")
    pip = 0.01
    # 市況（ATR/短期ボラ）を拾ってトリガーを動的調整
    def _coerce(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    atr_m1 = _coerce(fac_m1.get("atr_pips"), _coerce(fac_m1.get("atr"), 0.0) * 100.0)
    vol_5m = _coerce(fac_m1.get("vol_5m"), 1.0)

    strategy_cfg = _load_strategy_protection_config()
    defaults_cfg = strategy_cfg.get("defaults", {}) if isinstance(strategy_cfg, dict) else {}
    apply_when_soft_default = _coerce_bool(defaults_cfg.get("apply_when_soft_tp"), False)
    tp_move_min_gap_default = _coerce(
        defaults_cfg.get("tp_move_min_gap_pips"),
        _env_float("TP_MOVE_MIN_GAP_PIPS", 0.3),
    )

    defaults = {
        "macro": {
            "trigger": _env_float("BE_TRAIL_TRIGGER_MACRO", 6.8),
            "lock_ratio": _env_float("BE_TRAIL_LOCK_RATIO_MACRO", 0.55),
            "min_lock": _env_float("BE_TRAIL_MIN_LOCK_MACRO", 2.6),
            "cooldown": _env_float("BE_TRAIL_COOLDOWN_MACRO_SEC", 90.0),
        },
        "micro": {
            "trigger": _env_float("BE_TRAIL_TRIGGER_MICRO", 2.2),
            "lock_ratio": _env_float("BE_TRAIL_LOCK_RATIO_MICRO", 0.50),
            "min_lock": _env_float("BE_TRAIL_MIN_LOCK_MICRO", 0.60),
            "cooldown": _env_float("BE_TRAIL_COOLDOWN_MICRO_SEC", 45.0),
        },
        "scalp": {
            "trigger": _env_float("BE_TRAIL_TRIGGER_SCALP", 1.6),
            "lock_ratio": _env_float("BE_TRAIL_LOCK_RATIO_SCALP", 0.35),
            "min_lock": _env_float("BE_TRAIL_MIN_LOCK_SCALP", 0.50),
            "cooldown": _env_float("BE_TRAIL_COOLDOWN_SCALP_SEC", 20.0),
        },
        "scalp_fast": {
            "trigger": _env_float(
                "BE_TRAIL_TRIGGER_SCALP_FAST",
                _env_float("BE_TRAIL_TRIGGER_SCALP", 1.6),
            ),
            "lock_ratio": _env_float(
                "BE_TRAIL_LOCK_RATIO_SCALP_FAST",
                _env_float("BE_TRAIL_LOCK_RATIO_SCALP", 0.35),
            ),
            "min_lock": _env_float(
                "BE_TRAIL_MIN_LOCK_SCALP_FAST",
                _env_float("BE_TRAIL_MIN_LOCK_SCALP", 0.50),
            ),
            "cooldown": _env_float(
                "BE_TRAIL_COOLDOWN_SCALP_FAST_SEC",
                _env_float("BE_TRAIL_COOLDOWN_SCALP_SEC", 20.0),
            ),
        },
    }
    # トレーリング開始を ATR/ボラで動的に決める（micro/scalp）
    base_start_delay = {
        "micro": _env_float("BE_TRAIL_START_DELAY_MICRO_SEC", 25.0),
        "scalp": _env_float("BE_TRAIL_START_DELAY_SCALP_SEC", 12.0),
        "scalp_fast": _env_float(
            "BE_TRAIL_START_DELAY_SCALP_FAST_SEC",
            _env_float("BE_TRAIL_START_DELAY_SCALP_SEC", 12.0),
        ),
    }
    max_start_delay = {
        "micro": _env_float("BE_TRAIL_MAX_DELAY_MICRO_SEC", 70.0),
        "scalp": _env_float("BE_TRAIL_MAX_DELAY_SCALP_SEC", 35.0),
        "scalp_fast": _env_float(
            "BE_TRAIL_MAX_DELAY_SCALP_FAST_SEC",
            _env_float("BE_TRAIL_MAX_DELAY_SCALP_SEC", 35.0),
        ),
    }
    tp_move_enabled = _env_bool("TP_MOVE_ENABLED", True)
    tp_move_cfg = {
        "macro": {
            "trigger": _env_float("TP_MOVE_TRIGGER_MACRO", 6.0),
            "buffer": _env_float("TP_MOVE_BUFFER_MACRO", 2.5),
        },
        "micro": {
            "trigger": _env_float("TP_MOVE_TRIGGER_MICRO", 2.0),
            "buffer": _env_float("TP_MOVE_BUFFER_MICRO", 1.0),
        },
        "scalp": {
            "trigger": _env_float("TP_MOVE_TRIGGER_SCALP", 1.0),
            "buffer": _env_float("TP_MOVE_BUFFER_SCALP", 0.8),
        },
        "scalp_fast": {
            "trigger": _env_float(
                "TP_MOVE_TRIGGER_SCALP_FAST",
                _env_float("TP_MOVE_TRIGGER_SCALP", 1.0),
            ),
            "buffer": _env_float(
                "TP_MOVE_BUFFER_SCALP_FAST",
                _env_float("TP_MOVE_BUFFER_SCALP", 0.8),
            ),
        },
    }

    if isinstance(defaults_cfg, dict):
        be_defaults = defaults_cfg.get("be_profile")
        if isinstance(be_defaults, dict):
            for pocket_key in ("macro", "micro", "scalp", "scalp_fast"):
                override = be_defaults.get(pocket_key)
                if not isinstance(override, dict) and pocket_key == "scalp_fast":
                    override = be_defaults.get("scalp")
                if isinstance(override, dict):
                    defaults[pocket_key]["trigger"] = _coerce(
                        override.get("trigger_pips"), defaults[pocket_key]["trigger"]
                    )
                    defaults[pocket_key]["lock_ratio"] = _coerce(
                        override.get("lock_ratio"), defaults[pocket_key]["lock_ratio"]
                    )
                    defaults[pocket_key]["min_lock"] = _coerce(
                        override.get("min_lock_pips"), defaults[pocket_key]["min_lock"]
                    )
                    defaults[pocket_key]["cooldown"] = _coerce(
                        override.get("cooldown_sec"), defaults[pocket_key]["cooldown"]
                    )
        start_defaults = defaults_cfg.get("start_delay_sec")
        if isinstance(start_defaults, dict):
            base_start_delay["micro"] = _coerce(
                start_defaults.get("micro"), base_start_delay.get("micro", 25.0)
            )
            base_start_delay["scalp"] = _coerce(
                start_defaults.get("scalp"), base_start_delay.get("scalp", 12.0)
            )
            base_start_delay["scalp_fast"] = _coerce(
                start_defaults.get("scalp_fast", start_defaults.get("scalp")),
                base_start_delay.get("scalp_fast", base_start_delay.get("scalp", 12.0)),
            )
        max_delay_defaults = defaults_cfg.get("max_delay_sec")
        if isinstance(max_delay_defaults, dict):
            max_start_delay["micro"] = _coerce(
                max_delay_defaults.get("micro"), max_start_delay.get("micro", 70.0)
            )
            max_start_delay["scalp"] = _coerce(
                max_delay_defaults.get("scalp"), max_start_delay.get("scalp", 35.0)
            )
            max_start_delay["scalp_fast"] = _coerce(
                max_delay_defaults.get("scalp_fast", max_delay_defaults.get("scalp")),
                max_start_delay.get("scalp_fast", max_start_delay.get("scalp", 35.0)),
            )
        tp_defaults = defaults_cfg.get("tp_move")
        if isinstance(tp_defaults, dict):
            tp_enabled = tp_defaults.get("enabled")
            if tp_enabled is not None:
                tp_move_enabled = _coerce_bool(tp_enabled, tp_move_enabled)
            for pocket_key in ("macro", "micro", "scalp", "scalp_fast"):
                override = tp_defaults.get(pocket_key)
                if not isinstance(override, dict) and pocket_key == "scalp_fast":
                    override = tp_defaults.get("scalp")
                if isinstance(override, dict):
                    tp_move_cfg[pocket_key]["trigger"] = _coerce(
                        override.get("trigger_pips"), tp_move_cfg[pocket_key]["trigger"]
                    )
                    tp_move_cfg[pocket_key]["buffer"] = _coerce(
                        override.get("buffer_pips"), tp_move_cfg[pocket_key]["buffer"]
                    )

    def _strategy_tag_from_trade(tr: dict, thesis: dict) -> str:
        tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or tr.get("strategy_tag")
            or tr.get("strategy")
        )
        return str(tag).strip() if tag else ""

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        trades = info.get("open_trades") or []
        if not trades:
            continue

        plan = pockets_policy.get(pocket) if isinstance(pockets_policy, dict) else {}
        be_profile = plan.get("be_profile", {}) if isinstance(plan, dict) else {}
        default_cfg = defaults.get(pocket, defaults["macro"])
        trigger = _coerce(be_profile.get("trigger_pips"), default_cfg["trigger"])
        lock_ratio = max(0.0, min(1.0, _coerce(be_profile.get("lock_ratio"), default_cfg["lock_ratio"])))
        min_lock = max(0.0, _coerce(be_profile.get("min_lock_pips"), default_cfg["min_lock"]))
        cooldown_sec = max(0.0, _coerce(be_profile.get("cooldown_sec"), default_cfg["cooldown"]))

        # ATR/ボラに応じて動的スケール
        atr_val = atr_m1
        if pocket == "macro":
            # マクロは大きめのATRを見て少しだけ拡げる
            trigger = max(trigger, atr_val * (1.2 if vol_5m < 1.0 else 1.35 if vol_5m < 1.8 else 1.5))
            min_lock = max(min_lock, atr_val * 0.3)
            lock_ratio = max(lock_ratio, 0.50 if vol_5m >= 1.5 else 0.45)
        elif pocket == "micro":
            if vol_5m < 0.8:
                trigger = max(trigger, atr_val * 0.9)
                lock_ratio = max(lock_ratio, 0.45)
            elif vol_5m > 1.6:
                trigger = max(trigger, atr_val * 1.1)
                lock_ratio = max(lock_ratio, 0.40)
            else:
                trigger = max(trigger, atr_val * 1.0)
                lock_ratio = max(lock_ratio, 0.42)
            min_lock = max(min_lock, atr_val * 0.22)
            trigger = min(trigger, _env_float("BE_TRAIL_MAX_TRIGGER_MICRO", 4.0))
        elif pocket in {"scalp", "scalp_fast"}:
            if vol_5m < 0.8:
                trigger = max(trigger, atr_val * 0.6)
                lock_ratio = max(lock_ratio, 0.32)
            elif vol_5m > 1.6:
                trigger = max(trigger, atr_val * 0.85)
                lock_ratio = max(lock_ratio, 0.30)
            else:
                trigger = max(trigger, atr_val * 0.7)
                lock_ratio = max(lock_ratio, 0.33)
            min_lock = max(min_lock, atr_val * 0.20)
            trigger_cap = _env_float("BE_TRAIL_MAX_TRIGGER_SCALP", 3.0)
            if pocket == "scalp_fast":
                trigger_cap = _env_float("BE_TRAIL_MAX_TRIGGER_SCALP_FAST", trigger_cap)
            trigger = min(trigger, trigger_cap)
        base_trigger = trigger
        base_lock_ratio = lock_ratio
        base_min_lock = min_lock
        base_cooldown_sec = cooldown_sec
        pocket_start_delay = base_start_delay.get(pocket, 45.0)
        pocket_max_delay = max_start_delay.get(pocket, 0.0)
        pocket_tp_cfg = tp_move_cfg.get(pocket, tp_move_cfg["macro"])
        # 経過時間に応じてロック強度を少し引き上げる
        def _age_scaled_lock(age_sec: float, base_ratio: float) -> float:
            if age_sec <= 0:
                return base_ratio
            bump = min(0.2, (age_sec / 180.0) * 0.2)  # 3分で+20%まで
            return min(0.65, base_ratio * (1.0 + bump))

        for tr in trades:
            trade_id = tr.get("trade_id")
            price = tr.get("price")
            side = tr.get("side")
            if not trade_id or price is None or not side:
                continue
            if _trade_matches_rollover_sl_strip(tr, str(pocket), rollover_ctx):
                continue

            client_id = str(tr.get("client_id") or "")
            if not client_id.startswith("qr-"):
                continue
            if pocket == "scalp" and client_id.startswith("qr-mirror-s5-"):
                continue
            thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
            strategy_tag = _strategy_tag_from_trade(tr, thesis)
            override = _strategy_override(strategy_cfg, strategy_tag)
            if _coerce_bool(override.get("enabled"), True) is False:
                continue
            apply_when_soft = _coerce_bool(override.get("apply_when_soft_tp"), apply_when_soft_default)
            if _soft_tp_mode(thesis) and not apply_when_soft:
                continue

            trigger = base_trigger
            lock_ratio = base_lock_ratio
            min_lock = base_min_lock
            cooldown_sec = base_cooldown_sec
            start_delay = pocket_start_delay
            max_delay = pocket_max_delay
            tp_move_enabled_local = tp_move_enabled
            tp_trigger = _coerce(pocket_tp_cfg.get("trigger"), 0.0)
            tp_buffer = _coerce(pocket_tp_cfg.get("buffer"), 0.6)
            if isinstance(override, dict):
                be_override = override.get("be_profile")
                if isinstance(be_override, dict):
                    trigger = _coerce(be_override.get("trigger_pips"), trigger)
                    lock_ratio = _coerce(be_override.get("lock_ratio"), lock_ratio)
                    min_lock = _coerce(be_override.get("min_lock_pips"), min_lock)
                    cooldown_sec = _coerce(be_override.get("cooldown_sec"), cooldown_sec)
                start_delay = _coerce(override.get("start_delay_sec"), start_delay)
                max_delay = _coerce(override.get("max_delay_sec"), max_delay)
                tp_override = override.get("tp_move")
                if isinstance(tp_override, dict):
                    if tp_override.get("enabled") is not None:
                        tp_move_enabled_local = _coerce_bool(tp_override.get("enabled"), tp_move_enabled_local)
                    tp_trigger = _coerce(tp_override.get("trigger_pips"), tp_trigger)
                    tp_buffer = _coerce(tp_override.get("buffer_pips"), tp_buffer)

            entry = float(price)
            trade_tp_info = tr.get("take_profit") or {}
            try:
                trade_tp = float(trade_tp_info.get("price"))
            except (TypeError, ValueError):
                trade_tp = None
            trade_sl_info = tr.get("stop_loss") or {}
            try:
                trade_sl = float(trade_sl_info.get("price"))
            except (TypeError, ValueError):
                trade_sl = None

            if side == "long":
                gain_pips = ((current_price or entry) - entry) * 100.0
            else:
                gain_pips = (entry - (current_price or entry)) * 100.0

            # micro/scalp: ATR/ボラに応じた開始遅延＋経過時間でロック強化
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            if pocket in {"micro", "scalp", "scalp_fast"} and opened_at:
                age_sec = max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds())
                if pocket == "micro":
                    delay_mult = 10.0
                elif pocket == "scalp_fast":
                    delay_mult = 5.0
                else:
                    delay_mult = 6.0
                start_delay = max(start_delay, atr_val * delay_mult)
                if vol_5m > 1.6:
                    start_delay *= 0.75
                elif vol_5m < 0.8:
                    start_delay *= 1.1
                if max_delay > 0.0:
                    start_delay = min(start_delay, max_delay)
                if age_sec < start_delay:
                    continue
                lock_ratio = _age_scaled_lock(age_sec - start_delay, lock_ratio)

            if gain_pips < trigger:
                continue

            last_record = _LAST_PROTECTIONS.get(trade_id) or {}
            last_ts = float(last_record.get("ts") or 0.0)
            if cooldown_sec > 0.0 and (now_ts - last_ts) < cooldown_sec:
                continue

            lock_pips = max(min_lock, gain_pips * lock_ratio)
            if side == "long":
                desired_sl = entry + lock_pips * pip
                if current_price is not None:
                    desired_sl = min(desired_sl, float(current_price) - 0.003)
                if trade_sl is not None and desired_sl <= trade_sl + 1e-6:
                    continue
            else:
                desired_sl = entry - lock_pips * pip
                if current_price is not None:
                    desired_sl = max(desired_sl, float(current_price) + 0.003)
                if trade_sl is not None and desired_sl >= trade_sl - 1e-6:
                    continue

            desired_sl = round(desired_sl, 3)
            tp_price = trade_tp
            if tp_move_enabled_local and current_price is not None:
                if gain_pips >= tp_trigger:
                    locked = (side == "long" and desired_sl > entry) or (side == "short" and desired_sl < entry)
                    if locked:
                        min_gap = max(0.3, tp_move_min_gap_default)
                        if side == "long":
                            target_tp = max(entry + min_gap * pip, float(current_price) + tp_buffer * pip)
                            if trade_tp is None or trade_tp - target_tp > 1e-6:
                                tp_price = round(target_tp, 3)
                        else:
                            target_tp = min(entry - min_gap * pip, float(current_price) - tp_buffer * pip)
                            if trade_tp is None or target_tp - trade_tp > 1e-6:
                                tp_price = round(target_tp, 3)

            _maybe_update_protections(
                trade_id,
                desired_sl,
                tp_price,
                context="dynamic_protection_v2",
                ref_price=current_price,
            )


async def set_trade_protections(
    trade_id: str,
    *,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> bool:
    """
    Legacy compatibility layer – update SL/TP for an open trade and report success.
    """
    service_result = await _order_manager_service_request_async(
        "/order/set_trade_protections",
        {
            "trade_id": trade_id,
            "sl_price": sl_price,
            "tp_price": tp_price,
        },
    )
    if service_result is not None:
        return bool(service_result)

    if not TRAILING_SL_ALLOWED and sl_price is not None:
        return False
    if not trade_id:
        return False
    try:
        _maybe_update_protections(
            trade_id,
            sl_price,
            tp_price,
            context="legacy_set_trade_protections",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] set_trade_protections failed trade=%s sl=%s tp=%s exc=%s",
            trade_id,
            sl_price,
            tp_price,
            exc,
        )
        return False


def _macro_partial_profile(
    fac_h4: Optional[dict],
    range_mode: bool,
) -> tuple[Tuple[float, float], Tuple[float, float]]:
    if range_mode:
        return (2.8, 4.6), (0.75, 0.18)
    if not fac_h4:
        return (3.2, 5.6), (0.78, 0.12)
    adx = float(fac_h4.get("adx", 0.0) or 0.0)
    ma10 = fac_h4.get("ma10", 0.0) or 0.0
    ma20 = fac_h4.get("ma20", 0.0) or 0.0
    atr_raw = fac_h4.get("atr") or 0.0
    atr_pips = atr_raw * 100.0
    gap_pips = abs(ma10 - ma20) * 100.0
    strength_ratio = gap_pips / atr_pips if atr_pips > 1e-6 else 0.0
    if strength_ratio >= 0.9 or adx >= 28.0:
        return (4.0, 6.8), (1.0, 0.0)
    if strength_ratio >= 0.6 or adx >= 24.0:
        return (3.6, 6.2), (0.98, 0.0)
    return (3.2, 5.2), (0.95, 0.0)


def plan_partial_reductions(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: Optional[dict] = None,
    *,
    range_mode: bool = False,
    stage_state: Optional[dict[str, dict[str, int]]] = None,
    pocket_profiles: Optional[dict[str, dict[str, float]]] = None,
    now: Optional[datetime] = None,
    threshold_overrides: Optional[dict[str, tuple[float, float]]] = None,
) -> list[tuple[str, str, int]]:
    price = fac_m1.get("close")
    if price is None:
        return []
    pip_scale = 100
    try:
        atr_m1 = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
    except Exception:
        atr_m1 = 0.0
    try:
        vol_5m = float(fac_m1.get("vol_5m") or 1.0)
    except Exception:
        vol_5m = 1.0
    current_time = _ensure_utc(now)
    actions: list[tuple[str, str, int]] = []
    policy = policy_bus.latest()
    pockets_policy = policy.pockets if policy else {}

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        thresholds = _PARTIAL_THRESHOLDS.get(pocket)
        fractions = _PARTIAL_FRACTIONS
        if pocket == "macro":
            thresholds, fractions = _macro_partial_profile(fac_h4, range_mode)
        elif range_mode:
            thresholds = _PARTIAL_THRESHOLDS_RANGE.get(pocket, thresholds)
        if thresholds and range_mode:
            try:
                atr_hint = float(atr_m1 or 0.0)
            except Exception:
                atr_hint = 0.0
            if atr_hint <= 2.0:
                thresholds = (1.6, 2.6) if pocket == "scalp" else (2.2, 3.6)
            elif atr_hint <= 3.0:
                thresholds = (min(thresholds[0], 2.2), min(thresholds[1], 3.8))
        if thresholds:
            thresholds = _scaled_thresholds(pocket, thresholds, atr_m1, vol_5m)
        plan = pockets_policy.get(pocket) if isinstance(pockets_policy, dict) else {}
        partial_plan = plan.get("partial_profile", {}) if isinstance(plan, dict) else {}
        min_units_override: Optional[int] = None
        if isinstance(partial_plan, dict):
            plan_thresholds = partial_plan.get("thresholds_pips")
            if isinstance(plan_thresholds, (list, tuple)) and plan_thresholds:
                try:
                    thresholds = [float(x) for x in plan_thresholds]
                except (TypeError, ValueError):
                    pass
            plan_fractions = partial_plan.get("fractions")
            if isinstance(plan_fractions, (list, tuple)) and plan_fractions:
                try:
                    fractions = tuple(float(x) for x in plan_fractions)
                except (TypeError, ValueError):
                    pass
            override_units = partial_plan.get("min_units")
            try:
                min_units_override = int(override_units) if override_units is not None else None
            except (TypeError, ValueError):
                min_units_override = None
        if not thresholds:
            continue
        pocket_stage = (stage_state or {}).get(pocket, {})
        pocket_profile = (pocket_profiles or {}).get(pocket, {})
        trades = info.get("open_trades") or []
        for tr in trades:
            trade_id = tr.get("trade_id")
            side = tr.get("side")
            entry = tr.get("price")
            units = int(tr.get("units", 0) or 0)
            if not trade_id or not side or entry is None or units == 0:
                continue
            thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
            if _soft_tp_mode(thesis):
                continue
            client_id = str(tr.get("client_id") or "")
            if not client_id.startswith("qr-"):
                continue
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            age_minutes = None
            if opened_at:
                age_minutes = max(0.0, (current_time - opened_at).total_seconds() / 60.0)
            if range_mode and pocket == "macro":
                if age_minutes is None or age_minutes < _PARTIAL_RANGE_MACRO_MIN_AGE_MIN:
                    continue
            current_stage = _PARTIAL_STAGE.get(trade_id, 0)
            min_hold_sec = _trade_min_hold_seconds(tr, pocket)
            age_seconds = _trade_age_seconds(tr, current_time)
            if age_seconds is not None and age_seconds < min_hold_sec:
                if range_mode:
                    thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
                    strategy_tag = thesis.get("strategy_tag") or tr.get("strategy_tag")
                    tags = {
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                    }
                    log_metric("partial_hold_guard", 1.0, tags=tags)
                continue
            gain_pips = 0.0
            stage_level = (pocket_stage or {}).get(side, 0)
            profile = pocket_profile or {}
            effective_thresholds = list(thresholds)
            if stage_level >= 3:
                effective_thresholds = [max(2.0, t * 0.75) for t in effective_thresholds]
            elif stage_level >= 1:
                effective_thresholds = [max(2.0, t * 0.85) for t in effective_thresholds]
            if profile.get("win_rate", 0.0) >= 0.55:
                effective_thresholds = [max(2.0, t * 0.9) for t in effective_thresholds]
            if profile.get("avg_loss_pips", 0.0) > 5.0:
                effective_thresholds = [max(1.5, t * 0.8) for t in effective_thresholds]
            thresholds_eff = tuple(effective_thresholds)
            if side == "long":
                gain_pips = (price - entry) * pip_scale
            else:
                gain_pips = (entry - price) * pip_scale
            if gain_pips <= thresholds_eff[0]:
                continue
            stage = 0
            for idx, threshold in enumerate(thresholds_eff, start=1):
                if gain_pips >= threshold:
                    stage = idx
            if stage <= current_stage:
                continue
            fraction_idx = min(stage - 1, len(fractions) - 1)
            fraction = fractions[fraction_idx]
            reduce_units = int(abs(units) * fraction)
            min_units_threshold = min_units_override if min_units_override is not None else _PARTIAL_MIN_UNITS
            if reduce_units < min_units_threshold:
                continue
            reduce_units = min(reduce_units, abs(units))
            sign = 1 if units > 0 else -1
            actions.append((pocket, trade_id, sign * reduce_units))
            _PARTIAL_STAGE[trade_id] = stage
    return actions


async def market_order(
    instrument: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp", "scalp_fast", "manual"],
    *,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
    stage_index: Optional[int] = None,
    arbiter_final: bool = False,
) -> Optional[str]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns trade id（約定時のみ）。未約定（submitted）や失敗は None。
    """
    if not strategy_tag:
        strategy_tag = _strategy_tag_from_client_id(client_order_id)
    entry_thesis = _ensure_entry_intent_payload(
        units=units,
        confidence=confidence,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    entry_probability = _entry_probability_value(confidence, entry_thesis)
    if not strategy_tag and isinstance(entry_thesis, dict):
        strategy_tag = _strategy_tag_from_thesis(entry_thesis)
    preserve_strategy_intent = (
        _ORDER_MANAGER_PRESERVE_STRATEGY_INTENT
        and not reduce_only
        and (pocket or "").lower() != "manual"
    )
    if preserve_strategy_intent:
        scaled_units, probability_reason = _probability_scaled_units(
            units,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_probability=entry_probability,
        )
        if probability_reason is not None:
            reason_note = probability_reason
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id or "",
                note=f"entry_probability:{reason_note}",
            )
            if _should_persist_preservice_order_log():
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="entry_probability_reject",
                    attempt=0,
                    request_payload={
                        "entry_probability": entry_probability,
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                        "entry_probability_reject_reason": reason_note,
                    },
                )
            else:
                _cache_order_status(
                    ts=datetime.now(timezone.utc).isoformat(),
                    client_order_id=client_order_id,
                    status="entry_probability_reject",
                    attempt=0,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    error_message=reason_note,
                    request_payload={
                        "entry_probability": entry_probability,
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                        "entry_probability_reject_reason": reason_note,
                    },
                )
            log_metric(
                "order_probability_reject",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": reason_note,
                },
            )
            return None
        if scaled_units != units:
            _console_order_log(
                "OPEN_SCALE",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side="buy" if units > 0 else "sell",
                units=scaled_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"probability_scale:{entry_probability:.3f}" if entry_probability is not None else "probability_scale",
            )
            if _should_persist_preservice_order_log():
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=scaled_units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="probability_scaled",
                    attempt=0,
                    request_payload={
                        "entry_probability": entry_probability,
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                        "scaled_units": scaled_units,
                        "raw_units": abs(units) if units else 0,
                    },
                )
            log_metric(
                "order_probability_scale",
                max(
                    0.0,
                    min(
                        1.0,
                        float(entry_probability)
                        if entry_probability is not None
                        else 1.0,
                    ),
                ),
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                },
            )
            units = scaled_units
    service_result = await _order_manager_service_request_async(
        "/order/market_order",
        {
            "instrument": instrument,
            "units": units,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "pocket": pocket,
            "client_order_id": client_order_id,
            "strategy_tag": strategy_tag,
            "reduce_only": reduce_only,
            "entry_thesis": entry_thesis,
            "meta": meta,
            "confidence": confidence,
            "entry_probability": entry_probability,
            "stage_index": stage_index,
            "arbiter_final": arbiter_final,
        },
    )
    if service_result is not None:
        if service_result is None:
            return None
        return str(service_result) if service_result is not None else None

    sl_disabled = stop_loss_disabled_for_pocket(pocket)
    if strategy_tag is not None:
        strategy_tag = str(strategy_tag)
        if not strategy_tag:
            strategy_tag = None
    else:
        strategy_tag = _strategy_tag_from_client_id(client_order_id)
    if (
        sl_disabled
        and isinstance(strategy_tag, str)
        and strategy_tag.strip().lower().startswith("scalp_ping_5s_b")
    ):
        sl_disabled = False
    thesis_disable_hard_stop = _disable_hard_stop_by_strategy(
        strategy_tag,
        pocket,
        entry_thesis if isinstance(entry_thesis, dict) else None,
    )
    if thesis_disable_hard_stop:
        sl_price = None
    # client_order_id は必須。欠損したまま送れば OANDA 側で空白になり、追跡不能となる。
    if not client_order_id:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag=strategy_tag or "unknown",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id="",
            note="missing_client_order_id",
        )
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=None,
            status="missing_client_order_id",
            attempt=0,
            request_payload={
                "strategy_tag": strategy_tag,
                "meta": meta,
                "entry_thesis": entry_thesis,
            },
        )
        log_metric(
            "order_missing_client_id",
            1.0,
            tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
        )
        return None
    virtual_sl_price: Optional[float] = None
    virtual_tp_price: Optional[float] = None
    side_label = "buy" if units > 0 else "sell"
    order_t0 = time.monotonic()
    entry_deadline_sec = _entry_execution_deadline_sec(pocket) if not reduce_only else 0.0

    def _merge_virtual(payload: Optional[dict] = None) -> dict:
        base: dict = {}
        if virtual_sl_price is not None:
            base["virtual_sl_price"] = virtual_sl_price
        if virtual_tp_price is not None:
            base["virtual_tp_price"] = virtual_tp_price
        if payload:
            base.update(payload)
        return base

    def log_order(**kwargs):
        kwargs["request_payload"] = _merge_virtual(kwargs.get("request_payload"))
        return _log_order(**kwargs)
    thesis_sl_pips: Optional[float] = None
    thesis_tp_pips: Optional[float] = None
    soft_tp = _soft_tp_mode(entry_thesis)
    if stage_index is None and isinstance(entry_thesis, dict):
        try:
            raw_stage = entry_thesis.get("stage_index")
            if raw_stage is not None:
                stage_index = int(raw_stage)
        except Exception:
            stage_index = None
    if isinstance(entry_thesis, dict):
        raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
        if raw_tag and not strategy_tag:
            strategy_tag = str(raw_tag)
        try:
            value = entry_thesis.get("sl_pips")
            if value is not None:
                thesis_sl_pips = float(value)
        except (TypeError, ValueError):
            thesis_sl_pips = None
    if thesis_disable_hard_stop:
        thesis_sl_pips = None
        if isinstance(entry_thesis, dict):
            entry_thesis = dict(entry_thesis)
            entry_thesis.pop("sl_pips", None)
            entry_thesis.pop("sl_before", None)
            entry_thesis.pop("dynamic_sl_applied", None)
            entry_thesis.pop("min_rr_adjusted", None)
            entry_thesis.pop("sl_cap_applied", None)
            entry_thesis.pop("forecast_execution", None)
    try:
        value = entry_thesis.get("tp_pips") or entry_thesis.get("target_tp_pips")
        if value is not None:
            thesis_tp_pips = float(value)
    except (TypeError, ValueError):
        thesis_tp_pips = None
    if soft_tp:
        tp_price = None
        thesis_tp_pips = None

    # strategy_tag は必須: entry_thesis から補完しても欠損なら拒否
    raw_tag = None
    if isinstance(entry_thesis, dict):
        raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
    if not strategy_tag and raw_tag:
        strategy_tag = str(raw_tag)
    if not strategy_tag:
        inferred = _strategy_tag_from_client_id(client_order_id)
        if inferred:
            strategy_tag = inferred
    if isinstance(entry_thesis, dict) and strategy_tag and not entry_thesis.get("strategy_tag"):
        entry_thesis = dict(entry_thesis)
        entry_thesis["strategy_tag"] = strategy_tag
    # In non-hedging (netting) accounts, an opposite-direction entry can net out / close
    # the user's manual trades. In hedging mode (positionFill=OPEN_ONLY) this cannot happen,
    # so allow opposite-direction entries and rely on exposure/risk guards instead.
    if (
        _BLOCK_MANUAL_NETTING
        and not HEDGING_ENABLED
        and not reduce_only
        and (pocket or "").lower() != "manual"
    ):
        manual_net, manual_trades = _manual_net_units()
        if manual_trades > 0 or manual_net != 0:
            block = False
            note = "manual_netting_block"
            if manual_net == 0:
                # Conservative: if manual trades exist but net is zero, block all new entries.
                block = True
                note = "manual_netting_block_zero_net"
            elif manual_net > 0 and units < 0:
                block = True
            elif manual_net < 0 and units > 0:
                block = True
            if block:
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    request_payload={
                        "manual_net_units": manual_net,
                        "manual_trades": manual_trades,
                        "strategy_tag": strategy_tag,
                    },
                )
                log_metric(
                    "order_manual_netting_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "manual_net_units": str(manual_net),
                    },
                )
                return None
    if not reduce_only and pocket != "manual":
        entry_thesis = _apply_default_entry_thesis_tfs(entry_thesis, pocket)
    if isinstance(entry_thesis, dict) and not reduce_only:
        entry_thesis = attach_section_axis(entry_thesis, pocket=pocket)
    if isinstance(entry_thesis, dict):
        entry_thesis = _augment_entry_thesis_regime(entry_thesis, pocket)
        entry_thesis = _augment_entry_thesis_flags(entry_thesis)
        entry_thesis = _augment_entry_thesis_policy_generation(
            entry_thesis,
            reduce_only=reduce_only,
        )

    trace_enabled = os.getenv("ORDER_TRACE_PROGRESS", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }

    def _trace(step: str) -> None:
        if not trace_enabled:
            return
        logging.info(
            "[ORDER_TRACE] step=%s pocket=%s strategy=%s units=%s client=%s",
            step,
            pocket,
            strategy_tag or "-",
            units,
            client_order_id or "-",
        )

    # strategy_tag も必須。entry_thesis から補完した上で欠損なら拒否。
    if not strategy_tag:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag="missing",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="missing_strategy_tag",
        )
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="missing_strategy_tag",
            attempt=0,
            request_payload={"meta": meta, "entry_thesis": entry_thesis},
        )
        log_metric(
            "order_missing_strategy_tag",
            1.0,
            tags={"pocket": pocket, "strategy": "missing"},
        )
        return None

    if _reject_entry_by_control(strategy_tag, pocket=pocket):
        note = "strategy_control_entry_disabled"
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=side_label,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note=note,
        )
        log_order(
            pocket=pocket,
            instrument=instrument,
            side=side_label,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status=note,
            attempt=0,
            request_payload={"strategy_tag": strategy_tag, "meta": meta, "entry_thesis": entry_thesis},
        )
        log_metric(
            "order_blocked_by_strategy_control",
            1.0,
            tags={"pocket": pocket, "strategy": strategy_tag, "action": "entry"},
        )
        return None

    if not reduce_only and pocket != "manual":
        policy_allowed, policy_reason, policy_details = _policy_gate_allows_entry(
            pocket,
            side_label,
            strategy_tag,
            reduce_only=reduce_only,
        )
        if not policy_allowed:
            reason = policy_reason or "policy_block"
            _console_order_log(
                "OPEN_REJECT",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=reason,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="policy_block",
                attempt=0,
                request_payload={
                    "reason": reason,
                    "policy": policy_details,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
            log_metric(
                "entry_policy_block",
                1.0,
                tags={"pocket": pocket, "strategy": strategy_tag or "unknown", "reason": reason},
            )
            return None

    log_order(
        pocket=pocket,
        instrument=instrument,
        side=side_label,
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status="preflight_start",
        attempt=0,
        stage_index=stage_index,
        request_payload={
            "note": "preflight_start",
            "arbiter_final": arbiter_final,
            "reduce_only": reduce_only,
        },
    )
    _trace("preflight_start")


    # Perf guard (PF/win-rate gate). Support per-strategy overrides by passing env_prefix
    # via meta/entry_thesis so a multi-strategy worker can tune each strategy independently.
    meta_env_prefix = None
    entry_env_prefix = None
    if isinstance(meta, dict):
        meta_env_prefix = _coerce_env_prefix(
            meta.get("env_prefix") or meta.get("ENV_PREFIX")
        )
    if isinstance(entry_thesis, dict):
        entry_env_prefix = _coerce_env_prefix(
            entry_thesis.get("env_prefix") or entry_thesis.get("ENV_PREFIX")
        )
    env_prefix = _resolve_env_prefix_for_order(
        entry_env_prefix,
        meta_env_prefix,
        strategy_tag,
    )
    if entry_env_prefix and meta_env_prefix and entry_env_prefix != meta_env_prefix:
        logging.debug(
            "[ORDER] env_prefix mismatch pocket=%s strategy=%s meta=%s entry=%s",
            pocket,
            strategy_tag,
            meta_env_prefix,
            entry_env_prefix,
        )
        logging.debug(
            "[ORDER] env_prefix resolved env_prefix=%s",
            env_prefix,
        )

    if pocket != "manual":
        try:
            pocket_decision = perf_guard.is_pocket_allowed(pocket, env_prefix=env_prefix)
        except Exception:
            pocket_decision = None
        if pocket_decision is not None and not pocket_decision.allowed:
            note = f"perf_block_pocket:{pocket_decision.reason}"
            _console_order_log(
                "OPEN_REJECT",
                pocket=pocket,
                strategy_tag=str(strategy_tag or "unknown"),
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=note,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="perf_block_pocket",
                attempt=0,
                stage_index=stage_index,
                request_payload={"pocket": pocket, "meta": meta, "entry_thesis": entry_thesis},
            )
            log_metric(
                "order_perf_block_pocket",
                1.0,
                tags={
                    "pocket": pocket,
                    "reason": pocket_decision.reason,
                },
            )
            return None
    if pocket != "manual" and strategy_tag:
        _trace("perf_guard")
        try:
            current_hour = datetime.now(timezone.utc).hour
        except Exception:
            current_hour = None
        decision = perf_guard.is_allowed(
            str(strategy_tag),
            pocket,
            hour=current_hour,
            side=side_label,
            env_prefix=env_prefix,
        )
        if not decision.allowed:
            note = f"perf_block:{decision.reason}"
            _console_order_log(
                "OPEN_REJECT",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=note,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="perf_block",
                attempt=0,
                stage_index=stage_index,
                request_payload={"strategy_tag": strategy_tag, "meta": meta, "entry_thesis": entry_thesis},
            )
            log_metric(
                "order_perf_block",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": str(strategy_tag),
                    "reason": decision.reason,
                },
            )
            return None

    if not reduce_only and pocket != "manual":
        decision = profit_guard.is_allowed(pocket, strategy_tag=strategy_tag, env_prefix=env_prefix)
        if not decision.allowed:
            range_active = False
            if _PROFIT_GUARD_BYPASS_RANGE and pocket in {"scalp", "micro"}:
                range_active = _range_active_for_entry()
            if range_active:
                log_metric(
                    "order_profit_guard_bypass",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "reason": decision.reason,
                        "mode": "range",
                    },
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="profit_guard_bypass",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "profit_guard_reason": decision.reason,
                        "range_active": True,
                    },
                )
            else:
                note = f"profit_guard:{decision.reason}"
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="profit_guard",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={"strategy_tag": strategy_tag, "meta": meta, "entry_thesis": entry_thesis},
                )
                log_metric(
                    "order_profit_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "reason": decision.reason,
                    },
                )
                return None

    # LLM brain gate (per-strategy human-like filter)
    if (
        not reduce_only
        and pocket != "manual"
        and not preserve_strategy_intent
        and _ORDER_MANAGER_BRAIN_GATE_ENABLED
    ):
        try:
            brain_decision = brain.decide(
                strategy_tag=strategy_tag,
                pocket=pocket,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
                meta=meta if isinstance(meta, dict) else None,
                confidence=confidence,
            )
        except Exception as exc:
            brain_decision = None
            logging.debug("[BRAIN] decision failed: %s", exc)
        if brain_decision is not None:
            if not brain_decision.allowed:
                note = f"brain_block:{brain_decision.reason}"
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="brain_block",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "brain_reason": brain_decision.reason,
                        "brain_action": brain_decision.action,
                    },
                )
                log_metric(
                    "order_brain_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "reason": brain_decision.reason,
                    },
                )
                return None
            if 0.0 < brain_decision.scale < 1.0:
                scaled_units = int(round(abs(units) * brain_decision.scale))
                min_allowed = min_units_for_strategy(strategy_tag, pocket=pocket)
                if scaled_units < min_allowed:
                    note = f"brain_scale_below_min:{brain_decision.reason}"
                    _console_order_log(
                        "OPEN_REJECT",
                        pocket=pocket,
                        strategy_tag=str(strategy_tag or "unknown"),
                        side=side_label,
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        note=note,
                    )
                    log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side=side_label,
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="brain_scale_below_min",
                        attempt=0,
                        stage_index=stage_index,
                        request_payload={
                            "strategy_tag": strategy_tag,
                            "meta": meta,
                            "entry_thesis": entry_thesis,
                            "brain_reason": brain_decision.reason,
                            "brain_action": brain_decision.action,
                            "brain_scale": brain_decision.scale,
                            "scaled_units": scaled_units,
                            "min_units": min_allowed,
                        },
                    )
                    log_metric(
                        "order_brain_block",
                        1.0,
                        tags={
                            "pocket": pocket,
                            "strategy": str(strategy_tag or "unknown"),
                            "reason": "scale_below_min",
                        },
                    )
                    return None
                if scaled_units > 0 and scaled_units != abs(units):
                    sign = 1 if units > 0 else -1
                    units = int(sign * scaled_units)
                    log_metric(
                        "order_brain_scale",
                        1.0,
                        tags={
                            "pocket": pocket,
                            "strategy": str(strategy_tag or "unknown"),
                            "reason": brain_decision.reason,
                            "scale": f"{brain_decision.scale:.2f}",
                        },
                    )

    # Probabilistic forecast gate (scikit-learn, offline bundle)
    if (
        not reduce_only
        and pocket != "manual"
        and not preserve_strategy_intent
        and _ORDER_MANAGER_FORECAST_GATE_ENABLED
    ):
        try:
            forecast_meta: dict[str, Any] = {"instrument": instrument}
            if isinstance(meta, dict):
                forecast_meta.update(meta)
            fc_decision = _forecast_decide_with_service(
                strategy_tag=strategy_tag,
                pocket=pocket,
                side=side_label,
                units=units,
                entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
                meta=forecast_meta,
            )
        except Exception as exc:
            fc_decision = None
            logging.debug("[FORECAST] decision failed: %s", exc)
        if fc_decision is not None:
            if not fc_decision.allowed:
                note = f"forecast_block:{fc_decision.reason}"
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="forecast_block",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "forecast_reason": fc_decision.reason,
                        "forecast_horizon": fc_decision.horizon,
                        "forecast_source": fc_decision.source,
                        "forecast_style": fc_decision.style,
                        "forecast_edge": fc_decision.edge,
                        "forecast_p_up": fc_decision.p_up,
                        "forecast_rebound_probability": fc_decision.rebound_probability,
                        "forecast_trend_strength": fc_decision.trend_strength,
                        "forecast_range_pressure": fc_decision.range_pressure,
                        "forecast_future_flow": fc_decision.future_flow,
                        "forecast_volatility_state": fc_decision.volatility_state,
                        "forecast_trend_state": fc_decision.trend_state,
                        "forecast_range_state": fc_decision.range_state,
                        "forecast_volatility_rank": fc_decision.volatility_rank,
                        "forecast_regime_score": fc_decision.regime_score,
                        "forecast_leading_indicator": fc_decision.leading_indicator,
                        "forecast_leading_indicator_strength": fc_decision.leading_indicator_strength,
                        "forecast_tf_confluence_score": fc_decision.tf_confluence_score,
                        "forecast_tf_confluence_count": fc_decision.tf_confluence_count,
                        "forecast_tf_confluence_horizons": fc_decision.tf_confluence_horizons,
                        "forecast_expected_pips": fc_decision.expected_pips,
                        "forecast_target_reach_prob": fc_decision.target_reach_prob,
                        "forecast_range_low_pips": fc_decision.range_low_pips,
                        "forecast_range_high_pips": fc_decision.range_high_pips,
                        "forecast_range_sigma_pips": fc_decision.range_sigma_pips,
                        "forecast_range_low_price": fc_decision.range_low_price,
                        "forecast_range_high_price": fc_decision.range_high_price,
                        "forecast_feature_ts": fc_decision.feature_ts,
                    },
                )
                log_metric(
                    "order_forecast_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "reason": fc_decision.reason,
                        "horizon": fc_decision.horizon,
                        "source": str(fc_decision.source or "unknown"),
                        "style": str(fc_decision.style or "n/a"),
                    },
                )
                return None
            if 0.0 < fc_decision.scale < 1.0:
                scaled_units = int(round(abs(units) * fc_decision.scale))
                min_allowed = min_units_for_strategy(strategy_tag, pocket=pocket)
                if scaled_units < min_allowed:
                    note = f"forecast_scale_below_min:{fc_decision.reason}"
                    _console_order_log(
                        "OPEN_REJECT",
                        pocket=pocket,
                        strategy_tag=str(strategy_tag or "unknown"),
                        side=side_label,
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        note=note,
                    )
                    log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side=side_label,
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="forecast_scale_below_min",
                        attempt=0,
                        stage_index=stage_index,
                        request_payload={
                            "strategy_tag": strategy_tag,
                            "meta": meta,
                            "entry_thesis": entry_thesis,
                            "forecast_reason": fc_decision.reason,
                            "forecast_horizon": fc_decision.horizon,
                            "forecast_source": fc_decision.source,
                            "forecast_style": fc_decision.style,
                            "forecast_edge": fc_decision.edge,
                            "forecast_p_up": fc_decision.p_up,
                            "forecast_rebound_probability": fc_decision.rebound_probability,
                            "forecast_trend_strength": fc_decision.trend_strength,
                            "forecast_range_pressure": fc_decision.range_pressure,
                            "forecast_future_flow": fc_decision.future_flow,
                            "forecast_volatility_state": fc_decision.volatility_state,
                            "forecast_trend_state": fc_decision.trend_state,
                            "forecast_range_state": fc_decision.range_state,
                            "forecast_volatility_rank": fc_decision.volatility_rank,
                            "forecast_regime_score": fc_decision.regime_score,
                            "forecast_leading_indicator": fc_decision.leading_indicator,
                            "forecast_leading_indicator_strength": fc_decision.leading_indicator_strength,
                            "forecast_tf_confluence_score": fc_decision.tf_confluence_score,
                            "forecast_tf_confluence_count": fc_decision.tf_confluence_count,
                            "forecast_tf_confluence_horizons": fc_decision.tf_confluence_horizons,
                            "forecast_target_reach_prob": fc_decision.target_reach_prob,
                            "forecast_range_low_pips": fc_decision.range_low_pips,
                            "forecast_range_high_pips": fc_decision.range_high_pips,
                            "forecast_range_sigma_pips": fc_decision.range_sigma_pips,
                            "forecast_range_low_price": fc_decision.range_low_price,
                            "forecast_range_high_price": fc_decision.range_high_price,
                            "forecast_scale": fc_decision.scale,
                            "scaled_units": scaled_units,
                            "min_units": min_allowed,
                        },
                    )
                    log_metric(
                        "order_forecast_block",
                        1.0,
                        tags={
                            "pocket": pocket,
                            "strategy": str(strategy_tag or "unknown"),
                            "reason": "scale_below_min",
                            "horizon": fc_decision.horizon,
                            "source": str(fc_decision.source or "unknown"),
                            "style": str(fc_decision.style or "n/a"),
                        },
                    )
                    return None
                if scaled_units > 0 and scaled_units != abs(units):
                    sign = 1 if units > 0 else -1
                    units = int(sign * scaled_units)
                    log_metric(
                        "order_forecast_scale",
                        1.0,
                        tags={
                            "pocket": pocket,
                            "strategy": str(strategy_tag or "unknown"),
                            "reason": fc_decision.reason,
                            "horizon": fc_decision.horizon,
                            "source": str(fc_decision.source or "unknown"),
                            "style": str(fc_decision.style or "n/a"),
                            "scale": f"{fc_decision.scale:.2f}",
                        },
                    )
            edge_strength = max(0.0, min(1.0, (float(fc_decision.edge) - 0.5) / 0.5))
            tp_hint = _as_float(fc_decision.tp_pips_hint)
            sl_cap = _as_float(fc_decision.sl_pips_cap)
            rr_floor = _as_float(fc_decision.rr_floor)
            base_tp_hint = thesis_tp_pips
            base_sl_hint = thesis_sl_pips

            if isinstance(entry_thesis, dict):
                entry_thesis = dict(entry_thesis)
            elif any(v is not None for v in (tp_hint, sl_cap, rr_floor)):
                entry_thesis = {}
            if isinstance(entry_thesis, dict):
                forecast_meta = dict(entry_thesis.get("forecast") or {})
                forecast_meta.update(
                    {
                        "reason": fc_decision.reason,
                        "horizon": fc_decision.horizon,
                        "source": fc_decision.source,
                        "style": fc_decision.style,
                        "edge": round(float(fc_decision.edge), 6),
                        "p_up": round(float(fc_decision.p_up), 6),
                        "rebound_probability": (
                            round(float(fc_decision.rebound_probability), 6)
                            if fc_decision.rebound_probability is not None
                            else None
                        ),
                        "expected_pips": (
                            round(float(fc_decision.expected_pips), 4)
                            if fc_decision.expected_pips is not None
                            else None
                        ),
                        "anchor_price": (
                            round(float(fc_decision.anchor_price), 5)
                            if fc_decision.anchor_price is not None
                            else None
                        ),
                        "target_price": (
                            round(float(fc_decision.target_price), 5)
                            if fc_decision.target_price is not None
                            else None
                        ),
                        "range_low_pips": (
                            round(float(fc_decision.range_low_pips), 4)
                            if fc_decision.range_low_pips is not None
                            else None
                        ),
                        "range_high_pips": (
                            round(float(fc_decision.range_high_pips), 4)
                            if fc_decision.range_high_pips is not None
                            else None
                        ),
                        "range_sigma_pips": (
                            round(float(fc_decision.range_sigma_pips), 4)
                            if fc_decision.range_sigma_pips is not None
                            else None
                        ),
                        "range_low_price": (
                            round(float(fc_decision.range_low_price), 5)
                            if fc_decision.range_low_price is not None
                            else None
                        ),
                        "range_high_price": (
                            round(float(fc_decision.range_high_price), 5)
                            if fc_decision.range_high_price is not None
                            else None
                        ),
                        "tp_pips_hint": (
                            round(float(fc_decision.tp_pips_hint), 4)
                            if fc_decision.tp_pips_hint is not None
                            else None
                        ),
                        "target_reach_prob": (
                            round(float(fc_decision.target_reach_prob), 6)
                            if fc_decision.target_reach_prob is not None
                            else None
                        ),
                        "sl_pips_cap": (
                            round(float(fc_decision.sl_pips_cap), 4)
                            if fc_decision.sl_pips_cap is not None
                            else None
                        ),
                        "rr_floor": (
                            round(float(fc_decision.rr_floor), 4)
                            if fc_decision.rr_floor is not None
                            else None
                        ),
                        "trend_strength": (
                            round(float(fc_decision.trend_strength), 6)
                            if fc_decision.trend_strength is not None
                            else None
                        ),
                        "range_pressure": (
                            round(float(fc_decision.range_pressure), 6)
                            if fc_decision.range_pressure is not None
                            else None
                        ),
                        "future_flow": fc_decision.future_flow,
                        "volatility_state": fc_decision.volatility_state,
                        "trend_state": fc_decision.trend_state,
                        "range_state": fc_decision.range_state,
                        "volatility_rank": fc_decision.volatility_rank,
                        "regime_score": fc_decision.regime_score,
                        "leading_indicator": fc_decision.leading_indicator,
                        "leading_indicator_strength": fc_decision.leading_indicator_strength,
                        "tf_confluence_score": (
                            round(float(fc_decision.tf_confluence_score), 6)
                            if fc_decision.tf_confluence_score is not None
                            else None
                        ),
                        "tf_confluence_count": (
                            int(fc_decision.tf_confluence_count)
                            if fc_decision.tf_confluence_count is not None
                            else None
                        ),
                        "tf_confluence_horizons": fc_decision.tf_confluence_horizons,
                        "feature_ts": fc_decision.feature_ts,
                        "edge_strength": round(edge_strength, 6),
                    }
                )
                if tp_hint is not None and tp_hint > 0.0:
                    forecast_meta["tp_pips_hint"] = round(float(tp_hint), 4)
                if sl_cap is not None and sl_cap > 0.0:
                    forecast_meta["sl_pips_cap"] = round(float(sl_cap), 4)
                if rr_floor is not None and rr_floor > 0.0:
                    forecast_meta["rr_floor"] = round(float(rr_floor), 4)
                entry_thesis["forecast"] = forecast_meta

            if tp_hint is not None and tp_hint > 0.0:
                blend = max(0.25, min(0.9, 0.35 + 0.45 * edge_strength))
                if thesis_tp_pips is None or thesis_tp_pips <= 0.0:
                    thesis_tp_pips = float(tp_hint)
                else:
                    thesis_tp_pips = (1.0 - blend) * float(thesis_tp_pips) + blend * float(tp_hint)
                thesis_tp_pips = max(0.5, float(thesis_tp_pips))

            if sl_cap is not None and sl_cap > 0.0 and thesis_sl_pips is not None and thesis_sl_pips > sl_cap:
                thesis_sl_pips = float(sl_cap)

            if (
                rr_floor is not None
                and rr_floor > 0.0
                and thesis_sl_pips is not None
                and thesis_tp_pips is not None
            ):
                min_tp = float(thesis_sl_pips) * float(rr_floor)
                if thesis_tp_pips < min_tp:
                    thesis_tp_pips = min_tp

            if isinstance(entry_thesis, dict):
                if thesis_tp_pips is not None and thesis_tp_pips > 0.0:
                    entry_thesis["tp_pips"] = round(float(thesis_tp_pips), 3)
                if thesis_sl_pips is not None and thesis_sl_pips > 0.0:
                    entry_thesis["sl_pips"] = round(float(thesis_sl_pips), 3)
                entry_thesis["forecast_execution"] = {
                    "tp_before": round(float(base_tp_hint), 4) if base_tp_hint is not None else None,
                    "tp_after": round(float(thesis_tp_pips), 4) if thesis_tp_pips is not None else None,
                    "sl_before": round(float(base_sl_hint), 4) if base_sl_hint is not None else None,
                    "sl_after": round(float(thesis_sl_pips), 4) if thesis_sl_pips is not None else None,
                    "tp_hint": round(float(tp_hint), 4) if tp_hint is not None else None,
                    "target_reach_prob": (
                        round(float(fc_decision.target_reach_prob), 6)
                        if fc_decision.target_reach_prob is not None
                        else None
                    ),
                    "sl_cap": round(float(sl_cap), 4) if sl_cap is not None else None,
                    "rr_floor": round(float(rr_floor), 4) if rr_floor is not None else None,
                    "edge_strength": round(edge_strength, 6),
                    "rebound_probability": (
                        round(float(fc_decision.rebound_probability), 6)
                        if fc_decision.rebound_probability is not None
                        else None
                    ),
                    "anchor_price": round(float(fc_decision.anchor_price), 5)
                    if fc_decision.anchor_price is not None
                    else None,
                    "target_price": round(float(fc_decision.target_price), 5)
                    if fc_decision.target_price is not None
                    else None,
                    "range_low_pips": round(float(fc_decision.range_low_pips), 4)
                    if fc_decision.range_low_pips is not None
                    else None,
                    "range_high_pips": round(float(fc_decision.range_high_pips), 4)
                    if fc_decision.range_high_pips is not None
                    else None,
                    "range_sigma_pips": round(float(fc_decision.range_sigma_pips), 4)
                    if fc_decision.range_sigma_pips is not None
                    else None,
                    "range_low_price": round(float(fc_decision.range_low_price), 5)
                    if fc_decision.range_low_price is not None
                    else None,
                    "range_high_price": round(float(fc_decision.range_high_price), 5)
                    if fc_decision.range_high_price is not None
                    else None,
                    "tf_confluence_score": (
                        round(float(fc_decision.tf_confluence_score), 6)
                        if fc_decision.tf_confluence_score is not None
                        else None
                    ),
                    "tf_confluence_count": (
                        int(fc_decision.tf_confluence_count)
                        if fc_decision.tf_confluence_count is not None
                        else None
                    ),
                    "tf_confluence_horizons": fc_decision.tf_confluence_horizons,
                }

            if base_tp_hint != thesis_tp_pips or base_sl_hint != thesis_sl_pips:
                log_metric(
                    "order_forecast_exec_profile",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "horizon": fc_decision.horizon,
                        "source": str(fc_decision.source or "unknown"),
                        "style": str(fc_decision.style or "n/a"),
                    },
                )

    # Pattern gate (pattern_book-driven block/scale; strategy worker opt-in)
    if (
        not reduce_only
        and pocket != "manual"
        and not preserve_strategy_intent
        and _ORDER_MANAGER_PATTERN_GATE_ENABLED
    ):
        try:
            pattern_decision = pattern_gate.decide(
                strategy_tag=strategy_tag,
                pocket=pocket,
                side=side_label,
                units=units,
                entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
                meta=meta if isinstance(meta, dict) else None,
            )
        except Exception as exc:
            pattern_decision = None
            logging.debug("[PATTERN_GATE] decision failed: %s", exc)
        if pattern_decision is not None:
            if isinstance(entry_thesis, dict):
                entry_thesis = dict(entry_thesis)
                entry_thesis["pattern_gate"] = pattern_decision.to_payload()
            if not pattern_decision.allowed:
                note = f"pattern_block:{pattern_decision.reason}"
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="pattern_block",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "pattern_gate": pattern_decision.to_payload(),
                    },
                )
                log_metric(
                    "order_pattern_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "reason": pattern_decision.reason,
                        "quality": pattern_decision.quality,
                        "source": pattern_decision.source,
                    },
                )
                return None

            if pattern_decision.scale != 1.0:
                scaled_units = int(round(abs(units) * pattern_decision.scale))
                min_allowed = min_units_for_strategy(strategy_tag, pocket=pocket)
                pattern_scale_floored = False
                if scaled_units < min_allowed:
                    if (
                        _PATTERN_GATE_SCALE_TO_MIN_UNITS
                        and min_allowed > 0
                    ):
                        scaled_units = min_allowed
                        pattern_scale_floored = True
                    else:
                        note = f"pattern_scale_below_min:{pattern_decision.reason}"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=str(strategy_tag or "unknown"),
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=note,
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status="pattern_scale_below_min",
                            attempt=0,
                            stage_index=stage_index,
                            request_payload={
                                "strategy_tag": strategy_tag,
                                "meta": meta,
                                "entry_thesis": entry_thesis,
                                "pattern_gate": pattern_decision.to_payload(),
                                "scaled_units": scaled_units,
                                "min_units": min_allowed,
                            },
                        )
                        log_metric(
                            "order_pattern_block",
                            1.0,
                            tags={
                                "pocket": pocket,
                                "strategy": str(strategy_tag or "unknown"),
                                "reason": "scale_below_min",
                                "quality": pattern_decision.quality,
                                "source": pattern_decision.source,
                            },
                        )
                        return None
                if scaled_units > 0 and scaled_units != abs(units):
                    sign = 1 if units > 0 else -1
                    units = int(sign * scaled_units)
                    log_metric(
                        "order_pattern_scale",
                        float(abs(scaled_units) / abs(units)) if abs(units) > 0 else 1.0,
                        tags={
                            "pocket": pocket,
                            "strategy": str(strategy_tag or "unknown"),
                            "reason": "floor_to_min"
                            if pattern_scale_floored
                            else pattern_decision.reason,
                            "quality": pattern_decision.quality,
                            "source": pattern_decision.source,
                        },
                    )

    exec_cfg = None
    if isinstance(entry_thesis, dict):
        exec_cfg = entry_thesis.get("execution")
    if exec_cfg is None and isinstance(meta, dict):
        exec_cfg = meta.get("execution")
    if (
        not reduce_only
        and isinstance(exec_cfg, dict)
        and exec_cfg.get("order_policy") == "market_guarded"
    ):
        ideal = _as_float(exec_cfg.get("ideal_entry"))
        chase_max = _as_float(exec_cfg.get("chase_max"))
        price_hint = (
            _as_float((meta or {}).get("entry_price"))
            or _as_float((entry_thesis or {}).get("entry_ref") if isinstance(entry_thesis, dict) else None)
            or _estimate_price(meta)
            or _latest_mid_price()
        )
        if ideal is not None and chase_max is not None and price_hint is not None:
            if abs(price_hint - ideal) > chase_max:
                _console_order_log(
                    "OPEN_SKIP",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note="market_guarded",
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="market_guarded_skip",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "price_hint": price_hint,
                        "ideal_entry": ideal,
                        "chase_max": chase_max,
                        "execution": exec_cfg,
                    },
                )
                return None

    # 強制マージンガード（reduce_only 以外）。直近スナップショットから
    # 現在の使用率と注文後の想定使用率を確認し、上限超えは即リジェクト。
    if not reduce_only:
        _trace("margin_guard")
        try:
            from utils.oanda_account import get_account_snapshot
        except Exception:
            get_account_snapshot = None  # type: ignore
        if get_account_snapshot is not None:
            try:
                snap = get_account_snapshot(cache_ttl_sec=1.0)
            except Exception as exc:
                note = "margin_snapshot_failed"
                logging.warning("[ORDER] margin guard snapshot failed: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None
            try:
                nav = float(snap.nav or 0.0)
                margin_used = float(snap.margin_used or 0.0)
                margin_rate = float(snap.margin_rate or 0.0)
                soft_cap = min(float(os.getenv("MAX_MARGIN_USAGE", "0.92") or 0.92), 0.99)
                hard_cap = min(float(os.getenv("MAX_MARGIN_USAGE_HARD", "0.96") or 0.96), 0.995)
                cap = min(hard_cap, max(soft_cap, 0.0))
                side_cap_enabled = str(os.getenv("MARGIN_SIDE_CAP_ENABLED", "1")).strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "off",
                }
                net_reducing = False
                net_before_units = 0.0
                long_u = None
                short_u = None
                try:
                    from utils.oanda_account import get_position_summary

                    long_u, short_u = get_position_summary()
                    net_before_units = float(long_u) - float(short_u)
                    net_after_units = (
                        net_before_units + abs(units) if side_label.lower() == "buy" else net_before_units - abs(units)
                    )
                    net_reducing = abs(net_after_units) < abs(net_before_units)
                except Exception:
                    net_reducing = False
                    net_before_units = 0.0
                if nav > 0:
                    usage_total = margin_used / nav
                    usage = usage_total
                    projected_usage = _projected_usage_with_netting(
                        nav,
                        margin_rate,
                        side_label,
                        units,
                        margin_used=margin_used,
                        meta=meta,
                    )
                    usage_for_cap = projected_usage if projected_usage is not None else usage
                    side_units = None
                    side_usage = None
                    side_projected = None
                    if side_cap_enabled and long_u is not None and short_u is not None and margin_rate > 0:
                        price_hint = _estimate_price(meta) or _latest_mid_price() or 0.0
                        if price_hint > 0:
                            if side_label.lower() == "buy":
                                side_units = abs(float(long_u))
                            else:
                                side_units = abs(float(short_u))
                            side_usage = (side_units * price_hint * margin_rate) / nav
                            side_projected = ((side_units + abs(units)) * price_hint * margin_rate) / nav
                            usage = side_usage
                            projected_usage = side_projected
                            usage_for_cap = side_projected
                            net_reducing = False
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and not (
                            net_reducing
                            and projected_usage is not None
                            and usage is not None
                            and projected_usage < usage
                        )
                    ):
                        price_hint = _estimate_price(meta) or _latest_mid_price() or 0.0
                        scaled_units = 0
                        cap_target = hard_cap * 0.99
                        if side_cap_enabled and side_units is not None and price_hint > 0 and margin_rate > 0:
                            try:
                                allowed_side = (cap_target * nav) / (price_hint * margin_rate) - side_units
                                if allowed_side > 0:
                                    scaled_units = int(math.floor(min(abs(units), allowed_side)))
                            except Exception:
                                scaled_units = 0
                        elif projected_usage and projected_usage > 0 and abs(units) > 0:
                            factor = cap_target / projected_usage
                            scaled_units = int(math.floor(abs(units) * factor))
                        elif nav > 0 and margin_rate > 0 and price_hint > 0:
                            try:
                                allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                                room = allowed_net - abs(net_before_units)
                                scaled_units = int(math.floor(min(abs(units), room)))
                            except Exception:
                                scaled_units = 0
                        if scaled_units > 0:
                            new_units = scaled_units if units > 0 else -scaled_units
                            logging.info(
                                "[ORDER] margin cap scale units %s -> %s usage=%.3f cap=%.3f",
                                units,
                                new_units,
                                usage_for_cap,
                                cap_target,
                            )
                            units = new_units
                        else:
                            note = "margin_usage_exceeds_cap"
                            _console_order_log(
                                "OPEN_REJECT",
                                pocket=pocket,
                                strategy_tag=strategy_tag or "unknown",
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                note=note,
                            )
                            log_order(
                                pocket=pocket,
                                instrument=instrument,
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                status=note,
                                attempt=0,
                                stage_index=stage_index,
                                request_payload={
                                    "strategy_tag": strategy_tag,
                                    "meta": meta,
                                    "entry_thesis": entry_thesis,
                                    "margin_usage": usage,
                                    "projected_usage": projected_usage,
                                    "margin_usage_total": usage_total,
                                    "side_usage": side_usage,
                                    "side_projected": side_projected,
                                    "cap": hard_cap,
                                },
                            )
                            log_metric(
                                "order_margin_block",
                                1.0,
                                tags={
                                    "pocket": pocket,
                                    "strategy": strategy_tag or "unknown",
                                    "reason": note,
                                },
                            )
                            return None
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and net_reducing
                        and projected_usage is not None
                        and usage is not None
                        and projected_usage < usage
                    ):
                        logging.info(
                            "[ORDER] allow net-reducing order usage=%.3f->%.3f cap=%.3f units=%d",
                            usage,
                            projected_usage,
                            hard_cap,
                            units,
                        )
                price_hint = _estimate_price(meta) or 0.0
                projected_usage = None
                if nav > 0 and margin_rate > 0:
                    if side_cap_enabled and long_u is not None and short_u is not None and price_hint > 0:
                        if side_label.lower() == "buy":
                            side_units = abs(float(long_u))
                        else:
                            side_units = abs(float(short_u))
                        projected_usage = ((side_units + abs(units)) * price_hint * margin_rate) / nav
                    else:
                        projected_usage = _projected_usage_with_netting(
                            nav,
                            margin_rate,
                            side_label,
                            units,
                            margin_used=margin_used,
                            meta=meta,
                        )
                        if projected_usage is None and price_hint > 0:
                            # フォールバック: 片側加算のみ
                            projected_used = margin_used + abs(units) * price_hint * margin_rate
                            projected_usage = projected_used / nav

                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and not (net_reducing and usage is not None and projected_usage < usage)
                ):
                    price_hint = _estimate_price(meta) or _latest_mid_price() or 0.0
                    scaled_units = 0
                    cap_target = cap * 0.99
                    try:
                        if side_cap_enabled and long_u is not None and short_u is not None and price_hint > 0:
                            if side_label.lower() == "buy":
                                side_units = abs(float(long_u))
                            else:
                                side_units = abs(float(short_u))
                            allowed_side = (cap_target * nav) / (price_hint * margin_rate) - side_units
                            if allowed_side > 0:
                                scaled_units = int(math.floor(min(abs(units), allowed_side)))
                        else:
                            factor = cap_target / projected_usage if projected_usage > 0 else 0.0
                            if factor > 0 and abs(units) > 0:
                                scaled_units = int(math.floor(abs(units) * factor))
                            elif nav > 0 and margin_rate > 0 and price_hint > 0:
                                allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                                room = allowed_net - abs(net_before_units)
                                scaled_units = int(math.floor(min(abs(units), room)))
                    except Exception:
                        scaled_units = 0
                    if scaled_units > 0:
                        new_units = scaled_units if units > 0 else -scaled_units
                        logging.info(
                            "[ORDER] projected margin scale units %s -> %s usage=%.3f cap=%.3f",
                            units,
                            new_units,
                            projected_usage,
                            cap_target,
                        )
                        units = new_units
                    else:
                        note = "margin_usage_projected_cap"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=strategy_tag or "unknown",
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=note,
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status=note,
                            attempt=0,
                            stage_index=stage_index,
                            request_payload={
                                "strategy_tag": strategy_tag,
                                "meta": meta,
                                "entry_thesis": entry_thesis,
                                "projected_usage": projected_usage,
                                "cap": cap,
                            },
                        )
                        log_metric(
                            "order_margin_block",
                            1.0,
                            tags={
                                "pocket": pocket,
                                "strategy": strategy_tag or "unknown",
                                "reason": note,
                            },
                        )
                        return None
                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and net_reducing
                    and usage is not None
                    and projected_usage < usage
                ):
                    logging.info(
                        "[ORDER] allow net-reducing projected usage=%.3f->%.3f cap=%.3f units=%d",
                        usage,
                        projected_usage,
                        cap,
                        units,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                note = "margin_guard_error"
                logging.warning("[ORDER] margin guard error: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None
    if not strategy_tag:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag="missing_tag",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="missing_strategy_tag",
        )
        log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="missing_strategy_tag",
            attempt=0,
            request_payload={
                "strategy_tag": strategy_tag,
                "meta": meta,
                "entry_thesis": entry_thesis,
            },
        )
        log_metric(
            "order_missing_strategy_tag",
            1.0,
            tags={"pocket": pocket, "side": "buy" if units > 0 else "sell"},
        )
        return None
    side_label = "buy" if units > 0 else "sell"

    entry_price_meta = _as_float((meta or {}).get("entry_price"))

    if strategy_tag and not reduce_only:
        _trace("strategy_cooldown")
        blocked, remain, reason = strategy_guard.is_blocked(strategy_tag)
        if blocked:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"strategy_cooldown:{reason}",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="strategy_cooldown",
                attempt=0,
                request_payload={
                    "strategy_tag": strategy_tag,
                    "cooldown_reason": reason,
                    "cooldown_remaining_sec": remain,
                },
            )
            return None
        price_hint = entry_price_meta
        if price_hint is None:
            price_hint = _entry_price_hint(entry_thesis, meta)
        open_positions = None
        if reentry_gate.needs_open_positions(strategy_tag):
            global _DIR_CAP_CACHE
            if _DIR_CAP_CACHE is None:
                _DIR_CAP_CACHE = PositionManager()
            open_positions = _DIR_CAP_CACHE.get_open_positions()
        allow_reentry, reentry_reason, reentry_details = reentry_gate.allow_entry(
            strategy_tag=strategy_tag,
            units=units,
            price=price_hint,
            open_positions=open_positions,
            now=datetime.now(timezone.utc),
        )
        if not allow_reentry:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"reentry_gate:{reentry_reason}",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="reentry_block",
                attempt=0,
                request_payload={
                    "strategy_tag": strategy_tag,
                    "reentry_reason": reentry_reason,
                    "reentry_details": reentry_details,
                },
            )
            return None

    # Pocket-level cooldown after margin rejection
    try:
        if pocket and _MARGIN_REJECT_UNTIL.get(pocket, 0.0) > time.monotonic():
            _trace("margin_cooldown")
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="margin_cooldown_active",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="margin_cooldown",
                attempt=0,
                request_payload={"cooldown_until": _MARGIN_REJECT_UNTIL.get(pocket)},
            )
            return None
    except Exception:
        pass

    if sl_disabled:
        sl_price = None

    if (
        _ENTRY_FACTOR_MAX_AGE_SEC > 0
        and not reduce_only
        and (pocket or "").lower() != "manual"
        and is_market_open()
    ):
        age_sec = _factor_age_seconds("M1")
        if age_sec is not None and age_sec > _ENTRY_FACTOR_MAX_AGE_SEC:
            pocket_key = (pocket or "").lower()
            if pocket_key and pocket_key in _ENTRY_FACTOR_STALE_ALLOW_POCKETS:
                log_metric(
                    "factor_stale_allow",
                    float(age_sec),
                    tags={"pocket": pocket_key, "tf": "M1"},
                )
            else:
                _trace("factor_stale")
                _console_order_log(
                    "OPEN_SKIP",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note="factor_stale",
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="factor_stale",
                    attempt=0,
                    request_payload={"age_sec": round(float(age_sec), 2)},
                )
                return None

    if (
        _FORWARD_TO_SIGNAL_GATE
        and not reduce_only
        and not arbiter_final
    ):
        price_hint = entry_price_meta or _estimate_price(meta) or _latest_mid_price()
        sl_pips_hint = _as_float((entry_thesis or {}).get("sl_pips")) if isinstance(entry_thesis, dict) else None
        if sl_pips_hint is None and isinstance(entry_thesis, dict):
            for alt in ("profile_sl_pips", "loss_guard_pips", "hard_stop_pips"):
                alt_val = _as_float(entry_thesis.get(alt))
                if alt_val:
                    sl_pips_hint = alt_val
                    break
        tp_pips_hint = _as_float((entry_thesis or {}).get("tp_pips")) if isinstance(entry_thesis, dict) else None
        if tp_pips_hint is None and isinstance(entry_thesis, dict):
            for alt in ("profile_tp_pips", "target_tp_pips"):
                alt_val = _as_float(entry_thesis.get(alt))
                if alt_val:
                    tp_pips_hint = alt_val
                    break
        tag_hint = (strategy_tag or "").strip().lower()
        if not tag_hint and isinstance(entry_thesis, dict):
            tag_hint = str(entry_thesis.get("strategy_tag") or entry_thesis.get("strategy") or "").strip().lower()
        if tag_hint == "fast_scalp":
            sl_pips_hint = None
        if sl_pips_hint is None and price_hint is not None and sl_price is not None:
            sl_pips_hint = abs(price_hint - sl_price) / 0.01
        if tp_pips_hint is None and price_hint is not None and tp_price is not None:
            tp_pips_hint = abs(price_hint - tp_price) / 0.01
        conf_val = int(round(_entry_confidence_score(confidence, entry_thesis)))
        payload = {
            "source": "order_manager",
            "strategy": strategy_tag,
            "pocket": pocket,
            "action": "OPEN_LONG" if units > 0 else "OPEN_SHORT",
            "confidence": conf_val,
            "entry_probability": _entry_probability_value(confidence, entry_thesis),
            "sl_pips": sl_pips_hint,
            "tp_pips": tp_pips_hint,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_price": price_hint,
            "client_order_id": client_order_id,
            "proposed_units": abs(units),
            "entry_type": (entry_thesis or {}).get("entry_type"),
            "entry_thesis": entry_thesis,
            "meta": meta or {},
        }
        try:
            signal_bus.enqueue(payload)
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="queued_to_gate",
                attempt=0,
                stage_index=stage_index,
                request_payload=payload,
            )
            _console_order_log(
                "OPEN_QUEUE",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="signal_gate",
            )
            return None
        except Exception as exc:
            logging.warning("[ORDER_GATE] enqueue failed, fall back to live order: %s", exc)

    if not is_market_open():
        _trace("market_closed")
        logging.info(
            "[ORDER] Market closed window. Skip order pocket=%s units=%s client_id=%s",
            pocket,
            units,
            client_order_id,
        )
        _console_order_log(
            "OPEN_SKIP",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=side_label,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="market_closed",
        )
        attempt_payload: dict = {"reason": "market_closed"}
        if entry_thesis is not None:
            attempt_payload["entry_thesis"] = entry_thesis
        if meta is not None:
            attempt_payload["meta"] = meta
        log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="market_closed",
            attempt=0,
            request_payload=attempt_payload,
        )
        return None

    # Entry-quality gates (opt-in): these are deterministic and can be enabled per VM
    # via systemd Environment overrides.
    if (
        not reduce_only
        and (pocket or "").lower() != "manual"
        and not preserve_strategy_intent
    ):
        allowed, reason, details = _entry_quality_regime_gate_decision(
            pocket=str(pocket or ""),
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            confidence=confidence,
        )
        if not allowed and reason:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=reason,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status=reason,
                attempt=0,
                stage_index=stage_index,
                request_payload={
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                    "details": details,
                },
            )
            log_metric(
                "entry_quality_regime_block",
                1.0,
                tags={
                    "pocket": pocket or "unknown",
                    "strategy": strategy_tag or "unknown",
                    "reason": str(details.get("mismatch_reason") or "regime_confidence"),
                },
            )
            return None

    if (
        instrument == "USD_JPY"
        and not reduce_only
        and (pocket or "").lower() != "manual"
    ):
        allowed, reason, details = _entry_quality_microstructure_gate_decision(
            pocket=str(pocket or ""),
        )
        if not allowed and reason:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=reason,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status=reason,
                attempt=0,
                stage_index=stage_index,
                request_payload={
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                    "details": details,
                },
            )
            log_metric(
                "entry_quality_microstructure_block",
                1.0,
                tags={
                    "pocket": pocket or "unknown",
                    "strategy": strategy_tag or "unknown",
                    "reason": reason,
                },
            )
            return None

    quote = _fetch_quote(instrument)
    if (
        quote
        and quote.get("spread_pips") is not None
        and not preserve_strategy_intent
    ):
        _trace("spread_check")
        spread_threshold = _order_spread_block_pips(
            pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
        )
        if spread_threshold > 0.0 and quote["spread_pips"] >= spread_threshold and not reduce_only:
            spread_pips = float(quote["spread_pips"])
            is_spike_tolerated, spike_info = _is_isolated_spread_spike(
                spread_pips=spread_pips,
                threshold_pips=spread_threshold,
            )
            if not is_spike_tolerated:
                note = f"spread_block:{spread_pips:.2f}p"
                _console_order_log(
                    "OPEN_SKIP",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="spread_block",
                    attempt=0,
                    request_payload={"quote": quote, "threshold": spread_threshold, "spread_spike": spike_info},
                )
                log_metric(
                    "entry_spread_block",
                    1.0,
                    tags={
                        "pocket": pocket or "unknown",
                        "strategy": strategy_tag or "unknown",
                        "reason": str((spike_info or {}).get("reason") or "spread_block"),
                        "decision": "blocked",
                    },
                )
                return None
            log_metric(
                "entry_spread_spike_tolerated",
                1.0,
                tags={
                    "pocket": pocket or "unknown",
                    "strategy": strategy_tag or "unknown",
                    "spread_pips": f"{spread_pips:.2f}",
                    "spread_threshold": f"{spread_threshold:.2f}",
                },
            )
            _trace("spread_spike_tolerated")

    estimated_entry = _estimate_entry_price(
        units=units, sl_price=sl_price, tp_price=tp_price, meta=meta
    )
    entry_basis = None
    if quote:
        entry_basis = quote["ask"] if units > 0 else quote["bid"]
        estimated_entry = entry_basis
    if entry_basis is None and entry_price_meta is not None:
        entry_basis = entry_price_meta

    # Market-adaptive SL: widen loss buffer when volatility/spread expands.
    # NOTE: This updates thesis_sl_pips (virtual SL) even when stopLossOnFill is disabled.
    if (
        not thesis_disable_hard_stop
        and _DYNAMIC_SL_ENABLE
        and (pocket or "").lower() in _DYNAMIC_SL_POCKETS
        and not preserve_strategy_intent
        and not reduce_only
    ):
        loss_guard_pips = None
        sl_hint_pips = thesis_sl_pips
        if isinstance(entry_thesis, dict):
            loss_guard_pips = _as_float(entry_thesis.get("loss_guard_pips"))
            if loss_guard_pips is None:
                loss_guard_pips = _as_float(entry_thesis.get("loss_guard"))
            if sl_hint_pips is None:
                sl_hint_pips = _as_float(entry_thesis.get("sl_pips"))
        if sl_hint_pips is None and entry_basis is not None and sl_price is not None:
            sl_hint_pips = abs(entry_basis - sl_price) / 0.01

        dynamic_sl_pips, dynamic_sl_meta = _dynamic_entry_sl_target_pips(
            pocket,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            quote=quote,
            sl_hint_pips=sl_hint_pips,
            loss_guard_pips=loss_guard_pips,
        )
        if dynamic_sl_pips is not None and (
            thesis_sl_pips is None or dynamic_sl_pips > thesis_sl_pips + 1e-6
        ):
            thesis_sl_pips = dynamic_sl_pips
            ref_price = entry_basis if entry_basis is not None else entry_price_meta
            if ref_price is not None:
                sl_price = _sl_price_from_pips(ref_price, units, dynamic_sl_pips)
            if isinstance(entry_thesis, dict):
                entry_thesis = dict(entry_thesis)
                entry_thesis["sl_pips"] = round(dynamic_sl_pips, 3)
                entry_thesis["dynamic_sl_applied"] = dynamic_sl_meta
            log_metric(
                "entry_dynamic_sl_pips",
                float(dynamic_sl_pips),
                tags={
                    "pocket": pocket or "unknown",
                    "strategy": strategy_tag or "unknown",
                },
            )

    # Recalculate SL/TP from thesis gaps using live quote to preserve intended RR
    if entry_basis is not None:
        if thesis_sl_pips is not None:
            if units > 0:
                sl_price = round(entry_basis - thesis_sl_pips * 0.01, 3)
            else:
                sl_price = round(entry_basis + thesis_sl_pips * 0.01, 3)
        if thesis_tp_pips is not None:
            if units > 0:
                tp_price = round(entry_basis + thesis_tp_pips * 0.01, 3)
            else:
                tp_price = round(entry_basis - thesis_tp_pips * 0.01, 3)

    if (
        not reduce_only
        and entry_basis is not None
        and sl_price is not None
        and tp_price is not None
        and not preserve_strategy_intent
    ):
        min_rr = _min_rr_for(pocket)
        if min_rr > 0.0:
            sl_pips = abs(entry_basis - sl_price) / 0.01
            tp_pips = abs(tp_price - entry_basis) / 0.01
            if sl_pips > 0.0 and tp_pips > 0.0 and tp_pips < sl_pips * min_rr:
                max_sl_pips = tp_pips / min_rr if min_rr > 0 else 0.0
                mode = _min_rr_adjust_mode_for(
                    pocket,
                    strategy_tag=strategy_tag,
                )
                sl_adjusted = False
                if mode in {"sl", "sl_first", "both"} and max_sl_pips > 0.0 and sl_pips > max_sl_pips:
                    if units > 0:
                        sl_price = round(entry_basis - max_sl_pips * 0.01, 3)
                    else:
                        sl_price = round(entry_basis + max_sl_pips * 0.01, 3)
                    sl_pips = max_sl_pips
                    sl_adjusted = True
                    if isinstance(entry_thesis, dict):
                        entry_thesis = dict(entry_thesis)
                        entry_thesis["sl_pips"] = round(max_sl_pips, 2)
                        entry_thesis["min_rr_adjusted"] = {
                            "min_rr": min_rr,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "mode": "sl",
                        }
                    log_metric(
                        "min_rr_adjust_sl",
                        float(max_sl_pips),
                        tags={
                            "pocket": pocket,
                            "strategy": strategy_tag or "unknown",
                            "min_rr": f"{min_rr:.2f}",
                        },
                    )
                    logging.info(
                        "[ORDER] min_rr adjust sl pocket=%s strategy=%s sl=%.2fp tp=%.2fp min_rr=%.2f",
                        pocket,
                        strategy_tag or "-",
                        sl_pips,
                        tp_pips,
                        min_rr,
                    )
                if mode in {"sl"} or (mode == "sl_first" and sl_adjusted):
                    pass
                else:
                    adj_tp_pips = sl_pips * min_rr
                    if units > 0:
                        tp_price = round(entry_basis + adj_tp_pips * 0.01, 3)
                    else:
                        tp_price = round(entry_basis - adj_tp_pips * 0.01, 3)
                    thesis_tp_pips = adj_tp_pips
                    if isinstance(entry_thesis, dict):
                        entry_thesis = dict(entry_thesis)
                        entry_thesis["tp_pips"] = round(adj_tp_pips, 2)
                        entry_thesis["min_rr_adjusted"] = {
                            "min_rr": min_rr,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "mode": "tp",
                        }
                    log_metric(
                        "min_rr_adjust",
                        float(adj_tp_pips),
                        tags={
                            "pocket": pocket,
                            "strategy": strategy_tag or "unknown",
                            "min_rr": f"{min_rr:.2f}",
                        },
                    )
                    logging.info(
                        "[ORDER] min_rr adjust tp pocket=%s strategy=%s sl=%.2fp tp=%.2fp min_rr=%.2f",
                        pocket,
                        strategy_tag or "-",
                        sl_pips,
                        adj_tp_pips,
                        min_rr,
                    )

    if (
        not reduce_only
        and entry_basis is not None
        and tp_price is not None
        and not preserve_strategy_intent
    ):
        tp_cap, tp_cap_meta = _tp_cap_for(pocket, entry_thesis)
        if tp_cap > 0.0:
            tp_pips = abs(tp_price - entry_basis) / 0.01
            if tp_pips > tp_cap:
                adj_tp_pips = tp_cap
                if units > 0:
                    tp_price = round(entry_basis + adj_tp_pips * 0.01, 3)
                else:
                    tp_price = round(entry_basis - adj_tp_pips * 0.01, 3)
                min_rr = _min_rr_for(pocket)
                if sl_price is not None and min_rr > 0.0:
                    sl_pips = abs(entry_basis - sl_price) / 0.01
                    max_sl_pips = adj_tp_pips / min_rr
                    if max_sl_pips > 0.0 and sl_pips > max_sl_pips:
                        if units > 0:
                            sl_price = round(entry_basis - max_sl_pips * 0.01, 3)
                        else:
                            sl_price = round(entry_basis + max_sl_pips * 0.01, 3)
                if isinstance(entry_thesis, dict):
                    entry_thesis = dict(entry_thesis)
                    payload = {
                        "cap_pips": round(tp_cap, 2),
                        "tp_pips_before": round(tp_pips, 2),
                        "tp_pips_after": round(adj_tp_pips, 2),
                        "sl_pips_after": round(
                            abs(entry_basis - sl_price) / 0.01, 2
                        ) if sl_price is not None else None,
                    }
                    if isinstance(tp_cap_meta, dict):
                        payload.update(
                            {
                                "cap_base": round(float(tp_cap_meta.get("cap_base", tp_cap)), 2),
                                "cap_mult": round(float(tp_cap_meta.get("cap_mult", 1.0)), 3),
                                "cap_final": round(float(tp_cap_meta.get("cap_final", tp_cap)), 2),
                                "cap_tf": tp_cap_meta.get("tf"),
                                "cap_score": (
                                    round(float(tp_cap_meta.get("score")), 3)
                                    if tp_cap_meta.get("score") is not None
                                    else None
                                ),
                                "cap_adx": (
                                    round(float(tp_cap_meta.get("adx")), 2)
                                    if tp_cap_meta.get("adx") is not None
                                    else None
                                ),
                                "cap_atr_pips": (
                                    round(float(tp_cap_meta.get("atr_pips")), 2)
                                    if tp_cap_meta.get("atr_pips") is not None
                                    else None
                                ),
                            }
                        )
                    entry_thesis["tp_cap_applied"] = payload
                log_metric(
                    "tp_cap_adjust",
                    float(adj_tp_pips),
                    tags={
                        "pocket": pocket or "unknown",
                        "strategy": strategy_tag or "unknown",
                    },
                )

    if (
        not reduce_only
        and entry_basis is not None
        and sl_price is not None
        and not preserve_strategy_intent
    ):
        max_sl_pips = _entry_max_sl_pips(pocket, strategy_tag=strategy_tag)
        if max_sl_pips > 0.0:
            sl_pips = abs(entry_basis - sl_price) / 0.01
            if sl_pips > max_sl_pips + 1e-6:
                if units > 0:
                    sl_price = round(entry_basis - max_sl_pips * 0.01, 3)
                else:
                    sl_price = round(entry_basis + max_sl_pips * 0.01, 3)
                thesis_sl_pips = max_sl_pips
                if isinstance(entry_thesis, dict):
                    entry_thesis = dict(entry_thesis)
                    entry_thesis["sl_pips"] = round(max_sl_pips, 2)
                    entry_thesis["sl_cap_applied"] = {
                        "sl_pips_before": round(sl_pips, 2),
                        "sl_pips_after": round(max_sl_pips, 2),
                    }
                log_metric(
                    "entry_sl_cap",
                    float(max_sl_pips),
                    tags={
                        "pocket": pocket or "unknown",
                        "strategy": strategy_tag or "unknown",
                    },
                )
                logging.warning(
                    "[ORDER] entry SL cap applied pocket=%s strategy=%s sl=%.2fp (cap=%.2fp)",
                    pocket,
                    strategy_tag or "-",
                    sl_pips,
                    max_sl_pips,
                )

    if not reduce_only and not preserve_strategy_intent:
        conf_score = _entry_confidence_score(confidence, entry_thesis)
        sl_pips_live: Optional[float] = None
        tp_pips_live: Optional[float] = None
        if entry_basis is not None:
            if sl_price is not None:
                sl_pips_live = abs(entry_basis - sl_price) / 0.01
            elif thesis_sl_pips is not None:
                sl_pips_live = float(thesis_sl_pips)
            if tp_price is not None:
                tp_pips_live = abs(tp_price - entry_basis) / 0.01
            elif thesis_tp_pips is not None:
                tp_pips_live = float(thesis_tp_pips)
        if sl_pips_live is None and thesis_sl_pips is not None:
            sl_pips_live = float(thesis_sl_pips)
        if tp_pips_live is None and thesis_tp_pips is not None:
            tp_pips_live = float(thesis_tp_pips)

        quality_ok, quality_reason, quality_meta = _entry_quality_gate(
            pocket,
            confidence=conf_score,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            quote=quote,
            sl_pips=sl_pips_live,
            tp_pips=tp_pips_live,
        )
        if not quality_ok:
            if (
                quality_reason == "entry_quality_spread_tp_softened"
                and isinstance(quality_meta, dict)
            ):
                soften_scale = _as_float(quality_meta.get("recommended_units_scale"))
                if soften_scale is not None and 0.0 < soften_scale < 1.0:
                    scaled_units = int(round(abs(units) * soften_scale))
                    min_allowed_units = min_units_for_strategy(strategy_tag, pocket=pocket)
                    if scaled_units <= 0:
                        reason = "entry_quality_scale_below_min"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=strategy_tag,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=reason,
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status=reason,
                            attempt=0,
                            request_payload={
                                "reason": reason,
                                "quality": quality_meta,
                                "confidence": conf_score,
                                "entry_thesis": entry_thesis,
                                "meta": meta,
                                "scaled_units": scaled_units,
                                "min_units": min_allowed_units,
                            },
                        )
                        log_metric(
                            "entry_quality_block",
                            1.0,
                            tags={
                                "pocket": pocket or "unknown",
                                "strategy": strategy_tag or "unknown",
                                "reason": reason,
                            },
                        )
                        return None
                    if min_allowed_units > 0 and scaled_units < min_allowed_units:
                        reason = "entry_quality_scale_below_min"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=strategy_tag,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=reason,
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status=reason,
                            attempt=0,
                            request_payload={
                                "reason": reason,
                                "quality": quality_meta,
                                "confidence": conf_score,
                                "entry_thesis": entry_thesis,
                                "meta": meta,
                                "scaled_units": scaled_units,
                                "min_units": min_allowed_units,
                            },
                        )
                        log_metric(
                            "entry_quality_block",
                            1.0,
                            tags={
                                "pocket": pocket or "unknown",
                                "strategy": strategy_tag or "unknown",
                                "reason": reason,
                            },
                        )
                        return None

                    if scaled_units > 0 and scaled_units != abs(units):
                        previous_units = units
                        units = scaled_units if units > 0 else -scaled_units
                        if isinstance(entry_thesis, dict):
                            entry_thesis = dict(entry_thesis)
                            entry_thesis["entry_quality_soften_scale"] = round(
                                soften_scale, 4
                            )
                        _console_order_log(
                            "OPEN_SCALE",
                            pocket=pocket,
                            strategy_tag=strategy_tag,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=f"entry_quality_spread_tp_softened:{soften_scale:.2f}",
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status="entry_quality_softened",
                            attempt=0,
                            request_payload={
                                "reason": "entry_quality_spread_tp_softened",
                                "quality": quality_meta,
                                "confidence": conf_score,
                                "previous_units": previous_units,
                                "scaled_units": scaled_units,
                                "entry_thesis": entry_thesis,
                                "meta": meta,
                            },
                        )
                        log_metric(
                            "entry_quality_softened",
                            float(soften_scale),
                            tags={
                                "pocket": pocket or "unknown",
                                "strategy": strategy_tag or "unknown",
                                "reason": "spread_tp_softened",
                            },
                        )
                    quality_ok = True

            if not quality_ok:
                reason = quality_reason or "entry_quality_block"
            if not quality_ok:
                _console_order_log(
                    "OPEN_SKIP",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=reason,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=reason,
                    attempt=0,
                    request_payload={
                        "reason": reason,
                        "quality": quality_meta,
                        "confidence": conf_score,
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                    },
                )
                log_metric(
                    "entry_quality_block",
                    1.0,
                    tags={
                        "pocket": pocket or "unknown",
                        "strategy": strategy_tag or "unknown",
                        "reason": reason,
                    },
                )
                return None

    # Margin preflight (new entriesのみ)
    preflight_units = units
    original_units = units
    min_allowed_units = min_units_for_strategy(strategy_tag, pocket=pocket)
    requested_units = units
    clamped_to_minimum = False
    if (
        not reduce_only
        and min_allowed_units > 0
        and 0 < abs(requested_units) < min_allowed_units
    ):
        requested_units = min_allowed_units if requested_units > 0 else -min_allowed_units
        clamped_to_minimum = True
        units = requested_units
        preflight_units = requested_units
        logging.info(
            "[ORDER] units clamped to pocket minimum pocket=%s requested=%s -> %s",
            pocket,
            original_units,
            requested_units,
        )

    if not reduce_only and estimated_entry is not None:
        _trace("preflight_units")
        allowed_units, req_margin = _preflight_units(
            estimated_price=estimated_entry, requested_units=requested_units
        )
        if allowed_units == 0:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="insufficient_margin",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="insufficient_margin_skip",
                attempt=0,
                request_payload={
                    "estimated_price": estimated_entry,
                    "required_margin": req_margin,
                },
            )
            return None
        if allowed_units != units:
            logging.info(
                "[ORDER] Preflight scaled units %s -> %s (pocket=%s)",
                units,
                allowed_units,
                pocket,
            )
            preflight_units = allowed_units
        if clamped_to_minimum and abs(preflight_units) >= min_allowed_units:
            logging.info(
                "[ORDER] Raised %s pocket units to minimum %s from %s",
                pocket,
                min_allowed_units,
                units,
            )

    if (
        not reduce_only
        and not sl_disabled
        and not thesis_disable_hard_stop
        and not preserve_strategy_intent
        and estimated_entry is not None
    ):
        hard_stop_pips = _entry_hard_stop_pips(pocket, strategy_tag=strategy_tag)
        if hard_stop_pips > 0.0:
            max_sl_pips = _entry_max_sl_pips(pocket, strategy_tag=strategy_tag)
            if max_sl_pips > 0.0 and hard_stop_pips > max_sl_pips + 1e-6:
                logging.warning(
                    "[ORDER] entry hard SL exceeds max SL cap; clamping pocket=%s strategy=%s hard=%.2fp cap=%.2fp",
                    pocket,
                    strategy_tag or "-",
                    hard_stop_pips,
                    max_sl_pips,
                )
                hard_stop_pips = max_sl_pips
            hard_sl_price = _sl_price_from_pips(estimated_entry, units, hard_stop_pips)
            if hard_sl_price is not None:
                current_gap_pips: float | None = None
                wrong_side = False
                if sl_price is not None:
                    wrong_side = (units > 0 and sl_price >= estimated_entry) or (
                        units < 0 and sl_price <= estimated_entry
                    )
                    try:
                        current_gap_pips = abs(estimated_entry - sl_price) / 0.01
                    except Exception:
                        current_gap_pips = None
                if (
                    sl_price is None
                    or wrong_side
                    or current_gap_pips is None
                    or current_gap_pips < hard_stop_pips - 1e-6
                ):
                    logging.info(
                        "[ORDER] hard SL on fill applied pocket=%s pips=%.1f sl=%s client=%s",
                        pocket,
                        hard_stop_pips,
                        f"{hard_sl_price:.3f}",
                        client_order_id or "-",
                    )
                    sl_price = hard_sl_price
                    thesis_sl_pips = hard_stop_pips
                    if isinstance(entry_thesis, dict):
                        entry_thesis = dict(entry_thesis)
                        entry_thesis["sl_pips"] = round(float(hard_stop_pips), 2)
                        entry_thesis["entry_hard_sl_applied"] = {
                            "pips": round(float(hard_stop_pips), 2),
                            "sl_price": round(float(hard_sl_price), 3),
                            "prev_gap_pips": round(float(current_gap_pips), 2)
                            if current_gap_pips is not None
                            else None,
                        }

    if (
        not reduce_only
        and estimated_entry is not None
        and not preserve_strategy_intent
    ):
        norm_sl = None if sl_disabled else sl_price
        norm_tp = tp_price
        norm_sl, norm_tp, normalized = _normalize_protections(
            estimated_entry,
            norm_sl,
            norm_tp,
            units > 0,
        )
        sl_price = None if sl_disabled else norm_sl
        tp_price = norm_tp
        if normalized:
            logging.debug(
                "[ORDER] normalized SL/TP client=%s sl=%s tp=%s entry=%.3f",
                client_order_id,
                f"{sl_price:.3f}" if sl_price is not None else "None",
                f"{tp_price:.3f}" if tp_price is not None else "None",
                estimated_entry,
            )

    if (
        not reduce_only
        and not thesis_disable_hard_stop
        and not preserve_strategy_intent
    ):
        loss_cap_jpy = _entry_loss_cap_jpy(
            pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            nav_hint=_entry_nav_hint(meta, entry_thesis if isinstance(entry_thesis, dict) else None),
        )
        if loss_cap_jpy > 0.0:
            sl_pips_for_cap: float | None = None
            if estimated_entry is not None and sl_price is not None:
                try:
                    sl_pips_for_cap = abs(estimated_entry - sl_price) / 0.01
                except Exception:
                    sl_pips_for_cap = None
            if sl_pips_for_cap is None and thesis_sl_pips is not None:
                try:
                    sl_pips_for_cap = float(thesis_sl_pips)
                except Exception:
                    sl_pips_for_cap = None
            if sl_pips_for_cap is None or sl_pips_for_cap <= 0.0:
                logging.warning(
                    "[ORDER] loss-cap requested but SL unavailable pocket=%s strategy=%s cap=%.1f",
                    pocket,
                    strategy_tag or "-",
                    loss_cap_jpy,
                )
            else:
                cap_buffer_pips = _entry_loss_cap_buffer_pips(
                    pocket,
                    strategy_tag=strategy_tag,
                    entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
                )
                effective_sl_pips = max(0.0, float(sl_pips_for_cap) + max(0.0, cap_buffer_pips))
                capped_abs_units = _loss_cap_units_from_sl(
                    loss_cap_jpy=loss_cap_jpy,
                    sl_pips=effective_sl_pips,
                )
                if capped_abs_units <= 0:
                    _console_order_log(
                        "OPEN_SKIP",
                        pocket=pocket,
                        strategy_tag=strategy_tag,
                        side=side_label,
                        units=preflight_units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        note="loss_cap_zero",
                    )
                    log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side=side_label,
                        units=preflight_units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="loss_cap_skip",
                        attempt=0,
                        request_payload={
                            "loss_cap_jpy": loss_cap_jpy,
                            "sl_pips": sl_pips_for_cap,
                            "buffer_pips": cap_buffer_pips,
                            "effective_sl_pips": effective_sl_pips,
                            "reason": "zero_units",
                        },
                    )
                    return None
                if abs(preflight_units) > capped_abs_units:
                    if min_allowed_units > 0 and capped_abs_units < min_allowed_units:
                        _console_order_log(
                            "OPEN_SKIP",
                            pocket=pocket,
                            strategy_tag=strategy_tag,
                            side=side_label,
                            units=preflight_units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note="loss_cap_lt_min_units",
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=preflight_units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status="loss_cap_skip",
                            attempt=0,
                            request_payload={
                                "loss_cap_jpy": loss_cap_jpy,
                                "sl_pips": sl_pips_for_cap,
                                "buffer_pips": cap_buffer_pips,
                                "effective_sl_pips": effective_sl_pips,
                                "loss_cap_units": capped_abs_units,
                                "min_units": min_allowed_units,
                                "reason": "below_min_units",
                            },
                        )
                        return None
                    pre_loss_cap_units = preflight_units
                    capped_units = capped_abs_units if preflight_units > 0 else -capped_abs_units
                    log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side=side_label,
                        units=preflight_units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="loss_cap_applied",
                        attempt=0,
                        request_payload={
                            "from_units": pre_loss_cap_units,
                            "to_units": capped_units,
                            "loss_cap_jpy": loss_cap_jpy,
                            "sl_pips": sl_pips_for_cap,
                            "buffer_pips": cap_buffer_pips,
                            "effective_sl_pips": effective_sl_pips,
                        },
                    )
                    preflight_units = capped_units
                    if isinstance(entry_thesis, dict):
                        entry_thesis = dict(entry_thesis)
                        entry_thesis["loss_cap_applied"] = {
                            "loss_cap_jpy": round(loss_cap_jpy, 2),
                            "sl_pips": round(float(sl_pips_for_cap), 3),
                            "buffer_pips": round(float(cap_buffer_pips), 3),
                            "effective_sl_pips": round(float(effective_sl_pips), 3),
                            "units_before": int(abs(pre_loss_cap_units)),
                            "units_after": int(abs(preflight_units)),
                        }
                    logging.info(
                        "[ORDER] loss-cap scaled units %s -> %s pocket=%s strategy=%s cap=%.1f sl=%.2fp",
                        units,
                        preflight_units,
                        pocket,
                        strategy_tag or "-",
                        loss_cap_jpy,
                        effective_sl_pips,
                    )

    # Virtual SL/TP logging (even if SL is disabled)
    if estimated_entry is not None:
        if thesis_sl_pips is not None:
            virtual_sl_price = round(
                estimated_entry - thesis_sl_pips * 0.01, 3
            ) if units > 0 else round(estimated_entry + thesis_sl_pips * 0.01, 3)
        if thesis_tp_pips is not None:
            virtual_tp_price = round(
                estimated_entry + thesis_tp_pips * 0.01, 3
            ) if units > 0 else round(estimated_entry - thesis_tp_pips * 0.01, 3)
    if sl_price is not None:
        virtual_sl_price = sl_price
    if tp_price is not None:
        virtual_tp_price = tp_price

    # Real-time freshness: if preflight takes too long, skip to avoid stale fills.
    if not reduce_only and entry_deadline_sec > 0:
        elapsed = time.monotonic() - order_t0
        if elapsed > entry_deadline_sec:
            note = f"stale_deadline elapsed={elapsed:.2f}s cap={entry_deadline_sec:.2f}s"
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=note,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="stale_deadline_skip",
                attempt=0,
                stage_index=stage_index,
                request_payload={
                    "note": "stale_deadline_skip",
                    "elapsed_sec": round(elapsed, 3),
                    "deadline_sec": float(entry_deadline_sec),
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
            log_metric(
                "order_stale_skip",
                1.0,
                tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
            )
            return None

    # Directional exposure cap: scale down instead of rejecting
    if not reduce_only and not preserve_strategy_intent:
        _trace("dir_cap")
        adjusted = _apply_directional_cap(preflight_units, pocket, side_label, meta)
        if adjusted == 0:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=preflight_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="dir_cap",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=preflight_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="dir_cap_block",
                attempt=0,
                stage_index=stage_index,
                request_payload={
                    "note": "dir_cap",
                    "preflight_units": preflight_units,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
            return None
        if adjusted != preflight_units:
            logging.info(
                "[ORDER] dir_cap scaled units %s -> %s pocket=%s side=%s",
                preflight_units,
                adjusted,
                pocket,
                side_label,
            )
            preflight_units = adjusted

    if isinstance(entry_thesis, dict):
        entry_thesis = _augment_entry_thesis_regime(entry_thesis, pocket)
        entry_thesis = _augment_entry_thesis_flags(entry_thesis)
        entry_thesis = _augment_entry_thesis_policy_generation(
            entry_thesis,
            reduce_only=reduce_only,
        )
    comment = _encode_thesis_comment(entry_thesis)
    client_ext = {"tag": f"pocket={pocket}"}
    trade_ext = {"tag": f"pocket={pocket}"}
    if comment:
        client_ext["comment"] = comment
        trade_ext["comment"] = comment
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(preflight_units),
            "timeInForce": "FOK",
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": client_ext,
            "tradeClientExtensions": trade_ext,
        }
    }
    if client_order_id:
        order_data["order"]["clientExtensions"]["id"] = client_order_id
        order_data["order"]["tradeClientExtensions"]["id"] = client_order_id
    if (
        not sl_disabled
        and not reduce_only
        and not thesis_disable_hard_stop
        and sl_price is not None
        and _allow_stop_loss_on_fill(pocket, strategy_tag=strategy_tag)
    ):
        order_data["order"]["stopLossOnFill"] = {"price": f"{sl_price:.3f}"}
    if tp_price is not None:
        order_data["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.3f}"}

    side = "buy" if preflight_units > 0 else "sell"
    units_to_send = preflight_units
    _trace("open_req")
    _console_order_log(
        "OPEN_REQ",
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=side,
        units=units_to_send,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        note="reduce_only" if reduce_only else None,
    )
    # Default behavior (2 attempts, 1 protection fallback) is kept unless overridden.
    # This is primarily used to reduce "TAKE_PROFIT_ON_FILL_LOSS"/"LOSING_TAKE_PROFIT"
    # opportunity loss on fast markets without changing sizing logic.
    # Allow operators to force single-attempt mode in service path to keep RPC latency bounded.
    max_submit_attempts = max(1, int(os.getenv("ORDER_SUBMIT_MAX_ATTEMPTS", "2") or 2))
    max_protection_fallbacks = max(
        0, int(os.getenv("ORDER_PROTECTION_FALLBACK_MAX_RETRIES", "1") or 1)
    )
    protection_fallbacks = 0

    for attempt in range(max_submit_attempts):
        payload = order_data.copy()
        payload["order"] = dict(order_data["order"], units=str(units_to_send))
        # Log attempt payload (include non-OANDA context for analytics)
        attempt_payload: dict = {"oanda": payload}
        if entry_thesis is not None:
            attempt_payload["entry_thesis"] = entry_thesis
        if meta is not None:
            attempt_payload["meta"] = meta
        if quote:
            attempt_payload["quote"] = quote
        attempt_payload = _merge_virtual(attempt_payload)
        log_order(
            pocket=pocket,
            instrument=instrument,
            side=side,
            units=units_to_send,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="submit_attempt",
            attempt=attempt + 1,
            request_payload=attempt_payload,
        )
        r = OrderCreate(accountID=ACCOUNT, data=payload)
        try:
            api.request(r)
            response = r.response
            reject = response.get("orderRejectTransaction") or response.get(
                "orderCancelTransaction"
            )
            if reject:
                reason = reject.get("rejectReason") or reject.get("reason")
                reason_key = str(reason or "").upper()
                logging.error(
                    "OANDA order rejected (attempt %d) pocket=%s units=%s reason=%s",
                    attempt + 1,
                    pocket,
                    units_to_send,
                    reason,
                )
                logging.error("Reject payload: %s", reject)
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="rejected",
                    attempt=attempt + 1,
                    stage_index=stage_index,
                    error_code=reject.get("errorCode"),
                    error_message=reject.get("errorMessage") or reason,
                    response_payload=response,
                )
                log_metric(
                    "order_success_rate",
                    0.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": reason_key.lower() if reason_key else "rejected",
                    },
                )
                log_metric(
                    "reject_rate",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": reason_key.lower() if reason_key else "rejected",
                    },
                )
                if reason_key == "INSUFFICIENT_MARGIN" and pocket:
                    _MARGIN_REJECT_UNTIL[pocket] = time.monotonic() + 120.0
                if (
                    attempt < max_submit_attempts - 1
                    and not reduce_only
                    and protection_fallbacks < max_protection_fallbacks
                    and reason_key in _PROTECTION_RETRY_REASONS
                    and (sl_price is not None or tp_price is not None)
                ):
                    base_gap_price = _protection_fallback_gap_price(
                        pocket,
                        strategy_tag=strategy_tag,
                    )
                    # Widen the fallback gap each time to handle continued price movement.
                    fallback_gap_price = float(base_gap_price) * (1.0 + float(protection_fallbacks))
                    retry_quote = _fetch_quote(instrument)
                    if retry_quote:
                        quote = retry_quote
                    retry_basis: Optional[float] = None
                    if retry_quote:
                        retry_basis = (
                            _as_float(retry_quote.get("ask"))
                            if units_to_send > 0
                            else _as_float(retry_quote.get("bid"))
                        )
                    if retry_basis is None:
                        retry_basis = entry_basis if entry_basis is not None else estimated_entry
                    sl_gap_retry_pips = thesis_sl_pips
                    if sl_gap_retry_pips is None and retry_basis is not None and sl_price is not None:
                        sl_gap_retry_pips = abs(float(retry_basis) - float(sl_price)) / 0.01
                    elif sl_gap_retry_pips is None and estimated_entry is not None and sl_price is not None:
                        sl_gap_retry_pips = abs(float(estimated_entry) - float(sl_price)) / 0.01
                    tp_gap_retry_pips = thesis_tp_pips
                    if tp_gap_retry_pips is None and retry_basis is not None and tp_price is not None:
                        tp_gap_retry_pips = abs(float(tp_price) - float(retry_basis)) / 0.01
                    elif tp_gap_retry_pips is None and estimated_entry is not None and tp_price is not None:
                        tp_gap_retry_pips = abs(float(tp_price) - float(estimated_entry)) / 0.01

                    fallback_basis = _derive_fallback_basis(
                        retry_basis,
                        sl_price,
                        tp_price,
                        units_to_send > 0,
                        fallback_gap_price=fallback_gap_price,
                    )
                    fallback_sl, fallback_tp = _fallback_protections(
                        fallback_basis,
                        is_buy=units_to_send > 0,
                        has_sl=sl_price is not None,
                        has_tp=tp_price is not None,
                        reason_key=reason_key,
                        sl_gap_pips=sl_gap_retry_pips,
                        tp_gap_pips=tp_gap_retry_pips,
                        fallback_gap_price=fallback_gap_price,
                    )
                    if sl_disabled:
                        fallback_sl = None
                    if fallback_sl is not None or fallback_tp is not None:
                        if (
                            fallback_sl is not None
                            and not thesis_disable_hard_stop
                            and (not reduce_only)
                            and _allow_stop_loss_on_fill(
                                pocket, strategy_tag=strategy_tag
                            )
                        ):
                            order_data["order"]["stopLossOnFill"] = {
                                "price": f"{fallback_sl:.3f}"
                            }
                            sl_price = fallback_sl
                        elif "stopLossOnFill" in order_data["order"]:
                            order_data["order"].pop("stopLossOnFill", None)

                        if fallback_tp is not None:
                            order_data["order"]["takeProfitOnFill"] = {
                                "price": f"{fallback_tp:.3f}"
                            }
                            tp_price = fallback_tp
                        elif "takeProfitOnFill" in order_data["order"]:
                            order_data["order"].pop("takeProfitOnFill", None)

                        protection_fallbacks += 1
                        logging.warning(
                            "[ORDER] protection fallback applied client=%s reason=%s basis=%.3f sl=%s tp=%s gap=%.4f (retry=%d/%d)",
                            client_order_id,
                            reason_key,
                            float(fallback_basis) if fallback_basis is not None else -1.0,
                            f"{fallback_sl:.3f}" if fallback_sl is not None else "-",
                            f"{fallback_tp:.3f}" if fallback_tp is not None else "-",
                            fallback_gap_price,
                            protection_fallbacks,
                            max_protection_fallbacks,
                        )
                        continue
                if attempt < max_submit_attempts - 1 and abs(units_to_send) >= 2000:
                    units_to_send = int(units_to_send * 0.5)
                    if units_to_send == 0:
                        break
                    logging.info(
                        "Retrying order with reduced units=%s (half).", units_to_send
                    )
                    continue
                return None

            trade_id = _extract_trade_id(response)
            if trade_id:
                fill_type = response.get("orderFillTransaction", {}).get(
                    "tradeOpened"
                )
                if not fill_type:
                    logging.info(
                        "OANDA order filled by adjusting existing trade(s): %s", trade_id
                    )
                # Extract executed_price if present
                executed_price = None
                ofill = response.get("orderFillTransaction") or {}
                if ofill.get("price"):
                    try:
                        executed_price = float(ofill.get("price"))
                    except Exception:
                        executed_price = None
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="filled",
                    attempt=attempt + 1,
                    stage_index=stage_index,
                    ticket_id=trade_id,
                    executed_price=executed_price,
                    # Keep the original request for post-hoc analysis (side/units/TP含む)
                    request_payload=attempt_payload,
                    response_payload=response,
                )
                log_metric(
                    "order_success_rate",
                    1.0,
                    tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
                )
                log_metric(
                    "reject_rate",
                    0.0,
                    tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
                )
                _console_order_log(
                    "OPEN_FILLED",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    ticket_id=trade_id,
                    note=f"attempt={attempt+1}",
                )
                target_sl = None if sl_disabled else sl_price
                # EXIT_NO_NEGATIVE_CLOSE=1 is meant to enforce profit-only exits. Honor that policy by not
                # attaching a broker SL on the loss side unless explicitly allowed (rollout flag).
                if target_sl is not None and not _allow_stop_loss_on_fill(
                    pocket, strategy_tag=strategy_tag
                ):
                    basis = executed_price if executed_price is not None else estimated_entry
                    try:
                        basis_val = float(basis) if basis is not None else None
                    except Exception:
                        basis_val = None
                    if not basis_val or basis_val <= 0.0:
                        logging.info(
                            "[ORDER] on_fill_protection skipped SL (no basis) pocket=%s client=%s sl=%s",
                            pocket,
                            client_order_id or "-",
                            f"{target_sl:.3f}",
                        )
                        target_sl = None
                    else:
                        loss_side = (units_to_send > 0 and target_sl < basis_val - 1e-6) or (
                            units_to_send < 0 and target_sl > basis_val + 1e-6
                        )
                        if loss_side:
                            logging.info(
                                "[ORDER] on_fill_protection skipped SL (EXIT_NO_NEGATIVE_CLOSE) pocket=%s client=%s sl=%s basis=%s",
                                pocket,
                                client_order_id or "-",
                                f"{target_sl:.3f}",
                                f"{basis_val:.3f}",
                            )
                            target_sl = None
                _maybe_update_protections(
                    trade_id,
                    target_sl,
                    tp_price,
                    context="on_fill_protection",
                    ref_price=executed_price,
                )
                return trade_id

            logging.error(
                "OANDA order fill lacked trade identifiers (attempt %d): %s",
                attempt + 1,
                response,
            )
            _console_order_log(
                "OPEN_FAIL",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"no_trade_id attempt={attempt+1}",
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="fill_no_tradeid",
                attempt=attempt + 1,
                stage_index=stage_index,
                response_payload=response,
            )
            return None
        except V20Error as e:
            logging.error(
                "OANDA API Error (attempt %d) pocket=%s units=%s: %s",
                attempt + 1,
                pocket,
                units_to_send,
                e,
            )
            resp = getattr(e, "response", None)
            if resp:
                logging.error("OANDA response: %s", resp)
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="api_error",
                attempt=attempt + 1,
                stage_index=stage_index,
                error_message=str(e),
                response_payload=resp if isinstance(resp, dict) else None,
            )
            log_metric(
                "order_success_rate",
                0.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": "api_error",
                },
            )
            log_metric(
                "reject_rate",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": "api_error",
                },
            )

            if (
                attempt == 0
                and abs(units_to_send) >= 2000
                and pocket != "scalp_fast"
            ):
                units_to_send = int(units_to_send * 0.5)
                if units_to_send == 0:
                    break
                logging.info(
                    "Retrying order with reduced units=%s (half).", units_to_send
                )
                continue
            return None
        except Exception as exc:
            logging.exception(
                "Unexpected error submitting order (attempt %d): %s",
                attempt + 1,
                exc,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="unexpected_error",
                attempt=attempt + 1,
                stage_index=stage_index,
                error_message=str(exc),
            )
            return None
    log_order(
        pocket=pocket,
        instrument=instrument,
        side=side,
        units=units_to_send,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status="order_fallthrough",
        attempt=0,
        stage_index=stage_index,
        request_payload={"note": "market_order_fallthrough"},
    )
    return None


async def limit_order(
    instrument: str,
    units: int,
    price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp"],
    *,
    current_bid: Optional[float] = None,
    current_ask: Optional[float] = None,
    require_passive: bool = True,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
    ttl_ms: float = 800.0,
    entry_thesis: Optional[dict] = None,
    confidence: Optional[int] = None,
    meta: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Place a passive limit order. Returns (trade_id, order_id)."""
    strategy_tag = _strategy_tag_from_client_id(client_order_id)
    if not strategy_tag:
        strategy_tag = _strategy_tag_from_thesis(entry_thesis)
    entry_thesis = _ensure_entry_intent_payload(
        units=units,
        confidence=confidence,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    entry_probability = _entry_probability_value(confidence, entry_thesis)
    preserve_strategy_intent = (
        _ORDER_MANAGER_PRESERVE_STRATEGY_INTENT
        and not reduce_only
        and (pocket or "").lower() != "manual"
    )
    if preserve_strategy_intent:
        scaled_units, probability_reason = _probability_scaled_units(
            units,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_probability=entry_probability,
        )
        if probability_reason is not None:
            reason_note = probability_reason
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"entry_probability:{reason_note}",
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="entry_probability_reject",
                attempt=0,
                request_payload={
                    "entry_probability": entry_probability,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                    "entry_probability_reject_reason": reason_note,
                    "strategy_tag": strategy_tag,
                },
            )
            log_metric(
                "order_probability_reject",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": reason_note,
                },
            )
            return None, None
        if scaled_units != units:
            _console_order_log(
                "OPEN_SCALE",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side="buy" if units > 0 else "sell",
                units=scaled_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"probability_scale:{entry_probability:.3f}" if entry_probability is not None else "probability_scale",
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=scaled_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="probability_scaled",
                attempt=0,
                request_payload={
                    "entry_probability": entry_probability,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                    "scaled_units": scaled_units,
                    "raw_units": abs(units) if units else 0,
                },
            )
            log_metric(
                "order_probability_scale",
                max(
                    0.0,
                    min(
                        1.0,
                        float(entry_probability)
                        if entry_probability is not None
                        else 1.0,
                    ),
                ),
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                },
            )
            units = scaled_units
    service_result = await _order_manager_service_request_async(
        "/order/limit_order",
        {
            "instrument": instrument,
            "units": units,
            "price": price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "pocket": pocket,
            "current_bid": current_bid,
            "current_ask": current_ask,
            "require_passive": require_passive,
            "client_order_id": client_order_id,
            "reduce_only": reduce_only,
            "ttl_ms": ttl_ms,
            "entry_thesis": entry_thesis,
            "meta": meta,
            "confidence": confidence,
            "entry_probability": entry_probability,
        },
    )
    if service_result is not None:
        if isinstance(service_result, dict):
            trade_id = service_result.get("trade_id")
            order_id = service_result.get("order_id")
            return (
                str(trade_id) if trade_id is not None else None,
                str(order_id) if order_id is not None else None,
            )
        return None, None

    sl_disabled = stop_loss_disabled_for_pocket(pocket)
    if (
        sl_disabled
        and isinstance(strategy_tag, str)
        and strategy_tag.strip().lower().startswith("scalp_ping_5s_b")
    ):
        sl_disabled = False
    thesis_disable_hard_stop = _disable_hard_stop_by_strategy(
        strategy_tag,
        pocket,
        entry_thesis if isinstance(entry_thesis, dict) else None,
    )
    if thesis_disable_hard_stop:
        sl_price = None
    if sl_disabled:
        sl_price = None

    if units == 0:
        return None, None

    if _soft_tp_mode(entry_thesis):
        tp_price = None

    if (
        not reduce_only
        and not sl_disabled
        and not thesis_disable_hard_stop
        and not preserve_strategy_intent
        and price > 0
    ):
        hard_stop_pips = _entry_hard_stop_pips(pocket, strategy_tag=strategy_tag)
        if hard_stop_pips > 0.0:
            hard_sl_price = _sl_price_from_pips(price, units, hard_stop_pips)
            if hard_sl_price is not None:
                current_gap_pips: float | None = None
                wrong_side = False
                if sl_price is not None:
                    wrong_side = (units > 0 and sl_price >= price) or (units < 0 and sl_price <= price)
                    try:
                        current_gap_pips = abs(price - sl_price) / 0.01
                    except Exception:
                        current_gap_pips = None
                if (
                    sl_price is None
                    or wrong_side
                    or current_gap_pips is None
                    or current_gap_pips < hard_stop_pips - 1e-6
                ):
                    logging.info(
                        "[ORDER] hard SL on fill applied (limit) pocket=%s pips=%.1f sl=%s client=%s",
                        pocket,
                        hard_stop_pips,
                        f"{hard_sl_price:.3f}",
                        client_order_id or "-",
                    )
                    sl_price = hard_sl_price

    if not reduce_only and pocket != "manual":
        entry_thesis = _apply_default_entry_thesis_tfs(entry_thesis, pocket)
    if isinstance(entry_thesis, dict):
        entry_thesis = _augment_entry_thesis_regime(entry_thesis, pocket)
        entry_thesis = _augment_entry_thesis_flags(entry_thesis)
        entry_thesis = _augment_entry_thesis_policy_generation(
            entry_thesis,
            reduce_only=reduce_only,
        )

    if (
        not reduce_only
        and not preserve_strategy_intent
        and not thesis_disable_hard_stop
    ):
        loss_cap_jpy = _entry_loss_cap_jpy(
            pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            nav_hint=_entry_nav_hint(meta, entry_thesis if isinstance(entry_thesis, dict) else None),
        )
        if loss_cap_jpy > 0.0:
            sl_pips_for_cap: float | None = None
            if sl_price is not None and price > 0.0:
                try:
                    sl_pips_for_cap = abs(price - sl_price) / 0.01
                except Exception:
                    sl_pips_for_cap = None
            if sl_pips_for_cap is None and isinstance(entry_thesis, dict):
                sl_pips_for_cap = _as_float(entry_thesis.get("sl_pips"))
            if sl_pips_for_cap is not None and sl_pips_for_cap > 0.0:
                cap_buffer_pips = _entry_loss_cap_buffer_pips(
                    pocket,
                    strategy_tag=strategy_tag,
                    entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
                )
                effective_sl_pips = max(0.0, float(sl_pips_for_cap) + max(0.0, cap_buffer_pips))
                capped_abs_units = _loss_cap_units_from_sl(
                    loss_cap_jpy=loss_cap_jpy,
                    sl_pips=effective_sl_pips,
                )
                min_allowed_units = min_units_for_strategy(strategy_tag, pocket=pocket)
                if capped_abs_units <= 0 or (
                    min_allowed_units > 0 and capped_abs_units < min_allowed_units
                ):
                    _console_order_log(
                        "OPEN_SKIP",
                        pocket=pocket,
                        strategy_tag=str(strategy_tag or "unknown"),
                        side="buy" if units > 0 else "sell",
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        note="loss_cap_skip",
                    )
                    _log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side="buy" if units > 0 else "sell",
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="loss_cap_skip",
                        attempt=0,
                        request_payload={
                            "loss_cap_jpy": loss_cap_jpy,
                            "sl_pips": sl_pips_for_cap,
                            "buffer_pips": cap_buffer_pips,
                            "effective_sl_pips": effective_sl_pips,
                            "loss_cap_units": capped_abs_units,
                            "min_units": min_allowed_units,
                        },
                    )
                    return None, None
                if abs(units) > capped_abs_units:
                    prev_units = units
                    units = capped_abs_units if units > 0 else -capped_abs_units
                    _log_order(
                        pocket=pocket,
                        instrument=instrument,
                        side="buy" if prev_units > 0 else "sell",
                        units=prev_units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        client_order_id=client_order_id,
                        status="loss_cap_applied",
                        attempt=0,
                        request_payload={
                            "from_units": prev_units,
                            "to_units": units,
                            "loss_cap_jpy": loss_cap_jpy,
                            "sl_pips": sl_pips_for_cap,
                            "buffer_pips": cap_buffer_pips,
                            "effective_sl_pips": effective_sl_pips,
                        },
                    )
                    if isinstance(entry_thesis, dict):
                        entry_thesis = dict(entry_thesis)
                        entry_thesis["loss_cap_applied"] = {
                            "loss_cap_jpy": round(loss_cap_jpy, 2),
                            "sl_pips": round(float(sl_pips_for_cap), 3),
                            "buffer_pips": round(float(cap_buffer_pips), 3),
                            "effective_sl_pips": round(float(effective_sl_pips), 3),
                            "units_before": int(abs(prev_units)),
                            "units_after": int(abs(units)),
                        }
            elif loss_cap_jpy > 0.0:
                logging.warning(
                    "[ORDER] limit loss-cap requested but SL unavailable pocket=%s strategy=%s cap=%.1f",
                    pocket,
                    strategy_tag or "-",
                    loss_cap_jpy,
                )

    if require_passive and not _is_passive_price(
        units=units,
        price=price,
        current_bid=current_bid,
        current_ask=current_ask,
    ):
        logging.info(
            "[ORDER] Passive guard blocked limit order pocket=%s units=%s price=%.3f bid=%s ask=%s",
            pocket,
            units,
            price,
            f"{current_bid:.3f}" if current_bid is not None else "NA",
            f"{current_ask:.3f}" if current_ask is not None else "NA",
        )
        return None, None
    if not is_market_open():
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="market_closed",
            attempt=0,
            request_payload={"price": price, "reason": "market_closed"},
        )
        return None, None

    # 強制マージンガード（reduce_only 以外）。market_order と同等の使用率チェックを行う。
    if not reduce_only:
        side_label = "buy" if units > 0 else "sell"
        meta_guard: dict = {}
        if isinstance(meta, dict):
            meta_guard = dict(meta)
        if meta_guard.get("entry_price") is None and meta_guard.get("price") is None:
            meta_guard["entry_price"] = float(price)
        try:
            from utils.oanda_account import get_account_snapshot
        except Exception:
            get_account_snapshot = None  # type: ignore
        if get_account_snapshot is not None:
            try:
                snap = get_account_snapshot(cache_ttl_sec=1.0)
            except Exception as exc:
                note = "margin_snapshot_failed"
                logging.warning("[ORDER] margin guard snapshot failed: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None, None
            try:
                nav = float(snap.nav or 0.0)
                margin_used = float(snap.margin_used or 0.0)
                margin_rate = float(snap.margin_rate or 0.0)
                soft_cap = min(float(os.getenv("MAX_MARGIN_USAGE", "0.92") or 0.92), 0.99)
                hard_cap = min(float(os.getenv("MAX_MARGIN_USAGE_HARD", "0.96") or 0.96), 0.995)
                cap = min(hard_cap, max(soft_cap, 0.0))
                side_cap_enabled = str(os.getenv("MARGIN_SIDE_CAP_ENABLED", "1")).strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "off",
                }
                net_reducing = False
                net_before_units = 0.0
                long_u = None
                short_u = None
                try:
                    from utils.oanda_account import get_position_summary

                    long_u, short_u = get_position_summary()
                    net_before_units = float(long_u) - float(short_u)
                    net_after_units = (
                        net_before_units + abs(units) if side_label.lower() == "buy" else net_before_units - abs(units)
                    )
                    net_reducing = abs(net_after_units) < abs(net_before_units)
                except Exception:
                    net_reducing = False
                    net_before_units = 0.0
                if nav > 0:
                    usage_total = margin_used / nav
                    usage = usage_total
                    projected_usage = _projected_usage_with_netting(
                        nav,
                        margin_rate,
                        side_label,
                        units,
                        margin_used=margin_used,
                        meta=meta_guard,
                    )
                    usage_for_cap = projected_usage if projected_usage is not None else usage
                    side_units = None
                    side_usage = None
                    side_projected = None
                    if side_cap_enabled and long_u is not None and short_u is not None and margin_rate > 0:
                        price_hint = _estimate_price(meta_guard) or _latest_mid_price() or 0.0
                        if price_hint > 0:
                            if side_label.lower() == "buy":
                                side_units = abs(float(long_u))
                            else:
                                side_units = abs(float(short_u))
                            side_usage = (side_units * price_hint * margin_rate) / nav
                            side_projected = ((side_units + abs(units)) * price_hint * margin_rate) / nav
                            usage = side_usage
                            projected_usage = side_projected
                            usage_for_cap = side_projected
                            net_reducing = False
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and not (
                            net_reducing
                            and projected_usage is not None
                            and usage is not None
                            and projected_usage < usage
                        )
                    ):
                        price_hint = _estimate_price(meta_guard) or _latest_mid_price() or 0.0
                        scaled_units = 0
                        cap_target = hard_cap * 0.99
                        if side_cap_enabled and side_units is not None and price_hint > 0 and margin_rate > 0:
                            try:
                                allowed_side = (cap_target * nav) / (price_hint * margin_rate) - side_units
                                if allowed_side > 0:
                                    scaled_units = int(math.floor(min(abs(units), allowed_side)))
                            except Exception:
                                scaled_units = 0
                        elif projected_usage and projected_usage > 0 and abs(units) > 0:
                            factor = cap_target / projected_usage
                            scaled_units = int(math.floor(abs(units) * factor))
                        elif nav > 0 and margin_rate > 0 and price_hint > 0:
                            try:
                                allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                                room = allowed_net - abs(net_before_units)
                                scaled_units = int(math.floor(min(abs(units), room)))
                            except Exception:
                                scaled_units = 0
                        if scaled_units > 0:
                            new_units = scaled_units if units > 0 else -scaled_units
                            logging.info(
                                "[ORDER] margin cap scale units %s -> %s usage=%.3f cap=%.3f",
                                units,
                                new_units,
                                usage_for_cap,
                                cap_target,
                            )
                            units = new_units
                        else:
                            note = "margin_usage_exceeds_cap"
                            _console_order_log(
                                "OPEN_REJECT",
                                pocket=pocket,
                                strategy_tag=str(strategy_tag or "unknown"),
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                note=note,
                            )
                            _log_order(
                                pocket=pocket,
                                instrument=instrument,
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                status=note,
                                attempt=0,
                                request_payload={
                                    "strategy_tag": strategy_tag,
                                    "meta": meta,
                                    "entry_thesis": entry_thesis,
                                    "margin_usage": usage,
                                    "projected_usage": projected_usage,
                                    "margin_usage_total": usage_total,
                                    "side_usage": side_usage,
                                    "side_projected": side_projected,
                                    "cap": hard_cap,
                                },
                            )
                            log_metric(
                                "order_margin_block",
                                1.0,
                                tags={
                                    "pocket": pocket,
                                    "strategy": strategy_tag or "unknown",
                                    "reason": note,
                                },
                            )
                            return None, None
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and net_reducing
                        and projected_usage is not None
                        and usage is not None
                        and projected_usage < usage
                    ):
                        logging.info(
                            "[ORDER] allow net-reducing order usage=%.3f->%.3f cap=%.3f units=%d",
                            usage,
                            projected_usage,
                            hard_cap,
                            units,
                        )
                price_hint = _estimate_price(meta_guard) or 0.0
                projected_usage = None
                if nav > 0 and margin_rate > 0:
                    if side_cap_enabled and long_u is not None and short_u is not None and price_hint > 0:
                        if side_label.lower() == "buy":
                            side_units = abs(float(long_u))
                        else:
                            side_units = abs(float(short_u))
                        projected_usage = ((side_units + abs(units)) * price_hint * margin_rate) / nav
                    else:
                        projected_usage = _projected_usage_with_netting(
                            nav,
                            margin_rate,
                            side_label,
                            units,
                            margin_used=margin_used,
                            meta=meta_guard,
                        )
                        if projected_usage is None and price_hint > 0:
                            projected_used = margin_used + abs(units) * price_hint * margin_rate
                            projected_usage = projected_used / nav

                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and not (net_reducing and usage is not None and projected_usage < usage)
                ):
                    price_hint = _estimate_price(meta_guard) or _latest_mid_price() or 0.0
                    scaled_units = 0
                    cap_target = cap * 0.99
                    try:
                        if side_cap_enabled and long_u is not None and short_u is not None and price_hint > 0:
                            if side_label.lower() == "buy":
                                side_units = abs(float(long_u))
                            else:
                                side_units = abs(float(short_u))
                            allowed_side = (cap_target * nav) / (price_hint * margin_rate) - side_units
                            if allowed_side > 0:
                                scaled_units = int(math.floor(min(abs(units), allowed_side)))
                        else:
                            factor = cap_target / projected_usage if projected_usage > 0 else 0.0
                            if factor > 0 and abs(units) > 0:
                                scaled_units = int(math.floor(abs(units) * factor))
                            elif nav > 0 and margin_rate > 0 and price_hint > 0:
                                allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                                room = allowed_net - abs(net_before_units)
                                scaled_units = int(math.floor(min(abs(units), room)))
                    except Exception:
                        scaled_units = 0
                    if scaled_units > 0:
                        new_units = scaled_units if units > 0 else -scaled_units
                        logging.info(
                            "[ORDER] projected margin scale units %s -> %s usage=%.3f cap=%.3f",
                            units,
                            new_units,
                            projected_usage,
                            cap_target,
                        )
                        units = new_units
                    else:
                        note = "margin_usage_projected_cap"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=str(strategy_tag or "unknown"),
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=note,
                        )
                        _log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status=note,
                            attempt=0,
                            request_payload={
                                "strategy_tag": strategy_tag,
                                "meta": meta,
                                "entry_thesis": entry_thesis,
                                "projected_usage": projected_usage,
                                "cap": cap,
                            },
                        )
                        log_metric(
                            "order_margin_block",
                            1.0,
                            tags={
                                "pocket": pocket,
                                "strategy": strategy_tag or "unknown",
                                "reason": note,
                            },
                        )
                        return None, None
                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and net_reducing
                    and usage is not None
                    and projected_usage < usage
                ):
                    logging.info(
                        "[ORDER] allow net-reducing projected usage=%.3f->%.3f cap=%.3f units=%d",
                        usage,
                        projected_usage,
                        cap,
                        units,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                note = "margin_guard_error"
                logging.warning("[ORDER] margin guard error: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None, None

    ttl_sec = max(0.0, ttl_ms / 1000.0)
    # OANDA GTD granularity is seconds; clamp sub-second TTL up to 1s instead
    # of accidentally leaving a GTC limit order around.
    if 0.0 < ttl_sec < 1.0:
        ttl_sec = 1.0
    time_in_force = "GTC"
    gtd_time = None
    if ttl_sec > 0.0:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl_sec)
        gtd_time = expiry.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        time_in_force = "GTD"

    comment = _encode_thesis_comment(entry_thesis)
    client_ext = {"tag": f"pocket={pocket}"}
    trade_ext = {"tag": f"pocket={pocket}"}
    if comment:
        client_ext["comment"] = comment
        trade_ext["comment"] = comment

    payload = {
        "order": {
            "type": "LIMIT",
            "instrument": instrument,
            "units": str(units),
            "price": f"{price:.3f}",
            "timeInForce": time_in_force,
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": client_ext,
            "tradeClientExtensions": trade_ext,
        }
    }
    if gtd_time:
        payload["order"]["gtdTime"] = gtd_time
    if client_order_id:
        payload["order"]["clientExtensions"]["id"] = client_order_id
        payload["order"]["tradeClientExtensions"]["id"] = client_order_id
    if (
        not sl_disabled
        and not reduce_only
        and not thesis_disable_hard_stop
        and sl_price is not None
        and _allow_stop_loss_on_fill(pocket, strategy_tag=strategy_tag)
    ):
        payload["order"]["stopLossOnFill"] = {"price": f"{sl_price:.3f}"}
    if tp_price is not None:
        payload["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.3f}"}

    attempt_payload: dict = {"oanda": payload}
    if entry_thesis is not None:
        attempt_payload["entry_thesis"] = entry_thesis
        attempt_payload["entry_probability"] = entry_probability
    if meta is not None:
        attempt_payload["meta"] = meta

    _log_order(
        pocket=pocket,
        instrument=instrument,
        side="buy" if units > 0 else "sell",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status="submit_attempt",
        attempt=1,
        request_payload=attempt_payload,
    )

    endpoint = OrderCreate(accountID=ACCOUNT, data=payload)
    try:
        api.request(endpoint)
    except V20Error as exc:
        logging.error("[ORDER] Limit order error: %s", exc)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="api_error",
            attempt=1,
            error_code=getattr(exc, "code", None),
            error_message=str(exc),
            request_payload=attempt_payload,
        )
        log_metric(
            "order_success_rate",
            0.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": "api_error",
            },
        )
        log_metric(
            "reject_rate",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": "api_error",
            },
        )
        return None, None
    except Exception as exc:  # noqa: BLE001
        logging.exception("[ORDER] Limit order unexpected error: %s", exc)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="unexpected_error",
            attempt=1,
            error_message=str(exc),
            request_payload=attempt_payload,
        )
        log_metric(
            "order_success_rate",
            0.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": "unexpected_error",
            },
        )
        log_metric(
            "reject_rate",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": "unexpected_error",
            },
        )
        return None, None

    response = endpoint.response
    reject = response.get("orderRejectTransaction")
    if reject:
        reason = reject.get("rejectReason") or reject.get("reason")
        reason_key = str(reason or "").lower() or "rejected"
        logging.error("[ORDER] Limit order rejected: %s", reason)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="rejected",
            attempt=1,
            error_code=reject.get("errorCode"),
            error_message=reject.get("errorMessage") or reason,
            response_payload=response,
        )
        log_metric(
            "order_success_rate",
            0.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": reason_key,
            },
        )
        log_metric(
            "reject_rate",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": strategy_tag or "unknown",
                "reason": reason_key,
            },
        )
        return None, None

    trade_id = _extract_trade_id(response)
    executed_price = None
    if response.get("orderFillTransaction") and response["orderFillTransaction"].get(
        "price"
    ):
        try:
            executed_price = float(response["orderFillTransaction"]["price"])
        except Exception:
            executed_price = None

    order_id = None
    create_tx = response.get("orderCreateTransaction")
    if create_tx and create_tx.get("id") is not None:
        order_id = str(create_tx["id"])

    status = "submitted"
    if trade_id:
        status = "filled"

    _log_order(
        pocket=pocket,
        instrument=instrument,
        side="buy" if units > 0 else "sell",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status=status,
        attempt=1,
        ticket_id=trade_id,
        executed_price=executed_price,
        request_payload=attempt_payload,
        response_payload=response,
    )
    log_metric(
        "order_success_rate",
        1.0,
        tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
    )
    log_metric(
        "reject_rate",
        0.0,
        tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
    )

    if trade_id:
        return trade_id, order_id

    return None, order_id
