"""
workers.common.forecast_gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional probabilistic forecast gate for entry sizing / blocking.

This gate is deliberately conservative and opt-in:
- It requires a trained joblib bundle (see scripts/train_forecast_bundle.py).
- It reads candles from indicators.factor_cache (M5/H1/D1) and computes
  the latest per-horizon probability.
- It returns allow / scale-down / block decisions.

The intent is to provide a "human-like" filter without using LLMs, and to keep
behavior unchanged unless explicitly enabled via env.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from utils.metrics_logger import log_metric

LOG = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_set(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


_ENABLED = _env_bool("FORECAST_GATE_ENABLED", False)
_BUNDLE_PATH = os.getenv(
    "FORECAST_BUNDLE_PATH",
    "config/forecast_models/USD_JPY_bundle.joblib",
).strip()
_TTL_SEC = max(1.0, _env_float("FORECAST_GATE_TTL_SEC", 15.0))

_EDGE_BLOCK = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BLOCK", 0.38)))
_EDGE_BAD = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BAD", 0.45)))
_EDGE_REF = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_REF", 0.55)))
_SCALE_MIN = max(0.0, min(1.0, _env_float("FORECAST_GATE_SCALE_MIN", 0.5)))

_REQUIRE_FRESH = _env_bool("FORECAST_GATE_REQUIRE_FRESH", False)
_MAX_AGE_SEC = max(1.0, _env_float("FORECAST_GATE_MAX_AGE_SEC", 120.0))

_STRATEGY_ALLOWLIST = _env_set("FORECAST_GATE_STRATEGY_ALLOWLIST")
_POCKET_ALLOWLIST = _env_set("FORECAST_GATE_POCKET_ALLOWLIST")

_HORIZON_FORCE = os.getenv("FORECAST_GATE_HORIZON", "").strip()
_HORIZON_SCALP_FAST = os.getenv("FORECAST_GATE_HORIZON_SCALP_FAST", "1h").strip()
_HORIZON_SCALP = os.getenv("FORECAST_GATE_HORIZON_SCALP", "1h").strip()
_HORIZON_MICRO = os.getenv("FORECAST_GATE_HORIZON_MICRO", "8h").strip()
_HORIZON_MACRO = os.getenv("FORECAST_GATE_HORIZON_MACRO", "1d").strip()


@dataclass(frozen=True)
class ForecastDecision:
    allowed: bool
    scale: float
    reason: str
    horizon: str
    edge: float
    p_up: float
    expected_pips: Optional[float] = None
    feature_ts: Optional[str] = None


_BUNDLE_CACHE = None
_BUNDLE_MTIME = 0.0
_PRED_CACHE: dict | None = None
_PRED_CACHE_TS = 0.0


def _should_use(strategy_tag: Optional[str], pocket: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    if not strategy_tag or not pocket:
        return False
    if pocket.strip().lower() == "manual":
        return False
    if _POCKET_ALLOWLIST and pocket.strip().lower() not in _POCKET_ALLOWLIST:
        return False
    if _STRATEGY_ALLOWLIST:
        key = strategy_tag.strip().lower()
        base = key.split("-", 1)[0]
        if key not in _STRATEGY_ALLOWLIST and base not in _STRATEGY_ALLOWLIST:
            return False
    return True


def _horizon_for(pocket: str, entry_thesis: Optional[dict]) -> str | None:
    if isinstance(entry_thesis, dict):
        hinted = entry_thesis.get("forecast_horizon") or entry_thesis.get("horizon")
        if hinted:
            return str(hinted).strip()
    if _HORIZON_FORCE:
        return _HORIZON_FORCE
    p = (pocket or "").strip().lower()
    if p == "scalp_fast":
        return _HORIZON_SCALP_FAST
    if p == "scalp":
        return _HORIZON_SCALP
    if p == "micro":
        return _HORIZON_MICRO
    if p == "macro":
        return _HORIZON_MACRO
    return None


def _load_bundle_cached():
    global _BUNDLE_CACHE, _BUNDLE_MTIME, _PRED_CACHE, _PRED_CACHE_TS
    if not _BUNDLE_PATH:
        return None
    path = Path(_BUNDLE_PATH)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    if _BUNDLE_CACHE is not None and mtime > 0 and abs(mtime - _BUNDLE_MTIME) < 1e-3:
        return _BUNDLE_CACHE
    try:
        from analysis.forecast_sklearn import load_bundle
    except Exception as exc:
        LOG.debug("[FORECAST] sklearn bundle import failed: %s", exc)
        return None
    try:
        bundle = load_bundle(path)
    except Exception as exc:
        LOG.warning("[FORECAST] failed to load bundle path=%s err=%s", path, exc)
        return None
    _BUNDLE_CACHE = bundle
    _BUNDLE_MTIME = mtime
    _PRED_CACHE = None
    _PRED_CACHE_TS = 0.0
    return bundle


def _scale_from_edge(edge: float) -> float:
    e = max(0.0, min(1.0, float(edge)))
    lo = max(0.0, min(1.0, float(_EDGE_BAD)))
    hi = max(0.0, min(1.0, float(_EDGE_REF)))
    if hi <= lo + 1e-9:
        return 1.0 if e >= hi else float(_SCALE_MIN)
    if e <= lo:
        return float(_SCALE_MIN)
    if e >= hi:
        return 1.0
    score = (e - lo) / (hi - lo)
    return float(_SCALE_MIN) + score * (1.0 - float(_SCALE_MIN))


def _ensure_predictions(bundle) -> dict | None:  # noqa: ANN001 - sklearn bundle type
    global _PRED_CACHE, _PRED_CACHE_TS
    now = time.time()
    if _PRED_CACHE is not None and now - _PRED_CACHE_TS < _TTL_SEC:
        return _PRED_CACHE
    try:
        from indicators.factor_cache import get_candles_snapshot
    except Exception:
        return None
    candles_by_tf = {
        "M5": get_candles_snapshot("M5", limit=1200, include_live=True),
        "H1": get_candles_snapshot("H1", limit=1000, include_live=True),
        "D1": get_candles_snapshot("D1", limit=500, include_live=True),
    }
    try:
        from analysis.forecast_sklearn import predict_latest
    except Exception:
        return None
    try:
        preds = predict_latest(bundle, candles_by_tf)
    except Exception as exc:
        LOG.debug("[FORECAST] predict_latest failed: %s", exc)
        preds = None
    if isinstance(preds, dict):
        _PRED_CACHE = preds
        _PRED_CACHE_TS = now
        return preds
    return None


def decide(
    *,
    strategy_tag: Optional[str],
    pocket: str,
    side: str,
    units: int,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> ForecastDecision | None:
    if not _should_use(strategy_tag, pocket):
        return None

    # Only gate known instrument (avoid accidental use on other pairs).
    instrument = None
    if isinstance(meta, dict):
        instrument = meta.get("instrument")
    if instrument and str(instrument).strip().upper() != "USD_JPY":
        return None

    horizon = _horizon_for(pocket, entry_thesis)
    if not horizon:
        return None

    bundle = _load_bundle_cached()
    if bundle is None:
        return None
    preds = _ensure_predictions(bundle)
    if not isinstance(preds, dict):
        return None

    row = preds.get(horizon)
    if not isinstance(row, dict):
        return None

    try:
        p_up = float(row.get("p_up"))
    except Exception:
        return None
    p_up = max(0.0, min(1.0, p_up))
    side_key = (side or "").strip().lower()
    edge = p_up if side_key == "buy" or units > 0 else 1.0 - p_up
    edge = max(0.0, min(1.0, float(edge)))

    if edge < float(_EDGE_BLOCK):
        log_metric(
            "forecast_gate_block",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": str(strategy_tag or "unknown"),
                "horizon": horizon,
                "reason": "edge_block",
            },
        )
        return ForecastDecision(
            allowed=False,
            scale=0.0,
            reason="edge_block",
            horizon=horizon,
            edge=edge,
            p_up=p_up,
            expected_pips=row.get("expected_pips"),
            feature_ts=row.get("feature_ts"),
        )

    feature_ts = row.get("feature_ts")
    if _REQUIRE_FRESH and feature_ts:
        try:
            from datetime import datetime, timezone

            ts = datetime.fromisoformat(str(feature_ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > _MAX_AGE_SEC:
                log_metric(
                    "forecast_gate_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "horizon": horizon,
                        "reason": "stale",
                    },
                )
                return ForecastDecision(
                    allowed=False,
                    scale=0.0,
                    reason="stale",
                    horizon=horizon,
                    edge=edge,
                    p_up=p_up,
                    expected_pips=row.get("expected_pips"),
                    feature_ts=str(feature_ts),
                )
        except Exception:
            # If timestamp parsing fails, do not block; behave as no-op.
            pass

    scale = _scale_from_edge(edge)
    if scale >= 0.999:
        return None
    log_metric(
        "forecast_gate_scale",
        1.0,
        tags={
            "pocket": pocket,
            "strategy": str(strategy_tag or "unknown"),
            "horizon": horizon,
            "reason": "edge_scale",
        },
    )
    return ForecastDecision(
        allowed=True,
        scale=scale,
        reason="edge_scale",
        horizon=horizon,
        edge=edge,
        p_up=p_up,
        expected_pips=row.get("expected_pips"),
        feature_ts=row.get("feature_ts"),
    )


__all__ = [
    "ForecastDecision",
    "decide",
]
