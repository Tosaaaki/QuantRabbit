"""Dedicated forecast decision service.

This worker provides:
- /forecast/decide: same schema as local forecast_gate.decide()
- /forecast/predictions: raw cached forecast rows for debugging/observation
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Body, FastAPI
import uvicorn

from workers.common import forecast_gate

LOG = logging.getLogger(__name__)

app = FastAPI(
    title="QuantRabbit Forecast Service",
    version="v1",
)


def _as_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_decision(decision: forecast_gate.ForecastDecision) -> dict[str, Any]:
    return {
        "allowed": bool(decision.allowed),
        "scale": decision.scale,
        "reason": decision.reason,
        "horizon": decision.horizon,
        "edge": decision.edge,
        "p_up": decision.p_up,
        "expected_pips": decision.expected_pips,
        "anchor_price": decision.anchor_price,
        "target_price": decision.target_price,
        "tp_pips_hint": decision.tp_pips_hint,
        "sl_pips_cap": decision.sl_pips_cap,
        "rr_floor": decision.rr_floor,
        "feature_ts": decision.feature_ts,
        "source": decision.source,
        "style": decision.style,
        "trend_strength": decision.trend_strength,
        "range_pressure": decision.range_pressure,
        "future_flow": decision.future_flow,
        "volatility_state": decision.volatility_state,
        "trend_state": decision.trend_state,
        "range_state": decision.range_state,
        "volatility_rank": decision.volatility_rank,
        "regime_score": decision.regime_score,
        "leading_indicator": decision.leading_indicator,
        "leading_indicator_strength": decision.leading_indicator_strength,
    }


def _success(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def _failure(message: str) -> dict[str, Any]:
    LOG.warning("[FORECAST_WORKER] request failed: %s", message)
    return {"ok": False, "error": message}


@app.post("/forecast/decide")
async def forecast_decide(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    decision = None

    try:
        strategy_tag = body.get("strategy_tag")
        pocket = str(body.get("pocket") or "").strip().lower()
        side = str(body.get("side") or "").strip().lower()
        units = _to_int(body.get("units"), 0)
        entry_thesis = body.get("entry_thesis")
        meta = body.get("meta")
        if not pocket:
            return _failure("pocket is required")
        if units == 0:
            return _failure("units must be non-zero")
        if isinstance(meta, dict):
            normalized_meta = dict(meta)
        else:
            normalized_meta = None

        decision = forecast_gate.decide(
            strategy_tag=strategy_tag,
            pocket=pocket,
            side=side,
            units=units,
            entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
            meta=normalized_meta,
        )
    except Exception as exc:
        return _failure(str(exc))

    if decision is None:
        return _success(None)
    return _success(_serialize_decision(decision))


@app.get("/forecast/predictions")
def forecast_predictions() -> dict[str, Any]:
    try:
        bundle = forecast_gate._load_bundle_cached()  # noqa: SLF001
        rows = forecast_gate._ensure_predictions(bundle)  # noqa: SLF001
        if not isinstance(rows, dict) or not rows:
            return _failure("NO_PREDICTIONS")
        return _success(rows)
    except Exception as exc:
        return _failure(str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "quant-forecast"}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    host = os.getenv("FORECAST_SERVICE_HOST", "0.0.0.0")
    raw_port = os.getenv("FORECAST_SERVICE_PORT", "8302")
    try:
        port = int(float(raw_port))
    except (TypeError, ValueError):
        port = 8302
    uvicorn.run(
        "workers.forecast.worker:app",
        host=host,
        port=port,
        log_config=None,
    )
