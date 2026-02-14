"""Dedicated order-manager worker.

This worker owns OANDA order placement and close paths by exposing a small HTTP
control plane and internally delegating to ``execution.order_manager``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Body, FastAPI
import uvicorn

os.environ["ORDER_MANAGER_SERVICE_ENABLED"] = "0"
os.environ["ORDER_MANAGER_SERVICE_FALLBACK_LOCAL"] = "1"

from execution import order_manager

# Force this worker to operate in pure local mode even if shared runtime env files
# still contain service-mode values.
order_manager._ORDER_MANAGER_SERVICE_ENABLED = False
order_manager._ORDER_MANAGER_SERVICE_FALLBACK_LOCAL = True

LOG = logging.getLogger(__name__)

app = FastAPI(
    title="QuantRabbit Order Manager",
    version="v2",
)


def _to_bool(value: Any, default: bool = False) -> bool:
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


def _as_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _success(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def _failure(message: str) -> dict[str, Any]:
    LOG.warning("[ORDER_MANAGER_WORKER] request failed: %s", message)
    return {"ok": False, "error": message}


@app.post("/order/cancel_order")
async def cancel_order(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    order_id = str(body.get("order_id") or "").strip()
    if not order_id:
        return _failure("order_id is required")
    try:
        result = await order_manager.cancel_order(
            order_id=order_id,
            pocket=body.get("pocket"),
            client_order_id=body.get("client_order_id"),
            reason=str(body.get("reason") or "user_cancel"),
        )
    except Exception as exc:
        return _failure(str(exc))
    return _success(result)


@app.post("/order/close_trade")
async def close_trade(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    trade_id = str(body.get("trade_id") or "").strip()
    if not trade_id:
        return _failure("trade_id is required")
    try:
        result = await order_manager.close_trade(
            trade_id=trade_id,
            units=_to_int(body.get("units"), 0) if body.get("units") is not None else None,
            client_order_id=body.get("client_order_id"),
            allow_negative=_to_bool(body.get("allow_negative"), False),
            exit_reason=body.get("exit_reason"),
        )
    except Exception as exc:
        return _failure(str(exc))
    return _success(result)


@app.post("/order/set_trade_protections")
async def set_trade_protections(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    trade_id = str(body.get("trade_id") or "").strip()
    if not trade_id:
        return _failure("trade_id is required")
    try:
        sl = _to_float(body.get("sl_price"))
        tp = _to_float(body.get("tp_price"))
        result = await order_manager.set_trade_protections(
            trade_id=trade_id,
            sl_price=sl,
            tp_price=tp,
        )
    except Exception as exc:
        return _failure(str(exc))
    return _success(result)


@app.post("/order/market_order")
async def market_order(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    instrument = str(body.get("instrument") or "USD_JPY").strip()
    units = _to_int(body.get("units"), 0)
    if not instrument:
        return _failure("instrument is required")
    if units == 0:
        return _failure("units must be non-zero")

    try:
        result = await order_manager.market_order(
            instrument=instrument,
            units=units,
            sl_price=_to_float(body.get("sl_price")),
            tp_price=_to_float(body.get("tp_price")),
            pocket=str(body.get("pocket") or "manual").strip().lower(),
            client_order_id=body.get("client_order_id"),
            strategy_tag=body.get("strategy_tag"),
            reduce_only=_to_bool(body.get("reduce_only"), False),
            entry_thesis=body.get("entry_thesis"),
            meta=body.get("meta"),
            confidence=_to_int(body.get("confidence"), 0) if body.get("confidence") is not None else None,
            stage_index= _to_int(body.get("stage_index"), 0) if body.get("stage_index") is not None else None,
            arbiter_final=_to_bool(body.get("arbiter_final"), False),
        )
    except Exception as exc:
        return _failure(str(exc))
    return _success(result)


@app.post("/order/limit_order")
async def limit_order(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    body = _as_dict(payload)
    instrument = str(body.get("instrument") or "USD_JPY").strip()
    units = _to_int(body.get("units"), 0)
    price = _to_float(body.get("price"))
    if not instrument:
        return _failure("instrument is required")
    if units == 0:
        return _failure("units must be non-zero")
    if price is None:
        return _failure("price is required")

    try:
        trade_id, order_id = await order_manager.limit_order(
            instrument=instrument,
            units=units,
            price=price,
            sl_price=_to_float(body.get("sl_price")),
            tp_price=_to_float(body.get("tp_price")),
            pocket=str(body.get("pocket") or "manual").strip().lower(),
            current_bid=_to_float(body.get("current_bid")),
            current_ask=_to_float(body.get("current_ask")),
            require_passive=_to_bool(body.get("require_passive"), True),
            client_order_id=body.get("client_order_id"),
            reduce_only=_to_bool(body.get("reduce_only"), False),
            ttl_ms=_to_float(body.get("ttl_ms"), 800.0) or 800.0,
            entry_thesis=body.get("entry_thesis"),
            confidence=_to_int(body.get("confidence"), 0) if body.get("confidence") is not None else None,
            meta=body.get("meta"),
        )
    except Exception as exc:
        return _failure(str(exc))
    return _success(
        {
            "trade_id": trade_id,
            "order_id": order_id,
        }
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "quant-order-manager"}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    host = os.getenv("ORDER_MANAGER_SERVICE_HOST", "0.0.0.0")
    port = _to_int(os.getenv("ORDER_MANAGER_SERVICE_PORT"), 8300)
    uvicorn.run(
        "workers.order_manager.worker:app",
        host=host,
        port=port,
        log_config=None,
    )
