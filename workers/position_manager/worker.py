"""Dedicated position-manager worker.

This worker owns all position/trade persistence and summary query paths so
strategy workers can consume a single source of truth.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import Body, FastAPI, Request
import uvicorn

os.environ["POSITION_MANAGER_SERVICE_ENABLED"] = "0"
os.environ["POSITION_MANAGER_SERVICE_FALLBACK_LOCAL"] = "1"

from execution import position_manager

# Force this worker to operate in pure local mode even if shared runtime env files
# still contain service-mode values.
position_manager._POSITION_MANAGER_SERVICE_ENABLED = False
position_manager._POSITION_MANAGER_SERVICE_FALLBACK_LOCAL = True

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    pm = position_manager.PositionManager()
    app.state.position_manager = pm
    try:
        yield
    finally:
        try:
            pm.close()
        except Exception:
            LOG.debug("[POSITION_MANAGER_WORKER] close failed", exc_info=True)


app = FastAPI(
    title="QuantRabbit Position Manager",
    version="v2",
    lifespan=_lifespan,
)


def _as_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "f", "no", "off", "disabled"}:
            return False
    try:
        return bool(int(float(value)))
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _success(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def _failure(message: str) -> dict[str, Any]:
    LOG.warning("[POSITION_MANAGER_WORKER] request failed: %s", message)
    return {"ok": False, "error": message}


def _manager(request: Request) -> position_manager.PositionManager:
    manager = getattr(request.app.state, "position_manager", None)
    if manager is None:
        raise RuntimeError("position manager not initialized")
    return manager


@app.post("/position/sync_trades")
def sync_trades(request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    body = _as_dict(payload)
    max_fetch = _to_int(body.get("max_fetch"), 1000)
    try:
        result = _manager(request).sync_trades(max_fetch=max_fetch)
    except Exception as exc:
        return _failure(str(exc))
    return _success(result)


@app.get("/position/open_positions")
def get_open_positions(request: Request, include_unknown: bool = True) -> dict[str, Any]:
    try:
        result = _manager(request).get_open_positions(include_unknown=include_unknown)
    except Exception as exc:
        return _failure(str(exc))
    if not isinstance(result, dict):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/open_positions")
def post_open_positions(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    include_unknown = _to_bool(body.get("include_unknown"), default=True)
    try:
        result = _manager(request).get_open_positions(include_unknown=include_unknown)
    except Exception as exc:
        return _failure(str(exc))
    if not isinstance(result, dict):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/performance_summary")
def get_performance_summary(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    parsed_now = _to_datetime(body.get("now"))
    try:
        result = _manager(request).get_performance_summary(now=parsed_now)
    except Exception as exc:
        return _failure(str(exc))
    if not isinstance(result, dict):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/fetch_recent_trades")
def fetch_recent_trades(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    limit = _to_int(body.get("limit"), 50)
    try:
        result = _manager(request).fetch_recent_trades(limit=limit)
    except Exception as exc:
        return _failure(str(exc))
    if not isinstance(result, list):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/close")
def close_manager(request: Request) -> dict[str, Any]:
    try:
        _manager(request).close()
        return _success(True)
    except Exception as exc:
        return _failure(str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "quant-position-manager"}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    uvicorn.run(
        "workers.position_manager.worker:app",
        host="0.0.0.0",
        port=8301,
        log_config=None,
    )
