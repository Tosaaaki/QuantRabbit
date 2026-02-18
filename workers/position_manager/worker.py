"""Dedicated position-manager worker.

This worker owns all position/trade persistence and summary query paths so
strategy workers can consume a single source of truth.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import logging
import os
import threading
import time
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_WORKER_OPEN_POSITIONS_TIMEOUT_SEC = max(
    0.5,
    _env_float("POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC", 8.0),
)
_WORKER_OPEN_POSITIONS_CACHE_TTL_SEC = max(
    0.0,
    _env_float("POSITION_MANAGER_WORKER_OPEN_POSITIONS_CACHE_TTL_SEC", 0.9),
)
_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC = max(
    _WORKER_OPEN_POSITIONS_CACHE_TTL_SEC,
    _env_float("POSITION_MANAGER_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC", 15.0),
)
_WORKER_SYNC_TRADES_TIMEOUT_SEC = max(
    0.5,
    _env_float("POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC", 6.0),
)
_WORKER_SYNC_TRADES_CACHE_TTL_SEC = max(
    0.0,
    _env_float("POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC", 1.0),
)
_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC = max(
    _WORKER_SYNC_TRADES_CACHE_TTL_SEC,
    _env_float("POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC", 12.0),
)
_WORKER_FAILURE_LOG_INTERVAL_SEC = max(
    1.0,
    _env_float("POSITION_MANAGER_WORKER_FAILURE_LOG_INTERVAL_SEC", 12.0),
)
_WORKER_BUSY_LOG_INTERVAL_SEC = max(
    1.0,
    _env_float("POSITION_MANAGER_WORKER_BUSY_LOG_INTERVAL_SEC", 30.0),
)
_FAILURE_LAST_LOG_TS: dict[str, float] = {}
_FAILURE_LOG_LOCK = threading.Lock()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    pm = None
    init_error: Exception | None = None
    for attempt in range(3):
        try:
            pm = position_manager.PositionManager()
            init_error = None
            break
        except Exception as exc:
            init_error = exc
            if attempt < 2:
                sleep_seconds = 2 ** attempt
                LOG.warning(
                    "[POSITION_MANAGER_WORKER] init attempt failed (attempt=%s): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(sleep_seconds)
            else:
                LOG.exception(
                    "[POSITION_MANAGER_WORKER] init failed after retries: %s",
                    exc,
                )

    if pm is None:
        # Keep service process alive; callers get explicit failure and can retry.
        LOG.error(
            "[POSITION_MANAGER_WORKER] position_manager unavailable: %s",
            init_error,
        )
    app.state.position_manager = pm
    app.state.position_manager_init_error = str(init_error) if init_error else None
    app.state.position_manager_open_positions_call_lock = threading.Lock()
    app.state.position_manager_db_call_lock = threading.Lock()
    app.state.open_positions_cache: dict[bool, tuple[float, dict[str, Any]]] = {}
    app.state.sync_trades_cache: dict[int, tuple[float, list[dict[str, Any]]]] = {}
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
    text = str(message)
    key = text.strip().lower() or "unknown"
    interval = (
        _WORKER_BUSY_LOG_INTERVAL_SEC
        if key.startswith("position manager busy")
        else _WORKER_FAILURE_LOG_INTERVAL_SEC
    )
    should_log = False
    now = time.monotonic()
    with _FAILURE_LOG_LOCK:
        last = _FAILURE_LAST_LOG_TS.get(key, 0.0)
        if now - last >= interval:
            _FAILURE_LAST_LOG_TS[key] = now
            should_log = True
    if should_log:
        LOG.warning("[POSITION_MANAGER_WORKER] request failed: %s", text)
    return {"ok": False, "error": text}


def _manager(request: Request) -> position_manager.PositionManager:
    manager = getattr(request.app.state, "position_manager", None)
    if manager is None:
        init_error = getattr(request.app.state, "position_manager_init_error", None)
        raise RuntimeError(
            f"position manager not initialized: {init_error or 'unknown'}"
        )
    return manager


def _cache_lookup(
    cache: dict[Any, tuple[float, Any]],
    key: Any,
    *,
    max_age_sec: float,
) -> tuple[Any, float] | None:
    entry = cache.get(key)
    if not entry:
        return None
    ts, value = entry
    age = max(0.0, time.monotonic() - float(ts))
    if age > max_age_sec:
        return None
    return copy.deepcopy(value), age


def _cache_store(cache: dict[Any, tuple[float, Any]], key: Any, value: Any) -> None:
    cache[key] = (time.monotonic(), copy.deepcopy(value))


def _try_acquire_call_lock(request: Request, lock_name: str) -> bool:
    lock = getattr(request.app.state, lock_name, None)
    if lock is None:
        lock = threading.Lock()
        setattr(request.app.state, lock_name, lock)
    return bool(lock.acquire(blocking=False))


def _release_call_lock(request: Request, lock_name: str) -> None:
    lock = getattr(request.app.state, lock_name, None)
    if lock is None:
        return
    try:
        lock.release()
    except RuntimeError:
        pass


def _mark_open_positions_cache_meta(
    payload: dict[str, Any],
    *,
    age_sec: float,
    stale: bool,
    reason: str | None,
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    meta = result.get("__meta__")
    if not isinstance(meta, dict):
        meta = {}
        result["__meta__"] = meta
    meta["worker_cache"] = True
    meta["worker_cache_age_sec"] = round(max(age_sec, 0.0), 2)
    if stale:
        meta["stale"] = True
    if reason:
        meta["worker_cache_reason"] = reason
    return result


async def _call_manager_with_timeout(
    request: Request,
    method_name: str,
    *,
    timeout_sec: float,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    manager = _manager(request)
    call_kwargs = kwargs or {}

    def _invoke() -> Any:
        method = getattr(manager, method_name)
        return method(**call_kwargs)

    return await asyncio.wait_for(
        asyncio.to_thread(_invoke),
        timeout=max(0.5, float(timeout_sec)),
    )


async def _handle_open_positions(
    request: Request,
    *,
    include_unknown: bool,
) -> dict[str, Any]:
    key = bool(include_unknown)
    cache = request.app.state.open_positions_cache

    cached = _cache_lookup(
        cache,
        key,
        max_age_sec=_WORKER_OPEN_POSITIONS_CACHE_TTL_SEC,
    )
    if cached is not None:
        payload, age_sec = cached
        if isinstance(payload, dict):
            return _success(
                _mark_open_positions_cache_meta(
                    payload,
                    age_sec=age_sec,
                    stale=False,
                    reason="fresh",
                )
            )

    if not _try_acquire_call_lock(
        request,
        "position_manager_open_positions_call_lock",
    ):
        stale = _cache_lookup(
            cache,
            key,
            max_age_sec=_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC,
        )
        if stale is not None:
            payload, age_sec = stale
            if isinstance(payload, dict):
                return _success(
                    _mark_open_positions_cache_meta(
                        payload,
                        age_sec=age_sec,
                        stale=True,
                        reason="manager_busy",
                    )
                )
        return _failure("position manager busy")

    try:
        try:
            result = await _call_manager_with_timeout(
                request,
                "get_open_positions",
                timeout_sec=_WORKER_OPEN_POSITIONS_TIMEOUT_SEC,
                kwargs={"include_unknown": include_unknown},
            )
        except asyncio.TimeoutError:
            stale = _cache_lookup(
                cache,
                key,
                max_age_sec=_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC,
            )
            if stale is not None:
                payload, age_sec = stale
                if isinstance(payload, dict):
                    return _success(
                        _mark_open_positions_cache_meta(
                            payload,
                            age_sec=age_sec,
                            stale=True,
                            reason="timeout",
                        )
                    )
            return _failure(
                f"open_positions timeout ({_WORKER_OPEN_POSITIONS_TIMEOUT_SEC:.1f}s)"
            )
        except Exception as exc:
            stale = _cache_lookup(
                cache,
                key,
                max_age_sec=_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC,
            )
            if stale is not None:
                payload, age_sec = stale
                if isinstance(payload, dict):
                    return _success(
                        _mark_open_positions_cache_meta(
                            payload,
                            age_sec=age_sec,
                            stale=True,
                            reason="error_fallback",
                        )
                    )
            return _failure(str(exc))

        if not isinstance(result, dict):
            return _failure("unexpected response type")
        _cache_store(cache, key, result)
        return _success(result)
    finally:
        _release_call_lock(request, "position_manager_open_positions_call_lock")


@app.post("/position/sync_trades")
async def sync_trades(
    request: Request, payload: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    body = _as_dict(payload)
    max_fetch = max(1, _to_int(body.get("max_fetch"), 1000))
    key = int(max_fetch)
    cache = request.app.state.sync_trades_cache

    cached = _cache_lookup(
        cache,
        key,
        max_age_sec=_WORKER_SYNC_TRADES_CACHE_TTL_SEC,
    )
    if cached is not None:
        result, _ = cached
        if isinstance(result, list):
            return _success(result)

    if not _try_acquire_call_lock(
        request,
        "position_manager_db_call_lock",
    ):
        stale = _cache_lookup(
            cache,
            key,
            max_age_sec=_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC,
        )
        if stale is not None:
            result, _ = stale
            if isinstance(result, list):
                return _success(result)
        return _failure("position manager busy")

    try:
        try:
            raw = await _call_manager_with_timeout(
                request,
                "sync_trades",
                timeout_sec=_WORKER_SYNC_TRADES_TIMEOUT_SEC,
                kwargs={"max_fetch": max_fetch},
            )
        except asyncio.TimeoutError:
            stale = _cache_lookup(
                cache,
                key,
                max_age_sec=_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC,
            )
            if stale is not None:
                result, _ = stale
                if isinstance(result, list):
                    return _success(result)
            return _failure(
                f"sync_trades timeout ({_WORKER_SYNC_TRADES_TIMEOUT_SEC:.1f}s)"
            )
        except Exception as exc:
            stale = _cache_lookup(
                cache,
                key,
                max_age_sec=_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC,
            )
            if stale is not None:
                result, _ = stale
                if isinstance(result, list):
                    return _success(result)
            return _failure(str(exc))

        if isinstance(raw, list):
            result = raw
        elif isinstance(raw, dict):
            result = list(raw.values()) if raw else []
        else:
            return _failure("unexpected response type")
        _cache_store(cache, key, result)
        return _success(result)
    finally:
        _release_call_lock(request, "position_manager_db_call_lock")


@app.get("/position/open_positions")
async def get_open_positions(
    request: Request, include_unknown: bool = True
) -> dict[str, Any]:
    return await _handle_open_positions(
        request,
        include_unknown=include_unknown,
    )


@app.post("/position/open_positions")
async def post_open_positions(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    include_unknown = _to_bool(body.get("include_unknown"), default=True)
    return await _handle_open_positions(
        request,
        include_unknown=include_unknown,
    )


@app.post("/position/performance_summary")
async def get_performance_summary(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    parsed_now = _to_datetime(body.get("now"))
    if not _try_acquire_call_lock(
        request,
        "position_manager_db_call_lock",
    ):
        return _failure("position manager busy")
    try:
        result = await _call_manager_with_timeout(
            request,
            "get_performance_summary",
            timeout_sec=_WORKER_SYNC_TRADES_TIMEOUT_SEC,
            kwargs={"now": parsed_now},
        )
    except Exception as exc:
        return _failure(str(exc))
    finally:
        _release_call_lock(request, "position_manager_db_call_lock")
    if not isinstance(result, dict):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/fetch_recent_trades")
async def fetch_recent_trades(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    body = _as_dict(payload)
    limit = _to_int(body.get("limit"), 50)
    if not _try_acquire_call_lock(
        request,
        "position_manager_db_call_lock",
    ):
        return _failure("position manager busy")
    try:
        result = await _call_manager_with_timeout(
            request,
            "fetch_recent_trades",
            timeout_sec=_WORKER_SYNC_TRADES_TIMEOUT_SEC,
            kwargs={"limit": limit},
        )
    except Exception as exc:
        return _failure(str(exc))
    finally:
        _release_call_lock(request, "position_manager_db_call_lock")
    if not isinstance(result, list):
        return _failure("unexpected response type")
    return _success(result)


@app.post("/position/close")
async def close_manager(request: Request) -> dict[str, Any]:
    if not _try_acquire_call_lock(
        request,
        "position_manager_db_call_lock",
    ):
        return _failure("position manager busy")
    try:
        await _call_manager_with_timeout(
            request,
            "close",
            timeout_sec=_WORKER_SYNC_TRADES_TIMEOUT_SEC,
            kwargs={},
        )
        return _success(True)
    except Exception as exc:
        return _failure(str(exc))
    finally:
        _release_call_lock(request, "position_manager_db_call_lock")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "quant-position-manager"}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


if __name__ == "__main__":
    _configure_logging()
    uvicorn.run(
        "workers.position_manager.worker:app",
        host="0.0.0.0",
        port=8301,
        log_config=None,
        access_log=_env_bool("POSITION_MANAGER_ACCESS_LOG", False),
    )
