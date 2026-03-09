from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

from utils.metrics_logger import log_metric
from utils.secrets import get_secret


@dataclasses.dataclass
class AccountSnapshot:
    nav: float
    balance: float
    margin_available: float
    margin_used: float
    margin_rate: float
    unrealized_pl: float
    free_margin_ratio: Optional[float]
    health_buffer: Optional[float]
    # Optional: callers may attach per-pocket exposure snapshots.
    positions: Optional[dict[str, Any]] = None


@dataclasses.dataclass
class AccountSnapshotState:
    snapshot: AccountSnapshot
    source: str
    age_sec: float
    stale: bool
    error_kind: Optional[str] = None


_SIDE_MARGIN_RATIO_ENABLED = (
    os.getenv("SIDE_SEPARATE_FREE_MARGIN_RATIO", "1").strip().lower()
    not in {"", "0", "false", "off"}
)


def _side_usage_ratio(
    units: float,
    price: float,
    nav: float,
    margin_rate: float,
) -> Optional[float]:
    if units <= 0.0 or nav <= 0.0 or margin_rate <= 0.0 or price <= 0.0:
        return 0.0 if units <= 0.0 else None
    used = units * price * margin_rate
    if used <= 0.0:
        return 0.0
    return max(0.0, min(1.0, used / nav))


def _side_free_margin_ratio(snapshot: "AccountSnapshot") -> Optional[float]:
    """
    Estimate side-separate free margin ratio.

    Strategy entry guards that currently use the global
    free margin ratio can use this value to keep long/short side capacity
    independent when deciding whether to allow new entries.
    """
    if not _SIDE_MARGIN_RATIO_ENABLED:
        return None
    if snapshot is None:
        return None
    try:
        nav = float(snapshot.nav)
        margin_rate = float(snapshot.margin_rate)
        margin_used = float(snapshot.margin_used)
    except Exception:
        return None
    if nav <= 0.0 or margin_rate <= 0.0:
        return None

    try:
        long_units, short_units = get_position_summary()
        long_units = max(0.0, float(long_units or 0.0))
        short_units = max(0.0, float(short_units or 0.0))
    except Exception:
        return None

    if long_units <= 0.0 and short_units <= 0.0 and margin_used > 0.0:
        return None

    net_units = long_units - short_units
    long_price_hint = None
    short_price_hint = None
    if margin_used > 0.0 and abs(net_units) > 0.0:
        inferred_price = margin_used / (abs(net_units) * margin_rate)
        if long_units > 0.0 and short_units <= 0.0:
            long_price_hint = inferred_price
        elif short_units > 0.0 and long_units <= 0.0:
            short_price_hint = inferred_price

    long_usage = _side_usage_ratio(long_units, long_price_hint or 0.0, nav, margin_rate)
    short_usage = _side_usage_ratio(short_units, short_price_hint or 0.0, nav, margin_rate)

    if long_units > 0.0 and long_usage is None:
        return None
    if short_units > 0.0 and short_usage is None:
        return None

    long_free_ratio = 1.0 - (long_usage or 0.0)
    short_free_ratio = 1.0 - (short_usage or 0.0)
    # A shared snapshot is consumed without side context in many workers.
    # Never widen the global free-margin ratio based on the "empty" side,
    # otherwise one-sided exposure can be misread as fully healthy.
    return max(0.0, min(1.0, min(long_free_ratio, short_free_ratio)))


def _apply_side_free_margin_ratio(snapshot: Optional[AccountSnapshot]) -> Optional[AccountSnapshot]:
    """Attach side-separate free-margin ratio when available."""
    if snapshot is None or not _SIDE_MARGIN_RATIO_ENABLED:
        return snapshot
    side_free_ratio = _side_free_margin_ratio(snapshot)
    global_free_ratio = snapshot.free_margin_ratio
    if (
        side_free_ratio is None
        or side_free_ratio == global_free_ratio
        or (
            global_free_ratio is not None
            and side_free_ratio > float(global_free_ratio)
        )
    ):
        return snapshot
    try:
        return dataclasses.replace(snapshot, free_margin_ratio=side_free_ratio)
    except Exception:
        snapshot.free_margin_ratio = side_free_ratio
        return snapshot


_BASE_DIR = Path(__file__).resolve().parents[1]
_LOG_DIR = (_BASE_DIR / "logs").resolve()

_SHARED_CACHE_ENABLED = os.getenv("OANDA_SHARED_CACHE_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_LOCK_STALE_SEC = max(2.0, float(os.getenv("OANDA_SHARED_CACHE_LOCK_STALE_SEC", "12") or 12.0))
_ALLOW_STALE_SEC = max(0.0, float(os.getenv("OANDA_SHARED_CACHE_ALLOW_STALE_SEC", "5") or 5.0))

_LAST_SNAPSHOT: AccountSnapshot | None = None
_LAST_SNAPSHOT_TS: float | None = None
_LAST_SNAPSHOT_ENV: str | None = None

# Disk-cache state (per process)
_ACCOUNT_DISK_STATE: dict[str, Any] = {"path": "", "mtime": 0.0, "ts": 0.0, "data": None}
_POS_STATE: dict[tuple[str, str], dict[str, Any]] = {}


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _account_cache_paths(env_name: str) -> tuple[Path, Path]:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cache = _LOG_DIR / f"oanda_account_snapshot_{env_name}.json"
    lock = _LOG_DIR / f"oanda_account_snapshot_{env_name}.lock"
    return cache, lock


def _pos_cache_paths(env_name: str, instrument: str) -> tuple[Path, Path]:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(instrument).upper()) or "UNKNOWN"
    cache = _LOG_DIR / f"oanda_open_positions_{env_name}_{safe}.json"
    lock = _LOG_DIR / f"oanda_open_positions_{env_name}.lock"
    return cache, lock


def _try_acquire_lock(path: Path) -> bool:
    """Best-effort singleflight across processes using an exclusive lockfile."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(time.time()).encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - float(path.stat().st_mtime)
            if age > _LOCK_STALE_SEC:
                path.unlink(missing_ok=True)
                return _try_acquire_lock(path)
        except Exception:
            pass
        return False
    except Exception:
        return False


def _release_lock(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _read_json(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # cache is best-effort; never fail the caller
        return


def _snapshot_from_payload(payload: dict) -> Optional[AccountSnapshot]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    margin_rate = _as_float(data.get("margin_rate"), 0.0)
    if margin_rate <= 0:
        return None
    free_ratio = data.get("free_margin_ratio")
    health_buffer = data.get("health_buffer")
    return AccountSnapshot(
        nav=_as_float(data.get("nav"), 0.0),
        balance=_as_float(data.get("balance"), 0.0),
        margin_available=_as_float(data.get("margin_available"), 0.0),
        margin_used=_as_float(data.get("margin_used"), 0.0),
        margin_rate=margin_rate,
        unrealized_pl=_as_float(data.get("unrealized_pl"), 0.0),
        free_margin_ratio=None if free_ratio is None else _as_float(free_ratio, 0.0),
        health_buffer=None if health_buffer is None else _as_float(health_buffer, 0.0),
        positions=data.get("positions") if isinstance(data.get("positions"), dict) else None,
    )


def _snapshot_state(
    snapshot: AccountSnapshot,
    *,
    source: str,
    ts: float | None,
    now: float | None = None,
    stale: bool = False,
    error_kind: Optional[str] = None,
) -> AccountSnapshotState:
    now_ts = time.time() if now is None else float(now)
    snap_ts = float(ts or 0.0)
    age_sec = max(0.0, now_ts - snap_ts) if snap_ts > 0.0 else 0.0
    return AccountSnapshotState(
        snapshot=snapshot,
        source=str(source),
        age_sec=age_sec,
        stale=bool(stale),
        error_kind=error_kind,
    )


def _account_allow_stale_sec(ttl: float, allow_stale_sec: float | None) -> float:
    cfg = _ALLOW_STALE_SEC if allow_stale_sec is None else max(0.0, float(allow_stale_sec))
    return max(float(ttl), cfg)


def _classify_snapshot_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            return f"http_{int(status)}"
        return "http_error"
    if isinstance(exc, requests.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, requests.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.RequestException):
        return "request_exception"
    return exc.__class__.__name__.lower()


def _load_account_disk(cache_path: Path, *, force: bool = False) -> tuple[AccountSnapshot | None, float | None]:
    global _ACCOUNT_DISK_STATE
    if not _SHARED_CACHE_ENABLED:
        return None, None
    try:
        stat = cache_path.stat()
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

    if _ACCOUNT_DISK_STATE.get("path") != str(cache_path):
        _ACCOUNT_DISK_STATE = {"path": str(cache_path), "mtime": 0.0, "ts": 0.0, "data": None}

    mtime = float(stat.st_mtime)
    cached_mtime = _as_float(_ACCOUNT_DISK_STATE.get("mtime"), 0.0)
    if force or (mtime > 0 and mtime != cached_mtime):
        payload = _read_json(cache_path)
        ts = _as_float((payload or {}).get("ts"), 0.0) if payload else 0.0
        snap = _snapshot_from_payload(payload or {})
        if snap is not None and ts > 0:
            _ACCOUNT_DISK_STATE.update({"mtime": mtime, "ts": ts, "data": snap})

    data = _ACCOUNT_DISK_STATE.get("data")
    if isinstance(data, AccountSnapshot):
        ts = _as_float(_ACCOUNT_DISK_STATE.get("ts"), 0.0)
        return data, (ts or None)
    return None, None


def get_account_snapshot_state(
    timeout: float = 7.0,
    *,
    cache_ttl_sec: float | None = None,
    allow_stale_sec: float | None = None,
) -> AccountSnapshotState:
    """
    Retrieve OANDA account summary and return cache metadata.

    - In-process cache: avoids repeated REST calls within a loop.
    - Shared disk cache: logs/*.json enables multi-worker reuse.
    """
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except KeyError:
        practice = False
    env_name = "practice" if practice else "live"
    ttl = cache_ttl_sec
    if ttl is None:
        ttl = _as_float(os.getenv("OANDA_ACCOUNT_SNAPSHOT_TTL_SEC"), 1.0)
    ttl = max(0.0, float(ttl))
    allow_stale = _account_allow_stale_sec(ttl, allow_stale_sec)
    base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"

    headers = {"Authorization": f"Bearer {token}"}
    global _LAST_SNAPSHOT, _LAST_SNAPSHOT_TS, _LAST_SNAPSHOT_ENV

    now = time.time()
    if (
        ttl > 0
        and _LAST_SNAPSHOT is not None
        and _LAST_SNAPSHOT_TS is not None
        and _LAST_SNAPSHOT_ENV == env_name
        and (now - _LAST_SNAPSHOT_TS) <= ttl
    ):
        snapshot = _apply_side_free_margin_ratio(_LAST_SNAPSHOT)
        _LAST_SNAPSHOT = snapshot
        return _snapshot_state(snapshot, source="memory_cache", ts=_LAST_SNAPSHOT_TS, now=now, stale=False)

    cache_path, lock_path = _account_cache_paths(env_name)
    disk_snapshot, disk_ts = _load_account_disk(cache_path)
    if ttl > 0 and disk_snapshot is not None and disk_ts is not None and (now - disk_ts) <= ttl:
        snapshot = _apply_side_free_margin_ratio(disk_snapshot)
        _LAST_SNAPSHOT = snapshot
        _LAST_SNAPSHOT_TS = disk_ts
        _LAST_SNAPSHOT_ENV = env_name
        return _snapshot_state(snapshot, source="disk_cache", ts=disk_ts, now=now, stale=False)

    lock_acquired = False
    if _SHARED_CACHE_ENABLED and ttl > 0:
        lock_acquired = _try_acquire_lock(lock_path)

    try:
        # Another process is refreshing; return stale cache if still usable.
        if _SHARED_CACHE_ENABLED and ttl > 0 and not lock_acquired:
            now = time.time()
            disk_snapshot, disk_ts = _load_account_disk(cache_path)
            if disk_snapshot is not None and disk_ts is not None and (now - disk_ts) <= allow_stale:
                snapshot = _apply_side_free_margin_ratio(disk_snapshot)
                _LAST_SNAPSHOT = snapshot
                _LAST_SNAPSHOT_TS = disk_ts
                _LAST_SNAPSHOT_ENV = env_name
                return _snapshot_state(
                    snapshot,
                    source="disk_cache",
                    ts=disk_ts,
                    now=now,
                    stale=(now - disk_ts) > ttl,
                    error_kind="refresh_in_progress" if (now - disk_ts) > ttl else None,
                )
            if (
                _LAST_SNAPSHOT is not None
                and _LAST_SNAPSHOT_TS is not None
                and _LAST_SNAPSHOT_ENV == env_name
                and (now - _LAST_SNAPSHOT_TS) <= allow_stale
            ):
                snapshot = _apply_side_free_margin_ratio(_LAST_SNAPSHOT)
                _LAST_SNAPSHOT = snapshot
                return _snapshot_state(
                    snapshot,
                    source="memory_cache",
                    ts=_LAST_SNAPSHOT_TS,
                    now=now,
                    stale=(now - _LAST_SNAPSHOT_TS) > ttl,
                    error_kind="refresh_in_progress" if (now - _LAST_SNAPSHOT_TS) > ttl else None,
                )

        # Re-check disk cache after acquiring the lock (another process may have refreshed earlier).
        if lock_acquired and _SHARED_CACHE_ENABLED and ttl > 0:
            now = time.time()
            disk_snapshot, disk_ts = _load_account_disk(cache_path, force=True)
            if disk_snapshot is not None and disk_ts is not None and (now - disk_ts) <= ttl:
                snapshot = _apply_side_free_margin_ratio(disk_snapshot)
                _LAST_SNAPSHOT = snapshot
                _LAST_SNAPSHOT_TS = disk_ts
                _LAST_SNAPSHOT_ENV = env_name
                return _snapshot_state(snapshot, source="disk_cache", ts=disk_ts, now=now, stale=False)

        resp = requests.get(
            f"{base}/v3/accounts/{account}/summary",
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        acc = resp.json().get("account", {})

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(acc.get(key, default))
            except (TypeError, ValueError):
                return default

        nav = _f("NAV", 0.0)
        balance = _f("balance", nav)
        margin_available = _f("marginAvailable", 0.0)
        margin_used = _f("marginUsed", 0.0)
        margin_rate = _f("marginRate", 0.0)
        unrealized = _f("unrealizedPL", 0.0)
        margin_closeout = _f("marginCloseoutPercent", 0.0)

        if margin_rate <= 0:
            # Prefer a recent "healthy" snapshot over a broken one.
            disk_snapshot, disk_ts = _load_account_disk(cache_path, force=True)
            if disk_snapshot is not None and (disk_snapshot.margin_rate or 0) > 0:
                snapshot = _apply_side_free_margin_ratio(disk_snapshot)
                _LAST_SNAPSHOT = snapshot
                _LAST_SNAPSHOT_TS = disk_ts or time.time()
                _LAST_SNAPSHOT_ENV = env_name
                return _snapshot_state(
                    snapshot,
                    source="disk_cache",
                    ts=_LAST_SNAPSHOT_TS,
                    now=time.time(),
                    stale=(time.time() - float(_LAST_SNAPSHOT_TS or 0.0)) > ttl,
                    error_kind="margin_rate_missing",
                )
            if _LAST_SNAPSHOT is not None and (_LAST_SNAPSHOT.margin_rate or 0) > 0 and _LAST_SNAPSHOT_ENV == env_name:
                _LAST_SNAPSHOT = _apply_side_free_margin_ratio(_LAST_SNAPSHOT)
                return _snapshot_state(
                    _LAST_SNAPSHOT,
                    source="memory_cache",
                    ts=_LAST_SNAPSHOT_TS,
                    now=time.time(),
                    stale=(time.time() - float(_LAST_SNAPSHOT_TS or 0.0)) > ttl,
                    error_kind="margin_rate_missing",
                )
            raise RuntimeError("margin_rate_missing")

        free_ratio: Optional[float]
        if nav > 0:
            free_ratio = margin_available / nav
        else:
            free_ratio = None

        health_buffer: Optional[float]
        if margin_closeout > 0:
            health_buffer = max(0.0, 1.0 - margin_closeout)
        elif margin_used > 0:
            health_buffer = max(0.0, margin_available / (margin_used + 1e-9))
        else:
            health_buffer = free_ratio

        snapshot = AccountSnapshot(
            nav=nav,
            balance=balance,
            margin_available=margin_available,
            margin_used=margin_used,
            margin_rate=margin_rate,
            unrealized_pl=unrealized,
            free_margin_ratio=free_ratio,
            health_buffer=health_buffer,
        )
        snapshot = _apply_side_free_margin_ratio(snapshot) or snapshot
        free_ratio = snapshot.free_margin_ratio

        # record a small set of health metrics for downstream hazard tuning
        try:
            log_metric("account.nav", nav, tags={"practice": str(practice).lower()})
            log_metric("account.balance", balance, tags={"practice": str(practice).lower()})
            log_metric(
                "account.health_buffer",
                health_buffer if health_buffer is not None else -1.0,
                tags={"practice": str(practice).lower()},
            )
            if free_ratio is not None:
                log_metric("account.free_margin_ratio", free_ratio, tags={"practice": str(practice).lower()})
            if nav > 0:
                log_metric("account.margin_usage_ratio", margin_used / nav, tags={"practice": str(practice).lower()})
        except Exception:
            pass

        ts_now = time.time()
        _LAST_SNAPSHOT = snapshot
        _LAST_SNAPSHOT_TS = ts_now
        _LAST_SNAPSHOT_ENV = env_name

        if _SHARED_CACHE_ENABLED:
            payload = {
                "ts": ts_now,
                "data": {
                    "nav": nav,
                    "balance": balance,
                    "margin_available": margin_available,
                    "margin_used": margin_used,
                    "margin_rate": margin_rate,
                    "unrealized_pl": unrealized,
                    "free_margin_ratio": snapshot.free_margin_ratio,
                    "health_buffer": health_buffer,
                },
            }
            _atomic_write_json(cache_path, payload)
            try:
                mtime = float(cache_path.stat().st_mtime)
                _ACCOUNT_DISK_STATE.update({"path": str(cache_path), "mtime": mtime, "ts": ts_now, "data": snapshot})
            except Exception:
                pass
        return _snapshot_state(snapshot, source="live", ts=ts_now, now=ts_now, stale=False)
    except Exception as exc:
        # fall back to cache if possible
        now = time.time()
        error_kind = _classify_snapshot_error(exc)
        disk_snapshot, disk_ts = _load_account_disk(cache_path, force=True)
        if disk_snapshot is not None and disk_ts is not None and (now - disk_ts) <= allow_stale:
            snapshot = _apply_side_free_margin_ratio(disk_snapshot)
            _LAST_SNAPSHOT = snapshot
            _LAST_SNAPSHOT_TS = disk_ts
            _LAST_SNAPSHOT_ENV = env_name
            return _snapshot_state(
                snapshot,
                source="disk_cache",
                ts=disk_ts,
                now=now,
                stale=(now - disk_ts) > ttl,
                error_kind=error_kind,
            )
        if (
            _LAST_SNAPSHOT is not None
            and _LAST_SNAPSHOT_TS is not None
            and _LAST_SNAPSHOT_ENV == env_name
            and (now - _LAST_SNAPSHOT_TS) <= allow_stale
        ):
            snapshot = _apply_side_free_margin_ratio(_LAST_SNAPSHOT)
            _LAST_SNAPSHOT = snapshot
            return _snapshot_state(
                snapshot,
                source="memory_cache",
                ts=_LAST_SNAPSHOT_TS,
                now=now,
                stale=(now - _LAST_SNAPSHOT_TS) > ttl,
                error_kind=error_kind,
            )
        raise
    finally:
        if lock_acquired:
            _release_lock(lock_path)


def get_account_snapshot(
    timeout: float = 7.0,
    *,
    cache_ttl_sec: float | None = None,
    allow_stale_sec: float | None = None,
) -> AccountSnapshot:
    return get_account_snapshot_state(
        timeout=timeout,
        cache_ttl_sec=cache_ttl_sec,
        allow_stale_sec=allow_stale_sec,
    ).snapshot


def get_position_summary(
    instrument: str = "USD_JPY",
    timeout: float = 7.0,
    *,
    cache_ttl_sec: float | None = None,
) -> tuple[float, float]:
    """
    Return (long_units, short_units) for the given instrument from OANDA openPositions.
    Falls back to (0, 0) on errors.
    """
    ttl = cache_ttl_sec
    if ttl is None:
        ttl = _as_float(os.getenv("OANDA_POSITION_SUMMARY_TTL_SEC"), 1.0)
    ttl = max(0.0, float(ttl))
    allow_stale = _account_allow_stale_sec(ttl, None)

    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")
        try:
            practice = get_secret("oanda_practice").lower() == "true"
        except KeyError:
            practice = False
        env_name = "practice" if practice else "live"
        base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        headers = {"Authorization": f"Bearer {token}"}

        cache_path, lock_path = _pos_cache_paths(env_name, instrument)
        key = (env_name, str(instrument).upper())
        now = time.time()

        state = _POS_STATE.get(key)
        if ttl > 0 and state:
            ts = _as_float(state.get("ts"), 0.0)
            data = state.get("data")
            if ts > 0 and isinstance(data, tuple) and (now - ts) <= ttl:
                return float(data[0]), float(data[1])

        if _SHARED_CACHE_ENABLED:
            try:
                mtime = float(cache_path.stat().st_mtime)
                if not state or mtime != _as_float(state.get("mtime"), 0.0):
                    payload = _read_json(cache_path) or {}
                    ts = _as_float(payload.get("ts"), 0.0)
                    long_u = _as_float(payload.get("long_units"), 0.0)
                    short_u = _as_float(payload.get("short_units"), 0.0)
                    if ts > 0:
                        state = {"mtime": mtime, "ts": ts, "data": (long_u, short_u)}
                        _POS_STATE[key] = state
                if ttl > 0 and state:
                    ts = _as_float(state.get("ts"), 0.0)
                    data = state.get("data")
                    if ts > 0 and isinstance(data, tuple) and (now - ts) <= ttl:
                        return float(data[0]), float(data[1])
            except Exception:
                pass

        lock_acquired = False
        if _SHARED_CACHE_ENABLED and ttl > 0:
            lock_acquired = _try_acquire_lock(lock_path)

        try:
            if _SHARED_CACHE_ENABLED and ttl > 0 and not lock_acquired and state:
                ts = _as_float(state.get("ts"), 0.0)
                data = state.get("data")
                if ts > 0 and isinstance(data, tuple) and (now - ts) <= allow_stale:
                    return float(data[0]), float(data[1])

            if lock_acquired and _SHARED_CACHE_ENABLED and ttl > 0:
                try:
                    payload = _read_json(cache_path) or {}
                    ts = _as_float(payload.get("ts"), 0.0)
                    long_u = _as_float(payload.get("long_units"), 0.0)
                    short_u = _as_float(payload.get("short_units"), 0.0)
                    if ts > 0 and (now - ts) <= ttl:
                        _POS_STATE[key] = {
                            "mtime": _as_float(cache_path.stat().st_mtime, 0.0),
                            "ts": ts,
                            "data": (long_u, short_u),
                        }
                        return long_u, short_u
                except Exception:
                    pass

            resp = requests.get(
                f"{base}/v3/accounts/{account}/openPositions",
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            positions = data.get("positions") or []

            long_units = 0.0
            short_units = 0.0
            for pos in positions:
                if str(pos.get("instrument") or "").upper() != str(instrument).upper():
                    continue
                longs = pos.get("long") or {}
                shorts = pos.get("short") or {}
                long_units = _as_float(longs.get("units"), 0.0)
                short_units = abs(_as_float(shorts.get("units"), 0.0))
                break

            ts_now = time.time()
            _POS_STATE[key] = {"mtime": ts_now, "ts": ts_now, "data": (long_units, short_units)}
            if _SHARED_CACHE_ENABLED:
                payload = {
                    "ts": ts_now,
                    "instrument": str(instrument).upper(),
                    "long_units": long_units,
                    "short_units": short_units,
                }
                _atomic_write_json(cache_path, payload)
                try:
                    _POS_STATE[key]["mtime"] = float(cache_path.stat().st_mtime)
                except Exception:
                    pass
            return long_units, short_units
        finally:
            if lock_acquired:
                _release_lock(lock_path)
    except Exception as exc:  # noqa: BLE001
        now = time.time()
        state = _POS_STATE.get(key) if "key" in locals() else None
        if state:
            ts = _as_float(state.get("ts"), 0.0)
            data = state.get("data")
            if ts > 0 and isinstance(data, tuple) and (now - ts) <= allow_stale:
                return float(data[0]), float(data[1])
        cache_path = locals().get("cache_path")
        if isinstance(cache_path, Path):
            try:
                payload = _read_json(cache_path) or {}
                ts = _as_float(payload.get("ts"), 0.0)
                long_u = _as_float(payload.get("long_units"), 0.0)
                short_u = _as_float(payload.get("short_units"), 0.0)
                if ts > 0 and (now - ts) <= allow_stale:
                    env_key = locals().get("key")
                    if isinstance(env_key, tuple):
                        _POS_STATE[env_key] = {
                            "mtime": _as_float(cache_path.stat().st_mtime, 0.0),
                            "ts": ts,
                            "data": (long_u, short_u),
                        }
                    return long_u, short_u
            except Exception:
                pass
        try:
            log_metric("oanda.positions.error", 1.0, tags={"msg": str(exc)[:120]})
        except Exception:
            pass
    return 0.0, 0.0
