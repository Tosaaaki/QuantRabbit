from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

_FALSEY = {"", "0", "false", "no", "off"}

# ---------------------------------------------------------------------------
# Hot-reload .env file loader: automatically re-reads the file when its
# mtime changes so that autotune / config writes are picked up without a
# systemd restart.
# ---------------------------------------------------------------------------
_env_file_cache: dict[str, dict] = {}
_env_file_mtime: dict[str, float] = {}
_env_file_lock = threading.Lock()


def load_env_file_hot(path: str | Path | None = None) -> dict:
    """
    Parse a KEY=VALUE .env file into a dict, automatically reloading
    when the file's modification timestamp changes.  Thread-safe.

    If *path* is None, falls back to QUANTRABBIT_ENV_FILE or the
    default production path.
    """
    if path is None:
        path = os.getenv(
            "QUANTRABBIT_ENV_FILE",
            "/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env",
        )
    key = str(path)
    p = Path(key)
    if not p.exists():
        return _env_file_cache.get(key, {})
    try:
        current_mtime = p.stat().st_mtime
    except OSError:
        return _env_file_cache.get(key, {})
    cached_mtime = _env_file_mtime.get(key, 0.0)
    if current_mtime != cached_mtime or key not in _env_file_cache:
        with _env_file_lock:
            try:
                current_mtime = p.stat().st_mtime
            except OSError:
                return _env_file_cache.get(key, {})
            if (
                current_mtime != _env_file_mtime.get(key, 0.0)
                or key not in _env_file_cache
            ):
                data: dict = {}
                try:
                    content = p.read_text(encoding="utf-8")
                except OSError:
                    content = ""
                for raw in content.splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, val = line.split("=", 1)
                    data[k.strip()] = val.strip().strip('"').strip("'")
                old_mtime = _env_file_mtime.get(key, 0.0)
                _env_file_cache[key] = data
                _env_file_mtime[key] = current_mtime
                if old_mtime != 0.0 and old_mtime != current_mtime:
                    logging.info(
                        "[env_utils] hot-reloaded %s (mtime %.0f -> %.0f)",
                        key,
                        old_mtime,
                        current_mtime,
                    )
    return _env_file_cache.get(key, {})


def env_get(
    name: str,
    default: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    allow_global_fallback: bool = True,
) -> Optional[str]:
    """
    Environment lookup with optional per-worker prefix.

    Precedence (when prefix is provided):
    1) {PREFIX}_UNIT_{NAME}  (per-systemd-unit / single-strategy override)
    2) {PREFIX}_{NAME}       (per-worker override)
    3) {NAME}                (global fallback, optional)
    """
    if prefix:
        p = str(prefix).strip()
        if p:
            unit_key = f"{p}_UNIT_{name}"
            if unit_key in os.environ:
                return os.environ[unit_key]
            pref_key = f"{p}_{name}"
            if pref_key in os.environ:
                return os.environ[pref_key]
            if not allow_global_fallback:
                return default
    return os.getenv(name, default)


def env_bool(
    name: str,
    default: bool,
    *,
    prefix: Optional[str] = None,
    allow_global_fallback: bool = True,
) -> bool:
    raw = env_get(
        name, None, prefix=prefix, allow_global_fallback=allow_global_fallback
    )
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSEY


def env_float(
    name: str,
    default: float,
    *,
    prefix: Optional[str] = None,
    allow_global_fallback: bool = True,
) -> float:
    raw = env_get(
        name, None, prefix=prefix, allow_global_fallback=allow_global_fallback
    )
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def env_int(
    name: str,
    default: int,
    *,
    prefix: Optional[str] = None,
    allow_global_fallback: bool = True,
) -> int:
    raw = env_get(
        name, None, prefix=prefix, allow_global_fallback=allow_global_fallback
    )
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)
