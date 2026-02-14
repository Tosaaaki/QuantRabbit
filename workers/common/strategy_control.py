from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple


LOG = __import__("logging").getLogger(__name__)


_DB_PATH = Path(os.getenv("STRATEGY_CONTROL_DB_PATH", "logs/strategy_control.db"))
_GLOBAL_SLUG = "__global__"

_DB: Optional[sqlite3.Connection] = None
_LOCK = threading.Lock()


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def normalize_strategy_slug(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    for suffix in ("_live", "-live"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_connection() -> sqlite3.Connection:
    global _DB
    if _DB is not None:
        return _DB

    with _LOCK:
        if _DB is not None:
            return _DB

        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_control_flags (
                strategy_slug TEXT PRIMARY KEY,
                entry_enabled INTEGER NOT NULL CHECK(entry_enabled IN (0, 1)),
                exit_enabled INTEGER NOT NULL CHECK(exit_enabled IN (0, 1)),
                global_lock INTEGER NOT NULL DEFAULT 0 CHECK(global_lock IN (0, 1)),
                updated_at TEXT NOT NULL,
                note TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO strategy_control_flags
            (strategy_slug, entry_enabled, exit_enabled, global_lock, updated_at, note)
            VALUES (?, 1, 1, 0, ?, 'seeded')
            """,
            (_GLOBAL_SLUG, _now()),
        )
        conn.commit()
        _DB = conn
        return _DB


def _upsert(
    strategy_slug: str,
    *,
    entry: Optional[bool] = None,
    exit: Optional[bool] = None,
    lock: Optional[bool] = None,
    note: Optional[str] = None,
) -> None:
    slug = normalize_strategy_slug(strategy_slug)
    if not slug:
        return

    row = get_flags(slug)
    if row is None:
        current_entry, current_exit, current_lock = True, True, False
    else:
        current_entry, current_exit, current_lock = row

    next_entry = current_entry if entry is None else bool(entry)
    next_exit = current_exit if exit is None else bool(exit)
    next_lock = current_lock if lock is None else bool(lock)

    conn = _ensure_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO strategy_control_flags
            (strategy_slug, entry_enabled, exit_enabled, global_lock, updated_at, note)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_slug) DO UPDATE SET
                entry_enabled=excluded.entry_enabled,
                exit_enabled=excluded.exit_enabled,
                global_lock=excluded.global_lock,
                updated_at=excluded.updated_at,
                note=excluded.note
            """,
            (
                slug,
                1 if next_entry else 0,
                1 if next_exit else 0,
                1 if next_lock else 0,
                _now(),
                (note or "")[:255],
            ),
        )
        conn.commit()


def set_strategy_flags(
    strategy_slug: str,
    *,
    entry: Optional[bool] = None,
    exit: Optional[bool] = None,
    lock: Optional[bool] = None,
    note: Optional[str] = None,
) -> None:
    _upsert(strategy_slug, entry=entry, exit=exit, lock=lock, note=note)


def get_flags(strategy_slug: str) -> Optional[tuple[bool, bool, bool]]:
    slug = normalize_strategy_slug(strategy_slug)
    if not slug:
        return None

    conn = _ensure_connection()
    row = conn.execute(
        "SELECT entry_enabled, exit_enabled, global_lock FROM strategy_control_flags WHERE strategy_slug=?",
        (slug,),
    ).fetchone()
    if row is None:
        return None
    try:
        return bool(row[0]), bool(row[1]), bool(row[2])
    except Exception:
        return None


def get_global_flags() -> tuple[bool, bool, bool]:
    conn = _ensure_connection()
    row = conn.execute(
        "SELECT entry_enabled, exit_enabled, global_lock FROM strategy_control_flags WHERE strategy_slug=?",
        (_GLOBAL_SLUG,),
    ).fetchone()
    if row is None:
        with _LOCK:
            conn.execute(
                "INSERT OR REPLACE INTO strategy_control_flags"
                " (strategy_slug, entry_enabled, exit_enabled, global_lock, updated_at, note)"
                " VALUES (?, 1, 1, 0, ?, 'seeded')",
                (_GLOBAL_SLUG, _now()),
            )
            conn.commit()
        return True, True, False
    return bool(row[0]), bool(row[1]), bool(row[2])


def can_enter(strategy_slug: str, *, default: bool = True) -> bool:
    slug = normalize_strategy_slug(strategy_slug)
    if not slug:
        return default

    g_entry, _, g_lock = get_global_flags()
    if g_lock or not g_entry:
        return False

    row = get_flags(slug)
    if row is None:
        return default
    return bool(row[0])


def can_exit(strategy_slug: str, *, default: bool = True) -> bool:
    slug = normalize_strategy_slug(strategy_slug)
    if not slug:
        return default

    _, g_exit, g_lock = get_global_flags()
    if g_lock or not g_exit:
        return False

    row = get_flags(slug)
    if row is None:
        return default
    return bool(row[1])


def set_global_flags(
    *,
    entry: Optional[bool] = None,
    exit: Optional[bool] = None,
    lock: Optional[bool] = None,
    note: Optional[str] = None,
) -> None:
    _upsert(_GLOBAL_SLUG, entry=entry, exit=exit, lock=lock, note=note)


def ensure_strategy(
    strategy_slug: str,
    *,
    entry: bool = True,
    exit: bool = True,
    lock: bool = False,
    note: Optional[str] = None,
) -> None:
    _upsert(
        strategy_slug,
        entry=entry,
        exit=exit,
        lock=lock,
        note=note,
    )


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for part in str(value).replace("\n", ",").split(","):
        slug = normalize_strategy_slug(part)
        if slug:
            out.append(slug)
    return out


def sync_env_overrides() -> None:
    """Apply control overrides from environment variables.

    Supported keys:
    - STRATEGY_CONTROL_GLOBAL_ENTRY_ENABLED / STRATEGY_CONTROL_GLOBAL_EXIT_ENABLED
    - STRATEGY_CONTROL_GLOBAL_LOCK
    - STRATEGY_CONTROL_SEED_STRATEGIES (comma-separated list)
    - STRATEGY_CONTROL_ENTRY_<STRATEGY> (0/1)
    - STRATEGY_CONTROL_EXIT_<STRATEGY> (0/1)
    - STRATEGY_CONTROL_LOCK_<STRATEGY> (0/1)
    """

    global_entry = _env_bool("STRATEGY_CONTROL_GLOBAL_ENTRY_ENABLED")
    global_exit = _env_bool("STRATEGY_CONTROL_GLOBAL_EXIT_ENABLED")
    global_lock = _env_bool("STRATEGY_CONTROL_GLOBAL_LOCK")
    if global_entry is not None or global_exit is not None or global_lock is not None:
        set_global_flags(
            entry=global_entry,
            exit=global_exit,
            lock=global_lock,
            note="env_sync",
        )

    for strategy in _split_csv(os.getenv("STRATEGY_CONTROL_SEED_STRATEGIES")):
        ensure_strategy(strategy)

    for key, value in os.environ.items():
        if not key.startswith("STRATEGY_CONTROL_ENTRY_"):
            continue
        strategy = key[len("STRATEGY_CONTROL_ENTRY_"):]
        strategy = normalize_strategy_slug(strategy)
        if not strategy:
            continue
        flag = _env_bool(value)
        if flag is not None:
            set_strategy_flags(strategy, entry=flag)

    for key, value in os.environ.items():
        if not key.startswith("STRATEGY_CONTROL_EXIT_"):
            continue
        strategy = key[len("STRATEGY_CONTROL_EXIT_"):]
        strategy = normalize_strategy_slug(strategy)
        if not strategy:
            continue
        flag = _env_bool(value)
        if flag is not None:
            set_strategy_flags(strategy, exit=flag)

    for key, value in os.environ.items():
        if not key.startswith("STRATEGY_CONTROL_LOCK_"):
            continue
        strategy = key[len("STRATEGY_CONTROL_LOCK_"):]
        strategy = normalize_strategy_slug(strategy)
        if not strategy:
            continue
        flag = _env_bool(value)
        if flag is not None:
            set_strategy_flags(strategy, lock=flag)


def list_enabled_strategies() -> list[tuple[str, bool, bool, bool]]:
    conn = _ensure_connection()
    rows = conn.execute(
        "SELECT strategy_slug, entry_enabled, exit_enabled, global_lock FROM strategy_control_flags WHERE strategy_slug != ? ORDER BY strategy_slug",
        (_GLOBAL_SLUG,),
    ).fetchall()
    out: list[tuple[str, bool, bool, bool]] = []
    for row in rows:
        try:
            out.append((row[0], bool(row[1]), bool(row[2]), bool(row[3])))
        except Exception:
            continue
    return out
