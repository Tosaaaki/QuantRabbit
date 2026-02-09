from __future__ import annotations

import importlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _init_trades_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              pocket TEXT,
              strategy_tag TEXT,
              strategy TEXT,
              close_time TEXT,
              pl_pips REAL,
              realized_pl REAL
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _insert_trade(
    db_path: Path,
    *,
    pocket: str,
    strategy_tag: str,
    pl_pips: float,
    realized_pl: float,
    close_time: str,
) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO trades (pocket, strategy_tag, strategy, close_time, pl_pips, realized_pl)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pocket, strategy_tag, strategy_tag, close_time, float(pl_pips), float(realized_pl)),
        )
        con.commit()
    finally:
        con.close()


def _reload_profit_guard(monkeypatch, *, db_path: Path, env: dict[str, str]):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import workers.common.profit_guard as profit_guard

    profit_guard = importlib.reload(profit_guard)
    profit_guard._DB = Path(db_path)
    profit_guard._cache.clear()
    profit_guard._CFG_CACHE.clear()
    return profit_guard


def test_profit_guard_reason_uses_cfg_lookback(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="TestStrat",
        pl_pips=5.0,
        realized_pl=500.0,
        close_time=now,
    )
    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="TestStrat",
        pl_pips=-4.0,
        realized_pl=-400.0,
        close_time=now,
    )

    profit_guard = _reload_profit_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PROFIT_GUARD_ENABLED": "1",
            "PROFIT_GUARD_MODE": "block",
            "PROFIT_GUARD_LOOKBACK_MIN": "180",
            "PROFIT_GUARD_POCKETS": "scalp",
            "PROFIT_GUARD_MIN_PEAK_PIPS": "1",
            "PROFIT_GUARD_MAX_GIVEBACK_PIPS": "1",
        },
    )

    dec = profit_guard.is_allowed("scalp", strategy_tag="TestStrat")
    assert dec.allowed is False
    assert "win=180m" in dec.reason


def test_profit_guard_prefix_does_not_fallback_to_global(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="M1Scalp",
        pl_pips=5.0,
        realized_pl=500.0,
        close_time=now,
    )
    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="M1Scalp",
        pl_pips=-4.0,
        realized_pl=-400.0,
        close_time=now,
    )

    profit_guard = _reload_profit_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PROFIT_GUARD_ENABLED": "0",
            "M1SCALP_PROFIT_GUARD_ENABLED": "1",
            "M1SCALP_PROFIT_GUARD_MODE": "block",
            "M1SCALP_PROFIT_GUARD_POCKETS": "scalp",
            "M1SCALP_PROFIT_GUARD_MIN_PEAK_PIPS": "1",
            "M1SCALP_PROFIT_GUARD_MAX_GIVEBACK_PIPS": "1",
            "M1SCALP_PROFIT_GUARD_LOOKBACK_MIN": "180",
        },
    )

    dec = profit_guard.is_allowed("scalp", strategy_tag="M1Scalp", env_prefix="M1SCALP")
    assert dec.allowed is False
    assert "giveback=" in dec.reason
