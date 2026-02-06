import importlib
import os
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
              pocket TEXT,
              strategy_tag TEXT,
              strategy TEXT,
              close_time TEXT,
              close_reason TEXT,
              pl_pips REAL,
              realized_pl REAL,
              micro_regime TEXT,
              macro_regime TEXT
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
    close_reason: str,
    pl_pips: float,
    close_time: str,
) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO trades (pocket, strategy_tag, strategy, close_time, close_reason, pl_pips, realized_pl, micro_regime, macro_regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pocket,
                strategy_tag,
                strategy_tag,
                close_time,
                close_reason,
                float(pl_pips),
                float(pl_pips) * 100.0,  # dummy
                "",
                "",
            ),
        )
        con.commit()
    finally:
        con.close()


def _reload_perf_guard(monkeypatch, *, db_path: Path, env: dict[str, str]):
    # Ensure predictable module-level constants on import.
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import workers.common.perf_guard as perf_guard

    perf_guard = importlib.reload(perf_guard)
    perf_guard._DB = Path(db_path)
    perf_guard._cache.clear()
    perf_guard._pocket_cache.clear()
    perf_guard._scale_cache.clear()
    return perf_guard


def test_perf_guard_failfast_blocks_before_warmup(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(12):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="BadStrat",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "30",  # warmup would normally allow
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "12",
            "PERF_GUARD_FAILFAST_PF": "0.75",
            "PERF_GUARD_FAILFAST_WIN": "0.40",
        },
    )

    dec = perf_guard.is_allowed("BadStrat", "scalp")
    assert dec.allowed is False
    assert "failfast:" in dec.reason


def test_perf_guard_sl_loss_rate_blocks_before_warmup(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # 9 stop-loss losses, 3 small wins => PF < 1.0 and SL loss rate 0.75
    for _ in range(9):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SLBleeder",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
        )
    for _ in range(3):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SLBleeder",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=0.5,
            close_time=now,
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "30",  # warmup would normally allow
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            # Disable failfast so this test exercises sl_loss_rate.
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            # Make sl_loss guard deterministic.
            "PERF_GUARD_SL_LOSS_RATE_MIN_TRADES": "12",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0.65",
        },
    )

    dec = perf_guard.is_allowed("SLBleeder", "scalp")
    assert dec.allowed is False
    assert dec.reason.startswith("sl_loss_rate=")


def test_perf_guard_margin_closeout_blocks_immediately(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="AnyStrat",
        close_reason="MARKET_ORDER_MARGIN_CLOSEOUT",
        pl_pips=-100.0,
        close_time=now,
    )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "30",
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
        },
    )

    dec = perf_guard.is_allowed("AnyStrat", "scalp")
    assert dec.allowed is False
    assert dec.reason.startswith("margin_closeout_n=")

