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
              units REAL,
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
    units: float = 100.0,
    micro_regime: str = "",
    macro_regime: str = "",
) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO trades (pocket, strategy_tag, strategy, units, close_time, close_reason, pl_pips, realized_pl, micro_regime, macro_regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pocket,
                strategy_tag,
                strategy_tag,
                float(units),
                close_time,
                close_reason,
                float(pl_pips),
                float(pl_pips) * 100.0,  # dummy
                micro_regime,
                macro_regime,
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


def test_perf_guard_failfast_matches_hashed_strategy_tag_suffix(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(12):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="HashBleed-l0abc1234",
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

    dec = perf_guard.is_allowed("HashBleed", "scalp")
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
    assert "sl_loss_rate=" in dec.reason


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
    assert "margin_closeout_n=" in dec.reason


def test_perf_guard_failfast_soft_in_reduce_mode(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(8):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SoftFail",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=0.5,
            close_time=now,
        )
    for _ in range(4):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SoftFail",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-2.0,
            close_time=now,
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "reduce",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "30",
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "12",
            "PERF_GUARD_FAILFAST_PF": "0.75",
            "PERF_GUARD_FAILFAST_WIN": "0.40",
            "PERF_GUARD_FAILFAST_HARD_PF": "0.30",
            "PERF_GUARD_FAILFAST_HARD_REQUIRE_BOTH": "1",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
        },
    )

    dec = perf_guard.is_allowed("SoftFail", "scalp")
    assert dec.allowed is True
    assert "failfast_soft:" in dec.reason


def test_perf_guard_margin_closeout_soft_in_reduce_mode(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(99):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SoftCloseout",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=1.0,
            close_time=now,
        )
    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="SoftCloseout",
        close_reason="MARKET_ORDER_MARGIN_CLOSEOUT",
        pl_pips=-5.0,
        close_time=now,
    )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "reduce",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "10",
            "PERF_GUARD_PF_MIN": "0.5",
            "PERF_GUARD_WIN_MIN": "0.10",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES": "24",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE": "0.03",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT": "1",
        },
    )

    dec = perf_guard.is_allowed("SoftCloseout", "scalp")
    assert dec.allowed is True
    assert "margin_closeout_soft_n=" in dec.reason


def test_perf_guard_margin_closeout_warmup_soft_in_reduce_mode(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(4):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SoftCloseoutWarmup",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=0.8,
            close_time=now,
        )
    for _ in range(2):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SoftCloseoutWarmup",
            close_reason="MARKET_ORDER_MARGIN_CLOSEOUT",
            pl_pips=-4.0,
            close_time=now,
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "reduce",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "10",
            "PERF_GUARD_PF_MIN": "0.5",
            "PERF_GUARD_WIN_MIN": "0.10",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES": "24",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE": "0.03",
            "PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT": "1",
        },
    )

    dec = perf_guard.is_allowed("SoftCloseoutWarmup", "scalp")
    assert dec.allowed is True
    assert "margin_closeout_soft_warmup_n=" in dec.reason


def test_perf_guard_regime_slice_cannot_bypass_global_hard_block(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # Good performance in live regime slice.
    for _ in range(25):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="RegimeLeak",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=1.0,
            close_time=now,
            micro_regime="trend",
        )
    # One forced liquidation outside the current regime must still hard-block.
    _insert_trade(
        db_path,
        pocket="scalp",
        strategy_tag="RegimeLeak",
        close_reason="MARKET_ORDER_MARGIN_CLOSEOUT",
        pl_pips=-10.0,
        close_time=now,
        micro_regime="range",
    )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "10",
            "PERF_GUARD_PF_MIN": "0.5",
            "PERF_GUARD_WIN_MIN": "0.5",
            "PERF_GUARD_REGIME_FILTER": "1",
            "PERF_GUARD_REGIME_MIN_TRADES": "20",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
        },
    )
    monkeypatch.setattr(perf_guard, "current_regime", lambda tf, event_mode=False: "trend")

    dec = perf_guard.is_allowed("RegimeLeak", "scalp")
    assert dec.allowed is False
    assert "margin_closeout_n=" in dec.reason


def test_perf_guard_prefix_does_not_fallback_to_global(monkeypatch, tmp_path: Path) -> None:
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
            "PERF_GUARD_ENABLED": "0",
            "M1SCALP_PERF_GUARD_ENABLED": "1",
            "M1SCALP_PERF_GUARD_MODE": "block",
            "M1SCALP_PERF_GUARD_LOOKBACK_DAYS": "3",
            "M1SCALP_PERF_GUARD_MIN_TRADES": "30",
            "M1SCALP_PERF_GUARD_PF_MIN": "1.0",
            "M1SCALP_PERF_GUARD_WIN_MIN": "0.50",
            "M1SCALP_PERF_GUARD_REGIME_FILTER": "0",
            "M1SCALP_PERF_GUARD_RELAX_TAGS": "",
            "M1SCALP_PERF_GUARD_FAILFAST_MIN_TRADES": "12",
            "M1SCALP_PERF_GUARD_FAILFAST_PF": "0.75",
            "M1SCALP_PERF_GUARD_FAILFAST_WIN": "0.40",
        },
    )

    dec = perf_guard.is_allowed("BadStrat", "scalp", env_prefix="M1SCALP")
    assert dec.allowed is False
    assert "failfast:" in dec.reason


def test_perf_guard_setup_blocks_bad_direction_only(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    current_hour = now_dt.hour
    for _ in range(8):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SideSplit",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )
    for _ in range(8):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="SideSplit",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=1.1,
            close_time=now,
            units=-100.0,
            micro_regime="trend",
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "40",
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_SETUP_ENABLED": "1",
            "PERF_GUARD_SETUP_USE_HOUR": "1",
            "PERF_GUARD_SETUP_USE_DIRECTION": "1",
            "PERF_GUARD_SETUP_USE_REGIME": "0",
            "PERF_GUARD_SETUP_MIN_TRADES": "8",
            "PERF_GUARD_SETUP_PF_MIN": "1.0",
            "PERF_GUARD_SETUP_WIN_MIN": "0.50",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN": "0.0",
        },
    )

    dec_buy = perf_guard.is_allowed("SideSplit", "scalp", hour=current_hour, side="buy")
    assert dec_buy.allowed is False
    assert dec_buy.reason.startswith("setup_pf=")

    dec_sell = perf_guard.is_allowed("SideSplit", "scalp", hour=current_hour, side="sell")
    assert dec_sell.allowed is True
    assert dec_sell.reason.startswith("warmup_n=") or dec_sell.reason.startswith("pf=")


def test_perf_guard_setup_strategy_override_relaxes_threshold(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    current_hour = now_dt.hour
    # weak setup by default: 3 wins / 7 losses
    for _ in range(7):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="Tag-A",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )
    for _ in range(3):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="Tag-A",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=0.6,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "50",
            "PERF_GUARD_PF_MIN": "1.0",
            "PERF_GUARD_WIN_MIN": "0.50",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_SETUP_ENABLED": "1",
            "PERF_GUARD_SETUP_USE_HOUR": "1",
            "PERF_GUARD_SETUP_USE_DIRECTION": "1",
            "PERF_GUARD_SETUP_USE_REGIME": "0",
            "PERF_GUARD_SETUP_MIN_TRADES": "10",
            "PERF_GUARD_SETUP_PF_MIN": "1.00",
            "PERF_GUARD_SETUP_WIN_MIN": "0.50",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN": "0.00",
            # strategy-specific override for Tag-A (suffix TAG_A)
            "PERF_GUARD_SETUP_PF_MIN_STRATEGY_TAG_A": "0.20",
            "PERF_GUARD_SETUP_WIN_MIN_STRATEGY_TAG_A": "0.25",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN_STRATEGY_TAG_A": "-0.60",
        },
    )

    dec = perf_guard.is_allowed("Tag-A", "scalp", hour=current_hour, side="buy")
    assert dec.allowed is True


def test_perf_guard_setup_strategy_override_global_fallback_with_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    current_hour = now_dt.hour
    for _ in range(7):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="Tag-A",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )
    for _ in range(3):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="Tag-A",
            close_reason="TAKE_PROFIT_ORDER",
            pl_pips=0.6,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "0",
            "M1SCALP_PERF_GUARD_ENABLED": "1",
            "M1SCALP_PERF_GUARD_MODE": "block",
            "M1SCALP_PERF_GUARD_LOOKBACK_DAYS": "3",
            "M1SCALP_PERF_GUARD_MIN_TRADES": "50",
            "M1SCALP_PERF_GUARD_PF_MIN": "1.0",
            "M1SCALP_PERF_GUARD_WIN_MIN": "0.50",
            "M1SCALP_PERF_GUARD_REGIME_FILTER": "0",
            "M1SCALP_PERF_GUARD_RELAX_TAGS": "",
            "M1SCALP_PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "M1SCALP_PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "M1SCALP_PERF_GUARD_SETUP_ENABLED": "1",
            "M1SCALP_PERF_GUARD_SETUP_USE_HOUR": "1",
            "M1SCALP_PERF_GUARD_SETUP_USE_DIRECTION": "1",
            "M1SCALP_PERF_GUARD_SETUP_USE_REGIME": "0",
            "M1SCALP_PERF_GUARD_SETUP_MIN_TRADES": "10",
            "M1SCALP_PERF_GUARD_SETUP_PF_MIN": "1.00",
            "M1SCALP_PERF_GUARD_SETUP_WIN_MIN": "0.50",
            "M1SCALP_PERF_GUARD_SETUP_AVG_PIPS_MIN": "0.00",
            # global strategy override should still apply as fallback
            "PERF_GUARD_SETUP_PF_MIN_STRATEGY_TAG_A": "0.20",
            "PERF_GUARD_SETUP_WIN_MIN_STRATEGY_TAG_A": "0.25",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN_STRATEGY_TAG_A": "-0.60",
        },
    )

    dec = perf_guard.is_allowed(
        "Tag-A",
        "scalp",
        hour=current_hour,
        side="buy",
        env_prefix="M1SCALP",
    )
    assert dec.allowed is True


def test_perf_guard_setup_strategy_min_trades_override(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    current_hour = now_dt.hour
    for _ in range(6):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="QuickBlock",
            close_reason="STOP_LOSS_ORDER",
            pl_pips=-1.0,
            close_time=now,
            units=100.0,
            micro_regime="trend",
        )

    perf_guard = _reload_perf_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "PERF_GUARD_ENABLED": "1",
            "PERF_GUARD_MODE": "block",
            "PERF_GUARD_LOOKBACK_DAYS": "3",
            "PERF_GUARD_MIN_TRADES": "50",
            "PERF_GUARD_PF_MIN": "0.1",
            "PERF_GUARD_WIN_MIN": "0.0",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_SETUP_ENABLED": "1",
            "PERF_GUARD_SETUP_USE_HOUR": "1",
            "PERF_GUARD_SETUP_USE_DIRECTION": "1",
            "PERF_GUARD_SETUP_USE_REGIME": "0",
            "PERF_GUARD_SETUP_MIN_TRADES": "10",
            "PERF_GUARD_SETUP_PF_MIN": "0.1",
            "PERF_GUARD_SETUP_WIN_MIN": "0.0",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN": "-10.0",
            "PERF_GUARD_SETUP_MIN_TRADES_STRATEGY_QUICKBLOCK": "6",
            "PERF_GUARD_SETUP_PF_MIN_STRATEGY_QUICKBLOCK": "1.1",
            "PERF_GUARD_SETUP_WIN_MIN_STRATEGY_QUICKBLOCK": "0.55",
            "PERF_GUARD_SETUP_AVG_PIPS_MIN_STRATEGY_QUICKBLOCK": "0.0",
        },
    )

    dec = perf_guard.is_allowed("QuickBlock", "scalp", hour=current_hour, side="buy")
    assert dec.allowed is False
    assert dec.reason.startswith("setup_pf=")


def test_perf_guard_strategy_min_trades_override_for_base_guard(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    _init_trades_db(db_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(6):
        _insert_trade(
            db_path,
            pocket="scalp",
            strategy_tag="QuickBase",
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
            "PERF_GUARD_MIN_TRADES": "20",
            "PERF_GUARD_PF_MIN": "0.1",
            "PERF_GUARD_WIN_MIN": "0.0",
            "PERF_GUARD_REGIME_FILTER": "0",
            "PERF_GUARD_RELAX_TAGS": "",
            "PERF_GUARD_FAILFAST_MIN_TRADES": "0",
            "PERF_GUARD_SL_LOSS_RATE_MAX_SCALP": "0",
            "PERF_GUARD_SETUP_ENABLED": "0",
            "PERF_GUARD_MIN_TRADES_STRATEGY_QUICKBASE": "6",
            "PERF_GUARD_PF_MIN_STRATEGY_QUICKBASE": "1.1",
            "PERF_GUARD_WIN_MIN_STRATEGY_QUICKBASE": "0.55",
        },
    )

    dec = perf_guard.is_allowed("QuickBase", "scalp")
    assert dec.allowed is False
    assert dec.reason.startswith("pf=")
