from __future__ import annotations

import os
import pathlib
import sqlite3
import sys

import pytest

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s import worker as scalp_worker


def _set_band_alloc_config(monkeypatch) -> None:
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD",
        0.70,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD",
        0.90,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_MIN_TRADES_PER_BAND",
        10,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_GAP_PIPS_REF",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_GAP_WIN_RATE_REF",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_GAP_SL_RATE_REF",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SAMPLE_STRONG_TRADES",
        20,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX",
        0.45,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT",
        0.55,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT",
        1.35,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED",
        False,
        raising=False,
    )


def test_probability_band_units_multiplier_reduces_high_prob_size(monkeypatch) -> None:
    _set_band_alloc_config(monkeypatch)

    metrics = scalp_worker.EntryProbabilityBandMetrics(
        side="short",
        sample=120,
        high_sample=50,
        high_mean_pips=-1.20,
        high_win_rate=0.38,
        high_sl_rate=0.63,
        low_sample=40,
        low_mean_pips=1.08,
        low_win_rate=0.72,
        low_sl_rate=0.29,
    )
    monkeypatch.setattr(
        scalp_worker,
        "_load_entry_probability_band_metrics",
        lambda **_: metrics,
        raising=False,
    )

    units_mult, meta = scalp_worker._entry_probability_band_units_multiplier(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="short",
        entry_probability=0.93,
        now_mono=10.0,
    )

    assert meta["reason"] == "ok"
    assert meta["bucket"] == "high"
    assert units_mult == pytest.approx(0.55, abs=1e-9)


def test_probability_band_units_multiplier_boosts_low_prob_size(monkeypatch) -> None:
    _set_band_alloc_config(monkeypatch)

    metrics = scalp_worker.EntryProbabilityBandMetrics(
        side="long",
        sample=140,
        high_sample=60,
        high_mean_pips=-0.95,
        high_win_rate=0.41,
        high_sl_rate=0.58,
        low_sample=44,
        low_mean_pips=0.88,
        low_win_rate=0.69,
        low_sl_rate=0.25,
    )
    monkeypatch.setattr(
        scalp_worker,
        "_load_entry_probability_band_metrics",
        lambda **_: metrics,
        raising=False,
    )

    units_mult, meta = scalp_worker._entry_probability_band_units_multiplier(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="long",
        entry_probability=0.55,
        now_mono=10.0,
    )

    assert meta["reason"] == "ok"
    assert meta["bucket"] == "low"
    assert units_mult == pytest.approx(1.30, abs=1e-9)


def test_probability_band_units_multiplier_uses_side_metrics_penalty(monkeypatch) -> None:
    _set_band_alloc_config(monkeypatch)
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT",
        0.85,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT",
        1.08,
        raising=False,
    )

    neutral_metrics = scalp_worker.EntryProbabilityBandMetrics(
        side="short",
        sample=80,
        high_sample=40,
        high_mean_pips=-0.20,
        high_win_rate=0.50,
        high_sl_rate=0.40,
        low_sample=32,
        low_mean_pips=-0.20,
        low_win_rate=0.50,
        low_sl_rate=0.40,
    )
    monkeypatch.setattr(
        scalp_worker,
        "_load_entry_probability_band_metrics",
        lambda **_: neutral_metrics,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker,
        "_load_recent_side_close_metrics_for_allocation",
        lambda **_: scalp_worker.SideCloseMetrics(
            long_sl_hits=1,
            short_sl_hits=12,
            long_market_plus=3,
            short_market_plus=2,
            long_trades=8,
            short_trades=20,
            sample=28,
        ),
        raising=False,
    )

    units_mult, meta = scalp_worker._entry_probability_band_units_multiplier(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="short",
        entry_probability=0.85,
        now_mono=10.0,
    )

    assert meta["reason"] == "ok"
    assert meta["side_sl_hits"] == 12
    assert meta["side_market_plus"] == 2
    assert units_mult == pytest.approx(0.85, abs=1e-9)


def test_load_entry_probability_band_metrics_from_trades_db(monkeypatch, tmp_path: pathlib.Path) -> None:
    _set_band_alloc_config(monkeypatch)
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_CACHE_TTL_SEC",
        0.1,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES",
        20,
        raising=False,
    )

    db_path = tmp_path / "trades.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE trades (
          close_time TEXT,
          units INTEGER,
          close_reason TEXT,
          strategy_tag TEXT,
          pocket TEXT,
          entry_thesis TEXT,
          pl_pips REAL
        )
        """
    )
    con.executemany(
        """
        INSERT INTO trades (close_time, units, close_reason, strategy_tag, pocket, entry_thesis, pl_pips)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("2026-02-19T10:00:00+00:00", -1000, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast", '{"entry_probability":0.95}', -1.5),
            ("2026-02-19T09:59:00+00:00", -1000, "MARKET_ORDER_TRADE_CLOSE", "scalp_ping_5s_b_live", "scalp_fast", '{"entry_probability":0.92}', 0.2),
            ("2026-02-19T09:58:00+00:00", -1000, "MARKET_ORDER_TRADE_CLOSE", "scalp_ping_5s_b_live", "scalp_fast", '{"entry_probability":0.55}', 1.0),
            ("2026-02-19T09:57:00+00:00", -1000, "MARKET_ORDER_TRADE_CLOSE", "scalp_ping_5s_b_live", "scalp_fast", '{"entry_probability":0.60}', 0.4),
            ("2026-02-19T09:56:00+00:00", 900, "MARKET_ORDER_TRADE_CLOSE", "scalp_ping_5s_b_live", "scalp_fast", '{"entry_probability":0.95}', 2.0),
        ],
    )
    con.commit()
    con.close()

    monkeypatch.setattr(scalp_worker, "_TRADES_DB", db_path, raising=False)
    monkeypatch.setattr(scalp_worker, "_ENTRY_PROB_BAND_METRICS_CACHE", {}, raising=False)

    metrics = scalp_worker._load_entry_probability_band_metrics(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="short",
        now_mono=10.0,
    )

    assert metrics is not None
    assert metrics.side == "short"
    assert metrics.sample == 4
    assert metrics.high_sample == 2
    assert metrics.low_sample == 2
    assert metrics.high_mean_pips == pytest.approx(-0.65, abs=1e-9)
    assert metrics.low_mean_pips == pytest.approx(0.7, abs=1e-9)
    assert metrics.high_win_rate == pytest.approx(0.5, abs=1e-9)
    assert metrics.low_win_rate == pytest.approx(1.0, abs=1e-9)
    assert metrics.high_sl_rate == pytest.approx(0.5, abs=1e-9)
    assert metrics.low_sl_rate == pytest.approx(0.0, abs=1e-9)
