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


def _sample_signal_for_prob(side: str = "long", mode: str = "momentum") -> scalp_worker.TickSignal:
    return scalp_worker.TickSignal(
        side=side,
        mode=mode,
        mode_score=1.0,
        momentum_score=0.8,
        revert_score=0.0,
        confidence=92,
        momentum_pips=0.2 if side == "long" else -0.2,
        trigger_pips=0.1,
        imbalance=0.65,
        tick_rate=5.0,
        span_sec=1.2,
        tick_age_ms=90.0,
        spread_pips=0.8,
        bid=154.5,
        ask=154.51,
        mid=154.505,
        range_pips=1.1,
        instant_range_pips=0.7,
        signal_window_sec=1.5,
    )


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


def _set_probability_align_config(monkeypatch) -> None:
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_DIRECTION_WEIGHT",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_HORIZON_WEIGHT",
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_M1_WEIGHT",
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_BOOST_MAX",
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_PENALTY_MAX",
        0.90,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX",
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_REVERT_PENALTY_MULT",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN",
        0.70,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_FLOOR",
        0.46,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_MIN",
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_MAX",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_UNITS_FOLLOW_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT",
        0.10,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT",
        1.0,
        raising=False,
    )


def test_entry_probability_alignment_floor_is_blocked_when_counter_dominates(monkeypatch) -> None:
    _set_probability_align_config(monkeypatch)

    direction_bias = scalp_worker.DirectionBias(
        side="short",
        score=-0.8,
        momentum_pips=-0.4,
        flow=-0.6,
        range_pips=1.2,
        vol_norm=0.6,
        tick_rate=6.0,
        span_sec=1.2,
    )

    adjusted, units_mult, meta = scalp_worker._adjust_entry_probability_alignment(
        signal=_sample_signal_for_prob("long"),
        raw_probability=0.90,
        direction_bias=direction_bias,
        horizon=None,
        m1_score=None,
    )

    assert adjusted < 0.46
    assert units_mult < 1.0
    assert meta["floor_applied"] is False
    assert meta["floor_block_reason"] == "support_lt_counter"


def test_entry_probability_alignment_floor_applies_when_support_not_weaker(monkeypatch) -> None:
    _set_probability_align_config(monkeypatch)
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_DIRECTION_WEIGHT",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_HORIZON_WEIGHT",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX",
        0.90,
        raising=False,
    )

    direction_bias = scalp_worker.DirectionBias(
        side="long",
        score=0.6,
        momentum_pips=0.4,
        flow=0.5,
        range_pips=1.1,
        vol_norm=0.6,
        tick_rate=6.0,
        span_sec=1.2,
    )
    horizon = scalp_worker.HorizonBias(
        long_side="neutral",
        long_score=0.0,
        mid_side="neutral",
        mid_score=0.0,
        short_side="short",
        short_score=0.6,
        micro_side="short",
        micro_score=0.6,
        composite_side="short",
        composite_score=-0.6,
        agreement=2,
    )

    adjusted, units_mult, meta = scalp_worker._adjust_entry_probability_alignment(
        signal=_sample_signal_for_prob("long"),
        raw_probability=0.90,
        direction_bias=direction_bias,
        horizon=horizon,
        m1_score=None,
    )

    assert adjusted == pytest.approx(0.46, abs=1e-9)
    assert units_mult == pytest.approx(0.511111111, rel=1e-6)
    assert meta["floor_applied"] is True
    assert meta["floor_block_reason"] == ""
