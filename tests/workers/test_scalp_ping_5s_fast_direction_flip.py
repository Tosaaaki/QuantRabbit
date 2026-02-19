from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s import worker as scalp_worker


def _set_fast_flip_config(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "FAST_DIRECTION_FLIP_ENABLED", True, raising=False)
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN",
        0.4,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN",
        0.2,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS",
        0.05,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_CONFIDENCE_ADD",
        3,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_COOLDOWN_SEC",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE",
        0.65,
        raising=False,
    )


def _sample_signal(side: str) -> scalp_worker.TickSignal:
    return scalp_worker.TickSignal(
        side=side,
        mode="momentum",
        mode_score=1.0,
        momentum_score=1.0,
        revert_score=0.0,
        confidence=80,
        momentum_pips=-0.2 if side == "short" else 0.2,
        trigger_pips=0.1,
        imbalance=0.7,
        tick_rate=6.0,
        span_sec=1.5,
        tick_age_ms=120.0,
        spread_pips=0.8,
        bid=154.7,
        ask=154.71,
        mid=154.705,
        range_pips=1.2,
        instant_range_pips=0.8,
        signal_window_sec=1.5,
    )


def test_fast_direction_flip_applies_when_bias_and_horizon_align(monkeypatch) -> None:
    _set_fast_flip_config(monkeypatch)
    monkeypatch.setattr(scalp_worker, "_LAST_FAST_FLIP_MONO", 0.0)

    signal = _sample_signal("short")
    bias = scalp_worker.DirectionBias(
        side="long",
        score=0.72,
        momentum_pips=0.18,
        flow=0.5,
        range_pips=1.3,
        vol_norm=0.7,
        tick_rate=7.0,
        span_sec=1.2,
    )
    horizon = scalp_worker.HorizonBias(
        long_side="long",
        long_score=0.5,
        mid_side="long",
        mid_score=0.4,
        short_side="neutral",
        short_score=0.05,
        micro_side="long",
        micro_score=0.3,
        composite_side="long",
        composite_score=0.42,
        agreement=3,
    )
    regime = scalp_worker.MtfRegime(
        side="neutral",
        mode="balanced",
        trend_score=0.1,
        heat_score=0.4,
        adx_m1=15.0,
        adx_m5=16.0,
        atr_m1=0.6,
        atr_m5=1.0,
    )

    flipped, reason = scalp_worker._maybe_fast_direction_flip(
        signal,
        direction_bias=bias,
        horizon=horizon,
        regime=regime,
        now_mono=10.0,
    )

    assert flipped is not None
    assert flipped.side == "long"
    assert flipped.mode.endswith("_fflip")
    assert "short->long" in reason


def test_fast_direction_flip_skips_on_horizon_mismatch(monkeypatch) -> None:
    _set_fast_flip_config(monkeypatch)
    monkeypatch.setattr(scalp_worker, "_LAST_FAST_FLIP_MONO", 0.0)

    signal = _sample_signal("short")
    bias = scalp_worker.DirectionBias(
        side="long",
        score=0.72,
        momentum_pips=0.18,
        flow=0.5,
        range_pips=1.3,
        vol_norm=0.7,
        tick_rate=7.0,
        span_sec=1.2,
    )
    horizon = scalp_worker.HorizonBias(
        long_side="neutral",
        long_score=0.05,
        mid_side="short",
        mid_score=-0.30,
        short_side="short",
        short_score=-0.50,
        micro_side="short",
        micro_score=-0.25,
        composite_side="short",
        composite_score=-0.34,
        agreement=3,
    )

    flipped, reason = scalp_worker._maybe_fast_direction_flip(
        signal,
        direction_bias=bias,
        horizon=horizon,
        regime=None,
        now_mono=10.0,
    )

    assert flipped is None
    assert reason == "horizon_mismatch"


def test_fast_direction_flip_respects_cooldown(monkeypatch) -> None:
    _set_fast_flip_config(monkeypatch)
    monkeypatch.setattr(scalp_worker, "_LAST_FAST_FLIP_MONO", 9.8)

    signal = _sample_signal("short")
    bias = scalp_worker.DirectionBias(
        side="long",
        score=0.8,
        momentum_pips=0.2,
        flow=0.6,
        range_pips=1.4,
        vol_norm=0.8,
        tick_rate=8.0,
        span_sec=1.1,
    )
    horizon = scalp_worker.HorizonBias(
        long_side="long",
        long_score=0.6,
        mid_side="long",
        mid_score=0.5,
        short_side="neutral",
        short_score=0.0,
        micro_side="long",
        micro_score=0.4,
        composite_side="long",
        composite_score=0.5,
        agreement=3,
    )

    flipped, reason = scalp_worker._maybe_fast_direction_flip(
        signal,
        direction_bias=bias,
        horizon=horizon,
        regime=None,
        now_mono=10.0,
    )

    assert flipped is None
    assert reason == "cooldown"


def test_fast_direction_flip_blocks_on_strong_counter_regime(monkeypatch) -> None:
    _set_fast_flip_config(monkeypatch)
    monkeypatch.setattr(scalp_worker, "_LAST_FAST_FLIP_MONO", 0.0)

    signal = _sample_signal("short")
    bias = scalp_worker.DirectionBias(
        side="long",
        score=0.8,
        momentum_pips=0.2,
        flow=0.6,
        range_pips=1.4,
        vol_norm=0.8,
        tick_rate=8.0,
        span_sec=1.1,
    )
    horizon = scalp_worker.HorizonBias(
        long_side="long",
        long_score=0.6,
        mid_side="long",
        mid_score=0.5,
        short_side="neutral",
        short_score=0.0,
        micro_side="long",
        micro_score=0.4,
        composite_side="long",
        composite_score=0.5,
        agreement=3,
    )
    regime = scalp_worker.MtfRegime(
        side="short",
        mode="continuation",
        trend_score=-0.8,
        heat_score=0.8,
        adx_m1=24.0,
        adx_m5=27.0,
        atr_m1=0.8,
        atr_m5=1.2,
    )

    flipped, reason = scalp_worker._maybe_fast_direction_flip(
        signal,
        direction_bias=bias,
        horizon=horizon,
        regime=regime,
        now_mono=10.0,
    )

    assert flipped is None
    assert reason == "regime_counter"
