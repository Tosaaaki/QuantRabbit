from __future__ import annotations

from workers.scalp_wick_reversal_blend.policy import (
    wick_blend_entry_quality,
    wick_blend_exit_adjustments,
)


def test_wick_blend_entry_quality_rejects_neutral_short_without_strong_reset() -> None:
    quality = wick_blend_entry_quality(
        side="short",
        rsi=53.2,
        adx=15.7,
        atr_pips=2.16,
        range_score=0.423,
        wick_ratio=0.46,
        tick_strength=0.24,
        follow_pips=0.84,
        retrace_from_extreme_pips=0.48,
        projection_score=-0.04,
    )

    assert quality["allow"] is False
    assert float(quality["quality"]) < float(quality["threshold"])


def test_wick_blend_entry_quality_allows_stretched_short_with_strong_reset() -> None:
    quality = wick_blend_entry_quality(
        side="short",
        rsi=63.5,
        adx=28.1,
        atr_pips=2.69,
        range_score=0.211,
        wick_ratio=0.71,
        tick_strength=0.46,
        follow_pips=1.15,
        retrace_from_extreme_pips=0.92,
        projection_score=0.24,
    )

    assert quality["allow"] is True
    assert float(quality["quality"]) >= float(quality["threshold"])


def test_wick_blend_entry_quality_penalizes_counter_projection() -> None:
    aligned = wick_blend_entry_quality(
        side="long",
        rsi=28.0,
        adx=24.0,
        atr_pips=2.0,
        range_score=0.34,
        wick_ratio=0.78,
        tick_strength=0.62,
        follow_pips=2.4,
        retrace_from_extreme_pips=3.8,
        projection_score=0.08,
    )
    headwind = wick_blend_entry_quality(
        side="long",
        rsi=28.0,
        adx=24.0,
        atr_pips=2.0,
        range_score=0.34,
        wick_ratio=0.78,
        tick_strength=0.62,
        follow_pips=2.4,
        retrace_from_extreme_pips=3.8,
        projection_score=-0.14,
    )

    assert float(headwind["quality"]) < float(aligned["quality"])
    assert float(headwind["components"]["projection_headwind"]) > 0.0


def test_wick_blend_exit_adjustments_use_trade_quality_and_entry_sl() -> None:
    adjusted = wick_blend_exit_adjustments(
        side="short",
        thesis={
            "sl_pips": 2.0,
            "tp_pips": 3.0,
            "atr_pips": 2.2,
            "wick_blend_quality": 0.62,
        },
        atr_pips=2.2,
        profit_take=1.2,
        trail_start=1.7,
        trail_backoff=0.55,
        lock_buffer=0.25,
        loss_cut_hard_pips=6.0,
        loss_cut_max_hold_sec=420.0,
    )

    assert adjusted["profit_take"] > 2.0
    assert adjusted["trail_start"] >= 1.7
    assert adjusted["trail_backoff"] < 0.55
    assert adjusted["loss_cut_hard_pips"] < 3.0
    assert adjusted["loss_cut_max_hold_sec"] < 420.0


def test_wick_blend_exit_adjustments_rebuild_low_quality_without_wick_payload() -> None:
    adjusted = wick_blend_exit_adjustments(
        side="short",
        thesis={
            "sl_pips": 1.83,
            "tp_pips": 2.48,
            "rsi": 53.19,
            "adx": 15.73,
            "atr_pips": 2.16,
            "range_score": 0.423,
            "vwap_gap": -6.16,
        },
        atr_pips=2.16,
        profit_take=1.2,
        trail_start=1.7,
        trail_backoff=0.55,
        lock_buffer=0.25,
        loss_cut_hard_pips=6.0,
        loss_cut_max_hold_sec=420.0,
    )

    assert adjusted["profit_take"] < 2.0
    assert adjusted["loss_cut_hard_pips"] < 2.5
    assert adjusted["loss_cut_max_hold_sec"] < 260.0


def test_wick_blend_exit_adjustments_tighten_on_projection_headwind() -> None:
    aligned = wick_blend_exit_adjustments(
        side="long",
        thesis={
            "sl_pips": 1.68,
            "tp_pips": 2.27,
            "atr_pips": 1.97,
            "wick_blend_quality": 0.894,
            "projection": {"score": 0.06},
        },
        atr_pips=1.97,
        profit_take=1.2,
        trail_start=1.7,
        trail_backoff=0.55,
        lock_buffer=0.25,
        loss_cut_hard_pips=6.0,
        loss_cut_max_hold_sec=420.0,
    )
    headwind = wick_blend_exit_adjustments(
        side="long",
        thesis={
            "sl_pips": 1.68,
            "tp_pips": 2.27,
            "atr_pips": 1.97,
            "wick_blend_quality": 0.894,
            "projection": {"score": -0.125},
        },
        atr_pips=1.97,
        profit_take=1.2,
        trail_start=1.7,
        trail_backoff=0.55,
        lock_buffer=0.25,
        loss_cut_hard_pips=6.0,
        loss_cut_max_hold_sec=420.0,
    )

    assert headwind["profit_take"] < aligned["profit_take"]
    assert headwind["loss_cut_hard_pips"] < aligned["loss_cut_hard_pips"]
    assert headwind["loss_cut_max_hold_sec"] < aligned["loss_cut_max_hold_sec"]
