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
