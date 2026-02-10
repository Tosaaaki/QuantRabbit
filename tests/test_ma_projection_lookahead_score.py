from __future__ import annotations

from analysis.ma_projection import MACrossProjection, score_ma_for_side


def _ma(*, gap_pips: float, slope_pips: float, eta_bars: float | None) -> MACrossProjection:
    return MACrossProjection(
        fast_ma=150.0,
        slow_ma=150.0,
        gap_pips=gap_pips,
        prev_gap_pips=gap_pips - slope_pips,
        gap_slope_pips=slope_pips,
        fast_slope_pips=0.0,
        slow_slope_pips=0.0,
        price_to_fast_pips=0.0,
        price_to_slow_pips=0.0,
        projected_cross_bars=eta_bars,
        projected_cross_minutes=None,
    )


def test_score_ma_align_without_cross_is_positive() -> None:
    ma = _ma(gap_pips=1.0, slope_pips=0.1, eta_bars=None)
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=True) == 0.7
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=False) == 0.7


def test_score_ma_align_with_adverse_cross_keeps_legacy_penalty() -> None:
    # Long-aligned but projected bearish cross soon -> mild negative.
    ma = _ma(gap_pips=1.0, slope_pips=-0.2, eta_bars=1.0)
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=True) == -0.4
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=False) == -0.4


def test_score_ma_pre_cross_lookahead_can_turn_positive() -> None:
    # MA gap is still bearish, but projected bullish cross soon -> allow early positioning when enabled.
    ma = _ma(gap_pips=-1.0, slope_pips=0.4, eta_bars=1.0)
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=True) > 0.5
    assert score_ma_for_side(ma, "long", 5.0, lookahead_enabled=False) == -0.8


def test_score_ma_pre_cross_lookahead_works_for_short_side() -> None:
    # MA gap is still bullish, but projected bearish cross soon -> favourable for short when enabled.
    ma = _ma(gap_pips=1.0, slope_pips=-0.2, eta_bars=2.0)
    assert score_ma_for_side(ma, "short", 5.0, lookahead_enabled=True) > 0.3
    assert score_ma_for_side(ma, "short", 5.0, lookahead_enabled=False) == -0.8

