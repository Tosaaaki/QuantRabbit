from __future__ import annotations

from types import SimpleNamespace

from quant_rabbit.analysis.market_state import (
    PHASES,
    classify_market_state,
    summarize_market_states,
)


def _indicators(**updates):
    values = {
        "valid": True,
        "plus_di_14": 28.0,
        "minus_di_14": 14.0,
        "ema_slope_5": 2.0,
        "ema_slope_20": 1.0,
        "atr_percentile_100": 0.5,
        "bb_squeeze": 0,
        "close": 1.1050,
        "donchian_low": 1.1000,
        "donchian_high": 1.1100,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _regime(state: str, *, confidence: float = 0.75, atr: float = 50.0):
    return SimpleNamespace(state=state, confidence=confidence, atr_percentile=atr)


def _families(trend: float, mean_rev: float, breakout: float):
    return SimpleNamespace(
        trend_score=trend,
        mean_rev_score=mean_rev,
        breakout_score=breakout,
    )


def test_every_statistical_regime_maps_to_one_of_four_action_phases() -> None:
    expected = {
        "TREND_STRONG": "TREND",
        "RANGE": "RANGE",
        "BREAKOUT_PENDING": "PRE_TREND",
        "TREND_WEAK": "PRE_TREND",
        "TRANSITION": "PRE_RANGE",
        "UNKNOWN": "PRE_RANGE",
    }
    for state, phase in expected.items():
        result = classify_market_state(
            indicators=_indicators(),
            regime_reading=_regime(state),
            family_scores=_families(0.8, 0.2, 0.1),
            legacy_regime="UNCLEAR",
        )
        assert result.phase in PHASES
        assert result.phase == phase
        assert result.entry_mode
        assert result.invalidation_phase in PHASES


def test_weak_trend_with_mean_reversion_dominance_is_pre_range() -> None:
    result = classify_market_state(
        indicators=_indicators(ema_slope_5=0.5, ema_slope_20=1.0),
        regime_reading=_regime("TREND_WEAK"),
        family_scores=_families(0.2, -0.9, 0.1),
        legacy_regime="FAILURE_RISK",
    )
    assert result.phase == "PRE_RANGE"
    assert result.momentum == "DECELERATING"
    assert result.strategy_family == "REVERSAL"


def test_breakout_pending_is_armed_pre_trend_not_unknown_or_no_trade() -> None:
    result = classify_market_state(
        indicators=_indicators(atr_percentile_100=0.1, bb_squeeze=1),
        regime_reading=_regime("BREAKOUT_PENDING", atr=10.0),
        family_scores=_families(0.3, 0.1, 0.9),
        legacy_regime="UNCLEAR",
    )
    assert result.phase == "PRE_TREND"
    assert result.volatility == "COMPRESSION"
    assert result.readiness == "ARMED"
    assert result.entry_mode == "STOP_ENTRY_AFTER_CONFIRMATION"


def test_pair_summary_calls_lower_timeframe_transition_early_not_conflict() -> None:
    short_state = classify_market_state(
        indicators=_indicators(),
        regime_reading=_regime("BREAKOUT_PENDING", atr=10.0),
        family_scores=_families(0.8, 0.1, 0.9),
    )
    long_state = classify_market_state(
        indicators=_indicators(),
        regime_reading=_regime("RANGE", atr=30.0),
        family_scores=_families(0.1, -0.8, 0.1),
    )
    summary = summarize_market_states([
        SimpleNamespace(granularity="M5", market_state=short_state),
        SimpleNamespace(granularity="H1", market_state=long_state),
    ])
    assert summary["phase_alignment"] == "LOWER_TF_LEADING"
    assert summary["classified_timeframe_count"] == 2
