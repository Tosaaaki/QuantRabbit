from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s import worker as scalp_worker


def _sample_signal(side: str) -> scalp_worker.TickSignal:
    return scalp_worker.TickSignal(
        side=side,
        mode="momentum",
        mode_score=1.1,
        momentum_score=0.9,
        revert_score=0.0,
        confidence=78,
        momentum_pips=0.3 if side == "long" else -0.3,
        trigger_pips=0.1,
        imbalance=0.68,
        tick_rate=6.0,
        span_sec=1.0,
        tick_age_ms=80.0,
        spread_pips=0.7,
        bid=154.40,
        ask=154.41,
        mid=154.405,
        range_pips=1.3,
        instant_range_pips=0.8,
        signal_window_sec=1.2,
    )


def test_extrema_reversal_respects_long_to_short_disable(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "EXTREMA_REVERSAL_ENABLED", True, raising=False)
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(scalp_worker.config, "EXTREMA_REVERSAL_MIN_SCORE", 1.0, raising=False)

    signal = _sample_signal("long")
    extrema_decision = scalp_worker.ExtremaGateDecision(
        allow_entry=True,
        reason="long_top_soft",
        units_mult=0.6,
        m1_pos=0.92,
        m5_pos=0.90,
        h4_pos=0.78,
    )
    reversed_signal, units_mult, reason, score = scalp_worker._extrema_reversal_route(
        signal,
        extrema_decision,
        regime=None,
        horizon=None,
        factors={},
    )

    assert reversed_signal is None
    assert units_mult == 1.0
    assert reason == ""
    assert score == 0.0


def test_extrema_reversal_keeps_short_to_long_path(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "EXTREMA_REVERSAL_ENABLED", True, raising=False)
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT",
        False,
        raising=False,
    )
    monkeypatch.setattr(scalp_worker.config, "EXTREMA_REVERSAL_MIN_SCORE", 1.45, raising=False)

    signal = _sample_signal("short")
    extrema_decision = scalp_worker.ExtremaGateDecision(
        allow_entry=True,
        reason="short_bottom_soft",
        units_mult=0.5,
        m1_pos=0.06,
        m5_pos=0.08,
        h4_pos=0.12,
    )
    reversed_signal, units_mult, reason, score = scalp_worker._extrema_reversal_route(
        signal,
        extrema_decision,
        regime=None,
        horizon=None,
        factors={},
    )

    assert reversed_signal is not None
    assert reversed_signal.side == "long"
    assert units_mult >= 0.1
    assert reason == "short_bottom_soft_reverse"
    assert score >= 1.45


def test_extrema_gate_uses_tighter_short_soft_mult_in_balanced_regime(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "EXTREMA_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_BOTTOM_BLOCK_POS",
        0.10,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_BOTTOM_SOFT_POS",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_H4_LOW_SOFT_POS",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_H4_LOW_BLOCK_POS",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT",
        0.42,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_SOFT_UNITS_MULT",
        0.60,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "EXTREMA_REQUIRE_M1_M5_AGREE_SHORT",
        True,
        raising=False,
    )

    def _fake_range_pos_from_candles(*, tf: str, lookback: int, min_span_pips: float) -> float:
        if tf == "M1":
            return 0.18
        if tf == "M5":
            return 0.26
        return 0.35

    monkeypatch.setattr(
        scalp_worker,
        "_range_pos_from_candles",
        _fake_range_pos_from_candles,
        raising=False,
    )

    regime = scalp_worker.MtfRegime(
        side="neutral",
        mode="balanced",
        trend_score=0.05,
        heat_score=0.30,
        adx_m1=15.0,
        adx_m5=14.0,
        atr_m1=0.8,
        atr_m5=1.2,
    )
    decision = scalp_worker._extrema_gate_decision("short", factors={}, regime=regime)

    assert decision.allow_entry is True
    assert decision.reason == "short_bottom_soft_balanced"
    assert decision.units_mult == 0.30
