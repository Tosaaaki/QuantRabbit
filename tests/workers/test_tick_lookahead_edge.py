from __future__ import annotations

from workers.common.tick_lookahead_edge import decide_tick_lookahead_edge


def _ticks(mids: list[float], start: float = 100.0, step: float = 0.2) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for idx, mid in enumerate(mids):
        out.append(
            {
                "epoch": start + idx * step,
                "bid": mid - 0.001,
                "ask": mid + 0.001,
                "mid": mid,
            }
        )
    return out


def _decide(**kwargs):
    base = {
        "signal_span_sec": 1.2,
        "pip_value": 0.01,
        "horizon_sec": 3.0,
        "edge_min_pips": 0.10,
        "edge_ref_pips": 0.60,
        "units_min_mult": 0.55,
        "units_max_mult": 1.25,
        "slippage_base_pips": 0.05,
        "slippage_spread_mult": 0.28,
        "slippage_range_mult": 0.14,
        "latency_penalty_pips": 0.03,
        "safety_margin_pips": 0.02,
        "momentum_decay": 0.60,
        "momentum_weight": 0.72,
        "flow_weight": 0.30,
        "rate_weight": 0.22,
        "bias_weight": 0.32,
        "trigger_weight": 0.25,
        "counter_penalty": 0.80,
        "allow_thin_edge": True,
    }
    base.update(kwargs)
    return decide_tick_lookahead_edge(**base)


def test_tick_lookahead_blocks_negative_edge() -> None:
    decision = _decide(
        ticks=_ticks([150.03, 150.02, 150.01, 150.00, 149.99, 149.98]),
        side="long",
        spread_pips=0.30,
        momentum_pips=-1.4,
        trigger_pips=0.9,
        imbalance=0.62,
        tick_rate=5.5,
        direction_bias_score=-0.4,
    )
    assert decision.allow_entry is False
    assert decision.reason == "edge_negative_block"
    assert decision.edge_pips <= 0.0
    assert decision.units_mult == 0.0


def test_tick_lookahead_thin_edge_scales_units() -> None:
    decision = _decide(
        ticks=_ticks([150.00, 150.003, 150.005, 150.006, 150.007, 150.008]),
        side="long",
        spread_pips=0.25,
        momentum_pips=0.5,
        trigger_pips=0.7,
        imbalance=0.56,
        tick_rate=3.8,
        direction_bias_score=0.0,
    )
    assert decision.allow_entry is True
    assert decision.reason == "edge_thin_scale"
    assert 0.0 < decision.edge_pips < 0.10
    assert 0.55 <= decision.units_mult < 1.0


def test_tick_lookahead_thin_edge_can_block_when_disabled() -> None:
    decision = _decide(
        ticks=_ticks([150.00, 150.003, 150.005, 150.006, 150.007, 150.008]),
        side="long",
        spread_pips=0.25,
        momentum_pips=0.5,
        trigger_pips=0.7,
        imbalance=0.56,
        tick_rate=3.8,
        direction_bias_score=0.0,
        allow_thin_edge=False,
    )
    assert decision.allow_entry is False
    assert decision.reason == "edge_thin_block"
    assert 0.0 < decision.edge_pips < 0.10


def test_tick_lookahead_scales_up_when_edge_is_strong() -> None:
    decision = _decide(
        ticks=_ticks([150.00, 150.02, 150.03, 150.05, 150.06, 150.08]),
        side="long",
        spread_pips=0.15,
        momentum_pips=2.8,
        trigger_pips=0.9,
        imbalance=0.74,
        tick_rate=8.0,
        direction_bias_score=0.55,
    )
    assert decision.allow_entry is True
    assert decision.reason == "edge_ok"
    assert decision.edge_pips >= 0.10
    assert decision.units_mult > 1.0

