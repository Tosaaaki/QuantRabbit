from __future__ import annotations

from analysis.pattern_book import (
    PatternAggregate,
    build_pattern_id,
    classify_pattern_action,
)


def test_build_pattern_id_from_entry_thesis() -> None:
    thesis = {
        "strategy_tag": "scalp_ping_5s_live",
        "signal_mode": "momentum_hz",
        "mtf_regime_gate": "mtf_reversion_aligned",
        "horizon_gate": "horizon_align",
        "extrema_gate_reason": "short_bottom_soft",
        "section_axis": {
            "high": 153.06,
            "low": 152.82,
        },
        "entry_ref": 152.876,
        "pattern_tag": "c:spin_dn|w:lower|tr:dn_mild",
    }
    pattern_id = build_pattern_id(
        entry_thesis=thesis,
        units=-1400,
        pocket="scalp_fast",
        strategy_tag_fallback="fallback_tag",
    )
    assert "st:scalp_ping_5s_live" in pattern_id
    assert "pk:scalp_fast" in pattern_id
    assert "sd:short" in pattern_id
    assert "sg:momentum_hz" in pattern_id
    assert "mtf:mtf_reversion_aligned" in pattern_id
    assert "hz:horizon_align" in pattern_id
    assert "ex:short_bottom_soft" in pattern_id
    assert "rg:low" in pattern_id


def test_classify_pattern_action_block_reduce_boost() -> None:
    block = classify_pattern_action(
        PatternAggregate(
            trades=180,
            wins=65,
            losses=115,
            win_rate=0.361,
            avg_pips=-0.21,
            total_pips=-37.8,
            gross_profit=18.0,
            gross_loss=42.0,
            profit_factor=0.428,
        ),
        min_samples_soft=30,
        min_samples_block=120,
    )
    assert block.action == "block"
    assert block.lot_multiplier == 0.0

    reduce = classify_pattern_action(
        PatternAggregate(
            trades=80,
            wins=35,
            losses=45,
            win_rate=0.4375,
            avg_pips=-0.06,
            total_pips=-4.8,
            gross_profit=12.0,
            gross_loss=14.0,
            profit_factor=0.857,
        ),
        min_samples_soft=30,
        min_samples_block=120,
    )
    assert reduce.action == "reduce"
    assert reduce.lot_multiplier < 1.0

    boost = classify_pattern_action(
        PatternAggregate(
            trades=95,
            wins=60,
            losses=35,
            win_rate=0.632,
            avg_pips=0.13,
            total_pips=12.35,
            gross_profit=30.0,
            gross_loss=20.0,
            profit_factor=1.5,
        ),
        min_samples_soft=30,
        min_samples_block=120,
    )
    assert boost.action == "boost"
    assert boost.lot_multiplier > 1.0
