from __future__ import annotations

import execution.strategy_entry as strategy_entry


def test_apply_participation_alloc_trims_and_boosts_probability(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_MULT_MAX", 1.12, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MomentumBurst",
            "lot_multiplier": 0.8,
            "probability_boost": 0.05,
            "preflights": 120,
            "filled": 18,
            "fill_rate": 0.15,
            "hard_block_rate": 0.22,
            "quality_score": 0.61,
            "current_share": 0.32,
            "target_share": 0.18,
        },
        raising=False,
    )

    thesis: dict = {}
    units, prob, payload = strategy_entry._apply_participation_alloc(
        strategy_tag="MomentumBurst",
        pocket="micro",
        units=100,
        min_units=10,
        entry_probability=0.60,
        entry_thesis=thesis,
    )

    assert units == 80
    assert prob is not None and prob > 0.60
    assert isinstance(payload, dict)
    assert thesis["participation_alloc"]["reason"] == "rebalance"


def test_apply_participation_alloc_trims_units_and_lowers_probability_for_loser_lane(
    monkeypatch,
) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_MULT_MAX", 1.12, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MicroTrendRetest-long",
            "action": "trim_units",
            "units_multiplier": 0.82,
            "lot_multiplier": 0.82,
            "probability_offset": -0.04,
            "max_probability_boost": 0.03,
            "preflights": 144,
            "filled": 19,
            "fill_rate": 0.1319,
            "hard_block_rate": 0.24,
            "quality_score": 0.42,
            "current_share": 0.31,
            "target_share": 0.15,
        },
        raising=False,
    )

    thesis: dict = {}
    units, prob, payload = strategy_entry._apply_participation_alloc(
        strategy_tag="MicroTrendRetest-long",
        pocket="micro",
        units=100,
        min_units=10,
        entry_probability=0.60,
        entry_thesis=thesis,
    )

    assert units == 82
    assert prob == 0.57
    assert isinstance(payload, dict)
    assert payload["reason"] == "overused_trim"
    assert payload["entry_probability_before"] == 0.6
    assert payload["entry_probability_after"] == 0.57
    assert payload["probability_offset"] == -0.03
    assert thesis["participation_alloc"]["action"] == "trim_units"


def test_apply_participation_alloc_boosts_units_for_explicit_boost_signal(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_MULT_MAX", 1.12, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MomentumBurst",
            "action": "boost_participation",
            "units_multiplier": 1.18,
            "lot_multiplier": 1.18,
            "max_units_boost": 0.12,
            "probability_boost": 0.05,
            "max_probability_boost": 0.05,
            "preflights": 42,
            "filled": 21,
            "fill_rate": 0.5,
            "hard_block_rate": 0.0,
            "quality_score": 0.92,
            "current_share": 0.08,
            "target_share": 0.22,
        },
        raising=False,
    )

    thesis: dict = {}
    units, prob, payload = strategy_entry._apply_participation_alloc(
        strategy_tag="MomentumBurst",
        pocket="micro",
        units=-100,
        min_units=10,
        entry_probability=0.60,
        entry_thesis=thesis,
    )

    assert units == -112
    assert prob == 0.62
    assert isinstance(payload, dict)
    assert payload["reason"] == "boost_participation"
    assert payload["lot_multiplier"] == 1.12
    assert payload["applied_units"] == 112
    assert thesis["participation_alloc"]["action"] == "boost_participation"


def test_apply_participation_alloc_does_not_boost_units_without_explicit_signal(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_MULT_MAX", 1.12, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MomentumBurst",
            "action": "hold",
            "units_multiplier": 1.18,
            "lot_multiplier": 1.18,
            "max_units_boost": 0.12,
            "probability_boost": 0.05,
            "max_probability_boost": 0.05,
            "preflights": 42,
            "filled": 21,
            "fill_rate": 0.5,
            "hard_block_rate": 0.0,
            "quality_score": 0.92,
            "current_share": 0.08,
            "target_share": 0.22,
        },
        raising=False,
    )

    thesis: dict = {}
    units, prob, payload = strategy_entry._apply_participation_alloc(
        strategy_tag="MomentumBurst",
        pocket="micro",
        units=100,
        min_units=10,
        entry_probability=0.60,
        entry_thesis=thesis,
    )

    assert units == 100
    assert prob == 0.62
    assert isinstance(payload, dict)
    assert payload["reason"] == "underused_boost"
    assert thesis["participation_alloc"]["lot_multiplier"] == 1.0


def test_inject_market_context_records_slow_market_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_MARKET_CONTEXT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry.market_context,
        "current_context",
        lambda **_kwargs: {
            "generated_at": "2026-03-10T07:00:00+00:00",
            "age_sec": 12.5,
            "stale": False,
            "pair": "USD_JPY",
            "pair_price": 157.88,
            "pair_change_pct_24h": 0.21,
            "dxy_change_pct_24h": 0.18,
            "us_jp_10y_spread": 3.1,
            "risk_mode": "risk_on",
            "high_impact_events": 1,
            "total_events": 3,
            "minutes_to_next_event": 42,
            "next_event_name": "US CPI",
            "event_severity": "medium",
            "bias_score": 0.44,
            "bias_label": "usd_jpy_bullish",
        },
        raising=False,
    )

    thesis: dict = {}
    updated, payload = strategy_entry._inject_market_context(
        thesis,
        instrument="USD_JPY",
    )

    assert updated is thesis
    assert isinstance(payload, dict)
    assert thesis["market_context"]["bias_label"] == "usd_jpy_bullish"
    assert thesis["market_bias_label"] == "usd_jpy_bullish"


def test_apply_auto_canary_caps_units_and_probability(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_AUTO_CANARY_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry.auto_canary,
        "current_override",
        lambda _strategy_tag: {
            "enabled": True,
            "strategy_key": "MicroTrendRetest",
            "units_multiplier": 0.85,
            "probability_offset": -0.03,
            "reason": "loser_cluster_canary",
            "confidence": 0.72,
        },
        raising=False,
    )

    thesis: dict = {}
    units, prob, payload = strategy_entry._apply_auto_canary(
        strategy_tag="MicroTrendRetest-long",
        pocket="micro",
        units=100,
        min_units=10,
        entry_probability=0.62,
        entry_thesis=thesis,
    )

    assert units == 85
    assert prob == 0.59
    assert isinstance(payload, dict)
    assert thesis["auto_canary"]["reason"] == "loser_cluster_canary"


def test_apply_dynamic_alloc_trim_skips_stale_profile(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_TRIM_ONLY", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_strategy_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MicroRangeBreak",
            "lot_multiplier": 0.25,
            "payload_stale": True,
        },
        raising=False,
    )

    thesis: dict = {}
    units, reason = strategy_entry._apply_dynamic_alloc_trim(
        strategy_tag="MicroRangeBreak",
        pocket="micro",
        units=100,
        min_units=10,
        entry_thesis=thesis,
    )

    assert units == 100
    assert reason is None
    assert "dynamic_alloc" not in thesis


def test_inject_live_setup_context_records_flow_regime_and_fingerprint() -> None:
    thesis = {
        "strategy_tag": "RangeFader-sell-fade",
        "range_mode": "transition",
        "range_score": 0.24,
        "technical_context": {
            "indicators": {
                "M1": {
                    "atr_pips": 2.6,
                    "rsi": 69.4,
                    "adx": 28.2,
                    "plus_di": 31.0,
                    "minus_di": 14.0,
                    "ma10": 158.05,
                    "ma20": 158.02,
                }
            },
            "ticks": {
                "spread_pips": 0.8,
                "tick_rate": 9.4,
            },
        },
    }

    updated, payload = strategy_entry._inject_live_setup_context(thesis, units=-120)

    assert updated is thesis
    assert isinstance(payload, dict)
    assert payload["flow_regime"] == "trend_long"
    assert payload["microstructure_bucket"] == "tight_fast"
    assert payload["gap_bucket"] == "up_strong"
    assert thesis["flow_regime"] == "trend_long"
    assert thesis["microstructure_bucket"] == "tight_fast"
    assert thesis["setup_fingerprint"].startswith("RangeFader-sell-fade|short|trend_long|tight_fast|")
    assert thesis["live_setup_context"]["setup_fingerprint"] == thesis["setup_fingerprint"]
