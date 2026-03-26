from __future__ import annotations


def _base_thresholds() -> dict[str, float]:
    return {
        "min_hold_sec": 10.0,
        "max_hold_sec": 720.0,
        "max_adverse_pips": 6.0,
        "profit_take_pips": 2.2,
        "lock_trigger_pips": 0.77,
        "trail_start_pips": 2.8,
        "trail_backoff_pips": 0.9,
        "lock_buffer_pips": 0.5,
        "rsi_fade_long": 44.0,
        "rsi_fade_short": 56.0,
        "vwap_gap_pips": 0.8,
        "structure_adx": 20.0,
        "structure_gap_pips": 1.8,
        "atr_spike_pips": 5.0,
    }


def test_trade_local_exit_thresholds_relax_aligned_breakout_setup() -> None:
    from workers.scalp_m1scalper import exit_worker

    thresholds = exit_worker._trade_local_exit_thresholds(
        side="long",
        entry_thesis={
            "m1_setup": {
                "strategy_mode": "breakout_retest",
                "setup_quality": 0.84,
                "flow_regime": "trend_long",
                "continuation_pressure": 0,
                "setup_fingerprint": "M1Scalper|long|breakout_retest|trend_long|p0|normal_fast",
            }
        },
        **_base_thresholds(),
    )

    assert thresholds["setup_dynamicized"] is True
    assert thresholds["strategy_mode"] == "breakout_retest"
    assert (
        thresholds["setup_fingerprint"]
        == "M1Scalper|long|breakout_retest|trend_long|p0|normal_fast"
    )
    assert thresholds["flow_regime"] == "trend_long"
    assert thresholds["max_hold_sec"] > 720.0
    assert thresholds["max_adverse_pips"] > 6.0
    assert thresholds["profit_take_pips"] > 2.2
    assert thresholds["lock_trigger_pips"] > 0.77
    assert thresholds["trail_start_pips"] > 2.8
    assert thresholds["trail_backoff_pips"] > 0.9
    assert thresholds["lock_buffer_pips"] > 0.5
    assert thresholds["rsi_fade_long"] < 44.0
    assert thresholds["rsi_fade_short"] > 56.0
    assert thresholds["vwap_gap_pips"] < 0.8
    assert thresholds["structure_adx"] < 20.0
    assert thresholds["structure_gap_pips"] < 1.8
    assert thresholds["atr_spike_pips"] > 5.0


def test_trade_local_exit_thresholds_tighten_headwind_vshape_setup_from_fingerprint() -> (
    None
):
    from workers.scalp_m1scalper import exit_worker

    thresholds = exit_worker._trade_local_exit_thresholds(
        side="long",
        entry_thesis={
            "setup_quality": 0.32,
            "flow_regime": "trend_short",
            "continuation_pressure": 2,
            "setup_fingerprint": "M1Scalper|long|vshape_rebound|trend_short|p2|normal_fast",
        },
        **_base_thresholds(),
    )

    assert thresholds["setup_dynamicized"] is True
    assert thresholds["strategy_mode"] == "vshape_rebound"
    assert thresholds["flow_regime"] == "trend_short"
    assert thresholds["max_hold_sec"] < 720.0
    assert thresholds["max_adverse_pips"] < 6.0
    assert thresholds["profit_take_pips"] < 2.2
    assert thresholds["lock_trigger_pips"] < 0.77
    assert thresholds["trail_start_pips"] < 2.8
    assert thresholds["trail_backoff_pips"] < 0.9
    assert thresholds["lock_buffer_pips"] < 0.5
    assert thresholds["rsi_fade_long"] > 44.0
    assert thresholds["rsi_fade_short"] < 56.0
    assert thresholds["vwap_gap_pips"] > 0.8
    assert thresholds["structure_adx"] > 20.0
    assert thresholds["structure_gap_pips"] > 1.8
    assert thresholds["atr_spike_pips"] < 5.0


def test_trade_local_exit_thresholds_prioritize_nested_m1_setup_context() -> None:
    from workers.scalp_m1scalper import exit_worker

    thresholds = exit_worker._trade_local_exit_thresholds(
        side="long",
        entry_thesis={
            "strategy_mode": "vshape_rebound",
            "setup_quality": 0.21,
            "flow_regime": "trend_short",
            "continuation_pressure": 2,
            "setup_fingerprint": "M1Scalper|long|vshape_rebound|trend_short|p2|normal_fast",
            "m1_setup": {
                "strategy_mode": "breakout_retest",
                "setup_quality": 0.86,
                "flow_regime": "trend_long",
                "continuation_pressure": 0,
                "setup_fingerprint": "M1Scalper|long|breakout_retest|trend_long|p0|tight_fast",
            },
        },
        **_base_thresholds(),
    )

    assert thresholds["setup_dynamicized"] is True
    assert thresholds["strategy_mode"] == "breakout_retest"
    assert thresholds["flow_regime"] == "trend_long"
    assert (
        thresholds["setup_fingerprint"]
        == "M1Scalper|long|breakout_retest|trend_long|p0|tight_fast"
    )
    assert thresholds["setup_quality"] == 0.86
    assert thresholds["continuation_pressure"] == 0.0
    assert thresholds["profit_take_pips"] > 2.2
    assert thresholds["max_hold_sec"] > 720.0
