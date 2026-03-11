from __future__ import annotations


def test_apply_signal_setup_context_carries_dynamic_setup_metadata():
    from workers.scalp_m1scalper import worker

    thesis: dict[str, object] = {}
    signal = {
        "entry_probability": 0.71,
        "setup_quality": 0.64,
        "setup_size_mult": 1.08,
        "flow_regime": "trend_long",
        "continuation_pressure": 1,
        "microstructure_bucket": "normal_fast",
        "setup_fingerprint": "M1Scalper|long|breakout_retest|trend_long|p1|normal_fast",
        "quality_components": {"breakout": 0.83, "pullback": 0.61},
        "notes": {"mode": "breakout_retest"},
    }

    payload = worker._apply_signal_setup_context(thesis, signal)

    assert payload["strategy_mode"] == "breakout_retest"
    assert thesis["strategy_mode"] == "breakout_retest"
    assert thesis["flow_regime"] == "trend_long"
    assert thesis["microstructure_bucket"] == "normal_fast"
    assert thesis["setup_fingerprint"] == signal["setup_fingerprint"]
    assert thesis["setup_quality"] == 0.64
    assert thesis["continuation_pressure"] == 1.0
    assert thesis["signal_entry_probability"] == 0.71
    assert thesis["m1_setup"]["setup_size_mult"] == 1.08
    assert thesis["setup_quality_components"]["breakout"] == 0.83
