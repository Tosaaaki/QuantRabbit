from __future__ import annotations

import os

import pytest

from workers.micro_runtime import worker


def _patch_trend_snapshot(monkeypatch, direction: str = "short", adx: float = 30.0) -> None:
    monkeypatch.setattr(
        worker,
        "_trend_snapshot",
        lambda *_args, **_kwargs: {
            "tf": "H4",
            "gap_pips": -20.0 if direction == "short" else 20.0,
            "direction": direction,
            "adx": adx,
        },
    )
    monkeypatch.setattr(worker.config, "TREND_FLIP_ENABLED", True)
    monkeypatch.setattr(worker.config, "TREND_FLIP_ADX_MIN", 20.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_TP_MULT", 1.12)
    monkeypatch.setattr(worker.config, "TREND_FLIP_SL_MULT", 0.95)


def test_trend_flip_blocklist_prevents_flip(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_ALLOWLIST", frozenset())
    monkeypatch.setattr(
        worker.config,
        "TREND_FLIP_STRATEGY_BLOCKLIST",
        frozenset({"MicroLevelReactor"}),
    )

    side, tag, flip_meta, tp_mult, sl_mult, trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroLevelReactor-bounce-lower",
        strategy_name="MicroLevelReactor",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert trend is not None
    assert side == "long"
    assert tag == "MicroLevelReactor-bounce-lower"
    assert flip_meta is None
    assert tp_mult == 1.0
    assert sl_mult == 1.0


def test_trend_flip_allowlist_limits_targets(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(
        worker.config,
        "TREND_FLIP_STRATEGY_ALLOWLIST",
        frozenset({"MicroVWAPBound"}),
    )
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_BLOCKLIST", frozenset())

    side, tag, flip_meta, tp_mult, sl_mult, _trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroCompressionRevert-long",
        strategy_name="MicroCompressionRevert",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert side == "long"
    assert tag == "MicroCompressionRevert-long"
    assert flip_meta is None
    assert tp_mult == 1.0
    assert sl_mult == 1.0


def test_trend_flip_still_applies_when_allowed(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_ALLOWLIST", frozenset())
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_BLOCKLIST", frozenset())

    side, tag, flip_meta, tp_mult, sl_mult, trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroVWAPBound-short",
        strategy_name="MicroVWAPBound",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert trend is not None
    assert side == "short"
    assert tag.endswith("-trendflip")
    assert flip_meta == {
        "from": "long",
        "to": "short",
        "tf": "H4",
        "gap_pips": -20.0,
        "adx": 35.0,
    }
    assert tp_mult == 1.12
    assert sl_mult == 0.95

def test_strategy_cooldown_active_when_recent(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", False)
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == 45.0
    assert worker._strategy_cooldown_active("MicroLevelReactor", 120.0) is True
    assert worker._strategy_cooldown_active("MicroLevelReactor", 146.0) is False


def test_strategy_cooldown_disabled_by_default(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", False)
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == 0.0
    assert worker._strategy_cooldown_active("MicroLevelReactor", 110.0) is False


def test_momentumburst_entry_thesis_marks_only_reaccel_signals() -> None:
    reaccel_signal = {"metadata": {"momentum_burst": {"reaccel": True}}}
    normal_signal = {"metadata": {"momentum_burst": {"reaccel": False}}}

    assert worker._momentumburst_entry_thesis_reaccel("MomentumBurst", reaccel_signal) is True
    assert worker._momentumburst_entry_thesis_reaccel("MomentumBurst", normal_signal) is False
    assert worker._momentumburst_entry_thesis_reaccel("MicroLevelReactor", reaccel_signal) is False


def test_history_profile_uses_override_skip_threshold(monkeypatch):
    monkeypatch.setattr(worker.config, "HIST_ENABLED", True)
    monkeypatch.setattr(worker.config, "HIST_MIN_TRADES", 12)
    monkeypatch.setattr(worker.config, "HIST_REGIME_MIN_TRADES", 8)
    monkeypatch.setattr(worker.config, "HIST_SKIP_SCORE", 0.20)
    monkeypatch.setattr(worker.config, "HIST_SKIP_SCORE_OVERRIDE", 0.18)
    monkeypatch.setattr(worker.config, "HIST_LOT_MIN", 0.72)
    monkeypatch.setattr(worker.config, "HIST_LOT_MAX", 1.28)
    monkeypatch.setattr(worker.config, "HIST_PF_CAP", 2.0)
    monkeypatch.setattr(worker.config, "HIST_TTL_SEC", 30.0)
    monkeypatch.setattr(
        worker,
        "_query_strategy_history",
        lambda **_kwargs: {
            "n": 312,
            "pf": 0.9,
            "win_rate": 0.35,
            "avg_pips": 0.2,
        },
    )
    monkeypatch.setattr(worker, "_derive_history_score", lambda _row: 0.182)
    worker._HISTORY_PROFILE_CACHE.clear()

    profile = worker._history_profile(
        strategy_key="MicroLevelReactor",
        pocket="micro",
        regime_label=None,
    )

    assert profile["enabled"] is True
    assert profile["skip"] is False
    assert profile["skip_score_threshold"] == pytest.approx(0.18)


def test_setup_history_profile_marks_recent_winner(monkeypatch):
    monkeypatch.setattr(worker.config, "HIST_SETUP_WINNER_PROTECT_ENABLED", True)
    monkeypatch.setattr(worker.config, "HIST_SETUP_WINNER_MIN_TRADES", 2)
    monkeypatch.setattr(worker.config, "HIST_SETUP_WINNER_SCORE", 0.58)
    monkeypatch.setattr(worker.config, "HIST_MIN_TRADES", 12)
    monkeypatch.setattr(worker.config, "HIST_PF_CAP", 2.0)
    monkeypatch.setattr(worker.config, "HIST_TTL_SEC", 30.0)
    monkeypatch.setattr(
        worker,
        "_query_setup_history",
        lambda **_kwargs: {
            "n": 2,
            "pf": float("inf"),
            "win_rate": 1.0,
            "avg_pips": 3.0,
        },
    )
    worker._SETUP_HISTORY_PROFILE_CACHE.clear()

    profile = worker._setup_history_profile(
        "MicroLevelReactor-breakout-long|long|range_compression|tight_normal",
        "micro",
    )

    assert profile["enabled"] is True
    assert profile["winner_protect"] is True
    assert profile["score"] == pytest.approx(0.58, rel=1e-3)


def test_apply_setup_history_winner_override_unblocks_hist_skip(monkeypatch):
    monkeypatch.setattr(worker.config, "HIST_SETUP_WINNER_PROTECT_ENABLED", True)
    monkeypatch.setattr(
        worker,
        "_setup_history_profile",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "winner_protect": True,
            "n": 6,
            "score": 0.747,
            "pf": 2.0,
            "win_rate": 1.0,
            "avg_pips": 3.7,
        },
    )

    hist_profile, setup_profile = worker._apply_setup_history_winner_override(
        {
            "skip": True,
            "source": "global",
            "n": 312,
            "score": 0.182,
        },
        setup_fingerprint="MicroLevelReactor-bounce-lower|long|range_fade|tight_normal",
        pocket="micro",
    )

    assert setup_profile["winner_protect"] is True
    assert hist_profile["skip"] is False
    assert hist_profile["source"] == "global+setup_winner"
    assert hist_profile["winner_setup_override"]["setup_fingerprint"].startswith(
        "MicroLevelReactor-bounce-lower"
    )


def test_strategy_cooldown_extends_with_fresh_participation_trim(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", False)
    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "trim_units",
            "cadence_floor": 0.90,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == pytest.approx(50.0)


def test_strategy_cooldown_shortens_with_fresh_participation_boost(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", False)
    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "boost_participation",
            "cadence_floor": 1.10,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MomentumBurst") == pytest.approx(45.0 / 1.10)


def test_strategy_cooldown_boost_offsets_mild_dynamic_alloc_trim(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MULT_MIN", 0.7)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MULT_MAX", 1.8)
    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "boost_participation",
            "cadence_floor": 1.10,
        },
        raising=False,
    )
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 24,
            "lot_multiplier": 0.95,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("PrecisionLowVol") == pytest.approx(
        45.0 / (1.10 * 0.95),
        rel=1e-3,
    )


def test_strategy_cooldown_uses_dynamic_alloc_when_base_is_zero(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(worker.config, "LOOP_INTERVAL_SEC", 8.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MULT_MIN", 0.7)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MULT_MAX", 1.8)
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 24,
            "lot_multiplier": 0.80,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == pytest.approx(10.0)


def test_strategy_cooldown_ignores_stale_missing_and_noop_dynamic_inputs(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": True,
            "protect_frequency": True,
            "action": "trim_units",
            "cadence_floor": 0.90,
        },
        raising=False,
    )
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": True,
            "trades": 24,
            "lot_multiplier": 0.80,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == 45.0

    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 8,
            "lot_multiplier": 0.80,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == 45.0

    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {"found": False},
        raising=False,
    )
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 24,
            "lot_multiplier": 1.05,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == 45.0


def test_momentumburst_reaccel_shortens_before_dynamic_extension(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker.config, "MOMENTUMBURST_REACCEL_COOLDOWN_SEC", 20.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 24,
            "lot_multiplier": 0.80,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec(
        "MomentumBurst",
        {"metadata": {"momentum_burst": {"reaccel": True}}},
    ) == pytest.approx(25.0)


def test_trendretest_side_specific_dynamic_alloc_key_is_resolved(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    seen = []

    def _fake_load(strategy, *_args, **_kwargs):
        seen.append(strategy)
        if strategy == "MicroTrendRetest-long":
            return {
                "found": True,
                "payload_stale": False,
                "trades": 24,
                "lot_multiplier": 0.75,
            }
        if strategy == "MicroTrendRetest-long-trendflip":
            return {
                "found": False,
                "payload_stale": False,
                "trades": 0,
                "lot_multiplier": 1.0,
            }
        return {
            "found": True,
            "payload_stale": False,
            "trades": 18,
            "lot_multiplier": 0.95,
        }

    monkeypatch.setattr(worker, "load_dynamic_alloc_profile", _fake_load, raising=False)

    assert worker._strategy_effective_cooldown_sec(
        "MicroTrendRetest",
        {"tag": "MicroTrendRetest-long-trendflip"},
    ) == pytest.approx(60.0)
    assert seen[:2] == ["MicroTrendRetest-long-trendflip", "MicroTrendRetest-long"]


def test_momentumburst_open_long_dynamic_alloc_key_is_resolved(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    seen = []

    def _fake_load(strategy, *_args, **_kwargs):
        seen.append(strategy)
        if strategy == "MomentumBurst-open_long":
            return {
                "found": True,
                "payload_stale": False,
                "trades": 18,
                "lot_multiplier": 0.75,
            }
        if strategy == "MomentumBurst-open_long-reaccel":
            return {
                "found": False,
                "payload_stale": False,
                "trades": 0,
                "lot_multiplier": 1.0,
            }
        return {
            "found": True,
            "payload_stale": False,
            "trades": 18,
            "lot_multiplier": 0.95,
        }

    monkeypatch.setattr(worker, "load_dynamic_alloc_profile", _fake_load, raising=False)

    assert worker._strategy_effective_cooldown_sec(
        "MomentumBurst",
        {"tag": "MomentumBurst-open_long-reaccel"},
    ) == pytest.approx(60.0)
    assert seen[:2] == ["MomentumBurst-open_long-reaccel", "MomentumBurst-open_long"]


def test_setup_tag_prefers_exact_profile_key_before_strategy_fallback(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    seen = []

    def _fake_load(strategy, *_args, **_kwargs):
        seen.append(strategy)
        if strategy == "MicroLevelReactor-bounce-lower":
            return {
                "found": True,
                "payload_stale": False,
                "trades": 18,
                "lot_multiplier": 0.75,
            }
        return {
            "found": True,
            "payload_stale": False,
            "trades": 18,
            "lot_multiplier": 0.95,
        }

    monkeypatch.setattr(worker, "load_dynamic_alloc_profile", _fake_load, raising=False)

    assert worker._strategy_effective_cooldown_sec(
        "MicroLevelReactor",
        {"tag": "MicroLevelReactor-bounce-lower"},
    ) == pytest.approx(60.0)
    assert seen == ["MicroLevelReactor-bounce-lower"]


def test_strategy_cooldown_uses_stronger_dynamic_extension_when_both_present(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    monkeypatch.setattr(worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(
        worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "trim_units",
            "cadence_floor": 0.90,
        },
        raising=False,
    )
    monkeypatch.setattr(
        worker,
        "load_dynamic_alloc_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "trades": 24,
            "lot_multiplier": 0.80,
        },
        raising=False,
    )

    assert worker._strategy_effective_cooldown_sec("MicroLevelReactor") == pytest.approx(56.25)


def test_mlr_strict_range_gate_blocks_without_range_context(monkeypatch):
    monkeypatch.setattr(worker.config, "MLR_STRICT_RANGE_GATE", True)
    monkeypatch.setattr(worker.config, "MLR_MIN_RANGE_SCORE", 0.62)
    monkeypatch.setattr(worker.config, "MLR_MAX_ADX", 20.0)
    monkeypatch.setattr(worker.config, "MLR_MAX_MA_GAP_PIPS", 2.2)

    ok, diag = worker._mlr_strict_range_ok(
        {"adx": 24.0, "ma10": 150.050, "ma20": 150.020},
        range_active=False,
        range_score=0.48,
    )

    assert ok is False
    assert diag["adx"] == 24.0
    assert diag["range_score"] == 0.48
    assert diag["ma_gap_pips"] > 2.2


def test_mlr_strict_range_gate_allows_strong_range_context(monkeypatch):
    monkeypatch.setattr(worker.config, "MLR_STRICT_RANGE_GATE", True)
    monkeypatch.setattr(worker.config, "MLR_MIN_RANGE_SCORE", 0.62)
    monkeypatch.setattr(worker.config, "MLR_MAX_ADX", 20.0)
    monkeypatch.setattr(worker.config, "MLR_MAX_MA_GAP_PIPS", 2.2)

    ok, diag = worker._mlr_strict_range_ok(
        {"adx": 16.0, "ma10": 150.023, "ma20": 150.010},
        range_active=True,
        range_score=0.70,
    )

    assert ok is True
    assert diag["adx"] == 16.0
    assert diag["range_active"] == 1.0


def test_micro_chop_context_detects_rotational_m1(monkeypatch):
    monkeypatch.setattr(worker.config, "CHOP_ENABLED", True)
    monkeypatch.setattr(worker.config, "CHOP_LOOKBACK_BARS", 6)
    monkeypatch.setattr(worker.config, "CHOP_SIGN_FLIP_MIN", 0.40)
    monkeypatch.setattr(worker.config, "CHOP_DIRECTIONAL_EFF_MAX", 0.20)
    monkeypatch.setattr(worker.config, "CHOP_MEAN_RANGE_MIN_PIPS", 2.0)

    ctx = worker._micro_chop_context(
        {
            "candles": [
                {"open": 150.00, "high": 150.03, "low": 149.99, "close": 150.02},
                {"open": 150.02, "high": 150.03, "low": 149.99, "close": 150.00},
                {"open": 150.00, "high": 150.04, "low": 149.99, "close": 150.03},
                {"open": 150.03, "high": 150.04, "low": 150.00, "close": 150.01},
                {"open": 150.01, "high": 150.05, "low": 150.00, "close": 150.04},
                {"open": 150.04, "high": 150.05, "low": 150.01, "close": 150.02},
            ]
        }
    )

    assert ctx["active"] is True
    assert ctx["score"] > 0.55
    assert ctx["sign_flip_ratio"] >= 0.40
    assert ctx["directional_eff"] <= 0.20


def test_mlr_strict_range_gate_allows_chop_override(monkeypatch):
    monkeypatch.setattr(worker.config, "MLR_STRICT_RANGE_GATE", True)
    monkeypatch.setattr(worker.config, "MLR_MIN_RANGE_SCORE", 0.62)
    monkeypatch.setattr(worker.config, "MLR_MAX_ADX", 20.0)
    monkeypatch.setattr(worker.config, "MLR_MAX_MA_GAP_PIPS", 2.2)
    monkeypatch.setattr(worker.config, "MLR_CHOP_OVERRIDE_ENABLED", True)
    monkeypatch.setattr(worker.config, "MLR_CHOP_SCORE_MIN", 0.55)
    monkeypatch.setattr(worker.config, "MLR_CHOP_MAX_ADX", 40.0)
    monkeypatch.setattr(worker.config, "MLR_CHOP_MAX_MA_GAP_PIPS", 5.0)

    ok, diag = worker._mlr_strict_range_ok(
        {"adx": 31.0, "ma10": 150.048, "ma20": 150.010},
        range_active=False,
        range_score=0.44,
        chop_ctx={"active": True, "score": 0.72},
    )

    assert ok is True
    assert diag["chop_active"] == 1.0
    assert diag["chop_override"] == 1.0


def test_strategy_chop_units_multiplier_reduces_momentumburst_in_chop(monkeypatch):
    monkeypatch.setattr(worker.config, "CHOP_STRATEGY_UNITS_MULT", {"MomentumBurst": 0.70})

    mult = worker._strategy_chop_units_multiplier(
        "MomentumBurst",
        {"active": True, "score": 0.75},
    )

    assert abs(mult - 0.775) < 1e-6


def test_allowed_strategies_matches_momentumburst_strategy_name(monkeypatch):
    monkeypatch.setenv("MICRO_STRATEGY_ALLOWLIST", "MomentumBurst")

    allowed = worker._allowed_strategies()
    names = [getattr(cls, "name", cls.__name__) for cls in allowed]

    assert names == ["MomentumBurst"]


def test_strategy_fac_view_includes_mtf_and_trend_snapshot(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="long", adx=35.0)

    fac = worker._strategy_fac_view(
        fac_m1={"close": 158.30, "candles": [{"close": 158.30}]},
        fac_m5={"candles": [{"close": 158.28}]},
        fac_h1={"candles": [{"close": 158.12}]},
        fac_h4={"candles": [{"close": 157.98}]},
    )

    assert fac["trend_snapshot"] == {
        "tf": "H4",
        "gap_pips": 20.0,
        "direction": "long",
        "adx": 35.0,
    }
    assert fac["mtf"] == {
        "candles_m5": [{"close": 158.28}],
        "candles_h1": [{"close": 158.12}],
        "candles_h4": [{"close": 157.98}],
    }


def test_resolve_account_snapshot_uses_last_snapshot_fallback(monkeypatch):
    cached = object()

    def _boom(**_kwargs):
        raise RuntimeError("summary 503")

    monkeypatch.setattr(worker, "get_account_snapshot", _boom)

    snap, unavailable, err = worker._resolve_account_snapshot(cached)

    assert snap is cached
    assert unavailable is False
    assert isinstance(err, RuntimeError)


def test_resolve_account_snapshot_marks_unavailable_without_cache(monkeypatch):
    def _boom(**_kwargs):
        raise RuntimeError("summary 503")

    monkeypatch.setattr(worker, "get_account_snapshot", _boom)

    snap, unavailable, err = worker._resolve_account_snapshot(None)

    assert snap is None
    assert unavailable is True
    assert isinstance(err, RuntimeError)


def test_clamp_dynamic_alloc_multiplier_when_recent_history_is_underperforming(monkeypatch):
    monkeypatch.setattr(worker.config, "HIST_REGIME_MIN_TRADES", 8)
    monkeypatch.setattr(worker.config, "HIST_MIN_TRADES", 12)

    dyn_mult, meta = worker._clamp_dynamic_alloc_multiplier(
        1.65,
        hist_profile={
            "source": "regime",
            "n": 8,
            "pf": 0.928,
            "lot_multiplier": 0.625,
        },
    )

    assert dyn_mult == 1.0
    assert meta["reason"] == "history_underperforming"
    assert meta["dyn_mult_before"] == 1.65


def test_clamp_dynamic_alloc_multiplier_keeps_boost_when_history_is_healthy(monkeypatch):
    monkeypatch.setattr(worker.config, "HIST_REGIME_MIN_TRADES", 8)
    monkeypatch.setattr(worker.config, "HIST_MIN_TRADES", 12)

    dyn_mult, meta = worker._clamp_dynamic_alloc_multiplier(
        1.45,
        hist_profile={
            "source": "regime",
            "n": 9,
            "pf": 1.12,
            "lot_multiplier": 1.08,
        },
    )

    assert dyn_mult == 1.45
    assert meta == {}


def test_clamp_strategy_units_multiplier_skips_positive_boost_for_dyn_alloc_loser(monkeypatch):
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_LOSER_SCORE", 0.28)

    mult, meta = worker._clamp_strategy_units_multiplier(
        1.30,
        dyn_profile={"found": True},
        dyn_mult=0.62,
        dyn_score=0.12,
        dyn_trades=24,
        dyn_clamp_meta={},
    )

    assert mult == 1.0
    assert meta["reason"] == "respect_dynamic_alloc_reduce"
    assert meta["configured_mult"] == 1.3


def test_clamp_strategy_units_multiplier_keeps_positive_boost_for_dyn_alloc_winner(monkeypatch):
    monkeypatch.setattr(worker.config, "DYN_ALLOC_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_MIN_TRADES", 10)
    monkeypatch.setattr(worker.config, "DYN_ALLOC_LOSER_SCORE", 0.28)

    mult, meta = worker._clamp_strategy_units_multiplier(
        1.20,
        dyn_profile={"found": True},
        dyn_mult=1.08,
        dyn_score=0.46,
        dyn_trades=24,
        dyn_clamp_meta={},
    )

    assert mult == 1.2
    assert meta == {}
