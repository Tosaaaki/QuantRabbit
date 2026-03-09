from __future__ import annotations

import os

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
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_cooldown_active("MicroLevelReactor", 120.0) is True
    assert worker._strategy_cooldown_active("MicroLevelReactor", 146.0) is False


def test_strategy_cooldown_disabled_by_default(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 0.0)
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_cooldown_active("MicroLevelReactor", 110.0) is False


def test_momentumburst_entry_thesis_marks_only_reaccel_signals() -> None:
    reaccel_signal = {"metadata": {"momentum_burst": {"reaccel": True}}}
    normal_signal = {"metadata": {"momentum_burst": {"reaccel": False}}}

    assert worker._momentumburst_entry_thesis_reaccel("MomentumBurst", reaccel_signal) is True
    assert worker._momentumburst_entry_thesis_reaccel("MomentumBurst", normal_signal) is False
    assert worker._momentumburst_entry_thesis_reaccel("MicroLevelReactor", reaccel_signal) is False


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
