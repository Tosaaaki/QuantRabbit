from __future__ import annotations

import asyncio
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.strategy_entry as strategy_entry


def _set_default_fusion_knobs(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_FORECAST_FUSION_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_UNITS_CUT_MAX",
        0.65,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_UNITS_BOOST_MAX",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_UNITS_MIN_SCALE",
        0.25,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_UNITS_MAX_SCALE",
        1.20,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_DISALLOW_UNITS_MULT",
        0.65,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_PROB_GAIN",
        0.22,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_DISALLOW_PROB_MULT",
        0.70,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_SYNTH_PROB_IF_MISSING",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_TP_BLEND_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_TP_BLEND_BASE",
        0.20,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_TP_BLEND_EDGE_GAIN",
        0.40,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_TF_CUT_MAX",
        0.35,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_TF_BOOST_MAX",
        0.12,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_PROB_MAX",
        0.22,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_EDGE_MIN",
        0.65,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_WEAK_CONTRA_PROB_MAX",
        0.50,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_WEAK_CONTRA_EDGE_MAX",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_UNITS_BOOST_MAX",
        0.18,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_UNITS_CUT_MAX",
        0.30,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_PROB_GAIN",
        0.10,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_STRONG_CONTRA",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_PROB_MIN",
        0.82,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_DIR_PROB_MAX",
        0.18,
        raising=False,
    )


def test_inject_entry_forecast_context_keeps_allow_decision(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_FORECAST_CONTEXT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "_build_entry_forecast_profile",
        lambda *args, **kwargs: {},
        raising=False,
    )
    decision = strategy_entry.forecast_gate.ForecastDecision(
        allowed=True,
        scale=1.0,
        reason="edge_allow",
        horizon="1m",
        edge=0.81,
        p_up=0.81,
        expected_pips=1.4,
        source="technical",
    )
    monkeypatch.setattr(strategy_entry.forecast_gate, "decide", lambda **_: decision)

    thesis, context = strategy_entry._inject_entry_forecast_context(
        instrument="USD_JPY",
        strategy_tag="scalp_ping_5s_c_live",
        pocket="scalp_fast",
        units=1200,
        entry_thesis={},
        meta=None,
    )
    assert context is not None
    assert context.get("reason") == "edge_allow"
    assert context.get("p_up") == 0.81
    assert isinstance(thesis, dict)
    assert isinstance(thesis.get("forecast"), dict)
    assert thesis["forecast"].get("reason") == "edge_allow"
    assert thesis["forecast"].get("p_up") == 0.81


def test_forecast_fusion_scales_down_on_mismatch(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {"tp_pips": 1.5}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="MicroRangeBreak-long",
        pocket="micro",
        units=1000,
        entry_probability=0.62,
        entry_thesis=thesis,
        forecast_context={
            "allowed": False,
            "reason": "style_mismatch_range",
            "p_up": 0.34,
            "edge": 0.44,
            "tp_pips_hint": 2.0,
        },
    )

    assert units < 1000
    assert prob is not None and prob < 0.62
    assert applied.get("forecast_allowed") is False
    # Opposite forecast should not overwrite TP with hint.
    assert thesis.get("tp_pips") == 1.5


def test_forecast_fusion_scales_up_and_sets_tp_hint(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="SCALP_M1SCALPER-long",
        pocket="scalp",
        units=1000,
        entry_probability=0.55,
        entry_thesis=thesis,
        forecast_context={
            "allowed": True,
            "reason": "ok",
            "p_up": 0.82,
            "edge": 0.76,
            "tp_pips_hint": 2.4,
            "horizon": "5m",
        },
    )

    assert units > 1000
    assert prob is not None and prob > 0.55
    assert thesis.get("tp_pips") is not None
    assert float(thesis.get("tp_pips")) > 0.0
    assert applied.get("forecast_horizon") == "5m"


def test_forecast_fusion_synthesizes_probability_when_missing(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {}
    units, prob, _ = strategy_entry._apply_forecast_fusion(
        strategy_tag="SCALP_PING_5S_B-short",
        pocket="scalp_fast",
        units=-800,
        entry_probability=None,
        entry_thesis=thesis,
        forecast_context={
            "allowed": True,
            "p_up": 0.22,
            "edge": 0.66,
        },
    )

    assert units < 0
    assert prob is not None
    assert 0.5 < prob <= 1.0
    assert thesis.get("entry_probability") == prob


def test_forecast_fusion_no_context_keeps_values(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {"tp_pips": 1.2}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="MicroVWAPBound-long",
        pocket="micro",
        units=1200,
        entry_probability=0.58,
        entry_thesis=thesis,
        forecast_context=None,
    )

    assert units == 1200
    assert prob == 0.58
    assert applied == {}
    assert thesis.get("tp_pips") == 1.2


def test_forecast_fusion_rejects_strong_contra(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {"tp_pips": 2.1, "sl_pips": 1.8}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1000,
        entry_probability=0.66,
        entry_thesis=thesis,
        forecast_context={
            "allowed": False,
            "reason": "hard_opposite",
            "p_up": 0.08,
            "edge": 0.93,
            "tp_pips_hint": 3.0,
            "sl_pips_cap": 1.0,
        },
    )

    assert units == 0
    assert prob is not None and 0.0 <= prob <= 0.22
    assert applied.get("strong_contra_reject") is True
    assert applied.get("reject_reason") == "strong_contra_forecast"
    # Strong contra reject keeps TP/SL untouched because order is not sent.
    assert thesis.get("tp_pips") == 2.1
    assert thesis.get("sl_pips") == 1.8


def test_forecast_fusion_rejects_strong_contra_on_bearish_edge(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {"tp_pips": 1.8, "sl_pips": 1.4}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1200,
        entry_probability=0.62,
        entry_thesis=thesis,
        forecast_context={
            "allowed": False,
            "reason": "hard_bearish_flow",
            "p_up": 0.10,
            "edge": 0.08,
            "tp_pips_hint": 2.4,
            "sl_pips_cap": 1.0,
        },
    )

    assert units == 0
    assert prob is not None and 0.0 <= prob <= 0.22
    assert applied.get("strong_contra_reject") is True
    assert applied.get("reject_reason") == "strong_contra_forecast"
    assert thesis.get("tp_pips") == 1.8
    assert thesis.get("sl_pips") == 1.4


def test_forecast_fusion_rejects_weak_contra_on_low_edge_strength(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED",
        True,
        raising=False,
    )
    thesis: dict[str, object] = {"tp_pips": 1.7, "sl_pips": 1.3}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1200,
        entry_probability=0.61,
        entry_thesis=thesis,
        forecast_context={
            "allowed": False,
            "reason": "weak_contra",
            "p_up": 0.44,
            "edge": 0.54,
        },
    )

    assert units == 0
    assert prob is not None and 0.0 <= prob <= 0.44
    assert applied.get("weak_contra_reject") is True
    assert applied.get("strong_contra_reject") is False
    assert applied.get("reject_reason") == "weak_contra_forecast"
    assert thesis.get("tp_pips") == 1.7
    assert thesis.get("sl_pips") == 1.3


def test_forecast_fusion_keeps_high_edge_contra_when_weak_contra_enabled(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    monkeypatch.setattr(
        strategy_entry,
        "_STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED",
        True,
        raising=False,
    )
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1000,
        entry_probability=0.60,
        entry_thesis={},
        forecast_context={
            "allowed": False,
            "reason": "contra_but_high_edge",
            "p_up": 0.44,
            "edge": 0.88,
        },
    )

    assert units > 0
    assert prob is not None and prob > 0.0
    assert applied.get("weak_contra_reject") is False
    assert applied.get("strong_contra_reject") is False


def test_forecast_fusion_rebound_supports_contra_buy(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    units_base, prob_base, _ = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1000,
        entry_probability=0.60,
        entry_thesis={},
        forecast_context={
            "allowed": False,
            "reason": "edge_block",
            "p_up": 0.32,
            "edge": 0.40,
        },
    )
    units_rebound, prob_rebound, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1000,
        entry_probability=0.60,
        entry_thesis={},
        forecast_context={
            "allowed": False,
            "reason": "edge_block",
            "p_up": 0.32,
            "edge": 0.40,
            "rebound_probability": 0.90,
        },
    )

    assert units_rebound > units_base
    assert prob_rebound is not None and prob_base is not None and prob_rebound > prob_base
    assert applied.get("rebound_probability") == 0.9
    assert float(applied.get("rebound_side_support") or 0.0) > 0.0


def test_forecast_fusion_rebound_overrides_strong_contra_for_buy(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    thesis: dict[str, object] = {"tp_pips": 2.1, "sl_pips": 1.8}
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=1000,
        entry_probability=0.66,
        entry_thesis=thesis,
        forecast_context={
            "allowed": False,
            "reason": "hard_opposite",
            "p_up": 0.08,
            "edge": 0.93,
            "rebound_probability": 0.90,
        },
    )

    assert units > 0
    assert prob is not None and prob > 0.0
    assert applied.get("strong_contra_reject") is False
    assert applied.get("rebound_override_strong_contra") is True


def test_forecast_fusion_tf_confluence_cuts_units(monkeypatch) -> None:
    _set_default_fusion_knobs(monkeypatch)
    units, prob, applied = strategy_entry._apply_forecast_fusion(
        strategy_tag="MicroRangeBreak-long",
        pocket="micro",
        units=1000,
        entry_probability=0.60,
        entry_thesis={},
        forecast_context={
            "allowed": True,
            "reason": "mtf_divergence",
            "p_up": 0.62,
            "edge": 0.65,
            "tf_confluence_score": -0.85,
            "tf_confluence_count": 3,
        },
    )

    assert units < 1000
    assert prob is not None and prob < 0.60
    assert applied.get("tf_confluence_score") == -0.85
    assert applied.get("tf_confluence_count") == 3


def test_entry_leading_profile_boosts_probability_with_strategy_prefix(monkeypatch) -> None:
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_ENABLED", "1")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_BOOST_MAX", "0.18")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_PENALTY_MAX", "0.05")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_WEIGHT_FORECAST", "0.5")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_WEIGHT_TECH", "0.3")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_WEIGHT_RANGE", "0.2")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_UNITS_MIN_MULT", "0.90")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_UNITS_MAX_MULT", "1.20")
    thesis: dict[str, object] = {
        "env_prefix": "SCALP_TEST",
        "tech_score": 0.72,
        "range_score": 0.82,
    }

    units, prob, applied = strategy_entry._apply_strategy_leading_profile(
        strategy_tag="scalp_test_live",
        pocket="scalp",
        units=1000,
        entry_probability=0.55,
        entry_thesis=thesis,
        forecast_context={"p_up": 0.84},
        meta={},
    )

    assert units > 1000
    assert prob is not None and prob > 0.55
    assert applied.get("reject") is False
    assert isinstance(thesis.get("entry_probability_leading_profile"), dict)
    assert thesis.get("entry_probability") == prob


def test_entry_leading_profile_rejects_under_strategy_threshold(monkeypatch) -> None:
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_ENABLED", "1")
    monkeypatch.setenv("SCALP_TEST_ENTRY_LEADING_PROFILE_REJECT_BELOW", "0.70")
    thesis: dict[str, object] = {"env_prefix": "SCALP_TEST"}

    units, prob, applied = strategy_entry._apply_strategy_leading_profile(
        strategy_tag="scalp_test_live",
        pocket="scalp",
        units=-900,
        entry_probability=0.62,
        entry_thesis=thesis,
        forecast_context={"p_up": 0.58},
        meta={},
    )

    assert units == 0
    assert prob is not None and prob < 0.70
    assert applied.get("reject") is True
    assert applied.get("reason") == "entry_leading_profile_reject"


def test_entry_leading_profile_skips_manual_pocket(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_ENTRY_LEADING_PROFILE_ENABLED", "1")
    thesis: dict[str, object] = {"env_prefix": "MANUAL_TEST"}

    units, prob, applied = strategy_entry._apply_strategy_leading_profile(
        strategy_tag="manual_test_live",
        pocket="manual",
        units=700,
        entry_probability=0.51,
        entry_thesis=thesis,
        forecast_context={"p_up": 0.99},
        meta={},
    )

    assert units == 700
    assert prob == 0.51
    assert applied == {}


def test_market_order_reject_reason_from_forecast_is_cached(monkeypatch) -> None:
    cached: list[dict[str, object]] = []

    monkeypatch.setattr(
        strategy_entry,
        "_resolve_strategy_tag",
        lambda strategy_tag, client_order_id, entry_thesis: strategy_tag or "scalp_ping_5s_b_live",
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_entry_technical_context",
        lambda **kwargs: kwargs.get("entry_thesis") or {},
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_resolve_entry_probability",
        lambda entry_thesis, confidence: 0.64,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_entry_forecast_context",
        lambda **kwargs: (kwargs.get("entry_thesis") or {}, {"allowed": False, "p_up": 0.08, "edge": 0.92}),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_strategy_feedback",
        lambda *args, **kwargs: (
            kwargs.get("units"),
            kwargs.get("entry_probability"),
            kwargs.get("sl_price"),
            kwargs.get("tp_price"),
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_forecast_fusion",
        lambda **kwargs: (
            0,
            kwargs.get("entry_probability"),
            {"reject_reason": "strong_contra_forecast"},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_env_prefix_context",
        lambda entry_thesis, meta, strategy_tag: (entry_thesis, meta),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_strategy_leading_profile",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("leading profile should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry.order_manager,
        "_cache_order_status",
        lambda **kwargs: cached.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry.order_manager,
        "market_order",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("order dispatch should not run")),
        raising=False,
    )

    result = asyncio.run(
        strategy_entry.market_order(
            instrument="USD_JPY",
            units=120,
            sl_price=150.0,
            tp_price=151.0,
            pocket="scalp",
            client_order_id="test-forecast-reject",
            strategy_tag="scalp_ping_5s_b_live",
            entry_thesis={},
        )
    )

    assert result is None
    assert len(cached) == 1
    assert cached[0]["status"] == "strong_contra_forecast"
    assert cached[0]["side"] == "buy"
    assert cached[0]["units"] == 0


def test_limit_order_reject_reason_from_leading_profile_is_cached(monkeypatch) -> None:
    cached: list[dict[str, object]] = []

    monkeypatch.setattr(
        strategy_entry,
        "_resolve_strategy_tag",
        lambda strategy_tag, client_order_id, entry_thesis: strategy_tag or "scalp_ping_5s_c_live",
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_entry_technical_context",
        lambda **kwargs: kwargs.get("entry_thesis") or {},
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_resolve_entry_probability",
        lambda entry_thesis, confidence: 0.58,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_entry_forecast_context",
        lambda **kwargs: (kwargs.get("entry_thesis") or {}, {"allowed": True, "p_up": 0.62, "edge": 0.71}),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_strategy_feedback",
        lambda *args, **kwargs: (
            kwargs.get("units"),
            kwargs.get("entry_probability"),
            kwargs.get("sl_price"),
            kwargs.get("tp_price"),
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_forecast_fusion",
        lambda **kwargs: (
            kwargs.get("units"),
            kwargs.get("entry_probability"),
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_inject_env_prefix_context",
        lambda entry_thesis, meta, strategy_tag: (entry_thesis, meta),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_apply_strategy_leading_profile",
        lambda **kwargs: (
            0,
            kwargs.get("entry_probability"),
            {"reason": "entry_leading_profile_reject"},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "_coordinate_entry_units",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("coordination should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry.order_manager,
        "_cache_order_status",
        lambda **kwargs: cached.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry.order_manager,
        "limit_order",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("order dispatch should not run")),
        raising=False,
    )

    order_id, rejected_reason = asyncio.run(
        strategy_entry.limit_order(
            instrument="USD_JPY",
            units=-180,
            price=149.8,
            sl_price=150.1,
            tp_price=149.2,
            pocket="scalp",
            client_order_id="test-leading-reject",
            strategy_tag="scalp_ping_5s_c_live",
            entry_thesis={},
        )
    )

    assert order_id is None
    assert rejected_reason is None
    assert len(cached) == 1
    assert cached[0]["status"] == "entry_leading_profile_reject"
    assert cached[0]["side"] == "sell"
    assert cached[0]["units"] == 0
