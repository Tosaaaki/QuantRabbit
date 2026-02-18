from __future__ import annotations

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
