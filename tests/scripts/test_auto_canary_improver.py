from __future__ import annotations

from scripts import auto_canary_improver


def test_build_auto_canary_reinforces_loser_cluster_with_counterfactual_reduce() -> None:
    loser_cluster = {
        "strategies": {
            "MomentumBurst": {
                "worst_severity": 0.72,
                "suggestion": {
                    "units_multiplier": 0.9,
                    "probability_offset": -0.02,
                },
            }
        }
    }
    replay_quality_gate = {"gate_status": "ok"}
    trade_counterfactual = {
        "strategy_like": "MomentumBurst%",
        "recommendations": [
            {
                "action": "reduce",
                "noise_lcb_uplift_pips": 12.5,
            }
        ],
    }

    payload = auto_canary_improver.build_auto_canary(
        loser_cluster,
        replay_quality_gate,
        trade_counterfactual,
        min_confidence=0.5,
    )

    rec = payload["strategies"]["MomentumBurst"]
    assert rec["enabled"] is True
    assert rec["units_multiplier"] < 0.9
    assert rec["probability_offset"] < 0.0
    assert "counterfactual_reduce" in rec["reasons"]


def test_build_auto_canary_emits_setup_overrides_when_cluster_has_live_setup() -> None:
    loser_cluster = {
        "strategies": {
            "RangeFader-sell-fade": {
                "worst_severity": 0.74,
                "suggestion": {
                    "units_multiplier": 0.91,
                    "probability_offset": -0.018,
                },
                "clusters": [
                    {
                        "cluster_key": "sell-fade-tight-fast",
                        "samples": 23,
                        "severity": 0.78,
                        "setup_context": {
                            "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
                            "flow_regime": "trend_long",
                            "microstructure_bucket": "tight_fast",
                        },
                        "suggestion": {
                            "units_multiplier": 0.84,
                            "probability_offset": -0.024,
                        },
                    }
                ],
            }
        }
    }

    payload = auto_canary_improver.build_auto_canary(
        loser_cluster,
        {"gate_status": "ok"},
        {"strategy_like": "", "recommendations": []},
        min_confidence=0.5,
    )

    rec = payload["strategies"]["RangeFader-sell-fade"]
    assert rec["enabled"] is True
    assert isinstance(rec.get("setup_overrides"), list)
    override = rec["setup_overrides"][0]
    assert override["match_dimension"] == "setup_fingerprint"
    assert override["setup_fingerprint"].startswith("RangeFader-sell-fade|short|trend_long|tight_fast|")
    assert override["units_multiplier"] == 0.84
