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
