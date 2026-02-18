from __future__ import annotations

from workers.common import forecast_gate
from workers.forecast import worker


def test_serialize_decision_includes_rebound_and_target_prob() -> None:
    decision = forecast_gate.ForecastDecision(
        allowed=True,
        scale=0.92,
        reason="edge_scale",
        horizon="5m",
        edge=0.61,
        p_up=0.64,
        rebound_probability=0.71,
        target_reach_prob=0.56,
    )
    payload = worker._serialize_decision(decision)
    assert payload["rebound_probability"] == 0.71
    assert payload["target_reach_prob"] == 0.56

