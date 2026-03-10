from __future__ import annotations

from scripts import loser_cluster_worker


def test_build_loser_clusters_extracts_strategy_cluster_and_suggestion() -> None:
    rows = [
        {
            "strategy_key": "MicroTrendRetest-long",
            "pocket": "micro",
            "units": 100,
            "pl_pips": -1.4,
            "realized_pl": -1.1,
            "entry_thesis": '{"rsi": 38.0, "adx": 18.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_key": "MicroTrendRetest-long",
            "pocket": "micro",
            "units": 100,
            "pl_pips": -1.2,
            "realized_pl": -1.0,
            "entry_thesis": '{"rsi": 39.0, "adx": 19.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_key": "MicroTrendRetest-long",
            "pocket": "micro",
            "units": 100,
            "pl_pips": -1.8,
            "realized_pl": -1.5,
            "entry_thesis": '{"rsi": 37.0, "adx": 20.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_key": "MicroTrendRetest-long",
            "pocket": "micro",
            "units": 100,
            "pl_pips": -0.9,
            "realized_pl": -0.7,
            "entry_thesis": '{"rsi": 36.0, "adx": 18.5, "range_score": 0.48, "spread_pips": 0.7}',
        },
    ]

    payload = loser_cluster_worker.build_loser_clusters(rows, min_cluster_size=4, top_k=3)

    strategy = payload["strategies"]["MicroTrendRetest-long"]
    assert strategy["cluster_count"] == 1
    assert strategy["worst_severity"] > 0.0
    assert strategy["suggestion"]["units_multiplier"] < 1.0
    assert strategy["suggestion"]["probability_offset"] < 0.0
