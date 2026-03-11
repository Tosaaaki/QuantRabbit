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


def test_build_loser_clusters_prefers_lane_from_entry_thesis() -> None:
    rows = [
        {
            "strategy_tag": "RangeFader",
            "strategy": "RangeFader",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.4,
            "realized_pl": -120.0,
            "entry_thesis": '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-buy-fade","rsi": 38.0, "adx": 18.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_tag": "RangeFader",
            "strategy": "RangeFader",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.2,
            "realized_pl": -110.0,
            "entry_thesis": '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-buy-fade","rsi": 39.0, "adx": 19.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_tag": "RangeFader",
            "strategy": "RangeFader",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.8,
            "realized_pl": -150.0,
            "entry_thesis": '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-buy-fade","rsi": 37.0, "adx": 20.0, "range_score": 0.48, "spread_pips": 0.7}',
        },
        {
            "strategy_tag": "RangeFader",
            "strategy": "RangeFader",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -0.9,
            "realized_pl": -90.0,
            "entry_thesis": '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-buy-fade","rsi": 36.0, "adx": 18.5, "range_score": 0.48, "spread_pips": 0.7}',
        },
    ]

    payload = loser_cluster_worker.build_loser_clusters(rows, min_cluster_size=4, top_k=3)

    assert "RangeFader-buy-fade" in payload["strategies"]
    assert "RangeFader" not in payload["strategies"]


def test_build_loser_clusters_exposes_setup_context_when_present() -> None:
    rows = [
        {
            "strategy_key": "RangeFader-sell-fade",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.4,
            "realized_pl": -120.0,
            "entry_thesis": '{"setup_fingerprint":"RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression","flow_regime":"trend_long","microstructure_bucket":"tight_fast","spread_pips":0.7}',
        },
        {
            "strategy_key": "RangeFader-sell-fade",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.2,
            "realized_pl": -110.0,
            "entry_thesis": '{"setup_fingerprint":"RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression","flow_regime":"trend_long","microstructure_bucket":"tight_fast","spread_pips":0.7}',
        },
        {
            "strategy_key": "RangeFader-sell-fade",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -1.8,
            "realized_pl": -150.0,
            "entry_thesis": '{"setup_fingerprint":"RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression","flow_regime":"trend_long","microstructure_bucket":"tight_fast","spread_pips":0.7}',
        },
        {
            "strategy_key": "RangeFader-sell-fade",
            "pocket": "scalp",
            "units": -100,
            "pl_pips": -0.9,
            "realized_pl": -90.0,
            "entry_thesis": '{"setup_fingerprint":"RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression","flow_regime":"trend_long","microstructure_bucket":"tight_fast","spread_pips":0.7}',
        },
    ]

    payload = loser_cluster_worker.build_loser_clusters(rows, min_cluster_size=4, top_k=3)

    cluster = payload["strategies"]["RangeFader-sell-fade"]["clusters"][0]
    assert cluster["setup_context"]["flow_regime"] == "trend_long"
    assert cluster["setup_context"]["microstructure_bucket"] == "tight_fast"
    assert cluster["setup_fingerprint"].startswith("RangeFader-sell-fade|short|trend_long|tight_fast|")
