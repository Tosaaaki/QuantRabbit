from __future__ import annotations

from pathlib import Path

from analysis import trade_counterfactual_worker as worker


def _cfg(tmp_path: Path, **kwargs: object) -> worker.ReviewConfig:
    base = worker.ReviewConfig(
        trades_db=tmp_path / "trades.db",
        orders_db=tmp_path / "orders.db",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
        strategy_like="scalp_ping_5s_b_live%",
        lookback_days=14,
        min_samples=6,
        fold_count=5,
        min_fold_samples=1,
        min_fold_consistency=0.6,
        block_lb_pips=-0.25,
        boost_lb_pips=0.2,
        reduce_factor=0.5,
        boost_factor=0.3,
        jst_offset_hours=9,
        top_k=20,
        oos_enabled=True,
        oos_min_folds=2,
        oos_min_action_match_ratio=0.6,
        oos_min_positive_ratio=0.6,
        oos_min_lb_uplift_pips=0.0,
    )
    return worker.ReviewConfig(**{**base.__dict__, **kwargs})


def _sample(day: str, pl: float, *, ticket: str) -> worker.TradeSample:
    return worker.TradeSample(
        ticket_id=ticket,
        client_order_id=f"cid-{ticket}",
        strategy_tag="scalp_ping_5s_b_live",
        side="short",
        hour_jst=23,
        day_jst=day,
        pl_pips=pl,
        entry_probability=0.55,
        spread_pips=1.35,
    )


def test_normalize_probability() -> None:
    assert worker._normalize_probability(0.73) == 0.73
    assert worker._normalize_probability(73) == 0.73
    assert worker._normalize_probability(-1) == 0.0
    assert worker._normalize_probability(None) is None
    assert worker._normalize_probability("x") is None


def test_recommendations_block_when_negative_and_consistent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, min_fold_consistency=0.6)
    rows = [
        _sample("2026-02-01", -1.1, ticket="a1"),
        _sample("2026-02-01", -0.8, ticket="a2"),
        _sample("2026-02-02", -0.9, ticket="a3"),
        _sample("2026-02-02", -1.0, ticket="a4"),
        _sample("2026-02-03", -0.7, ticket="a5"),
        _sample("2026-02-04", -0.6, ticket="a6"),
        _sample("2026-02-05", -0.8, ticket="a7"),
    ]

    recs = worker._build_recommendations(rows, cfg)
    assert recs
    block = next((r for r in recs if r["feature"] == "side" and r["bucket"] == "short"), None)
    assert block is not None
    assert block["action"] == "block"
    assert float(block["expected_uplift_pips"]) > 0.0
    assert float(block["certainty"]) > 0.0


def test_recommendations_skip_low_fold_consistency(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, min_fold_consistency=0.95)
    rows = [
        _sample("2026-02-01", -1.0, ticket="b1"),
        _sample("2026-02-01", +1.0, ticket="b2"),
        _sample("2026-02-02", -1.0, ticket="b3"),
        _sample("2026-02-02", +1.0, ticket="b4"),
        _sample("2026-02-03", -1.0, ticket="b5"),
        _sample("2026-02-03", +1.0, ticket="b6"),
        _sample("2026-02-04", -1.0, ticket="b7"),
        _sample("2026-02-04", +1.0, ticket="b8"),
    ]

    recs = worker._build_recommendations(rows, cfg)
    assert recs == []


def test_recommendations_skip_when_oos_positive_ratio_low(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        min_samples=4,
        fold_count=4,
        min_fold_samples=1,
        min_fold_consistency=0.5,
        oos_enabled=True,
        oos_min_folds=3,
        oos_min_action_match_ratio=0.5,
        oos_min_positive_ratio=0.8,
        oos_min_lb_uplift_pips=0.0,
    )
    rows = [
        _sample("2026-02-01", -1.2, ticket="c1"),
        _sample("2026-02-01", -0.8, ticket="c2"),
        _sample("2026-02-02", -1.1, ticket="c3"),
        _sample("2026-02-02", -0.9, ticket="c4"),
        _sample("2026-02-03", +1.4, ticket="c5"),
        _sample("2026-02-03", +1.1, ticket="c6"),
        _sample("2026-02-04", -1.0, ticket="c7"),
        _sample("2026-02-04", +0.9, ticket="c8"),
    ]

    recs = worker._build_recommendations(rows, cfg)
    assert recs == []
