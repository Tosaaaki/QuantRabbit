from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
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


def test_load_trade_rows_from_replay_json(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "trade_id": "sim-1",
                        "strategy_tag": "scalp_ping_5s_b_live",
                        "entry_time": now.isoformat(),
                        "exit_time": (now + timedelta(seconds=160)).isoformat(),
                        "units": 1200,
                        "pnl_pips": -0.7,
                        "reason": "time_stop",
                    },
                    {
                        "trade_id": "sim-2",
                        "strategy_tag": "other_strategy",
                        "entry_time": now.isoformat(),
                        "exit_time": (now + timedelta(seconds=90)).isoformat(),
                        "units": -1000,
                        "pnl_pips": 0.5,
                        "reason": "tp_hit",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(
        tmp_path,
        include_live_trades=False,
        replay_json_globs=(str(replay_path),),
        lookback_days=5,
    )
    rows = worker._load_trade_rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row.source == "replay"
    assert row.strategy_tag == "scalp_ping_5s_b_live"
    assert row.reason == "time_stop"
    assert row.hold_sec is not None and row.hold_sec >= 150.0


def test_recommendations_include_stuck_block_signal(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        min_samples=6,
        min_fold_samples=1,
        min_fold_consistency=0.5,
        oos_enabled=False,
        stuck_hold_sec=120.0,
        stuck_loss_pips=-0.2,
        block_stuck_rate=0.5,
    )
    rows = [
        worker.TradeSample(
            ticket_id=f"s{i}",
            client_order_id=f"cid-s{i}",
            strategy_tag="scalp_ping_5s_b_live",
            side="long",
            hour_jst=3,
            day_jst=f"2026-02-0{1 + (i % 4)}",
            pl_pips=-0.6,
            entry_probability=0.52,
            spread_pips=0.9,
            reason="time_stop",
            hold_sec=180.0,
            source="replay",
        )
        for i in range(8)
    ]
    recs = worker._build_recommendations(rows, cfg)
    stuck_rec = next((rec for rec in recs if rec["feature"] == "stuck" and rec["bucket"] == "stuck"), None)
    assert stuck_rec is not None
    assert stuck_rec["action"] == "block"
    assert float(stuck_rec["stuck_rate"]) >= 0.5


def test_main_skips_when_market_open(monkeypatch) -> None:
    monkeypatch.setenv("COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN", "1")
    monkeypatch.setattr(worker, "is_market_open", lambda: True)

    def _fail_parse_args():
        raise AssertionError("parse_args should not be called when market-open skip is active")

    monkeypatch.setattr(worker, "parse_args", _fail_parse_args)
    assert worker.main() == 0
