from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts import benchmark_brain_local_llm as bench


def _init_orders_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE orders (
              id INTEGER PRIMARY KEY,
              ts TEXT,
              pocket TEXT,
              instrument TEXT,
              side TEXT,
              units INTEGER,
              sl_price REAL,
              tp_price REAL,
              client_order_id TEXT,
              status TEXT,
              request_json TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _init_trades_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE trades (
              id INTEGER PRIMARY KEY,
              client_order_id TEXT,
              realized_pl REAL,
              pl_pips REAL,
              close_time TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _init_brain_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE brain_decisions (
              id INTEGER PRIMARY KEY,
              ts TEXT,
              ts_epoch REAL,
              strategy_tag TEXT,
              pocket TEXT,
              side TEXT,
              units INTEGER,
              sl_price REAL,
              tp_price REAL,
              confidence REAL,
              client_order_id TEXT,
              context_json TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _insert_brain_decision(
    brain_db: Path,
    *,
    row_id: int,
    ts: str,
    strategy_tag: str,
    pocket: str,
    side: str,
    units: int,
    client_order_id: str,
) -> None:
    context = {
        "ts": ts,
        "strategy_tag": strategy_tag,
        "pocket": pocket,
        "side": side,
        "units": units,
        "entry_thesis": {"entry_probability": 0.6},
        "meta": {},
    }
    ts_epoch = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    con = sqlite3.connect(brain_db)
    try:
        con.execute(
            """
            INSERT INTO brain_decisions(
              id, ts, ts_epoch, strategy_tag, pocket, side, units, sl_price, tp_price,
              confidence, client_order_id, context_json
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row_id,
                ts,
                ts_epoch,
                strategy_tag,
                pocket,
                side,
                units,
                None,
                None,
                0.7,
                client_order_id,
                json.dumps(context),
            ),
        )
        con.commit()
    finally:
        con.close()


def _insert_order(
    orders_db: Path,
    *,
    row_id: int,
    ts: str,
    client_order_id: str,
    strategy_tag: str = "scalp_ping_5s_b_live",
    side: str = "buy",
) -> None:
    req = {
        "strategy_tag": strategy_tag,
        "entry_thesis": {
            "strategy_tag": strategy_tag,
            "entry_probability": 0.6,
        },
        "meta": {"env_prefix": "SCALP_PING_5S_B"},
    }
    con = sqlite3.connect(orders_db)
    try:
        con.execute(
            """
            INSERT INTO orders(id, ts, pocket, instrument, side, units, sl_price, tp_price,
                               client_order_id, status, request_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row_id,
                ts,
                "scalp_fast",
                "USD_JPY",
                side,
                7,
                156.968,
                None,
                client_order_id,
                "preflight_start",
                json.dumps(req),
            ),
        )
        con.commit()
    finally:
        con.close()


def _insert_trade(trades_db: Path, *, client_order_id: str, realized_pl: float, pl_pips: float) -> None:
    con = sqlite3.connect(trades_db)
    try:
        con.execute(
            """
            INSERT INTO trades(id, client_order_id, realized_pl, pl_pips, close_time)
            VALUES(NULL, ?, ?, ?, ?)
            """,
            (
                client_order_id,
                realized_pl,
                pl_pips,
                "2026-03-05T03:10:00+00:00",
            ),
        )
        con.commit()
    finally:
        con.close()


def test_load_samples_from_orders_parses_entry_thesis(tmp_path: Path) -> None:
    orders_db = tmp_path / "orders.db"
    trades_db = tmp_path / "trades.db"
    _init_orders_db(orders_db)
    _init_trades_db(trades_db)

    req = {
        "strategy_tag": "scalp_ping_5s_b_live",
        "meta": {"env_prefix": "SCALP_PING_5S_B"},
        "entry_thesis": {
            "strategy_tag": "scalp_ping_5s_b_live",
            "entry_probability": 0.67,
            "confidence": 92,
            "spread_pips": 0.8,
        },
    }

    con = sqlite3.connect(orders_db)
    try:
        con.execute(
            """
            INSERT INTO orders(id, ts, pocket, instrument, side, units, sl_price, tp_price,
                               client_order_id, status, request_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                1,
                "2026-03-05T03:02:53.591373+00:00",
                "scalp_fast",
                "USD_JPY",
                "buy",
                7,
                156.968,
                None,
                "cid-1",
                "preflight_start",
                json.dumps(req),
            ),
        )
        con.commit()
    finally:
        con.close()

    outcomes = {}
    samples = bench._load_samples_from_orders(
        orders_db,
        lookback_hours=48.0,
        max_samples=10,
        outcomes=outcomes,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.source == "orders_preflight"
    assert sample.strategy_tag == "scalp_ping_5s_b_live"
    assert sample.pocket == "scalp_fast"
    assert sample.side == "buy"
    assert sample.units == 7
    assert sample.confidence == 0.67
    assert sample.context["entry_thesis"]["spread_pips"] == 0.8


def test_evaluate_variant_reports_parse_and_alignment() -> None:
    samples = [
        bench.Sample(
            sample_id="s1",
            ts="2026-03-05T03:00:00+00:00",
            source="brain_decisions",
            strategy_tag="a",
            pocket="scalp_fast",
            side="buy",
            units=10,
            sl_price=None,
            tp_price=None,
            confidence=0.7,
            client_order_id="cid-pos",
            context={"strategy_tag": "a", "pocket": "scalp_fast", "side": "buy", "units": 10},
            realized_pl=10.0,
            pl_pips=1.2,
        ),
        bench.Sample(
            sample_id="s2",
            ts="2026-03-05T03:00:05+00:00",
            source="brain_decisions",
            strategy_tag="a",
            pocket="scalp_fast",
            side="sell",
            units=12,
            sl_price=None,
            tp_price=None,
            confidence=0.55,
            client_order_id="cid-neg",
            context={"strategy_tag": "a", "pocket": "scalp_fast", "side": "sell", "units": 12},
            realized_pl=-8.0,
            pl_pips=-1.1,
        ),
    ]

    variant = bench.VariantSpec(
        name="test",
        model="dummy",
        url="http://127.0.0.1:11434/api/chat",
        temperature=0.2,
        max_tokens=128,
        timeout_sec=2.0,
    )

    def _fake_caller(prompt: str, **_kwargs):
        if '"side": "buy"' in prompt:
            return bench.CallOutcome(
                payload={"action": "ALLOW", "scale": 1.0, "reason": "ok"},
                raw_content='{"action":"ALLOW","scale":1.0,"reason":"ok"}',
            )
        return bench.CallOutcome(
            payload={"action": "BLOCK", "scale": 0.0, "reason": "avoid"},
            raw_content='{"action":"BLOCK","scale":0.0,"reason":"avoid"}',
        )

    result = bench._evaluate_variant(
        samples=samples,
        variant=variant,
        enable_alignment=True,
        include_sample_details=False,
        caller=_fake_caller,
    )

    assert result["sample_count"] == 2
    assert result["parse"]["pass"] == 2
    assert result["parse"]["fail"] == 0
    assert result["actions"]["ALLOW"] == 1
    assert result["actions"]["BLOCK"] == 1
    assert result["action_mix"]["allow"] == 0.5
    assert result["outcome_alignment"]["scored_trades"] == 2
    assert result["outcome_alignment"]["score_mean"] == 1.0


def test_normalize_decision_rejects_invalid_action() -> None:
    normalized, reason = bench._normalize_decision({"action": "MAYBE", "scale": 0.4})
    assert normalized is None
    assert reason == "invalid_action"


def test_evaluate_variant_reports_caller_fail_reason_and_example() -> None:
    samples = [
        bench.Sample(
            sample_id="s1",
            ts="2026-03-05T03:00:00+00:00",
            source="brain_decisions",
            strategy_tag="a",
            pocket="scalp_fast",
            side="buy",
            units=10,
            sl_price=None,
            tp_price=None,
            confidence=0.7,
            client_order_id="cid-pos",
            context={"strategy_tag": "a", "pocket": "scalp_fast", "side": "buy", "units": 10},
            realized_pl=None,
            pl_pips=None,
        )
    ]
    variant = bench.VariantSpec(
        name="timeout",
        model="dummy",
        url="http://127.0.0.1:11434/api/chat",
        temperature=0.2,
        max_tokens=128,
        timeout_sec=2.0,
    )

    def _fake_caller(_prompt: str, **_kwargs):
        return bench.CallOutcome(payload=None, fail_reason="http_timeout", raw_content="timeout")

    result = bench._evaluate_variant(
        samples=samples,
        variant=variant,
        enable_alignment=False,
        include_sample_details=False,
        caller=_fake_caller,
    )
    assert result["parse"]["pass"] == 0
    assert result["parse"]["fail"] == 1
    assert result["parse"]["fail_reasons"]["http_timeout"] == 1
    assert result["parse"]["fail_examples"]["http_timeout"] == "timeout"


def test_select_samples_prioritize_outcomes_and_reason(tmp_path: Path) -> None:
    orders_db = tmp_path / "orders.db"
    trades_db = tmp_path / "trades.db"
    brain_db = tmp_path / "brain.db"
    _init_orders_db(orders_db)
    _init_trades_db(trades_db)

    _insert_order(
        orders_db,
        row_id=1,
        ts="2026-03-05T03:00:00+00:00",
        client_order_id="cid-old-no",
    )
    _insert_order(
        orders_db,
        row_id=2,
        ts="2026-03-05T03:01:00+00:00",
        client_order_id="cid-hit",
    )
    _insert_order(
        orders_db,
        row_id=3,
        ts="2026-03-05T03:02:00+00:00",
        client_order_id="cid-new-no",
    )
    _insert_trade(trades_db, client_order_id="cid-hit", realized_pl=12.5, pl_pips=1.1)

    _source, samples, load_meta = bench._select_samples(
        source="orders",
        brain_db=brain_db,
        orders_db=orders_db,
        trades_db=trades_db,
        lookback_hours=72.0,
        max_samples=2,
        sample_mode="recent",
        seed=7,
        outcome_sample_policy="prioritize",
    )

    assert len(samples) == 2
    assert samples[0].client_order_id == "cid-hit"
    outcome_meta = load_meta["outcome_policy"]
    assert outcome_meta["selected_with_trade_outcome"] == 1
    assert outcome_meta["selected_without_trade_outcome"] == 1
    assert outcome_meta["insufficient_data_reason"] == "insufficient_realized_outcome_samples:1/2"


def test_select_samples_require_outcomes_no_data_reason(tmp_path: Path) -> None:
    orders_db = tmp_path / "orders.db"
    trades_db = tmp_path / "trades.db"
    brain_db = tmp_path / "brain.db"
    _init_orders_db(orders_db)
    _init_trades_db(trades_db)
    _insert_order(
        orders_db,
        row_id=1,
        ts="2026-03-05T03:00:00+00:00",
        client_order_id="cid-only-no-outcome",
    )

    _source, samples, load_meta = bench._select_samples(
        source="orders",
        brain_db=brain_db,
        orders_db=orders_db,
        trades_db=trades_db,
        lookback_hours=72.0,
        max_samples=2,
        sample_mode="recent",
        seed=7,
        outcome_sample_policy="require",
    )

    assert samples == []
    outcome_meta = load_meta["outcome_policy"]
    assert outcome_meta["selected_total_samples"] == 0
    assert outcome_meta["insufficient_data_reason"] == "outcome_required_but_no_realized_outcome_samples"


def test_rank_variants_uses_outcome_score_only_when_enough_samples() -> None:
    ranking = bench._rank_variants(
        [
            {
                "variant": {"name": "A", "model": "m"},
                "parse": {"pass_rate": 0.8},
                "latency_ms_parse_pass": {"p95": 10.0},
                "outcome_alignment": {"score_mean": 1.0, "scored_trades": 5},
            },
            {
                "variant": {"name": "B", "model": "m"},
                "parse": {"pass_rate": 0.85},
                "latency_ms_parse_pass": {"p95": 8.0},
                "outcome_alignment": {"score_mean": 0.2, "scored_trades": 1},
            },
        ],
        min_outcome_samples=3,
    )

    assert ranking[0]["name"] == "A"
    assert ranking[0]["outcome_score_used"] is True
    assert ranking[0]["outcome_score"] == 1.0
    assert ranking[1]["outcome_score_used"] is False
    assert ranking[1]["outcome_score_reason"] == "insufficient_scored_trades:1<3"


def test_auto_source_prefers_higher_outcome_coverage(tmp_path: Path) -> None:
    orders_db = tmp_path / "orders.db"
    trades_db = tmp_path / "trades.db"
    brain_db = tmp_path / "brain.db"
    _init_orders_db(orders_db)
    _init_trades_db(trades_db)
    _init_brain_db(brain_db)

    _insert_order(
        orders_db,
        row_id=1,
        ts="2026-03-05T03:00:00+00:00",
        client_order_id="cid-order-1",
    )
    _insert_order(
        orders_db,
        row_id=2,
        ts="2026-03-05T03:01:00+00:00",
        client_order_id="cid-order-2",
    )
    _insert_trade(trades_db, client_order_id="cid-order-1", realized_pl=5.0, pl_pips=0.5)
    _insert_trade(trades_db, client_order_id="cid-order-2", realized_pl=8.0, pl_pips=0.9)

    _insert_brain_decision(
        brain_db,
        row_id=1,
        ts="2026-03-05T03:00:00+00:00",
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=3,
        client_order_id="cid-brain-1",
    )
    _insert_trade(trades_db, client_order_id="cid-brain-1", realized_pl=4.0, pl_pips=0.4)

    selected_source, samples, load_meta = bench._select_samples(
        source="auto",
        brain_db=brain_db,
        orders_db=orders_db,
        trades_db=trades_db,
        lookback_hours=72.0,
        max_samples=2,
        sample_mode="recent",
        seed=42,
        outcome_sample_policy="prioritize",
    )

    assert selected_source == "orders"
    assert len(samples) == 2
    assert load_meta["auto_source_reason"] == "orders_higher_outcome_coverage"
    assert load_meta["order_outcome_samples"] == 2
    assert load_meta["brain_outcome_samples"] == 1


def test_rank_variants_outcome_score_scales_by_coverage() -> None:
    ranking = bench._rank_variants(
        [
            {
                "variant": {"name": "A", "model": "m"},
                "parse": {"pass_rate": 0.8},
                "latency_ms_parse_pass": {"p95": 10.0},
                "sample_count": 10,
                "outcome_alignment": {"score_mean": 1.0, "scored_trades": 5},
            },
            {
                "variant": {"name": "B", "model": "m"},
                "parse": {"pass_rate": 0.8},
                "latency_ms_parse_pass": {"p95": 12.0},
                "sample_count": 10,
                "outcome_alignment": {"score_mean": 1.0, "scored_trades": 10},
            },
        ],
        min_outcome_samples=3,
    )

    top = ranking[0]
    assert top["name"] == "B"
    assert top["alignment_coverage"] == 1.0
    second = ranking[1]
    assert second["name"] == "A"
    assert second["alignment_coverage"] == 0.5
    assert second["outcome_score"] == 0.5
