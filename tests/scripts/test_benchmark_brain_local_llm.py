from __future__ import annotations

import json
import sqlite3
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
