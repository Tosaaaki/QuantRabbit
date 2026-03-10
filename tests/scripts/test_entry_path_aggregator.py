from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import entry_path_aggregator


def _seed_orders_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                pocket TEXT,
                status TEXT,
                request_json TEXT
            )
            """
        )
        rows = [
            (
                "2026-03-10 00:00:01",
                "micro",
                "preflight_start",
                json.dumps(
                    {
                        "strategy_tag": "MomentumBurst",
                        "entry_thesis": {
                            "strategy_tag": "MomentumBurst",
                            "entry_path_attribution": [
                                {"stage": "technical_context", "status": "pass"},
                                {"stage": "forecast_fusion", "status": "reduce", "reason": "soft_contra"},
                            ],
                        },
                    }
                ),
            ),
            (
                "2026-03-10 00:00:02",
                "micro",
                "filled",
                json.dumps(
                    {
                        "strategy_tag": "MomentumBurst",
                        "entry_thesis": {"strategy_tag": "MomentumBurst"},
                    }
                ),
            ),
            (
                "2026-03-10 00:00:03",
                "scalp",
                "preflight_start",
                json.dumps(
                    {
                        "strategy_tag": "RangeFader-neutral-fade",
                        "entry_thesis": {
                            "strategy_tag": "RangeFader-neutral-fade",
                            "entry_path_attribution": [
                                {"stage": "technical_context", "status": "pass"},
                                {"stage": "entry_net_edge_gate", "status": "block", "reason": "negative"},
                            ],
                        },
                    }
                ),
            ),
            (
                "2026-03-10 00:00:04",
                "scalp",
                "perf_block",
                json.dumps(
                    {
                        "strategy_tag": "RangeFader-neutral-fade",
                        "entry_thesis": {"strategy_tag": "RangeFader-neutral-fade"},
                    }
                ),
            ),
            (
                "2026-03-10 00:00:05",
                "scalp",
                "close_ok",
                json.dumps(
                    {
                        "strategy_tag": "RangeFader-neutral-fade",
                        "entry_thesis": {"strategy_tag": "RangeFader-neutral-fade"},
                    }
                ),
            ),
        ]
        conn.executemany(
            "INSERT INTO orders(ts, pocket, status, request_json) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def test_build_report_aggregates_entry_statuses_and_shares(tmp_path: Path) -> None:
    db_path = tmp_path / "orders.db"
    _seed_orders_db(db_path)

    payload = entry_path_aggregator.build_report(
        db_path,
        lookback_hours=24,
        limit=100,
        top_k=4,
    )

    assert payload["orders_considered"] == 4
    assert payload["strategies_count"] == 2
    momentum = payload["strategies"]["MomentumBurst"]
    assert momentum["attempts"] == 1
    assert momentum["fills"] == 1
    assert momentum["filled_rate"] == 1.0
    assert momentum["fill_share"] == 1.0

    range_fader = payload["strategies"]["RangeFader"]
    assert range_fader["attempts"] == 1
    assert range_fader["fills"] == 0
    assert range_fader["terminal_status_counts"]["perf_block"] == 1
    assert range_fader["hard_blocks"] >= 1
    assert any(item["key"] == "entry_net_edge_gate:negative" for item in range_fader["top_blockers"])
