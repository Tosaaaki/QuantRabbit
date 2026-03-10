from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from analysis import strategy_feedback_worker as feedback_worker
from scripts import publish_health_snapshot as snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _seed_entry_intent_board(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE entry_intent_board (
                ts TEXT,
                ts_epoch REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO entry_intent_board(ts, ts_epoch) VALUES (?, strftime('%s','now'))",
            ("2026-03-10T00:00:00Z",),
        )
        conn.commit()


def _seed_health_artifacts(project_root: Path, *, include_feedback_tag: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        project_root / "config" / "dynamic_alloc.json",
        {"as_of": now, "strategies": {"MicroTrendRetest": {"lot_multiplier": 0.75}}},
    )
    _write_json(
        project_root / "config" / "pattern_book.json",
        {"updated_at": now, "patterns": {"sample": {"quality": "avoid"}}},
    )
    _write_json(
        project_root / "logs" / "forecast_improvement_latest.json",
        {"generated_at": now, "verdict": "mixed", "runtime_overrides": {"enabled": True}},
    )
    _write_json(
        project_root / "logs" / "entry_path_summary_latest.json",
        {
            "generated_at": now,
            "orders_considered": 24,
            "strategies": {"MicroTrendRetest": {"attempts": 12, "fills": 3}},
        },
    )
    _write_json(
        project_root / "config" / "participation_alloc.json",
        {"as_of": now, "strategies": {"MicroTrendRetest": {"lot_multiplier": 0.92}}},
    )
    _write_json(
        project_root / "logs" / "loser_cluster_latest.json",
        {
            "generated_at": now,
            "strategies": {"MicroTrendRetest": {"cluster_count": 1}},
            "top_clusters": [{"strategy_tag": "MicroTrendRetest"}],
        },
    )
    _write_json(
        project_root / "config" / "auto_canary_overrides.json",
        {
            "generated_at": now,
            "strategies": {"MicroTrendRetest": {"enabled": True, "units_multiplier": 0.9}},
        },
    )
    _write_json(
        project_root / "logs" / "macro_news_context.json",
        {
            "generated_at": now,
            "event_severity": "low",
            "caution_window_active": False,
            "usd_jpy_bias": "neutral",
            "headlines": [],
            "sources": ["market_context_latest"],
        },
    )
    strategies = {"MomentumBurst": {"entry_units_multiplier": 0.9}}
    if include_feedback_tag:
        strategies["MicroTrendRetest"] = {"entry_units_multiplier": 0.75}
    _write_json(
        project_root / "logs" / "strategy_feedback.json",
        {"updated_at": now, "strategies": strategies},
    )
    for rel in (
        "ops/env/quant-v2-runtime.env",
        "ops/env/quant-strategy-feedback.env",
        "ops/env/local-v2-stack.env",
    ):
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "quant-strategy-feedback.env":
            path.write_text(
                "STRATEGY_FEEDBACK_LOOKBACK_DAYS=3\nSTRATEGY_FEEDBACK_MIN_TRADES=12\n",
                encoding="utf-8",
            )
        else:
            path.write_text("", encoding="utf-8")


def _mock_feedback_discovery(monkeypatch) -> None:
    record = feedback_worker.StrategyRecord(
        canonical_tag="MicroTrendRetest",
        active=True,
        entry_active=True,
        exit_active=False,
    )
    stats = feedback_worker.StrategyStats(
        tag="MicroTrendRetest",
        trades=19,
        wins=3,
        losses=16,
        sum_pips=-56.3,
        avg_pips=-2.96,
        avg_abs_pips=4.11,
        gross_win=10.9,
        gross_loss=67.2,
        avg_hold_sec=39.0,
        last_closed="2026-03-10T00:00:00+00:00",
    )
    monkeypatch.setattr(feedback_worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(feedback_worker, "_local_stack_running_services", lambda _pid_dir: set())
    monkeypatch.setattr(feedback_worker, "_discover_from_control", lambda: {})
    monkeypatch.setattr(
        feedback_worker,
        "_discover_from_systemd",
        lambda _systemd_dir, _running, _now: {"MicroTrendRetest": record},
    )
    monkeypatch.setattr(
        feedback_worker,
        "_discover_from_trades",
        lambda _db_path, _lookback_days: ({"MicroTrendRetest": stats}, {"MicroTrendRetest": stats.last_closed or ""}),
    )
    monkeypatch.setattr(
        feedback_worker,
        "_remap_stats_to_known_keys",
        lambda stats_by_tag, latest_by_tag, _known_keys: (stats_by_tag, latest_by_tag),
    )


def test_build_mechanism_integrity_flags_missing_strategy_feedback_coverage(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path
    logs_dir = project_root / "logs"
    orders_db = logs_dir / "orders.db"
    trades_db = logs_dir / "trades.db"

    _seed_health_artifacts(project_root, include_feedback_tag=False)
    _seed_entry_intent_board(orders_db)
    trades_db.touch()
    _mock_feedback_discovery(monkeypatch)
    monkeypatch.setattr(snapshot, "_port_listening", lambda _port: True)
    monkeypatch.setattr(snapshot, "_http_json", lambda _url, timeout_sec=1.5: {"ok": True})

    integrity = snapshot._build_mechanism_integrity(
        project_root=project_root,
        logs_dir=logs_dir,
        orders_db=orders_db,
        trades_db=trades_db,
    )

    assert integrity["ok"] is False
    assert "strategy_feedback_coverage_gap" in integrity["missing_mechanisms"]
    assert integrity["strategy_feedback"]["eligible_missing_strategies"] == ["MicroTrendRetest"]
    assert integrity["blackboard"]["entry_intent_board_table"] is True


def test_build_mechanism_integrity_is_ok_when_artifacts_and_feedback_are_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    logs_dir = project_root / "logs"
    orders_db = logs_dir / "orders.db"
    trades_db = logs_dir / "trades.db"

    _seed_health_artifacts(project_root, include_feedback_tag=True)
    _seed_entry_intent_board(orders_db)
    trades_db.touch()
    _mock_feedback_discovery(monkeypatch)
    monkeypatch.setattr(snapshot, "_port_listening", lambda _port: True)
    monkeypatch.setattr(snapshot, "_http_json", lambda _url, timeout_sec=1.5: {"ok": True})

    integrity = snapshot._build_mechanism_integrity(
        project_root=project_root,
        logs_dir=logs_dir,
        orders_db=orders_db,
        trades_db=trades_db,
    )

    assert integrity["ok"] is True
    assert integrity["missing_mechanisms"] == []
    assert integrity["strategy_feedback"]["eligible_missing_strategies"] == []
    assert integrity["dynamic_alloc"]["fresh"] is True
    assert integrity["forecast_service"]["health"] == {"ok": True}
    assert integrity["entry_path_summary"]["fresh"] is True
    assert integrity["participation_alloc"]["fresh"] is True
    assert integrity["loser_cluster"]["fresh"] is True
    assert integrity["auto_canary"]["fresh"] is True
    assert integrity["macro_news_context"]["fresh"] is True


def test_build_mechanism_integrity_accepts_forecast_health_when_port_probe_misses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    logs_dir = project_root / "logs"
    orders_db = logs_dir / "orders.db"
    trades_db = logs_dir / "trades.db"

    _seed_health_artifacts(project_root, include_feedback_tag=True)
    _seed_entry_intent_board(orders_db)
    trades_db.touch()
    _mock_feedback_discovery(monkeypatch)
    monkeypatch.setattr(snapshot, "_port_listening", lambda _port: False)
    monkeypatch.setattr(snapshot, "_http_json", lambda _url, timeout_sec=1.5: {"ok": True, "service": "quant-forecast"})

    integrity = snapshot._build_mechanism_integrity(
        project_root=project_root,
        logs_dir=logs_dir,
        orders_db=orders_db,
        trades_db=trades_db,
    )

    assert integrity["forecast_service"]["ok"] is True
    assert "forecast_service_down" not in integrity["missing_mechanisms"]


def test_build_mechanism_integrity_flags_missing_blackboard_when_orders_db_absent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    logs_dir = project_root / "logs"
    orders_db = logs_dir / "orders.db"
    trades_db = logs_dir / "trades.db"

    _seed_health_artifacts(project_root, include_feedback_tag=True)
    trades_db.touch()
    _mock_feedback_discovery(monkeypatch)
    monkeypatch.setattr(snapshot, "_port_listening", lambda _port: True)
    monkeypatch.setattr(snapshot, "_http_json", lambda _url, timeout_sec=1.5: {"ok": True})

    integrity = snapshot._build_mechanism_integrity(
        project_root=project_root,
        logs_dir=logs_dir,
        orders_db=orders_db,
        trades_db=trades_db,
    )

    assert integrity["ok"] is False
    assert integrity["blackboard"]["entry_intent_board_table"] is None
    assert "entry_intent_board_missing" in integrity["missing_mechanisms"]
