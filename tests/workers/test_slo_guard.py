from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sqlite3


def _init_metrics_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE metrics (
              ts TEXT NOT NULL,
              metric TEXT NOT NULL,
              value REAL NOT NULL,
              tags TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _insert_metric(
    db_path: Path,
    *,
    metric: str,
    value: float,
    tags: str = '{"mode":"strategy_control"}',
) -> None:
    con = sqlite3.connect(db_path)
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        con.execute(
            "INSERT INTO metrics (ts, metric, value, tags) VALUES (?, ?, ?, ?)",
            (ts, metric, float(value), tags),
        )
        con.commit()
    finally:
        con.close()


def _reload_slo_guard(monkeypatch, *, db_path: Path, env: dict[str, str]):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ORDER_SLO_GUARD_DB_PATH", str(db_path))
    import workers.common.slo_guard as slo_guard

    slo_guard = importlib.reload(slo_guard)
    slo_guard._CACHE = None
    return slo_guard


def test_slo_guard_disabled(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    _init_metrics_db(db_path)
    guard = _reload_slo_guard(
        monkeypatch,
        db_path=db_path,
        env={"ORDER_SLO_GUARD_ENABLED": "0"},
    )
    decision = guard.decide(pocket="scalp_fast", strategy_tag="scalp_ping_5s_b_live")
    assert decision.allowed is True
    assert decision.reason == "disabled"


def test_slo_guard_scope_skip(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    _init_metrics_db(db_path)
    guard = _reload_slo_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "ORDER_SLO_GUARD_ENABLED": "1",
            "ORDER_SLO_GUARD_APPLY_POCKETS": "scalp_fast,scalp,micro",
        },
    )
    decision = guard.decide(pocket="macro", strategy_tag="macro_worker")
    assert decision.allowed is True
    assert decision.reason == "scope_skip"


def test_slo_guard_blocks_on_latest_data_lag(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    _init_metrics_db(db_path)
    for value in (1200.0, 1300.0, 7100.0):
        _insert_metric(db_path, metric="data_lag_ms", value=value)
    for value in (800.0, 900.0, 950.0):
        _insert_metric(db_path, metric="decision_latency_ms", value=value)

    guard = _reload_slo_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "ORDER_SLO_GUARD_ENABLED": "1",
            "ORDER_SLO_GUARD_SAMPLE_MIN": "3",
            "ORDER_SLO_GUARD_DATA_LAG_MAX_MS": "1000",
            "ORDER_SLO_GUARD_DECISION_LATENCY_MAX_MS": "4000",
            "ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS": "9000",
            "ORDER_SLO_GUARD_DECISION_LATENCY_P95_MAX_MS": "9000",
        },
    )
    decision = guard.decide(pocket="scalp_fast", strategy_tag="scalp_ping_5s_b_live")
    assert decision.allowed is False
    assert decision.reason == "data_lag_latest_exceeded"
    assert decision.sample >= 3


def test_slo_guard_blocks_on_latency_p95(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    _init_metrics_db(db_path)
    for value in (200.0, 220.0, 210.0, 240.0):
        _insert_metric(db_path, metric="data_lag_ms", value=value)
    for value in (500.0, 700.0, 9000.0, 12000.0):
        _insert_metric(db_path, metric="decision_latency_ms", value=value)

    guard = _reload_slo_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "ORDER_SLO_GUARD_ENABLED": "1",
            "ORDER_SLO_GUARD_SAMPLE_MIN": "4",
            "ORDER_SLO_GUARD_DATA_LAG_MAX_MS": "10000",
            "ORDER_SLO_GUARD_DECISION_LATENCY_MAX_MS": "20000",
            "ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS": "10000",
            "ORDER_SLO_GUARD_DECISION_LATENCY_P95_MAX_MS": "2500",
        },
    )
    decision = guard.decide(pocket="scalp_fast", strategy_tag="scalp_ping_5s_c_live")
    assert decision.allowed is False
    assert decision.reason == "decision_latency_p95_exceeded"
    assert decision.decision_latency_p95_ms is not None
    assert decision.decision_latency_p95_ms > 2500


def test_slo_guard_allows_healthy_window(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    _init_metrics_db(db_path)
    for value in (300.0, 350.0, 420.0, 380.0):
        _insert_metric(db_path, metric="data_lag_ms", value=value)
    for value in (410.0, 500.0, 620.0, 560.0):
        _insert_metric(db_path, metric="decision_latency_ms", value=value)

    guard = _reload_slo_guard(
        monkeypatch,
        db_path=db_path,
        env={
            "ORDER_SLO_GUARD_ENABLED": "1",
            "ORDER_SLO_GUARD_SAMPLE_MIN": "4",
            "ORDER_SLO_GUARD_DATA_LAG_MAX_MS": "1500",
            "ORDER_SLO_GUARD_DECISION_LATENCY_MAX_MS": "1500",
            "ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS": "2000",
            "ORDER_SLO_GUARD_DECISION_LATENCY_P95_MAX_MS": "2000",
        },
    )
    decision = guard.decide(pocket="micro", strategy_tag="MicroRangeBreak")
    assert decision.allowed is True
    assert decision.reason == "healthy"
    assert decision.sample >= 4
