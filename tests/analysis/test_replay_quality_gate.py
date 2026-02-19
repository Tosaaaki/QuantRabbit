from __future__ import annotations

from analytics.replay_quality_gate import (
    GateThreshold,
    build_walk_forward_folds,
    compute_trade_metrics,
    evaluate_fold_gate,
    summarize_worker_folds,
)


def test_build_walk_forward_folds_basic() -> None:
    items = [f"f{i}" for i in range(1, 7)]
    folds = build_walk_forward_folds(items, train_files=3, test_files=2, step_files=1)

    assert folds == [
        {"train": ["f1", "f2", "f3"], "test": ["f4", "f5"]},
        {"train": ["f2", "f3", "f4"], "test": ["f5", "f6"]},
    ]


def test_compute_trade_metrics_profit_factor_and_drawdown() -> None:
    trades = [
        {"pnl_pips": 1.0, "exit_time": "2026-02-01T00:00:01Z"},
        {"pnl_pips": -0.5, "exit_time": "2026-02-01T00:00:02Z"},
        {"pnl_pips": 2.0, "exit_time": "2026-02-01T00:00:03Z"},
        {"pnl_pips": -3.0, "exit_time": "2026-02-01T00:00:04Z"},
        {"pnl_pips": 1.0, "exit_time": "2026-02-01T00:00:05Z"},
    ]

    metrics = compute_trade_metrics(trades)

    assert metrics["trade_count"] == 5.0
    assert round(metrics["total_pips"], 6) == 0.5
    assert round(metrics["win_rate"], 6) == 0.6
    assert round(metrics["profit_factor"], 6) == round(4.0 / 3.5, 6)
    assert round(metrics["max_drawdown_pips"], 6) == 3.0


def test_evaluate_fold_gate_fail_on_multiple_checks() -> None:
    train = {
        "trade_count": 30.0,
        "profit_factor": 1.6,
        "win_rate": 0.58,
        "total_pips": 12.0,
        "max_drawdown_pips": 10.0,
    }
    test = {
        "trade_count": 9.0,
        "profit_factor": 0.9,
        "win_rate": 0.45,
        "total_pips": -2.0,
        "max_drawdown_pips": 55.0,
    }
    threshold = GateThreshold(
        min_train_trades=20,
        min_test_trades=10,
        min_test_pf=1.0,
        min_test_win_rate=0.5,
        min_test_total_pips=0.0,
        max_test_drawdown_pips=45.0,
        min_pf_stability_ratio=0.6,
    )

    gate = evaluate_fold_gate(train, test, threshold)

    assert gate["passed"] is False
    assert "test_trade_count" in gate["failed_checks"]
    assert "test_profit_factor" in gate["failed_checks"]
    assert "test_win_rate" in gate["failed_checks"]
    assert "test_total_pips" in gate["failed_checks"]
    assert "test_max_drawdown_pips" in gate["failed_checks"]


def test_summarize_worker_folds_pass_rate() -> None:
    fold_results = [
        {"test_metrics": {"profit_factor": 1.2, "win_rate": 0.55, "max_drawdown_pips": 20.0}, "gate": {"passed": True}},
        {"test_metrics": {"profit_factor": 0.95, "win_rate": 0.49, "max_drawdown_pips": 25.0}, "gate": {"passed": False}},
        {"test_metrics": {"profit_factor": 1.1, "win_rate": 0.52, "max_drawdown_pips": 22.0}, "gate": {"passed": True}},
    ]

    summary = summarize_worker_folds(fold_results, min_fold_pass_rate=0.66)

    assert summary["folds"] == 3
    assert summary["passed_folds"] == 2
    assert round(summary["pass_rate"], 6) == round(2 / 3, 6)
    assert summary["status"] == "pass"
