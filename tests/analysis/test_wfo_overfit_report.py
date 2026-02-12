from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analytics.wfo_overfit_report import TradeRow, build_report


def _synthetic_rows() -> list[TradeRow]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    out: list[TradeRow] = []
    for day in range(70):
        ts = start + timedelta(days=day)
        # Strategy A: strong in early phase, weak later.
        a_pips = 1.8 if day < 35 else -1.2
        # Strategy B: weak in early phase, strong later.
        b_pips = -1.0 if day < 35 else 1.4
        out.append(TradeRow(close_time=ts, strategy="A", pl_pips=a_pips))
        out.append(TradeRow(close_time=ts, strategy="B", pl_pips=b_pips))
    return out


def test_build_report_outputs_wfo_summary_and_strategy_stats() -> None:
    rows = _synthetic_rows()
    report = build_report(
        rows,
        train_days=21,
        test_days=7,
        step_days=7,
        min_train_trades=7,
        min_test_trades=4,
        metric="avg_pips",
    )
    summary = report["summary"]
    assert summary["windows_evaluated"] > 0
    assert 0.0 <= summary["pbo_lite"] <= 1.0
    assert 0.0 <= summary["selected_positive_rate"] <= 1.0

    stats = report["strategy_stats"]
    assert len(stats) == 2
    assert "dsr" in stats[0]
    assert "sr_star" in stats[0]

