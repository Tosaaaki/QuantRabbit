from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.crypto.improvement import CryptoImprovementEvaluator
from quant_rabbit.crypto.ledger import CryptoLedger
from quant_rabbit.crypto.reporting import PaperShadowReportingWriter


def _epoch(ledger: CryptoLedger, mode: str, run_id: str) -> None:
    ledger.append(
        "FAST_EPOCH_SUMMARY",
        run_id,
        {
            "run_id": run_id,
            "started_at_utc": "2026-07-28T00:29:00+00:00",
            "completed_at_utc": "2026-07-28T00:30:00+00:00",
            "mode": mode,
            "pairs": ["btc_jpy"],
            "runtime": {
                "events_processed": 100,
                "elapsed_sec": 60,
                "books_ready": 1,
            },
            "latency": {
                "decision_us_p95": 50,
                "exchange_to_receive_ms_p95": 100,
            },
            "decisions": {
                "actions": {"WAIT": 100},
                "reasons": {
                    "NET_EDGE_BELOW_BUFFER": 80,
                    "IMBALANCE_BELOW_ENTRY": 20,
                },
            },
            "decision_diagnostics": {
                "market_regimes": {"RANGE_LIQUID": 100},
                "gross_edge_bps_p50": 0.2,
                "expected_cost_bps_p50": 12,
                "net_edge_bps_p50": -11.8,
                "net_edge_bps_max": -11,
                "near_threshold_waits": 0,
                "prediction_candidate_count": 0,
                "prediction_duplicate_count": 0,
                "shadow_sibling_candidates": {
                    "RANGE_MAKER_REVERSION": 100
                },
                "no_future_data": True,
            },
            "guardian": {
                "state": "GREEN",
                "kill_switch": False,
            },
            "metrics": {
                "equity_jpy": "10000",
                "discipline_violations": 0,
            },
            "safety": {
                "order_authority": "NONE",
                "broker_mutation_allowed": False,
            },
        },
        dedupe_key=f"fast-epoch-summary:{run_id}",
        created_at=datetime(
            2026, 7, 28, 0, 30, tzinfo=timezone.utc
        ),
    )


def test_improvement_loop_ranks_zero_fill_causes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    for mode in ("spot", "margin"):
        _epoch(
            CryptoLedger(tmp_path / mode / "ledger.db"),
            mode.upper(),
            f"{mode}-run",
        )
    evaluator = CryptoImprovementEvaluator(tmp_path)
    now = datetime(2026, 7, 28, 1, 30, tzinfo=timezone.utc)

    first = evaluator.run_once(now)
    second = evaluator.run_once(now)

    assert first["evaluation_added"] == 2
    assert first["experiment_added"] == 2
    assert second["evaluation_added"] == 0
    assert second["experiment_added"] == 0
    for evaluation in first["evaluations"]:
        assert evaluation["performance"]["completed_trades"] == 0
        assert (
            evaluation["root_causes_top3"][0]["code"]
            == "STRATEGY_EDGE_BELOW_COST"
        )
        assert len(evaluation["root_causes_top3"]) == 3
        assert {
            row["code"] for row in evaluation["root_causes_top3"]
        } >= {"NO_ACTIONABLE_PREDICTION_CANDIDATES"}
        assert evaluation["causality"]["future_data_used"] is False
        assert evaluation["adoption_gate"]["eligible_now"] is False
    for experiment in first["experiments"]:
        assert experiment["baseline"]["preserved"] is True
        assert (
            experiment["variant"]["only_one_category_changed"] is True
        )
        assert experiment["adoption_conditions"]["live_order_promotion"] == (
            "FORBIDDEN"
        )

    summary = PaperShadowReportingWriter(tmp_path).run_once(now)
    assert summary["local_summary_added"] == 2
    hourly = [
        row
        for row in (
            tmp_path / "summary_outbox.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if '"period":"hour"' in row
    ]
    assert len(hourly) == 1
    assert '"evaluation_count":2' in hourly[0]
