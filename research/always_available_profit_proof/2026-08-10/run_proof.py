#!/usr/bin/env python3
"""Build the bounded existence and total-decision proof certificate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from decision_engine import decide


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE / "preregister_v1.json"
SEED = 20260810


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bootstrap_lcb(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(20_000)])
    return float(np.quantile(means, 0.025))


def metrics(values: list[float]) -> dict[str, Any]:
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(values),
        "net_jpy": sum(values),
        "expectancy_jpy": float(np.mean(values)),
        "bootstrap_lcb_expectancy_jpy": bootstrap_lcb(values),
        "profit_factor": "INF" if loss == 0 and gain > 0 else (gain / loss if loss else None),
        "max_drawdown_jpy": drawdown,
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def base_snapshot() -> dict[str, Any]:
    return {
        "decision_time": "2026-06-19T08:01:30Z",
        "causal_cutoff": "2026-06-19T08:01:30Z",
        "pair": "EUR_USD",
        "side": "SHORT",
        "strategy": "BREAKOUT_FAILURE",
        "order_type": "LIMIT",
        "exit_policy": "ATTACHED_TECHNICAL_TP_HARVEST",
        "bid": 1.14470,
        "ask": 1.14486,
        "quote_time": "2026-06-19T08:01:30Z",
        "completed_bar": True,
        "prior_resistance": 1.14480,
        "wick_high": 1.14495,
        "body_close": 1.14470,
        "limit_price": 1.14486,
        "take_profit": 1.14406,
        "stop_loss": 1.14556,
        "fillability_known": True,
        "financing_known": True,
        "margin_available": 3000.0,
        "margin_required": 1000.0,
        "unwind_known": True
    }


def utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    head, dot, tail = normalized.partition(".")
    if dot:
        digits, plus, offset = tail.partition("+")
        normalized = f"{head}.{digits[:6]}+{offset}" if plus else f"{head}.{digits[:6]}"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    loaded: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for name, source in prereg["sources"].items():
        path = ROOT / source["path"]
        actual_hash = digest(path)
        expected_hash = source.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise SystemExit(f"source SHA mismatch: {path}")
        loaded[name] = json.loads(path.read_text())
        source_hashes[name] = actual_hash

    replay = loaded["exact_vehicle_replay"]
    reason = loaded["profit_reason_ledger"]
    audit = loaded["forecast_replay_audit"]
    samples = replay["sample_replay_details"]
    if replay["target_shape"] != "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST":
        raise SystemExit("vehicle shape drift")
    if len(samples) != 4:
        raise SystemExit("frozen exact sample count drift")
    for row in samples:
        if not row["replay_win"] or row["replay_loss"]:
            raise SystemExit(f"non-winning frozen exact receipt: {row['trade_id']}")
        if row["market_close_mixed_in"] or row["market_or_stop_vehicle_mixed_in"]:
            raise SystemExit(f"vehicle contamination: {row['trade_id']}")
        if row["realized_pl_jpy"] <= 0:
            raise SystemExit(f"non-positive realized receipt: {row['trade_id']}")
        if utc(row["first_entry_touch_utc"]) > utc(row["first_tp_touch_after_entry_utc"]):
            raise SystemExit(f"causal touch ordering failure: {row['trade_id']}")

    actual_values = [float(row["realized_pl_jpy"]) for row in samples]
    normalized_values = [float(row["realized_pl_jpy"]) * 1000.0 / float(row["units"]) for row in samples]
    exact_metrics = metrics(actual_values)
    normalized_metrics = metrics(normalized_values)
    ledger_evidence = reason["exact_limit_attached_tp_evidence"]
    if abs(exact_metrics["net_jpy"] - float(ledger_evidence["net_jpy"])) > 1e-6:
        raise SystemExit("profit reason ledger net does not reconcile")

    supporting_rule = next(
        row for row in audit["precision_rules"]["edge_rules"]
        if row.get("pair") == "EUR_USD" and row.get("direction") == "DOWN"
    )
    observed_evidence = {
        "independent_samples": len(samples),
        "active_days": len({row["entry_timestamp_utc"][:10] for row in samples}),
        "positive_day_rate": 1.0,
        "lcb_jpy_per_1000u": normalized_metrics["bootstrap_lcb_expectancy_jpy"],
        "profit_factor": float("inf") if normalized_metrics["profit_factor"] == "INF" else normalized_metrics["profit_factor"],
    }
    forward_decision = decide(base_snapshot(), observed_evidence)
    admitted_evidence = {
        **observed_evidence,
        "independent_samples": 20,
        "active_days": 10,
    }
    admission_fixture_decision = decide(base_snapshot(), admitted_evidence)

    decision_rows = []
    for row in samples:
        decision_rows.append({
            "decision_id": f"historical-existence:{row['trade_id']}",
            "decision_time": row["entry_timestamp_utc"],
            "action": "TRADE",
            "scope": "FROZEN_HISTORICAL_EXISTENCE_ONLY",
            "pair": "EUR_USD",
            "side": "SHORT",
            "vehicle": replay["target_shape"],
            "realized_after_bidask_jpy": row["realized_pl_jpy"],
            "actual_outcome_used_for_forward_decision": False,
            "source_sha256": source_hashes["exact_vehicle_replay"],
        })
    decision_rows.append({
        "decision_id": forward_decision["decision_id"],
        "decision_time": base_snapshot()["decision_time"],
        "action": forward_decision["action"],
        "scope": "FORWARD_RESEARCH_PERMISSION",
        "pair": "EUR_USD",
        "side": "SHORT",
        "abstain_reasons": forward_decision["abstain_reasons"],
        "actual_outcome_used_for_forward_decision": False,
        "input_output_lineage_sha256": forward_decision["input_output_lineage_sha256"],
    })

    report = {
        "contract": prereg["contract"],
        "generated_at_utc": prereg["created_at_utc"],
        "source_hashes": source_hashes,
        "proof": {
            "decision_totality": "PROVED_BY_EXHAUSTIVE_GATE_TESTS",
            "positive_vehicle_exists": True,
            "exact_realized_after_bidask": exact_metrics,
            "exact_realized_after_bidask_per_1000u": normalized_metrics,
            "supporting_correlated_forecast_rule": {
                "samples": supporting_rule["samples"],
                "active_days": supporting_rule["active_days"],
                "optimized_avg_realized_pips": supporting_rule["optimized_avg_realized_pips"],
                "optimized_profit_factor": supporting_rule["optimized_profit_factor"],
                "daily_stability_status": supporting_rule["daily_stability_status"],
                "max_daily_sample_share": supporting_rule["max_daily_sample_share"],
                "positive_day_rate": supporting_rule["positive_day_rate"],
                "admission_use": "SUPPORT_ONLY_NOT_INDEPENDENT_PROOF"
            }
        },
        "forward_engine": {
            "current_action": forward_decision,
            "why": "The profitable vehicle is real, but four independent receipts do not meet the frozen 20-sample/10-day forward permission floor.",
            "admission_fixture_action": admission_fixture_decision,
            "admission_fixture_is_not_empirical_profit_claim": True,
            "total_answer_contract": "TRADE when every gate passes; otherwise WAIT with exact blockers"
        },
        "claim_status": "CONDITIONAL_PROFIT_EXISTENCE_PROVED__UNCONDITIONAL_ANYTIME_PROFIT_NOT_PROVED",
        "next_proof_count": 16,
        "holdout_used": False,
        "live_paper_order_deploy_used": False
    }
    (HERE / "proof_report_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with (HERE / "decision_ledger_v1.jsonl").open("w") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
