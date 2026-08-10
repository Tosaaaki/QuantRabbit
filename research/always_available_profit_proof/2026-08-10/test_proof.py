from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("always_decision_engine", HERE / "decision_engine.py")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
assert spec.loader is not None
spec.loader.exec_module(engine)


def snapshot() -> dict:
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
        "margin_available": 3000,
        "margin_required": 1000,
        "unwind_known": True,
    }


def evidence() -> dict:
    return {
        "independent_samples": 20,
        "active_days": 10,
        "positive_day_rate": 0.7,
        "lcb_jpy_per_1000u": 1.0,
        "profit_factor": 1.1,
    }


def test_fully_admitted_snapshot_trades() -> None:
    assert engine.decide(snapshot(), evidence())["action"] == "TRADE"


def test_current_four_sample_evidence_waits() -> None:
    current = evidence()
    current["independent_samples"] = 4
    current["active_days"] = 4
    result = engine.decide(snapshot(), current)
    assert result["action"] == "WAIT"
    assert "SAMPLE_FLOOR_NOT_MET" in result["abstain_reasons"]


def test_wick_sweep_needs_completed_body_confirmation() -> None:
    row = snapshot()
    row["completed_bar"] = False
    assert engine.decide(row, evidence())["action"] == "WAIT"
    row = snapshot()
    row["body_close"] = row["prior_resistance"] + 0.00001
    assert engine.decide(row, evidence())["action"] == "WAIT"


def test_future_input_and_mid_price_are_blocked() -> None:
    row = snapshot()
    row["causal_cutoff"] = "2026-06-19T08:01:31Z"
    assert "FUTURE_INPUT" in engine.decide(row, evidence())["abstain_reasons"]
    row = snapshot()
    row["bid"] = row["ask"] + 0.00001
    assert "INVALID_BID_ASK" in engine.decide(row, evidence())["abstain_reasons"]


def test_realized_outcome_cannot_change_forward_action() -> None:
    clean = engine.decide(snapshot(), evidence())
    contaminated = snapshot()
    contaminated["actual_after_cost_net"] = -999999
    contaminated["terminal_reason"] = "STOP_LOSS_ORDER"
    dirty = engine.decide(contaminated, evidence())
    assert clean["action"] == dirty["action"] == "TRADE"
    assert clean["outcome_fields_consumed"] == dirty["outcome_fields_consumed"] == []


def test_missing_margin_financing_fill_and_unwind_fail_closed() -> None:
    row = snapshot()
    for key in ("fillability_known", "financing_known", "unwind_known"):
        row[key] = False
    row["margin_available"] = None
    result = engine.decide(row, evidence())
    assert result["action"] == "WAIT"
    assert {"FILLABILITY_MISSING", "FINANCING_MISSING", "UNWIND_MISSING", "MARGIN_MISSING"} <= set(result["abstain_reasons"])


def test_totality_and_determinism_over_1024_gate_combinations() -> None:
    boolean_fields = (
        "completed_bar",
        "fillability_known",
        "financing_known",
        "unwind_known",
    )
    for values in itertools.product((False, True), repeat=len(boolean_fields)):
        row = snapshot()
        for key, value in zip(boolean_fields, values):
            row[key] = value
        for enough_samples, enough_days, good_lcb, good_pf, good_days, fresh_quote in itertools.product((False, True), repeat=6):
            ev = evidence()
            ev["independent_samples"] = 20 if enough_samples else 4
            ev["active_days"] = 10 if enough_days else 4
            ev["lcb_jpy_per_1000u"] = 1 if good_lcb else -1
            ev["profit_factor"] = 1.1 if good_pf else 0.9
            ev["positive_day_rate"] = 0.7 if good_days else 0.5
            if not fresh_quote:
                row["quote_time"] = "2026-06-19T08:01:20Z"
            else:
                row["quote_time"] = "2026-06-19T08:01:30Z"
            first = engine.decide(row, ev)
            second = engine.decide(row, ev)
            assert first == second
            assert first["action"] in {"TRADE", "WAIT"}
            assert bool(first["abstain_reasons"]) == (first["action"] == "WAIT")


def test_generated_report_reconciles_frozen_receipts() -> None:
    report = json.loads((HERE / "proof_report_v1.json").read_text())
    exact = report["proof"]["exact_realized_after_bidask"]
    assert exact["trades"] == 4
    assert exact["wins"] == 4
    assert exact["losses"] == 0
    assert abs(exact["net_jpy"] - 3255.0938) < 1e-6
    assert exact["bootstrap_lcb_expectancy_jpy"] > 0
    assert report["forward_engine"]["current_action"]["action"] == "WAIT"
    assert report["forward_engine"]["admission_fixture_action"]["action"] == "TRADE"
