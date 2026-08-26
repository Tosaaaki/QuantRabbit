from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from counterparty_response_v4 import FEATURES
from indicator_factory_v3 import canonical, metrics, period
from run_v250_partial_holdout_v3 import sha256_file
from run_counterparty_response_study_v4 import HORIZONS, ROLES


MODEL_TYPES = ("MULTINOMIAL_RESPONSE", "RIDGE_RETURN")
FAMILY_TESTS = len(MODEL_TYPES) * len(HORIZONS)


def load_source(path: Path) -> tuple[dict[str, dict], dict[tuple[str, int, str], dict]]:
    events: dict[str, dict] = {}
    outcomes: dict[tuple[str, int, str], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["type"] == "RAW_EVENT":
                values = [float(row[name]) for name in FEATURES]
                if all(math.isfinite(value) for value in values):
                    events[row["signal_id"]] = {
                        "signal_id": row["signal_id"], "pair": row["pair"],
                        "features": values,
                    }
                continue
            key = (row["signal_id"], int(row["horizon"]), row["role"])
            item = outcomes.setdefault(key, {
                "signal_id": row["signal_id"], "pair": row["pair"],
                "fill_time": row["fill_time"], "exit_time": row["exit_time"],
                "direction": int(row["direction"]), "returns": {},
            })
            item["returns"][row["arm"]] = float(
                row["gross_return"] if row["arm"] == "RAW_SIGNAL" else row["net_return"]
            )
    outcomes = {key: value for key, value in outcomes.items()
                if value["signal_id"] in events and set(value["returns"]) == {
                    "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
                }}
    return events, outcomes


def paired_rows(events: dict[str, dict], outcomes: dict, horizon: int) -> list[dict]:
    rows = []
    for signal_id, event in events.items():
        continuation = outcomes.get((signal_id, horizon, "CONTINUATION_RESPONSE"))
        reversal = outcomes.get((signal_id, horizon, "FAILED_AUCTION_REVERSAL"))
        if continuation is None or reversal is None:
            continue
        rows.append({
            "signal_id": signal_id,
            "pair": event["pair"],
            "features": event["features"],
            "fill_time": continuation["fill_time"],
            "exit_time": continuation["exit_time"],
            "roles": {
                "CONTINUATION_RESPONSE": continuation,
                "FAILED_AUCTION_REVERSAL": reversal,
            },
        })
    return sorted(rows, key=lambda row: (row["fill_time"], row["signal_id"]))


def response_label(row: dict) -> str:
    values = {role: row["roles"][role]["returns"]["RAW_SIGNAL"] for role in ROLES}
    role = max(values, key=values.get)
    return role if values[role] > 0 else "UNRESOLVED_NO_ORDER"


def fit_and_decide(rows: list[dict], model_type: str) -> tuple[object, list[dict]]:
    tuning = [row for row in rows if period(row["fill_time"]) == "TUNING"
              and row["exit_time"] < "2026-05-01"]
    if len(tuning) < 100:
        raise ValueError("insufficient non-overlapping tuning outcomes")
    x_train = np.asarray([row["features"] for row in tuning], dtype=float)
    if model_type == "MULTINOMIAL_RESPONSE":
        labels = np.asarray([response_label(row) for row in tuning])
        if len(set(labels)) < 2:
            raise ValueError("classifier needs at least two response states")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=404),
        )
        model.fit(x_train, labels)
        selected = model.predict(np.asarray([row["features"] for row in rows], dtype=float))
        confidence = np.max(model.predict_proba(
            np.asarray([row["features"] for row in rows], dtype=float)
        ), axis=1)
    elif model_type == "RIDGE_RETURN":
        targets = np.asarray([
            [row["roles"][role]["returns"]["RAW_SIGNAL"] for role in ROLES]
            for row in tuning
        ], dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x_train, targets)
        prediction = model.predict(np.asarray([row["features"] for row in rows], dtype=float))
        best = np.argmax(prediction, axis=1)
        best_value = prediction[np.arange(len(rows)), best]
        selected = np.asarray([
            ROLES[index] if value > 0 else "UNRESOLVED_NO_ORDER"
            for index, value in zip(best, best_value)
        ])
        confidence = best_value
    else:
        raise ValueError(model_type)

    decisions = []
    for row, state, score in zip(rows, selected, confidence):
        expected = state in ROLES
        decision = {
            "source_signal_id": row["signal_id"],
            "pair": row["pair"],
            "fill_time": row["fill_time"],
            "exit_time": row["exit_time"],
            "selected_state": str(state),
            "model_score": float(score),
            "expected_order": expected,
            "cost_used_to_generate_signal": False,
            "cost_used_to_select_state": False,
        }
        if expected:
            chosen = row["roles"][str(state)]
            decision.update({"direction": chosen["direction"], "returns": chosen["returns"]})
        decisions.append(decision)
    return model, decisions


def evaluate(source: Path, output_root: Path) -> dict:
    events, outcomes = load_source(source)
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "models"
    model_dir.mkdir(exist_ok=True)
    z = statistics.NormalDist().inv_cdf(1 - 0.05 / (2 * FAMILY_TESTS))
    candidates = []
    ledger_rows = []
    model_hashes = {}
    raw_proposals = 0
    for horizon in HORIZONS:
        rows = paired_rows(events, outcomes, horizon)
        raw_proposals += len(rows)
        for model_type in MODEL_TYPES:
            model, decisions = fit_and_decide(rows, model_type)
            candidate_id = f"FX_CRS_{model_type}_H{horizon}_V4"
            artifact = model_dir / f"{candidate_id}.joblib"
            joblib.dump(model, artifact)
            model_hashes[candidate_id] = sha256_file(artifact)
            for decision in decisions:
                ledger_rows.append({"candidate_id": candidate_id, "horizon_m5_bars": horizon, **decision})
            summaries = {}
            for split in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC"):
                all_split = [row for row in decisions if period(row["fill_time"]) == split]
                selected = [row for row in all_split if row["expected_order"]]
                summaries[split] = {
                    "raw_proposals": len(all_split),
                    "expected_orders": len(selected),
                    "unresolved": sum(not row["expected_order"] for row in all_split),
                    "direction_accuracy": (
                        sum(row["returns"]["RAW_SIGNAL"] > 0 for row in selected) / len(selected)
                        if selected else None
                    ),
                    "arms": {arm: metrics(selected, arm, z) for arm in (
                        "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
                    )},
                }
            corrected = [
                summaries["TUNING"]["arms"]["RAW_SIGNAL"]["family_corrected_cluster_lower_bps"],
                summaries["WALK_FORWARD"]["arms"]["EXECUTABLE_BASE"]["family_corrected_cluster_lower_bps"],
                summaries["WALK_FORWARD"]["arms"]["ADVERSE_STRESS"]["family_corrected_cluster_lower_bps"],
            ]
            finite = [value for value in corrected if value is not None]
            candidates.append({
                "candidate_id": candidate_id,
                "model_type": model_type,
                "horizon_m5_bars": horizon,
                "periods": summaries,
                "development_admitted": all(value is not None and value > 0 for value in corrected),
                "ranking_floor_bps": min(finite) if finite else None,
            })
    candidates.sort(key=lambda item: (
        item["development_admitted"],
        item["ranking_floor_bps"] if item["ranking_floor_bps"] is not None else -math.inf,
    ), reverse=True)
    ledger_path = output_root / "decision_ledger_counterparty_v4.jsonl"
    ledger_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows), encoding="utf-8")
    payload = {
        "experiment": "FX_COUNTERPARTY_RESPONSE_MODELS_V4",
        "evidence_class": "opened_development_not_future_holdout",
        "family_tests": FAMILY_TESTS,
        "family_corrected_z": z,
        "source_raw_events": len(events),
        "raw_proposals_across_fixed_horizons": raw_proposals,
        "cost_suppressed_raw_proposals": 0,
        "same_source_signal_id_all_cost_arms": True,
        "candidates": candidates,
        "development_admitted_count": sum(item["development_admitted"] for item in candidates),
        "final_admitted_count": 0,
        "models": model_hashes,
        "source_ledger": str(source),
        "source_ledger_sha256": sha256_file(source),
        "decision_ledger": str(ledger_path),
        "decision_ledger_sha256": sha256_file(ledger_path),
        "fixed_horizon_exit": True,
        "terminal_open_inventory": 0,
        "terminal_inventory_mtm_hidden": False,
        "portfolio_admission_blockers": [
            "opened 2026 data are development evidence",
            "no untouched future FX holdout is available",
            "portfolio leverage/currency-cap replay is required if a proposal-level candidate passes",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    result_path = output_root / "result_counterparty_models_v4.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.source_ledger, args.output_root)
    print(json.dumps({
        "source_raw_events": result["source_raw_events"],
        "raw_proposals": result["raw_proposals_across_fixed_horizons"],
        "development_admitted": result["development_admitted_count"],
        "top_candidate": result["candidates"][0],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
