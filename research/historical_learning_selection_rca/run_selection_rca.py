#!/usr/bin/env python3
"""Frozen, research-only RCA for historical trade/skip selection.

Inputs are bound to commit f35e8c176 artifacts.  The runner never reads the
post-anchor holdout, never calls a broker, and never invents a skip outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
from statistics import mean
import sys
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20_260_809
BOOTSTRAPS = 10_000
MIN_N = 30
MIN_RETENTION = 0.80
ANCHOR = "2026-07-09T07:46:03.151624347Z"
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", ANCHOR),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", ANCHOR),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", ANCHOR),
)
FROZEN = {
    "research/historical_learning_admission/all_entry_episodes_v1.jsonl": "efcf6b0fb675050d6a08efc0119065e0874e50e1c51373a0c0fb61bb6ebd815e",
    "research/historical_learning_admission/admission_report_v1.json": "5a5ddaca413b3b1db6352e9f35afcece417e996cf58feede0c2356c5f857264a",
    "research/historical_learning_admission/preregister_v1.json": "39da302c5a74c0f429f7d128577af97c6c5ff381cd4046a2db4c73e1cb70cf0c",
    "data/execution_ledger.db": "545feb1d62410904bf3f86b4290986caf3932546ef858abec6c3eb27a58b38eb",
    "data/forecast_history.jsonl": "771c4fce9c2f8782fe51c8068198fdc1870b22eceddf58157b01f66218fffe05",
    "data/projection_ledger.jsonl": "9b6f7582f44850e94238eadfc1d54f5ec7eb9743de2c685a746a323844423b0a",
    "data/entry_thesis_ledger.jsonl": "2ccbe49b00de30849cad3f5d8e8f1a61fae8d84d8b1d89b4d4a13c9de4ae5651",
}


def load_parent(repo: Path) -> Any:
    path = repo / "research/historical_learning_admission/run_admission.py"
    spec = importlib.util.spec_from_file_location("frozen_admission_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen(repo: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in FROZEN.items():
        value = sha256(repo / relative)
        if value != expected:
            raise RuntimeError(f"frozen input changed: {relative}: {value} != {expected}")
        actual[relative] = value
    return actual


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lane_family(row: dict[str, Any]) -> str:
    lane = str(row.get("lane_id") or "MISSING")
    return lane.split(":", 1)[0]


def pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vector", DictVectorizer(sparse=False)),
            ("scale", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=100,
                    max_depth=3,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_predict(parent: Any, train: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[Pipeline, np.ndarray]:
    model = pipeline()
    model.fit([parent.features(row) for row in train], np.asarray([float(row["net_jpy"]) for row in train]))
    prediction = model.predict([parent.features(row) for row in rows])
    return model, np.asarray(prediction, dtype=float)


def bootstrap_mean_ci(values: list[float], seed_offset: int = 0) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(SEED + seed_offset)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAPS))
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def drawdown(values: Iterable[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def profit_factor(values: list[float]) -> float | str | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses:
        return gains / losses
    if gains:
        return "Infinity"
    return None


def metric(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    baseline = [float(row["net_jpy"]) for row in rows]
    values = [value if take else 0.0 for value, take in zip(baseline, selected)]
    deltas = [value - base for value, base in zip(values, baseline)]
    ci = bootstrap_mean_ci(deltas)
    financing = sum(float(row.get("financing_jpy") or 0.0) for row, take in zip(rows, selected) if take)
    known_margin: list[float] = []
    for row, take in zip(rows, selected):
        if take and str(row["pair"]).endswith("_JPY") and row.get("intended_price") is not None:
            known_margin.append(float(row["units"]) * float(row["intended_price"]) * 0.04)
    pf = profit_factor(values)
    return {
        "trades_available": len(rows),
        "trades_selected": sum(selected),
        "retention_ratio": sum(selected) / len(rows) if rows else None,
        "net_jpy": sum(values),
        "baseline_net_jpy": sum(baseline),
        "incremental_net_jpy": sum(deltas),
        "incremental_expectancy_jpy": mean(deltas) if deltas else None,
        "paired_bootstrap_95pct_ci_jpy": ci,
        "paired_lcb_jpy": ci[0],
        "profit_factor": pf,
        "expectancy_per_available_episode_jpy": mean(values) if values else None,
        "max_drawdown_jpy": drawdown(values),
        "financing_jpy_explicit": financing,
        "cost_breakdown": {
            "spread": "implicit in actual broker realized P/L",
            "fill_slippage": "implicit in actual broker realized P/L",
            "fee": "zero-account contract",
            "financing_jpy": financing,
            "opportunity_cost": "missing",
        },
        "peak_margin_jpy_known_jpy_pairs_only": max(known_margin, default=None),
        "margin_coverage": len(known_margin) / max(1, sum(selected)),
        "margin_status": "INCOMPLETE_DECISION_TIME_CROSS_CURRENCY_CONVERSION",
        "ruin_proxy": "UNPROVEN_WITHOUT_COMPLETE_MARGIN_AND_INITIAL_EQUITY_SNAPSHOT",
        "selected_episode_ids": [row["episode_id"] for row, take in zip(rows, selected) if take],
    }


def candidate_gates(report: dict[str, Any]) -> dict[str, bool]:
    pf = report["profit_factor"]
    return {
        "minimum_validation_events": report["trades_available"] >= MIN_N,
        "minimum_retention": report["retention_ratio"] is not None and report["retention_ratio"] >= MIN_RETENTION,
        "net_strictly_above_all_trades": report["incremental_net_jpy"] > 0,
        "paired_lcb_positive": report["paired_lcb_jpy"] is not None and report["paired_lcb_jpy"] > 0,
        "profit_factor_above_one": pf == "Infinity" or (isinstance(pf, (int, float)) and pf > 1),
        "margin_complete": report["margin_coverage"] == 1.0,
    }


def coverage_binding_take(row: dict[str, Any], predicted_net_jpy: float) -> bool:
    """A missing causal forecast abstains from filtering, so the trade passes."""
    return (not bool(row.get("forecast_present"))) or predicted_net_jpy > 0


def category_table(rows: list[dict[str, Any]], key: str, label: str = "label_available") -> dict[str, Any]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        category = str(row[key])
        available = bool(row[label])
        counts[category][1 if available else 0] += 1
    observed = np.asarray(list(counts.values()), dtype=float)
    if observed.shape[0] >= 2 and np.all(observed.sum(axis=0) > 0):
        chi2, p_value, _, _ = chi2_contingency(observed)
        total = observed.sum()
        phi2 = chi2 / total
        rows_n, cols_n = observed.shape
        correction = ((cols_n - 1) * (rows_n - 1)) / max(total - 1, 1)
        phi2_corrected = max(0.0, phi2 - correction)
        rows_corrected = rows_n - ((rows_n - 1) ** 2) / max(total - 1, 1)
        cols_corrected = cols_n - ((cols_n - 1) ** 2) / max(total - 1, 1)
        denominator = min(cols_corrected - 1, rows_corrected - 1)
        cramers_v = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
    else:
        chi2 = p_value = cramers_v = None
    return {
        "categories": {
            category: {"missing": values[0], "available": values[1], "availability_rate": values[1] / sum(values)}
            for category, values in sorted(counts.items())
        },
        "chi_square": chi2,
        "p_value": p_value,
        "cramers_v_bias_corrected": cramers_v,
    }


def observed_net_missingness(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    present = [float(row["net_jpy"]) for row in rows if row[key]]
    missing = [float(row["net_jpy"]) for row in rows if not row[key]]
    if present and missing:
        statistic, p_value = mannwhitneyu(present, missing, alternative="two-sided")
        difference = mean(present) - mean(missing)
        rng = random.Random(SEED + len(key))
        samples = sorted(
            mean(rng.choice(present) for _ in present) - mean(rng.choice(missing) for _ in missing)
            for _ in range(BOOTSTRAPS)
        )
        ci: list[float | None] = [samples[249], samples[9749]]
    else:
        statistic = p_value = difference = None
        ci = [None, None]
    return {
        "present_n": len(present),
        "missing_n": len(missing),
        "present_net_jpy": sum(present),
        "missing_net_jpy": sum(missing),
        "present_mean_jpy": mean(present) if present else None,
        "missing_mean_jpy": mean(missing) if missing else None,
        "mean_difference_present_minus_missing_jpy": difference,
        "bootstrap_95pct_ci_jpy": ci,
        "mann_whitney_u": statistic,
        "mann_whitney_p": p_value,
        "present_win_rate": sum(value > 0 for value in present) / len(present) if present else None,
        "missing_win_rate": sum(value > 0 for value in missing) / len(missing) if missing else None,
    }


def missingness_predictability(parent: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: parent.parse_time(row["feature_at_utc"]))
    cut = math.floor(len(ordered) * 0.60)
    train, validation = ordered[:cut], ordered[cut:]
    y_train = np.asarray([int(row["label_available"]) for row in train])
    y_validation = np.asarray([int(row["label_available"]) for row in validation])
    if len(set(y_train.tolist())) < 2 or len(set(y_validation.tolist())) < 2:
        return {"status": "SINGLE_CLASS", "train_n": len(train), "validation_n": len(validation)}
    model = Pipeline(
        [
            ("vector", DictVectorizer(sparse=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]
    )
    model.fit([parent.features(row) for row in train], y_train)
    scores = model.predict_proba([parent.features(row) for row in validation])[:, 1]
    auc = float(roc_auc_score(y_validation, scores))
    rng = np.random.default_rng(SEED)
    permutation = [float(roc_auc_score(rng.permutation(y_validation), scores)) for _ in range(1000)]
    p_value = (1 + sum(value >= auc for value in permutation)) / (len(permutation) + 1)
    return {
        "status": "OK",
        "train_n": len(train),
        "validation_n": len(validation),
        "validation_auc": auc,
        "permutation_auc_mean": mean(permutation),
        "permutation_p_value_one_sided": p_value,
        "features": "same causal/static feature contract as frozen admission; administrative episode_status excluded",
    }


def load_thesis(repo: Path, parent: Any) -> dict[str, list[tuple[datetime, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for row in read_jsonl(repo / "data/entry_thesis_ledger.jsonl"):
        if row.get("trade_id") is None:
            continue
        timestamp = parent.parse_time(str(row["timestamp_utc"]))
        if timestamp <= parent.parse_time(ANCHOR):
            grouped[str(row["trade_id"])].append((timestamp, row))
    return grouped


def enrich_rows(repo: Path, parent: Any, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thesis = load_thesis(repo, parent)
    enriched = []
    for source in episodes:
        row = dict(source)
        feature_at = parent.parse_time(row["feature_at_utc"])
        records = thesis.get(str(row.get("trade_id")), [])
        row["label_available"] = row["label_status"] == "ACTUAL_AFTER_COST"
        row["forecast_present"] = row.get("forecast_at_utc") is not None
        row["intended_price_present"] = row.get("intended_price") is not None
        row["thesis_record_present"] = bool(records)
        row["thesis_causal_present"] = any(timestamp <= feature_at for timestamp, _ in records)
        row["thesis_post_feature_only"] = bool(records) and not row["thesis_causal_present"]
        row["lane_family"] = lane_family(row)
        timestamp = parent.parse_time(row["feature_at_utc"])
        row["hour_utc"] = f"{timestamp.hour:02d}"
        row["weekday_utc"] = str(timestamp.weekday())
        enriched.append(row)
    return enriched


def source_inventory(repo: Path, parent: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = repo / "data/execution_ledger.db"
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        event_counts = {
            event: {"count": count, "min_utc": first, "max_utc": last}
            for event, count, first, last in connection.execute(
                "SELECT event_type,COUNT(*),MIN(ts_utc),MAX(ts_utc) FROM execution_events WHERE ts_utc<=? GROUP BY event_type",
                (ANCHOR,),
            )
        }
        receipts = list(
            connection.execute(
                "SELECT receipt_uid,ts_utc,kind,status,sent,lane_id,payload_json FROM gateway_receipts WHERE ts_utc<=? ORDER BY ts_utc",
                (ANCHOR,),
            )
        )
    finally:
        connection.close()
    receipt_status = Counter(str(item[3]) for item in receipts)
    explicit_no_action = sum(str(item[3]) == "NO_ACTION" or "NO_ACTION" in str(item[0]) for item in receipts)
    matched_episode_ids: set[str] = set()
    order_ids = {str(row["order_id"]): str(row["episode_id"]) for row in rows if row.get("order_id")}
    for item in receipts:
        payload = json.loads(item[6]) if item[6] else {}
        text = json.dumps(payload, sort_keys=True)
        for order_id, episode_id in order_ids.items():
            if f'"order_id": "{order_id}"' in text:
                matched_episode_ids.add(episode_id)
    projections = [row for row in read_jsonl(repo / "data/projection_ledger.jsonl") if parent.parse_time(str(row["timestamp_emitted_utc"])) <= parent.parse_time(ANCHOR)]
    forecasts = [row for row in read_jsonl(repo / "data/forecast_history.jsonl") if parent.parse_time(str(row["timestamp_utc"])) <= parent.parse_time(ANCHOR)]
    thesis = [row for row in read_jsonl(repo / "data/entry_thesis_ledger.jsonl") if parent.parse_time(str(row["timestamp_utc"])) <= parent.parse_time(ANCHOR)]
    snapshots = []
    for relative in ("data/order_intents.json", "data/candidate_intents.json", "data/trader_decision.json", "data/gpt_trader_decision.json", "data/scout_execution_receipt.json"):
        path = repo / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = payload.get("generated_at_utc") or payload.get("timestamp_utc")
        snapshots.append({
            "path": relative,
            "sha256": sha256(path),
            "generated_at_utc": timestamp,
            "append_only": False,
            "historical_cohort_usable": bool(timestamp) and parent.parse_time(str(timestamp)) <= parent.parse_time(ANCHOR),
            "reason": "single mutable snapshot; does not establish a complete historical skip ledger",
        })
    return {
        "execution_events_to_anchor": event_counts,
        "gateway_receipts_to_anchor": {
            "count": len(receipts),
            "status_counts": dict(sorted(receipt_status.items())),
            "explicit_no_action_count": explicit_no_action,
            "episode_ids_joined_by_explicit_order_id": len(matched_episode_ids),
            "assessment": "sparse receipts are admissible evidence but not a complete decision opportunity stream",
        },
        "forecast_history_to_anchor": {"rows": len(forecasts), "first_utc": forecasts[0]["timestamp_utc"] if forecasts else None, "last_utc": forecasts[-1]["timestamp_utc"] if forecasts else None},
        "projection_ledger_to_anchor": {
            "rows": len(projections),
            "first_utc": projections[0]["timestamp_emitted_utc"] if projections else None,
            "last_utc": projections[-1]["timestamp_emitted_utc"] if projections else None,
            "usable_feature_fields": ["pair", "signal_name", "direction", "lead_time_min", "confidence", "entry_price", "predicted_target_price", "predicted_invalidation_price", "regime_at_emission"],
            "forbidden_post_outcome_fields": ["resolution_status", "resolved_at_utc", "resolution_evidence"],
            "episode_join_status": "NO_STABLE_ORDER_OR_EPISODE_ID; not joined to the 549 cohort",
        },
        "entry_thesis_to_anchor": {
            "rows": len(thesis),
            "assessment": "trade_id join exists, but timestamp causality is checked per episode; post-feature rows are not features",
        },
        "mutable_snapshots": snapshots,
        "skip_ledger_status": "MISSING_APPEND_ONLY_COMPLETE_DECISION_LEDGER",
    }


def shadow_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["label_available"] and row["forecast_present"]:
            grade = "EXECUTED_LABELED_CAUSAL_FORECAST"
        elif row["label_available"]:
            grade = "EXECUTED_LABELED_FEATURE_PARTIAL"
        elif row["episode_status"] in {"CANCELED_UNFILLED", "REJECTED", "ACCEPTED_UNRESOLVED"}:
            grade = "OBSERVED_DECISION_OR_ORDER_NO_COUNTERFACTUAL_LABEL"
        else:
            grade = "UNRESOLVED"
        result.append({
            "episode_id": row["episode_id"],
            "feature_at_utc": row["feature_at_utc"],
            "order_id": row.get("order_id"),
            "trade_id": row.get("trade_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "episode_status": row["episode_status"],
            "label_status": row["label_status"],
            "causal_forecast_present": row["forecast_present"],
            "thesis_record_present": row["thesis_record_present"],
            "decision_time_thesis_present": row["thesis_causal_present"],
            "post_feature_thesis_rejected": row["thesis_post_feature_only"],
            "reconstruction_grade": grade,
            "skip_inference": "NOT_INFERRED",
        })
    return result


def oof_predictions(parent: Any, train: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    ordered = sorted(train, key=lambda row: parent.parse_time(row["feature_at_utc"]))
    predictions: list[tuple[dict[str, Any], float]] = []
    boundaries = [0.40, 0.55, 0.70, 0.85, 1.00]
    start = max(MIN_N, math.floor(len(ordered) * 0.25))
    previous = start
    for boundary in boundaries:
        stop = math.floor(len(ordered) * boundary)
        if stop <= previous:
            continue
        validation = ordered[previous:stop]
        if not validation:
            continue
        validation_start = parent.parse_time(validation[0]["feature_at_utc"])
        fit_rows = [row for row in ordered[:previous] if parent.parse_time(row["close_at_utc"]) < validation_start - parent.EMBARGO]
        if len(fit_rows) < MIN_N:
            previous = stop
            continue
        _, values = fit_predict(parent, fit_rows, validation)
        predictions.extend(zip(validation, values.tolist()))
        previous = stop
    return [item[0] for item in predictions], np.asarray([item[1] for item in predictions], dtype=float)


def choose_threshold(oof_rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    if len(oof_rows) < MIN_N:
        return {"status": "FALLBACK_ALL_TRADES_INSUFFICIENT_OOF", "threshold_jpy": None, "oof_n": len(oof_rows)}
    candidates = []
    for quantile in (0.00, 0.05, 0.10, 0.15, 0.20):
        threshold = float(np.quantile(predictions, quantile))
        selected = (predictions > threshold).tolist()
        report = metric(oof_rows, selected)
        qualifies = (
            report["retention_ratio"] >= MIN_RETENTION
            and report["incremental_net_jpy"] > 0
            and report["paired_lcb_jpy"] is not None
            and report["paired_lcb_jpy"] > 0
        )
        candidates.append({"quantile": quantile, "threshold_jpy": threshold, "qualifies": qualifies, "metrics": report})
    qualified = [item for item in candidates if item["qualifies"]]
    if not qualified:
        return {"status": "FALLBACK_ALL_TRADES_NO_TRAIN_QUALIFYING_THRESHOLD", "threshold_jpy": None, "oof_n": len(oof_rows), "grid": candidates}
    best = max(qualified, key=lambda item: (item["metrics"]["incremental_net_jpy"], item["metrics"]["retention_ratio"]))
    return {"status": "TRAIN_THRESHOLD_FIXED", "threshold_jpy": best["threshold_jpy"], "quantile": best["quantile"], "oof_n": len(oof_rows), "grid": candidates}


def calibration_offsets(oof_rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    if not oof_rows:
        return {"global": 0.0, "groups": {}, "oof_n": 0, "status": "NO_OOF_FALLBACK_ZERO"}
    residuals = np.asarray([float(row["net_jpy"]) for row in oof_rows]) - predictions
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, residual in zip(oof_rows, residuals.tolist()):
        grouped[f'{row["pair"]}|{row["side"]}'].append(float(residual))
    eligible = {key: mean(values) for key, values in grouped.items() if len(values) >= 20}
    return {"global": float(residuals.mean()), "groups": eligible, "group_counts": {key: len(values) for key, values in sorted(grouped.items())}, "oof_n": len(oof_rows), "status": "TRAIN_OOF_RESIDUALS"}


def attribution(rows: list[dict[str, Any]], predictions: np.ndarray, selected: list[bool]) -> dict[str, Any]:
    values = np.asarray([float(row["net_jpy"]) for row in rows])
    selected_array = np.asarray(selected, dtype=bool)
    false_negative = (~selected_array) & (values > 0)
    false_positive = selected_array & (values <= 0)
    excluded_loser = (~selected_array) & (values < 0)
    missed_winners = float(values[false_negative].sum())
    avoided_losers = float(-values[excluded_loser].sum())
    order = np.argsort(predictions)
    deciles = []
    for index, positions in enumerate(np.array_split(order, min(10, len(order))), start=1):
        if not len(positions):
            continue
        deciles.append({
            "prediction_decile_low_to_high": index,
            "n": len(positions),
            "mean_predicted_net_jpy": float(predictions[positions].mean()),
            "mean_actual_net_jpy": float(values[positions].mean()),
            "actual_net_jpy": float(values[positions].sum()),
            "actual_win_rate": float((values[positions] > 0).mean()),
        })
    correlation, p_value = spearmanr(predictions, values)
    top_missed = sorted(
        ({"episode_id": row["episode_id"], "actual_net_jpy": float(row["net_jpy"]), "predicted_net_jpy": float(prediction)} for row, prediction, take in zip(rows, predictions, selected) if not take and float(row["net_jpy"]) > 0),
        key=lambda item: item["actual_net_jpy"], reverse=True,
    )[:10]
    top_kept_losers = sorted(
        ({"episode_id": row["episode_id"], "actual_net_jpy": float(row["net_jpy"]), "predicted_net_jpy": float(prediction)} for row, prediction, take in zip(rows, predictions, selected) if take and float(row["net_jpy"]) <= 0),
        key=lambda item: item["actual_net_jpy"],
    )[:10]
    return {
        "selected_net_jpy": float(values[selected_array].sum()),
        "excluded_actual_net_jpy": float(values[~selected_array].sum()),
        "false_negative_winners": int(false_negative.sum()),
        "false_negative_winner_jpy": missed_winners,
        "false_positive_losers": int(false_positive.sum()),
        "false_positive_loser_jpy": float(values[false_positive].sum()),
        "avoided_losers": int(excluded_loser.sum()),
        "avoided_loser_magnitude_jpy": avoided_losers,
        "incremental_identity_jpy": avoided_losers - missed_winners,
        "spearman_prediction_actual": None if math.isnan(float(correlation)) else float(correlation),
        "spearman_p_value": None if math.isnan(float(p_value)) else float(p_value),
        "calibration_deciles": deciles,
        "top_missed_winners": top_missed,
        "top_kept_losers": top_kept_losers,
    }


def evaluate_windows(parent: Any, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labeled = [row for row in rows if row["label_available"]]
    reports = []
    prediction_rows: list[dict[str, Any]] = []
    for window_id, start_text, end_text in WINDOWS:
        start, end = parent.parse_time(start_text), parent.parse_time(end_text)
        scoped = [row for row in labeled if start <= parent.parse_time(row["feature_at_utc"]) <= end]
        train, validation, purged = parent.split_rows(scoped)
        base = metric(validation, [True] * len(validation))
        window: dict[str, Any] = {
            "id": window_id,
            "labeled_events": len(scoped),
            "train_events": len(train),
            "validation_events": len(validation),
            "purged_train_events": purged,
            "ALL_TRADES": base,
        }
        if len(train) < MIN_N or len(validation) < MIN_N:
            window["status"] = "NOT_FIT_MINIMUM_SAMPLE_GATE"
            window["candidates"] = {name: {"status": "NOT_FIT_MINIMUM_SAMPLE_GATE", "accepted": False} for name in ("FROZEN_HGB", "A_COVERAGE_BINDING", "B_COST_AWARE_ABSTAIN", "C_PAIR_SIDE_CALIBRATION")}
            reports.append(window)
            continue
        model, predictions = fit_predict(parent, train, validation)
        frozen_selected = (predictions > 0).tolist()
        frozen_report = metric(validation, frozen_selected)
        frozen_report["gates"] = candidate_gates(frozen_report)
        frozen_report["accepted"] = all(frozen_report["gates"].values())
        frozen_report["attribution"] = attribution(validation, predictions, frozen_selected)

        a_selected = [coverage_binding_take(row, float(prediction)) for row, prediction in zip(validation, predictions)]
        a_report = metric(validation, a_selected)
        a_report["gates"] = candidate_gates(a_report)
        a_report["accepted"] = all(a_report["gates"].values())

        oof_rows, oof_values = oof_predictions(parent, train)
        threshold = choose_threshold(oof_rows, oof_values)
        if threshold["threshold_jpy"] is None:
            b_selected = [True] * len(validation)
        else:
            b_selected = (predictions > float(threshold["threshold_jpy"])).tolist()
        b_report = metric(validation, b_selected)
        b_report["train_only_threshold"] = threshold
        b_report["validation_labels_used_for_threshold"] = False
        b_report["gates"] = candidate_gates(b_report)
        b_report["accepted"] = all(b_report["gates"].values())

        offsets = calibration_offsets(oof_rows, oof_values)
        calibrated = np.asarray([
            prediction + offsets["groups"].get(f'{row["pair"]}|{row["side"]}', offsets["global"])
            for row, prediction in zip(validation, predictions)
        ])
        c_selected = (calibrated > 0).tolist()
        c_report = metric(validation, c_selected)
        c_report["train_only_calibration"] = offsets
        c_report["validation_labels_used_for_calibration"] = False
        c_report["gates"] = candidate_gates(c_report)
        c_report["accepted"] = all(c_report["gates"].values())

        window["status"] = "EVALUATED"
        window["candidates"] = {
            "FROZEN_HGB": frozen_report,
            "A_COVERAGE_BINDING": a_report,
            "B_COST_AWARE_ABSTAIN": b_report,
            "C_PAIR_SIDE_CALIBRATION": c_report,
        }
        for row, prediction, frozen, a, b, c, calibrated_prediction in zip(validation, predictions, frozen_selected, a_selected, b_selected, c_selected, calibrated):
            prediction_rows.append({
                "window_id": window_id,
                "episode_id": row["episode_id"],
                "feature_at_utc": row["feature_at_utc"],
                "pair": row["pair"],
                "side": row["side"],
                "actual_net_jpy": row["net_jpy"],
                "frozen_hgb_predicted_net_jpy": float(prediction),
                "frozen_hgb_selected": frozen,
                "coverage_binding_selected": a,
                "cost_aware_selected": b,
                "calibrated_predicted_net_jpy": float(calibrated_prediction),
                "pair_side_calibration_selected": c,
            })
        reports.append(window)
    return reports, prediction_rows


def x_contract_status(repo: Path) -> dict[str, Any]:
    queue_path = repo / "research/x_fx_methods/2026-08-09/validation_queue.json"
    ledger_path = repo / "research/x_fx_methods/2026-08-09/ledger.json"
    if not queue_path.exists() or not ledger_path.exists():
        return {"status": "WAITING_NO_OUTPUT", "admitted": False}
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    candidates = {item["id"]: item for item in ledger.get("candidates", [])}
    audited = []
    for contract in queue.get("queue", []):
        source = candidates.get(contract.get("source_candidate"), {})
        claim = source.get("explicit_claim", {})
        complete = all(
            [
                source.get("status_url"),
                claim.get("entry"),
                claim.get("exit"),
                claim.get("invalidation"),
                queue.get("common_contract", {}).get("costs"),
            ]
        )
        standalone = complete and not any(
            marker in str(claim.get(field, "")).lower()
            for field in ("entry", "exit", "invalidation")
            for marker in ("not a standalone", "no single executable", "keep the strategy's original", "keep the baseline", "baseline setup")
        )
        audited.append({
            "contract_id": contract.get("contract_id"),
            "source_status_url": source.get("status_url"),
            "field_complete": complete,
            "standalone_executable": standalone,
            "current_status": contract.get("current_status"),
        })
    return {
        "status": "NOT_ADMITTED_NO_STANDALONE_COMPLETE_CONTRACT" if not any(item["standalone_executable"] for item in audited) else "ADMISSIBLE_CONTRACT_AVAILABLE_NOT_EVALUATED_IN_THIS_RUN",
        "admitted": False,
        "reason": "The visible X task is active; existing queue items inherit baseline entry/exit/invalidation or leave a source exit non-unique, so no standalone D variable is inferred.",
        "audited": audited,
        "files_read_only": [str(queue_path.relative_to(repo)), str(ledger_path.relative_to(repo))],
    }


def run(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    parent = load_parent(repo)
    bindings = verify_frozen(repo)
    episodes = read_jsonl(repo / "research/historical_learning_admission/all_entry_episodes_v1.jsonl")
    rows = enrich_rows(repo, parent, episodes)
    inventory = source_inventory(repo, parent, rows)
    windows, predictions = evaluate_windows(parent, rows)
    categorical = {
        key: category_table(rows, key)
        for key in ("episode_status", "pair", "side", "lane_family", "hour_utc", "weekday_utc", "forecast_present", "intended_price_present")
    }
    labeled = [row for row in rows if row["label_available"]]
    missingness = {
        "episodes": len(rows),
        "labeled": len(labeled),
        "label_coverage": len(labeled) / len(rows),
        "causal_forecast_coverage_all_episodes": sum(row["forecast_present"] for row in rows) / len(rows),
        "causal_forecast_coverage_labeled": sum(row["forecast_present"] for row in labeled) / len(labeled),
        "thesis_record_coverage_labeled": sum(row["thesis_record_present"] for row in labeled) / len(labeled),
        "decision_time_thesis_coverage_labeled": sum(row["thesis_causal_present"] for row in labeled) / len(labeled),
        "post_feature_only_thesis_records_labeled": sum(row["thesis_post_feature_only"] for row in labeled),
        "categorical_associations": categorical,
        "predictability": missingness_predictability(parent, rows),
        "observed_labeled_net_associations": {
            "forecast_present": observed_net_missingness(labeled, "forecast_present"),
            "thesis_record_present": observed_net_missingness(labeled, "thesis_record_present"),
            "decision_time_thesis_present": observed_net_missingness(labeled, "thesis_causal_present"),
        },
        "mnar_conclusion": "MCAR_REJECTABLE_IF_ASSOCIATIONS_OR_PREDICTABILITY_ARE_SIGNIFICANT; MNAR_NOT_IDENTIFIABLE_WITHOUT_MISSING_COUNTERFACTUAL_RETURNS",
    }
    candidates = ("A_COVERAGE_BINDING", "B_COST_AWARE_ABSTAIN", "C_PAIR_SIDE_CALIBRATION")
    stability = {}
    for candidate in candidates:
        evaluated = [window["candidates"][candidate] for window in windows if window["status"] == "EVALUATED"]
        stability[candidate] = {
            "evaluable_windows": len(evaluated),
            "positive_incremental_windows": sum(report["incremental_net_jpy"] > 0 for report in evaluated),
            "positive_lcb_windows": sum(report["paired_lcb_jpy"] is not None and report["paired_lcb_jpy"] > 0 for report in evaluated),
            "all_evaluable_accepted": bool(evaluated) and all(report["accepted"] for report in evaluated),
            "final_decision": "REJECT",
        }
    report = {
        "contract": "historical_learning_selection_rca_result_v1",
        "preregister_sha256": sha256(repo / "research/historical_learning_selection_rca/preregister_v1.json"),
        "frozen_bindings_verified": bindings,
        "holdout_used": False,
        "source_inventory": inventory,
        "shadow_cohort": {
            "rows": len(rows),
            "status_counts": dict(sorted(Counter(row["episode_status"] for row in rows).items())),
            "skip_rows_inferred": 0,
            "grade_counts": dict(sorted(Counter(row["reconstruction_grade"] for row in shadow_rows(rows)).items())),
        },
        "missingness": missingness,
        "windows": windows,
        "candidate_stability": stability,
        "x_contract": x_contract_status(repo),
        "overall_decision": "REJECT_MODEL_SELECTION_CHANGES",
        "policy_admission": "BLOCKED_ALL_ENTRY_COUNTERFACTUAL_AND_MARGIN_COVERAGE_INCOMPLETE",
        "permissions": {"read_only_inputs": True, "holdout_used": False, "paper": False, "live": False, "broker_order": False, "deploy": False},
    }
    return report, shadow_rows(rows), predictions


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("research/historical_learning_selection_rca"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report, shadow, predictions = run(repo)
    write_json(output / "selection_rca_report_v1.json", report)
    write_json(output / "source_inventory_v1.json", report["source_inventory"])
    write_jsonl(output / "shadow_decision_cohort_v1.jsonl", shadow)
    write_jsonl(output / "selection_predictions_v1.jsonl", predictions)
    print(json.dumps({"episodes": report["shadow_cohort"]["rows"], "decision": report["overall_decision"], "holdout_used": report["holdout_used"]}, sort_keys=True))


if __name__ == "__main__":
    main()
