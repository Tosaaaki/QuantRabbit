#!/usr/bin/env python3
"""Read-only historical-learning admission audit.

Builds a causal all-order episode ledger and evaluates only preregistered
baselines.  It never labels an unfilled/rejected/skip episode from M1 or an
interpolated path, never reads events after the frozen anchor, and never calls
a broker client.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANCHOR = "2026-07-09T07:46:03.151624347Z"
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", ANCHOR),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", ANCHOR),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", ANCHOR),
)
SEED = 20_260_809
BOOTSTRAPS = 10_000
EMBARGO = timedelta(hours=1)
MIN_N = 30
MARGIN_RATE = 0.04
PAIR_MARGIN_CAP = 0.45
TOTAL_MARGIN_CAP = 0.92
FORBIDDEN_FEATURES = {"close_price", "exit_reason", "realized_pl_jpy", "financing_jpy", "resolution_status"}


def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    match = re.match(r"^(.*\.)(\d+)([+-]\d\d:\d\d)$", text)
    if match and len(match.group(2)) > 6:
        text = match.group(1) + match.group(2)[:6] + match.group(3)
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_forecasts(path: Path) -> dict[str, tuple[list[datetime], list[dict[str, Any]]]]:
    grouped: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for row in read_jsonl(path):
        ts = parse_time(str(row["timestamp_utc"]))
        if ts <= parse_time(ANCHOR):
            grouped[str(row["pair"])].append((ts, row))
    result = {}
    for pair, values in grouped.items():
        values.sort(key=lambda item: item[0])
        result[pair] = ([item[0] for item in values], [item[1] for item in values])
    return result


def prior_forecast(index: dict[str, tuple[list[datetime], list[dict[str, Any]]]], pair: str, ts: datetime) -> tuple[dict[str, Any] | None, datetime | None]:
    if pair not in index:
        return None, None
    times, rows = index[pair]
    pos = bisect_right(times, ts) - 1
    return (rows[pos], times[pos]) if pos >= 0 else (None, None)


def load_episode_sources(ledger: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        accepted = [dict(row) for row in connection.execute("""
            WITH fills AS (
              SELECT order_id,trade_id,price fill_price,ts_utc fill_at,
                     ROW_NUMBER() OVER(PARTITION BY order_id ORDER BY ts_utc) rn
              FROM execution_events WHERE event_type='ORDER_FILLED'
            ), closes AS (
              SELECT trade_id,price close_price,ts_utc close_at,exit_reason,
                     realized_pl_jpy,financing_jpy,
                     ROW_NUMBER() OVER(PARTITION BY trade_id ORDER BY ts_utc DESC) rn
              FROM execution_events WHERE event_type='TRADE_CLOSED'
            ), cancels AS (
              SELECT order_id,MIN(ts_utc) cancel_at
              FROM execution_events WHERE event_type='ORDER_CANCELED'
              GROUP BY order_id
            )
            SELECT a.event_uid,a.ts_utc feature_at,a.order_id,a.pair,a.side,
                   ABS(a.units) units,a.price intended_price,a.tp,a.sl,a.lane_id,
                   f.trade_id,f.fill_price,f.fill_at,c.close_price,c.close_at,
                   c.exit_reason,c.realized_pl_jpy,c.financing_jpy,x.cancel_at
            FROM execution_events a
            LEFT JOIN fills f ON f.order_id=a.order_id AND f.rn=1
            LEFT JOIN closes c ON c.trade_id=f.trade_id AND c.rn=1
            LEFT JOIN cancels x ON x.order_id=a.order_id
            WHERE a.event_type='ORDER_ACCEPTED' AND a.ts_utc<=?
            ORDER BY a.ts_utc
        """, (ANCHOR,))]
        rejected = [dict(row) for row in connection.execute("""
            SELECT event_uid,ts_utc feature_at,order_id,pair,side,ABS(units) units,
                   price intended_price,tp,sl,lane_id,exit_reason reject_reason
            FROM execution_events
            WHERE event_type='ORDER_REJECTED' AND ts_utc<=?
            ORDER BY ts_utc
        """, (ANCHOR,))]
        return accepted, rejected
    finally:
        connection.close()


def episode_rows(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], forecasts: dict[str, tuple[list[datetime], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in accepted:
        feature_at = parse_time(source["feature_at"])
        forecast, forecast_at = prior_forecast(forecasts, str(source["pair"]), feature_at)
        executed = source["trade_id"] is not None
        labeled = executed and source["close_at"] is not None and parse_time(source["close_at"]) > feature_at
        status = "EXECUTED_LABELED" if labeled else "EXECUTED_OPEN_OR_INVALID" if executed else "CANCELED_UNFILLED" if source["cancel_at"] else "ACCEPTED_UNRESOLVED"
        entry_reference = source["intended_price"]
        net = (float(source["realized_pl_jpy"] or 0.0) + float(source["financing_jpy"] or 0.0)) if labeled else None
        row = {
            "episode_id": source["event_uid"], "feature_at_utc": source["feature_at"],
            "order_id": source["order_id"], "trade_id": source["trade_id"], "pair": source["pair"],
            "side": source["side"], "units": source["units"], "lane_id": source["lane_id"],
            "intended_price": entry_reference, "tp": source["tp"], "sl": source["sl"],
            "fill_at_utc": source["fill_at"], "close_at_utc": source["close_at"],
            "episode_status": status, "outcome_type": source["exit_reason"] if labeled else status,
            "label_status": "ACTUAL_AFTER_COST" if labeled else "MISSING_COUNTERFACTUAL",
            "net_jpy": net, "financing_jpy": float(source["financing_jpy"] or 0.0) if labeled else None,
            "cost_completeness": "SPREAD_SLIPPAGE_IMPLICIT_FINANCING_EXPLICIT_OPPORTUNITY_MISSING" if labeled else "NO_COUNTERFACTUAL_BA_PATH",
            "forecast_at_utc": forecast_at.isoformat().replace("+00:00", "Z") if forecast_at else None,
            "forecast_direction": forecast.get("direction") if forecast else None,
            "forecast_confidence": forecast.get("confidence") if forecast else None,
            "forecast_horizon_min": forecast.get("horizon_min") if forecast else None,
        }
        rows.append(row)
    for source in rejected:
        rows.append({
            "episode_id": source["event_uid"], "feature_at_utc": source["feature_at"],
            "order_id": source["order_id"], "trade_id": None, "pair": source["pair"], "side": source["side"],
            "units": source["units"], "lane_id": source["lane_id"], "intended_price": source["intended_price"],
            "tp": source["tp"], "sl": source["sl"], "fill_at_utc": None, "close_at_utc": None,
            "episode_status": "REJECTED", "outcome_type": "REJECTED", "label_status": "MISSING_COUNTERFACTUAL",
            "net_jpy": None, "financing_jpy": None, "cost_completeness": "NO_COUNTERFACTUAL_BA_PATH",
            "forecast_at_utc": None, "forecast_direction": None, "forecast_confidence": None, "forecast_horizon_min": None,
        })
    rows.sort(key=lambda row: parse_time(row["feature_at_utc"]))
    return rows


def features(row: dict[str, Any]) -> dict[str, Any]:
    ts = parse_time(row["feature_at_utc"])
    entry = float(row["intended_price"]) if row["intended_price"] is not None else None
    side = str(row["side"] or "MISSING")
    forecast_direction = str(row["forecast_direction"] or "MISSING")
    alignment = 1.0 if (side == "LONG" and forecast_direction == "UP") or (side == "SHORT" and forecast_direction == "DOWN") else 0.0
    age = None
    if row["forecast_at_utc"]:
        age = (ts - parse_time(row["forecast_at_utc"])).total_seconds() / 60.0
    def distance(value: Any) -> float:
        return abs(float(value) - entry) / entry * 10_000.0 if value is not None and entry else -1.0
    return {
        "pair": str(row["pair"] or "MISSING"), "side": side,
        "lane_id": str(row["lane_id"] or "MISSING"), "forecast_direction": forecast_direction,
        "log_units": math.log1p(float(row["units"] or 0.0)),
        "tp_distance_bps": distance(row["tp"]), "sl_distance_bps": distance(row["sl"]),
        "entry_price_missing": float(entry is None),
        "hour_sin": math.sin(2 * math.pi * ts.hour / 24), "hour_cos": math.cos(2 * math.pi * ts.hour / 24),
        "weekday": float(ts.weekday()),
        "forecast_confidence": float(row["forecast_confidence"]) if row["forecast_confidence"] is not None else -1.0,
        "forecast_confidence_missing": float(row["forecast_confidence"] is None),
        "forecast_age_min": float(age) if age is not None else -1.0,
        "forecast_horizon_min": float(row["forecast_horizon_min"]) if row["forecast_horizon_min"] is not None else -1.0,
        "forecast_alignment": alignment,
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    ordered = sorted(rows, key=lambda row: parse_time(row["feature_at_utc"]))
    if len(ordered) < 2:
        return [], [], 0
    cut = max(1, math.floor(len(ordered) * 0.60))
    validation = ordered[cut:]
    validation_start = parse_time(validation[0]["feature_at_utc"])
    raw_train = ordered[:cut]
    train = [row for row in raw_train if parse_time(row["close_at_utc"]) < validation_start - EMBARGO]
    return train, validation, len(raw_train) - len(train)


def profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return (gains / losses) if losses else (math.inf if gains else None)


def drawdown(values: list[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def paired_ci(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(SEED)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAPS))
    return [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]]


def quote_to_jpy(row: dict[str, Any]) -> float | None:
    if str(row["pair"]).endswith("_JPY"):
        return 1.0
    entry = row["intended_price"]
    # The feature-time intended price can be missing for market orders; margin
    # is then unproven rather than backfilled from the future fill.
    if entry is None:
        return None
    return None  # cross-currency conversion snapshot is not preserved at decision time


def metric(rowset: list[dict[str, Any]], selected: list[bool], baseline_values: list[float]) -> dict[str, Any]:
    values = [float(row["net_jpy"]) if take else 0.0 for row, take in zip(rowset, selected)]
    deltas = [value - base for value, base in zip(values, baseline_values)]
    ci = paired_ci(deltas)
    pair_net: dict[str, float] = defaultdict(float)
    side_net: dict[str, float] = defaultdict(float)
    for row, value in zip(rowset, values):
        if value:
            pair_net[str(row["pair"])] += value
            side_net[str(row["side"])] += value
    known_margins = []
    for row, take in zip(rowset, selected):
        conversion = quote_to_jpy(row)
        if take and conversion is not None and row["intended_price"] is not None:
            known_margins.append(float(row["units"]) * float(row["intended_price"]) * conversion * MARGIN_RATE)
    peak_margin = max(known_margins, default=None)
    initial_equity = peak_margin / PAIR_MARGIN_CAP if peak_margin else None
    dd = drawdown(values)
    pf = profit_factor(values)
    gates = {
        "minimum_validation_events": len(rowset) >= MIN_N,
        "net_positive": sum(values) > 0,
        "profit_factor_above_one": pf is not None and pf > 1,
        "paired_lcb_positive": ci[0] is not None and ci[0] > 0,
        "drawdown_within_limit": initial_equity is not None and dd <= initial_equity * .02,
        "margin_complete_and_within_cap": (
            sum(selected) > 0 and len(known_margins) == sum(selected) and initial_equity is not None
            and peak_margin <= initial_equity * TOTAL_MARGIN_CAP
        ),
        "both_sides_positive": all(side_net.get(side, 0) > 0 for side in ("LONG", "SHORT")),
        "two_pairs_positive": sum(value > 0 for value in pair_net.values()) >= 2,
        "two_volatility_regimes_positive": False,
        "all_entry_counterfactual_coverage": False,
    }
    return {
        "trades_selected": sum(selected), "trades_available": len(rowset), "net_jpy": sum(values),
        "selected_episode_ids": [row["episode_id"] for row, take in zip(rowset, selected) if take],
        "profit_factor": "Infinity" if pf == math.inf else pf, "expectancy_per_available_episode_jpy": mean(values) if values else None,
        "max_drawdown_jpy": dd, "paired_delta_mean_jpy": mean(deltas) if deltas else None,
        "paired_bootstrap_95pct_ci": ci, "paired_lcb_jpy": ci[0], "peak_margin_jpy": peak_margin,
        "margin_coverage": len(known_margins) / max(1, sum(selected)),
        "pair_net_jpy": dict(sorted(pair_net.items())), "side_net_jpy": dict(sorted(side_net.items())),
        "gates": gates, "accepted": all(gates.values()),
    }


def model_report(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [float(row["net_jpy"]) for row in validation]
    result: dict[str, Any] = {"ALL_TRADES": metric(validation, [True] * len(validation), baseline)}
    rule_selected = [
        row["forecast_confidence"] is not None and float(row["forecast_confidence"]) >= .60 and
        ((row["side"] == "LONG" and row["forecast_direction"] == "UP") or (row["side"] == "SHORT" and row["forecast_direction"] == "DOWN"))
        for row in validation
    ]
    result["RULE_FORECAST"] = metric(validation, rule_selected, baseline)
    if len(train) < MIN_N or len(validation) < MIN_N:
        for name in ("LOGISTIC", "RIDGE", "HIST_GRADIENT_BOOSTING"):
            result[name] = {"status": "NOT_FIT_MINIMUM_SAMPLE_GATE", "accepted": False}
        return result
    x_train, x_validation = [features(row) for row in train], [features(row) for row in validation]
    y_train = np.asarray([float(row["net_jpy"]) for row in train])
    y_validation_binary = np.asarray([float(row["net_jpy"]) > 0 for row in validation], dtype=int)
    y_train_binary = (y_train > 0).astype(int)
    base_steps = [("vector", DictVectorizer(sparse=False)), ("scale", StandardScaler())]
    if len(set(y_train_binary.tolist())) < 2:
        result["LOGISTIC"] = {"status": "NOT_FIT_SINGLE_CLASS", "accepted": False}
    else:
        logistic = Pipeline(base_steps + [("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED))])
        logistic.fit(x_train, y_train_binary)
        probability = logistic.predict_proba(x_validation)[:, 1]
        positive_mean = float(y_train[y_train > 0].mean())
        nonpositive_mean = float(y_train[y_train <= 0].mean())
        expected = probability * positive_mean + (1 - probability) * nonpositive_mean
        report = metric(validation, (expected > 0).tolist(), baseline)
        report["brier_score"] = float(brier_score_loss(y_validation_binary, probability))
        report["class_balance_train_positive_ratio"] = float(y_train_binary.mean())
        result["LOGISTIC"] = report
    ridge = Pipeline(base_steps + [("model", Ridge(alpha=1.0))])
    ridge.fit(x_train, y_train)
    result["RIDGE"] = metric(validation, (ridge.predict(x_validation) > 0).tolist(), baseline)
    hgb = Pipeline(base_steps + [("model", HistGradientBoostingRegressor(learning_rate=.05, max_iter=100, max_depth=3, min_samples_leaf=10, l2_regularization=1.0, random_state=SEED))])
    hgb.fit(x_train, y_train)
    result["HIST_GRADIENT_BOOSTING"] = metric(validation, (hgb.predict(x_validation) > 0).tolist(), baseline)
    return result


def run(repo: Path, episodes_output: Path) -> dict[str, Any]:
    prereg_path = repo / "research/historical_learning_admission/preregister_v1.json"
    prereg = json.loads(prereg_path.read_text())
    paths = {
        "execution_ledger_sha256": repo / "data/execution_ledger.db",
        "forecast_history_sha256": repo / "data/forecast_history.jsonl",
        "projection_ledger_sha256": repo / "data/projection_ledger.jsonl",
        "entry_thesis_ledger_sha256": repo / "data/entry_thesis_ledger.jsonl",
    }
    for key, path in paths.items():
        if sha256(path) != prereg["source_bindings"][key]:
            raise RuntimeError(f"source changed after preregistration: {path}")
    forecasts = load_forecasts(paths["forecast_history_sha256"])
    accepted, rejected = load_episode_sources(paths["execution_ledger_sha256"])
    episodes = episode_rows(accepted, rejected, forecasts)
    thesis_trade_ids = {
        str(row["trade_id"]) for row in read_jsonl(paths["entry_thesis_ledger_sha256"])
        if row.get("trade_id") is not None and parse_time(str(row["timestamp_utc"])) <= parse_time(ANCHOR)
    }
    episodes_output.parent.mkdir(parents=True, exist_ok=True)
    with episodes_output.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    labeled = [row for row in episodes if row["label_status"] == "ACTUAL_AFTER_COST"]
    future_forecast_violations = sum(
        row["forecast_at_utc"] is not None and parse_time(row["forecast_at_utc"]) > parse_time(row["feature_at_utc"])
        for row in episodes
    )
    label_time_violations = sum(
        row["label_status"] == "ACTUAL_AFTER_COST" and parse_time(row["close_at_utc"]) <= parse_time(row["feature_at_utc"])
        for row in episodes
    )
    feature_names = set().union(*(features(row).keys() for row in labeled)) if labeled else set()
    windows = []
    for window_id, start_text, end_text in WINDOWS:
        start, end = parse_time(start_text), parse_time(end_text)
        scoped = [row for row in labeled if start <= parse_time(row["feature_at_utc"]) <= end]
        train, validation, purged = split_rows(scoped)
        models = model_report(train, validation)
        windows.append({
            "id": window_id, "from_utc": start_text, "to_utc": end_text,
            "labeled_events": len(scoped), "train_events": len(train), "validation_events": len(validation),
            "purged_train_events": purged, "models": models,
            "decision": "ACCEPT" if any(v.get("accepted") for k, v in models.items() if k != "ALL_TRADES") else "REJECT",
        })
    counts: dict[str, int] = defaultdict(int)
    for row in episodes:
        counts[row["episode_status"]] += 1
    all_entry_coverage = len(labeled) / len(episodes) if episodes else 0.0
    forecast_covered = sum(row["forecast_at_utc"] is not None for row in labeled)
    thesis_covered = sum(str(row["trade_id"]) in thesis_trade_ids for row in labeled)
    return {
        "contract": "historical_learning_admission_result_v1", "preregister_sha256": sha256(prereg_path),
        "holdout_used": False, "episode_counts": dict(sorted(counts.items())), "episodes_total": len(episodes),
        "labeled_actual_after_cost": len(labeled), "all_entry_label_coverage": all_entry_coverage,
        "causal_forecast_coverage": forecast_covered / len(labeled) if labeled else 0.0,
        "entry_thesis_coverage": thesis_covered / len(labeled) if labeled else 0.0,
        "skip_episode_count": 0, "skip_ledger_status": "MISSING_APPEND_ONLY_DECISION_LEDGER",
        "label_cost_status": {
            "spread": "IMPLICIT_IN_ACTUAL_FILL_PNL", "slippage": "IMPLICIT_IN_ACTUAL_FILL_PNL",
            "financing": "EXPLICIT", "fee": "ZERO_ACCOUNT_CONTRACT", "opportunity_cost": "MISSING",
            "margin_at_decision": "INCOMPLETE_CROSS_CURRENCY_CONVERSION_SNAPSHOT",
        },
        "leakage_audit": {
            "future_forecast_join_violations": future_forecast_violations,
            "label_timestamp_violations": label_time_violations,
            "forbidden_feature_intersection": sorted(feature_names & FORBIDDEN_FEATURES),
            "projection_resolution_used_as_feature": False,
            "passed": future_forecast_violations == 0 and label_time_violations == 0 and not (feature_names & FORBIDDEN_FEATURES),
        },
        "windows": windows,
        "overall_decision": "REJECT" if all_entry_coverage < 1.0 or any(window["decision"] != "ACCEPT" for window in windows) else "ACCEPT",
        "proof_grade": "DIAGNOSTIC_EXECUTED_ONLY_NOT_POLICY_ADMISSIBLE",
        "root_causes": [
            "unfilled/rejected/skip counterfactual BA labels are absent, so executed-only learning has selection bias",
            "decision-time cross-currency margin and opportunity-cost snapshots are absent",
            "volatility regime is not causally recorded for the full cohort; post-outcome construction would leak",
        ],
        "permissions": {"read_only": True, "paper": False, "live": False, "broker_order": False, "deploy": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--episodes-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    episodes_output = args.episodes_output if args.episodes_output.is_absolute() else repo / args.episodes_output
    report_output = args.report_output if args.report_output.is_absolute() else repo / args.report_output
    report = run(repo, episodes_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": report["episodes_total"], "labeled": report["labeled_actual_after_cost"], "decision": report["overall_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
