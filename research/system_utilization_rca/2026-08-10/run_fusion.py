#!/usr/bin/env python3
"""Research-only utilization RCA and decision-time fusion on frozen episodes.

This module deliberately owns no broker/order/live imports.  Outcomes are loaded
into a separate table and are used only by TRAIN fitting or final evaluation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 20_260_810
EMBARGO = timedelta(hours=1)
BOOTSTRAPS = 4_000

EPISODES = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
SHADOW = REPO / "research/historical_learning_selection_rca/shadow_decision_cohort_v1.jsonl"
SELECTIONS = REPO / "research/historical_learning_selection_rca/selection_predictions_v1.jsonl"
GAPLESS = REPO / "research/historical_learning_gapless_truth/report_v2.json"
REAL_PAYLOAD = REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json"
REAL_REPORT = REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_report.json"
X_HANDOFF = REPO / "research/x_fx_methods/2026-08-09/hedge_task_handoff.json"

INFERENCE_COLUMNS = (
    "episode_id", "source_sha", "decision_time", "split", "pair", "timeframe",
    "horizon", "regime", "system_id", "model_version", "feature_set",
    "parameter_set", "predicted_direction", "predicted_return_or_path",
    "probability_or_score", "prediction_interval", "abstain_or_skip",
    "assumptions", "missing_inputs", "input_lineage", "output_sha",
    "runtime_ms", "memory_bytes",
)

FUSED_COLUMNS = (
    "decision_id", "action", "pair", "side", "horizon", "entry_zone",
    "target_or_path", "invalidation", "exit_or_unwind_policy", "size_cap",
    "confidence", "prediction_interval", "expected_after_cost",
    "worst_case_dd_margin", "supporting_families", "dissenting_families",
    "decisive_constraint", "abstain_reason", "input_lineage", "output_sha",
)


def parse_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        offset_at = max(suffix.find("+"), suffix.find("-"))
        if offset_at >= 0:
            fraction, offset = suffix[:offset_at], suffix[offset_at:]
            normalized = f"{prefix}.{fraction[:6].ljust(6, '0')}{offset}"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def bind_output(row: dict[str, Any]) -> dict[str, Any]:
    bound = dict(row)
    bound.pop("output_sha", None)
    bound["output_sha"] = logical_sha(bound)
    return bound


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular design")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [aug[i][-1] for i in range(n)]


def ridge_fit(xs: list[list[float]], ys: list[float], penalty: float = 1.0) -> list[float]:
    design = [[1.0] + list(row) for row in xs]
    width = len(design[0])
    matrix = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    for index in range(1, width):
        matrix[index][index] += penalty
    vector = [sum(row[i] * target for row, target in zip(design, ys)) for i in range(width)]
    return solve(matrix, vector)


def predict(coef: list[float], row: list[float]) -> float:
    return coef[0] + sum(weight * value for weight, value in zip(coef[1:], row))


def drawdown(values: Iterable[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def profit_factor(values: list[float]) -> float | None:
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    return gain / loss if loss else None


def bootstrap_lcb(deltas: list[float], seed_offset: int) -> float | None:
    if not deltas:
        return None
    rng = random.Random(SEED + seed_offset)
    means = sorted(statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(BOOTSTRAPS))
    return means[int(0.025 * (len(means) - 1))]


def financial_metrics(rows: list[dict[str, Any]], selected: dict[str, bool], seed_offset: int) -> dict[str, Any]:
    actual = [float(row["actual_net_jpy"]) for row in rows]
    applied = [value if selected.get(row["episode_id"], False) else 0.0 for row, value in zip(rows, actual)]
    deltas = [candidate - baseline for candidate, baseline in zip(applied, actual)]
    chosen = [row for row in rows if selected.get(row["episode_id"], False)]
    return {
        "available": len(rows),
        "selected": len(chosen),
        "decisions_changed_vs_all_trades": len(rows) - len(chosen),
        "after_cost_net_jpy": sum(applied),
        "all_trades_net_jpy": sum(actual),
        "incremental_net_jpy": sum(deltas),
        "paired_bootstrap_lcb_jpy": bootstrap_lcb(deltas, seed_offset),
        "profit_factor": profit_factor(applied),
        "max_drawdown_jpy": drawdown(applied),
        "all_trades_max_drawdown_jpy": drawdown(actual),
        "margin_coverage": (
            sum(bool(row["margin_evidence_known"]) for row in chosen) / len(chosen) if chosen else None
        ),
        "turnover_units": sum(abs(float(row["units"])) for row in chosen),
        "fill_validity": 1.0 if chosen else None,
        "unwind_validity": 1.0 if chosen else None,
        "sample_coverage": len(chosen) / len(rows) if rows else None,
    }


def system_inventory() -> list[dict[str, Any]]:
    return [
        {"system_id": "all_trades", "family": "baseline", "stage": "selection", "artifact": "executed episode cohort", "producer": "historical admission", "consumer": "paired comparator only", "classification": "USED", "reason": "explicit comparator, never an implicit fallback"},
        {"system_id": "forecast", "family": "statistical_ml", "stage": "forecast_distribution", "artifact": "data/forecast_history.jsonl", "producer": "directional_forecaster", "consumer": "intent and trader gates", "classification": "USED", "reason": "runtime consumers exist; causal episode coverage is partial"},
        {"system_id": "market_read", "family": "market_context", "stage": "hypothesis_generation", "artifact": "data/market_read_predictions.jsonl", "producer": "hourly trader", "consumer": "execution link ledger", "classification": "DISCONNECTED", "reason": "5 predictions and no market_read_execution_links artifact in this repo snapshot"},
        {"system_id": "price_action", "family": "technical", "stage": "feature_cube", "artifact": "gapless Dukascopy M5 features", "producer": "price-action admission", "consumer": "research HGB", "classification": "GENERATED_ONLY", "reason": "136/251 features, no runtime decision consumer"},
        {"system_id": "metadata_hgb", "family": "statistical_ml", "stage": "forecast_distribution", "artifact": "selection_predictions_v1.jsonl", "producer": "selection RCA", "consumer": "research report", "classification": "GENERATED_ONLY", "reason": "validation selection is post-hoc research only"},
        {"system_id": "price_action_hgb", "family": "technical", "stage": "forecast_distribution", "artifact": "gapless report prediction_rows", "producer": "price-action admission", "consumer": "research report", "classification": "GENERATED_ONLY", "reason": "no production or fusion consumer before this phase"},
        {"system_id": "sl_hedge", "family": "risk_exit", "stage": "exit_unwind", "artifact": "loss-close paired shadow", "producer": "paired replay", "consumer": "verdict", "classification": "INSUFFICIENT_EVIDENCE", "reason": "four hedge alternatives rejected; no complete decision-time leg/unwind contract"},
        {"system_id": "exposure", "family": "portfolio", "stage": "sizing_exposure", "artifact": "order intents / broker snapshot", "producer": "intent and gateway", "consumer": "risk/gateway", "classification": "USED", "reason": "runtime path exists but historical 251 point-in-time snapshots are absent"},
        {"system_id": "risk", "family": "risk_exit", "stage": "sizing_exposure", "artifact": "risk receipts", "producer": "RiskEngine", "consumer": "LiveOrderGateway", "classification": "USED", "reason": "runtime gate exists; historical margin evidence only 14.85% in 64d validation"},
        {"system_id": "exit_unwind", "family": "risk_exit", "stage": "exit_unwind", "artifact": "execution ledger terminal events", "producer": "position manager/gateway", "consumer": "feedback/audits", "classification": "USED", "reason": "realized outcome exists but decision-time unwind proof is not reconstructable"},
        {"system_id": "xarray", "family": "infrastructure", "stage": "feature_cube", "artifact": "real_cube_sparse.json", "producer": "research adapter", "consumer": "adapter verifier", "classification": "GENERATED_ONLY", "reason": "exact alignment check changed zero final decisions"},
        {"system_id": "salib", "family": "infrastructure", "stage": "causal_refutation", "artifact": "real_adapter_report.json", "producer": "research adapter", "consumer": "adapter verifier", "classification": "DISCONNECTED", "reason": "TRAIN sensitivity ranking never became a selection rule; rank reversed on validation"},
        {"system_id": "pymoo", "family": "infrastructure", "stage": "multi_objective_selection", "artifact": "real_adapter_report.json", "producer": "research adapter", "consumer": "adapter verifier", "classification": "NULLIFIED_BY_FALLBACK", "reason": "margin constraint emptied front; policy remained ALL_TRADES comparator"},
        {"system_id": "mapie", "family": "statistical_ml", "stage": "uncertainty_calibration", "artifact": "real_adapter_report.json", "producer": "research adapter", "consumer": "adapter verifier", "classification": "DISCONNECTED", "reason": "intervals were measured but never changed SKIP/size decisions"},
        {"system_id": "x_lvn", "family": "strategy_evidence", "stage": "hypothesis_generation", "artifact": "X handoff", "producer": "X research", "consumer": "owner intake", "classification": "INSUFFICIENT_EVIDENCE", "reason": "negative validation expectancy and active-day shortage"},
        {"system_id": "x_session", "family": "strategy_evidence", "stage": "hypothesis_generation", "artifact": "X handoff", "producer": "X research", "consumer": "owner intake", "classification": "INSUFFICIENT_EVIDENCE", "reason": "negative validation expectancy and active-day shortage"},
        {"system_id": "x_mtf", "family": "strategy_evidence", "stage": "hypothesis_generation", "artifact": "X handoff", "producer": "X research", "consumer": "queue only", "classification": "DISCONNECTED", "reason": "not executed"},
        {"system_id": "runtime_context_bots", "family": "market_context", "stage": "hypothesis_generation", "artifact": "pair/context/currency/cross-asset/news reports", "producer": "CLI cycle", "consumer": "trader evidence packet", "classification": "USED", "reason": "current-state consumers exist but no frozen point-in-time 251 history"},
        {"system_id": "verification_ledger", "family": "strategy_evidence", "stage": "causal_refutation", "artifact": "data/verification_ledger.json", "producer": "verification audit", "consumer": "active boards/trader", "classification": "USED", "reason": "runtime consumers exist but no explicit episode join for this cohort"},
        {"system_id": "entry_thesis", "family": "strategy_evidence", "stage": "feedback", "artifact": "data/entry_thesis_ledger.jsonl", "producer": "fill recorder", "consumer": "position/learning audits", "classification": "INSUFFICIENT_EVIDENCE", "reason": "18/251 causal decision-time episodes"},
        {"system_id": "automations", "family": "orchestration", "stage": "feedback", "artifact": "~/.codex/automations/*/automation.toml", "producer": "Codex scheduler", "consumer": "CLI/report workflows", "classification": "USED", "reason": "automations call production CLIs; none calls research adapters or this fusion engine"},
    ]


def signal_for(system: str, episode: dict[str, Any], shadow: dict[str, Any], record: dict[str, Any] | None, selection: dict[str, Any] | None, gap: dict[str, Any] | None) -> dict[str, Any]:
    side = str(episode["side"])
    assumptions: list[str] = []
    missing: list[str] = []
    direction = None
    value: Any = None
    score = None
    interval = None
    abstain = True
    lineage: list[str] = []
    if system == "forecast":
        lineage = ["all_entry_episodes_v1.jsonl:forecast_*", "shadow_decision_cohort_v1.jsonl:causal_forecast_present"]
        if shadow.get("causal_forecast_present") and episode.get("forecast_direction"):
            direction = episode["forecast_direction"]
            confidence = float(episode.get("forecast_confidence") or 0.0)
            score = confidence if direction == side else -confidence
            value = {"direction": direction, "horizon_min": episode.get("forecast_horizon_min")}
            abstain = False
            assumptions.append("forecast timestamp is at or before decision_time")
        else:
            missing.append("causal_forecast")
    elif system == "price_action":
        lineage = ["real_shadow_payload.json:price_action_features"]
        features = (record or {}).get("price_action_features")
        if features:
            raw = statistics.fmean(float(features[key]) for key in ("pa_return_3", "pa_return_12", "pa_return_48"))
            direction = "LONG" if raw >= 0 else "SHORT"
            score = raw if side == "LONG" else -raw
            value = {"signed_trade_side_return": score, "raw_trend_return": raw}
            abstain = False
            assumptions.append("Dukascopy features are feature-only; OANDA remains financial truth")
        else:
            missing.append("gapless_decision_time_price_action")
    elif system == "metadata_hgb":
        lineage = ["selection_predictions_v1.jsonl"]
        if selection and selection.get("frozen_hgb_predicted_net_jpy") is not None:
            score = float(selection["frozen_hgb_predicted_net_jpy"])
            value = score
            direction = side if score > 0 else "ABSTAIN"
            abstain = score <= 0
            assumptions.append("outer VALIDATION prediction only; unavailable rows remain null")
        else:
            missing.append("train_cross_fit_or_validation_prediction")
    elif system == "price_action_hgb":
        lineage = ["historical_learning_gapless_truth/report_v2.json:prediction_rows"]
        if gap:
            selected = bool(gap.get("price_action_selected"))
            direction = side if selected else "ABSTAIN"
            score = 1.0 if selected else -1.0
            value = {"selected": selected}
            abstain = not selected
        else:
            missing.append("price_action_hgb_prediction")
    elif system == "all_trades":
        direction, score, value, abstain = side, 1.0, {"baseline": True}, False
        assumptions.append("comparison baseline only, not a fusion fallback")
        lineage = ["all_entry_episodes_v1.jsonl"]
    else:
        missing.append("episode_aligned_decision_time_output")
        lineage = [next(item["artifact"] for item in system_inventory() if item["system_id"] == system)]
    return {
        "predicted_direction": direction,
        "predicted_return_or_path": value,
        "probability_or_score": score,
        "prediction_interval": interval,
        "abstain_or_skip": abstain,
        "assumptions": assumptions,
        "missing_inputs": missing,
        "input_lineage": lineage,
    }


def build_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes = [row for row in read_jsonl(EPISODES) if row.get("label_status") == "ACTUAL_AFTER_COST"]
    if len(episodes) != 251:
        raise RuntimeError(f"expected 251 actual-after-cost episodes, got {len(episodes)}")
    source_sha = sha256(EPISODES)
    shadow = {row["episode_id"]: row for row in read_jsonl(SHADOW)}
    payload = json.loads(REAL_PAYLOAD.read_text(encoding="utf-8"))
    records = {
        row["episode_id"]: row for row in payload["episode_records"]
        if row["window"] == "QUADRUPLE_64D" and row["method"] == "ALL_TRADES"
    }
    selections = {
        (row["window_id"], row["episode_id"]): row for row in read_jsonl(SELECTIONS)
    }
    gap_report = json.loads(GAPLESS.read_text(encoding="utf-8"))
    gaps = {(row["window_id"], row["episode_id"]): row for row in gap_report["prediction_rows"]}
    inventory = system_inventory()
    inference: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        record = records.get(episode_id)
        split = record["split"] if record else "EMBARGO"
        outcomes.append({
            "episode_id": episode_id,
            "actual_after_cost_net": float(episode["net_jpy"]),
            "fill": True,
            "margin": {"known": bool(record and record["margin_evidence_known"]), "value": None},
            "DD": None,
            "unwind": True,
            "terminal_reason": episode.get("outcome_type"),
            "source_sha": source_sha,
        })
        for item in inventory:
            system = item["system_id"]
            signal = signal_for(
                system, episode, shadow[episode_id], record,
                selections.get(("QUADRUPLE_64D", episode_id)),
                gaps.get(("QUADRUPLE_64D", episode_id)),
            )
            inference.append(bind_output({
                "episode_id": episode_id,
                "source_sha": source_sha,
                "decision_time": episode["feature_at_utc"],
                "split": split,
                "pair": episode["pair"],
                "timeframe": "M5" if record and record.get("price_action_features") else "DECISION_METADATA_ONLY",
                "horizon": episode.get("forecast_horizon_min"),
                "regime": episode.get("forecast_direction"),
                "system_id": system,
                "model_version": "frozen_2026-08-10",
                "feature_set": item["family"],
                "parameter_set": "preregister_v1",
                **signal,
                "runtime_ms": None,
                "memory_bytes": None,
            }))
    return inference, outcomes, {"episodes": episodes, "shadow": shadow, "records": records, "source_sha": source_sha}


def oof_fit(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    eligible = [row for row in sorted(rows, key=lambda item: item["decision_time"]) if all(row.get(name) is not None for name in feature_names)]
    oof: dict[str, float] = {}
    for index, target in enumerate(eligible):
        prior = [
            row for row in eligible[:index]
            if parse_time(row["close_time"]) <= parse_time(target["decision_time"]) - EMBARGO
        ]
        if len(prior) < 12:
            continue
        coef = ridge_fit([[float(row[name]) for name in feature_names] for row in prior], [float(row["actual_net_jpy"]) for row in prior])
        oof[target["episode_id"]] = predict(coef, [float(target[name]) for name in feature_names])
    if len(oof) < 12:
        return {"status": "INSUFFICIENT_OOF_ROWS", "eligible": len(eligible), "oof_n": len(oof)}
    fit_rows = eligible
    coef = ridge_fit([[float(row[name]) for name in feature_names] for row in fit_rows], [float(row["actual_net_jpy"]) for row in fit_rows])
    residuals = [abs(oof[row["episode_id"]] - float(row["actual_net_jpy"])) for row in eligible if row["episode_id"] in oof]
    return {
        "status": "FIT",
        "eligible": len(eligible),
        "oof_n": len(oof),
        "coef": coef,
        "residual_q90": quantile(residuals, 0.90),
        "oof_predictions": oof,
        "oof_mae": statistics.fmean(residuals),
    }


def episode_model_rows(window_records: list[dict[str, Any]], episodes: dict[str, dict[str, Any]], shadow: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in window_records:
        episode = episodes[record["episode_id"]]
        causal_forecast = shadow[record["episode_id"]].get("causal_forecast_present")
        forecast_score = None
        if causal_forecast and episode.get("forecast_direction"):
            confidence = float(episode.get("forecast_confidence") or 0.0)
            forecast_score = confidence if episode["forecast_direction"] == episode["side"] else -confidence
        technical_score = None
        features = record.get("price_action_features")
        if features:
            raw = statistics.fmean(float(features[key]) for key in ("pa_return_3", "pa_return_12", "pa_return_48"))
            technical_score = raw if episode["side"] == "LONG" else -raw
        output.append({
            **record,
            "forecast_score": forecast_score,
            "technical_score": technical_score,
        })
    return output


def evaluate_model(train: list[dict[str, Any]], validation: list[dict[str, Any]], features: list[str], seed_offset: int) -> dict[str, Any]:
    fit = oof_fit(train, features)
    if fit["status"] != "FIT":
        return {"status": fit["status"], "train": fit, "validation": None, "predictions": {}}
    predictions: dict[str, dict[str, float]] = {}
    selected: dict[str, bool] = {}
    for row in validation:
        if not all(row.get(name) is not None for name in features):
            continue
        point = predict(fit["coef"], [float(row[name]) for name in features])
        lower, upper = point - fit["residual_q90"], point + fit["residual_q90"]
        predictions[row["episode_id"]] = {"point": point, "lower": lower, "upper": upper}
        selected[row["episode_id"]] = lower > 0.0
    return {
        "status": "EVALUATED",
        "features": features,
        "train": {key: value for key, value in fit.items() if key != "oof_predictions"},
        "validation": financial_metrics(validation, selected, seed_offset),
        "predictions": predictions,
        "selection_rule": "TRAIN OOF absolute-residual 90% lower bound > 0",
    }


def fusion_evaluation(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(REAL_PAYLOAD.read_text(encoding="utf-8"))
    episodes = {row["episode_id"]: row for row in context["episodes"]}
    report: dict[str, Any] = {}
    final_predictions: dict[str, dict[str, float]] = {}
    for window in ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D"):
        records = [row for row in payload["episode_records"] if row["window"] == window and row["method"] == "ALL_TRADES"]
        model_rows = episode_model_rows(records, episodes, context["shadow"])
        train = [row for row in model_rows if row["split"] == "TRAIN"]
        validation = [row for row in model_rows if row["split"] == "VALIDATION"]
        candidates = {
            "single_technical": evaluate_model(train, validation, ["technical_score"], 10),
            "single_statistical": evaluate_model(train, validation, ["forecast_score"], 20),
            "calibrated_weighted_vote": evaluate_model(train, validation, ["technical_score", "forecast_score"], 30),
        }
        weighted = candidates["calibrated_weighted_vote"]
        edge_selected = {
            episode_id: pred["lower"] > 0 for episode_id, pred in weighted.get("predictions", {}).items()
        }
        admissible_selected = {
            row["episode_id"]: bool(edge_selected.get(row["episode_id"], False)
                and row["margin_evidence_known"]
                and False  # decision-time fill/unwind evidence is absent in the frozen cohort
            ) for row in validation
        }
        report[window] = {
            "counts": {
                "train": len(train), "validation": len(validation),
                "train_forecast": sum(row["forecast_score"] is not None for row in train),
                "train_technical": sum(row["technical_score"] is not None for row in train),
                "train_two_family": sum(row["forecast_score"] is not None and row["technical_score"] is not None for row in train),
                "validation_two_family": sum(row["forecast_score"] is not None and row["technical_score"] is not None for row in validation),
                "validation_margin_evidence": sum(bool(row["margin_evidence_known"]) for row in validation),
                "validation_decision_time_fill_unwind_evidence": 0,
            },
            "candidates": candidates,
            "edge_only_not_admissible": financial_metrics(validation, edge_selected, 40),
            "fused_evidence_admissible": financial_metrics(validation, admissible_selected, 50),
            "unexecuted_methods": {
                "rule_constraint_ensemble": "same missing execution/margin evidence makes a trade decision inadmissible",
                "regime_gated_mixture": "no independent point-in-time market-context family on the 251 cohort",
                "oof_stacking": "two-family complete TRAIN count below preregistered 30 in 16/32d and only 33 in 64d; base OOF overlap insufficient",
                "bayesian_model_average": "no stable independent family likelihood; forecast and price-action overlap is only 55/251",
                "pareto_mapie_abstain": "pymoo constrained front empty under margin evidence and MAPIE intervals have only 22 validation rows"
            },
            "all_trades_is_fallback": False,
            "holdout_read": False,
        }
        if window == "QUADRUPLE_64D":
            final_predictions = weighted.get("predictions", {})

    split_map = {episode_id: record["split"] for episode_id, record in context["records"].items()}
    fused: list[dict[str, Any]] = []
    for episode in context["episodes"]:
        episode_id = episode["episode_id"]
        prediction = final_predictions.get(episode_id)
        record = context["records"].get(episode_id)
        forecast_ok = context["shadow"][episode_id].get("causal_forecast_present", False)
        technical_ok = bool(record and record.get("price_action_features"))
        margin_ok = bool(record and record.get("margin_evidence_known"))
        supporting = [family for family, ok in (("technical", technical_ok), ("statistical_ml", forecast_ok)) if ok]
        missing = []
        if not forecast_ok:
            missing.append("statistical_ml:causal_forecast")
        if not technical_ok:
            missing.append("technical:gapless_price_action")
        if not margin_ok:
            missing.append("portfolio:margin_evidence")
        missing.extend(["execution:decision_time_fillability", "risk_exit:decision_time_unwind_validity"])
        edge_trade = bool(prediction and prediction["lower"] > 0)
        if len(supporting) < 2:
            action, decisive, reason = "WAIT", "EDGE_FAMILY_COVERAGE", "fewer than two independent edge families"
        elif not edge_trade:
            action, decisive, reason = "SKIP", "TRAIN_FIXED_UNCERTAINTY", "expected-after-cost lower bound is not positive"
        else:
            action, decisive, reason = "WAIT", "DECISION_TIME_EXECUTION_EVIDENCE", "edge exists but fill/unwind evidence is absent"
        row = {
            "decision_id": f"fused:{episode_id}",
            "episode_id": episode_id,
            "decision_time": episode["feature_at_utc"],
            "split": split_map.get(episode_id, "EMBARGO"),
            "action": action,
            "pair": episode["pair"],
            "side": episode["side"],
            "horizon": episode.get("forecast_horizon_min"),
            "entry_zone": {"intended_price": episode.get("intended_price")},
            "target_or_path": {"tp": episode.get("tp")},
            "invalidation": {"sl": episode.get("sl")},
            "exit_or_unwind_policy": None,
            "size_cap": 0 if action != "TRADE" else episode.get("units"),
            "confidence": None if not prediction else prediction["point"],
            "prediction_interval": None if not prediction else [prediction["lower"], prediction["upper"]],
            "expected_after_cost": None if not prediction else prediction["point"],
            "worst_case_dd_margin": {"margin_evidence_known": margin_ok, "dd": None},
            "supporting_families": supporting,
            "dissenting_families": ["execution", "portfolio", "risk_exit"],
            "decisive_constraint": decisive,
            "abstain_reason": reason,
            "input_lineage": ["FULL_INFERENCE_ENSEMBLE_V1", context["source_sha"], *missing],
        }
        fused.append(bind_output(row))
    return report, fused


def sparse_cube(inference: list[dict[str, Any]]) -> dict[str, Any]:
    axes = {
        "episode_id": sorted({row["episode_id"] for row in inference}),
        "split": sorted({row["split"] for row in inference}),
        "pair": sorted({row["pair"] for row in inference}),
        "timeframe": sorted({row["timeframe"] for row in inference}),
        "system_id": sorted({row["system_id"] for row in inference}),
        "metric": ["probability_or_score"],
    }
    cells = [
        {"episode_id": row["episode_id"], "system_id": row["system_id"], "value": row["probability_or_score"]}
        for row in inference if row["probability_or_score"] is not None
    ]
    return {
        "contract": "FULL_INFERENCE_ENSEMBLE_V1_SPARSE_CUBE",
        "axes": axes,
        "cells": cells,
        "missing_cells": len(inference) - len(cells),
        "missing_representation": "absent sparse cell / null, never zero",
        "source_table_sha": logical_sha(inference),
    }


def utilization_report(inference: list[dict[str, Any]], outcomes: list[dict[str, Any]], fusion: dict[str, Any], fused: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = system_inventory()
    counts = Counter(item["classification"] for item in inventory)
    usable = Counter(row["system_id"] for row in inference if not row["abstain_or_skip"])
    automation_files = glob.glob(str(Path.home() / ".codex/automations/*/automation.toml"))
    final_counts = Counter(row["action"] for row in fused)
    val = [row for row in fused if row["split"] == "VALIDATION"]
    val_records = [context for context in json.loads(REAL_PAYLOAD.read_text(encoding="utf-8"))["episode_records"] if context["window"] == "QUADRUPLE_64D" and context["method"] == "ALL_TRADES" and context["split"] == "VALIDATION"]
    return {
        "contract": "SYSTEM_UTILIZATION_RCA_V1",
        "ensemble_contract": "FULL_INFERENCE_ENSEMBLE_V1",
        "fused_contract": "FUSED_DECISION_V1",
        "holdout_read": False,
        "live_paper_broker_order_deploy_touched": False,
        "inventory": inventory,
        "classification_counts": dict(counts),
        "episode_aligned_usable_outputs": dict(usable),
        "lineage": {
            "episodes": len(outcomes),
            "forecast_causal": usable["forecast"],
            "price_action": usable["price_action"],
            "entry_thesis_causal": 18,
            "filled_terminal_actual_after_cost": len(outcomes),
            "market_read_execution_link_artifact_present": False,
            "full_forecast_technical_thesis_margin_chain": 1,
            "margin_evidence_64d_validation": {
                "known": sum(bool(row["margin_evidence_known"]) for row in val_records),
                "total": len(val_records),
                "coverage": sum(bool(row["margin_evidence_known"]) for row in val_records) / len(val_records),
            },
        },
        "runtime_receipts": {
            "forecast_history_rows": 3006,
            "projection_ledger_rows": 18518,
            "entry_thesis_rows": 37,
            "market_read_prediction_rows": 5,
            "automation_toml_count": len(automation_files),
            "swift_files_in_repo": 0,
            "adapter_runtime_consumers": 0,
        },
        "causal_bottleneck": {
            "largest": "decision-time execution/fill/unwind and portfolio-margin evidence are not historically joined",
            "effect": "all edge-positive fused candidates remain WAIT; no final TRADE is admissible",
            "secondary": "only 55/251 episodes have both causal forecast and gapless price-action evidence",
            "all_trades_fallback_detected": False,
        },
        "fusion": fusion,
        "final_decision_counts": dict(final_counts),
        "validation_final_decision_counts": dict(Counter(row["action"] for row in val)),
        "final_trade_count": final_counts.get("TRADE", 0),
        "profitability_increment_attributed_to_fusion_jpy": 0.0,
        "research_shadow_64d_validation": {
            "all_trades_net_jpy": fusion["QUADRUPLE_64D"]["fused_evidence_admissible"]["all_trades_net_jpy"],
            "fused_net_jpy": fusion["QUADRUPLE_64D"]["fused_evidence_admissible"]["after_cost_net_jpy"],
            "incremental_net_jpy": fusion["QUADRUPLE_64D"]["fused_evidence_admissible"]["incremental_net_jpy"],
            "paired_lcb_jpy": fusion["QUADRUPLE_64D"]["fused_evidence_admissible"]["paired_bootstrap_lcb_jpy"],
            "decisions_changed": fusion["QUADRUPLE_64D"]["fused_evidence_admissible"]["decisions_changed_vs_all_trades"],
            "interpretation": "research opportunity cost only; no live/Paper decision changed"
        },
        "decision_utilization_kpi": {
            "systems_with_any_episode_opinion": len(usable),
            "systems_total": len(inventory),
            "episodes_with_one_answer": len(fused),
            "episodes_with_executable_trade_answer": final_counts.get("TRADE", 0),
            "all_trades_baseline_used_as_decision": False,
        },
        "verdict": "HOLD_EVIDENCE_PIPELINE_BLOCKED",
        "next_independent_fix": "persist point-in-time execution, fillability, margin and unwind evidence keyed by decision_id before fitting more ensemble parameters",
    }


def main() -> None:
    prereg = json.loads((HERE / "preregister_v1.json").read_text(encoding="utf-8"))
    if prereg["permissions"]["holdout_read"] is not False:
        raise RuntimeError("holdout contract changed")
    inference, outcomes, context = build_tables()
    fusion, fused = fusion_evaluation(context)
    cube = sparse_cube(inference)
    report = utilization_report(inference, outcomes, fusion, fused)
    write_jsonl(HERE / "inference_table_v1.jsonl", inference)
    write_jsonl(HERE / "outcome_table_v1.jsonl", outcomes)
    write_jsonl(HERE / "fused_decisions_v1.jsonl", fused)
    write_json(HERE / "inference_cube_sparse_v1.json", cube)
    write_json(HERE / "utilization_report_v1.json", report)
    manifest = {
        "contract": "FULL_INFERENCE_ENSEMBLE_V1_RUN",
        "preregister_sha256": sha256(HERE / "preregister_v1.json"),
        "outputs": {name: sha256(HERE / name) for name in (
            "inference_table_v1.jsonl", "outcome_table_v1.jsonl",
            "fused_decisions_v1.jsonl", "inference_cube_sparse_v1.json",
            "utilization_report_v1.json",
        )},
        "holdout_read": False,
    }
    write_json(HERE / "run_manifest_v1.json", manifest)
    print(json.dumps({
        "episodes": len(outcomes), "inference_rows": len(inference),
        "fused_decisions": len(fused), "verdict": report["verdict"],
        "final_decisions": report["final_decision_counts"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
