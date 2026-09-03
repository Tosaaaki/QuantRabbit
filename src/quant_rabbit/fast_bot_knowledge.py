"""Versioned knowledge ledger for resolved fast-bot shadow trades.

The raw signal, exact S5 outcome, and corrective challenger ledgers remain
untouched.  This module links them by the existing shadow ``signal_id`` (the
shadow trade identity), derives one content-addressed episode, classifies the
failure layer from precommitted same-path counterfactuals, and appends a
versioned knowledge assessment.  It has no broker client and cannot activate a
strategy or grant live permission.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quant_rabbit.fast_bot import SIGNAL_CONTRACT
from quant_rabbit.fast_bot_corrective_challenger import (
    ARM_ORDER,
    ROW_CONTRACT,
    arm_order_for_config,
    canonical_sha,
    load_config,
    load_jsonl,
    sealed_valid,
)
from quant_rabbit.fast_bot_truth import OUTCOME_CONTRACT


EPISODE_CONTRACT = "QR_FAST_BOT_SHADOW_LEARNING_EPISODE_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_SHADOW_LEARNING_SCORECARD_V1"
KNOWLEDGE_CONTRACT = "QR_FAST_BOT_SHADOW_KNOWLEDGE_V1"
DERIVATION_POLICY = "QR_FAST_BOT_SHADOW_KNOWLEDGE_DERIVATION_V1"


def run_fast_bot_knowledge(
    *,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    challenger_ledger_path: Path,
    config_path: Path,
    episode_ledger_path: Path,
    knowledge_ledger_path: Path,
    scorecard_path: Path,
) -> dict[str, Any]:
    """Derive immutable learning artifacts without changing source ledgers."""

    config, config_sha = load_config(config_path)
    arm_order = arm_order_for_config(config)
    signals = load_jsonl(shadow_ledger_path)
    outcomes = load_jsonl(outcome_ledger_path)
    challenger_rows = load_jsonl(challenger_ledger_path)
    episodes, missing = build_learning_episodes(
        signals=signals,
        outcomes=outcomes,
        challenger_rows=challenger_rows,
        config_sha256=config_sha,
        arm_order=arm_order,
    )
    appended_episodes = _append_once(
        episode_ledger_path,
        episodes,
        contract=EPISODE_CONTRACT,
        identity_key="episode_id",
    )
    all_episodes = [
        row
        for row in load_jsonl(episode_ledger_path)
        if row.get("contract") == EPISODE_CONTRACT
        and row.get("config_sha256") == config_sha
    ]
    scorecard = build_learning_scorecard(
        all_episodes,
        config=config,
        config_sha256=config_sha,
    )
    _write_json_atomic(scorecard_path, scorecard)
    knowledge = build_knowledge_record(scorecard=scorecard, config=config)
    appended_knowledge = _append_once(
        knowledge_ledger_path,
        [knowledge],
        contract=KNOWLEDGE_CONTRACT,
        identity_key="knowledge_id",
    )
    return {
        "contract": "QR_FAST_BOT_SHADOW_KNOWLEDGE_RUN_V1",
        "status": str(scorecard["assessment_status"]),
        "collection_action": str(scorecard["collection_action"]),
        "config_sha256": config_sha,
        "resolved_episode_count": len(all_episodes),
        "new_episode_count": appended_episodes,
        "missing_complete_counterfactual_count": missing,
        "new_knowledge_record_count": appended_knowledge,
        "episode_ledger_path": str(episode_ledger_path),
        "knowledge_ledger_path": str(knowledge_ledger_path),
        "scorecard_path": str(scorecard_path),
        "target_arm_id": scorecard["target_arm_id"],
        "target_net_delta_pips": scorecard["paired_delta"]["net_delta_pips"],
        "execution_authority": "NONE",
        "broker_http_methods_used": [],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
    }


def build_learning_episodes(
    *,
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    challenger_rows: Sequence[Mapping[str, Any]],
    config_sha256: str,
    arm_order: Sequence[str] = ARM_ORDER,
) -> tuple[list[dict[str, Any]], int]:
    signal_by_id: dict[str, Mapping[str, Any]] = {}
    for signal in signals:
        if not _signal_valid(signal):
            raise ValueError("shadow signal seal mismatch")
        signal_id = str(signal["signal_id"])
        if signal_id in signal_by_id:
            raise ValueError("duplicate shadow signal_id")
        signal_by_id[signal_id] = signal
    outcome_by_id: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not sealed_valid(outcome, OUTCOME_CONTRACT):
            raise ValueError("shadow outcome seal mismatch")
        signal_id = str(outcome.get("signal_id") or "")
        if signal_id in outcome_by_id:
            raise ValueError("duplicate shadow outcome signal_id")
        outcome_by_id[signal_id] = outcome
    rows_by_signal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in challenger_rows:
        if row.get("config_sha256") != config_sha256:
            continue
        if not sealed_valid(row, ROW_CONTRACT):
            raise ValueError("challenger row seal mismatch")
        rows_by_signal[str(row.get("signal_id") or "")].append(row)

    episodes: list[dict[str, Any]] = []
    missing = 0
    for signal_id, signal in signal_by_id.items():
        outcome = outcome_by_id.get(signal_id)
        rows = rows_by_signal.get(signal_id, [])
        if outcome is None or not _outcome_matches_signal(outcome, signal):
            continue
        if (
            tuple(
                str(row.get("arm_id"))
                for row in sorted(rows, key=lambda item: _arm_index(item, arm_order))
            )
            != tuple(arm_order)
        ):
            missing += 1
            continue
        episodes.append(
            build_learning_episode(
                signal=signal,
                outcome=outcome,
                challenger_rows=rows,
                config_sha256=config_sha256,
                arm_order=arm_order,
            )
        )
    return episodes, missing


def build_learning_episode(
    *,
    signal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    challenger_rows: Sequence[Mapping[str, Any]],
    config_sha256: str,
    arm_order: Sequence[str] = ARM_ORDER,
) -> dict[str, Any]:
    ordered = sorted(
        challenger_rows,
        key=lambda item: _arm_index(item, arm_order),
    )
    baseline = next(row for row in ordered if row["arm_id"] == "BASELINE")
    realized = float(baseline["after_cost_net_pips"])
    filled = baseline.get("filled") is True and baseline.get("vetoed") is not True
    stop_loss_pips = float(baseline["stop_loss_pips"])
    take_profit_pips = float(baseline["take_profit_pips"])
    counterfactuals = [
        {
            "arm_id": str(row["arm_id"]),
            "row_sha256": str(row["contract_sha256"]),
            "filled": row.get("filled") is True,
            "vetoed": row.get("vetoed") is True,
            "veto_reason": row.get("veto_reason"),
            "after_cost_net_pips": float(row["after_cost_net_pips"]),
            "delta_pips_vs_baseline": round(
                float(row["after_cost_net_pips"]) - realized, 6
            ),
            "mfe_pips": float(row["mfe_pips"]),
            "mae_pips": float(row["mae_pips"]),
        }
        for row in ordered
    ]
    failure = _failure_layer(
        filled=filled,
        realized_pips=realized,
        counterfactuals=counterfactuals,
    )
    exit_reason = str(baseline.get("exit_reason") or "")
    stop_gap = (
        max(0.0, abs(min(0.0, realized)) - stop_loss_pips)
        if filled and "STOP_LOSS" in exit_reason
        else 0.0 if filled else None
    )
    source_rows = [str(row["contract_sha256"]) for row in ordered]
    identity = {
        "derivation_policy": DERIVATION_POLICY,
        "config_sha256": config_sha256,
        "signal_id": str(signal["signal_id"]),
        "signal_sha256": str(signal["signal_sha256"]),
        "outcome_sha256": str(outcome["contract_sha256"]),
        "challenger_row_sha256": source_rows,
    }
    body = {
        "contract": EPISODE_CONTRACT,
        "schema_version": 1,
        "derivation_policy": DERIVATION_POLICY,
        "episode_id": canonical_sha(identity),
        "trade_id": str(signal["signal_id"]),
        "trade_id_namespace": "SHADOW_SIGNAL_ID",
        "config_sha256": config_sha256,
        "generated_at_utc": str(signal["generated_at_utc"]),
        "resolved_at_utc": str(outcome["resolved_at_utc"]),
        "pair": str(signal["pair"]),
        "side": str(signal["side"]),
        "method": str(signal["method"]),
        "horizon_lane": str(signal.get("horizon_lane") or ""),
        "raw_source_refs": identity,
        "outcome": {
            "filled": filled,
            "fill_at_utc": baseline.get("fill_at_utc"),
            "exit_at_utc": baseline.get("exit_at_utc"),
            "exit_reason": exit_reason,
            "realized_pips": realized,
            "mfe_pips": float(baseline["mfe_pips"]),
            "mae_pips": float(baseline["mae_pips"]),
            "fill_gap_slippage_like_pips": 0.0 if filled else None,
            "stop_gap_slippage_like_pips": (
                round(stop_gap, 6) if stop_gap is not None else None
            ),
            "gap_measurement_basis": (
                "PASSIVE_TOUCH_FILL_AND_STOP_OVERRUN_V1" if filled else "UNFILLED"
            ),
        },
        "expectation_gap": {
            "planned_take_profit_pips": take_profit_pips,
            "planned_stop_loss_pips": stop_loss_pips,
            "actual_minus_planned_take_profit_pips": (
                round(realized - take_profit_pips, 6) if filled else None
            ),
            "statistical_expectancy_gap_pips": None,
            "statistical_expectancy_status": (
                "REQUIRES_PREREGISTERED_FORWARD_SAMPLE_FLOORS"
            ),
        },
        "entry_context": {
            "spread_pips": float(signal["spread_pips"]),
            "m5_atr_pips": float(signal["m5_atr_pips"]),
            "regime_score": float(signal["regime_score"]),
            "regime_bucket": str(baseline["regime_bucket"]),
            "atr_bucket": str(baseline["atr_bucket"]),
            "spread_bucket": str(baseline["spread_bucket"]),
        },
        "failure_classification": failure,
        "counterfactuals": counterfactuals,
        "raw_ledgers_mutated": False,
        "derived_record_versioned": True,
        "automatic_adoption_allowed": False,
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "shadow_only": True,
        "live_permission": False,
    }
    return _seal(body)


def build_learning_scorecard(
    episodes: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    for row in episodes:
        if not _sealed(row, EPISODE_CONTRACT):
            raise ValueError("learning episode seal mismatch")
    prereg = dict(config["preregistration"])
    target_arm = str(prereg["target_arm_id"])
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        for row in episode["counterfactuals"]:
            arm_rows[str(row["arm_id"])].append(
                {
                    **dict(row),
                    "trade_id": episode["trade_id"],
                    "generated_at_utc": episode["generated_at_utc"],
                    "regime_bucket": episode["entry_context"]["regime_bucket"],
                }
            )
    arm_order = arm_order_for_config(config)
    metrics = {arm: _metrics(arm_rows.get(arm, [])) for arm in arm_order}
    baseline = metrics["BASELINE"]
    target = metrics[target_arm]
    paired_delta = _paired_delta(
        baseline_rows=arm_rows.get("BASELINE", []),
        target_rows=arm_rows.get(target_arm, []),
    )
    criteria = _criteria(prereg["success_criteria"], baseline, target, paired_delta)
    adverse = _adverse_conditions(baseline, target, preregistration=prereg)
    status = (
        "STOP_REVIEW_REQUIRED"
        if adverse["stop_condition_observed"]
        else "ELIGIBLE_FOR_OWNER_REVIEW"
        if all(criteria.values())
        else "COLLECTING_FORWARD_EVIDENCE"
    )
    evidence_through = max(
        (str(row["resolved_at_utc"]) for row in episodes),
        default=None,
    )
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "derivation_policy": DERIVATION_POLICY,
        "config_sha256": config_sha256,
        "evidence_through_utc": evidence_through,
        "episode_count": len(episodes),
        "trade_ids": sorted(str(row["trade_id"]) for row in episodes),
        "target_arm_id": target_arm,
        "arm_metrics": metrics,
        "paired_delta": paired_delta,
        "regime_delta": _regime_delta(
            arm_rows.get("BASELINE", []), arm_rows.get(target_arm, [])
        ),
        "success_criteria_results": criteria,
        "adverse_condition_results": adverse,
        "assessment_status": status,
        "collection_action": (
            "HALT_TARGET_COHORT" if status == "STOP_REVIEW_REQUIRED" else "CONTINUE_TARGET_COHORT"
        ),
        "same_scorecard_metrics": [
            "profit_factor",
            "net_pips",
            "wins",
            "losses",
            "resolved_count",
            "filled_count",
            "unfilled_count",
            "mean_mfe_pips",
            "mean_mae_pips",
            "regime_delta",
            "one_sided_95_daily_delta_lower_pips",
        ],
        "uncertainty_claim_allowed": paired_delta["daily_observation_count"] >= 2,
        "positive_profitability_claim_allowed": False,
        "owner_financial_judgment_required": True,
        "once_only_activation_ready": False,
        "once_only_activation_blocker": "OWNER_REVIEW_AND_SOURCE_BUNDLE_BINDING_REQUIRED",
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
    }
    return _seal(body)


def build_knowledge_record(
    *, scorecard: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    prereg = dict(config["preregistration"])
    identity = {
        "derivation_policy": DERIVATION_POLICY,
        "hypothesis_id": prereg["hypothesis_id"],
        "hypothesis_version": prereg["hypothesis_version"],
        "config_sha256": scorecard["config_sha256"],
        "scorecard_sha256": scorecard["contract_sha256"],
    }
    body = {
        "contract": KNOWLEDGE_CONTRACT,
        "schema_version": 1,
        "knowledge_id": canonical_sha(identity),
        "derivation_policy": DERIVATION_POLICY,
        "hypothesis_id": prereg["hypothesis_id"],
        "hypothesis_version": prereg["hypothesis_version"],
        "preregistration": prereg,
        "evidence_refs": {
            "config_sha256": scorecard["config_sha256"],
            "scorecard_sha256": scorecard["contract_sha256"],
            "trade_ids": list(scorecard["trade_ids"]),
        },
        "assessment_status": scorecard["assessment_status"],
        "collection_action": scorecard["collection_action"],
        "success_criteria_results": dict(scorecard["success_criteria_results"]),
        "adverse_condition_results": dict(scorecard["adverse_condition_results"]),
        "adoption_state": "NOT_ADOPTED_OWNER_REVIEW_REQUIRED",
        "immutable_raw_preserved": True,
        "versioned_derived_record": True,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
    }
    return _seal(body)


def _failure_layer(
    *, filled: bool, realized_pips: float, counterfactuals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not filled:
        return {
            "layer": "EXECUTION_FILL_LAYER",
            "status": "NO_FILL_NO_REALIZED_LOSS",
            "supporting_arm_id": None,
            "recovered_pips_counterfactual": 0.0,
            "interpretation_limit": "UNFILLED_IS_NOT_DIRECTIONAL_FAILURE",
        }
    if realized_pips >= 0.0:
        return {
            "layer": "NO_FAILURE",
            "status": "NONNEGATIVE_REALIZED_OUTCOME",
            "supporting_arm_id": None,
            "recovered_pips_counterfactual": 0.0,
            "interpretation_limit": "SINGLE_WIN_IS_NOT_PROFITABILITY_PROOF",
        }
    by_arm = {str(row["arm_id"]): row for row in counterfactuals}
    priority = (
        ("LANE_COOLDOWN", "PORTFOLIO_CONCURRENCY_LAYER"),
        ("VOL_SHOCK_VETO", "REGIME_VOLATILITY_LAYER"),
        ("ATR_NORMALIZED_GEOMETRY", "GEOMETRY_LAYER"),
        ("EURUSD_RANGE_ROTATION_EXCLUDE", "STRATEGY_LANE_SELECTION_LAYER"),
        ("COMBINED", "MULTI_LAYER"),
    )
    for arm_id, layer in priority:
        row = by_arm.get(arm_id)
        if row and float(row["delta_pips_vs_baseline"]) > 0.0:
            return {
                "layer": layer,
                "status": "SUPPORTED_BY_PRECOMMITTED_SAME_PATH_COUNTERFACTUAL",
                "supporting_arm_id": arm_id,
                "recovered_pips_counterfactual": float(
                    row["delta_pips_vs_baseline"]
                ),
                "interpretation_limit": (
                    "PAIRED_COUNTERFACTUAL_SUPPORT_IS_NOT_CAUSAL_PROOF_OR_ADOPTION"
                ),
            }
    return {
        "layer": "DIRECTION_OR_ENTRY_LAYER",
        "status": "UNRESOLVED_BY_PRECOMMITTED_COUNTERFACTUALS",
        "supporting_arm_id": None,
        "recovered_pips_counterfactual": 0.0,
        "interpretation_limit": "DO_NOT_INVENT_CAUSE_FROM_RESIDUAL_LOSS",
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row["generated_at_utc"]), str(row["trade_id"])))
    eligible = [row for row in ordered if row.get("vetoed") is not True]
    filled = [row for row in eligible if row.get("filled") is True]
    values = [float(row["after_cost_net_pips"]) for row in filled]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    days = {str(row["generated_at_utc"])[:10] for row in filled}
    gross_loss = abs(sum(losses))
    pf: float | str | None = (
        round(sum(wins) / gross_loss, 6)
        if gross_loss
        else "INF" if wins else None
    )
    streak = 0
    max_streak = 0
    for value in values:
        streak = streak + 1 if value < 0.0 else 0
        max_streak = max(max_streak, streak)
    return {
        "resolved_count": len(ordered),
        "eligible_count": len(eligible),
        "vetoed_count": len(ordered) - len(eligible),
        "filled_count": len(filled),
        "unfilled_count": len(eligible) - len(filled),
        "active_days": len(days),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(filled), 6) if filled else None,
        "net_pips": round(sum(values), 6),
        "profit_factor": pf,
        "max_consecutive_losses": max_streak,
        "mean_mfe_pips": _mean(float(row["mfe_pips"]) for row in filled),
        "mean_mae_pips": _mean(float(row["mae_pips"]) for row in filled),
    }


def _paired_delta(
    *, baseline_rows: Sequence[Mapping[str, Any]], target_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline = {str(row["trade_id"]): row for row in baseline_rows}
    target = {str(row["trade_id"]): row for row in target_rows}
    if set(baseline) != set(target):
        raise ValueError("paired scorecard trade identities differ")
    daily: dict[str, float] = defaultdict(float)
    total = 0.0
    for trade_id in sorted(baseline):
        left = _effective_pips(baseline[trade_id])
        right = _effective_pips(target[trade_id])
        delta = right - left
        total += delta
        daily[str(baseline[trade_id]["generated_at_utc"])[:10]] += delta
    lower = _one_sided_95_lower(list(daily.values()))
    return {
        "paired_trade_count": len(baseline),
        "net_delta_pips": round(total, 6),
        "daily_observation_count": len(daily),
        "one_sided_95_daily_delta_lower_pips": (
            round(lower, 6) if lower is not None and math.isfinite(lower) else None
        ),
        "uncertainty_status": (
            "ESTIMATED" if len(daily) >= 2 else "INSUFFICIENT_ACTIVE_DAYS"
        ),
    }


def _regime_delta(
    baseline_rows: Sequence[Mapping[str, Any]], target_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    baseline: dict[str, float] = defaultdict(float)
    target: dict[str, float] = defaultdict(float)
    for row in baseline_rows:
        baseline[str(row["regime_bucket"])] += _effective_pips(row)
    for row in target_rows:
        target[str(row["regime_bucket"])] += _effective_pips(row)
    return [
        {
            "regime_bucket": regime,
            "baseline_net_pips": round(baseline[regime], 6),
            "target_net_pips": round(target[regime], 6),
            "delta_pips": round(target[regime] - baseline[regime], 6),
        }
        for regime in sorted(set(baseline) | set(target))
    ]


def _criteria(
    expected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, bool]:
    pf = target["profit_factor"]
    pf_value = math.inf if pf == "INF" else float(pf or 0.0)
    return {
        "minimum_resolved_forward_fills": int(target["filled_count"])
        >= int(expected["minimum_resolved_forward_fills"]),
        "minimum_active_forward_days": int(target["active_days"])
        >= int(expected["minimum_active_forward_days"]),
        "minimum_profit_factor": pf_value
        >= float(expected["minimum_profit_factor"]),
        "minimum_net_delta_pips": float(delta["net_delta_pips"])
        > float(expected["minimum_net_delta_pips_exclusive"]),
        "minimum_daily_delta_lower_pips": (
            delta["one_sided_95_daily_delta_lower_pips"] is not None
            and float(delta["one_sided_95_daily_delta_lower_pips"])
            > float(expected["minimum_daily_delta_lower_pips_exclusive"])
        ),
        "maximum_loss_streak_delta": int(target["max_consecutive_losses"])
        - int(baseline["max_consecutive_losses"])
        <= int(expected["maximum_loss_streak_delta_vs_baseline"]),
    }


def _adverse_conditions(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    sample_floor_met = int(target["filled_count"]) >= 100
    early_stop = preregistration.get("early_futility_stop")
    early_floor = (
        int(early_stop.get("minimum_target_fills") or 0)
        if isinstance(early_stop, Mapping)
        else 0
    )
    early_floor_met = early_floor > 0 and int(target["filled_count"]) >= early_floor
    baseline_pf = _profit_factor_value(baseline.get("profit_factor"))
    target_pf = _profit_factor_value(target.get("profit_factor"))
    dual_metric_futility = bool(
        early_floor_met
        and float(target["net_pips"]) <= float(baseline["net_pips"])
        and target_pf <= baseline_pf
    )
    target_mae = target.get("mean_mae_pips")
    baseline_mae = baseline.get("mean_mae_pips")
    observed = {
        "loss_streak_worse_than_baseline": int(target["max_consecutive_losses"])
        > int(baseline["max_consecutive_losses"]),
        "mean_mae_worse_after_minimum_fill_floor": bool(
            sample_floor_met
            and target_mae is not None
            and baseline_mae is not None
            and float(target_mae) > float(baseline_mae)
        ),
        "net_pips_not_above_baseline_after_minimum_fill_floor": bool(
            sample_floor_met
            and float(target["net_pips"]) <= float(baseline["net_pips"])
        ),
        "dual_metric_futility_after_early_floor": dual_metric_futility,
        "early_futility_minimum_target_fills": early_floor or None,
        "early_futility_floor_met": early_floor_met,
    }
    stop_condition_keys = (
        "loss_streak_worse_than_baseline",
        "mean_mae_worse_after_minimum_fill_floor",
        "net_pips_not_above_baseline_after_minimum_fill_floor",
        "dual_metric_futility_after_early_floor",
    )
    return {
        **observed,
        # early_futility_floor_met is an informational prerequisite, not an
        # adverse observation by itself.
        "stop_condition_observed": any(
            bool(observed[key]) for key in stop_condition_keys
        ),
    }


def _profit_factor_value(value: Any) -> float:
    if value == "INF":
        return math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -math.inf


def _signal_valid(value: Mapping[str, Any]) -> bool:
    stored = str(value.get("signal_sha256") or "")
    body = {key: item for key, item in value.items() if key != "signal_sha256"}
    return (
        value.get("contract") == SIGNAL_CONTRACT
        and len(stored) == 64
        and stored == canonical_sha(body)
    )


def _outcome_matches_signal(
    outcome: Mapping[str, Any], signal: Mapping[str, Any]
) -> bool:
    return (
        outcome.get("signal_id") == signal.get("signal_id")
        and outcome.get("signal_sha256") == signal.get("signal_sha256")
        and outcome.get("pair") == signal.get("pair")
        and outcome.get("side") == signal.get("side")
        and outcome.get("method") == signal.get("method")
    )


def _effective_pips(row: Mapping[str, Any]) -> float:
    return (
        float(row["after_cost_net_pips"])
        if row.get("vetoed") is not True and row.get("filled") is True
        else 0.0
    )


def _arm_index(
    row: Mapping[str, Any], arm_order: Sequence[str] = ARM_ORDER
) -> int:
    try:
        return tuple(arm_order).index(str(row.get("arm_id") or ""))
    except ValueError:
        return len(arm_order)


def _one_sided_95_lower(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return mean
    return mean - _student_t_one_sided_95(len(values) - 1) * stdev / math.sqrt(len(values))


def _student_t_one_sided_95(df: int) -> float:
    # Fixed statistical critical values define the preregistered 95% inference
    # level; they are not market parameters and must not be tuned on outcomes.
    table = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        12: 1.782,
        15: 1.753,
        20: 1.725,
        25: 1.708,
        30: 1.697,
        40: 1.684,
        60: 1.671,
        120: 1.658,
    }
    for bound in sorted(table):
        if df <= bound:
            return table[bound]
    return 1.645


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def _sealed(value: Mapping[str, Any], contract: str) -> bool:
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and stored == canonical_sha(body)


def _append_once(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: str,
    identity_key: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            identity = str(value.get(identity_key) or "") if isinstance(value, dict) else ""
            if not identity or identity in seen or not _sealed(value, contract):
                raise ValueError(f"invalid knowledge ledger row at line {number}")
            seen.add(identity)
        handle.seek(0, os.SEEK_END)
        appended = 0
        for row in rows:
            identity = str(row.get(identity_key) or "")
            if not identity or not _sealed(row, contract):
                raise ValueError("new knowledge ledger row is invalid")
            if identity in seen:
                continue
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(identity)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "DERIVATION_POLICY",
    "EPISODE_CONTRACT",
    "KNOWLEDGE_CONTRACT",
    "SCORECARD_CONTRACT",
    "build_knowledge_record",
    "build_learning_episode",
    "build_learning_episodes",
    "build_learning_scorecard",
    "run_fast_bot_knowledge",
]
