"""Causal, fixed-denominator Phase A/B supervision study contract.

The module preregisters bot-only and AI-managed replays over identical sealed
market windows.  It validates point-in-time supervisor inputs and compares
cadences only on untouched walk-forward/OOS blocks.  Missing or failed cells
remain in the denominator and force an UNRANKED result.

This is research/paper infrastructure.  It cannot grant live, broker, order, or
automatic deployment authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any, Final

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


PLAN_CONTRACT: Final = "QR_DOJO_COUNTERFACTUAL_SUPERVISION_PLAN_V1"
CELL_CONTRACT: Final = "QR_DOJO_COUNTERFACTUAL_SUPERVISION_CELL_V1"
EVALUATION_CONTRACT: Final = "QR_DOJO_COUNTERFACTUAL_SUPERVISION_EVALUATION_V1"
SCHEMA_VERSION: Final = 1

BOT_ONLY: Final = "BOT_ONLY"
CADENCE_IDS: Final = (
    "FIXED_5M",
    "FIXED_15M",
    "FIXED_30M",
    "FIXED_60M",
    "FIXED_120M",
    "EVENT_DRIVEN",
    "ADAPTIVE_60M_15M_EVENT",
)
BASELINE_CADENCE_ID: Final = "ADAPTIVE_60M_15M_EVENT"
SUPERVISOR_ACTIONS: Final = (
    "CONTINUE",
    "STOP_NEW_RISK",
    "REDUCE",
    "CLOSE",
    "RESUME",
)
EVENT_SIGNALS: Final = (
    "MARGIN_UTILIZATION_THRESHOLD",
    "NET_EXPOSURE_SPIKE",
    "GROSS_EXPOSURE_SPIKE",
    "CORRELATION_CONCENTRATION",
    "DRAWDOWN_DETERIORATION",
    "VOLATILITY_REGIME_CHANGE",
    "STRATEGY_THESIS_INVALIDATION",
    "CONSECUTIVE_LOSSES",
    "POSITION_AGE",
)
METRIC_FIELDS: Final = (
    "net_pnl_jpy",
    "max_drawdown_fraction",
    "peak_margin_usage_fraction",
    "margin_call_count",
    "ruin_event_count",
    "turnover_jpy",
    "mean_gross_exposure_jpy",
    "loss_avoidance_jpy",
    "opportunity_loss_jpy",
    "ai_call_count",
    "ai_token_count",
    "ai_latency_ms_total",
    "stop_count",
    "resume_count",
)
EVIDENCE_COUNT_FIELDS: Final = (
    "orders",
    "fills",
    "tp_exits",
    "sl_exits",
    "inventory_snapshots",
    "margin_snapshots",
    "unrealized_pnl_snapshots",
    "realized_pnl_events",
)
_AUTHORITY: Final = {
    "research_only": True,
    "paper_replay_only": True,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "order_authority": "NONE",
    "automatic_deployment_allowed": False,
    "promotion_eligible": False,
}


class DojoCounterfactualSupervisionError(ValueError):
    """The study is mutable, incomplete, unpaired, or causally invalid."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoCounterfactualSupervisionError(f"{field} must be an object")
    return dict(value)


def _sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoCounterfactualSupervisionError(f"{field} must be an array")
    return list(value)


def _exact(value: Mapping[str, Any], keys: set[str], field: str) -> None:
    if set(value) != keys:
        raise DojoCounterfactualSupervisionError(f"{field} schema mismatch")


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise DojoCounterfactualSupervisionError(
            f"{field} must be visible ASCII with 1..240 characters"
        )
    return value


def _sha256(value: Any, field: str) -> str:
    digest = _identifier(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DojoCounterfactualSupervisionError(f"{field} must be lowercase SHA-256")
    return digest


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoCounterfactualSupervisionError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoCounterfactualSupervisionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DojoCounterfactualSupervisionError(f"{field} is outside its bounds")
    return result


def _validate_windows(value: Any) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    prior_end: int | None = None
    for index, raw in enumerate(_sequence(value, "market_windows")):
        row = _mapping(raw, f"market_windows[{index}]")
        _exact(
            row,
            {
                "window_id",
                "partition",
                "start_epoch",
                "end_epoch",
                "source_slice_sha256",
            },
            f"market_windows[{index}]",
        )
        window_id = _identifier(row["window_id"], f"market_windows[{index}].window_id")
        if window_id in seen:
            raise DojoCounterfactualSupervisionError("window IDs must be unique")
        partition = row["partition"]
        if partition not in {"TRAIN", "OOS"}:
            raise DojoCounterfactualSupervisionError("partition must be TRAIN or OOS")
        start = _integer(row["start_epoch"], f"market_windows[{index}].start_epoch")
        end = _integer(
            row["end_epoch"], f"market_windows[{index}].end_epoch", minimum=start + 1
        )
        if prior_end is not None and start < prior_end:
            raise DojoCounterfactualSupervisionError(
                "market windows must be chronological and non-overlapping"
            )
        seen.add(window_id)
        prior_end = end
        windows.append(
            {
                "window_id": window_id,
                "partition": partition,
                "start_epoch": start,
                "end_epoch": end,
                "source_slice_sha256": _sha256(
                    row["source_slice_sha256"],
                    f"market_windows[{index}].source_slice_sha256",
                ),
            }
        )
    if not windows or not any(row["partition"] == "TRAIN" for row in windows):
        raise DojoCounterfactualSupervisionError("at least one TRAIN window is required")
    if not any(row["partition"] == "OOS" for row in windows):
        raise DojoCounterfactualSupervisionError("at least one OOS window is required")
    return windows


def build_counterfactual_supervision_plan(
    *,
    study_id: str,
    sealed_before_epoch: int,
    market_windows: Sequence[Mapping[str, Any]],
    source_manifest_sha256: str,
    initial_capital_jpy: int | float,
    cost_model_sha256: str,
    execution_model_sha256: str,
    supervisor_policy_sha256: str,
    strategy_compatible_resume_signal_ids: Sequence[str],
    regime_ids: Sequence[str],
) -> dict[str, Any]:
    """Seal the paired replay denominator and all cadence choices before reading OOS."""

    windows = _validate_windows(market_windows)
    sealed_epoch = _integer(sealed_before_epoch, "sealed_before_epoch")
    resume_ids = sorted(
        {_identifier(value, "resume signal") for value in strategy_compatible_resume_signal_ids}
    )
    if not resume_ids:
        raise DojoCounterfactualSupervisionError(
            "at least one strategy-compatible resume signal is required"
        )
    regimes = sorted({_identifier(value, "regime id") for value in regime_ids})
    if not regimes:
        raise DojoCounterfactualSupervisionError("at least one regime is required")
    cadence_policies = [
        {
            "cadence_id": f"FIXED_{minutes}M",
            "mode": "FIXED",
            "fixed_interval_seconds": minutes * 60,
            "normal_heartbeat_seconds": None,
            "high_risk_interval_seconds": None,
            "major_event_immediate": False,
        }
        for minutes in (5, 15, 30, 60, 120)
    ]
    cadence_policies.extend(
        [
            {
                "cadence_id": "EVENT_DRIVEN",
                "mode": "EVENT_DRIVEN",
                "fixed_interval_seconds": None,
                "normal_heartbeat_seconds": None,
                "high_risk_interval_seconds": None,
                "major_event_immediate": True,
            },
            {
                "cadence_id": BASELINE_CADENCE_ID,
                "mode": "ADAPTIVE_HYBRID",
                "fixed_interval_seconds": None,
                "normal_heartbeat_seconds": 3_600,
                "high_risk_interval_seconds": 900,
                "major_event_immediate": True,
            },
        ]
    )
    body = {
        "contract": PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_id": _identifier(study_id, "study_id"),
        "sealed_before_epoch": sealed_epoch,
        "market_windows": windows,
        "source_manifest_sha256": _sha256(
            source_manifest_sha256, "source_manifest_sha256"
        ),
        "initial_capital_jpy": _finite(
            initial_capital_jpy, "initial_capital_jpy", minimum=1.0
        ),
        "cost_model_sha256": _sha256(cost_model_sha256, "cost_model_sha256"),
        "execution_model_sha256": _sha256(
            execution_model_sha256, "execution_model_sha256"
        ),
        "supervisor_policy_sha256": _sha256(
            supervisor_policy_sha256, "supervisor_policy_sha256"
        ),
        "phase_a_arm": BOT_ONLY,
        "phase_b_cadence_ids": list(CADENCE_IDS),
        "baseline_cadence_id": BASELINE_CADENCE_ID,
        "cadence_policies": cadence_policies,
        "supervisor_actions": list(SUPERVISOR_ACTIONS),
        "event_signals": list(EVENT_SIGNALS),
        "strategy_compatible_resume_signal_ids": resume_ids,
        "regime_ids": regimes,
        "metric_fields": list(METRIC_FIELDS),
        "evidence_count_fields": list(EVIDENCE_COUNT_FIELDS),
        "closed_market_call_policy": (
            "SKIP_UNLESS_OPEN_POSITION_OR_PENDING_ORDER"
        ),
        "selection_policy": {
            "rank_partition": "OOS_ONLY",
            "minimum_oos_blocks": 8,
            "familywise_alpha": 0.05,
            "multiple_comparison_method": "BONFERRONI_ONE_SIDED_SIGN_TEST",
            "positive_median_net_pnl_delta_required": True,
            "non_positive_median_max_drawdown_delta_required": True,
            "no_aggregate_margin_call_or_ruin_increase_required": True,
            "all_declared_cells_remain_in_denominator": True,
            "single_six_month_fit_can_rank": False,
        },
        "anti_bias": {
            "decision_input_available_through_must_not_exceed_decision_epoch": True,
            "same_source_initial_capital_cost_and_execution_required": True,
            "train_can_rank": False,
            "failed_or_missing_cell_can_be_dropped": False,
            "cadence_reselection_after_oos_allowed": False,
            "survivor_only_summary_allowed": False,
        },
        "authority": dict(_AUTHORITY),
    }
    return {**body, "plan_sha256": canonical_portfolio_sha256(body)}


def validate_counterfactual_supervision_plan(value: Any) -> dict[str, Any]:
    plan = _mapping(value, "plan")
    claimed = plan.pop("plan_sha256", None)
    rebuilt = build_counterfactual_supervision_plan(
        study_id=plan.get("study_id"),
        sealed_before_epoch=plan.get("sealed_before_epoch"),
        market_windows=plan.get("market_windows"),
        source_manifest_sha256=plan.get("source_manifest_sha256"),
        initial_capital_jpy=plan.get("initial_capital_jpy"),
        cost_model_sha256=plan.get("cost_model_sha256"),
        execution_model_sha256=plan.get("execution_model_sha256"),
        supervisor_policy_sha256=plan.get("supervisor_policy_sha256"),
        strategy_compatible_resume_signal_ids=plan.get(
            "strategy_compatible_resume_signal_ids"
        ),
        regime_ids=plan.get("regime_ids"),
    )
    if claimed != rebuilt["plan_sha256"] or value != rebuilt:
        raise DojoCounterfactualSupervisionError("plan content or digest mismatch")
    return rebuilt


def _validate_decision(
    raw: Any,
    *,
    index: int,
    cadence_id: str,
    plan: Mapping[str, Any],
    window: Mapping[str, Any],
) -> dict[str, Any]:
    decision = _mapping(raw, f"decisions[{index}]")
    _exact(
        decision,
        {
            "decision_id",
            "decision_epoch",
            "input_available_through_epoch",
            "observation_sha256",
            "action",
            "trigger_kind",
            "event_signal_ids",
            "market_open",
            "has_open_position_or_pending_order",
            "resume_signal_ids",
            "regime_id",
            "margin_usage_fraction",
            "net_exposure_jpy",
            "gross_exposure_jpy",
            "drawdown_fraction",
            "position_age_seconds",
            "consecutive_losses",
            "strategy_thesis_valid",
        },
        f"decisions[{index}]",
    )
    epoch = _integer(decision["decision_epoch"], "decision_epoch")
    available = _integer(
        decision["input_available_through_epoch"], "input_available_through_epoch"
    )
    if not int(window["start_epoch"]) <= epoch < int(window["end_epoch"]):
        raise DojoCounterfactualSupervisionError(
            "supervisor decision is outside its sealed market window"
        )
    if available > epoch:
        raise DojoCounterfactualSupervisionError(
            "supervisor input contains future information"
        )
    for field in ("market_open", "has_open_position_or_pending_order", "strategy_thesis_valid"):
        if decision[field].__class__ is not bool:
            raise DojoCounterfactualSupervisionError(f"{field} must be boolean")
    if not decision["market_open"] and not decision["has_open_position_or_pending_order"]:
        raise DojoCounterfactualSupervisionError(
            "closed-market AI call lacks the position/order exception"
        )
    action = decision["action"]
    if action not in SUPERVISOR_ACTIONS:
        raise DojoCounterfactualSupervisionError("supervisor action is unsupported")
    event_ids = sorted(
        {_identifier(value, "event signal") for value in _sequence(
            decision["event_signal_ids"], "event_signal_ids"
        )}
    )
    if not set(event_ids).issubset(EVENT_SIGNALS):
        raise DojoCounterfactualSupervisionError("event signal is not preregistered")
    resume_ids = sorted(
        {_identifier(value, "resume signal") for value in _sequence(
            decision["resume_signal_ids"], "resume_signal_ids"
        )}
    )
    if action == "RESUME":
        if not resume_ids or not set(resume_ids).issubset(
            plan["strategy_compatible_resume_signal_ids"]
        ):
            raise DojoCounterfactualSupervisionError(
                "RESUME lacks a preregistered strategy-compatible signal"
            )
    elif resume_ids:
        raise DojoCounterfactualSupervisionError(
            "resume signals are only allowed on RESUME"
        )
    trigger = decision["trigger_kind"]
    if cadence_id.startswith("FIXED_") and trigger != "FIXED":
        raise DojoCounterfactualSupervisionError("fixed cadence requires FIXED trigger")
    if cadence_id == "EVENT_DRIVEN" and trigger != "MAJOR_EVENT":
        raise DojoCounterfactualSupervisionError(
            "event-driven cadence requires MAJOR_EVENT trigger"
        )
    if cadence_id == BASELINE_CADENCE_ID and trigger not in {
        "HEARTBEAT_60M",
        "HIGH_RISK_15M",
        "MAJOR_EVENT",
    }:
        raise DojoCounterfactualSupervisionError("adaptive trigger is unsupported")
    if trigger == "MAJOR_EVENT" and not event_ids:
        raise DojoCounterfactualSupervisionError(
            "major event trigger requires a preregistered event signal"
        )
    if trigger == "HIGH_RISK_15M" and not event_ids:
        raise DojoCounterfactualSupervisionError(
            "high-risk trigger requires a preregistered event signal"
        )
    if decision["regime_id"] not in plan["regime_ids"]:
        raise DojoCounterfactualSupervisionError("decision regime is outside the plan")
    for field in (
        "margin_usage_fraction",
        "gross_exposure_jpy",
        "drawdown_fraction",
        "position_age_seconds",
    ):
        _finite(decision[field], field, minimum=0.0)
    _finite(decision["net_exposure_jpy"], "net_exposure_jpy")
    _integer(decision["consecutive_losses"], "consecutive_losses")
    _sha256(decision["observation_sha256"], "observation_sha256")
    _identifier(decision["decision_id"], "decision_id")
    return dict(decision)


def seal_counterfactual_supervision_cell(
    *, plan: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and seal one complete Phase A or Phase B replay cell."""

    validated_plan = validate_counterfactual_supervision_plan(plan)
    row = _mapping(cell, "cell")
    _exact(
        row,
        {
            "window_id",
            "arm",
            "cadence_id",
            "source_slice_sha256",
            "status",
            "economic_transcript_sha256",
            "orders_sha256",
            "fills_sha256",
            "inventory_sha256",
            "evidence_counts",
            "metrics",
            "regime_net_pnl_jpy",
            "decisions",
        },
        "cell",
    )
    window_id = _identifier(row["window_id"], "window_id")
    windows = {item["window_id"]: item for item in validated_plan["market_windows"]}
    if window_id not in windows:
        raise DojoCounterfactualSupervisionError("cell window is outside the plan")
    arm = row["arm"]
    cadence_id = row["cadence_id"]
    if arm == BOT_ONLY:
        if cadence_id is not None:
            raise DojoCounterfactualSupervisionError(
                "BOT_ONLY must not have an AI cadence"
            )
    elif arm == "AI_MANAGED":
        if cadence_id not in CADENCE_IDS:
            raise DojoCounterfactualSupervisionError(
                "AI_MANAGED cadence is outside the fixed denominator"
            )
    else:
        raise DojoCounterfactualSupervisionError("cell arm is unsupported")
    if (
        row["source_slice_sha256"] != windows[window_id]["source_slice_sha256"]
        or row["status"] != "COMPLETE"
    ):
        raise DojoCounterfactualSupervisionError(
            "cell source differs or the cell is not complete"
        )
    evidence_counts = _mapping(row["evidence_counts"], "evidence_counts")
    _exact(evidence_counts, set(EVIDENCE_COUNT_FIELDS), "evidence_counts")
    for field in EVIDENCE_COUNT_FIELDS:
        _integer(evidence_counts[field], f"evidence_counts.{field}")
    metrics = _mapping(row["metrics"], "metrics")
    _exact(metrics, set(METRIC_FIELDS), "metrics")
    for field in METRIC_FIELDS:
        minimum = 0.0 if field != "net_pnl_jpy" else None
        _finite(metrics[field], f"metrics.{field}", minimum=minimum)
    for field in (
        "margin_call_count",
        "ruin_event_count",
        "ai_call_count",
        "ai_token_count",
        "stop_count",
        "resume_count",
    ):
        _integer(metrics[field], f"metrics.{field}")
    regime_metrics = _mapping(row["regime_net_pnl_jpy"], "regime_net_pnl_jpy")
    if set(regime_metrics) != set(validated_plan["regime_ids"]):
        raise DojoCounterfactualSupervisionError(
            "regime metrics must preserve the fixed regime denominator"
        )
    for regime_id, value in regime_metrics.items():
        _finite(value, f"regime_net_pnl_jpy.{regime_id}")
    decisions = [
        _validate_decision(
            value,
            index=index,
            cadence_id=cadence_id,
            plan=validated_plan,
            window=windows[window_id],
        )
        for index, value in enumerate(_sequence(row["decisions"], "decisions"))
    ]
    decision_ids = [decision["decision_id"] for decision in decisions]
    decision_epochs = [int(decision["decision_epoch"]) for decision in decisions]
    if (
        len(decision_ids) != len(set(decision_ids))
        or decision_epochs != sorted(decision_epochs)
    ):
        raise DojoCounterfactualSupervisionError(
            "decision IDs must be unique and epochs chronological"
        )
    if cadence_id is not None and cadence_id.startswith("FIXED_"):
        fixed_seconds = int(cadence_id.removeprefix("FIXED_").removesuffix("M")) * 60
        if any(
            later - earlier < fixed_seconds
            for earlier, later in zip(decision_epochs, decision_epochs[1:])
        ):
            raise DojoCounterfactualSupervisionError(
                "fixed-cadence decisions are more frequent than preregistered"
            )
    if arm == BOT_ONLY:
        if decisions or any(
            metrics[field] != 0
            for field in ("ai_call_count", "ai_token_count", "ai_latency_ms_total")
        ):
            raise DojoCounterfactualSupervisionError(
                "BOT_ONLY contains AI decisions or AI costs"
            )
    elif int(metrics["ai_call_count"]) != len(decisions):
        raise DojoCounterfactualSupervisionError(
            "AI call count differs from the decision evidence denominator"
        )
    for field in (
        "economic_transcript_sha256",
        "orders_sha256",
        "fills_sha256",
        "inventory_sha256",
    ):
        _sha256(row[field], field)
    body = {
        "contract": CELL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": validated_plan["plan_sha256"],
        **row,
        "decisions": decisions,
        "authority": dict(_AUTHORITY),
    }
    return {**body, "cell_sha256": canonical_portfolio_sha256(body)}


def _one_sided_sign_test_probability(positive: int, total: int) -> float:
    return sum(math.comb(total, index) for index in range(positive, total + 1)) / (
        2**total
    )


def evaluate_counterfactual_supervision(
    *, plan: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return an OOS-only cadence evaluation, or UNRANKED with exact blockers."""

    validated_plan = validate_counterfactual_supervision_plan(plan)
    windows = {row["window_id"]: row for row in validated_plan["market_windows"]}
    expected = {
        (window_id, BOT_ONLY, None)
        for window_id in windows
    } | {
        (window_id, "AI_MANAGED", cadence_id)
        for window_id in windows
        for cadence_id in CADENCE_IDS
    }
    observed: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    invalid_cells: list[str] = []
    for index, value in enumerate(cells):
        try:
            sealed = seal_counterfactual_supervision_cell(
                plan=validated_plan, cell=value
            )
        except (DojoCounterfactualSupervisionError, TypeError, ValueError) as exc:
            invalid_cells.append(f"cell[{index}]:{exc}")
            continue
        key = (sealed["window_id"], sealed["arm"], sealed["cadence_id"])
        if key in observed:
            invalid_cells.append(f"duplicate:{key}")
        else:
            observed[key] = sealed
    missing = sorted(
        f"{window_id}|{arm}|{cadence_id or 'NONE'}"
        for window_id, arm, cadence_id in expected - set(observed)
    )
    blockers = []
    if invalid_cells:
        blockers.append("INVALID_OR_FAILED_CELL")
    if missing:
        blockers.append("FIXED_DENOMINATOR_INCOMPLETE")
    oos_ids = [
        window_id
        for window_id, row in windows.items()
        if row["partition"] == "OOS"
    ]
    if len(oos_ids) < validated_plan["selection_policy"]["minimum_oos_blocks"]:
        blockers.append("INSUFFICIENT_WALK_FORWARD_OOS_BLOCKS")

    cadence_rows: list[dict[str, Any]] = []
    if not blockers:
        corrected_alpha = (
            validated_plan["selection_policy"]["familywise_alpha"]
            / len(CADENCE_IDS)
        )
        for cadence_id in CADENCE_IDS:
            pnl_deltas = []
            drawdown_deltas = []
            margin_call_delta = 0
            ruin_delta = 0
            opportunity_loss = 0.0
            ai_calls = 0.0
            ai_tokens = 0.0
            ai_latency = 0.0
            for window_id in oos_ids:
                bot_metrics = observed[(window_id, BOT_ONLY, None)]["metrics"]
                ai_metrics = observed[
                    (window_id, "AI_MANAGED", cadence_id)
                ]["metrics"]
                pnl_deltas.append(
                    float(ai_metrics["net_pnl_jpy"])
                    - float(bot_metrics["net_pnl_jpy"])
                )
                drawdown_deltas.append(
                    float(ai_metrics["max_drawdown_fraction"])
                    - float(bot_metrics["max_drawdown_fraction"])
                )
                margin_call_delta += int(ai_metrics["margin_call_count"]) - int(
                    bot_metrics["margin_call_count"]
                )
                ruin_delta += int(ai_metrics["ruin_event_count"]) - int(
                    bot_metrics["ruin_event_count"]
                )
                opportunity_loss += float(ai_metrics["opportunity_loss_jpy"])
                ai_calls += float(ai_metrics["ai_call_count"])
                ai_tokens += float(ai_metrics["ai_token_count"])
                ai_latency += float(ai_metrics["ai_latency_ms_total"])
            positive = sum(delta > 0 for delta in pnl_deltas)
            p_value = _one_sided_sign_test_probability(positive, len(pnl_deltas))
            eligible = (
                median(pnl_deltas) > 0
                and median(drawdown_deltas) <= 0
                and margin_call_delta <= 0
                and ruin_delta <= 0
                and p_value <= corrected_alpha
            )
            cadence_rows.append(
                {
                    "cadence_id": cadence_id,
                    "oos_block_count": len(oos_ids),
                    "positive_net_pnl_delta_block_count": positive,
                    "median_net_pnl_delta_jpy": median(pnl_deltas),
                    "median_max_drawdown_fraction_delta": median(drawdown_deltas),
                    "aggregate_margin_call_count_delta": margin_call_delta,
                    "aggregate_ruin_event_count_delta": ruin_delta,
                    "aggregate_opportunity_loss_jpy": opportunity_loss,
                    "aggregate_ai_call_count": int(ai_calls),
                    "aggregate_ai_token_count": int(ai_tokens),
                    "aggregate_ai_latency_ms": ai_latency,
                    "one_sided_sign_test_p_value": p_value,
                    "bonferroni_alpha": corrected_alpha,
                    "paper_shadow_eligible": eligible,
                }
            )
    eligible_rows = [row for row in cadence_rows if row["paper_shadow_eligible"]]
    eligible_rows.sort(
        key=lambda row: (
            -row["median_net_pnl_delta_jpy"],
            row["median_max_drawdown_fraction_delta"],
            row["aggregate_ai_call_count"],
            row["cadence_id"],
        )
    )
    selected = eligible_rows[0]["cadence_id"] if eligible_rows else None
    status = "RANKED_OOS" if selected is not None else "UNRANKED"
    if not blockers and selected is None:
        blockers.append("NO_CADENCE_PASSED_PREREGISTERED_ROBUSTNESS_GATE")
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": validated_plan["plan_sha256"],
        "status": status,
        "fixed_expected_cell_count": len(expected),
        "valid_cell_count": len(observed),
        "missing_cell_ids": missing,
        "invalid_cell_errors": invalid_cells,
        "blockers": blockers,
        "rank_partition": "OOS_ONLY",
        "cadence_rows": cadence_rows,
        "baseline_cadence_id": BASELINE_CADENCE_ID,
        "selected_paper_shadow_cadence_id": selected,
        "paper_shadow_eligible": selected is not None,
        "live_or_broker_authority_granted": False,
        "profit_guaranteed": False,
        "authority": dict(_AUTHORITY),
    }
    return {**body, "evaluation_sha256": canonical_portfolio_sha256(body)}


__all__ = [
    "BASELINE_CADENCE_ID",
    "BOT_ONLY",
    "CADENCE_IDS",
    "CELL_CONTRACT",
    "DojoCounterfactualSupervisionError",
    "EVALUATION_CONTRACT",
    "PLAN_CONTRACT",
    "build_counterfactual_supervision_plan",
    "evaluate_counterfactual_supervision",
    "seal_counterfactual_supervision_cell",
    "validate_counterfactual_supervision_plan",
]
