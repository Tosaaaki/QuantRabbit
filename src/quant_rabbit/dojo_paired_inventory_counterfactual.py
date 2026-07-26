"""Causal paired inventory-supervision replay over immutable DOJO transcripts.

The runner branches only the worker proposal stream. It consumes the exact
recorded quote path, portfolio policy, initial capital, costs, and worker
intents, while a frozen research-only inventory policy may suppress new risk
or add owner-bound reductions. It has no broker, live, network, model-provider,
or promotion capability.

The current implementation intentionally distinguishes a *model-authored
frozen policy* from provider calls at each checkpoint. Results produced without
an externally sealed decision provider remain experimental and UNRANKED.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Final

from quant_rabbit.dojo_economic_transcript import (
    TRANSCRIPT_HEADER_CONTRACT,
    TRANSCRIPT_RECORD_CONTRACT,
    _strict_json_line,
    verify_economic_transcript_header,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    seal_worker_proposal,
    seal_worker_proposal_batch,
    verify_post_exit_snapshot,
    verify_worker_proposal_batch,
)


RESULT_CONTRACT: Final = "QR_DOJO_PAIRED_INVENTORY_COUNTERFACTUAL_RESULT_V1"
PLAN_CONTRACT: Final = "QR_DOJO_PAIRED_INVENTORY_COUNTERFACTUAL_PLAN_V1"
SCHEMA_VERSION: Final = 1
CADENCE_IDS: Final = (
    "FIXED_5M",
    "FIXED_15M",
    "FIXED_30M",
    "FIXED_60M",
    "FIXED_120M",
    "EVENT_DRIVEN",
    "ADAPTIVE_60M_15M_EVENT",
)
ACTION_IDS: Final = (
    "HOLD",
    "PAUSE_NEW_ENTRIES",
    "RESUME",
    "REDUCE_LONG",
    "REDUCE_SHORT",
    "PARTIAL_CLOSE",
    "CLOSE_RISKY",
    "CLOSE_ALL",
    "BLOCK_LONG_ENTRIES",
    "BLOCK_SHORT_ENTRIES",
)
AUTHORITY: Final = {
    "research_only": True,
    "paper_replay_only": True,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "order_authority": "NONE",
    "automatic_deployment_allowed": False,
    "promotion_eligible": False,
}


class DojoPairedInventoryCounterfactualError(ValueError):
    """The transcript, pairing, causal packet, or research boundary is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def build_paired_inventory_plan(
    *,
    study_id: str,
    source_job_sha256: str,
    source_job_result_sha256: str,
    transcript_sha256_by_coordinate: Mapping[str, str],
    calibration_start_epoch: int,
    calibration_end_epoch: int,
    oos_blocks: Sequence[Mapping[str, Any]],
    source_quote_coverage_proved: bool,
    researcher_prior_aggregate_outcome_exposure: bool,
) -> dict[str, Any]:
    """Build the immutable experimental denominator before OOS replay."""

    blocks = [dict(row) for row in oos_blocks]
    if len(blocks) != 8:
        raise DojoPairedInventoryCounterfactualError(
            "exactly eight non-overlapping OOS blocks are required"
        )
    prior_end = calibration_end_epoch
    for index, row in enumerate(blocks):
        if set(row) != {"block_id", "start_epoch", "end_epoch"}:
            raise DojoPairedInventoryCounterfactualError("OOS block schema mismatch")
        if (
            row["start_epoch"] != prior_end
            or not isinstance(row["end_epoch"], int)
            or row["end_epoch"] <= row["start_epoch"]
        ):
            raise DojoPairedInventoryCounterfactualError(
                "OOS blocks must be contiguous and chronological"
            )
        row["block_id"] = f"OOS_{index + 1:02d}"
        prior_end = row["end_epoch"]
    if not isinstance(source_quote_coverage_proved, bool):
        raise DojoPairedInventoryCounterfactualError(
            "source quote coverage flag must be boolean"
        )
    body = {
        "contract": PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "source_job_sha256": source_job_sha256,
        "source_job_result_sha256": source_job_result_sha256,
        "transcript_sha256_by_coordinate": dict(
            sorted(transcript_sha256_by_coordinate.items())
        ),
        "calibration_window": {
            "start_epoch": calibration_start_epoch,
            "end_epoch": calibration_end_epoch,
            "outcome_use": "STATE_DISTRIBUTION_ONLY_NO_OOS_OUTCOME",
        },
        "oos_blocks": blocks,
        "cadence_ids": list(CADENCE_IDS),
        "action_ids": list(ACTION_IDS),
        "decision_information_policy": (
            "SNAPSHOT_AND_HISTORY_AVAILABLE_THROUGH_DECISION_EPOCH_ONLY"
        ),
        "append_wall_clock_allowed": False,
        "future_quote_allowed": False,
        "terminal_result_allowed_in_decision": False,
        "model_execution_mode": (
            "FROZEN_MODEL_AUTHORED_CAUSAL_POLICY_NO_PROVIDER_CALL"
        ),
        "actual_model_checkpoint_call_required_for_rank": True,
        "source_quote_coverage_proved": source_quote_coverage_proved,
        "researcher_prior_aggregate_outcome_exposure": (
            researcher_prior_aggregate_outcome_exposure
        ),
        "classification": "EXPERIMENTAL_WORN_TRAIN",
        "authority": dict(AUTHORITY),
    }
    return {**body, "plan_sha256": canonical_portfolio_sha256(body)}


def _quote_to_jpy(
    amount: float,
    currency: str,
    quote_map: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Mapping[str, Any]],
) -> float:
    if currency == "JPY" or amount == 0:
        return amount
    route = routes[currency]
    quote = quote_map[route["pair"]]
    if route["orientation"] == "JPY_PER_CURRENCY":
        factor = float(quote["bid"] if amount > 0 else quote["ask"])
    else:
        denominator = float(quote["ask"] if amount > 0 else quote["bid"])
        factor = 1.0 / denominator
    return amount * factor


def _position_values(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    valuation_quotes: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    quotes = (
        dict(valuation_quotes)
        if valuation_quotes is not None
        else {row["pair"]: row for row in snapshot["quotes"]}
    )
    routes = {row["currency"]: row for row in policy["conversion_routes"]}
    values = []
    for position in snapshot["positions"]:
        base, quote_currency = position["pair"].split("_", 1)
        quote = quotes[position["pair"]]
        mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
        notional = abs(
            _quote_to_jpy(
                float(position["units"]) * mid,
                quote_currency,
                quotes,
                routes,
            )
        )
        exit_price = float(
            quote["bid"] if position["side"] == "LONG" else quote["ask"]
        )
        quote_pnl = float(position["units"]) * (
            exit_price - float(position["entry_price"])
            if position["side"] == "LONG"
            else float(position["entry_price"]) - exit_price
        )
        values.append(
            {
                **position,
                "notional_jpy": notional,
                "unrealized_pnl_jpy": _quote_to_jpy(
                    quote_pnl, quote_currency, quotes, routes
                ),
                "base_currency": base,
            }
        )
    return values


@dataclass
class _Arm:
    cadence_id: str
    session: PortfolioReplaySession
    initial_balance_jpy: float
    policy: Mapping[str, Any]
    current_epoch: int = 0
    paused: bool = False
    direction_block: str | None = None
    last_decision_epoch: int | None = None
    last_packet_sha256: str | None = None
    last_gross_exposure_jpy: float = 0.0
    last_drawdown_fraction: float = 0.0
    last_regime_id: str = "UNKNOWN"
    peak_equity_jpy: float = 0.0
    peak_balance_jpy: float = 0.0
    consecutive_losses: int = 0
    close_pnls_jpy: list[float] = field(default_factory=list)
    close_events: list[dict[str, Any]] = field(default_factory=list)
    close_reason_counts: dict[str, int] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    skipped_cached_calls: int = 0
    close_history: dict[str, deque[float]] = field(default_factory=dict)
    block_equity: dict[str, list[float]] = field(default_factory=dict)
    block_margin: dict[str, list[float]] = field(default_factory=dict)
    block_start_balance: dict[str, float] = field(default_factory=dict)
    block_end_balance: dict[str, float] = field(default_factory=dict)

    def observe_event(self, event: Mapping[str, Any]) -> None:
        if event["kind"] != "POSITION_CLOSE":
            return
        payload = event["payload"]
        pnl = float(payload["pnl_jpy"])
        self.close_pnls_jpy.append(pnl)
        reason = str(payload["reason"])
        self.close_events.append(
            {"epoch": self.current_epoch, "pnl_jpy": pnl, "reason": reason}
        )
        self.close_reason_counts[reason] = self.close_reason_counts.get(reason, 0) + 1
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0


def _new_arm(
    cadence_id: str, policy: Mapping[str, Any], initial_balance_jpy: float
) -> _Arm:
    holder: dict[str, _Arm] = {}

    def listener(event: Mapping[str, Any]) -> None:
        holder["arm"].observe_event(event)

    session = PortfolioReplaySession(
        policy=policy,
        initial_balance_jpy=initial_balance_jpy,
        event_listener=listener,
    )
    arm = _Arm(
        cadence_id=cadence_id,
        session=session,
        initial_balance_jpy=initial_balance_jpy,
        policy=policy,
        peak_equity_jpy=initial_balance_jpy,
        peak_balance_jpy=initial_balance_jpy,
    )
    holder["arm"] = arm
    return arm


def _regime(
    arm: _Arm, snapshot: Mapping[str, Any], values: Sequence[Mapping[str, Any]]
) -> str:
    if snapshot["phase"] != "C":
        return arm.last_regime_id
    quotes = {row["pair"]: row for row in snapshot["quotes"]}
    returns = []
    for value in values:
        pair = str(value["pair"])
        mid = (float(quotes[pair]["bid"]) + float(quotes[pair]["ask"])) / 2.0
        history = arm.close_history.setdefault(pair, deque(maxlen=72))
        if history and history[-1] > 0:
            returns.append(mid / history[-1] - 1.0)
        history.append(mid)
    if not returns:
        return "UNKNOWN"
    mean_abs = sum(abs(value) for value in returns) / len(returns)
    directional = abs(sum(returns) / len(returns))
    if mean_abs >= 0.0015:
        return "HIGH_VOLATILITY"
    if directional >= max(0.00035, mean_abs * 0.65):
        return "TREND"
    return "RANGE"


def _packet(
    arm: _Arm,
    snapshot: Mapping[str, Any],
    *,
    family_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = arm.session._prepared  # reducer-owned causal valuation view
    valuation_quotes = None if prepared is None else prepared["quotes"]
    values = _position_values(
        snapshot,
        arm.policy,
        valuation_quotes=valuation_quotes,
    )
    equity = float(snapshot["account"]["equity_jpy"])
    balance = float(snapshot["account"]["balance_jpy"])
    margin = float(snapshot["account"]["margin_used_jpy"])
    arm.peak_equity_jpy = max(arm.peak_equity_jpy, equity)
    arm.peak_balance_jpy = max(arm.peak_balance_jpy, balance)
    drawdown = (
        max(0.0, (arm.peak_equity_jpy - equity) / arm.peak_equity_jpy)
        if arm.peak_equity_jpy > 0
        else 1.0
    )
    gross = sum(float(row["notional_jpy"]) for row in values)
    signed = sum(
        float(row["notional_jpy"]) * (1.0 if row["side"] == "LONG" else -1.0)
        for row in values
    )
    long_gross = sum(
        float(row["notional_jpy"]) for row in values if row["side"] == "LONG"
    )
    short_gross = gross - long_gross
    regime_id = _regime(arm, snapshot, values)
    compatible = (
        regime_id
        in (
            {"RANGE", "UNKNOWN"}
            if any(token in family_id for token in ("fade", "mean_revert", "round"))
            else {"TREND", "UNKNOWN"}
        )
    )
    packet = {
        "decision_epoch": int(snapshot["epoch"]),
        "input_available_through_epoch": int(snapshot["epoch"]),
        "phase": snapshot["phase"],
        "equity_jpy": round(equity, 6),
        "balance_jpy": round(balance, 6),
        "drawdown_fraction": round(drawdown, 9),
        "margin_utilization_fraction": round(margin / max(equity, 1e-9), 9),
        "gross_exposure_jpy": round(gross, 4),
        "net_exposure_jpy": round(signed, 4),
        "long_gross_exposure_jpy": round(long_gross, 4),
        "short_gross_exposure_jpy": round(short_gross, 4),
        "hedge_buildup_fraction": round(
            0.0 if gross == 0 else 1.0 - abs(signed) / gross, 9
        ),
        "directional_skew_fraction": round(
            0.0 if gross == 0 else signed / gross, 9
        ),
        "unrealized_pnl_jpy": round(
            sum(float(row["unrealized_pnl_jpy"]) for row in values), 6
        ),
        "realized_profit_giveback_jpy": round(
            max(0.0, arm.peak_balance_jpy - balance), 6
        ),
        "position_count": len(values),
        "pending_order_count": len(snapshot["pending_orders"]),
        "stale_valuation_pair_count": sum(
            value is None or int(value) > 0
            for value in snapshot.get(
                "pair_local_quote_age_seconds",
                {row["pair"]: 0 for row in snapshot["quotes"]},
            ).values()
        ),
        "maximum_position_age_seconds": max(
            (int(snapshot["epoch"]) - int(row["opened_epoch"]) for row in values),
            default=0,
        ),
        "consecutive_losses": arm.consecutive_losses,
        "regime_id": regime_id,
        "strategy_regime_compatible": compatible,
        "paused": arm.paused,
        "direction_block": arm.direction_block,
        "terminal_result_visible": False,
        "future_quote_visible": False,
        "append_wall_clock_visible": False,
    }
    return packet, values


def _calibrated_thresholds(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    drawdowns = [float(row["drawdown_fraction"]) for row in samples]
    margins = [float(row["margin_utilization_fraction"]) for row in samples]
    gross = [float(row["gross_exposure_jpy"]) for row in samples]
    givebacks = [float(row["realized_profit_giveback_jpy"]) for row in samples]
    return {
        # Floors are ex-ante account-risk conventions; percentiles adapt them
        # upward to the calibration state distribution without consulting OOS.
        "pause_drawdown_fraction": max(0.03, _percentile(drawdowns, 0.75)),
        "close_drawdown_fraction": max(0.06, _percentile(drawdowns, 0.95)),
        "reduce_margin_fraction": max(0.08, _percentile(margins, 0.90)),
        "close_margin_fraction": max(0.16, _percentile(margins, 0.99)),
        "gross_spike_jpy": max(20_000.0, _percentile(gross, 0.90) * 0.25),
        "profit_giveback_jpy": max(4_000.0, _percentile(givebacks, 0.90)),
        "resume_drawdown_fraction": max(0.01, _percentile(drawdowns, 0.50)),
    }


def _event_ids(
    arm: _Arm, packet: Mapping[str, Any], thresholds: Mapping[str, float]
) -> list[str]:
    events = []
    if float(packet["margin_utilization_fraction"]) >= thresholds["reduce_margin_fraction"]:
        events.append("MARGIN_UTILIZATION_THRESHOLD")
    if (
        float(packet["gross_exposure_jpy"]) - arm.last_gross_exposure_jpy
        >= thresholds["gross_spike_jpy"]
    ):
        events.append("GROSS_EXPOSURE_SPIKE")
    if (
        float(packet["drawdown_fraction"]) - arm.last_drawdown_fraction
        >= 0.01
    ):
        events.append("DRAWDOWN_DETERIORATION")
    if packet["regime_id"] != arm.last_regime_id:
        events.append("VOLATILITY_REGIME_CHANGE")
    if packet["strategy_regime_compatible"] is False:
        events.append("STRATEGY_THESIS_INVALIDATION")
    if int(packet["consecutive_losses"]) >= 3:
        events.append("CONSECUTIVE_LOSSES")
    if int(packet["maximum_position_age_seconds"]) >= 7_200:
        events.append("POSITION_AGE")
    return events


def _cadence_due(
    arm: _Arm,
    packet: Mapping[str, Any],
    events: Sequence[str],
    thresholds: Mapping[str, float],
) -> tuple[bool, str]:
    if packet["phase"] != "C":
        return False, "NOT_CANDLE_CLOSE"
    epoch = int(packet["decision_epoch"])
    elapsed = math.inf if arm.last_decision_epoch is None else epoch - arm.last_decision_epoch
    if arm.cadence_id.startswith("FIXED_"):
        minutes = int(arm.cadence_id.removeprefix("FIXED_").removesuffix("M"))
        return elapsed >= minutes * 60, f"FIXED_{minutes}M"
    if arm.cadence_id == "EVENT_DRIVEN":
        return bool(events), "MAJOR_EVENT"
    if events:
        return True, "MAJOR_EVENT"
    high_risk = (
        float(packet["drawdown_fraction"]) >= thresholds["pause_drawdown_fraction"]
        or float(packet["margin_utilization_fraction"])
        >= thresholds["reduce_margin_fraction"]
    )
    interval = 900 if high_risk else 3_600
    return elapsed >= interval, "HIGH_RISK_15M" if high_risk else "HEARTBEAT_60M"


def _decide(
    arm: _Arm,
    packet: Mapping[str, Any],
    values: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    drawdown = float(packet["drawdown_fraction"])
    margin = float(packet["margin_utilization_fraction"])
    giveback = float(packet["realized_profit_giveback_jpy"])
    unrealized = float(packet["unrealized_pnl_jpy"])
    skew = float(packet["directional_skew_fraction"])
    if values and (
        drawdown >= thresholds["close_drawdown_fraction"]
        or margin >= thresholds["close_margin_fraction"]
    ):
        reasons.append("ACCOUNT_TAIL_RISK")
        return "CLOSE_RISKY", reasons
    if values and margin >= thresholds["reduce_margin_fraction"]:
        reasons.append("MARGIN_BUFFER")
        return "PARTIAL_CLOSE", reasons
    if values and giveback >= thresholds["profit_giveback_jpy"] and unrealized < 0:
        reasons.append("REALIZED_PROFIT_GIVEBACK")
        return "CLOSE_RISKY", reasons
    if values and abs(skew) >= 0.85 and packet["strategy_regime_compatible"] is False:
        reasons.append("REGIME_DIRECTION_CONCENTRATION")
        return ("REDUCE_LONG" if skew > 0 else "REDUCE_SHORT"), reasons
    if drawdown >= thresholds["pause_drawdown_fraction"]:
        reasons.append("DRAWDOWN_PAUSE")
        return "PAUSE_NEW_ENTRIES", reasons
    if (arm.paused or arm.direction_block is not None) and (
        drawdown <= thresholds["resume_drawdown_fraction"]
        and packet["strategy_regime_compatible"] is True
    ):
        reasons.append("PREREGISTERED_COMPATIBLE_REGIME_RESUME")
        return "RESUME", reasons
    if packet["strategy_regime_compatible"] is False:
        reasons.append("REGIME_DIRECTION_BLOCK")
        return (
            "BLOCK_LONG_ENTRIES" if skew < 0 else "BLOCK_SHORT_ENTRIES"
        ), reasons
    return "HOLD", ["NO_MATERIAL_RISK_CHANGE"]


def _matching_position(
    baseline_position: Mapping[str, Any],
    ai_snapshot: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in ai_snapshot["positions"]
        if row["worker_id"] == baseline_position["worker_id"]
        and row["pair"] == baseline_position["pair"]
        and row["side"] == baseline_position["side"]
    ]
    return matches[0] if len(matches) == 1 else None


def _matching_order(
    baseline_order: Mapping[str, Any],
    ai_snapshot: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in ai_snapshot["pending_orders"]
        if row["worker_id"] == baseline_order["worker_id"]
        and row["pair"] == baseline_order["pair"]
        and row["side"] == baseline_order["side"]
    ]
    return matches[0] if len(matches) == 1 else None


def _transform_batch(
    *,
    original_snapshot: Mapping[str, Any],
    original_batch: Mapping[str, Any],
    ai_snapshot: Mapping[str, Any],
    arm: _Arm,
    action: str,
) -> dict[str, Any]:
    baseline_positions = {
        row["position_id"]: row for row in original_snapshot["positions"]
    }
    baseline_orders = {
        row["order_id"]: row for row in original_snapshot["pending_orders"]
    }
    prepared = arm.session._prepared
    valuation_quotes = None if prepared is None else prepared["quotes"]
    ai_values = _position_values(
        ai_snapshot,
        arm.policy,
        valuation_quotes=valuation_quotes,
    )
    fresh_pairs = set(
        ai_snapshot.get("fresh_quote_pairs", ai_snapshot["expected_quote_pairs"])
    )
    routes = {
        row["currency"]: row for row in arm.policy["conversion_routes"]
    }

    def close_dependencies_are_fresh(position: Mapping[str, Any]) -> bool:
        if position["pair"] not in fresh_pairs:
            return False
        quote_currency = str(position["pair"]).split("_", 1)[1]
        if quote_currency == "JPY":
            return True
        return routes[quote_currency]["pair"] in fresh_pairs
    proposals = []
    for proposal_index, sealed in enumerate(original_batch["proposals"]):
        reductions = []
        for intent in sealed["risk_reducing_intents"]:
            copied = {
                "intent_id": intent["intent_id"],
                "action": intent["action"],
                "parameters": dict(intent["parameters"]),
                "reason_code": intent["reason_code"],
            }
            if intent["action"] == "CANCEL_ORDER":
                baseline = baseline_orders.get(intent["parameters"]["order_id"])
                target = None if baseline is None else _matching_order(baseline, ai_snapshot)
                if target is None:
                    continue
                copied["parameters"]["order_id"] = target["order_id"]
            else:
                baseline = baseline_positions.get(
                    intent["parameters"]["position_id"]
                )
                target = (
                    None
                    if baseline is None
                    else _matching_position(baseline, ai_snapshot)
                )
                if target is None:
                    continue
                copied["parameters"]["position_id"] = target["position_id"]
                if intent["action"] == "CLOSE_POSITION" and copied["parameters"]["units"] is not None:
                    copied["parameters"]["units"] = min(
                        float(copied["parameters"]["units"]), float(target["units"])
                    )
            reductions.append(copied)
        additions: list[tuple[Mapping[str, Any], float | None]] = []
        if proposal_index == 0:
            if action == "CLOSE_ALL":
                additions = [(row, None) for row in ai_values]
            elif action == "CLOSE_RISKY" and ai_values:
                additions = [
                    (min(ai_values, key=lambda row: row["unrealized_pnl_jpy"]), None)
                ]
            elif action == "PARTIAL_CLOSE":
                additions = [
                    (row, max(1e-9, float(row["units"]) * 0.25))
                    for row in ai_values
                ]
            elif action in {"REDUCE_LONG", "REDUCE_SHORT"}:
                side = action.removeprefix("REDUCE_")
                additions = [
                    (row, max(1e-9, float(row["units"]) * 0.5))
                    for row in ai_values
                    if row["side"] == side
                ]
        existing_targets = {
            row["parameters"].get("position_id")
            for row in reductions
            if row["action"] == "CLOSE_POSITION"
        }
        for index, (position, units) in enumerate(additions):
            if position["position_id"] in existing_targets:
                continue
            # A stale last-observed conversion mark is valid for causal MTM,
            # but never for an AI-initiated executable close.
            if not close_dependencies_are_fresh(position):
                continue
            reductions.append(
                {
                    "intent_id": (
                        f"ai-{arm.cadence_id.lower()}-{ai_snapshot['quote_watermark']}-{index}"
                    ),
                    "action": "CLOSE_POSITION",
                    "parameters": {
                        "position_id": position["position_id"],
                        "units": units,
                    },
                    "reason_code": "AI_CAUSAL_INVENTORY_RISK_CONTROL",
                }
            )
        new_risk = []
        for intent in sealed["new_risk_intents"]:
            copied = {
                "intent_id": intent["intent_id"],
                "action": intent["action"],
                "parameters": {
                    key: value
                    for key, value in intent["parameters"].items()
                    if key != "activation_policy"
                },
                "reason_code": intent["reason_code"],
            }
            side = copied["parameters"]["side"]
            blocked = (
                arm.paused
                or action == "PAUSE_NEW_ENTRIES"
                or action == f"BLOCK_{side}_ENTRIES"
                or arm.direction_block == side
            )
            if not blocked:
                new_risk.append(copied)
        raw = {
            "worker_id": sealed["worker_id"],
            "owner_id": sealed["owner_id"],
            "family_id": sealed["family_id"],
            "config_sha256": sealed["config_sha256"],
            "snapshot_sha256": ai_snapshot["snapshot_sha256"],
            "risk_reducing_intents": reductions,
            "new_risk_intents": new_risk,
        }
        proposals.append(seal_worker_proposal(ai_snapshot, raw))
    return seal_worker_proposal_batch(ai_snapshot, proposals)


def _apply_action_state(arm: _Arm, action: str) -> None:
    if action == "PAUSE_NEW_ENTRIES":
        arm.paused = True
    elif action == "RESUME":
        arm.paused = False
        arm.direction_block = None
    elif action == "BLOCK_LONG_ENTRIES":
        arm.direction_block = "LONG"
    elif action == "BLOCK_SHORT_ENTRIES":
        arm.direction_block = "SHORT"


def _block_id(epoch: int, blocks: Sequence[Mapping[str, Any]]) -> str | None:
    for block in blocks:
        if int(block["start_epoch"]) <= epoch < int(block["end_epoch"]):
            return str(block["block_id"])
    return None


def _profit_metrics(values: Sequence[float]) -> dict[str, Any]:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "close_event_count": len(values),
        "win_rate": None if not values else len(wins) / len(values),
        "profit_factor": (
            None
            if gross_loss == 0
            else gross_profit / gross_loss
        ),
        "expectancy_jpy_per_close": (
            None if not values else sum(values) / len(values)
        ),
        "gross_profit_jpy": gross_profit,
        "gross_loss_jpy": gross_loss,
    }


def replay_paired_inventory_transcript(
    *,
    transcript_path: Path,
    plan: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    cost_scenario: str,
) -> dict[str, Any]:
    """Run all seven AI cadence arms from one immutable transcript read."""

    if plan.get("contract") != PLAN_CONTRACT:
        raise DojoPairedInventoryCounterfactualError("plan contract is invalid")
    blocks = list(plan["oos_blocks"])
    calibration = plan["calibration_window"]
    expected_file_sha = plan["transcript_sha256_by_coordinate"]
    header: dict[str, Any] | None = None
    arms: dict[str, _Arm] = {}
    recorded_snapshot: dict[str, Any] | None = None
    recorded_batch: dict[str, Any] | None = None
    current_quote: dict[str, Any] | None = None
    ai_snapshots: dict[str, dict[str, Any]] = {}
    calibration_samples: list[dict[str, Any]] = []
    thresholds: dict[str, float] | None = None
    prior_record_sha = "0" * 64
    record_index = 0
    file_digest = hashlib.sha256()
    terminal_seen = False
    family_id = ""
    baseline_block_equity: dict[str, list[float]] = {
        row["block_id"]: [] for row in blocks
    }
    baseline_block_margin: dict[str, list[float]] = {
        row["block_id"]: [] for row in blocks
    }
    baseline_block_balance: dict[str, list[float]] = {
        row["block_id"]: [] for row in blocks
    }

    with Path(transcript_path).open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            file_digest.update(raw)
            row = dict(_strict_json_line(raw, line_number=line_number))
            if (
                row.get("contract") != TRANSCRIPT_RECORD_CONTRACT
                or row.get("record_index") != record_index
                or row.get("previous_record_sha256") != prior_record_sha
            ):
                raise DojoPairedInventoryCounterfactualError(
                    "transcript record chain is invalid"
                )
            unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
            if canonical_portfolio_sha256(unsigned) != row.get("record_sha256"):
                raise DojoPairedInventoryCounterfactualError(
                    "transcript record digest is invalid"
                )
            event_type = row["event_type"]
            payload = row["payload"]
            if event_type == "HEADER":
                if header is not None or payload.get("contract") != TRANSCRIPT_HEADER_CONTRACT:
                    raise DojoPairedInventoryCounterfactualError("header is invalid")
                header = verify_economic_transcript_header(payload)
                family_id = header["portfolio_policy"]["active_worker_bindings"][0][
                    "family_id"
                ]
                arms = {
                    cadence: _new_arm(
                        cadence,
                        header["portfolio_policy"],
                        float(header["initial_balance_jpy"]),
                    )
                    for cadence in CADENCE_IDS
                }
            elif event_type == "QUOTE_BATCH":
                current_quote = dict(payload)
            elif event_type == "POST_EXIT_SNAPSHOT":
                if header is None or current_quote is None:
                    raise DojoPairedInventoryCounterfactualError(
                        "snapshot precedes header/quote"
                    )
                recorded_snapshot = verify_post_exit_snapshot(payload["snapshot"])
                epoch = int(recorded_snapshot["epoch"])
                block_id = _block_id(epoch, blocks)
                if block_id is not None:
                    account = recorded_snapshot["account"]
                    baseline_block_equity[block_id].append(float(account["equity_jpy"]))
                    baseline_block_margin[block_id].append(
                        float(account["margin_used_jpy"])
                        / max(float(account["equity_jpy"]), 1e-9)
                    )
                    baseline_block_balance[block_id].append(
                        float(account["balance_jpy"])
                    )
                for cadence, arm in arms.items():
                    arm.current_epoch = epoch
                    sparse = "fresh_quote_pairs" in recorded_snapshot
                    kwargs = {}
                    if sparse:
                        kwargs = {
                            "fresh_quote_pairs": recorded_snapshot["fresh_quote_pairs"],
                            "unavailable_quote_pairs": recorded_snapshot[
                                "unavailable_quote_pairs"
                            ],
                            "pair_local_quote_age_seconds": recorded_snapshot[
                                "pair_local_quote_age_seconds"
                            ],
                            "quote_policy": recorded_snapshot["quote_policy"],
                        }
                    ai_snapshots[cadence] = arm.session.prepare_coordinate(
                        coordinate_id=current_quote["coordinate_id"],
                        epoch=current_quote["epoch"],
                        phase=current_quote["phase"],
                        intrabar=current_quote["intrabar"],
                        quote_watermark=current_quote["quote_watermark"],
                        quotes=current_quote["quotes"],
                        quote_batch_sha256_value=current_quote["quote_batch_sha256"],
                        **kwargs,
                    )
                if (
                    calibration["start_epoch"]
                    <= epoch
                    < calibration["end_epoch"]
                    and recorded_snapshot["phase"] == "C"
                ):
                    for cadence, arm in arms.items():
                        calibration_packet, _ = _packet(
                            arm,
                            ai_snapshots[cadence],
                            family_id=family_id,
                        )
                        if cadence == CADENCE_IDS[0]:
                            calibration_samples.append(calibration_packet)
                if block_id is not None:
                    for cadence, arm in arms.items():
                        account = ai_snapshots[cadence]["account"]
                        arm.block_equity.setdefault(block_id, []).append(
                            float(account["equity_jpy"])
                        )
                        arm.block_margin.setdefault(block_id, []).append(
                            float(account["margin_used_jpy"])
                            / max(float(account["equity_jpy"]), 1e-9)
                        )
            elif event_type == "WORKER_PROPOSAL_BATCH":
                if header is None or recorded_snapshot is None:
                    raise DojoPairedInventoryCounterfactualError(
                        "proposal precedes snapshot"
                    )
                recorded_batch = verify_worker_proposal_batch(
                    recorded_snapshot, payload["proposal_batch"]
                )
                epoch = int(recorded_snapshot["epoch"])
                if thresholds is None and epoch >= calibration["end_epoch"]:
                    thresholds = _calibrated_thresholds(calibration_samples)
                for cadence, arm in arms.items():
                    ai_snapshot = ai_snapshots[cadence]
                    action = "HOLD"
                    decision: dict[str, Any] | None = None
                    if thresholds is not None and _block_id(epoch, blocks) is not None:
                        packet, values = _packet(
                            arm, ai_snapshot, family_id=family_id
                        )
                        events = _event_ids(arm, packet, thresholds)
                        due, trigger_kind = _cadence_due(
                            arm, packet, events, thresholds
                        )
                        packet_sha = _sha(packet)
                        if due and packet_sha == arm.last_packet_sha256 and not events:
                            arm.skipped_cached_calls += 1
                            due = False
                        if due:
                            action, reasons = _decide(
                                arm, packet, values, thresholds
                            )
                            decision = {
                                "decision_id": (
                                    f"{cadence}:{header['coordinate_id']}:{epoch}:"
                                    f"{len(arm.decisions) + 1}"
                                ),
                                "decision_epoch": epoch,
                                "input_available_through_epoch": epoch,
                                "packet_sha256": packet_sha,
                                "packet": packet,
                                "trigger_kind": trigger_kind,
                                "event_signal_ids": events,
                                "action": action,
                                "reason_ids": reasons,
                                "pre_action_equity_jpy": packet["equity_jpy"],
                                "pre_action_balance_jpy": packet["balance_jpy"],
                                "post_outcome": None,
                                "provider_model_called": False,
                                "future_information_used": False,
                            }
                            if arm.decisions and arm.decisions[-1]["post_outcome"] is None:
                                prior = arm.decisions[-1]
                                prior["post_outcome"] = {
                                    "observed_through_epoch": epoch,
                                    "equity_delta_jpy": (
                                        float(packet["equity_jpy"])
                                        - float(prior["pre_action_equity_jpy"])
                                    ),
                                    "balance_delta_jpy": (
                                        float(packet["balance_jpy"])
                                        - float(prior["pre_action_balance_jpy"])
                                    ),
                                }
                            arm.decisions.append(decision)
                            arm.last_decision_epoch = epoch
                            arm.last_packet_sha256 = packet_sha
                            arm.last_gross_exposure_jpy = float(
                                packet["gross_exposure_jpy"]
                            )
                            arm.last_drawdown_fraction = float(
                                packet["drawdown_fraction"]
                            )
                            arm.last_regime_id = str(packet["regime_id"])
                            _apply_action_state(arm, action)
                    transformed = _transform_batch(
                        original_snapshot=recorded_snapshot,
                        original_batch=recorded_batch,
                        ai_snapshot=ai_snapshot,
                        arm=arm,
                        action=action,
                    )
                    ai_receipt = arm.session.consume_proposal_batch(transformed)
                    block_id = _block_id(epoch, blocks)
                    if block_id is not None:
                        arm.block_equity.setdefault(block_id, []).append(
                            float(ai_receipt["ending_equity_jpy"])
                        )
                        arm.block_margin.setdefault(block_id, []).append(
                            float(ai_receipt["reserved_margin_jpy"])
                            / max(float(ai_receipt["ending_equity_jpy"]), 1e-9)
                        )
                        arm.block_start_balance.setdefault(
                            block_id, float(ai_snapshot["account"]["balance_jpy"])
                        )
                        arm.block_end_balance[block_id] = float(
                            ai_receipt["ending_balance_jpy"]
                        )
            elif event_type == "ALLOCATION_RECEIPT":
                if current_quote is None:
                    raise DojoPairedInventoryCounterfactualError(
                        "allocation receipt has no quote coordinate"
                    )
                block_id = _block_id(int(current_quote["epoch"]), blocks)
                if block_id is not None:
                    receipt = payload["receipt"]
                    baseline_block_equity[block_id].append(
                        float(receipt["ending_equity_jpy"])
                    )
                    baseline_block_margin[block_id].append(
                        float(receipt["reserved_margin_jpy"])
                        / max(float(receipt["ending_equity_jpy"]), 1e-9)
                    )
                    baseline_block_balance[block_id].append(
                        float(receipt["ending_balance_jpy"])
                    )
            elif event_type == "TERMINAL_SUCCESS":
                terminal_seen = True
            elif event_type == "TERMINAL_FAILURE":
                raise DojoPairedInventoryCounterfactualError(
                    "failed transcript cannot produce a paired result"
                )
            prior_record_sha = row["record_sha256"]
            record_index += 1
    if header is None or not terminal_seen or thresholds is None:
        raise DojoPairedInventoryCounterfactualError(
            "transcript is incomplete or has no OOS boundary"
        )
    actual_file_sha = file_digest.hexdigest()
    expected = expected_file_sha.get(header["coordinate_id"])
    if expected != actual_file_sha:
        raise DojoPairedInventoryCounterfactualError(
            "transcript file digest differs from the sealed plan"
        )
    baseline_net = float(baseline_result["end_equity_jpy"]) - float(
        baseline_result["start_equity_jpy"]
    )
    cadence_rows = []
    for cadence, arm in arms.items():
        result = arm.session.finalize(terminal_policy=MONTH_END_FLAT_SETTLEMENT)
        if arm.decisions and arm.decisions[-1]["post_outcome"] is None:
            arm.decisions[-1]["post_outcome"] = {
                "observed_through_epoch": result["end_epoch"],
                "equity_delta_jpy": (
                    float(result["end_equity_jpy"])
                    - float(arm.decisions[-1]["pre_action_equity_jpy"])
                ),
                "balance_delta_jpy": (
                    float(result["end_balance_jpy"])
                    - float(arm.decisions[-1]["pre_action_balance_jpy"])
                ),
            }
        block_rows = []
        for block in blocks:
            block_id = block["block_id"]
            base_equity = baseline_block_equity[block_id]
            ai_equity = arm.block_equity.get(block_id, [])
            base_balances = baseline_block_balance[block_id]
            if not base_equity or not ai_equity or not base_balances:
                block_rows.append(
                    {"block_id": block_id, "status": "MISSING_FIXED_DENOMINATOR"}
                )
                continue
            base_peak = base_equity[0]
            base_dd = 0.0
            for value in base_equity:
                base_peak = max(base_peak, value)
                base_dd = max(base_dd, (base_peak - value) / max(base_peak, 1e-9))
            ai_peak = ai_equity[0]
            ai_dd = 0.0
            for value in ai_equity:
                ai_peak = max(ai_peak, value)
                ai_dd = max(ai_dd, (ai_peak - value) / max(ai_peak, 1e-9))
            base_net = base_balances[-1] - base_balances[0]
            ai_net = (
                arm.block_end_balance[block_id]
                - arm.block_start_balance[block_id]
            )
            block_rows.append(
                {
                    "block_id": block_id,
                    "status": "MEASURED_EXPERIMENTAL",
                    "bot_only_net_jpy": base_net,
                    "ai_managed_net_jpy": ai_net,
                    "net_delta_jpy": ai_net - base_net,
                    "bot_only_max_drawdown_fraction": base_dd,
                    "ai_managed_max_drawdown_fraction": ai_dd,
                    "max_drawdown_delta": ai_dd - base_dd,
                    "bot_only_peak_margin_usage_fraction": max(
                        baseline_block_margin[block_id]
                    ),
                    "ai_managed_peak_margin_usage_fraction": max(
                        arm.block_margin.get(block_id, [0.0])
                    ),
                }
            )
        ai_net = float(result["end_equity_jpy"]) - float(result["start_equity_jpy"])
        oos_close_pnls = [
            float(row["pnl_jpy"])
            for row in arm.close_events
            if int(row["epoch"]) >= int(calibration["end_epoch"])
        ]
        cadence_rows.append(
            {
                "cadence_id": cadence,
                "portfolio_result": result,
                "bot_only_full_month_net_jpy": baseline_net,
                "ai_managed_full_month_net_jpy": ai_net,
                "full_month_net_delta_jpy": ai_net - baseline_net,
                "ai_close_metrics": _profit_metrics(oos_close_pnls),
                "ai_call_count": len(arm.decisions),
                "provider_model_call_count": 0,
                "state_packet_cache_skip_count": arm.skipped_cached_calls,
                "intervention_count": sum(
                    row["action"] != "HOLD" for row in arm.decisions
                ),
                "intervention_audit_log": arm.decisions,
                "close_reason_counts": dict(sorted(arm.close_reason_counts.items())),
                "oos_block_rows": block_rows,
                "phase_b_actual_model_checkpoint_decisions_measured": False,
            }
        )
    body = {
        "contract": RESULT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "coordinate_id": header["coordinate_id"],
        "family_id": family_id,
        "cost_scenario": cost_scenario,
        "source_transcript_sha256": actual_file_sha,
        "source_policy_sha256": header["policy_sha256"],
        "source_initial_balance_jpy": header["initial_balance_jpy"],
        "source_quote_coverage_proved": plan["source_quote_coverage_proved"],
        "calibrated_thresholds": thresholds,
        "cadence_rows": cadence_rows,
        "classification": "EXPERIMENTAL_UNRANKED",
        "blockers": [
            "SOURCE_QUOTE_COVERAGE_NOT_PROVED",
            "WORN_TRAIN_RESEARCHER_PRIOR_AGGREGATE_OUTCOME_EXPOSURE",
            "ACTUAL_MODEL_CHECKPOINT_CALLS_NOT_EXECUTED",
            "BOT_ONLY_TRADE_LEVEL_PROFIT_FACTOR_NOT_IN_IMMUTABLE_EVIDENCE",
        ],
        "authority": dict(AUTHORITY),
    }
    return {**body, "result_sha256": canonical_portfolio_sha256(body)}


__all__ = [
    "ACTION_IDS",
    "CADENCE_IDS",
    "DojoPairedInventoryCounterfactualError",
    "PLAN_CONTRACT",
    "RESULT_CONTRACT",
    "build_paired_inventory_plan",
    "replay_paired_inventory_transcript",
]
