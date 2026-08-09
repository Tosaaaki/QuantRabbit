"""Causal price-action context and paired inventory ablation primitives.

The module is deliberately read-only.  It neither chooses nor places an
order.  Its first job is to separate one/two-candle shape from multi-bar price
structure without future leakage.  Its second job is to compare already
realised, after-cost shadow arms on the exact same event identities.

The output can reject the claim that multi-bar price action adds information
over inventory-only and candle-shape controls.  It cannot establish an
"always profitable" claim, grant Paper/live permission, or authorise an AI
trader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Mapping, Sequence

from quant_rabbit.loss_close_hedge_paired_shadow import _freeze_full_s5
from quant_rabbit.loss_close_paired_shadow import (
    S5BidAskCandle,
    _iso_utc,
    _parse_canonical_utc_parts,
    validate_paired_shadow_state,
)


PRICE_ACTION_CONTEXT_CONTRACT = "loss_close_price_action_context_v1"
PRICE_ACTION_ABLATION_CONTRACT = "loss_close_price_action_ablation_v1"
INVENTORY_ARM = "INVENTORY_ONLY"
CANDLE_ARM = "CANDLE_1_2"
PRICE_ACTION_ARM = "PRICE_ACTION_MULTI_BAR"
_ARMS = (INVENTORY_ARM, CANDLE_ARM, PRICE_ACTION_ARM)
_ALLOWED_SPLITS = ("TRAIN", "VALIDATION")


@dataclass(frozen=True)
class PriceActionFeatureSpec:
    """Precommitted multi-bar feature geometry.

    ``frames_seconds`` deliberately contains M1 and M5 by default.  The
    windows are bar counts inside each frame, not thresholds fitted on the
    outcome.  Sweeps should change this geometry on TRAIN, retain plateaus on
    VALIDATION, and leave any holdout untouched.
    """

    frames_seconds: tuple[int, ...] = (60, 300)
    structure_bars: int = 12
    regime_bars: int = 24
    breakout_bars: int = 8
    acceptance_bars: int = 2
    attack_tolerance_ratio: float = 0.08


@dataclass(frozen=True)
class PairedAblationSpec:
    """Acceptance floor for a pre-holdout paired ablation."""

    min_events_per_split: int = 30
    min_increment_jpy: float = 0.0


@dataclass(frozen=True)
class _Bar:
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    mean_spread: float


def build_price_action_context(
    paired_state: object,
    candles: Sequence[S5BidAskCandle],
    *,
    spec: PriceActionFeatureSpec = PriceActionFeatureSpec(),
    holdout_used: bool = False,
) -> dict[str, Any]:
    """Build candle and multi-bar context using only completed past bars."""

    blockers: list[str] = []
    if holdout_used is not False:
        blockers.append("HOLDOUT_USE_FORBIDDEN")
    if spec.__class__ is not PriceActionFeatureSpec:
        blockers.append("INVALID_FEATURE_SPEC")
    else:
        blockers.extend(_feature_spec_issues(spec))
    if not isinstance(paired_state, Mapping):
        blockers.append("PAIRED_STATE_NOT_MAPPING")
        return _context_result(blockers=blockers)
    try:
        state = dict(paired_state)
    except Exception:
        return _context_result(blockers=blockers + ["PAIRED_STATE_SNAPSHOT_UNREADABLE"])
    validation = validate_paired_shadow_state(state)
    if not validation["valid"]:
        blockers.extend(f"INVALID_PAIRED_STATE:{x}" for x in validation["issues"])
        return _context_result(
            blockers=blockers, state_sha256=validation.get("state_sha256")
        )
    if blockers:
        return _context_result(blockers=blockers, state_sha256=state["state_sha256"])

    frozen, candle_issues = _freeze_full_s5(candles, pair=str(state["pair"]))
    # OANDA can omit a five-second candle when there was no price update.
    # That gap is fatal for intra-candle fill ordering, but this function does
    # not score fills.  Keep it visible as a warning and aggregate only time
    # buckets that have ended before the decision.  Every other S5 issue still
    # fails closed.
    gap_warnings = [issue for issue in candle_issues if issue.startswith("S5_TRUTH_GAP:")]
    blockers.extend(issue for issue in candle_issues if not issue.startswith("S5_TRUTH_GAP:"))
    decision_parts = _parse_canonical_utc_parts(state["decision_timestamp_utc"])
    assert decision_parts is not None
    decision_time = decision_parts[0].astimezone(timezone.utc)
    past = tuple(candle for candle in frozen if candle.timestamp_utc < decision_time)
    if not past:
        blockers.append("NO_COMPLETE_S5_HISTORY_BEFORE_DECISION")
    if blockers:
        return _context_result(blockers=blockers, state_sha256=state["state_sha256"])

    by_frame: dict[str, Any] = {}
    for seconds in spec.frames_seconds:
        bars = _aggregate_complete_bars(past, seconds=seconds, decision_time=decision_time)
        if len(bars) < spec.regime_bars + spec.acceptance_bars + 1:
            blockers.append(f"INSUFFICIENT_COMPLETED_BARS:{seconds}")
            continue
        by_frame[_frame_label(seconds)] = {
            "last_completed_bar_end_utc": _iso_utc(bars[-1].end),
            "completed_bar_count": len(bars),
            "candle_1_2": _candle_features(bars),
            "price_action_multi_bar": _multi_bar_features(bars, spec),
        }
    if blockers:
        return _context_result(blockers=blockers, state_sha256=state["state_sha256"])

    side_sign = 1 if state["side"] == "LONG" else -1
    pa_dirs = [
        int(frame["price_action_multi_bar"]["direction"])
        for frame in by_frame.values()
    ]
    candle_dirs = [int(frame["candle_1_2"]["last_direction"]) for frame in by_frame.values()]
    pa_consensus = _sign(sum(pa_dirs)) if len(set(pa_dirs)) == 1 else 0
    candle_consensus = _sign(sum(candle_dirs)) if len(set(candle_dirs)) == 1 else 0
    pattern_candidates = {
        label: frame["price_action_multi_bar"]["chart_pattern_candidate"]
        for label, frame in by_frame.items()
    }
    return _context_result(
        blockers=[],
        state_sha256=state["state_sha256"],
        status="CONTEXT_CALCULATED_OUTCOME_NOT_EVALUATED",
        payload={
            "decision_timestamp_utc": state["decision_timestamp_utc"],
            "pair": state["pair"],
            "inventory_side": state["side"],
            "uses_only_candles_strictly_before_decision": True,
            "s5_gap_warning_count": len(gap_warnings),
            "s5_gap_warnings": gap_warnings,
            "s5_gaps_allowed_for_feature_context_not_fill_order": True,
            "frames": by_frame,
            "cross_frame": {
                "price_action_consensus_direction": pa_consensus,
                "price_action_against_inventory": pa_consensus == -side_sign,
                "candle_consensus_direction": candle_consensus,
                "candle_against_inventory": candle_consensus == -side_sign,
                "price_action_persistent_across_frames": pa_consensus != 0,
                "chart_pattern_candidates": pattern_candidates,
                "setup_gate": (
                    "EVALUATE_PAIRED_SHADOW_ONLY"
                    if pa_consensus != 0 and any(x != "NONE" for x in pattern_candidates.values())
                    else "SKIP_NO_PRECOMMITTED_MULTI_BAR_SETUP"
                ),
            },
            "hypothesis": (
                "PRICE_ACTION_MULTI_BAR_ADDS_AFTER_COST_INVENTORY_AND_UNWIND_"
                "INFORMATION_BEYOND_INVENTORY_ONLY_AND_CANDLE_1_2"
            ),
            "hypothesis_proven": False,
        },
    )


def evaluate_paired_price_action_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: PairedAblationSpec = PairedAblationSpec(),
    holdout_used: bool = False,
) -> dict[str, Any]:
    """Compare identical events for inventory, candle, and price-action arms.

    Each row must contain ``event_uid``, ``split``, ``cost_model_sha256`` and
    an exact three-arm mapping.  Each arm supplies after-cost ``net_jpy`` plus
    risk/unwind fields.  TEST/HOLDOUT rows are rejected: this evaluator is a
    TRAIN/VALIDATION gate only.
    """

    blockers: list[str] = []
    if holdout_used is not False:
        blockers.append("HOLDOUT_USE_FORBIDDEN")
    if spec.__class__ is not PairedAblationSpec:
        blockers.append("INVALID_ABLATION_SPEC")
    else:
        if spec.min_events_per_split.__class__ is not int or spec.min_events_per_split < 1:
            blockers.append("INVALID_MIN_EVENTS_PER_SPLIT")
        if not _exact_finite_float(spec.min_increment_jpy) or spec.min_increment_jpy < 0.0:
            blockers.append("INVALID_MIN_INCREMENT_JPY")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        blockers.append("ROWS_NOT_SEQUENCE")
        return _ablation_result(blockers=blockers)
    try:
        frozen = tuple(dict(row) for row in rows)
    except Exception:
        return _ablation_result(blockers=blockers + ["ROWS_SNAPSHOT_UNREADABLE"])
    if not frozen:
        blockers.append("ROWS_EMPTY")

    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(frozen):
        uid = row.get("event_uid")
        split = row.get("split")
        digest = row.get("cost_model_sha256")
        arms = row.get("arms")
        if not isinstance(uid, str) or not uid or uid in seen:
            blockers.append(f"INVALID_OR_DUPLICATE_EVENT_UID:{index}")
        else:
            seen.add(uid)
        if split not in _ALLOWED_SPLITS:
            blockers.append(f"FORBIDDEN_OR_INVALID_SPLIT:{index}")
        if not isinstance(digest, str) or len(digest) != 64:
            blockers.append(f"INVALID_COST_MODEL_SHA256:{index}")
        if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
            blockers.append(f"ARMS_NOT_EXACT_PAIRED_ABLATION:{index}")
            continue
        clean_arms: dict[str, dict[str, Any]] = {}
        for arm_name in _ARMS:
            arm = arms.get(arm_name)
            if not isinstance(arm, Mapping):
                blockers.append(f"INVALID_ARM:{index}:{arm_name}")
                continue
            arm = dict(arm)
            for key in ("net_jpy", "max_drawdown_jpy"):
                if not _exact_finite_float(arm.get(key)):
                    blockers.append(f"INVALID_ARM_NUMBER:{index}:{arm_name}:{key}")
            for key in (
                "ruin_floor_breached",
                "margin_closeout_proxy_breached",
                "unwind_complete",
                "fill_order_resolved",
            ):
                if arm.get(key).__class__ is not bool:
                    blockers.append(f"INVALID_ARM_BOOLEAN:{index}:{arm_name}:{key}")
            clean_arms[arm_name] = arm
        if len(clean_arms) == len(_ARMS):
            cleaned.append({"event_uid": uid, "split": split, "arms": clean_arms})
    if blockers:
        return _ablation_result(blockers=blockers)

    summaries: dict[str, Any] = {}
    split_passes: list[bool] = []
    for split in _ALLOWED_SPLITS:
        selected = [row for row in cleaned if row["split"] == split]
        if len(selected) < spec.min_events_per_split:
            blockers.append(f"INSUFFICIENT_EVENTS:{split}")
            continue
        arm_summary = {arm: _summarise_arm(selected, arm) for arm in _ARMS}
        pa = arm_summary[PRICE_ACTION_ARM]
        inv = arm_summary[INVENTORY_ARM]
        candle = arm_summary[CANDLE_ARM]
        increment_inv = pa["mean_net_jpy"] - inv["mean_net_jpy"]
        increment_candle = pa["mean_net_jpy"] - candle["mean_net_jpy"]
        risk_ok = (
            pa["max_drawdown_jpy"] <= inv["max_drawdown_jpy"]
            and pa["max_drawdown_jpy"] <= candle["max_drawdown_jpy"]
            and pa["ruin_floor_breach_count"] <= inv["ruin_floor_breach_count"]
            and pa["ruin_floor_breach_count"] <= candle["ruin_floor_breach_count"]
            and pa["margin_closeout_breach_count"] <= inv["margin_closeout_breach_count"]
            and pa["margin_closeout_breach_count"] <= candle["margin_closeout_breach_count"]
            and pa["incomplete_unwind_count"] == 0
            and pa["unresolved_fill_order_count"] == 0
        )
        passed = (
            increment_inv > spec.min_increment_jpy
            and increment_candle > spec.min_increment_jpy
            and risk_ok
        )
        split_passes.append(passed)
        summaries[split] = {
            "event_count": len(selected),
            "arms": arm_summary,
            "price_action_increment_vs_inventory_jpy": increment_inv,
            "price_action_increment_vs_candle_jpy": increment_candle,
            "risk_and_unwind_not_worse": risk_ok,
            "passes_precommitted_increment_gate": passed,
        }
    if blockers:
        return _ablation_result(blockers=blockers)

    return _ablation_result(
        blockers=[],
        status="PRE_HOLDOUT_ABLATION_CALCULATED",
        payload={
            "splits": summaries,
            "hypothesis_survives_pre_holdout": all(split_passes),
            "holdout_unlock_allowed": False,
            "ai_supervisor_evaluation_allowed": all(split_passes),
            "decision_rule": (
                "PRICE_ACTION_MUST_BE_STRICTLY_BETTER_THAN_BOTH_CONTROLS_ON_"
                "TRAIN_AND_VALIDATION_WITHOUT_WORSE_DD_RUIN_MARGIN_FILL_OR_UNWIND"
            ),
        },
    )


def _feature_spec_issues(spec: PriceActionFeatureSpec) -> list[str]:
    issues: list[str] = []
    if (
        spec.frames_seconds.__class__ is not tuple
        or not spec.frames_seconds
        or any(x.__class__ is not int or x < 60 or x % 5 for x in spec.frames_seconds)
        or len(set(spec.frames_seconds)) != len(spec.frames_seconds)
    ):
        issues.append("INVALID_FRAMES_SECONDS")
    for name in ("structure_bars", "regime_bars", "breakout_bars", "acceptance_bars"):
        value = getattr(spec, name)
        if value.__class__ is not int or value < 2:
            issues.append(f"INVALID_WINDOW:{name}")
    if (
        spec.regime_bars.__class__ is int
        and spec.structure_bars.__class__ is int
        and spec.regime_bars < spec.structure_bars
    ):
        issues.append("REGIME_WINDOW_SHORTER_THAN_STRUCTURE_WINDOW")
    if spec.structure_bars.__class__ is int and spec.structure_bars < 4:
        issues.append("STRUCTURE_WINDOW_REQUIRES_AT_LEAST_FOUR_BARS")
    if all(
        getattr(spec, name).__class__ is int
        for name in ("regime_bars", "breakout_bars", "acceptance_bars")
    ) and spec.regime_bars < spec.breakout_bars + spec.acceptance_bars:
        issues.append("REGIME_WINDOW_TOO_SHORT_FOR_BREAKOUT_AND_ACCEPTANCE")
    if not _exact_finite_float(spec.attack_tolerance_ratio) or not 0.0 < spec.attack_tolerance_ratio <= 0.5:
        issues.append("INVALID_ATTACK_TOLERANCE_RATIO")
    return issues


def _aggregate_complete_bars(
    candles: Sequence[S5BidAskCandle], *, seconds: int, decision_time: datetime
) -> tuple[_Bar, ...]:
    buckets: dict[datetime, list[S5BidAskCandle]] = {}
    for candle in candles:
        epoch = int(candle.timestamp_utc.timestamp())
        start = datetime.fromtimestamp(epoch - epoch % seconds, tz=timezone.utc)
        end = start + timedelta(seconds=seconds)
        if end <= decision_time:
            buckets.setdefault(start, []).append(candle)
    out: list[_Bar] = []
    for start in sorted(buckets):
        values = buckets[start]
        mids_o = [(x.bid.open + x.ask.open) / 2.0 for x in values]
        mids_h = [(x.bid.high + x.ask.high) / 2.0 for x in values]
        mids_l = [(x.bid.low + x.ask.low) / 2.0 for x in values]
        mids_c = [(x.bid.close + x.ask.close) / 2.0 for x in values]
        spreads = [x.ask.close - x.bid.close for x in values]
        out.append(
            _Bar(
                start=start,
                end=start + timedelta(seconds=seconds),
                open=mids_o[0],
                high=max(mids_h),
                low=min(mids_l),
                close=mids_c[-1],
                mean_spread=sum(spreads) / len(spreads),
            )
        )
    return tuple(out)


def _candle_features(bars: Sequence[_Bar]) -> dict[str, Any]:
    last, previous = bars[-1], bars[-2]
    span = max(last.high - last.low, 1e-12)
    body = last.close - last.open
    previous_body = previous.close - previous.open
    bullish_engulfing = (
        body > 0.0
        and previous_body < 0.0
        and last.open <= previous.close
        and last.close >= previous.open
    )
    bearish_engulfing = (
        body < 0.0
        and previous_body > 0.0
        and last.open >= previous.close
        and last.close <= previous.open
    )
    return {
        "last_direction": _sign(body),
        "body_to_range": abs(body) / span,
        "upper_wick_to_range": (last.high - max(last.open, last.close)) / span,
        "lower_wick_to_range": (min(last.open, last.close) - last.low) / span,
        "engulfing_direction": 1 if bullish_engulfing else -1 if bearish_engulfing else 0,
        "bars_used": 2,
    }


def _multi_bar_features(bars: Sequence[_Bar], spec: PriceActionFeatureSpec) -> dict[str, Any]:
    recent = list(bars[-spec.regime_bars :])
    structure = recent[-spec.structure_bars :]
    half = max(2, len(structure) // 2)
    older, newer = structure[:-half], structure[-half:]
    older_high, newer_high = max(x.high for x in older), max(x.high for x in newer)
    older_low, newer_low = min(x.low for x in older), min(x.low for x in newer)
    if newer_high > older_high and newer_low > older_low:
        pattern, direction = "HH_HL", 1
    elif newer_high < older_high and newer_low < older_low:
        pattern, direction = "LH_LL", -1
    elif newer_high > older_high and newer_low < older_low:
        pattern, direction = "EXPANDING", _sign(structure[-1].close - structure[0].open)
    elif newer_high < older_high and newer_low > older_low:
        pattern, direction = "CONTRACTING", 0
    else:
        pattern, direction = "MIXED", _sign(structure[-1].close - structure[0].open)

    ranges = [x.high - x.low for x in recent]
    path = sum(abs(b.close - a.close) for a, b in zip(structure, structure[1:]))
    displacement = structure[-1].close - structure[0].open
    rail_source = recent[-(spec.breakout_bars + spec.acceptance_bars) : -spec.acceptance_bars]
    high_rail = max(x.high for x in rail_source)
    low_rail = min(x.low for x in rail_source)
    accepted = recent[-spec.acceptance_bars :]
    accepted_up = all(x.close > high_rail for x in accepted)
    accepted_down = all(x.close < low_rail for x in accepted)
    last = recent[-1]
    failed_up = last.high > high_rail and last.close <= high_rail
    failed_down = last.low < low_rail and last.close >= low_rail
    broke_up_earlier = any(x.close > high_rail for x in recent[-4:-1])
    broke_down_earlier = any(x.close < low_rail for x in recent[-4:-1])
    span = max(max(x.high for x in structure) - min(x.low for x in structure), 1e-12)
    tolerance = span * spec.attack_tolerance_ratio
    high_attacks = sum(abs(x.high - high_rail) <= tolerance for x in structure)
    low_attacks = sum(abs(x.low - low_rail) <= tolerance for x in structure)
    short_ranges = ranges[-max(3, spec.structure_bars // 3) :]
    long_ranges = ranges[: max(1, len(ranges) - len(short_ranges))]
    compression = (sum(short_ranges) / len(short_ranges)) / max(
        sum(long_ranges) / len(long_ranges), 1e-12
    )
    flat_high = abs(newer_high - older_high) <= tolerance
    flat_low = abs(newer_low - older_low) <= tolerance
    midpoint = low_rail + (high_rail - low_rail) / 2.0
    if flat_high and newer_low > older_low and compression <= 1.15:
        chart_pattern = "ASCENDING_TRIANGLE_CANDIDATE"
    elif flat_low and newer_high < older_high and compression <= 1.15:
        chart_pattern = "DESCENDING_TRIANGLE_CANDIDATE"
    elif low_attacks >= 2 and newer_low >= older_low - tolerance and last.close > midpoint:
        chart_pattern = "DOUBLE_BOTTOM_CANDIDATE"
    elif high_attacks >= 2 and newer_high <= older_high + tolerance and last.close < midpoint:
        chart_pattern = "DOUBLE_TOP_CANDIDATE"
    else:
        chart_pattern = "NONE"
    return {
        "pattern": pattern,
        "direction": direction,
        "directional_efficiency": abs(displacement) / max(path, 1e-12),
        "range_position": (last.close - min(x.low for x in structure)) / span,
        "range_compression_ratio": compression,
        "breakout_acceptance_direction": 1 if accepted_up else -1 if accepted_down else 0,
        "failed_break_direction": -1 if failed_up else 1 if failed_down else 0,
        "retest_hold_direction": (
            1
            if broke_up_earlier and last.low <= high_rail + tolerance and last.close > high_rail
            else -1
            if broke_down_earlier and last.high >= low_rail - tolerance and last.close < low_rail
            else 0
        ),
        "high_rail": high_rail,
        "low_rail": low_rail,
        "high_attack_count": high_attacks,
        "low_attack_count": low_attacks,
        "chart_pattern_candidate": chart_pattern,
        "raw_break_without_close_acceptance": (
            (last.high > high_rail or last.low < low_rail)
            and not accepted_up
            and not accepted_down
        ),
        "close_confirmation_required": True,
        "mean_executable_spread": sum(x.mean_spread for x in structure) / len(structure),
        "bars_used": len(recent),
    }


def _summarise_arm(rows: Sequence[Mapping[str, Any]], arm_name: str) -> dict[str, Any]:
    arms = [row["arms"][arm_name] for row in rows]
    return {
        "mean_net_jpy": sum(float(arm["net_jpy"]) for arm in arms) / len(arms),
        "total_net_jpy": sum(float(arm["net_jpy"]) for arm in arms),
        "max_drawdown_jpy": max(float(arm["max_drawdown_jpy"]) for arm in arms),
        "ruin_floor_breach_count": sum(bool(arm["ruin_floor_breached"]) for arm in arms),
        "margin_closeout_breach_count": sum(
            bool(arm["margin_closeout_proxy_breached"]) for arm in arms
        ),
        "incomplete_unwind_count": sum(not bool(arm["unwind_complete"]) for arm in arms),
        "unresolved_fill_order_count": sum(
            not bool(arm["fill_order_resolved"]) for arm in arms
        ),
    }


def _context_result(
    *,
    blockers: Sequence[str],
    state_sha256: str | None = None,
    status: str = "BLOCKED",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _safety_envelope(PRICE_ACTION_CONTEXT_CONTRACT, blockers, status)
    result["state_sha256"] = state_sha256
    if payload:
        result.update(payload)
    return result


def _ablation_result(
    *, blockers: Sequence[str], status: str = "BLOCKED", payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result = _safety_envelope(PRICE_ACTION_ABLATION_CONTRACT, blockers, status)
    if payload:
        result.update(payload)
    return result


def _safety_envelope(contract: str, blockers: Sequence[str], status: str) -> dict[str, Any]:
    return {
        "contract": contract,
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "read_only": True,
        "paper_permission_allowed": False,
        "live_permission_allowed": False,
        "broker_order_allowed": False,
        "deployment_allowed": False,
        "proof_eligible": False,
        "always_profit_claim_allowed": False,
        "statistical_claim_allowed": False,
        "holdout_used": False,
    }


def _frame_label(seconds: int) -> str:
    return f"M{seconds // 60}" if seconds % 60 == 0 else f"S{seconds}"


def _sign(value: float | int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _exact_finite_float(value: object) -> bool:
    return value.__class__ is float and math.isfinite(value)
