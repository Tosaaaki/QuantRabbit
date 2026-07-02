from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _trader_no_broker_tp_runner(position) -> bool:
    """Return true when a trader-owned TP-less position is an intentional
    SL-free runner rather than a protection blocker.

    In the SL-free runtime, missing broker TP is preserved unless explicit TP
    repair is enabled. Risk still uses margin and portfolio checks for new
    entries; this helper only prevents the TP-less runner from freezing every
    future lane.
    """
    return (
        position.owner == Owner.TRADER
        and position.take_profit is None
        and _trader_sl_repair_disabled()
        and not _missing_tp_repair_enabled()
    )


def _layerable_trader_position(position) -> bool:
    if position.owner != Owner.TRADER:
        return False
    sl_ok = position.stop_loss is not None or _trader_sl_repair_disabled()
    tp_ok = position.take_profit is not None or _trader_no_broker_tp_runner(position)
    return sl_ok and tp_ok

from .models import (
    AccountSummary,
    BrokerOrder,
    BrokerPosition,
    BrokerSnapshot,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    RiskDecision,
    RiskIssue,
    RiskMetrics,
    Side,
    TradeMethod,
)
from .forecast_precision import (
    bidask_replay_negative_precision_issue,
    bidask_replay_precision_support,
    hit_rate_wilson_lower,
    support_signal_clears_live_precision,
    technical_harvest_negative_precision_issue,
    technical_harvest_precision_support,
)
from .instruments import DEFAULT_TRADER_PAIRS, NORMAL_SPREAD_PIPS, instrument_pip_factor
from .guardian_receipt_consumption import guardian_receipt_new_entry_blockers_from_paths
from .operator_manual import (
    is_operator_managed_manual_owner,
    operator_manual_jpy_add_block_issue,
    operator_manual_same_theme_add_block_issue,
)

# OANDA Japan retail FX margin in the current account is 25:1 leverage, i.e.
# 4% initial margin. Recent broker truth confirms the same scale: USD_JPY
# 25,000u filled near 155.962 required roughly 155,954 JPY initial margin.
# This is broker/account policy, not market data; replace it with per-instrument
# `/accounts/{id}/instruments` marginRate once that adapter is wired.
OANDA_JP_RETAIL_FX_MARGIN_RATE = 0.04

# Minimum order size (units) the production trader will emit or accept.
#
# What this represents: the lot size below which expected pip-target reward is
# dominated by the OANDA spread cost on the round trip. At 1u in a JPY-quoted
# pair the JPY-per-pip is 0.01 JPY; a 1.3-pip normal spread already costs more
# than any realistic pip target captured at micro size, so the trade is
# guaranteed to lose money once spread is paid. The 1000u floor matches the
# existing rounding granularity used in `_risk_budgeted_units` (which floors
# >=1000-units results to the nearest 1000) and the broker-supplied 1000u
# default trade granularity for FX retail accounts.
#
# Why it is a constant rather than market-derived: spread × micro-lot
# economics is a broker-policy reality, not a session-by-session market
# condition. The floor moves only when the broker offers a fundamentally
# different minimum trade size; intra-day liquidity does not change it.
# Per AGENT_CONTRACT §3.5, this constant carries its (a)/(b)/(c) docstring
# right here.
#
# What should replace it: if the broker contract changes (e.g. tighter
# spreads + true micro-lot pricing where 100u becomes economic), revisit
# this floor — do not bypass it in the moment.
MIN_PRODUCTION_LOT_UNITS = 1000

# Hedge timing metadata is an execution contract, not prompt-only prose.
#
# What this represents: every same-pair HEDGE intent must declare why the
# opposite-side leg exists and when it gets reviewed/unwound, otherwise a
# replayed or hand-written receipt can bypass the time-boxing discipline and
# become passive loss-freezing.
#
# Why it is constant rather than derived: these are receipt contract labels.
# The market-derived part is the generator's choice of class and size.
# Replace only when docs/AGENT_CONTRACT.md changes the class taxonomy.
HEDGE_TIMING_CLASSES = {"LOCK_GAIN", "REVERSAL", "CONTINUATION", "OPPOSITE_EXPOSURE"}

# Mirrors intent_generator.RECOVERY_HEDGE_CONTINUATION_MAX_SCALE. Duplicated
# here intentionally so RiskEngine can defend manual stage-live-order and
# replayed receipts without importing strategy code.
HEDGE_CONTINUATION_MAX_SCALE = 0.35

# Exit-reason payoff relaxation needs enough realized broker-TP outcomes to
# distinguish a real capture shape from one lucky burst. This mirrors the
# strategy generator evidence floor so RiskEngine can defend hand-written or
# replayed receipts without importing strategy code.
LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES = 20
LOSS_ASYMMETRY_TP_PROOF_COLLECTION_MIN_EXIT_TRADES = 5
LOSS_ASYMMETRY_TP_PROOF_COLLECTION_MIN_LOT_MODE = "TP_PROOF_COLLECTION_MIN_LOT"
LOSS_ASYMMETRY_OANDA_CAMPAIGN_FIREPOWER_MIN_LOT_MODE = "OANDA_CAMPAIGN_FIREPOWER_MIN_LOT"
LOSS_ASYMMETRY_OANDA_CAMPAIGN_FIREPOWER_RELAXED_MODE = "OANDA_CAMPAIGN_FIREPOWER_RELAXED"
POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE = "OANDA_CAMPAIGN_FIREPOWER_HARVEST"
SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_MODE = "TP_HARVEST_REPAIR"
OANDA_CAMPAIGN_FIREPOWER_TARGET_OK_STATUSES = {
    "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
    "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
}
# Mirrors intent_generator's OANDA vehicle RR tolerance. This is a receipt
# consistency bound, not a market-risk setting: RiskEngine rechecks replayed
# receipts against the same audited vehicle geometry that generated them.
OANDA_CAMPAIGN_EXIT_SHAPE_RR_REL_TOLERANCE = 0.10
OANDA_CAMPAIGN_EXIT_SHAPE_RR_ABS_TOLERANCE = 0.05
CAPTURE_ECONOMICS_STALE_BLOCK_CODE = "CAPTURE_ECONOMICS_STALE"
SPREAD_FLOOR_COMPARISON_EPSILON_PIPS = 1e-6


def _min_lot_test_override_active() -> bool:
    """Whether the production minimum-lot gate is disabled for the current
    process.

    Production behavior: `MIN_PRODUCTION_LOT_UNITS` is enforced by both
    `intent_generator._risk_budgeted_units` (sub-floor → 0 units → DRY_RUN_BLOCKED)
    and `RiskEngine.validate` (`MIN_LOT_VIOLATION` BLOCK). Some unit tests
    deliberately exercise sub-1000 unit fixtures (broker-API edge cases,
    legacy receipt replay). They opt out by setting `QR_ALLOW_TEST_MICRO_LOT=1`
    in the test's setUp.
    """
    return os.environ.get("QR_ALLOW_TEST_MICRO_LOT", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


# I (2026-05-13) — session-aware spread tolerance multipliers.
# Read once at module load from env so the operator can tune without
# code edit. Defaults mirror `intent_generator` constants — both must
# agree because the trader prompt narrates the tolerance band.
def _env_float_or(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < minimum:
        return minimum
    return value


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "none", "off", "false", "disabled"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def _env_optional_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "none", "off", "false", "disabled"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


# Forecast-market-support audit floors mirrored from
# strategy.intent_generator. They are validation-contract constants: a RANGE
# forecast may be tradable as rail rotation, but if the same current packet
# says an audited directional projection was unselected, the broker gateway
# must not send the lower-quality opposite entry just because another blocker
# happened to be absent.
FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE = _env_float_or(
    "QR_FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE",
    0.55,
    minimum=0.50,
)
FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE = _env_float_or(
    "QR_FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE",
    0.55,
    minimum=0.0,
)
FORECAST_MARKET_SUPPORT_MIN_SAMPLES = (
    _env_optional_int("QR_FORECAST_MARKET_SUPPORT_MIN_SAMPLES", None)
    or _env_optional_int("QR_PROJECTION_CONFIDENCE_MIN_SAMPLES", None)
    or 10
)
FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL = _env_float_or(
    "QR_FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL",
    0.10,
    minimum=0.0,
)
FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE = _env_float_or(
    "QR_FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE",
    0.65,
    minimum=0.50,
)
# RANGE rotation confidence mirrors intent_generator: 50% is the minimum
# probability that the box remains tradeable, and live send may go below it
# only when a current audited same-side projection supports a passive rail
# LIMIT. This is gateway defense-in-depth, not a separate tuning surface.
FORECAST_RANGE_ROTATION_MIN_CONFIDENCE = _env_float_or(
    "QR_FORECAST_RANGE_ROTATION_MIN_CONFIDENCE",
    0.50,
    minimum=0.50,
)
FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE = _env_float_or(
    "QR_FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE",
    0.45,
    minimum=0.0,
)
FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES = max(
    FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
    _env_optional_int("QR_FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES", None)
    or FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
)
# Adverse-path calibration is the final detector's realized path truth. A
# bucket that hits invalidation before target in a majority of calibrated
# samples is unsafe even when its headline direction confidence is high; mirror
# intent_generator/self_improvement_audit so gateway validation cannot be
# weaker than intent construction.
FORECAST_DIRECTIONAL_LIVE_MAX_INVALIDATION_FIRST_RATE = _env_float_or(
    "QR_FORECAST_DIRECTIONAL_LIVE_MAX_INVALIDATION_FIRST_RATE",
    0.60,
    minimum=0.0,
)
FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER = _env_float_or(
    "QR_FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER",
    0.90,
    minimum=0.50,
)
FORECAST_LIVE_PRECISION_MIN_SAMPLES = max(
    FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
    _env_optional_int("QR_FORECAST_LIVE_PRECISION_MIN_SAMPLES", None)
    or max(30, FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES),
)
FORECAST_LIVE_PRECISION_MIN_TARGET_PIPS = _env_float_or(
    "QR_FORECAST_LIVE_PRECISION_MIN_TARGET_PIPS",
    2.0,
    minimum=0.0,
)


_SPREAD_SESSION_MULTS: dict[str, float] = {
    # Deep-liquidity sessions: tighten the spread cap below policy.
    "LONDON_NY_OVERLAP": _env_float_or("QR_SESSION_SPREAD_MULT_LONDON_NY", 0.8, minimum=0.5),
    "LONDON_KILLZONE": _env_float_or("QR_SESSION_SPREAD_MULT_LONDON", 1.0, minimum=0.5),
    "LONDON_OPEN": _env_float_or("QR_SESSION_SPREAD_MULT_LONDON", 1.0, minimum=0.5),
    "LONDON": _env_float_or("QR_SESSION_SPREAD_MULT_LONDON", 1.0, minimum=0.5),
    "NY_AM_KILLZONE": _env_float_or("QR_SESSION_SPREAD_MULT_NY_AM", 1.0, minimum=0.5),
    "NY_OPEN": _env_float_or("QR_SESSION_SPREAD_MULT_NY_AM", 1.0, minimum=0.5),
    "NY": _env_float_or("QR_SESSION_SPREAD_MULT_NY_AM", 1.0, minimum=0.5),
    # Thin-liquidity sessions: loosen so we don't reject every Tokyo
    # entry as overspread. JP_HOLIDAY uses the OFF_HOURS widening.
    "TOKYO_KILLZONE": _env_float_or("QR_SESSION_SPREAD_MULT_TOKYO", 1.25, minimum=0.5),
    "TOKYO_OPEN": _env_float_or("QR_SESSION_SPREAD_MULT_TOKYO", 1.25, minimum=0.5),
    "TOKYO": _env_float_or("QR_SESSION_SPREAD_MULT_TOKYO", 1.25, minimum=0.5),
    "ASIA": _env_float_or("QR_SESSION_SPREAD_MULT_TOKYO", 1.25, minimum=0.5),
    "OFF_HOURS": _env_float_or("QR_SESSION_SPREAD_MULT_OFF_HOURS", 1.5, minimum=0.5),
    "JP_HOLIDAY": _env_float_or("QR_SESSION_SPREAD_MULT_OFF_HOURS", 1.5, minimum=0.5),
}


def _spread_session_multiplier_from_tag(tag_raw: object) -> float:
    """Return the spread-cap multiplier for a chart/session liquidity tag.

    The multipliers are the documented session-liquidity tiers from §3.5.
    Missing tags fall back to 1.0 so broker-spec spread limits still apply
    when an older artifact lacks session context.
    """
    if not tag_raw:
        return 1.0
    tag = str(tag_raw).upper().strip()
    return _SPREAD_SESSION_MULTS.get(tag, 1.0)


def _spread_session_multiplier(intent: OrderIntent) -> float:
    """Return the session-aware multiplier on top of
    `RiskPolicy.max_spread_multiple`. Reads from intent.metadata
    (producer: intent_generator._chart_context_for). Missing metadata
    falls back to 1.0 so the policy default still applies.
    """
    metadata = intent.metadata or {}
    tag_raw = metadata.get("session_current_tag") or metadata.get("session_bucket")
    return _spread_session_multiplier_from_tag(tag_raw)


def _below_spread_floor(distance_pips: float, floor_pips: float) -> bool:
    return distance_pips + SPREAD_FLOOR_COMPARISON_EPSILON_PIPS < floor_pips


def _uses_range_reward_floor(intent: OrderIntent, regime_state: str) -> bool:
    """Return true when the setup is executable as a range rotation.

    Multi-timeframe entries can be local range trades inside a higher-TF trend.
    In that case `intent.metadata['regime_state']` may carry the dominant
    higher-TF trend label, while the method and geometry are still rail/box
    rotation. Applying the default trend RR floor to those TF-local range
    entries hides valid scalps; use the range RR floor whenever method or
    geometry proves a range-rotation setup.
    """
    if "RANGE" in regime_state:
        return True
    context = intent.market_context
    if context is not None and context.method == TradeMethod.RANGE_ROTATION:
        return True
    metadata = intent.metadata or {}
    geometry_model = str(metadata.get("geometry_model") or "").upper()
    method_name = str(metadata.get("method") or "").upper()
    return "RANGE" in geometry_model or method_name == TradeMethod.RANGE_ROTATION.value


def _range_countertrend_low_rr_issue(
    intent: OrderIntent,
    metrics: RiskMetrics,
    policy: RiskPolicy,
) -> RiskIssue | None:
    if metrics.reward_risk >= policy.technical_harvest_min_reward_risk:
        return None
    context = intent.market_context
    method = context.method if context is not None else None
    metadata = intent.metadata or {}
    method_name = str(metadata.get("method") or "").upper()
    geometry_model = str(metadata.get("geometry_model") or "").upper()
    if (
        method != TradeMethod.RANGE_ROTATION
        and method_name != TradeMethod.RANGE_ROTATION.value
        and "RANGE" not in geometry_model
    ):
        return None
    evidence_text = _range_countertrend_evidence_text(intent)
    if intent.side == Side.LONG:
        adverse_lean = "SHORT_LEAN" in evidence_text or "SCORE_BALANCE=SHORT" in evidence_text
        adverse_regime = "TREND_DOWN" in evidence_text or "DOMINANT_REGIME=TREND_DOWN" in evidence_text
        adverse_side = "SHORT"
    else:
        adverse_lean = "LONG_LEAN" in evidence_text or "SCORE_BALANCE=LONG" in evidence_text
        adverse_regime = "TREND_UP" in evidence_text or "DOMINANT_REGIME=TREND_UP" in evidence_text
        adverse_side = "LONG"
    if not (adverse_lean and adverse_regime):
        return None
    oanda_tolerance_issue = _oanda_campaign_firepower_countertrend_rr_tolerance_issue(
        intent,
        metrics,
        policy,
    )
    if oanda_tolerance_issue is not None:
        return oanda_tolerance_issue
    return RiskIssue(
        "RANGE_COUNTERTREND_RR_TOO_LOW",
        f"{intent.pair} {intent.side.value} RANGE_ROTATION is counter to {adverse_side}-leaning "
        f"matrix/higher-timeframe evidence with reward/risk {metrics.reward_risk:.2f}x below "
        f"{policy.technical_harvest_min_reward_risk:.2f}x; wait for alignment or demand at least 1R.",
    )


def _range_countertrend_evidence_text(intent: OrderIntent) -> str:
    context = intent.market_context
    metadata = intent.metadata or {}
    parts: list[str] = []
    if context is not None:
        parts.extend((context.regime, context.chart_story, context.narrative))
    for key in (
        "strongest_matrix_reject",
        "strongest_matrix_warning",
        "strongest_matrix_support",
        "regime_state",
        "dominant_regime",
    ):
        value = metadata.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("matrix_warning_context", "matrix_reject_context", "matrix_support_context"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).upper()


def _oanda_campaign_firepower_countertrend_rr_tolerance_issue(
    intent: OrderIntent,
    metrics: RiskMetrics,
    policy: RiskPolicy,
) -> RiskIssue | None:
    """Downgrade one narrow OANDA HARVEST repair shape to WARN.

    A RANGE rail fade can sit just below 1R after broker precision and attached
    HARVEST TP anchoring, while still matching the audited OANDA campaign
    vehicle that generated the repair lane. This is not a generic low-RR escape:
    the receipt must prove non-market attached-TP HARVEST shape, range-box
    geometry, P0 repair mode, current vehicle match, and matching-vehicle
    positive active-day return. RiskEngine also recomputes the RR tolerance so
    stale or hand-written metadata cannot turn a 0.6R fade into a live send.
    """

    metadata = intent.metadata or {}
    if intent.order_type == OrderType.MARKET:
        return None
    if str(metadata.get("position_intent") or "NEW").upper() == "HEDGE":
        return None
    if metadata.get("attach_take_profit_on_fill") is not True:
        return None
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return None
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return None
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return None
    if str(metadata.get("positive_rotation_mode") or "") != POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE:
        return None
    if metadata.get("positive_rotation_live_ready") is not True:
        return None
    if metadata.get("positive_rotation_oanda_campaign_firepower_vehicle_match") is not True:
        return None
    if metadata.get("positive_rotation_oanda_campaign_minimum_floor_reachable") is not True:
        return None
    if metadata.get("self_improvement_p0_repair_live_ready") is not True:
        return None
    if (
        str(metadata.get("self_improvement_p0_repair_mode") or "")
        != SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_MODE
    ):
        return None
    if str(metadata.get("forecast_direction") or "").upper() != "RANGE":
        return None
    if metadata.get("range_tp_is_inside_box") is not True:
        return None
    if metadata.get("range_sl_outside_box") is not True:
        return None
    entry_side = str(metadata.get("range_entry_side") or "").upper()
    if intent.side == Side.LONG and entry_side != "SUPPORT":
        return None
    if intent.side == Side.SHORT and entry_side != "RESISTANCE":
        return None
    status = str(metadata.get("positive_rotation_oanda_campaign_firepower_status") or "").upper()
    if status not in OANDA_CAMPAIGN_FIREPOWER_TARGET_OK_STATUSES:
        return None
    matching_return = _to_float(
        metadata.get(
            "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day"
        )
    )
    if matching_return is None or matching_return <= 0.0:
        return None
    current_rr = _to_float(metadata.get("positive_rotation_oanda_campaign_current_reward_risk"))
    if current_rr is None or not math.isclose(current_rr, metrics.reward_risk, abs_tol=1e-6):
        return None
    expected_rr = _to_float(
        metadata.get("positive_rotation_oanda_campaign_matching_vehicle_expected_reward_risk")
    )
    if expected_rr is None or expected_rr < policy.technical_harvest_min_reward_risk:
        return None
    if metrics.reward_risk < policy.range_min_reward_risk:
        return None
    tolerance = max(
        OANDA_CAMPAIGN_EXIT_SHAPE_RR_ABS_TOLERANCE,
        abs(expected_rr) * OANDA_CAMPAIGN_EXIT_SHAPE_RR_REL_TOLERANCE,
    )
    if expected_rr - metrics.reward_risk > tolerance + 1e-9:
        return None
    return RiskIssue(
        "OANDA_CAMPAIGN_FIREPOWER_RANGE_COUNTERTREND_RR_TOLERANCE",
        f"{intent.pair} {intent.side.value} RANGE_ROTATION is countertrend, but the non-market "
        f"attached-TP HARVEST repair receipt matches the audited OANDA vehicle within RR tolerance "
        f"({metrics.reward_risk:.2f}x vs vehicle {expected_rr:.2f}x); keep all other live gates active.",
        severity="WARN",
    )


def _uses_technical_harvest_reward_floor(intent: OrderIntent) -> bool:
    """Return true for failed-break scalps with an attached structural TP.

    These entries are not trend runners. The intent generator can explicitly
    build them with a 1R operating HARVEST target when no better structural
    anchor exists; RiskEngine must not then reclassify the same receipt under
    the generic 1.2R runner floor. This exception is deliberately unavailable
    to TREND_CONTINUATION so weak forecast-first trend chases still need their
    higher payoff gate.
    """
    metadata = intent.metadata or {}
    context = intent.market_context
    method = context.method if context is not None else None
    if method is None:
        try:
            method = TradeMethod.parse(str(metadata.get("method") or ""))
        except ValueError:
            method = None
    if method != TradeMethod.BREAKOUT_FAILURE:
        return False
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    source = str(metadata.get("tp_target_source") or "").upper()
    if not source or "FORECAST" in source or "RUNNER" in source:
        return False
    return "HARVEST" in source


def _truthy_metadata(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


TARGET_PATH_GRADE_RANK = {
    "C": 0,
    "B-": 1,
    "B0": 2,
    "B": 2,
    "B+": 3,
    "A": 4,
    "S": 5,
}
TARGET_PATH_MAIN_ROLES = {"MAIN", "HERO", "PATH_A", "5PCT_PATH", "GUARANTEE_5", "PACE_5"}
TARGET_PATH_SUPPORT_ROLES = {"SCOUT", "RELOAD", "SECOND_SHOT", "SUPPORT", "PATH_B"}


def _target_path_guard_issues(intent: OrderIntent, *, for_live_send: bool) -> list[RiskIssue]:
    """Enforce target-path receipt metadata when the receipt supplies it.

    Existing strategy-generated intents that do not carry target-path fields
    keep their current validation behavior. Once a dry-run/manual receipt names
    the daily target mode, 5% path, attack stack, grade, or recent-thesis-loss
    facts, the risk engine treats those fields as executable guard evidence.
    """

    metadata = intent.metadata or {}
    if not metadata:
        return []
    contract_present = _target_path_contract_present(metadata)
    explicit_loss_repeat = (
        _truthy_metadata(metadata.get("same_thesis_lost_recently"))
        or _truthy_metadata(metadata.get("same_thesis_recent_loss"))
    )
    if not contract_present and not explicit_loss_repeat:
        return []

    issues: list[RiskIssue] = []
    grade = _target_grade(metadata)
    rank = TARGET_PATH_GRADE_RANK.get(grade)
    mode = str(
        metadata.get("daily_target_mode")
        or metadata.get("target_mode")
        or metadata.get("mode")
        or ""
    ).strip().upper()
    role = str(
        metadata.get("target_path_role")
        or metadata.get("path_role")
        or metadata.get("daily_target_layer")
        or ""
    ).strip().upper()
    remaining_5 = _target_remaining_to_5pct(metadata)
    minimum_progress_pct = _to_float(metadata.get("minimum_progress_pct"))
    progress_pct = _to_float(
        metadata.get("daily_progress_pct")
        or metadata.get("day_progress_pct")
        or metadata.get("total_day_progress_pct")
    )
    under_5 = remaining_5 is not None and remaining_5 > 0
    base_reached = (
        (remaining_5 is not None and remaining_5 <= 0)
        or (minimum_progress_pct is not None and minimum_progress_pct >= 100)
        or (progress_pct is not None and progress_pct >= 5.0)
    )
    extension_gate_yes = _extension_gate_yes(metadata)
    fresh_risk = str(metadata.get("position_intent") or "NEW").upper() != "HEDGE"

    if mode == "EXTEND" and (rank is None or rank < TARGET_PATH_GRADE_RANK["A"]):
        issues.append(
            RiskIssue(
                "EXTEND_REQUIRES_A_GRADE",
                "EXTEND mode requires A/S grade risk; B+ and lower cannot add extension risk.",
            )
        )
    if base_reached and not extension_gate_yes and fresh_risk and grade.startswith("B"):
        issues.append(
            RiskIssue(
                "BASE_TARGET_REACHED_B_RISK_BLOCKED",
                "+5% is reached and the 10% Extension Gate is NO; fresh B risk is blocked.",
            )
        )
    if (
        under_5
        and rank is not None
        and rank <= TARGET_PATH_GRADE_RANK["B0"]
        and (
            role in TARGET_PATH_MAIN_ROLES
            or role in TARGET_PATH_SUPPORT_ROLES
            or _metadata_claims_main_target_path(metadata, role)
        )
    ):
        issues.append(
            RiskIssue(
                "TARGET_PATH_GRADE_TOO_LOW",
                "B0/B-/C trades are not valid +5% target-path risk.",
            )
        )
    if under_5 and grade == "B+" and _metadata_claims_main_target_path(metadata, role):
        issues.append(
            RiskIssue(
                "B_PLUS_NOT_MAIN_TARGET_PATH",
                "B+ can support a scout/reload, but it cannot be the main +5% path trade.",
            )
        )
    if explicit_loss_repeat and _truthy_metadata(metadata.get("vehicle_unchanged_after_loss")):
        issues.append(
            RiskIssue(
                "SAME_THESIS_LOST_RECENTLY",
                "same thesis lost recently and the vehicle is unchanged; require a new vehicle or structure proof.",
            )
        )
    if _path_or_attack_stack_available(metadata) and not _maps_to_attack_stack(metadata):
        issues.append(
            RiskIssue(
                "PATH_ATTACK_STACK_MAPPING_MISSING",
                "order must map to the 5% PACE BOARD / ATTACK STACK when that board is available.",
            )
        )
    return issues


def _target_path_contract_present(metadata: dict[str, object]) -> bool:
    keys = {
        "daily_target_mode",
        "target_mode",
        "remaining_to_5pct_yen",
        "remaining_to_5pct",
        "remaining_minimum_jpy",
        "minimum_progress_pct",
        "daily_progress_pct",
        "total_day_progress_pct",
        "ten_pct_extension_gate",
        "extension_gate_10pct",
        "extension_gate",
        "target_path_role",
        "path_role",
        "valid_as_target_path",
        "path_board_available",
        "five_pct_path_available",
        "attack_stack_available",
        "maps_to_attack_stack",
        "path_board_slot",
        "attack_stack_slot",
    }
    return any(key in metadata for key in keys)


def _target_grade(metadata: dict[str, object]) -> str:
    raw = (
        metadata.get("conviction_grade")
        or metadata.get("grade")
        or metadata.get("allocation_band")
        or metadata.get("pretrade_allocation_band")
        or ""
    )
    grade = str(raw).strip().upper().replace("_", "").replace(" ", "")
    if grade == "B":
        return "B0"
    return grade


def _target_remaining_to_5pct(metadata: dict[str, object]) -> float | None:
    for key in ("remaining_to_5pct_yen", "remaining_to_5pct", "remaining_minimum_jpy"):
        value = _to_float(metadata.get(key))
        if value is not None:
            return value
    return None


def _extension_gate_yes(metadata: dict[str, object]) -> bool:
    for key in ("ten_pct_extension_gate", "extension_gate_10pct", "extension_gate"):
        if key in metadata:
            value = metadata.get(key)
            if isinstance(value, str):
                return value.strip().upper() == "YES" or _truthy_metadata(value)
            return _truthy_metadata(value)
    return False


def _metadata_claims_main_target_path(metadata: dict[str, object], role: str) -> bool:
    if role in TARGET_PATH_MAIN_ROLES:
        return True
    if role in TARGET_PATH_SUPPORT_ROLES:
        return False
    valid = str(metadata.get("valid_as_target_path") or "").strip().upper()
    return valid == "YES" or _truthy_metadata(metadata.get("target_path_required"))


def _path_or_attack_stack_available(metadata: dict[str, object]) -> bool:
    return (
        _truthy_metadata(metadata.get("path_board_available"))
        or _truthy_metadata(metadata.get("five_pct_path_available"))
        or _truthy_metadata(metadata.get("attack_stack_available"))
    )


def _maps_to_attack_stack(metadata: dict[str, object]) -> bool:
    if _truthy_metadata(metadata.get("maps_to_attack_stack")):
        return True
    return bool(str(metadata.get("path_board_slot") or "").strip()) and bool(
        str(metadata.get("attack_stack_slot") or "").strip()
    )


def _loss_asymmetry_guard_issues(intent: OrderIntent, metrics: RiskMetrics) -> list[RiskIssue]:
    """Block fresh entries whose planned loss exceeds the proven average winner.

    The guard activates only from machine-readable capture_economics metadata.
    It is deliberately JPY-value-free: when recent realized exits are
    NEGATIVE_EXPECTANCY and the average loss is larger than the average win,
    the observed average winner becomes the temporary per-entry loss ceiling.
    """
    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "NEW").upper() == "HEDGE":
        return []
    mode = str(metadata.get("loss_asymmetry_guard_mode") or "").upper()
    if _truthy_metadata(metadata.get("capture_economics_stale")) or mode == CAPTURE_ECONOMICS_STALE_BLOCK_CODE:
        generated_at = str(metadata.get("capture_economics_generated_at_utc") or "unknown")
        latest_close = str(metadata.get("capture_economics_latest_realized_ts_utc") or "unknown")
        reason = str(metadata.get("capture_economics_stale_reason") or "").strip()
        detail = f" ({reason})" if reason else ""
        return [
            RiskIssue(
                CAPTURE_ECONOMICS_STALE_BLOCK_CODE,
                "capture_economics is stale relative to execution_ledger realized closes: "
                f"generated_at_utc={generated_at}, latest_realized_ts_utc={latest_close}{detail}; "
                "refresh capture-economics before adding fresh one-way risk.",
            )
        ]
    if (
        mode == "TP_PROVEN_RELAXED"
        and _loss_asymmetry_tp_relaxation_shape_allowed(intent, metadata)
    ):
        return []
    if mode == LOSS_ASYMMETRY_TP_PROOF_COLLECTION_MIN_LOT_MODE:
        proof_cap = _to_float(metadata.get("loss_asymmetry_guard_effective_max_loss_jpy"))
        normal_cap = _to_float(metadata.get("loss_asymmetry_guard_base_max_loss_jpy"))
        original_cap = _to_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
        if (
            proof_cap is not None
            and normal_cap is not None
            and original_cap is not None
            and original_cap < proof_cap <= normal_cap
            and metrics.risk_jpy <= proof_cap + 1e-9
            and _loss_asymmetry_tp_proof_collection_shape_allowed(intent, metadata)
        ):
            return []
    if mode == LOSS_ASYMMETRY_OANDA_CAMPAIGN_FIREPOWER_MIN_LOT_MODE:
        proof_cap = _to_float(metadata.get("loss_asymmetry_guard_effective_max_loss_jpy"))
        normal_cap = _to_float(metadata.get("loss_asymmetry_guard_base_max_loss_jpy"))
        original_cap = _to_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
        if (
            proof_cap is not None
            and normal_cap is not None
            and original_cap is not None
            and original_cap < proof_cap <= normal_cap
            and metrics.risk_jpy <= proof_cap + 1e-9
            and _loss_asymmetry_oanda_campaign_firepower_min_lot_shape_allowed(intent, metadata)
        ):
            return []
    status = str(metadata.get("capture_economics_status") or "").upper()
    avg_win = _to_float(metadata.get("capture_avg_win_jpy"))
    avg_loss = _to_float(metadata.get("capture_avg_loss_jpy"))
    active = _truthy_metadata(metadata.get("loss_asymmetry_guard_active"))
    if not active and not (
        status == "NEGATIVE_EXPECTANCY"
        and avg_win is not None
        and avg_win > 0
        and avg_loss is not None
        and avg_loss > avg_win
    ):
        return []
    cap = _to_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    if cap is None or cap <= 0:
        cap = avg_win
    if cap is None or cap <= 0:
        return [
            RiskIssue(
                "LOSS_ASYMMETRY_GUARD_CAP_MISSING",
                "loss-asymmetry guard is active but capture_avg_win_jpy / "
                "loss_asymmetry_guard_loss_cap_jpy is missing; refresh capture-economics "
                "before live send.",
            )
        ]
    if metrics.risk_jpy <= cap + 1e-9:
        return []
    loss_text = f"{avg_loss:.0f} JPY" if avg_loss is not None else "unknown"
    return [
        RiskIssue(
            "LOSS_ASYMMETRY_GUARD_EXCEEDED",
            f"planned worst-case loss {metrics.risk_jpy:.0f} JPY exceeds the "
            f"observed average winner cap {cap:.0f} JPY while capture_economics "
            f"status={status or 'UNKNOWN'} and avg_loss={loss_text}; size down "
            "or repair TP/exit payoff before adding fresh one-way risk.",
        )
    ]


def _loss_asymmetry_tp_relaxation_shape_allowed(
    intent: OrderIntent,
    metadata: dict,
) -> bool:
    """Validate the only exit shape allowed to bypass the average-winner cap."""

    if intent.order_type == OrderType.MARKET:
        return False
    if metadata.get("attach_take_profit_on_fill") is not True:
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("capture_take_profit_scope") or "").upper() not in {"PAIR_SIDE_METHOD", "PAIR_SIDE"}:
        return False
    tp_trades = _to_float(metadata.get("capture_take_profit_trades"))
    tp_expectancy = _to_float(metadata.get("capture_take_profit_expectancy_jpy"))
    tp_losses = _to_float(metadata.get("capture_take_profit_losses"))
    return (
        tp_trades is not None
        and tp_trades >= LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES
        and tp_expectancy is not None
        and tp_expectancy > 0
        and tp_losses is not None
        and tp_losses <= 0
    )


def _loss_asymmetry_tp_proof_collection_shape_allowed(
    intent: OrderIntent,
    metadata: dict,
) -> bool:
    """Validate thin exact TP proof before allowing min-lot evidence collection."""

    if intent.order_type == OrderType.MARKET:
        return False
    if str(metadata.get("position_intent") or "").upper() == "HEDGE":
        return False
    if metadata.get("attach_take_profit_on_fill") is not True:
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("capture_take_profit_scope") or "").upper() != "PAIR_SIDE_METHOD":
        return False
    tp_trades = _to_int(metadata.get("capture_take_profit_trades"))
    tp_wins = _to_int(metadata.get("capture_take_profit_wins"))
    tp_expectancy = _to_float(metadata.get("capture_take_profit_expectancy_jpy"))
    tp_avg_win = _to_float(metadata.get("capture_take_profit_avg_win_jpy"))
    tp_avg_loss = _to_float(metadata.get("capture_take_profit_avg_loss_jpy"))
    tp_losses = _to_int(metadata.get("capture_take_profit_losses"))
    market_close_expectancy = _to_float(metadata.get("capture_market_close_expectancy_jpy"))
    avg_loss = _to_float(metadata.get("capture_avg_loss_jpy"))
    if tp_trades is None or not (
        LOSS_ASYMMETRY_TP_PROOF_COLLECTION_MIN_EXIT_TRADES
        <= tp_trades
        < LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES
    ):
        return False
    if tp_expectancy is None or tp_expectancy <= 0 or tp_avg_win is None or tp_avg_win <= 0:
        return False
    if tp_losses is None or tp_losses > 0:
        return False
    if market_close_expectancy is None or market_close_expectancy >= 0:
        return False
    wins = tp_wins if tp_wins is not None else max(tp_trades - tp_losses, 0)
    hit_rate = wins / tp_trades if tp_trades > 0 else 0.0
    lower = hit_rate_wilson_lower(hit_rate, tp_trades)
    loss_proxy_candidates = (tp_avg_loss, avg_loss)
    loss_proxy = max(
        (value for value in loss_proxy_candidates if value is not None and value > 0),
        default=None,
    )
    if lower is None or loss_proxy is None:
        return False
    pessimistic_expectancy = (lower * tp_avg_win) - ((1.0 - lower) * loss_proxy)
    return pessimistic_expectancy > 0


def _loss_asymmetry_oanda_campaign_firepower_min_lot_shape_allowed(
    intent: OrderIntent,
    metadata: dict,
) -> bool:
    """Validate OANDA firepower min-lot metadata before accepting the cap lift."""

    if intent.order_type == OrderType.MARKET:
        return False
    if str(metadata.get("position_intent") or "").upper() == "HEDGE":
        return False
    if metadata.get("attach_take_profit_on_fill") is not True:
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return False
    if metadata.get("positive_rotation_oanda_campaign_min_lot_sizing") is not True:
        return False
    if metadata.get("positive_rotation_oanda_campaign_firepower_vehicle_match") is not True:
        return False
    if metadata.get("positive_rotation_oanda_campaign_minimum_floor_reachable") is not True:
        return False
    status = str(metadata.get("positive_rotation_oanda_campaign_firepower_status") or "").upper()
    if status not in OANDA_CAMPAIGN_FIREPOWER_TARGET_OK_STATUSES:
        return False
    matching_return = _to_float(
        metadata.get(
            "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day"
        )
    )
    if matching_return is None or matching_return <= 0:
        return False
    min_lot_loss = _to_float(metadata.get("positive_rotation_oanda_campaign_min_lot_loss_jpy"))
    effective_cap = _to_float(metadata.get("loss_asymmetry_guard_effective_max_loss_jpy"))
    min_lot_units = _to_int(metadata.get("positive_rotation_oanda_campaign_min_lot_units"))
    if min_lot_loss is None or effective_cap is None or min_lot_loss > effective_cap + 1e-9:
        return False
    return min_lot_units == MIN_PRODUCTION_LOT_UNITS


def _loss_asymmetry_oanda_campaign_firepower_relaxed_shape_allowed(
    intent: OrderIntent,
    metadata: dict,
    metrics: RiskMetrics,
) -> bool:
    """Validate OANDA firepower normal-cap relaxation metadata."""

    if intent.order_type == OrderType.MARKET:
        return False
    if str(metadata.get("position_intent") or "").upper() == "HEDGE":
        return False
    if metadata.get("attach_take_profit_on_fill") is not True:
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return False
    if metadata.get("positive_rotation_oanda_campaign_normal_cap_relaxed") is not True:
        return False
    if metadata.get("positive_rotation_oanda_campaign_firepower_vehicle_match") is not True:
        return False
    if metadata.get("positive_rotation_oanda_campaign_minimum_floor_reachable") is not True:
        return False
    if (
        metadata.get("positive_rotation_oanda_campaign_normal_cap_minimum_floor_reachable")
        is not True
    ):
        return False
    status = str(metadata.get("positive_rotation_oanda_campaign_firepower_status") or "").upper()
    if status not in OANDA_CAMPAIGN_FIREPOWER_TARGET_OK_STATUSES:
        return False
    original_cap = _to_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    normal_cap = _to_float(metadata.get("positive_rotation_oanda_campaign_normal_cap_jpy"))
    effective_cap = _to_float(metadata.get("loss_asymmetry_guard_effective_max_loss_jpy"))
    if (
        original_cap is None
        or normal_cap is None
        or effective_cap is None
        or not (original_cap < effective_cap <= normal_cap)
        or metrics.risk_jpy > effective_cap + 1e-9
    ):
        return False
    required_trades = _to_int(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_required_minimum_trades")
    )
    target_trades = _to_int(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_target_trades_per_day")
    )
    observed_attempts = _to_float(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_observed_attempts_per_day")
    )
    if (
        required_trades is None
        or target_trades is None
        or observed_attempts is None
        or required_trades < 0
    ):
        return False
    if required_trades > 0 and (
        required_trades > target_trades
        or required_trades > int(math.floor(observed_attempts))
    ):
        return False
    matching_return = _to_float(
        metadata.get(
            "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day"
        )
    )
    weighted_return = _to_float(
        metadata.get(
            "positive_rotation_oanda_campaign_normal_cap_weighted_return_pct_per_trade"
        )
    )
    return (
        weighted_return is not None
        and weighted_return > 0
        and matching_return is not None
        and matching_return > 0
    )


@dataclass(frozen=True)
class InstrumentSpec:
    pair: str
    pip_factor: int
    normal_spread_pips: float
    margin_rate: float = OANDA_JP_RETAIL_FX_MARGIN_RATE

    @property
    def pip_size(self) -> float:
        return 1.0 / self.pip_factor


DEFAULT_SPECS: dict[str, InstrumentSpec] = {
    pair: InstrumentSpec(pair, instrument_pip_factor(pair), NORMAL_SPREAD_PIPS[pair])
    for pair in DEFAULT_TRADER_PAIRS
}


def estimate_required_margin_jpy(*, units: int, entry_price: float, quote_to_jpy: float, spec: InstrumentSpec) -> float:
    """Estimate initial margin in account JPY for a candidate FX order."""
    return max(0.0, abs(units) * abs(entry_price) * quote_to_jpy * spec.margin_rate)


def margin_budget_jpy(account: AccountSummary, *, max_margin_utilization_pct: float) -> float:
    """Return the smaller of broker free margin and operator utilization headroom."""
    utilization_budget = account.nav_jpy * (max_margin_utilization_pct / 100.0) - account.margin_used_jpy
    return min(account.margin_available_jpy, utilization_budget)


def hedge_margin_free_units(
    *,
    pair: str,
    side: Side,
    snapshot: BrokerSnapshot,
    position_intent: str | None = None,
) -> int:
    """Return trader-owned units that can be hedged before the bot nets out.

    OANDA's broker margin calculation sees every same-pair position, but the
    trader is not allowed to use manual/tagless exposure as a hedge reference.
    A bot HEDGE may only cover trader-owned opposite exposure; otherwise a
    manual LONG could silently authorize a fresh bot SHORT and turn operator
    risk into strategy-owned exposure.
    """
    if not _account_hedging_enabled(snapshot) or str(position_intent or "").upper() != "HEDGE":
        return 0
    long_units, short_units = _same_pair_position_units(snapshot, pair, owner=Owner.TRADER)
    if side == Side.LONG:
        return max(0, short_units - long_units)
    return max(0, long_units - short_units)


def broker_margin_free_units(*, pair: str, side: Side, snapshot: BrokerSnapshot) -> int:
    """Return same-pair units that add no broker margin on a hedging account.

    OANDA v20 hedging margin is based on the larger side of all same-pair
    open trades in the account, including operator-managed manual/tagless
    trades. This broker-truth number is separate from
    `hedge_margin_free_units`, which is the strategy-owned hedge cap and
    intentionally ignores manual exposure.
    """
    if not _account_hedging_enabled(snapshot):
        return 0
    long_units, short_units = _same_pair_position_units(snapshot, pair)
    if side == Side.LONG:
        return max(0, short_units - long_units)
    return max(0, long_units - short_units)


def incremental_margin_units(
    *,
    pair: str,
    side: Side,
    units: int,
    snapshot: BrokerSnapshot,
    position_intent: str | None = None,
) -> int:
    """Return units that increase broker-required margin for a candidate order."""
    requested_units = max(0, abs(int(units)))
    if requested_units <= 0:
        return 0
    if not _account_hedging_enabled(snapshot):
        return requested_units

    long_units, short_units = _same_pair_position_units(snapshot, pair)
    before_larger_side = max(long_units, short_units)
    if side == Side.LONG:
        long_units += requested_units
    else:
        short_units += requested_units
    return max(0, max(long_units, short_units) - before_larger_side)


def estimate_incremental_margin_jpy(
    *,
    pair: str,
    side: Side,
    units: int,
    entry_price: float,
    quote_to_jpy: float,
    spec: InstrumentSpec,
    snapshot: BrokerSnapshot,
    position_intent: str | None = None,
) -> float:
    """Estimate account-JPY margin increase after same-pair hedging offsets."""
    margin_units = incremental_margin_units(
        pair=pair,
        side=side,
        units=units,
        snapshot=snapshot,
        position_intent=position_intent,
    )
    return estimate_required_margin_jpy(
        units=margin_units,
        entry_price=entry_price,
        quote_to_jpy=quote_to_jpy,
        spec=spec,
    )


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _forecast_unselected_projection_conflict_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    metadata = intent.metadata or {}
    direction = str(metadata.get("forecast_direction") or "").upper()
    if direction not in {"RANGE", "UP", "DOWN"}:
        return []
    support = metadata.get("forecast_market_support")
    if not isinstance(support, dict):
        return []
    signals = support.get("unselected_signals")
    if not isinstance(signals, list):
        return []

    conflicts: list[dict] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_direction = str(signal.get("direction") or "").upper()
        signal_side = Side.LONG if signal_direction == "UP" else Side.SHORT if signal_direction == "DOWN" else None
        if signal_side is None or signal_side == intent.side:
            continue
        hit_rate = _to_float(signal.get("hit_rate"))
        samples = _to_int(signal.get("samples"))
        if hit_rate is None or samples is None:
            continue
        if not _forecast_support_signal_clears_live_precision(signal):
            continue
        conflicts.append(signal)

    if not conflicts:
        return []
    top = conflicts[0]
    severity = "BLOCK" if for_live_send else "WARN"
    signal_name = str(top.get("name") or "projection")
    signal_direction = str(top.get("direction") or "").upper()
    hit_rate = _to_float(top.get("hit_rate")) or 0.0
    samples = _to_int(top.get("samples")) or 0
    if direction in {"UP", "DOWN"}:
        forecast_side = Side.LONG if direction == "UP" else Side.SHORT
        if intent.side != forecast_side:
            return []
        if _technical_harvest_precision_support_for_intent(intent) is not None:
            return []
        if _bidask_replay_precision_support_for_intent(intent) is not None:
            return []
        if _forecast_selected_direction_has_audited_support(
            metadata,
            support,
            direction=direction,
        ):
            return []
        return [
            RiskIssue(
                "FORECAST_UNSELECTED_OPPOSITE_PROJECTION",
                (
                    f"{intent.pair} {intent.side.value} forecast {direction} selected the "
                    f"{forecast_side.value} side, but audited unselected {signal_name} "
                    f"projection supports {signal_direction} (hit_rate={hit_rate:.2f}, "
                    f"samples={samples}); keep this entry dry-run until the selected "
                    "direction has current audited projection support or the opposing "
                    "projection no longer conflicts."
                ),
                severity=severity,
            )
        ]
    return [
        RiskIssue(
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
            (
                f"{intent.pair} {intent.side.value} has forecast RANGE, but audited "
                f"{signal_name} projection supports {signal_direction} "
                f"(hit_rate={hit_rate:.2f}, samples={samples}); do not send the "
                "opposite entry until the forecast resolves or the projection no longer conflicts."
            ),
            severity=severity,
        )
    ]


def _forecast_market_support(metadata: dict) -> dict:
    support = metadata.get("forecast_market_support")
    return support if isinstance(support, dict) else {}


def _bidask_replay_precision_support_for_intent(intent: OrderIntent) -> dict | None:
    metadata = intent.metadata or {}
    method = intent.market_context.method if intent.market_context is not None else None
    support = bidask_replay_precision_support(
        metadata,
        pair=intent.pair,
        side=intent.side.value,
        order_type=intent.order_type.value,
        method=method.value if isinstance(method, TradeMethod) else str(method or ""),
        entry=_to_float(intent.entry),
        take_profit=_to_float(intent.tp),
        stop_loss=_to_float(intent.sl),
    )
    if support is not None:
        metadata["bidask_replay_precision_live_ready"] = True
        metadata["bidask_replay_precision_support"] = support
    return support


def _bidask_replay_negative_precision_issue_for_intent(intent: OrderIntent) -> RiskIssue | None:
    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "").upper() == "HEDGE":
        return None
    method = intent.market_context.method if intent.market_context is not None else None
    issue = bidask_replay_negative_precision_issue(
        metadata,
        pair=intent.pair,
        side=intent.side.value,
        order_type=intent.order_type.value,
        method=method.value if isinstance(method, TradeMethod) else str(method or ""),
        entry=_to_float(intent.entry),
        take_profit=_to_float(intent.tp),
        stop_loss=_to_float(intent.sl),
    )
    if issue is None:
        return None
    metadata["bidask_replay_precision_negative"] = issue
    return RiskIssue(
        "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
        (
            f"{intent.pair} {intent.side.value} forecast {issue['direction']} matches S5 bid/ask "
            f"negative replay bucket {issue['name']}: hit_rate="
            f"{float(issue.get('directional_hit_rate') or 0.0):.2f}, avg_final="
            f"{float(issue.get('avg_final_pips') or 0.0):.2f}pip, avg_MAE="
            f"{float(issue.get('avg_mae_pips') or 0.0):.2f}pip over "
            f"{int(issue.get('samples') or 0)} sample(s). Do not send inside a proven "
            "bid/ask losing pair-direction."
        ),
        severity="BLOCK",
    )


def _technical_harvest_precision_support_for_intent(intent: OrderIntent) -> dict | None:
    metadata = intent.metadata or {}
    method = intent.market_context.method if intent.market_context is not None else None
    support = technical_harvest_precision_support(
        metadata,
        pair=intent.pair,
        side=intent.side.value,
        order_type=intent.order_type.value,
        method=method.value if isinstance(method, TradeMethod) else str(method or ""),
        entry=_to_float(intent.entry),
        take_profit=_to_float(intent.tp),
        stop_loss=_to_float(intent.sl),
    )
    if support is not None:
        metadata["technical_harvest_precision_live_ready"] = True
        metadata["technical_harvest_precision_support"] = support
    return support


def _technical_harvest_negative_precision_issue_for_intent(intent: OrderIntent) -> RiskIssue | None:
    metadata = intent.metadata or {}
    method = intent.market_context.method if intent.market_context is not None else None
    issue = technical_harvest_negative_precision_issue(
        metadata,
        pair=intent.pair,
        side=intent.side.value,
        order_type=intent.order_type.value,
        method=method.value if isinstance(method, TradeMethod) else str(method or ""),
        entry=_to_float(intent.entry),
        take_profit=_to_float(intent.tp),
        stop_loss=_to_float(intent.sl),
    )
    if issue is None:
        return None
    metadata["technical_harvest_precision_negative"] = issue
    return RiskIssue(
        "TECHNICAL_HARVEST_NEGATIVE_BUCKET_FOR_LIVE",
        (
            f"{intent.pair} {intent.side.value} attached HARVEST TP matches audited negative "
            f"technical bucket {issue['name']} ({issue['feature']}): TP-first "
            f"{float(issue.get('scalp_tp_first_hit_rate') or 0.0):.2f}, Wilson95_lower="
            f"{float(issue.get('scalp_tp_first_wilson95_lower') or 0.0):.2f} over "
            f"{int(issue.get('samples') or 0)} sample(s). Do not send inside a proven losing "
            "technical state."
        ),
        severity="BLOCK",
    )


def _forecast_support_signal_clears_live_precision(signal: dict) -> bool:
    return support_signal_clears_live_precision(
        signal,
        min_wilson_lower=FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
        min_samples=FORECAST_LIVE_PRECISION_MIN_SAMPLES,
        min_target_pips=FORECAST_LIVE_PRECISION_MIN_TARGET_PIPS,
    )


def _forecast_market_support_has_current_directional_signal(
    support: dict,
    *,
    direction: str,
) -> bool:
    for signal in support.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        if str(signal.get("direction") or "").upper() != direction:
            continue
        if _forecast_support_signal_clears_live_precision(signal):
            return True
    return False


def _forecast_selected_direction_has_audited_support(
    metadata: dict,
    support: dict,
    *,
    direction: str,
) -> bool:
    if _forecast_directional_bucket_is_known_weak(metadata, support):
        return False
    if not bool(support.get("ok")):
        return False
    support_direction = str(support.get("direction") or "").upper()
    if support_direction and support_direction != direction:
        return False
    if bool(support.get("bootstrap_projection_support")):
        raw_confidence = _to_float(metadata.get("forecast_raw_confidence"))
        return raw_confidence is not None and raw_confidence >= FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE
    aligned_count = _to_int(support.get("aligned_projection_count")) or 0
    if aligned_count <= 0:
        return False
    samples = _to_int(support.get("best_aligned_samples"))
    if samples is None or samples <= 0:
        samples = _to_int(support.get("best_samples")) or 0
    if samples < FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
        return False
    hit_rate = _to_float(support.get("best_aligned_hit_rate"))
    if hit_rate is None:
        hit_rate = _to_float(support.get("best_hit_rate"))
    if hit_rate is None or hit_rate < FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE:
        return False
    return _forecast_market_support_has_current_directional_signal(
        support,
        direction=direction,
    )


def _forecast_directional_hit_rate(metadata: dict, support: dict) -> tuple[float | None, int, str]:
    hit_rate = _to_float(metadata.get("forecast_directional_hit_rate"))
    if hit_rate is None:
        hit_rate = _to_float(support.get("directional_hit_rate"))
    samples = _to_int(metadata.get("forecast_directional_samples")) or 0
    if samples <= 0:
        samples = _to_int(support.get("directional_samples")) or 0
    calibration_name = str(
        metadata.get("forecast_directional_calibration_name")
        or support.get("directional_calibration_name")
        or "directional_forecast"
    )
    return hit_rate, samples, calibration_name


def _forecast_directional_economic_hit_rate(metadata: dict, support: dict) -> tuple[float | None, int]:
    hit_rate = _to_float(metadata.get("forecast_directional_economic_hit_rate"))
    if hit_rate is None:
        hit_rate = _to_float(support.get("directional_economic_hit_rate"))
    samples = _to_int(metadata.get("forecast_directional_economic_samples")) or 0
    if samples <= 0:
        samples = _to_int(support.get("directional_economic_samples")) or 0
    return hit_rate, samples


def _forecast_directional_bucket_clears_live_precision(metadata: dict, support: dict) -> bool:
    hit_rate, samples, _ = _forecast_directional_hit_rate(metadata, support)
    lower = hit_rate_wilson_lower(hit_rate, samples)
    headline_ok = (
        hit_rate is not None
        and samples >= FORECAST_LIVE_PRECISION_MIN_SAMPLES
        and lower is not None
        and lower >= FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER
    )
    if not headline_ok:
        return False
    economic_hit_rate, economic_samples = _forecast_directional_economic_hit_rate(metadata, support)
    if economic_hit_rate is None:
        return _forecast_directional_timeout_rate(metadata, support) <= 0.0
    economic_lower = hit_rate_wilson_lower(economic_hit_rate, economic_samples)
    return (
        economic_samples >= FORECAST_LIVE_PRECISION_MIN_SAMPLES
        and economic_lower is not None
        and economic_lower >= FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER
    )


def _forecast_directional_timeout_rate(metadata: dict, support: dict) -> float:
    rate = _to_float(metadata.get("forecast_directional_timeout_rate"))
    if rate is None:
        rate = _to_float(support.get("directional_timeout_rate"))
    return max(0.0, min(1.0, rate or 0.0))


def _forecast_directional_bucket_is_known_weak(metadata: dict, support: dict) -> bool:
    hit_rate, samples, _ = _forecast_directional_hit_rate(metadata, support)
    weak_hit_rate = (
        hit_rate is not None
        and samples >= FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES
        and hit_rate < FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE
    )
    invalidation_first_rate, _, _ = _forecast_directional_invalidation_first(metadata, support)
    adverse_path = (
        invalidation_first_rate is not None
        and samples >= FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES
        and invalidation_first_rate >= FORECAST_DIRECTIONAL_LIVE_MAX_INVALIDATION_FIRST_RATE
    )
    return weak_hit_rate or adverse_path


def _forecast_directional_invalidation_first(metadata: dict, support: dict) -> tuple[float | None, int | None, str]:
    invalidation_first_rate = _to_float(metadata.get("forecast_directional_invalidation_first_rate"))
    if invalidation_first_rate is None:
        invalidation_first_rate = _to_float(support.get("directional_invalidation_first_rate"))
    invalidation_first_count = _to_int(metadata.get("forecast_directional_invalidation_first_count"))
    if invalidation_first_count is None:
        invalidation_first_count = _to_int(support.get("directional_invalidation_first_count"))
    calibration_name = str(
        metadata.get("forecast_directional_calibration_name")
        or support.get("directional_calibration_name")
        or "directional_forecast"
    )
    return invalidation_first_rate, invalidation_first_count, calibration_name


def _forecast_directional_bucket_issue(
    intent: OrderIntent,
    *,
    direction: str,
    metadata: dict,
    support: dict,
) -> RiskIssue:
    invalidation_first_rate, invalidation_first_count, calibration_name = _forecast_directional_invalidation_first(
        metadata,
        support,
    )
    _hit_rate, samples, hit_rate_calibration_name = _forecast_directional_hit_rate(metadata, support)
    if (
        invalidation_first_rate is not None
        and samples >= FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES
        and invalidation_first_rate >= FORECAST_DIRECTIONAL_LIVE_MAX_INVALIDATION_FIRST_RATE
    ):
        count_text = (
            f"{invalidation_first_count}/{samples}"
            if invalidation_first_count is not None
            else f"{samples} sample(s)"
        )
        return RiskIssue(
            "FORECAST_DIRECTIONAL_INVALIDATION_FIRST_FOR_LIVE",
            (
                f"{intent.pair} {intent.side.value} forecast {direction} bucket "
                f"{calibration_name} touched invalidation before target in {count_text} "
                f"sample(s) ({invalidation_first_rate:.2f}); this adverse-path bucket cannot "
                "veto the opposite side or authorize live send without independent audited "
                "projection support."
            ),
            severity="BLOCK",
        )
    return _forecast_directional_hit_rate_weak_issue(
        intent,
        direction=direction,
        metadata=metadata
        | {
            "forecast_directional_calibration_name": hit_rate_calibration_name,
        },
        support=support,
    )


def _forecast_supported_opposite_side_blocks(
    metadata: dict,
    *,
    forecast_side: Side,
    min_confidence: float,
) -> bool:
    direction = str(metadata.get("forecast_direction") or "").upper()
    expected_side = Side.LONG if direction == "UP" else Side.SHORT if direction == "DOWN" else None
    if expected_side != forecast_side:
        return False
    raw_confidence = _to_float(metadata.get("forecast_raw_confidence"))
    support_floor = max(0.0, min_confidence - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL)
    if raw_confidence is None or raw_confidence < support_floor:
        return False
    chart_direction_bias = str(metadata.get("chart_direction_bias") or "").upper()
    if chart_direction_bias and chart_direction_bias != forecast_side.value:
        return False
    support = _forecast_market_support(metadata)
    if not bool(support.get("ok")):
        return False
    support_direction = str(support.get("direction") or "").upper()
    if support_direction and support_direction != direction:
        return False
    if bool(support.get("bootstrap_projection_support")):
        return True
    aligned_count = _to_int(support.get("aligned_projection_count")) or 0
    if aligned_count <= 0:
        return False
    return _forecast_market_support_has_current_directional_signal(
        support,
        direction=direction,
    )


def _forecast_confidence_required_issue(
    intent: OrderIntent,
    *,
    direction: str,
    confidence: float | None,
    min_confidence: float,
) -> RiskIssue:
    return RiskIssue(
        "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
        (
            f"{intent.pair} {intent.side.value} forecast {direction} confidence "
            f"{0.0 if confidence is None else confidence:.2f} < {min_confidence:.2f}; "
            "the weak forecast cannot veto the opposite side, but live send still needs "
            "an executable forecast or audited replacement evidence."
        ),
        severity="BLOCK",
    )


def _range_rail_limit_metadata_ok(metadata: dict) -> bool:
    return (
        str(metadata.get("geometry_model") or "").upper() == "RANGE_RAIL_LIMIT"
        and bool(metadata.get("range_tp_is_inside_box"))
        and bool(metadata.get("range_sl_outside_box"))
    )


def _support_signal_within_forecast_horizon(
    signal: dict,
    *,
    forecast_horizon_min: float | None,
) -> bool:
    if forecast_horizon_min is None or forecast_horizon_min <= 0:
        return True
    lead_time = _to_float(signal.get("lead_time_min"))
    if lead_time is None:
        return True
    return max(0.0, lead_time) <= forecast_horizon_min


def _forecast_range_unselected_projection_support_allows_side(
    intent: OrderIntent,
    metadata: dict,
    support: dict,
    *,
    confidence: float | None,
    min_confidence: float,
) -> bool:
    """Mirror intent_generator's RANGE rail support override at the gateway."""
    if intent.side not in {Side.LONG, Side.SHORT}:
        return False
    if intent.order_type != OrderType.LIMIT:
        return False
    method = intent.market_context.method if intent.market_context is not None else None
    if method != TradeMethod.RANGE_ROTATION:
        return False
    if not _range_rail_limit_metadata_ok(metadata):
        return False
    if confidence is None:
        return False
    support_floor = max(0.0, min_confidence - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL)
    if confidence < support_floor:
        return False
    expected_direction = "UP" if intent.side == Side.LONG else "DOWN"
    forecast_horizon_min = _to_float(metadata.get("forecast_horizon_min"))
    signals = support.get("unselected_signals")
    if not isinstance(signals, list):
        return False
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        if str(signal.get("direction") or "").upper() != expected_direction:
            continue
        if not _support_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        ):
            continue
        signal_confidence = _to_float(signal.get("confidence")) or 0.0
        hit_rate = _to_float(signal.get("hit_rate")) or 0.0
        samples = _to_int(signal.get("samples")) or 0
        if (
            signal_confidence >= FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE
            and hit_rate >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE
            and samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
            and _forecast_support_signal_clears_live_precision(signal)
        ):
            return True
    return False


def _forecast_unclear_unselected_projection_support_allows_side(
    intent: OrderIntent,
    metadata: dict,
    support: dict,
) -> bool:
    """Mirror intent_generator's UNCLEAR forecast passive LIMIT override."""
    if intent.side not in {Side.LONG, Side.SHORT}:
        return False
    if intent.order_type != OrderType.LIMIT:
        return False
    if str(metadata.get("forecast_direction") or "").upper() != "UNCLEAR":
        return False
    chart_direction_bias = str(metadata.get("chart_direction_bias") or "").upper()
    if chart_direction_bias and chart_direction_bias != intent.side.value:
        return False
    expected_direction = "UP" if intent.side == Side.LONG else "DOWN"
    opposite_direction = "DOWN" if expected_direction == "UP" else "UP"
    forecast_horizon_min = _to_float(metadata.get("forecast_horizon_min"))
    signals = support.get("unselected_signals")
    if not isinstance(signals, list):
        return False
    has_same_side_signal = False
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_direction = str(signal.get("direction") or "").upper()
        if signal_direction not in {"UP", "DOWN"}:
            continue
        if not _support_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        ):
            continue
        signal_confidence = _to_float(signal.get("confidence")) or 0.0
        hit_rate = _to_float(signal.get("hit_rate")) or 0.0
        samples = _to_int(signal.get("samples")) or 0
        strong_enough = (
            signal_confidence >= FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE
            and hit_rate >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE
            and samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
            and _forecast_support_signal_clears_live_precision(signal)
        )
        if not strong_enough:
            continue
        if signal_direction == opposite_direction:
            return False
        if signal_direction == expected_direction:
            has_same_side_signal = True
    return has_same_side_signal


def _forecast_range_confidence_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    if not for_live_send:
        return []
    metadata = intent.metadata or {}
    if str(metadata.get("forecast_direction") or "").upper() != "RANGE":
        return []
    if (
        _intent_declares_recovery_hedge(intent)
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    ):
        return []
    confidence = _to_float(metadata.get("forecast_confidence"))
    min_confidence = FORECAST_RANGE_ROTATION_MIN_CONFIDENCE
    if confidence is not None and confidence >= min_confidence:
        return []
    support = _forecast_market_support(metadata)
    if _forecast_range_unselected_projection_support_allows_side(
        intent,
        metadata,
        support,
        confidence=confidence,
        min_confidence=min_confidence,
    ):
        return []
    return [
        _forecast_confidence_required_issue(
            intent,
            direction="RANGE",
            confidence=confidence,
            min_confidence=min_confidence,
        )
    ]


def _forecast_watch_only_live_send_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    if not for_live_send:
        return []
    metadata = intent.metadata or {}
    campaign_role = str(metadata.get("campaign_role") or "").upper()
    watch_only = bool(metadata.get("forecast_watch_only")) or campaign_role == "FORECAST_WATCH"
    if not watch_only:
        return []
    if _forecast_watch_only_live_override_valid(intent, metadata):
        return []
    direction = str(metadata.get("forecast_direction") or "").upper() or "UNKNOWN"
    confidence = _to_float(metadata.get("forecast_confidence"))
    reason = str(metadata.get("forecast_watch_only_reason") or "").strip()
    reason_tail = f" Reason: {reason}" if reason else ""
    return [
        RiskIssue(
            "FORECAST_WATCH_ONLY",
            (
                f"{intent.pair} {intent.side.value} is labeled as a watch-only forecast lane "
                f"({direction} conf={0.0 if confidence is None else confidence:.2f}); "
                "gateway live send is blocked unless the intent carries a current watch-only "
                f"live override and the order remains a non-market audited support/rail entry.{reason_tail}"
            ),
            severity="BLOCK",
        )
    ]


def _forecast_watch_only_live_override_valid(intent: OrderIntent, metadata: dict) -> bool:
    if not bool(metadata.get("forecast_watch_only_live_override")):
        return False
    # The generator's watch overrides explicitly say "Do not convert to MARKET";
    # the gateway repeats that contract for replayed receipts and manual staging.
    if intent.order_type == OrderType.MARKET:
        return False
    direction = str(metadata.get("forecast_direction") or "").upper()
    confidence = _to_float(metadata.get("forecast_confidence"))
    method = intent.market_context.method if intent.market_context is not None else None
    support = _forecast_market_support(metadata)
    if direction == "RANGE":
        return (
            method == TradeMethod.RANGE_ROTATION
            and intent.order_type == OrderType.LIMIT
            and _range_rail_limit_metadata_ok(metadata)
            and (
                (confidence is not None and confidence >= FORECAST_RANGE_ROTATION_MIN_CONFIDENCE)
                or _forecast_range_unselected_projection_support_allows_side(
                    intent,
                    metadata,
                    support,
                    confidence=confidence,
                    min_confidence=FORECAST_RANGE_ROTATION_MIN_CONFIDENCE,
                )
            )
        )
    if direction not in {"UP", "DOWN"}:
        return False
    expected_side = Side.LONG if direction == "UP" else Side.SHORT
    if intent.side != expected_side:
        return False
    support_floor = max(
        0.0,
        FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL,
    )
    raw_confidence = _to_float(metadata.get("forecast_raw_confidence"))
    if (
        (confidence is None or confidence < support_floor)
        and (raw_confidence is None or raw_confidence < support_floor)
    ):
        return False
    return _forecast_selected_direction_has_audited_support(
        metadata,
        support,
        direction=direction,
    )


def _forecast_executable_live_readiness_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    if not for_live_send:
        return []
    metadata = intent.metadata or {}
    direction = str(metadata.get("forecast_direction") or "").upper()
    if not direction or direction in {"UP", "DOWN", "RANGE"}:
        return []
    if (
        _intent_declares_recovery_hedge(intent)
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    ):
        return []
    confidence = _to_float(metadata.get("forecast_confidence"))
    support = _forecast_market_support(metadata)
    if _forecast_unclear_unselected_projection_support_allows_side(intent, metadata, support):
        return []
    return [
        RiskIssue(
            "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
            (
                f"{intent.pair} {intent.side.value} current pair forecast is {direction} "
                f"conf={0.0 if confidence is None else confidence:.2f}; fresh live entries need "
                "an executable UP/DOWN/RANGE prediction before gateway send."
            ),
            severity="BLOCK",
        )
    ]


def _forecast_directional_hit_rate_weak_issue(
    intent: OrderIntent,
    *,
    direction: str,
    metadata: dict,
    support: dict,
) -> RiskIssue:
    hit_rate, samples, calibration_name = _forecast_directional_hit_rate(metadata, support)
    lower = hit_rate_wilson_lower(hit_rate, samples)
    economic_hit_rate, economic_samples = _forecast_directional_economic_hit_rate(metadata, support)
    economic_lower = hit_rate_wilson_lower(economic_hit_rate, economic_samples)
    return RiskIssue(
        "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE",
        (
            f"{intent.pair} {intent.side.value} forecast {direction} bucket "
            f"{calibration_name} hit_rate={0.0 if hit_rate is None else hit_rate:.2f}, "
            f"Wilson95_lower={0.0 if lower is None else lower:.2f} over {samples} sample(s); "
            f"economic_hit_rate={0.0 if economic_hit_rate is None else economic_hit_rate:.2f}, "
            f"economic_Wilson95_lower={0.0 if economic_lower is None else economic_lower:.2f} "
            f"over {economic_samples} economic sample(s); "
            f"live requires Wilson95_lower>={FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER:.2f} "
            f"and samples>={FORECAST_LIVE_PRECISION_MIN_SAMPLES}. This bucket cannot "
            "authorize live send without independent audited projection support."
        ),
        severity="BLOCK",
    )


def _forecast_range_method_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    metadata = intent.metadata or {}
    if str(metadata.get("forecast_direction") or "").upper() != "RANGE":
        return []
    method = intent.market_context.method if intent.market_context is not None else None
    if method == TradeMethod.RANGE_ROTATION:
        return []
    if _range_forecast_tp_proven_breakout_failure_allowed(intent, method):
        return []
    if (
        _intent_declares_recovery_hedge(intent)
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    ):
        return []
    severity = "BLOCK" if for_live_send else "WARN"
    return [
        RiskIssue(
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            (
                f"{intent.pair} {intent.side.value} has a RANGE forecast; only executable "
                "RANGE_ROTATION rail geometry may become LIVE_READY from a RANGE prediction."
            ),
            severity=severity,
        )
    ]


def _range_forecast_tp_proven_breakout_failure_allowed(
    intent: OrderIntent,
    method: TradeMethod | None,
) -> bool:
    """Allow only the exact broker-TP-proven failed-break fade inside RANGE.

    A RANGE forecast still must not authorize generic trend/failure chasing.
    The exception is the already realized TP_PROVEN_HARVEST BREAKOUT_FAILURE
    LIMIT shape: it is a passive failed-break fade with exact pair/side/method
    TAKE_PROFIT_ORDER proof and positive Wilson-stressed expectancy.
    """

    if method != TradeMethod.BREAKOUT_FAILURE:
        return False
    if intent.order_type == OrderType.MARKET:
        return False
    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "NEW").upper() == "HEDGE":
        return False
    if metadata.get("attach_take_profit_on_fill") is not True:
        return False
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("opportunity_mode") or "").upper() != "HARVEST":
        return False
    if str(metadata.get("positive_rotation_mode") or "").upper() != "TP_PROVEN_HARVEST":
        return False
    if metadata.get("positive_rotation_live_ready") is not True:
        return False
    if str(metadata.get("capture_take_profit_scope") or "").upper() != "PAIR_SIDE_METHOD":
        return False
    expected_scope = (
        f"{intent.pair}|{intent.side.value}|{method.value}|TAKE_PROFIT_ORDER"
    ).upper()
    if str(metadata.get("capture_take_profit_scope_key") or "").upper() != expected_scope:
        return False
    tp_trades = int(_to_float(metadata.get("capture_take_profit_trades")) or 0)
    if tp_trades < LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES:
        return False
    tp_losses = int(_to_float(metadata.get("capture_take_profit_losses")) or 0)
    if tp_losses != 0:
        return False
    tp_expectancy = _to_float(metadata.get("capture_take_profit_expectancy_jpy"))
    if tp_expectancy is None or tp_expectancy <= 0:
        return False
    pessimistic = _to_float(metadata.get("positive_rotation_pessimistic_expectancy_jpy"))
    if pessimistic is None or pessimistic <= 0:
        return False
    return True


@dataclass(frozen=True)
class RiskPolicy:
    # No JPY literal fallback: production and tests must inject a cap from
    # intent.metadata, an explicit policy, or an equity-derived daily target
    # ledger. RiskEngine fails closed with LOSS_CAP_MISSING when absent.
    max_loss_jpy: float | None = None
    # Equity-percent cap used by daily-target-state to derive the day's risk
    # budget from starting balance (e.g. 2.0 = 2% of equity per trading day).
    daily_risk_pct: float | None = 2.0
    # Fallback number of independent trade attempts the campaign expects to
    # make in a day. DailyTargetLedger's CLI/automation path first reads
    # ai-test-bot firepower evidence; this policy value is used only when that
    # observed-expectancy pace is unavailable.
    #
    # Per AGENT_CONTRACT §3.5:
    # (a) market reality: a realistic FX scalp/swing day fires 5–30 trades
    #     depending on regime, session liquidity, and pacing. 10 is a safe
    #     minimum fallback; current production pace should come from
    #     ai_test_bot_backtest.firepower.required_trades_per_day_at_observed_expectancy.
    # (b) constant rather than derived: this value is only the no-evidence
    #     operator-policy fallback. When backtest firepower is present, the
    #     ledger uses that market/history-derived pace instead.
    # (c) replace via: --target-trades-per-day on daily-target-state, or by
    #     improving ai-test-bot firepower/expectancy wiring.
    target_trades_per_day: int | None = 10
    # Sanity ceiling on the pace divisor used to derive per_trade_risk_budget_jpy.
    # ai-test-bot.firepower can return required_trades_per_day_at_observed_expectancy
    # values into the hundreds when current strategy expectancy is too thin to
    # hit the daily target at any practical pace (e.g. 229 trades/day). Dividing
    # daily_risk_budget_jpy by such a number sizes each order at ~10–20 JPY
    # worst-case loss, which floors out near broker minimum units and silently
    # makes execution operationally meaningless.
    #
    # Per AGENT_CONTRACT §3.5:
    # (a) market reality: a sustained autonomous FX scalp/swing day rarely
    #     supports more than ~30 independent risk-bounded shots before slippage,
    #     spread cost, and decision-quality degradation dominate edge.
    # (b) constant rather than derived: this is the operator's declared maximum
    #     practical attempt count, not market output. Backtest expectancy gaps
    #     are still surfaced in ai_test_bot.firepower so the operator sees the
    #     gap; the cap only prevents the gap from silently sabotaging sizing.
    # (c) replace via: pass --target-trades-per-day on daily-target-state for an
    #     explicit operator pace, or improve strategy expectancy so backtest
    #     pace falls naturally below the cap.
    max_target_trades_per_day: int | None = 30
    # Floor on per_trade_risk_budget as a fraction of starting equity.
    # (a) market reality: the per-trade slice must fund a position whose
    #     ATR-derived TP meaningfully exceeds round-trip spread, or every
    #     "win" is noise-scale. The old 0.05% floor (~100 JPY at 200k NAV)
    #     was decorative: 2026-06-11 live showed the stale-backtest pace
    #     (30 trades/day, derived from the pre-TP-fix clipped-win era)
    #     slicing a 10% daily budget into 585 JPY shots, which the SL-free
    #     sizing min() then used to cut NAV%-sized entries (~5,000u) down
    #     to 1,000u micro-lots — the exact "single-lane × micro-size"
    #     death spiral of feedback_basket_and_pace_cap.md (2026-05-06),
    #     where micro wins re-teach the backtest that 30+ micro trades are
    #     "required". At 1.0% of equity the floor funds a MIN-lot-multiple
    #     position with ATR-scale geometry, and the whole-day protection
    #     stays intact because remaining_risk_budget_jpy still decrements
    #     per shot (≈10 full-loss trades exhaust a 10% day).
    # (b) constant rather than derived: this is operator policy preventing
    #     "math break" cycles where pace × budget drives per-trade into
    #     units the broker cannot honor — see
    #     feedback_high_conviction_execution.md, feedback_attack_mode_sizing.md,
    #     and feedback_basket_and_pace_cap.md.
    # (c) replace via: improve strategy expectancy so backtest firepower
    #     pace falls naturally, or raise daily_risk_pct intentionally.
    min_per_trade_risk_pct: float | None = 1.0
    # Default reward/risk floor for non-range entries.
    # (a) market reality: trend / breakout-failure setups need their TP to clear
    #     spread + slippage by a margin that compensates for losing trades; 1.2R
    #     is the conservative floor where +EV holds at modest hit rate.
    # (b) constant rather than derived: this is operator policy on minimum
    #     trade quality. Per-regime floors override (see range_min_reward_risk).
    # (c) replace via: tune from post-trade-learning hit-rate distribution per
    #     regime if the floor proves systematically too tight or too loose.
    min_reward_risk: float = 1.2
    # Reward/risk floor when the intent's regime context is RANGE.
    # Per AGENT_CONTRACT §3.5: range regimes deserve faster rotation. Hit rate
    # in clean RANGE is materially above trend, so a lower R floor is +EV when
    # the actual move is bounded by the opposite rail. Range geometry already
    # caps TP at the opposing rail (`_range_geometry`), so this floor only
    # gates the loss/reward ratio, not the absolute target distance.
    range_min_reward_risk: float = 0.6
    # Reward/risk floor for non-trend failed-break HARVEST entries whose broker
    # TP is explicitly attached to a technical/structural target.
    # (a) market reality: failed-break and small-wave harvests are short-cycle
    #     capture trades; forcing the 1.2R runner floor makes the bot skip or
    #     over-stretch the nearest executable structural take-profit. 1.0R still
    #     requires winners to pay for one full loser before spread/slippage gates.
    # (b) constant rather than derived: this is the execution contract that
    #     matches intent_generator's attached HARVEST fallback. It is narrower
    #     than range_min_reward_risk and never applies to TREND_CONTINUATION.
    # (c) replace via: train a method/source-specific expectancy floor from
    #     post-trade HARVEST outcomes, then wire that distribution into policy.
    technical_harvest_min_reward_risk: float = 1.0
    max_quote_age_seconds: int = 20
    max_spread_multiple: float = 2.5
    min_target_spread_multiple: float = 5.0
    min_stop_spread_multiple: float = 5.0
    block_new_entries_with_open_positions: bool = True
    block_new_entries_with_external_risk: bool = True
    block_unprotected_positions: bool = True
    block_new_entries_with_pending_entry_orders: bool = True
    require_live_enabled_for_send: bool = True
    require_market_context_for_live_send: bool = True
    # Operator-set margin utilization ceiling for autonomous entries.
    # (a) market reality: OANDA rejects orders when marginAvailable cannot
    #     cover initial margin; capping marginUsed keeps the rejection in our
    #     risk gate instead of in broker-side order cancellation.
    # (b) constant rather than derived: this is the current operator policy.
    #     92% means the system may use most NAV while leaving 8% headroom for
    #     spread, slippage, and mark-to-market movement.
    # (c) replace via: pass RiskPolicy(max_margin_utilization_pct=...) from
    #     CLI/config when an operator-facing knob is introduced.
    max_margin_utilization_pct: float | None = 92.0
    require_margin_account: bool = True
    allow_protected_trader_position_adds: bool = False
    # OANDA trades without the vNext trader tag are operator-managed manual
    # exposure. They remain visible in broker truth, but the autonomous trader
    # must not protect, close, or count them against its own entry budget.
    allow_operator_managed_manual_exposure: bool = True
    # Concurrent trader-owned positions cap. Default 4 caps simultaneous
    # exposure to ~4 lanes; live env can override via `QR_MAX_PORTFOLIO_POSITIONS`
    # for attack-mode multi-lane participation (`feedback_attack_mode_sizing.md`).
    # NOTE: use `field(default_factory=...)` so the env override is honored at
    # instance construction. With a bare `= int(os.environ.get(...))` the value
    # was frozen at module import — bootstrap-time setdefault arrived too late
    # and the cap stayed at 4 even when QR_MAX_PORTFOLIO_POSITIONS=10 was set
    # (regression seen 2026-05-11 15:44 JST: 48 intents DRY_RUN_BLOCKED with
    # "open trader positions 5 reached portfolio limit 4").
    max_portfolio_positions: int = field(
        default_factory=lambda: int(os.environ.get("QR_MAX_PORTFOLIO_POSITIONS", "4") or "4")
    )
    # Same-pair concentration guard for protected-position add mode.
    # (a) market reality: the trader scans 28 FX pairs, but one trapped pair can
    #     consume position slots, margin headroom, and GPT selection priority;
    #     2026-06-01 live truth had 6 trader-owned EUR_USD positions and
    #     high-confidence GBP_USD/EUR_GBP candidates blocked by margin.
    # (b) constant rather than derived: this is an operator portfolio-shape
    #     policy. Two same-pair trader positions allow one thesis plus one
    #     controlled follow-on for ordinary adds, while the bounded adverse-add
    #     exception below covers the operator's replayable 2025 averaging shape.
    # (c) replace via: QR_MAX_SAME_PAIR_TRADER_POSITIONS, or a future
    #     expectancy-derived allocator that budgets slots by pair correlation
    #     and realized turnover.
    max_same_pair_trader_positions: int | None = field(
        default_factory=lambda: _env_optional_int("QR_MAX_SAME_PAIR_TRADER_POSITIONS", 2)
    )
    # Bounded same-side averaging slot cap.
    # (a) market reality: the 2025 manual-history replay that the campaign is
    #     trying to reproduce had profitable averaging-into-adverse clusters only
    #     after excluding >=12h / margin-closeout tails; that bounded profile had
    #     median 3 entries and max 4 entries, while unbounded and with-move
    #     stacking created the blow-up tail.
    # (b) constant rather than derived: this is an empirical replay boundary for
    #     the operator precedent, not a market threshold. Current ATR still gates
    #     the add distance separately.
    # (c) replace via: QR_MAX_BOUNDED_ADVERSE_ADD_POSITIONS, or a future
    #     post-trade cluster allocator that derives the cap per pair/regime.
    max_bounded_adverse_add_positions: int | None = field(
        default_factory=lambda: _env_optional_int("QR_MAX_BOUNDED_ADVERSE_ADD_POSITIONS", 4)
    )
    # Per-pair margin concentration cap for fresh, non-hedge adds.
    # (a) market reality: margin utilization is the binding opportunity cost;
    #     when one pair uses roughly half of NAV as initial margin, independent
    #     pairs cannot reach the 1000u production lot even if their forecast is
    #     stronger.
    # (b) constant rather than derived: 45% is a portfolio-shape reserve under
    #     the 92% total margin ceiling, forcing headroom for at least one other
    #     independent pair before a same-pair add can continue stacking.
    # (c) replace via: QR_MAX_SAME_PAIR_MARGIN_UTILIZATION_PCT, or a broker
    #     instrument-margin allocator once pair-level margin budgets are wired
    #     from account/instrument telemetry.
    max_same_pair_margin_utilization_pct: float | None = field(
        default_factory=lambda: _env_optional_float("QR_MAX_SAME_PAIR_MARGIN_UTILIZATION_PCT", 45.0)
    )
    # Bound same-pair same-side averaging into adverse movement by current
    # operating volatility, not by clock/session.
    # (a) market reality: the operator's replayable 2025 "nanpin" clusters
    #     were small adverse retests, not unlimited martingale adds; an add
    #     several current ATRs away is a new trapped leg, even if the time of
    #     day is normally liquid.
    # (b) constant rather than derived: 2x current operating ATR is the
    #     execution-policy ceiling that separates local retest/absorption from
    #     stale-loss averaging. The ATR itself is market-derived per intent.
    # (c) replace via: QR_MAX_ADVERSE_ADD_ATR_MULTIPLE, or a future
    #     expectancy-derived add allocator trained on post-trade clusters.
    max_adverse_add_atr_multiple: float | None = field(
        default_factory=lambda: _env_optional_float("QR_MAX_ADVERSE_ADD_ATR_MULTIPLE", 2.0)
    )
    max_portfolio_loss_jpy: float | None = None


class RiskEngine:
    def __init__(
        self,
        *,
        policy: RiskPolicy | None = None,
        specs: dict[str, InstrumentSpec] | None = None,
        live_enabled: bool = False,
        validation_time_utc: datetime | None = None,
    ) -> None:
        self.policy = policy or RiskPolicy()
        self.specs = specs or DEFAULT_SPECS
        self.live_enabled = live_enabled
        self.validation_time_utc = (
            validation_time_utc.astimezone(timezone.utc)
            if validation_time_utc is not None
            else None
        )

    def _now(self) -> datetime:
        return self.validation_time_utc or datetime.now(timezone.utc)

    def validate(self, intent: OrderIntent, snapshot: BrokerSnapshot, *, for_live_send: bool = False) -> RiskDecision:
        issues: list[RiskIssue] = []
        spec = self._spec(intent.pair)
        quote = snapshot.quotes.get(intent.pair)

        if for_live_send and self.policy.require_live_enabled_for_send and not self.live_enabled:
            issues.append(RiskIssue("LIVE_DISABLED", "live execution is disabled; dry-run only"))
        if for_live_send:
            issues.extend(
                RiskIssue(
                    str(item.get("code") or "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY"),
                    str(item.get("message") or "unresolved guardian receipt issue blocks normal new entries"),
                )
                for item in guardian_receipt_new_entry_blockers_from_paths()
            )

        if intent.owner != Owner.TRADER:
            issues.append(RiskIssue("OWNER_NOT_TRADER", f"order owner must be trader, got {intent.owner.value}"))
        if intent.units <= 0:
            issues.append(RiskIssue("BAD_UNITS", f"units must be positive, got {intent.units}"))
        # Fix C (2026-05-12) defense-in-depth: even if intent_generator's
        # Fix B path is bypassed (manual stage-live-order, replayed legacy
        # receipt, ad-hoc CLI scripts), the gateway must refuse sub-floor
        # lots. Sub-MIN_PRODUCTION_LOT_UNITS lots cannot capture a pip
        # target larger than the broker spread on the round trip, so the
        # trade is structurally unprofitable. `QR_ALLOW_TEST_MICRO_LOT=1`
        # disables this for fixtures that intentionally exercise micro
        # sizes.
        if (
            0 < abs(int(intent.units)) < MIN_PRODUCTION_LOT_UNITS
            and not _min_lot_test_override_active()
        ):
            issues.append(
                RiskIssue(
                    "MIN_LOT_VIOLATION",
                    f"order size {abs(int(intent.units))}u is below the "
                    f"{MIN_PRODUCTION_LOT_UNITS}u production floor; round-trip "
                    "spread cost would dominate any realistic pip target.",
                )
            )
        if not intent.thesis.strip():
            issues.append(RiskIssue("MISSING_THESIS", "order intent must carry a non-empty thesis"))
        issues.extend(_hedge_metadata_issues(intent))
        issues.extend(_hedge_balance_issues(intent, snapshot))
        issues.extend(_forecast_unselected_projection_conflict_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_executable_live_readiness_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_range_method_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_range_confidence_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_watch_only_live_send_issues(intent, for_live_send=for_live_send))
        issues.extend(self._market_context_issues(intent, for_live_send=for_live_send))
        operator_manual_block = operator_manual_jpy_add_block_issue(intent, snapshot)
        if operator_manual_block is not None:
            issues.append(RiskIssue(operator_manual_block["code"], operator_manual_block["message"]))
        operator_manual_same_theme_block = operator_manual_same_theme_add_block_issue(intent, snapshot)
        if operator_manual_same_theme_block is not None:
            issues.append(
                RiskIssue(
                    operator_manual_same_theme_block["code"],
                    operator_manual_same_theme_block["message"],
                )
            )

        entry_relevant_positions = self._entry_relevant_positions(snapshot)
        portfolio_add_mode = self.policy.allow_protected_trader_position_adds
        if self.policy.block_new_entries_with_open_positions and not portfolio_add_mode:
            for position in entry_relevant_positions:
                issues.append(
                    RiskIssue(
                        "OPEN_POSITION_EXISTS",
                        f"open broker position blocks fresh entry: {position.pair} {position.side.value} "
                        f"id={position.trade_id} {position.units}u; manage exposure before adding risk",
                    )
                )
        elif self.policy.block_new_entries_with_open_positions and portfolio_add_mode:
            if len(entry_relevant_positions) >= self.policy.max_portfolio_positions:
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_POSITION_LIMIT",
                        f"open trader positions {len(entry_relevant_positions)} reached portfolio limit "
                        f"{self.policy.max_portfolio_positions}",
                    )
                )
            for position in entry_relevant_positions:
                # SL-free regime: trader-owned SL=None is intentional, and
                # TP-less positions are allowed as no-broker-TP runners unless
                # explicit missing-TP repair is enabled. Margin, hedging, and
                # portfolio caps remain the executable add gates.
                pos_eligible = _layerable_trader_position(position)
                if not pos_eligible:
                    issues.append(
                        RiskIssue(
                            "OPEN_POSITION_EXISTS",
                            f"only protected trader-owned positions can be layered; "
                            f"{position.pair} {position.side.value} id={position.trade_id} is not eligible",
                        )
                    )
                elif position.pair == intent.pair and position.side != intent.side and (
                    not _account_hedging_enabled(snapshot) or not _intent_declares_hedge(intent)
                ):
                    if _account_hedging_enabled(snapshot) and _candidate_adds_to_trader_net_side(intent, snapshot):
                        continue
                    issues.append(
                        RiskIssue(
                            "OPPOSING_POSITION_NEEDS_HEDGING",
                            f"fresh {intent.pair} {intent.side.value} entry opposes protected "
                            f"{position.pair} {position.side.value} id={position.trade_id}; "
                            "opposite-side adds require both broker hedging proof and explicit "
                            "intent.metadata['position_intent']='HEDGE'",
                        )
                    )

        if self.policy.block_new_entries_with_external_risk:
            for position in entry_relevant_positions:
                if position.owner != Owner.TRADER:
                    issues.append(
                        RiskIssue(
                            "EXTERNAL_RISK_OPEN",
                            f"external/manual risk is open: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u; adopt or close before new entries",
                        )
                    )

        if self.policy.block_unprotected_positions:
            sl_free_active = _trader_sl_repair_disabled()
            for position in entry_relevant_positions:
                missing = []
                if position.take_profit is None:
                    if _trader_no_broker_tp_runner(position):
                        issues.append(
                            RiskIssue(
                                "TP_LESS_RUNNER_OPEN",
                                f"open trader SL-free runner has no broker TP: {position.pair} "
                                f"{position.side.value} id={position.trade_id} {position.units}u",
                                severity="WARN",
                            )
                        )
                    else:
                        missing.append("TP")
                if position.stop_loss is None:
                    if sl_free_active and position.owner == Owner.TRADER:
                        # User directive 「SLいらない」 — trader-owned SL=None is
                        # deliberate and does not block fresh entries. TP-only.
                        pass
                    else:
                        missing.append("SL")
                if missing:
                    issues.append(
                        RiskIssue(
                            "UNPROTECTED_POSITION",
                            f"open position lacks {'/'.join(missing)}: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u",
                        )
                    )

        if self.policy.block_new_entries_with_pending_entry_orders:
            for order in snapshot.orders:
                if _is_pending_entry_order(order) and not _is_operator_managed_manual_order(order):
                    issues.append(
                        RiskIssue(
                            "PENDING_ENTRY_ORDER_OPEN",
                            f"pending entry order is already open: {order.pair or '(unknown)'} "
                            f"{order.order_type} id={order.order_id}; resolve it before new entries",
                        )
                    )

        if quote is None:
            issues.append(RiskIssue("MISSING_QUOTE", f"missing live quote for {intent.pair}"))
            return RiskDecision(False, None, tuple(issues))

        quote_age = max(0.0, (self._now() - quote.timestamp_utc).total_seconds())
        if quote_age > self.policy.max_quote_age_seconds:
            issues.append(
                RiskIssue(
                    "STALE_QUOTE",
                    f"{intent.pair} quote is stale: {quote_age:.1f}s > {self.policy.max_quote_age_seconds}s",
                )
            )

        spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
        # I (2026-05-13) — session-aware spread tolerance per
        # AGENT_CONTRACT §3.5 "spread tolerance must be liquidity-
        # derived". Read the session tag from intent.metadata
        # (producer: intent_generator._chart_context_for). Multipliers
        # are operator-tuned per session liquidity tier; the policy
        # `max_spread_multiple` remains the anchor.
        session_mult = _spread_session_multiplier(intent)
        effective_spread_cap_mult = self.policy.max_spread_multiple * session_mult
        if spread_pips > spec.normal_spread_pips * effective_spread_cap_mult:
            issues.append(
                RiskIssue(
                    "SPREAD_TOO_WIDE",
                    f"{intent.pair} spread {spread_pips:.1f}pip exceeds "
                    f"{effective_spread_cap_mult:.2f}x normal {spec.normal_spread_pips:.1f}pip "
                    f"(policy={self.policy.max_spread_multiple:.1f}x, session_mult={session_mult:.2f})",
                )
            )

        issues.extend(self._entry_contract_issues(intent, quote, spec, spread_pips, for_live_send=for_live_send))
        entry_price = self._entry_price(intent, quote)
        issues.extend(self._same_pair_position_concentration_issues(intent, entry_relevant_positions, entry_price, spec))
        issues.extend(self._same_pair_adverse_add_issues(intent, snapshot, entry_price, spec))
        issues.extend(self._conversion_quote_issues(intent.pair, snapshot))
        quote_to_jpy = self._quote_to_jpy(intent.pair, snapshot)
        if quote_to_jpy is None:
            return RiskDecision(False, None, tuple(issues))

        issues.extend(self._same_pair_margin_concentration_issues(intent, snapshot, entry_price, quote_to_jpy, spec))
        metrics = self._metrics(intent, quote, spec, entry_price, quote_to_jpy, snapshot)
        issues.extend(self._margin_issues(snapshot, metrics))
        issues.extend(_target_path_guard_issues(intent, for_live_send=for_live_send))
        if metrics.loss_pips <= 0:
            issues.append(RiskIssue("SL_NOT_LOSS_SIDE", f"SL is not on the loss side for {intent.side.value}"))
        if metrics.reward_pips <= 0:
            issues.append(RiskIssue("TP_NOT_REWARD_SIDE", f"TP is not on the reward side for {intent.side.value}"))
        loss_cap = self._resolved_loss_cap(intent)
        if loss_cap is None:
            issues.append(
                RiskIssue(
                    "LOSS_CAP_MISSING",
                    "no per-trade loss cap available: pass policy.max_loss_jpy or "
                    "intent.metadata['max_loss_jpy'] (equity-derived); refusing to validate without one.",
                )
            )
        elif metrics.risk_jpy > loss_cap:
            issues.append(
                RiskIssue(
                    "LOSS_CAP_EXCEEDED",
                    f"planned worst-case loss {metrics.risk_jpy:.0f} JPY exceeds cap {loss_cap:.0f} JPY",
                )
            )
        issues.extend(_loss_asymmetry_guard_issues(intent, metrics))
        # Regime/mode-derived reward/risk floor. RANGE regimes and explicit
        # recovery hedges are allowed below the default min_reward_risk because
        # rotation geometry caps TP at the opposing rail, and recovery hedges
        # monetize a trapped opposite leg instead of opening a fresh one-way
        # thesis. Failed-break technical HARVEST is a separate short-cycle
        # capture mode: it may use the 1R HARVEST floor, but trend continuation
        # keeps the default/higher trend payoff gates.
        # Falls back to default min when regime is missing/unclear so silent
        # data gaps cannot relax the floor (per AGENT_CONTRACT §3.5).
        regime_state = ""
        if intent.metadata:
            raw_state = intent.metadata.get("regime_state")
            if isinstance(raw_state, str):
                regime_state = raw_state.upper()
        if _intent_declares_recovery_hedge(intent) or _uses_range_reward_floor(intent, regime_state):
            active_min_rr = self.policy.range_min_reward_risk
            floor_label = "range/recovery"
        elif _uses_technical_harvest_reward_floor(intent):
            active_min_rr = self.policy.technical_harvest_min_reward_risk
            floor_label = "technical_harvest"
        else:
            active_min_rr = self.policy.min_reward_risk
            floor_label = "default"
        if metrics.reward_risk < active_min_rr:
            issues.append(
                RiskIssue(
                    "REWARD_RISK_TOO_LOW",
                    f"planned reward/risk {metrics.reward_risk:.2f}x is below {active_min_rr:.2f}x"
                    + (f" (regime={regime_state})" if regime_state else ""),
                )
            )
        elif floor_label == "technical_harvest" and metrics.reward_risk < self.policy.min_reward_risk:
            issues.append(
                RiskIssue(
                    "TECHNICAL_HARVEST_REWARD_RISK_FLOOR",
                    f"technical HARVEST uses {active_min_rr:.2f}x floor instead of "
                    f"default {self.policy.min_reward_risk:.2f}x; target remains "
                    f"{metrics.reward_pips:.1f}pip with spread {spread_pips:.1f}pip",
                    severity="WARN",
                )
            )
        countertrend_low_rr_issue = _range_countertrend_low_rr_issue(intent, metrics, self.policy)
        if countertrend_low_rr_issue is not None:
            issues.append(countertrend_low_rr_issue)
        if _below_spread_floor(metrics.reward_pips, spread_pips * self.policy.min_target_spread_multiple):
            issues.append(
                RiskIssue(
                    "TARGET_TOO_THIN_FOR_SPREAD",
                    f"target {metrics.reward_pips:.1f}pip is less than "
                    f"{self.policy.min_target_spread_multiple:.1f}x spread {spread_pips:.1f}pip",
                )
            )
        if _below_spread_floor(metrics.loss_pips, spread_pips * self.policy.min_stop_spread_multiple):
            issues.append(
                RiskIssue(
                    "STOP_TOO_THIN_FOR_SPREAD",
                    f"stop {metrics.loss_pips:.1f}pip is less than "
                    f"{self.policy.min_stop_spread_multiple:.1f}x spread {spread_pips:.1f}pip",
                )
            )
        issues.extend(
            self._forecast_direction_issues(
                intent,
                for_live_send=for_live_send,
            )
        )
        issues.extend(
            self._forecast_directional_live_readiness_issues(
                intent,
                for_live_send=for_live_send,
            )
        )
        issues.extend(
            self._forecast_geometry_issues(
                intent,
                entry_price=entry_price,
                spec=spec,
                spread_pips=spread_pips,
                for_live_send=for_live_send,
            )
        )
        if portfolio_add_mode and self.policy.max_portfolio_loss_jpy is not None:
            portfolio_risk, risk_issue = self._open_portfolio_risk_jpy(snapshot)
            if risk_issue:
                issues.append(risk_issue)
            elif portfolio_risk + metrics.risk_jpy > self.policy.max_portfolio_loss_jpy:
                # Under SL-free the per-day loss budget is advisory only —
                # margin-utilization is the real ceiling, not a JPY literal.
                # User 2026-05-08「市況>リスク」+ `feedback_offense_sizing.md`
                # 「loss cap撤廃」+ `feedback_market_over_risk_budget.md`
                # 「含み損%/JPYは判断材料にしない」.
                # Surface the gate as WARN under SL-free so the operator
                # sees the exposure but the cycle isn't blocked.
                severity = "WARN" if _trader_sl_repair_disabled() else "BLOCK"
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_LOSS_CAP_EXCEEDED",
                        f"open risk {portfolio_risk:.0f} JPY + candidate risk {metrics.risk_jpy:.0f} JPY "
                        f"exceeds portfolio cap {self.policy.max_portfolio_loss_jpy:.0f} JPY",
                        severity=severity,
                    )
                )

        return RiskDecision(
            allowed=not any(issue.severity == "BLOCK" for issue in issues),
            metrics=metrics,
            issues=tuple(issues),
        )

    def _same_pair_position_concentration_issues(
        self,
        intent: OrderIntent,
        entry_relevant_positions: tuple[BrokerPosition, ...],
        entry_price: float,
        spec: InstrumentSpec,
    ) -> list[RiskIssue]:
        cap = self.policy.max_same_pair_trader_positions
        if cap is None or _intent_declares_hedge(intent):
            return []
        if cap <= 0:
            return [
                RiskIssue(
                    "INVALID_PAIR_CONCENTRATION_POLICY",
                    f"max_same_pair_trader_positions must be positive or None, got {cap}",
                )
            ]
        same_pair_positions = tuple(
            position
            for position in entry_relevant_positions
            if position.owner == Owner.TRADER and position.pair == intent.pair
        )
        same_pair_count = len(same_pair_positions)
        if same_pair_count < cap:
            return []

        bounded_cap = self.policy.max_bounded_adverse_add_positions
        if bounded_cap is not None and bounded_cap <= 0:
            return [
                RiskIssue(
                    "INVALID_BOUNDED_ADVERSE_ADD_POLICY",
                    f"max_bounded_adverse_add_positions must be positive or None, got {bounded_cap}",
                )
            ]
        same_side_positions = tuple(position for position in same_pair_positions if position.side == intent.side)
        all_same_side = len(same_side_positions) == same_pair_count
        if (
            bounded_cap is not None
            and bounded_cap > cap
            and same_pair_count < bounded_cap
            and all_same_side
        ):
            adverse_pips = _same_pair_adverse_add_pips(intent.side, same_side_positions, entry_price, spec)
            if adverse_pips is not None and adverse_pips > 0.0:
                # Let the dedicated adverse-add gate below enforce explicit
                # classification and current-ATR distance. This exception only
                # keeps the ordinary two-position cap from blocking the manual
                # replay's bounded 3rd/4th averaging entries.
                return []

        if bounded_cap is not None and all_same_side:
            adverse_pips = _same_pair_adverse_add_pips(intent.side, same_side_positions, entry_price, spec)
            if adverse_pips is not None and adverse_pips > 0.0 and same_pair_count >= bounded_cap:
                return [
                    RiskIssue(
                        "PAIR_CONCENTRATION_LIMIT",
                        f"open trader {intent.pair} positions {same_pair_count} reached bounded adverse-add "
                        f"cap {bounded_cap}; 2025 manual replay supports small averaging only up to that "
                        "entry count, so wait for TP/position-management before adding same-pair risk",
                    )
                ]
        return [
            RiskIssue(
                "PAIR_CONCENTRATION_LIMIT",
                f"open trader {intent.pair} positions {same_pair_count} reached same-pair cap {cap}; "
                "wait for TP/position-management or emit an explicit HEDGE intent before adding same-pair risk",
            )
        ]

    def _same_pair_margin_concentration_issues(
        self,
        intent: OrderIntent,
        snapshot: BrokerSnapshot,
        entry_price: float,
        quote_to_jpy: float,
        spec: InstrumentSpec,
    ) -> list[RiskIssue]:
        cap_pct = self.policy.max_same_pair_margin_utilization_pct
        if cap_pct is None or _intent_declares_hedge(intent):
            return []
        if cap_pct <= 0 or cap_pct > 100:
            return [
                RiskIssue(
                    "INVALID_PAIR_MARGIN_POLICY",
                    f"max_same_pair_margin_utilization_pct must be within 0-100 or None, got {cap_pct}",
                )
            ]
        account = snapshot.account
        if account is None or account.nav_jpy <= 0:
            return []

        long_units, short_units = _same_pair_position_units(snapshot, intent.pair, owner=Owner.TRADER)
        candidate_units = max(0, abs(int(intent.units)))
        if intent.side == Side.LONG:
            long_units += candidate_units
        else:
            short_units += candidate_units
        pair_margin_units = max(long_units, short_units)
        if pair_margin_units <= 0:
            return []
        estimated_pair_margin = estimate_required_margin_jpy(
            units=pair_margin_units,
            entry_price=entry_price,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
        )
        cap_jpy = account.nav_jpy * (cap_pct / 100.0)
        if estimated_pair_margin <= cap_jpy:
            return []
        return [
            RiskIssue(
                "PAIR_MARGIN_CONCENTRATION_LIMIT",
                f"{intent.pair} trader margin after candidate would be {estimated_pair_margin:.0f} JPY "
                f"({estimated_pair_margin / account.nav_jpy * 100.0:.1f}% NAV), above same-pair cap "
                f"{cap_pct:.1f}% ({cap_jpy:.0f} JPY); wait for that pair to harvest or route an explicit HEDGE",
            )
        ]

    def _same_pair_adverse_add_issues(
        self,
        intent: OrderIntent,
        snapshot: BrokerSnapshot,
        entry_price: float,
        spec: InstrumentSpec,
    ) -> list[RiskIssue]:
        cap_mult = self.policy.max_adverse_add_atr_multiple
        if cap_mult is None or _intent_declares_hedge(intent):
            return []
        if cap_mult <= 0:
            return [
                RiskIssue(
                    "INVALID_ADVERSE_ADD_POLICY",
                    f"max_adverse_add_atr_multiple must be positive or None, got {cap_mult}",
                )
            ]

        same_side_positions = tuple(
            position
            for position in snapshot.positions
            if (
                position.owner == Owner.TRADER
                and position.pair == intent.pair
                and position.side == intent.side
            )
        )
        if not same_side_positions:
            return []

        adverse_pips = _same_pair_adverse_add_pips(intent.side, same_side_positions, entry_price, spec)
        if adverse_pips is None:
            return []
        if adverse_pips <= 0.0:
            return []

        metadata = intent.metadata or {}
        add_type = str(metadata.get("same_pair_add_type") or "").upper().strip()
        if add_type != "AVERAGE_INTO_ADVERSE":
            return [
                RiskIssue(
                    "ADVERSE_ADD_CLASSIFICATION_MISSING",
                    f"{intent.pair} {intent.side.value} adds {adverse_pips:.1f}pip into adverse "
                    "same-pair exposure, but intent metadata does not classify it as "
                    "same_pair_add_type=AVERAGE_INTO_ADVERSE; refuse unclassified averaging.",
                )
            ]

        atr_pips = _to_float(metadata.get("tp_atr_pips"))
        if atr_pips is None or atr_pips <= 0:
            return [
                RiskIssue(
                    "ADVERSE_ADD_ATR_MISSING",
                    f"{intent.pair} {intent.side.value} adverse same-pair add needs current "
                    "tp_atr_pips metadata so the add is bounded by market volatility, not by time/session.",
                )
            ]

        cap_pips = atr_pips * cap_mult
        if adverse_pips <= cap_pips:
            return []
        return [
            RiskIssue(
                "ADVERSE_ADD_DISTANCE_TOO_WIDE",
                f"{intent.pair} {intent.side.value} adverse same-pair add is {adverse_pips:.1f}pip "
                f"from trader avg entry, above {cap_mult:.1f}x current ATR cap "
                f"({cap_pips:.1f}pip from tp_atr_pips={atr_pips:.1f}); this is stale-loss averaging, "
                "not bounded retest/nanpin.",
            )
        ]

    def _entry_relevant_positions(self, snapshot: BrokerSnapshot) -> tuple[BrokerPosition, ...]:
        if not self.policy.allow_operator_managed_manual_exposure:
            return tuple(snapshot.positions)
        return tuple(position for position in snapshot.positions if not _is_operator_managed_manual(position))

    def _spec(self, pair: str) -> InstrumentSpec:
        try:
            return self.specs[pair]
        except KeyError as exc:
            raise ValueError(f"unsupported instrument: {pair}") from exc

    def _resolved_loss_cap(self, intent: OrderIntent) -> float | None:
        """Return the per-trade loss cap to enforce.

        Resolution order (no JPY literal fallback):
            1. intent.metadata['max_loss_jpy'] — caller (intent generator) injected an
               equity-derived cap for this specific lane.
            2. policy.max_loss_jpy — explicit policy-wide cap from CLI / config.
            3. None — validator emits LOSS_CAP_MISSING and refuses the trade.
        """
        meta = intent.metadata or {}
        cap = meta.get("max_loss_jpy")
        if cap is not None:
            try:
                cap_value = float(cap)
            except (TypeError, ValueError):
                return None
            return cap_value if cap_value > 0 else None
        if self.policy.max_loss_jpy is not None and self.policy.max_loss_jpy > 0:
            return float(self.policy.max_loss_jpy)
        return None

    def _entry_price(self, intent: OrderIntent, quote: Quote) -> float:
        if intent.order_type == OrderType.MARKET:
            return quote.ask if intent.side == Side.LONG else quote.bid
        if intent.entry is not None:
            return float(intent.entry)
        return quote.ask if intent.side == Side.LONG else quote.bid

    def _entry_contract_issues(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        spread_pips: float,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        if intent.order_type == OrderType.MARKET:
            return self._market_entry_issues(intent, quote, spec, spread_pips, for_live_send=for_live_send)
        if intent.entry is None:
            return [RiskIssue("PENDING_ENTRY_REQUIRES_ENTRY", f"{intent.order_type.value} requires an entry price")]
        entry = float(intent.entry)
        issues: list[RiskIssue] = []
        if intent.order_type == OrderType.STOP_ENTRY:
            if intent.side == Side.LONG and entry <= quote.ask:
                issues.append(
                    RiskIssue(
                        "STOP_ENTRY_NOT_ABOVE_MARKET",
                        f"LONG stop-entry must be above current ask: entry={entry} ask={quote.ask}",
                    )
                )
            if intent.side == Side.SHORT and entry >= quote.bid:
                issues.append(
                    RiskIssue(
                        "STOP_ENTRY_NOT_BELOW_MARKET",
                        f"SHORT stop-entry must be below current bid: entry={entry} bid={quote.bid}",
                    )
                )
        elif intent.order_type == OrderType.LIMIT:
            if intent.side == Side.LONG and entry >= quote.ask:
                issues.append(
                    RiskIssue(
                        "LIMIT_ENTRY_NOT_BELOW_MARKET",
                        f"LONG limit must be below current ask: entry={entry} ask={quote.ask}",
                    )
                )
            if intent.side == Side.SHORT and entry <= quote.bid:
                issues.append(
                    RiskIssue(
                        "LIMIT_ENTRY_NOT_ABOVE_MARKET",
                        f"SHORT limit must be above current bid: entry={entry} bid={quote.bid}",
                    )
                )
        return issues

    def _market_entry_issues(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        spread_pips: float,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        if intent.entry is None:
            return []
        expected = float(intent.entry)
        executable = quote.ask if intent.side == Side.LONG else quote.bid
        drift_pips = abs(expected - executable) * spec.pip_factor
        allowed_drift = max(spread_pips * 2.0, 1.0)
        if drift_pips <= allowed_drift:
            return []
        severity = "BLOCK" if for_live_send else "WARN"
        return [
            RiskIssue(
                "MARKET_ENTRY_DRIFT",
                f"MARKET expected entry is stale versus broker quote: expected={expected} "
                f"executable={executable} drift={drift_pips:.1f}pip > {allowed_drift:.1f}pip",
                severity=severity,
            )
        ]

    def _forecast_direction_issues(
        self,
        intent: OrderIntent,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        metadata = intent.metadata or {}
        direction = str(metadata.get("forecast_direction") or "").upper()
        if direction not in {"UP", "DOWN"}:
            return []
        forecast_side = Side.LONG if direction == "UP" else Side.SHORT
        if intent.side == forecast_side:
            return []
        if _bidask_replay_precision_support_for_intent(intent) is not None:
            return []
        confidence = _to_float(metadata.get("forecast_confidence"))
        min_confidence = FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE
        support = _forecast_market_support(metadata)
        unsupported_weak_forecast = (
            confidence is None
            or confidence < min_confidence
            or _forecast_directional_bucket_is_known_weak(metadata, support)
            or not _forecast_directional_bucket_clears_live_precision(metadata, support)
        )
        if unsupported_weak_forecast and not _forecast_supported_opposite_side_blocks(
            metadata,
            forecast_side=forecast_side,
            min_confidence=min_confidence,
        ):
            if not for_live_send:
                return []
            if (
                _forecast_directional_bucket_is_known_weak(metadata, support)
                or not _forecast_directional_bucket_clears_live_precision(metadata, support)
            ):
                return [
                    _forecast_directional_bucket_issue(
                        intent,
                        direction=direction,
                        metadata=metadata,
                        support=support,
                    )
                ]
            return [
                _forecast_confidence_required_issue(
                    intent,
                    direction=direction,
                    confidence=confidence,
                    min_confidence=min_confidence,
                )
            ]
        target = _to_float(metadata.get("forecast_target_price"))
        invalidation = _to_float(metadata.get("forecast_invalidation_price"))
        details = [f"forecast {direction}"]
        if confidence is not None:
            details.append(f"conf={confidence:.2f}")
        if target is not None:
            details.append(f"target={target:.5f}")
        if invalidation is not None:
            details.append(f"invalidation={invalidation:.5f}")
        severity = "BLOCK" if for_live_send else "WARN"
        return [
            RiskIssue(
                "FORECAST_DIRECTION_CONFLICT",
                f"{intent.pair} {intent.side.value} conflicts with current directional forecast "
                f"({', '.join(details)}); only {forecast_side.value} may be sent while this "
                "forecast is fresh.",
                severity=severity,
            )
        ]

    def _forecast_directional_live_readiness_issues(
        self,
        intent: OrderIntent,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        if not for_live_send:
            return []
        metadata = intent.metadata or {}
        direction = str(metadata.get("forecast_direction") or "").upper()
        if direction not in {"UP", "DOWN"}:
            return []
        forecast_side = Side.LONG if direction == "UP" else Side.SHORT
        if intent.side != forecast_side:
            return []
        support = _forecast_market_support(metadata)
        negative_issue = _technical_harvest_negative_precision_issue_for_intent(intent)
        if negative_issue is not None:
            return [negative_issue]
        negative_issue = _bidask_replay_negative_precision_issue_for_intent(intent)
        if negative_issue is not None:
            return [negative_issue]
        if _technical_harvest_precision_support_for_intent(intent) is not None:
            return []
        if _bidask_replay_precision_support_for_intent(intent) is not None:
            return []
        if _forecast_selected_direction_has_audited_support(
            metadata,
            support,
            direction=direction,
        ):
            return []
        confidence = _to_float(metadata.get("forecast_confidence"))
        min_confidence = FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE
        if confidence is None or confidence < min_confidence:
            return [
                _forecast_confidence_required_issue(
                    intent,
                    direction=direction,
                    confidence=confidence,
                    min_confidence=min_confidence,
                )
            ]
        if not _forecast_directional_bucket_clears_live_precision(metadata, support):
            return [
                _forecast_directional_bucket_issue(
                    intent,
                    direction=direction,
                    metadata=metadata,
                    support=support,
                )
            ]
        return []

    def _forecast_geometry_issues(
        self,
        intent: OrderIntent,
        *,
        entry_price: float,
        spec: InstrumentSpec,
        spread_pips: float,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        metadata = intent.metadata or {}
        direction = str(metadata.get("forecast_direction") or "").upper()
        if direction not in {"UP", "DOWN"}:
            return []
        forecast_side = Side.LONG if direction == "UP" else Side.SHORT
        if intent.side != forecast_side:
            return []

        severity = "BLOCK" if for_live_send else "WARN"
        issues: list[RiskIssue] = []
        target_floor = spread_pips * self.policy.min_target_spread_multiple
        invalidation_floor = spread_pips * self.policy.min_stop_spread_multiple

        target_price = _to_float(metadata.get("forecast_target_price"))
        if target_price is not None and target_price > 0.0:
            if intent.side == Side.LONG:
                target_pips = (target_price - entry_price) * spec.pip_factor
            else:
                target_pips = (entry_price - target_price) * spec.pip_factor
            if target_pips <= 0.0:
                issues.append(
                    RiskIssue(
                        "FORECAST_TARGET_NOT_REWARD_SIDE",
                        f"{intent.pair} {intent.side.value} forecast {direction} target "
                        f"{target_price:.5f} is not on the reward side of entry {entry_price:.5f}; "
                        "refresh the forecast/entry geometry before live send.",
                        severity=severity,
                    )
                )
            elif _below_spread_floor(target_pips, target_floor):
                issues.append(
                    RiskIssue(
                        "FORECAST_TARGET_TOO_THIN_FOR_SPREAD",
                        f"{intent.pair} {intent.side.value} forecast {direction} target is only "
                        f"{target_pips:.1f}pip from entry, below "
                        f"{self.policy.min_target_spread_multiple:.1f}x spread floor "
                        f"({target_floor:.1f}pip from spread {spread_pips:.1f}pip); "
                        "prediction edge is inside execution noise.",
                        severity=severity,
                    )
                )

        invalidation_price = _to_float(metadata.get("forecast_invalidation_price"))
        if invalidation_price is not None and invalidation_price > 0.0:
            if intent.side == Side.LONG:
                invalidation_pips = (entry_price - invalidation_price) * spec.pip_factor
            else:
                invalidation_pips = (invalidation_price - entry_price) * spec.pip_factor
            if invalidation_pips <= 0.0:
                issues.append(
                    RiskIssue(
                        "FORECAST_INVALIDATION_NOT_LOSS_SIDE",
                        f"{intent.pair} {intent.side.value} forecast {direction} invalidation "
                        f"{invalidation_price:.5f} is not on the loss side of entry {entry_price:.5f}; "
                        "refresh the forecast/entry geometry before live send.",
                        severity=severity,
                    )
                )
            elif _below_spread_floor(invalidation_pips, invalidation_floor):
                issues.append(
                    RiskIssue(
                        "FORECAST_INVALIDATION_TOO_THIN_FOR_SPREAD",
                        f"{intent.pair} {intent.side.value} forecast {direction} invalidation is only "
                        f"{invalidation_pips:.1f}pip from entry, below "
                        f"{self.policy.min_stop_spread_multiple:.1f}x spread floor "
                        f"({invalidation_floor:.1f}pip from spread {spread_pips:.1f}pip); "
                        "the thesis can be invalidated by normal execution noise.",
                        severity=severity,
                    )
                )
        return issues

    def _market_context_issues(self, intent: OrderIntent, *, for_live_send: bool) -> list[RiskIssue]:
        severity = "BLOCK" if for_live_send and self.policy.require_market_context_for_live_send else "WARN"
        context = intent.market_context
        if context is None:
            return [
                RiskIssue(
                    "MISSING_MARKET_CONTEXT",
                    "order intent must state market regime, narrative, chart story, method, and invalidation",
                    severity=severity,
                )
            ]
        issues: list[RiskIssue] = []
        missing = [
            name
            for name, value in (
                ("regime", context.regime),
                ("narrative", context.narrative),
                ("chart_story", context.chart_story),
                ("invalidation", context.invalidation),
            )
            if not value.strip()
        ]
        if missing:
            issues.append(
                RiskIssue(
                    "INCOMPLETE_MARKET_CONTEXT",
                    f"market context is missing {', '.join(missing)}",
                    severity=severity,
                )
            )
        method_issue = self._method_regime_issue(intent, severity)
        if method_issue:
            issues.append(method_issue)
        return issues

    def _method_regime_issue(self, intent: OrderIntent, severity: str) -> RiskIssue | None:
        context = intent.market_context
        if context is None:
            return None
        regime_text = f"{context.regime} {context.chart_story} {context.narrative}".upper()
        method = context.method
        if method == TradeMethod.RANGE_ROTATION and _contains_any(regime_text, ("TREND", "IMPULSE", "BAND WALK")):
            if not _contains_any(regime_text, ("RANGE", "BOX", "RAIL", "ROTATION")):
                return RiskIssue(
                    "METHOD_REGIME_MISMATCH",
                    "range rotation method needs a range/box/rail story, not a one-way trend or impulse",
                    severity=severity,
                )
        if method == TradeMethod.TREND_CONTINUATION and not _contains_any(
            regime_text, ("TREND", "CONTINUATION", "STAIRCASE", "BAND WALK", "LADDER", "BREAKOUT")
        ):
            return RiskIssue(
                "METHOD_REGIME_MISMATCH",
                "trend continuation method needs a trend/continuation chart story",
                severity=severity,
            )
        if method == TradeMethod.BREAKOUT_FAILURE and not _contains_any(
            regime_text, ("FAIL", "REJECT", "RETEST", "RECLAIM", "TRAP", "BREAK")
        ):
            return RiskIssue(
                "METHOD_REGIME_MISMATCH",
                "breakout-failure method needs a failed break, rejection, retest, reclaim, or trapped-side story",
                severity=severity,
            )
        return None

    def _metrics(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        entry_price: float,
        quote_to_jpy: float,
        snapshot: BrokerSnapshot,
    ) -> RiskMetrics:
        if intent.side == Side.LONG:
            loss_pips = (entry_price - intent.sl) * spec.pip_factor
            reward_pips = (intent.tp - entry_price) * spec.pip_factor
        else:
            loss_pips = (intent.sl - entry_price) * spec.pip_factor
            reward_pips = (entry_price - intent.tp) * spec.pip_factor
        spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
        jpy_per_pip = (intent.units / spec.pip_factor) * quote_to_jpy
        risk_jpy = max(0.0, loss_pips) * jpy_per_pip
        reward_jpy = max(0.0, reward_pips) * jpy_per_pip
        # Reward/risk is geometry, not size. When upstream sizing correctly
        # resolves to 0 units because margin cannot fund the production lot,
        # risk_jpy/reward_jpy are both zero; reporting RR as 0 then creates a
        # misleading secondary `REWARD_RISK_TOO_LOW` blocker. Use the pip
        # geometry so diagnostics can distinguish "cannot size" from "bad RR".
        reward_risk = max(0.0, reward_pips) / loss_pips if loss_pips > 0 else 0.0
        estimated_margin = estimate_incremental_margin_jpy(
            pair=intent.pair,
            side=intent.side,
            units=intent.units,
            entry_price=entry_price,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
            snapshot=snapshot,
            position_intent=str((intent.metadata or {}).get("position_intent") or ""),
        )
        account = snapshot.account
        max_margin_pct = self.policy.max_margin_utilization_pct
        margin_budget = None
        margin_after_utilization = None
        margin_used = None
        margin_available = None
        if account is not None:
            margin_used = account.margin_used_jpy
            margin_available = account.margin_available_jpy
            if max_margin_pct is not None and account.nav_jpy > 0:
                margin_budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
                margin_after_utilization = (account.margin_used_jpy + estimated_margin) / account.nav_jpy * 100.0
        return RiskMetrics(
            entry_price=entry_price,
            loss_pips=loss_pips,
            reward_pips=reward_pips,
            risk_jpy=risk_jpy,
            reward_jpy=reward_jpy,
            reward_risk=reward_risk,
            spread_pips=spread_pips,
            jpy_per_pip=jpy_per_pip,
            estimated_margin_jpy=estimated_margin,
            margin_used_jpy=margin_used,
            margin_available_jpy=margin_available,
            margin_budget_jpy=margin_budget,
            margin_utilization_after_pct=margin_after_utilization,
            max_margin_utilization_pct=max_margin_pct,
        )

    def _margin_issues(self, snapshot: BrokerSnapshot, metrics: RiskMetrics) -> list[RiskIssue]:
        max_margin_pct = self.policy.max_margin_utilization_pct
        if max_margin_pct is None:
            return []
        if max_margin_pct <= 0 or max_margin_pct > 100:
            return [
                RiskIssue(
                    "INVALID_MARGIN_POLICY",
                    f"max_margin_utilization_pct must be within 0-100, got {max_margin_pct}",
                )
            ]
        account = snapshot.account
        if account is None:
            if not self.policy.require_margin_account:
                return []
            return [
                RiskIssue(
                    "MARGIN_ACCOUNT_MISSING",
                    "broker account summary is required to enforce margin availability and utilization cap",
                )
            ]
        issues: list[RiskIssue] = []
        if account.nav_jpy <= 0:
            issues.append(
                RiskIssue(
                    "MARGIN_NAV_INVALID",
                    f"account NAV must be positive to enforce {max_margin_pct:.1f}% margin cap; got {account.nav_jpy:.0f} JPY",
                )
            )
        if account.margin_used_jpy < 0:
            issues.append(
                RiskIssue(
                    "MARGIN_USED_INVALID",
                    f"account margin_used_jpy must be non-negative; got {account.margin_used_jpy:.0f} JPY",
                )
            )
        if account.margin_available_jpy < 0:
            issues.append(
                RiskIssue(
                    "MARGIN_AVAILABLE_INVALID",
                    f"account margin_available_jpy must be non-negative; got {account.margin_available_jpy:.0f} JPY",
                )
            )
        if issues:
            return issues

        budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
        cap_jpy = account.nav_jpy * (max_margin_pct / 100.0)
        if metrics.estimated_margin_jpy <= 0:
            return issues
        if metrics.estimated_margin_jpy > account.margin_available_jpy:
            issues.append(
                RiskIssue(
                    "MARGIN_AVAILABLE_EXCEEDED",
                    f"estimated initial margin {metrics.estimated_margin_jpy:.0f} JPY exceeds "
                    f"broker marginAvailable {account.margin_available_jpy:.0f} JPY",
                )
            )
        if budget <= 0 and metrics.estimated_margin_jpy > 0:
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_REACHED",
                    f"current marginUsed {account.margin_used_jpy:.0f} JPY already reaches/exceeds "
                    f"{max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY",
                )
            )
        elif metrics.estimated_margin_jpy > budget:
            after = account.margin_used_jpy + metrics.estimated_margin_jpy
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_EXCEEDED",
                    f"candidate margin {metrics.estimated_margin_jpy:.0f} JPY would raise marginUsed to "
                    f"{after:.0f} JPY, above {max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY "
                    f"(remaining budget {budget:.0f} JPY)",
                )
            )
        return issues

    def _conversion_quote_issues(self, pair: str, snapshot: BrokerSnapshot) -> list[RiskIssue]:
        quote_ccy = pair.split("_", 1)[1]
        if quote_ccy == "JPY":
            return []
        if snapshot.home_conversions.get(quote_ccy, 0.0) > 0:
            return []
        conversion_pair = f"{quote_ccy}_JPY"
        conversion_quote = snapshot.quotes.get(conversion_pair)
        if conversion_quote is None:
            return [
                RiskIssue(
                    "MISSING_CONVERSION_QUOTE",
                    f"{conversion_pair} quote is required to compute broker-truth JPY risk for {pair}",
                )
            ]
        issues: list[RiskIssue] = []
        quote_age = max(0.0, (self._now() - conversion_quote.timestamp_utc).total_seconds())
        if quote_age > self.policy.max_quote_age_seconds:
            issues.append(
                RiskIssue(
                    "STALE_CONVERSION_QUOTE",
                    f"{conversion_pair} conversion quote is stale: "
                    f"{quote_age:.1f}s > {self.policy.max_quote_age_seconds}s",
                )
            )
        spec = self._spec(conversion_pair)
        spread_pips = abs(conversion_quote.ask - conversion_quote.bid) * spec.pip_factor
        if spread_pips > spec.normal_spread_pips * self.policy.max_spread_multiple:
            issues.append(
                RiskIssue(
                    "CONVERSION_SPREAD_TOO_WIDE",
                    f"{conversion_pair} conversion spread {spread_pips:.1f}pip exceeds "
                    f"{self.policy.max_spread_multiple:.1f}x normal {spec.normal_spread_pips:.1f}pip",
                )
            )
        return issues

    def _quote_to_jpy(self, pair: str, snapshot: BrokerSnapshot) -> float | None:
        quote_ccy = pair.split("_", 1)[1]
        if quote_ccy == "JPY":
            return 1.0
        home_conversion = snapshot.home_conversions.get(quote_ccy)
        if home_conversion is not None and home_conversion > 0:
            return float(home_conversion)
        conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
        if conversion_quote is not None:
            return max(conversion_quote.bid, conversion_quote.ask)
        return None

    def _open_portfolio_risk_jpy(self, snapshot: BrokerSnapshot) -> tuple[float, RiskIssue | None]:
        total = 0.0
        sl_free_active = _trader_sl_repair_disabled()
        # Synthetic-SL distance for trader-owned SL-free positions: assume the
        # discretionary close happens within the SL-free invalidation budget
        # (5x M5 ATR ≈ 25 pips on majors). Hard-coded conservative estimate
        # so basket math has a real number; refine later from pair_charts ATR.
        SL_FREE_SYNTHETIC_PIPS = 30.0
        for position in self._entry_relevant_positions(snapshot):
            spec = self._spec(position.pair)
            quote_to_jpy = self._quote_to_jpy(position.pair, snapshot)
            if quote_to_jpy is None:
                return 0.0, RiskIssue(
                    "PORTFOLIO_RISK_UNKNOWN",
                    f"missing conversion quote for open position {position.trade_id} {position.pair}",
                )
            if position.stop_loss is None:
                if sl_free_active and position.owner == Owner.TRADER:
                    loss_pips = SL_FREE_SYNTHETIC_PIPS
                else:
                    return 0.0, RiskIssue(
                        "PORTFOLIO_RISK_UNKNOWN",
                        f"open position {position.trade_id} has no SL; cannot compute portfolio risk",
                    )
            else:
                if position.side == Side.LONG:
                    loss_pips = (position.entry_price - position.stop_loss) * spec.pip_factor
                else:
                    loss_pips = (position.stop_loss - position.entry_price) * spec.pip_factor
            jpy_per_pip = (position.units / spec.pip_factor) * quote_to_jpy
            total += max(0.0, loss_pips) * jpy_per_pip
        return total, None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _same_pair_adverse_add_pips(
    side: Side,
    same_side_positions: tuple[BrokerPosition, ...],
    entry_price: float,
    spec: InstrumentSpec,
) -> float | None:
    total_units = sum(abs(int(position.units)) for position in same_side_positions)
    if total_units <= 0:
        return None
    avg_entry = (
        sum(float(position.entry_price) * abs(int(position.units)) for position in same_side_positions)
        / total_units
    )
    raw_distance_pips = (float(entry_price) - avg_entry) * spec.pip_factor
    if side == Side.LONG:
        return max(0.0, -raw_distance_pips)
    return max(0.0, raw_distance_pips)


def _same_pair_position_units(snapshot: BrokerSnapshot, pair: str, *, owner: Owner | None = None) -> tuple[int, int]:
    long_units = 0
    short_units = 0
    for position in snapshot.positions:
        if position.pair != pair:
            continue
        if owner is not None and position.owner != owner:
            continue
        units = max(0, abs(int(position.units)))
        if position.side == Side.LONG:
            long_units += units
        elif position.side == Side.SHORT:
            short_units += units
    return long_units, short_units


def _account_hedging_enabled(snapshot: BrokerSnapshot) -> bool:
    return bool(snapshot.account and snapshot.account.hedging_enabled)


def _candidate_adds_to_trader_net_side(intent: OrderIntent, snapshot: BrokerSnapshot) -> bool:
    """True when a same-pair add extends the trader-owned dominant side.

    The per-position opposite-side guard prevents accidental hedges. Once
    trader-owned exposure is already net LONG or net SHORT, adding to that
    same net side is a pyramid, not a new hedge, even if an older opposite
    trade still exists in the account.
    """
    long_units, short_units = _same_pair_position_units(snapshot, intent.pair, owner=Owner.TRADER)
    if intent.side == Side.LONG:
        return long_units >= short_units
    return short_units >= long_units


def _intent_declares_hedge(intent: OrderIntent) -> bool:
    return str((intent.metadata or {}).get("position_intent") or "").upper() == "HEDGE"


def _intent_declares_recovery_hedge(intent: OrderIntent) -> bool:
    return _intent_declares_hedge(intent) and bool((intent.metadata or {}).get("hedge_recovery"))


def _hedge_metadata_issues(intent: OrderIntent) -> list[RiskIssue]:
    if not _intent_declares_hedge(intent):
        return []
    metadata = intent.metadata or {}
    issues: list[RiskIssue] = []
    timing_class = str(metadata.get("hedge_timing_class") or "").upper()
    if timing_class not in HEDGE_TIMING_CLASSES:
        issues.append(
            RiskIssue(
                "HEDGE_TIMING_METADATA_MISSING",
                "HEDGE intents must carry metadata.hedge_timing_class "
                f"in {sorted(HEDGE_TIMING_CLASSES)}",
            )
        )
    if metadata.get("hedge_unwind_plan_required") is not True:
        issues.append(
            RiskIssue(
                "HEDGE_UNWIND_PLAN_MISSING",
                "HEDGE intents must set metadata.hedge_unwind_plan_required=true",
            )
        )
    if not str(metadata.get("hedge_review_trigger") or "").strip():
        issues.append(
            RiskIssue(
                "HEDGE_REVIEW_TRIGGER_MISSING",
                "HEDGE intents must carry metadata.hedge_review_trigger so the leg is time-boxed",
            )
        )
    if _intent_declares_recovery_hedge(intent) and timing_class == "CONTINUATION":
        try:
            scale = float(metadata.get("hedge_recovery_size_scale"))
        except (TypeError, ValueError):
            scale = None
        if scale is None or scale > HEDGE_CONTINUATION_MAX_SCALE:
            issues.append(
                RiskIssue(
                    "HEDGE_CONTINUATION_SIZE_TOO_LARGE",
                    "CONTINUATION recovery hedges must publish "
                    f"hedge_recovery_size_scale <= {HEDGE_CONTINUATION_MAX_SCALE:.2f}",
                )
            )
    return issues


def _hedge_balance_issues(intent: OrderIntent, snapshot: BrokerSnapshot) -> list[RiskIssue]:
    if not _intent_declares_hedge(intent):
        return []
    margin_free_units = hedge_margin_free_units(
        pair=intent.pair,
        side=intent.side,
        snapshot=snapshot,
        position_intent="HEDGE",
    )
    if margin_free_units <= 0:
        return [
            RiskIssue(
                "HEDGE_REFERENCE_ALREADY_COVERED",
                f"{intent.pair} {intent.side.value} HEDGE has no uncovered opposite-side units; "
                "additional units would become a net directional add, not a hedge",
            )
        ]
    if abs(int(intent.units)) > margin_free_units:
        return [
            RiskIssue(
                "HEDGE_UNITS_EXCEED_OPPOSITE_EXPOSURE",
                f"{intent.pair} {intent.side.value} HEDGE requests {abs(int(intent.units))}u but "
                f"only {margin_free_units}u can be added before the opposite exposure is fully covered",
            )
        ]
    return []


def _is_operator_managed_manual(position: BrokerPosition) -> bool:
    return is_operator_managed_manual_owner(position.owner)


def _is_operator_managed_manual_order(order: BrokerOrder) -> bool:
    return is_operator_managed_manual_owner(order.owner)


def _is_pending_entry_order(order: BrokerOrder) -> bool:
    if order.trade_id:
        return False
    order_type = order.order_type.upper()
    return order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}


def resolve_max_loss_jpy(
    *,
    max_loss_jpy: float | None,
    max_loss_pct: float | None,
    equity_jpy: float | None,
    default_max_loss_jpy: float | None = None,
    label: str = "max-loss",
) -> float:
    """Resolve a risk cap from explicit JPY value or percentage of equity."""
    if max_loss_jpy is not None:
        if max_loss_jpy <= 0:
            raise ValueError(f"{label}: --max-loss-jpy must be positive")
        return float(max_loss_jpy)
    if max_loss_pct is not None:
        if max_loss_pct <= 0:
            raise ValueError(f"{label}: --max-loss-pct must be positive")
        if equity_jpy is None:
            raise ValueError(f"{label}: --max-loss-pct requires --risk-equity-jpy or a daily target state")
        if equity_jpy <= 0:
            raise ValueError(f"{label}: --risk-equity-jpy must be positive")
        return equity_jpy * (max_loss_pct / 100.0)
    if default_max_loss_jpy is None:
        raise ValueError(
            f"{label}: no risk cap available. Provide --max-loss-jpy, "
            f"--max-loss-pct + --risk-equity-jpy, or have the daily-target ledger emit "
            f"daily_risk_budget_jpy from current equity. No JPY literal fallback."
        )
    return float(default_max_loss_jpy)
