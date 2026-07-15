from __future__ import annotations

import os
import math
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType


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
from .forecast_learning import (
    FORECAST_LEARNING_EXECUTION_DESK_BY_METHOD,
    forecast_learning_selected_method,
    validate_forecast_learning_execution_geometry,
)
from .instruments import DEFAULT_TRADER_PAIRS, NORMAL_SPREAD_PIPS, instrument_pip_factor
from .guardian_receipt_consumption import guardian_receipt_new_entry_blockers_from_paths
from .guardian_tuning_overrides import resolve_forecast_confidence_floor_state
from .market_close_leak_gate import market_close_leak_family_block_issue
from .month_scale_residual_gate import month_scale_residual_metadata_issue
from .operator_manual import (
    is_operator_managed_manual_owner,
    operator_manual_jpy_add_block_issue,
    operator_manual_same_theme_add_block_issue,
)
from .paths import DEFAULT_FORECAST_HISTORY, DEFAULT_PAIR_CHARTS
from .strategy.m15_recovery_contract import (
    RECOVERY_ORDER_TYPE as M15_RECOVERY_ORDER_TYPE,
    recovery_claimed as m15_recovery_claimed,
)


_M15_RECOVERY_HISTORY_CACHE_MAX_ENTRIES = 32
_M15_RECOVERY_HISTORY_CACHE: dict[
    tuple[str, str, str, tuple[int, ...]], dict
] = {}
_M15_RECOVERY_HISTORY_CACHE_LOCK = threading.Lock()

# OANDA Japan retail FX margin in the current account is 25:1 leverage, i.e.
# 4% initial margin. Recent broker truth confirms the same scale: USD_JPY
# 25,000u filled near 155.962 required roughly 155,954 JPY initial margin.
# This is broker/account policy, not market data; replace it with per-instrument
# `/accounts/{id}/instruments` marginRate once that adapter is wired.
OANDA_JP_RETAIL_FX_MARGIN_RATE = 0.04

# OANDA has already accepted integer-size orders below 1,000 units on this
# account (including 201u, 322u and 2u).  Lot size scales both gross reward and
# spread cost linearly, so a sub-1,000 order is not intrinsically negative;
# economic viability comes from the existing bid/ask spread, TP/SL geometry,
# and post-cost expectancy gates.  The only broker-size floor is therefore one
# integer unit.  Keep the historical name as a compatibility alias because it
# is referenced across risk, sizing, and receipt code.
MIN_PRODUCTION_LOT_UNITS = 1

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
# Public legacy receipt label retained for compatibility and forged-packet
# regression coverage. It is deliberately not an accepted relaxation mode.
LOSS_ASYMMETRY_OANDA_CAMPAIGN_FIREPOWER_RELAXED_MODE = "OANDA_CAMPAIGN_FIREPOWER_RELAXED"
CAPTURE_ECONOMICS_STALE_BLOCK_CODE = "CAPTURE_ECONOMICS_STALE"
SPREAD_FLOOR_COMPARISON_EPSILON_PIPS = 1e-6


def _min_lot_test_override_active() -> bool:
    """Legacy test switch retained for compatibility.

    The production floor is now the broker integer minimum (1u), so positive
    integer test orders no longer need this override.  Older fixtures may
    still set it harmlessly.
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
    It is deliberately JPY-value-free: an explicit active guard or
    NEGATIVE_EXPECTANCY status makes the observed average winner the temporary
    per-entry loss ceiling for every ordinary, non-exempt fresh-risk shape.
    """
    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "NEW").upper() == "HEDGE":
        return []
    mode = str(metadata.get("loss_asymmetry_guard_mode") or "").upper()
    status = str(metadata.get("capture_economics_status") or "").upper()
    active = _truthy_metadata(metadata.get("loss_asymmetry_guard_active"))
    invalid_numeric_fields = _invalid_loss_asymmetry_numeric_fields(metadata)
    if invalid_numeric_fields and (active or status == "NEGATIVE_EXPECTANCY" or mode):
        return [
            RiskIssue(
                "LOSS_ASYMMETRY_GUARD_NONFINITE",
                "loss-asymmetry metadata contains a non-finite or malformed numeric "
                f"field ({', '.join(invalid_numeric_fields)}); refresh capture-economics "
                "before adding fresh one-way risk.",
            )
        ]
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
            and _loss_asymmetry_tp_proof_collection_shape_allowed(
                intent,
                metadata,
                metrics,
            )
        ):
            return []
    avg_win = _to_float(metadata.get("capture_avg_win_jpy"))
    avg_loss = _to_float(metadata.get("capture_avg_loss_jpy"))
    guard_required = active or status == "NEGATIVE_EXPECTANCY"
    if guard_required and (avg_win is None or avg_win <= 0):
        return [
            RiskIssue(
                "LOSS_ASYMMETRY_GUARD_CAP_MISSING",
                "ordinary fresh risk under an active loss-asymmetry guard or "
                "NEGATIVE_EXPECTANCY requires a finite positive "
                "capture_avg_win_jpy from realized local payoff evidence; a "
                "declared, effective, or legacy OANDA firepower cap cannot "
                "substitute. Refresh capture-economics before live send.",
            )
        ]
    if not guard_required:
        return []
    declared_cap = _to_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    cap_candidates = [
        value
        for value in (declared_cap, avg_win)
        if value is not None and value > 0
    ]
    # A receipt may lie about either its effective or declared cap. Under an
    # active NEGATIVE_EXPECTANCY guard, the tighter positive value wins; OANDA
    # firepower metadata can never widen the observed average-winner ceiling.
    cap = min(cap_candidates) if cap_candidates else None
    if cap is None or cap <= 0:
        return [
            RiskIssue(
                "LOSS_ASYMMETRY_GUARD_CAP_MISSING",
                "loss-asymmetry guard is active but no finite positive local "
                "average-winner cap is available; refresh capture-economics "
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


def _tp_harvest_vehicle_for_order_type(order_type: OrderType) -> str | None:
    if order_type == OrderType.LIMIT:
        return "LIMIT"
    if order_type == OrderType.MARKET:
        return "MARKET"
    if order_type == OrderType.STOP_ENTRY:
        return "STOP"
    return None


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
    if str(metadata.get("capture_take_profit_scope") or "").upper() not in {
        "PAIR_SIDE_METHOD_VEHICLE",
        "PAIR_SIDE_METHOD",
        "PAIR_SIDE",
    }:
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
    metrics: RiskMetrics,
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
    if str(metadata.get("capture_take_profit_scope") or "").upper() not in {
        "PAIR_SIDE_METHOD_VEHICLE",
        "PAIR_SIDE_METHOD",
    }:
        return False
    if metadata.get("positive_rotation_proof_collection_min_lot_sizing") is not True:
        return False
    if not _exact_min_lot_cap_lift_matches(
        intent,
        metadata,
        metrics,
        units_key="positive_rotation_proof_collection_min_lot_units",
        loss_key="positive_rotation_proof_collection_min_lot_loss_jpy",
    ):
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


def _m15_recovery_evidence_weighted_expectancy_jpy(
    intent: OrderIntent,
    metrics: RiskMetrics,
    validation: "M15RecoveryMicroValidation",
) -> float | None:
    """Return conservative current-geometry expectancy for a verified recovery.

    A fixed 1R gate discards the interaction between hit rate and payoff.  For
    the bounded M15 recovery route, the producer and this RiskEngine have
    already revalidated an exact-vehicle TP cohort and its Wilson lower bound.
    Apply that lower bound to the *current* TP/SL geometry, after charging one
    current spread to both the winning and losing paths.  This can relax only
    the generic RR floor; every loss, margin, freshness, event, GPT, Guardian,
    allocation, and final pre-POST check remains in force.
    """

    if not validation.applicable or not validation.allowed:
        return None
    metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
    if str(metadata.get("positive_rotation_mode") or "").upper() not in {
        "TP_PROOF_COLLECTION_HARVEST",
        "TP_PROVEN_HARVEST",
    }:
        return None
    if (
        str(metadata.get("positive_rotation_confidence_method") or "")
        != "WILSON_LOWER_BOUND_STRESS_EXPECTANCY"
        or not math.isclose(
            _to_float(metadata.get("positive_rotation_confidence_z")) or 0.0,
            1.96,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None
    lower = _to_float(metadata.get("positive_rotation_tp_win_rate_lower"))
    trades = _to_int(metadata.get("positive_rotation_tp_trades"))
    wins = _to_int(metadata.get("positive_rotation_tp_wins"))
    losses = _to_int(metadata.get("capture_take_profit_losses"))
    if (
        lower is None
        or not 0.0 < lower < 1.0
        or trades is None
        or trades < LOSS_ASYMMETRY_TP_PROOF_COLLECTION_MIN_EXIT_TRADES
        or wins != trades
        or losses != 0
    ):
        return None
    spread_cost_jpy = max(0.0, metrics.spread_pips * metrics.jpy_per_pip)
    net_reward_jpy = max(0.0, metrics.reward_jpy - spread_cost_jpy)
    net_loss_jpy = metrics.risk_jpy + spread_cost_jpy
    if net_reward_jpy <= 0.0 or net_loss_jpy <= 0.0:
        return None
    expectancy = lower * net_reward_jpy - (1.0 - lower) * net_loss_jpy
    return expectancy if math.isfinite(expectancy) else None


def _exact_min_lot_cap_lift_matches(
    intent: OrderIntent,
    metadata: dict,
    metrics: RiskMetrics,
    *,
    units_key: str,
    loss_key: str,
) -> bool:
    """Bind a legacy minimum-size cap lift to the exact 1u receipt.

    These narrow evidence-collection modes may only lift an average-winner
    cap enough to fund the current broker minimum.  After the production floor
    moved from 1,000u to 1u, accepting an old 1,000u metadata packet would turn
    that exception into a large risk-cap bypass.  Require the actual order,
    declared units, declared minimum loss, effective cap, and freshly computed
    RiskEngine loss to describe the same one-unit exposure.
    """

    actual_units = abs(int(intent.units))
    declared_units = _to_int(metadata.get(units_key))
    declared_loss = _to_float(metadata.get(loss_key))
    effective_cap = _to_float(
        metadata.get("loss_asymmetry_guard_effective_max_loss_jpy")
    )
    if (
        actual_units != MIN_PRODUCTION_LOT_UNITS
        or declared_units != MIN_PRODUCTION_LOT_UNITS
        or declared_loss is None
        or declared_loss <= 0.0
        or effective_cap is None
        or effective_cap <= 0.0
        or metrics.risk_jpy <= 0.0
    ):
        return False
    return bool(
        math.isclose(
            declared_loss,
            metrics.risk_jpy,
            rel_tol=1e-6,
            abs_tol=1e-4,
        )
        and math.isclose(
            effective_cap,
            declared_loss,
            rel_tol=1e-6,
            abs_tol=1e-4,
        )
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


DEFAULT_SPECS: Mapping[str, InstrumentSpec] = MappingProxyType(
    {
        pair: InstrumentSpec(
            pair,
            instrument_pip_factor(pair),
            NORMAL_SPREAD_PIPS[pair],
        )
        for pair in DEFAULT_TRADER_PAIRS
    }
)

# C-4 margin-aware basket tolerance. This is engineering headroom for
# intra-cycle broker-margin drift, not a market edge threshold.
MARGIN_AWARE_BASKET_BUFFER = 0.9


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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _invalid_loss_asymmetry_numeric_fields(metadata: dict) -> list[str]:
    invalid: list[str] = []
    for key in (
        "capture_avg_win_jpy",
        "capture_avg_loss_jpy",
        "loss_asymmetry_guard_loss_cap_jpy",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            invalid.append(key)
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            invalid.append(key)
            continue
        if not math.isfinite(parsed):
            invalid.append(key)
    return invalid


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
    forecast_direction = str(metadata.get("forecast_direction") or "").upper()
    direction_phrase = (
        f"forecast {issue['direction']}"
        if forecast_direction in {"UP", "DOWN"}
        else f"entry direction {issue['direction']} under {forecast_direction or 'UNKNOWN'} forecast"
    )
    return RiskIssue(
        "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
        (
            f"{intent.pair} {intent.side.value} {direction_phrase} matches S5 bid/ask "
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


def _forecast_learning_scout_forward_evidence_supported(
    intent: OrderIntent,
) -> bool:
    """Authenticate the one narrow broad-replay learning exception.

    This does not grant live permission. It only prevents an aggregate
    pair/direction replay bucket from making forward model learning
    impossible. The intent must still pass the independent predictive-SCOUT,
    strategy-profile, spread/geometry, RiskEngine, GPT, Guardian, and broker
    gateway checks.
    """

    metadata = intent.metadata or {}
    if (
        str(metadata.get("predictive_scout_source") or "").upper()
        != "FORECAST_ORIENTATION_LEARNING"
        or intent.order_type != OrderType.LIMIT
        or abs(int(intent.units)) < MIN_PRODUCTION_LOT_UNITS
        or intent.entry is None
        or intent.tp is None
        or intent.sl is None
    ):
        return False
    receipt = metadata.get("forecast_learning_v1")
    if not isinstance(receipt, dict):
        return False
    method = intent.market_context.method if intent.market_context is not None else None
    selected_method = forecast_learning_selected_method(receipt)
    if method is None or method.value != selected_method:
        return False
    if (
        str(metadata.get("desk") or "").lower()
        != FORECAST_LEARNING_EXECUTION_DESK_BY_METHOD.get(selected_method)
    ):
        return False
    rank_direction = str(receipt.get("rank_direction") or "").upper()
    expected_side = (
        Side.LONG
        if rank_direction == "UP"
        else Side.SHORT
        if rank_direction == "DOWN"
        else None
    )
    if intent.side != expected_side:
        return False
    # Local import avoids a module-load cycle: predictive_scout imports the
    # broker production-lot compatibility constant from this module.
    from .predictive_scout import predictive_scout_metadata_supported

    execution_geometry = metadata.get("forecast_learning_execution_geometry_v1")
    decision_sha = str(receipt.get("decision_sha256") or "")
    return bool(
        predictive_scout_metadata_supported(metadata)
        and validate_forecast_learning_execution_geometry(
            execution_geometry,
            pair=intent.pair,
            side=intent.side.value,
            method=method.value,
            entry=intent.entry,
            take_profit=intent.tp,
            stop_loss=intent.sl,
            source_decision_sha256=decision_sha,
        )
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


def _guardian_tuning_override_issues(
    intent: OrderIntent,
    *,
    for_live_send: bool,
) -> list[RiskIssue]:
    """Enforce evidence-activated pair/method confidence tightening at send."""

    if not for_live_send:
        return []
    metadata = intent.metadata or {}
    confidence = _to_float(metadata.get("forecast_confidence"))
    method = intent.market_context.method if intent.market_context is not None else None
    method_name = method.value if isinstance(method, TradeMethod) else str(method or "").upper()
    if not method_name:
        return []
    fallback = (
        FORECAST_RANGE_ROTATION_MIN_CONFIDENCE
        if method == TradeMethod.RANGE_ROTATION
        else FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE
    )
    lane_id = str(
        metadata.get("lane_id") or metadata.get("parent_lane_id") or ""
    ).strip()
    resolution = resolve_forecast_confidence_floor_state(
        pair=intent.pair,
        method=method_name,
        lane_id=lane_id or None,
        fallback=fallback,
    )
    if resolution["status"] in {
        "OVERRIDE_STATE_INVALID",
        "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
        "OVERRIDE_CONFIRMATION_PENDING",
        "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING",
        "OVERRIDE_LANE_QUARANTINED",
        "OVERRIDE_LANE_ID_REQUIRED",
        "OVERRIDE_LANE_ID_INVALID",
    }:
        issue_code = (
            "GUARDIAN_TUNING_LANE_QUARANTINED"
            if resolution["status"] == "OVERRIDE_LANE_QUARANTINED"
            else (
                "GUARDIAN_TUNING_POST_ACTIVATION_MONITOR_PENDING"
                if resolution["status"]
                == "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING"
                else (
                    "GUARDIAN_TUNING_OVERRIDE_CONFIRMATION_PENDING"
                    if resolution["status"] == "OVERRIDE_CONFIRMATION_PENDING"
                    else (
                        "GUARDIAN_TUNING_OVERRIDE_LANE_ID_INVALID"
                        if resolution["status"] in {
                            "OVERRIDE_LANE_ID_REQUIRED",
                            "OVERRIDE_LANE_ID_INVALID",
                        }
                        else "GUARDIAN_TUNING_OVERRIDE_STATE_INVALID"
                    )
                )
            )
        )
        return [
            RiskIssue(
                issue_code,
                (
                    f"{intent.pair} {method_name} guardian tuning state is not "
                    f"ready ({resolution['status']}); "
                    "block live send until the accepted setting provenance is repaired."
                ),
                severity="BLOCK",
            )
        ]
    active_floor = float(resolution["resolved_value"])
    if active_floor <= fallback or (
        confidence is not None and confidence >= active_floor
    ):
        return []
    confidence_text = "missing" if confidence is None else f"{confidence:.4f}"
    return [
        RiskIssue(
            "GUARDIAN_TUNING_FORECAST_CONFIDENCE_FLOOR",
            (
                f"{intent.pair} {method_name} forecast confidence {confidence_text} "
                f"is below evidence-activated floor {active_floor:.4f}; preserve the "
                "previous gates and wait for a qualifying entry-time forecast."
            ),
            severity="BLOCK",
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
    scope = str(metadata.get("capture_take_profit_scope") or "").upper()
    if scope == "PAIR_SIDE_METHOD":
        expected_scope = (
            f"{intent.pair}|{intent.side.value}|{method.value}|TAKE_PROFIT_ORDER"
        ).upper()
    elif scope == "PAIR_SIDE_METHOD_VEHICLE":
        vehicle = _tp_harvest_vehicle_for_order_type(intent.order_type)
        if vehicle is None:
            return False
        expected_scope = (
            f"{intent.pair}|{intent.side.value}|{method.value}|{vehicle}|TAKE_PROFIT_ORDER"
        ).upper()
    else:
        return False
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
class M15RecoveryMicroValidation:
    """Independent risk/gateway result for the bounded M15 recovery claim."""

    applicable: bool
    allowed: bool
    evidence: dict
    issues: tuple[RiskIssue, ...] = ()


def m15_recovery_micro_claimed(intent: OrderIntent) -> bool:
    """Detect the recovery contract without changing ordinary intent behavior."""

    metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
    return m15_recovery_claimed(metadata)


def _strict_aware_utc_iso(value: object) -> datetime | None:
    if value.__class__ is not str or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _m15_recovery_pair_chart_from_path(
    pair_charts_path: Path,
    *,
    expected_pair: str,
) -> tuple[dict | None, dict, RiskIssue | None]:
    """Read one canonical artifact and bind its root clock into one exact row."""

    try:
        raw = pair_charts_path.read_bytes()
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return None, {"status": "BLOCKED", "path": str(pair_charts_path)}, RiskIssue(
            "M15_RECOVERY_PAIR_CHARTS_UNREADABLE",
            "M15 recovery requires the current canonical pair_charts.json; "
            f"the artifact could not be read exactly: {type(exc).__name__}: {exc}",
        )
    source_sha256 = hashlib.sha256(raw).hexdigest()
    generated_at_raw = payload.get("generated_at_utc") if isinstance(payload, dict) else None
    generated_at = _strict_aware_utc_iso(generated_at_raw)
    charts = payload.get("charts") if isinstance(payload, dict) else None
    if generated_at is None or not isinstance(charts, list):
        return None, {
            "status": "BLOCKED",
            "path": str(pair_charts_path),
            "source_sha256": source_sha256,
        }, RiskIssue(
            "M15_RECOVERY_PAIR_CHARTS_SHAPE_INVALID",
            "M15 recovery requires an aware root generated_at_utc and a charts array",
        )
    exact_rows = [
        chart
        for chart in charts
        if isinstance(chart, dict) and chart.get("pair") == expected_pair
    ]
    all_pairs = [
        chart.get("pair") for chart in charts if isinstance(chart, dict)
    ]
    if (
        len(exact_rows) != 1
        or len(all_pairs) != len(charts)
        or any(pair.__class__ is not str for pair in all_pairs)
        or len(set(all_pairs)) != len(all_pairs)
    ):
        return None, {
            "status": "BLOCKED",
            "path": str(pair_charts_path),
            "source_sha256": source_sha256,
        }, RiskIssue(
            "M15_RECOVERY_PAIR_CHART_IDENTITY_INVALID",
            f"pair_charts.json must contain exactly one {expected_pair} row and no duplicate/malformed pair identities",
        )
    row = dict(exact_rows[0])
    row_clock = row.get("generated_at_utc")
    if row_clock is not None and row_clock != generated_at_raw:
        return None, {
            "status": "BLOCKED",
            "path": str(pair_charts_path),
            "source_sha256": source_sha256,
        }, RiskIssue(
            "M15_RECOVERY_PAIR_CHART_CLOCK_CONFLICT",
            "the selected pair row conflicts with the canonical artifact root generated_at_utc",
        )
    # The producer envelope owns the artifact clock. This mirrors the normal
    # chart loaders and prevents a row-local timestamp from refreshing itself.
    row["generated_at_utc"] = generated_at_raw
    row_sha256 = hashlib.sha256(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row, {
        "status": "LOADED",
        "path": str(pair_charts_path),
        "source_sha256": source_sha256,
        "source_size_bytes": len(raw),
        "root_generated_at_utc": generated_at.isoformat(),
        "pair": expected_pair,
        "pair_chart_sha256": row_sha256,
        "root_clock_bound_into_pair_chart": True,
    }, None


def _m15_recovery_forecast_history_row(
    path: Path,
    *,
    pair: str,
    cycle_id: str,
) -> tuple[dict | None, RiskIssue | None]:
    # Production history is append-only and currently exceeds 64 MiB.  A
    # whole-file size veto made every otherwise valid recovery lane impossible
    # in live trading.  Stream every JSONL row with a per-row memory bound so
    # uniqueness remains proven across the entire ledger without loading the
    # ledger into memory.
    max_line_bytes = 1024 * 1024
    try:
        initial_signature = _m15_recovery_file_signature(path.stat())
        cache_key = (
            str(path.resolve()),
            pair,
            cycle_id,
            initial_signature,
        )
        with _M15_RECOVERY_HISTORY_CACHE_LOCK:
            cached = _M15_RECOVERY_HISTORY_CACHE.get(cache_key)
        if cached is not None:
            if _m15_recovery_file_signature(path.stat()) == initial_signature:
                return dict(cached), None
        matches: list[dict] = []
        with path.open("rb") as handle:
            opened_signature = _m15_recovery_file_signature(
                os.fstat(handle.fileno())
            )
            line_number = 0
            while True:
                raw = handle.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes and not raw.endswith(b"\n"):
                    raise ValueError(
                        f"forecast history line {line_number} exceeds the 1 MiB row bound"
                    )
                if not raw.strip():
                    continue
                item = json.loads(
                    raw,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
                if not isinstance(item, dict):
                    raise ValueError(
                        f"forecast history line {line_number} is not an object"
                    )
                if item.get("pair") == pair and item.get("cycle_id") == cycle_id:
                    matches.append(item)
            closed_signature = _m15_recovery_file_signature(
                os.fstat(handle.fileno())
            )
        current_signature = _m15_recovery_file_signature(path.stat())
        if (
            opened_signature != closed_signature
            or opened_signature != current_signature
        ):
            raise ValueError(
                "forecast history changed while the exact cycle row was being validated"
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return None, RiskIssue(
            "M15_RECOVERY_FORECAST_HISTORY_UNREADABLE",
            "M15 recovery requires the exact same-cycle forecast history row; "
            f"the ledger could not be read safely: {type(exc).__name__}: {exc}",
        )
    if len(matches) != 1:
        return None, RiskIssue(
            "M15_RECOVERY_FORECAST_HISTORY_IDENTITY_INVALID",
            f"M15 recovery requires exactly one {pair}/{cycle_id} forecast history row; found {len(matches)}",
        )
    with _M15_RECOVERY_HISTORY_CACHE_LOCK:
        _M15_RECOVERY_HISTORY_CACHE[cache_key] = dict(matches[0])
        while (
            len(_M15_RECOVERY_HISTORY_CACHE)
            > _M15_RECOVERY_HISTORY_CACHE_MAX_ENTRIES
        ):
            _M15_RECOVERY_HISTORY_CACHE.pop(
                next(iter(_M15_RECOVERY_HISTORY_CACHE))
            )
    return matches[0], None


def _m15_recovery_file_signature(stat_result: os.stat_result) -> tuple[int, ...]:
    """Return the stable identity needed around a streaming ledger scan."""

    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def validate_m15_recovery_micro_live_claim(
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    *,
    pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
    forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
    validation_time_utc: datetime | None = None,
) -> M15RecoveryMicroValidation:
    """Revalidate the non-live receipt against source bytes and broker quote.

    Passing this function is one necessary risk/gateway proof only. It never
    mutates ``live_permission`` and does not bypass any ordinary RiskEngine,
    GPT, Guardian, margin, loss, or order-building gate.
    """

    if not m15_recovery_micro_claimed(intent):
        return M15RecoveryMicroValidation(
            applicable=False,
            allowed=True,
            evidence={"status": "NOT_APPLICABLE"},
        )

    from quant_rabbit.strategy.directional_forecaster import (
        M15_RECOVERY_MICRO_CONTRACT,
        M15_RECOVERY_MICRO_MAX_UNITS,
        M15_RECOVERY_MICRO_MODE,
        validate_m15_recovery_micro_receipt,
    )
    from quant_rabbit.strategy.m15_recovery_contract import (
        FORECAST_CONTRACT,
        LANE_CONTRACT,
        validate_forecast_binding,
        validate_lane_binding,
    )

    metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
    receipt = metadata.get("m15_recovery_micro_receipt")
    issues: list[RiskIssue] = []
    if not isinstance(receipt, dict):
        issues.append(
            RiskIssue(
                "M15_RECOVERY_RECEIPT_INVALID",
                "M15 recovery metadata must carry one exact receipt object",
            )
        )
        receipt = {}
    receipt_sha = receipt.get("receipt_sha256")
    outer_receipt_sha = metadata.get("m15_recovery_micro_receipt_sha256")
    forecast_receipt = metadata.get("forecast_m15_recovery_receipt")
    producer_units = metadata.get("m15_recovery_micro_units")
    if (
        receipt.get("contract") != M15_RECOVERY_MICRO_CONTRACT
        or receipt.get("mode") != M15_RECOVERY_MICRO_MODE
        or receipt.get("status") != "ELIGIBLE_FOR_MICRO_REVALIDATION"
        or receipt.get("pair") != intent.pair
        or receipt.get("live_permission") is not False
        or receipt.get("requires_risk_gateway_revalidation") is not True
        or receipt.get("manual_position_mutation_allowed") is not False
        or not isinstance(receipt_sha, str)
        or outer_receipt_sha != receipt_sha
        or metadata.get("m15_recovery_micro_contract") != M15_RECOVERY_MICRO_CONTRACT
        or metadata.get("m15_recovery_micro_mode") != M15_RECOVERY_MICRO_MODE
        or metadata.get("m15_recovery_micro_live_permission") is not False
        or metadata.get("m15_recovery_micro_requires_risk_gateway_revalidation") is not True
        or metadata.get("m15_recovery_micro_manual_position_mutation_allowed") is not False
        or metadata.get("m15_recovery_micro_full_size_allowed") is not False
        or metadata.get("m15_recovery_micro_max_units") != M15_RECOVERY_MICRO_MAX_UNITS
        or producer_units.__class__ is not int
        or not 1 <= producer_units <= M15_RECOVERY_MICRO_MAX_UNITS
        or intent.units > producer_units
        or forecast_receipt != receipt
        or metadata.get("forecast_m15_recovery_mode") != M15_RECOVERY_MICRO_MODE
        or metadata.get("forecast_m15_recovery_live_permission") is not False
    ):
        issues.append(
            RiskIssue(
                "M15_RECOVERY_RECEIPT_CONTRACT_MISMATCH",
                "M15 recovery receipt/metadata must preserve its exact non-live, no-manual-mutation, 999u contract and digest",
            )
        )

    forecast_direction = metadata.get("forecast_direction")
    direction_matches = (
        (forecast_direction == "UP" and intent.side == Side.LONG)
        or (forecast_direction == "DOWN" and intent.side == Side.SHORT)
    )
    method = intent.market_context.method if intent.market_context is not None else None
    vehicle = _tp_harvest_vehicle_for_order_type(intent.order_type)
    expected_scope_key = (
        f"{intent.pair}|{intent.side.value}|{method.value}|{vehicle}|TAKE_PROFIT_ORDER"
        if method is not None and vehicle is not None
        else None
    )
    proof_mode = metadata.get("positive_rotation_mode")
    if (
        not direction_matches
        or intent.order_type != OrderType.STOP_ENTRY
        or intent.tp is None
        or str(metadata.get("position_intent") or "NEW").upper() == "HEDGE"
        or metadata.get("attach_take_profit_on_fill") is not True
        or metadata.get("tp_execution_mode") != "ATTACHED_TECHNICAL_TP"
        or metadata.get("tp_target_intent") != "HARVEST"
        or metadata.get("opportunity_mode") != "HARVEST"
        or proof_mode not in {"TP_PROOF_COLLECTION_HARVEST", "TP_PROVEN_HARVEST"}
        or metadata.get("capture_take_profit_exact_vehicle_required") is not True
        or metadata.get("capture_take_profit_scope") != "PAIR_SIDE_METHOD_VEHICLE"
        or metadata.get("capture_take_profit_vehicle") != vehicle
        or metadata.get("capture_take_profit_scope_key") != expected_scope_key
        or intent.units.__class__ is not int
        or not 1 <= intent.units <= M15_RECOVERY_MICRO_MAX_UNITS
    ):
        issues.append(
            RiskIssue(
                "M15_RECOVERY_LIVE_SHAPE_INVALID",
                f"M15 recovery is limited to a direction-aligned 1..999u {M15_RECOVERY_ORDER_TYPE}, attached-TP, exact-vehicle HARVEST intent",
            )
        )
    try:
        from quant_rabbit.strategy.intent_generator import (
            _positive_rotation_source_integrity_failures,
            positive_rotation_proof_acquisition_contract,
        )

        source_failures = _positive_rotation_source_integrity_failures(
            pair=intent.pair,
            side=intent.side,
            method=method,
            order_type=intent.order_type,
            metadata=metadata,
            mode=str(proof_mode or ""),
        )
        proof_contract = positive_rotation_proof_acquisition_contract(intent)
    except Exception as exc:
        source_failures = [f"producer proof validator raised {type(exc).__name__}: {exc}"]
        proof_contract = {"reachable": False, "failed_checks": source_failures}
    if source_failures or proof_contract.get("reachable") is not True:
        issues.append(
            RiskIssue(
                "M15_RECOVERY_POSITIVE_EXACT_VEHICLE_PROOF_INVALID",
                "M15 recovery requires the current producer-owned positive exact-vehicle HARVEST proof contract",
            )
        )

    forecast_binding = metadata.get("forecast_m15_recovery_binding")
    lane_binding = metadata.get("m15_recovery_lane_binding")
    cycle_id = metadata.get("forecast_cycle_id")
    history_row: dict | None = None
    history_issue: RiskIssue | None = None
    if cycle_id.__class__ is str and cycle_id:
        history_row, history_issue = _m15_recovery_forecast_history_row(
            forecast_history_path,
            pair=intent.pair,
            cycle_id=cycle_id,
        )
    else:
        history_issue = RiskIssue(
            "M15_RECOVERY_FORECAST_HISTORY_IDENTITY_INVALID",
            "M15 recovery forecast_cycle_id is missing or malformed",
        )
    if history_issue is not None:
        issues.append(history_issue)
    forecast_binding_valid, forecast_binding_error = validate_forecast_binding(
        forecast_binding,
        recovery_receipt=receipt,
        metadata=metadata,
        history_row=history_row,
    )
    if (
        not forecast_binding_valid
        or not isinstance(forecast_binding, dict)
        or forecast_binding.get("contract") != FORECAST_CONTRACT
        or metadata.get("forecast_m15_recovery_binding_sha256")
        != forecast_binding.get("binding_sha256")
    ):
        issues.append(
            RiskIssue(
                "M15_RECOVERY_FORECAST_BINDING_INVALID",
                "M15 recovery forecast direction/calibration/geometry/cycle is not bound to the source receipt and same-cycle history"
                + (f": {forecast_binding_error}" if forecast_binding_error else ""),
            )
        )
    method_value = method.value if method is not None else ""
    lane_binding_valid, lane_binding_error = validate_lane_binding(
        lane_binding,
        forecast_binding=(forecast_binding if isinstance(forecast_binding, dict) else {}),
        pair=intent.pair,
        side=intent.side.value,
        method=method_value,
        order_type=intent.order_type.value,
        entry=intent.entry,
        tp=intent.tp,
        sl=intent.sl,
        current_units=intent.units,
        metadata=metadata,
    )
    if (
        not lane_binding_valid
        or not isinstance(lane_binding, dict)
        or lane_binding.get("contract") != LANE_CONTRACT
        or metadata.get("m15_recovery_lane_contract") != LANE_CONTRACT
        or metadata.get("m15_recovery_lane_binding_sha256")
        != lane_binding.get("binding_sha256")
    ):
        issues.append(
            RiskIssue(
                "M15_RECOVERY_LANE_BINDING_INVALID",
                "M15 recovery exact side/method/vehicle/geometry/producer-units/TP-proof source binding is invalid"
                + (f": {lane_binding_error}" if lane_binding_error else ""),
            )
        )

    quote = snapshot.quotes.get(intent.pair)
    pair_chart, source_evidence, source_issue = _m15_recovery_pair_chart_from_path(
        pair_charts_path,
        expected_pair=intent.pair,
    )
    if source_issue is not None:
        issues.append(source_issue)
    spread_pips: float | None = None
    quote_time: datetime | None = None
    boundary_time = validation_time_utc or snapshot.fetched_at_utc
    if boundary_time.tzinfo is None or boundary_time.utcoffset() is None:
        issues.append(
            RiskIssue(
                "M15_RECOVERY_VALIDATION_TIME_INVALID",
                "M15 recovery validation boundary must use an aware broker/risk clock",
            )
        )
        boundary_time = None
    if quote is None:
        issues.append(
            RiskIssue(
                "M15_RECOVERY_CURRENT_QUOTE_MISSING",
                f"M15 recovery requires a current broker quote for {intent.pair}",
            )
        )
    else:
        quote_time = quote.timestamp_utc
        spread_pips = abs(quote.ask - quote.bid) * instrument_pip_factor(intent.pair)
    receipt_valid = bool(
        pair_chart is not None
        and quote_time is not None
        and boundary_time is not None
        and validate_m15_recovery_micro_receipt(
            receipt,
            pair_chart=pair_chart,
            expected_pair=intent.pair,
            now_utc=boundary_time,
            current_spread_pips=spread_pips,
        )
    )
    if not receipt_valid:
        issues.append(
            RiskIssue(
                "M15_RECOVERY_CURRENT_SOURCE_REVALIDATION_FAILED",
                "M15 recovery receipt no longer matches the canonical current pair chart and broker spread/time",
            )
        )
    blocking = [issue.code for issue in issues if issue.severity == "BLOCK"]
    evidence = {
        **source_evidence,
        "status": "BLOCKED" if blocking else "PASSED",
        "contract": M15_RECOVERY_MICRO_CONTRACT,
        "pair": intent.pair,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": intent.units,
        "producer_units": producer_units,
        "receipt_sha256": receipt_sha,
        "quote_timestamp_utc": quote_time.isoformat() if quote_time is not None else None,
        "validation_time_utc": boundary_time.isoformat() if boundary_time is not None else None,
        "current_spread_pips": spread_pips,
        "positive_rotation_mode": proof_mode,
        "forecast_binding_sha256": (
            forecast_binding.get("binding_sha256")
            if isinstance(forecast_binding, dict)
            else None
        ),
        "lane_binding_sha256": (
            lane_binding.get("binding_sha256")
            if isinstance(lane_binding, dict)
            else None
        ),
        "forecast_history_path": str(forecast_history_path),
        "positive_exact_vehicle_proof_reachable": proof_contract.get("reachable") is True,
        "manual_position_mutation_allowed": False,
        "live_permission_granted": False,
        "blocking_codes": blocking,
    }
    return M15RecoveryMicroValidation(
        applicable=True,
        allowed=not blocking,
        evidence=evidence,
        issues=tuple(issues),
    )


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
    # Floor on per_trade_risk_budget as a fraction of the latest broker NAV.
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
    #     stays intact because gross trader-attributed realized losses consume
    #     non-refillable capacity and fresh open/candidate risk consumes the
    #     remainder (≈10 full-loss trades exhaust a 10% day).
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
    #     95% means the system may use most NAV while leaving 5% headroom for
    #     spread, slippage, and mark-to-market movement.
    # (c) replace via: pass RiskPolicy(max_margin_utilization_pct=...) from
    #     CLI/config when an operator-facing knob is introduced.
    max_margin_utilization_pct: float | None = 95.0
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
    #     the 95% total margin ceiling, forcing headroom for at least one other
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
        specs: Mapping[str, InstrumentSpec] | None = None,
        live_enabled: bool = False,
        validation_time_utc: datetime | None = None,
        guardian_receipt_watchdog_path: Path | None = None,
        guardian_receipt_consumption_path: Path | None = None,
        guardian_receipt_operator_review_path: Path | None = None,
        guardian_receipt_broker_snapshot_path: Path | None = None,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
    ) -> None:
        self.policy = policy or RiskPolicy()
        self.specs = (
            DEFAULT_SPECS
            if specs is None
            else MappingProxyType(dict(specs))
        )
        self.live_enabled = live_enabled
        self.guardian_receipt_watchdog_path = guardian_receipt_watchdog_path
        self.guardian_receipt_consumption_path = guardian_receipt_consumption_path
        self.guardian_receipt_operator_review_path = guardian_receipt_operator_review_path
        self.guardian_receipt_broker_snapshot_path = guardian_receipt_broker_snapshot_path
        self.pair_charts_path = pair_charts_path
        self.forecast_history_path = forecast_history_path
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
        if intent.owner != Owner.TRADER:
            issues.append(RiskIssue("OWNER_NOT_TRADER", f"order owner must be trader, got {intent.owner.value}"))
        if intent.units <= 0:
            issues.append(RiskIssue("BAD_UNITS", f"units must be positive, got {intent.units}"))
        if not intent.thesis.strip():
            issues.append(RiskIssue("MISSING_THESIS", "order intent must carry a non-empty thesis"))
        issues.extend(_hedge_metadata_issues(intent))
        issues.extend(_hedge_balance_issues(intent, snapshot))
        if for_live_send:
            family_block = market_close_leak_family_block_issue(intent)
            if family_block is not None:
                issues.append(
                    RiskIssue(
                        str(family_block["code"]),
                        str(family_block["message"]),
                        str(family_block.get("severity") or "BLOCK"),
                    )
                )
            residual_block = month_scale_residual_metadata_issue(
                {
                    "pair": intent.pair,
                    "side": intent.side.value if isinstance(intent.side, Side) else str(intent.side),
                    "metadata": intent.metadata,
                }
            )
            if residual_block is not None:
                issues.append(
                    RiskIssue(
                        str(residual_block["code"]),
                        str(residual_block["message"]),
                        str(residual_block.get("severity") or "BLOCK"),
                    )
                )
        issues.extend(_forecast_unselected_projection_conflict_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_executable_live_readiness_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_range_method_issues(intent, for_live_send=for_live_send))
        issues.extend(_forecast_range_confidence_issues(intent, for_live_send=for_live_send))
        issues.extend(_guardian_tuning_override_issues(intent, for_live_send=for_live_send))
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

        # The producer receipt is deliberately non-live. RiskEngine must
        # independently bind it back to the current canonical chart bytes and
        # the broker quote used for this exact validation call. Ordinary
        # intents do not enter this branch.
        m15_recovery_validation = validate_m15_recovery_micro_live_claim(
            intent,
            snapshot,
            pair_charts_path=self.pair_charts_path,
            forecast_history_path=self.forecast_history_path,
            validation_time_utc=self._now(),
        )
        issues.extend(m15_recovery_validation.issues)
        if m15_recovery_validation.applicable and m15_recovery_validation.allowed:
            issues.append(
                RiskIssue(
                    "M15_RECOVERY_RISK_REVALIDATED",
                    "RiskEngine independently revalidated the bounded recovery receipt against current canonical chart bytes and broker spread/time; gateway/GPT/Guardian gates remain required",
                    "WARN",
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
        if for_live_send:
            account_payload: dict[str, object] = {}
            current_margin_utilization_pct = None
            margin_available_jpy = None
            if snapshot.account is not None:
                account_payload = {
                    "nav_jpy": snapshot.account.nav_jpy,
                    "margin_used_jpy": snapshot.account.margin_used_jpy,
                    "margin_available_jpy": snapshot.account.margin_available_jpy,
                }
                margin_available_jpy = snapshot.account.margin_available_jpy
                if snapshot.account.nav_jpy > 0:
                    current_margin_utilization_pct = (
                        snapshot.account.margin_used_jpy
                        / snapshot.account.nav_jpy
                        * 100.0
                    )
            issues.extend(
                RiskIssue(
                    str(
                        item.get("code")
                        or "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY"
                    ),
                    str(
                        item.get("message")
                        or "unresolved guardian receipt issue blocks normal new entries"
                    ),
                    str(item.get("severity") or "BLOCK"),
                )
                for item in guardian_receipt_new_entry_blockers_from_paths(
                    watchdog_path=self.guardian_receipt_watchdog_path,
                    consumption_path=self.guardian_receipt_consumption_path,
                    operator_review_path=self.guardian_receipt_operator_review_path,
                    broker_snapshot_path=self.guardian_receipt_broker_snapshot_path,
                    broker_snapshot_payload={"account": account_payload},
                    allow_p1_margin_warning=True,
                    current_margin_utilization_pct=(
                        current_margin_utilization_pct
                    ),
                    projected_margin_utilization_pct=(
                        metrics.margin_utilization_after_pct
                    ),
                    max_margin_utilization_pct=(
                        metrics.max_margin_utilization_pct
                    ),
                    margin_available_jpy=margin_available_jpy,
                )
            )
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
        elif (
            m15_recovery_validation.applicable
            and m15_recovery_validation.allowed
        ):
            # The bounded M15 recovery contract is a STOP-ENTRY, attached-TP
            # BREAKOUT_FAILURE HARVEST shape whose forecast target and
            # invalidation have already been content-addressed and revalidated
            # against current chart/spread evidence above.  Its live
            # 2026-07-14 bounds have a theoretical maximum below 1.2R, so using
            # the generic runner floor makes the only valid shape impossible.
            # Keep the established 1.0R technical-harvest floor; an unbound or
            # stale recovery claim never reaches this branch.
            active_min_rr = self.policy.technical_harvest_min_reward_risk
            floor_label = "technical_harvest"
        elif _uses_technical_harvest_reward_floor(intent):
            active_min_rr = self.policy.technical_harvest_min_reward_risk
            floor_label = "technical_harvest"
        else:
            active_min_rr = self.policy.min_reward_risk
            floor_label = "default"
        evidence_weighted_expectancy_jpy = (
            _m15_recovery_evidence_weighted_expectancy_jpy(
                intent,
                metrics,
                m15_recovery_validation,
            )
            if metrics.reward_risk < active_min_rr
            else None
        )
        if (
            metrics.reward_risk < active_min_rr
            and evidence_weighted_expectancy_jpy is None
        ):
            issues.append(
                RiskIssue(
                    "REWARD_RISK_TOO_LOW",
                    f"planned reward/risk {metrics.reward_risk:.2f}x is below {active_min_rr:.2f}x"
                    + (f" (regime={regime_state})" if regime_state else ""),
                )
            )
        elif (
            metrics.reward_risk < active_min_rr
            and evidence_weighted_expectancy_jpy <= 0.0
        ):
            issues.append(
                RiskIssue(
                    "M15_RECOVERY_NEGATIVE_CURRENT_GEOMETRY_EXPECTANCY",
                    "verified exact-vehicle TP Wilson lower bound applied to "
                    "the current TP/SL geometry is non-positive after charging "
                    "the current spread on both paths: "
                    f"{evidence_weighted_expectancy_jpy:.2f} JPY <= 0; "
                    f"reward/risk={metrics.reward_risk:.2f}x, "
                    f"reward={metrics.reward_jpy:.2f} JPY, "
                    f"risk={metrics.risk_jpy:.2f} JPY, "
                    f"spread={metrics.spread_pips:.2f}pip",
                )
            )
        elif metrics.reward_risk < active_min_rr:
            issues.append(
                RiskIssue(
                    "M15_RECOVERY_EVIDENCE_WEIGHTED_RR",
                    f"planned reward/risk {metrics.reward_risk:.2f}x is below "
                    f"the generic {active_min_rr:.2f}x floor, but the verified "
                    "exact-vehicle TP Wilson lower bound applied to current "
                    "spread-adjusted TP/SL geometry remains positive at "
                    f"{evidence_weighted_expectancy_jpy:.2f} JPY; permit only "
                    "the already bounded recovery size and retain all other gates",
                    severity="WARN",
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
                m15_recovery_validation=m15_recovery_validation,
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
                # This cap is the equity-derived, non-refillable campaign-day
                # loss capacity. It is a hard accounting boundary even under
                # SL-free execution; margin capacity cannot restore loss budget.
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_LOSS_CAP_EXCEEDED",
                        f"open risk {portfolio_risk:.0f} JPY + candidate risk {metrics.risk_jpy:.0f} JPY "
                        f"exceeds portfolio cap {self.policy.max_portfolio_loss_jpy:.0f} JPY",
                        severity="BLOCK",
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

        Both the lane receipt and the policy/gateway may provide a cap.  The
        receipt is allowed to tighten a later broker-truth cap, never widen it:
        final pre-POST reconciliation can discover a same-day loss after the
        intent packet was generated.  Treating the stale metadata value as an
        override would let that old packet exceed the freshly reduced cap.
        When neither source is available, return None so the validator emits
        LOSS_CAP_MISSING rather than inventing a JPY fallback.
        """
        meta = intent.metadata or {}
        caps: list[float] = []
        metadata_cap = meta.get("max_loss_jpy")
        if metadata_cap is not None:
            try:
                cap_value = float(metadata_cap)
            except (TypeError, ValueError):
                return None
            if cap_value <= 0:
                return None
            caps.append(cap_value)
        if self.policy.max_loss_jpy is not None and self.policy.max_loss_jpy > 0:
            caps.append(float(self.policy.max_loss_jpy))
        return min(caps) if caps else None

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
        if _forecast_learning_scout_forward_evidence_supported(intent):
            # The authenticated orientation learner explicitly ranks DIRECT
            # versus INVERSE.  Applying the ordinary point-forecast conflict
            # gate to an INVERSE SCOUT makes that bounded forward-learning
            # route impossible by construction.  The exception is limited to
            # the content-addressed rank decision and its exact passive LIMIT
            # entry/TP/SL binding; all other SCOUT, strategy, spread, margin,
            # Guardian, GPT, and gateway checks still run independently.
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
        m15_recovery_validation: M15RecoveryMicroValidation | None = None,
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
        if _forecast_learning_scout_forward_evidence_supported(intent):
            # Broad pair/direction replay is historical context, not exact
            # technical-vehicle proof.  A canonical, content-addressed,
            # minimum-risk learning SCOUT exists specifically to resolve that
            # uncertainty forward.  Exact technical-negative evidence above
            # still blocks, and the separate SCOUT/gateway validators still
            # enforce LIMIT geometry, attached exits, risk, TTL, cooldown,
            # concurrency, and manual-pair isolation.
            return []
        negative_issue = _bidask_replay_negative_precision_issue_for_intent(intent)
        if negative_issue is not None:
            return [negative_issue]
        if _technical_harvest_precision_support_for_intent(intent) is not None:
            return []
        if _bidask_replay_precision_support_for_intent(intent) is not None:
            return []
        if (
            m15_recovery_validation is not None
            and m15_recovery_validation.applicable
            and m15_recovery_validation.allowed
        ):
            # The current chart bytes, broker spread/time, same-cycle history,
            # forecast/lane bindings, exact TP vehicle proof, and <=999u shape
            # were all checked above in this same RiskEngine invocation.  That
            # lane-specific evidence replaces only the generic confidence /
            # bucket-precision floor; direction, geometry, spread, margin,
            # loss, GPT, Guardian, and final gateway checks remain active.
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
        learning_scout = (
            str(metadata.get("predictive_scout_source") or "").upper()
            == "FORECAST_ORIENTATION_LEARNING"
        )
        forecast_side = Side.LONG if direction == "UP" else Side.SHORT
        if intent.side != forecast_side and not learning_scout:
            return []

        severity = "BLOCK" if for_live_send else "WARN"
        issues: list[RiskIssue] = []
        target_floor = spread_pips * self.policy.min_target_spread_multiple
        invalidation_floor = spread_pips * self.policy.min_stop_spread_multiple

        target_price = _to_float(metadata.get("forecast_target_price"))
        invalidation_price = _to_float(metadata.get("forecast_invalidation_price"))
        if learning_scout:
            receipt = metadata.get("forecast_learning_v1")
            execution_geometry = metadata.get(
                "forecast_learning_execution_geometry_v1"
            )
            method = (
                intent.market_context.method.value
                if intent.market_context is not None
                else ""
            )
            decision_sha = (
                str(receipt.get("decision_sha256") or "")
                if isinstance(receipt, dict)
                else ""
            )
            if not validate_forecast_learning_execution_geometry(
                execution_geometry,
                pair=intent.pair,
                side=intent.side.value,
                method=method,
                entry=entry_price,
                take_profit=_to_float(intent.tp),
                stop_loss=_to_float(intent.sl),
                source_decision_sha256=decision_sha,
            ):
                return [
                    RiskIssue(
                        "FORECAST_LEARNING_EXECUTION_GEOMETRY_INVALID",
                        (
                            f"{intent.pair} {intent.side.value} learning forecast is not bound "
                            "to this exact passive LIMIT entry/TP/SL vehicle; regenerate the "
                            "current technical execution receipt before live send."
                        ),
                        severity=severity,
                    )
                ]
            assert isinstance(execution_geometry, dict)
            target_price = _to_float(
                execution_geometry.get("execution_target_price")
            )
            invalidation_price = _to_float(
                execution_geometry.get("execution_invalidation_price")
            )

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

        cap_jpy = account.nav_jpy * (max_margin_pct / 100.0)
        utilization_room = cap_jpy - account.margin_used_jpy
        # The portfolio hard cap is a state boundary, not merely a budget for
        # positive incremental margin.  A hedging-account offset can make a
        # fresh order's estimated increment zero; that must not let any fresh
        # entry route through when the account is already at/over the cap or
        # the broker reports no available margin.
        if utilization_room <= 0:
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_REACHED",
                    f"current marginUsed {account.margin_used_jpy:.0f} JPY already reaches/exceeds "
                    f"{max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY",
                )
            )
        if account.margin_available_jpy <= 0:
            issues.append(
                RiskIssue(
                    "MARGIN_AVAILABLE_EXHAUSTED",
                    "broker marginAvailable must remain positive for a fresh entry; "
                    f"got {account.margin_available_jpy:.0f} JPY",
                )
            )
        if issues:
            return issues
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
        if metrics.estimated_margin_jpy > utilization_room:
            after = account.margin_used_jpy + metrics.estimated_margin_jpy
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_EXCEEDED",
                    f"candidate margin {metrics.estimated_margin_jpy:.0f} JPY would raise marginUsed to "
                    f"{after:.0f} JPY, above {max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY "
                    f"(remaining budget {utilization_room:.0f} JPY)",
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
