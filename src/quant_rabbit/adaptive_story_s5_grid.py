"""Finite causal story/vehicle diagnostics on exact S5 bid/ask truth.

The story names make multi-timeframe context auditable; they do not give an
LLM, a narrative string, or this historical evaluator order authority.  V2 is
append-only and predeclares ten story policies, one contextual order policy
per story, and five exit policies.  Every decision uses completed UTC buckets
only, observes the first real S5 quote after the trigger, and can fill no
earlier than the following real S5 candle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_rabbit.causal_multitf_s5_grid import (
    TIMEFRAMES_SECONDS,
    UtcSplit,
    _Bar,
    _Bucket,
    _Feature,
    _feature,
    _new_bucket,
    _normalise_splits,
    _split_for,
    _validate_candle,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


STORY_GRID_CONTRACT_V2 = "QR_ADAPTIVE_STORY_S5_GRID_V2"
STORY_GRID_COMBINED_CONTRACT_V2 = "QR_ADAPTIVE_STORY_S5_GRID_COMBINED_V2"
CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1 = (
    "QR_ADAPTIVE_STORY_CURRENCY_TRIGGER_M1_FACTOR_CLUSTERS_V1"
)
STORY_CATALOG_POLICY_V2 = "PREDECLARED_CAUSAL_STORY_CONTEXTUAL_ORDER_V2"
STORY_TRUTH_POLICY_V2 = "EXACT_S5_BID_ASK_EQUAL_INITIAL_R_V2"
CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1 = (
    "SPLIT_CANDIDATE_EXACT_TRIGGER_M1_UTC_CURRENCY_SIGN_V1"
)
CURRENCY_FACTOR_VIEW_POLICY_V1 = "LONG_BASE_PLUS_QUOTE_MINUS_SHORT_INVERSE_V1"
EXIT_POLICY_IDS = (
    "PROFIT_FIRST_24H",
    "TIME_1H",
    "TIME_4H",
    "TIME_24H",
    "TRAILING_STRUCTURAL_24H",
)

# This is an ex-ante research TTL for STOP/LIMIT observation, chosen to cover
# one M15 execution window.  It is not a production timeout or tuned result;
# changing it requires a new append-only catalog version.
ENTRY_TTL_SECONDS = 15 * 60
# The longest predeclared exit observes one complete UTC day.  It is fixed to
# define split embargo and not inferred from inspected outcomes.
MAX_HOLD_SECONDS = 24 * 60 * 60
# Two executable spread widths stress entry plus exit cost.  It is a symmetric
# cost feasibility check, never a directional input; a new stress model must be
# introduced under a new policy version.
SPREAD_STRESS_ROUND_TRIPS = 2.0
# STOP and LIMIT offsets are fixed fractions of the already-frozen M5 ATR so
# the order rests outside/inside immediate noise without using fixed pips.
STOP_OFFSET_ATR = 0.10
LIMIT_OFFSET_ATR = 0.25
# A market order is allowed only for a completed M1 body at least one frozen
# M5 ATR.  This is a preregistered high-impulse class boundary, not an adaptive
# outcome-tuned threshold.
MARKET_IMPULSE_MIN_ATR = 1.0
# All structural stops retain at least one frozen M5 ATR of room.  The market
# structure may make them wider; this floor cannot be retuned inside V2.
STRUCTURAL_STOP_ATR_FLOOR = 1.0
# Bounded diagnostic rows prevent multi-month streams from being copied into
# artifacts. Aggregate statistics remain complete after this audit sample cap.
MAX_AUDIT_ROWS = 1024
# The time-shift counterfactual moves one complete M1 decision boundary.  It is
# an audit-only shadow label, never a selectable or retrospectively scored arm.
TIME_SHIFT_SHADOW_SECONDS = 60
# Completed-history context is capped above the longest V1 EMA/breakout
# lookback (50 + 20 bars). This bounds memory without changing a story gate.
FEATURE_HISTORY_BAR_CAP = 80
# Story V2 uses one half-M5-ATR completed-body threshold for an established
# impulse and one full ATR for the exceptional MARKET class. These are frozen
# ex-ante class boundaries; later values require a new story catalog version.
STORY_IMPULSE_MIN_ATR = 0.50
CLIMAX_IMPULSE_MIN_ATR = 1.50
# Compression requires eight strictly earlier completed M15 ATR observations
# and the setup ATR at or below their lower quartile. Eight is the smallest
# bounded history that gives two observations to that quartile; future work
# may add a new version with a longer distribution, never retune this one.
COMPRESSION_MIN_PRIOR_ATR_OBSERVATIONS = 8
COMPRESSION_ATR_QUANTILE = 0.25
# The range story treats ADX below 25 as non-trending and uses the outer
# quartiles as rails. These conventional diagnostic boundaries are frozen in
# V2 rather than tuned from its results.
RANGE_ADX_MAX = 25.0
# Opening stories use local 08:00-08:15 in Europe/London and America/New_York.
# ZoneInfo supplies DST conversion; a future session-calendar study must add a
# new version rather than replace these local-time definitions in place.
SESSION_LOCAL_OPEN_HOUR = 8
SESSION_OPEN_WINDOW_MINUTES = 15
SESSION_TIMEZONES = (ZoneInfo("Europe/London"), ZoneInfo("America/New_York"))
# These validation-screen floors are frozen economic consistency checks, not
# statistical proof or live promotion. They require 30 resolved trades, eight
# active UTC days, and four contributing pairs before a pooled story can be
# called an economic-screen survivor.
SCREEN_MIN_RESOLVED_TRADES = 30
SCREEN_MIN_ACTIVE_DAYS = 8
SCREEN_MIN_CONTRIBUTING_PAIRS = 4
SCREEN_MIN_PROFIT_FACTOR_R = 1.0
# OANDA quotes the supported FX instruments at one tenth of a pip.  This is a
# broker price-grid fact rather than a tuned market threshold; a different
# venue must replace it through a new evaluator policy/version.
BROKER_TICKS_PER_PIP = 10
# Binary floats can represent an exact decimal story threshold a few ulps
# below its intended boundary.  This computational tolerance is constant (not
# market-derived) solely to preserve equality at preregistered cutoffs; a
# Decimal-native feature pipeline should replace it if price features migrate.
FLOAT_BOUNDARY_ABS_TOL = 1e-12
# Pooling identical daily values in pair/day order can reassociate binary-float
# sums.  This constant accepts only machine-level reassociation, not economic
# drift; exact Decimal aggregate storage should replace it if introduced.
AGGREGATE_REASSOCIATION_TOL = 1e-12
# Every FX trade has exactly two currency-leg views: one base leg and one quote
# leg.  Two is a market-identity invariant rather than a tuned threshold; a
# non-FX instrument model must introduce a new factor-view policy.
CURRENCY_FACTOR_VIEWS_PER_TRADE = 2
PAIR_RUN_ALLOWED_STATUSES = ("COMPLETE", "NO_DATA")

PRICE_PRECISION_POLICY_V2 = "BROKER_TICK_ONE_TENTH_PIP_DIRECTION_PRESERVING_V2"
PRICE_COST_SCOPE_V2: dict[str, Any] = {
    "entry_and_exit_spread": "EXACT_S5_BID_ASK",
    "resting_entry_slippage": "EXACT_OPEN_GAP_OR_FROZEN_TRIGGER_PRICE",
    "market_entry_slippage": "NEXT_REAL_S5_EXECUTABLE_OPEN",
    "barrier_exit_slippage": "FROZEN_BARRIER_EXCEPT_EXECUTABLE_OPEN_GAP",
    "time_exit_slippage": "FIRST_REAL_S5_EXECUTABLE_OPEN_AT_OR_AFTER_DUE",
    "broker_on_fill_dependent_order_loss_cancel": "NO_FILL_NO_PNL",
    "explicit_commission": "NOT_MODELED",
    "financing_and_swap": "NOT_MODELED",
    "market_impact": "NOT_MODELED",
    "broker_tick_policy": PRICE_PRECISION_POLICY_V2,
}

_AUTHORITY: dict[str, Any] = {
    "historical_only": True,
    "diagnostic_only": True,
    "shadow_only": True,
    "order_authority": "NONE",
    "forward_proof_eligible": False,
    "live_permission": False,
    "live_order_enabled": False,
    "promotion_allowed": False,
    "automatic_promotion_allowed": False,
    "broker_mutation_allowed": False,
}


@dataclass(frozen=True, slots=True)
class StoryTemplateV2:
    """One causal setup/trigger narrative with bounded order modes."""

    hypothesis_id: str
    story_name: str
    setup_role: str
    trigger_role: str
    allowed_order_modes: tuple[str, ...]
    ordinary_order_mode: str
    market_on_high_impulse: bool
    no_trade_control: bool = False


@dataclass(frozen=True, slots=True)
class StoryExitV2:
    """One predeclared exit policy sharing the same contextual entry."""

    exit_policy_id: str
    max_hold_seconds: int
    profit_target_r: float | None
    trailing_structural: bool
    complexity: int


@dataclass(frozen=True, slots=True)
class StoryVehicleV2:
    """One selectable story-by-exit candidate or the H31 zero control."""

    candidate_id: str
    hypothesis_id: str
    story_name: str
    exit_policy_id: str
    contextual_order_policy: str
    allowed_order_modes: tuple[str, ...]
    max_hold_seconds: int
    profit_target_r: float | None
    trailing_structural: bool
    complexity: int
    no_trade_control: bool = False


def build_story_templates_v2() -> tuple[StoryTemplateV2, ...]:
    """Return the immutable H21-H31 causal story catalog."""

    return (
        StoryTemplateV2(
            "H21",
            "PULLBACK_CONTINUATION",
            "PRIOR_M1_PULLBACK_INSIDE_M15_H1_TREND",
            "M1_RECLAIM_OF_PRIOR_PULLBACK_EXTREME",
            ("LIMIT",),
            "LIMIT",
            False,
        ),
        StoryTemplateV2(
            "H22",
            "CHOCH_REVERSAL",
            "PRIOR_M15_TREND_AND_M1_EXHAUSTION",
            "M1_BREAK_OF_PRIOR_COUNTER_SWING",
            ("STOP",),
            "STOP",
            False,
        ),
        StoryTemplateV2(
            "H23",
            "RANGE_SWEEP_REVERSION",
            "PRIOR_M15_LOW_ADX_WITH_FROZEN_M15_RAILS",
            "M1_SWEEP_AND_REACCEPTANCE",
            ("LIMIT",),
            "LIMIT",
            False,
        ),
        StoryTemplateV2(
            "H24",
            "RANGE_BREAK_RETEST",
            "PRIOR_M15_RANGE_AND_COMPLETED_BREAK",
            "M1_RETEST_HOLDS_FROZEN_M15_RAIL",
            ("LIMIT",),
            "LIMIT",
            False,
        ),
        StoryTemplateV2(
            "H25",
            "COMPRESSION_RELEASE",
            "PRIOR_M15_ATR_IN_LOWER_PRIOR_ONLY_QUANTILE",
            "M1_BREAK_WITH_M5_ATR_EXPANSION",
            ("STOP", "MARKET"),
            "STOP",
            True,
        ),
        StoryTemplateV2(
            "H26",
            "FALSE_COMPRESSION_RELEASE",
            "PRIOR_M15_COMPRESSION_AND_BREAK_ATTEMPT",
            "M1_FALSE_BREAK_REACCEPTANCE",
            ("STOP",),
            "STOP",
            False,
        ),
        StoryTemplateV2(
            "H27",
            "IMPULSE_PAUSE_CONTINUATION",
            "PRIOR_M1_IMPULSE_WITH_M5_DIRECTION",
            "M1_SHALLOW_PAUSE_HOLDS_HALF_IMPULSE",
            ("LIMIT",),
            "LIMIT",
            False,
        ),
        StoryTemplateV2(
            "H28",
            "CLIMAX_FADE",
            "PRIOR_M1_CLIMAX_BODY_AND_M15_DECELERATION",
            "M1_COUNTER_BREAK",
            ("STOP",),
            "STOP",
            False,
        ),
        StoryTemplateV2(
            "H29",
            "DST_OPENING_BREAK",
            "LOCAL_DST_AWARE_OPEN_AND_PRIOR_M15_RANGE",
            "M1_OPENING_RANGE_BREAK",
            ("STOP", "MARKET"),
            "STOP",
            True,
        ),
        StoryTemplateV2(
            "H30",
            "DST_OPENING_FAILED_BREAK",
            "LOCAL_DST_AWARE_OPEN_AND_PRIOR_M15_RANGE",
            "M1_OPENING_FALSE_BREAK_REACCEPTANCE",
            ("STOP",),
            "STOP",
            False,
        ),
        StoryTemplateV2(
            "H31",
            "NO_TRADE_CONTROL",
            "CONTROL",
            "NONE",
            (),
            "NONE",
            False,
            True,
        ),
    )


def _exit_catalog_v2() -> tuple[StoryExitV2, ...]:
    return (
        StoryExitV2("PROFIT_FIRST_24H", MAX_HOLD_SECONDS, None, False, 1),
        StoryExitV2("TIME_1H", 60 * 60, None, False, 0),
        StoryExitV2("TIME_4H", 4 * 60 * 60, None, False, 1),
        StoryExitV2("TIME_24H", MAX_HOLD_SECONDS, None, False, 2),
        StoryExitV2("TRAILING_STRUCTURAL_24H", MAX_HOLD_SECONDS, None, True, 3),
    )


def build_story_vehicle_catalog_v2() -> tuple[StoryVehicleV2, ...]:
    """Return exactly 50 contextual-order candidates plus H31 control."""

    rows: list[StoryVehicleV2] = []
    for template in build_story_templates_v2():
        if template.no_trade_control:
            rows.append(
                StoryVehicleV2(
                    candidate_id="H31:NO_TRADE_CONTROL",
                    hypothesis_id="H31",
                    story_name=template.story_name,
                    exit_policy_id="NO_TRADE_CONTROL",
                    contextual_order_policy="NO_ORDER",
                    allowed_order_modes=(),
                    max_hold_seconds=0,
                    profit_target_r=None,
                    trailing_structural=False,
                    complexity=0,
                    no_trade_control=True,
                )
            )
            continue
        for exit_policy in _exit_catalog_v2():
            rows.append(
                StoryVehicleV2(
                    candidate_id=(
                        f"{template.hypothesis_id}:{exit_policy.exit_policy_id}"
                    ),
                    hypothesis_id=template.hypothesis_id,
                    story_name=template.story_name,
                    exit_policy_id=exit_policy.exit_policy_id,
                    contextual_order_policy="STORY_CONTEXT_SELECTS_ONE_ORDER_MODE_V2",
                    allowed_order_modes=template.allowed_order_modes,
                    max_hold_seconds=exit_policy.max_hold_seconds,
                    profit_target_r=exit_policy.profit_target_r,
                    trailing_structural=exit_policy.trailing_structural,
                    complexity=exit_policy.complexity,
                )
            )
    return tuple(rows)


@dataclass(slots=True)
class _StoryDecision:
    hypothesis_id: str
    side: str
    setup_at: datetime
    trigger_at: datetime
    structural_target: float | None
    structural_stop_anchor: float | None
    setup_evidence: dict[str, Any]
    trigger_evidence: dict[str, Any]

    @property
    def scorable(self) -> bool:
        if self.structural_target is None or not math.isfinite(self.structural_target):
            return False
        trigger_close = self.trigger_evidence.get("trigger_close")
        if not isinstance(trigger_close, (int, float)) or isinstance(
            trigger_close, bool
        ):
            return False
        return (
            self.structural_target > float(trigger_close)
            if self.side == "LONG"
            else self.structural_target < float(trigger_close)
        )


@dataclass(slots=True)
class _StoryPosition:
    candidate: StoryVehicleV2
    split_name: str
    side: str
    setup_at: datetime
    trigger_at: datetime
    direct_story_side: str
    frozen_m1: _Feature
    frozen_m5: _Feature
    frozen_structural_target: float
    frozen_structural_stop_anchor: float | None
    chosen_order_mode: str
    order_selection_reason: str
    quote_observed_at: datetime | None = None
    observed_bid: float | None = None
    observed_ask: float | None = None
    entry_target: float | None = None
    structural_stop: float | None = None
    order_creation_geometry_issue: str | None = None
    cost_gate_passed: bool | None = None
    cost_gate_evidence: dict[str, Any] | None = None
    filled_at: datetime | None = None
    entry_exec: float | None = None
    entry_mid: float | None = None
    entry_fill_kind: str | None = None
    initial_risk_price: float | None = None
    take_profit: float | None = None
    exit_due_at: datetime | None = None
    trail_stop: float | None = None


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CurrencyTriggerClusterKey = tuple[str, str, str, str, int]
_CurrencyTriggerClusterMap = dict[_CurrencyTriggerClusterKey, dict[str, dict[str, Any]]]


def _parse_canonical_utc_instant(
    value: Any,
    *,
    field: str,
    exact_minute: bool = False,
) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a canonical UTC instant")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a canonical UTC instant") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be UTC")
    canonical = parsed.astimezone(timezone.utc)
    if value != canonical.isoformat():
        raise ValueError(f"{field} must use canonical UTC ISO format")
    if exact_minute and (canonical.second != 0 or canonical.microsecond != 0):
        raise ValueError(f"{field} must be an exact UTC minute")
    return canonical


def _pair_currency_factor_views(
    pair: str,
    side: str,
) -> tuple[tuple[str, int], tuple[str, int]]:
    if not _is_canonical_pair_name(pair) or pair[:3] == pair[4:]:
        raise ValueError("factor pair must be canonical distinct-currency AAA_BBB")
    if side == "LONG":
        return ((pair[:3], 1), (pair[4:], -1))
    if side == "SHORT":
        return ((pair[:3], -1), (pair[4:], 1))
    raise ValueError("factor trade side must be LONG or SHORT")


def _trade_identity_body(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": member["candidate_id"],
        "pair": member["pair"],
        "side": member["side"],
        "setup_at_utc": member["setup_at_utc"],
        "trigger_at_utc": member["trigger_at_utc"],
        "quote_observed_at_utc": member["quote_observed_at_utc"],
        "entry_at_utc": member["entry_at_utc"],
        "entry_exec": member["entry_exec"],
        "exit_policy_id": member["exit_policy_id"],
    }


def _validated_factor_member(
    member: Mapping[str, Any],
    *,
    expected_candidate_id: str,
    expected_pair: str | None,
) -> dict[str, Any]:
    expected_fields = {
        "trade_identity_sha256",
        "candidate_id",
        "pair",
        "side",
        "setup_at_utc",
        "trigger_at_utc",
        "quote_observed_at_utc",
        "entry_at_utc",
        "entry_exec",
        "exit_policy_id",
        "exact_net_r",
    }
    if not isinstance(member, Mapping) or set(member) != expected_fields:
        raise ValueError("currency-trigger factor member schema mismatch")
    candidate_id = member.get("candidate_id")
    if candidate_id != expected_candidate_id:
        raise ValueError("currency-trigger factor member candidate mismatch")
    pair = member.get("pair")
    if not isinstance(pair, str) or not _is_canonical_pair_name(pair):
        raise ValueError("currency-trigger factor member pair is noncanonical")
    if expected_pair is not None and pair != expected_pair:
        raise ValueError("currency-trigger factor member pair scope mismatch")
    side = member.get("side")
    if not isinstance(side, str):
        raise ValueError("currency-trigger factor member side is invalid")
    _pair_currency_factor_views(pair, side)
    setup_at = _parse_canonical_utc_instant(
        member.get("setup_at_utc"), field="factor member setup_at_utc"
    )
    trigger_at = _parse_canonical_utc_instant(
        member.get("trigger_at_utc"),
        field="factor member trigger_at_utc",
        exact_minute=True,
    )
    quote_at = _parse_canonical_utc_instant(
        member.get("quote_observed_at_utc"),
        field="factor member quote_observed_at_utc",
    )
    entry_at = _parse_canonical_utc_instant(
        member.get("entry_at_utc"), field="factor member entry_at_utc"
    )
    if not setup_at < trigger_at < quote_at < entry_at:
        raise ValueError("currency-trigger factor member causal clocks are invalid")
    entry_exec = _require_finite_number(
        member.get("entry_exec"), field="factor member entry_exec"
    )
    if entry_exec <= 0.0:
        raise ValueError("currency-trigger factor member entry_exec must be positive")
    exact_net_r = _require_finite_number(
        member.get("exact_net_r"), field="factor member exact_net_r"
    )
    exit_policy_id = member.get("exit_policy_id")
    if not isinstance(exit_policy_id, str) or not exit_policy_id:
        raise ValueError("currency-trigger factor member exit policy is invalid")
    expected_exit_policy_id = expected_candidate_id.partition(":")[2]
    if not expected_exit_policy_id or exit_policy_id != expected_exit_policy_id:
        raise ValueError("currency-trigger factor member exit policy mismatch")
    normalized = {
        "trade_identity_sha256": member["trade_identity_sha256"],
        "candidate_id": candidate_id,
        "pair": pair,
        "side": side,
        "setup_at_utc": setup_at.isoformat(),
        "trigger_at_utc": trigger_at.isoformat(),
        "quote_observed_at_utc": quote_at.isoformat(),
        "entry_at_utc": entry_at.isoformat(),
        "entry_exec": entry_exec,
        "exit_policy_id": exit_policy_id,
        "exact_net_r": exact_net_r,
    }
    identity = normalized["trade_identity_sha256"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or identity != _canonical_sha(_trade_identity_body(normalized))
    ):
        raise ValueError("currency-trigger factor trade identity is invalid")
    return normalized


def _factor_member_from_trade(
    pair: str,
    trade: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = trade.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("resolved factor trade candidate is invalid")
    provisional = {
        "trade_identity_sha256": "",
        "candidate_id": candidate_id,
        "pair": pair,
        "side": trade.get("side"),
        "setup_at_utc": trade.get("setup_at_utc"),
        "trigger_at_utc": trade.get("trigger_at_utc"),
        "quote_observed_at_utc": trade.get("quote_observed_at_utc"),
        "entry_at_utc": trade.get("entry_at_utc"),
        "entry_exec": trade.get("entry_exec"),
        "exit_policy_id": trade.get("exit_policy_id"),
        "exact_net_r": trade.get("exact_net_r"),
    }
    provisional["trade_identity_sha256"] = _canonical_sha(
        _trade_identity_body(provisional)
    )
    return _validated_factor_member(
        provisional,
        expected_candidate_id=candidate_id,
        expected_pair=pair,
    )


def _record_currency_trigger_factor_trade(
    pair: str,
    trade: Mapping[str, Any],
    clusters: _CurrencyTriggerClusterMap,
    ownership: dict[str, tuple[_CurrencyTriggerClusterKey, ...]],
) -> str:
    member = _factor_member_from_trade(pair, trade)
    identity = str(member["trade_identity_sha256"])
    if identity in ownership:
        raise ValueError("duplicate resolved factor trade identity")
    split_name = trade.get("split")
    if not isinstance(split_name, str) or not split_name:
        raise ValueError("resolved factor trade split is invalid")
    candidate_id = str(member["candidate_id"])
    trigger_at = str(member["trigger_at_utc"])
    factor_keys = tuple(
        (split_name, candidate_id, trigger_at, currency, sign)
        for currency, sign in _pair_currency_factor_views(pair, str(member["side"]))
    )
    if len(set(factor_keys)) != CURRENCY_FACTOR_VIEWS_PER_TRADE:
        raise ValueError("resolved factor trade does not own two distinct factor views")
    for key in factor_keys:
        cluster = clusters.setdefault(key, {})
        if identity in cluster:
            raise ValueError("duplicate trade identity inside factor cluster")
        cluster[identity] = dict(member)
    ownership[identity] = factor_keys
    return identity


def _currency_trigger_cluster_key_from_row(
    row: Mapping[str, Any],
) -> _CurrencyTriggerClusterKey:
    return (
        str(row["split"]),
        str(row["candidate_id"]),
        str(row["trigger_at_utc"]),
        str(row["currency"]),
        int(row["sign"]),
    )


def _currency_trigger_cluster_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "candidate_id": row["candidate_id"],
        "trigger_at_utc": row["trigger_at_utc"],
        "currency": row["currency"],
        "sign": row["sign"],
        "member_count": row["member_count"],
        "exact_net_r": row["exact_net_r"],
        "member_identity_digest": row["member_identity_digest"],
    }


def _currency_trigger_metric_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_total_exact_net_r: float,
    cluster_digest: str,
) -> dict[str, Any]:
    top_row = (
        min(
            rows,
            key=lambda row: (
                -float(row["exact_net_r"]),
                _currency_trigger_cluster_key_from_row(row),
            ),
        )
        if rows
        else None
    )
    return {
        "currency_trigger_factor_cluster_count": len(rows),
        "currency_trigger_min_leave_one_cluster_out_total_r": (
            min(candidate_total_exact_net_r - float(row["exact_net_r"]) for row in rows)
            if rows
            else None
        ),
        "currency_trigger_top_cluster": (
            _currency_trigger_cluster_summary(top_row) if top_row is not None else None
        ),
        "currency_trigger_cluster_digest": cluster_digest,
    }


def _build_currency_trigger_cluster_payload(
    candidate_id: str,
    split_name: str,
    clusters: Mapping[_CurrencyTriggerClusterKey, Mapping[str, Mapping[str, Any]]],
    ownership: Mapping[str, tuple[_CurrencyTriggerClusterKey, ...]],
    *,
    resolved_trade_count: int,
    candidate_total_exact_net_r: float,
    expected_pair: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(clusters):
        if key[0] != split_name or key[1] != candidate_id:
            continue
        members_by_id = clusters[key]
        members = [dict(members_by_id[identity]) for identity in sorted(members_by_id)]
        identities = [str(member["trade_identity_sha256"]) for member in members]
        rows.append(
            {
                "split": key[0],
                "candidate_id": key[1],
                "trigger_at_utc": key[2],
                "currency": key[3],
                "sign": key[4],
                "member_count": len(members),
                "exact_net_r": sum(float(member["exact_net_r"]) for member in members),
                "member_identity_digest": _canonical_sha(identities),
                "members": members,
            }
        )
    cluster_digest = _canonical_sha(rows)
    factor_membership_count = sum(int(row["member_count"]) for row in rows)
    factor_exact_net_r_sum = sum(float(row["exact_net_r"]) for row in rows)
    payload: dict[str, Any] = {
        "contract": CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1,
        "schema_version": 1,
        "cluster_key_policy": CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1,
        "factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
        "split": split_name,
        "candidate_id": candidate_id,
        "cluster_rows_untruncated": True,
        "factor_clusters_are_not_candidate_summed": True,
        "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
        "resolved_trade_count": resolved_trade_count,
        "factor_membership_count": factor_membership_count,
        "candidate_total_exact_net_r": candidate_total_exact_net_r,
        "factor_membership_exact_net_r_sum_for_ownership_check": (
            factor_exact_net_r_sum
        ),
        "cluster_rows": rows,
        "cluster_rows_sha256": cluster_digest,
        "ownership_invariant_passed": True,
    }
    metric_fields = _currency_trigger_metric_fields(
        rows,
        candidate_total_exact_net_r=candidate_total_exact_net_r,
        cluster_digest=cluster_digest,
    )
    _validate_currency_trigger_cluster_payload(
        payload,
        expected_candidate_id=candidate_id,
        expected_split_name=split_name,
        expected_pair=expected_pair,
        expected_resolved_trade_count=resolved_trade_count,
        expected_candidate_total_exact_net_r=candidate_total_exact_net_r,
        expected_metric_fields=metric_fields,
        expected_ownership=ownership,
    )
    return payload, metric_fields


def _story_catalog_receipt_v2() -> dict[str, Any]:
    return {
        "story_catalog_policy": STORY_CATALOG_POLICY_V2,
        "templates": [
            {
                "hypothesis_id": row.hypothesis_id,
                "story_name": row.story_name,
                "setup_role": row.setup_role,
                "trigger_role": row.trigger_role,
                "allowed_order_modes": list(row.allowed_order_modes),
                "ordinary_order_mode": row.ordinary_order_mode,
                "market_on_high_impulse": row.market_on_high_impulse,
                "no_trade_control": row.no_trade_control,
            }
            for row in build_story_templates_v2()
        ],
        "vehicles": [
            {
                "candidate_id": row.candidate_id,
                "hypothesis_id": row.hypothesis_id,
                "story_name": row.story_name,
                "exit_policy_id": row.exit_policy_id,
                "contextual_order_policy": row.contextual_order_policy,
                "allowed_order_modes": list(row.allowed_order_modes),
                "max_hold_seconds": row.max_hold_seconds,
                "profit_target_r": row.profit_target_r,
                "trailing_structural": row.trailing_structural,
                "complexity": row.complexity,
                "no_trade_control": row.no_trade_control,
            }
            for row in build_story_vehicle_catalog_v2()
        ],
    }


def _truth_evaluator_receipt_v2() -> dict[str, Any]:
    return {
        "truth_policy": STORY_TRUTH_POLICY_V2,
        "entry_ttl_seconds_exclusive": ENTRY_TTL_SECONDS,
        "maximum_hold_seconds": MAX_HOLD_SECONDS,
        "spread_stress_round_trips": SPREAD_STRESS_ROUND_TRIPS,
        "stop_offset_atr": STOP_OFFSET_ATR,
        "limit_offset_atr": LIMIT_OFFSET_ATR,
        "market_impulse_min_atr": MARKET_IMPULSE_MIN_ATR,
        "story_impulse_min_atr": STORY_IMPULSE_MIN_ATR,
        "h22_reversal_impulse_min_atr": STORY_IMPULSE_MIN_ATR,
        "frozen_threshold_float_boundary_abs_tolerance": FLOAT_BOUNDARY_ABS_TOL,
        "climax_impulse_min_atr": CLIMAX_IMPULSE_MIN_ATR,
        "structural_stop_atr_floor": STRUCTURAL_STOP_ATR_FLOOR,
        "compression_min_prior_atr_observations": (
            COMPRESSION_MIN_PRIOR_ATR_OBSERVATIONS
        ),
        "compression_atr_quantile": COMPRESSION_ATR_QUANTILE,
        "range_adx_max_exclusive": RANGE_ADX_MAX,
        "session_local_open_hour": SESSION_LOCAL_OPEN_HOUR,
        "session_open_window_minutes": SESSION_OPEN_WINDOW_MINUTES,
        "session_timezones": [str(zone) for zone in SESSION_TIMEZONES],
        "session_open_window_end_clock_policy": "LOCAL_(08:00,08:15]",
        "quote_observation_policy": (
            "FIRST_REAL_S5_AFTER_TRIGGER_OBSERVES_ONLY;"
            "FOLLOWING_REAL_S5_IS_EARLIEST_FILL"
        ),
        "entry_execution_side": {"LONG": "ASK", "SHORT": "BID"},
        "exit_execution_side": {"LONG": "BID", "SHORT": "ASK"},
        "same_s5_barrier_policy": "STRUCTURAL_STOP_FIRST",
        "barrier_open_gap_policy": (
            "STOP_GAP_FIRST;THEN_TARGET_GAP_AT_EXECUTABLE_OPEN;"
            "THEN_INTRABAR_STOP_FIRST"
        ),
        "mandatory_time_close_policy": "FIRST_REAL_S5_OPEN_AT_OR_AFTER_DUE",
        "trailing_input_policy": "COMPLETED_M1_AFTER_FILL_ONLY",
        "split_embargo_seconds": ENTRY_TTL_SECONDS + MAX_HOLD_SECONDS,
        "gross_mid_decomposition_policy": (
            "ONLY_WHEN_ENTRY_AND_EXIT_HAVE_SYNCHRONIZED_EXECUTABLE_OPENS"
        ),
        "resting_fill_s5_policy": (
            "NO_TARGET;STOP_RANGE_CHARGED_CONSERVATIVELY;NO_PREFILL_OPEN_GAP"
        ),
        "entry_gap_invalid_geometry_policy": (
            "BROKER_ON_FILL_DEPENDENT_ORDER_LOSS_CANCEL_NO_FILL"
        ),
        "price_precision_policy": PRICE_PRECISION_POLICY_V2,
        "broker_ticks_per_pip": BROKER_TICKS_PER_PIP,
        "resting_entry_tick_rounding": {
            "LONG_LIMIT": "FLOOR",
            "SHORT_LIMIT": "CEILING",
            "LONG_STOP": "CEILING",
            "SHORT_STOP": "FLOOR",
        },
        "structural_barrier_tick_rounding": {
            "LONG_STOP": "FLOOR",
            "LONG_TARGET": "CEILING",
            "SHORT_STOP": "CEILING",
            "SHORT_TARGET": "FLOOR",
        },
        "trailing_candidate_tick_rounding": {
            "LONG": "FLOOR_THEN_MAX",
            "SHORT": "CEILING_THEN_MIN",
        },
        "price_cost_scope": dict(PRICE_COST_SCOPE_V2),
        "currency_trigger_factor_clusters": {
            "contract": CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1,
            "cluster_key_policy": CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1,
            "cluster_key_fields": [
                "split",
                "candidate_id",
                "trigger_at_utc",
                "currency",
                "sign",
            ],
            "factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
            "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
            "trade_identity_fields": [
                "candidate_id",
                "pair",
                "side",
                "setup_at_utc",
                "trigger_at_utc",
                "quote_observed_at_utc",
                "entry_at_utc",
                "entry_exec",
                "exit_policy_id",
            ],
            "trigger_clock_policy": "CANONICAL_EXACT_UTC_MINUTE",
            "trade_identity_policy": "UNIQUE_SHA256_CANONICAL_BODY",
            "accumulation_policy": "BEFORE_BOUNDED_AUDIT_ROW_CAP",
            "cluster_rows_untruncated": True,
            "factor_clusters_are_not_candidate_summed": True,
            "factor_membership_total_policy": (
                "VERIFY_ONLY_EQUALS_TWO_TIMES_CANDIDATE_TOTAL_R"
            ),
            "leave_one_cluster_out_policy": (
                "CANDIDATE_TOTAL_R_MINUS_FACTOR_CLUSTER_R"
            ),
            "top_cluster_policy": ("MAX_EXACT_NET_R_THEN_CANONICAL_CLUSTER_KEY"),
            "global_merge_policy": (
                "MERGE_EQUAL_CLUSTER_KEYS_ACROSS_PAIRS;"
                "REJECT_CROSS_PAIR_DUPLICATE_TRADE_IDENTITIES"
            ),
            "economic_gate": (
                "leave_one_currency_trigger_cluster_min_total_r_positive"
            ),
            "economic_gate_boundary": "STRICTLY_GREATER_THAN_ZERO",
        },
        "economic_screen": {
            "minimum_resolved_trades": SCREEN_MIN_RESOLVED_TRADES,
            "minimum_active_days": SCREEN_MIN_ACTIVE_DAYS,
            "minimum_contributing_pairs": SCREEN_MIN_CONTRIBUTING_PAIRS,
            "minimum_profit_factor_r_exclusive": SCREEN_MIN_PROFIT_FACTOR_R,
            "unresolved_filled_count_required": 0,
            "leave_one_currency_trigger_cluster_min_total_r_required_positive": (True),
        },
    }


def _opposite(side: str) -> str:
    return "SHORT" if side == "LONG" else "LONG"


def _is_canonical_pair_name(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 7
        and value[3] == "_"
        and all("A" <= character <= "Z" for character in (*value[:3], *value[4:]))
    )


def _trend_side(feature: _Feature | None) -> str | None:
    if (
        feature is None
        or feature.ema_fast is None
        or feature.ema_slow is None
        or feature.plus_di is None
        or feature.minus_di is None
    ):
        return None
    if feature.ema_fast > feature.ema_slow and feature.plus_di > feature.minus_di:
        return "LONG"
    if feature.ema_fast < feature.ema_slow and feature.minus_di > feature.plus_di:
        return "SHORT"
    return None


def _completed_impulse_ratio(m1: _Feature, m5: _Feature) -> float:
    if m5.atr is None or m5.atr <= 0.0:
        return 0.0
    return abs(m1.close - m1.current_open) / m5.atr


def _meets_frozen_threshold(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=FLOAT_BOUNDARY_ABS_TOL
    )


def _prior_atr_quantile(
    history: Sequence[_Feature],
    *,
    before: datetime,
) -> float | None:
    values = sorted(
        float(row.atr)
        for row in history
        if row.completed_at < before and row.atr is not None and row.atr > 0.0
    )
    if len(values) < COMPRESSION_MIN_PRIOR_ATR_OBSERVATIONS:
        return None
    index = int((len(values) - 1) * COMPRESSION_ATR_QUANTILE)
    return values[index]


def _is_dst_aware_local_open(trigger_at: datetime) -> tuple[bool, str | None]:
    for zone in SESSION_TIMEZONES:
        local = trigger_at.astimezone(zone)
        # ``trigger_at`` is the completed M1 end clock.  Therefore the local
        # 08:00-08:15 opening bars end in the half-open-to-closed interval
        # (08:00, 08:15], not [08:00, 08:15).
        completed_minute = local.hour * 60 + local.minute
        open_minute = SESSION_LOCAL_OPEN_HOUR * 60
        if open_minute < completed_minute <= open_minute + SESSION_OPEN_WINDOW_MINUTES:
            return True, str(zone)
    return False, None


def _story_decision(
    template: StoryTemplateV2,
    features: Mapping[str, _Feature],
    prior_features: Mapping[str, _Feature],
    feature_history: Mapping[str, Sequence[_Feature]],
    *,
    trigger_at: datetime,
) -> _StoryDecision | None:
    """Bind an actual prior setup to a completed trigger and prior-only target."""

    m1 = features.get("M1")
    m5 = features.get("M5")
    prior_m1 = prior_features.get("M1")
    prior_m5 = prior_features.get("M5")
    prior_m15 = prior_features.get("M15")
    prior_h1 = prior_features.get("H1")
    if (
        m1 is None
        or m5 is None
        or prior_m1 is None
        or prior_m5 is None
        or m5.atr is None
        or m5.atr <= 0.0
        or not prior_m1.completed_at < trigger_at
    ):
        return None
    setup_at = prior_m1.completed_at
    m5_side = _trend_side(prior_m5)
    m15_side = _trend_side(prior_m15)
    h1_side = _trend_side(prior_h1)
    upper = prior_m15.prior_high if prior_m15 is not None else None
    lower = prior_m15.prior_low if prior_m15 is not None else None
    h1_upper = prior_h1.prior_high if prior_h1 is not None else None
    h1_lower = prior_h1.prior_low if prior_h1 is not None else None

    def emit(
        side: str,
        target: float | None,
        stop_anchor: float | None,
        setup_evidence: Mapping[str, Any],
        trigger_evidence: Mapping[str, Any],
    ) -> _StoryDecision:
        return _StoryDecision(
            hypothesis_id=template.hypothesis_id,
            side=side,
            setup_at=setup_at,
            trigger_at=trigger_at,
            structural_target=target,
            structural_stop_anchor=stop_anchor,
            setup_evidence=dict(setup_evidence),
            trigger_evidence={"trigger_close": m1.close, **dict(trigger_evidence)},
        )

    if template.hypothesis_id == "H21":
        trend = m15_side if m15_side is not None and m15_side == h1_side else None
        if (
            trend == "LONG"
            and prior_m1.ema_fast is not None
            and prior_m1.close <= prior_m1.ema_fast
            and prior_m1.current_low < prior_m1.current_open
            and m1.close > prior_m1.current_high
            and m5_side == "LONG"
        ):
            return emit(
                "LONG",
                upper,
                prior_m1.current_low,
                {"pullback_side": "DOWN", "trend_side": trend},
                {"reclaimed_prior_high": prior_m1.current_high},
            )
        if (
            trend == "SHORT"
            and prior_m1.ema_fast is not None
            and prior_m1.close >= prior_m1.ema_fast
            and prior_m1.current_high > prior_m1.current_open
            and m1.close < prior_m1.current_low
            and m5_side == "SHORT"
        ):
            return emit(
                "SHORT",
                lower,
                prior_m1.current_high,
                {"pullback_side": "UP", "trend_side": trend},
                {"reclaimed_prior_low": prior_m1.current_low},
            )
        return None
    if template.hypothesis_id == "H22":
        prior_impulse = _completed_impulse_ratio(prior_m1, prior_m5)
        if (
            m15_side == "LONG"
            and _meets_frozen_threshold(prior_impulse, STORY_IMPULSE_MIN_ATR)
            and m1.close < prior_m1.current_low
        ):
            return emit(
                "SHORT",
                lower,
                prior_m1.current_high,
                {"exhausted_trend_side": "LONG", "impulse_atr": prior_impulse},
                {"choch_below": prior_m1.current_low},
            )
        if (
            m15_side == "SHORT"
            and _meets_frozen_threshold(prior_impulse, STORY_IMPULSE_MIN_ATR)
            and m1.close > prior_m1.current_high
        ):
            return emit(
                "LONG",
                upper,
                prior_m1.current_low,
                {"exhausted_trend_side": "SHORT", "impulse_atr": prior_impulse},
                {"choch_above": prior_m1.current_high},
            )
        return None
    if template.hypothesis_id == "H23":
        if (
            prior_m15 is None
            or prior_m15.adx is None
            or prior_m15.adx >= RANGE_ADX_MAX
            or upper is None
            or lower is None
            or upper <= lower
        ):
            return None
        if m1.current_low < lower < m1.close:
            return emit(
                "LONG",
                upper,
                m1.current_low,
                {"frozen_m15_lower": lower, "m15_adx": prior_m15.adx},
                {"swept_m15_lower": lower, "reaccepted_above": lower},
            )
        if m1.current_high > upper > m1.close:
            return emit(
                "SHORT",
                lower,
                m1.current_high,
                {"frozen_m15_upper": upper, "m15_adx": prior_m15.adx},
                {"swept_m15_upper": upper, "reaccepted_below": upper},
            )
        return None
    if template.hypothesis_id == "H24":
        if (
            prior_m15 is None
            or prior_m15.adx is None
            or prior_m15.adx >= RANGE_ADX_MAX
            or upper is None
            or lower is None
            or upper <= lower
        ):
            return None
        if prior_m1.close > upper and m1.current_low <= upper < m1.close:
            return emit(
                "LONG",
                h1_upper,
                m1.current_low,
                {
                    "completed_break_above_m15_upper": upper,
                    "m15_adx": prior_m15.adx,
                },
                {"retest_held": upper},
            )
        if prior_m1.close < lower and m1.current_high >= lower > m1.close:
            return emit(
                "SHORT",
                h1_lower,
                m1.current_high,
                {
                    "completed_break_below_m15_lower": lower,
                    "m15_adx": prior_m15.adx,
                },
                {"retest_held": lower},
            )
        return None
    if template.hypothesis_id == "H25":
        compression_quantile = (
            _prior_atr_quantile(
                feature_history.get("M15", ()), before=prior_m15.completed_at
            )
            if prior_m15 is not None
            else None
        )
        compressed = bool(
            prior_m15 is not None
            and prior_m15.atr is not None
            and compression_quantile is not None
            and prior_m15.atr <= compression_quantile
        )
        expanded = bool(prior_m5.atr is not None and m5.atr > prior_m5.atr)
        if compressed and expanded and upper is not None and m1.close > upper:
            return emit(
                "LONG",
                h1_upper,
                lower,
                {"m15_atr": prior_m15.atr, "prior_q25_atr": compression_quantile},
                {"released_above": upper, "m5_atr_expanded": True},
            )
        if compressed and expanded and lower is not None and m1.close < lower:
            return emit(
                "SHORT",
                h1_lower,
                upper,
                {"m15_atr": prior_m15.atr, "prior_q25_atr": compression_quantile},
                {"released_below": lower, "m5_atr_expanded": True},
            )
        return None
    if template.hypothesis_id == "H26":
        compression_quantile = (
            _prior_atr_quantile(
                feature_history.get("M15", ()), before=prior_m15.completed_at
            )
            if prior_m15 is not None
            else None
        )
        compressed = bool(
            prior_m15 is not None
            and prior_m15.atr is not None
            and compression_quantile is not None
            and prior_m15.atr <= compression_quantile
        )
        if compressed and lower is not None and upper is not None:
            if m1.current_low < lower < m1.close:
                return emit(
                    "LONG",
                    upper,
                    m1.current_low,
                    {
                        "compressed_m15_atr": prior_m15.atr,
                        "prior_q25_atr": compression_quantile,
                    },
                    {
                        "false_release_below": lower,
                        "reaccepted_above": lower,
                    },
                )
            if m1.current_high > upper > m1.close:
                return emit(
                    "SHORT",
                    lower,
                    m1.current_high,
                    {
                        "compressed_m15_atr": prior_m15.atr,
                        "prior_q25_atr": compression_quantile,
                    },
                    {
                        "false_release_above": upper,
                        "reaccepted_below": upper,
                    },
                )
        return None
    if template.hypothesis_id == "H27":
        prior_impulse = _completed_impulse_ratio(prior_m1, prior_m5)
        midpoint = (prior_m1.current_open + prior_m1.close) / 2.0
        if (
            _meets_frozen_threshold(prior_impulse, STORY_IMPULSE_MIN_ATR)
            and m5_side == "LONG"
            and prior_m1.close > prior_m1.current_open
            and m1.current_low >= midpoint
            and m1.close > m1.current_open
        ):
            return emit(
                "LONG",
                upper,
                m1.current_low,
                {"impulse_side": "LONG", "impulse_atr": prior_impulse},
                {"pause_held_midpoint": midpoint},
            )
        if (
            _meets_frozen_threshold(prior_impulse, STORY_IMPULSE_MIN_ATR)
            and m5_side == "SHORT"
            and prior_m1.close < prior_m1.current_open
            and m1.current_high <= midpoint
            and m1.close < m1.current_open
        ):
            return emit(
                "SHORT",
                lower,
                m1.current_high,
                {"impulse_side": "SHORT", "impulse_atr": prior_impulse},
                {"pause_held_midpoint": midpoint},
            )
        return None
    if template.hypothesis_id == "H28":
        prior_impulse = _completed_impulse_ratio(prior_m1, prior_m5)
        midpoint = (prior_m1.current_open + prior_m1.close) / 2.0
        decelerating = bool(
            prior_m15 is not None
            and (
                (
                    prior_m15.adx is not None
                    and prior_m15.previous_adx is not None
                    and prior_m15.adx < prior_m15.previous_adx
                )
                or (
                    prior_m15.atr is not None
                    and prior_m15.previous_atr is not None
                    and prior_m15.atr < prior_m15.previous_atr
                )
            )
        )
        if prior_impulse >= CLIMAX_IMPULSE_MIN_ATR and decelerating:
            if prior_m1.close > prior_m1.current_open and m1.close < midpoint:
                return emit(
                    "SHORT",
                    lower,
                    prior_m1.current_high,
                    {"climax_side": "LONG", "impulse_atr": prior_impulse},
                    {"counter_break_below": midpoint},
                )
            if prior_m1.close < prior_m1.current_open and m1.close > midpoint:
                return emit(
                    "LONG",
                    upper,
                    prior_m1.current_low,
                    {"climax_side": "SHORT", "impulse_atr": prior_impulse},
                    {"counter_break_above": midpoint},
                )
        return None
    if template.hypothesis_id == "H29":
        at_open, session_zone = _is_dst_aware_local_open(trigger_at)
        if (
            not at_open
            or prior_m15 is None
            or prior_m15.adx is None
            or prior_m15.adx >= RANGE_ADX_MAX
            or upper is None
            or lower is None
            or upper <= lower
        ):
            return None
        if m1.close > upper:
            return emit(
                "LONG",
                h1_upper,
                lower,
                {
                    "session_zone": session_zone,
                    "frozen_m15_upper": upper,
                    "m15_adx": prior_m15.adx,
                },
                {"opening_break_above": upper},
            )
        if m1.close < lower:
            return emit(
                "SHORT",
                h1_lower,
                upper,
                {
                    "session_zone": session_zone,
                    "frozen_m15_lower": lower,
                    "m15_adx": prior_m15.adx,
                },
                {"opening_break_below": lower},
            )
        return None
    if template.hypothesis_id == "H30":
        at_open, session_zone = _is_dst_aware_local_open(trigger_at)
        if (
            not at_open
            or prior_m15 is None
            or prior_m15.adx is None
            or prior_m15.adx >= RANGE_ADX_MAX
            or upper is None
            or lower is None
            or upper <= lower
        ):
            return None
        if m1.current_high > upper > m1.close:
            return emit(
                "SHORT",
                lower,
                m1.current_high,
                {
                    "session_zone": session_zone,
                    "swept_upper": upper,
                    "m15_adx": prior_m15.adx,
                },
                {"opening_reaccepted_below": upper},
            )
        if m1.current_low < lower < m1.close:
            return emit(
                "LONG",
                upper,
                m1.current_low,
                {
                    "session_zone": session_zone,
                    "swept_lower": lower,
                    "m15_adx": prior_m15.adx,
                },
                {"opening_reaccepted_above": lower},
            )
    return None


def _choose_order_mode(
    template: StoryTemplateV2,
    *,
    impulse_ratio: float,
) -> tuple[str, str]:
    if (
        template.market_on_high_impulse
        and "MARKET" in template.allowed_order_modes
        and impulse_ratio >= MARKET_IMPULSE_MIN_ATR
    ):
        return "MARKET", "COMPLETED_M1_HIGH_IMPULSE_MARKET"
    if template.ordinary_order_mode not in template.allowed_order_modes:
        raise AssertionError("story ordinary order is outside its allowed modes")
    return template.ordinary_order_mode, "STORY_NATIVE_RESTING_ORDER"


def _mid_open(candle: S5BidAskCandle) -> float:
    return (candle.bid_o + candle.ask_o) / 2.0


def _entry_open(candle: S5BidAskCandle, side: str) -> float:
    return candle.ask_o if side == "LONG" else candle.bid_o


def _exit_open(candle: S5BidAskCandle, side: str) -> float:
    return candle.bid_o if side == "LONG" else candle.ask_o


def _broker_tick_size(pip_factor: float) -> Decimal:
    factor = Decimal(str(pip_factor))
    if not factor.is_finite() or factor <= 0:
        raise ValueError("pip_factor must be positive and finite")
    return Decimal(1) / (factor * BROKER_TICKS_PER_PIP)


def _round_price_to_tick(
    value: float,
    tick: Decimal,
    *,
    rounding: str,
) -> float:
    price = Decimal(str(value))
    if not price.is_finite() or price <= 0:
        raise ValueError("price must be positive and finite")
    ticks = (price / tick).to_integral_value(rounding=rounding)
    return float(ticks * tick)


def _tighten_trailing_stop_to_tick(
    existing_stop: float,
    candidate: float,
    *,
    side: str,
    pip_factor: float,
) -> float:
    if side not in {"LONG", "SHORT"}:
        raise ValueError("trailing side must be LONG or SHORT")
    rounded_candidate = _round_price_to_tick(
        candidate,
        _broker_tick_size(pip_factor),
        rounding=ROUND_FLOOR if side == "LONG" else ROUND_CEILING,
    )
    return (
        max(existing_stop, rounded_candidate)
        if side == "LONG"
        else min(existing_stop, rounded_candidate)
    )


def _observe_quote(
    position: _StoryPosition,
    candle: S5BidAskCandle,
    *,
    pip_factor: float = 10_000.0,
) -> None:
    position.quote_observed_at = candle.timestamp_utc
    position.observed_bid = candle.bid_o
    position.observed_ask = candle.ask_o
    atr = float(position.frozen_m5.atr or 0.0)
    if atr <= 0.0:
        raise AssertionError("frozen M5 ATR must be positive")
    tick = _broker_tick_size(pip_factor)
    mid = _mid_open(candle)
    if position.chosen_order_mode == "MARKET":
        raw_reference = candle.ask_o if position.side == "LONG" else candle.bid_o
        reference = _round_price_to_tick(raw_reference, tick, rounding=ROUND_HALF_UP)
        position.entry_target = None
    elif position.chosen_order_mode == "STOP":
        if position.side == "LONG":
            raw_reference = max(
                candle.ask_o, position.frozen_m1.current_high + STOP_OFFSET_ATR * atr
            )
            reference = _round_price_to_tick(
                raw_reference, tick, rounding=ROUND_CEILING
            )
        else:
            raw_reference = min(
                candle.bid_o, position.frozen_m1.current_low - STOP_OFFSET_ATR * atr
            )
            reference = _round_price_to_tick(raw_reference, tick, rounding=ROUND_FLOOR)
        position.entry_target = reference
    elif position.chosen_order_mode == "LIMIT":
        raw_reference = (
            min(candle.ask_o, mid - LIMIT_OFFSET_ATR * atr)
            if position.side == "LONG"
            else max(candle.bid_o, mid + LIMIT_OFFSET_ATR * atr)
        )
        reference = _round_price_to_tick(
            raw_reference,
            tick,
            rounding=ROUND_FLOOR if position.side == "LONG" else ROUND_CEILING,
        )
        position.entry_target = reference
    else:
        raise AssertionError("unknown contextual order mode")

    raw_structural_target = position.frozen_structural_target
    position.frozen_structural_target = _round_price_to_tick(
        raw_structural_target,
        tick,
        rounding=ROUND_CEILING if position.side == "LONG" else ROUND_FLOOR,
    )
    if position.side == "LONG":
        raw_structural_stop = min(
            value
            for value in (
                reference - STRUCTURAL_STOP_ATR_FLOOR * atr,
                position.frozen_structural_stop_anchor,
            )
            if value is not None
        )
        position.structural_stop = _round_price_to_tick(
            raw_structural_stop, tick, rounding=ROUND_FLOOR
        )
    else:
        raw_structural_stop = max(
            value
            for value in (
                reference + STRUCTURAL_STOP_ATR_FLOOR * atr,
                position.frozen_structural_stop_anchor,
            )
            if value is not None
        )
        position.structural_stop = _round_price_to_tick(
            raw_structural_stop, tick, rounding=ROUND_CEILING
        )
    planned_risk = abs(reference - position.structural_stop)
    spread = candle.ask_o - candle.bid_o
    signed_target_room = (
        position.frozen_structural_target - reference
        if position.side == "LONG"
        else reference - position.frozen_structural_target
    )
    planned_reward = max(0.0, signed_target_room)
    stressed_reward = planned_reward - SPREAD_STRESS_ROUND_TRIPS * spread
    fill_geometry_issue = _entry_geometry_issue(position, reference)
    if fill_geometry_issue == "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL":
        position.order_creation_geometry_issue = (
            "ORDER_CREATION_STOP_LOSS_GEOMETRY_INVALID"
        )
    elif fill_geometry_issue == "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL":
        position.order_creation_geometry_issue = (
            "ORDER_CREATION_TAKE_PROFIT_GEOMETRY_INVALID"
        )
    position.cost_gate_passed = bool(
        position.order_creation_geometry_issue is None
        and planned_risk > 0.0
        and stressed_reward > 0.0
    )
    position.cost_gate_evidence = {
        "broker_tick_size": float(tick),
        "price_precision_policy": PRICE_PRECISION_POLICY_V2,
        "raw_entry_reference_before_tick_rounding": raw_reference,
        "rounded_entry_reference": reference,
        "raw_structural_stop_before_tick_rounding": raw_structural_stop,
        "rounded_structural_stop": position.structural_stop,
        "raw_structural_target_before_tick_rounding": raw_structural_target,
        "rounded_structural_target": position.frozen_structural_target,
        "observed_spread_price": spread,
        "planned_initial_risk_price": planned_risk,
        "frozen_structural_target": position.frozen_structural_target,
        "planned_target_room_price": planned_reward,
        "spread_stress_price": SPREAD_STRESS_ROUND_TRIPS * spread,
        "planned_target_after_spread_stress_price": stressed_reward,
        "order_creation_geometry_issue": position.order_creation_geometry_issue,
        "direction_neutral": True,
        "passed": position.cost_gate_passed,
    }


def _fill_price(
    position: _StoryPosition, candle: S5BidAskCandle
) -> tuple[float, str] | None:
    if position.chosen_order_mode == "MARKET":
        return _entry_open(candle, position.side), "EXECUTABLE_OPEN"
    assert position.entry_target is not None
    target = position.entry_target
    if position.chosen_order_mode == "STOP":
        if position.side == "LONG":
            if candle.ask_o >= target:
                return candle.ask_o, "EXECUTABLE_OPEN_GAP"
            return (target, "INTRABAR_TRIGGER") if candle.ask_h >= target else None
        if candle.bid_o <= target:
            return candle.bid_o, "EXECUTABLE_OPEN_GAP"
        return (target, "INTRABAR_TRIGGER") if candle.bid_l <= target else None
    if position.side == "LONG":
        if candle.ask_o <= target:
            return candle.ask_o, "EXECUTABLE_OPEN_GAP"
        return (target, "INTRABAR_TRIGGER") if candle.ask_l <= target else None
    if candle.bid_o >= target:
        return candle.bid_o, "EXECUTABLE_OPEN_GAP"
    return (target, "INTRABAR_TRIGGER") if candle.bid_h >= target else None


def _entry_geometry_valid(position: _StoryPosition, fill_price: float) -> bool:
    if position.structural_stop is None:
        return False
    if position.side == "LONG":
        return position.structural_stop < fill_price < position.frozen_structural_target
    return position.frozen_structural_target < fill_price < position.structural_stop


def _entry_geometry_issue(position: _StoryPosition, fill_price: float) -> str | None:
    if position.structural_stop is None:
        return "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL"
    if position.side == "LONG":
        if fill_price <= position.structural_stop:
            return "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL"
        if fill_price >= position.frozen_structural_target:
            return "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL"
    else:
        if fill_price >= position.structural_stop:
            return "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL"
        if fill_price <= position.frozen_structural_target:
            return "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL"
    return None


def _finish(
    position: _StoryPosition,
    candle: S5BidAskCandle,
    *,
    exit_exec: float,
    reason: str,
    pip_factor: float,
    exit_mid_observable: bool,
) -> dict[str, Any]:
    assert position.entry_exec is not None
    assert position.initial_risk_price is not None
    sign = 1.0 if position.side == "LONG" else -1.0
    exact_price = sign * (exit_exec - position.entry_exec)
    exit_mid = _mid_open(candle) if exit_mid_observable else None
    gross_price = (
        sign * (exit_mid - position.entry_mid)
        if exit_mid is not None and position.entry_mid is not None
        else None
    )
    spread_drag_price = gross_price - exact_price if gross_price is not None else None
    risk = position.initial_risk_price
    return {
        "candidate_id": position.candidate.candidate_id,
        "hypothesis_id": position.candidate.hypothesis_id,
        "story_name": position.candidate.story_name,
        "split": position.split_name,
        "side": position.side,
        "order_mode": position.chosen_order_mode,
        "order_selection_reason": position.order_selection_reason,
        "exit_policy_id": position.candidate.exit_policy_id,
        "setup_at_utc": position.setup_at.isoformat(),
        "trigger_at_utc": position.trigger_at.isoformat(),
        "quote_observed_at_utc": position.quote_observed_at.isoformat()
        if position.quote_observed_at
        else None,
        "entry_at_utc": position.filled_at.isoformat() if position.filled_at else None,
        "exit_at_utc": candle.timestamp_utc.isoformat(),
        "entry_exec": position.entry_exec,
        "entry_fill_kind": position.entry_fill_kind,
        "exit_exec": exit_exec,
        "frozen_structural_stop": position.structural_stop,
        "final_trailing_stop": position.trail_stop,
        "initial_risk_pips": risk * pip_factor,
        "gross_mid_pips": gross_price * pip_factor if gross_price is not None else None,
        "spread_drag_pips": (
            spread_drag_price * pip_factor if spread_drag_price is not None else None
        ),
        "exact_net_pips": exact_price * pip_factor,
        "gross_mid_r": gross_price / risk if gross_price is not None else None,
        "spread_drag_r": (
            spread_drag_price / risk if spread_drag_price is not None else None
        ),
        "exact_net_r": exact_price / risk,
        "gross_spread_decomposition_status": (
            "EXACT_SYNCHRONIZED_OPENS"
            if gross_price is not None
            else "UNAVAILABLE_UNSYNCHRONIZED_INTRABAR_BID_ASK"
        ),
        "reason": reason,
        "cost_gate_evidence": dict(position.cost_gate_evidence or {}),
        "t_setup_lt_t_trigger_lt_t_entry": bool(
            position.setup_at < position.trigger_at
            and position.filled_at is not None
            and position.trigger_at < position.filled_at
        ),
        "exact_bid_ask": True,
        "equal_initial_risk_normalized": True,
        **_AUTHORITY,
    }


def _resolve_filled(
    position: _StoryPosition,
    candle: S5BidAskCandle,
    *,
    pip_factor: float,
) -> dict[str, Any] | None:
    assert position.structural_stop is not None
    assert position.exit_due_at is not None
    stop = (
        position.trail_stop
        if position.trail_stop is not None
        else position.structural_stop
    )
    target = position.take_profit
    side = position.side
    if candle.timestamp_utc >= position.exit_due_at:
        executable = _exit_open(candle, side)
        stop_gap = executable <= stop if side == "LONG" else executable >= stop
        return _finish(
            position,
            candle,
            exit_exec=executable,
            reason=(
                "STRUCTURAL_STOP_GAP_AT_TIME_CLOSE"
                if stop_gap
                else "MANDATORY_TIME_CLOSE"
            ),
            pip_factor=pip_factor,
            exit_mid_observable=True,
        )
    if side == "LONG":
        if candle.bid_o <= stop:
            return _finish(
                position,
                candle,
                exit_exec=candle.bid_o,
                reason="STRUCTURAL_STOP_GAP",
                pip_factor=pip_factor,
                exit_mid_observable=True,
            )
        if target is not None and candle.bid_o >= target:
            return _finish(
                position,
                candle,
                exit_exec=candle.bid_o,
                reason="PROFIT_FIRST_TARGET_GAP",
                pip_factor=pip_factor,
                exit_mid_observable=True,
            )
        stop_hit = candle.bid_l <= stop
        target_hit = target is not None and candle.bid_h >= target
        if stop_hit:
            return _finish(
                position,
                candle,
                exit_exec=stop,
                reason="STRUCTURAL_STOP_SAME_S5" if target_hit else "STRUCTURAL_STOP",
                pip_factor=pip_factor,
                exit_mid_observable=False,
            )
        if target_hit:
            return _finish(
                position,
                candle,
                exit_exec=float(target),
                reason="PROFIT_FIRST_TARGET",
                pip_factor=pip_factor,
                exit_mid_observable=False,
            )
    else:
        if candle.ask_o >= stop:
            return _finish(
                position,
                candle,
                exit_exec=candle.ask_o,
                reason="STRUCTURAL_STOP_GAP",
                pip_factor=pip_factor,
                exit_mid_observable=True,
            )
        if target is not None and candle.ask_o <= target:
            return _finish(
                position,
                candle,
                exit_exec=candle.ask_o,
                reason="PROFIT_FIRST_TARGET_GAP",
                pip_factor=pip_factor,
                exit_mid_observable=True,
            )
        stop_hit = candle.ask_h >= stop
        target_hit = target is not None and candle.ask_l <= target
        if stop_hit:
            return _finish(
                position,
                candle,
                exit_exec=stop,
                reason="STRUCTURAL_STOP_SAME_S5" if target_hit else "STRUCTURAL_STOP",
                pip_factor=pip_factor,
                exit_mid_observable=False,
            )
        if target_hit:
            return _finish(
                position,
                candle,
                exit_exec=float(target),
                reason="PROFIT_FIRST_TARGET",
                pip_factor=pip_factor,
                exit_mid_observable=False,
            )
    return None


def _resolve_intrabar_fill_candle(
    position: _StoryPosition,
    candle: S5BidAskCandle,
    *,
    pip_factor: float,
) -> dict[str, Any] | None:
    """Resolve only a conservative stop on the resting-order fill candle.

    Bid/ask OHLC has no event ordering inside one S5.  A LIMIT target may have
    printed before the later pullback fill, and a STOP candle's open may be a
    pre-fill price.  Therefore this fill candle can never claim a target or an
    open-gap stop.  If its executable-side range also contains the structural
    stop, the ambiguity is charged stop-first; otherwise resolution begins on
    the next real S5.
    """

    assert position.entry_fill_kind == "INTRABAR_TRIGGER"
    assert position.structural_stop is not None
    stop = (
        position.trail_stop
        if position.trail_stop is not None
        else position.structural_stop
    )
    stop_hit = candle.bid_l <= stop if position.side == "LONG" else candle.ask_h >= stop
    if not stop_hit:
        return None
    return _finish(
        position,
        candle,
        exit_exec=stop,
        reason="STRUCTURAL_STOP_ON_INTRABAR_FILL_S5_CONSERVATIVE",
        pip_factor=pip_factor,
        exit_mid_observable=False,
    )


def _blank_stat() -> dict[str, Any]:
    return {
        "signal_count": 0,
        "embargoed_count": 0,
        "deoverlap_count": 0,
        "unscorable_count": 0,
        "cost_blocked_count": 0,
        "order_creation_blocked_count": 0,
        "unfilled_count": 0,
        "canceled_before_fill_count": 0,
        "gap_attempt_canceled_count": 0,
        "filled_count": 0,
        "resolved_count": 0,
        "purged_count": 0,
        "truth_window_unresolved_count": 0,
        "unresolved_filled_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "gross_mid_pips": 0.0,
        "spread_drag_pips": 0.0,
        "gross_decomposition_unavailable_count": 0,
        "exact_net_pips": 0.0,
        "exact_net_r": 0.0,
        "gross_profit_r": 0.0,
        "gross_loss_r": 0.0,
        "reason_counts": defaultdict(int),
    }


def _blank_daily() -> dict[str, Any]:
    return {
        "filled_count": 0,
        "resolved_count": 0,
        "exact_net_r": 0.0,
        "gross_profit_r": 0.0,
        "gross_loss_r": 0.0,
    }


def _record(stat: dict[str, Any], trade: Mapping[str, Any]) -> None:
    net = float(trade["exact_net_pips"])
    net_r = float(trade["exact_net_r"])
    stat["resolved_count"] += 1
    if trade["gross_mid_pips"] is None or trade["spread_drag_pips"] is None:
        stat["gross_decomposition_unavailable_count"] += 1
    else:
        stat["gross_mid_pips"] += float(trade["gross_mid_pips"])
        stat["spread_drag_pips"] += float(trade["spread_drag_pips"])
    stat["exact_net_pips"] += net
    stat["exact_net_r"] += net_r
    if net_r > 0.0:
        stat["win_count"] += 1
        stat["gross_profit_r"] += net_r
    elif net_r < 0.0:
        stat["loss_count"] += 1
        stat["gross_loss_r"] += -net_r
    stat["reason_counts"][str(trade["reason"])] += 1


def _record_daily(daily: dict[str, Any], trade: Mapping[str, Any]) -> None:
    net_r = float(trade["exact_net_r"])
    daily["resolved_count"] += 1
    daily["exact_net_r"] += net_r
    if net_r > 0.0:
        daily["gross_profit_r"] += net_r
    elif net_r < 0.0:
        daily["gross_loss_r"] += -net_r


def _metric(stat: Mapping[str, Any]) -> dict[str, Any]:
    resolved = int(stat["resolved_count"])
    loss_r = float(stat["gross_loss_r"])
    profit_r = float(stat["gross_profit_r"])
    decomposition_unavailable = int(stat["gross_decomposition_unavailable_count"])
    return {
        "signal_count": int(stat["signal_count"]),
        "embargoed_count": int(stat["embargoed_count"]),
        "deoverlap_count": int(stat["deoverlap_count"]),
        "unscorable_count": int(stat["unscorable_count"]),
        "cost_blocked_count": int(stat["cost_blocked_count"]),
        "order_creation_blocked_count": int(stat["order_creation_blocked_count"]),
        "unfilled_count": int(stat["unfilled_count"]),
        "canceled_before_fill_count": int(stat["canceled_before_fill_count"]),
        "gap_attempt_canceled_count": int(stat["gap_attempt_canceled_count"]),
        "filled_count": int(stat["filled_count"]),
        "resolved_count": resolved,
        "purged_count": int(stat["purged_count"]),
        "truth_window_unresolved_count": int(stat["truth_window_unresolved_count"]),
        "unresolved_filled_count": int(stat["unresolved_filled_count"]),
        "win_count": int(stat["win_count"]),
        "loss_count": int(stat["loss_count"]),
        "gross_mid_pips": (
            None if decomposition_unavailable else float(stat["gross_mid_pips"])
        ),
        "spread_drag_pips": (
            None if decomposition_unavailable else float(stat["spread_drag_pips"])
        ),
        "observable_only_gross_mid_pips": float(stat["gross_mid_pips"]),
        "observable_only_spread_drag_pips": float(stat["spread_drag_pips"]),
        "gross_decomposition_unavailable_count": decomposition_unavailable,
        "gross_decomposition_complete": decomposition_unavailable == 0,
        "exact_net_pips": float(stat["exact_net_pips"]),
        "exact_net_r": float(stat["exact_net_r"]),
        "average_net_r": float(stat["exact_net_r"]) / resolved if resolved else None,
        "profit_factor_r": profit_r / loss_r if loss_r > 0.0 else None,
        "gross_profit_r": profit_r,
        "gross_loss_r": loss_r,
        "win_rate": int(stat["win_count"]) / resolved if resolved else None,
        "reason_counts": dict(sorted(stat["reason_counts"].items())),
    }


def _requested_vehicles(
    candidate_ids: Sequence[str] | None,
) -> tuple[StoryVehicleV2, ...]:
    catalog = build_story_vehicle_catalog_v2()
    if candidate_ids is None:
        return catalog
    if isinstance(candidate_ids, (str, bytes)) or not isinstance(
        candidate_ids, Sequence
    ):
        raise ValueError("candidate_ids must be a sequence or None")
    requested = tuple(str(value).strip() for value in candidate_ids)
    if any(not value for value in requested):
        raise ValueError("candidate_ids cannot contain empty values")
    if len(requested) != len(set(requested)):
        raise ValueError("candidate_ids must be unique")
    known = {row.candidate_id for row in catalog}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown adaptive story candidate_ids: {unknown}")
    requested_set = set(requested)
    return tuple(row for row in catalog if row.candidate_id in requested_set)


def run_adaptive_story_s5_grid(
    pair: str,
    candles: Iterable[S5BidAskCandle],
    splits: Sequence[UtcSplit],
    *,
    unavailable_pairs: Sequence[str] = (),
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one pair once through the finite causal story/vehicle grid."""

    pair_name = str(pair).strip().upper()
    if not pair_name:
        raise ValueError("pair is required")
    _pair_currency_factor_views(pair_name, "LONG")
    normalized_splits = _normalise_splits(splits)
    unavailable = tuple(str(item).strip().upper() for item in unavailable_pairs)
    if len(unavailable) != len(set(unavailable)):
        raise ValueError("unavailable_pairs must be unique")
    vehicles = _requested_vehicles(candidate_ids)
    executable = tuple(row for row in vehicles if not row.no_trade_control)
    stats = {
        (row.candidate_id, split.name): _blank_stat()
        for row in executable
        for split in normalized_splits
    }
    daily_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    currency_trigger_clusters: _CurrencyTriggerClusterMap = {}
    currency_trigger_ownership: dict[str, tuple[_CurrencyTriggerClusterKey, ...]] = {}
    if pair_name in unavailable:
        return _build_result(
            pair_name,
            normalized_splits,
            vehicles,
            stats,
            status="UNAVAILABLE",
            source_count=0,
            missing_slots=0,
            completed_counts={key: 0 for key in TIMEFRAMES_SECONDS},
            partial_counts={key: 0 for key in TIMEFRAMES_SECONDS},
            signal_rows=[],
            trade_rows=[],
            blocked_rows=[],
            daily_stats=daily_stats,
            currency_trigger_clusters=currency_trigger_clusters,
            currency_trigger_ownership=currency_trigger_ownership,
        )
    if isinstance(candles, (str, bytes)):
        raise ValueError("candles must be an iterable of S5BidAskCandle")
    try:
        iterator = iter(candles)
    except TypeError as error:
        raise ValueError("candles must be iterable") from error

    pip_factor = float(instrument_pip_factor(pair_name))
    requested_hypotheses = {row.hypothesis_id for row in executable}
    templates = tuple(
        row
        for row in build_story_templates_v2()
        if not row.no_trade_control and row.hypothesis_id in requested_hypotheses
    )
    vehicles_by_story = {
        template.hypothesis_id: tuple(
            row for row in executable if row.hypothesis_id == template.hypothesis_id
        )
        for template in templates
    }
    buckets: dict[str, _Bucket] = {}
    bars: dict[str, deque[_Bar]] = {
        timeframe: deque(maxlen=FEATURE_HISTORY_BAR_CAP)
        for timeframe in TIMEFRAMES_SECONDS
    }
    features: dict[str, _Feature] = {}
    feature_history: dict[str, deque[_Feature]] = {
        timeframe: deque(maxlen=FEATURE_HISTORY_BAR_CAP)
        for timeframe in TIMEFRAMES_SECONDS
    }
    active: dict[str, _StoryPosition] = {}
    completed_counts = {key: 0 for key in TIMEFRAMES_SECONDS}
    source_count = 0
    missing_slots = 0
    previous_stamp: datetime | None = None
    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for candle in iterator:
        stamp = _validate_candle(candle, previous_stamp)
        source_count += 1
        if previous_stamp is not None:
            missing_slots += max(
                0, int((stamp - previous_stamp).total_seconds() // 5) - 1
            )
        previous_stamp = stamp

        # Purge at the half-open split boundary; a prior partition position may
        # never leak P/L or de-overlap into the next partition.
        for candidate_id, position in tuple(active.items()):
            split = next(
                split
                for split in normalized_splits
                if split.name == position.split_name
            )
            if stamp >= split.to_utc:
                stat = stats[(candidate_id, position.split_name)]
                stat["purged_count"] += 1
                stat["unresolved_filled_count"] += int(position.filled_at is not None)
                stat["reason_counts"]["SPLIT_END_PURGE"] += 1
                del active[candidate_id]

        completed_events: list[_Bar] = []
        for timeframe in TIMEFRAMES_SECONDS:
            bucket = buckets.get(timeframe)
            if bucket is not None and bucket.end <= stamp:
                completed_events.append(bucket.finish())
                del buckets[timeframe]
        completed_events.sort(key=lambda row: (row.end, row.timeframe))
        m1_decisions: list[tuple[_Bar, dict[str, _Feature], dict[str, _Feature]]] = []
        index = 0
        while index < len(completed_events):
            clock = completed_events[index].end
            same_clock: list[_Bar] = []
            while (
                index < len(completed_events) and completed_events[index].end == clock
            ):
                same_clock.append(completed_events[index])
                index += 1
            prior_frozen = dict(features)
            for bar in same_clock:
                bars[bar.timeframe].append(bar)
                features[bar.timeframe] = _feature(
                    tuple(bars[bar.timeframe]), features.get(bar.timeframe)
                )
                feature_history[bar.timeframe].append(features[bar.timeframe])
                completed_counts[bar.timeframe] += 1
            for bar in same_clock:
                if bar.timeframe == "M1":
                    m1_decisions.append((bar, dict(features), prior_frozen))

        # A trailing stop may use the newly completed M1 structure, but never
        # the current S5 candle that follows that close boundary.
        current_completed_m1 = features.get("M1") if m1_decisions else None
        if current_completed_m1 is not None:
            for position in active.values():
                if (
                    position.filled_at is None
                    or not position.candidate.trailing_structural
                    or current_completed_m1.completed_at <= position.filled_at
                    or position.entry_exec is None
                ):
                    continue
                if (
                    position.side == "LONG"
                    and current_completed_m1.close > position.entry_exec
                ):
                    position.trail_stop = _tighten_trailing_stop_to_tick(
                        position.trail_stop or float(position.structural_stop),
                        current_completed_m1.current_low,
                        side="LONG",
                        pip_factor=pip_factor,
                    )
                elif (
                    position.side == "SHORT"
                    and current_completed_m1.close < position.entry_exec
                ):
                    position.trail_stop = _tighten_trailing_stop_to_tick(
                        position.trail_stop or float(position.structural_stop),
                        current_completed_m1.current_high,
                        side="SHORT",
                        pip_factor=pip_factor,
                    )

        for m1_bar, frozen, prior_frozen in m1_decisions:
            split = _split_for(m1_bar.end, normalized_splits)
            m5 = frozen.get("M5")
            prior_m1 = prior_frozen.get("M1")
            if (
                split is None
                or prior_m1 is None
                or m5 is None
                or m5.atr is None
                or m5.atr <= 0.0
            ):
                continue
            setup_at = prior_m1.completed_at
            trigger_at = m1_bar.end
            if not setup_at < trigger_at:
                raise AssertionError("story setup must precede its trigger")
            embargoed = (
                trigger_at + timedelta(seconds=ENTRY_TTL_SECONDS + MAX_HOLD_SECONDS)
                >= split.to_utc
            )
            for template in templates:
                decision = _story_decision(
                    template,
                    frozen,
                    prior_frozen,
                    feature_history,
                    trigger_at=trigger_at,
                )
                if decision is None:
                    continue
                side = decision.side
                setup_at = decision.setup_at
                impulse = _completed_impulse_ratio(frozen["M1"], m5)
                order_mode, order_reason = _choose_order_mode(
                    template, impulse_ratio=impulse
                )
                if len(signal_rows) < MAX_AUDIT_ROWS:
                    signal_rows.append(
                        {
                            "hypothesis_id": template.hypothesis_id,
                            "story_name": template.story_name,
                            "setup_at_utc": setup_at.isoformat(),
                            "trigger_at_utc": trigger_at.isoformat(),
                            "direct_story_side": side,
                            "setup_evidence": decision.setup_evidence,
                            "trigger_evidence": decision.trigger_evidence,
                            "frozen_structural_target": decision.structural_target,
                            "frozen_structural_stop_anchor": (
                                decision.structural_stop_anchor
                            ),
                            "structural_target_scorable": decision.scorable,
                            "chosen_order_mode": order_mode,
                            "allowed_order_modes": list(template.allowed_order_modes),
                            "order_selection_reason": order_reason,
                            "completed_feature_clocks": {
                                key: value.completed_at.isoformat()
                                for key, value in sorted(frozen.items())
                            },
                            "completed_m1_impulse_to_m5_atr": impulse,
                            "market_impulse_gate": {
                                "required_for_market": True,
                                "threshold_atr": MARKET_IMPULSE_MIN_ATR,
                                "passed": impulse >= MARKET_IMPULSE_MIN_ATR,
                                "uses_completed_m1_and_m5_only": True,
                            },
                            "split": split.name,
                            "split_end_embargoed": embargoed,
                            "inverse_shadow_only": True,
                            "time_shift_shadow_only": True,
                            "counterfactuals_scorecard_eligible": False,
                            "shadow_counterfactuals": [
                                {
                                    "counterfactual_kind": "INVERSE_SIDE",
                                    "side": _opposite(side),
                                    "shadow_only": True,
                                    "scorecard_eligible": False,
                                },
                                {
                                    "counterfactual_kind": "NEXT_M1_TIME_SHIFT",
                                    "shift_seconds": TIME_SHIFT_SHADOW_SECONDS,
                                    "shifted_trigger_at_utc": (
                                        trigger_at
                                        + timedelta(seconds=TIME_SHIFT_SHADOW_SECONDS)
                                    ).isoformat(),
                                    "shadow_only": True,
                                    "scorecard_eligible": False,
                                },
                            ],
                        }
                    )
                for vehicle in vehicles_by_story[template.hypothesis_id]:
                    stat = stats[(vehicle.candidate_id, split.name)]
                    stat["signal_count"] += 1
                    if not decision.scorable:
                        stat["unscorable_count"] += 1
                        stat["reason_counts"][
                            "PRIOR_ONLY_STRUCTURAL_TARGET_UNAVAILABLE_OR_WRONG_SIDE"
                        ] += 1
                        if len(blocked_rows) < MAX_AUDIT_ROWS:
                            blocked_rows.append(
                                {
                                    "candidate_id": vehicle.candidate_id,
                                    "trigger_at_utc": trigger_at.isoformat(),
                                    "reason": (
                                        "PRIOR_ONLY_STRUCTURAL_TARGET_"
                                        "UNAVAILABLE_OR_WRONG_SIDE"
                                    ),
                                    "structural_target": decision.structural_target,
                                    "side": side,
                                }
                            )
                        continue
                    if embargoed:
                        stat["embargoed_count"] += 1
                        stat["reason_counts"]["SPLIT_END_MAX_TTL_PLUS_24H_EMBARGO"] += 1
                        continue
                    if vehicle.candidate_id in active:
                        stat["deoverlap_count"] += 1
                        stat["reason_counts"]["DEOVERLAP_ACTIVE_STORY_VEHICLE"] += 1
                        continue
                    active[vehicle.candidate_id] = _StoryPosition(
                        candidate=vehicle,
                        split_name=split.name,
                        side=side,
                        setup_at=setup_at,
                        trigger_at=trigger_at,
                        direct_story_side=side,
                        frozen_m1=frozen["M1"],
                        frozen_m5=m5,
                        frozen_structural_target=float(decision.structural_target),
                        frozen_structural_stop_anchor=(decision.structural_stop_anchor),
                        chosen_order_mode=order_mode,
                        order_selection_reason=order_reason,
                    )

        resolved: list[str] = []
        for candidate_id, position in active.items():
            stat = stats[(candidate_id, position.split_name)]
            if position.quote_observed_at is None:
                if stamp <= position.trigger_at:
                    continue
                _observe_quote(position, candle, pip_factor=pip_factor)
                if stamp >= position.trigger_at + timedelta(seconds=ENTRY_TTL_SECONDS):
                    stat["unfilled_count"] += 1
                    stat["reason_counts"]["ENTRY_TTL_EXPIRED_AT_FIRST_REAL_QUOTE"] += 1
                    resolved.append(candidate_id)
                    continue
                if position.order_creation_geometry_issue is not None:
                    reason = position.order_creation_geometry_issue
                    stat["order_creation_blocked_count"] += 1
                    stat["reason_counts"][reason] += 1
                    if len(blocked_rows) < MAX_AUDIT_ROWS:
                        blocked_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "trigger_at_utc": position.trigger_at.isoformat(),
                                "quote_observed_at_utc": stamp.isoformat(),
                                "reason": reason,
                                "cost_gate_evidence": dict(
                                    position.cost_gate_evidence or {}
                                ),
                            }
                        )
                    resolved.append(candidate_id)
                    continue
                if position.cost_gate_passed is not True:
                    stat["cost_blocked_count"] += 1
                    stat["reason_counts"]["DIRECTION_NEUTRAL_COST_GATE_BLOCKED"] += 1
                    if len(blocked_rows) < MAX_AUDIT_ROWS:
                        blocked_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "trigger_at_utc": position.trigger_at.isoformat(),
                                "quote_observed_at_utc": stamp.isoformat(),
                                "reason": "DIRECTION_NEUTRAL_COST_GATE_BLOCKED",
                                "cost_gate_evidence": dict(
                                    position.cost_gate_evidence or {}
                                ),
                            }
                        )
                    resolved.append(candidate_id)
                continue
            if position.filled_at is None:
                if stamp <= position.quote_observed_at:
                    continue
                if stamp >= position.trigger_at + timedelta(seconds=ENTRY_TTL_SECONDS):
                    stat["unfilled_count"] += 1
                    stat["reason_counts"]["ENTRY_TTL_EXPIRED"] += 1
                    resolved.append(candidate_id)
                    continue
                fill = _fill_price(position, candle)
                if fill is None:
                    continue
                fill_price, fill_kind = fill
                assert position.structural_stop is not None
                geometry_issue = _entry_geometry_issue(position, fill_price)
                if geometry_issue is not None:
                    stat["canceled_before_fill_count"] += 1
                    stat["unfilled_count"] += 1
                    stat["gap_attempt_canceled_count"] += int(
                        fill_kind == "EXECUTABLE_OPEN_GAP"
                    )
                    stat["reason_counts"][geometry_issue] += 1
                    if len(blocked_rows) < MAX_AUDIT_ROWS:
                        blocked_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "trigger_at_utc": position.trigger_at.isoformat(),
                                "quote_observed_at_utc": (
                                    position.quote_observed_at.isoformat()
                                    if position.quote_observed_at is not None
                                    else None
                                ),
                                "gap_attempt_at_utc": stamp.isoformat(),
                                "gap_attempt_exec": fill_price,
                                "entry_fill_kind": fill_kind,
                                "reason": geometry_issue,
                                "broker_fill_occurred": False,
                                "economic_outcome_recorded": False,
                            }
                        )
                    resolved.append(candidate_id)
                    continue
                position.filled_at = stamp
                position.entry_exec = fill_price
                position.entry_fill_kind = fill_kind
                position.entry_mid = (
                    _mid_open(candle)
                    if fill_kind in {"EXECUTABLE_OPEN", "EXECUTABLE_OPEN_GAP"}
                    else None
                )
                position.initial_risk_price = abs(fill_price - position.structural_stop)
                if position.initial_risk_price <= 0.0:
                    raise AssertionError(
                        "executable entry-to-stop risk must be positive"
                    )
                position.exit_due_at = stamp + timedelta(
                    seconds=position.candidate.max_hold_seconds
                )
                if position.candidate.exit_policy_id == "PROFIT_FIRST_24H":
                    position.take_profit = position.frozen_structural_target
                stat["filled_count"] += 1
                daily_stats.setdefault(
                    (candidate_id, position.split_name, stamp.date().isoformat()),
                    _blank_daily(),
                )["filled_count"] += 1
                if fill_kind == "INTRABAR_TRIGGER":
                    trade = _resolve_intrabar_fill_candle(
                        position, candle, pip_factor=pip_factor
                    )
                else:
                    trade = _resolve_filled(position, candle, pip_factor=pip_factor)
            else:
                trade = _resolve_filled(position, candle, pip_factor=pip_factor)
            if trade is None:
                continue
            _record(stat, trade)
            entry_day = (
                position.filled_at.date().isoformat()
                if position.filled_at is not None
                else stamp.date().isoformat()
            )
            _record_daily(
                daily_stats.setdefault(
                    (candidate_id, position.split_name, entry_day),
                    _blank_daily(),
                ),
                trade,
            )
            _record_currency_trigger_factor_trade(
                pair_name,
                trade,
                currency_trigger_clusters,
                currency_trigger_ownership,
            )
            if len(trade_rows) < MAX_AUDIT_ROWS:
                trade_rows.append(trade)
            resolved.append(candidate_id)
        for candidate_id in resolved:
            active.pop(candidate_id, None)

        for timeframe in TIMEFRAMES_SECONDS:
            bucket = buckets.get(timeframe)
            expected = datetime.fromtimestamp(
                int(stamp.timestamp())
                - int(stamp.timestamp()) % TIMEFRAMES_SECONDS[timeframe],
                tz=timezone.utc,
            )
            if bucket is None or bucket.start != expected:
                buckets[timeframe] = _new_bucket(timeframe, candle)
            else:
                bucket.add(candle)

    for position in active.values():
        stat = stats[(position.candidate.candidate_id, position.split_name)]
        stat["truth_window_unresolved_count"] += 1
        stat["unresolved_filled_count"] += int(position.filled_at is not None)
        stat["reason_counts"]["TRUTH_WINDOW_ENDED_UNRESOLVED"] += 1
    partial_counts = {key: int(key in buckets) for key in TIMEFRAMES_SECONDS}
    return _build_result(
        pair_name,
        normalized_splits,
        vehicles,
        stats,
        status="COMPLETE" if source_count else "NO_DATA",
        source_count=source_count,
        missing_slots=missing_slots,
        completed_counts=completed_counts,
        partial_counts=partial_counts,
        signal_rows=signal_rows,
        trade_rows=trade_rows,
        blocked_rows=blocked_rows,
        daily_stats=daily_stats,
        currency_trigger_clusters=currency_trigger_clusters,
        currency_trigger_ownership=currency_trigger_ownership,
    )


def _split_receipt_rows(splits: Sequence[UtcSplit]) -> list[dict[str, str]]:
    return [
        {
            "name": split.name,
            "from_utc": split.from_utc.isoformat(),
            "to_utc": split.to_utc.isoformat(),
        }
        for split in splits
    ]


def _split_utc_dates(split: UtcSplit) -> tuple[str, ...]:
    cursor = datetime.combine(
        split.from_utc.date(), datetime.min.time(), tzinfo=timezone.utc
    )
    dates: list[str] = []
    while cursor < split.to_utc:
        if cursor + timedelta(days=1) > split.from_utc:
            dates.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    return tuple(dates)


def _build_result(
    pair: str,
    splits: Sequence[UtcSplit],
    vehicles: Sequence[StoryVehicleV2],
    stats: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    status: str,
    source_count: int,
    missing_slots: int,
    completed_counts: Mapping[str, int],
    partial_counts: Mapping[str, int],
    signal_rows: Sequence[Mapping[str, Any]],
    trade_rows: Sequence[Mapping[str, Any]],
    blocked_rows: Sequence[Mapping[str, Any]],
    daily_stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
    currency_trigger_clusters: Mapping[
        _CurrencyTriggerClusterKey, Mapping[str, Mapping[str, Any]]
    ],
    currency_trigger_ownership: Mapping[str, tuple[_CurrencyTriggerClusterKey, ...]],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for vehicle in vehicles:
        by_split: dict[str, dict[str, Any]] = {}
        currency_clusters_by_split: dict[str, dict[str, Any]] = {}
        if not vehicle.no_trade_control:
            for split in splits:
                metric = _metric(stats[(vehicle.candidate_id, split.name)])
                cluster_payload, cluster_metric_fields = (
                    _build_currency_trigger_cluster_payload(
                        vehicle.candidate_id,
                        split.name,
                        currency_trigger_clusters,
                        currency_trigger_ownership,
                        resolved_trade_count=int(metric["resolved_count"]),
                        candidate_total_exact_net_r=float(metric["exact_net_r"]),
                        expected_pair=pair,
                    )
                )
                metric.update(cluster_metric_fields)
                by_split[split.name] = metric
                currency_clusters_by_split[split.name] = cluster_payload
        daily_by_split = {
            split.name: [
                {
                    "utc_date": utc_date,
                    **dict(
                        daily_stats.get(
                            (vehicle.candidate_id, split.name, utc_date),
                            _blank_daily(),
                        )
                    ),
                }
                for utc_date in _split_utc_dates(split)
            ]
            for split in splits
            if not vehicle.no_trade_control
        }
        trials.append(
            {
                "candidate_id": vehicle.candidate_id,
                "hypothesis_id": vehicle.hypothesis_id,
                "story_name": vehicle.story_name,
                "exit_policy_id": vehicle.exit_policy_id,
                "contextual_order_policy": vehicle.contextual_order_policy,
                "allowed_order_modes": list(vehicle.allowed_order_modes),
                "max_hold_seconds": vehicle.max_hold_seconds,
                "profit_target_r": vehicle.profit_target_r,
                "trailing_structural": vehicle.trailing_structural,
                "complexity": vehicle.complexity,
                "no_trade_control": vehicle.no_trade_control,
                "by_split": by_split,
                "daily_aggregates_by_split": daily_by_split,
                "currency_trigger_factor_clusters_by_split": (
                    currency_clusters_by_split
                ),
                "scorecard_eligible": not vehicle.no_trade_control,
            }
        )
    body: dict[str, Any] = {
        "contract": STORY_GRID_CONTRACT_V2,
        "schema_version": 2,
        "status": status,
        "pair": pair,
        "story_catalog_policy": STORY_CATALOG_POLICY_V2,
        "truth_policy": STORY_TRUTH_POLICY_V2,
        "story_catalog_sha256": _canonical_sha(_story_catalog_receipt_v2()),
        "truth_evaluator_sha256": _canonical_sha(_truth_evaluator_receipt_v2()),
        "price_precision_policy": PRICE_PRECISION_POLICY_V2,
        "price_cost_scope": dict(PRICE_COST_SCOPE_V2),
        "currency_trigger_cluster_contract": (CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1),
        "currency_trigger_cluster_key_policy": (CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1),
        "currency_factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
        "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
        "currency_trigger_cluster_rows_untruncated": True,
        "currency_trigger_factor_clusters_are_not_candidate_summed": True,
        "entry_ttl_boundary": "EXCLUSIVE",
        "intrabar_resting_fill_s5_policy": (
            "NO_TARGET;STOP_RANGE_CHARGED_CONSERVATIVELY;NO_PREFILL_OPEN_GAP"
        ),
        "entry_gap_invalid_geometry_policy": (
            "BROKER_ON_FILL_DEPENDENT_ORDER_LOSS_CANCEL_NO_FILL"
        ),
        "story_template_count": len(build_story_templates_v2()),
        "predeclared_selectable_candidate_count": 50,
        "predeclared_control_candidate_count": 1,
        "requested_candidate_ids": [row.candidate_id for row in vehicles],
        "evaluated_candidate_ids": [
            row.candidate_id for row in vehicles if not row.no_trade_control
        ],
        "requested_control_candidate_ids": [
            row.candidate_id for row in vehicles if row.no_trade_control
        ],
        "candidate_whitelist_sha256": _canonical_sha(
            [row.candidate_id for row in vehicles]
        ),
        "split_receipt": _split_receipt_rows(splits),
        "split_digest": _canonical_sha(_split_receipt_rows(splits)),
        "daily_aggregates_complete": True,
        "daily_cluster_basis": "ENTRY_UTC_DATE",
        "exit_day_or_mark_to_market_used_for_selection": False,
        "candidate_count": len(vehicles),
        "contextual_order_cross_product_forbidden": True,
        "setup_trigger_entry_policy": "T_SETUP_LT_T_TRIGGER_LT_T_ENTRY",
        "quote_observation_policy": (
            "FIRST_REAL_S5_AFTER_TRIGGER_OBSERVES_ONLY;"
            "FOLLOWING_REAL_S5_IS_EARLIEST_FILL"
        ),
        "split_embargo_seconds": ENTRY_TTL_SECONDS + MAX_HOLD_SECONDS,
        "profit_first_unrealized_loss_zeroed": False,
        "profit_first_mandatory_max_hold_seconds": MAX_HOLD_SECONDS,
        "inverse_counterfactual_shadow_only": True,
        "time_shift_counterfactual_shadow_only": True,
        "counterfactuals_selection_eligible": False,
        "aggregation": {
            "source_candle_count": source_count,
            "observed_missing_s5_slots": missing_slots,
            "synthetic_s5_count": 0,
            "completed_bucket_counts": dict(completed_counts),
            "partial_bucket_counts": dict(partial_counts),
            "partial_buckets_used_as_features": False,
        },
        "all_trials": trials,
        "signal_audit_rows": list(signal_rows),
        "trade_audit_rows": list(trade_rows),
        "blocked_audit_rows": list(blocked_rows),
        "audit_rows_truncated": bool(
            len(signal_rows) >= MAX_AUDIT_ROWS
            or len(trade_rows) >= MAX_AUDIT_ROWS
            or len(blocked_rows) >= MAX_AUDIT_ROWS
        ),
        "raw_candles_copied_to_result": False,
        **_AUTHORITY,
    }
    return {**body, "result_sha256": _canonical_sha(body)}


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"pair run {field} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pair run {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"pair run {field} must be a finite number")
    return result


def _same_total(left: float, right: float) -> bool:
    # Aggregates can arrive in a different pair/day addition order; this
    # tolerance covers floating-point reassociation only, never economic drift.
    return math.isclose(
        left,
        right,
        rel_tol=AGGREGATE_REASSOCIATION_TOL,
        abs_tol=AGGREGATE_REASSOCIATION_TOL,
    )


def _validate_currency_trigger_cluster_payload(
    payload: Mapping[str, Any],
    *,
    expected_candidate_id: str,
    expected_split_name: str,
    expected_pair: str | None,
    expected_resolved_trade_count: int,
    expected_candidate_total_exact_net_r: float,
    expected_metric_fields: Mapping[str, Any],
    expected_ownership: Mapping[str, tuple[_CurrencyTriggerClusterKey, ...]]
    | None = None,
) -> tuple[tuple[dict[str, Any], ...], frozenset[str]]:
    expected_payload_fields = {
        "contract",
        "schema_version",
        "cluster_key_policy",
        "factor_view_policy",
        "split",
        "candidate_id",
        "cluster_rows_untruncated",
        "factor_clusters_are_not_candidate_summed",
        "currency_factor_views_per_trade",
        "resolved_trade_count",
        "factor_membership_count",
        "candidate_total_exact_net_r",
        "factor_membership_exact_net_r_sum_for_ownership_check",
        "cluster_rows",
        "cluster_rows_sha256",
        "ownership_invariant_passed",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_payload_fields:
        raise ValueError("currency-trigger factor payload schema mismatch")
    expected_policy = {
        "contract": CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1,
        "schema_version": 1,
        "cluster_key_policy": CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1,
        "factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
        "split": expected_split_name,
        "candidate_id": expected_candidate_id,
        "cluster_rows_untruncated": True,
        "factor_clusters_are_not_candidate_summed": True,
        "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
        "ownership_invariant_passed": True,
    }
    if any(payload.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("currency-trigger factor payload policy mismatch")
    resolved_count = _require_nonnegative_int(
        payload.get("resolved_trade_count"),
        field="currency-trigger resolved_trade_count",
    )
    if resolved_count != expected_resolved_trade_count:
        raise ValueError("currency-trigger resolved trade count mismatch")
    candidate_total = _require_finite_number(
        payload.get("candidate_total_exact_net_r"),
        field="currency-trigger candidate_total_exact_net_r",
    )
    if not _same_total(candidate_total, expected_candidate_total_exact_net_r):
        raise ValueError("currency-trigger candidate total R mismatch")
    rows_raw = payload.get("cluster_rows")
    if not isinstance(rows_raw, list):
        raise ValueError("currency-trigger cluster rows are invalid")
    expected_row_fields = {
        "split",
        "candidate_id",
        "trigger_at_utc",
        "currency",
        "sign",
        "member_count",
        "exact_net_r",
        "member_identity_digest",
        "members",
    }
    normalized_rows: list[dict[str, Any]] = []
    occurrence_keys: defaultdict[str, list[_CurrencyTriggerClusterKey]] = defaultdict(
        list
    )
    canonical_member_by_identity: dict[str, dict[str, Any]] = {}
    prior_key: _CurrencyTriggerClusterKey | None = None
    for row_raw in rows_raw:
        if not isinstance(row_raw, Mapping) or set(row_raw) != expected_row_fields:
            raise ValueError("currency-trigger cluster row schema mismatch")
        split_name = row_raw.get("split")
        candidate_id = row_raw.get("candidate_id")
        if split_name != expected_split_name or candidate_id != expected_candidate_id:
            raise ValueError("currency-trigger cluster row scope mismatch")
        trigger_at = _parse_canonical_utc_instant(
            row_raw.get("trigger_at_utc"),
            field="currency-trigger cluster trigger_at_utc",
            exact_minute=True,
        ).isoformat()
        currency = row_raw.get("currency")
        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or any(character < "A" or character > "Z" for character in currency)
        ):
            raise ValueError("currency-trigger cluster currency is noncanonical")
        sign = row_raw.get("sign")
        if isinstance(sign, bool) or sign not in {-1, 1}:
            raise ValueError("currency-trigger cluster sign must be -1 or +1")
        key: _CurrencyTriggerClusterKey = (
            expected_split_name,
            expected_candidate_id,
            trigger_at,
            currency,
            int(sign),
        )
        if prior_key is not None and key <= prior_key:
            raise ValueError("currency-trigger cluster rows are not canonical unique")
        prior_key = key
        members_raw = row_raw.get("members")
        if not isinstance(members_raw, list):
            raise ValueError("currency-trigger cluster members are invalid")
        members: list[dict[str, Any]] = []
        prior_identity: str | None = None
        for member_raw in members_raw:
            member = _validated_factor_member(
                member_raw,
                expected_candidate_id=expected_candidate_id,
                expected_pair=expected_pair,
            )
            identity = str(member["trade_identity_sha256"])
            if prior_identity is not None and identity <= prior_identity:
                raise ValueError(
                    "currency-trigger cluster member identities are not unique sorted"
                )
            prior_identity = identity
            if member["trigger_at_utc"] != trigger_at:
                raise ValueError("factor member trigger does not match cluster key")
            factor_views = _pair_currency_factor_views(
                str(member["pair"]), str(member["side"])
            )
            if (currency, int(sign)) not in factor_views:
                raise ValueError("factor member currency/sign does not match trade")
            prior_member = canonical_member_by_identity.get(identity)
            if prior_member is not None and prior_member != member:
                raise ValueError("factor trade identity has conflicting member values")
            canonical_member_by_identity[identity] = member
            occurrence_keys[identity].append(key)
            members.append(member)
        member_count = _require_nonnegative_int(
            row_raw.get("member_count"),
            field="currency-trigger cluster member_count",
        )
        if member_count != len(members):
            raise ValueError("currency-trigger cluster member count mismatch")
        identities = [str(member["trade_identity_sha256"]) for member in members]
        if row_raw.get("member_identity_digest") != _canonical_sha(identities):
            raise ValueError("currency-trigger cluster member digest mismatch")
        row_total = _require_finite_number(
            row_raw.get("exact_net_r"), field="currency-trigger cluster exact_net_r"
        )
        member_total = sum(float(member["exact_net_r"]) for member in members)
        if not _same_total(row_total, member_total):
            raise ValueError("currency-trigger cluster R total mismatch")
        normalized_rows.append(
            {
                "split": expected_split_name,
                "candidate_id": expected_candidate_id,
                "trigger_at_utc": trigger_at,
                "currency": currency,
                "sign": int(sign),
                "member_count": member_count,
                "exact_net_r": row_total,
                "member_identity_digest": row_raw["member_identity_digest"],
                "members": members,
            }
        )
    if payload.get("cluster_rows_sha256") != _canonical_sha(normalized_rows):
        raise ValueError("currency-trigger cluster rows digest mismatch")
    identities = frozenset(occurrence_keys)
    if len(identities) != resolved_count:
        raise ValueError("resolved trades do not uniquely own factor cluster rows")
    for identity, keys in occurrence_keys.items():
        if len(keys) != CURRENCY_FACTOR_VIEWS_PER_TRADE or len(set(keys)) != len(keys):
            raise ValueError("resolved trade does not own exactly two factor clusters")
        member = canonical_member_by_identity[identity]
        expected_keys = {
            (
                expected_split_name,
                expected_candidate_id,
                str(member["trigger_at_utc"]),
                currency,
                sign,
            )
            for currency, sign in _pair_currency_factor_views(
                str(member["pair"]), str(member["side"])
            )
        }
        if set(keys) != expected_keys:
            raise ValueError("resolved trade factor ownership is incomplete")
    factor_memberships = sum(len(keys) for keys in occurrence_keys.values())
    if factor_memberships != resolved_count * CURRENCY_FACTOR_VIEWS_PER_TRADE:
        raise ValueError("factor membership total invariant failed")
    claimed_memberships = _require_nonnegative_int(
        payload.get("factor_membership_count"),
        field="currency-trigger factor_membership_count",
    )
    if claimed_memberships != factor_memberships:
        raise ValueError("factor membership count mismatch")
    factor_total = sum(float(row["exact_net_r"]) for row in normalized_rows)
    claimed_factor_total = _require_finite_number(
        payload.get("factor_membership_exact_net_r_sum_for_ownership_check"),
        field="currency-trigger factor membership R sum",
    )
    if not _same_total(claimed_factor_total, factor_total) or not _same_total(
        factor_total,
        candidate_total * CURRENCY_FACTOR_VIEWS_PER_TRADE,
    ):
        raise ValueError("factor membership R ownership total invariant failed")
    metric_fields = _currency_trigger_metric_fields(
        normalized_rows,
        candidate_total_exact_net_r=candidate_total,
        cluster_digest=str(payload["cluster_rows_sha256"]),
    )
    if metric_fields != dict(expected_metric_fields):
        raise ValueError("currency-trigger cluster metric fields mismatch")
    if expected_ownership is not None:
        scoped_expected = {
            identity: set(keys)
            for identity, keys in expected_ownership.items()
            if keys
            and keys[0][0] == expected_split_name
            and keys[0][1] == expected_candidate_id
        }
        if scoped_expected != {
            identity: set(keys) for identity, keys in occurrence_keys.items()
        }:
            raise ValueError("producer factor ownership registry mismatch")
    return tuple(normalized_rows), identities


def _validate_trial_daily_summary(
    metric: Mapping[str, Any],
    daily_rows: Sequence[Mapping[str, Any]],
    dates: Sequence[str],
    *,
    field: str,
) -> None:
    if [row.get("utc_date") for row in daily_rows] != list(dates):
        raise ValueError("pair run daily UTC vector mismatch")
    metric_filled = _require_nonnegative_int(
        metric.get("filled_count"), field=f"{field}.filled_count"
    )
    metric_resolved = _require_nonnegative_int(
        metric.get("resolved_count"), field=f"{field}.resolved_count"
    )
    metric_net = _require_finite_number(
        metric.get("exact_net_r"), field=f"{field}.exact_net_r"
    )
    metric_profit = _require_finite_number(
        metric.get("gross_profit_r"), field=f"{field}.gross_profit_r"
    )
    metric_loss = _require_finite_number(
        metric.get("gross_loss_r"), field=f"{field}.gross_loss_r"
    )
    count_fields = (
        "signal_count",
        "embargoed_count",
        "deoverlap_count",
        "unscorable_count",
        "cost_blocked_count",
        "order_creation_blocked_count",
        "unfilled_count",
        "canceled_before_fill_count",
        "gap_attempt_canceled_count",
        "purged_count",
        "truth_window_unresolved_count",
        "unresolved_filled_count",
        "win_count",
        "loss_count",
        "gross_decomposition_unavailable_count",
    )
    counts = {
        name: _require_nonnegative_int(metric.get(name), field=f"{field}.{name}")
        for name in count_fields
    }
    if metric_profit < 0.0 or metric_loss < 0.0 or metric_resolved > metric_filled:
        raise ValueError("pair run split metric count/profit-loss invariant mismatch")
    if counts["win_count"] + counts["loss_count"] > metric_resolved:
        raise ValueError("pair run split metric win/loss count mismatch")
    if counts["unresolved_filled_count"] > (
        counts["purged_count"] + counts["truth_window_unresolved_count"]
    ):
        raise ValueError(
            "pair run unresolved filled count exceeds purge/truth-window ownership"
        )
    if not _same_total(metric_net, metric_profit - metric_loss):
        raise ValueError("pair run split metric net/profit-loss mismatch")
    reasons = metric.get("reason_counts")
    if not isinstance(reasons, Mapping):
        raise ValueError("pair run reason counts are invalid")
    for reason, count in reasons.items():
        if not isinstance(reason, str) or not reason:
            raise ValueError("pair run reason count key is invalid")
        _require_nonnegative_int(count, field=f"{field}.reason_counts.{reason}")

    daily_filled = 0
    daily_resolved = 0
    daily_net = 0.0
    daily_profit = 0.0
    daily_loss = 0.0
    for index, row in enumerate(daily_rows):
        if not isinstance(row, Mapping):
            raise ValueError("pair run daily row is invalid")
        if set(row) != {
            "utc_date",
            "filled_count",
            "resolved_count",
            "exact_net_r",
            "gross_profit_r",
            "gross_loss_r",
        }:
            raise ValueError("pair run daily row schema mismatch")
        prefix = f"{field}.daily[{index}]"
        filled = _require_nonnegative_int(
            row.get("filled_count"), field=f"{prefix}.filled_count"
        )
        resolved = _require_nonnegative_int(
            row.get("resolved_count"), field=f"{prefix}.resolved_count"
        )
        net = _require_finite_number(
            row.get("exact_net_r"), field=f"{prefix}.exact_net_r"
        )
        profit = _require_finite_number(
            row.get("gross_profit_r"), field=f"{prefix}.gross_profit_r"
        )
        loss = _require_finite_number(
            row.get("gross_loss_r"), field=f"{prefix}.gross_loss_r"
        )
        if profit < 0.0 or loss < 0.0 or resolved > filled:
            raise ValueError("pair run daily count/profit-loss invariant mismatch")
        if not _same_total(net, profit - loss):
            raise ValueError("pair run daily net/profit-loss mismatch")
        daily_filled += filled
        daily_resolved += resolved
        daily_net += net
        daily_profit += profit
        daily_loss += loss
    if (
        daily_filled != metric_filled
        or daily_resolved != metric_resolved
        or not _same_total(daily_net, metric_net)
        or not _same_total(daily_profit, metric_profit)
        or not _same_total(daily_loss, metric_loss)
    ):
        raise ValueError("pair run daily aggregates do not match split summary")


def _validated_trials_for_run(
    run: Mapping[str, Any],
    vehicles: Sequence[StoryVehicleV2],
    split_dates: Mapping[str, Sequence[str]],
    *,
    expected_pair: str,
) -> dict[str, Mapping[str, Any]]:
    trials = run.get("all_trials")
    if not isinstance(trials, list) or len(trials) != len(vehicles):
        raise ValueError("pair run all_trials is invalid")
    expected_ids = [row.candidate_id for row in vehicles]
    if [
        row.get("candidate_id") for row in trials if isinstance(row, Mapping)
    ] != expected_ids:
        raise ValueError("pair run candidate trials are incomplete or out of order")

    validated: dict[str, Mapping[str, Any]] = {}
    for vehicle, trial in zip(vehicles, trials, strict=True):
        if not isinstance(trial, Mapping):
            raise ValueError("pair run candidate trial is invalid")
        expected_metadata = {
            "candidate_id": vehicle.candidate_id,
            "hypothesis_id": vehicle.hypothesis_id,
            "story_name": vehicle.story_name,
            "exit_policy_id": vehicle.exit_policy_id,
            "contextual_order_policy": vehicle.contextual_order_policy,
            "allowed_order_modes": list(vehicle.allowed_order_modes),
            "max_hold_seconds": vehicle.max_hold_seconds,
            "profit_target_r": vehicle.profit_target_r,
            "trailing_structural": vehicle.trailing_structural,
            "complexity": vehicle.complexity,
            "no_trade_control": vehicle.no_trade_control,
            "scorecard_eligible": not vehicle.no_trade_control,
        }
        if set(trial) != {
            *expected_metadata,
            "by_split",
            "daily_aggregates_by_split",
            "currency_trigger_factor_clusters_by_split",
        }:
            raise ValueError("pair run candidate trial schema mismatch")
        if any(trial.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("pair run candidate trial metadata mismatch")
        metrics_by_split = trial.get("by_split")
        daily_by_split = trial.get("daily_aggregates_by_split")
        currency_clusters_by_split = trial.get(
            "currency_trigger_factor_clusters_by_split"
        )
        if (
            not isinstance(metrics_by_split, Mapping)
            or not isinstance(daily_by_split, Mapping)
            or not isinstance(currency_clusters_by_split, Mapping)
        ):
            raise ValueError("pair run candidate aggregates are invalid")
        expected_split_names = set() if vehicle.no_trade_control else set(split_dates)
        if (
            set(metrics_by_split) != expected_split_names
            or set(daily_by_split) != expected_split_names
            or set(currency_clusters_by_split) != expected_split_names
        ):
            raise ValueError("pair run candidate split trial definition mismatch")
        for split_name, dates in split_dates.items():
            if vehicle.no_trade_control:
                continue
            metric = metrics_by_split[split_name]
            daily_rows = daily_by_split[split_name]
            if not isinstance(metric, Mapping) or not isinstance(daily_rows, list):
                raise ValueError("pair run split candidate aggregates are missing")
            _validate_trial_daily_summary(
                metric,
                daily_rows,
                dates,
                field=f"{vehicle.candidate_id}.{split_name}",
            )
            cluster_payload = currency_clusters_by_split[split_name]
            cluster_metric_fields = {
                "currency_trigger_factor_cluster_count": metric.get(
                    "currency_trigger_factor_cluster_count"
                ),
                "currency_trigger_min_leave_one_cluster_out_total_r": metric.get(
                    "currency_trigger_min_leave_one_cluster_out_total_r"
                ),
                "currency_trigger_top_cluster": metric.get(
                    "currency_trigger_top_cluster"
                ),
                "currency_trigger_cluster_digest": metric.get(
                    "currency_trigger_cluster_digest"
                ),
            }
            _validate_currency_trigger_cluster_payload(
                cluster_payload,
                expected_candidate_id=vehicle.candidate_id,
                expected_split_name=split_name,
                expected_pair=expected_pair,
                expected_resolved_trade_count=int(metric["resolved_count"]),
                expected_candidate_total_exact_net_r=float(metric["exact_net_r"]),
                expected_metric_fields=cluster_metric_fields,
            )
        validated[vehicle.candidate_id] = trial
    return validated


def combine_adaptive_story_s5_grid_runs(
    pair_runs: Sequence[Mapping[str, Any]],
    splits: Sequence[UtcSplit],
    *,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Pool sealed pair runs on complete entry-UTC-day clusters.

    The combiner never reads capped trade audit rows.  It accepts only pair
    runs created with the exact same split and candidate whitelist receipts,
    then pools their complete per-entry-day aggregates, including zero days.
    """

    normalized_splits = _normalise_splits(splits)
    vehicles = _requested_vehicles(candidate_ids)
    requested_ids = [row.candidate_id for row in vehicles]
    evaluated = tuple(row for row in vehicles if not row.no_trade_control)
    expected_split_rows = _split_receipt_rows(normalized_splits)
    expected_split_digest = _canonical_sha(expected_split_rows)
    expected_candidate_digest = _canonical_sha(requested_ids)
    expected_catalog_digest = _canonical_sha(_story_catalog_receipt_v2())
    expected_evaluator_digest = _canonical_sha(_truth_evaluator_receipt_v2())
    split_dates = {split.name: _split_utc_dates(split) for split in normalized_splits}
    if isinstance(pair_runs, (str, bytes)) or not isinstance(pair_runs, Sequence):
        raise ValueError("pair_runs must be a sequence")
    runs = tuple(pair_runs)
    if not runs:
        raise ValueError("at least one sealed pair run is required")
    run_by_pair: dict[str, Mapping[str, Any]] = {}
    trials_by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("pair run must be an object")
        if (
            run.get("contract") != STORY_GRID_CONTRACT_V2
            or run.get("schema_version") != 2
        ):
            raise ValueError("pair run contract mismatch")
        if run.get("status") not in PAIR_RUN_ALLOWED_STATUSES:
            raise ValueError("pair run status is not combinable")
        claimed = run.get("result_sha256")
        if not isinstance(claimed, str) or claimed != _canonical_sha(
            {key: value for key, value in run.items() if key != "result_sha256"}
        ):
            raise ValueError("pair run result seal is invalid")
        if run.get("requested_candidate_ids") != requested_ids:
            raise ValueError("pair run candidate whitelist mismatch")
        if run.get("candidate_whitelist_sha256") != expected_candidate_digest:
            raise ValueError("pair run candidate whitelist digest mismatch")
        if run.get("evaluated_candidate_ids") != [
            row.candidate_id for row in evaluated
        ]:
            raise ValueError("pair run evaluated candidate scope mismatch")
        if run.get("requested_control_candidate_ids") != [
            row.candidate_id for row in vehicles if row.no_trade_control
        ]:
            raise ValueError("pair run control candidate scope mismatch")
        if run.get("candidate_count") != len(vehicles):
            raise ValueError("pair run candidate count mismatch")
        if run.get("split_receipt") != expected_split_rows:
            raise ValueError("pair run split receipt mismatch")
        if run.get("split_digest") != expected_split_digest:
            raise ValueError("pair run split digest mismatch")
        if (
            run.get("story_catalog_policy") != STORY_CATALOG_POLICY_V2
            or run.get("truth_policy") != STORY_TRUTH_POLICY_V2
        ):
            raise ValueError("pair run story/truth policy mismatch")
        if run.get("story_catalog_sha256") != expected_catalog_digest:
            raise ValueError("pair run story catalog digest mismatch")
        if run.get("truth_evaluator_sha256") != expected_evaluator_digest:
            raise ValueError("pair run truth evaluator digest mismatch")
        expected_run_policy_fields = {
            "story_template_count": len(build_story_templates_v2()),
            "predeclared_selectable_candidate_count": 50,
            "predeclared_control_candidate_count": 1,
            "contextual_order_cross_product_forbidden": True,
            "setup_trigger_entry_policy": "T_SETUP_LT_T_TRIGGER_LT_T_ENTRY",
            "entry_ttl_boundary": "EXCLUSIVE",
            "intrabar_resting_fill_s5_policy": (
                "NO_TARGET;STOP_RANGE_CHARGED_CONSERVATIVELY;NO_PREFILL_OPEN_GAP"
            ),
            "entry_gap_invalid_geometry_policy": (
                "BROKER_ON_FILL_DEPENDENT_ORDER_LOSS_CANCEL_NO_FILL"
            ),
            "split_embargo_seconds": ENTRY_TTL_SECONDS + MAX_HOLD_SECONDS,
            "currency_trigger_cluster_contract": (CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1),
            "currency_trigger_cluster_key_policy": (
                CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1
            ),
            "currency_factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
            "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
            "currency_trigger_cluster_rows_untruncated": True,
            "currency_trigger_factor_clusters_are_not_candidate_summed": True,
        }
        if any(
            run.get(key) != value for key, value in expected_run_policy_fields.items()
        ):
            raise ValueError("pair run evaluator policy metadata mismatch")
        if (
            run.get("price_precision_policy") != PRICE_PRECISION_POLICY_V2
            or run.get("price_cost_scope") != PRICE_COST_SCOPE_V2
        ):
            raise ValueError("pair run price/cost scope mismatch")
        if (
            run.get("daily_aggregates_complete") is not True
            or run.get("daily_cluster_basis") != "ENTRY_UTC_DATE"
        ):
            raise ValueError("pair run complete daily aggregates are missing")
        if not all(run.get(key) == value for key, value in _AUTHORITY.items()):
            raise ValueError("pair run authority invariant mismatch")
        pair_value = run.get("pair")
        if not _is_canonical_pair_name(pair_value):
            raise ValueError("pair run pair must use canonical uppercase AAA_BBB")
        pair = pair_value
        _pair_currency_factor_views(pair, "LONG")
        if pair in run_by_pair:
            raise ValueError("pair runs must have unique canonical pairs")
        aggregation = run.get("aggregation")
        if not isinstance(aggregation, Mapping):
            raise ValueError("pair run aggregation is invalid")
        source_count = _require_nonnegative_int(
            aggregation.get("source_candle_count"),
            field="aggregation.source_candle_count",
        )
        if (run.get("status") == "COMPLETE") != (source_count > 0):
            raise ValueError("pair run status/source-count mismatch")
        validated_trials = _validated_trials_for_run(
            run,
            vehicles,
            split_dates,
            expected_pair=pair,
        )
        if run.get("status") == "NO_DATA":
            for vehicle in evaluated:
                trial = validated_trials[vehicle.candidate_id]
                for split_name in split_dates:
                    metric = trial["by_split"][split_name]
                    if (
                        metric["filled_count"] != 0
                        or metric["resolved_count"] != 0
                        or float(metric["exact_net_r"]) != 0.0
                        or float(metric["gross_profit_r"]) != 0.0
                        or float(metric["gross_loss_r"]) != 0.0
                    ):
                        raise ValueError("NO_DATA pair run contains economic outcomes")
        run_by_pair[pair] = run
        trials_by_pair[pair] = validated_trials

    combined_rows: list[dict[str, Any]] = []
    for vehicle in vehicles:
        if vehicle.no_trade_control:
            combined_rows.append(
                {
                    "candidate_id": vehicle.candidate_id,
                    "hypothesis_id": vehicle.hypothesis_id,
                    "no_trade_control": True,
                    "by_split": {},
                    "economic_screen_by_split": {},
                    "currency_trigger_factor_clusters_by_split": {},
                }
            )
            continue
        by_split: dict[str, Any] = {}
        screen_by_split: dict[str, Any] = {}
        currency_clusters_by_split: dict[str, dict[str, Any]] = {}
        for split in normalized_splits:
            dates = split_dates[split.name]
            pooled_daily = {
                utc_date: {
                    "exact_net_r": 0.0,
                    "resolved_count": 0,
                    "filled_count": 0,
                    "gross_profit_r": 0.0,
                    "gross_loss_r": 0.0,
                }
                for utc_date in dates
            }
            resolved = 0
            unresolved_or_purged = 0
            unresolved_filled = 0
            gross_profit_r = 0.0
            gross_loss_r = 0.0
            exact_net_r = 0.0
            contributing_pairs: set[str] = set()
            reason_counts: defaultdict[str, int] = defaultdict(int)
            global_factor_clusters: _CurrencyTriggerClusterMap = {}
            global_factor_ownership_lists: defaultdict[
                str, list[_CurrencyTriggerClusterKey]
            ] = defaultdict(list)
            global_identity_pair: dict[str, str] = {}
            for pair in sorted(run_by_pair):
                trial = trials_by_pair[pair][vehicle.candidate_id]
                metrics_by_split = trial.get("by_split")
                daily_by_split = trial.get("daily_aggregates_by_split")
                factor_by_split = trial.get("currency_trigger_factor_clusters_by_split")
                if (
                    not isinstance(metrics_by_split, Mapping)
                    or not isinstance(daily_by_split, Mapping)
                    or not isinstance(factor_by_split, Mapping)
                ):
                    raise ValueError("pair run candidate aggregates are invalid")
                metric = metrics_by_split.get(split.name)
                daily_rows = daily_by_split.get(split.name)
                factor_payload = factor_by_split.get(split.name)
                if (
                    not isinstance(metric, Mapping)
                    or not isinstance(daily_rows, list)
                    or not isinstance(factor_payload, Mapping)
                ):
                    raise ValueError("pair run split candidate aggregates are missing")
                if [row.get("utc_date") for row in daily_rows] != list(dates):
                    raise ValueError("pair run daily UTC vector mismatch")
                pair_resolved = int(metric.get("resolved_count") or 0)
                resolved += pair_resolved
                if pair_resolved > 0:
                    contributing_pairs.add(pair)
                unresolved_or_purged += int(metric.get("purged_count") or 0)
                unresolved_or_purged += int(
                    metric.get("truth_window_unresolved_count") or 0
                )
                unresolved_filled += int(metric.get("unresolved_filled_count") or 0)
                exact_net_r += float(metric.get("exact_net_r") or 0.0)
                gross_profit_r += float(metric.get("gross_profit_r") or 0.0)
                gross_loss_r += float(metric.get("gross_loss_r") or 0.0)
                reasons = metric.get("reason_counts")
                if not isinstance(reasons, Mapping):
                    raise ValueError("pair run reason counts are invalid")
                for reason, count in reasons.items():
                    reason_counts[str(reason)] += int(count)
                for row in daily_rows:
                    if not isinstance(row, Mapping):
                        raise ValueError("pair run daily row is invalid")
                    day = pooled_daily[str(row["utc_date"])]
                    day["exact_net_r"] += float(row.get("exact_net_r") or 0.0)
                    day["resolved_count"] += int(row.get("resolved_count") or 0)
                    day["filled_count"] += int(row.get("filled_count") or 0)
                    day["gross_profit_r"] += float(row.get("gross_profit_r") or 0.0)
                    day["gross_loss_r"] += float(row.get("gross_loss_r") or 0.0)
                factor_metric_fields = {
                    "currency_trigger_factor_cluster_count": metric.get(
                        "currency_trigger_factor_cluster_count"
                    ),
                    "currency_trigger_min_leave_one_cluster_out_total_r": metric.get(
                        "currency_trigger_min_leave_one_cluster_out_total_r"
                    ),
                    "currency_trigger_top_cluster": metric.get(
                        "currency_trigger_top_cluster"
                    ),
                    "currency_trigger_cluster_digest": metric.get(
                        "currency_trigger_cluster_digest"
                    ),
                }
                factor_rows, factor_identities = (
                    _validate_currency_trigger_cluster_payload(
                        factor_payload,
                        expected_candidate_id=vehicle.candidate_id,
                        expected_split_name=split.name,
                        expected_pair=pair,
                        expected_resolved_trade_count=pair_resolved,
                        expected_candidate_total_exact_net_r=float(
                            metric.get("exact_net_r") or 0.0
                        ),
                        expected_metric_fields=factor_metric_fields,
                    )
                )
                duplicate_identities = set(factor_identities) & set(
                    global_identity_pair
                )
                if duplicate_identities:
                    raise ValueError(
                        "duplicate currency-trigger factor trade identity "
                        "across pair runs"
                    )
                for identity in factor_identities:
                    global_identity_pair[identity] = pair
                for factor_row in factor_rows:
                    cluster_key = _currency_trigger_cluster_key_from_row(factor_row)
                    global_cluster = global_factor_clusters.setdefault(cluster_key, {})
                    for factor_member in factor_row["members"]:
                        identity = str(factor_member["trade_identity_sha256"])
                        if identity in global_cluster:
                            raise ValueError(
                                "duplicate trade identity in merged currency-trigger "
                                "factor cluster"
                            )
                        global_cluster[identity] = dict(factor_member)
                        global_factor_ownership_lists[identity].append(cluster_key)
            daily_net_r = [
                {
                    "utc_date": utc_date,
                    "exact_net_r": pooled_daily[utc_date]["exact_net_r"],
                    "resolved_count": pooled_daily[utc_date]["resolved_count"],
                }
                for utc_date in dates
            ]
            active_days = sum(
                pooled_daily[utc_date]["filled_count"] > 0 for utc_date in dates
            )
            average_net_r = exact_net_r / resolved if resolved else None
            average_daily_net_r = exact_net_r / len(dates) if dates else None
            profit_factor_r = (
                gross_profit_r / gross_loss_r if gross_loss_r > 0.0 else None
            )
            profit_factor_infinite = gross_loss_r == 0.0 and gross_profit_r > 0.0
            loocv_min = (
                min(
                    exact_net_r - pooled_daily[utc_date]["exact_net_r"]
                    for utc_date in dates
                )
                if dates
                else None
            )
            global_factor_ownership = {
                identity: tuple(keys)
                for identity, keys in global_factor_ownership_lists.items()
            }
            global_factor_payload, global_factor_metric_fields = (
                _build_currency_trigger_cluster_payload(
                    vehicle.candidate_id,
                    split.name,
                    global_factor_clusters,
                    global_factor_ownership,
                    resolved_trade_count=resolved,
                    candidate_total_exact_net_r=exact_net_r,
                    expected_pair=None,
                )
            )
            currency_clusters_by_split[split.name] = global_factor_payload
            minimum_currency_trigger_loco_total_r = global_factor_metric_fields[
                "currency_trigger_min_leave_one_cluster_out_total_r"
            ]
            gates = {
                "resolved_trade_floor_passed": resolved >= SCREEN_MIN_RESOLVED_TRADES,
                "active_entry_day_floor_passed": active_days >= SCREEN_MIN_ACTIVE_DAYS,
                "contributing_pair_floor_passed": len(contributing_pairs)
                >= SCREEN_MIN_CONTRIBUTING_PAIRS,
                "average_net_r_positive": average_net_r is not None
                and average_net_r > 0.0,
                "average_daily_net_r_positive": average_daily_net_r is not None
                and average_daily_net_r > 0.0,
                "profit_factor_r_above_one": profit_factor_infinite
                or (
                    profit_factor_r is not None
                    and profit_factor_r > SCREEN_MIN_PROFIT_FACTOR_R
                ),
                "loocv_each_day_removed_total_r_positive": loocv_min is not None
                and loocv_min > 0.0,
                "leave_one_currency_trigger_cluster_min_total_r_positive": (
                    minimum_currency_trigger_loco_total_r is not None
                    and minimum_currency_trigger_loco_total_r > 0.0
                ),
                "no_unresolved_or_purged": unresolved_or_purged == 0,
                "no_unresolved_filled": unresolved_filled == 0,
            }
            screen = {
                "resolved_count": resolved,
                "unresolved_filled_count": unresolved_filled,
                "active_entry_day_count": active_days,
                "contributing_pair_count": len(contributing_pairs),
                "resolved_pair_count": len(contributing_pairs),
                **global_factor_metric_fields,
                "gates": gates,
                "eligible": all(gates.values()),
                "screen_is_statistical_proof": False,
            }
            by_split[split.name] = {
                "resolved_count": resolved,
                "unresolved_or_purged_count": unresolved_or_purged,
                "unresolved_filled_count": unresolved_filled,
                "active_entry_day_count": active_days,
                "contributing_pair_count": len(contributing_pairs),
                "resolved_pair_count": len(contributing_pairs),
                "exact_net_r": exact_net_r,
                "gross_profit_r": gross_profit_r,
                "gross_loss_r": gross_loss_r,
                "average_net_r": average_net_r,
                "average_daily_net_r": average_daily_net_r,
                "profit_factor_r": profit_factor_r,
                "profit_factor_r_infinite": profit_factor_infinite,
                "loocv_removed_day_min_total_r": loocv_min,
                "trades_per_day": resolved / len(dates) if dates else None,
                "reason_counts": dict(sorted(reason_counts.items())),
                "daily_net_r": daily_net_r,
                **global_factor_metric_fields,
            }
            screen_by_split[split.name] = screen
        combined_rows.append(
            {
                "candidate_id": vehicle.candidate_id,
                "hypothesis_id": vehicle.hypothesis_id,
                "story_name": vehicle.story_name,
                "exit_policy_id": vehicle.exit_policy_id,
                "no_trade_control": False,
                "by_split": by_split,
                "economic_screen_by_split": screen_by_split,
                "currency_trigger_factor_clusters_by_split": (
                    currency_clusters_by_split
                ),
            }
        )

    validation_names = [
        split.name for split in normalized_splits if split.name.upper() == "VALIDATION"
    ]
    economic_survivors = [
        row["candidate_id"]
        for row in combined_rows
        if not row["no_trade_control"]
        and validation_names
        and row["economic_screen_by_split"][validation_names[0]]["eligible"]
    ]
    body: dict[str, Any] = {
        "contract": STORY_GRID_COMBINED_CONTRACT_V2,
        "schema_version": 2,
        "status": "COMPLETE",
        "pair_count": len(run_by_pair),
        "pairs": sorted(run_by_pair),
        "pair_identity_policy": "STRICT_ASCII_UPPERCASE_AAA_BBB_NO_NORMALIZATION",
        "requested_candidate_ids": requested_ids,
        "evaluated_candidate_ids": [row.candidate_id for row in evaluated],
        "candidate_whitelist_sha256": expected_candidate_digest,
        "story_catalog_policy": STORY_CATALOG_POLICY_V2,
        "truth_policy": STORY_TRUTH_POLICY_V2,
        "story_catalog_sha256": expected_catalog_digest,
        "truth_evaluator_sha256": expected_evaluator_digest,
        "accepted_pair_run_statuses": list(PAIR_RUN_ALLOWED_STATUSES),
        "price_precision_policy": PRICE_PRECISION_POLICY_V2,
        "price_cost_scope": dict(PRICE_COST_SCOPE_V2),
        "currency_trigger_cluster_contract": (CURRENCY_TRIGGER_CLUSTER_CONTRACT_V1),
        "currency_trigger_cluster_key_policy": (CURRENCY_TRIGGER_CLUSTER_KEY_POLICY_V1),
        "currency_factor_view_policy": CURRENCY_FACTOR_VIEW_POLICY_V1,
        "currency_factor_views_per_trade": CURRENCY_FACTOR_VIEWS_PER_TRADE,
        "currency_trigger_cluster_rows_untruncated": True,
        "currency_trigger_factor_clusters_are_not_candidate_summed": True,
        "currency_trigger_global_merge_policy": (
            "MERGE_EQUAL_CLUSTER_KEYS_ACROSS_PAIRS;"
            "REJECT_CROSS_PAIR_DUPLICATE_TRADE_IDENTITIES"
        ),
        "split_receipt": expected_split_rows,
        "split_digest": expected_split_digest,
        "daily_cluster_basis": "ENTRY_UTC_DATE",
        "daily_zero_fill_policy": "ALL_SPLIT_CALENDAR_UTC_DAYS",
        "daily_aggregates_source": "COMPLETE_PAIR_RUN_AGGREGATES_NOT_AUDIT_ROWS",
        "candidate_metrics": combined_rows,
        "economic_survivor_ids": economic_survivors,
        "economic_screen_is_statistical_proof": False,
        **_AUTHORITY,
    }
    return {**body, "result_sha256": _canonical_sha(body)}


__all__ = [
    "StoryExitV2",
    "StoryTemplateV2",
    "StoryVehicleV2",
    "UtcSplit",
    "build_story_templates_v2",
    "build_story_vehicle_catalog_v2",
    "combine_adaptive_story_s5_grid_runs",
    "run_adaptive_story_s5_grid",
]
