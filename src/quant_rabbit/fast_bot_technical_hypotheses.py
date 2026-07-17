"""Prospective, regime-aware technical hypotheses for fast-bot episodes.

The chart producer exposes many indicators, but an indicator is not a vote:
ADX/ATR are directionless, oscillators change meaning by regime, and moving
average crosses can be stale.  This module freezes a small fixed catalog that
assigns one job to each input (regime, direction, momentum, volatility,
location, trigger) and emits every accepted and rejected hypothesis before
future S5 truth is known.

The output is diagnostic only.  Raw rule scores are not probabilities and no
row can grant order, risk, tuning, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_rabbit.analysis.market_state import TAXONOMY_SECTIONS


FEATURE_SNAPSHOT_CONTRACT = "QR_FAST_BOT_EPISODE_TECHNICAL_FEATURE_SNAPSHOT_V1"
CATALOG_CONTRACT = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_CATALOG_V1"
SHADOW_CONTRACT = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_SHADOW_V1"
HYPOTHESIS_CONTRACT = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_V1"
EVALUATOR_POLICY_V1 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_EVALUATOR_V1"

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
SIDES = {"LONG": "UP", "SHORT": "DOWN"}
OPPOSITE_SIDE = {"LONG": "SHORT", "SHORT": "LONG"}
HYPOTHESIS_FAMILIES = (
    "TREND",
    "PULLBACK",
    "BREAKOUT",
    "BREAKOUT_FAILURE",
    "RANGE",
    "EXHAUSTION",
)
KNOWN_ROUTE_FAMILIES = frozenset(
    {
        "BREAKOUT_CONTINUATION",
        "RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
        "UNRESOLVED_BREAK_ATTEMPT",
        "AMBIGUOUS",
    }
)
ROUTE_FAMILY_BY_BRANCH = {
    "ACCEPTED": "BREAKOUT_CONTINUATION",
    "REJECTED": "RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
}
MARKET_STATE_ENUM_FEATURES = {
    key: frozenset(values) for key, values in TAXONOMY_SECTIONS.items()
}
MARKET_STATE_FEATURES = frozenset(MARKET_STATE_ENUM_FEATURES) | {
    "confidence",
    "evidence_complete",
}
NUMERIC_INDICATOR_FEATURES = frozenset(
    {
        "close",
        "sma_20",
        "adx",
        "adx_14",
        "adx_percentile_100",
        "plus_di_14",
        "minus_di_14",
        "ema_12",
        "ema_20",
        "ema_50",
        "ema_slope_5",
        "ema_slope_20",
        "ema_gap_5_24",
        "ema_gap_12_48",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_scaled",
        "rsi_14",
        "stoch_rsi",
        "stoch_rsi_k",
        "stoch_rsi_d",
        "williams_r",
        "williams_r_14",
        "cci_14",
        "mfi_14",
        "bb_position",
        "bb_width",
        "bollinger_position",
        "bollinger_width",
        "donchian_position",
        "donchian_high",
        "donchian_low",
        "donchian_width_pips",
        "atr",
        "atr_14",
        "atr_pips",
        "atr_percentile",
        "atr_percentile_100",
        "atr_percentile_24h",
        "choppiness",
        "choppiness_14",
        "hurst",
        "hurst_100",
        "hurst_returns",
        "half_life",
        "half_life_60",
        "roc",
        "roc_5",
        "roc_10",
        "roc_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_span_pips",
        "keltner_width",
        "bb_squeeze",
        "bb_width_percentile_100",
        "aroon_up_14",
        "aroon_down_14",
        "aroon_osc",
        "aroon_osc_14",
        "vortex_plus_14",
        "vortex_minus_14",
        "supertrend",
        "supertrend_value",
        "supertrend_dir",
        "psar_value",
        "psar_dir",
        "hull_ma_20",
        "kama_10",
        "alma_20",
        "linreg_slope_20",
        "linreg_r2_20",
        "linreg_channel_upper",
        "linreg_channel_lower",
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_span_a",
        "ichimoku_span_b",
        "ichimoku_cloud_pos",
        "zscore",
        "zscore_20",
        "z_score_20",
        "realized_vol_20",
        "vwap",
        "vwap_gap",
        "vwap_gap_pips",
        "avwap_anchor",
        "avwap_upper_1sd",
        "avwap_lower_1sd",
        "avwap_upper_2sd",
        "avwap_lower_2sd",
        "avwap_swing_high",
        "avwap_swing_low",
        "price_vs_ema50_per_atr",
    }
)
ENUM_INDICATOR_FEATURES = frozenset(
    {
        "ema_order",
        "atr_regime",
        "regime_quantile",
    }
)
INDICATOR_SERIES_FEATURES = frozenset(
    {
        "rsi_14",
        "macd_hist",
        "adx_14",
        "atr_pips",
        "ema_12_minus_50_pips",
    }
)
MAX_INDICATOR_SERIES_VALUES = 30
MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES = 64 * 1024
FEATURE_SNAPSHOT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "pair",
        "handoff_cycle_generated_at_utc",
        "feature_allowlist_version",
        "timeframes",
        "hypothesis_families",
        "raw_chart_packet_embedded",
        "diagnostic_only",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "contract_sha256",
    }
)
FEATURE_ROW_KEYS = frozenset(
    {
        "timeframe",
        "complete_candle_close_utc",
        "market_state",
        "indicators",
        "indicator_series",
        "feature_sha256",
    }
)
FORECAST_TARGETS = (
    "P_SIDE_NET_POSITIVE_AFTER_COST",
    "NET_EXECUTABLE_EXCURSION_Q10_Q50_Q90_PIPS",
    "P_FILL",
    "P_TP_BEFORE_SL_GIVEN_FILL",
    "TIME_TO_FILL_TP_SL_COMPETING_RISK_CDF",
    "P_INITIAL_THESIS_FAILURE",
    "P_PROFITABLE_REVERSE_AFTER_CAUSAL_FAILURE",
)


# V1 is an append-only evaluator policy: future tuning must add V2 constants
# and dispatch instead of changing these rows or their evaluator functions.
CATALOG_V1: tuple[dict[str, Any], ...] = (
    {
        "hypothesis_id": "H01",
        "family": "TREND_CONTINUATION",
        "execution_method": "TREND_CONTINUATION",
        "entry_vehicle": "STOP_AFTER_CLOSE_CONFIRMATION",
        "arm_truth_join_policy": "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE",
        "regime_role": "M15_H1_TREND_OR_PRE_TREND",
        "direction_role": "HTF_DIRECTION_PLUS_M5_EMA_DI_SUPERTREND",
        "confirmer_role": "RSI_MACD_ROC_WITH_ADX_AND_ATR_SEQUENCE",
    },
    {
        "hypothesis_id": "H02",
        "family": "TREND_PULLBACK",
        "execution_method": "TREND_CONTINUATION",
        "entry_vehicle": "SHALLOW_PULLBACK_THEN_M1_REACCELERATION",
        "arm_truth_join_policy": "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE",
        "regime_role": "M15_H1_TREND_OR_PRE_TREND",
        "direction_role": "HTF_DIRECTION_PLUS_M5_TREND_CORE",
        "confirmer_role": "M1_RSI50_OR_MACD_RECLAIM_FROM_VALUE",
    },
    {
        "hypothesis_id": "H03",
        "family": "RANGE_ROTATION",
        "execution_method": "RANGE_ROTATION",
        "entry_vehicle": "PASSIVE_EDGE_AFTER_REENTRY",
        "arm_truth_join_policy": "EXISTING_PASSIVE_QUOTE_PROXY",
        "regime_role": "M5_M15_RANGE_WITH_LOW_TREND_STRENGTH",
        "direction_role": "EDGE_TOWARD_VALUE",
        "confirmer_role": "RSI_REENTRY_PLUS_ZSCORE_BB_DONCHIAN_LOCATION",
    },
    {
        "hypothesis_id": "H04",
        "family": "PRETREND_BREAKOUT",
        "execution_method": "TREND_CONTINUATION",
        "entry_vehicle": "STOP_AFTER_CLOSE_CONFIRMED_RAIL_BREAK",
        "arm_truth_join_policy": "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE",
        "regime_role": "M15_H1_COMPRESSION_OR_PRE_TREND",
        "direction_role": "CONFIRMED_ATTEMPT_DIRECTION",
        "confirmer_role": "M5_MOMENTUM_AND_VOLATILITY_EXPANSION",
    },
    {
        "hypothesis_id": "H05",
        "family": "BREAKOUT_FAILURE",
        "execution_method": "BREAKOUT_FAILURE",
        "entry_vehicle": "FADE_AFTER_CLOSE_REACCEPTANCE",
        "arm_truth_join_policy": "EXISTING_PASSIVE_QUOTE_PROXY",
        "regime_role": "SEALED_RANGE_BREAK_ATTEMPT",
        "direction_role": "OPPOSITE_REJECTED_ATTEMPT",
        "confirmer_role": "SEALED_NEXT_M1_CONFIRMATION",
    },
    {
        "hypothesis_id": "H06",
        "family": "EXHAUSTION_REVERSAL",
        "execution_method": "RANGE_ROTATION",
        "entry_vehicle": "REVERSAL_CLOSE_ONLY_NO_BLIND_FADE",
        "arm_truth_join_policy": "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE",
        "regime_role": "MATURE_OR_EXHAUSTING_TREND",
        "direction_role": "OPPOSITE_OLD_TREND",
        "confirmer_role": "EXTENSION_PLUS_MOMENTUM_ADX_ATR_DECELERATION",
    },
    {
        "hypothesis_id": "H07",
        "family": "SESSION_OPEN_EXPANSION",
        "execution_method": "TREND_CONTINUATION",
        "entry_vehicle": "SESSION_RAIL_STOP_AFTER_CONFIRMATION",
        "arm_truth_join_policy": "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE",
        "regime_role": "LONDON_OR_NEW_YORK_OPEN_AFTER_COMPRESSION",
        "direction_role": "M5_CONFIRMED_EXPANSION_DIRECTION",
        "confirmer_role": "M5_RSI_MACD_ROC_AND_ATR_EXPANSION",
    },
    {
        "hypothesis_id": "H08",
        "family": "NO_TRADE_CONTROL",
        "execution_method": None,
        "entry_vehicle": "NONE",
        "arm_truth_join_policy": "ZERO_PNL_CONTROL",
        "regime_role": "TRANSITION_UNKNOWN_CONFLICT_OR_NO_DIRECTIONAL_SURVIVOR",
        "direction_role": "EITHER",
        "confirmer_role": "CONTROL_ONLY",
    },
)


def technical_hypothesis_catalog() -> dict[str, Any]:
    """Return the sealed, fixed best-of-search universe."""

    body = {
        "contract": CATALOG_CONTRACT,
        "schema_version": 1,
        "evaluator_policy": EVALUATOR_POLICY_V1,
        "selection_policy": (
            "TRAIN_THRESHOLDS_THEN_VALIDATION_ONE_SE_SIMPLEST_PER_FAMILY_"
            "THEN_ONE_FIXED_HOLDOUT"
        ),
        "multiple_testing_policy": "FAMILY_HOLM_OR_SPA_PLUS_EPISODE_DAY_BASKET_CLUSTER",
        "maximum_one_regime_gate": True,
        "maximum_one_direction_core": True,
        "maximum_one_optional_confirmer_group": True,
        "correlated_indicator_majority_vote_forbidden": True,
        "discarded_hypotheses_remain_permanent_shadow": True,
        "probability_claim_requires_forward_calibration": True,
        "hypotheses": [dict(item) for item in CATALOG_V1],
        "forecast_targets": list(FORECAST_TARGETS),
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal(body)


def build_fast_bot_technical_hypotheses(
    feature_snapshot: Mapping[str, Any],
    *,
    attempt_direction: str,
    branch_outcome: str,
    route_family: str,
    spread_pips: float,
    m5_atr_pips: float,
    spread_to_m5_atr: float,
) -> dict[str, Any]:
    """Evaluate every fixed hypothesis against one causal feature snapshot."""

    attempt = attempt_direction if isinstance(attempt_direction, str) else ""
    branch = branch_outcome if isinstance(branch_outcome, str) else ""
    route = route_family if isinstance(route_family, str) else ""
    snapshot_valid = isinstance(feature_snapshot, Mapping) and _feature_snapshot_valid(
        feature_snapshot
    )
    context_valid = _episode_context_valid(
        attempt_direction=attempt,
        branch_outcome=branch,
        route_family=route,
    )
    snapshot_m5_atr = (
        _snapshot_m5_atr(feature_snapshot) if snapshot_valid else None
    )
    catalog = technical_hypothesis_catalog()
    cost_state = _cost_state(
        spread_pips=spread_pips,
        m5_atr_pips=m5_atr_pips,
        spread_to_m5_atr=spread_to_m5_atr,
        feature_m5_atr_pips=snapshot_m5_atr,
    )
    base = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "evaluator_policy": EVALUATOR_POLICY_V1,
        "catalog_contract_sha256": catalog["contract_sha256"],
        "catalog_hypothesis_ids": [item["hypothesis_id"] for item in CATALOG_V1],
        "attempt_direction": attempt,
        "branch_outcome": branch,
        "route_family": route,
        "cost_state": cost_state,
        "cost_state_sha256": _canonical_sha(cost_state),
        "forecast_targets": list(FORECAST_TARGETS),
        "forecast_values_status": "PENDING_PROSPECTIVE_FORWARD_CALIBRATION",
        "raw_rule_score_is_probability": False,
        "discarded_hypotheses_retained": True,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }
    if not snapshot_valid or not context_valid or not cost_state["valid"]:
        if not snapshot_valid:
            status = "INVALID_FEATURE_SNAPSHOT"
        elif not context_valid:
            status = "INVALID_EPISODE_CONTEXT"
        else:
            status = "INVALID_COST_STATE"
        return _seal(
            {
                **base,
                "status": status,
                "pair": (
                    str(feature_snapshot.get("pair") or "")
                    if isinstance(feature_snapshot, Mapping)
                    else ""
                ),
                "feature_snapshot_sha256": (
                    str(feature_snapshot["contract_sha256"])
                    if snapshot_valid
                    else None
                ),
                "active_directional_count": 0,
                "hypotheses": [],
            }
        )
    views = _views(feature_snapshot)
    context = {
        "attempt_direction": attempt,
        "branch_outcome": branch,
        "route_family": route,
        "cycle": _parse_utc(feature_snapshot["handoff_cycle_generated_at_utc"]),
    }
    evaluators: dict[str, Callable[[Mapping[str, _View], Mapping[str, Any]], _Evaluation]] = {
        "H01": _trend_continuation,
        "H02": _trend_pullback,
        "H03": _range_rotation,
        "H04": _pretrend_breakout,
        "H05": _breakout_failure,
        "H06": _exhaustion_reversal,
        "H07": _session_open_expansion,
    }
    rows: list[dict[str, Any]] = []
    active_directional = 0
    active_sides: set[str] = set()
    for spec in CATALOG_V1[:-1]:
        result = evaluators[str(spec["hypothesis_id"])](views, context)
        if result.active:
            active_directional += 1
            if result.side in SIDES:
                active_sides.add(result.side)
        rows.append(_hypothesis_row(spec, result, feature_snapshot=feature_snapshot))
    control = _no_trade_control(
        views,
        context,
        active_directional=active_directional,
        active_sides=active_sides,
    )
    rows.append(
        _hypothesis_row(CATALOG_V1[-1], control, feature_snapshot=feature_snapshot)
    )
    body = {
        **base,
        "status": "EMITTED",
        "pair": str(feature_snapshot["pair"]),
        "feature_snapshot_sha256": str(feature_snapshot["contract_sha256"]),
        "active_directional_count": active_directional,
        "active_hypothesis_ids": [
            row["hypothesis_id"] for row in rows if row["status"] == "ACTIVE_SHADOW"
        ],
        "hypotheses": rows,
    }
    return _seal(body)


def technical_hypothesis_shadow_valid(
    value: Any,
    *,
    feature_snapshot: Mapping[str, Any],
    attempt_direction: str,
    branch_outcome: str,
    route_family: str,
    spread_pips: float,
    m5_atr_pips: float,
    spread_to_m5_atr: float,
) -> bool:
    """Deep-validate a shadow by deterministically rebuilding frozen input."""

    if not isinstance(value, Mapping) or not _sealed_valid(value, SHADOW_CONTRACT):
        return False
    validator = {
        EVALUATOR_POLICY_V1: _technical_hypothesis_shadow_v1_valid,
    }.get(value.get("evaluator_policy"))
    if validator is None:
        return False
    return validator(
        value,
        feature_snapshot=feature_snapshot,
        attempt_direction=attempt_direction,
        branch_outcome=branch_outcome,
        route_family=route_family,
        spread_pips=spread_pips,
        m5_atr_pips=m5_atr_pips,
        spread_to_m5_atr=spread_to_m5_atr,
    )


def _technical_hypothesis_shadow_v1_valid(
    value: Mapping[str, Any],
    *,
    feature_snapshot: Mapping[str, Any],
    attempt_direction: str,
    branch_outcome: str,
    route_family: str,
    spread_pips: float,
    m5_atr_pips: float,
    spread_to_m5_atr: float,
) -> bool:
    """Validate immutable V1; never reinterpret it when adding a later policy."""

    try:
        expected = build_fast_bot_technical_hypotheses(
            feature_snapshot,
            attempt_direction=attempt_direction,
            branch_outcome=branch_outcome,
            route_family=route_family,
            spread_pips=spread_pips,
            m5_atr_pips=m5_atr_pips,
            spread_to_m5_atr=spread_to_m5_atr,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        expected.get("status") == "EMITTED"
        and value.get("status") == "EMITTED"
        and dict(value) == expected
    )


class _View:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.market = row.get("market_state") if isinstance(row.get("market_state"), Mapping) else {}
        self.indicators = row.get("indicators") if isinstance(row.get("indicators"), Mapping) else {}
        self.series = row.get("indicator_series") if isinstance(row.get("indicator_series"), Mapping) else {}

    def market_text(self, key: str) -> str:
        return str(self.market.get(key) or "UNKNOWN").upper()

    def number(self, *keys: str) -> float | None:
        for key in keys:
            value = self.indicators.get(key)
            parsed = _number(value)
            if parsed is not None:
                return parsed
        return None

    def values(self, key: str) -> tuple[float, ...]:
        raw = self.series.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        parsed = tuple(value for item in raw if (value := _number(item)) is not None)
        return parsed[-30:]


class _Evaluation:
    def __init__(
        self,
        *,
        active: bool,
        side: str | None,
        score: int,
        evidence: Sequence[str],
        blockers: Sequence[str],
    ) -> None:
        self.active = active
        self.side = side
        self.score = score
        self.evidence = tuple(evidence)
        self.blockers = tuple(blockers)


def _trend_continuation(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    del context
    side, higher_evidence = _higher_timeframe_side(views)
    evidence = list(higher_evidence)
    blockers: list[str] = []
    if side is None:
        blockers.append("HTF_DIRECTION_NOT_ALIGNED")
        return _Evaluation(active=False, side=None, score=0, evidence=evidence, blockers=blockers)
    regime_ok = any(
        views[tf].market_text("phase") in {"PRE_TREND", "TREND"}
        for tf in ("M15", "H1")
    )
    if regime_ok:
        evidence.append("M15_H1_TREND_REGIME")
    else:
        blockers.append("M15_H1_TREND_REGIME_MISSING")
    direction_ok, core_evidence = _trend_core(views["M5"], side)
    evidence.extend(core_evidence)
    if not direction_ok:
        blockers.append("M5_DIRECTION_CORE_WEAK")
    momentum_ok, momentum_evidence = _momentum_core(views["M5"], side)
    evidence.extend(momentum_evidence)
    if not momentum_ok:
        blockers.append("M5_MOMENTUM_NOT_CONFIRMED")
    adx = views["M5"].number("adx_14", "adx")
    adx_slope = _series_slope(views["M5"].values("adx_14"), lookback=5)
    adx_ok = adx is not None and adx >= 18.0 and adx_slope is not None and adx_slope > 0.0
    if adx_ok:
        evidence.append("M5_ADX_STRENGTHENING_DIRECTIONLESS")
    else:
        blockers.append("M5_ADX_WEAK_OR_FALLING")
    atr = views["M5"].number("atr_pips")
    atr_slope = _series_slope(views["M5"].values("atr_pips"), lookback=5)
    atr_ok = atr is not None and atr > 0.0 and atr_slope is not None and atr_slope >= 0.0
    if atr_ok:
        evidence.append("M5_ATR_AVAILABLE_AND_NONCONTRACTING_DIRECTIONLESS")
    else:
        blockers.append("M5_ATR_UNAVAILABLE_OR_CONTRACTING")
    strength_volatility_ok = adx_ok and atr_ok
    active = regime_ok and direction_ok and momentum_ok and strength_volatility_ok
    return _Evaluation(
        active=active,
        side=side,
        score=sum((regime_ok, direction_ok, momentum_ok, strength_volatility_ok)),
        evidence=evidence,
        blockers=blockers,
    )


def _trend_pullback(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    del context
    side, higher_evidence = _higher_timeframe_side(views)
    evidence = list(higher_evidence)
    blockers: list[str] = []
    if side is None:
        return _Evaluation(
            active=False,
            side=None,
            score=0,
            evidence=evidence,
            blockers=("HTF_DIRECTION_NOT_ALIGNED",),
        )
    regime_ok = any(
        views[tf].market_text("phase") in {"PRE_TREND", "TREND"}
        for tf in ("M15", "H1")
    )
    direction_ok, core_evidence = _trend_core(views["M5"], side)
    evidence.extend(core_evidence)
    rsi = views["M1"].values("rsi_14")
    macd = views["M1"].values("macd_hist")
    ema_spread = views["M1"].values("ema_12_minus_50_pips")
    rsi_age = _cross_age(rsi, 50.0, side=side, lookback=6)
    macd_age = _cross_age(macd, 0.0, side=side, lookback=6)
    ema_age = _cross_age(ema_spread, 0.0, side=side, lookback=10)
    rsi_reclaim = rsi_age is not None and rsi_age <= 2
    macd_reclaim = macd_age is not None and macd_age <= 2
    ema_fresh = ema_age is not None and ema_age <= 3
    if rsi_reclaim:
        evidence.append(f"M1_RSI50_RECLAIM_AGE_{rsi_age}")
    if macd_reclaim:
        evidence.append(f"M1_MACD_RECLAIM_AGE_{macd_age}")
    if ema_fresh:
        evidence.append(f"M1_FRESH_EMA12_50_CROSS_AGE_{ema_age}")
    zscore = views["M1"].number("z_score_20", "zscore_20", "zscore")
    value_ok = zscore is not None and (
        -1.25 <= zscore <= 0.75
        if side == "LONG"
        else -0.75 <= zscore <= 1.25
    )
    if value_ok:
        evidence.append("M1_NOT_CHASING_EXTREME")
    else:
        blockers.append("M1_VALUE_LOCATION_MISSING_OR_EXTENDED")
    reacceleration = rsi_reclaim or macd_reclaim or ema_fresh
    if not reacceleration:
        blockers.append("M1_REACCELERATION_NOT_CONFIRMED")
    if not regime_ok:
        blockers.append("TREND_REGIME_MISSING")
    if not direction_ok:
        blockers.append("M5_TREND_CORE_WEAK")
    return _Evaluation(
        active=regime_ok and direction_ok and reacceleration and value_ok,
        side=side,
        score=sum((regime_ok, direction_ok, reacceleration, value_ok)),
        evidence=evidence,
        blockers=blockers,
    )


def _range_rotation(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    del context
    evidence: list[str] = []
    blockers: list[str] = []
    regime_ok = any(
        views[tf].market_text("phase") in {"PRE_RANGE", "RANGE"}
        for tf in ("M5", "M15")
    )
    if regime_ok:
        evidence.append("M5_M15_RANGE_REGIME_GROUP")
    else:
        blockers.append("M5_M15_RANGE_PHASE_MISSING")
    adx = views["M5"].number("adx_14", "adx")
    chop = views["M5"].number("choppiness_14", "choppiness")
    hurst = views["M15"].number("hurst_100", "hurst")
    range_strength_ok = any(
        (
            adx is not None and adx < 25.0,
            chop is not None and chop >= 55.0,
            hurst is not None and hurst < 0.5,
        )
    )
    if range_strength_ok:
        evidence.append("RANGE_STAT_FILTER_GROUP_CONFIRMED")
    else:
        blockers.append("RANGE_STAT_FILTERS_MISSING")
    side, location_evidence = _range_side(views)
    evidence.extend(location_evidence)
    location_ok = side is not None
    if not location_ok:
        blockers.append("RANGE_EDGE_OR_RAW_LOCATION_GROUP_UNAVAILABLE")
    rsi = views["M1"].values("rsi_14")
    reentry_age = (
        _oscillator_reentry_age(rsi, side=side) if side is not None else None
    )
    trigger_ok = reentry_age is not None and reentry_age <= 2
    if trigger_ok:
        evidence.append(f"M1_RSI_EXTREME_REENTRY_AGE_{reentry_age}")
    else:
        blockers.append("M1_RSI_REENTRY_MISSING_OR_STALE")
    active = regime_ok and range_strength_ok and location_ok and trigger_ok
    return _Evaluation(
        active=active,
        side=side,
        score=sum((regime_ok, range_strength_ok, location_ok, trigger_ok)),
        evidence=evidence,
        blockers=blockers,
    )


def _pretrend_breakout(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    attempt = str(context.get("attempt_direction") or "")
    branch = str(context.get("branch_outcome") or "")
    side = "LONG" if attempt == "UP" else "SHORT" if attempt == "DOWN" else None
    evidence: list[str] = []
    blockers: list[str] = []
    compression = any(
        views[tf].number("bb_squeeze") == 1.0
        or _at_most(views[tf].number("bb_width_percentile_100"), 0.35)
        or _at_most(views[tf].number("atr_percentile_100"), 0.35)
        or views[tf].market_text("phase") == "PRE_TREND"
        for tf in ("M15", "H1")
    )
    if compression:
        evidence.append("PRE_BREAK_COMPRESSION")
    else:
        blockers.append("PRE_BREAK_COMPRESSION_MISSING")
    accepted = branch == "ACCEPTED"
    if accepted:
        evidence.append("SEALED_BREAK_ACCEPTED")
    else:
        blockers.append("SEALED_BREAK_NOT_ACCEPTED")
    momentum_ok, momentum_evidence = (
        _momentum_core(views["M5"], side) if side else (False, [])
    )
    evidence.extend(momentum_evidence)
    if not momentum_ok:
        blockers.append("M5_BREAK_MOMENTUM_MISSING")
    atr_slope = _series_slope(views["M5"].values("atr_pips"), lookback=5)
    expansion = atr_slope is not None and atr_slope > 0.0
    if expansion:
        evidence.append("M5_ATR_EXPANDING")
    else:
        blockers.append("M5_ATR_NOT_EXPANDING")
    route_ok = side is not None and accepted
    return _Evaluation(
        active=compression and route_ok and momentum_ok and expansion,
        side=side,
        score=sum((compression, route_ok, momentum_ok, expansion)),
        evidence=evidence,
        blockers=blockers,
    )


def _breakout_failure(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    del views
    attempt = str(context.get("attempt_direction") or "")
    branch = str(context.get("branch_outcome") or "")
    attempted_side = "LONG" if attempt == "UP" else "SHORT" if attempt == "DOWN" else None
    side = OPPOSITE_SIDE[attempted_side] if attempted_side else None
    rejected = branch == "REJECTED"
    return _Evaluation(
        active=side is not None and rejected,
        side=side,
        score=int(side is not None) + int(rejected),
        evidence=(
            ("SEALED_RANGE_BREAK_ATTEMPT", "SEALED_BREAK_REJECTED")
            if rejected
            else ("SEALED_RANGE_BREAK_ATTEMPT",)
        ),
        blockers=(() if rejected else ("SEALED_BREAK_NOT_REJECTED",)),
    )


def _exhaustion_reversal(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    del context
    old_side, direction_evidence = _higher_timeframe_side(views)
    side = OPPOSITE_SIDE[old_side] if old_side else None
    evidence = list(direction_evidence)
    blockers: list[str] = []
    if side is None:
        return _Evaluation(active=False, side=None, score=0, evidence=evidence, blockers=("OLD_TREND_DIRECTION_UNAVAILABLE",))
    old_direction = SIDES[old_side]
    allowed_extension = (
        {"OVERBOUGHT", "STRETCHED_UP"}
        if old_side == "LONG"
        else {"OVERSOLD", "STRETCHED_DOWN"}
    )
    extension = any(
        views[tf].market_text("extension") in allowed_extension
        for tf in ("M5", "M15", "H1")
    )
    rsi = views["M5"].number("rsi_14")
    rsi_extreme = rsi is not None and (rsi >= 70.0 if old_side == "LONG" else rsi <= 30.0)
    adx = views["M5"].number("adx_14", "adx")
    adx_slope = _series_slope(views["M5"].values("adx_14"), lookback=5)
    adx_fading = adx is not None and adx >= 25.0 and adx_slope is not None and adx_slope < 0.0
    macd_slope = _series_slope(views["M5"].values("macd_hist"), lookback=5)
    macd_fading = macd_slope is not None and (macd_slope < 0.0 if old_side == "LONG" else macd_slope > 0.0)
    atr = views["M5"].number("atr_pips")
    atr_slope = _series_slope(views["M5"].values("atr_pips"), lookback=5)
    atr_decelerating = (
        atr is not None and atr > 0.0 and atr_slope is not None and atr_slope < 0.0
    )
    extension_role = extension and rsi_extreme
    deceleration_role = atr_decelerating and (adx_fading or macd_fading)
    m1_reversal = (
        views["M1"].market_text("direction") == SIDES[side]
        and views["M1"].market_text("readiness") == "TRIGGERED"
    )
    for active, label in (
        (extension, f"MULTITF_{old_direction}_EXTENSION"),
        (rsi_extreme, f"M5_RSI_{old_direction}_EXTREME"),
        (adx_fading, "M5_ADX_HIGH_BUT_FADING"),
        (macd_fading, "M5_MACD_MOMENTUM_FADING"),
        (atr_decelerating, "M5_ATR_DECELERATING_DIRECTIONLESS"),
        (m1_reversal, "M1_CLOSE_CONFIRMED_REVERSAL"),
    ):
        if active:
            evidence.append(label)
    direction_role = old_side is not None
    score = sum((direction_role, extension_role, deceleration_role, m1_reversal))
    if not extension_role:
        blockers.append(f"DIRECTIONAL_EXTENSION_NOT_CONFIRMED_{old_direction}")
    if not deceleration_role:
        blockers.append("ADX_OR_MACD_WITH_REQUIRED_ATR_DECELERATION_MISSING")
    if not m1_reversal:
        blockers.append("M1_CLOSE_CONFIRMED_REVERSAL_MISSING")
    active = direction_role and extension_role and deceleration_role and m1_reversal
    if not active:
        blockers.append(f"EXHAUSTION_CONFIRMATION_INSUFFICIENT_{old_direction}")
    return _Evaluation(
        active=active,
        side=side,
        score=score,
        evidence=evidence,
        blockers=blockers,
    )


def _session_open_expansion(
    views: Mapping[str, _View], context: Mapping[str, Any]
) -> _Evaluation:
    cycle = context.get("cycle")
    attempt = str(context.get("attempt_direction") or "")
    branch = str(context.get("branch_outcome") or "")
    side = "LONG" if attempt == "UP" else "SHORT" if attempt == "DOWN" else None
    sessions = _opening_sessions(cycle) if isinstance(cycle, datetime) else ()
    evidence = [f"{session}_OPEN_WINDOW" for session in sessions]
    blockers: list[str] = []
    if not sessions:
        blockers.append("OUTSIDE_LONDON_NEW_YORK_OPEN")
    accepted = branch == "ACCEPTED"
    if accepted:
        evidence.append("SEALED_BREAK_ACCEPTED")
    else:
        blockers.append("SEALED_BREAK_NOT_ACCEPTED")
    compression = (
        views["M15"].number("bb_squeeze") == 1.0
        or _at_most(views["M15"].number("bb_width_percentile_100"), 0.35)
        or _at_most(views["M15"].number("atr_percentile_100"), 0.35)
    )
    if compression:
        evidence.append("M15_SESSION_COMPRESSION_STATE")
    else:
        blockers.append("M15_COMPRESSION_MISSING")
    momentum_ok, momentum_evidence = (
        _momentum_core(views["M5"], side) if side else (False, [])
    )
    evidence.extend(momentum_evidence)
    if not momentum_ok:
        blockers.append("M5_SESSION_MOMENTUM_MISSING")
    atr_slope = _series_slope(views["M5"].values("atr_pips"), lookback=5)
    expansion = atr_slope is not None and atr_slope > 0.0
    if expansion:
        evidence.append("M5_ATR_EXPANDING")
    else:
        blockers.append("M5_ATR_NOT_EXPANDING")
    direction_role = side is not None and accepted
    expansion_confirmation = momentum_ok and expansion
    return _Evaluation(
        active=bool(sessions) and compression and direction_role and expansion_confirmation,
        side=side,
        score=sum((bool(sessions), compression, direction_role, expansion_confirmation)),
        evidence=evidence,
        blockers=blockers,
    )


def _no_trade_control(
    views: Mapping[str, _View],
    context: Mapping[str, Any],
    *,
    active_directional: int,
    active_sides: set[str],
) -> _Evaluation:
    del context
    incomplete = any(view.market.get("evidence_complete") is not True for view in views.values())
    uncertain = sum(
        views[tf].market_text("phase") in {"TRANSITION", "UNKNOWN"}
        for tf in ("M1", "M5", "M15", "H1")
    ) >= 2
    conflict = len(
        {
            views[tf].market_text("direction")
            for tf in ("M5", "M15", "H1", "H4")
            if views[tf].market_text("direction") in {"UP", "DOWN"}
        }
    ) > 1
    hypothesis_conflict = active_sides == set(SIDES)
    no_survivor = active_directional == 0
    evidence = [
        label
        for active, label in (
            (incomplete, "INCOMPLETE_EVIDENCE"),
            (uncertain, "TRANSITION_OR_UNKNOWN"),
            (conflict, "MULTITF_DIRECTION_CONFLICT"),
            (hypothesis_conflict, "ACTIVE_DIRECTIONAL_HYPOTHESIS_CONFLICT"),
            (no_survivor, "NO_DIRECTIONAL_HYPOTHESIS_SURVIVED"),
        )
        if active
    ]
    active = incomplete or uncertain or conflict or hypothesis_conflict or no_survivor
    return _Evaluation(
        active=active,
        side=None,
        score=sum((incomplete, uncertain, conflict, hypothesis_conflict, no_survivor)),
        evidence=evidence,
        blockers=(() if active else ("DIRECTIONAL_HYPOTHESES_AVAILABLE",)),
    )


def _higher_timeframe_side(views: Mapping[str, _View]) -> tuple[str | None, tuple[str, ...]]:
    directions = [views[tf].market_text("direction") for tf in ("M15", "H1", "H4")]
    up = directions.count("UP")
    down = directions.count("DOWN")
    if max(up, down) < 2 or up == down:
        return None, tuple()
    side = "LONG" if up > down else "SHORT"
    return side, (f"M15_H1_H4_{SIDES[side]}_MAJORITY",)


def _trend_core(view: _View, side: str) -> tuple[bool, list[str]]:
    long = side == "LONG"
    evidence: list[str] = []
    ema_slope_aligned = False
    slope = view.number("ema_slope_20", "ema_slope_5")
    if slope is not None and (slope > 0.0 if long else slope < 0.0):
        ema_slope_aligned = True
        evidence.append("M5_EMA_SLOPE_ALIGNED")
    di_aligned = False
    plus_di = view.number("plus_di_14")
    minus_di = view.number("minus_di_14")
    if plus_di is not None and minus_di is not None and (plus_di > minus_di if long else minus_di > plus_di):
        di_aligned = True
        evidence.append("M5_DI_DIRECTION_ALIGNED")
    supertrend_aligned = False
    supertrend = view.number("supertrend_dir")
    if supertrend is not None and (supertrend > 0.0 if long else supertrend < 0.0):
        supertrend_aligned = True
        evidence.append("M5_SUPERTREND_ALIGNED")
    price_ema_order_aligned = False
    close = view.number("close")
    ema20 = view.number("ema_20")
    ema50 = view.number("ema_50")
    if close is not None and ema20 is not None and ema50 is not None and (close > ema20 > ema50 if long else close < ema20 < ema50):
        price_ema_order_aligned = True
        evidence.append("M5_PRICE_EMA_ORDER_ALIGNED")
    # The aligned indicators are one direction role, never four votes.  Prefer
    # the price/EMA ordering; otherwise require EMA slope plus one independent
    # representation of the same direction before the role is satisfied.
    direction_role = price_ema_order_aligned or (
        ema_slope_aligned and (di_aligned or supertrend_aligned)
    )
    return direction_role, evidence


def _momentum_core(view: _View, side: str | None) -> tuple[bool, list[str]]:
    if side not in SIDES:
        return False, []
    long = side == "LONG"
    evidence: list[str] = []
    aligned = False
    rsi = view.number("rsi_14")
    if rsi is not None and (rsi >= 52.0 if long else rsi <= 48.0):
        aligned = True
        evidence.append("M5_RSI_MOMENTUM_ALIGNED")
    macd = view.number("macd_hist")
    if macd is not None and (macd > 0.0 if long else macd < 0.0):
        aligned = True
        evidence.append("M5_MACD_ALIGNED")
    roc = view.number("roc_5", "roc_10")
    if roc is not None and (roc > 0.0 if long else roc < 0.0):
        aligned = True
        evidence.append("M5_ROC_ALIGNED")
    return aligned, evidence


def _range_side(views: Mapping[str, _View]) -> tuple[str | None, list[str]]:
    evidence: list[str] = []
    locations = {views[tf].market_text("location") for tf in ("M1", "M5")}
    values = {views[tf].market_text("value_zone") for tf in ("M1", "M5")}
    semantic_sides: set[str] = set()
    if "LOWER_THIRD" in locations or values & {"DISCOUNT", "DEEP_DISCOUNT"}:
        semantic_sides.add("LONG")
    if "UPPER_THIRD" in locations or values & {"PREMIUM", "DEEP_PREMIUM"}:
        semantic_sides.add("SHORT")
    if len(semantic_sides) != 1:
        return None, evidence
    side = next(iter(semantic_sides))

    zscore = views["M5"].number("z_score_20", "zscore_20", "zscore")
    bb_position = views["M5"].number("bb_position", "bollinger_position")
    donchian_position = views["M5"].number("donchian_position")
    raw_sides: set[str] = set()
    if any(
        (
            zscore is not None and zscore <= -1.0,
            bb_position is not None and bb_position <= 0.2,
            donchian_position is not None and donchian_position <= 0.2,
        )
    ):
        raw_sides.add("LONG")
    if any(
        (
            zscore is not None and zscore >= 1.0,
            bb_position is not None and bb_position >= 0.8,
            donchian_position is not None and donchian_position >= 0.8,
        )
    ):
        raw_sides.add("SHORT")
    if raw_sides != {side}:
        return None, evidence
    evidence.append(f"M1_M5_RANGE_EDGE_{SIDES[side]}")
    evidence.append(f"M5_ZSCORE_BB_DONCHIAN_EDGE_GROUP_{SIDES[side]}")
    return side, evidence


def _series_slope(values: Sequence[float], *, lookback: int) -> float | None:
    if len(values) < 2:
        return None
    window = tuple(values[-max(2, lookback) :])
    return float(window[-1] - window[0]) / float(len(window) - 1)


def _cross_age(
    values: Sequence[float], level: float, *, side: str, lookback: int
) -> int | None:
    if len(values) < 2:
        return None
    window = tuple(values[-max(2, lookback) :])
    if side == "LONG" and window[-1] <= level:
        return None
    if side == "SHORT" and window[-1] >= level:
        return None
    if side not in SIDES:
        return None
    for current_index in range(len(window) - 1, 0, -1):
        previous = window[current_index - 1]
        current = window[current_index]
        crossed = (
            previous <= level < current
            if side == "LONG"
            else previous >= level > current
        )
        if crossed:
            return len(window) - 1 - current_index
    return None


def _oscillator_reentry_age(values: Sequence[float], *, side: str) -> int | None:
    return _cross_age(
        values,
        30.0 if side == "LONG" else 70.0,
        side=side,
        lookback=8,
    )


def _opening_sessions(value: datetime) -> tuple[str, ...]:
    utc = value.astimezone(timezone.utc)
    sessions: list[str] = []
    for name, zone in (("LONDON", "Europe/London"), ("NEW_YORK", "America/New_York")):
        local = utc.astimezone(ZoneInfo(zone))
        minute = local.hour * 60 + local.minute
        if 8 * 60 <= minute < 9 * 60 + 30:
            sessions.append(name)
    return tuple(sessions)


def _hypothesis_row(
    spec: Mapping[str, Any],
    result: _Evaluation,
    *,
    feature_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    side = result.side if result.side in SIDES else None
    body = {
        "contract": HYPOTHESIS_CONTRACT,
        "schema_version": 1,
        **dict(spec),
        "status": "ACTIVE_SHADOW" if result.active else "INACTIVE_SHADOW",
        "predicted_side": side,
        "predicted_direction": SIDES.get(side, "EITHER"),
        "raw_confluence_score": int(result.score),
        "raw_confluence_score_is_probability": False,
        "evidence": sorted(set(result.evidence)),
        "blockers": sorted(set(result.blockers)),
        "feature_snapshot_sha256": str(feature_snapshot["contract_sha256"]),
        "forecast_targets": list(FORECAST_TARGETS),
        "forecast_values_status": "PENDING_PROSPECTIVE_FORWARD_CALIBRATION",
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "hypothesis_sha256": _canonical_sha(body)}


def _views(feature_snapshot: Mapping[str, Any]) -> dict[str, _View]:
    return {
        str(row["timeframe"]): _View(row)
        for row in feature_snapshot["timeframes"]
    }


def _feature_snapshot_valid(value: Mapping[str, Any]) -> bool:
    if (
        set(value) != FEATURE_SNAPSHOT_KEYS
        or not _sealed_valid(value, FEATURE_SNAPSHOT_CONTRACT)
    ):
        return False
    rows = value.get("timeframes")
    if not bool(
        value.get("schema_version").__class__ is int
        and value.get("schema_version") == 1
        and isinstance(value.get("pair"), str)
        and bool(value.get("pair"))
        and value.get("pair") == str(value.get("pair")).strip().upper()
        and value.get("feature_allowlist_version") == 1
        and value.get("hypothesis_families") == list(HYPOTHESIS_FAMILIES)
        and value.get("raw_chart_packet_embedded") is False
        and value.get("diagnostic_only") is True
        and value.get("order_authority") == "NONE"
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and isinstance(rows, list)
        and len(rows) == len(TIMEFRAMES)
    ):
        return False
    try:
        cycle = _parse_utc(value["handoff_cycle_generated_at_utc"])
        if len(_canonical_json_bytes(value)) > MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES:
            return False
        for expected_timeframe, row in zip(TIMEFRAMES, rows, strict=True):
            if not isinstance(row, Mapping) or set(row) != FEATURE_ROW_KEYS:
                return False
            timeframe = str(row.get("timeframe") or "")
            close = _parse_utc(row["complete_candle_close_utc"])
            body = {
                key: item for key, item in row.items() if key != "feature_sha256"
            }
            if not bool(
                timeframe == expected_timeframe
                and close <= cycle
                and row.get("feature_sha256") == _canonical_sha(body)
            ):
                return False
            market = row.get("market_state")
            if not isinstance(market, Mapping) or not set(market).issubset(
                MARKET_STATE_FEATURES
            ):
                return False
            for key, item in market.items():
                if key in MARKET_STATE_ENUM_FEATURES:
                    if (
                        not isinstance(item, str)
                        or item not in MARKET_STATE_ENUM_FEATURES[key]
                    ):
                        return False
                elif key == "confidence":
                    parsed = _finite_feature_number(item)
                    if parsed is None or not 0.0 <= float(parsed) <= 1.0:
                        return False
                elif key == "evidence_complete":
                    if not isinstance(item, bool):
                        return False
                else:
                    return False

            indicators = row.get("indicators")
            if not isinstance(indicators, Mapping) or not set(indicators).issubset(
                NUMERIC_INDICATOR_FEATURES | ENUM_INDICATOR_FEATURES
            ):
                return False
            for key, item in indicators.items():
                if key in NUMERIC_INDICATOR_FEATURES:
                    if _finite_feature_number(item) is None:
                        return False
                elif _bounded_enum(item) is None:
                    return False

            series = row.get("indicator_series")
            if not isinstance(series, Mapping) or not set(series).issubset(
                INDICATOR_SERIES_FEATURES
            ):
                return False
            for items in series.values():
                if (
                    not isinstance(items, list)
                    or len(items) > MAX_INDICATOR_SERIES_VALUES
                    or any(_finite_feature_number(item) is None for item in items)
                ):
                    return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _episode_context_valid(
    *, attempt_direction: str, branch_outcome: str, route_family: str
) -> bool:
    return bool(
        attempt_direction in {"UP", "DOWN"}
        and branch_outcome in ROUTE_FAMILY_BY_BRANCH
        and route_family in KNOWN_ROUTE_FAMILIES
        and route_family == ROUTE_FAMILY_BY_BRANCH[branch_outcome]
    )


def _snapshot_m5_atr(value: Mapping[str, Any]) -> float | None:
    try:
        row = next(
            item for item in value["timeframes"] if item.get("timeframe") == "M5"
        )
    except (KeyError, StopIteration, TypeError):
        return None
    indicators = row.get("indicators")
    if not isinstance(indicators, Mapping):
        return None
    return _number(indicators.get("atr_pips"))


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if value.get("contract") != contract:
        return False
    sha = value.get("contract_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        return False
    try:
        expected = _canonical_sha(
            {key: item for key, item in value.items() if key != "contract_sha256"}
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return sha == expected


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("technical hypothesis clock must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and abs(parsed) <= 1_000_000_000_000.0 else None


def _finite_feature_number(value: Any) -> int | float | None:
    parsed = _number(value)
    if parsed is None:
        return None
    return int(value) if isinstance(value, int) else parsed


def _bounded_enum(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    normalized = value.upper()
    if value != normalized or len(value) > 64:
        return None
    return value


def _at_most(value: float | None, ceiling: float) -> bool:
    return value is not None and value <= ceiling


def _cost_state(
    *,
    spread_pips: Any,
    m5_atr_pips: Any,
    spread_to_m5_atr: Any,
    feature_m5_atr_pips: Any,
) -> dict[str, Any]:
    spread = _number(spread_pips)
    atr = _number(m5_atr_pips)
    ratio = _number(spread_to_m5_atr)
    feature_atr = _number(feature_m5_atr_pips)
    valid = bool(
        spread is not None
        and spread > 0.0
        and atr is not None
        and atr > 0.0
        and ratio is not None
        and ratio > 0.0
        and feature_atr is not None
        and math.isclose(atr, feature_atr, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(ratio, spread / atr, rel_tol=0.0, abs_tol=1e-6)
    )
    return {
        "valid": valid,
        "spread_pips": spread,
        "m5_atr_pips": atr,
        "feature_m5_atr_pips": feature_atr,
        "spread_to_m5_atr": ratio,
        "movement_capacity_after_one_spread_pips": (
            round(atr - spread, 6) if valid and atr is not None and spread is not None else None
        ),
        "cost_role": "STATE_VARIABLE_AND_REQUIRED_EDGE_THRESHOLD_NOT_DIRECTION",
        "cost_is_blanket_exclusion": False,
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    return {**body, "contract_sha256": _canonical_sha(body)}


__all__ = [
    "CATALOG_CONTRACT",
    "EVALUATOR_POLICY_V1",
    "FORECAST_TARGETS",
    "HYPOTHESIS_CONTRACT",
    "SHADOW_CONTRACT",
    "build_fast_bot_technical_hypotheses",
    "technical_hypothesis_shadow_valid",
    "technical_hypothesis_catalog",
]
