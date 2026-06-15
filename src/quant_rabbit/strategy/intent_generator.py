from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_STRATEGY_PROFILE,
    ROOT,
)
from quant_rabbit.analysis.market_context_matrix import matrix_summary_for_intent
from quant_rabbit.risk import (
    DEFAULT_SPECS,
    MIN_PRODUCTION_LOT_UNITS,
    RiskEngine,
    RiskPolicy,
    _min_lot_test_override_active,
    broker_margin_free_units,
    estimate_incremental_margin_jpy,
    estimate_required_margin_jpy,
    hedge_margin_free_units,
    margin_budget_jpy,
    resolve_max_loss_jpy,
)
from quant_rabbit.snapshot_json import snapshot_payload_order_raw
from quant_rabbit.strategy.lane_history_ledger import (
    LOSS_STREAK_BLOCK_THRESHOLD,
    LOSS_STREAK_SIZE_BACKOFF,
    SameDayLossStreak,
    compute_same_day_loss_streaks,
)
from quant_rabbit.strategy.price_action import structural_tp_target
from quant_rabbit.strategy.profile import StrategyProfile, issues_to_dicts


# Geometry tuning constants. Per AGENT_CONTRACT §3.5, every constant on the
# trader risk path needs a market-reality reason. These are *minimums* / floors,
# not the truth — the actual stop distance is the larger of ATR-derived and
# spread-derived candidates.
#
# - GEOMETRY_ATR_MULT: default 1.0 ATR is a "typical move" of the timeframe.
#   Live evidence (2026-05-06: 8 trader SL hits, ~-1075 JPY, all at 5–13 pips on
#   EUR_USD/EUR_JPY/GBP_USD/AUD_JPY) showed 1.0 ATR sits inside routine M5 wick
#   noise. The user's standing directive is "SLいらない" / SL-far
#   (`feedback_no_tight_sl_thin_market.md`, `project_sl_free_strategy.md`).
#   Override the default by setting `QR_GEOMETRY_ATR_MULT` (e.g. 5.0) in the
#   live env so production widens SL out of the noise band without rewriting
#   every regression-test fixture that was tuned to the 1.0-baseline geometry.
# - GEOMETRY_SPREAD_FLOOR_MULT: default 6.0 × spread protects against broker
#   fill jitter and wick noise around the entry. Must remain >=
#   RiskPolicy.min_stop_spread_multiple (5.0). Override via
#   `QR_GEOMETRY_SPREAD_FLOOR_MULT` (e.g. 12.0) for live SL-far operation.
# - GEOMETRY_ATR_TIMEFRAME: M5 is the operating timeframe of the scalp / swing
#   trader. M1 is too noisy for stop geometry, while M15/M30/H1/H4/D ATR
#   reflects slower structure than the trader is reacting to.
def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


GEOMETRY_ATR_MULT = _env_float("QR_GEOMETRY_ATR_MULT", 1.0, minimum=0.5)
GEOMETRY_SPREAD_FLOOR_MULT = _env_float("QR_GEOMETRY_SPREAD_FLOOR_MULT", 6.0, minimum=5.1)
GEOMETRY_ATR_TIMEFRAME = "M5"

# F — Noise-resistant initial SL geometry (2026-05-13).
#
# 2026-05-12T15:33 UTC mass-close incident drove the operator demand:
# new-entry SL must not be hit by routine M5 wick noise. The M5-ATR-based
# stop (`atr_pips × GEOMETRY_ATR_MULT`) sits in the wick band of higher
# timeframes; a tight broker SL there is the very failure mode the
# operator is preventing with the new broker-side SL push.
#
# When `QR_NEW_ENTRY_INITIAL_SL=1` is exported, intent geometry
# additionally floors the stop distance at:
#   `H4 atr × QR_NEW_ENTRY_SL_H4_ATR_MULT (default 1.5)`
# AND applies a session-aware widening multiplier when liquidity is
# thin. The override env keeps the multiplier tunable in production
# without code edits.
#
# Per AGENT_CONTRACT §3.5:
# (a) market reality: H4 ATR represents one typical multi-hour swing.
#     1.5× sits outside the noise band of the lower TFs the trader
#     actually reads; the SL only triggers when price has decisively
#     broken away from the structural setup.
# (b) constants rather than market-derived: the multiplier is an
#     operator-policy widening dial. The base (H4 ATR) is fully
#     market-derived; the multiplier is the operator's risk
#     preference. JPY/pip literals are zero.
# (c) replace via: tune `QR_NEW_ENTRY_SL_H4_ATR_MULT` per post-trade
#     learning if SL hits cluster at H4-ATR-band edges.
NEW_ENTRY_SL_H4_ATR_MULT = _env_float(
    "QR_NEW_ENTRY_SL_H4_ATR_MULT", 1.5, minimum=0.5
)
# Session-aware widening: thin-liquidity sessions (Tokyo open,
# OFF_HOURS overnight, JP holidays) print wider wicks per pip moved,
# so the noise floor must widen to compensate. Multipliers ride on top
# of the H4-ATR floor.
NEW_ENTRY_SL_THIN_SESSION_MULT = _env_float(
    "QR_NEW_ENTRY_SL_THIN_SESSION_MULT", 1.3, minimum=1.0
)
NEW_ENTRY_SL_OFF_HOURS_MULT = _env_float(
    "QR_NEW_ENTRY_SL_OFF_HOURS_MULT", 1.4, minimum=1.0
)


def _new_entry_initial_sl_active() -> bool:
    return os.environ.get("QR_NEW_ENTRY_INITIAL_SL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


# Disaster stop (2026-06-11, operator-approved reversal of strict SL-free:
# 「SLの件もやっていい」). A broker-side CATASTROPHE bound on every new entry,
# deliberately decoupled from intent.sl:
#
# - intent.sl stays the EXPECTED invalidation (sizing, reward/risk, and risk
#   validation all keep using it) — sizing against the disaster distance
#   would recreate the 2026-06-11 micro-lot spiral.
# - the disaster stop is only attached to the broker order
#   (`stopLossOnFill`) so a flash move / JPY intervention / weekend gap
#   during the 20-minute blind window between cycles cannot destroy the
#   account, and the give-up-close tail (avg -1,437 JPY, margin closeouts
#   -5,641 JPY on 2026-05-14) gets a hard ceiling.
# - it never trails (QR_DISABLE_TRAILING_SL=1 stays on) and existing
#   positions are never retro-fitted (existing-position invariant).
#
# Per AGENT_CONTRACT §3.5:
# (a) market reality: H4 ATR is one typical multi-hour swing; 2.5× that,
#     further widened in thin sessions, sits far outside every wick band
#     that harvested the 2026-05-13 noise stops (those were 4-42 pips;
#     this is 60-120+ pips on majors). A hit means the thesis is not
#     merely wrong but catastrophically wrong.
# (b) constants: the multiplier is operator risk policy (catastrophe
#     tolerance), the base (H4 ATR × session liquidity) is market-derived.
# (c) replace via: tune QR_DISASTER_SL_H4_ATR_MULT if live disaster-stop
#     hits ever cluster at the band edge instead of true dislocations.
DISASTER_SL_H4_ATR_MULT = _env_float("QR_DISASTER_SL_H4_ATR_MULT", 2.5, minimum=1.0)
# Strict-ordering buffer: the broker disaster stop must always sit beyond
# the expected (synthetic) stop, or the catastrophe bound would fire before
# the discretionary structure exit it is meant to backstop. 1.25 is a
# documented ordering margin, not a market estimate.
DISASTER_SL_MIN_EXPECTED_MULT = 1.25


def _disaster_sl_active() -> bool:
    return os.environ.get("QR_DISASTER_SL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _disaster_sl_metadata(
    pair: str,
    side: Side,
    *,
    entry: float,
    expected_sl: float,
    chart_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the broker-side catastrophe stop for a NEW entry.

    Returns metadata keys consumed by `_oanda_order_request`:
    `disaster_sl` (price), `disaster_sl_pips`, `disaster_sl_basis` — or
    `disaster_sl_missing` when H4 ATR evidence is unavailable (the entry
    still proceeds exactly as the pre-disaster-stop SL-free runtime did;
    the missing bound is surfaced, never silently substituted, per §3.5).
    """
    if not _disaster_sl_active():
        return {}
    h4_atr_pips = None
    session_tag = None
    if isinstance(chart_context, dict):
        h4_atr_pips = _optional_float(chart_context.get("h4_atr_pips"))
        session_tag = _text_or_none(
            chart_context.get("session_current_tag") or chart_context.get("session_bucket")
        )
    if h4_atr_pips is None or h4_atr_pips <= 0:
        return {"disaster_sl_missing": "H4_ATR_MISSING"}
    pip_factor = PIP_FACTORS[pair]
    session_mult = _session_widening_mult(session_tag)
    distance_pips = h4_atr_pips * DISASTER_SL_H4_ATR_MULT * session_mult
    expected_pips = abs(entry - expected_sl) * pip_factor
    if expected_pips > 0:
        distance_pips = max(distance_pips, expected_pips * DISASTER_SL_MIN_EXPECTED_MULT)
    pip = 1.0 / pip_factor
    if side == Side.LONG:
        price = entry - distance_pips * pip
    else:
        price = entry + distance_pips * pip
    return {
        "disaster_sl": _round_price(pair, price),
        "disaster_sl_pips": round(distance_pips, 1),
        "disaster_sl_basis": (
            f"H4_ATR {h4_atr_pips:.1f}p x {DISASTER_SL_H4_ATR_MULT:g} "
            f"x session {session_mult:g}"
        ),
    }


# I — Session-aware spread tolerance multipliers (2026-05-13).
#
# AGENT_CONTRACT §3.5 mandates spread tolerance be liquidity-derived,
# not a static 2.5× multiplier. Trading the same `max_spread_multiple`
# during Tokyo open and London-NY overlap means we either reject every
# Tokyo entry (cap too tight) or accept overspread London garbage (cap
# too loose). The session multiplier is applied on top of
# `RiskPolicy.max_spread_multiple` so the policy default still anchors
# the tolerance.
#
# Per AGENT_CONTRACT §3.5:
# (a) market reality: London_NY overlap is the deepest book of the day;
#     Tokyo and OFF_HOURS hold the highest spread variance. Tighten in
#     liquid hours, loosen in thin hours.
# (b) policy constants over market-derived: chart_reader publishes the
#     session bucket; these multipliers are the operator's response.
# (c) replace via: post-trade-learning evidence on spread-driven
#     rejections per session, or tighter broker-truth flow snapshots.
SESSION_SPREAD_MULT_LONDON_NY_OVERLAP = _env_float(
    "QR_SESSION_SPREAD_MULT_LONDON_NY", 0.8, minimum=0.5
)
SESSION_SPREAD_MULT_LONDON = _env_float(
    "QR_SESSION_SPREAD_MULT_LONDON", 1.0, minimum=0.5
)
SESSION_SPREAD_MULT_NY_AM = _env_float(
    "QR_SESSION_SPREAD_MULT_NY_AM", 1.0, minimum=0.5
)
SESSION_SPREAD_MULT_TOKYO = _env_float(
    "QR_SESSION_SPREAD_MULT_TOKYO", 1.25, minimum=0.5
)
SESSION_SPREAD_MULT_OFF_HOURS = _env_float(
    "QR_SESSION_SPREAD_MULT_OFF_HOURS", 1.5, minimum=0.5
)

# Regime-derived reward/risk multipliers per AGENT_CONTRACT §3.5: "Trend regimes
# deserve wider targets; range regimes deserve faster rotation." Applied after
# the lane's base target_reward_risk to convert regime context into geometry
# without forcing the lane definition to encode regime in advance.
#
# (a) market reality: range markets revert to mid before reaching distant targets,
#     so close TP captures more of the available rotation. Trend markets keep
#     extending, so wider TP rides the move. Impulse continuation goes furthest.
# (b) constants rather than derived: these are the operator's regime-handling
#     policy. The regime detector itself is data-derived; the *response* to a
#     regime is a deliberate policy choice and is intentionally constant.
# (c) replace via: switch a regime to a different multiplier here when post-trade
#     learning shows the current value mis-fits realized R distribution per
#     regime. Per-pair overrides should land in pair_charts metadata, not here.
REGIME_REWARD_RISK_RANGE_MULT = 0.6
REGIME_REWARD_RISK_TREND_MULT = 1.3
REGIME_REWARD_RISK_IMPULSE_MULT = 1.5
REGIME_REWARD_RISK_UNCLEAR_MULT = 1.0

# Dynamic reward_risk adders (AGENT_CONTRACT §3.5: target_reward_risk
# must be regime-derived, not a fixed literal). Each adder reads a
# current market input and pushes reward_risk up (room to run) or down
# (no room). Values are bounded operator knobs — tune via post-trade-
# learning, but never set to 0 because that would mute the dynamic
# response and revert to the static-1.5 anti-pattern the contract
# explicitly forbids.
DYNAMIC_RR_BASE = float(os.environ.get("QR_DYNAMIC_RR_BASE", "2.0"))
DYNAMIC_RR_ATR_PCTILE_HIGH = 0.7   # ≥ this → +ATR_HIGH bonus
DYNAMIC_RR_ATR_PCTILE_LOW = 0.3    # ≤ this → -ATR_LOW penalty
DYNAMIC_RR_ATR_HIGH_BONUS = float(os.environ.get("QR_DYNAMIC_RR_ATR_HIGH_BONUS", "1.5"))
DYNAMIC_RR_ATR_LOW_PENALTY = float(os.environ.get("QR_DYNAMIC_RR_ATR_LOW_PENALTY", "0.5"))
DYNAMIC_RR_ADX_TREND_THRESHOLD = 25.0
DYNAMIC_RR_ADX_RANGE_THRESHOLD = 18.0
DYNAMIC_RR_ADX_TREND_BONUS = float(os.environ.get("QR_DYNAMIC_RR_ADX_TREND_BONUS", "1.5"))
DYNAMIC_RR_ADX_RANGE_PENALTY = float(os.environ.get("QR_DYNAMIC_RR_ADX_RANGE_PENALTY", "0.5"))
DYNAMIC_RR_SIGMA_EXHAUSTED_THRESHOLD = 2.5
DYNAMIC_RR_SIGMA_EXHAUSTED_PENALTY = float(os.environ.get("QR_DYNAMIC_RR_SIGMA_EXHAUSTED_PENALTY", "0.8"))
DYNAMIC_RR_SESSION_OVERLAP_BONUS = float(os.environ.get("QR_DYNAMIC_RR_SESSION_OVERLAP_BONUS", "0.5"))
DYNAMIC_RR_SESSION_THIN_PENALTY = float(os.environ.get("QR_DYNAMIC_RR_SESSION_THIN_PENALTY", "0.5"))
DYNAMIC_RR_FLOOR = float(os.environ.get("QR_DYNAMIC_RR_FLOOR", "1.5"))
DYNAMIC_RR_CEILING = float(os.environ.get("QR_DYNAMIC_RR_CEILING", "5.0"))

# High-impact macro follow-through sizing. These constants do not decide
# direction; they only let a factual post-release event surprise use more of
# the already equity-derived daily risk budget once the forecast layer has
# produced same-direction event evidence.
#
# (a) Market reality: NFP/CPI/rate-decision beats can move 50-100 pips while a
#     normal technical scalp cannot. Treating a confirmed surprise like a
#     routine M5 setup leaves the account under-exposed to the few sessions
#     that can realistically cover the daily target.
# (b) Constants rather than derived: these are operator policy caps for the
#     event campaign. The base loss cap is still equity-derived by
#     daily_target_state; the daily-budget share prevents one event lane from
#     consuming the whole day.
# (c) Replace with event-family calibration once the projection ledger has
#     enough `event_surprise_followthrough` samples by event name/currency.
MACRO_EVENT_SIZE_UP_SIGNAL_NAMES = frozenset({"event_surprise_followthrough"})
MACRO_EVENT_SIZE_UP_MIN_SIGNAL_CONFIDENCE = _env_float(
    "QR_MACRO_EVENT_SIZE_UP_MIN_SIGNAL_CONF", 0.80, minimum=0.0
)
MACRO_EVENT_RISK_MULTIPLIER = _env_float(
    "QR_MACRO_EVENT_RISK_MULT", 3.0, minimum=1.0
)
MACRO_EVENT_MAX_DAILY_RISK_SHARE = _env_float(
    "QR_MACRO_EVENT_MAX_DAILY_RISK_SHARE", 0.50, minimum=0.0
)

# Recovery hedges are exposure management, not permission to auto-flatten the
# whole trapped leg. Use current directional conviction to choose a tranche size;
# full same-pair margin-free capacity remains a cap, not the default target.
RECOVERY_HEDGE_MIN_CONVICTION_SCALE = 0.25
RECOVERY_HEDGE_DEFAULT_CONVICTION_SCALE = 0.50
# Continuation recovery hedges are the lowest-quality hedge class: the original
# side is underwater and the hedge is chasing the same adverse move without a
# machine-readable reversal confirmation. Cap below the neutral 50% tranche so
# it can monetize urgent momentum without freezing the whole trapped leg at a
# likely exhaustion point. Replace this execution-policy cap with post-trade
# expectancy by hedge_timing_class once enough receipts exist.
RECOVERY_HEDGE_CONTINUATION_MAX_SCALE = 0.35

# Projection verification is a cycle preflight, while market-context refresh,
# broker snapshot, and intent generation can take several minutes. A projection
# that was still inside its verification window at preflight should not become
# a global live-entry blocker seconds later in the same cycle. The grace is an
# execution tolerance, not a forecast-validity extension: projections older
# than the grace remain BLOCKing until verify-projections resolves them.
# Grace = the worst-case same-cycle latency between the cycle's projection
# preflight (which resolves every already-expired PENDING row) and gateway
# staging. The consolidated trader cycle includes model reasoning between
# refresh and gateway, measured at ~10 minutes live (2026-06-11: preflight
# 11:17:12Z resolved all expired rows, staging 11:27:16Z re-blocked 3 of 4
# basket lanes on rows that crossed expiry in between). One full 20-minute
# scheduler cadence is the natural bound: by then the next preflight has run,
# so anything older is a genuine verification-pipeline defect, not boundary
# latency.
PROJECTION_PENDING_EXPIRY_GRACE_SECONDS = _env_float(
    "QR_PROJECTION_PENDING_EXPIRY_GRACE_SECONDS",
    1200.0,
    minimum=0.0,
)
FORECAST_CONFIDENCE_TELEMETRY_TOLERANCE = _env_float(
    "QR_FORECAST_CONFIDENCE_TELEMETRY_TOLERANCE",
    0.001,
    minimum=0.0,
)
# Fallback TP construction for fresh entries without a structural HARVEST
# anchor still asks for at least 1R before expanding the broker TP distance.
# Live permission itself must come from RiskEngine's regime-derived RR floor;
# otherwise this intent layer silently reintroduces the fixed-RR veto that
# range/failure scalps were explicitly designed to avoid.
FRESH_ENTRY_HARVEST_TP_FALLBACK_MIN_REWARD_RISK = 1.0
OPPORTUNITY_MODE_HARVEST_REWARD_RISK_MAX = 1.35
OPPORTUNITY_MODE_RUNNER_REWARD_RISK_MIN = 2.0


def _broker_price_tick_pips(pair: str) -> float:
    """Return one broker price tick expressed in pips for the pair precision."""
    digits = 3 if pair.endswith("_JPY") else 5
    return (10**-digits) * PIP_FACTORS[pair]


def _fresh_entry_live_floor_distance_pips(pair: str, stop_pips: float) -> float:
    """Minimum TP distance for NEW non-recovery entries to clear live RR."""
    return (FRESH_ENTRY_HARVEST_TP_FALLBACK_MIN_REWARD_RISK * stop_pips) + _broker_price_tick_pips(pair)


def _market_derived_reward_risk(chart_context: dict[str, Any] | None) -> tuple[float, list[str]]:
    """Compute reward_risk from current market state.

    Inputs (all from `chart_context`, which `_chart_context_for`
    flattens from pair_charts.json):
      - confluence.atr_percentile_24h: 0.0-1.0
      - h1_adx or h4_adx: trend strength (0-100)
      - confluence.range_24h_sigma_multiple: 0+ (exhaustion proxy)
      - session_current_tag: LONDON_NY_OVERLAP / OFF_HOURS / TOKYO_KILLZONE / ...

    Returns (reward_risk, rationale_lines). Falls back to
    DYNAMIC_RR_BASE when chart_context is missing entirely. Never
    silently returns the legacy 1.5 literal.
    """
    rationale: list[str] = []
    rr = DYNAMIC_RR_BASE

    if not chart_context:
        return DYNAMIC_RR_BASE, ["chart_context missing → base reward_risk"]

    # ATR percentile (24h band).
    confluence = chart_context.get("confluence") or {}
    try:
        atr_pct = float(confluence.get("atr_percentile_24h"))
    except (TypeError, ValueError):
        atr_pct = None
    if atr_pct is not None:
        if atr_pct >= DYNAMIC_RR_ATR_PCTILE_HIGH:
            rr += DYNAMIC_RR_ATR_HIGH_BONUS
            rationale.append(f"ATR %ile {atr_pct:.2f} ≥ {DYNAMIC_RR_ATR_PCTILE_HIGH} → +{DYNAMIC_RR_ATR_HIGH_BONUS}")
        elif atr_pct <= DYNAMIC_RR_ATR_PCTILE_LOW:
            rr -= DYNAMIC_RR_ATR_LOW_PENALTY
            rationale.append(f"ATR %ile {atr_pct:.2f} ≤ {DYNAMIC_RR_ATR_PCTILE_LOW} → -{DYNAMIC_RR_ATR_LOW_PENALTY}")

    # ADX (use H1 by default).
    try:
        adx = float(chart_context.get("h1_adx") or chart_context.get("h4_adx"))
    except (TypeError, ValueError):
        adx = None
    if adx is not None:
        if adx >= DYNAMIC_RR_ADX_TREND_THRESHOLD:
            rr += DYNAMIC_RR_ADX_TREND_BONUS
            rationale.append(f"ADX {adx:.1f} ≥ {DYNAMIC_RR_ADX_TREND_THRESHOLD} → +{DYNAMIC_RR_ADX_TREND_BONUS}")
        elif adx <= DYNAMIC_RR_ADX_RANGE_THRESHOLD:
            rr -= DYNAMIC_RR_ADX_RANGE_PENALTY
            rationale.append(f"ADX {adx:.1f} ≤ {DYNAMIC_RR_ADX_RANGE_THRESHOLD} → -{DYNAMIC_RR_ADX_RANGE_PENALTY}")

    # 24h range exhaustion.
    try:
        sigma_24h = float(confluence.get("range_24h_sigma_multiple"))
    except (TypeError, ValueError):
        sigma_24h = None
    if sigma_24h is not None and sigma_24h >= DYNAMIC_RR_SIGMA_EXHAUSTED_THRESHOLD:
        rr -= DYNAMIC_RR_SIGMA_EXHAUSTED_PENALTY
        rationale.append(f"24h σ {sigma_24h:.2f} ≥ {DYNAMIC_RR_SIGMA_EXHAUSTED_THRESHOLD} → -{DYNAMIC_RR_SIGMA_EXHAUSTED_PENALTY}")

    # Session adjustment.
    session = str(chart_context.get("session_current_tag") or chart_context.get("session_bucket") or "").upper()
    if "LONDON_NY_OVERLAP" in session:
        rr += DYNAMIC_RR_SESSION_OVERLAP_BONUS
        rationale.append(f"LDN/NY overlap session → +{DYNAMIC_RR_SESSION_OVERLAP_BONUS}")
    elif session in ("OFF_HOURS", "JP_HOLIDAY"):
        rr -= DYNAMIC_RR_SESSION_THIN_PENALTY
        rationale.append(f"{session} → -{DYNAMIC_RR_SESSION_THIN_PENALTY}")

    rr_clamped = max(DYNAMIC_RR_FLOOR, min(DYNAMIC_RR_CEILING, rr))
    if rr_clamped != rr:
        rationale.append(f"clamped to [{DYNAMIC_RR_FLOOR}, {DYNAMIC_RR_CEILING}]")
    return round(rr_clamped, 2), rationale

# Regime-derived stop widening per AGENT_CONTRACT §3.5 + feedback_no_tight_sl_thin_market.md.
# Wider SL is engaged when noise/uncertainty is elevated so the trade is not
# stopped out by the wick-noise floor of an unclear regime.
#
# (a) market reality: low-confidence regime classification means the four
#     constituent signals (Hurst/ADX/Choppiness/ATR percentile) disagree.
#     High atr_percentile means current ATR sits above its long-window
#     distribution — bigger candles, deeper wicks. Either condition expands
#     the realistic invalidation distance.
# (b) constants rather than derived: these are operator knobs for converting
#     measured noise into a stop multiplier. The detector outputs are derived;
#     the *response* is policy.
# (c) replace via: tune via post-trade-learning evidence — if SL hits cluster
#     near M5 wick range during low-confidence cycles, raise the multiplier.
REGIME_LOW_CONFIDENCE_THRESHOLD = 0.5  # below this fraction of agreeing signals
REGIME_LOW_CONFIDENCE_STOP_MULT = 1.3
REGIME_HIGH_VOL_PCTILE = 0.8  # ATR sits in the top 20% of trailing year
REGIME_HIGH_VOL_STOP_MULT = 1.4
REGIME_MAX_STOP_WIDEN = 1.5  # ceiling — never widen more than 1.5× ATR floor

# Range rotation needs an actual rail, not a generic "a few pips away" limit.
# The buffer places the pending order just inside the rail so a touch can fill
# before the exact support/resistance tick. It is spread-derived because broker
# cost is the immediate market reality around a limit fill.
RANGE_RAIL_ENTRY_BUFFER_SPREAD_MULT = 0.5
RANGE_OPPOSING_RAIL_BUFFER_SPREAD_MULT = 0.5

# A range LIMIT must still be pending, not effectively marketable. The minimum
# distance from current bid/ask is expressed in current spread multiples so it
# tightens and loosens with live liquidity rather than a fixed pip literal.
PENDING_ENTRY_OFFSET_SPREAD_MULT = 2.0

# Range MARKET participation is only valid when the current executable price is
# already in the rail zone. The zone is market-derived: the larger of current
# spread noise and a fraction of current M5 ATR. It is a trigger-shape
# constant, not a risk gate; risk still comes from ATR/spread geometry and
# RiskEngine.
RANGE_MARKET_EDGE_ZONE_SPREAD_MULT = 2.0
RANGE_MARKET_EDGE_ZONE_ATR_MULT = 0.25

# Directional range participation is the low-volatility variant of
# RANGE_ROTATION. It is still risk-capped by per-trade JPY budget, but when M5
# says RANGE + QUIET and the M5 directional bias agrees with the lane, the
# trader may take a market scalp from inside the box instead of waiting only at
# the rail.
#
# - RANGE_DIRECTIONAL_STOP_SPREAD_MULT: OANDA fill / rounding noise requires
#   the stop to clear RiskPolicy.min_stop_spread_multiple (5x spread). 5.1x is
#   the smallest generator-side cushion that still validates after price
#   rounding, so low-vol ATR can translate into larger units without raising
#   the loss cap.
# - RANGE_DIRECTIONAL_MARKET_TARGET_RR_CAP: low-vol range scalps should rotate
#   fast; cap the market target to one stop-distance even when historical lane
#   evidence asks for a runner. The lane's base RR remains recorded in
#   metadata, while this execution shape pursues small repeatable wins.
RANGE_DIRECTIONAL_STOP_SPREAD_MULT = 5.1
RANGE_DIRECTIONAL_MARKET_TARGET_RR_CAP = 1.0
# Directional range MARKET is the "small capture" lane. It must be taken from a
# real box edge, not the middle of the range, otherwise the bot sells lower-half
# ranges or buys upper-half ranges and pays spread for the worst part of the
# rotation. One-third of the box is the smallest geometric edge bucket that still
# leaves a distinct middle/no-trade zone; tune via range-scalp MAE/MFE once enough
# live receipts exist.
RANGE_DIRECTIONAL_MARKET_EDGE_POSITION = _env_float(
    "QR_RANGE_DIRECTIONAL_MARKET_EDGE_POSITION", 1.0 / 3.0, minimum=0.05
)

# Structural range rails. 1σ VWAP bands are deliberately excluded here: they
# are often the box interior / magnet zone, not the actual fail point. Treating
# them as rails creates tiny spread-dominated boxes and makes range trades look
# worse than they are. Use outer bands, Donchian rails, swing extremes, and
# linear-regression channel edges as the executable rotation boundaries.
RANGE_SUPPORT_LEVEL_KEYS = ("bb_lower", "donchian_low", "avwap_lower_2sd", "swing_low", "linreg_channel_lower")
RANGE_RESISTANCE_LEVEL_KEYS = ("bb_upper", "donchian_high", "avwap_upper_2sd", "swing_high", "linreg_channel_upper")

# Current-range auto lanes prevent forecast RANGE from starving when the daily
# campaign was mined from trend stories but the live chart is now a boxed tape.
# Market reality: a stable range is only exploitable when executable support
# and resistance rails exist; a squeeze / BREAKOUT_PENDING box is explicitly
# not a range-fade setup. The 1.0 evidence floor means one explicit current
# range read (dominant or M5/M15/H1 state) is enough only when rails are also
# present. ADX<20 and Chop>61.8 are the same Wilder/Dreiss stable-range
# defaults used elsewhere in the regime layer. RANGE_FORMING uses ADX<25 (not
# strong trend), Chop>=45 (mixed/choppy tape), or BB-width percentile<=35
# (compression beginning but not bottom-25% squeeze); those are pre-existing
# range-phase / breakout-pending boundaries from the forecaster, not
# pair-specific P/L decisions. Replace with ledger-calibrated pair/session
# thresholds once enough RANGE_ROTATION / range-breakout outcomes exist.
RANGE_AUTOLANE_TIMEFRAMES = ("M5", "M15", "M30", "H1")
RANGE_AUTOLANE_MIN_EVIDENCE = 1.0
RANGE_AUTOLANE_TARGET_RR_CAP = 2.0
RANGE_BREAKOUT_AUTOLANE_TARGET_RR_FLOOR = 2.0
RANGE_AUTOLANE_ADX_MAX = 20.0
RANGE_AUTOLANE_CHOP_MIN = 61.8
RANGE_AUTOLANE_SQUEEZE_PCT_MAX = 25.0
RANGE_AUTOLANE_FORMING_ADX_MAX = 25.0
RANGE_AUTOLANE_FORMING_CHOP_MIN = 45.0
RANGE_AUTOLANE_FORMING_BB_WIDTH_PCT_MAX = 35.0

# Market-location map: the trader must know which timeframe is boxed, which
# timeframe is trending, and which nearby levels form the actual battlefield.
# Cluster radius is a fraction of the current M5 ATR, so it expands/contracts
# with live volatility rather than a fixed pip literal. `0.15 × ATR` groups
# levels that are close enough to behave as the same decision zone on the
# operating TF; replace with ledger-calibrated pair/session fractions once
# level-cluster outcomes have enough samples.
MARKET_LOCATION_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
# Level clusters are metadata only, but the fraction must still express market
# reality: levels separated by less than the M5 sub-noise band behave as one
# execution zone rather than separate prices. Wider/narrower values should come
# from post-trade level-cluster outcome samples, not from a fixed pip guess.
LEVEL_CLUSTER_ATR_FRACTION = 0.15
# Keep only the nearest actionable levels/zones so receipts carry context
# without drowning GPT/the operator in stale far-away prices.
LEVEL_CONTEXT_LIMIT = 6
NEWS_CONTEXT_LIMIT = 4
LEVEL_MIDPOINT_KEYS = (
    "bb_middle",
    "vwap",
    "avwap_anchor",
    "avwap_upper_1sd",
    "avwap_lower_1sd",
    "sma_20",
    "ema_20",
    "ema_50",
    "ichimoku_kijun",
)
# Wilder's ADX 25 "trend is distinguishable from noise" boundary is already
# documented in analysis/regime.py and directional_forecaster.py. This local
# alias avoids importing the forecaster just to classify metadata.
FORECAST_STRONG_ADX_PROXY = 25.0

# Forecast-first lane seeding fixes the candidate-list blind spot: if the
# campaign plan only contains stale/archived directions, the predictor must be
# able to create the current pair/direction candidate before TraderBrain scores
# lanes. These are method *families*, not live permission. Risk, profile,
# spread, levels, and GPT verification still decide whether a seed can trade.
# The set intentionally covers trend continuation, failed-break reversal, and
# executable range rotation so the candidate generator does not decide the
# method before geometry and market context have been checked.
FORECAST_SEED_DIRECTIONAL_METHODS = (
    TradeMethod.BREAKOUT_FAILURE.value,
    TradeMethod.TREND_CONTINUATION.value,
)
FORECAST_SEED_RANGE_METHODS = (TradeMethod.RANGE_ROTATION.value,)
FORECAST_SEED_DESK_BY_METHOD = {
    TradeMethod.BREAKOUT_FAILURE.value: "failure_trader",
    TradeMethod.TREND_CONTINUATION.value: "trend_trader",
    TradeMethod.RANGE_ROTATION.value: "range_trader",
}
# RANGE forecasts answer "is the box still a tradeable box?", not "will the
# next directional leg trend far enough?" A calibrated probability above 50%
# is therefore the natural live floor, but only for executable RANGE_RAIL_LIMIT
# geometry with TP inside the box and SL outside it. Trend/breakout forecasts
# still use the stricter directional ENTRY_CONFIDENCE_MIN from the forecaster.
FORECAST_RANGE_ROTATION_MIN_CONFIDENCE = _env_float(
    "QR_FORECAST_RANGE_ROTATION_MIN_CONFIDENCE",
    0.50,
    minimum=0.50,
)
# 2026-06-15 live forecast audit:
# - OANDA M1 truth on 2,466 post-cycle UP/DOWN forecasts: hit=45.7%,
#   avg_move=-0.95p, avg_MAE=7.92p > avg_MFE=6.37p.
# - Calibrated confidence >=0.55 was still negative expectancy; >=0.65 was
#   the first bucket with hit>50% and positive average move. Keep candidate
#   seeding at the forecaster floor, but require the higher bar for LIVE_READY
#   directional entries unless audited projection support rescues a near miss.
FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE = _env_float(
    "QR_FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE",
    0.65,
    minimum=0.50,
)
# Watch-only forecast lanes expose geometry for pairs the campaign list omits
# when either the raw forecast or the current chart vote is materially strong,
# but calibrated confidence is still below the live-entry floor. This prevents
# the candidate surface from hiding obvious chart opportunities while the
# FORECAST_WATCH_ONLY blocker keeps them non-fillable until the predictor earns
# enough calibrated confidence.
FORECAST_WATCH_MIN_RAW_CONFIDENCE = _env_float(
    "QR_FORECAST_WATCH_MIN_RAW_CONFIDENCE",
    0.50,
    minimum=0.0,
)
FORECAST_WATCH_MIN_CHART_SCORE = _env_float(
    "QR_FORECAST_WATCH_MIN_CHART_SCORE",
    0.65,
    minimum=0.50,
)
# Post-harvest re-entry is a candidate-surface memory, not a risk gate. Keep it
# inside the same operating window as predictive LIMIT orders: a local M1/M5
# top/bottom close is relevant for the immediate retest, not as a day-long
# directional bias.
POST_HARVEST_REENTRY_LOOKBACK_MIN = _env_float("QR_POST_HARVEST_REENTRY_LOOKBACK_MIN", 90.0, minimum=5.0)
POST_HARVEST_REENTRY_MAX_SEEDS = _env_int("QR_POST_HARVEST_REENTRY_MAX_SEEDS", 8, minimum=0)
# Matrix-supported repair seeding is a candidate-surface repair, not live
# permission. A pair/side must have support from at least three independent
# market-context layers so a repair lane cannot be born from a single spread
# or level reading. The seed cap keeps the generator focused on the strongest
# current repairs before broad exploration; override it only after coverage
# reports show the queue is starved.
MATRIX_REPAIR_MIN_SUPPORT_LAYERS = _env_int(
    "QR_MATRIX_REPAIR_MIN_SUPPORT_LAYERS",
    3,
    minimum=1,
)
MATRIX_REPAIR_MAX_SEEDS = _env_int(
    "QR_MATRIX_REPAIR_MAX_SEEDS",
    9,
    minimum=0,
)
# Directional forecasts are calibrated as a final pair-level detector, but
# recent live evidence can show that the final detector is underperforming while
# specific market-condition detectors (liquidity sweep, squeeze, session timing)
# are still scoring well in projection_ledger. A near-miss confidence forecast
# may clear live context only when current, same-cycle projection evidence is
# auditable and materially better than chance. The confidence shortfall is
# bounded to 0.10 so this path rescues calibrated near-misses (for example
# 0.58 vs a 0.65 floor), not genuinely weak predictions.
FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL = _env_float(
    "QR_FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL",
    0.10,
    minimum=0.0,
)
# Weak forecast support override must not convert a range-edge fakeout into a
# live trend entry. The 2026-06-15 EUR_CHF loss was exactly this shape:
# calibrated forecast below the live floor, support override present, M5/M15
# boxed near the upper rail, then a LONG STOP filled and reversed. Keep the
# threshold as a location boundary inside the current box; replace it with
# pair/session-calibrated failed-break outcome stats once there are enough
# tagged support-override STOP fills.
FORECAST_SUPPORT_RANGE_EDGE_CHASE_POSITION = _env_float(
    "QR_FORECAST_SUPPORT_RANGE_EDGE_CHASE_POSITION",
    0.80,
    minimum=0.50,
)
# Directional projection support must beat random direction by at least five
# points before it can offset pair-forecast calibration. Spread/RR/risk gates
# still run after this, so this is only permission to avoid a forecast-only veto.
FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE = _env_float(
    "QR_FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE",
    0.55,
    minimum=0.50,
)
# EITHER signals forecast timing/volatility, not side. They can support a raw
# directional forecast only when their own hit-rate is materially high.
FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE = _env_float(
    "QR_FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE",
    0.70,
    minimum=0.50,
)
# A current projection signal that is itself below 55% confidence is not strong
# enough to offset a weak final forecast. This floor is detector-confidence, not
# a profit guarantee; the projection ledger hit-rate floor is checked separately.
FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE = _env_float(
    "QR_FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE",
    0.55,
    minimum=0.0,
)
# Reuse the projection-ledger calibration sample floor unless explicitly raised.
FORECAST_MARKET_SUPPORT_MIN_SAMPLES = _env_int(
    "QR_FORECAST_MARKET_SUPPORT_MIN_SAMPLES",
    _env_int("QR_PROJECTION_CONFIDENCE_MIN_SAMPLES", 10, minimum=1),
    minimum=1,
)
# Directional forecast calibration is the final detector's own realized truth,
# not a secondary market signal. If its current pair/regime/side bucket is
# below 45% HIT on enough samples, high raw confidence is not enough for live
# entry: that is the "confident but reverse-first" failure mode seen in recent
# USD_CAD/EUR_CHF loss-closes. These defaults mirror the self-improvement
# forecast warning floor so audit and live-readiness use the same market fact.
# Replace them only after projection_ledger bucket expectancy shows a different
# break-even floor for live entries, not by lowering the gate to create volume.
FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE = _env_float(
    "QR_FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE",
    0.45,
    minimum=0.0,
)
FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES = _env_int(
    "QR_FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES",
    FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
    minimum=FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
)
# Same-cycle projection bootstrap for newly added detectors. The normal path
# requires projection_ledger samples before a near-miss forecast can trade.
# That is correct for mature detectors, but it creates a cold-start dead zone:
# a factual macro surprise or other high-confidence projection cannot gather
# live samples because it is blocked for not already having live samples. This
# bootstrap is deliberately narrower than the audited path: raw forecast must
# already clear the live-entry floor, calibrated confidence must be only a
# near miss (bounded by FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL), and
# at least one aligned directional projection must be high-confidence. Spread,
# RR, structure, profile downgrades, telemetry, and gateway checks still run.
# Replace this with signal-specific Bayesian priors once each new projection
# has enough event-tagged samples in projection_ledger.
FORECAST_BOOTSTRAP_RAW_CONFIDENCE_MIN = _env_float(
    "QR_FORECAST_BOOTSTRAP_RAW_CONFIDENCE_MIN",
    0.60,
    minimum=0.50,
)
FORECAST_BOOTSTRAP_SIGNAL_CONFIDENCE_MIN = _env_float(
    "QR_FORECAST_BOOTSTRAP_SIGNAL_CONFIDENCE_MIN",
    0.70,
    minimum=0.50,
)
# Strong directional audited support (2026-06-15).
#
# The ordinary market-support path only rescues a calibrated near-miss forecast
# (default max shortfall 0.10). Live evidence can also produce a different
# shape: the aggregate final detector is pessimistic, but raw confidence clears
# the live floor and a same-direction, current projection bucket has substantial
# audited follow-through. That is usable only for STOP-ENTRY confirmation, where
# price has to trade through the trigger first; MARKET/LIMIT entries would still
# spend time underwater when the forecast is early.
#
# Per AGENT_CONTRACT §3.5:
# (a) market reality: event/liquidity projection buckets with >=75% hit-rate on
#     >=30 samples are materially different from timing-only EITHER signals.
#     Requiring raw forecast to be near the live floor preserves the pair-level
#     thesis, while the calibrated floor stops genuinely weak final predictions
#     from trading just because one detector is hot.
# (b) constants rather than derived: these are operator risk-policy caps for how
#     much audited same-direction evidence may offset the final detector.
# (c) replace via: tune by post-trade expectancy for this override path once the
#     execution ledger has enough STOP-ENTRY fills tagged with
#     strong_directional_projection_support.
FORECAST_STRONG_DIRECTIONAL_MAX_CONFIDENCE_SHORTFALL = _env_float(
    "QR_FORECAST_STRONG_DIRECTIONAL_MAX_CONFIDENCE_SHORTFALL",
    0.25,
    minimum=FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL,
)
FORECAST_STRONG_DIRECTIONAL_RAW_MAX_CONFIDENCE_SHORTFALL = _env_float(
    "QR_FORECAST_STRONG_DIRECTIONAL_RAW_MAX_CONFIDENCE_SHORTFALL",
    0.05,
    minimum=0.0,
)
FORECAST_STRONG_DIRECTIONAL_CALIBRATED_FLOOR = _env_float(
    "QR_FORECAST_STRONG_DIRECTIONAL_CALIBRATED_FLOOR",
    0.40,
    minimum=0.0,
)
FORECAST_STRONG_DIRECTIONAL_MIN_HIT_RATE = _env_float(
    "QR_FORECAST_STRONG_DIRECTIONAL_MIN_HIT_RATE",
    0.75,
    minimum=FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE,
)
FORECAST_STRONG_DIRECTIONAL_MIN_SIGNAL_CONFIDENCE = _env_float(
    "QR_FORECAST_STRONG_DIRECTIONAL_MIN_SIGNAL_CONFIDENCE",
    0.70,
    minimum=FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE,
)
FORECAST_STRONG_DIRECTIONAL_MIN_SAMPLES = _env_int(
    "QR_FORECAST_STRONG_DIRECTIONAL_MIN_SAMPLES",
    max(30, FORECAST_MARKET_SUPPORT_MIN_SAMPLES),
    minimum=FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
)
# Forecast-first trend continuation must pay for being wrong. RANGE-side
# breakout STOPs can be useful, but the 2026-06-15 EUR_CHF loss showed that
# a weak calibrated forecast + one higher-TF trend hint + ~1.2R geometry spends
# hours underwater and then exits by thesis invalidation. Keep those as dry-run
# evidence unless the live geometry offers at least a real trend-trade payoff.
FORECAST_SEED_TREND_MIN_LIVE_REWARD_RISK = _env_float(
    "QR_FORECAST_SEED_TREND_MIN_LIVE_REWARD_RISK",
    1.5,
    minimum=1.0,
)
# Forecast-first seeds are allowed only when the chart packet contains at
# least two independent TF readings with both regime and family-score context.
# A single TF can be enough for a unit fixture, but it is not enough market
# story to create a new candidate direction before the ordinary campaign lanes.
# Production chart_reader currently emits seven such TFs; if that source shape
# changes, replace this with an explicit chart-reader capability flag.
FORECAST_SEED_MIN_RICH_TF_VIEWS = 2

# Broker-side TP attachment mode. Small-wave / range / failed-break setups
# should bank at the nearest structural target, while strong trend setups
# need a runner path where broker TP is omitted and the position is managed
# by dynamic TP rebalance + profit partial closes.
#
# (a) market reality: ADX ≥ 25 is the same trend-strength boundary already
#     used by the dynamic reward/risk layer; two of three TFs agreeing is the
#     documented M15/M30/H1 majority threshold in AGENT_CONTRACT §3.5.
# (b) constants rather than derived: these convert market-derived context
#     into an execution mode. Missing context defaults to ATTACHED_TP so a data
#     gap never silently creates an uncapped runner.
# (c) replace via: post-trade learning on runner giveback vs capped TP fills.
TP_MODE_TF_AGREEMENT_MAJORITY = 2.0 / 3.0
TP_MODE_EXHAUSTION_SIGMA = DYNAMIC_RR_SIGMA_EXHAUSTED_THRESHOLD
# Attached HARVEST TPs are expected to be reachable in the operating tape.
# This mirrors tp_rebalancer's `MAX_TP_DISTANCE_ATR_MULT`: a target beyond
# 10× current operating ATR is a runner, not a failed-break harvest.
HARVEST_TP_MAX_OPERATING_ATR_MULT = _env_float("QR_HARVEST_TP_MAX_OPERATING_ATR_MULT", 10.0, minimum=1.0)

# Generic trend/failure stops should not sit inside the current wick shelf.
# The buffer is one live spread beyond the adverse structural level because the
# spread is the current broker noise floor; the actual distance remains driven
# by ATR, spread, and chart-derived levels, not a fixed pip literal.
STRUCTURAL_STOP_BUFFER_SPREAD_MULT = 1.0

# Trend-continuation direction gate. The chart reader labels aggregate
# long-vs-short vote gaps within +/-0.05 as TIED; a hard trend gate should only
# fire when the opposite side is decisively outside that documented noise band.
# Using 2x the tied boundary matches trader_brain's directional gate: it blocks
# genuine trend-continuation contradictions while leaving weak/mixed tape to
# range/failure desks so entries do not dry up.
CHART_DIRECTION_TIED_GAP_BOUNDARY = 0.05
TREND_CONTINUATION_STRONG_BIAS_MULT = 2.0
TREND_CONTINUATION_STRONG_BIAS_GAP = (
    CHART_DIRECTION_TIED_GAP_BOUNDARY * TREND_CONTINUATION_STRONG_BIAS_MULT
)

# Pattern-chase blocker names are the reversal / failed-break shapes emitted by
# pattern_signals.py. This is intentionally categorical rather than numeric:
# momentum-only patterns (Aroon, HVN magnets, inside-bar continuation) can still
# support a lane score, but only failed-break / exhaustion / reversal candle
# evidence is allowed to veto a MARKET/STOP chase.
PATTERN_CHASE_BLOCK_NAMES = frozenset(
    {
        "failed_breakout",
        "rsi_extreme_top",
        "rsi_extreme_bottom",
        "dealing_range_top",
        "dealing_range_bottom",
        "bullish_engulfing",
        "bearish_engulfing",
        "hammer",
        "shooting_star",
        "doji_after_bull",
        "doji_after_bear",
        "volume_spike_climax_up",
        "volume_spike_climax_down",
        "time_exhaustion",
        "morning_star",
        "evening_star",
    }
)


def _per_trade_risk_from_state(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> float | None:
    """Return per-trade JPY cap from the daily target ledger, or None if unavailable.

    Reads `per_trade_risk_budget_jpy` (= daily_risk_budget_jpy / target_trades_per_day),
    written by DailyTargetLedger. Falling back to `daily_risk_budget_jpy` (the
    whole-day total) would mean a single trade can burn the day's entire risk
    budget, which is exactly the failure mode this split was built to remove.
    Per AGENT_CONTRACT §3.5: no silent literal fallback; if the file is missing
    or the value is missing/zero, return None and let the caller raise.
    """
    return _state_field(state_path, "per_trade_risk_budget_jpy")


def _daily_risk_budget_from_state(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> float | None:
    """Return whole-day JPY risk budget from the daily target ledger.

    Per AGENT_CONTRACT §3.5 the **portfolio** cap (open + candidate exposure)
    must be the day's total budget, NOT the per-trade slice. Reusing the
    per-trade cap as the portfolio cap silently blocks every additional shot
    once one position opens, because `open_risk + candidate_risk` immediately
    exceeds a per-shot limit. Returns None when the ledger is absent so the
    caller can decide whether to skip the portfolio gate (no-op) rather than
    invent a JPY literal.
    """
    return _state_field(state_path, "daily_risk_budget_jpy")


def _daily_entry_budget_block_issue(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> dict[str, str] | None:
    """Return a BLOCK issue when the daily ledger says no fresh-entry risk remains.

    `per_trade_risk_budget_jpy` is intentionally a per-shot sizing cap and can
    stay positive after open risk has consumed the remaining day budget. The
    separate `remaining_risk_budget_jpy` / `RISK_BUDGET_EXHAUSTED` state is the
    market-day circuit breaker for additional entries.
    """
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    status = str(payload.get("status") or "")
    remaining_raw = payload.get("remaining_risk_budget_jpy")
    remaining: float | None
    try:
        remaining = float(remaining_raw) if remaining_raw is not None else None
    except (TypeError, ValueError):
        remaining = None
    if status != "RISK_BUDGET_EXHAUSTED" and not (remaining is not None and remaining <= 0):
        return None
    remaining_text = "unknown" if remaining is None else f"{remaining:.4f}"
    return {
        "code": "DAILY_RISK_BUDGET_EXHAUSTED",
        "message": (
            "daily target remaining_risk_budget_jpy is "
            f"{remaining_text} (status={status or 'UNKNOWN'}); block fresh entries until "
            "broker truth and the daily target ledger reopen risk budget"
        ),
        "severity": "BLOCK",
    }


def _state_field(state_path: Path, key: str) -> float | None:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get(key)
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _campaign_plan_staleness_issue(
    plan: dict[str, Any],
    *,
    campaign_plan_path: Path,
    strategy_profile_path: Path,
    data_root: Path,
    snapshot: BrokerSnapshot | None,
) -> str | None:
    """Return a loud production blocker when a dated plan is reused across states."""
    if snapshot is None:
        return None

    target_state_path = _campaign_target_state_path(campaign_plan_path, data_root)
    target_state = _load_json_dict(target_state_path) if target_state_path is not None else {}
    strategy_profile = _load_strategy_profile_for_plan(strategy_profile_path, data_root)
    if not target_state and not strategy_profile:
        return None

    target_generated_at = _parse_telemetry_time(target_state.get("generated_at_utc"))
    strategy_generated_at = _parse_telemetry_time(strategy_profile.get("generated_at_utc"))
    generated_at = _parse_telemetry_time(plan.get("generated_at_utc"))
    if generated_at is None:
        if target_state or strategy_generated_at is not None:
            return (
                f"campaign plan lacks generated_at_utc while broker snapshot is current: {campaign_plan_path}; "
                "run plan-campaign before generate-intents"
            )
        return None

    if target_generated_at is not None and generated_at < target_generated_at:
        return (
            "campaign plan stale relative to daily target state: "
            f"{campaign_plan_path} generated at {generated_at.isoformat()} before "
            f"{target_state_path} {target_generated_at.isoformat()}; run plan-campaign"
        )

    if strategy_generated_at is not None and generated_at < strategy_generated_at:
        return (
            "campaign plan stale relative to strategy profile: "
            f"{campaign_plan_path} generated at {generated_at.isoformat()} before "
            f"{strategy_profile_path} {strategy_generated_at.isoformat()}; run plan-campaign"
        )

    mismatch = _campaign_target_mismatch(plan, target_state)
    if mismatch:
        return (
            f"campaign plan target state mismatch while broker snapshot is current: {mismatch}; "
            f"plan={campaign_plan_path} target_state={target_state_path}; run plan-campaign"
        )
    return None


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _campaign_target_state_path(campaign_plan_path: Path, data_root: Path) -> Path | None:
    local_target_state = campaign_plan_path.parent / DEFAULT_DAILY_TARGET_STATE.name
    campaign_is_in_data_root = _path_is_same_or_under(campaign_plan_path, data_root)
    if local_target_state.exists() and campaign_is_in_data_root:
        return local_target_state
    if campaign_is_in_data_root or (
        _paths_equivalent(campaign_plan_path, DEFAULT_CAMPAIGN_PLAN)
        and _paths_equivalent(data_root, DEFAULT_CAMPAIGN_PLAN.parent)
    ):
        return data_root / DEFAULT_DAILY_TARGET_STATE.name
    return None


def _load_strategy_profile_for_plan(strategy_profile_path: Path, data_root: Path) -> dict[str, Any]:
    if not _path_is_same_or_under(strategy_profile_path, data_root):
        return {}
    return _load_json_dict(strategy_profile_path)


def _path_is_same_or_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _paths_equivalent(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _campaign_target_mismatch(plan: dict[str, Any], target_state: dict[str, Any]) -> str | None:
    if not target_state:
        return None
    for key in ("start_balance_jpy", "target_jpy"):
        planned = _optional_float(plan.get(key))
        current = _optional_float(target_state.get(key))
        if planned is None or current is None:
            continue
        tolerance = max(1.0, abs(current) * 0.0001)
        if abs(planned - current) > tolerance:
            return f"{key} plan={planned:.2f} target_state={current:.2f}"
    return None


def _load_pair_charts(charts_path: Path = DEFAULT_PAIR_CHARTS) -> dict[str, dict[str, Any]] | None:
    """Load pair_charts.json indexed by pair name.

    Returns a dict like {"EUR_USD": {"M5": {"atr_pips": 5.2, "regime": ...}, ...}, ...}.
    Returns None when the file is absent / malformed — the caller must decide
    whether to BLOCK the cycle (production) or proceed without ATR (tests).
    """
    if not charts_path.exists():
        return None
    try:
        payload = json.loads(charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw_charts = payload.get("charts")
    if not isinstance(raw_charts, list):
        return None
    indexed: dict[str, dict[str, Any]] = {}
    for chart in raw_charts:
        pair = chart.get("pair")
        if not isinstance(pair, str):
            continue
        per_tf: dict[str, Any] = {
            "__raw_chart": chart,
            "generated_at_utc": payload.get("generated_at_utc"),
            "dominant_regime": chart.get("dominant_regime"),
            "long_score": chart.get("long_score"),
            "short_score": chart.get("short_score"),
            "session": chart.get("session") if isinstance(chart.get("session"), dict) else {},
            # Carry the inline structural chart_story (M1...struct=BOS_UP@...)
            # so downstream scoring (`trader_brain._micro_structure_direction`,
            # `trader_brain._short_term_momentum_class`) can parse the same
            # micro-structure signal the operator reads, without re-loading
            # pair_charts.json. The narrative chart_story on `MarketContext`
            # is news/quality_audit excerpts and is left untouched.
            "chart_story": chart.get("chart_story", "") or "",
            # 2026-05-13 precision filters (B/C/D) sourced from
            # chart_reader._build_extended_confluence. Carrying the whole
            # confluence dict here means consumers (intent_generator
            # context issues, trader_brain gating, attack_advisor) all
            # see the same statistic without re-deriving anything.
            "confluence": chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {},
        }
        for view in chart.get("views", []) or []:
            granularity = view.get("granularity")
            if isinstance(granularity, str):
                per_tf[f"{granularity}__regime"] = view.get("regime")
                per_tf[f"{granularity}__long_bias"] = view.get("long_bias")
                per_tf[f"{granularity}__short_bias"] = view.get("short_bias")
                family_scores = view.get("family_scores")
                if isinstance(family_scores, dict):
                    per_tf[f"{granularity}__family_scores"] = family_scores
                per_tf[granularity] = view.get("indicators", {}) or {}
                # Expose regime_reading per timeframe so geometry can read
                # confidence / atr_percentile for regime-aware SL widening.
                # Per AGENT_CONTRACT §3.5: data-derived; no silent literal
                # replaces a missing reading — caller falls back to base stop.
                regime_reading = view.get("regime_reading")
                if isinstance(regime_reading, dict):
                    per_tf[f"{granularity}__regime_reading"] = regime_reading
                recent_candles = view.get("recent_candles")
                if isinstance(recent_candles, list):
                    per_tf[f"{granularity}__recent_candles"] = recent_candles
        indexed[pair] = per_tf
    return indexed if indexed else None


def _load_levels_snapshot(levels_path: Path = DEFAULT_LEVELS_SNAPSHOT) -> dict[str, Any] | None:
    if not levels_path.exists():
        return None
    try:
        payload = json.loads(levels_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_market_context_matrix(matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX) -> dict[str, Any] | None:
    if not matrix_path.exists():
        return None
    try:
        payload = json.loads(matrix_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atr_pips_for(pair: str, charts: dict[str, dict[str, Any]] | None, timeframe: str = GEOMETRY_ATR_TIMEFRAME) -> float | None:
    """Look up ATR(pips) for a pair on the given timeframe. None when missing."""
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    indicators = per_tf.get(timeframe)
    if not isinstance(indicators, dict):
        return None
    raw = indicators.get("atr_pips")
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _regime_state_for(pair: str, charts: dict[str, dict[str, Any]] | None) -> str | None:
    """Return the chart-level dominant_regime string for a pair, or None.

    The dominant_regime aggregates the per-timeframe reading layer into a
    single classification (TREND_UP/TREND_DOWN/RANGE/IMPULSE_*/FAILURE_RISK/
    UNCLEAR). It is the operator's primary signal for choosing whether the
    current cycle is rotation territory or trend continuation territory.
    Per AGENT_CONTRACT §3.5: returning None when missing forces the caller
    to fall back to the lane's base reward_risk — no silent literal swap.
    """
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    raw = per_tf.get("dominant_regime")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _regime_reading_for(
    pair: str,
    charts: dict[str, dict[str, Any]] | None,
    timeframe: str = GEOMETRY_ATR_TIMEFRAME,
) -> dict[str, Any] | None:
    """Return the per-timeframe regime_reading dict, or None when absent.

    Used by stop-widening to access regime confidence + atr_percentile from
    the chart reader's reading layer (Hurst/ADX/Choppiness/ATR-percentile).
    """
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    raw = per_tf.get(f"{timeframe}__regime_reading")
    if isinstance(raw, dict):
        return raw
    return None


def _session_bucket_for(pair: str, charts: dict[str, dict[str, Any]] | None) -> str | None:
    """Return a coarse session bucket aligned with archive outcome evidence.

    Pair charts expose richer tags such as NY_AM_KILLZONE or SILVER_BULLET.
    The archive outcome mart is intentionally coarser (ASIA/LONDON/NY/ROLLOVER)
    so condition matching stays comparable across old and current packets.
    """
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    session = per_tf.get("session")
    if not isinstance(session, dict):
        return None
    return _session_bucket_from_tag(session.get("current_tag"))


def _chart_context_for(pair: str, charts: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """Return current pair-chart direction/method context for executable gates.

    This is not a score-threshold selector. It records the chart reader's
    current long-vs-short orientation and operating timeframe regime so order
    intents cannot become LIVE_READY while directly contradicting the same
    evidence packet that supplied their ATR geometry.
    """
    if charts is None:
        return {}
    per_tf = charts.get(pair)
    if not per_tf:
        return {}
    range_phase, range_breakout_direction = _current_range_phase(pair, charts)
    long_score = _optional_float(per_tf.get("long_score"))
    short_score = _optional_float(per_tf.get("short_score"))
    conf = per_tf.get("confluence") if isinstance(per_tf.get("confluence"), dict) else {}
    bias = None
    score_balance = str(conf.get("score_balance") or "").upper()
    if score_balance == "LONG_LEAN":
        bias = Side.LONG.value
    elif score_balance == "SHORT_LEAN":
        bias = Side.SHORT.value
    elif (
        long_score is not None
        and short_score is not None
        and abs(long_score - short_score) > CHART_DIRECTION_TIED_GAP_BOUNDARY
    ):
        bias = Side.LONG.value if long_score > short_score else Side.SHORT.value
    m5_indicators = per_tf.get("M5") if isinstance(per_tf.get("M5"), dict) else {}
    h1_indicators = per_tf.get("H1") if isinstance(per_tf.get("H1"), dict) else {}
    h4_indicators = per_tf.get("H4") if isinstance(per_tf.get("H4"), dict) else {}
    m5_family = per_tf.get("M5__family_scores") if isinstance(per_tf.get("M5__family_scores"), dict) else {}
    context = {
        "chart_long_score": long_score,
        "chart_short_score": short_score,
        "chart_direction_bias": bias,
        "m5_regime": _text_or_none(per_tf.get("M5__regime")),
        "m15_regime": _text_or_none(per_tf.get("M15__regime")),
        "h1_regime": _text_or_none(per_tf.get("H1__regime")),
        "m5_long_bias": _optional_float(per_tf.get("M5__long_bias")),
        "m5_short_bias": _optional_float(per_tf.get("M5__short_bias")),
        "m15_long_bias": _optional_float(per_tf.get("M15__long_bias")),
        "m15_short_bias": _optional_float(per_tf.get("M15__short_bias")),
        "m5_regime_quantile": _text_or_none(m5_indicators.get("regime_quantile")),
        "m5_mean_rev_score": _optional_float(m5_family.get("mean_rev_score")),
        "m5_trend_score": _optional_float(m5_family.get("trend_score")),
        "m5_breakout_score": _optional_float(m5_family.get("breakout_score")),
        "m5_family_disagreement": _optional_float(m5_family.get("disagreement")),
        "h1_adx": _optional_float(h1_indicators.get("adx_14") or h1_indicators.get("adx") or h1_indicators.get("ADX")),
        "h4_adx": _optional_float(h4_indicators.get("adx_14") or h4_indicators.get("adx") or h4_indicators.get("ADX")),
        # Inline structural multi-TF chart_story for trader_brain micro-
        # structure / momentum scoring. Distinct key from market_context's
        # narrative `chart_story` (news/quality_audit excerpts).
        "chart_story_structural": str(per_tf.get("chart_story") or ""),
        # 2026-05-13 precision context (B/C/D feed). Producers:
        # chart_reader._build_extended_confluence. Consumers:
        # _method_context_issues for the C-gate (24h sigma BLOCK on
        # same-side entry after a 2+σ move), trader_brain
        # _apply_directional_gating for B and D scoring, and
        # attack_advisor confidence ranking.
        "confluence": conf,
        "price_percentile_24h": _optional_float(conf.get("price_percentile_24h")),
        "price_percentile_7d": _optional_float(conf.get("price_percentile_7d")),
        "price_range_24h_low": _price_range_low(conf, h1_indicators, horizon="24h"),
        "price_range_24h_high": _price_range_high(conf, h1_indicators, horizon="24h"),
        "price_range_24h_source": _price_range_source(conf, h1_indicators, horizon="24h"),
        "price_range_7d_low": _price_range_low(conf, h4_indicators, horizon="7d"),
        "price_range_7d_high": _price_range_high(conf, h4_indicators, horizon="7d"),
        "price_range_7d_source": _price_range_source(conf, h4_indicators, horizon="7d"),
        "atr_percentile_24h": _optional_float(conf.get("atr_percentile_24h")),
        "range_24h_sigma_multiple": _optional_float(conf.get("range_24h_sigma_multiple")),
        "tf_agreement_score": _optional_float(conf.get("tf_agreement_score")),
        "chart_score_balance": _text_or_none(conf.get("score_balance")),
        "chart_score_gap": _optional_float(conf.get("score_gap")),
        "higher_tf_regime": _text_or_none(conf.get("higher_tf_regime")),
        "higher_tf_alignment": _text_or_none(conf.get("higher_tf_alignment")),
        "range_phase": range_phase,
        "range_breakout_direction": range_breakout_direction,
        # F (2026-05-13) — H4 ATR pips for noise-resistant initial SL.
        # Consumed by `_generic_geometry` only when
        # QR_NEW_ENTRY_INITIAL_SL=1; raw value is published here so the
        # operator can audit the SL widening on a per-cycle basis.
        "h4_atr_pips": _optional_float(
            (per_tf.get("H4") if isinstance(per_tf.get("H4"), dict) else {}).get("atr_pips")
        ),
        # I (2026-05-13) — session liquidity tag for spread tolerance
        # and SL widening. Producer: chart_reader._build_session_for_pair.
        # Distinct key from the existing `session_bucket` which carries
        # the normalized session-name token (TOKYO/LONDON/NY/etc.);
        # this carries the chart_reader killzone label
        # (LONDON_KILLZONE/JUDAS/OFF_HOURS/etc.) used by the F/I rules.
        "session_current_tag": _text_or_none(
            (per_tf.get("session") if isinstance(per_tf.get("session"), dict) else {}).get("current_tag")
        ),
    }
    context.update(_pattern_context_for(per_tf.get("__raw_chart")))
    return context


def _price_range_low(conf: dict[str, Any], indicators: dict[str, Any], *, horizon: str) -> float | None:
    value = _optional_float(conf.get(f"price_range_{horizon}_low"))
    if value is not None:
        return value
    return _optional_float(indicators.get("donchian_low"))


def _price_range_high(conf: dict[str, Any], indicators: dict[str, Any], *, horizon: str) -> float | None:
    value = _optional_float(conf.get(f"price_range_{horizon}_high"))
    if value is not None:
        return value
    return _optional_float(indicators.get("donchian_high"))


def _price_range_source(conf: dict[str, Any], indicators: dict[str, Any], *, horizon: str) -> str | None:
    if _optional_float(conf.get(f"price_range_{horizon}_low")) is not None and _optional_float(
        conf.get(f"price_range_{horizon}_high")
    ) is not None:
        return f"confluence_{horizon}"
    if _optional_float(indicators.get("donchian_low")) is not None and _optional_float(
        indicators.get("donchian_high")
    ) is not None:
        return "donchian_proxy"
    return None


def _pattern_context_for(raw_chart: Any) -> dict[str, Any]:
    """Summarize current failed-break / candle-shape evidence for gates.

    The pattern detector already converts candle/structure events into
    direction-tagged signals. Intent generation keeps both all-pattern weights
    and the reversal-only subset so executable gates can distinguish "momentum
    agrees" from "price just failed to break and printed an exhaustion shape".
    Missing chart data returns no fields instead of inventing a neutral value.
    """
    if not isinstance(raw_chart, dict):
        return {}
    try:
        from quant_rabbit.strategy.pattern_signals import detect_pattern_signals
    except Exception:
        return {}
    try:
        signals = detect_pattern_signals(raw_chart)
    except Exception:
        return {}
    if not signals:
        return {"pattern_signal_count": 0}

    weights = {Side.LONG.value: 0.0, Side.SHORT.value: 0.0}
    reversal_weights = {Side.LONG.value: 0.0, Side.SHORT.value: 0.0}
    records: list[dict[str, Any]] = []
    for signal in signals:
        direction = str(getattr(signal, "direction", "") or "").upper()
        if direction == "UP":
            side = Side.LONG.value
        elif direction == "DOWN":
            side = Side.SHORT.value
        else:
            continue
        confidence = _optional_float(getattr(signal, "confidence", None)) or 0.0
        magnitude = _optional_float(getattr(signal, "bonus_magnitude", None)) or 0.0
        weight = max(0.0, confidence) * max(0.0, magnitude)
        name = str(getattr(signal, "name", "") or "")
        is_reversal = name in PATTERN_CHASE_BLOCK_NAMES
        weights[side] += weight
        if is_reversal:
            reversal_weights[side] += weight
        records.append(
            {
                "name": name,
                "timeframe": str(getattr(signal, "timeframe", "") or ""),
                "direction": direction,
                "side": side,
                "confidence": round(confidence, 3),
                "weight": round(weight, 2),
                "chase_block_evidence": is_reversal,
                "rationale": str(getattr(signal, "rationale", "") or ""),
            }
        )

    records.sort(key=lambda item: _optional_float(item.get("weight")) or 0.0, reverse=True)
    long_weight = round(weights[Side.LONG.value], 2)
    short_weight = round(weights[Side.SHORT.value], 2)
    reversal_long = round(reversal_weights[Side.LONG.value], 2)
    reversal_short = round(reversal_weights[Side.SHORT.value], 2)
    return {
        "pattern_signal_count": len(records),
        "pattern_weight_long": long_weight,
        "pattern_weight_short": short_weight,
        "pattern_dominant_side": _dominant_weight_side(long_weight, short_weight),
        "pattern_reversal_weight_long": reversal_long,
        "pattern_reversal_weight_short": reversal_short,
        "pattern_reversal_dominant_side": _dominant_weight_side(reversal_long, reversal_short),
        "pattern_signals": records[:12],
    }


def _dominant_weight_side(long_weight: float, short_weight: float) -> str | None:
    if long_weight > short_weight:
        return Side.LONG.value
    if short_weight > long_weight:
        return Side.SHORT.value
    return None


def _market_location_context_for(
    pair: str,
    current_price: float | None,
    charts: dict[str, dict[str, Any]] | None,
    levels_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if current_price is None or charts is None:
        return {}
    per_tf = charts.get(pair)
    if not per_tf:
        return {}
    tf_map: dict[str, dict[str, Any]] = {}
    range_tfs: list[str] = []
    trend_tfs: list[str] = []
    structural_levels: list[dict[str, Any]] = []
    for timeframe in MARKET_LOCATION_TIMEFRAMES:
        indicators = per_tf.get(timeframe) if isinstance(per_tf.get(timeframe), dict) else {}
        if not indicators:
            continue
        regime = _text_or_none(per_tf.get(f"{timeframe}__regime"))
        state = _regime_reading_state(per_tf, timeframe) or None
        classification = _tf_regime_classification(regime, state, indicators)
        support = _nearest_below(current_price, _numeric_levels(indicators, RANGE_SUPPORT_LEVEL_KEYS))
        resistance = _nearest_above(current_price, _numeric_levels(indicators, RANGE_RESISTANCE_LEVEL_KEYS))
        position = None
        if support is not None and resistance is not None and resistance > support:
            position = max(0.0, min(1.0, (current_price - support) / (resistance - support)))
        entry = {
            "regime": regime,
            "state": state,
            "classification": classification,
            "range_position": round(position, 3) if position is not None else None,
            "nearest_support": _round_price(pair, support) if support is not None else None,
            "nearest_support_distance_pips": _distance_pips(pair, support, current_price) if support is not None else None,
            "nearest_resistance": _round_price(pair, resistance) if resistance is not None else None,
            "nearest_resistance_distance_pips": _distance_pips(pair, resistance, current_price) if resistance is not None else None,
            "adx": _optional_float((indicators or {}).get("adx_14") or (indicators or {}).get("adx")),
            "choppiness": _optional_float((indicators or {}).get("choppiness_14")),
            "linreg_slope": _optional_float((indicators or {}).get("linreg_slope_20")),
            "atr_pips": _optional_float((indicators or {}).get("atr_pips")),
        }
        tf_map[timeframe] = {key: value for key, value in entry.items() if value is not None}
        if classification == "RANGE":
            range_tfs.append(timeframe)
        elif classification in {"TREND_UP", "TREND_DOWN"}:
            trend_tfs.append(f"{timeframe}:{classification}")
        structural_levels.extend(_indicator_levels(pair, timeframe, current_price, indicators))

    snapshot_levels = _levels_snapshot_levels(pair, current_price, levels_snapshot)
    all_levels = structural_levels + snapshot_levels
    nearest_below = _nearest_level_records(all_levels, side="below", limit=LEVEL_CONTEXT_LIMIT)
    nearest_above = _nearest_level_records(all_levels, side="above", limit=LEVEL_CONTEXT_LIMIT)
    cluster_radius_pips = _cluster_radius_pips(pair, charts)
    level_clusters = _level_clusters_near(pair, current_price, all_levels, cluster_radius_pips)
    story = _market_location_story(tf_map, nearest_below, nearest_above, level_clusters)
    return {
        "current_price_mid": _round_price(pair, current_price),
        "tf_regime_map": tf_map,
        "range_timeframes": range_tfs,
        "trend_timeframes": trend_tfs,
        "nearest_levels_below": nearest_below,
        "nearest_levels_above": nearest_above,
        "level_clusters_near": level_clusters,
        "level_cluster_radius_pips": round(cluster_radius_pips, 2) if cluster_radius_pips is not None else None,
        "market_location_story": story,
    }


def _tf_regime_classification(regime: str | None, state: str | None, indicators: dict[str, Any]) -> str:
    regime_text = str(regime or "").upper()
    state_text = str(state or "").upper()
    if "BREAKOUT_PENDING" in {regime_text, state_text}:
        return "BREAKOUT_PENDING"
    if "RANGE" in regime_text or state_text == "RANGE":
        return "RANGE"
    if regime_text.startswith("TREND_UP") or state_text == "TREND_UP":
        return "TREND_UP"
    if regime_text.startswith("TREND_DOWN") or state_text == "TREND_DOWN":
        return "TREND_DOWN"
    slope = _optional_float((indicators or {}).get("linreg_slope_20"))
    adx = _optional_float((indicators or {}).get("adx_14") or (indicators or {}).get("adx"))
    if adx is not None and adx >= FORECAST_STRONG_ADX_PROXY:
        if slope is not None and slope > 0:
            return "TREND_UP"
        if slope is not None and slope < 0:
            return "TREND_DOWN"
    if state_text:
        return state_text
    return regime_text or "UNKNOWN"


def _indicator_levels(
    pair: str,
    timeframe: str,
    current_price: float,
    indicators: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in (*RANGE_SUPPORT_LEVEL_KEYS, *RANGE_RESISTANCE_LEVEL_KEYS, *LEVEL_MIDPOINT_KEYS):
        price = _optional_float((indicators or {}).get(key))
        if price is None or price <= 0:
            continue
        out.append(_level_record(pair, current_price, price, f"{timeframe}:{key}"))
    return out


def _levels_snapshot_levels(
    pair: str,
    current_price: float,
    levels_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    reading = _levels_snapshot_pair(pair, levels_snapshot)
    if not reading:
        return []
    out: list[dict[str, Any]] = []
    for key in ("pdh", "pdl", "pdc", "pdo", "daily_open", "weekly_open", "monthly_open", "last_close"):
        price = _optional_float(reading.get(key))
        if price is not None and price > 0:
            out.append(_level_record(pair, current_price, price, f"levels:{key}"))
    for pivot in reading.get("pivots") or []:
        if not isinstance(pivot, dict):
            continue
        style = str(pivot.get("style") or "PIVOT").lower()
        for key in ("pp", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"):
            price = _optional_float(pivot.get(key))
            if price is not None and price > 0:
                out.append(_level_record(pair, current_price, price, f"levels:pivot:{style}:{key}"))
    for session in reading.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        name = str(session.get("name") or "SESSION").lower()
        for key in ("high", "low"):
            price = _optional_float(session.get(key))
            if price is not None and price > 0:
                out.append(_level_record(pair, current_price, price, f"levels:session:{name}:{key}"))
    for item in reading.get("round_numbers") or []:
        if not isinstance(item, dict):
            continue
        price = _optional_float(item.get("price"))
        if price is not None and price > 0:
            out.append(_level_record(pair, current_price, price, "levels:round_number"))
    return out


def _levels_snapshot_pair(pair: str, levels_snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(levels_snapshot, dict):
        return None
    for item in levels_snapshot.get("pairs") or []:
        if isinstance(item, dict) and item.get("pair") == pair:
            return item
    return None


def _level_record(pair: str, current_price: float, price: float, source: str) -> dict[str, Any]:
    distance = _distance_pips(pair, price, current_price)
    side = "below" if price <= current_price else "above"
    return {
        "price": _round_price(pair, price),
        "source": source,
        "side": side,
        "distance_pips": distance,
    }


def _distance_pips(pair: str, price: float | None, current_price: float) -> float | None:
    if price is None:
        return None
    return round((price - current_price) * PIP_FACTORS[pair], 2)


def _nearest_level_records(
    levels: list[dict[str, Any]],
    *,
    side: str,
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [item for item in levels if item.get("side") == side and item.get("distance_pips") is not None]
    filtered.sort(key=lambda item: abs(float(item["distance_pips"])))
    return filtered[:limit]


def _cluster_radius_pips(pair: str, charts: dict[str, dict[str, Any]] | None) -> float | None:
    atr = _atr_pips_for(pair, charts)
    if atr is None or atr <= 0:
        return None
    return atr * LEVEL_CLUSTER_ATR_FRACTION


def _level_clusters_near(
    pair: str,
    current_price: float,
    levels: list[dict[str, Any]],
    radius_pips: float | None,
) -> list[dict[str, Any]]:
    if not levels:
        return []
    if radius_pips is None or radius_pips <= 0:
        return []
    radius_price = radius_pips / PIP_FACTORS[pair]
    sorted_levels = sorted(levels, key=lambda item: float(item["price"]))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in sorted_levels:
        if not current:
            current = [item]
            continue
        center = sum(float(member["price"]) for member in current) / len(current)
        if abs(float(item["price"]) - center) <= radius_price:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)
    out: list[dict[str, Any]] = []
    for members in clusters:
        if len(members) < 2:
            continue
        center = sum(float(member["price"]) for member in members) / len(members)
        side = "below" if center <= current_price else "above"
        out.append(
            {
                "price": _round_price(pair, center),
                "side": side,
                "distance_pips": _distance_pips(pair, center, current_price),
                "count": len(members),
                "sources": sorted({str(member.get("source")) for member in members})[:LEVEL_CONTEXT_LIMIT],
            }
        )
    out.sort(key=lambda item: (abs(float(item.get("distance_pips") or 0.0)), -int(item.get("count") or 0)))
    return out[:LEVEL_CONTEXT_LIMIT]


def _market_location_story(
    tf_map: dict[str, dict[str, Any]],
    nearest_below: list[dict[str, Any]],
    nearest_above: list[dict[str, Any]],
    level_clusters: list[dict[str, Any]],
) -> str:
    tf_bits: list[str] = []
    for tf in MARKET_LOCATION_TIMEFRAMES:
        item = tf_map.get(tf)
        if not item:
            continue
        bit = f"{tf} {item.get('classification')}"
        if item.get("range_position") is not None:
            bit += f" pos={item['range_position']}"
        tf_bits.append(bit)
    below = nearest_below[0] if nearest_below else None
    above = nearest_above[0] if nearest_above else None
    level_bits: list[str] = []
    if below:
        level_bits.append(f"below {below['price']} ({below['distance_pips']}p {below['source']})")
    if above:
        level_bits.append(f"above {above['price']} (+{above['distance_pips']}p {above['source']})")
    if level_clusters:
        top = level_clusters[0]
        level_bits.append(f"cluster {top['price']} ({top['count']} levels, {top['distance_pips']}p)")
    return "; ".join(tf_bits[:7] + level_bits[:3])


def _session_bucket_from_tag(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if "LONDON" in text:
        return "LONDON"
    if text.startswith("NY") or "NEWYORK" in text or "NEW_YORK" in text or "SILVER_BULLET" in text:
        return "NY"
    if "ASIA" in text or "TOKYO" in text:
        return "ASIA"
    if "ROLLOVER" in text or "OFF_HOURS" in text:
        return "ROLLOVER"
    return None


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _regime_reward_risk_multiplier(regime_state: str | None) -> float:
    """Return the regime-derived multiplier on a lane's base target_reward_risk.

    Per AGENT_CONTRACT §3.5: trend regimes deserve wider targets, range regimes
    deserve faster rotation. Returns 1.0 when regime is unknown so missing data
    never silently shortens or extends targets.
    """
    if not regime_state:
        return 1.0
    upper = regime_state.upper()
    if "RANGE" in upper:
        return REGIME_REWARD_RISK_RANGE_MULT
    if "IMPULSE" in upper:
        return REGIME_REWARD_RISK_IMPULSE_MULT
    if "TREND" in upper:
        return REGIME_REWARD_RISK_TREND_MULT
    # FAILURE_RISK / UNCLEAR / TRANSITION fall through unchanged so the lane's
    # base target_reward_risk governs. Operator can size down via lane defs.
    return REGIME_REWARD_RISK_UNCLEAR_MULT


def _regime_stop_widening_multiplier(regime_reading: dict[str, Any] | None) -> float:
    """Return >=1.0 stop-widening multiplier based on regime confidence & vol.

    Widens (never narrows) when:
      - regime confidence is below `REGIME_LOW_CONFIDENCE_THRESHOLD` (signals
        disagree → realistic invalidation is wider than the cleanest case),
      - ATR percentile is above `REGIME_HIGH_VOL_PCTILE` (top of trailing-year
        volatility distribution → wick noise is structurally bigger).

    Returns 1.0 when the reading is missing — geometry stays on the ATR/spread
    floor without inventing a multiplier.
    """
    if not regime_reading:
        return 1.0
    multiplier = 1.0
    confidence = regime_reading.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < REGIME_LOW_CONFIDENCE_THRESHOLD:
        multiplier = max(multiplier, REGIME_LOW_CONFIDENCE_STOP_MULT)
    atr_pct = regime_reading.get("atr_percentile")
    normalized_atr_pct = _normalized_percentile(atr_pct)
    if normalized_atr_pct is not None and normalized_atr_pct > REGIME_HIGH_VOL_PCTILE:
        multiplier = max(multiplier, REGIME_HIGH_VOL_STOP_MULT)
    return min(REGIME_MAX_STOP_WIDEN, multiplier)


def _normalized_percentile(value: object) -> float | None:
    """Normalize percentile values emitted as either 0..1 or 0..100.

    pair_charts currently emits ATR percentile from different readers; some
    are fractional and some are percentage points. The risk response must not
    treat 35th percentile as 3500th percentile and widen quiet tape stops.
    """
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed < 0:
        return None
    return parsed / 100.0 if parsed > 1.0 else parsed


def _range_indicators_for(
    pair: str,
    charts: dict[str, dict[str, Any]] | None,
    timeframe: str = GEOMETRY_ATR_TIMEFRAME,
) -> dict[str, Any] | None:
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    indicators = per_tf.get(timeframe)
    return indicators if isinstance(indicators, dict) and indicators else None


def _append_current_range_phase_lanes(
    lanes: list[dict[str, Any]],
    charts: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if charts is None:
        return lanes
    out = list(lanes)
    seen = {(lane.get("desk"), lane.get("pair"), lane.get("direction"), lane.get("method")) for lane in out}
    for lane in lanes:
        pair = str(lane.get("pair") or "")
        source_direction = str(lane.get("direction") or "")
        if source_direction not in {Side.LONG.value, Side.SHORT.value}:
            continue
        phase, breakout_direction = _current_range_phase(pair, charts)
        if phase in {"IN_RANGE", "RANGE_FORMING"}:
            method = str(lane.get("method") or "")
            if method == TradeMethod.RANGE_ROTATION.value:
                continue
            if not _pair_has_current_range_rotation_edge(pair, charts, phase=phase):
                continue
            synthetic = dict(lane)
            base_rr = _optional_float(synthetic.get("target_reward_risk")) or RANGE_AUTOLANE_TARGET_RR_CAP
            forming = phase == "RANGE_FORMING"
            synthetic.update(
                {
                    "desk": "range_trader",
                    "method": TradeMethod.RANGE_ROTATION.value,
                    "campaign_role": (
                        f"{str(lane.get('campaign_role') or 'NOW')}_"
                        f"{'CURRENT_RANGE_FORMING' if forming else 'CURRENT_RANGE'}"
                    ),
                    "reason": (
                        f"{lane.get('reason') or 'evidence-backed lane'}; "
                        f"current chart phase={phase} with executable rails"
                    ),
                    "required_receipt": (
                        "Use exact forming-range rail/box LIMIT order intent; no market chase."
                        if forming
                        else "Use exact range rail/box order intent; market only if quote is already at executable rail."
                    ),
                    "target_reward_risk": min(base_rr, RANGE_AUTOLANE_TARGET_RR_CAP),
                }
            )
            key = (synthetic.get("desk"), synthetic.get("pair"), synthetic.get("direction"), synthetic.get("method"))
            if key in seen:
                continue
            out.append(synthetic)
            seen.add(key)
        elif phase in {"BREAKOUT_UP", "BREAKOUT_DOWN"} and breakout_direction in {"UP", "DOWN"}:
            breakout_side = Side.LONG.value if breakout_direction == "UP" else Side.SHORT.value
            method = str(lane.get("method") or "")
            if method == TradeMethod.TREND_CONTINUATION.value and source_direction == breakout_side:
                continue
            synthetic = dict(lane)
            base_rr = _optional_float(synthetic.get("target_reward_risk")) or 0.0
            synthetic.update(
                {
                    "desk": "trend_trader",
                    "direction": breakout_side,
                    "method": TradeMethod.TREND_CONTINUATION.value,
                    "campaign_role": f"{str(lane.get('campaign_role') or 'NOW')}_CURRENT_RANGE_BREAKOUT",
                    "reason": (
                        f"{lane.get('reason') or 'evidence-backed lane'}; "
                        f"current chart phase={phase} confirms range break {breakout_direction}"
                    ),
                    "required_receipt": (
                        "Use STOP-ENTRY or retest continuation after a close-confirmed range break; "
                        "do not fade the broken rail."
                    ),
                    "target_reward_risk": max(base_rr, RANGE_BREAKOUT_AUTOLANE_TARGET_RR_FLOOR),
                }
            )
            key = (synthetic.get("desk"), synthetic.get("pair"), synthetic.get("direction"), synthetic.get("method"))
            if key in seen:
                continue
            out.append(synthetic)
            seen.add(key)
    return out


def _append_post_harvest_reentry_lanes(
    lanes: list[dict[str, Any]],
    charts: dict[str, dict[str, Any]] | None,
    snapshot: BrokerSnapshot | None,
    *,
    data_root: Path,
) -> list[dict[str, Any]]:
    if snapshot is None or charts is None or POST_HARVEST_REENTRY_MAX_SEEDS <= 0:
        return lanes
    seeds = _recent_post_harvest_reentry_seeds(data_root, snapshot=snapshot)
    if not seeds:
        return lanes
    out = list(lanes)
    existing = {
        (lane.get("desk"), lane.get("pair"), lane.get("direction"), lane.get("method"))
        for lane in out
    }
    added = 0
    for seed in seeds:
        pair = str(seed.get("pair") or "")
        side = str(seed.get("side") or "").upper()
        if side not in {Side.LONG.value, Side.SHORT.value}:
            continue
        if pair not in snapshot.quotes:
            continue
        if _snapshot_has_trader_pair_exposure(snapshot, pair):
            continue
        if not _pair_has_current_range_rotation_edge(pair, charts):
            continue
        key = ("post_harvest_trader", pair, side, TradeMethod.RANGE_ROTATION.value)
        if key in existing:
            continue
        out.append(_post_harvest_reentry_lane(seed))
        existing.add(key)
        added += 1
        if added >= POST_HARVEST_REENTRY_MAX_SEEDS:
            break
    return out


def _recent_post_harvest_reentry_seeds(data_root: Path, *, snapshot: BrokerSnapshot) -> list[dict[str, Any]]:
    ledger = data_root / "execution_ledger.db"
    if not ledger.exists():
        return []
    now = _ensure_utc(snapshot.fetched_at_utc) or datetime.now(timezone.utc)
    cutoff_seconds = POST_HARVEST_REENTRY_LOOKBACK_MIN * 60.0
    rows: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            records = conn.execute(
                """
                SELECT ts_utc, pair, side, trade_id, order_id, exit_reason, raw_json
                FROM execution_events
                WHERE event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                  AND exit_reason = 'TAKE_PROFIT_MARKET'
                ORDER BY ts_utc DESC
                LIMIT 50
                """
            ).fetchall()
    except sqlite3.Error:
        return []
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        close_time = _parse_telemetry_time(record["ts_utc"])
        if close_time is None:
            continue
        if close_time > now:
            age_seconds = 0.0
        else:
            age_seconds = (now - close_time).total_seconds()
        if age_seconds > cutoff_seconds:
            continue
        raw = _json_dict(record["raw_json"])
        if not bool(raw.get("sent")):
            continue
        if str(raw.get("management_action") or "") != "TAKE_PROFIT_MARKET":
            continue
        if str(raw.get("owner") or "") != Owner.TRADER.value:
            continue
        reasons = [str(item) for item in raw.get("reasons") or [] if str(item)]
        reason_blob = " | ".join(reasons)
        if "temporary top profit-take" not in reason_blob and "temporary bottom profit-take" not in reason_blob:
            continue
        pair = str(record["pair"] or raw.get("pair") or "")
        side = _closed_position_side(record["side"], raw)
        if pair not in DEFAULT_TRADER_PAIRS or side not in {Side.LONG.value, Side.SHORT.value}:
            continue
        key = (pair, side)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        rows.append(
            {
                "pair": pair,
                "side": side,
                "trade_id": str(record["trade_id"] or raw.get("trade_id") or ""),
                "order_id": str(record["order_id"] or ""),
                "closed_at_utc": close_time.isoformat().replace("+00:00", "Z"),
                "age_minutes": round(age_seconds / 60.0, 2),
                "reasons": reasons[:4],
            }
        )
    return rows


def _post_harvest_reentry_lane(seed: dict[str, Any]) -> dict[str, Any]:
    pair = str(seed.get("pair") or "")
    side = str(seed.get("side") or "").upper()
    trade_id = str(seed.get("trade_id") or "unknown")
    closed_at = str(seed.get("closed_at_utc") or "unknown")
    reason = (
        f"post-harvest re-entry seed: trader-owned {pair} {side} trade {trade_id} "
        f"was profit-harvested at a temporary local extreme at {closed_at}; "
        "wait for the market to pull back to the range support/resistance rail before re-entering"
    )
    return {
        "desk": "post_harvest_trader",
        "pair": pair,
        "direction": side,
        "method": TradeMethod.RANGE_ROTATION.value,
        "adoption": "TRIGGER_RECEIPT_REQUIRED",
        "campaign_role": "POST_HARVEST_PULLBACK_REENTRY",
        "reason": reason,
        "required_receipt": (
            "Post-harvest re-entry lane: LIMIT only at the fresh range rail / pullback retest; "
            "no market chase and no same-pass re-entry. Forecast, telemetry, strategy profile, "
            "spread, range-location, and broker-truth gates must still pass before live use."
        ),
        "target_reward_risk": RANGE_AUTOLANE_TARGET_RR_CAP,
        "blockers": [],
        "story_examples": list(seed.get("reasons") or [])[:2],
        "post_harvest_reentry_seed": True,
        "post_harvest_trade_id": trade_id,
        "post_harvest_closed_at_utc": closed_at,
        "post_harvest_age_minutes": seed.get("age_minutes"),
    }


def _snapshot_has_trader_pair_exposure(snapshot: BrokerSnapshot, pair: str) -> bool:
    for position in snapshot.positions:
        if position.pair == pair and position.owner == Owner.TRADER:
            return True
    for order in snapshot.orders:
        if order.pair == pair and order.owner == Owner.TRADER:
            return True
    return False


def _closed_position_side(side_value: object, raw: dict[str, Any]) -> str | None:
    side = str(side_value or "").upper()
    if side in {Side.LONG.value, Side.SHORT.value}:
        return side
    units = _close_order_units(raw)
    if units is None:
        return None
    # OANDA close order units are opposite the position being reduced:
    # negative units reduce a LONG, positive units reduce a SHORT.
    if units < 0:
        return Side.LONG.value
    if units > 0:
        return Side.SHORT.value
    return None


def _close_order_units(raw: dict[str, Any]) -> int | None:
    candidates = [
        ((raw.get("response") or {}).get("orderCreateTransaction") or {}).get("units"),
        ((raw.get("response") or {}).get("orderFillTransaction") or {}).get("units"),
    ]
    for value in candidates:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            continue
    return None


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _append_matrix_supported_repair_lanes(
    lanes: list[dict[str, Any]],
    matrix: dict[str, Any] | None,
    strategy_profile: StrategyProfile | None,
    charts: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if MATRIX_REPAIR_MAX_SEEDS <= 0 or matrix is None or strategy_profile is None:
        return lanes
    pairs_payload = matrix.get("pairs") if isinstance(matrix, dict) else None
    if not isinstance(pairs_payload, dict):
        return lanes

    existing = {
        (lane.get("desk"), lane.get("pair"), lane.get("direction"), lane.get("method"))
        for lane in lanes
    }
    ranked: list[tuple[float, int, int, int, str, str, str, Any, dict[str, Any]]] = []
    for entry in strategy_profile.entries.values():
        if not _matrix_repair_entry_can_seed(entry):
            continue
        if entry.direction not in {Side.LONG.value, Side.SHORT.value}:
            continue
        side_payload = _matrix_side_payload(pairs_payload, entry.pair, entry.direction)
        if side_payload is None or not _matrix_side_is_strong_repair_support(side_payload):
            continue
        support_count = _optional_int(side_payload.get("support_count")) or len(side_payload.get("supports") or [])
        layer_count = len(_matrix_support_layers(side_payload))
        profit_score = _matrix_repair_profit_score(entry)
        evidence_n = _optional_int(getattr(entry, "positive_evidence_n", None)) or 0
        for method in _matrix_repair_methods(entry.method, entry.pair, charts):
            desk = FORECAST_SEED_DESK_BY_METHOD.get(method)
            if not desk:
                continue
            key = (desk, entry.pair, entry.direction, method)
            if key in existing:
                continue
            ranked.append((profit_score, support_count, layer_count, evidence_n, entry.pair, entry.direction, method, entry, side_payload))
            existing.add(key)

    if not ranked:
        return lanes
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5], item[6]))
    seeds = [
        _matrix_repair_seed_lane(entry, method=method, side_payload=side_payload)
        for _, _, _, _, _, _, method, entry, side_payload in ranked[:MATRIX_REPAIR_MAX_SEEDS]
    ]
    return seeds + lanes


def _matrix_side_payload(
    pairs_payload: dict[str, Any],
    pair: str,
    side: str,
) -> dict[str, Any] | None:
    per_pair = pairs_payload.get(pair)
    if not isinstance(per_pair, dict):
        return None
    payload = per_pair.get(side)
    return payload if isinstance(payload, dict) else None


def _matrix_side_is_strong_repair_support(side_payload: dict[str, Any]) -> bool:
    reject_count = _optional_int(side_payload.get("reject_count"))
    if reject_count is None:
        reject_count = len(side_payload.get("rejects") or [])
    if reject_count > 0:
        return False
    support_layers = _matrix_support_layers(side_payload)
    return len(support_layers) >= MATRIX_REPAIR_MIN_SUPPORT_LAYERS


def _matrix_repair_entry_can_seed(entry: Any) -> bool:
    status = str(getattr(entry, "status", "") or "").upper()
    if status in {"RISK_REPAIR_CANDIDATE", "BLOCK_UNTIL_NEW_EVIDENCE"}:
        return True
    if status != "WATCH_ONLY":
        return False
    return _watch_only_matrix_entry_has_positive_discovery(entry)


def _watch_only_matrix_entry_has_positive_discovery(entry: Any) -> bool:
    """Allow WATCH_ONLY matrix seeds only as diagnostic candidate coverage.

    WATCH_ONLY still blocks live send through StrategyProfile. The seed exists
    so coverage can price short-horizon and trigger-horizon geometry for a
    currently supported edge instead of hiding it as NO_CURRENT_LANE.
    """

    for attr in ("pretrade_net_jpy", "positive_best_jpy", "positive_tail_jpy"):
        value = _optional_float(getattr(entry, attr, None))
        if value is not None and value > 0:
            return True
    evidence_n = _optional_int(getattr(entry, "positive_evidence_n", None)) or 0
    return evidence_n > 0


def _matrix_support_layers(side_payload: dict[str, Any]) -> set[str]:
    layers: set[str] = set()
    for raw in side_payload.get("supports") or []:
        if not isinstance(raw, dict):
            continue
        layer = str(raw.get("layer") or "").strip()
        if layer:
            layers.add(layer)
    return layers


def _matrix_repair_methods(method: str | None, pair: str, charts: dict[str, dict[str, Any]] | None) -> tuple[str, ...]:
    parsed = str(method or "").strip().upper()
    if parsed in FORECAST_SEED_DESK_BY_METHOD:
        return (parsed,)
    methods: list[str] = []
    if _pair_has_current_range_rotation_edge(pair, charts):
        methods.append(TradeMethod.RANGE_ROTATION.value)
    methods.extend((TradeMethod.BREAKOUT_FAILURE.value, TradeMethod.TREND_CONTINUATION.value))
    return tuple(dict.fromkeys(methods))


def _matrix_repair_profit_score(entry: Any) -> float:
    best = _optional_float(getattr(entry, "positive_best_jpy", None)) or 0.0
    tail = _optional_float(getattr(entry, "positive_tail_jpy", None)) or 0.0
    return max(best, tail, 0.0)


def _matrix_repair_seed_lane(
    entry: Any,
    *,
    method: str,
    side_payload: dict[str, Any],
) -> dict[str, Any]:
    watch_only = str(getattr(entry, "status", "") or "").upper() == "WATCH_ONLY"
    support_messages = [
        str(item.get("message") or item.get("code") or "")
        for item in side_payload.get("supports") or []
        if isinstance(item, dict)
    ]
    support_messages = [message for message in support_messages if message]
    support_count = _optional_int(side_payload.get("support_count")) or len(side_payload.get("supports") or [])
    rr = max(_optional_float(getattr(entry, "target_reward_risk", None)) or 0.0, DYNAMIC_RR_BASE)
    return {
        "desk": FORECAST_SEED_DESK_BY_METHOD[method],
        "pair": entry.pair,
        "direction": entry.direction,
        "method": method,
        "adoption": "TRIGGER_RECEIPT_REQUIRED",
        "campaign_role": "MATRIX_SUPPORTED_WATCH_ONLY_REPAIR" if watch_only else "MATRIX_SUPPORTED_REPAIR",
        "reason": (
            f"matrix-supported repair seed: {side_payload.get('evidence_ref') or entry.pair + ':' + entry.direction} "
            f"support_count={support_count} layers={','.join(sorted(_matrix_support_layers(side_payload)))}; "
            f"profile={entry.status}"
            + ("; watch-only diagnostic geometry only" if watch_only else "")
        ),
        "required_receipt": (
            (
                "Watch-only matrix-supported lane: create pending dry-run geometry and blocker reasons only; "
                "do not send live until new evidence repairs the mined strategy profile."
            )
            if watch_only
            else (
                "Matrix-supported repair lane: create only a pending LIMIT/STOP-ENTRY dry-run receipt; "
                "do not market-chase the supported direction. RiskEngine, forecast freshness, spread, "
                "profile repair, and live gateway gates must still decide execution."
            )
        ),
        "target_reward_risk": rr,
        "blockers": [entry.required_fix] if entry.required_fix else [],
        "story_examples": support_messages[:2],
        "matrix_repair_seed": True,
        "matrix_watch_only_seed": watch_only,
        "matrix_repair_profile_status": entry.status,
    }


def _pre_entry_forecast_cycle_id(snapshot: BrokerSnapshot, *, pair_charts_path: Path) -> str:
    fetched = snapshot.fetched_at_utc.isoformat()
    charts_generated = "charts-unknown"
    try:
        payload = json.loads(pair_charts_path.read_text())
        if isinstance(payload, dict):
            charts_generated = str(payload.get("generated_at_utc") or charts_generated)
    except (OSError, json.JSONDecodeError):
        pass
    return f"pre-entry-forecast-refresh:{fetched}:{charts_generated}"


def _append_forecast_seed_lanes(
    lanes: list[dict[str, Any]],
    charts: dict[str, dict[str, Any]] | None,
    snapshot: BrokerSnapshot | None,
    *,
    data_root: Path | None = None,
    forecast_cycle_id: str | None = None,
) -> list[dict[str, Any]]:
    """Prepend predictor-created candidate lanes before campaign slicing.

    The old flow was candidate-first: campaign/outcome lanes were sliced, then
    TraderBrain applied the pair forecast as a veto/bonus. If the stale
    candidate list omitted the pair/direction the predictor currently liked,
    the predictor never got to express that opportunity. This helper inverts
    the first step: every pair in the fresh chart packet gets one pair-level
    forecast, and sufficiently confident forecasts seed same-direction lanes
    before `max_candidates` is applied. Lower-confidence forecasts are still
    attached to existing lanes so live readiness can say "weak forecast"
    instead of pretending forecast context is missing. They remain ordinary
    intents after that point; risk/profile/GPT can still block them.
    """
    if charts is None or snapshot is None:
        return lanes
    existing_by_key = {
        (lane.get("desk"), lane.get("pair"), lane.get("direction"), lane.get("method")): lane
        for lane in lanes
    }
    source_by_pair: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        pair = str(lane.get("pair") or "")
        if pair and pair not in source_by_pair:
            source_by_pair[pair] = lane

    seeds: list[dict[str, Any]] = []
    seeded_keys: set[tuple[Any, Any, Any, Any]] = set()
    forecasts_by_pair: dict[str, Any] = {}
    for pair in sorted(charts):
        quote = snapshot.quotes.get(pair)
        if quote is None:
            continue
        forecast = _forecast_seed_for_pair(pair, charts, snapshot, data_root=data_root)
        if forecast is None:
            continue
        _record_forecast_seed_telemetry(
            forecast,
            pair=pair,
            quote=quote,
            pair_chart=charts.get(pair),
            data_root=data_root,
            cycle_id=forecast_cycle_id,
            validation_time_utc=snapshot.fetched_at_utc,
        )
        direction = str(getattr(forecast, "direction", "") or "").upper()
        confidence = _optional_float(getattr(forecast, "confidence", None))
        if direction and confidence is not None:
            forecasts_by_pair[pair] = forecast
        min_confidence = _forecast_seed_min_confidence_for_direction(direction)
        watch_reason: str | None = None
        if confidence is None:
            continue
        if confidence < min_confidence:
            forecast_side = Side.LONG.value if direction == "UP" else Side.SHORT.value if direction == "DOWN" else None
            if not _forecast_market_support_allows_side(
                forecast_side,
                forecast,
                min_confidence=min_confidence,
            ):
                watch_reason = _forecast_watch_candidate_reason(
                    pair,
                    forecast,
                    charts,
                    source_by_pair=source_by_pair,
                    min_confidence=min_confidence,
                )
                if watch_reason is None:
                    continue
        range_geometry_watch_reason = _range_forecast_geometry_watch_reason(pair, forecast, charts)
        if range_geometry_watch_reason:
            watch_reason = (
                f"{watch_reason}; {range_geometry_watch_reason}"
                if watch_reason
                else range_geometry_watch_reason
            )
        methods = _forecast_seed_methods(pair, forecast, charts)
        if not methods:
            continue
        side = Side.LONG.value if direction == "UP" else Side.SHORT.value if direction == "DOWN" else None
        for method in methods:
            if direction == "RANGE":
                # RANGE forecasts are executable only through rail/box geometry.
                side = _range_seed_direction(pair, charts, quote.mid)
                if side is None:
                    continue
            if side not in {Side.LONG.value, Side.SHORT.value}:
                continue
            desk = FORECAST_SEED_DESK_BY_METHOD[method]
            key = (desk, pair, side, method)
            if key in seeded_keys:
                continue
            source = existing_by_key.get(key) or source_by_pair.get(pair)
            lane = _forecast_seed_lane(
                source,
                pair=pair,
                side=side,
                method=method,
                forecast=forecast,
                cycle_id=forecast_cycle_id,
            )
            if watch_reason is not None:
                lane["campaign_role"] = "FORECAST_WATCH"
                lane["forecast_watch_only"] = True
                lane["required_receipt"] = (
                    "Watch-only forecast-first lane: build dry-run geometry and blocker reasons only. "
                    "Do not send live until calibrated forecast confidence clears the live-entry floor "
                    "on a fresh snapshot."
                )
                lane["reason"] = f"{lane.get('reason') or 'forecast-first candidate discovery'}; {watch_reason}"
                lane["forecast_watch_only_reason"] = watch_reason
                lane["blockers"] = [
                    *list(lane.get("blockers") or []),
                    "watch-only forecast candidate below calibrated live-entry confidence",
                ]
            seeds.append(lane)
            seeded_keys.add(key)

    lanes = [
        _lane_with_forecast_context(
            lane,
            forecasts_by_pair.get(str(lane.get("pair") or "")),
            cycle_id=forecast_cycle_id,
        )
        for lane in lanes
    ]
    if not seeds:
        return lanes
    return seeds + [
        lane
        for lane in lanes
        if (lane.get("desk"), lane.get("pair"), lane.get("direction"), lane.get("method")) not in seeded_keys
    ]


def _record_forecast_seed_telemetry(
    forecast: Any,
    *,
    pair: str,
    quote: Quote,
    pair_chart: dict[str, Any] | None,
    data_root: Path | None,
    cycle_id: str | None,
    validation_time_utc: datetime | None = None,
) -> None:
    if not _require_telemetry_for_live_active() or data_root is None or not cycle_id:
        return
    raw_chart = pair_chart.get("__raw_chart") if isinstance(pair_chart, dict) else None
    try:
        current_price = float(quote.mid)
    except (TypeError, ValueError):
        return
    try:
        from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
        from quant_rabbit.strategy.projection_ledger import (
            projection_telemetry_market_open,
            record_directional_forecast,
            record_projections,
        )

        forecast_record = _forecast_with_pair(forecast, pair=pair)
        regime_label = _forecast_seed_regime_label(raw_chart) if isinstance(raw_chart, dict) else None
        emission_time = _ensure_utc(getattr(quote, "timestamp_utc", None))
        if not _quote_fresh_for_forecast_seed_telemetry(
            emission_time,
            validation_time_utc=validation_time_utc,
        ):
            return
        record_forecast(
            forecast_record,
            data_root=data_root,
            cycle_id=cycle_id,
            now=emission_time,
            replace_existing=True,
        )
        if not projection_telemetry_market_open(emission_time):
            return
        record_directional_forecast(
            forecast_record,
            pair=pair,
            current_price=current_price,
            data_root=data_root,
            regime_at_emission=regime_label,
            cycle_id=cycle_id,
            now=emission_time,
        )
        projection_signals = list(getattr(forecast_record, "projection_signals", ()) or ())
        if projection_signals:
            record_projections(
                projection_signals,
                pair=pair,
                current_price=current_price,
                data_root=data_root,
                regime_at_emission=regime_label,
                cycle_id=cycle_id,
                now=emission_time,
            )
    except Exception:
        return


def _quote_fresh_for_forecast_seed_telemetry(
    emission_time: datetime | None,
    *,
    validation_time_utc: datetime | None,
) -> bool:
    """Only record forecast telemetry when the price can be live-validated.

    Projection rows are HIT/MISS calibration samples. A stale weekend quote may
    still have an in-market timestamp, but it is not a tradable observation at
    the current snapshot time. Reuse RiskPolicy's quote-age contract so the
    telemetry ledger and live-entry risk gate agree on freshness.
    """
    emitted = _ensure_utc(emission_time)
    if emitted is None:
        return False
    validation_time = _ensure_utc(validation_time_utc) or datetime.now(timezone.utc)
    if emitted > validation_time:
        return True
    quote_age = (validation_time - emitted).total_seconds()
    return quote_age <= RiskPolicy().max_quote_age_seconds


class _ForecastPairProxy:
    def __init__(self, forecast: Any, *, pair: str) -> None:
        self._forecast = forecast
        self.pair = pair

    def __getattr__(self, name: str) -> Any:
        return getattr(self._forecast, name)


class _ForecastSeedProxy:
    def __init__(
        self,
        forecast: Any,
        *,
        pair: str,
        market_support: dict[str, Any],
        projection_signals: tuple[Any, ...],
    ) -> None:
        self._forecast = forecast
        self.pair = pair
        self.market_support = market_support
        self.projection_signals = projection_signals

    def __getattr__(self, name: str) -> Any:
        return getattr(self._forecast, name)

    def to_dict(self) -> dict[str, Any]:
        to_dict = getattr(self._forecast, "to_dict", None)
        payload = to_dict() if callable(to_dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("pair", self.pair)
        payload["market_support"] = self.market_support
        return payload


def _forecast_with_pair(forecast: Any, *, pair: str) -> Any:
    if str(getattr(forecast, "pair", "") or ""):
        return forecast
    return _ForecastPairProxy(forecast, pair=pair)


def _forecast_seed_for_pair(
    pair: str,
    charts: dict[str, dict[str, Any]],
    snapshot: BrokerSnapshot,
    *,
    data_root: Path | None = None,
) -> Any | None:
    per_tf = charts.get(pair)
    raw_chart = per_tf.get("__raw_chart") if isinstance(per_tf, dict) else None
    quote = snapshot.quotes.get(pair) if snapshot is not None else None
    if not isinstance(raw_chart, dict) or quote is None:
        return None
    if not _forecast_seed_has_rich_chart_context(raw_chart):
        return None
    try:
        current_price = float(quote.mid)
    except (TypeError, ValueError):
        return None
    if current_price <= 0:
        return None
    try:
        bid = float(quote.bid)
        ask = float(quote.ask)
        spread_pips = abs(ask - bid) * PIP_FACTORS.get(pair, instrument_pip_factor(pair))
    except (TypeError, ValueError):
        return None
    if spread_pips <= 0:
        return None
    full_charts = {
        chart_pair: chart_data.get("__raw_chart")
        for chart_pair, chart_data in charts.items()
        if isinstance(chart_data, dict) and isinstance(chart_data.get("__raw_chart"), dict)
    }
    try:
        from quant_rabbit.paths import ROOT
        from quant_rabbit.strategy.correlation_predictor import detect_correlation_lag
        from quant_rabbit.strategy.directional_forecaster import synthesize_forecast
        from quant_rabbit.strategy.forward_projection import detect_forward_projections
        from quant_rabbit.strategy.path_projection import detect_paths
        from quant_rabbit.strategy.pattern_signals import detect_pattern_signals
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates
        from quant_rabbit.strategy.reversal_signal import detect_reversal
    except Exception:
        return None

    # Forecast seeding itself stays side-effect free. IntentGenerator records
    # the synthesized pair forecast immediately after this helper returns when
    # live telemetry gates are active, so live-readiness validation can audit
    # the same snapshot instead of waiting for TraderBrain's later scoring pass.
    artifact_root = data_root or (ROOT / "data")
    cot_payload = _load_optional_json(artifact_root / "cot_snapshot.json")
    option_skew_payload = _load_optional_json(artifact_root / "option_skew_snapshot.json")
    try:
        pattern_signals = detect_pattern_signals(
            raw_chart,
            cot_payload=cot_payload,
            option_skew_payload=option_skew_payload,
        )
    except Exception:
        pattern_signals = []
    try:
        projection_signals = detect_forward_projections(
            raw_chart,
            pair=pair,
            current_price=current_price,
            calendar_path=artifact_root / "economic_calendar.json",
            news_digest_path=_news_digest_path_for_data_root(artifact_root),
            news_items_path=artifact_root / "news_items.json",
            cross_asset_path=artifact_root / "cross_asset_snapshot.json",
        )
    except Exception:
        projection_signals = []
    try:
        correlation_signals = detect_correlation_lag(pair, full_charts)
    except Exception:
        correlation_signals = []
    try:
        paths = list(detect_paths(raw_chart, Side.LONG.value, current_price))
        paths.extend(detect_paths(raw_chart, Side.SHORT.value, current_price))
    except Exception:
        paths = []
    try:
        reversal_long = detect_reversal(raw_chart, Side.LONG.value)
    except Exception:
        reversal_long = None
    try:
        reversal_short = detect_reversal(raw_chart, Side.SHORT.value)
    except Exception:
        reversal_short = None
    try:
        hit_rates = compute_hit_rates(artifact_root)
    except Exception:
        hit_rates = None
    regime_label = _forecast_seed_regime_label(raw_chart)
    try:
        forecast = synthesize_forecast(
            pair=pair,
            pair_chart=raw_chart,
            current_price=current_price,
            pattern_signals=pattern_signals,
            projection_signals=projection_signals,
            correlation_signals=correlation_signals,
            paths=paths,
            reversal_long=reversal_long,
            reversal_short=reversal_short,
            hit_rates=hit_rates,
            regime=regime_label,
            spread_pips=spread_pips,
        )
    except Exception:
        return None
    if (
        str(getattr(forecast, "direction", "") or "").upper() == "UNCLEAR"
        and (_optional_float(getattr(forecast, "confidence", None)) or 0.0) <= 0.0
    ):
        return None
    market_support = _forecast_market_support_for_forecast(
        pair=pair,
        forecast=forecast,
        projection_signals=projection_signals,
        hit_rates=hit_rates,
        regime=regime_label,
    )
    return _ForecastSeedProxy(
        forecast,
        pair=pair,
        market_support=market_support,
        projection_signals=tuple(projection_signals),
    )


def _forecast_seed_has_rich_chart_context(raw_chart: dict[str, Any]) -> bool:
    views = raw_chart.get("views")
    if not isinstance(views, list):
        return False
    rich_views = 0
    for view in views:
        if not isinstance(view, dict):
            continue
        if not isinstance(view.get("regime_reading"), dict):
            continue
        if not isinstance(view.get("family_scores"), dict):
            continue
        rich_views += 1
    return rich_views >= FORECAST_SEED_MIN_RICH_TF_VIEWS


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _news_digest_path_for_data_root(data_root: Path) -> Path:
    if data_root.name == "data":
        return data_root.parent / "logs" / "news_digest.md"
    return data_root / "news_digest.md"


def _intent_news_evidence_metadata(
    pair: str,
    *,
    data_root: Path | None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refs: list[str] = []
    signal_names: list[str] = []
    pair_context: list[str] = []
    digest_ref = ""

    def add_ref(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in refs:
            refs.append(text)

    def add_signal(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in signal_names:
            signal_names.append(text)

    if isinstance(source_metadata, dict):
        raw_refs = source_metadata.get("news_refs") or []
        if not isinstance(raw_refs, list):
            raw_refs = [raw_refs]
        for raw in raw_refs:
            add_ref(raw)
        digest_ref = str(source_metadata.get("news_digest_ref") or "").strip()
        raw_signals = source_metadata.get("news_signal_names") or []
        if not isinstance(raw_signals, list):
            raw_signals = [raw_signals]
        for raw in raw_signals:
            add_signal(raw)

    if data_root is not None:
        digest_path = _news_digest_path_for_data_root(data_root)
        if digest_path.exists():
            add_ref("news:digest")
            digest_ref = digest_ref or "news:digest"
            pair_context = _news_digest_pair_context(pair, digest_path)
        items_path = data_root / "news_items.json"
        if items_path.exists():
            add_ref("news:items")

    if not refs and not pair_context:
        return {}
    if refs and not signal_names:
        add_signal("market_story_news_artifact")
    payload: dict[str, Any] = {
        "news_refs": refs[:NEWS_CONTEXT_LIMIT],
        "news_signal_names": signal_names[:NEWS_CONTEXT_LIMIT],
    }
    if digest_ref:
        payload["news_digest_ref"] = digest_ref
    if "news:items" in refs:
        payload["news_items_ref"] = "news:items"
    if pair_context:
        payload["news_pair_context"] = pair_context[:NEWS_CONTEXT_LIMIT]
    return payload


def _news_digest_pair_context(pair: str, digest_path: Path) -> list[str]:
    try:
        text = digest_path.read_text(encoding="utf-8")
    except OSError:
        return []
    pair_key = pair.upper()
    pair_slash = pair_key.replace("_", "/")
    pair_flat = pair_key.replace("_", "")
    currencies = [part for part in pair_key.split("_") if part]
    exact: list[str] = []
    currency: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        line = line.lstrip("- ").strip()
        line_upper = line.upper()
        if pair_key in line_upper or pair_slash in line_upper or pair_flat in line_upper:
            if line not in exact:
                exact.append(line[:240])
            continue
        if any(ccy in line_upper for ccy in currencies) and line not in currency:
            currency.append(line[:240])
        if len(exact) >= NEWS_CONTEXT_LIMIT:
            break
    return (exact + currency)[:NEWS_CONTEXT_LIMIT]


def _forecast_seed_min_confidence() -> float:
    try:
        from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN

        return float(ENTRY_CONFIDENCE_MIN)
    except Exception:
        # ENTRY_CONFIDENCE_MIN itself is documented in the forecaster. This
        # branch is only for import failure in stripped tests; a high threshold
        # fails closed instead of seeding weak predictions.
        return 1.0


def _forecast_seed_min_confidence_for_direction(direction: str) -> float:
    entry_min = _forecast_seed_min_confidence()
    if direction == "RANGE":
        return min(entry_min, FORECAST_RANGE_ROTATION_MIN_CONFIDENCE)
    return entry_min


def _forecast_seed_regime_label(raw_chart: dict[str, Any]) -> str | None:
    conf = raw_chart.get("confluence") if isinstance(raw_chart.get("confluence"), dict) else {}
    raw = str((conf or {}).get("dominant_regime") or "").upper()
    if "TREND" in raw:
        return "TREND"
    if "RANGE" in raw:
        return "RANGE"
    return raw[:20] if raw else None


def _forecast_projection_support_sort_key(item: dict[str, Any]) -> tuple[float, int, float]:
    return (
        _optional_float(item.get("hit_rate")) or -1.0,
        _optional_int(item.get("samples")) or 0,
        _optional_float(item.get("confidence")) or 0.0,
    )


def _dedupe_forecast_projection_support(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A calibration bucket is one learned edge. Repeated macro events or
    # duplicate same-timeframe signals must not inflate the support count.
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for item in items:
        key = (
            str(item.get("calibration_name") or item.get("name") or ""),
            str(item.get("direction") or ""),
            str(item.get("timeframe") or ""),
        )
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            order.append(key)
            continue
        if _forecast_projection_support_sort_key(item) > _forecast_projection_support_sort_key(current):
            by_key[key] = item
    return sorted(
        [by_key[key] for key in order],
        key=_forecast_projection_support_sort_key,
        reverse=True,
    )


def _forecast_market_support_for_forecast(
    *,
    pair: str,
    forecast: Any,
    projection_signals: list[Any],
    hit_rates: dict[str, dict[str, Any]] | None,
    regime: str | None,
) -> dict[str, Any]:
    direction = str(getattr(forecast, "direction", "") or "").upper()
    raw_confidence = _optional_float(getattr(forecast, "raw_confidence", None))
    forecast_horizon_min = _optional_float(getattr(forecast, "horizon_min", None))
    out: dict[str, Any] = {
        "ok": False,
        "direction": direction,
        "aligned_projection_count": 0,
        "timing_projection_count": 0,
        "unselected_projection_count": 0,
        "best_hit_rate": None,
        "best_samples": 0,
        "best_unselected_hit_rate": None,
        "best_unselected_samples": 0,
        "bootstrap_projection_support": False,
        "reason": "",
        "signals": [],
        "unselected_signals": [],
        "unselected_reason": "",
    }
    out.update(
        _forecast_directional_calibration_for_forecast(
            pair=pair,
            direction=direction,
            hit_rates=hit_rates,
            regime=regime,
        )
    )
    if direction not in {"UP", "DOWN"} or not projection_signals:
        unselected = _forecast_unselected_projection_support(
            pair=pair,
            forecast_direction=direction,
            projection_signals=projection_signals,
            hit_rates=hit_rates,
            regime=regime,
            forecast_horizon_min=forecast_horizon_min,
        )
        if unselected:
            top = unselected[0]
            out.update(
                {
                    "unselected_projection_count": len(unselected),
                    "best_unselected_hit_rate": top["hit_rate"],
                    "best_unselected_samples": top["samples"],
                    "unselected_signals": unselected[:4],
                    "unselected_reason": (
                        f"{top['name']} {top['direction']} audited hit_rate={top['hit_rate']:.2f} "
                        f"samples={top['samples']} was unselected because forecast={direction or 'NONE'}"
                    ),
                }
            )
            out["reason"] = f"forecast {direction or 'NONE'} has no executable direction; audited projection unselected"
            return out
        out["reason"] = "no directional projection support"
        return out

    bootstrap = _forecast_bootstrap_projection_support(
        forecast=forecast,
        direction=direction,
        projection_signals=projection_signals,
        hit_rates=hit_rates,
        pair=pair,
        regime=regime,
    )
    bootstrap = _dedupe_forecast_projection_support(bootstrap)
    if not isinstance(hit_rates, dict):
        if bootstrap:
            out.update(
                {
                    "ok": True,
                    "aligned_projection_count": len(bootstrap),
                    "bootstrap_projection_support": True,
                    "reason": bootstrap[0]["reason"],
                    "signals": bootstrap[:4],
                }
            )
            return out
        out["reason"] = "no directional audited projection support"
        return out
    try:
        from quant_rabbit.strategy.projection_ledger import select_calibration_signal_name
    except Exception:
        if bootstrap:
            out.update(
                {
                    "ok": True,
                    "aligned_projection_count": len(bootstrap),
                    "bootstrap_projection_support": True,
                    "reason": bootstrap[0]["reason"],
                    "signals": bootstrap[:4],
                }
            )
            return out
        out["reason"] = "projection calibration unavailable"
        return out

    aligned: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    considered: list[dict[str, Any]] = []
    for signal in projection_signals:
        name = str(getattr(signal, "name", "") or "")
        signal_direction = str(getattr(signal, "direction", "") or "").upper()
        if not name or signal_direction not in {direction, "EITHER"}:
            continue
        if not _projection_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        ):
            continue
        confidence = _optional_float(getattr(signal, "confidence", None)) or 0.0
        bonus = _optional_float(getattr(signal, "bonus_magnitude", None)) or 0.0
        if confidence < FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE:
            continue
        if signal_direction == "EITHER" and bonus <= 0.0:
            continue
        calibration_name = select_calibration_signal_name(
            name,
            signal_direction,
            pair,
            hit_rates=hit_rates,
            regime=regime,
        )
        bucket = _projection_hit_rate_bucket(
            hit_rates,
            calibration_name,
            pair=pair,
            regime=regime,
        )
        if bucket is None:
            continue
        hit_rate = _optional_float(bucket.get("hit_rate"))
        samples = _optional_int(bucket.get("samples"))
        if hit_rate is None or samples is None:
            continue
        item = {
            "name": name,
            "calibration_name": calibration_name,
            "direction": signal_direction,
            "confidence": round(confidence, 4),
            "hit_rate": round(hit_rate, 4),
            "samples": samples,
            "timeframe": getattr(signal, "timeframe", None),
            "lead_time_min": _projection_signal_lead_time_payload(signal),
            "rationale": str(getattr(signal, "rationale", "") or "")[:180],
        }
        considered.append(item)
        if samples < FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
            continue
        if signal_direction == direction and hit_rate >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE:
            aligned.append(item)
        elif signal_direction == "EITHER" and hit_rate >= FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE:
            timing.append(item)

    aligned = _dedupe_forecast_projection_support(aligned)
    timing = _dedupe_forecast_projection_support(timing)
    considered = _dedupe_forecast_projection_support(considered)
    entry_min = _forecast_seed_min_confidence_for_direction(direction)
    timing_with_raw_direction = bool(timing) and raw_confidence is not None and raw_confidence >= entry_min
    ok = bool(aligned) or timing_with_raw_direction or bool(bootstrap)
    display_support = (aligned + timing) if (aligned or timing) else bootstrap
    best_aligned = aligned[0] if aligned else None
    best_timing = timing[0] if timing else None
    best = best_aligned or (best_timing if timing_with_raw_direction else None)
    unselected = _forecast_unselected_projection_support(
        pair=pair,
        forecast_direction=direction,
        projection_signals=projection_signals,
        hit_rates=hit_rates,
        regime=regime,
        forecast_horizon_min=forecast_horizon_min,
    )
    out.update(
        {
            "ok": ok,
            "aligned_projection_count": len(aligned) + (len(bootstrap) if not aligned else 0),
            "timing_projection_count": len(timing),
            "unselected_projection_count": len(unselected),
            "best_hit_rate": best["hit_rate"] if best else None,
            "best_samples": best["samples"] if best else 0,
            "best_aligned_hit_rate": best_aligned["hit_rate"] if best_aligned else None,
            "best_aligned_samples": best_aligned["samples"] if best_aligned else 0,
            "best_timing_hit_rate": best_timing["hit_rate"] if best_timing else None,
            "best_timing_samples": best_timing["samples"] if best_timing else 0,
            "best_unselected_hit_rate": unselected[0]["hit_rate"] if unselected else None,
            "best_unselected_samples": unselected[0]["samples"] if unselected else 0,
            "bootstrap_projection_support": bool(bootstrap) and not bool(aligned),
            "signals": display_support[:4] if ok else (considered + bootstrap)[:4],
            "unselected_signals": unselected[:4],
        }
    )
    if best and best["direction"] == direction:
        top = best
        out["reason"] = (
            f"{top['name']} {top['direction']} hit_rate={top['hit_rate']:.2f} "
            f"samples={top['samples']} supports weak calibrated forecast"
        )
    elif best and best["direction"] == "EITHER" and timing_with_raw_direction:
        top = best
        out["reason"] = (
            f"{top['name']} timing hit_rate={top['hit_rate']:.2f} samples={top['samples']} "
            f"supports raw forecast confidence {raw_confidence:.2f}"
        )
    elif bootstrap:
        out["reason"] = bootstrap[0]["reason"]
    else:
        out["reason"] = "no current projection clears audited support floors"
    if unselected:
        top = unselected[0]
        out["unselected_reason"] = (
            f"{top['name']} {top['direction']} audited hit_rate={top['hit_rate']:.2f} "
            f"samples={top['samples']} did not align with forecast={direction}"
        )
    return out


def _forecast_directional_calibration_for_forecast(
    *,
    pair: str,
    direction: str,
    hit_rates: dict[str, dict[str, Any]] | None,
    regime: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "directional_calibration_name": None,
        "directional_hit_rate": None,
        "directional_samples": 0,
    }
    direction = str(direction or "").upper()
    if direction not in {"UP", "DOWN"} or not isinstance(hit_rates, dict):
        return payload
    try:
        from quant_rabbit.strategy.projection_ledger import select_calibration_signal_name
    except Exception:
        return payload
    calibration_name = select_calibration_signal_name(
        "directional_forecast",
        direction,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    )
    bucket = _projection_hit_rate_bucket(
        hit_rates,
        calibration_name,
        pair=pair,
        regime=regime,
    )
    if bucket is None:
        return payload
    hit_rate = _optional_float(bucket.get("hit_rate"))
    samples = _optional_int(bucket.get("samples")) or 0
    if hit_rate is None:
        return payload
    payload.update(
        {
            "directional_calibration_name": calibration_name,
            "directional_hit_rate": round(hit_rate, 4),
            "directional_samples": samples,
        }
    )
    return payload


def _forecast_unselected_projection_support(
    *,
    pair: str,
    forecast_direction: str,
    projection_signals: list[Any],
    hit_rates: dict[str, dict[str, Any]] | None,
    regime: str | None,
    forecast_horizon_min: float | None = None,
) -> list[dict[str, Any]]:
    if not projection_signals or not isinstance(hit_rates, dict):
        return []
    try:
        from quant_rabbit.strategy.projection_ledger import select_calibration_signal_name
    except Exception:
        return []

    forecast_direction = str(forecast_direction or "").upper()
    out: list[dict[str, Any]] = []
    for signal in projection_signals:
        name = str(getattr(signal, "name", "") or "")
        signal_direction = str(getattr(signal, "direction", "") or "").upper()
        if not name or signal_direction not in {"UP", "DOWN"}:
            continue
        if not _projection_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        ):
            continue
        if forecast_direction in {"UP", "DOWN"} and signal_direction == forecast_direction:
            continue
        confidence = _optional_float(getattr(signal, "confidence", None)) or 0.0
        bonus = _optional_float(getattr(signal, "bonus_magnitude", None)) or 0.0
        if confidence < FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE or bonus <= 0.0:
            continue
        calibration_name = select_calibration_signal_name(
            name,
            signal_direction,
            pair,
            hit_rates=hit_rates,
            regime=regime,
        )
        bucket = _projection_hit_rate_bucket(
            hit_rates,
            calibration_name,
            pair=pair,
            regime=regime,
        )
        if bucket is None:
            continue
        hit_rate = _optional_float(bucket.get("hit_rate"))
        samples = _optional_int(bucket.get("samples"))
        if hit_rate is None or samples is None:
            continue
        if samples < FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
            continue
        if hit_rate < FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE:
            continue
        out.append(
            {
                "name": name,
                "calibration_name": calibration_name,
                "direction": signal_direction,
                "confidence": round(confidence, 4),
                "hit_rate": round(hit_rate, 4),
                "samples": samples,
                "timeframe": getattr(signal, "timeframe", None),
                "lead_time_min": _projection_signal_lead_time_payload(signal),
                "rationale": str(getattr(signal, "rationale", "") or "")[:180],
            }
        )
    return _dedupe_forecast_projection_support(out)


def _forecast_bootstrap_projection_support(
    *,
    forecast: Any,
    direction: str,
    projection_signals: list[Any],
    hit_rates: dict[str, dict[str, Any]] | None = None,
    pair: str | None = None,
    regime: str | None = None,
) -> list[dict[str, Any]]:
    if direction not in {"UP", "DOWN"}:
        return []
    confidence = _optional_float(getattr(forecast, "confidence", None))
    raw_confidence = _optional_float(getattr(forecast, "raw_confidence", None))
    if raw_confidence is None:
        raw_confidence = confidence
    entry_min = _forecast_seed_min_confidence_for_direction(direction)
    support_floor = max(0.0, entry_min - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL)
    if confidence is None or confidence < support_floor:
        return []
    if raw_confidence is None or raw_confidence < max(entry_min, FORECAST_BOOTSTRAP_RAW_CONFIDENCE_MIN):
        return []
    out: list[dict[str, Any]] = []
    for signal in projection_signals:
        signal_direction = str(getattr(signal, "direction", "") or "").upper()
        if signal_direction != direction:
            continue
        if not _projection_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=_optional_float(getattr(forecast, "horizon_min", None)),
        ):
            continue
        confidence_value = _optional_float(getattr(signal, "confidence", None)) or 0.0
        bonus = _optional_float(getattr(signal, "bonus_magnitude", None)) or 0.0
        if confidence_value < FORECAST_BOOTSTRAP_SIGNAL_CONFIDENCE_MIN or bonus <= 0.0:
            continue
        name = str(getattr(signal, "name", "") or "")
        if not name:
            continue
        calibration_name, audit_bucket = _forecast_bootstrap_audit_bucket(
            name=name,
            signal_direction=signal_direction,
            pair=pair,
            regime=regime,
            hit_rates=hit_rates,
        )
        audit_hit_rate = _optional_float(audit_bucket.get("hit_rate")) if audit_bucket else None
        audit_samples = _optional_int(audit_bucket.get("samples")) if audit_bucket else 0
        audit_samples = audit_samples or 0
        if (
            audit_hit_rate is not None
            and audit_samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
            and audit_hit_rate < FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE
        ):
            continue
        if audit_hit_rate is not None and audit_samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
            audit_text = f"audited hit_rate={audit_hit_rate:.2f} samples={audit_samples}"
        elif audit_samples > 0:
            audit_text = (
                f"ledger samples thin ({audit_samples}<{FORECAST_MARKET_SUPPORT_MIN_SAMPLES}); "
                "bootstrap pending"
            )
        else:
            audit_text = "ledger samples pending"
        out.append(
            {
                "name": name,
                "calibration_name": calibration_name,
                "direction": signal_direction,
                "confidence": round(confidence_value, 4),
                "hit_rate": round(audit_hit_rate, 4) if audit_hit_rate is not None else None,
                "samples": audit_samples,
                "bootstrap_projection_support": True,
                "timeframe": getattr(signal, "timeframe", None),
                "lead_time_min": _projection_signal_lead_time_payload(signal),
                "rationale": str(getattr(signal, "rationale", "") or "")[:180],
                "reason": (
                    f"{name} {signal_direction} same-cycle bootstrap: signal_conf={confidence_value:.2f}, "
                    f"raw_forecast_conf={raw_confidence:.2f}, calibrated_conf={confidence:.2f}; "
                    f"{audit_text}"
                ),
            }
        )
    return sorted(out, key=lambda item: item["confidence"], reverse=True)


def _projection_signal_lead_time_payload(signal: Any) -> float | None:
    lead_time = _optional_float(getattr(signal, "lead_time_min", None))
    return round(lead_time, 4) if lead_time is not None else None


def _projection_signal_within_forecast_horizon(
    signal: Any,
    *,
    forecast_horizon_min: float | None,
) -> bool:
    """Return whether a projection can support this forecast's time scope.

    A projection whose expected move starts after the forecast horizon is
    useful context, but it is not current live-entry support. This prevents a
    multi-day macro nowcast from rescuing a weak one-hour entry thesis.
    """
    if forecast_horizon_min is None or forecast_horizon_min <= 0:
        return True
    lead_time = _optional_float(getattr(signal, "lead_time_min", None))
    if lead_time is None:
        return True
    return max(0.0, lead_time) <= forecast_horizon_min


def _forecast_bootstrap_audit_bucket(
    *,
    name: str,
    signal_direction: str,
    pair: str | None,
    regime: str | None,
    hit_rates: dict[str, dict[str, Any]] | None,
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(hit_rates, dict) or not pair:
        return name, None
    try:
        from quant_rabbit.strategy.projection_ledger import select_calibration_signal_name
    except Exception:
        return name, None
    calibration_name = select_calibration_signal_name(
        name,
        signal_direction,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    )
    return calibration_name, _projection_hit_rate_bucket(
        hit_rates,
        calibration_name,
        pair=pair,
        regime=regime,
    )


def _projection_hit_rate_bucket(
    hit_rates: dict[str, dict[str, Any]],
    signal_name: str,
    *,
    pair: str,
    regime: str | None,
) -> dict[str, Any] | None:
    by_key = hit_rates.get(signal_name)
    if not isinstance(by_key, dict):
        return None
    keys: list[str] = []
    if regime is not None:
        keys.append(f"{pair}:{regime}")
    keys.append(f"{pair}:_all_regimes")
    if regime is not None:
        keys.append(f"_all_pairs:{regime}")
    keys.extend(["_all_pairs:_all_regimes", pair, "_all_pairs"])
    first_bucket: dict[str, Any] | None = None
    for key in keys:
        bucket = by_key.get(key)
        if isinstance(bucket, dict):
            if first_bucket is None:
                first_bucket = bucket
            samples = _optional_int(bucket.get("samples")) or 0
            if samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
                return bucket
    return first_bucket


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed


def _forecast_seed_methods(
    pair: str,
    forecast: Any,
    charts: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    direction = str(getattr(forecast, "direction", "") or "").upper()
    if direction == "RANGE":
        if _pair_has_current_range_rotation_edge(pair, charts):
            return FORECAST_SEED_RANGE_METHODS
        return FORECAST_SEED_RANGE_METHODS if _forecast_has_range_box(forecast) else ()
    if direction not in {"UP", "DOWN"}:
        return ()
    methods: list[str] = []
    if _pair_has_current_range_rotation_edge(pair, charts):
        methods.append(TradeMethod.RANGE_ROTATION.value)
    methods.extend(FORECAST_SEED_DIRECTIONAL_METHODS)
    return tuple(dict.fromkeys(methods))


def _forecast_has_range_box(forecast: Any) -> bool:
    low = _optional_float(getattr(forecast, "range_low_price", None))
    high = _optional_float(getattr(forecast, "range_high_price", None))
    return low is not None and high is not None and high > low


def _range_forecast_geometry_watch_reason(
    pair: str,
    forecast: Any,
    charts: dict[str, dict[str, Any]],
) -> str | None:
    if str(getattr(forecast, "direction", "") or "").upper() != "RANGE":
        return None
    if not _forecast_has_range_box(forecast):
        return None
    if _pair_has_current_range_rotation_edge(pair, charts):
        return None
    return (
        "RANGE forecast has a measured rail box, but the current range-rotation edge is not confirmed; "
        "expose RANGE_ROTATION dry-run geometry only until the chart phase clears the auto-lane gate."
    )


def _forecast_watch_candidate_reason(
    pair: str,
    forecast: Any,
    charts: dict[str, dict[str, Any]],
    *,
    source_by_pair: dict[str, dict[str, Any]],
    min_confidence: float,
) -> str | None:
    if pair in source_by_pair:
        return None
    direction = str(getattr(forecast, "direction", "") or "").upper()
    if direction not in {"UP", "DOWN", "RANGE"}:
        return None
    confidence = _optional_float(getattr(forecast, "confidence", None))
    raw_confidence = _optional_float(getattr(forecast, "raw_confidence", None))
    if raw_confidence is None:
        raw_confidence = confidence
    chart_score = _forecast_watch_chart_score(pair, direction, charts)
    strong_raw = raw_confidence is not None and raw_confidence >= FORECAST_WATCH_MIN_RAW_CONFIDENCE
    strong_chart = chart_score is not None and chart_score >= FORECAST_WATCH_MIN_CHART_SCORE
    if not strong_raw and not strong_chart:
        return None
    bits = [
        f"calibrated confidence {0.0 if confidence is None else confidence:.2f} < live floor {min_confidence:.2f}",
    ]
    if strong_raw:
        bits.append(f"raw forecast confidence {raw_confidence:.2f}")
    if strong_chart:
        bits.append(f"chart {direction} score {chart_score:.2f}")
    return "forecast watch-only candidate: " + ", ".join(bits)


def _forecast_watch_chart_score(
    pair: str,
    direction: str,
    charts: dict[str, dict[str, Any]],
) -> float | None:
    per_tf = charts.get(pair)
    raw_chart = per_tf.get("__raw_chart") if isinstance(per_tf, dict) else None
    if not isinstance(raw_chart, dict):
        return None
    if direction == "UP":
        return _optional_float(raw_chart.get("long_score"))
    if direction == "DOWN":
        return _optional_float(raw_chart.get("short_score"))
    if direction == "RANGE":
        long_score = _optional_float(raw_chart.get("long_score"))
        short_score = _optional_float(raw_chart.get("short_score"))
        if long_score is None and short_score is None:
            return None
        return max(value for value in (long_score, short_score) if value is not None)
    return None


def _range_seed_direction(pair: str, charts: dict[str, dict[str, Any]], current_price: float) -> str | None:
    indicators = _range_indicators_for(pair, charts)
    if not indicators:
        return None
    support = _nearest_below(current_price, _numeric_levels(indicators, RANGE_SUPPORT_LEVEL_KEYS))
    resistance = _nearest_above(current_price, _numeric_levels(indicators, RANGE_RESISTANCE_LEVEL_KEYS))
    if support is None or resistance is None or resistance <= support:
        return None
    midpoint = support + ((resistance - support) / 2.0)
    return Side.LONG.value if current_price <= midpoint else Side.SHORT.value


def _forecast_seed_lane(
    source: dict[str, Any] | None,
    *,
    pair: str,
    side: str,
    method: str,
    forecast: Any,
    cycle_id: str | None,
) -> dict[str, Any]:
    lane = dict(source or {})
    confidence = _optional_float(getattr(forecast, "confidence", None)) or 0.0
    target = getattr(forecast, "target_price", None)
    invalidation = getattr(forecast, "invalidation_price", None)
    rationale = str(getattr(forecast, "rationale_summary", "") or "")
    drivers_for = [str(item) for item in list(getattr(forecast, "drivers_for", ()) or ())[:3]]
    drivers_against = [str(item) for item in list(getattr(forecast, "drivers_against", ()) or ())[:3]]
    base_reason = str(lane.get("reason") or "forecast-first candidate discovery")
    trigger_only = _lane_forbids_market_chase(lane)
    adoption = "TRIGGER_RECEIPT_REQUIRED" if trigger_only else "ORDER_INTENT_REQUIRED"
    required_receipt = (
        "Forecast-first lane: use only a pending trigger receipt; no market chase. "
        "The same fresh forecast, market-location map, levels, spread, and risk geometry must still agree."
        if trigger_only
        else (
            "Forecast-first lane: create an intent only if this same fresh forecast, "
            "market-location map, levels, spread, and risk geometry still agree."
        )
    )
    lane.update(
        {
            "desk": FORECAST_SEED_DESK_BY_METHOD[method],
            "pair": pair,
            "direction": side,
            "method": method,
            "adoption": adoption,
            "campaign_role": "FORECAST_FIRST",
            "reason": (
                f"{base_reason}; forecast-first seed {getattr(forecast, 'direction', None)} "
                f"conf={confidence:.2f}: {rationale}"
            ),
            "required_receipt": required_receipt,
            "target_reward_risk": max(_optional_float(lane.get("target_reward_risk")) or 0.0, DYNAMIC_RR_BASE),
            "blockers": list(lane.get("blockers") or []),
            "story_examples": drivers_for or list(lane.get("story_examples") or [])[:2],
            "forecast_seed": True,
            **_forecast_context_payload(forecast, cycle_id=cycle_id),
        }
    )
    return lane


def _lane_with_forecast_context(
    lane: dict[str, Any],
    forecast: Any | None,
    *,
    cycle_id: str | None,
) -> dict[str, Any]:
    if forecast is None:
        return lane
    out = dict(lane)
    out.update(_forecast_context_payload(forecast, cycle_id=cycle_id))
    return out


def _forecast_context_payload(forecast: Any, *, cycle_id: str | None = None) -> dict[str, Any]:
    confidence = _optional_float(getattr(forecast, "confidence", None)) or 0.0
    market_support = _forecast_market_support_payload(getattr(forecast, "market_support", None))
    component_scores = getattr(forecast, "component_scores", None)
    if not isinstance(component_scores, dict):
        component_scores = {}
    payload = {
        "forecast_cycle_id": cycle_id,
        "forecast_direction": str(getattr(forecast, "direction", "") or ""),
        "forecast_confidence": round(confidence, 4),
        "forecast_raw_confidence": getattr(forecast, "raw_confidence", None),
        "forecast_calibration_multiplier": getattr(forecast, "calibration_multiplier", None),
        "forecast_current_price": getattr(forecast, "current_price", None),
        "forecast_target_price": getattr(forecast, "target_price", None),
        "forecast_invalidation_price": getattr(forecast, "invalidation_price", None),
        "forecast_range_low_price": getattr(forecast, "range_low_price", None),
        "forecast_range_high_price": getattr(forecast, "range_high_price", None),
        "forecast_range_width_pips": getattr(forecast, "range_width_pips", None),
        "forecast_horizon_min": getattr(forecast, "horizon_min", None),
        "forecast_rationale": str(getattr(forecast, "rationale_summary", "") or ""),
        "forecast_drivers_for": [str(item) for item in list(getattr(forecast, "drivers_for", ()) or ())[:3]],
        "forecast_drivers_against": [str(item) for item in list(getattr(forecast, "drivers_against", ()) or ())[:3]],
        "forecast_component_scores": {
            str(key): round(float(value), 4)
            for key, value in component_scores.items()
            if _optional_float(value) is not None
        },
        "forecast_market_support": market_support,
        "forecast_market_support_ok": bool(market_support.get("ok")),
        "forecast_market_support_reason": market_support.get("reason"),
        "forecast_directional_calibration_name": market_support.get("directional_calibration_name"),
        "forecast_directional_hit_rate": market_support.get("directional_hit_rate"),
        "forecast_directional_samples": market_support.get("directional_samples"),
    }
    payload.update(_forecast_news_ref_metadata(forecast, market_support))
    return payload


def _forecast_news_ref_metadata(forecast: Any, market_support: dict[str, Any]) -> dict[str, Any]:
    signal_names: list[str] = []
    for raw in list(market_support.get("signals") or []) + list(market_support.get("unselected_signals") or []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        name_key = name.lower()
        if "news" not in name_key and not name_key.startswith(("macro_event_nowcast", "us_employment_nowcast")):
            continue
        if name not in signal_names:
            signal_names.append(name)

    forecast_text = " ".join(
        [
            str(getattr(forecast, "rationale_summary", "") or ""),
            " ".join(str(item) for item in list(getattr(forecast, "drivers_for", ()) or ())[:3]),
            str(market_support.get("reason") or ""),
        ]
    ).lower()
    if not signal_names and ("news" in forecast_text or "headline" in forecast_text):
        signal_names.append("forecast_news_context")
    if not signal_names:
        return {}
    return {
        "news_refs": ["news:digest", "news:items"],
        "news_digest_ref": "news:digest",
        "news_signal_names": signal_names[:4],
    }


def _forecast_market_support_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "ok": bool(value.get("ok")),
        "direction": str(value.get("direction") or ""),
        "aligned_projection_count": _optional_int(value.get("aligned_projection_count")) or 0,
        "timing_projection_count": _optional_int(value.get("timing_projection_count")) or 0,
        "unselected_projection_count": _optional_int(value.get("unselected_projection_count")) or 0,
        "best_hit_rate": _optional_float(value.get("best_hit_rate")),
        "best_samples": _optional_int(value.get("best_samples")) or 0,
        "best_aligned_hit_rate": _optional_float(value.get("best_aligned_hit_rate")),
        "best_aligned_samples": _optional_int(value.get("best_aligned_samples")) or 0,
        "best_timing_hit_rate": _optional_float(value.get("best_timing_hit_rate")),
        "best_timing_samples": _optional_int(value.get("best_timing_samples")) or 0,
        "best_unselected_hit_rate": _optional_float(value.get("best_unselected_hit_rate")),
        "best_unselected_samples": _optional_int(value.get("best_unselected_samples")) or 0,
        "directional_calibration_name": (
            str(value.get("directional_calibration_name") or "")
            if value.get("directional_calibration_name") is not None
            else None
        ),
        "directional_hit_rate": _optional_float(value.get("directional_hit_rate")),
        "directional_samples": _optional_int(value.get("directional_samples")) or 0,
        "bootstrap_projection_support": bool(value.get("bootstrap_projection_support")),
        "reason": str(value.get("reason") or ""),
        "unselected_reason": str(value.get("unselected_reason") or ""),
        "signals": [
            item for item in list(value.get("signals") or [])[:4]
            if isinstance(item, dict)
        ],
        "unselected_signals": [
            item for item in list(value.get("unselected_signals") or [])[:4]
            if isinstance(item, dict)
        ],
    }


def _pair_has_current_range_rotation_edge(
    pair: str,
    charts: dict[str, dict[str, Any]],
    *,
    phase: str | None = None,
) -> bool:
    per_tf = charts.get(pair)
    if not per_tf:
        return False
    indicators = _range_indicators_for(pair, charts)
    if not indicators or not _has_range_rails(indicators):
        return False

    evidence = 0.0
    dominant = str(per_tf.get("dominant_regime") or "").upper()
    if "BREAKOUT" in dominant or "SQUEEZE" in dominant:
        return False
    if phase in {"BREAKOUT_PENDING", "BREAKOUT_UP", "BREAKOUT_DOWN"}:
        return False
    if phase in {"IN_RANGE", "RANGE_FORMING"}:
        evidence += 1.0
    if "RANGE" in dominant:
        evidence += 1.0

    for timeframe in RANGE_AUTOLANE_TIMEFRAMES:
        state = _regime_reading_state(per_tf, timeframe)
        if state == "BREAKOUT_PENDING":
            return False
        regime = str(per_tf.get(f"{timeframe}__regime") or "").upper()
        tf_indicators = per_tf.get(timeframe) if isinstance(per_tf.get(timeframe), dict) else {}
        if "BREAKOUT_PENDING" in regime or "SQUEEZE" in regime:
            return False
        if _has_squeeze_breakout_risk(tf_indicators, per_tf, timeframe):
            return False
        if state == "RANGE" or "RANGE" in regime:
            evidence += 1.0
        adx = _optional_float((tf_indicators or {}).get("adx_14") or (tf_indicators or {}).get("adx"))
        chop = _optional_float((tf_indicators or {}).get("choppiness_14"))
        if adx is not None and chop is not None and adx < RANGE_AUTOLANE_ADX_MAX and chop > RANGE_AUTOLANE_CHOP_MIN:
            evidence += 0.75
        if (
            timeframe == GEOMETRY_ATR_TIMEFRAME
            and "RANGE" in regime
            and str((tf_indicators or {}).get("regime_quantile") or "").upper() == "QUIET"
        ):
            evidence += 0.5
    return evidence >= RANGE_AUTOLANE_MIN_EVIDENCE


def _current_range_phase(pair: str, charts: dict[str, dict[str, Any]] | None) -> tuple[str, str | None]:
    if charts is None:
        return "NONE", None
    per_tf = charts.get(pair)
    if not per_tf:
        return "NONE", None
    raw_chart = per_tf.get("__raw_chart")
    if isinstance(raw_chart, dict):
        try:
            from quant_rabbit.strategy.directional_forecaster import _range_phase_analysis

            phase = _range_phase_analysis(raw_chart)
            phase_name = str(getattr(phase, "phase", "") or "NONE").upper()
            direction = getattr(phase, "direction", None)
            direction_name = str(direction).upper() if direction is not None else None
            if phase_name != "NONE":
                return phase_name, direction_name
        except Exception:
            pass
    return _fallback_range_phase_from_indexed_chart(per_tf)


def _fallback_range_phase_from_indexed_chart(per_tf: dict[str, Any]) -> tuple[str, str | None]:
    has_stable_range = False
    has_forming_range = False
    dominant = str(per_tf.get("dominant_regime") or "").upper()
    for timeframe in RANGE_AUTOLANE_TIMEFRAMES:
        state = _regime_reading_state(per_tf, timeframe)
        regime = str(per_tf.get(f"{timeframe}__regime") or "").upper()
        indicators = per_tf.get(timeframe) if isinstance(per_tf.get(timeframe), dict) else {}
        if state == "BREAKOUT_PENDING" or "BREAKOUT_PENDING" in regime or _has_squeeze_breakout_risk(indicators, per_tf, timeframe):
            return "BREAKOUT_PENDING", None
        if state == "RANGE" or "RANGE" in regime:
            has_stable_range = True
        if state in {"TREND_WEAK", "TRANSITION"} or regime in {"TREND_WEAK", "TRANSITION"}:
            adx = _optional_float((indicators or {}).get("adx_14") or (indicators or {}).get("adx"))
            chop = _optional_float((indicators or {}).get("choppiness_14"))
            bb_width_pct = _percent_0_100((indicators or {}).get("bb_width_percentile_100"))
            if (
                (adx is not None and adx < RANGE_AUTOLANE_FORMING_ADX_MAX)
                or (chop is not None and chop >= RANGE_AUTOLANE_FORMING_CHOP_MIN)
                or (bb_width_pct is not None and bb_width_pct <= RANGE_AUTOLANE_FORMING_BB_WIDTH_PCT_MAX)
            ):
                has_forming_range = True
    if has_stable_range:
        return "IN_RANGE", None
    if dominant.startswith("TREND_") or dominant.startswith("IMPULSE_"):
        return "NONE", None
    if has_forming_range:
        return "RANGE_FORMING", None
    return "NONE", None


def _has_range_rails(indicators: dict[str, Any]) -> bool:
    support = any(_optional_float(indicators.get(key)) is not None for key in RANGE_SUPPORT_LEVEL_KEYS)
    resistance = any(_optional_float(indicators.get(key)) is not None for key in RANGE_RESISTANCE_LEVEL_KEYS)
    return support and resistance


def _regime_reading_state(per_tf: dict[str, Any], timeframe: str) -> str:
    reading = per_tf.get(f"{timeframe}__regime_reading")
    if not isinstance(reading, dict):
        return ""
    return str(reading.get("state") or "").upper()


def _has_squeeze_breakout_risk(indicators: dict[str, Any], per_tf: dict[str, Any], timeframe: str) -> bool:
    if not _truthy_flag((indicators or {}).get("bb_squeeze") or (indicators or {}).get("squeeze")):
        return False
    reading = per_tf.get(f"{timeframe}__regime_reading")
    if not isinstance(reading, dict):
        reading = {}
    atr_pct = _percent_0_100(reading.get("atr_percentile"))
    if atr_pct is None:
        atr_pct = _percent_0_100((indicators or {}).get("atr_percentile_100"))
    bb_width_pct = _percent_0_100((indicators or {}).get("bb_width_percentile_100"))
    return (
        (atr_pct is not None and atr_pct <= RANGE_AUTOLANE_SQUEEZE_PCT_MAX)
        or (bb_width_pct is not None and bb_width_pct <= RANGE_AUTOLANE_SQUEEZE_PCT_MAX)
    )


def _percent_0_100(value: object) -> float | None:
    pct = _optional_float(value)
    if pct is None:
        return None
    return pct * 100.0 if 0.0 <= pct <= 1.0 else pct


def _truthy_flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    numeric = _optional_float(value)
    return numeric is not None and numeric > 0.0


def _pair_chart_for(pair: str, charts: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    raw = per_tf.get("__raw_chart")
    return raw if isinstance(raw, dict) else None


def _execution_regime_state(
    method: TradeMethod,
    dominant_regime: str | None,
    chart_context: dict[str, Any] | None,
) -> str | None:
    """Return the regime state that should drive executable geometry.

    Dominant multi-timeframe regime remains recorded, but RANGE_ROTATION is an
    M5 operating tactic. When M5 is explicitly RANGE/QUIET, use RANGE for the
    executable RR floor even if higher timeframes make the aggregate dominant
    regime look UNCLEAR.
    """
    if method == TradeMethod.RANGE_ROTATION and _is_low_vol_range_context(chart_context):
        return "RANGE"
    return dominant_regime


def _effective_stop_widening_multiplier(
    method: TradeMethod,
    regime_reading: dict[str, Any] | None,
    chart_context: dict[str, Any] | None,
) -> float:
    if method == TradeMethod.RANGE_ROTATION and _is_low_vol_range_context(chart_context):
        return 1.0
    return _regime_stop_widening_multiplier(regime_reading)


def _execution_target_reward_risk(
    base_reward_risk: float,
    method: TradeMethod,
    order_type: OrderType,
    execution_regime: str | None,
    chart_context: dict[str, Any] | None,
    side: Side,
) -> float:
    # AGENT_CONTRACT §3.5: reward_risk must be regime-derived. The
    # `base_reward_risk` (from lane definition or legacy 1.5 fallback) is
    # treated as a SUGGESTION; the live market-derived value from
    # `_market_derived_reward_risk` is used as the floor so static
    # base values cannot reduce a dynamic-high regime to a tight TP.
    market_rr, _rationale = _market_derived_reward_risk(chart_context)
    blended_rr = max(base_reward_risk, market_rr)
    target_reward_risk = round(blended_rr * _regime_reward_risk_multiplier(execution_regime), 2)
    if (
        method == TradeMethod.RANGE_ROTATION
        and order_type == OrderType.MARKET
        and _is_low_vol_directional_range(side, chart_context)
    ):
        target_reward_risk = min(target_reward_risk, RANGE_DIRECTIONAL_MARKET_TARGET_RR_CAP)
    return round(target_reward_risk, 2)


def _is_low_vol_range_context(chart_context: dict[str, Any] | None) -> bool:
    if not chart_context:
        return False
    return str(chart_context.get("m5_regime") or "").upper() == "RANGE" and str(
        chart_context.get("m5_regime_quantile") or ""
    ).upper() == "QUIET"


def _is_low_vol_directional_range(side: Side, chart_context: dict[str, Any] | None) -> bool:
    if not _is_low_vol_range_context(chart_context):
        return False
    if _direction_bias_from_m5(chart_context) != side.value:
        return False
    position = _m5_range_position(chart_context)
    if position is None:
        return False
    edge = max(0.05, min(0.45, RANGE_DIRECTIONAL_MARKET_EDGE_POSITION))
    if side == Side.LONG:
        return position <= edge
    return position >= 1.0 - edge


def _m5_range_position(chart_context: dict[str, Any] | None) -> float | None:
    if not chart_context:
        return None
    tf_map = chart_context.get("tf_regime_map")
    if not isinstance(tf_map, dict):
        return None
    m5 = tf_map.get("M5")
    if not isinstance(m5, dict):
        return None
    return _optional_float(m5.get("range_position"))


def _direction_bias_from_m5(chart_context: dict[str, Any] | None) -> str | None:
    if not chart_context:
        return None
    long_bias = _optional_float(chart_context.get("m5_long_bias"))
    short_bias = _optional_float(chart_context.get("m5_short_bias"))
    if (
        long_bias is None
        or short_bias is None
        or abs(long_bias - short_bias) <= CHART_DIRECTION_TIED_GAP_BOUNDARY
    ):
        return None
    return Side.LONG.value if long_bias > short_bias else Side.SHORT.value


def _operating_tf_opposes_side(side: Side, chart_context: dict[str, Any] | None) -> bool:
    """Return true when M5/M15 structure says the hedge side is a chase.

    Recovery hedges can monetize trapped exposure, but a higher-timeframe
    BOS in the hedge direction is not enough when the operating tape has just
    printed the opposite CHoCH/BOS with a clear M5/M15 bias. In that case the
    hedge is continuation/chase until price proves through a trigger.
    """
    if not chart_context:
        return False
    story = str(chart_context.get("chart_story_structural") or "")
    for timeframe in ("M5", "M15"):
        struct_dir = _tf_structure_direction(story, timeframe)
        if struct_dir is None or _direction_to_side(struct_dir) == side:
            continue
        long_bias = _optional_float(chart_context.get(f"{timeframe.lower()}_long_bias"))
        short_bias = _optional_float(chart_context.get(f"{timeframe.lower()}_short_bias"))
        if long_bias is None or short_bias is None:
            continue
        if side == Side.SHORT and long_bias - short_bias > CHART_DIRECTION_TIED_GAP_BOUNDARY:
            return True
        if side == Side.LONG and short_bias - long_bias > CHART_DIRECTION_TIED_GAP_BOUNDARY:
            return True
    return False


def _operating_tf_confirms_side(side: Side, chart_context: dict[str, Any] | None) -> bool:
    """Return true only when M5/M15 has a close-confirmed side break.

    Wick-only sweeps are excluded by `_tf_structure_direction`; this is the
    executable "抜けた" proof that lets a MARKET/STOP continuation override
    opposed reversal candle evidence.
    """
    if not chart_context:
        return False
    story = str(chart_context.get("chart_story_structural") or "")
    for timeframe in ("M5", "M15"):
        struct_dir = _tf_structure_direction(story, timeframe)
        if struct_dir is None or _direction_to_side(struct_dir) != side:
            continue
        long_bias = _optional_float(chart_context.get(f"{timeframe.lower()}_long_bias"))
        short_bias = _optional_float(chart_context.get(f"{timeframe.lower()}_short_bias"))
        if long_bias is None or short_bias is None:
            return True
        if side == Side.LONG and long_bias >= short_bias:
            return True
        if side == Side.SHORT and short_bias >= long_bias:
            return True
    return False


def _tf_structure_direction(chart_story: str, timeframe: str) -> str | None:
    text = chart_story.upper()
    marker = f"{timeframe.upper()}("
    start = text.find(marker)
    if start < 0:
        return None
    end = text.find(");", start)
    if end < 0:
        end = text.find(")", start)
    segment = text[start : end if end >= 0 else len(text)]
    for token, direction in (
        ("BOS_UP", "UP"),
        ("CHOCH_UP", "UP"),
        ("BOS_DOWN", "DOWN"),
        ("CHOCH_DOWN", "DOWN"),
    ):
        token_at = segment.find(token)
        if token_at < 0:
            continue
        if ":WICK" in segment[token_at : token_at + 48]:
            continue
        return direction
    return None


def _direction_to_side(direction: str) -> Side | None:
    upper = direction.upper()
    if upper == "UP":
        return Side.LONG
    if upper == "DOWN":
        return Side.SHORT
    return None


PIP_FACTORS = {pair: instrument_pip_factor(pair) for pair in DEFAULT_TRADER_PAIRS}


@dataclass(frozen=True)
class GeneratedIntent:
    lane_id: str
    status: str
    intent: dict[str, Any] | None
    risk_metrics: dict[str, Any] | None
    risk_allowed: bool | None
    risk_issues: tuple[dict[str, Any], ...]
    strategy_issues: tuple[dict[str, Any], ...]
    live_strategy_issues: tuple[dict[str, Any], ...]
    live_blockers: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class IntentGenerationSummary:
    output_path: Path
    report_path: Path
    candidates_seen: int
    generated: int
    needs_snapshot: int
    dry_run_passed: int
    live_ready: int


@dataclass(frozen=True)
class _TelemetryLiveReadinessCache:
    latest_forecasts_by_pair: dict[str, dict[str, Any]]
    forecasts_by_pair_cycle: dict[tuple[str, str], dict[str, Any]]
    directional_projection_keys: set[tuple[str, str]]
    projection_signal_keys: set[tuple[str, str, str]]
    expired_pending_projection_count: int


class IntentGenerator:
    def __init__(
        self,
        *,
        campaign_plan: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile: Path = DEFAULT_STRATEGY_PROFILE,
        output_path: Path = DEFAULT_ORDER_INTENTS,
        report_path: Path = DEFAULT_ORDER_INTENT_REPORT,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
        market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
        data_root: Path | None = None,
        max_loss_jpy: float | None = None,
        max_loss_pct: float | None = None,
        risk_equity_jpy: float | None = None,
    ) -> None:
        self.campaign_plan = campaign_plan
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path
        self.pair_charts_path = pair_charts_path
        self.levels_path = levels_path
        self.market_context_matrix_path = market_context_matrix_path
        self.data_root = data_root or (ROOT / "data")
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy

    def run(self, *, snapshot_path: Path | None = None, max_candidates: int = 12) -> IntentGenerationSummary:
        plan = json.loads(self.campaign_plan.read_text())
        snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text())) if snapshot_path else None
        stale_plan_issue = _campaign_plan_staleness_issue(
            plan,
            campaign_plan_path=self.campaign_plan,
            strategy_profile_path=self.strategy_profile,
            data_root=self.data_root,
            snapshot=snapshot,
        )
        if stale_plan_issue is not None:
            raise RuntimeError(stale_plan_issue)
        lanes = [lane for lane in plan.get("lanes", []) if _lane_can_attempt(lane)]
        validation_time_utc = _snapshot_validation_time(snapshot)
        # Load ATR / regime per pair from pair_charts.json before lane
        # expansion. Range phases need different executable tactics:
        # forming/stable boxes get rail LIMIT rotation coverage; confirmed
        # range breaks get a breakout-side STOP-ENTRY continuation candidate.
        # BREAKOUT_PENDING / squeeze remains a wait state so the bot does not
        # fade the box just before expansion.
        pair_charts = _load_pair_charts(self.pair_charts_path)
        levels_snapshot = _load_levels_snapshot(self.levels_path)
        market_context_matrix = _load_market_context_matrix(self.market_context_matrix_path)
        strategy_profile = StrategyProfile.load(self.strategy_profile) if self.strategy_profile.exists() else None
        if snapshot is not None:
            range_seed_count = len(lanes)
            lanes = _append_current_range_phase_lanes(lanes, pair_charts)
            if len(lanes) > range_seed_count:
                max_candidates = max(max_candidates, len(lanes))
            matrix_seed_count = len(lanes)
            lanes = _append_matrix_supported_repair_lanes(lanes, market_context_matrix, strategy_profile, pair_charts)
            if len(lanes) > matrix_seed_count:
                max_candidates = max(max_candidates, len(lanes))
            post_harvest_seed_count = len(lanes)
            lanes = _append_post_harvest_reentry_lanes(
                lanes,
                pair_charts,
                snapshot,
                data_root=self.data_root,
            )
            if len(lanes) > post_harvest_seed_count:
                max_candidates = max(max_candidates, len(lanes))
            forecast_seed_count = len(lanes)
            forecast_cycle_id = _pre_entry_forecast_cycle_id(
                snapshot,
                pair_charts_path=self.pair_charts_path,
            )
            lanes = _append_forecast_seed_lanes(
                lanes,
                pair_charts,
                snapshot,
                data_root=self.data_root,
                forecast_cycle_id=forecast_cycle_id,
            )
            if len(lanes) > forecast_seed_count:
                max_candidates = max(max_candidates, len(lanes))
        # Phase 2 (user 2026-05-08「短期SHORTなら長期LONGでもSHORTいけるでしょ。
        # 逆もまた然り」): under SL-free, also synthesize mirror lanes with
        # opposite direction for each (desk, pair, method) so the scoring
        # layer has both sides as candidates and the AI trader can pick
        # whichever the structural lens favors right now. Dedupe to avoid
        # producing the mirror twice when campaign_plan already lists both
        # directions. `max_candidates` doubles to keep the dual-direction
        # set room.
        if _sl_free_active():
            seen_keys = {(l.get("desk"), l.get("pair"), l.get("direction"), l.get("method")) for l in lanes}
            mirrors: list[dict[str, Any]] = []
            for lane in lanes:
                if lane.get("forecast_seed") or lane.get("matrix_repair_seed") or lane.get("post_harvest_reentry_seed"):
                    continue
                m = _mirror_lane(lane)
                key = (m.get("desk"), m.get("pair"), m.get("direction"), m.get("method"))
                if key in seen_keys:
                    continue
                mirrors.append(m)
                seen_keys.add(key)
            lanes = lanes + mirrors
            max_candidates = max(max_candidates, max_candidates * 2)
        lanes = _dedupe_lanes_for_generation(lanes)
        # Pull equity-derived per-trade cap from daily_target_state.json when
        # neither explicit JPY nor pct arguments were supplied. This is the
        # day's risk budget already divided by the target trade pace, so each
        # generated lane sizes against one shot — not the whole day. No JPY
        # literal fallback (§3.5).
        default_cap = _per_trade_risk_from_state()
        max_loss_jpy = resolve_max_loss_jpy(
            max_loss_jpy=self.max_loss_jpy,
            max_loss_pct=self.max_loss_pct,
            equity_jpy=self.risk_equity_jpy,
            default_max_loss_jpy=default_cap,
            label="generate-intents risk cap",
        )
        # Whole-day cap on `open_risk + candidate_risk` (portfolio cap).
        # Distinct from `max_loss_jpy` (per-trade cap). Per AGENT_CONTRACT §3.5
        # these caps must not be conflated: feeding the per-trade slice as the
        # portfolio cap blocks every additional shot once one position opens,
        # because `open_risk` is already on that single-shot scale. None when
        # the ledger is missing — the validator skips the portfolio gate
        # rather than synthesizing a literal.
        portfolio_loss_cap = _daily_risk_budget_from_state()
        telemetry_cache = (
            _build_telemetry_live_readiness_cache(
                data_root=self.data_root,
                validation_time_utc=validation_time_utc,
            )
            if snapshot is not None and _require_telemetry_for_live_active()
            else None
        )
        # Same-day loss-streak discipline (AGENT_CONTRACT §8, 2026-06-10).
        # Broker-truth realized closes for the current campaign day, trader-
        # attributed only. Computed once per run; the campaign-day key is the
        # UTC calendar date (JST9 boundary, same as target._campaign_day_key).
        loss_streak_day = (validation_time_utc or datetime.now(timezone.utc)).date().isoformat()
        loss_streaks = (
            compute_same_day_loss_streaks(self.data_root / "execution_ledger.db", loss_streak_day)
            if snapshot is not None
            else {}
        )
        results: list[GeneratedIntent] = []
        for lane in lanes[:max_candidates]:
            variants = (None,) if snapshot is None else _order_variants_for(lane)
            for order_type in variants:
                results.append(
                    self._build_for_lane(
                        lane,
                        snapshot,
                        strategy_profile,
                        max_loss_jpy=max_loss_jpy,
                        portfolio_loss_cap=portfolio_loss_cap,
                        pair_charts=pair_charts,
                        levels_snapshot=levels_snapshot,
                        market_context_matrix=market_context_matrix,
                        order_type_override=order_type,
                        validation_time_utc=validation_time_utc,
                        telemetry_cache=telemetry_cache,
                        data_root=self.data_root,
                        loss_streaks=loss_streaks,
                    )
                )
        generated_at = datetime.now(timezone.utc).isoformat()
        self._write_output(results, generated_at, snapshot_path)
        self._write_report(results, generated_at, snapshot_path)
        return IntentGenerationSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            candidates_seen=len(lanes),
            generated=sum(1 for item in results if item.intent is not None),
            needs_snapshot=sum(1 for item in results if item.status == "NEEDS_BROKER_SNAPSHOT"),
            dry_run_passed=sum(1 for item in results if item.status == "DRY_RUN_PASSED"),
            live_ready=sum(1 for item in results if item.status == "LIVE_READY"),
        )

    def _build_for_lane(
        self,
        lane: dict[str, Any],
        snapshot: BrokerSnapshot | None,
        strategy_profile: StrategyProfile | None,
        *,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None = None,
        pair_charts: dict[str, dict[str, Any]] | None = None,
        levels_snapshot: dict[str, Any] | None = None,
        market_context_matrix: dict[str, Any] | None = None,
        order_type_override: OrderType | None = None,
        validation_time_utc: datetime | None = None,
        telemetry_cache: _TelemetryLiveReadinessCache | None = None,
        data_root: Path | None = None,
        loss_streaks: dict[str, SameDayLossStreak] | None = None,
    ) -> GeneratedIntent:
        parent_lane_id = _lane_id(lane)
        method = TradeMethod.parse(str(lane["method"]))
        default_order_type = _order_type_for(method)
        lane_id = _variant_lane_id(
            parent_lane_id,
            order_type_override,
            default_order_type=default_order_type,
        )
        pair = str(lane["pair"])
        direction = str(lane["direction"])
        if snapshot is None:
            return GeneratedIntent(
                lane_id=lane_id,
                status="NEEDS_BROKER_SNAPSHOT",
                intent=None,
                risk_metrics=None,
                risk_allowed=None,
                risk_issues=(),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=("broker snapshot is required to price entry/TP/SL",),
                note="Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.",
            )
        quote = snapshot.quotes.get(pair)
        if quote is None:
            return GeneratedIntent(
                lane_id=lane_id,
                status="MISSING_QUOTE",
                intent=None,
                risk_metrics=None,
                risk_allowed=False,
                risk_issues=({"code": "MISSING_QUOTE", "message": f"missing quote for {pair}", "severity": "BLOCK"},),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=(f"snapshot has no quote for {pair}",),
                note="Cannot build priced intent without a live quote.",
            )
        atr_pips = _atr_pips_for(pair, pair_charts)
        range_indicators = _range_indicators_for(pair, pair_charts)
        pair_chart = _pair_chart_for(pair, pair_charts)
        regime_state = _regime_state_for(pair, pair_charts)
        regime_reading = _regime_reading_for(pair, pair_charts)
        session_bucket = _session_bucket_for(pair, pair_charts)
        chart_context = _chart_context_for(pair, pair_charts)
        chart_context.update(_market_location_context_for(pair, quote.mid, pair_charts, levels_snapshot))
        # Same-day loss-streak sizing backoff (AGENT_CONTRACT §8, 2026-06-10).
        # §6 instructs "size it down" instead of inventing prose blockers: each
        # consecutive trader-attributed realized loss on this pair today halves
        # the per-trade risk budget for the next attempt on the same pair.
        pair_loss_streak = (loss_streaks or {}).get(pair)
        streak_count = pair_loss_streak.consecutive_losses if pair_loss_streak else 0
        effective_max_loss_jpy = max_loss_jpy
        if streak_count > 0 and LOSS_STREAK_BLOCK_THRESHOLD > 0:
            effective_max_loss_jpy = max_loss_jpy * (LOSS_STREAK_SIZE_BACKOFF**streak_count)
        intent = _intent_from_lane(
            lane,
            quote,
            snapshot,
            max_loss_jpy=effective_max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            atr_pips=atr_pips,
            range_indicators=range_indicators,
            order_type_override=order_type_override,
            parent_lane_id=parent_lane_id,
            regime_state=regime_state,
            regime_reading=regime_reading,
            session_bucket=session_bucket,
            chart_context=chart_context,
            pair_chart=pair_chart,
            market_context_matrix=market_context_matrix,
            data_root=data_root,
        )
        risk_policy = RiskPolicy(
            block_new_entries_with_pending_entry_orders=False,
            allow_protected_trader_position_adds=True,
            max_portfolio_loss_jpy=portfolio_loss_cap,
        )
        risk = RiskEngine(
            policy=risk_policy,
            validation_time_utc=validation_time_utc,
        ).validate(
            intent,
            snapshot,
            for_live_send=False,
        )
        strategy_profile_evidence = strategy_profile.issue_evidence(intent) if strategy_profile else None
        strategy_issues = tuple(
            issues_to_dicts(
                strategy_profile.validate(intent, for_live_send=False),
                strategy_profile_evidence=strategy_profile_evidence,
            )
            if strategy_profile
            else ()
        )
        live_strategy_issues = tuple(
            issues_to_dicts(
                strategy_profile.validate(intent, for_live_send=True),
                strategy_profile_evidence=strategy_profile_evidence,
            )
            if strategy_profile
            else ()
        )
        live_blockers = tuple(issue["message"] for issue in live_strategy_issues if issue.get("severity") == "BLOCK")
        risk_issues = list(issue.__dict__ for issue in risk.issues)
        risk_allowed = risk.allowed
        daily_entry_budget_issue = _daily_entry_budget_block_issue(
            (data_root or self.data_root) / "daily_target_state.json"
        )
        if daily_entry_budget_issue is not None and str(intent.metadata.get("position_intent") or "") != "HEDGE":
            risk_issues.append(daily_entry_budget_issue)
            live_blockers = (*live_blockers, daily_entry_budget_issue["message"])
            risk_allowed = False
        # Per AGENT_CONTRACT §3.5: surface MISSING_ATR_DATA as a BLOCK issue so
        # the operator sees that geometry was built without market context.
        # No silent fallback — the lane is blocked from going LIVE_READY until
        # pair_charts are refreshed.
        if atr_pips is None:
            risk_issues.append(
                {
                    "code": "MISSING_ATR_DATA",
                    "message": (
                        f"pair_charts.json has no atr_pips for {pair} {GEOMETRY_ATR_TIMEFRAME}; "
                        "geometry is using spread-only floor (no ATR scaling)."
                    ),
                    "severity": "BLOCK",
                }
            )
            # Force DRY_RUN_BLOCKED downstream.
            risk_allowed = False
        # Fix B (2026-05-12): _risk_budgeted_units returns 0 when the
        # current margin headroom can only support a sub-1000u lot. Surface
        # that as a BLOCK so the intent becomes DRY_RUN_BLOCKED — never
        # LIVE_READY — and the gateway never receives a fillable receipt at
        # an unprofitable lot size. 2026-05-12T07:46 UTC produced 201u
        # EUR_USD, 322u AUD_JPY, 2u GBP_USD entries whose spread cost
        # dominated any pip target; this gate stops the same pattern.
        if int(intent.units) == 0 and not _min_lot_test_override_active():
            min_lot_issue = _min_lot_block_issue(
                pair=pair,
                entry=(
                    float(intent.entry)
                    if intent.entry is not None
                    else (quote.ask if intent.side == Side.LONG else quote.bid)
                ),
                sl=intent.sl,
                max_loss_jpy=effective_max_loss_jpy,
                snapshot=snapshot,
                side=intent.side,
                position_intent=str(intent.metadata.get("position_intent") or ""),
            )
            risk_issues.append(min_lot_issue)
            if min_lot_issue["message"] not in live_blockers:
                live_blockers = (*live_blockers, min_lot_issue["message"])
            risk_allowed = False
        if (
            method == TradeMethod.RANGE_ROTATION
            and order_type_override == OrderType.MARKET
            and intent.metadata.get("geometry_model") not in {"RANGE_RAIL_MARKET", "RANGE_DIRECTIONAL_MARKET"}
        ):
            risk_issues.append(
                {
                    "code": "RANGE_MARKET_NOT_AT_RAIL",
                    "message": (
                        f"{pair} range MARKET lane is not inside the rail zone; "
                        "keep the pending LIMIT rail order unless M5 RANGE/QUIET direction bias "
                        "supports a directional range market scalp."
                    ),
                    "severity": "BLOCK",
                }
            )
            risk_allowed = False
        context_issues = _method_context_issues(intent)
        if context_issues:
            risk_issues.extend(context_issues)
            # Only block when at least one context issue is BLOCK severity.
            # Under SL-free CHART_DIRECTION_CONFLICT is downgraded to WARN
            # (commit 1eee01e) so Phase 2 mirror lanes can reach LIVE_READY
            # — but the previous code blanket-set risk_allowed=False the
            # moment ANY context_issue existed, killing every WARN-only
            # mirror. User 2026-05-11「他の通貨入らないね」.
            if any(issue.get("severity") == "BLOCK" for issue in context_issues):
                risk_allowed = False
        loss_streak_issues = _same_day_loss_streak_issues(
            intent,
            pair_loss_streak,
            base_max_loss_jpy=max_loss_jpy,
            effective_max_loss_jpy=effective_max_loss_jpy,
        )
        if loss_streak_issues:
            risk_issues.extend(loss_streak_issues)
            if any(issue.get("severity") == "BLOCK" for issue in loss_streak_issues):
                risk_allowed = False
        forecast_live_issue = _forecast_live_readiness_issue(intent, intent.metadata or {}, method)
        if forecast_live_issue is not None:
            risk_issues.append(forecast_live_issue)
            live_blockers = (*live_blockers, forecast_live_issue["message"])
        forecast_watch_issue = _forecast_watch_only_issue(intent, intent.metadata or {})
        if forecast_watch_issue is not None:
            risk_issues.append(forecast_watch_issue)
            live_blockers = (*live_blockers, forecast_watch_issue["message"])
        telemetry_live_issues = _telemetry_live_readiness_issues(
            intent,
            intent.metadata or {},
            snapshot,
            validation_time_utc,
            cache=telemetry_cache,
            data_root=data_root,
        )
        if telemetry_live_issues:
            risk_issues.extend(telemetry_live_issues)
            live_blockers = (*live_blockers, *(issue["message"] for issue in telemetry_live_issues))
        fresh_rr_issue = _fresh_entry_live_reward_risk_issue(intent, risk.metrics)
        if fresh_rr_issue is not None:
            risk_issues.append(fresh_rr_issue)
            live_blockers = (*live_blockers, fresh_rr_issue["message"])
        live_risk = RiskEngine(
            policy=_live_send_preview_policy(risk_policy),
            live_enabled=True,
            validation_time_utc=validation_time_utc,
        ).validate(intent, snapshot, for_live_send=True)
        risk_issues, live_blockers = _merge_live_send_preview_blockers(
            risk_issues,
            live_blockers,
            live_risk.issues,
        )
        risk_issues = tuple(risk_issues)
        if risk_allowed and not live_blockers:
            status = "LIVE_READY"
        elif risk_allowed:
            status = "DRY_RUN_PASSED"
        else:
            status = "DRY_RUN_BLOCKED"
        return GeneratedIntent(
            lane_id=lane_id,
            status=status,
            intent=_intent_to_json(intent),
            risk_metrics=asdict(risk.metrics) if risk.metrics else None,
            risk_allowed=risk_allowed,
            risk_issues=risk_issues,
            strategy_issues=strategy_issues,
            live_strategy_issues=live_strategy_issues,
            live_blockers=live_blockers,
            note="Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.",
        )

    def _write_output(
        self,
        results: list[GeneratedIntent],
        generated_at: str,
        snapshot_path: Path | None,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": generated_at,
            "campaign_plan": str(self.campaign_plan),
            "strategy_profile": str(self.strategy_profile),
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "results": [asdict(item) for item in results],
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        results: list[GeneratedIntent],
        generated_at: str,
        snapshot_path: Path | None,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Order Intents Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Campaign plan: `{self.campaign_plan}`",
            f"- Snapshot: `{snapshot_path if snapshot_path else 'none'}`",
            f"- Results: `{len(results)}`",
            "",
            "## Status Counts",
            "",
        ]
        counts: dict[str, int] = {}
        for item in results:
            counts[item.status] = counts.get(item.status, 0) + 1
        for status, count in sorted(counts.items()):
            lines.append(f"- `{status}`: `{count}`")
        lines.extend(["", "## Candidates", ""])
        for item in results:
            lines.append(f"- `{item.lane_id}` status=`{item.status}`")
            lines.append(f"  - note: {item.note}")
            if item.intent:
                intent = item.intent
                lines.append(
                    f"  - intent: `{intent['pair']} {intent['side']} {intent['order_type']}` "
                    f"units={intent['units']} entry={intent.get('entry')} tp={intent['tp']} sl={intent['sl']}"
                )
                metadata = intent.get("metadata") or {}
                forecast_direction = str(metadata.get("forecast_direction") or "")
                if forecast_direction or metadata.get("forecast_horizon_min") is not None:
                    confidence = _optional_float(metadata.get("forecast_confidence"))
                    confidence_text = f"{confidence:.4f}" if confidence is not None else str(
                        metadata.get("forecast_confidence") or "n/a"
                    )
                    lines.append(
                        f"  - forecast: direction=`{forecast_direction or 'UNKNOWN'}` "
                        f"confidence=`{confidence_text}` "
                        f"horizon_min=`{metadata.get('forecast_horizon_min')}` "
                        f"watch_only=`{bool(metadata.get('forecast_watch_only'))}`"
                    )
            if item.risk_metrics:
                margin_tail = ""
                if item.risk_metrics.get("estimated_margin_jpy") is not None:
                    margin_tail = f" margin=`{item.risk_metrics['estimated_margin_jpy']:.1f} JPY`"
                    after = item.risk_metrics.get("margin_utilization_after_pct")
                    cap = item.risk_metrics.get("max_margin_utilization_pct")
                    if after is not None and cap is not None:
                        margin_tail += f" margin_after=`{after:.1f}%/{cap:.1f}%`"
                lines.append(
                    f"  - risk metrics: risk=`{item.risk_metrics['risk_jpy']:.1f} JPY` "
                    f"reward=`{item.risk_metrics['reward_jpy']:.1f} JPY` "
                    f"rr=`{item.risk_metrics['reward_risk']:.2f}` spread=`{item.risk_metrics['spread_pips']:.1f}pip`"
                    f"{margin_tail}"
                )
            for issue in item.risk_issues:
                lines.append(f"  - risk {issue['severity']}: {issue['code']} {issue['message']}")
            for issue in item.strategy_issues:
                lines.append(f"  - strategy {issue['severity']}: {issue['code']} {issue['message']}")
            for issue in item.live_strategy_issues:
                lines.append(f"  - live strategy {issue['severity']}: {issue['code']} {issue['message']}")
            for blocker in item.live_blockers:
                lines.append(f"  - live blocker: {blocker}")
        lines.extend(
            [
                "",
                "## Completion Rule",
                "",
                "- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.",
                "- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.",
                "- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _live_send_preview_policy(policy: RiskPolicy) -> RiskPolicy:
    """Return the send-gateway risk contract without the external live flag.

    Intent generation is still a dry-run layer, but `LIVE_READY` must not name
    lanes that the gateway will deterministically block for the same snapshot.
    The preview keeps the execution-risk checks and suppresses only
    LIVE_DISABLED, which is controlled by the operator outside lane quality.
    """
    return replace(policy, require_live_enabled_for_send=False)


def _merge_live_send_preview_blockers(
    risk_issues: list[dict[str, Any]],
    live_blockers: tuple[str, ...],
    live_issues: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    blockers = list(live_blockers)
    seen_blockers = set(blockers)

    for issue in live_issues:
        if getattr(issue, "severity", None) != "BLOCK":
            continue
        code = str(getattr(issue, "code", "") or "")
        if code == "LIVE_DISABLED":
            continue
        if code == "BAD_UNITS" and _has_specific_min_lot_issue(risk_issues):
            continue
        message = str(getattr(issue, "message", "") or "")
        match = next(
            (
                existing
                for existing in risk_issues
                if existing.get("code") == code and existing.get("message") == message
            ),
            None,
        )
        if match is None:
            risk_issues.append({"code": code, "message": message, "severity": "BLOCK"})
        else:
            match["severity"] = "BLOCK"
        if message and message not in seen_blockers:
            blockers.append(message)
            seen_blockers.add(message)

    return risk_issues, tuple(blockers)


_SPECIFIC_MIN_LOT_ISSUE_CODES = {
    "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
    "MARGIN_TOO_THIN_FOR_MIN_LOT",
    "CONVERSION_RATE_MISSING_FOR_MIN_LOT",
    "MIN_LOT_SIZE_UNAVAILABLE",
}


def _has_specific_min_lot_issue(risk_issues: list[dict[str, Any]]) -> bool:
    return any(str(issue.get("code") or "") in _SPECIFIC_MIN_LOT_ISSUE_CODES for issue in risk_issues)


def _lane_can_attempt(lane: dict[str, Any]) -> bool:
    return lane.get("adoption") in {"ORDER_INTENT_REQUIRED", "RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"} and lane.get(
        "direction"
    ) in {"LONG", "SHORT"}


def _mirror_lane(lane: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `lane` with `direction` flipped LONG↔SHORT.

    User 2026-05-08「短期SHORTなら、長期LONGでもSHORTいけるでしょ。逆もまた然り」:
    under SL-free mode the AI trader reads market direction in real time
    and should be free to take counter-trend pullbacks regardless of the
    historical campaign direction. Generating a mirror lane lets the
    scoring layer compare LONG vs SHORT on the same pair/method/desk
    and pick the side the structural lens actually favors right now.
    """
    flipped_direction = "SHORT" if lane.get("direction") == "LONG" else "LONG"
    mirror = dict(lane)
    mirror["direction"] = flipped_direction
    # Mark synthesised mirrors so downstream layers can recognise them
    # in audit / report output. The lane_id changes naturally because
    # `_lane_id` reads the new direction.
    mirror["mirror_of"] = lane.get("direction")
    return mirror


def _dedupe_lanes_for_generation(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one source lane per executable desk/pair/side/method key.

    Seed helpers are intentionally prepended in priority order. Keeping the
    first lane preserves that priority while preventing one market idea from
    expanding into duplicate LIMIT/STOP/MARKET variants and later duplicate
    live receipts.
    """

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for lane in lanes:
        key = _lane_generation_key(lane)
        if key is None:
            out.append(lane)
            continue
        if key in seen:
            continue
        out.append(lane)
        seen.add(key)
    return out


def _lane_generation_key(lane: dict[str, Any]) -> tuple[str, str, str, str] | None:
    desk = str(lane.get("desk") or "").strip().lower()
    pair = str(lane.get("pair") or "").strip().upper()
    direction = str(lane.get("direction") or "").strip().upper()
    method = str(lane.get("method") or "").strip().upper()
    if not (desk and pair and direction and method):
        return None
    return desk, pair, direction, method


def _sl_free_active() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }


def _lane_id(lane: dict[str, Any]) -> str:
    return f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}"


def _variant_lane_id(
    parent_lane_id: str,
    order_type: OrderType | None,
    *,
    default_order_type: OrderType | None = None,
) -> str:
    if order_type is None or order_type == default_order_type:
        return parent_lane_id
    suffix_by_type = {
        OrderType.LIMIT: "LIMIT",
        OrderType.MARKET: "MARKET",
        OrderType.STOP_ENTRY: "STOP",
    }
    suffix = suffix_by_type.get(order_type)
    if suffix:
        return f"{parent_lane_id}:{suffix}"
    return parent_lane_id


def _order_variants_for(lane: dict[str, Any]) -> tuple[OrderType, ...]:
    method = TradeMethod.parse(str(lane["method"]))
    base = _order_type_for(method)
    variants: list[OrderType] = []
    if method == TradeMethod.BREAKOUT_FAILURE:
        variants.append(OrderType.LIMIT)
    variants.append(base)
    if not _lane_forbids_market_chase(lane):
        variants.append(OrderType.MARKET)
    return tuple(dict.fromkeys(variants))


def _lane_forbids_market_chase(lane: dict[str, Any]) -> bool:
    """Return true when campaign evidence requires a pending trigger receipt.

    `TRIGGER_RECEIPT_REQUIRED` lanes come from missed-edge / trigger repair
    evidence. AGENT_CONTRACT §11 allows those to reopen only from LIMIT or
    STOP-ENTRY receipts; generating a same-geometry MARKET variant turns
    "arm the trigger" into quote chasing and lets TraderBrain select stale
    market entries that the live gateway later rejects.
    """
    if str(lane.get("adoption") or "") == "TRIGGER_RECEIPT_REQUIRED":
        return True
    receipt = str(lane.get("required_receipt") or "").lower()
    return "no market chase" in receipt


def _macro_event_sizing_plan(
    lane: dict[str, Any],
    *,
    side: Side,
    base_max_loss_jpy: float,
    portfolio_loss_cap: float | None,
    position_metadata: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Return an event-specific loss cap and audit metadata for the lane.

    Only factual post-release macro surprises can size up. Pre-release nowcasts
    remain directional evidence but do not get extra risk because the actual
    release is not known yet.
    """

    if str(position_metadata.get("position_intent") or "").upper() == "HEDGE":
        return base_max_loss_jpy, {}
    signal = _macro_event_size_up_signal(lane, side=side)
    if signal is None or base_max_loss_jpy <= 0:
        return base_max_loss_jpy, {}

    scaled_cap = base_max_loss_jpy * MACRO_EVENT_RISK_MULTIPLIER
    daily_share_cap: float | None = None
    if (
        portfolio_loss_cap is not None
        and portfolio_loss_cap > 0
        and MACRO_EVENT_MAX_DAILY_RISK_SHARE > 0
    ):
        daily_share_cap = portfolio_loss_cap * MACRO_EVENT_MAX_DAILY_RISK_SHARE
        if daily_share_cap > base_max_loss_jpy:
            scaled_cap = min(scaled_cap, daily_share_cap)
        else:
            scaled_cap = base_max_loss_jpy

    effective_cap = max(base_max_loss_jpy, scaled_cap)
    if effective_cap <= base_max_loss_jpy:
        return base_max_loss_jpy, {}

    confidence = _optional_float(signal.get("confidence")) or 0.0
    return effective_cap, {
        "macro_event_size_up": True,
        "macro_event_signal_name": signal.get("name"),
        "macro_event_signal_direction": signal.get("direction"),
        "macro_event_signal_confidence": round(confidence, 4),
        "macro_event_base_max_loss_jpy": round(base_max_loss_jpy, 4),
        "macro_event_risk_multiplier": MACRO_EVENT_RISK_MULTIPLIER,
        "macro_event_daily_risk_share_cap_jpy": round(daily_share_cap, 4) if daily_share_cap is not None else None,
        "macro_event_loss_budget_target": True,
    }


def _macro_event_size_up_signal(lane: dict[str, Any], *, side: Side) -> dict[str, Any] | None:
    support = lane.get("forecast_market_support")
    if not isinstance(support, dict) or not bool(support.get("ok")):
        return None
    expected_direction = "UP" if side == Side.LONG else "DOWN"
    support_direction = str(support.get("direction") or "").upper()
    if support_direction and support_direction != expected_direction:
        return None
    for raw in support.get("signals") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if name not in MACRO_EVENT_SIZE_UP_SIGNAL_NAMES:
            continue
        direction = str(raw.get("direction") or "").upper()
        if direction != expected_direction:
            continue
        confidence = _optional_float(raw.get("confidence")) or 0.0
        if confidence < MACRO_EVENT_SIZE_UP_MIN_SIGNAL_CONFIDENCE:
            continue
        return raw
    return None


def _intent_from_lane(
    lane: dict[str, Any],
    quote: Quote,
    snapshot: BrokerSnapshot,
    *,
    max_loss_jpy: float,
    portfolio_loss_cap: float | None = None,
    atr_pips: float | None = None,
    range_indicators: dict[str, Any] | None = None,
    order_type_override: OrderType | None = None,
    parent_lane_id: str | None = None,
    regime_state: str | None = None,
    regime_reading: dict[str, Any] | None = None,
    session_bucket: str | None = None,
    chart_context: dict[str, Any] | None = None,
    pair_chart: dict[str, Any] | None = None,
    market_context_matrix: dict[str, Any] | None = None,
    data_root: Path | None = None,
) -> OrderIntent:
    pair = str(lane["pair"])
    side = Side.parse(str(lane["direction"]))
    method = TradeMethod.parse(str(lane["method"]))
    order_type = order_type_override or _order_type_for(method)
    base_reward_risk = _target_reward_risk(lane)
    # Regime-derived reward_risk per AGENT_CONTRACT §3.5: range → close TP for
    # rotation, trend → wider TP to ride. The lane's base value is preserved
    # for audit; the active geometry reflects current regime.
    execution_regime = _execution_regime_state(method, regime_state, chart_context)
    rr_multiplier = _regime_reward_risk_multiplier(execution_regime)
    target_reward_risk = _execution_target_reward_risk(
        base_reward_risk,
        method,
        order_type,
        execution_regime,
        chart_context,
        side,
    )
    stop_widen_mult = _effective_stop_widening_multiplier(method, regime_reading, chart_context)
    entry, tp, sl = _geometry(
        pair,
        side,
        order_type,
        quote,
        reward_risk=target_reward_risk,
        atr_pips=atr_pips,
        range_indicators=range_indicators if method == TradeMethod.RANGE_ROTATION else None,
        chart_indicators=range_indicators,
        stop_widen_mult=stop_widen_mult,
        chart_context=chart_context,
    )
    position_metadata = _position_intent_metadata(pair, side, snapshot, entry=entry)
    if str(position_metadata.get("position_intent") or "").upper() == "HEDGE":
        position_metadata.update(_hedge_timing_metadata(side, position_metadata, chart_context, lane))
    recovery_target_units: int | None = None
    if _is_hedge_recovery_metadata(position_metadata):
        recovery_metadata = _recovery_hedge_sizing_metadata(side, position_metadata, chart_context, lane)
        position_metadata.update(recovery_metadata)
        recovery_target_units = int(position_metadata.get("hedge_recovery_units") or 0) or None
    effective_max_loss_jpy, macro_event_sizing_metadata = _macro_event_sizing_plan(
        lane,
        side=side,
        base_max_loss_jpy=max_loss_jpy,
        portfolio_loss_cap=portfolio_loss_cap,
        position_metadata=position_metadata,
    )
    tp, tp_execution_metadata = _take_profit_execution_plan(
        pair=pair,
        side=side,
        method=method,
        order_type=order_type,
        quote=quote,
        entry=entry,
        tp=tp,
        sl=sl,
        reward_risk=target_reward_risk,
        execution_regime=execution_regime,
        chart_context=chart_context,
        pair_chart=pair_chart,
        atr_pips=atr_pips,
        forecast_direction=lane.get("forecast_direction"),
        forecast_confidence=_optional_float(lane.get("forecast_confidence")),
        forecast_target_price=_optional_float(lane.get("forecast_target_price")),
        hedge_recovery=_is_hedge_recovery_metadata(position_metadata),
    )
    geometry_metadata = _geometry_metadata(
        pair,
        side,
        order_type,
        quote,
        entry=entry,
        tp=tp,
        sl=sl,
        range_indicators=range_indicators if method == TradeMethod.RANGE_ROTATION else None,
        chart_indicators=range_indicators,
        chart_context=chart_context,
        atr_pips=atr_pips,
    )
    geometry_metadata.update(tp_execution_metadata)
    # Disaster stop: broker-side catastrophe bound computed AFTER geometry so
    # the expected intent.sl (sizing / reward-risk anchor) is final, and the
    # strict-ordering buffer can guarantee the broker stop sits beyond it.
    geometry_metadata.update(
        _disaster_sl_metadata(
            pair,
            side,
            entry=entry,
            expected_sl=sl,
            chart_context=chart_context,
        )
    )
    position_intent = str(position_metadata.get("position_intent") or "")
    units = _risk_budgeted_units(
        pair,
        entry,
        sl,
        max_loss_jpy=effective_max_loss_jpy,
        snapshot=snapshot,
        side=side,
        position_intent=position_intent,
        target_units_override=recovery_target_units,
        loss_budget_target=bool(macro_event_sizing_metadata.get("macro_event_size_up")),
    )
    margin_metadata = _margin_sizing_metadata(pair, entry, units, snapshot, side=side, position_intent=position_intent)
    required_receipt, watch_override_reason = _forecast_watch_live_override_receipt(
        lane,
        side,
        order_type,
        chart_context=chart_context,
        geometry_metadata=geometry_metadata,
    )
    event_blockers = list(lane.get("blockers") or [])
    if watch_override_reason:
        event_blockers = [
            item
            for item in event_blockers
            if "watch-only forecast candidate" not in str(item).lower()
        ]
        event_blockers.insert(0, watch_override_reason)
    thesis = f"{lane['desk']} {pair} {side.value} {method.value} {target_reward_risk:.2f}R: {required_receipt}"
    if execution_regime:
        regime_context = f"{execution_regime} current; {method.value} campaign lane"
        if regime_state and regime_state != execution_regime:
            regime_context += f"; dominant={regime_state}"
    else:
        regime_context = f"{method.value} campaign lane"
    location_story = str((chart_context or {}).get("market_location_story") or "")
    lane_story = " | ".join(str(item) for item in lane.get("story_examples", [])[:2])
    matrix_metadata = matrix_summary_for_intent(market_context_matrix, pair, side.value)
    news_metadata = _intent_news_evidence_metadata(pair, data_root=data_root, source_metadata=lane)
    matrix_story = ""
    if matrix_metadata:
        matrix_support_context = matrix_metadata.get("matrix_support_context")
        if isinstance(matrix_support_context, list) and matrix_support_context:
            matrix_support = str(matrix_support_context[0])
        else:
            matrix_support = str(matrix_metadata.get("strongest_matrix_support") or "none")
        matrix_story = (
            f"matrix {matrix_metadata.get('market_context_matrix_ref')}: "
            f"supports={matrix_metadata.get('matrix_support_count', 0)} "
            f"rejects={matrix_metadata.get('matrix_reject_count', 0)} "
            f"warnings={matrix_metadata.get('matrix_warning_count', 0)}; "
            f"support={matrix_support}; "
            f"counter={matrix_metadata.get('strongest_matrix_reject') or 'none'}"
        )
    context = MarketContext(
        regime=regime_context,
        narrative=str(lane.get("reason") or ""),
        chart_story=" | ".join(item for item in (location_story, lane_story, matrix_story) if item)
        or "campaign lane requires current chart read",
        method=method,
        invalidation=f"invalid if SL {sl} trades or campaign overlay vetoes the setup",
        event_risk="; ".join(str(item) for item in event_blockers[:2]),
        session=session_bucket or "generated dry-run",
    )
    return OrderIntent(
        pair=pair,
        side=side,
        order_type=order_type,
        units=units,
        entry=entry,
        tp=tp,
        sl=sl,
        thesis=thesis,
        owner=Owner.TRADER,
        market_context=context,
        metadata={
            "desk": lane.get("desk"),
            "adoption": lane.get("adoption"),
            "campaign_role": lane.get("campaign_role"),
            "required_receipt": required_receipt,
            "forecast_seed": bool(lane.get("forecast_seed")),
            "forecast_watch_only": bool(lane.get("forecast_watch_only")),
            "forecast_watch_only_reason": lane.get("forecast_watch_only_reason"),
            "forecast_watch_only_live_override": bool(watch_override_reason),
            "forecast_watch_only_live_override_reason": watch_override_reason,
            "matrix_repair_seed": bool(lane.get("matrix_repair_seed")),
            "matrix_watch_only_seed": bool(lane.get("matrix_watch_only_seed")),
            "matrix_repair_profile_status": lane.get("matrix_repair_profile_status"),
            "post_harvest_reentry_seed": bool(lane.get("post_harvest_reentry_seed")),
            "post_harvest_trade_id": lane.get("post_harvest_trade_id"),
            "post_harvest_closed_at_utc": lane.get("post_harvest_closed_at_utc"),
            "post_harvest_age_minutes": lane.get("post_harvest_age_minutes"),
            "forecast_cycle_id": lane.get("forecast_cycle_id"),
            "forecast_direction": lane.get("forecast_direction"),
            "forecast_confidence": lane.get("forecast_confidence"),
            "forecast_raw_confidence": lane.get("forecast_raw_confidence"),
            "forecast_calibration_multiplier": lane.get("forecast_calibration_multiplier"),
            "forecast_current_price": lane.get("forecast_current_price"),
            "forecast_target_price": lane.get("forecast_target_price"),
            "forecast_invalidation_price": lane.get("forecast_invalidation_price"),
            "forecast_range_low_price": lane.get("forecast_range_low_price"),
            "forecast_range_high_price": lane.get("forecast_range_high_price"),
            "forecast_range_width_pips": lane.get("forecast_range_width_pips"),
            "forecast_horizon_min": lane.get("forecast_horizon_min"),
            "forecast_rationale": lane.get("forecast_rationale"),
            "forecast_drivers_for": lane.get("forecast_drivers_for"),
            "forecast_drivers_against": lane.get("forecast_drivers_against"),
            "forecast_component_scores": lane.get("forecast_component_scores"),
            "forecast_market_support": lane.get("forecast_market_support"),
            "forecast_market_support_ok": lane.get("forecast_market_support_ok"),
            "forecast_market_support_reason": lane.get("forecast_market_support_reason"),
            "forecast_directional_calibration_name": lane.get("forecast_directional_calibration_name"),
            "forecast_directional_hit_rate": lane.get("forecast_directional_hit_rate"),
            "forecast_directional_samples": lane.get("forecast_directional_samples"),
            "mirror_of": lane.get("mirror_of"),
            "target_reward_risk": target_reward_risk,
            "base_target_reward_risk": base_reward_risk,
            "regime_reward_risk_mult": rr_multiplier,
            "regime_state": execution_regime,
            "dominant_regime_state": regime_state,
            "regime_stop_widen_mult": stop_widen_mult,
            "session_bucket": session_bucket,
            **(chart_context or {}),
            "evidence_tail_jpy": lane.get("evidence_tail_jpy"),
            "evidence_best_jpy": lane.get("evidence_best_jpy"),
            "sizing_rule": (
                f"floor units to the largest broker size under the {effective_max_loss_jpy:.0f} JPY loss cap "
                f"and {RiskPolicy().max_margin_utilization_pct:.1f}% margin utilization cap"
            ),
            "max_loss_jpy": effective_max_loss_jpy,
            **macro_event_sizing_metadata,
            "parent_lane_id": parent_lane_id or _lane_id(lane),
            "order_timing": "NOW_MARKET" if order_type == OrderType.MARKET else "PENDING_TRIGGER",
            **position_metadata,
            **geometry_metadata,
            **margin_metadata,
            **news_metadata,
            **matrix_metadata,
        },
    )


def _same_day_loss_streak_issues(
    intent: OrderIntent,
    streak: SameDayLossStreak | None,
    *,
    base_max_loss_jpy: float,
    effective_max_loss_jpy: float,
) -> list[dict[str, str]]:
    """Same-day per-pair consecutive-loss re-entry discipline (§8, 2026-06-10).

    Regression target: 2026-06-04 EUR_USD lost LONG -2,181, LONG -2,642, then
    revenge-flip SHORT -2,333 within hours (-7,157 JPY, ~3.8% NAV) — the
    lane_history ±25 score nudge and trader_overrides feedback were advisory
    and did not change entry *timing*. Once a pair has hit
    LOSS_STREAK_BLOCK_THRESHOLD consecutive trader-attributed realized losses
    this campaign day:

    - MARKET / STOP-ENTRY re-engagement on that pair (either side) is BLOCKED.
      Active chasing is what compounded the streaks; the pair stays tradable
      through passive LIMIT timing at a structural retest, which forces the
      market to come to the trader's level first.
    - LIMIT lanes get a WARN plus the exponential sizing backoff already
      applied upstream, so a third attempt risks a fraction of the per-trade
      budget instead of full size.

    Below the threshold the streak is advisory: WARN + sizing backoff only.
    Recovery hedges (`position_intent=HEDGE`) manage existing trapped
    exposure under their own §3.5 timing rules and are never hard-blocked
    here. The streak resets on any winning close and dies at the campaign-day
    boundary, so this cannot become a standing direction-bias rule
    (feedback_no_direction_bias_rules).
    """
    if streak is None or streak.consecutive_losses <= 0:
        return []
    if LOSS_STREAK_BLOCK_THRESHOLD <= 0:
        return []
    # Mutate the intent's own metadata dict (never `or {}` — that would write
    # the audit keys to a throwaway dict when metadata starts empty).
    metadata = intent.metadata
    is_hedge = str(metadata.get("position_intent") or "").upper() == "HEDGE"
    count = streak.consecutive_losses
    detail = (
        f"{intent.pair} has {count} consecutive trader-attributed realized "
        f"losses today (net {streak.net_loss_jpy:+.0f} JPY, last at "
        f"{streak.last_loss_ts_utc or 'unknown'}); per-trade budget backed off "
        f"{base_max_loss_jpy:.0f} -> {effective_max_loss_jpy:.0f} JPY"
    )
    metadata["same_day_loss_streak"] = count
    metadata["same_day_loss_streak_net_jpy"] = round(streak.net_loss_jpy, 2)
    metadata["loss_streak_max_loss_scale"] = (
        round(effective_max_loss_jpy / base_max_loss_jpy, 4) if base_max_loss_jpy else 1.0
    )
    active_chase = intent.order_type in (OrderType.MARKET, OrderType.STOP_ENTRY)
    if count >= LOSS_STREAK_BLOCK_THRESHOLD and active_chase and not is_hedge:
        return [
            {
                "code": "SAME_DAY_LOSS_STREAK_CHASE",
                "message": (
                    f"{detail}. Re-engagement after {LOSS_STREAK_BLOCK_THRESHOLD}+ same-day "
                    "losses requires passive LIMIT retest timing, not MARKET/STOP chase."
                ),
                "severity": "BLOCK",
            }
        ]
    return [
        {
            "code": "SAME_DAY_LOSS_STREAK",
            "message": detail,
            "severity": "WARN",
        }
    ]


def _method_context_issues(intent: OrderIntent) -> list[dict[str, str]]:
    """Block receipts whose current chart context contradicts the lane.

    The campaign planner may offer both sides and multiple desks for coverage.
    `generate-intents` is the first executable layer that has live broker
    quotes plus pair_charts in the same packet, so it must prevent a generic
    coverage lane from becoming LIVE_READY when the current chart packet says
    the opposite side or a non-trend operating tape.
    """
    metadata = intent.metadata or {}
    issues: list[dict[str, str]] = []
    method = intent.market_context.method if intent.market_context is not None else None
    hedge_recovery = _is_hedge_recovery_metadata(metadata)
    bias = _method_direction_bias(metadata, method)
    forecast_issue = _forecast_direction_conflict_issue(intent, metadata)
    if forecast_issue is not None:
        issues.append(forecast_issue)
    trend_hard_block = _trend_continuation_hard_block_reason(intent, metadata, method)
    if bias in {Side.LONG.value, Side.SHORT.value} and bias != intent.side.value:
        long_score = metadata.get("m5_long_bias") if method == TradeMethod.RANGE_ROTATION else metadata.get("chart_long_score")
        short_score = metadata.get("m5_short_bias") if method == TradeMethod.RANGE_ROTATION else metadata.get("chart_short_score")
        scope = "M5 range" if method == TradeMethod.RANGE_ROTATION else "pair_charts"
        # Phase 2 (user 2026-05-08「逆もまた然り」): under SL-free the AI
        # trader is the discretionary direction picker — historical
        # chart_score bias is one input, not a hard veto. The MTF + PA +
        # micro-override scoring in trader_brain decides the side; this
        # gate becomes WARN so symmetric mirror lanes reach LIVE_READY.
        severity = "BLOCK" if trend_hard_block else ("WARN" if (_sl_free_active() or hedge_recovery) else "BLOCK")
        tail = (
            "recovery hedge may run against the stale score while it monetizes trapped exposure."
            if hedge_recovery
            else (trend_hard_block or "wait for this side to dominate or choose the aligned lane.")
        )
        issues.append(
            {
                "code": "CHART_DIRECTION_CONFLICT",
                "message": (
                    f"{intent.pair} {intent.side.value} conflicts with current {scope} direction "
                    f"bias={bias} (long_score={long_score}, short_score={short_score}); "
                    f"{tail}"
                ),
                "severity": severity,
            }
        )
    elif trend_hard_block:
        issues.append(
            {
                "code": "TREND_CONTINUATION_DIRECTION_CONFLICT",
                "message": f"{intent.pair} {intent.side.value} {trend_hard_block}",
                "severity": "BLOCK",
            }
        )

    range_phase = str(metadata.get("range_phase") or "").upper()
    if method == TradeMethod.RANGE_ROTATION and range_phase in {"BREAKOUT_PENDING", "BREAKOUT_UP", "BREAKOUT_DOWN"}:
        issues.append(
            {
                "code": "RANGE_PHASE_NOT_ROTATION",
                "message": (
                    f"{intent.pair} {intent.side.value} RANGE_ROTATION is invalid in range_phase={range_phase}; "
                    "wait during pending squeeze or use the confirmed breakout-side continuation lane."
                ),
                "severity": "BLOCK",
            }
        )
    if method == TradeMethod.RANGE_ROTATION and _range_rotation_chases_broader_location(intent, metadata):
        pct_24h = _optional_float(metadata.get("price_percentile_24h"))
        pct_7d = _optional_float(metadata.get("price_percentile_7d"))
        location_bits = []
        if pct_24h is not None:
            location_bits.append(f"p24h={pct_24h:.2f}")
        if pct_7d is not None:
            location_bits.append(f"p7d={pct_7d:.2f}")
        issues.append(
            {
                "code": "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                "message": (
                    f"{intent.pair} {intent.side.value} RANGE_ROTATION is on the wrong side of broader "
                    f"market location ({', '.join(location_bits) or 'percentile unavailable'}). "
                    "Range rotation must buy discount/lower-half rails and sell premium/upper-half rails; "
                    "do not turn a tiny local rail into a late chase."
                ),
                "severity": "BLOCK",
            }
        )

    if method == TradeMethod.TREND_CONTINUATION and intent.order_type == OrderType.MARKET:
        m5_regime = str(metadata.get("m5_regime") or "").upper()
        expected = "TREND_UP" if intent.side == Side.LONG else "TREND_DOWN"
        if m5_regime and m5_regime != expected:
            issues.append(
                {
                    "code": "TREND_MARKET_NOT_OPERATING_TREND",
                    "message": (
                        f"{intent.pair} {intent.side.value} MARKET trend-continuation needs M5 {expected}; "
                        f"current M5 regime is {m5_regime}. Use a pending trigger instead of chasing inside chop/range."
                    ),
                    "severity": "BLOCK",
                }
            )

    breakout_market_issue = _breakout_failure_market_retest_issue(intent, metadata, method)
    if breakout_market_issue is not None:
        issues.append(breakout_market_issue)
    breakout_stop_issue = _breakout_failure_stop_chase_issue(intent, metadata, method)
    if breakout_stop_issue is not None:
        issues.append(breakout_stop_issue)
    pattern_chase_issue = _pattern_reversal_chase_issue(intent, metadata, method)
    if pattern_chase_issue is not None:
        issues.append(pattern_chase_issue)

    if hedge_recovery and _operating_tf_opposes_side(intent.side, metadata):
        market = intent.order_type == OrderType.MARKET
        issues.append(
            {
                "code": "RECOVERY_HEDGE_MARKET_OPPOSED_BY_M5"
                if market
                else "RECOVERY_HEDGE_CONTINUATION_MICRO_OPPOSED",
                "message": (
                    f"{intent.pair} {intent.side.value} recovery hedge is opposed by current M5/M15 "
                    "structure and bias; use a pending trigger and keep continuation size capped until "
                    "the operating timeframe confirms."
                ),
                "severity": "BLOCK" if market else "WARN",
            }
        )

    tp_mode = str(metadata.get("tp_execution_mode") or "").upper()
    tp_intent = str(metadata.get("tp_target_intent") or "").upper()
    tp_source = str(metadata.get("tp_target_source") or "").upper()
    tp_distance = _optional_float(metadata.get("tp_target_distance_pips"))
    tp_atr = _optional_float(metadata.get("tp_atr_pips"))
    harvest_tp_too_far = (
        tp_distance is not None
        and tp_atr is not None
        and tp_atr > 0
        and tp_distance > HARVEST_TP_MAX_OPERATING_ATR_MULT * tp_atr
    )
    if (
        tp_mode == "ATTACHED_TECHNICAL_TP"
        and tp_intent == "HARVEST"
        and "ATR_RR" in tp_source
        and (hedge_recovery or (method == TradeMethod.BREAKOUT_FAILURE and harvest_tp_too_far))
    ):
        issues.append(
            {
                "code": "HARVEST_TP_STRUCTURE_MISSING",
                "message": (
                    f"{intent.pair} {intent.side.value} {method.value if method else 'UNKNOWN'} needs a usable "
                    "nearby structural HARVEST TP; refusing to fall back to a distant ATR/RR target for a "
                    "failed-break or recovery-hedge trade."
                ),
                "severity": "BLOCK",
            }
        )

    # C — 2026-05-13 "no chasing exhausted moves" filter. When the pair
    # has already covered >= EXHAUSTION_RANGE_SIGMA_MULTIPLE (= 2.0,
    # standard 2σ boundary) of its typical H1 range in the last 24
    # hours, AND the new entry direction is aligned with where the
    # move travelled (LONG into a 24h high, SHORT into a 24h low),
    # block it. Fading the exhausted move (opposite direction) is not
    # blocked; same-direction chasing is the failure mode
    # 2026-05-12T15:33 UTC drove the operator to demand killing.
    sigma_mult = _optional_float(metadata.get("range_24h_sigma_multiple"))
    ppct_24h = _gate_location_percentile(intent, metadata, "24h")
    if sigma_mult is not None and sigma_mult >= EXHAUSTION_RANGE_SIGMA_MULTIPLE:
        chasing = False
        if intent.side == Side.LONG and ppct_24h is not None and ppct_24h >= 0.5:
            chasing = True
        if intent.side == Side.SHORT and ppct_24h is not None and ppct_24h <= 0.5:
            chasing = True
        if chasing and _is_structural_retest_entry(intent, metadata, method):
            chasing = False
        if chasing:
            severity = _same_side_chase_severity(intent, hedge_recovery=hedge_recovery)
            issues.append(
                {
                    "code": "EXHAUSTION_RANGE_CHASE",
                    "message": (
                        f"{intent.pair} {intent.side.value} chases a move already "
                        f"{sigma_mult:.2f}× typical hourly range over 24h "
                        f"(p24h={ppct_24h:.2f}); "
                        + _same_side_chase_tail(intent, hedge_recovery=hedge_recovery)
                    ),
                    "severity": severity,
                }
            )
    return issues


def _same_side_chase_severity(intent: OrderIntent, *, hedge_recovery: bool) -> str:
    if not hedge_recovery:
        return "BLOCK"
    if intent.order_type == OrderType.MARKET:
        return "BLOCK"
    return "WARN"


def _same_side_chase_tail(intent: OrderIntent, *, hedge_recovery: bool) -> str:
    if not hedge_recovery:
        return (
            "refuse same-direction entry after the "
            f"{EXHAUSTION_RANGE_SIGMA_MULTIPLE:.1f}σ-equivalent extension."
        )
    if intent.order_type == OrderType.MARKET:
        return (
            "recovery hedge must wait for a pullback, retest, or non-market trigger "
            "instead of buying/selling the already extended price."
        )
    return "allowing as recovery hedge against trapped opposite exposure."


# A support/resistance box midpoint is the geometry boundary between "at the
# lower half" and "at the upper half"; it is not a tuned strategy threshold.
BREAKOUT_FAILURE_RETEST_MIDPOINT = 0.5


def _breakout_failure_market_retest_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> dict[str, str] | None:
    if method != TradeMethod.BREAKOUT_FAILURE or intent.order_type != OrderType.MARKET:
        return None
    tf_map = metadata.get("tf_regime_map")
    if not isinstance(tf_map, dict):
        return None

    wrong_side: list[str] = []
    for timeframe in ("M5", "M15"):
        tf_data = tf_map.get(timeframe)
        if not isinstance(tf_data, dict):
            continue
        position = _optional_float(tf_data.get("range_position"))
        if position is None:
            continue
        if intent.side == Side.SHORT and position <= BREAKOUT_FAILURE_RETEST_MIDPOINT:
            wrong_side.append(_breakout_retest_location_text(timeframe, position, tf_data))
        elif intent.side == Side.LONG and position >= BREAKOUT_FAILURE_RETEST_MIDPOINT:
            wrong_side.append(_breakout_retest_location_text(timeframe, position, tf_data))

    if not wrong_side:
        return None

    expected_side = "upper-half resistance retest" if intent.side == Side.SHORT else "lower-half support retest"
    current_side = "lower-half chase" if intent.side == Side.SHORT else "upper-half chase"
    return {
        "code": "BREAKOUT_FAILURE_MARKET_NOT_RETESTED",
        "message": (
            f"{intent.pair} {intent.side.value} BREAKOUT_FAILURE MARKET is a {current_side} "
            f"({'; '.join(wrong_side)}). If the break has not continued, wait for the "
            f"{expected_side} or use the pending trigger instead of entering at market."
        ),
        "severity": "BLOCK",
    }


def _breakout_retest_location_text(timeframe: str, position: float, tf_data: dict[str, Any]) -> str:
    bits = [f"{timeframe} pos={position:.2f}"]
    support_distance = _optional_float(tf_data.get("nearest_support_distance_pips"))
    resistance_distance = _optional_float(tf_data.get("nearest_resistance_distance_pips"))
    if support_distance is not None:
        bits.append(f"support {support_distance:+.1f}p")
    if resistance_distance is not None:
        bits.append(f"resistance {resistance_distance:+.1f}p")
    return " ".join(bits)


def _breakout_failure_stop_chase_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> dict[str, str] | None:
    if method != TradeMethod.BREAKOUT_FAILURE or intent.order_type != OrderType.STOP_ENTRY:
        return None
    if intent.entry is None:
        return None
    tf_map = metadata.get("tf_regime_map")
    if not isinstance(tf_map, dict):
        return None

    wrong_side: list[str] = []
    for timeframe in ("M5", "M15"):
        tf_data = tf_map.get(timeframe)
        if not isinstance(tf_data, dict):
            continue
        position = _entry_range_position(intent.entry, tf_data)
        if position is None:
            continue
        if intent.side == Side.SHORT and position <= BREAKOUT_FAILURE_RETEST_MIDPOINT:
            wrong_side.append(_breakout_retest_location_text(timeframe, position, tf_data))
        elif intent.side == Side.LONG and position >= BREAKOUT_FAILURE_RETEST_MIDPOINT:
            wrong_side.append(_breakout_retest_location_text(timeframe, position, tf_data))

    if not wrong_side:
        return None

    expected_side = "upper-half resistance retest/LIMIT" if intent.side == Side.SHORT else "lower-half support retest/LIMIT"
    failed_side = "lower-half/support sell-stop" if intent.side == Side.SHORT else "upper-half/resistance buy-stop"
    return {
        "code": "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
        "message": (
            f"{intent.pair} {intent.side.value} BREAKOUT_FAILURE STOP_ENTRY is a {failed_side} "
            f"({'; '.join(wrong_side)}). Use the {expected_side} or a true TREND_CONTINUATION "
            "breakout lane; do not arm a failed-break stop that sells the low or buys the high."
        ),
        "severity": "BLOCK",
    }


def _is_structural_retest_entry(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> bool:
    """True when a range/failure receipt waits on the retest side.

    This is the carve-out for the 24h exhaustion filter: selling after a large
    down move is a chase when it sells the low, but not when the receipt is a
    structural upper-half retest/rejection. The midpoint is the existing box
    geometry boundary used by breakout-failure timing, not a new threshold.
    """
    if method not in {TradeMethod.BREAKOUT_FAILURE, TradeMethod.RANGE_ROTATION}:
        return False
    if intent.order_type == OrderType.STOP_ENTRY:
        return False
    if method == TradeMethod.RANGE_ROTATION and _range_rotation_chases_broader_location(intent, metadata):
        return False
    tf_map = metadata.get("tf_regime_map")
    if not isinstance(tf_map, dict):
        return False

    positions: list[float] = []
    for timeframe in ("M5", "M15"):
        tf_data = tf_map.get(timeframe)
        if not isinstance(tf_data, dict):
            continue
        position = None
        if intent.order_type == OrderType.LIMIT and intent.entry is not None:
            position = _entry_range_position(intent.entry, tf_data)
        if position is None:
            position = _optional_float(tf_data.get("range_position"))
        if position is not None:
            positions.append(position)

    if not positions:
        return False
    if intent.side == Side.SHORT:
        return all(position >= BREAKOUT_FAILURE_RETEST_MIDPOINT for position in positions)
    return all(position <= BREAKOUT_FAILURE_RETEST_MIDPOINT for position in positions)


def _range_rotation_chases_broader_location(intent: OrderIntent, metadata: dict[str, Any]) -> bool:
    """True when a rail fade is on the wrong side of broader location.

    M5/M15 can mark a tiny local rail after a large move. That does not make a
    SHORT at the 24h/7d discount a resistance fade, nor a LONG at the premium a
    support fade. For pending LIMIT receipts, evaluate the planned entry price
    when the chart packet carries entry percentiles; the current price may be
    extended while the order waits for a valid retest. Use the existing midpoint
    boundary: below half is discount, above half is premium.
    """
    percentiles = [
        value
        for value in (
            _gate_location_percentile(intent, metadata, "24h"),
            _gate_location_percentile(intent, metadata, "7d"),
        )
        if value is not None
    ]
    if not percentiles:
        return False
    if intent.side == Side.SHORT:
        return any(value <= BREAKOUT_FAILURE_RETEST_MIDPOINT for value in percentiles)
    return any(value >= BREAKOUT_FAILURE_RETEST_MIDPOINT for value in percentiles)


def _gate_location_percentile(intent: OrderIntent, metadata: dict[str, Any], horizon: str) -> float | None:
    """Return the location percentile a timing gate should evaluate.

    MARKET and STOP_ENTRY entries are evaluated at current location because
    they chase or trigger through the current move. LIMIT receipts wait for a
    pullback/retest; when the generator computed an entry percentile, use that
    planned entry location instead of current price.
    """
    if intent.order_type == OrderType.LIMIT and intent.entry is not None:
        entry_value = _optional_float(metadata.get(f"entry_price_percentile_{horizon}"))
        if entry_value is not None:
            return entry_value
    return _optional_float(metadata.get(f"price_percentile_{horizon}"))


def _pattern_reversal_chase_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> dict[str, str] | None:
    if method not in {TradeMethod.TREND_CONTINUATION, TradeMethod.BREAKOUT_FAILURE}:
        return None
    if intent.order_type not in {OrderType.MARKET, OrderType.STOP_ENTRY}:
        return None
    opposing_side = str(metadata.get("pattern_reversal_dominant_side") or "").upper()
    if opposing_side not in {Side.LONG.value, Side.SHORT.value}:
        return None
    if opposing_side == intent.side.value:
        return None
    evidence = _opposing_pattern_chase_evidence(metadata, opposing_side)
    if not evidence:
        return None
    if _operating_tf_confirms_side(intent.side, metadata):
        return None

    hedge_recovery = _is_hedge_recovery_metadata(metadata)
    long_weight = _optional_float(metadata.get("pattern_reversal_weight_long"))
    short_weight = _optional_float(metadata.get("pattern_reversal_weight_short"))
    weight_text = []
    if long_weight is not None:
        weight_text.append(f"LONG={long_weight:.1f}")
    if short_weight is not None:
        weight_text.append(f"SHORT={short_weight:.1f}")
    weights = f" ({', '.join(weight_text)})" if weight_text else ""
    return {
        "code": "PATTERN_REVERSAL_CHASE",
        "message": (
            f"{intent.pair} {intent.side.value} {intent.order_type.value} {method.value} "
            f"chases into {opposing_side} failed-break/reversal candle evidence{weights}: "
            f"{'; '.join(evidence)}. Wait for M5/M15 close-confirmed {intent.side.value} "
            "BOS/CHOCH, or use a retest LIMIT instead of chasing the failed side."
        ),
        "severity": _same_side_chase_severity(intent, hedge_recovery=hedge_recovery),
    }


def _opposing_pattern_chase_evidence(metadata: dict[str, Any], opposing_side: str) -> list[str]:
    raw_signals = metadata.get("pattern_signals")
    if not isinstance(raw_signals, list):
        return []
    evidence: list[str] = []
    for raw in raw_signals:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("side") or "").upper() != opposing_side:
            continue
        if not bool(raw.get("chase_block_evidence")):
            continue
        name = str(raw.get("name") or "pattern")
        timeframe = str(raw.get("timeframe") or "?")
        rationale = str(raw.get("rationale") or "").strip()
        if rationale:
            evidence.append(f"{timeframe} {name}: {rationale}")
        else:
            evidence.append(f"{timeframe} {name}")
        if len(evidence) >= 3:
            break
    return evidence


def _entry_range_position(entry: float, tf_data: dict[str, Any]) -> float | None:
    support = _optional_float(tf_data.get("nearest_support"))
    resistance = _optional_float(tf_data.get("nearest_resistance"))
    if support is None or resistance is None or resistance <= support:
        return None
    return (entry - support) / (resistance - support)


def _forecast_direction_conflict_issue(intent: OrderIntent, metadata: dict[str, Any]) -> dict[str, str] | None:
    direction = str(metadata.get("forecast_direction") or "").upper()
    if direction not in {"UP", "DOWN"}:
        return None
    confidence = _optional_float(metadata.get("forecast_confidence"))
    forecast_side = Side.LONG.value if direction == "UP" else Side.SHORT.value
    if forecast_side == intent.side.value:
        return None
    min_confidence = _forecast_live_min_confidence(metadata)
    if confidence is None or confidence < min_confidence:
        if not _forecast_supported_opposite_side_blocks(
            metadata,
            forecast_side=forecast_side,
            min_confidence=min_confidence,
        ):
            return None
    target = metadata.get("forecast_target_price")
    invalidation = metadata.get("forecast_invalidation_price")
    extra = []
    if target is not None:
        extra.append(f"target={target}")
    if invalidation is not None:
        extra.append(f"invalidation={invalidation}")
    tail = f" ({', '.join(extra)})" if extra else ""
    return {
        "code": "FORECAST_DIRECTION_CONFLICT",
        "message": (
            f"{intent.pair} {intent.side.value} conflicts with current pair forecast "
            f"{direction} conf={confidence:.2f}; only {forecast_side} lanes may become LIVE_READY"
            f" while this forecast is fresh{tail}."
        ),
        "severity": "BLOCK",
    }


def _forecast_supported_opposite_side_blocks(
    metadata: dict[str, Any],
    *,
    forecast_side: str,
    min_confidence: float,
) -> bool:
    """Block the opposite side when audited projection supports the forecast.

    The support path may be insufficient to authorize the forecast side when
    calibrated confidence is deeply below the live floor. It still must stop the
    opposite lane: otherwise a news/liquidity-supported DOWN forecast can allow
    a LONG dry-run candidate to pass simply because final forecast calibration
    was pessimistic.
    """
    direction = str(metadata.get("forecast_direction") or "").upper()
    expected_side = Side.LONG.value if direction == "UP" else Side.SHORT.value if direction == "DOWN" else None
    if expected_side != forecast_side:
        return False
    raw_confidence = _optional_float(metadata.get("forecast_raw_confidence"))
    support_floor = max(0.0, min_confidence - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL)
    if raw_confidence is None or raw_confidence < support_floor:
        return False
    chart_direction_bias = str(metadata.get("chart_direction_bias") or "").upper()
    if chart_direction_bias and chart_direction_bias != forecast_side:
        return False
    support = _forecast_market_support_payload(metadata.get("forecast_market_support"))
    if not bool(support.get("ok")):
        return False
    support_direction = str(support.get("direction") or "").upper()
    if support_direction and support_direction != direction:
        return False
    if bool(support.get("bootstrap_projection_support")):
        return True
    aligned_count = _optional_int(support.get("aligned_projection_count")) or 0
    if aligned_count <= 0:
        return False
    samples = _optional_int(support.get("best_samples")) or 0
    if samples < FORECAST_MARKET_SUPPORT_MIN_SAMPLES:
        return False
    hit_rate = _optional_float(support.get("best_hit_rate")) or 0.0
    return hit_rate >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE


def _trend_timeframe_support_count(metadata: dict[str, Any], side: Side) -> int:
    expected = "TREND_UP" if side == Side.LONG else "TREND_DOWN"
    seen: set[str] = set()
    trend_timeframes = metadata.get("trend_timeframes")
    if isinstance(trend_timeframes, list):
        for raw in trend_timeframes:
            text = str(raw or "").upper()
            if ":" not in text:
                continue
            timeframe, classification = text.split(":", 1)
            if timeframe in {"H1", "H4", "D"} and classification == expected:
                seen.add(timeframe)
    tf_map = metadata.get("tf_regime_map")
    if isinstance(tf_map, dict):
        for timeframe in ("H1", "H4", "D"):
            tf_data = tf_map.get(timeframe)
            if not isinstance(tf_data, dict):
                continue
            classification = str(
                tf_data.get("classification") or tf_data.get("regime") or tf_data.get("state") or ""
            ).upper()
            if classification == expected:
                seen.add(timeframe)
    return len(seen)


def _has_higher_tf_confirmation_data(metadata: dict[str, Any]) -> bool:
    if isinstance(metadata.get("trend_timeframes"), list):
        return True
    tf_map = metadata.get("tf_regime_map")
    return isinstance(tf_map, dict) and any(timeframe in tf_map for timeframe in ("H1", "H4", "D"))


def _weak_forecast_trend_continuation_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
    *,
    confidence: float | None,
    min_confidence: float,
) -> dict[str, str] | None:
    if method != TradeMethod.TREND_CONTINUATION:
        return None
    if not metadata.get("forecast_seed"):
        return None
    if confidence is None or confidence >= min_confidence:
        return None
    if not _has_higher_tf_confirmation_data(metadata):
        return None
    support_count = _trend_timeframe_support_count(metadata, intent.side)
    if support_count >= 2:
        return None
    return {
        "code": "FORECAST_TREND_CONTINUATION_HIGHER_TF_REQUIRED_FOR_LIVE",
        "message": (
            f"{intent.pair} {intent.side.value} forecast-first TREND_CONTINUATION has weak calibrated "
            f"confidence {confidence:.2f} < {min_confidence:.2f} and only {support_count} aligned H1/H4/D "
            "trend timeframe(s); keep the STOP as dry-run until at least two higher timeframes confirm "
            "the continuation side or the calibrated forecast clears the live floor."
        ),
        "severity": "WARN",
    }


def _forecast_live_readiness_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> dict[str, str] | None:
    if not _require_forecast_for_live_active():
        return None
    direction = str(metadata.get("forecast_direction") or "").upper()
    confidence = _optional_float(metadata.get("forecast_confidence"))
    min_confidence = _forecast_live_min_confidence(metadata)
    recovery_reversal_override = _reversal_recovery_chart_forecast_override(intent, metadata)
    if direction not in {"UP", "DOWN", "RANGE"}:
        if recovery_reversal_override:
            return None
        if direction:
            return {
                "code": "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                "message": (
                    f"{intent.pair} {intent.side.value} current pair forecast is {direction} "
                    f"conf={0.0 if confidence is None else confidence:.2f}; fresh entries need "
                    "an executable UP/DOWN/RANGE prediction before they can become LIVE_READY."
                ),
                "severity": "WARN",
            }
        return {
            "code": "FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
            "message": (
                f"{intent.pair} {intent.side.value} has no fresh executable pair forecast; "
                "fresh entries must be predicted before they can become LIVE_READY."
            ),
            "severity": "WARN",
        }
    if confidence is None or confidence < min_confidence:
        if recovery_reversal_override:
            return None
        weak_trend_issue = _weak_forecast_trend_continuation_issue(
            intent,
            metadata,
            method,
            confidence=confidence,
            min_confidence=min_confidence,
        )
        if weak_trend_issue is not None:
            return weak_trend_issue
        if _forecast_market_support_override(intent, metadata, min_confidence=min_confidence):
            return None
        return {
            "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
            "message": (
                f"{intent.pair} {intent.side.value} forecast {direction} confidence "
                f"{0.0 if confidence is None else confidence:.2f} < {min_confidence:.2f}; "
                "do not trade a weak prediction just to satisfy campaign exposure."
            ),
            "severity": "WARN",
        }
    if (
        direction == "RANGE"
        and method != TradeMethod.RANGE_ROTATION
        and not (
            _is_hedge_recovery_metadata(metadata)
            and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
        )
    ):
        return {
            "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "message": (
                f"{intent.pair} {intent.side.value} has a RANGE forecast; only executable "
                "RANGE_ROTATION rail geometry may become LIVE_READY from a RANGE prediction."
            ),
            "severity": "WARN",
        }
    weak_calibration_issue = _forecast_directional_hit_rate_issue(intent, metadata, direction=direction)
    if weak_calibration_issue is not None:
        return weak_calibration_issue
    return None


def _forecast_directional_hit_rate_issue(
    intent: OrderIntent,
    metadata: dict[str, Any],
    *,
    direction: str,
) -> dict[str, str] | None:
    if direction not in {"UP", "DOWN"}:
        return None
    expected_side = Side.LONG.value if direction == "UP" else Side.SHORT.value
    if intent.side.value != expected_side:
        return None
    support = _forecast_market_support_payload(metadata.get("forecast_market_support"))
    hit_rate = _optional_float(metadata.get("forecast_directional_hit_rate"))
    if hit_rate is None:
        hit_rate = _optional_float(support.get("directional_hit_rate"))
    samples = _optional_int(metadata.get("forecast_directional_samples")) or 0
    if samples <= 0:
        samples = _optional_int(support.get("directional_samples")) or 0
    if hit_rate is None or samples < FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES:
        return None
    if hit_rate >= FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE:
        return None
    calibration_name = str(
        metadata.get("forecast_directional_calibration_name")
        or support.get("directional_calibration_name")
        or "directional_forecast"
    )
    return {
        "code": "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE",
        "message": (
            f"{intent.pair} {intent.side.value} forecast {direction} bucket "
            f"{calibration_name} hit_rate={hit_rate:.2f} over {samples} sample(s) is below "
            f"{FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE:.2f}; keep this forecast as dry-run "
            "until the calibrated direction recovers or independent live evidence replaces it."
        ),
        "severity": "WARN",
    }


def _forecast_watch_only_issue(intent: OrderIntent, metadata: dict[str, Any]) -> dict[str, str] | None:
    if not metadata.get("forecast_watch_only"):
        return None
    if _forecast_market_support_override(
        intent,
        metadata,
        min_confidence=_forecast_live_min_confidence(metadata),
    ):
        return None
    direction = str(metadata.get("forecast_direction") or "").upper() or "UNKNOWN"
    confidence = _optional_float(metadata.get("forecast_confidence"))
    reason = str(metadata.get("forecast_watch_only_reason") or "").strip()
    reason_tail = f" Reason: {reason}" if reason else ""
    return {
        "code": "FORECAST_WATCH_ONLY",
        "message": (
            f"{intent.pair} {intent.side.value} is a watch-only forecast candidate "
            f"({direction} conf={0.0 if confidence is None else confidence:.2f}); "
            "dry-run geometry is exposed for review, but live send is blocked until "
            f"a fresh calibrated forecast clears the entry floor.{reason_tail}"
        ),
        "severity": "WARN",
    }


def _forecast_market_support_override(
    intent: OrderIntent,
    metadata: dict[str, Any],
    *,
    min_confidence: float,
) -> bool:
    return _forecast_market_support_allows_side(
        intent.side.value,
        metadata,
        min_confidence=min_confidence,
        order_type=intent.order_type,
    )


def _forecast_watch_live_override_receipt(
    lane: dict[str, Any],
    side: Side,
    order_type: OrderType,
    *,
    chart_context: dict[str, Any] | None,
    geometry_metadata: dict[str, Any] | None,
) -> tuple[str, str | None]:
    original = str(lane.get("required_receipt") or "")
    if not bool(lane.get("forecast_watch_only")):
        return original, None
    source: dict[str, Any] = {}
    source.update(chart_context or {})
    source.update(geometry_metadata or {})
    source.update(
        {
            "forecast_direction": lane.get("forecast_direction"),
            "forecast_confidence": lane.get("forecast_confidence"),
            "forecast_raw_confidence": lane.get("forecast_raw_confidence"),
            "forecast_market_support": lane.get("forecast_market_support"),
        }
    )
    if not _forecast_market_support_allows_side(
        side.value,
        source,
        min_confidence=_forecast_live_min_confidence(source),
        order_type=order_type,
    ):
        return original, None
    support = _forecast_market_support_payload(source.get("forecast_market_support"))
    support_reason = support.get("reason") or "audited same-direction projection support"
    receipt = (
        f"Forecast support override: {order_type.value} may be armed only because raw forecast "
        "is near the live floor and audited same-direction projection support cleared "
        f"({support_reason}). Do not convert to MARKET; refresh forecast and broker snapshot before send."
    )
    return receipt, f"forecast support override: {support_reason}"


def _forecast_market_support_has_strong_directional_signal(
    support: dict[str, Any],
    *,
    direction: str,
    forecast_horizon_min: float | None = None,
) -> bool:
    if not bool(support.get("ok")):
        return False
    if bool(support.get("bootstrap_projection_support")):
        return False
    support_direction = str(support.get("direction") or "").upper()
    if support_direction and support_direction != direction:
        return False
    aligned_count = _optional_int(support.get("aligned_projection_count")) or 0
    if aligned_count <= 0:
        return False
    for signal in support.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        if str(signal.get("direction") or "").upper() != direction:
            continue
        if not _support_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        ):
            continue
        confidence = _optional_float(signal.get("confidence")) or 0.0
        hit_rate = _optional_float(signal.get("hit_rate")) or 0.0
        samples = _optional_int(signal.get("samples")) or 0
        if (
            confidence >= FORECAST_STRONG_DIRECTIONAL_MIN_SIGNAL_CONFIDENCE
            and hit_rate >= FORECAST_STRONG_DIRECTIONAL_MIN_HIT_RATE
            and samples >= FORECAST_STRONG_DIRECTIONAL_MIN_SAMPLES
        ):
            return True
    return False


def _forecast_market_support_has_current_directional_signal(
    support: dict[str, Any],
    *,
    direction: str,
    forecast_horizon_min: float | None = None,
) -> bool:
    directional_signals = [
        signal
        for signal in support.get("signals") or []
        if isinstance(signal, dict) and str(signal.get("direction") or "").upper() == direction
    ]
    if not directional_signals:
        # Legacy/unit payloads may expose only aggregate aligned counts.
        return True
    return any(
        _support_signal_within_forecast_horizon(
            signal,
            forecast_horizon_min=forecast_horizon_min,
        )
        for signal in directional_signals
    )


def _support_signal_within_forecast_horizon(
    signal: dict[str, Any],
    *,
    forecast_horizon_min: float | None,
) -> bool:
    if forecast_horizon_min is None or forecast_horizon_min <= 0:
        return True
    lead_time = _optional_float(signal.get("lead_time_min"))
    if lead_time is None:
        return True
    return max(0.0, lead_time) <= forecast_horizon_min


def _forecast_directional_bucket_is_known_weak(
    metadata: dict[str, Any],
    support: dict[str, Any],
) -> bool:
    """Return true when the current direction bucket is audited weak.

    EITHER/timing signals predict expansion timing, not side. If the same
    pair/direction/regime forecast bucket is already below the live hit-rate
    floor, timing-only support must not rescue the lane; only an audited
    same-direction projection can replace the weak final detector.
    """
    hit_rate = _optional_float(metadata.get("forecast_directional_hit_rate"))
    if hit_rate is None:
        hit_rate = _optional_float(support.get("directional_hit_rate"))
    samples = _optional_int(metadata.get("forecast_directional_samples")) or 0
    if samples <= 0:
        samples = _optional_int(support.get("directional_samples")) or 0
    return (
        hit_rate is not None
        and samples >= FORECAST_DIRECTIONAL_LIVE_MIN_SAMPLES
        and hit_rate < FORECAST_DIRECTIONAL_LIVE_MIN_HIT_RATE
    )


def _forecast_market_support_allows_side(
    side: str | None,
    source: Any,
    *,
    min_confidence: float,
    order_type: OrderType | None = None,
) -> bool:
    if side not in {Side.LONG.value, Side.SHORT.value}:
        return False
    if isinstance(source, dict):
        direction = str(source.get("forecast_direction") or "").upper()
        confidence = _optional_float(source.get("forecast_confidence"))
        raw_confidence = _optional_float(source.get("forecast_raw_confidence"))
        forecast_horizon_min = _optional_float(source.get("forecast_horizon_min"))
        support = _forecast_market_support_payload(source.get("forecast_market_support"))
        chart_direction_bias = str(source.get("chart_direction_bias") or "").upper()
    else:
        direction = str(getattr(source, "direction", "") or "").upper()
        confidence = _optional_float(getattr(source, "confidence", None))
        raw_confidence = _optional_float(getattr(source, "raw_confidence", None))
        forecast_horizon_min = _optional_float(getattr(source, "horizon_min", None))
        support = _forecast_market_support_payload(getattr(source, "market_support", None))
        chart_direction_bias = ""
    expected_side = Side.LONG.value if direction == "UP" else Side.SHORT.value if direction == "DOWN" else None
    if expected_side != side:
        return False
    if confidence is None:
        return False
    if chart_direction_bias and chart_direction_bias != side:
        return False
    samples = _optional_int(support.get("best_samples")) or 0
    aligned_count = _optional_int(support.get("aligned_projection_count")) or 0
    timing_count = _optional_int(support.get("timing_projection_count")) or 0
    hit_rate = _optional_float(support.get("best_hit_rate")) or 0.0
    aligned_samples = _optional_int(support.get("best_aligned_samples"))
    if (aligned_samples is None or aligned_samples <= 0) and aligned_count > 0:
        aligned_samples = samples
    aligned_hit_rate = _optional_float(support.get("best_aligned_hit_rate"))
    if aligned_hit_rate is None and aligned_count > 0:
        aligned_hit_rate = hit_rate
    timing_samples = _optional_int(support.get("best_timing_samples"))
    if (timing_samples is None or timing_samples <= 0) and timing_count > 0:
        timing_samples = samples
    timing_hit_rate = _optional_float(support.get("best_timing_hit_rate"))
    if timing_hit_rate is None and timing_count > 0:
        timing_hit_rate = hit_rate
    support_floor = max(0.0, min_confidence - FORECAST_MARKET_SUPPORT_MAX_CONFIDENCE_SHORTFALL)
    # Timing-evidence breakout stop (AGENT_CONTRACT §8, 2026-06-10).
    # Projection-ledger truth (87k scored rows): volatility-timing detectors
    # are the system's strongest predictions (bb_squeeze_expansion 82-88%
    # n≈17.8k, session_expansion 77-91%) while the aggregate directional
    # forecast reaches its target only ~43-50%. A resting STOP-ENTRY beyond
    # the breakout rail only fills when the market itself trades through the
    # level, so direction is proven by price — the weak aggregate confidence
    # is the wrong gate for that shape. Requirements, all fail-closed:
    #   - order_type is STOP-ENTRY (market-proof entry; MARKET/LIMIT keep the
    #     near-miss confidence floor below),
    #   - the lane side matches the pair forecast direction (§5 alignment is
    #     preserved — this path never authorizes fading the forecast),
    #   - the structural chart lean (confluence score_balance) actively
    #     agrees with the side (TIED or missing lean does not qualify),
    #   - a current EITHER timing signal passed the audited
    #     FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE + sample floors
    #     (timing_count is only populated from signals that cleared both).
    # Spread, RR, exhaustion, telemetry, strategy-profile, and gateway checks
    # all still run after this; it only lifts a near-miss forecast-confidence
    # veto. It must not turn a genuinely weak directional forecast into
    # LIVE_READY from a timing-only EITHER signal; timing predicts movement, not
    # side, and those entries are the exact reverse-first failure mode this gate
    # is meant to avoid.
    breakout_proof = (
        order_type == OrderType.STOP_ENTRY
        and bool(chart_direction_bias)
        and chart_direction_bias == side
    )
    timing_evidence = (
        timing_count > 0
        and (timing_samples or 0) >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
        and (timing_hit_rate or 0.0) >= FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE
    )
    known_weak_direction_bucket = _forecast_directional_bucket_is_known_weak(
        source if isinstance(source, dict) else {},
        support,
    )
    if _forecast_support_override_stop_entry_chases_range_edge(
        source,
        side=side,
        direction=direction,
        order_type=order_type,
    ):
        return False
    if breakout_proof and timing_evidence and confidence >= support_floor and not known_weak_direction_bucket:
        return True
    strong_directional_floor = max(
        FORECAST_STRONG_DIRECTIONAL_CALIBRATED_FLOOR,
        min_confidence - FORECAST_STRONG_DIRECTIONAL_MAX_CONFIDENCE_SHORTFALL,
    )
    strong_directional_raw_floor = max(
        FORECAST_BOOTSTRAP_RAW_CONFIDENCE_MIN,
        min_confidence - FORECAST_STRONG_DIRECTIONAL_RAW_MAX_CONFIDENCE_SHORTFALL,
    )
    strong_directional_projection = (
        breakout_proof
        and raw_confidence is not None
        and raw_confidence >= strong_directional_raw_floor
        and confidence >= strong_directional_floor
        and _forecast_market_support_has_strong_directional_signal(
            support,
            direction=direction,
            forecast_horizon_min=forecast_horizon_min,
        )
    )
    if strong_directional_projection:
        return True
    if confidence < support_floor:
        return False
    if not bool(support.get("ok")):
        return False
    if bool(support.get("bootstrap_projection_support")):
        return raw_confidence is not None and raw_confidence >= min_confidence
    if aligned_count > 0:
        if not _forecast_market_support_has_current_directional_signal(
            support,
            direction=direction,
            forecast_horizon_min=forecast_horizon_min,
        ):
            return False
        return (
            (aligned_samples or 0) >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
            and (aligned_hit_rate or 0.0) >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE
        )
    return (
        timing_count > 0
        and not known_weak_direction_bucket
        and raw_confidence is not None
        and raw_confidence >= min_confidence
        and (timing_samples or 0) >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
        and (timing_hit_rate or 0.0) >= FORECAST_MARKET_SUPPORT_MIN_TIMING_HIT_RATE
    )


def _forecast_support_override_stop_entry_chases_range_edge(
    source: Any,
    *,
    side: str | None,
    direction: str,
    order_type: OrderType | None,
) -> bool:
    if order_type != OrderType.STOP_ENTRY or side not in {Side.LONG.value, Side.SHORT.value}:
        return False
    if not isinstance(source, dict):
        return False
    expected_trend = "TREND_UP" if side == Side.LONG.value else "TREND_DOWN"
    m5_regime = str(source.get("m5_regime") or "").upper()
    if m5_regime == expected_trend:
        return False
    range_phase = str(source.get("range_phase") or "").upper()
    breakout_direction = str(source.get("range_breakout_direction") or "").upper()
    if range_phase in {"BREAKOUT_UP", "BREAKOUT_DOWN"} and breakout_direction == direction:
        return False

    tf_map = source.get("tf_regime_map")
    if not isinstance(tf_map, dict):
        return False
    edge = FORECAST_SUPPORT_RANGE_EDGE_CHASE_POSITION
    lower_edge = 1.0 - edge
    for timeframe in ("M5", "M15"):
        tf_data = tf_map.get(timeframe)
        if not isinstance(tf_data, dict):
            continue
        classification = str(
            tf_data.get("classification") or tf_data.get("regime") or tf_data.get("state") or ""
        ).upper()
        if classification == expected_trend:
            continue
        position = _optional_float(tf_data.get("range_position"))
        if position is None:
            continue
        if side == Side.LONG.value and position >= edge:
            return True
        if side == Side.SHORT.value and position <= lower_edge:
            return True
    return False


def _fresh_entry_live_reward_risk_issue(intent: OrderIntent, metrics: Any | None) -> dict[str, str] | None:
    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "NEW").upper() != "NEW":
        return None
    if bool(metadata.get("hedge_recovery")):
        return None
    reward_risk = _optional_float(getattr(metrics, "reward_risk", None))
    method = intent.market_context.method if intent.market_context else None
    if (
        metadata.get("forecast_seed")
        and method == TradeMethod.TREND_CONTINUATION
        and reward_risk is not None
        and reward_risk < FORECAST_SEED_TREND_MIN_LIVE_REWARD_RISK
    ):
        return {
            "code": "FORECAST_TREND_CONTINUATION_REWARD_RISK_TOO_LOW",
            "message": (
                f"{intent.pair} {intent.side.value} forecast-first TREND_CONTINUATION reward/risk "
                f"{reward_risk:.2f}x < {FORECAST_SEED_TREND_MIN_LIVE_REWARD_RISK:.2f}x; keep as dry-run "
                "because weak continuation forecasts need enough payoff to absorb invalidation risk."
            ),
            "severity": "WARN",
        }
    if reward_risk is None or reward_risk > 0.0:
        return None
    return {
        "code": "FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE",
        "message": (
            f"{intent.pair} {intent.side.value} fresh entry reward/risk {reward_risk:.2f}x "
            "is not positive; keep as dry-run until TP/entry geometry is reward-side."
        ),
        "severity": "BLOCK",
    }


def _reversal_recovery_chart_forecast_override(intent: OrderIntent, metadata: dict[str, Any]) -> bool:
    if not (
        _is_hedge_recovery_metadata(metadata)
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    ):
        return False
    if intent.order_type != OrderType.MARKET:
        return False
    side = intent.side.value
    balance = str(metadata.get("chart_score_balance") or "").upper()
    if side not in balance:
        return False
    score_gap = _optional_float(metadata.get("chart_score_gap"))
    if score_gap is None:
        confluence = metadata.get("confluence")
        if isinstance(confluence, dict):
            score_gap = _optional_float(confluence.get("score_gap"))
    if score_gap is None or abs(score_gap) < 0.15:
        return False

    dominant = str(metadata.get("pattern_reversal_dominant_side") or "").upper()
    if dominant != side:
        return False
    side_key = "pattern_reversal_weight_long" if side == Side.LONG.value else "pattern_reversal_weight_short"
    other_key = "pattern_reversal_weight_short" if side == Side.LONG.value else "pattern_reversal_weight_long"
    side_weight = _optional_float(metadata.get(side_key))
    other_weight = _optional_float(metadata.get(other_key))
    if side_weight is None or side_weight < 20.0:
        return False
    if other_weight is not None and side_weight - other_weight < 6.0:
        return False

    expected_regime = "TREND_UP" if side == Side.LONG.value else "TREND_DOWN"
    trend_timeframes = [str(item).upper() for item in metadata.get("trend_timeframes") or []]
    if trend_timeframes and not any(expected_regime in item for item in trend_timeframes):
        return False
    return True


def _forecast_live_min_confidence(metadata: dict[str, Any]) -> float:
    entry_min = _forecast_seed_min_confidence()
    if _is_range_rotation_forecast_metadata(metadata):
        return min(entry_min, FORECAST_RANGE_ROTATION_MIN_CONFIDENCE)
    if (
        _is_hedge_recovery_metadata(metadata)
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    ):
        return min(entry_min, RECOVERY_HEDGE_DEFAULT_CONVICTION_SCALE)
    return max(entry_min, FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE)


def _is_range_rotation_forecast_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("forecast_direction") or "").upper() == "RANGE"
        and str(metadata.get("geometry_model") or "").upper() == "RANGE_RAIL_LIMIT"
        and bool(metadata.get("range_tp_is_inside_box"))
        and bool(metadata.get("range_sl_outside_box"))
    )


def _telemetry_live_readiness_issues(
    intent: OrderIntent,
    metadata: dict[str, Any],
    snapshot: BrokerSnapshot,
    validation_time_utc: datetime,
    *,
    cache: _TelemetryLiveReadinessCache | None = None,
    data_root: Path | None = None,
) -> tuple[dict[str, str], ...]:
    if not _require_telemetry_for_live_active():
        return ()

    issues: list[dict[str, str]] = []
    data_root = data_root or (ROOT / "data")
    direction = str(metadata.get("forecast_direction") or "").upper()
    confidence = _optional_float(metadata.get("forecast_confidence"))
    recovery_reversal_override = _reversal_recovery_chart_forecast_override(intent, metadata)
    if direction not in {"UP", "DOWN", "RANGE"} and not recovery_reversal_override:
        if direction:
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} forecast telemetry is {direction}; "
                        "the forecast trail exists, but it cannot authorize live entry until "
                        "the current forecast resolves to UP/DOWN/RANGE."
                    ),
                )
            )
        else:
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} has no executable forecast metadata; "
                        "telemetry cannot prove what the live entry is predicting."
                    ),
                )
            )
    expected_cycle_id = str(metadata.get("forecast_cycle_id") or "")
    latest_is_current = False
    projection_cycle_id = ""
    quote_fresh, quote_issue = _forecast_telemetry_quote_fresh_for_live(
        intent.pair,
        snapshot,
        validation_time_utc=validation_time_utc,
    )
    if not quote_fresh:
        issues.append(
            _telemetry_issue(
                "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
                quote_issue,
            )
        )
    elif cache is not None:
        latest = cache.latest_forecasts_by_pair.get(intent.pair)
        cycle_matched = (
            cache.forecasts_by_pair_cycle.get((intent.pair, expected_cycle_id))
            if expected_cycle_id
            else None
        )
        audit_forecast = cycle_matched or latest
    else:
        latest = _latest_forecast_history_for_pair(intent.pair, data_root=data_root)
        cycle_matched = _forecast_history_for_pair_cycle(
            intent.pair,
            expected_cycle_id,
            data_root=data_root,
        ) if expected_cycle_id else None
        audit_forecast = cycle_matched or latest

    if not quote_fresh:
        pass
    elif audit_forecast is None:
        issues.append(
            _telemetry_issue(
                "TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE",
                (
                    f"{intent.pair} {intent.side.value} cannot become LIVE_READY because "
                    "forecast_history.jsonl has no current row for this pair; live entries "
                    "must leave an auditable forecast trail before the broker can receive them."
                ),
            )
        )
    else:
        latest_ts = _parse_telemetry_time(audit_forecast.get("timestamp_utc"))
        snapshot_ts = _ensure_utc(getattr(snapshot, "fetched_at_utc", None))
        latest_cycle_id = str(audit_forecast.get("cycle_id") or "")
        latest_matches_intent_cycle = bool(expected_cycle_id and latest_cycle_id == expected_cycle_id)
        projection_cycle_id = latest_cycle_id
        if latest_ts is None:
            latest_text = audit_forecast.get("timestamp_utc") or "unknown"
            snapshot_text = snapshot_ts.isoformat() if snapshot_ts is not None else "unknown"
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} forecast telemetry is stale "
                        f"(forecast_history={latest_text}, snapshot={snapshot_text}); "
                        "refresh the forecast from the same broker snapshot before live entry."
                    ),
                )
            )
        elif expected_cycle_id and latest_cycle_id != expected_cycle_id:
            latest_text = latest_cycle_id or "missing"
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} forecast telemetry cycle_id "
                        f"{latest_text} does not match intent forecast_cycle_id "
                        f"{expected_cycle_id}; refresh the forecast from the same broker snapshot before live entry."
                    ),
                )
            )
        elif snapshot_ts is not None and latest_ts < snapshot_ts and not latest_matches_intent_cycle:
            latest_text = audit_forecast.get("timestamp_utc") or "unknown"
            snapshot_text = snapshot_ts.isoformat() if snapshot_ts is not None else "unknown"
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} forecast telemetry is stale "
                        f"(forecast_history={latest_text}, snapshot={snapshot_text}); "
                        "refresh the forecast from the same broker snapshot before live entry."
                    ),
                )
            )
        else:
            latest_is_current = True

        latest_direction = str(audit_forecast.get("direction") or "").upper()
        if direction in {"UP", "DOWN", "RANGE"} and latest_direction != direction:
            latest_is_current = False
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} intent forecast {direction} "
                        f"does not match latest forecast_history direction {latest_direction or 'missing'}; "
                        "do not send a broker order when the executable lane and audit trail disagree."
                    ),
                )
            )
        latest_confidence = _optional_float(audit_forecast.get("confidence"))
        if (
            direction in {"UP", "DOWN", "RANGE"}
            and confidence is not None
            and latest_confidence is not None
            and not _forecast_confidence_matches(latest_confidence, confidence)
        ):
            latest_is_current = False
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} intent forecast confidence "
                        f"{confidence:.4f} does not match latest forecast_history confidence "
                        f"{latest_confidence:.4f}; refresh before live entry."
                    ),
                )
            )

        # A later position-management or TraderBrain pass can append the same
        # pair forecast under its own cycle id while this intent is being
        # generated. Treat that as harmless only if it agrees with the exact
        # intent forecast; a newer contradictory row still blocks live use.
        if latest is not None and latest is not audit_forecast and latest_is_current:
            latest_row_ts = _parse_telemetry_time(latest.get("timestamp_utc"))
            if latest_row_ts is not None and latest_ts is not None and latest_row_ts > latest_ts:
                later_direction = str(latest.get("direction") or "").upper()
                if direction in {"UP", "DOWN", "RANGE"} and later_direction != direction:
                    latest_is_current = False
                    issues.append(
                        _telemetry_issue(
                            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
                            (
                                f"{intent.pair} {intent.side.value} newer forecast_history direction "
                                f"{later_direction or 'missing'} does not match intent forecast {direction}; "
                                "refresh before live entry."
                            ),
                        )
                    )
                later_confidence = _optional_float(latest.get("confidence"))
                if (
                    direction in {"UP", "DOWN", "RANGE"}
                    and confidence is not None
                    and later_confidence is not None
                    and not _forecast_confidence_matches(later_confidence, confidence)
                ):
                    latest_is_current = False
                    issues.append(
                        _telemetry_issue(
                            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
                            (
                                f"{intent.pair} {intent.side.value} newer forecast_history confidence "
                                f"{later_confidence:.4f} does not match intent forecast confidence "
                                f"{confidence:.4f}; refresh before live entry."
                            ),
                        )
                    )

    if latest_is_current and direction in {"UP", "DOWN", "RANGE"}:
        cycle_id = projection_cycle_id
        projection_recorded = (
            (intent.pair, cycle_id) in cache.directional_projection_keys
            if cache is not None
            else _directional_projection_recorded(
                intent.pair,
                cycle_id,
                data_root=data_root,
            )
        )
        if not cycle_id or not projection_recorded:
            issues.append(
                _telemetry_issue(
                    "TELEMETRY_DIRECTIONAL_PROJECTION_REQUIRED_FOR_LIVE",
                    (
                        f"{intent.pair} {intent.side.value} has a {direction} forecast but "
                        "projection_ledger.jsonl has no matching directional_forecast row for "
                        f"cycle_id={cycle_id or 'missing'}; the prediction must be logged for "
                        "future hit/miss calibration before live entry."
                    ),
                )
            )
        support_signal_names = _forecast_market_support_signal_names(metadata)
        if support_signal_names:
            missing_support = []
            for signal_name in support_signal_names:
                support_recorded = (
                    (intent.pair, cycle_id, signal_name) in cache.projection_signal_keys
                    if cache is not None
                    else _projection_signal_recorded(
                        intent.pair,
                        cycle_id,
                        signal_name,
                        data_root=data_root,
                    )
                )
                if not cycle_id or not support_recorded:
                    missing_support.append(signal_name)
            if missing_support:
                issues.append(
                    _telemetry_issue(
                        "TELEMETRY_MARKET_SUPPORT_PROJECTION_REQUIRED_FOR_LIVE",
                        (
                            f"{intent.pair} {intent.side.value} uses forecast market support "
                            f"{', '.join(missing_support[:3])}, but projection_ledger.jsonl has "
                            f"no matching same-cycle projection row for cycle_id={cycle_id or 'missing'}; "
                            "the support signal must be logged for future hit/miss calibration before live entry."
                        ),
                    )
                )

    expired_pending = (
        cache.expired_pending_projection_count
        if cache is not None
        else _expired_pending_projection_count(
            data_root=data_root,
            validation_time_utc=validation_time_utc,
        )
    )
    if expired_pending:
        issues.append(
            _telemetry_issue(
                "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
                (
                    f"projection_ledger.jsonl has {expired_pending} expired PENDING projection(s); "
                    "run projection verification before adding new live exposure."
                ),
            )
        )

    execution_issue = _execution_ledger_sync_live_issue(snapshot, data_root=data_root)
    if execution_issue is not None:
        issues.append(execution_issue)
    return tuple(issues)


def _forecast_telemetry_quote_fresh_for_live(
    pair: str,
    snapshot: BrokerSnapshot,
    *,
    validation_time_utc: datetime,
) -> tuple[bool, str]:
    quote = snapshot.quotes.get(pair)
    snapshot_ts = _ensure_utc(getattr(snapshot, "fetched_at_utc", None))
    validation_ts = _ensure_utc(validation_time_utc) or snapshot_ts or datetime.now(timezone.utc)
    if quote is None:
        return (
            False,
            (
                f"{pair} has no broker quote in the current snapshot; forecast telemetry "
                "cannot be recorded or matched for live entry until broker truth refreshes."
            ),
        )
    quote_ts = _ensure_utc(getattr(quote, "timestamp_utc", None))
    if quote_ts is None:
        return (
            False,
            (
                f"{pair} quote has no timestamp; forecast telemetry cannot prove the "
                "prediction was made from current tradable price truth."
            ),
        )
    if quote_ts > validation_ts:
        return True, ""
    max_age = float(RiskPolicy().max_quote_age_seconds)
    age = (validation_ts - quote_ts).total_seconds()
    if age <= max_age:
        return True, ""
    return (
        False,
        (
            f"{pair} quote is {age:.0f}s old versus the {max_age:.0f}s live freshness "
            "contract; skip forecast_history direction/confidence matching because a "
            "same-cycle forecast cannot be recorded from stale price truth."
        ),
    )


def _build_telemetry_live_readiness_cache(
    *,
    data_root: Path,
    validation_time_utc: datetime,
) -> _TelemetryLiveReadinessCache:
    latest_forecasts_by_pair: dict[str, dict[str, Any]] = {}
    forecasts_by_pair_cycle: dict[tuple[str, str], dict[str, Any]] = {}
    forecast_path = data_root / "forecast_history.jsonl"
    for item in _iter_jsonl_dicts(forecast_path):
        pair = str(item.get("pair") or "")
        if not pair:
            continue
        latest_forecasts_by_pair[pair] = item
        cycle_id = str(item.get("cycle_id") or "")
        if cycle_id:
            forecasts_by_pair_cycle[(pair, cycle_id)] = item

    directional_projection_keys: set[tuple[str, str]] = set()
    projection_signal_keys: set[tuple[str, str, str]] = set()
    expired_pending = 0
    now = _ensure_utc(validation_time_utc) or datetime.now(timezone.utc)
    projection_path = data_root / "projection_ledger.jsonl"
    for item in _iter_jsonl_dicts(projection_path):
        pair = str(item.get("pair") or "")
        cycle_id = str(item.get("cycle_id") or "")
        signal_name = str(item.get("signal_name") or "")
        if pair and cycle_id and signal_name:
            projection_signal_keys.add((pair, cycle_id, signal_name))
        if pair and cycle_id and signal_name == "directional_forecast":
            directional_projection_keys.add((pair, cycle_id))
        if str(item.get("resolution_status") or "").upper() != "PENDING":
            continue
        if _pending_projection_is_expired(item, validation_time_utc=now):
            expired_pending += 1

    return _TelemetryLiveReadinessCache(
        latest_forecasts_by_pair=latest_forecasts_by_pair,
        forecasts_by_pair_cycle=forecasts_by_pair_cycle,
        directional_projection_keys=directional_projection_keys,
        projection_signal_keys=projection_signal_keys,
        expired_pending_projection_count=expired_pending,
    )


def _forecast_confidence_matches(latest_confidence: float, intent_confidence: float) -> bool:
    return abs(latest_confidence - intent_confidence) <= FORECAST_CONFIDENCE_TELEMETRY_TOLERANCE


def _telemetry_issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "severity": "WARN"}


def _latest_forecast_history_for_pair(pair: str, *, data_root: Path) -> dict[str, Any] | None:
    path = data_root / "forecast_history.jsonl"
    latest: dict[str, Any] | None = None
    for item in _iter_jsonl_dicts(path):
        if str(item.get("pair") or "") == pair:
            latest = item
    return latest


def _forecast_history_for_pair_cycle(pair: str, cycle_id: str, *, data_root: Path) -> dict[str, Any] | None:
    if not cycle_id:
        return None
    path = data_root / "forecast_history.jsonl"
    latest: dict[str, Any] | None = None
    for item in _iter_jsonl_dicts(path):
        if str(item.get("pair") or "") == pair and str(item.get("cycle_id") or "") == cycle_id:
            latest = item
    return latest


def _directional_projection_recorded(pair: str, cycle_id: str, *, data_root: Path) -> bool:
    if not cycle_id:
        return False
    path = data_root / "projection_ledger.jsonl"
    for item in _iter_jsonl_dicts(path):
        if (
            str(item.get("pair") or "") == pair
            and str(item.get("cycle_id") or "") == cycle_id
            and str(item.get("signal_name") or "") == "directional_forecast"
        ):
            return True
    return False


def _projection_signal_recorded(pair: str, cycle_id: str, signal_name: str, *, data_root: Path) -> bool:
    if not cycle_id or not signal_name:
        return False
    path = data_root / "projection_ledger.jsonl"
    for item in _iter_jsonl_dicts(path):
        if (
            str(item.get("pair") or "") == pair
            and str(item.get("cycle_id") or "") == cycle_id
            and str(item.get("signal_name") or "") == signal_name
        ):
            return True
    return False


def _forecast_market_support_signal_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    support = _forecast_market_support_payload(metadata.get("forecast_market_support"))
    if not bool(support.get("ok")):
        return ()
    names: list[str] = []
    for item in support.get("signals") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name:
            names.append(name)
    return tuple(dict.fromkeys(names))


def _expired_pending_projection_count(*, data_root: Path, validation_time_utc: datetime) -> int:
    path = data_root / "projection_ledger.jsonl"
    now = _ensure_utc(validation_time_utc) or datetime.now(timezone.utc)
    expired = 0
    for item in _iter_jsonl_dicts(path):
        if str(item.get("resolution_status") or "").upper() != "PENDING":
            continue
        if _pending_projection_is_expired(item, validation_time_utc=now):
            expired += 1
    return expired


def _pending_projection_is_expired(item: dict[str, Any], *, validation_time_utc: datetime) -> bool:
    emitted = _parse_telemetry_time(item.get("timestamp_emitted_utc"))
    now = _ensure_utc(validation_time_utc) or datetime.now(timezone.utc)
    if emitted is None:
        return True
    if emitted > now:
        return False
    window_min = _optional_float(item.get("resolution_window_min"))
    if window_min is None or window_min <= 0:
        return True
    expiry_age_seconds = (now - emitted).total_seconds() - (window_min * 60.0)
    return expiry_age_seconds >= PROJECTION_PENDING_EXPIRY_GRACE_SECONDS


def _execution_ledger_sync_live_issue(snapshot: BrokerSnapshot, *, data_root: Path) -> dict[str, str] | None:
    account = getattr(snapshot, "account", None)
    expected = str(getattr(account, "last_transaction_id", "") or "").strip()
    if not expected:
        return None
    actual = _execution_ledger_last_transaction_id(data_root / "execution_ledger.db")
    if actual is None:
        return _telemetry_issue(
            "TELEMETRY_EXECUTION_LEDGER_STALE_FOR_LIVE",
            (
                f"execution ledger is missing last_oanda_transaction_id while broker snapshot "
                f"is at transaction {expected}; sync OANDA transactions before live entry."
            ),
        )
    if _transaction_id_is_behind(actual, expected):
        return _telemetry_issue(
            "TELEMETRY_EXECUTION_LEDGER_STALE_FOR_LIVE",
            (
                f"execution ledger last_oanda_transaction_id={actual} is behind broker "
                f"snapshot transaction {expected}; sync OANDA transactions before live entry."
            ),
        )
    return None


def _execution_ledger_last_transaction_id(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "select value from sync_state where key = ?",
                ("last_oanda_transaction_id",),
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    value = str(row[0] or "").strip()
    return value or None


def _transaction_id_is_behind(actual: str, expected: str) -> bool:
    try:
        return int(actual) < int(expected)
    except (TypeError, ValueError):
        return actual != expected


def _iter_jsonl_dicts(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    items: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    items.append(item)
    except OSError:
        return ()
    return tuple(items)


def _parse_telemetry_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_utc(parsed)


def _ensure_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_forecast_for_live_active() -> bool:
    return os.environ.get("QR_REQUIRE_FORECAST_FOR_LIVE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _require_telemetry_for_live_active() -> bool:
    return os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


# C — 2σ-equivalent extension boundary. (high_24h - low_24h) divided
# by median H1 range over the same window above this multiple means
# the pair has already exhausted >= 2× typical hourly motion in 24h.
# Same-side entries after this point chase a move that has already
# happened; spread + slippage on the late entry dominate the
# remaining edge. 2.0 is the documented 2σ boundary used in
# standard-deviation extreme detection — not a tuned market literal.
EXHAUSTION_RANGE_SIGMA_MULTIPLE = 2.0


def _method_direction_bias(metadata: dict[str, Any], method: TradeMethod | None) -> str:
    if method == TradeMethod.RANGE_ROTATION:
        m5_bias = _direction_bias_from_m5(metadata)
        if m5_bias:
            return m5_bias
    return str(metadata.get("chart_direction_bias") or "").upper()


def _trend_continuation_hard_block_reason(
    intent: OrderIntent,
    metadata: dict[str, Any],
    method: TradeMethod | None,
) -> str | None:
    if method != TradeMethod.TREND_CONTINUATION:
        return None
    side = intent.side.value
    reasons: list[str] = []

    dominant = str(metadata.get("dominant_regime_state") or "").upper()
    if _regime_points_against_side(dominant, side):
        reasons.append(f"dominant_regime_state={dominant} points against {side}")

    decisive_bias = _decisive_pair_chart_bias(metadata)
    if decisive_bias in {Side.LONG.value, Side.SHORT.value} and decisive_bias != side:
        gap = _optional_float(metadata.get("chart_score_gap"))
        gap_text = f"{gap:.3f}" if gap is not None else "unknown"
        reasons.append(
            f"decisive pair_charts bias={decisive_bias} "
            f"(|score_gap|={gap_text} >= {TREND_CONTINUATION_STRONG_BIAS_GAP:.3f})"
        )

    if not reasons:
        return None
    return (
        "trend-continuation hard gate: "
        + "; ".join(reasons)
        + ". Use the aligned trend lane, or a RANGE_ROTATION/BREAKOUT_FAILURE receipt "
        "with explicit range/failure geometry instead."
    )


def _regime_points_against_side(regime: str, side: str) -> bool:
    if not regime:
        return False
    if side == Side.LONG.value:
        return regime.startswith("TREND_DOWN") or regime.startswith("IMPULSE_DOWN")
    if side == Side.SHORT.value:
        return regime.startswith("TREND_UP") or regime.startswith("IMPULSE_UP")
    return False


def _decisive_pair_chart_bias(metadata: dict[str, Any]) -> str | None:
    balance = str(metadata.get("chart_score_balance") or "").upper()
    if balance not in {"LONG_LEAN", "SHORT_LEAN"}:
        return None
    gap = _optional_float(metadata.get("chart_score_gap"))
    if gap is None or abs(gap) < TREND_CONTINUATION_STRONG_BIAS_GAP:
        return None
    return Side.LONG.value if balance == "LONG_LEAN" else Side.SHORT.value


def _order_type_for(method: TradeMethod) -> OrderType:
    if method == TradeMethod.RANGE_ROTATION:
        return OrderType.LIMIT
    return OrderType.STOP_ENTRY


def _take_profit_execution_plan(
    *,
    pair: str,
    side: Side,
    method: TradeMethod,
    order_type: OrderType,
    quote: Quote,
    entry: float,
    tp: float,
    sl: float,
    reward_risk: float,
    execution_regime: str | None,
    chart_context: dict[str, Any] | None,
    pair_chart: dict[str, Any] | None,
    atr_pips: float | None,
    forecast_direction: object | None = None,
    forecast_confidence: float | None = None,
    forecast_target_price: float | None = None,
    hedge_recovery: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Return the virtual TP and broker-attachment metadata.

    The intent always carries a virtual TP so risk/reward validation and
    reports remain comparable. `attach_take_profit_on_fill=False` means the
    broker order omits `takeProfitOnFill`; the virtual TP becomes the first
    management target for TP rebalance / profit partial close.
    """
    attach_tp, attach_reason = _should_attach_take_profit_on_fill(
        method=method,
        order_type=order_type,
        execution_regime=execution_regime,
        chart_context=chart_context,
    )
    target_intent = "HARVEST" if attach_tp else "EXTEND"
    target_source = "RANGE_RAIL" if method == TradeMethod.RANGE_ROTATION else "ATR_RR"
    target_reason = "range geometry already anchored to opposing rail" if method == TradeMethod.RANGE_ROTATION else ""
    effective_tp = tp
    structural_target_found = False
    if method != TradeMethod.RANGE_ROTATION:
        technical_tp, technical_reason = _technical_tp_candidate(
            pair=pair,
            side=side,
            entry=entry,
            spread_pips=abs(quote.ask - quote.bid) * PIP_FACTORS[pair],
            pair_chart=pair_chart,
            intent=target_intent,
        )
        if technical_tp is not None:
            effective_tp = technical_tp
            structural_target_found = True
            target_source = f"STRUCTURAL_{target_intent}"
            target_reason = technical_reason
        else:
            target_reason = technical_reason

    pip_factor = PIP_FACTORS[pair]
    stop_pips = abs(entry - sl) * pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    harvest_target_too_far = (
        atr_pips is not None
        and atr_pips > 0
        and abs(effective_tp - entry) * pip_factor > HARVEST_TP_MAX_OPERATING_ATR_MULT * atr_pips
    )
    if (
        attach_tp
        and target_intent == "HARVEST"
        and (method == TradeMethod.BREAKOUT_FAILURE or hedge_recovery)
        and not structural_target_found
        and (harvest_target_too_far or hedge_recovery)
    ):
        fallback_tp, fallback_reason = _attached_harvest_floor_tp_candidate(
            pair=pair,
            side=side,
            entry=entry,
            stop_pips=stop_pips,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            fresh_new_entry=not hedge_recovery,
        )
        if fallback_tp is not None:
            effective_tp = fallback_tp
            target_source = "OPERATING_HARVEST_FLOOR"
            target_reason = f"{target_reason}; {fallback_reason}" if target_reason else fallback_reason
    forecast_tp, forecast_reason = _forecast_tp_candidate(
        pair=pair,
        side=side,
        entry=entry,
        forecast_direction=forecast_direction,
        forecast_confidence=forecast_confidence,
        forecast_target_price=forecast_target_price,
    )
    if forecast_tp is not None:
        forecast_ok, forecast_gate_reason = _forecast_tp_respects_execution_floor(
            method=method,
            execution_regime=execution_regime,
            entry=entry,
            target=forecast_tp,
            stop_pips=stop_pips,
            spread_pips=spread_pips,
            pip_factor=pip_factor,
            pair=pair,
            fresh_new_entry=not hedge_recovery,
        )
        if forecast_ok and _tp_closer_to_entry(entry, forecast_tp, effective_tp):
            effective_tp = forecast_tp
            target_source = f"FORECAST_CAPPED_{target_source}"
            target_reason = f"{forecast_reason}; capped {target_reason or target_source}"
        elif not forecast_ok:
            target_reason = f"{target_reason}; forecast target skipped: {forecast_gate_reason}"
        else:
            target_reason = f"{target_reason}; forecast target farther: {forecast_reason}"

    target_pips = abs(effective_tp - entry) * pip_factor
    virtual_rr = target_pips / stop_pips if stop_pips > 0 else 0.0
    opportunity_mode, opportunity_reason = _opportunity_mode_from_execution_plan(
        method=method,
        target_intent=target_intent,
        reward_risk=virtual_rr,
    )
    metadata = {
        "opportunity_mode": opportunity_mode,
        "opportunity_mode_reason": opportunity_reason,
        "opportunity_mode_reward_risk": round(virtual_rr, 3),
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP" if attach_tp else "RUNNER_NO_BROKER_TP",
        "attach_take_profit_on_fill": attach_tp,
        "tp_attach_reason": attach_reason,
        "tp_target_source": target_source,
        "tp_target_reason": target_reason,
        "tp_target_intent": target_intent,
        "virtual_take_profit": _round_price(pair, effective_tp),
        "virtual_take_profit_reward_risk": round(virtual_rr, 3),
        "tp_target_distance_pips": round(target_pips, 3),
        "tp_requested_reward_risk": reward_risk,
        "tp_atr_pips": atr_pips,
    }
    return _round_price(pair, effective_tp), metadata


def _opportunity_mode_from_execution_plan(
    *,
    method: TradeMethod,
    target_intent: str,
    reward_risk: float,
) -> tuple[str, str]:
    intent = str(target_intent or "").upper()
    method_name = method.value if isinstance(method, TradeMethod) else str(method or "").upper()
    if any(token in intent for token in ("RUNNER", "TRAIL", "EXTEND", "SWING", "HOLD", "ADD")):
        return "RUNNER", f"tp_target_intent={intent}"
    if any(token in intent for token in ("HARVEST", "SCALP", "QUICK")):
        return "HARVEST", f"tp_target_intent={intent}"
    if reward_risk >= OPPORTUNITY_MODE_RUNNER_REWARD_RISK_MIN:
        return "RUNNER", f"reward_risk>={OPPORTUNITY_MODE_RUNNER_REWARD_RISK_MIN:.2f}"
    if 0.0 < reward_risk <= OPPORTUNITY_MODE_HARVEST_REWARD_RISK_MAX:
        return "HARVEST", f"reward_risk<={OPPORTUNITY_MODE_HARVEST_REWARD_RISK_MAX:.2f}"
    if method_name == TradeMethod.TREND_CONTINUATION.value:
        return "RUNNER", "method=TREND_CONTINUATION"
    if method_name in {TradeMethod.RANGE_ROTATION.value, TradeMethod.BREAKOUT_FAILURE.value}:
        return "HARVEST", f"method={method_name}"
    return "BALANCED", "no dominant harvest/runner signal"


def _attached_harvest_floor_tp_candidate(
    *,
    pair: str,
    side: Side,
    entry: float,
    stop_pips: float,
    spread_pips: float,
    atr_pips: float | None,
    fresh_new_entry: bool,
) -> tuple[float | None, str]:
    """Fallback for attached HARVEST TP when no structural anchor is usable.

    Failed-break / recovery entries are not runners. If structural_tp_target
    cannot provide a nearby market level, use the smallest reward-side target
    that still clears the active range/hedge reward floor and the live spread
    floor, bounded by the operating ATR cap. This keeps initial broker TP from
    inheriting a distant SL-free virtual risk distance.
    """
    if atr_pips is None or atr_pips <= 0 or stop_pips <= 0:
        return None, "attached HARVEST fallback skipped: missing operating ATR or stop distance"
    policy = RiskPolicy()
    rr_floor_distance_pips = policy.range_min_reward_risk * stop_pips
    if fresh_new_entry:
        rr_floor_distance_pips = max(rr_floor_distance_pips, _fresh_entry_live_floor_distance_pips(pair, stop_pips))
    floor_distance_pips = max(
        rr_floor_distance_pips,
        policy.min_target_spread_multiple * spread_pips,
    )
    max_distance_pips = HARVEST_TP_MAX_OPERATING_ATR_MULT * atr_pips
    if floor_distance_pips > max_distance_pips:
        return None, (
            f"attached HARVEST fallback skipped: minimum acceptable target "
            f"{floor_distance_pips:.1f}pip exceeds {HARVEST_TP_MAX_OPERATING_ATR_MULT:.1f}× "
            f"operating ATR {atr_pips:.1f}pip"
        )

    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    if side == Side.LONG:
        candidate = entry + floor_distance_pips * pip
    else:
        candidate = entry - floor_distance_pips * pip
    candidate = _round_price(pair, candidate)

    # Rounding to broker precision can shave a fraction of a pip; nudge one
    # price tick outward only if needed to keep the active RR floor true.
    rounded_distance = abs(candidate - entry) * pip_factor
    if rounded_distance < floor_distance_pips:
        tick = _broker_price_tick_pips(pair) / pip_factor
        candidate = _round_price(pair, candidate + tick if side == Side.LONG else candidate - tick)
        rounded_distance = abs(candidate - entry) * pip_factor

    rr_label = "fresh_live_rr_floor" if fresh_new_entry else "range_rr_floor"
    rr_floor = rounded_distance / stop_pips if stop_pips > 0 else 0.0
    return candidate, (
        f"attached HARVEST structural anchor missing; using minimum acceptable "
        f"operating target {rounded_distance:.1f}pip "
        f"({rr_label}={rr_floor:.2f}, "
        f"max={max_distance_pips:.1f}pip/{HARVEST_TP_MAX_OPERATING_ATR_MULT:.1f}×ATR)"
    )


def _forecast_tp_candidate(
    *,
    pair: str,
    side: Side,
    entry: float,
    forecast_direction: object | None,
    forecast_confidence: float | None,
    forecast_target_price: float | None,
) -> tuple[float | None, str]:
    if forecast_target_price is None or forecast_confidence is None:
        return None, "forecast target missing"
    if forecast_confidence < _forecast_seed_min_confidence():
        return None, f"forecast confidence {forecast_confidence:.2f} below TP threshold"
    direction = str(forecast_direction or "").upper()
    forecast_side = Side.LONG if direction == "UP" else Side.SHORT if direction == "DOWN" else None
    if forecast_side != side:
        return None, f"forecast direction {direction or 'missing'} does not match {side.value}"
    target = _round_price(pair, float(forecast_target_price))
    if not _target_on_reward_side(side, entry, target):
        return None, f"forecast target {target} is not on reward side"
    return target, f"forecast {direction} conf={forecast_confidence:.2f} target@{target}"


def _forecast_tp_respects_execution_floor(
    *,
    method: TradeMethod,
    execution_regime: str | None,
    entry: float,
    target: float,
    stop_pips: float,
    spread_pips: float,
    pip_factor: int,
    pair: str,
    fresh_new_entry: bool,
) -> tuple[bool, str]:
    target_pips = abs(target - entry) * pip_factor
    policy = RiskPolicy()
    regime = str(execution_regime or "").upper()
    active_min_rr = (
        policy.range_min_reward_risk
        if method == TradeMethod.RANGE_ROTATION or "RANGE" in regime
        else policy.min_reward_risk
    )
    min_target_pips = active_min_rr * stop_pips if stop_pips > 0 else 0.0
    floor_label = "execution"
    if fresh_new_entry and stop_pips > 0:
        live_floor_pips = _fresh_entry_live_floor_distance_pips(pair, stop_pips)
        if live_floor_pips > min_target_pips:
            min_target_pips = live_floor_pips
            floor_label = "fresh-live"
    if stop_pips > 0 and target_pips < min_target_pips:
        return False, f"forecast TP RR {(target_pips / stop_pips):.2f} < {floor_label} floor {(min_target_pips / stop_pips):.2f}"
    min_target_pips = spread_pips * policy.min_target_spread_multiple
    if target_pips < min_target_pips:
        return False, f"forecast TP {target_pips:.1f}pip < spread floor {min_target_pips:.1f}pip"
    return True, "forecast TP respects execution floor"


def _tp_closer_to_entry(entry: float, candidate: float, current: float) -> bool:
    return abs(candidate - entry) < abs(current - entry)


def _technical_tp_candidate(
    *,
    pair: str,
    side: Side,
    entry: float,
    spread_pips: float,
    pair_chart: dict[str, Any] | None,
    intent: str,
) -> tuple[float | None, str]:
    if not pair_chart:
        return None, "no pair_chart structural ladder; using ATR/RR virtual target"
    pip_factor = PIP_FACTORS[pair]
    target, reason = structural_tp_target(
        pair_chart,
        side=side.value,
        current_price=entry,
        pip_factor=pip_factor,
        intent=intent,
    )
    if target is None:
        return None, reason
    target = _round_price(pair, float(target))
    if not _target_on_reward_side(side, entry, target):
        return None, f"structural target {target} is not on reward side; using ATR/RR virtual target"
    distance_pips = abs(target - entry) * pip_factor
    min_target_pips = spread_pips * RiskPolicy().min_target_spread_multiple
    if distance_pips < min_target_pips:
        return None, (
            f"structural target {target} is only {distance_pips:.1f}pip away "
            f"(< {min_target_pips:.1f}pip spread floor); using ATR/RR virtual target"
        )
    return target, reason


def _target_on_reward_side(side: Side, entry: float, target: float) -> bool:
    if side == Side.LONG:
        return target > entry
    return target < entry


def _should_attach_take_profit_on_fill(
    *,
    method: TradeMethod,
    order_type: OrderType,
    execution_regime: str | None,
    chart_context: dict[str, Any] | None,
) -> tuple[bool, str]:
    regime = str(execution_regime or "").upper()
    if method == TradeMethod.RANGE_ROTATION:
        return True, "range rotation harvests the opposing technical rail"
    if method == TradeMethod.BREAKOUT_FAILURE:
        return True, "failed-break setups bank at the reclaimed structural level"
    if order_type == OrderType.LIMIT:
        return True, "pending limit entries use broker TP to capture the planned rotation"
    if "RANGE" in regime or "UNCLEAR" in regime or "FAILURE" in regime:
        return True, f"{regime or 'missing'} regime is not a clean runner trend"

    ctx = chart_context or {}
    adx = _chart_adx(ctx)
    if adx is None:
        return True, "ADX missing; attach TP instead of creating an uncapped runner"
    if adx < DYNAMIC_RR_ADX_TREND_THRESHOLD:
        return True, f"ADX {adx:.1f} below trend threshold {DYNAMIC_RR_ADX_TREND_THRESHOLD}"

    tf_agree = _optional_float(ctx.get("tf_agreement_score"))
    if tf_agree is None:
        return True, "TF agreement missing; attach TP instead of creating an uncapped runner"
    if tf_agree < TP_MODE_TF_AGREEMENT_MAJORITY:
        return True, f"TF agreement {tf_agree:.2f} below majority {TP_MODE_TF_AGREEMENT_MAJORITY:.2f}"

    atr_pct = _optional_float(ctx.get("atr_percentile_24h"))
    if atr_pct is not None and atr_pct <= DYNAMIC_RR_ATR_PCTILE_LOW:
        return True, f"ATR percentile {atr_pct:.2f} is small-wave tape"

    sigma_24h = _optional_float(ctx.get("range_24h_sigma_multiple"))
    if sigma_24h is not None and sigma_24h >= TP_MODE_EXHAUSTION_SIGMA:
        return True, f"24h range {sigma_24h:.2f}σ is exhausted; harvest rather than run"

    session = str(ctx.get("session_current_tag") or ctx.get("session_bucket") or "").upper()
    if session in {"OFF_HOURS", "JP_HOLIDAY"}:
        return True, f"{session} liquidity is thin; attach TP"

    if "TREND" in regime or "IMPULSE" in regime:
        return False, f"{regime} with ADX {adx:.1f} and TF agreement {tf_agree:.2f} qualifies as runner"
    return True, f"{regime or 'missing'} regime does not qualify as runner"


def _chart_adx(chart_context: dict[str, Any]) -> float | None:
    for key in ("h1_adx", "h4_adx"):
        value = _optional_float(chart_context.get(key))
        if value is not None:
            return value
    return None


def _geometry(
    pair: str,
    side: Side,
    order_type: OrderType,
    quote: Quote,
    *,
    reward_risk: float = 1.5,
    atr_pips: float | None = None,
    range_indicators: dict[str, Any] | None = None,
    chart_indicators: dict[str, Any] | None = None,
    stop_widen_mult: float = 1.0,
    chart_context: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Build (entry, tp, sl) prices.

    Stop distance comes from market reality, not a fixed pip literal:
        stop_pips = max(atr_pips * GEOMETRY_ATR_MULT, spread_pips * GEOMETRY_SPREAD_FLOOR_MULT)

    `stop_widen_mult` (≥1.0) lets a regime-aware caller widen the stop when
    confidence is low or ATR sits in the top of its trailing distribution —
    per AGENT_CONTRACT §3.5 the response to noise is wider invalidation, not
    a narrower target. Defaults to 1.0 so callers without regime context get
    the previous behavior unchanged.

    When atr_pips is None (pair_charts missing), we fall back to a *spread-only*
    distance; the caller is responsible for emitting MISSING_ATR so the operator
    sees that geometry was built without the primary market input.
    """
    if order_type in {OrderType.LIMIT, OrderType.MARKET} and range_indicators:
        range_geometry = _range_geometry(
            pair,
            side,
            order_type,
            quote,
            reward_risk=reward_risk,
            atr_pips=atr_pips,
            indicators=range_indicators,
            stop_widen_mult=stop_widen_mult,
            chart_context=chart_context,
        )
        if range_geometry is not None:
            return range_geometry
    h4_atr_pips = None
    session_bucket: str | None = None
    if isinstance(chart_context, dict):
        h4_atr_pips = _optional_float(chart_context.get("h4_atr_pips"))
        # Prefer the chart_reader killzone label
        # (`session_current_tag`); fall back to the older
        # `session_bucket` name if a caller passed only that.
        session_bucket = _text_or_none(
            chart_context.get("session_current_tag") or chart_context.get("session_bucket")
        )
    return _generic_geometry(
        pair,
        side,
        order_type,
        quote,
        reward_risk=reward_risk,
        atr_pips=atr_pips,
        chart_indicators=chart_indicators,
        stop_widen_mult=stop_widen_mult,
        h4_atr_pips=h4_atr_pips,
        session_bucket=session_bucket,
    )


def _session_widening_mult(session_bucket: str | None) -> float:
    """F (2026-05-13) — multiplier on top of H4-ATR noise floor by
    session liquidity tier. Thin / OFF_HOURS sessions widen the
    floor; deep London-NY overlap is the default 1.0.

    Falls back to 1.0 when the session bucket is unknown so missing
    data never silently shrinks the noise floor — AGENT_CONTRACT §3.5.
    """
    if not session_bucket:
        return 1.0
    tag = str(session_bucket).upper().strip()
    if tag in {"OFF_HOURS", "OFF-HOURS", "OFFHOURS", "JP_HOLIDAY"}:
        return NEW_ENTRY_SL_OFF_HOURS_MULT
    if tag in {"TOKYO_KILLZONE", "ASIA_OPEN", "TOKYO", "ASIA"}:
        return NEW_ENTRY_SL_THIN_SESSION_MULT
    # London / NY (and the LONDON_NY overlap) are the deep-liquidity
    # band — no extra widening.
    return 1.0


def _generic_geometry(
    pair: str,
    side: Side,
    order_type: OrderType,
    quote: Quote,
    *,
    reward_risk: float,
    atr_pips: float | None,
    chart_indicators: dict[str, Any] | None = None,
    stop_widen_mult: float = 1.0,
    h4_atr_pips: float | None = None,
    session_bucket: str | None = None,
) -> tuple[float, float, float]:
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    spread_floor = spread_pips * GEOMETRY_SPREAD_FLOOR_MULT
    if atr_pips is not None and atr_pips > 0:
        stop_pips = max(atr_pips * GEOMETRY_ATR_MULT, spread_floor)
    else:
        stop_pips = spread_floor
    # Regime-aware stop widening: noise / low-confidence regimes get a wider
    # invalidation distance so wick noise does not stop the trade prematurely.
    # Floor at the spread-only distance to keep the spread guarantee.
    stop_pips = max(stop_pips * max(stop_widen_mult, 1.0), spread_floor)
    # F (2026-05-13) — noise-resistant initial-SL floor.
    # When the operator wants a broker-side SL on new entries
    # (`QR_NEW_ENTRY_INITIAL_SL=1`), the stop distance must sit outside
    # the M5/M15 wick band so routine noise cannot hit it. The floor is
    # H4 ATR × NEW_ENTRY_SL_H4_ATR_MULT; the M5-derived stop is kept as
    # the lower bound, never the upper. Session-thin liquidity expands
    # the floor further (Tokyo open / OFF_HOURS print wider wicks).
    if _new_entry_initial_sl_active() and h4_atr_pips is not None and h4_atr_pips > 0:
        h4_floor = h4_atr_pips * NEW_ENTRY_SL_H4_ATR_MULT
        session_mult = _session_widening_mult(session_bucket)
        stop_pips = max(stop_pips, h4_floor * session_mult)
    trigger_offset_pips = spread_pips * PENDING_ENTRY_OFFSET_SPREAD_MULT
    if order_type == OrderType.LIMIT:
        entry = quote.bid - trigger_offset_pips * pip if side == Side.LONG else quote.ask + trigger_offset_pips * pip
    elif order_type == OrderType.MARKET:
        entry = quote.ask if side == Side.LONG else quote.bid
    else:
        entry = quote.ask + trigger_offset_pips * pip if side == Side.LONG else quote.bid - trigger_offset_pips * pip
    stop_pips = _structural_stop_pips(
        pair,
        side,
        entry,
        base_stop_pips=stop_pips,
        spread_pips=spread_pips,
        indicators=chart_indicators,
    )
    reward_pips = stop_pips * reward_risk
    if side == Side.LONG:
        tp = entry + reward_pips * pip
        sl = entry - stop_pips * pip
    else:
        tp = entry - reward_pips * pip
        sl = entry + stop_pips * pip
    return _round_price(pair, entry), _round_price(pair, tp), _round_price(pair, sl)


def _structural_stop_pips(
    pair: str,
    side: Side,
    entry: float,
    *,
    base_stop_pips: float,
    spread_pips: float,
    indicators: dict[str, Any] | None,
) -> float:
    if not indicators:
        return base_stop_pips
    pip_factor = PIP_FACTORS[pair]
    if side == Side.LONG:
        level = _nearest_below(entry, _numeric_levels(indicators, RANGE_SUPPORT_LEVEL_KEYS))
        if level is None:
            return base_stop_pips
        structural_pips = (entry - level) * pip_factor
    else:
        level = _nearest_above(entry, _numeric_levels(indicators, RANGE_RESISTANCE_LEVEL_KEYS))
        if level is None:
            return base_stop_pips
        structural_pips = (level - entry) * pip_factor
    if structural_pips <= 0:
        return base_stop_pips
    buffer_pips = spread_pips * STRUCTURAL_STOP_BUFFER_SPREAD_MULT
    return max(base_stop_pips, structural_pips + buffer_pips)


def _range_geometry(
    pair: str,
    side: Side,
    order_type: OrderType,
    quote: Quote,
    *,
    reward_risk: float,
    atr_pips: float | None,
    indicators: dict[str, Any],
    stop_widen_mult: float = 1.0,
    chart_context: dict[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    support_levels = _numeric_levels(indicators, RANGE_SUPPORT_LEVEL_KEYS)
    resistance_levels = _numeric_levels(indicators, RANGE_RESISTANCE_LEVEL_KEYS)
    support = _nearest_below(quote.ask, support_levels)
    resistance = _nearest_above(quote.bid, resistance_levels)
    if support is None or resistance is None:
        return None

    spread_buffer = spread_pips * RANGE_RAIL_ENTRY_BUFFER_SPREAD_MULT * pip
    pending_offset = spread_pips * PENDING_ENTRY_OFFSET_SPREAD_MULT * pip
    spread_floor = spread_pips * GEOMETRY_SPREAD_FLOOR_MULT
    stop_pips = max((atr_pips or 0.0) * GEOMETRY_ATR_MULT, spread_floor)
    # Regime-aware stop widening keeps SL outside wick noise; floor at spread.
    stop_pips = max(stop_pips * max(stop_widen_mult, 1.0), spread_floor)
    if stop_pips <= 0:
        return None

    if side == Side.LONG:
        if order_type == OrderType.MARKET:
            edge_zone = _range_market_edge_zone_pips(atr_pips, stop_pips, spread_pips)
            if quote.ask > support + (edge_zone * pip):
                return _directional_range_market_geometry(
                    pair,
                    side,
                    quote,
                    reward_risk=reward_risk,
                    atr_pips=atr_pips,
                    spread_pips=spread_pips,
                    chart_context=chart_context,
                )
            entry = quote.ask
        else:
            entry = min(support + spread_buffer, quote.ask - pending_offset)
        sl = min(entry - (stop_pips * pip), support - spread_buffer)
        loss_pips = abs(entry - sl) * pip_factor
        resistance = _target_resistance(entry, loss_pips, spread_pips, resistance_levels, pip)
        if resistance is None or resistance <= support:
            return None
        opposing_rail = resistance - (spread_pips * RANGE_OPPOSING_RAIL_BUFFER_SPREAD_MULT * pip)
        rr_target = entry + (loss_pips * reward_risk * pip)
        tp = min(rr_target, opposing_rail)
        if tp <= entry:
            return None
    else:
        if order_type == OrderType.MARKET:
            edge_zone = _range_market_edge_zone_pips(atr_pips, stop_pips, spread_pips)
            if quote.bid < resistance - (edge_zone * pip):
                return _directional_range_market_geometry(
                    pair,
                    side,
                    quote,
                    reward_risk=reward_risk,
                    atr_pips=atr_pips,
                    spread_pips=spread_pips,
                    chart_context=chart_context,
                )
            entry = quote.bid
        else:
            entry = max(resistance - spread_buffer, quote.bid + pending_offset)
        sl = max(entry + (stop_pips * pip), resistance + spread_buffer)
        loss_pips = abs(entry - sl) * pip_factor
        support = _target_support(entry, loss_pips, spread_pips, support_levels, pip)
        if support is None or resistance <= support:
            return None
        opposing_rail = support + (spread_pips * RANGE_OPPOSING_RAIL_BUFFER_SPREAD_MULT * pip)
        rr_target = entry - (loss_pips * reward_risk * pip)
        tp = max(rr_target, opposing_rail)
        if tp >= entry:
            return None
    return _round_price(pair, entry), _round_price(pair, tp), _round_price(pair, sl)


def _directional_range_market_geometry(
    pair: str,
    side: Side,
    quote: Quote,
    *,
    reward_risk: float,
    atr_pips: float | None,
    spread_pips: float,
    chart_context: dict[str, Any] | None,
) -> tuple[float, float, float] | None:
    if not _is_low_vol_directional_range(side, chart_context):
        return None
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    stop_floor = spread_pips * RANGE_DIRECTIONAL_STOP_SPREAD_MULT
    stop_pips = max((atr_pips or 0.0) * GEOMETRY_ATR_MULT, stop_floor)
    if stop_pips <= 0:
        return None
    target_pips = max(
        stop_pips * min(reward_risk, RANGE_DIRECTIONAL_MARKET_TARGET_RR_CAP),
        spread_pips * RiskPolicy().min_target_spread_multiple,
    )
    if side == Side.LONG:
        entry = quote.ask
        tp = entry + (target_pips * pip)
        sl = entry - (stop_pips * pip)
    else:
        entry = quote.bid
        tp = entry - (target_pips * pip)
        sl = entry + (stop_pips * pip)
    return _round_price(pair, entry), _round_price(pair, tp), _round_price(pair, sl)


def _range_market_edge_zone_pips(atr_pips: float | None, stop_pips: float, spread_pips: float) -> float:
    atr_component = (atr_pips if atr_pips is not None and atr_pips > 0 else stop_pips) * RANGE_MARKET_EDGE_ZONE_ATR_MULT
    spread_component = spread_pips * RANGE_MARKET_EDGE_ZONE_SPREAD_MULT
    return max(atr_component, spread_component)


def _numeric_levels(indicators: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, ...]:
    levels: list[float] = []
    for key in keys:
        raw = indicators.get(key)
        try:
            value = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            continue
        if value > 0:
            levels.append(value)
    return tuple(levels)


def _nearest_below(price: float, levels: tuple[float, ...]) -> float | None:
    candidates = [level for level in levels if level < price]
    return max(candidates) if candidates else None


def _nearest_above(price: float, levels: tuple[float, ...]) -> float | None:
    candidates = [level for level in levels if level > price]
    return min(candidates) if candidates else None


def _target_resistance(
    entry: float,
    stop_pips: float,
    spread_pips: float,
    levels: tuple[float, ...],
    pip: float,
) -> float | None:
    min_target_pips = _minimum_range_target_pips(stop_pips, spread_pips)
    candidates = sorted(level for level in levels if level > entry)
    if not candidates:
        return None
    threshold = entry + (min_target_pips * pip)
    for level in candidates:
        if level >= threshold:
            return level
    return candidates[-1]


def _target_support(
    entry: float,
    stop_pips: float,
    spread_pips: float,
    levels: tuple[float, ...],
    pip: float,
) -> float | None:
    min_target_pips = _minimum_range_target_pips(stop_pips, spread_pips)
    candidates = sorted((level for level in levels if level < entry), reverse=True)
    if not candidates:
        return None
    threshold = entry - (min_target_pips * pip)
    for level in candidates:
        if level <= threshold:
            return level
    return candidates[-1]


def _minimum_range_target_pips(stop_pips: float, spread_pips: float) -> float:
    # RANGE geometry uses the regime-specific floor so rail-tagged rotations
    # are not pushed past the opposite rail just to satisfy a 1.2R floor.
    # Spread-multiple floor still applies; the overall target distance is the
    # max of the two so spread cost is always covered.
    policy = RiskPolicy()
    return max(
        stop_pips * policy.range_min_reward_risk,
        spread_pips * policy.min_target_spread_multiple,
    )


def _same_side_add_metadata(
    pair: str,
    side: Side,
    positions: list[BrokerPosition],
    *,
    entry: float | None,
) -> dict[str, Any]:
    units_by_position = [abs(int(position.units)) for position in positions]
    total_units = sum(units_by_position)
    metadata: dict[str, Any] = {
        "position_intent": "PYRAMID",
        "position_fill": "OPEN_ONLY",
        "same_pair_existing_entries": len(positions),
        "same_pair_existing_units": total_units,
    }
    if total_units <= 0:
        metadata["same_pair_add_type"] = "UNKNOWN_SAME_SIDE_ADD"
        return metadata
    avg_entry = (
        sum(
            float(position.entry_price) * units
            for position, units in zip(positions, units_by_position)
        )
        / total_units
    )
    metadata["same_pair_existing_avg_entry"] = _round_price(pair, avg_entry)
    if entry is None:
        metadata["same_pair_add_type"] = "UNKNOWN_SAME_SIDE_ADD"
        return metadata
    pip_factor = PIP_FACTORS[pair]
    raw_distance_pips = (float(entry) - avg_entry) * pip_factor
    metadata["same_pair_add_entry"] = _round_price(pair, float(entry))
    metadata["same_pair_add_distance_from_avg_pips"] = round(raw_distance_pips, 3)
    if side == Side.LONG:
        adverse_add_pips = max(0.0, -raw_distance_pips)
        with_move_add_pips = max(0.0, raw_distance_pips)
    else:
        adverse_add_pips = max(0.0, raw_distance_pips)
        with_move_add_pips = max(0.0, -raw_distance_pips)
    metadata["same_pair_adverse_add_pips"] = round(adverse_add_pips, 3)
    metadata["same_pair_with_move_add_pips"] = round(with_move_add_pips, 3)
    if adverse_add_pips > 0.0:
        metadata["same_pair_add_type"] = "AVERAGE_INTO_ADVERSE"
    elif with_move_add_pips > 0.0:
        metadata["same_pair_add_type"] = "PYRAMID_WITH_MOVE"
    else:
        metadata["same_pair_add_type"] = "FLAT_RETEST_ADD"
    return metadata


def _position_intent_metadata(
    pair: str,
    side: Side,
    snapshot: BrokerSnapshot,
    *,
    entry: float | None = None,
) -> dict[str, Any]:
    pair_positions = [position for position in snapshot.positions if position.pair == pair]
    same_pair = [position for position in pair_positions if position.owner == Owner.TRADER]
    if not same_pair:
        return {}
    if any(position.side != side for position in same_pair):
        same_side_positions = [position for position in same_pair if position.side == side]
        opposing_positions = [position for position in same_pair if position.side != side]
        non_trader_positions = [position for position in pair_positions if position.owner != Owner.TRADER]
        ignored_same_side_units = sum(
            abs(int(position.units)) for position in non_trader_positions if position.side == side
        )
        ignored_opposing_units = sum(
            abs(int(position.units)) for position in non_trader_positions if position.side != side
        )
        ignored_opposing_pl = sum(
            float(position.unrealized_pl_jpy or 0.0) for position in non_trader_positions if position.side != side
        )
        metadata: dict[str, Any] = {
            "position_intent": "HEDGE",
            "position_fill": "OPEN_ONLY",
            "hedge_reference_scope": "trader_owned_only",
            "hedge_non_trader_same_side_units_ignored": ignored_same_side_units,
            "hedge_non_trader_opposing_units_ignored": ignored_opposing_units,
            "hedge_non_trader_opposing_unrealized_pl_jpy_ignored": round(ignored_opposing_pl, 4),
        }
        gross_reference_units = sum(abs(int(position.units)) for position in opposing_positions)
        same_side_units = sum(abs(int(position.units)) for position in same_side_positions)
        reference_units = max(0, gross_reference_units - same_side_units)
        existing_pl = sum(float(position.unrealized_pl_jpy or 0.0) for position in opposing_positions)
        if reference_units <= 0:
            metadata = _same_side_add_metadata(pair, side, same_side_positions, entry=entry)
            metadata.update(
                {
                    "hedge_reference_units": 0,
                    "hedge_gross_opposing_units": gross_reference_units,
                    "hedge_existing_same_side_units": same_side_units,
                    "hedge_covered_reference_units": gross_reference_units,
                    "hedge_suppressed_reason": "opposite_exposure_already_covered",
                    "hedge_reference_scope": "trader_owned_only",
                    "hedge_non_trader_same_side_units_ignored": ignored_same_side_units,
                    "hedge_non_trader_opposing_units_ignored": ignored_opposing_units,
                    "hedge_non_trader_opposing_unrealized_pl_jpy_ignored": round(
                        ignored_opposing_pl,
                        4,
                    ),
                }
            )
            return metadata
        metadata.update(
            {
                "hedge_reference_units": reference_units,
                "hedge_gross_opposing_units": gross_reference_units,
                "hedge_existing_same_side_units": same_side_units,
                "hedge_covered_reference_units": min(gross_reference_units, same_side_units),
                "hedge_existing_unrealized_pl_jpy": round(existing_pl, 4),
            }
        )
        underwater_positions = [position for position in opposing_positions if position.unrealized_pl_jpy < 0]
        if underwater_positions:
            metadata.update(
                {
                    "hedge_recovery": True,
                    "hedge_recovery_reason": "opposing_same_pair_underwater",
                    "hedge_recovery_units": reference_units,
                    "hedge_recovery_unrealized_pl_jpy": round(
                        sum(position.unrealized_pl_jpy for position in underwater_positions),
                        4,
                    ),
                }
            )
        elif existing_pl > 0:
            metadata.update(
                {
                    "hedge_lock_gain": True,
                    "hedge_lock_gain_unrealized_pl_jpy": round(existing_pl, 4),
                }
            )
        return metadata
    return _same_side_add_metadata(pair, side, same_pair, entry=entry)


def _is_hedge_recovery_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("position_intent") or "").upper() == "HEDGE"
        and bool(metadata.get("hedge_recovery"))
    )


def _hedge_timing_metadata(
    side: Side,
    position_metadata: dict[str, Any],
    chart_context: dict[str, Any] | None,
    lane: dict[str, Any],
) -> dict[str, Any]:
    if str(position_metadata.get("position_intent") or "").upper() != "HEDGE":
        return {}
    if bool(position_metadata.get("hedge_lock_gain")):
        return {
            "hedge_timing_class": "LOCK_GAIN",
            "hedge_time_efficiency_role": "protect_existing_mfe_while_retesting_next_structure",
            "hedge_review_trigger": "next_m15_close_or_nearest_structure_hit",
            "hedge_unwind_plan_required": True,
        }
    if _is_hedge_recovery_metadata(position_metadata):
        if _hedge_reversal_confirmed(side, chart_context, lane):
            return {
                "hedge_timing_class": "REVERSAL",
                "hedge_time_efficiency_role": "monetize_confirmed_reversal_against_trapped_leg",
                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
                "hedge_unwind_plan_required": True,
            }
        return {
            "hedge_timing_class": "CONTINUATION",
            "hedge_time_efficiency_role": "small_tranche_only_until_reversal_or_exhaustion_confirms",
            "hedge_review_trigger": "next_m15_close_or_failed_break_trigger",
            "hedge_unwind_plan_required": True,
        }
    return {
        "hedge_timing_class": "OPPOSITE_EXPOSURE",
        "hedge_time_efficiency_role": "opposite_same_pair_exposure_requires_explicit_unwind",
        "hedge_review_trigger": "next_m15_close_or_structure_change",
        "hedge_unwind_plan_required": True,
    }


def _hedge_reversal_confirmed(
    side: Side,
    chart_context: dict[str, Any] | None,
    lane: dict[str, Any],
) -> bool:
    context = chart_context or {}
    if _operating_tf_opposes_side(side, context):
        return False
    if _structure_story_confirms_side(side, str(context.get("chart_story_structural") or "")):
        return True
    tf_agree = _bounded_unit(_optional_float(context.get("tf_agreement_score")))
    gap_score = _directional_chart_gap(side, context)
    if tf_agree is not None and gap_score is not None:
        if tf_agree >= TP_MODE_TF_AGREEMENT_MAJORITY and gap_score >= CHART_DIRECTION_TIED_GAP_BOUNDARY:
            return True
    forecast_confidence = _bounded_unit(_optional_float(lane.get("forecast_confidence")))
    forecast_direction = str(lane.get("forecast_direction") or "").upper()
    forecast_side = Side.LONG if forecast_direction == "UP" else Side.SHORT if forecast_direction == "DOWN" else None
    return (
        forecast_side == side
        and forecast_confidence is not None
        and forecast_confidence >= RECOVERY_HEDGE_DEFAULT_CONVICTION_SCALE
    )


def _structure_story_confirms_side(side: Side, chart_story: str) -> bool:
    text = chart_story.upper()
    tokens = ("BOS_UP", "CHOCH_UP") if side == Side.LONG else ("BOS_DOWN", "CHOCH_DOWN")
    for token in tokens:
        start = text.find(token)
        if start < 0:
            continue
        local = text[start : start + 48]
        if ":WICK" not in local:
            return True
    return False


def _recovery_hedge_sizing_metadata(
    side: Side,
    position_metadata: dict[str, Any],
    chart_context: dict[str, Any] | None,
    lane: dict[str, Any],
) -> dict[str, Any]:
    reference_units = int(position_metadata.get("hedge_recovery_units") or 0)
    if reference_units <= 0:
        return {
            "hedge_recovery_reference_units": reference_units,
            "hedge_recovery_size_scale": RECOVERY_HEDGE_DEFAULT_CONVICTION_SCALE,
        }
    scale = _recovery_hedge_conviction_scale(side, chart_context, lane)
    if str(position_metadata.get("hedge_timing_class") or "").upper() == "CONTINUATION":
        scale = min(scale, RECOVERY_HEDGE_CONTINUATION_MAX_SCALE)
    scaled_units = int((reference_units * scale) // MIN_PRODUCTION_LOT_UNITS) * MIN_PRODUCTION_LOT_UNITS
    if scaled_units <= 0 and reference_units >= MIN_PRODUCTION_LOT_UNITS:
        scaled_units = MIN_PRODUCTION_LOT_UNITS
    return {
        "hedge_recovery_reference_units": reference_units,
        "hedge_recovery_size_scale": scale,
        "hedge_recovery_units": min(reference_units, scaled_units),
    }


def _recovery_hedge_conviction_scale(
    side: Side,
    chart_context: dict[str, Any] | None,
    lane: dict[str, Any],
) -> float:
    context = chart_context or {}
    side_key = "short" if side == Side.SHORT else "long"
    aligned_score = _bounded_unit(_optional_float(context.get(f"chart_{side_key}_score")))
    gap_score = _directional_chart_gap(side, context)
    tf_score = _bounded_unit(_optional_float(context.get("tf_agreement_score")))
    weighted: list[tuple[float, float]] = []
    if aligned_score is not None:
        weighted.append((aligned_score, 0.55))
    if gap_score is not None:
        weighted.append((gap_score, 0.30))
    if tf_score is not None:
        weighted.append((tf_score, 0.15))
    conviction = (
        sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
        if weighted
        else RECOVERY_HEDGE_DEFAULT_CONVICTION_SCALE
    )
    forecast_confidence = _bounded_unit(_optional_float(lane.get("forecast_confidence")))
    forecast_direction = str(lane.get("forecast_direction") or "").upper()
    forecast_side = Side.LONG if forecast_direction == "UP" else Side.SHORT if forecast_direction == "DOWN" else None
    if forecast_confidence is not None and forecast_side is not None:
        if forecast_side == side:
            conviction = (conviction * 0.7) + (forecast_confidence * 0.3)
        else:
            conviction *= max(0.35, 1.0 - forecast_confidence)
    return round(
        max(RECOVERY_HEDGE_MIN_CONVICTION_SCALE, min(1.0, conviction)),
        2,
    )


def _directional_chart_gap(side: Side, chart_context: dict[str, Any]) -> float | None:
    gap = _optional_float(chart_context.get("chart_score_gap"))
    if gap is None:
        return None
    if side == Side.LONG:
        return _bounded_unit(max(0.0, gap))
    return _bounded_unit(max(0.0, -gap))


def _bounded_unit(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _geometry_metadata(
    pair: str,
    side: Side,
    order_type: OrderType,
    quote: Quote,
    *,
    entry: float,
    tp: float,
    sl: float,
    range_indicators: dict[str, Any] | None,
    chart_indicators: dict[str, Any] | None,
    chart_context: dict[str, Any] | None = None,
    atr_pips: float | None = None,
) -> dict[str, Any]:
    entry_location = _entry_location_percentile_metadata(pair, entry, chart_context)
    if order_type not in {OrderType.LIMIT, OrderType.MARKET} or not range_indicators:
        return {
            "geometry_model": "ATR_SPREAD_STRUCTURE",
            **entry_location,
            **_structural_stop_metadata(pair, side, entry, sl, chart_indicators),
        }
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    stop_pips = abs(entry - sl) * pip_factor
    support_levels = _numeric_levels(range_indicators, RANGE_SUPPORT_LEVEL_KEYS)
    resistance_levels = _numeric_levels(range_indicators, RANGE_RESISTANCE_LEVEL_KEYS)
    support = _nearest_below(quote.ask, support_levels)
    resistance = _nearest_above(quote.bid, resistance_levels)
    if support is None or resistance is None:
        return {"geometry_model": "ATR_SPREAD", **entry_location}
    if side == Side.LONG:
        resistance = _target_resistance(entry, stop_pips, spread_pips, resistance_levels, pip) or resistance
    else:
        support = _target_support(entry, stop_pips, spread_pips, support_levels, pip) or support
    inside_box = support < tp < resistance
    outside_box = sl < support if side == Side.LONG else sl > resistance
    if order_type == OrderType.MARKET and _is_low_vol_directional_range(side, chart_context):
        model = "RANGE_DIRECTIONAL_MARKET"
    elif order_type == OrderType.MARKET:
        edge_zone = _range_market_edge_zone_pips(atr_pips, stop_pips, spread_pips)
        if side == Side.LONG:
            at_rail = quote.ask <= support + (edge_zone * pip)
        else:
            at_rail = quote.bid >= resistance - (edge_zone * pip)
        model = "RANGE_RAIL_MARKET" if at_rail and inside_box and outside_box else "ATR_SPREAD"
    else:
        model = "RANGE_RAIL_LIMIT"
    return {
        "geometry_model": model,
        "range_support": round(support, 3 if pair.endswith("_JPY") else 5),
        "range_resistance": round(resistance, 3 if pair.endswith("_JPY") else 5),
        "range_entry_side": "support" if side == Side.LONG else "resistance",
        "range_tp_is_inside_box": inside_box,
        "range_sl_outside_box": outside_box,
        "range_directional_market": model == "RANGE_DIRECTIONAL_MARKET",
        "range_directional_target_rr_cap": (
            RANGE_DIRECTIONAL_MARKET_TARGET_RR_CAP if model == "RANGE_DIRECTIONAL_MARKET" else None
        ),
        **entry_location,
    }


def _entry_location_percentile_metadata(
    pair: str,
    entry: float,
    chart_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(chart_context, dict):
        return {}
    out: dict[str, Any] = {}
    for horizon in ("24h", "7d"):
        low = _optional_float(chart_context.get(f"price_range_{horizon}_low"))
        high = _optional_float(chart_context.get(f"price_range_{horizon}_high"))
        if low is None or high is None or high <= low:
            continue
        percentile = max(0.0, min(1.0, (float(entry) - low) / (high - low)))
        out[f"entry_price_percentile_{horizon}"] = round(percentile, 4)
        out[f"entry_price_range_{horizon}_low"] = _round_price(pair, low)
        out[f"entry_price_range_{horizon}_high"] = _round_price(pair, high)
        source = _text_or_none(chart_context.get(f"price_range_{horizon}_source"))
        if source:
            out[f"entry_price_percentile_{horizon}_source"] = source
    return out


def _structural_stop_metadata(
    pair: str,
    side: Side,
    entry: float,
    sl: float,
    indicators: dict[str, Any] | None,
) -> dict[str, Any]:
    if not indicators:
        return {}
    pip_factor = PIP_FACTORS[pair]
    if side == Side.LONG:
        level = _nearest_below(entry, _numeric_levels(indicators, RANGE_SUPPORT_LEVEL_KEYS))
        outside = level is not None and sl < level
    else:
        level = _nearest_above(entry, _numeric_levels(indicators, RANGE_RESISTANCE_LEVEL_KEYS))
        outside = level is not None and sl > level
    if level is None:
        return {}
    return {
        "structural_stop_level": round(level, 3 if pair.endswith("_JPY") else 5),
        "structural_stop_distance_pips": round(abs(entry - level) * pip_factor, 3),
        "structural_stop_outside_level": outside,
    }


def _target_reward_risk(lane: dict[str, Any]) -> float:
    try:
        value = float(lane.get("target_reward_risk") or 1.5)
    except (TypeError, ValueError):
        value = 1.5
    return round(min(8.0, max(1.2, value)), 2)


def _round_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _risk_budgeted_units(
    pair: str,
    entry: float,
    sl: float,
    *,
    max_loss_jpy: float,
    snapshot: BrokerSnapshot,
    side: Side | None = None,
    position_intent: str | None = None,
    target_units_override: int | None = None,
    loss_budget_target: bool = False,
) -> int:
    pip_factor = PIP_FACTORS[pair]
    stop_pips = abs(entry - sl) * pip_factor
    if stop_pips <= 0:
        return 1
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return 1
    loss_budget_units = max_loss_jpy * pip_factor / (stop_pips * quote_to_jpy)
    margin_budget_units = _margin_budgeted_units(pair, entry, snapshot, side=side, position_intent=position_intent)
    margin_free_hedge_units = (
        hedge_margin_free_units(pair=pair, side=side, snapshot=snapshot, position_intent=position_intent)
        if side is not None
        else 0
    )
    # SL-free mode (`QR_TRADER_DISABLE_SL_REPAIR=1`) is still loss-cap bounded:
    # the operator/NAV/hedge target picks the desired size, but the executable
    # result must fit the equity-derived per-trade loss cap before it can be
    # LIVE_READY.
    # Sizing precedence (highest first):
    #   1. QR_TRADER_POSITION_NAV_PCT — % of NAV used as margin per position.
    #      Auto-scales with equity (user 2026-05-08「BaseUnitを決めると、
    #      資産が増えたときに追従できないよ。％で決めないといけなくない？」).
    #   2. QR_TRADER_BASE_UNITS — legacy fixed-unit fallback.
    #   3. Hard-coded 3000 unit fallback.
    # Margin headroom (`_margin_budgeted_units`) still caps the result so the
    # 92% portfolio margin utilization gate is never breached. Explicit same-pair
    # HEDGE intents are the exception to NAV% sizing: they target the current
    # margin-free opposite leg so a protective hedge can flatten directional
    # exposure without becoming a new speculative position size.
    sl_free_active = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if sl_free_active:
        if loss_budget_target:
            target_units = loss_budget_units
        elif target_units_override is not None and target_units_override > 0:
            target_units = float(target_units_override)
        elif margin_free_hedge_units >= MIN_PRODUCTION_LOT_UNITS:
            target_units = float(margin_free_hedge_units)
        else:
            nav_pct_units = _nav_pct_position_units(pair, entry, snapshot)
            if nav_pct_units is not None:
                target_units = nav_pct_units
            else:
                try:
                    target_units = float(os.environ.get("QR_TRADER_BASE_UNITS", "3000") or "3000")
                except ValueError:
                    target_units = 3000.0
        candidates = [target_units, loss_budget_units]
        if margin_budget_units is not None:
            candidates.append(margin_budget_units)
        max_units = max(1.0, min(candidates))
    else:
        max_units = min(loss_budget_units, margin_budget_units) if margin_budget_units is not None else loss_budget_units
    if max_units >= 1000:
        return max(1000, int(max_units // 1000) * 1000)
    # 2026-05-12 emergency fix B: when margin headroom can only support
    # sub-MIN_PRODUCTION_LOT_UNITS, refuse to emit a fillable intent.
    # Returning 0 propagates `intent.units == 0`, which the build-intent
    # caller turns into a `MARGIN_TOO_THIN_FOR_MIN_LOT` BLOCK so the lane
    # becomes DRY_RUN_BLOCKED instead of being staged at a few hundred
    # units (where the OANDA spread cost dominates any pip target — the
    # 470901/470904/470907 sequence on 2026-05-12T07:46 UTC).
    #
    # The floor only fires when the broker snapshot carries an account
    # (production). Test fixtures that construct a snapshot without an
    # `AccountSummary` keep the historical micro-lot fallback so they
    # can still exercise legacy geometry/narrative edge cases without
    # rewriting every fixture; `QR_ALLOW_TEST_MICRO_LOT=1` is the
    # explicit opt-out for callers that want the old behavior even with
    # an account present.
    if (
        max_units < MIN_PRODUCTION_LOT_UNITS
        and snapshot.account is not None
        and not _min_lot_test_override_active()
    ):
        return 0
    return max(1, int(max_units))


def _min_lot_block_issue(
    *,
    pair: str,
    entry: float,
    sl: float,
    max_loss_jpy: float,
    snapshot: BrokerSnapshot,
    side: Side | None = None,
    position_intent: str | None = None,
) -> dict[str, str]:
    pip_factor = PIP_FACTORS[pair]
    stop_pips = abs(entry - sl) * pip_factor
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if stop_pips > 0 and quote_to_jpy is not None:
        loss_budget_units = max_loss_jpy * pip_factor / (stop_pips * quote_to_jpy)
        margin_budget_units = _margin_budgeted_units(
            pair,
            entry,
            snapshot,
            side=side,
            position_intent=position_intent,
        )
        loss_subfloor = loss_budget_units < MIN_PRODUCTION_LOT_UNITS
        margin_subfloor = margin_budget_units is not None and margin_budget_units < MIN_PRODUCTION_LOT_UNITS
        if loss_subfloor and not margin_subfloor:
            return {
                "code": "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
                "message": (
                    f"equity-derived loss budget can only fund "
                    f"{loss_budget_units:.0f}u for {pair} at the current "
                    f"{stop_pips:.1f}pip stop; refusing to emit a "
                    f"sub-{MIN_PRODUCTION_LOT_UNITS}u receipt because "
                    "round-trip spread cost would dominate the pip target. "
                    "Wait for tighter market-derived geometry or explicit "
                    "operator pace/equity evidence that raises the per-trade budget."
                ),
                "severity": "BLOCK",
            }
        if margin_subfloor:
            return {
                "code": "MARGIN_TOO_THIN_FOR_MIN_LOT",
                "message": (
                    f"available margin headroom can only fund "
                    f"{margin_budget_units:.0f}u for {pair}; refusing to "
                    f"emit a sub-{MIN_PRODUCTION_LOT_UNITS}u receipt because "
                    "round-trip spread cost would dominate the pip target. "
                    "Free margin or wait for open positions to harvest TP."
                ),
                "severity": "BLOCK",
            }
    if quote_to_jpy is None:
        return {
            "code": "CONVERSION_RATE_MISSING_FOR_MIN_LOT",
            "message": (
                f"{pair} cannot be sized against JPY because the quote-currency "
                "conversion is missing; refusing to infer a production lot."
            ),
            "severity": "BLOCK",
        }
    return {
        "code": "MIN_LOT_SIZE_UNAVAILABLE",
        "message": (
            f"{pair} resolved to 0 units before the production "
            f"{MIN_PRODUCTION_LOT_UNITS}u floor; refusing to emit a "
            "sub-floor receipt until sizing inputs are repaired."
        ),
        "severity": "BLOCK",
    }


def _nav_pct_position_units(pair: str, entry: float, snapshot: BrokerSnapshot) -> float | None:
    """Compute per-position units from QR_TRADER_POSITION_NAV_PCT × NAV.

    Returns the desired unit count when the env var is set to a valid
    positive number AND the broker snapshot carries a margin-eligible
    account/quote/spec. Returns None when the operator has not configured
    NAV-pct sizing (caller falls back to QR_TRADER_BASE_UNITS).

    The percentage is consumed as MARGIN per position (not notional), so
    "30" means each new position locks ~30% of NAV. With OANDA Japan's
    25:1 leverage that translates to ≈7.5x NAV notional per position.
    Three concurrent 30% positions reach ~90% margin utilization, which
    sits just inside the 92% portfolio cap.
    """
    pct_str = os.environ.get("QR_TRADER_POSITION_NAV_PCT", "").strip()
    if not pct_str:
        return None
    try:
        pct = float(pct_str)
    except ValueError:
        return None
    if pct <= 0:
        return None
    account = snapshot.account
    if account is None or account.nav_jpy <= 0:
        return None
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    spec = DEFAULT_SPECS.get(pair)
    if quote_to_jpy is None or spec is None or spec.margin_rate <= 0:
        return None
    margin_per_unit = estimate_required_margin_jpy(
        units=1,
        entry_price=entry,
        quote_to_jpy=quote_to_jpy,
        spec=spec,
    )
    if margin_per_unit <= 0:
        return None
    target_margin_jpy = account.nav_jpy * (pct / 100.0)
    return target_margin_jpy / margin_per_unit


def _margin_budgeted_units(
    pair: str,
    entry: float,
    snapshot: BrokerSnapshot,
    *,
    side: Side | None = None,
    position_intent: str | None = None,
) -> float | None:
    account = snapshot.account
    if account is None:
        return None
    policy = RiskPolicy()
    max_margin_pct = policy.max_margin_utilization_pct
    if max_margin_pct is None or max_margin_pct <= 0:
        return None
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    spec = DEFAULT_SPECS.get(pair)
    if quote_to_jpy is None or spec is None or spec.margin_rate <= 0:
        return None
    broker_margin_free = (
        broker_margin_free_units(pair=pair, side=side, snapshot=snapshot)
        if side is not None
        else 0
    )
    budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
    if budget <= 0:
        return float(broker_margin_free)
    margin_per_unit = estimate_required_margin_jpy(units=1, entry_price=entry, quote_to_jpy=quote_to_jpy, spec=spec)
    if margin_per_unit <= 0:
        return float(broker_margin_free)
    return float(broker_margin_free) + (budget / margin_per_unit)


def _margin_sizing_metadata(
    pair: str,
    entry: float,
    units: int,
    snapshot: BrokerSnapshot,
    *,
    side: Side | None = None,
    position_intent: str | None = None,
) -> dict[str, Any]:
    policy = RiskPolicy()
    max_margin_pct = policy.max_margin_utilization_pct
    metadata: dict[str, Any] = {"max_margin_utilization_pct": max_margin_pct}
    account = snapshot.account
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    spec = DEFAULT_SPECS.get(pair)
    if account is None or quote_to_jpy is None or spec is None or max_margin_pct is None:
        return metadata
    if side is None:
        estimated_margin = estimate_required_margin_jpy(
            units=units,
            entry_price=entry,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
        )
        margin_free_hedge_units = 0
        broker_margin_free = 0
    else:
        estimated_margin = estimate_incremental_margin_jpy(
            pair=pair,
            side=side,
            units=units,
            entry_price=entry,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
            snapshot=snapshot,
            position_intent=position_intent,
        )
        margin_free_hedge_units = hedge_margin_free_units(
            pair=pair,
            side=side,
            snapshot=snapshot,
            position_intent=position_intent,
        )
        broker_margin_free = broker_margin_free_units(pair=pair, side=side, snapshot=snapshot)
    metadata.update(
        {
            "estimated_margin_jpy": round(estimated_margin, 3),
            "hedge_margin_free_units": margin_free_hedge_units,
            "broker_margin_free_units": broker_margin_free,
            "margin_budget_jpy": round(margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct), 3),
            "margin_used_jpy": round(account.margin_used_jpy, 3),
            "margin_available_jpy": round(account.margin_available_jpy, 3),
            "margin_utilization_after_pct": (
                round((account.margin_used_jpy + estimated_margin) / account.nav_jpy * 100.0, 3)
                if account.nav_jpy > 0
                else None
            ),
            "margin_rate": spec.margin_rate,
        }
    )
    return metadata


def _quote_to_jpy(pair: str, snapshot: BrokerSnapshot) -> float | None:
    quote_ccy = pair.split("_", 1)[1]
    if quote_ccy == "JPY":
        return 1.0
    home_conversion = snapshot.home_conversions.get(quote_ccy)
    if home_conversion is not None and home_conversion > 0:
        return float(home_conversion)
    conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is None:
        return None
    return max(conversion_quote.bid, conversion_quote.ask)


def _intent_to_json(intent: OrderIntent) -> dict[str, Any]:
    return {
        "pair": intent.pair,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": intent.units,
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
        "thesis": intent.thesis,
        "owner": intent.owner.value,
        "market_context": {
            "regime": intent.market_context.regime if intent.market_context else "",
            "narrative": intent.market_context.narrative if intent.market_context else "",
            "chart_story": intent.market_context.chart_story if intent.market_context else "",
            "method": intent.market_context.method.value if intent.market_context else "",
            "invalidation": intent.market_context.invalidation if intent.market_context else "",
            "event_risk": intent.market_context.event_risk if intent.market_context else "",
            "session": intent.market_context.session if intent.market_context else "",
        },
        "metadata": intent.metadata,
    }


def _snapshot_from_json(payload: dict[str, Any]) -> BrokerSnapshot:
    positions = tuple(
        BrokerPosition(
            trade_id=str(item["trade_id"]),
            pair=str(item["pair"]),
            side=Side.parse(str(item["side"])),
            units=int(item["units"]),
            entry_price=float(item["entry_price"]),
            unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
            take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
            stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
        )
        for item in payload.get("positions", []) or []
    )
    orders = tuple(
        BrokerOrder(
            order_id=str(item["order_id"]),
            pair=item.get("pair"),
            order_type=str(item.get("order_type") or ""),
            trade_id=item.get("trade_id"),
            price=float(item["price"]) if item.get("price") is not None else None,
            state=item.get("state"),
            units=int(item["units"]) if item.get("units") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            raw=snapshot_payload_order_raw(item),
        )
        for item in payload.get("orders", []) or []
    )
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        timestamp = item.get("timestamp_utc")
        quotes[pair] = Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc),
        )
    fetched = payload.get("fetched_at_utc")
    account = _account_summary_from_payload(payload.get("account"))
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=positions,
        orders=orders,
        quotes=quotes,
        account=account,
        home_conversions={str(k).upper(): float(v) for k, v in (payload.get("home_conversions") or {}).items()},
    )


def _snapshot_validation_time(snapshot: BrokerSnapshot | None) -> datetime:
    if snapshot is None:
        return datetime.now(timezone.utc)
    fetched_at = getattr(snapshot, "fetched_at_utc", None)
    if isinstance(fetched_at, datetime):
        return fetched_at.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _account_summary_from_payload(payload: object):
    from quant_rabbit.models import AccountSummary

    if not isinstance(payload, dict):
        return None
    fetched = payload.get("fetched_at_utc")
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        hedging_enabled=bool(payload.get("hedging_enabled") or False),
        fetched_at_utc=(
            datetime.fromisoformat(fetched) if isinstance(fetched, str) and fetched else datetime.now(timezone.utc)
        ),
    )
