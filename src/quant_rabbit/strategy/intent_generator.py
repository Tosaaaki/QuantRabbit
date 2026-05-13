from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.risk import (
    DEFAULT_SPECS,
    MIN_PRODUCTION_LOT_UNITS,
    RiskEngine,
    RiskPolicy,
    _min_lot_test_override_active,
    estimate_required_margin_jpy,
    margin_budget_jpy,
    resolve_max_loss_jpy,
)
from quant_rabbit.strategy.profile import StrategyProfile


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

# Structural range rails. 1σ VWAP bands are deliberately excluded here: they
# are often the box interior / magnet zone, not the actual fail point. Treating
# them as rails creates tiny spread-dominated boxes and makes range trades look
# worse than they are. Use outer bands, Donchian rails, swing extremes, and
# linear-regression channel edges as the executable rotation boundaries.
RANGE_SUPPORT_LEVEL_KEYS = ("bb_lower", "donchian_low", "avwap_lower_2sd", "swing_low", "linreg_channel_lower")
RANGE_RESISTANCE_LEVEL_KEYS = ("bb_upper", "donchian_high", "avwap_upper_2sd", "swing_high", "linreg_channel_upper")

# Generic trend/failure stops should not sit inside the current wick shelf.
# The buffer is one live spread beyond the adverse structural level because the
# spread is the current broker noise floor; the actual distance remains driven
# by ATR, spread, and chart-derived levels, not a fixed pip literal.
STRUCTURAL_STOP_BUFFER_SPREAD_MULT = 1.0


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
        indexed[pair] = per_tf
    return indexed if indexed else None


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
    long_score = _optional_float(per_tf.get("long_score"))
    short_score = _optional_float(per_tf.get("short_score"))
    bias = None
    if long_score is not None and short_score is not None and long_score != short_score:
        bias = Side.LONG.value if long_score > short_score else Side.SHORT.value
    m5_indicators = per_tf.get("M5") if isinstance(per_tf.get("M5"), dict) else {}
    m5_family = per_tf.get("M5__family_scores") if isinstance(per_tf.get("M5__family_scores"), dict) else {}
    conf = per_tf.get("confluence") if isinstance(per_tf.get("confluence"), dict) else {}
    return {
        "chart_long_score": long_score,
        "chart_short_score": short_score,
        "chart_direction_bias": bias,
        "m5_regime": _text_or_none(per_tf.get("M5__regime")),
        "m15_regime": _text_or_none(per_tf.get("M15__regime")),
        "h1_regime": _text_or_none(per_tf.get("H1__regime")),
        "m5_long_bias": _optional_float(per_tf.get("M5__long_bias")),
        "m5_short_bias": _optional_float(per_tf.get("M5__short_bias")),
        "m5_regime_quantile": _text_or_none(m5_indicators.get("regime_quantile")),
        "m5_mean_rev_score": _optional_float(m5_family.get("mean_rev_score")),
        "m5_trend_score": _optional_float(m5_family.get("trend_score")),
        "m5_breakout_score": _optional_float(m5_family.get("breakout_score")),
        "m5_family_disagreement": _optional_float(m5_family.get("disagreement")),
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
        "price_percentile_24h": _optional_float(conf.get("price_percentile_24h")),
        "price_percentile_7d": _optional_float(conf.get("price_percentile_7d")),
        "atr_percentile_24h": _optional_float(conf.get("atr_percentile_24h")),
        "range_24h_sigma_multiple": _optional_float(conf.get("range_24h_sigma_multiple")),
        "tf_agreement_score": _optional_float(conf.get("tf_agreement_score")),
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
    target_reward_risk = round(base_reward_risk * _regime_reward_risk_multiplier(execution_regime), 2)
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
    return _direction_bias_from_m5(chart_context) == side.value


def _direction_bias_from_m5(chart_context: dict[str, Any] | None) -> str | None:
    if not chart_context:
        return None
    long_bias = _optional_float(chart_context.get("m5_long_bias"))
    short_bias = _optional_float(chart_context.get("m5_short_bias"))
    if long_bias is None or short_bias is None or long_bias == short_bias:
        return None
    return Side.LONG.value if long_bias > short_bias else Side.SHORT.value


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


class IntentGenerator:
    def __init__(
        self,
        *,
        campaign_plan: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile: Path = DEFAULT_STRATEGY_PROFILE,
        output_path: Path = DEFAULT_ORDER_INTENTS,
        report_path: Path = DEFAULT_ORDER_INTENT_REPORT,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        max_loss_jpy: float | None = None,
        max_loss_pct: float | None = None,
        risk_equity_jpy: float | None = None,
    ) -> None:
        self.campaign_plan = campaign_plan
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path
        self.pair_charts_path = pair_charts_path
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy

    def run(self, *, snapshot_path: Path | None = None, max_candidates: int = 12) -> IntentGenerationSummary:
        plan = json.loads(self.campaign_plan.read_text())
        lanes = [lane for lane in plan.get("lanes", []) if _lane_can_attempt(lane)]
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
                m = _mirror_lane(lane)
                key = (m.get("desk"), m.get("pair"), m.get("direction"), m.get("method"))
                if key in seen_keys:
                    continue
                mirrors.append(m)
                seen_keys.add(key)
            lanes = lanes + mirrors
            max_candidates = max(max_candidates, max_candidates * 2)
        snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text())) if snapshot_path else None
        strategy_profile = StrategyProfile.load(self.strategy_profile) if self.strategy_profile.exists() else None
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
        # Load ATR / regime per pair from pair_charts.json. None when the file
        # is missing — _build_for_lane will surface MISSING_ATR_DATA so the
        # operator sees that geometry was built without market context.
        pair_charts = _load_pair_charts(self.pair_charts_path)
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
                        order_type_override=order_type,
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
        order_type_override: OrderType | None = None,
    ) -> GeneratedIntent:
        parent_lane_id = _lane_id(lane)
        lane_id = _variant_lane_id(parent_lane_id, order_type_override)
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
                live_blockers=(f"snapshot has no quote for {pair}",),
                note="Cannot build priced intent without a live quote.",
            )
        atr_pips = _atr_pips_for(pair, pair_charts)
        range_indicators = _range_indicators_for(pair, pair_charts)
        regime_state = _regime_state_for(pair, pair_charts)
        regime_reading = _regime_reading_for(pair, pair_charts)
        session_bucket = _session_bucket_for(pair, pair_charts)
        chart_context = _chart_context_for(pair, pair_charts)
        intent = _intent_from_lane(
            lane,
            quote,
            snapshot,
            max_loss_jpy=max_loss_jpy,
            atr_pips=atr_pips,
            range_indicators=range_indicators,
            order_type_override=order_type_override,
            parent_lane_id=parent_lane_id,
            regime_state=regime_state,
            regime_reading=regime_reading,
            session_bucket=session_bucket,
            chart_context=chart_context,
        )
        risk = RiskEngine(
            policy=RiskPolicy(
                block_new_entries_with_pending_entry_orders=False,
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=portfolio_loss_cap,
            )
        ).validate(
            intent,
            snapshot,
            for_live_send=False,
        )
        strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=False) if strategy_profile else ())
        )
        live_strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=True) if strategy_profile else ())
        )
        live_blockers = tuple(issue["message"] for issue in live_strategy_issues if issue.get("severity") == "BLOCK")
        risk_issues = list(issue.__dict__ for issue in risk.issues)
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
        else:
            risk_allowed = risk.allowed
        # Fix B (2026-05-12): _risk_budgeted_units returns 0 when the
        # current margin headroom can only support a sub-1000u lot. Surface
        # that as a BLOCK so the intent becomes DRY_RUN_BLOCKED — never
        # LIVE_READY — and the gateway never receives a fillable receipt at
        # an unprofitable lot size. 2026-05-12T07:46 UTC produced 201u
        # EUR_USD, 322u AUD_JPY, 2u GBP_USD entries whose spread cost
        # dominated any pip target; this gate stops the same pattern.
        if int(intent.units) == 0 and not _min_lot_test_override_active():
            risk_issues.append(
                {
                    "code": "MARGIN_TOO_THIN_FOR_MIN_LOT",
                    "message": (
                        f"available margin headroom can only fund "
                        f"<{MIN_PRODUCTION_LOT_UNITS}u for {pair}; refusing to "
                        "emit a sub-floor receipt because round-trip spread cost "
                        "would dominate the pip target. Free margin or wait for "
                        "open positions to harvest TP."
                    ),
                    "severity": "BLOCK",
                }
            )
            risk_allowed = False
        method = TradeMethod.parse(str(lane["method"]))
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


def _variant_lane_id(parent_lane_id: str, order_type: OrderType | None) -> str:
    if order_type == OrderType.MARKET:
        return f"{parent_lane_id}:MARKET"
    return parent_lane_id


def _order_variants_for(lane: dict[str, Any]) -> tuple[OrderType, ...]:
    method = TradeMethod.parse(str(lane["method"]))
    base = _order_type_for(method)
    if _lane_forbids_market_chase(lane):
        return (base,)
    return (base, OrderType.MARKET)


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


def _intent_from_lane(
    lane: dict[str, Any],
    quote: Quote,
    snapshot: BrokerSnapshot,
    *,
    max_loss_jpy: float,
    atr_pips: float | None = None,
    range_indicators: dict[str, Any] | None = None,
    order_type_override: OrderType | None = None,
    parent_lane_id: str | None = None,
    regime_state: str | None = None,
    regime_reading: dict[str, Any] | None = None,
    session_bucket: str | None = None,
    chart_context: dict[str, Any] | None = None,
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
    position_metadata = _position_intent_metadata(pair, side, snapshot)
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
    units = _risk_budgeted_units(pair, entry, sl, max_loss_jpy=max_loss_jpy, snapshot=snapshot)
    margin_metadata = _margin_sizing_metadata(pair, entry, units, snapshot)
    thesis = f"{lane['desk']} {pair} {side.value} {method.value} {target_reward_risk:.2f}R: {lane['required_receipt']}"
    if execution_regime:
        regime_context = f"{execution_regime} current; {method.value} campaign lane"
        if regime_state and regime_state != execution_regime:
            regime_context += f"; dominant={regime_state}"
    else:
        regime_context = f"{method.value} campaign lane"
    context = MarketContext(
        regime=regime_context,
        narrative=str(lane.get("reason") or ""),
        chart_story=" | ".join(str(item) for item in lane.get("story_examples", [])[:2]) or "campaign lane requires current chart read",
        method=method,
        invalidation=f"invalid if SL {sl} trades or campaign overlay vetoes the setup",
        event_risk="; ".join(str(item) for item in lane.get("blockers", [])[:2]),
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
            "required_receipt": lane.get("required_receipt"),
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
                f"floor units to the largest broker size under the {max_loss_jpy:.0f} JPY loss cap "
                f"and {RiskPolicy().max_margin_utilization_pct:.1f}% margin utilization cap"
            ),
            "max_loss_jpy": max_loss_jpy,
            "parent_lane_id": parent_lane_id or _lane_id(lane),
            "order_timing": "NOW_MARKET" if order_type == OrderType.MARKET else "PENDING_TRIGGER",
            **position_metadata,
            **geometry_metadata,
            **margin_metadata,
        },
    )


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
    bias = _method_direction_bias(metadata, method)
    if bias in {Side.LONG.value, Side.SHORT.value} and bias != intent.side.value:
        long_score = metadata.get("m5_long_bias") if method == TradeMethod.RANGE_ROTATION else metadata.get("chart_long_score")
        short_score = metadata.get("m5_short_bias") if method == TradeMethod.RANGE_ROTATION else metadata.get("chart_short_score")
        scope = "M5 range" if method == TradeMethod.RANGE_ROTATION else "pair_charts"
        # Phase 2 (user 2026-05-08「逆もまた然り」): under SL-free the AI
        # trader is the discretionary direction picker — historical
        # chart_score bias is one input, not a hard veto. The MTF + PA +
        # micro-override scoring in trader_brain decides the side; this
        # gate becomes WARN so symmetric mirror lanes reach LIVE_READY.
        severity = "WARN" if _sl_free_active() else "BLOCK"
        issues.append(
            {
                "code": "CHART_DIRECTION_CONFLICT",
                "message": (
                    f"{intent.pair} {intent.side.value} conflicts with current {scope} direction "
                    f"bias={bias} (long_score={long_score}, short_score={short_score}); "
                    "wait for this side to dominate or choose the aligned lane."
                ),
                "severity": severity,
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

    # C — 2026-05-13 "no chasing exhausted moves" filter. When the pair
    # has already covered >= EXHAUSTION_RANGE_SIGMA_MULTIPLE (= 2.0,
    # standard 2σ boundary) of its typical H1 range in the last 24
    # hours, AND the new entry direction is aligned with where the
    # move travelled (LONG into a 24h high, SHORT into a 24h low),
    # block it. Fading the exhausted move (opposite direction) is not
    # blocked; same-direction chasing is the failure mode
    # 2026-05-12T15:33 UTC drove the operator to demand killing.
    sigma_mult = _optional_float(metadata.get("range_24h_sigma_multiple"))
    ppct_24h = _optional_float(metadata.get("price_percentile_24h"))
    if sigma_mult is not None and sigma_mult >= EXHAUSTION_RANGE_SIGMA_MULTIPLE:
        chasing = False
        if intent.side == Side.LONG and ppct_24h is not None and ppct_24h >= 0.5:
            chasing = True
        if intent.side == Side.SHORT and ppct_24h is not None and ppct_24h <= 0.5:
            chasing = True
        if chasing:
            issues.append(
                {
                    "code": "EXHAUSTION_RANGE_CHASE",
                    "message": (
                        f"{intent.pair} {intent.side.value} chases a move already "
                        f"{sigma_mult:.2f}× typical hourly range over 24h "
                        f"(p24h={ppct_24h:.2f}); refuse same-direction entry after the "
                        f"{EXHAUSTION_RANGE_SIGMA_MULTIPLE:.1f}σ-equivalent extension."
                    ),
                    "severity": "BLOCK",
                }
            )
    return issues


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


def _order_type_for(method: TradeMethod) -> OrderType:
    if method == TradeMethod.RANGE_ROTATION:
        return OrderType.LIMIT
    return OrderType.STOP_ENTRY


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
    if tag in {"TOKYO_KILLZONE", "ASIA_OPEN", "TOKYO"}:
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


def _position_intent_metadata(pair: str, side: Side, snapshot: BrokerSnapshot) -> dict[str, Any]:
    same_pair = [position for position in snapshot.positions if position.pair == pair and position.owner == Owner.TRADER]
    if not same_pair:
        return {}
    if any(position.side != side for position in same_pair):
        return {"position_intent": "HEDGE", "position_fill": "OPEN_ONLY"}
    return {"position_intent": "PYRAMID", "position_fill": "OPEN_ONLY"}


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
    if order_type not in {OrderType.LIMIT, OrderType.MARKET} or not range_indicators:
        return {
            "geometry_model": "ATR_SPREAD_STRUCTURE",
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
        return {"geometry_model": "ATR_SPREAD"}
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
    }


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


def _risk_budgeted_units(pair: str, entry: float, sl: float, *, max_loss_jpy: float, snapshot: BrokerSnapshot) -> int:
    pip_factor = PIP_FACTORS[pair]
    stop_pips = abs(entry - sl) * pip_factor
    if stop_pips <= 0:
        return 1
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return 1
    margin_budget_units = _margin_budgeted_units(pair, entry, snapshot)
    # SL-free mode (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive 「損失を出さない
    # で稼ぎまくる」, 2026-05-07): sizing is operator-anchored, not loss-anchored.
    # Sizing precedence (highest first):
    #   1. QR_TRADER_POSITION_NAV_PCT — % of NAV used as margin per position.
    #      Auto-scales with equity (user 2026-05-08「BaseUnitを決めると、
    #      資産が増えたときに追従できないよ。％で決めないといけなくない？」).
    #   2. QR_TRADER_BASE_UNITS — legacy fixed-unit fallback.
    #   3. Hard-coded 3000 unit fallback.
    # Margin headroom (`_margin_budgeted_units`) still caps the result so the
    # 92% portfolio margin utilization gate is never breached.
    sl_free_active = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if sl_free_active:
        nav_pct_units = _nav_pct_position_units(pair, entry, snapshot)
        if nav_pct_units is not None:
            target_units = nav_pct_units
        else:
            try:
                target_units = float(os.environ.get("QR_TRADER_BASE_UNITS", "3000") or "3000")
            except ValueError:
                target_units = 3000.0
        candidates = [target_units]
        if margin_budget_units is not None:
            candidates.append(margin_budget_units)
        max_units = max(1.0, min(candidates))
    else:
        loss_budget_units = max_loss_jpy * pip_factor / (stop_pips * quote_to_jpy)
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


def _margin_budgeted_units(pair: str, entry: float, snapshot: BrokerSnapshot) -> float | None:
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
    budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
    if budget <= 0:
        return 0.0
    margin_per_unit = estimate_required_margin_jpy(units=1, entry_price=entry, quote_to_jpy=quote_to_jpy, spec=spec)
    if margin_per_unit <= 0:
        return 0.0
    return budget / margin_per_unit


def _margin_sizing_metadata(pair: str, entry: float, units: int, snapshot: BrokerSnapshot) -> dict[str, Any]:
    policy = RiskPolicy()
    max_margin_pct = policy.max_margin_utilization_pct
    metadata: dict[str, Any] = {"max_margin_utilization_pct": max_margin_pct}
    account = snapshot.account
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    spec = DEFAULT_SPECS.get(pair)
    if account is None or quote_to_jpy is None or spec is None or max_margin_pct is None:
        return metadata
    estimated_margin = estimate_required_margin_jpy(units=units, entry_price=entry, quote_to_jpy=quote_to_jpy, spec=spec)
    metadata.update(
        {
            "estimated_margin_jpy": round(estimated_margin, 3),
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
