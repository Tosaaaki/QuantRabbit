from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Short-term momentum thresholds (M1/M5 ADX). Above HIGH the move is live and
# MARKET catches it; below LOW the tape is quiet and STOP-ENTRY/LIMIT save the
# spread by waiting for the trigger. Tuned for FX major pairs in 2026 sessions
# where M5 ADX rarely exceeds 60.
SHORT_TERM_MOMENTUM_HIGH_ADX = 25.0
SHORT_TERM_MOMENTUM_LOW_ADX = 18.0
_CHART_STORY_M1_ADX_PATTERN = re.compile(r"M1\([^)]*ADX=([\d.]+)")
_CHART_STORY_M5_ADX_PATTERN = re.compile(r"M5\([^)]*ADX=([\d.]+)")

# Micro-structure alignment: M1/M5 most-recent BOS/CHOCH direction. Kept
# for backward compatibility with the prior commit; the production scoring
# now uses `_mtf_confluence_score` which spans all 7 timeframes and 5
# indicator lenses (struct + regime + Supertrend + cloud + RSI extreme).
# These constants survive only as the simpler M1/M5-only scorer used by
# legacy unit tests; new direction scoring goes through MTF_CONFLUENCE_*.
MICRO_STRUCTURE_ALIGNED_BONUS = 6.0
MICRO_STRUCTURE_OPPOSED_PENALTY = -12.0
_CHART_STORY_M1_STRUCT_PATTERN = re.compile(r"M1\([^)]*struct=(BOS|CHOCH)_(UP|DOWN)@")
_CHART_STORY_M5_STRUCT_PATTERN = re.compile(r"M5\([^)]*struct=(BOS|CHOCH)_(UP|DOWN)@")

# MTF confluence scoring (production): 7 timeframes × 5 lenses, positive-
# biased per user directive 2026-05-08 「エントリーしない理由ではなく、
# エントリーする理由をみつけてほしい」「分析を広く」 (`feedback_analysis_breadth.md`,
# `feedback_s_always_exists.md`). The negative penalty is hard-capped so a
# single contrary signal cannot zero out an otherwise-aligned setup; the
# positive bonus scales with the number of agreeing lenses across the TF
# stack.
MTF_TF_WEIGHTS: dict[str, float] = {
    # Daily is the bias arbiter — slow but decisive. Weights sum to 1.0.
    "D": 0.20,
    "H4": 0.18,
    "H1": 0.16,
    "M30": 0.14,
    "M15": 0.12,
    "M5": 0.10,
    "M1": 0.10,
}
MTF_CONFLUENCE_POSITIVE_GAIN = 22.0   # max positive multiplier (≈ +30 ceiling)
MTF_CONFLUENCE_NEGATIVE_GAIN = 12.0   # capped per `MTF_CONFLUENCE_FLOOR`
MTF_CONFLUENCE_CEILING = 30.0
MTF_CONFLUENCE_FLOOR = -10.0
_TF_BLOCK_PATTERN = re.compile(r"\b(D|H4|H1|M30|M15|M5|M1)\(([^)]+)\)")
_LENS_PATTERNS: dict[str, "re.Pattern[str]"] = {
    "adx": re.compile(r"ADX=([-\d.]+)"),
    "rsi": re.compile(r"RSI=([-\d.]+)"),
    "atr_pips": re.compile(r"ATR=([-\d.]+)p"),
    "supertrend": re.compile(r"ST=([+-])"),
    "cloud": re.compile(r"cloud=(above|below)"),
    "struct": re.compile(r"struct=(BOS|CHOCH)_(UP|DOWN)@"),
    "read_label": re.compile(r"Read=([A-Z_]+):([\d.]+)"),
}
# Regime → directional vote. RANGE / UNCLEAR / TRANSITION leave the regime
# lens neutral but other lenses (struct/ST/cloud/RSI) still vote.
_REGIME_UP_TOKENS = {"TREND_UP", "IMPULSE_UP", "BULL"}
_REGIME_DOWN_TOKENS = {"TREND_DOWN", "IMPULSE_DOWN", "BEAR", "FAILURE_RISK"}
_REGIME_NEUTRAL_TOKENS = {"RANGE", "UNCLEAR", "TRANSITION"}


def _structural_chart_story(intent: dict[str, Any] | None) -> str:
    """Return the inline multi-TF structural chart_story for an intent.

    Looks first in `intent.metadata.chart_story_structural` (the
    intent_generator-injected pair_charts string), falls back to the
    legacy market_context.chart_story when the metadata key is absent
    (older receipts / tests). Returns "" when neither is available.
    """
    if not isinstance(intent, dict):
        return ""
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    metadata_story = str(metadata.get("chart_story_structural") or "")
    if metadata_story:
        return metadata_story
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    return str(market_context.get("chart_story") or "")


def _micro_structure_direction(intent_or_context: dict[str, Any] | None) -> str:
    """Return the dominant M1/M5 BOS/CHOCH direction.

    "UP" / "DOWN" when both M1 and M5 agree, or when only one timeframe
    publishes a struct event. "UNCLEAR" when the two timeframes conflict
    or when neither carries a struct field. Reads the inline structural
    chart_story injected by `intent_generator._chart_context_for`; a bare
    `market_context` dict still works for backward-compat tests.
    """
    if not isinstance(intent_or_context, dict):
        return "UNCLEAR"
    # Accept either a full intent dict (preferred) or a raw market_context
    # dict (legacy tests).
    if "metadata" in intent_or_context or "market_context" in intent_or_context:
        chart_story = _structural_chart_story(intent_or_context)
    else:
        chart_story = str(intent_or_context.get("chart_story") or "")
    if not chart_story:
        return "UNCLEAR"
    m1 = _CHART_STORY_M1_STRUCT_PATTERN.search(chart_story)
    m5 = _CHART_STORY_M5_STRUCT_PATTERN.search(chart_story)
    m1_dir = m1.group(2) if m1 else None
    m5_dir = m5.group(2) if m5 else None
    if m1_dir and m5_dir:
        return m1_dir if m1_dir == m5_dir else "UNCLEAR"
    if m5_dir:
        return m5_dir
    if m1_dir:
        return m1_dir
    return "UNCLEAR"


def _micro_structure_alignment_score(
    intent: dict[str, Any],
    rationale: list[str],
    blockers: list[str],
) -> float:
    """Legacy M1/M5-only scorer. Production now uses `_mtf_confluence_score`."""
    direction = str(intent.get("side") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return 0.0
    micro = _micro_structure_direction(intent)
    if micro == "UNCLEAR":
        return 0.0
    aligned = (direction == "LONG" and micro == "UP") or (direction == "SHORT" and micro == "DOWN")
    if aligned:
        rationale.append(f"M1/M5 micro-structure {micro} agrees with {direction} thesis")
        return MICRO_STRUCTURE_ALIGNED_BONUS
    rationale.append(
        f"M1/M5 micro-structure {micro} opposes {direction} thesis — fresh flip risk against entry"
    )
    return MICRO_STRUCTURE_OPPOSED_PENALTY


def _parse_chart_story_full(chart_story: str) -> dict[str, dict[str, Any]]:
    """Parse the inline chart_story into per-TF indicator dicts.

    Returns a mapping like {"M1": {"regime": "RANGE", "adx": 23.5,
    "rsi": 60.1, "supertrend": "UP", "cloud": "above",
    "struct_type": "BOS", "struct_dir": "UP", "read_label": "TREND_WEAK",
    "read_confidence": 0.33}, "M5": {...}, ...}. Each field is optional —
    callers must check for presence (not all TFs publish cloud, e.g. H4).
    """
    if not chart_story:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for match in _TF_BLOCK_PATTERN.finditer(chart_story):
        tf = match.group(1)
        body = match.group(2)
        # First token before the comma is the regime label.
        head, _, rest = body.partition(",")
        data: dict[str, Any] = {"regime": head.strip()}
        full_body = rest if rest else body
        for key in ("adx", "rsi", "atr_pips"):
            m = _LENS_PATTERNS[key].search(full_body)
            if m:
                try:
                    data[key] = float(m.group(1))
                except ValueError:
                    continue
        m = _LENS_PATTERNS["supertrend"].search(full_body)
        if m:
            data["supertrend"] = "UP" if m.group(1) == "+" else "DOWN"
        m = _LENS_PATTERNS["cloud"].search(full_body)
        if m:
            data["cloud"] = m.group(1)
        m = _LENS_PATTERNS["struct"].search(full_body)
        if m:
            data["struct_type"] = m.group(1)
            data["struct_dir"] = m.group(2)
        m = _LENS_PATTERNS["read_label"].search(full_body)
        if m:
            data["read_label"] = m.group(1)
            try:
                data["read_confidence"] = float(m.group(2))
            except ValueError:
                pass
        result[tf] = data
    return result


def _tf_lens_support(tf_data: dict[str, Any], direction: str) -> tuple[float, float, list[str]]:
    """Per-TF support for `direction` across 5 lenses.

    Returns (raw_support, max_possible, lens_reasons). raw_support is the
    sum of lens points that vote with `direction`; max_possible is the
    total points available from lenses that actually published a signal.
    The caller normalizes raw_support / max_possible × ADX/Read multiplier
    to produce a per-TF support score in roughly [0, 1.5]. Lens points:
        struct (BOS/CHOCH): 1.0 — most recent swing event, strongest signal
        regime (TREND_UP/DOWN, IMPULSE_*): 1.0 — stable directional bias
        supertrend (ST=+/-): 0.7 — trend confirmation
        cloud (above/below): 0.5 — Ichimoku bias
        rsi extreme (>70 SHORT, <30 LONG): 0.5 — mean-reversion bias
    """
    if direction not in {"LONG", "SHORT"}:
        return 0.0, 0.0, []
    target_up = direction == "LONG"
    raw = 0.0
    max_possible = 0.0
    reasons: list[str] = []

    struct_dir = tf_data.get("struct_dir")
    if struct_dir in {"UP", "DOWN"}:
        max_possible += 1.0
        if (struct_dir == "UP") == target_up:
            raw += 1.0
            reasons.append(f"{tf_data.get('struct_type', 'STRUCT')}_{struct_dir}")

    regime = str(tf_data.get("regime") or "")
    if regime in _REGIME_UP_TOKENS or regime in _REGIME_DOWN_TOKENS:
        max_possible += 1.0
        regime_up = regime in _REGIME_UP_TOKENS
        if regime_up == target_up:
            raw += 1.0
            reasons.append(regime)
    # RANGE / UNCLEAR / TRANSITION: no regime contribution either way.

    supertrend = tf_data.get("supertrend")
    if supertrend in {"UP", "DOWN"}:
        max_possible += 0.7
        if (supertrend == "UP") == target_up:
            raw += 0.7
            reasons.append(f"ST={'+' if supertrend == 'UP' else '-'}")

    cloud = tf_data.get("cloud")
    if cloud in {"above", "below"}:
        max_possible += 0.5
        if (cloud == "above") == target_up:
            raw += 0.5
            reasons.append(f"cloud={cloud}")

    rsi = tf_data.get("rsi")
    if isinstance(rsi, (int, float)):
        if rsi >= 70 and not target_up:
            raw += 0.5
            max_possible += 0.5
            reasons.append(f"RSI={rsi:.0f} OB")
        elif rsi <= 30 and target_up:
            raw += 0.5
            max_possible += 0.5
            reasons.append(f"RSI={rsi:.0f} OS")

    return raw, max_possible, reasons


def _tf_strength_multiplier(tf_data: dict[str, Any]) -> float:
    """ADX strength × Read confidence = per-TF signal weight."""
    multiplier = 1.0
    adx = tf_data.get("adx")
    if isinstance(adx, (int, float)):
        if adx >= 30.0:
            multiplier *= 1.30
        elif adx >= 20.0:
            multiplier *= 1.10
        elif adx < 15.0:
            multiplier *= 0.70
    read_conf = tf_data.get("read_confidence")
    if isinstance(read_conf, (int, float)):
        # 0.0..1.0 confidence → 0.5..1.0 multiplier so a high-confidence Read
        # boosts the weight without zeroing low-confidence TFs entirely.
        multiplier *= 0.5 + 0.5 * float(read_conf)
    return multiplier


def _mtf_confluence_score(
    intent: dict[str, Any],
    rationale: list[str],
    blockers: list[str],
) -> float:
    """7-TF × 5-lens confluence scoring with positive bias.

    Aggregates per-TF lens support for the lane direction, weights by TF
    importance (`MTF_TF_WEIGHTS`), and applies an ADX × Read confidence
    multiplier per TF. Positive (aligned) signals scale to a +30 ceiling;
    negative (opposing) signals are hard-capped at -10 so a single contrary
    signal cannot zero out an otherwise-aligned setup. The rationale
    surfaces the aligned lenses for the operator so the GPT verifier sees
    why the lane scored positive, even when the net score is moderate.
    """
    direction = str(intent.get("side") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return 0.0
    chart_story = _structural_chart_story(intent)
    if not chart_story:
        return 0.0
    tf_data = _parse_chart_story_full(chart_story)
    if not tf_data:
        return 0.0
    opposite = "SHORT" if direction == "LONG" else "LONG"

    # Dynamic TF weights (user 2026-05-11「TFの組み合わせは状況で変わる」+
    # 残研究 1/2/3): session × method × pair × ATR percentile × news
    # calendar weighting overrides the legacy MTF_TF_WEIGHTS literal.
    # Falls back to the literal when no situation context is available
    # so existing tests stay stable.
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    session_str = str(market_context.get("session") or metadata.get("session_bucket") or "")
    dominant_regime = str(market_context.get("regime") or "")
    method_str = str(market_context.get("method") or "")
    pair_str = str(intent.get("pair") or "")
    try:
        from quant_rabbit.strategy.tf_weights import dynamic_tf_weights
        weights, situation_label = dynamic_tf_weights(
            session=session_str,
            chart_story=chart_story,
            dominant_regime=dominant_regime,
            method=method_str,
            pair=pair_str,
            # pair_chart not available here cheaply; ATR percentile boost
            # therefore relies on chart_story_structural ADX values via
            # the situation classifier. Acceptable: PA aggregate (which
            # has pair_chart) gets the full ATR percentile boost.
        )
    except Exception:
        weights, situation_label = dict(MTF_TF_WEIGHTS), "baseline"

    aligned_weighted = 0.0
    opposed_weighted = 0.0
    aligned_summary: list[str] = []
    opposed_summary: list[str] = []
    aligned_lens_count = 0

    for tf, weight in weights.items():
        data = tf_data.get(tf)
        if not data:
            continue
        multiplier = _tf_strength_multiplier(data)
        aligned_raw, aligned_max, aligned_reasons = _tf_lens_support(data, direction)
        opposed_raw, opposed_max, _ = _tf_lens_support(data, opposite)

        if aligned_max > 0:
            tf_aligned = (aligned_raw / aligned_max) * multiplier
            aligned_weighted += tf_aligned * weight
            if aligned_raw >= 0.5:
                aligned_lens_count += len(aligned_reasons)
                aligned_summary.append(
                    f"{tf}({','.join(aligned_reasons)})"
                )
        if opposed_max > 0 and opposed_raw >= 0.7:
            tf_opposed = (opposed_raw / opposed_max) * multiplier
            opposed_weighted += tf_opposed * weight
            if opposed_raw >= 1.0:
                opposed_summary.append(tf)

    # Confluence bonus: each aligned lens beyond the 4th adds a small extra
    # boost so a setup confirmed by many lenses outscores one confirmed by
    # only the highest-weight TF. Caps at +5 to avoid runaway.
    confluence_bonus = min(5.0, max(0.0, (aligned_lens_count - 4) * 0.8))

    positive_score = aligned_weighted * MTF_CONFLUENCE_POSITIVE_GAIN
    negative_score = -min(opposed_weighted * MTF_CONFLUENCE_NEGATIVE_GAIN, abs(MTF_CONFLUENCE_FLOOR))
    total = max(MTF_CONFLUENCE_FLOOR, min(MTF_CONFLUENCE_CEILING, positive_score + negative_score + confluence_bonus))

    if aligned_summary:
        rationale.append(
            f"MTF confluence [{situation_label}]: {direction} aligned at "
            + " ".join(aligned_summary)
            + f" (+{positive_score + confluence_bonus:.1f})"
        )
    if opposed_summary:
        rationale.append(
            f"MTF caution [{situation_label}]: opposing signals at {','.join(opposed_summary)} ({negative_score:.1f}, capped)"
        )
    if not aligned_summary and not opposed_summary:
        rationale.append(f"MTF confluence [{situation_label}]: no decisive lens for {direction} (net 0)")

    return total


def _short_term_momentum_class(intent_or_context: dict[str, Any] | None) -> str:
    """Read M1/M5 ADX off the inline structural chart_story to classify momentum.

    Returns "HIGH" when the average M1+M5 ADX is at/above the breakout
    threshold (move in progress, MARKET fills catch it), "LOW" when the
    average is at/below the quiet threshold (range/transition, pending
    triggers fill more cheaply), and "NEUTRAL" otherwise. Accepts either
    a full intent dict (preferred — reads `metadata.chart_story_structural`)
    or a raw market_context dict for backward compatibility.
    """
    if not isinstance(intent_or_context, dict):
        return "NEUTRAL"
    if "metadata" in intent_or_context or "market_context" in intent_or_context:
        chart_story = _structural_chart_story(intent_or_context)
    else:
        chart_story = str(intent_or_context.get("chart_story") or "")
    if not chart_story:
        return "NEUTRAL"
    m1 = _CHART_STORY_M1_ADX_PATTERN.search(chart_story)
    m5 = _CHART_STORY_M5_ADX_PATTERN.search(chart_story)
    if not m1 or not m5:
        return "NEUTRAL"
    try:
        avg_adx = (float(m1.group(1)) + float(m5.group(1))) / 2.0
    except (TypeError, ValueError):
        return "NEUTRAL"
    if avg_adx >= SHORT_TERM_MOMENTUM_HIGH_ADX:
        return "HIGH"
    if avg_adx <= SHORT_TERM_MOMENTUM_LOW_ADX:
        return "LOW"
    return "NEUTRAL"

from quant_rabbit.models import BrokerSnapshot, Owner, Side, TradeMethod


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_TRADER_SETTINGS,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_DECISION,
    DEFAULT_TRADER_DECISION_REPORT,
)
from quant_rabbit.strategy.price_action import aggregate_price_action_score
from quant_rabbit.strategy.lane_history_ledger import (
    LaneHistorySnapshot,
    compute_lane_history,
    lane_history_modifier,
)
from quant_rabbit.strategy.regime_classifier import (
    RegimeSnapshot,
    classify_all as regime_classify_all,
    regime_score_modifier,
)
from quant_rabbit.strategy.entry_timing_gate import check_entry_timing
from quant_rabbit.strategy.trader_overrides import (
    TraderOverrides,
    load_trader_overrides,
    overrides_block_check,
    overrides_score_delta,
)
from quant_rabbit.strategy.news_themes import NewsThemes, parse_news_themes
from quant_rabbit.strategy.reversal_signal import detect_reversal
from quant_rabbit.strategy.pattern_signals import (
    aggregate_pattern_score,
    detect_pattern_signals,
)
from quant_rabbit.strategy.forward_projection import (
    aggregate_projection_score,
    detect_forward_projections,
)
from quant_rabbit.strategy.correlation_predictor import (
    aggregate_correlation_lag_score,
    build_correlation_map,
    detect_correlation_lag,
)
from quant_rabbit.strategy.path_projection import (
    aggregate_path_score,
    detect_paths,
)
from quant_rabbit.strategy.directional_forecaster import (
    DirectionalForecast,
    ENTRY_CONFIDENCE_MIN,
    synthesize_forecast,
)
from quant_rabbit.strategy.forecast_persistence_tracker import (
    record_forecast,
)


# Rank ceiling for "primary attack" lanes consumed by the trader_brain
# prefilter. Mirrors `gpt_trader.PRIMARY_ATTACK_RANK_CEILING` so the
# prefilter and the GPT verifier agree on which advised lanes are
# high-conviction enough to count as primary-basket overlay. Both
# constants must stay in sync; a regression test enforces equality.
# The same K cap that gates verifier basket coverage gates the trader_brain
# promotion. Per AGENT_CONTRACT §8 the operator's deterministic prefilter
# must surface advised lanes alongside its own ranking; the rank gap below
# K is the conviction gate that prevents low-rank lane spam from leaking
# in. K=4 matches `RiskPolicy.max_portfolio_positions` for the live config.
ATTACK_ADVICE_PROMOTION_RANK_CEILING = 4
# Flat score bonus applied to LIVE_READY lanes that appear in the top-K
# attack_advice list. Sized so it edges advised lanes ahead of unadvised
# peers at score parity (typical _score_lane delta between adjacent
# LIVE_READY lanes is 5–15), without dwarfing the MTF / price-action / RR
# components that should still discriminate between advised candidates.
ATTACK_ADVICE_PROMOTION_BONUS = 10.0

# === Directional gating constants (C-1 + C-2, 2026-05-12) ===
# The trader_brain prefilter previously emitted both LONG and SHORT
# LIVE_READY lanes for the same pair regardless of `pair_charts`
# directional bias. Under margin pressure the gateway then rejected
# every candidate at the staging step
# (`BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED`), so neither side entered
# on a clear reversal. The directional gate runs over the scored lane
# tuple after `_score_lane` but before basket construction and demotes
# the losing direction so the surviving side can claim available
# margin. These constants drive that gate.
#
# Threshold for what counts as a "decisive" pair_charts bias. The
# chart_reader's `_build_confluence` (chart_reader.py:604–610)
# already documents 0.05 as the natural noise floor between LONG and
# SHORT bias votes — gaps inside ±0.05 are labeled TIED. The strong
# threshold is twice that floor: a gap of |0.10|+ means the bias is
# twice as decisive as the documented noise band, which empirically
# corresponds to score_balance leans that survive an intra-cycle
# refresh. The multiplier is the smallest integer step that respects
# the chart_reader contract without copying a new market-derived
# literal into trader_brain.
_TIED_SCORE_GAP_BOUNDARY = 0.05  # mirrors chart_reader.py:605
DIRECTIONAL_GATING_STRONG_GAP_MULTIPLIER = 2.0
DIRECTIONAL_GATING_STRONG_GAP = (
    _TIED_SCORE_GAP_BOUNDARY * DIRECTIONAL_GATING_STRONG_GAP_MULTIPLIER
)
# C-2 attack_advice directional veto: score penalty applied to lanes
# whose direction is opposite to the top-K attack_advice majority for
# their pair. Magnitude matches the existing `-25` penalty applied
# elsewhere in `_score_lane` for non-CANDIDATE strategy profiles and
# direction-conflict downgrades — so the veto is internally
# consistent with the rest of the discretionary penalty grid.
ATTACK_ADVICE_VETO_PENALTY = 25.0

# === Precision filters (B/C/D, 2026-05-13) ===
# 2026-05-12T15:33 UTC mass-close incident drove the operator demand:
# the trader must stop chasing trends after the move has already
# happened, must reject entries when the multi-timeframe regime
# picture disagrees, and must penalize same-side participation at
# 24h-distribution extremes. The thresholds below are documented
# market-statistic boundaries, never JPY/pip literals.

# B — price percentile extremes. 0.95 / 0.05 are the standard 95%/5%
# distribution boundaries used everywhere from VaR to RSI extreme
# detection; entries at these tails are statistically late.
PRICE_PERCENTILE_EXTREME_HIGH = 0.95
PRICE_PERCENTILE_EXTREME_LOW = 0.05
PRICE_PERCENTILE_EXTREME_PENALTY = 25.0  # same magnitude as the other §6 penalty grid
PRICE_PERCENTILE_MEAN_REV_BONUS = 15.0   # smaller bonus for fading the extreme

# D — multi-TF agreement. Below this score the M15/M30/H1 regime
# picture is too fractured for confident participation. 0.67 = 2/3
# of timeframes agree — the smallest integer majority for a 3-TF
# panel, so it is the documented "majority" minimum, not a tuned
# market threshold.
TF_AGREEMENT_MAJORITY_THRESHOLD = 2.0 / 3.0
TF_AGREEMENT_DISAGREEMENT_PENALTY = 25.0


def _load_full_pair_charts_for_brain(pair_charts_path: Path = DEFAULT_PAIR_CHARTS) -> dict[str, dict[str, Any]]:
    """Load pair_charts.json keyed by pair, preserving the full views array.

    intent_generator's `_load_pair_charts` flattens views into per-TF keys
    for ATR/regime extraction; the price-action lens needs the raw views
    list because swings, structure_events, liquidity, order_blocks, and
    dealing_range live there. Returns {} on missing/malformed file so
    scoring can degrade to MTF-only without crashing.
    """
    if not pair_charts_path.exists():
        return {}
    try:
        payload = json.loads(pair_charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for chart in payload.get("charts", []) or []:
        pair = chart.get("pair")
        if isinstance(pair, str):
            out[pair] = chart
    return out


def _current_price_for(pair: str, snapshot: BrokerSnapshot, side: str) -> float | None:
    """Pick the appropriate side of the current quote for an entry."""
    if not pair or snapshot is None:
        return None
    quote = (snapshot.quotes or {}).get(pair) if hasattr(snapshot, "quotes") else None
    if quote is None:
        return None
    # Use ASK for LONG entries, BID for SHORT — that's what the trade fills at.
    return quote.ask if side == "LONG" else quote.bid


def _current_mid_price_for(pair: str, snapshot: BrokerSnapshot) -> float | None:
    """Return the quote midpoint for pair-level forecasting.

    Pair forecasts must be independent of candidate lane direction. Entry
    scoring still uses ask/bid through `_current_price_for`; the forecast uses
    midpoint so a LONG lane and a SHORT lane cannot synthesize different
    "single" forecasts just because they read opposite sides of the spread.
    """
    if not pair or snapshot is None:
        return None
    quote = (snapshot.quotes or {}).get(pair) if hasattr(snapshot, "quotes") else None
    if quote is None:
        return None
    try:
        bid = float(quote.bid)
        ask = float(quote.ask)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def _forecast_regime_label(pair_chart: dict[str, Any] | None) -> str | None:
    if not isinstance(pair_chart, dict):
        return None
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    raw = str((confluence or {}).get("dominant_regime") or "").upper()
    if "TREND" in raw:
        return "TREND"
    if "RANGE" in raw:
        return "RANGE"
    return raw[:20] if raw else None


def _pair_forecast(
    *,
    pair: str,
    pair_chart: dict[str, Any] | None,
    full_pair_charts: dict[str, dict[str, Any]] | None,
    snapshot: BrokerSnapshot | None,
    forecast_cache: dict[str, DirectionalForecast | None] | None,
    forecast_cycle_id: str | None,
) -> DirectionalForecast | None:
    """Build one pair-level forecast and cache it for the whole cycle.

    The previous implementation synthesized inside each candidate lane, so a
    LONG lane could see a LONG-biased forecast while a SHORT lane for the same
    pair saw a different SHORT-biased forecast, and both writes counted in the
    persistence ledger. This helper makes prediction a pair-level fact first;
    lane scoring then asks whether the lane agrees with that fact.
    """
    if not pair or pair_chart is None or snapshot is None:
        return None
    if forecast_cache is not None and pair in forecast_cache:
        return forecast_cache[pair]

    current_price = _current_mid_price_for(pair, snapshot)
    if current_price is None:
        if forecast_cache is not None:
            forecast_cache[pair] = None
        return None

    cot_payload = None
    option_skew_payload = None
    try:
        from quant_rabbit.paths import ROOT as _QR_ROOT

        cot_path = _QR_ROOT / "data" / "cot_snapshot.json"
        if cot_path.exists():
            cot_payload = json.loads(cot_path.read_text())
        option_path = _QR_ROOT / "data" / "option_skew_snapshot.json"
        if option_path.exists():
            option_skew_payload = json.loads(option_path.read_text())
    except Exception:
        pass

    pattern_signals = []
    projection_signals = []
    correlation_signals = []
    paths = []
    hit_rates = None
    regime_label = _forecast_regime_label(pair_chart)
    try:
        pattern_signals = detect_pattern_signals(
            pair_chart,
            cot_payload=cot_payload,
            option_skew_payload=option_skew_payload,
        )
    except Exception:
        pattern_signals = []
    try:
        from quant_rabbit.paths import (
            DEFAULT_CALENDAR_SNAPSHOT,
            DEFAULT_CROSS_ASSET_SNAPSHOT,
            ROOT as _QR_ROOT,
        )
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates as _compute_hit_rates

        hit_rates = _compute_hit_rates(_QR_ROOT / "data")
        projection_signals = detect_forward_projections(
            pair_chart,
            pair=pair,
            current_price=current_price,
            calendar_path=DEFAULT_CALENDAR_SNAPSHOT,
            cross_asset_path=DEFAULT_CROSS_ASSET_SNAPSHOT,
        )
    except Exception:
        projection_signals = []
    try:
        if full_pair_charts and pair in full_pair_charts:
            correlation_signals = detect_correlation_lag(pair, full_pair_charts)
    except Exception:
        correlation_signals = []
    try:
        # Evaluate both sides so the forecast is not biased by whichever lane
        # happened to call `_score_lane` first.
        paths = list(detect_paths(pair_chart, "LONG", current_price))
        paths.extend(detect_paths(pair_chart, "SHORT", current_price))
    except Exception:
        paths = []
    try:
        reversal_long = detect_reversal(pair_chart, "LONG")
    except Exception:
        reversal_long = None
    try:
        reversal_short = detect_reversal(pair_chart, "SHORT")
    except Exception:
        reversal_short = None

    forecast = synthesize_forecast(
        pair=pair,
        pair_chart=pair_chart,
        current_price=current_price,
        pattern_signals=pattern_signals,
        projection_signals=projection_signals,
        correlation_signals=correlation_signals,
        paths=paths,
        reversal_long=reversal_long,
        reversal_short=reversal_short,
        hit_rates=hit_rates,
        regime=regime_label,
    )
    if (
        forecast.direction == "UNCLEAR"
        and forecast.confidence <= 0.0
        and forecast.rationale_summary in {"no detector evidence", "forecaster disabled"}
    ):
        if forecast_cache is not None:
            forecast_cache[pair] = None
        return None
    if forecast_cache is not None:
        forecast_cache[pair] = forecast
    try:
        from quant_rabbit.paths import ROOT as _QR_ROOT

        record_forecast(
            forecast,
            data_root=_QR_ROOT / "data",
            cycle_id=forecast_cycle_id,
        )
    except Exception:
        pass
    return forecast


def _range_rotation_forecast_ready(intent: dict[str, Any]) -> bool:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    model = str(metadata.get("geometry_model") or "")
    if model == "RANGE_DIRECTIONAL_MARKET":
        return bool(metadata.get("range_directional_market"))
    if model not in {"RANGE_RAIL_LIMIT", "RANGE_RAIL_MARKET"}:
        return False
    return bool(metadata.get("range_tp_is_inside_box")) and bool(metadata.get("range_sl_outside_box"))


def _forecast_lane_gate(
    forecast: DirectionalForecast,
    *,
    direction: str,
    method: str,
    intent: dict[str, Any],
) -> tuple[bool, str]:
    """Return whether a candidate lane agrees with the pair forecast."""
    side = direction.upper()
    if forecast.direction == "UP":
        return side == "LONG", f"forecast UP {'aligned' if side == 'LONG' else 'opposes'} {side}"
    if forecast.direction == "DOWN":
        return side == "SHORT", f"forecast DOWN {'aligned' if side == 'SHORT' else 'opposes'} {side}"
    if forecast.direction == "RANGE":
        if method == TradeMethod.RANGE_ROTATION.value and _range_rotation_forecast_ready(intent):
            metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
            return (
                True,
                "forecast RANGE supports range rotation "
                f"({metadata.get('geometry_model')}, rails="
                f"{metadata.get('range_support')}–{metadata.get('range_resistance')})",
            )
        return (
            False,
            "forecast RANGE requires executable RANGE_ROTATION rail geometry, not trend/breakout chase",
        )
    return False, f"forecast {forecast.direction} has no executable edge"


JPY_CROSSES = {"AUD_JPY", "EUR_JPY", "GBP_JPY", "USD_JPY"}
PENDING_ENTRY_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
ACTION_SEND_ENTRY = "SEND_ENTRY"
ACTION_MONITOR_EXISTING = "MONITOR_EXISTING_EXPOSURE"
ACTION_NO_TRADE = "NO_TRADE"

# A historical worst loss becomes "large" only after it exceeds 1.8x the
# current per-trade cap. This preserves the old proportional warning behavior
# (-900 JPY when the cap was 500 JPY), while keeping the threshold tied to the
# active campaign cap instead of a stale JPY literal.
HISTORICAL_LARGE_LOSS_CAP_MULTIPLE = 1.8

# Risk-geometry scoring buckets are fractions of the current per-trade cap:
# <=60% is high-quality unused risk budget, <=90% is acceptable, above 90%
# leaves little room for spread/slippage drift. These are scoring weights only;
# the RiskEngine remains the executable risk authority.
LOW_RISK_CAP_FRACTION = 0.60
MEDIUM_RISK_CAP_FRACTION = 0.90

# Pending-order replacement tolerance. This is deliberately a spread multiple,
# not a fixed pip distance: a valid trigger can drift by more raw pips in thin
# liquidity, while liquid tape should tolerate less. Above this many current
# spreads, the pending price is no longer the same executable neighborhood.
PENDING_ENTRY_REPLACE_SPREAD_MULT = 8.0

# Narrative penalties are score/ranking inputs, not risk gates. The JPY
# intervention penalty must clear the size-multiple rounding step, otherwise
# rate-check / intervention risk can be visible in rationale but still round
# back to 1.00 size. It remains advisory: LIVE_READY receipts stay executable
# under AGENT_CONTRACT §6.
JPY_INTERVENTION_SCORE_PENALTY = 100.0
JPY_LIQUIDITY_SCORE_PENALTY = 25.0


@dataclass(frozen=True)
class LaneScore:
    lane_id: str
    pair: str
    direction: str
    method: str
    order_type: str
    entry: float | None
    tp: float | None
    sl: float | None
    status: str
    score: float
    action: str
    blockers: tuple[str, ...]
    rationale: tuple[str, ...]
    size_multiple: float = 1.0
    judgment: tuple[str, ...] = ()
    spread_pips: float | None = None
    # Per-lane margin estimate from intent_generator
    # (`risk_metrics.estimated_margin_jpy`). Threaded through so the
    # automation-layer margin-aware basket truncation (C-4, 2026-05-12)
    # can stop adding lanes when cumulative margin would exceed the
    # broker's `margin_available_jpy × MARGIN_AWARE_BASKET_BUFFER`
    # ceiling. Old smoke runs left this implicit and the gateway
    # rejected every candidate at the staging step with
    # `BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED`; surfacing it earlier
    # lets the basket converge on a sendable subset instead of
    # whack-a-mole rejecting all 12 lanes.
    estimated_margin_jpy: float | None = None


@dataclass(frozen=True)
class TraderDecision:
    action: str
    selected_lane_id: str | None
    generated_at_utc: str
    reason: str
    scores: tuple[LaneScore, ...]
    positions: int
    orders: int
    selected_lane_score: float | None = None
    selected_lane_size_multiple: float | None = None
    pending_cancel_order_ids: tuple[str, ...] = ()
    loss_cap_jpy: float | None = None
    loss_cap_source: str = ""


@dataclass(frozen=True)
class TraderSettings:
    score_bias: float = 0.0
    score_size_enabled: bool = True
    size_multiple_min: float = 0.7
    size_multiple_max: float = 1.8
    size_multiple_anchor_score: float = 110.0
    size_multiple_per_score_point: float = 0.005
    default_max_loss_jpy: float | None = None
    default_max_loss_pct: float | None = None


class TraderBrain:
    """Compare live-ready lanes using mined history, market story, and current risk state."""

    def __init__(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        trader_settings_path: Path = DEFAULT_TRADER_SETTINGS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        output_path: Path = DEFAULT_TRADER_DECISION,
        report_path: Path = DEFAULT_TRADER_DECISION_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.trader_settings_path = trader_settings_path
        self.target_state_path = target_state_path
        self.attack_advice_path = attack_advice_path
        self.pair_charts_path = pair_charts_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self, snapshot: BrokerSnapshot) -> TraderDecision:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents_payload = _load_json(self.intents_path)
        campaign_payload = _load_json(self.campaign_plan_path)
        strategy_payload = _load_json(self.strategy_profile_path)
        story_payload = _load_json(self.market_story_profile_path)
        attack_payload = _load_json(self.attack_advice_path)
        strategies = _strategy_index(strategy_payload)
        stories = _story_index(story_payload)
        campaign = _campaign_index(campaign_payload)
        # Map lane_id -> rank for the top-K advised lanes (rank 0..K-1).
        # Used by `_score_lane` to add a documented promotion bonus +
        # rationale so high-conviction advised setups outrank unadvised
        # peers in the prefilter SEND_ENTRY basket without overriding any
        # AGENT_CONTRACT §11 hard block (BLOCK_UNTIL_NEW_EVIDENCE,
        # missing-receipt, exposure, etc.). Missing/malformed advice
        # degrades silently — the map is empty and scoring is unchanged.
        attack_ranks = _attack_advice_top_k_ranks(attack_payload)
        # Full pair_charts entries (with views array) so `_score_lane` can
        # consult the price-action lens (swings, structure_events, dealing
        # range, order blocks, liquidity) on every candidate. Missing /
        # malformed file degrades gracefully — PA delta becomes 0 and the
        # legacy MTF confluence scoring still runs.
        full_pair_charts = _load_full_pair_charts_for_brain(self.pair_charts_path)
        trader_settings = _load_trader_settings(self.trader_settings_path)
        loss_cap_jpy, loss_cap_source = _resolve_trader_loss_cap(
            strategy_payload=strategy_payload,
            settings=trader_settings,
            target_state_path=self.target_state_path,
            snapshot=snapshot,
        )
        # Per-cycle context: lane history (recent (pair, direction) P&L
        # → bias modifier) and per-pair regime snapshot. Both computed
        # once per cycle and passed to _score_lane to avoid recomputing
        # on every candidate. They degrade gracefully — missing
        # execution_ledger.db or pair_charts → empty dicts → 0 modifier.
        from quant_rabbit.paths import ROOT as _QR_ROOT
        lane_history = compute_lane_history(_QR_ROOT / "data" / "execution_ledger.db")
        # full_pair_charts is keyed-by-pair dict; rebuild minimal payload
        # for regime_classifier which expects {"charts": [...]}.
        regime_snapshots = regime_classify_all({"charts": list(full_pair_charts.values())}) if full_pair_charts else {}
        # Module C: daily-review feedback. Empty when file is absent or
        # expired — no behavior change in that case.
        trader_overrides = load_trader_overrides(_QR_ROOT / "data")
        # Module A-extension: news theme parser. Reads the curated
        # `logs/news_digest.md` (produced by qr-news-digest routine) and
        # converts macro themes (USD strong, risk-off, pair-specific
        # bearish/bullish notes) into bounded per-(pair, direction)
        # score biases. Empty when digest is missing or unparseable.
        news_themes = parse_news_themes(_QR_ROOT / "logs" / "news_digest.md")
        positions = len(snapshot.positions)
        orders = len(snapshot.orders)
        pending_entries = _pending_entry_order_count(snapshot)
        portfolio_add_allowed = _portfolio_add_allowed(snapshot)
        exposure_blockers = () if portfolio_add_allowed else _exposure_blockers(snapshot)
        forecast_cache: dict[str, DirectionalForecast | None] = {}
        scores = tuple(
            sorted(
                (
                    self._score_lane(
                        result,
                        strategies,
                        stories,
                        campaign,
                        exposure_blockers,
                        trader_settings,
                        loss_cap_jpy=loss_cap_jpy,
                        full_pair_charts=full_pair_charts,
                        snapshot=snapshot,
                        attack_ranks=attack_ranks,
                        lane_history=lane_history,
                        regime_snapshots=regime_snapshots,
                        trader_overrides=trader_overrides,
                        news_themes=news_themes,
                        forecast_cache=forecast_cache,
                        forecast_cycle_id=generated_at,
                    )
                    for result in intents_payload.get("results", [])
                    if isinstance(result, dict) and isinstance(result.get("intent"), dict)
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        # C-1 + C-2 directional gating (2026-05-12). Operates on the
        # scored lane tuple BEFORE entry selection / basket construction.
        # Existing-position management (`_position_manager()` /
        # `_position_gateway()`) is invoked in `automation.py` against
        # `snapshot.positions` directly and does not read this tuple,
        # so the directional gate cannot reach SL/TP/CLOSE behavior on
        # any open trade — only the NEW-entry lane set is reshaped.
        scores = _apply_directional_gating(scores, full_pair_charts, attack_ranks)
        # Compute contaminated pending ids unconditionally so stale pendings
        # are flagged for cleanup whether we MONITOR or layer (AGENT_CONTRACT §11
        # — protected trader-owned exposure is not by itself a no-trade gate,
        # and a self-contradicting pending must not silently lock the basket).
        pending_cancel_order_ids = _contaminated_pending_order_ids(snapshot, scores)
        # MONITOR only when the open exposure is unprotected (portfolio_add_allowed
        # is False) and there's something to monitor. A pending entry alone does
        # NOT lock the basket when portfolio_add_allowed=True: the pending will
        # either get canceled here (if contaminated) or coexist with the new
        # entry via gateway basket validation. Prior behavior (2026-05-13 fix):
        # `if exposure_blockers or pending_entries:` self-locked the basket on a
        # single stale pending while 9 portfolio slots were free, contradicting
        # §11 and starving daily-target progress when the cycle held one
        # protected position.
        if exposure_blockers or (pending_entries and not portfolio_add_allowed):
            decision = TraderDecision(
                action=ACTION_MONITOR_EXISTING,
                selected_lane_id=None,
                selected_lane_score=None,
                selected_lane_size_multiple=None,
                generated_at_utc=generated_at,
                reason="Pending entry or non-layerable exposure is open; evaluate but do not add fresh risk.",
                scores=scores,
                positions=positions,
                orders=orders,
                pending_cancel_order_ids=pending_cancel_order_ids,
                loss_cap_jpy=loss_cap_jpy,
                loss_cap_source=loss_cap_source,
            )
        else:
            selected = _select_entry_lane(
                scores,
                target_state_path=self.target_state_path,
                snapshot=snapshot,
            )
            if selected:
                decision = TraderDecision(
                    action=ACTION_SEND_ENTRY,
                    selected_lane_id=selected.lane_id,
                    selected_lane_score=selected.score,
                    selected_lane_size_multiple=selected.size_multiple,
                    generated_at_utc=generated_at,
                    reason=_entry_selection_reason(scores, selected),
                    scores=scores,
                    positions=positions,
                    orders=orders,
                    pending_cancel_order_ids=pending_cancel_order_ids,
                    loss_cap_jpy=loss_cap_jpy,
                    loss_cap_source=loss_cap_source,
                )
            else:
                decision = TraderDecision(
                    action=ACTION_NO_TRADE,
                    selected_lane_id=None,
                    selected_lane_score=None,
                    selected_lane_size_multiple=None,
                    generated_at_utc=generated_at,
                    reason="No lane cleared trader-brain discretionary gates.",
                    scores=scores,
                    positions=positions,
                    orders=orders,
                    pending_cancel_order_ids=pending_cancel_order_ids,
                    loss_cap_jpy=loss_cap_jpy,
                    loss_cap_source=loss_cap_source,
                )
        self._write(decision)
        return decision

    def _score_lane(
        self,
        result: dict[str, Any],
        strategies: dict[tuple[str, str], dict[str, Any]],
        stories: dict[str, dict[str, Any]],
        campaign: dict[str, dict[str, Any]],
        exposure_blockers: tuple[str, ...],
        settings: TraderSettings,
        *,
        loss_cap_jpy: float | None,
        full_pair_charts: dict[str, dict[str, Any]] | None = None,
        snapshot: BrokerSnapshot | None = None,
        attack_ranks: dict[str, int] | None = None,
        lane_history: dict[tuple[str, str], LaneHistorySnapshot] | None = None,
        regime_snapshots: dict[str, RegimeSnapshot] | None = None,
        trader_overrides: TraderOverrides | None = None,
        news_themes: NewsThemes | None = None,
        forecast_cache: dict[str, DirectionalForecast | None] | None = None,
        forecast_cycle_id: str | None = None,
    ) -> LaneScore:
        intent = result["intent"]
        lane_id = str(result.get("lane_id") or "")
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        method = str((intent.get("market_context") or {}).get("method") or "")
        order_type = str(intent.get("order_type") or "")
        entry = _optional_float(intent.get("entry"))
        tp = _optional_float(intent.get("tp"))
        sl = _optional_float(intent.get("sl"))
        status = str(result.get("status") or "")
        risk_metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
        spread_pips = _optional_float(risk_metrics.get("spread_pips"))
        strategy = strategies.get((pair, direction), {})
        story = stories.get(pair, {})
        parent_lane_id = str((intent.get("metadata") or {}).get("parent_lane_id") or "")
        lane = campaign.get(lane_id) or campaign.get(parent_lane_id) or campaign.get(_parent_lane_id(lane_id)) or {}
        blockers: list[str] = list(exposure_blockers)
        rationale: list[str] = []
        score = 0.0

        if status == "LIVE_READY":
            score += 100.0
            rationale.append("live-ready risk/profile receipt")
        elif status == "DRY_RUN_PASSED":
            score += 35.0
            blockers.append("strategy still has live blockers")
        else:
            score -= 250.0
            blockers.append(f"intent status is {status}")

        profile_status = str(strategy.get("status") or "")
        if profile_status == "CANDIDATE":
            score += 25.0
            rationale.append("strategy profile candidate")
        elif profile_status:
            score -= 25.0
            # Mirror `strategy_profile.validate()` (profile.py:125): under
            # SL-free the per_trade cap bounds the loss, so non-CANDIDATE
            # profiles (BLOCK_UNTIL_NEW_EVIDENCE, RISK_REPAIR_CANDIDATE,
            # MINE_MISSED_EDGE, WATCH_ONLY) become advisory rationale
            # rather than hard blockers. §11 still applies — the profile
            # status itself is never auto-promoted — but the trader can
            # still take risk-bounded entries while waiting for the next
            # mine-strategy / receipt-promotion cycle. Mirrors
            # feedback_no_direction_bias_rules.md (sample-period BLOCK
            # should not become a permanent veto). Without SL-free the
            # legacy hard block remains.
            if _trader_sl_repair_disabled():
                rationale.append(
                    f"strategy profile is {profile_status} (advisory under SL-free; per_trade cap bounds loss)"
                )
            else:
                blockers.append(f"strategy profile is {profile_status}")
        else:
            score -= 40.0
            blockers.append("missing strategy profile")

        pretrade_net = float(strategy.get("pretrade_net_jpy") or 0.0)
        live_net = float(strategy.get("live_net_jpy") or 0.0)
        live_worst = _optional_float(strategy.get("live_worst_jpy"))
        if loss_cap_jpy is None:
            blockers.append("trader loss cap missing for historical evidence scaling")
        else:
            pretrade_component = _clamp(pretrade_net / loss_cap_jpy, -25.0, 25.0)
            live_component = _clamp(live_net / loss_cap_jpy, -30.0, 30.0)
            if status != "LIVE_READY" or pretrade_component > 0:
                score += pretrade_component
            if status != "LIVE_READY" or live_component > 0:
                score += live_component
        if pretrade_net > 0:
            rationale.append(f"positive pretrade evidence {pretrade_net:.0f} JPY")
        if live_net > 0:
            rationale.append(f"positive live evidence {live_net:.0f} JPY")
        if live_net < 0:
            rationale.append(f"negative live execution history {live_net:.0f} JPY; current receipt is the authority")
            if status != "LIVE_READY":
                score -= 20.0
        if loss_cap_jpy is not None and live_worst is not None and live_worst <= -loss_cap_jpy:
            if live_worst <= -(loss_cap_jpy * HISTORICAL_LARGE_LOSS_CAP_MULTIPLE):
                rationale.append(
                    f"historical live worst loss is large: {live_worst:.0f} JPY; current receipt repairs sizing"
                )
            else:
                rationale.append(f"old worst loss repaired only by current sizing: {live_worst:.0f} JPY")
            if status != "LIVE_READY":
                score -= 8.0

        method_pressure = int((story.get("methods") or {}).get(method, 0))
        score += _clamp(method_pressure * 0.25, 0.0, 30.0)
        if method_pressure:
            rationale.append(f"market-story method pressure {method_pressure}")
        themes = dict(story.get("themes") or {})
        examples = tuple(str(item) for item in story.get("examples", [])[:4])
        score += _method_theme_score(method, themes, rationale)
        score += _campaign_score(lane, rationale)
        score += _narrative_risk_score(pair, direction, method, themes, examples, blockers, rationale, status=status)
        score += _technical_consensus_score(
            intent=intent,
            method=method,
            status=status,
            strategy=strategy,
            story=story,
            lane=lane,
            risk_metrics=risk_metrics,
            method_pressure=method_pressure,
            loss_cap_jpy=loss_cap_jpy,
            rationale=rationale,
            blockers=blockers,
        )
        score += _direction_conflict_penalty(result, rationale)
        score += _mtf_confluence_score(intent, rationale, blockers)

        # Micro override: M1+M5 struct opposite to lane direction trumps the
        # historical-bias score. User 2026-05-08「ミクロのグラデーションでしょ？
        # 今の赤もショートで入ってたら勝てた」: when both micro TFs print struct
        # against the lane, the immediate move is the wrong direction — refuse
        # to enter regardless of how strong the historical evidence looks.
        # Single-TF flip is half-strength so a noisy M1 alone doesn't kill an
        # otherwise-clean lane.
        intent_side = str(intent.get("side") or "").upper()
        chart_story = _structural_chart_story(intent)
        m1_struct = _CHART_STORY_M1_STRUCT_PATTERN.search(chart_story)
        m5_struct = _CHART_STORY_M5_STRUCT_PATTERN.search(chart_story)
        target_up = intent_side == "LONG"
        m1_opp = bool(m1_struct and ((m1_struct.group(2) == "DOWN") == target_up))
        m5_opp = bool(m5_struct and ((m5_struct.group(2) == "DOWN") == target_up))
        if m1_opp and m5_opp:
            score -= 30.0
            rationale.append(
                f"micro override: M1+M5 both struct opposite to {intent_side} — historical bias ignored"
            )
        elif m1_opp or m5_opp:
            score -= 10.0
            which = "M1" if m1_opp else "M5"
            rationale.append(f"micro caution: {which} struct opposite to {intent_side}")
        # Price-action lens (user 2026-05-08「市況をちゃんとみれるようにして」):
        # SMC structural read across H4-M5 (swings, BOS/CHOCH events, dealing
        # range, order blocks, liquidity touches). Adds ±25 envelope to the
        # MTF confluence score so high-conviction entries earn extra weight
        # and weak setups (LONG into resistance, SHORT into expanding range)
        # are demoted before reaching live_ready selection.
        pa_pair = str(intent.get("pair") or "")
        pa_side = str(intent.get("side") or "").upper()
        pa_chart = (full_pair_charts or {}).get(pa_pair) if full_pair_charts else None
        pa_price = _current_price_for(pa_pair, snapshot, pa_side) if snapshot else None
        if pa_chart and pa_price and pa_side in {"LONG", "SHORT"}:
            pa_pip_factor = 100.0 if pa_pair.endswith("_JPY") else 10000.0
            pa_delta, pa_reasons = aggregate_price_action_score(
                pa_chart, pa_side, pa_price, pa_pip_factor
            )
            score += pa_delta
            if pa_reasons:
                # Surface the strongest PA reasons inline so the operator sees
                # the structural read in the lane rationale.
                rationale.append(f"price-action ({pa_delta:+.1f}): " + " ; ".join(pa_reasons[:2]))
            elif abs(pa_delta) >= 1.0:
                rationale.append(f"price-action ({pa_delta:+.1f}): no decisive structural lens")
        if order_type.upper() == "MARKET" and status == "LIVE_READY":
            # Regime-aware MARKET preference: catch live momentum, but don't
            # pay spread on quiet tape (user 2026-05-08 「市況によって柔軟に」
            # 「エントリー機会は逃さない」). Pending entries (LIMIT/STOP) cost
            # nothing until the trigger fires, so when M1/M5 ADX is quiet we
            # let those win the variant race; when ADX is hot we boost MARKET
            # to outscore pendings and grab the move.
            momentum = _short_term_momentum_class(intent)
            if momentum == "HIGH":
                score += 12.0
                rationale.append("MARKET catches active short-term momentum (M1/M5 ADX≥25)")
            elif momentum == "LOW":
                score -= 8.0
                rationale.append("short-term tape quiet (M1/M5 ADX≤18); pending entry preferred over MARKET")
            else:
                score += 5.0
                rationale.append("market receipt can execute the current quote instead of waiting for a trigger")

        # Severity-aware: intent_generator emits BLOCK for hard receipt issues
        # (geometry/units/missing-context) and WARN for advisory caveats like
        # CHART_DIRECTION_CONFLICT under SL-free. Only BLOCK-severity drops the
        # lane out of SEND_ENTRY; WARN issues surface in rationale and a small
        # score nudge so aligned lanes still outrank against-bias ones at parity
        # without becoming a hard veto. Drops the 100-point penalty that turned
        # the intent_generator WARN downgrade into a trader_brain hard block
        # (2026-05-11 incident: EUR_USD SHORT BREAKOUT_FAILURE held LIVE_READY
        # in intents but trader_brain blocked it, leaving the prefilter LONG-
        # only while ai_attack_advice ranked the SHORT side #1).
        risk_issues = result.get("risk_issues") or []
        hard_risk_issues = [
            issue for issue in risk_issues
            if str(issue.get("severity") or "").upper() == "BLOCK"
        ]
        warn_risk_issues = [
            issue for issue in risk_issues
            if str(issue.get("severity") or "").upper() == "WARN"
        ]
        if hard_risk_issues:
            blockers.extend(
                str(issue.get("message") or issue.get("code")) for issue in hard_risk_issues
            )
            score -= 100.0
        # Surface WARN issues at the front of rationale so the operator always
        # sees them even after truncation; they document a *deliberately
        # non-blocking* condition (e.g. SHORT into a LONG-biased pair_charts
        # under SL-free) that the trader must remain aware of.
        for warn in reversed(warn_risk_issues):
            code = str(warn.get("code") or "")
            msg = str(warn.get("message") or code)
            rationale.insert(0, f"risk warn {code}: {msg}")
            score -= 5.0
        if result.get("live_blockers"):
            blockers.extend(str(item) for item in result.get("live_blockers", []))
            score -= 100.0

        gate_blockers, judgment = _discretionary_gate_check(
            intent=intent,
            status=status,
            profile_status=profile_status,
            strategy=strategy,
            lane=lane,
            method=method,
            method_pressure=method_pressure,
        )
        blockers.extend(gate_blockers)

        # ai_attack_advice overlay (AGENT_CONTRACT §8): the operator's
        # deterministic prefilter must surface advised lanes alongside its
        # own ranking. A LIVE_READY lane sitting in the top-K of
        # ai_attack_advice.recommended_now_lane_ids gets a flat bonus +
        # rationale so it edges past unadvised peers at score parity. The
        # promotion never overrides §11 hard blocks (BLOCK_UNTIL_NEW_EVIDENCE,
        # missing strategy profile, missing receipt, exposure blockers) —
        # those still keep the lane out of SEND_ENTRY. The bonus only applies
        # when status == "LIVE_READY"; lanes that intent_generator already
        # demoted stay demoted.
        if attack_ranks and status == "LIVE_READY" and lane_id in attack_ranks:
            rank = attack_ranks[lane_id]
            score += ATTACK_ADVICE_PROMOTION_BONUS
            # Insert at the front so the overlay annotation survives the
            # rationale truncation alongside other primary context.
            rationale.insert(
                0,
                f"attack_advice rank #{rank + 1} (top-{ATTACK_ADVICE_PROMOTION_RANK_CEILING}) "
                f"promoted +{ATTACK_ADVICE_PROMOTION_BONUS:.0f}",
            )

        # Detect reversal-from-extreme EARLIER in the modifier cascade so
        # we can suppress lane_history / trader_overrides at confirmed
        # reversal points. A losing-streak direction stays penalised in
        # normal trend conditions, but when price is at an extreme AND
        # structure confirms the reversal, the historical bias is
        # exactly the WRONG signal — that's the bottom we should buy.
        pair_chart_for_reversal = (full_pair_charts or {}).get(pair) if full_pair_charts else None
        _reversal_detected = detect_reversal(pair_chart_for_reversal, direction)

        # Module B: lane history bias — recent (pair, direction) P&L
        # adjusts score so a losing-streak direction gets downweighted
        # and a winning-streak direction gets upweighted. Bounded ±25.
        # SUPPRESSED when reversal_signal fires for the same direction:
        # history is a look-back signal that contradicts a confirmed
        # turn-from-extreme.
        if lane_history and _reversal_detected is None:
            lh_delta, lh_rationale = lane_history_modifier(lane_history, pair, direction)
            if lh_delta != 0.0:
                score += lh_delta
                if lh_rationale:
                    rationale.insert(0, lh_rationale)
        elif _reversal_detected is not None and lane_history:
            # Surface the suppression in rationale for audit.
            lh_delta_check, _ = lane_history_modifier(lane_history, pair, direction)
            if lh_delta_check < 0:
                rationale.insert(0, f"lane_history {lh_delta_check:+.1f} SUPPRESSED by reversal signal")

        # Module A: regime classifier — REVERSAL_RISK pair + same-side
        # entry direction gets penalty proportional to risk score.
        # STABLE_TREND pair + trend-aligned entry gets a modest reward.
        # Prevents the "buying the top / selling the bottom" pattern.
        if regime_snapshots:
            rg_snap = regime_snapshots.get(pair)
            rg_delta, rg_rationale = regime_score_modifier(rg_snap, direction)
            if rg_delta != 0.0:
                score += rg_delta
                if rg_rationale:
                    rationale.insert(0, rg_rationale)

        # Module D: entry timing gate — last 3 M5 candles vs intent
        # direction. ALIGNED rewards, MIXED small penalty, AGAINST big
        # penalty. Catches "entering at the top" timing errors that the
        # existing micro-override hard veto misses when M1 disagrees but
        # the 3-candle slope is still hostile.
        pa_chart_for_timing = (full_pair_charts or {}).get(pair) if full_pair_charts else None
        timing = check_entry_timing(pa_chart_for_timing, direction)
        if timing.score_delta != 0.0:
            score += timing.score_delta
            if timing.rationale:
                rationale.insert(0, timing.rationale)

        # Module C: daily-review overrides (lane_id blocks + (pair, direction)
        # score bias). Empty overrides → no-op. Expired overrides already
        # short-circuited at load time.
        # Bias overrides are suppressed when reversal_signal fires (same
        # rationale as lane_history); explicit `blocked_lanes` are kept
        # in force because they represent the operator's hard decision
        # (e.g. "do not touch this lane today" from daily-review).
        if trader_overrides is not None:
            blocked, block_msg = overrides_block_check(trader_overrides, lane_id)
            if blocked:
                blockers.append(block_msg or f"trader_overrides blocked {lane_id}")
                score -= 100.0
            ov_delta, ov_rationale = overrides_score_delta(trader_overrides, pair, direction)
            if ov_delta != 0.0:
                if _reversal_detected is None:
                    score += ov_delta
                    if ov_rationale:
                        rationale.insert(0, ov_rationale)
                elif ov_delta < 0:
                    rationale.insert(0, f"trader_overrides {ov_delta:+.1f} SUPPRESSED by reversal signal")

        # News themes: macro narrative converted to per-(pair, direction)
        # bias by news_themes.parse_news_themes. Currency-strength themes,
        # risk-on/off, and explicit pair-specific notes all roll into a
        # single bounded delta. Empty when news_digest.md is missing.
        if news_themes is not None:
            nt_delta, nt_rationale = news_themes.for_pair(pair, direction)
            if nt_delta != 0.0:
                score += nt_delta
                if nt_rationale:
                    rationale.insert(0, nt_rationale)

        # Reversal-from-extreme override (2026-05-13): when price is at
        # an extreme percentile (≤0.15 for LONG, ≥0.85 for SHORT) on
        # 24h OR 7d AND M1/M5/M15 prints a close-confirmed BOS/CHOCH
        # in the entry direction, add REVERSAL_BONUS. lane_history and
        # trader_overrides bias overrides were already suppressed
        # earlier in this function so the bonus is added on top of a
        # cleaned-up base. User feedback 2026-05-13:
        # 「安く買って高く売る、高く売って安く買う。こういう基本的なことが
        # できてない」.
        if _reversal_detected is not None:
            score += _reversal_detected.bonus
            rationale.insert(0, _reversal_detected.rationale)

        # Pattern-recognition layer (2026-05-14): failed breakout (trap
        # fade), RSI extreme + BB rail, dealing-range edge exhaustion,
        # Aroon strong cross. Each detected signal contributes a bounded
        # score; aligned with intent gets the full bonus, against gets
        # half-magnitude penalty. The whole layer is clamped to
        # ±PATTERN_TOTAL_CAP (default 30) so it can't dominate the model.
        # Discretionary "そろそろ感" via existing indicator data.
        if pair_chart_for_reversal is not None:  # already loaded above
            # COT payload loaded lazily per cycle (per-pair detector
            # filters internally). File-not-present → cot_payload=None
            # → COT shift detector silently skipped.
            _cot_payload = None
            _option_skew_payload = None
            try:
                from quant_rabbit.paths import ROOT as __QR_ROOT
                _cot_path = __QR_ROOT / "data" / "cot_snapshot.json"
                if _cot_path.exists():
                    _cot_payload = json.loads(_cot_path.read_text())
                _opt_path = __QR_ROOT / "data" / "option_skew_snapshot.json"
                if _opt_path.exists():
                    _option_skew_payload = json.loads(_opt_path.read_text())
            except Exception:
                pass
            _pattern_signals = detect_pattern_signals(
                pair_chart_for_reversal,
                cot_payload=_cot_payload,
                option_skew_payload=_option_skew_payload,
            )
            if _pattern_signals:
                _pattern_delta, _pattern_rationales = aggregate_pattern_score(
                    _pattern_signals, direction
                )
                if _pattern_delta != 0.0:
                    score += _pattern_delta
                    # Keep just the strongest 2-3 rationales on the lane
                    # report to avoid bloating; signals themselves are
                    # available via the structured dict if a caller needs.
                    rationale.insert(0, f"patterns {_pattern_delta:+.1f}: " + "; ".join(_pattern_rationales[:3]))

        # Forward-projection layer (2026-05-14): BB squeeze→expansion,
        # liquidity sweep targets, news-catalyst lookahead, cross-asset
        # lag, session expansion. UNlike pattern_signals (which read
        # past events), these are predictive — "what's about to happen".
        # User directive 2026-05-14:「相場が動いてから動いちゃだめ」.
        # 2026-05-14 (later): each projection signal is RECORDED to
        # projection_ledger.jsonl and the layer's confidence weights
        # are CALIBRATED from the rolling hit-rate of past predictions.
        # Detectors that don't pan out get dampened; strong ones get
        # boosted. User directive:「予測の精度を最大限高める」.
        from quant_rabbit.paths import ROOT as _QR_ROOT, DEFAULT_CALENDAR_SNAPSHOT, DEFAULT_CROSS_ASSET_SNAPSHOT
        from quant_rabbit.strategy.projection_ledger import (
            compute_hit_rates as _compute_hit_rates,
            record_projections as _record_projections,
        )
        if pair_chart_for_reversal is not None and snapshot is not None:
            cur_price_for_proj = _current_price_for(pair, snapshot, direction) if snapshot else None
            _projection_signals = detect_forward_projections(
                pair_chart_for_reversal,
                pair=pair,
                current_price=cur_price_for_proj,
                calendar_path=DEFAULT_CALENDAR_SNAPSHOT,
                cross_asset_path=DEFAULT_CROSS_ASSET_SNAPSHOT,
            )
            if _projection_signals:
                # Load rolling hit rates for calibration. Empty when
                # ledger hasn't accumulated samples yet — calibration
                # multiplier defaults to 1.0 in that case.
                _hit_rates = _compute_hit_rates(_QR_ROOT / "data")
                # Regime label (for hit_rate bucketing)
                _proj_regime = None
                if pair_chart_for_reversal:
                    _conf2 = pair_chart_for_reversal.get("confluence") or {}
                    _rg_raw = str(_conf2.get("dominant_regime") or "").upper()
                    if "TREND" in _rg_raw:
                        _proj_regime = "TREND"
                    elif "RANGE" in _rg_raw:
                        _proj_regime = "RANGE"
                _proj_delta, _proj_rationales = aggregate_projection_score(
                    _projection_signals, direction,
                    hit_rates=_hit_rates, pair=pair, regime=_proj_regime,
                )
                # Record predictions once per cycle/pair/signal. `_score_lane`
                # evaluates multiple candidate lanes for the same pair; the
                # ledger key keeps that scoring fan-out from inflating hit-rate
                # evidence.
                try:
                    # Extract regime label from confluence for
                    # hit_rate bucketing.
                    _regime_label = None
                    if pair_chart_for_reversal:
                        _conf = pair_chart_for_reversal.get("confluence") or {}
                        _regime_raw = str(_conf.get("dominant_regime") or "").upper()
                        if "TREND" in _regime_raw:
                            _regime_label = "TREND"
                        elif "RANGE" in _regime_raw:
                            _regime_label = "RANGE"
                        elif _regime_raw:
                            _regime_label = _regime_raw[:20]
                    _record_projections(
                        _projection_signals,
                        pair=pair,
                        current_price=cur_price_for_proj,
                        data_root=_QR_ROOT / "data",
                        regime_at_emission=_regime_label,
                        cycle_id=forecast_cycle_id,
                    )
                except Exception:
                    pass  # ledger write failure must not break scoring
                if _proj_delta != 0.0:
                    score += _proj_delta
                    rationale.insert(0, f"forward-proj {_proj_delta:+.1f}: " + "; ".join(_proj_rationales[:3]))

        # Cross-pair correlation lag (2026-05-14): when a strongly
        # correlated leader pair has moved but this pair hasn't yet,
        # project catch-up direction. Pair-level signal, runs once per
        # pair using the full pair_charts dict (other charts as
        # leaders). Skipped silently when full_pair_charts is sparse.
        if full_pair_charts and pair in full_pair_charts:
            try:
                _corr_signals = detect_correlation_lag(pair, full_pair_charts)
                if _corr_signals:
                    _corr_delta, _corr_rat = aggregate_correlation_lag_score(_corr_signals, direction)
                    if _corr_delta != 0.0:
                        score += _corr_delta
                        rationale.insert(0, f"corr-lag {_corr_delta:+.1f}: " + "; ".join(_corr_rat[:2]))
            except Exception:
                pass

        # Multi-step path projection (2026-05-14): sweep → FVG fill →
        # continuation. Requires M15 view with liquidity + FVG data.
        # Silent no-op when chart is missing structural artifacts.
        _paths_for_forecast: list = []
        if pair_chart_for_reversal is not None and snapshot is not None:
            try:
                _cur_for_path = _current_price_for(pair, snapshot, direction) if snapshot else None
                if _cur_for_path is not None:
                    _paths_for_forecast = list(detect_paths(pair_chart_for_reversal, direction, _cur_for_path))
                    if _paths_for_forecast:
                        _path_delta, _path_rat = aggregate_path_score(_paths_for_forecast, direction)
                        if _path_delta != 0.0:
                            score += _path_delta
                            rationale.insert(0, f"path-proj {_path_delta:+.1f}: " + "; ".join(_path_rat[:2]))
            except Exception:
                pass

        # 2026-05-15: Pair-level forecaster — synthesize all detectors into
        # ONE forecast per pair per cycle: UP / DOWN / RANGE / UNCLEAR. This
        # is the user-defined Stage 1 (予測). Directional forecasts must align
        # with the lane side; RANGE forecasts are tradable only through an
        # executable RANGE_ROTATION rail setup, because "レンジ幅を見極める" is a
        # valid market read, not a blanket no-trade state.
        try:
            metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
            forecast_context_present = bool(
                metadata.get("chart_story_structural")
                or metadata.get("chart_direction_bias")
                or metadata.get("m5_regime")
                or metadata.get("price_percentile_24h") is not None
                or metadata.get("range_support") is not None
                or metadata.get("range_resistance") is not None
            )
            # Unit tests and replay fixtures often carry synthetic intents
            # without the chart metadata generated by `generate-intents`.
            # Do not let the developer worktree's live `data/pair_charts.json`
            # leak into those synthetic receipts and veto them. Production
            # intents include these metadata keys, so live cycles still get the
            # pair-level forecast gate.
            _forecast = None
            if forecast_context_present:
                _forecast = _pair_forecast(
                    pair=pair,
                    pair_chart=pair_chart_for_reversal,
                    full_pair_charts=full_pair_charts,
                    snapshot=snapshot,
                    forecast_cache=forecast_cache,
                    forecast_cycle_id=forecast_cycle_id,
                )
            if _forecast is not None:
                gate_ok, gate_reason = _forecast_lane_gate(
                    _forecast,
                    direction=direction,
                    method=method,
                    intent=intent,
                )
                if _forecast.confidence < ENTRY_CONFIDENCE_MIN:
                    blockers.append(
                        f"forecast confidence {_forecast.confidence:.2f} < {ENTRY_CONFIDENCE_MIN} threshold"
                    )
                    score -= 30.0
                    rationale.insert(
                        0,
                        f"forecast {_forecast.direction} conf {_forecast.confidence:.2f} too low",
                    )
                elif not gate_ok:
                    blockers.append(f"{gate_reason}; rationale: {_forecast.rationale_summary}")
                    score -= 80.0 if _forecast.direction in {"UP", "DOWN"} else 60.0
                    rationale.insert(
                        0,
                        f"forecast {_forecast.direction} conf={_forecast.confidence:.2f} → BLOCK: {gate_reason}",
                    )
                else:
                    reward = (15.0 if _forecast.direction == "RANGE" else 20.0) * _forecast.confidence
                    score += reward
                    rationale.insert(
                        0,
                        f"forecast {_forecast.direction} conf={_forecast.confidence:.2f} → +{reward:.1f}: {gate_reason}",
                    )
        except Exception:
            pass

        adjusted_score = round(score + settings.score_bias, 2)
        size_multiple = _size_multiple(adjusted_score, settings)
        action = ACTION_SEND_ENTRY if status == "LIVE_READY" and not blockers else ACTION_NO_TRADE
        estimated_margin_jpy = _optional_float(risk_metrics.get("estimated_margin_jpy"))
        return LaneScore(
            lane_id=lane_id,
            pair=pair,
            direction=direction,
            method=method,
            order_type=order_type,
            entry=entry,
            tp=tp,
            sl=sl,
            status=status,
            score=adjusted_score,
            size_multiple=size_multiple,
            action=action,
            blockers=tuple(blockers[:8]),
            # Rationale cap raised from 8 → 12 to keep WARN risk_issues and
            # attack_advice promotion lines visible (added 2026-05-11).
            # The existing reasoning lines averaged 6–8 entries before the
            # overlay; 12 leaves headroom for the WARN advisory + the
            # attack_advice rank line without dropping mining/PA/MTF
            # context. Resize this if the rationale grows again.
            rationale=tuple(rationale[:12]),
            judgment=tuple(judgment[:8]),
            spread_pips=spread_pips,
            estimated_margin_jpy=estimated_margin_jpy,
        )

    def _write(self, decision: TraderDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "action": decision.action,
                    "selected_lane_id": decision.selected_lane_id,
                    "selected_lane_score": decision.selected_lane_score,
                    "selected_lane_size_multiple": decision.selected_lane_size_multiple,
                    "generated_at_utc": decision.generated_at_utc,
                    "reason": decision.reason,
                    "positions": decision.positions,
                    "orders": decision.orders,
                    "pending_cancel_order_ids": list(decision.pending_cancel_order_ids),
                    "loss_cap_jpy": decision.loss_cap_jpy,
                    "loss_cap_source": decision.loss_cap_source,
                    "scores": [asdict(item) for item in decision.scores],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Trader Decision Report",
            "",
            f"- Generated at UTC: `{decision.generated_at_utc}`",
            f"- Action: `{decision.action}`",
            f"- Selected lane: `{decision.selected_lane_id}`",
            f"- Selected lane score: `{decision.selected_lane_score}`",
            f"- Selected lane size multiple: `{decision.selected_lane_size_multiple}`",
            f"- Positions: `{decision.positions}`",
            f"- Orders: `{decision.orders}`",
            f"- Pending cancel ids: `{', '.join(decision.pending_cancel_order_ids) if decision.pending_cancel_order_ids else 'none'}`",
            f"- Loss cap: `{decision.loss_cap_jpy if decision.loss_cap_jpy is not None else 'missing'}` (`{decision.loss_cap_source or 'missing'}`)",
            f"- Reason: {decision.reason}",
            "",
            "## Ranked Lanes",
            "",
        ]
        for item in decision.scores[:12]:
            lines.append(
                f"- `{item.lane_id}` score=`{item.score}` action=`{item.action}` "
                f"`{item.pair} {item.direction} {item.method}`"
            )
            lines.append(f"  - size_multiple: `{item.size_multiple}`")
            if item.rationale:
                lines.append(f"  - why: {'; '.join(item.rationale)}")
            if item.judgment:
                lines.append(f"  - judgment: {'; '.join(item.judgment)}")
            if item.blockers:
                lines.append(f"  - blockers: {'; '.join(item.blockers)}")
        lines.extend(
            [
                "",
                "## Trader-Brain Contract",
                "",
                "- This layer must compare lanes; it must not send the first live-ready candidate mechanically.",
                "- Scores rank attention only; live entry requires explicit discretionary gates, not a single score threshold.",
                "- Pending entry or non-layerable exposure makes TraderBrain monitor-only; automation may pass compatible pending entries to gateway basket validation.",
                "- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.",
                "- The execution gateway remains the final authority for live risk.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _attack_advice_top_k_ranks(
    payload: dict[str, Any],
    *,
    k: int = ATTACK_ADVICE_PROMOTION_RANK_CEILING,
) -> dict[str, int]:
    """Map the top-K attack-advised lane_ids to their rank (0..k-1).

    Returns {} for missing/empty advice so callers degrade silently.
    `recommended_now_lane_ids` is the canonical ranked list emitted by
    `ai-attack-advice`; only string entries are retained, duplicates are
    de-duped at the original ranking, and the result is bounded to k
    entries per AGENT_CONTRACT §8 conviction-gap rationale.
    """
    if not isinstance(payload, dict):
        return {}
    advised = payload.get("recommended_now_lane_ids") or []
    ranks: dict[str, int] = {}
    for raw in advised:
        lane_id = str(raw or "")
        if not lane_id or lane_id in ranks:
            continue
        ranks[lane_id] = len(ranks)
        if len(ranks) >= k:
            break
    return ranks


def _pair_charts_directional_bias(
    full_pair_charts: dict[str, dict[str, Any]],
    pair: str,
) -> tuple[str, float] | None:
    """Read `pair_charts[pair].confluence` and return (lean, |gap|) if the
    bias is decisive enough to gate the opposite direction.

    Decisive = `score_balance` is LONG_LEAN or SHORT_LEAN AND
    `|score_gap| >= DIRECTIONAL_GATING_STRONG_GAP`. Returns None when
    pair_charts is missing/malformed or the bias is too weak.
    """
    chart = full_pair_charts.get(pair) if isinstance(full_pair_charts, dict) else None
    if not isinstance(chart, dict):
        return None
    conf = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else None
    if not isinstance(conf, dict):
        return None
    balance = str(conf.get("score_balance") or "")
    if balance not in {"LONG_LEAN", "SHORT_LEAN"}:
        return None
    try:
        gap = abs(float(conf.get("score_gap") or 0.0))
    except (TypeError, ValueError):
        return None
    if gap < DIRECTIONAL_GATING_STRONG_GAP:
        return None
    return balance, gap


def _attack_advice_pair_majority(
    advised_lane_ids: list[str],
    pair: str,
) -> str | None:
    """Return 'LONG' or 'SHORT' if the top-K advised lanes for this pair
    have a strict majority in one direction.

    Returns None when there are 0 advised lanes for the pair (no signal)
    or when the lanes are evenly split (no decisive direction).
    """
    longs = 0
    shorts = 0
    for lane_id in advised_lane_ids:
        parts = lane_id.split(":")
        # lane_id pattern: desk:PAIR:DIRECTION:METHOD[:MARKET]
        if len(parts) >= 3 and parts[1] == pair:
            direction = parts[2].upper()
            if direction == "LONG":
                longs += 1
            elif direction == "SHORT":
                shorts += 1
    if longs == 0 and shorts == 0:
        return None
    if longs == shorts:
        return None
    return "LONG" if longs > shorts else "SHORT"


def _apply_directional_gating(
    scores: tuple[LaneScore, ...],
    full_pair_charts: dict[str, dict[str, Any]],
    attack_ranks: dict[str, int],
) -> tuple[LaneScore, ...]:
    """Apply the C-1 / C-2 directional gating + 2026-05-13 precision
    filters (B price-percentile, D multi-TF agreement) to the scored
    lane tuple.

    Pass order:
      C-2 — attack_advice veto on opposite direction
      C-1 — pair_charts bias + advice agreement demotion
      B   — price-percentile extreme penalty / mean-rev bonus
      D   — multi-TF disagreement penalty

    All passes run only on the basket-construction view of the lane
    set. Existing-position management (`PositionManager` /
    `PositionProtectionGateway`) reads `snapshot.positions` directly
    and never consults this function or the LaneScore tuple, so this
    pass cannot modify SL/TP/CLOSE behavior on any open trade.

    Returns a new tuple — `LaneScore` is frozen, so mutations are
    expressed as `dataclasses.replace`.
    """
    if not scores:
        return scores
    advised_lane_ids = sorted(attack_ranks.keys(), key=lambda lid: attack_ranks[lid])

    # Pre-compute per-pair gating context once.
    pairs_in_scores = {item.pair for item in scores if item.pair}
    gating_ctx: dict[str, dict[str, Any]] = {}
    for pair in pairs_in_scores:
        bias = _pair_charts_directional_bias(full_pair_charts, pair)
        majority = _attack_advice_pair_majority(advised_lane_ids, pair)
        # B/D context comes from chart_reader's extended confluence.
        chart = full_pair_charts.get(pair) if isinstance(full_pair_charts, dict) else None
        conf = chart.get("confluence") if isinstance(chart, dict) else None
        ppct_24h = _optional_float((conf or {}).get("price_percentile_24h"))
        tf_agree = _optional_float((conf or {}).get("tf_agreement_score"))
        gating_ctx[pair] = {
            "bias": bias,
            "majority": majority,
            "price_percentile_24h": ppct_24h,
            "tf_agreement_score": tf_agree,
        }

    new_scores: list[LaneScore] = []
    for item in scores:
        ctx = gating_ctx.get(item.pair, {})
        bias = ctx.get("bias")
        majority = ctx.get("majority")
        direction = (item.direction or "").upper()
        if direction not in {"LONG", "SHORT"}:
            new_scores.append(item)
            continue

        new_score = item.score
        new_action = item.action
        new_rationale: list[str] = list(item.rationale)
        new_blockers: list[str] = list(item.blockers)

        # C-2: attack_advice veto for opposite-direction lanes.
        if majority is not None and majority != direction:
            new_score = round(new_score - ATTACK_ADVICE_VETO_PENALTY, 2)
            new_rationale.insert(
                0,
                f"attack_advice_veto: top-{ATTACK_ADVICE_PROMOTION_RANK_CEILING} "
                f"majority={majority}, lane is {direction}, penalty="
                f"-{ATTACK_ADVICE_VETO_PENALTY:.0f}",
            )

        # C-1: directional gating demotion. Requires BOTH a decisive
        # pair_charts bias AND attack_advice majority agreement.
        if bias is not None and majority is not None:
            bias_balance, bias_gap = bias
            bias_direction = "LONG" if bias_balance == "LONG_LEAN" else "SHORT"
            if bias_direction == majority and direction != bias_direction:
                if item.action == ACTION_SEND_ENTRY:
                    new_action = ACTION_NO_TRADE
                    new_blockers.insert(
                        0,
                        f"directional_gating_demoted: bias={bias_balance}, "
                        f"|gap|={bias_gap:.3f} >= {DIRECTIONAL_GATING_STRONG_GAP:.3f}, "
                        f"advice_majority={majority}",
                    )
                else:
                    new_rationale.insert(
                        0,
                        f"directional_gating: bias={bias_balance}, |gap|={bias_gap:.3f}, "
                        f"advice_majority={majority} (already NO_TRADE)",
                    )

        # B — price percentile extremes (2026-05-13). Same-side entry at
        # the top of 24h distribution (LONG @ >= 0.95) or bottom
        # (SHORT @ <= 0.05) is statistically late; opposite-side entry
        # at the same extreme is the mean-reversion side and gets a
        # smaller bonus.
        ppct = ctx.get("price_percentile_24h")
        if ppct is not None:
            if direction == "LONG":
                if ppct >= PRICE_PERCENTILE_EXTREME_HIGH:
                    new_score = round(new_score - PRICE_PERCENTILE_EXTREME_PENALTY, 2)
                    new_rationale.insert(
                        0,
                        f"price_percentile_extreme: LONG @ p24h={ppct:.2f} "
                        f">= {PRICE_PERCENTILE_EXTREME_HIGH:.2f} "
                        f"(-{PRICE_PERCENTILE_EXTREME_PENALTY:.0f})",
                    )
                elif ppct <= PRICE_PERCENTILE_EXTREME_LOW:
                    new_score = round(new_score + PRICE_PERCENTILE_MEAN_REV_BONUS, 2)
                    new_rationale.insert(
                        0,
                        f"price_percentile_mean_rev: LONG @ p24h={ppct:.2f} "
                        f"<= {PRICE_PERCENTILE_EXTREME_LOW:.2f} "
                        f"(+{PRICE_PERCENTILE_MEAN_REV_BONUS:.0f})",
                    )
            elif direction == "SHORT":
                if ppct <= PRICE_PERCENTILE_EXTREME_LOW:
                    new_score = round(new_score - PRICE_PERCENTILE_EXTREME_PENALTY, 2)
                    new_rationale.insert(
                        0,
                        f"price_percentile_extreme: SHORT @ p24h={ppct:.2f} "
                        f"<= {PRICE_PERCENTILE_EXTREME_LOW:.2f} "
                        f"(-{PRICE_PERCENTILE_EXTREME_PENALTY:.0f})",
                    )
                elif ppct >= PRICE_PERCENTILE_EXTREME_HIGH:
                    new_score = round(new_score + PRICE_PERCENTILE_MEAN_REV_BONUS, 2)
                    new_rationale.insert(
                        0,
                        f"price_percentile_mean_rev: SHORT @ p24h={ppct:.2f} "
                        f">= {PRICE_PERCENTILE_EXTREME_HIGH:.2f} "
                        f"(+{PRICE_PERCENTILE_MEAN_REV_BONUS:.0f})",
                    )

        # D — multi-TF disagreement (2026-05-13). When M15/M30/H1
        # regimes do not have a 2/3 majority, the directional picture is
        # too fractured to chase. Penalise any direction; the operator's
        # response is to wait for alignment, not to fade.
        tf_agree = ctx.get("tf_agreement_score")
        if tf_agree is not None and tf_agree < TF_AGREEMENT_MAJORITY_THRESHOLD:
            new_score = round(new_score - TF_AGREEMENT_DISAGREEMENT_PENALTY, 2)
            new_rationale.insert(
                0,
                f"tf_disagreement: M15/M30/H1 agreement={tf_agree:.2f} "
                f"< {TF_AGREEMENT_MAJORITY_THRESHOLD:.2f} "
                f"(-{TF_AGREEMENT_DISAGREEMENT_PENALTY:.0f})",
            )

        if (
            new_score == item.score
            and new_action == item.action
            and new_rationale == list(item.rationale)
            and new_blockers == list(item.blockers)
        ):
            new_scores.append(item)
            continue

        new_scores.append(
            replace(
                item,
                score=new_score,
                action=new_action,
                rationale=tuple(new_rationale[:12]),
                blockers=tuple(new_blockers[:8]),
            )
        )
    # Re-sort by score (descending) so the basket-construction layer
    # sees the gating-adjusted ranking, not the pre-gating order.
    return tuple(sorted(new_scores, key=lambda it: it.score, reverse=True))


def _select_entry_lane(
    scores: tuple[LaneScore, ...],
    *,
    target_state_path: Path,
    snapshot: BrokerSnapshot,
) -> LaneScore | None:
    sendable = tuple(item for item in scores if item.action == ACTION_SEND_ENTRY)
    if not sendable:
        return None
    # The previous FLAT-with-open-target rule force-picked MARKET unconditionally,
    # which over-paid the spread on quiet tape (user 2026-05-08 「市況によって
    # 柔軟に」). `_score_lane` now adjusts MARKET ±points by short-term momentum
    # (M1/M5 ADX), so the score-ranked top variant already reflects regime
    # preference: MARKET when ADX is high (momentum live), pending entry when
    # quiet. Trust the score; do not re-override here.
    return sendable[0]


def _entry_selection_reason(scores: tuple[LaneScore, ...], selected: LaneScore) -> str:
    top_sendable = next((item for item in scores if item.action == ACTION_SEND_ENTRY), None)
    if selected.order_type.upper() == "MARKET":
        if top_sendable is not None and top_sendable.lane_id != selected.lane_id:
            return f"Selected live-ready MARKET lane for target-open immediate exposure: {selected.lane_id}"
        return f"Selected highest-scoring live-ready MARKET lane for target-open immediate exposure: {selected.lane_id}"
    return f"Selected highest-scoring live-ready lane: {selected.lane_id}"


def _target_open_needs_immediate_entry(target_state_path: Path, snapshot: BrokerSnapshot) -> bool:
    if not _target_open(target_state_path):
        return False
    trader_positions = sum(1 for position in snapshot.positions if position.owner == Owner.TRADER)
    return trader_positions == 0 and _pending_entry_order_count(snapshot) == 0


def _target_open(target_state_path: Path) -> bool:
    target = _load_json(target_state_path)
    if str(target.get("status") or "") != "PURSUE_TARGET":
        return False
    return float(target.get("remaining_target_jpy") or 0.0) > 0


def load_trader_settings(path: Path) -> TraderSettings:
    return _load_trader_settings(path)


def _load_trader_settings(path: Path) -> TraderSettings:
    payload = _load_json(path)
    settings_payload = payload.get("size_by_score")
    if not isinstance(settings_payload, dict):
        settings_payload = {}
    risk_payload = payload.get("risk")
    if not isinstance(risk_payload, dict):
        risk_payload = {}
    score_bias = _coalesce_float(settings_payload.get("score_bias"), 0.0)
    score_size_enabled = settings_payload.get("enabled")
    if not isinstance(score_size_enabled, bool):
        score_size_enabled = True
    size_multiple_min = _coalesce_float(settings_payload.get("size_multiple_min"), 0.7)
    size_multiple_max = _coalesce_float(settings_payload.get("size_multiple_max"), 1.8)
    if size_multiple_max < size_multiple_min:
        size_multiple_min, size_multiple_max = size_multiple_max, size_multiple_min
    if size_multiple_min <= 0:
        size_multiple_min = 0.05
    size_multiple_anchor_score = _coalesce_float(settings_payload.get("size_multiple_anchor_score"), 110.0)
    size_multiple_per_score_point = _coalesce_float(
        settings_payload.get("size_multiple_per_score_point"), 0.005
    )
    if size_multiple_per_score_point < 0:
        size_multiple_per_score_point = 0.0
    default_max_loss_jpy = _optional_float(risk_payload.get("max_loss_jpy"))
    default_max_loss_pct = _optional_float(risk_payload.get("max_loss_pct"))
    return TraderSettings(
        score_bias=score_bias,
        score_size_enabled=bool(score_size_enabled),
        size_multiple_min=size_multiple_min,
        size_multiple_max=size_multiple_max,
        size_multiple_anchor_score=size_multiple_anchor_score,
        size_multiple_per_score_point=size_multiple_per_score_point,
        default_max_loss_jpy=default_max_loss_jpy,
        default_max_loss_pct=default_max_loss_pct,
    )


def _size_multiple(score: float, settings: TraderSettings) -> float:
    if not settings.score_size_enabled:
        return 1.0
    multiple = 1.0 + ((score - settings.size_multiple_anchor_score) * settings.size_multiple_per_score_point)
    return round(_clamp(multiple, settings.size_multiple_min, settings.size_multiple_max), 2)


def _coalesce_float(value: object, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _resolve_trader_loss_cap(
    *,
    strategy_payload: dict[str, Any],
    settings: TraderSettings,
    target_state_path: Path,
    snapshot: BrokerSnapshot,
) -> tuple[float | None, str]:
    cap = _loss_cap_from_target_state(target_state_path)
    if cap is not None:
        return cap, f"daily target state {target_state_path}"
    cap = _loss_cap_from_strategy_payload(strategy_payload)
    if cap is not None:
        return cap, "strategy profile system_contract.loss_cap_jpy"
    if settings.default_max_loss_jpy is not None and settings.default_max_loss_jpy > 0:
        return round(settings.default_max_loss_jpy, 4), "trader settings risk.max_loss_jpy"
    if (
        settings.default_max_loss_pct is not None
        and settings.default_max_loss_pct > 0
        and snapshot.account is not None
        and snapshot.account.balance_jpy > 0
    ):
        cap = snapshot.account.balance_jpy * (settings.default_max_loss_pct / 100.0)
        return round(cap, 4), "trader settings risk.max_loss_pct of broker balance"
    return None, "missing loss cap"


def _loss_cap_from_strategy_payload(payload: dict[str, Any]) -> float | None:
    contract = payload.get("system_contract")
    if not isinstance(contract, dict):
        return None
    return _positive_float(contract.get("loss_cap_jpy"))


def _loss_cap_from_target_state(path: Path) -> float | None:
    payload = _load_json(path)
    return _positive_float(payload.get("per_trade_risk_budget_jpy"))


def _positive_float(value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed <= 0:
        return None
    return round(parsed, 4)


def _strategy_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("profiles", []) or []:
        if isinstance(item, dict):
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            if pair and direction:
                index[(pair, direction)] = item
    return index


def _story_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in payload.get("pair_profiles", []) or []:
        if isinstance(item, dict) and item.get("pair"):
            index[str(item["pair"])] = item
    return index


def _campaign_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for lane in payload.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}"
        index[lane_id] = lane
    return index


def _parent_lane_id(lane_id: str) -> str:
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
    return lane_id


def _exposure_blockers(snapshot: BrokerSnapshot) -> tuple[str, ...]:
    blockers: list[str] = []
    for position in snapshot.positions:
        if position.owner in {Owner.MANUAL, Owner.UNKNOWN}:
            continue
        blockers.append(f"open position exists: {position.pair} {position.side.value} id={position.trade_id}")
    for order in snapshot.orders:
        if order.owner in {Owner.MANUAL, Owner.UNKNOWN}:
            continue
        if not order.trade_id and order.order_type.upper() in PENDING_ENTRY_TYPES:
            blockers.append(f"pending entry exists: {order.pair} {order.order_type} id={order.order_id}")
    return tuple(blockers)


def _pending_entry_order_count(snapshot: BrokerSnapshot) -> int:
    return sum(
        1
        for order in snapshot.orders
        if not order.trade_id and order.order_type.upper() in PENDING_ENTRY_TYPES
        and order.owner not in {Owner.MANUAL, Owner.UNKNOWN}
    )


def _portfolio_add_allowed(snapshot: BrokerSnapshot) -> bool:
    trader_positions = tuple(position for position in snapshot.positions if position.owner == Owner.TRADER)
    if not trader_positions:
        return False
    sl_free_active = _trader_sl_repair_disabled()
    # SL-free regime: trader-owned SL=None is intentional; TP must still
    # be present for layering (TP is the harvest plan in SL-free mode).
    return all(
        position.owner == Owner.TRADER
        and position.take_profit is not None
        and (position.stop_loss is not None or sl_free_active)
        for position in trader_positions
    )


def _contaminated_pending_order_ids(snapshot: BrokerSnapshot, scores: tuple[LaneScore, ...]) -> tuple[str, ...]:
    scores_by_key: dict[tuple[str, str, str], list[LaneScore]] = {}
    for score in scores:
        key = (score.pair, score.direction, _normalized_entry_type(score.order_type))
        scores_by_key.setdefault(key, []).append(score)
    contaminated: list[str] = []
    for order in snapshot.orders:
        if order.trade_id or order.order_type.upper() not in PENDING_ENTRY_TYPES:
            continue
        direction = _order_direction(order.units)
        if not order.pair or direction is None:
            continue
        if order.owner != Owner.TRADER:
            continue
        compatible_scores = scores_by_key.get((order.pair, direction, _normalized_entry_type(order.order_type)), [])
        if not compatible_scores:
            contaminated.append(order.order_id)
            continue
        if not any(_keeps_pending_order(order, score) for score in compatible_scores):
            contaminated.append(order.order_id)
    return tuple(contaminated)


def _order_direction(units: int | None) -> str | None:
    if units is None:
        return None
    if units > 0:
        return Side.LONG.value
    if units < 0:
        return Side.SHORT.value
    return None


def _same_entry_type(order_type: str, lane_order_type: str) -> bool:
    return _normalized_entry_type(order_type) == _normalized_entry_type(lane_order_type)


def _normalized_entry_type(order_type: str) -> str:
    return "STOP-ENTRY" if order_type.upper() == "STOP" else order_type.upper()


def _keeps_pending_order(order: BrokerOrder, score: LaneScore) -> bool:
    blocker_text = " ".join(score.blockers).upper()
    if "INTERVENTION" in blocker_text or "VISUAL STORY" in blocker_text:
        return False
    if score.action != ACTION_SEND_ENTRY and not _blocked_only_by_existing_pending(score):
        return False
    pair = order.pair or ""
    return not (
        _entry_drift_exceeds_current_spread(pair, order.price, score.entry, score.spread_pips)
        or _entry_drift_exceeds_current_spread(
            pair,
            _raw_dependent_price(order.raw, "takeProfitOnFill"),
            score.tp,
            score.spread_pips,
        )
        or _entry_drift_exceeds_current_spread(
            pair,
            _raw_dependent_price(order.raw, "stopLossOnFill"),
            score.sl,
            score.spread_pips,
        )
    )


def _blocked_only_by_existing_pending(score: LaneScore) -> bool:
    return bool(score.blockers) and all(str(item).startswith("pending entry exists:") for item in score.blockers)


def _entry_drift_pips(pair: str, order_price: float | None, lane_entry: float | None) -> float:
    if order_price is None or lane_entry is None:
        return 0.0
    pip_factor = 100 if pair.endswith("_JPY") else 10000
    return abs(order_price - lane_entry) * pip_factor


def _entry_drift_exceeds_current_spread(
    pair: str,
    order_price: float | None,
    lane_entry: float | None,
    spread_pips: float | None,
) -> bool:
    if spread_pips is None or spread_pips <= 0:
        return False
    # Pending entries should not be canceled just because the next broker snapshot
    # reprices the same setup by a few ticks. The replacement threshold is tied to
    # current spread, so it expands in thin liquidity and tightens in normal tape.
    return _entry_drift_pips(pair, order_price, lane_entry) > spread_pips * PENDING_ENTRY_REPLACE_SPREAD_MULT


def _raw_dependent_price(raw: dict[str, Any], key: str) -> float | None:
    if not isinstance(raw, dict):
        return None
    nested = raw.get(key)
    if not isinstance(nested, dict):
        return None
    return _optional_float(nested.get("price"))


def _method_theme_score(method: str, themes: dict[str, Any], rationale: list[str]) -> float:
    score = 0.0
    if method == TradeMethod.TREND_CONTINUATION.value and int(themes.get("momentum") or 0) > 0:
        score += 12.0
        rationale.append("momentum theme supports trend")
    if method == TradeMethod.RANGE_ROTATION.value and int(themes.get("range_rail") or 0) > 0:
        score += 12.0
        rationale.append("range rail theme supports rotation")
    if method == TradeMethod.BREAKOUT_FAILURE.value and int(themes.get("breakout_failure") or 0) > 0:
        score += 14.0
        rationale.append("breakout-failure theme supports trap/reclaim")
    if int(themes.get("event_risk") or 0) > 0:
        score -= 8.0
        rationale.append("event risk requires restraint")
    if int(themes.get("spread_liquidity") or 0) > 0:
        score -= 10.0
        rationale.append("spread/liquidity theme reduces urgency")
    return score


def _campaign_score(lane: dict[str, Any], rationale: list[str]) -> float:
    role = str(lane.get("campaign_role") or "")
    adoption = str(lane.get("adoption") or "")
    score = 0.0
    if "NOW" in role:
        score += 14.0
        rationale.append(f"campaign role {role}")
    elif "BACKUP" in role:
        score += 6.0
    if adoption == "ORDER_INTENT_REQUIRED":
        score += 8.0
    return score


def _narrative_risk_score(
    pair: str,
    direction: str,
    method: str,
    themes: dict[str, Any],
    examples: tuple[str, ...],
    blockers: list[str],
    rationale: list[str],
    *,
    status: str,
) -> float:
    # AGENT_CONTRACT §6: current discretionary narrative concerns size the lane
    # down via score -> size_multiple. They MUST NOT block the lane in prose. Per §3.5
    # per_trade_risk_budget_jpy already shrinks the per-shot exposure; layering
    # an "intervention narrative" or "visual story rejected" gate on top is an
    # invented threshold not enumerated in §3.5/§9/§10/§11. Surface the concern
    # in rationale so the operator/GPT sees it, but let the lane stay tradable.
    text = " ".join(examples).upper()
    score = 0.0
    is_live_ready = status == "LIVE_READY"
    if pair in JPY_CROSSES and direction == Side.LONG.value:
        intervention = int(themes.get("intervention") or 0)
        liquidity = int(themes.get("spread_liquidity") or 0)
        if intervention or "INTERVENTION" in text or "RATE CHECK" in text:
            score -= JPY_INTERVENTION_SCORE_PENALTY
            rationale.append("JPY-cross long under intervention/rate-check narrative; size multiple reduced")
        if liquidity or "GOLDEN WEEK" in text:
            score -= JPY_LIQUIDITY_SCORE_PENALTY
            rationale.append("JPY liquidity theme requires smaller/fewer entries")
    if "WAIT" in text:
        if is_live_ready:
            rationale.append("stale narrative WAIT language ignored for live-ready receipt")
        else:
            score -= 18.0
            rationale.append("recent narrative contained WAIT language")
    if "NO:" in text and method == TradeMethod.RANGE_ROTATION.value:
        if is_live_ready:
            rationale.append("stale visual rejection marker ignored for live-ready receipt")
        else:
            score -= 28.0
            rationale.append("visual story rejected range rotation; size multiple reduced")
    if "TREND-BULL" in text and direction == Side.LONG.value:
        score += 10.0
    if "TREND-BEAR" in text and direction == Side.SHORT.value:
        score += 10.0
    return score


def _technical_consensus_score(
    *,
    intent: dict[str, Any],
    method: str,
    status: str,
    strategy: dict[str, Any],
    story: dict[str, Any],
    lane: dict[str, Any],
    risk_metrics: dict[str, Any] | None,
    method_pressure: int,
    loss_cap_jpy: float | None,
    rationale: list[str],
    blockers: list[str],
) -> float:
    score = 0.0
    support_ticks = 0
    evidence_ticks = 0

    positive_evidence_n = int(strategy.get("positive_evidence_n") or 0)
    positive_tail_jpy = float(strategy.get("positive_tail_jpy") or 0.0)
    positive_best_jpy = float(strategy.get("positive_best_jpy") or 0.0)
    seat_discovered = int(strategy.get("seat_discovered") or 0)
    seat_orderable = int(strategy.get("seat_orderable") or 0)
    seat_captured = int(strategy.get("seat_captured") or 0)
    live_worst = _optional_float(strategy.get("live_worst_jpy"))
    required_fix = str(strategy.get("required_fix") or "")

    # Evidence depth and quality are one pillar.
    if positive_evidence_n >= 120:
        score += 8.0
        evidence_ticks += 1
        rationale.append(f"broad positive evidence count={positive_evidence_n}")
    elif positive_evidence_n >= 40:
        score += 5.0
        evidence_ticks += 1
        rationale.append(f"positive evidence count={positive_evidence_n}")
    elif positive_evidence_n > 0:
        score += 1.0
        rationale.append(f"some positive evidence count={positive_evidence_n}")
    else:
        if status == "LIVE_READY":
            rationale.append("missing positive mined evidence on this pair/direction; advisory only")
        else:
            score -= 3.0
            rationale.append("missing positive mined evidence on this pair/direction; repair required before live-ready")

    if seat_orderable > 0 and seat_discovered > 0:
        capture_rate = seat_captured / seat_discovered
        if capture_rate >= 0.50:
            score += 4.0
            support_ticks += 1
            rationale.append(f"high capture quality={capture_rate:.0%} ({seat_captured}/{seat_discovered})")
        elif capture_rate >= 0.30:
            score += 2.0
            support_ticks += 1
        elif capture_rate >= 0.10:
            score += 0.0
        else:
            if status == "LIVE_READY":
                rationale.append(f"low capture rate={capture_rate:.0%} ({seat_captured}/{seat_discovered}); advisory only")
            else:
                score -= 4.0
                rationale.append(
                    f"low capture rate={capture_rate:.0%} ({seat_captured}/{seat_discovered}); repair required before live-ready"
                )
    if positive_tail_jpy > 0:
        score += 2.0
    if positive_best_jpy > 0:
        score += 1.5

    # Risk geometry quality is second pillar.
    risk_metrics = risk_metrics or {}
    reward_risk = _optional_float(risk_metrics.get("reward_risk"))
    spread_pips = _optional_float(risk_metrics.get("spread_pips"))
    risk_jpy = _optional_float(risk_metrics.get("risk_jpy"))
    reward_jpy = _optional_float(risk_metrics.get("reward_jpy"))
    plan_rr = _optional_float(lane.get("target_reward_risk"))

    if reward_risk is None or spread_pips is None or risk_jpy is None:
        score -= 2.5
        blockers.append("missing dry-run risk geometry metric")
    else:
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        if method == TradeMethod.RANGE_ROTATION.value and metadata.get("geometry_model") == "RANGE_RAIL_LIMIT":
            score += 8.0
            support_ticks += 1
            rationale.append(
                f"range LIMIT is anchored to {metadata.get('range_entry_side')} rail "
                f"{metadata.get('range_support')}–{metadata.get('range_resistance')}"
            )
        if reward_risk >= 1.2:
            score += 2.0
            support_ticks += 1
        else:
            score -= 6.0
            blockers.append(f"reward/risk below minimum floor: {reward_risk:.2f}R")
        if reward_risk >= 2.0:
            score += 2.5
            rationale.append(f"reward/risk geometry supports edge: {reward_risk:.2f}R")
        if spread_pips <= 1.2:
            score += 2.0
            support_ticks += 1
            rationale.append(f"tight spread={spread_pips:.1f}pip")
        elif spread_pips <= 2.0:
            score += 0.8
        else:
            score -= 2.0
            blockers.append(f"wide spread for fresh edge={spread_pips:.1f}pip")
        if loss_cap_jpy is None:
            blockers.append("trader loss cap missing for risk geometry ranking")
        else:
            risk_fraction = risk_jpy / loss_cap_jpy
            if risk_fraction <= LOW_RISK_CAP_FRACTION:
                score += 2.0
                support_ticks += 1
            elif risk_fraction <= MEDIUM_RISK_CAP_FRACTION:
                score += 1.0
            else:
                score -= 2.0
        if plan_rr is not None and reward_risk >= plan_rr * 0.85:
            score += 1.5
            rationale.append(f"geometry reward/risk {reward_risk:.2f}R matches lane target {plan_rr:.2f}R")
        if reward_jpy is not None and reward_jpy > risk_jpy:
            score += 1.0

    # Strategy context and campaign contract consistency is third pillar.
    if method_pressure <= 0 and not story.get("methods"):
        if status == "LIVE_READY" and support_ticks >= 2 and evidence_ticks >= 1:
            rationale.append("entry can still pass with strong statistical edge despite weak live story pressure")
        elif status == "LIVE_READY":
            rationale.append("live technical story lacks method pressure; advisory only")
        else:
            score -= 5.0
            blockers.append("live technical story lacks method pressure for this setup")

    if required_fix and "watch-only" in required_fix.lower():
        if status == "LIVE_READY":
            rationale.append("strategy required_fix still mentions watch-only restrictions; advisory only")
        else:
            score -= 2.0
            rationale.append("strategy required_fix still mentions watch-only restrictions; repair required before live-ready")

    if (
        loss_cap_jpy is not None
        and live_worst is not None
        and live_worst <= -(loss_cap_jpy * HISTORICAL_LARGE_LOSS_CAP_MULTIPLE)
    ):
        if status != "LIVE_READY":
            score -= 2.0

    if status == "LIVE_READY" and intent.get("order_type"):
        support_ratio = (support_ticks + 1) / 5.0
        if support_ratio >= 0.7:
            score += 3.0
        elif support_ratio <= 0.2:
            score -= 3.0

    score += _story_fusion_score(
        method=method,
        direction=str(intent.get("side") or ""),
        examples=tuple(str(item) for item in story.get("examples", ())),
        score=score,
        rationale=rationale,
        blockers=blockers,
        support_ticks=support_ticks,
        status=status,
    )

    return score


def _story_fusion_score(
    *,
    method: str,
    direction: str,
    examples: tuple[str, ...],
    score: float,
    rationale: list[str],
    blockers: list[str],
    support_ticks: int,
    status: str,
) -> float:
    delta = 0.0
    if not examples:
        if status == "LIVE_READY":
            rationale.append("story has no concrete technical/news/chart examples; advisory only")
            return 0.0
        blockers.append("story has no concrete technical/news/chart examples")
        return -3.0

    source_counts: dict[str, int] = {}
    for item in examples:
        upper = item.upper()
        source = "OTHER"
        if upper.startswith("NEWS_DIGEST"):
            source = "NEWS"
        elif upper.startswith("NEWS_FLOW"):
            source = "FLOW"
        elif upper.startswith("QUALITY_AUDIT"):
            source = "QUALITY"
        source_counts[source] = source_counts.get(source, 0) + 1
        if "TREND-BULL" in upper and direction == Side.LONG.value:
            delta += 1.5
            support_ticks += 1
        elif "TREND-BEAR" in upper and direction == Side.SHORT.value:
            delta += 1.5
            support_ticks += 1
        elif "TREND-BULL" in upper and direction == Side.SHORT.value:
            delta -= 1.2
        elif "TREND-BEAR" in upper and direction == Side.LONG.value:
            delta -= 1.2

    method_token = {
        "TREND_CONTINUATION": "TREND",
        "RANGE_ROTATION": "RANGE",
        "BREAKOUT_FAILURE": "BREAKOUT",
        "EVENT_RISK": "EVENT",
        "POSITION_MANAGEMENT": "POSITION",
    }.get(method, "")

    source_diversity = len(source_counts)
    if source_diversity >= 2:
        delta += 2.0
        rationale.append(f"multi-source story coverage ({', '.join(sorted(source_counts))})")
        if source_counts.get("QUALITY", 0) > 0 and source_counts.get("NEWS", 0) > 0:
            delta += 1.0
            rationale.append("news and chart-quality evidence agree on setup")
    else:
        if status == "LIVE_READY":
            rationale.append("story evidence lacks source diversity; advisory only")
        else:
            delta -= 1.0
            blockers.append("story evidence lacks source diversity")

    conflict_hits = sum(1 for item in examples if "NO:" in item.upper())
    if conflict_hits >= 2:
        if status == "LIVE_READY":
            rationale.append("story explicitly contains mixed rejection markers; advisory only")
        else:
            delta -= 2.5
            blockers.append("story explicitly contains mixed rejection markers")
    elif conflict_hits == 1:
        if status != "LIVE_READY":
            delta -= 1.0

    if method_token and any(method_token in item.upper() for item in examples):
        delta += 2.0
        support_ticks += 1
        rationale.append(f"story examples confirm method token={method_token}")

    if support_ticks >= 3 and score >= 80:
        delta += 1.0
    return delta


def _direction_conflict_penalty(result: dict[str, Any], rationale: list[str]) -> float:
    intent = result.get("intent") or {}
    pair = str(intent.get("pair") or "")
    direction = str(intent.get("side") or "")
    context = intent.get("market_context") or {}
    narrative = f"{context.get('narrative') or ''} {context.get('chart_story') or ''}".upper()
    if pair == "EUR_USD" and "DIRECTIONLESS" in narrative:
        rationale.append("EUR_USD narrative is directionless; require cleaner proof")
        return -10.0 if direction else 0.0
    return 0.0


def _discretionary_gate_check(
    *,
    intent: dict[str, Any],
    status: str,
    profile_status: str,
    strategy: dict[str, Any],
    lane: dict[str, Any],
    method: str,
    method_pressure: int,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    judgment: list[str] = []
    if status == "LIVE_READY":
        judgment.append("fresh live-ready receipt exists")
    else:
        blockers.append(f"receipt is not live-ready: {status}")

    if profile_status == "CANDIDATE":
        judgment.append("strategy profile is live-eligible")
    elif profile_status:
        # Mirror `strategy_profile.validate()` (profile.py:125) and the
        # `_score_lane` profile-status branch above: under SL-free the
        # per_trade cap is the loss bound, so non-CANDIDATE profile
        # status (BLOCK_UNTIL_NEW_EVIDENCE / RISK_REPAIR_CANDIDATE /
        # MINE_MISSED_EDGE / WATCH_ONLY) downgrades to advisory judgment
        # rather than a hard blocker. The profile status field is never
        # auto-promoted (AGENT_CONTRACT §11) — this only relaxes the
        # trader_brain hard-veto, not the profile classification.
        if _trader_sl_repair_disabled():
            judgment.append(
                f"strategy profile is {profile_status} (advisory under SL-free)"
            )
        else:
            blockers.append(f"strategy profile is not live-eligible: {profile_status}")
    else:
        blockers.append("missing strategy profile")

    if not str(intent.get("thesis") or "").strip():
        blockers.append("missing trader thesis")
    context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else None
    if context is None:
        blockers.append("missing market context")
    else:
        missing_context = [
            name
            for name in ("regime", "narrative", "chart_story", "method", "invalidation")
            if not str(context.get(name) or "").strip()
        ]
        if missing_context:
            blockers.append(f"incomplete market context: {', '.join(missing_context)}")
        else:
            judgment.append("thesis, narrative, chart story, method, and invalidation are explicit")

    if _optional_float(intent.get("tp")) is None or _optional_float(intent.get("sl")) is None:
        blockers.append("missing TP/SL geometry")
    if int(intent.get("units") or 0) <= 0:
        blockers.append("missing executable units")

    adoption = str(lane.get("adoption") or "")
    if adoption in {"ORDER_INTENT_REQUIRED", "RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"}:
        judgment.append(f"campaign lane is executable after receipts: {adoption}")
    elif adoption and _trader_sl_repair_disabled():
        # `plan-campaign` mirrors the strategy_profile status into
        # adoption (REJECTED when profile is BLOCK_UNTIL_NEW_EVIDENCE,
        # etc.). Under SL-free the per_trade cap (already enforced by
        # RiskEngine) bounds the loss, so a stale REJECTED adoption
        # downgrades to advisory the same way as the profile-status
        # branches above. Without SL-free the legacy hard block remains.
        judgment.append(f"campaign lane adoption is {adoption} (advisory under SL-free)")
    else:
        blockers.append(f"campaign lane is not executable: {adoption or 'missing'}")

    pretrade_net = float(strategy.get("pretrade_net_jpy") or 0.0)
    live_net = float(strategy.get("live_net_jpy") or 0.0)
    if pretrade_net > 0 or live_net > 0 or strategy.get("receipt_promotion"):
        judgment.append("mined or repaired edge evidence is positive")
    elif status == "LIVE_READY" and (
        profile_status == "CANDIDATE"
        or (profile_status and _trader_sl_repair_disabled())
    ):
        # Under SL-free, the per_trade cap (`per_trade_risk_budget_jpy`)
        # bounds the loss, so a LIVE_READY receipt with a non-CANDIDATE
        # profile (BLOCK_UNTIL_NEW_EVIDENCE / RISK_REPAIR_CANDIDATE /
        # MINE_MISSED_EDGE / WATCH_ONLY) is not vetoed by stale negative
        # history. Otherwise sample-period statistics perpetuate
        # themselves (feedback_no_direction_bias_rules.md) and the only
        # path to refresh the profile is to trade — chicken-and-egg.
        judgment.append("past edge evidence is weak/negative, but the current live-ready receipt remains executable")
    else:
        blockers.append("no positive mined or repaired edge evidence")

    if method and method_pressure > 0:
        judgment.append(f"current story contains method pressure for {method}")
    elif method and (
        int(strategy.get("positive_evidence_n") or 0) >= 40
        or str(strategy.get("status")) == "CANDIDATE"
    ):
        judgment.append("market story is weak, but evidence is strong enough to keep under review")
    else:
        blockers.append("market story does not support the selected method")
    return blockers, judgment


def _optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
