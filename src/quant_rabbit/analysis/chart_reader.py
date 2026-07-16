"""Pair-level chart view that lets the trader rank pairs by edge.

For each pair, fetch candles across execution, setup, intraday, and higher
timeframes, run the indicator stack, and emit:

- A multi-timeframe regime tag (TREND_UP / TREND_DOWN / RANGE / IMPULSE / FAILURE_RISK)
- A bias score per direction (long/short) from indicator agreement
- A volatility/spread health flag
- A short, deterministic chart_story string suitable for `MarketContext`

The trader (Codex) consumes the score table to pick which pair to attack and
the chart_story is what gets stamped on the order intent. The score is *not* a
trading signal by itself — it shows where the indicators line up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from quant_rabbit.analysis.candles import (
    PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
    TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    TECHNICAL_CANDLE_PROVENANCE_INVALID,
    TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE,
    TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES,
    TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE,
    Candle,
    fetch_candles_via_client,
    fetch_technical_candles_via_client,
)
from quant_rabbit.analysis.indicators import (
    IndicatorSet,
    compute_indicators,
    compute_rsi_series,
    compute_macd_hist_series,
)
from quant_rabbit.analysis.structure import StructureReading, analyze_structure
from quant_rabbit.broker.oanda import OandaReadOnlyClient

# Reading-layer additions (phases 1-3 per docs/research/ + AGENT_CONTRACT §6).
# These are fail-soft: when underlying data is insufficient the readings are
# emitted as None, never silently faked.
from quant_rabbit.analysis.regime import RegimeReading, classify_regime, classify_regime_from_values
from quant_rabbit.analysis.families import FamilyScores, compute_family_scores
from quant_rabbit.analysis.market_state import (
    MarketStateReading,
    classify_market_state,
    summarize_market_states,
)
from quant_rabbit.analysis.statfilters import StatFilterReading, compute_stat_filters
from quant_rabbit.analysis.sessions import SessionContext, build_session_context
from quant_rabbit.analysis.smc import SMCReading, analyze_smc
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS


# How many of the most-recent OHLCV bars to publish per view. 30 is
# enough for candlestick-pattern detectors (look back 1-3 bars),
# volume-spike rolling averages (20-bar baseline), and short-window
# time-exhaustion runs (≤10 bars). Bounded so pair_charts.json stays
# under a few hundred KB even with 7 timeframes × N pairs.
RECENT_CANDLES_PUBLISH = 30


# Default trader chart stack. These are roles, not equal votes:
#
# - M1: execution microstructure / last-minute confirmation.
# - M5/M15: main setup and operating chart.
# - M30/H1: intraday bias and structure.
# - H4/D: higher-timeframe context and "are we fighting the tape?" audit.
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D")

# Timeframe-specific annualization for ATR percentiles. These are calendar /
# bar-frequency constants, not strategy thresholds. The classifier uses them to
# make "1y ATR percentile" mean roughly the same market memory on M1 and D.
_BARS_PER_YEAR_BY_TIMEFRAME: dict[str, int] = {
    "M1": 252 * 1440,
    "M5": 252 * 288,
    "M15": 252 * 96,
    "M30": 252 * 48,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D": 252,
}

# Statistical filter windows by timeframe. Each value is a count of bars, so
# it scales the memory horizon with the chart being read instead of treating an
# H4 candle like an M5 candle. Windows remain estimator policy; missing history
# emits None rather than substituting a price or risk literal.
_STAT_FILTER_WINDOWS_BY_TIMEFRAME: dict[str, dict[str, int]] = {
    "M1": {"autocorr": 200, "moment": 300, "jump": 80, "dfa": 300},
    "M5": {"autocorr": 100, "moment": 200, "jump": 50, "dfa": 200},
    "M15": {"autocorr": 80, "moment": 160, "jump": 40, "dfa": 160},
    "M30": {"autocorr": 64, "moment": 128, "jump": 32, "dfa": 128},
    "H1": {"autocorr": 48, "moment": 96, "jump": 24, "dfa": 96},
    "H4": {"autocorr": 32, "moment": 80, "jump": 20, "dfa": 80},
    "D": {"autocorr": 24, "moment": 64, "jump": 16, "dfa": 64},
}


@dataclass(frozen=True)
class ChartView:
    """One timeframe slice of indicators + derived regime tags + reading layer."""

    granularity: str
    indicators: IndicatorSet
    regime: str  # legacy heuristic regime (TREND_UP/TREND_DOWN/RANGE/IMPULSE_*/FAILURE_RISK/UNCLEAR)
    long_bias: float  # 0..1 — agreement strength toward long
    short_bias: float  # 0..1 — agreement strength toward short
    structure: StructureReading | None = None
    # Reading layer (phases 1-3). See docs/research/03-quant-ensemble-regime.md
    # and AGENT_CONTRACT §6: trader MUST cite these in `chart_story`, not raw
    # indicator values, when present.
    regime_reading: RegimeReading | None = None  # Hurst+ADX+Choppiness+ATR_pct → 4-state
    family_scores: FamilyScores | None = None  # Trend/MeanRev/Breakout composites
    stat_filters: StatFilterReading | None = None  # jumps / autocorr / ACF / kurt
    smc: SMCReading | None = None  # SwingPivot/BOS/OB/FVG/Sweep/Breaker/iFVG/Displacement/DealingRange/OTE
    # 2026-05-14: raw OHLCV history for downstream candlestick-pattern,
    # volume-spike, and time-exhaustion detectors. Capped to the most
    # recent RECENT_CANDLES_PUBLISH bars to keep pair_charts.json size
    # bounded; downstream consumers should not expect a long history.
    recent_candles: tuple[Candle, ...] = field(default_factory=tuple)
    # 2026-05-14: indicator time series for true divergence detectors.
    # Last RECENT_CANDLES_PUBLISH values of key oscillators aligned to
    # `recent_candles` so a downstream consumer can take the i-th
    # element of both. Empty tuples when there aren't enough bars to
    # seed the indicator (early-startup case).
    indicator_series: dict[str, tuple[float, ...]] = field(default_factory=dict)
    candle_integrity: dict[str, object] = field(default_factory=dict)
    market_state: MarketStateReading | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "granularity": self.granularity,
            "regime": self.regime,
            "long_bias": round(self.long_bias, 4),
            "short_bias": round(self.short_bias, 4),
            "indicators": self.indicators.to_dict(),
            "structure": self.structure.to_dict() if self.structure else None,
            "regime_reading": _regime_reading_to_dict(self.regime_reading),
            "family_scores": _family_scores_to_dict(self.family_scores),
            "stat_filters": _stat_filters_to_dict(self.stat_filters),
            "smc": _smc_to_dict(self.smc),
            "recent_candles": [
                {
                    "t": c.timestamp_utc.isoformat() if hasattr(c.timestamp_utc, "isoformat") else str(c.timestamp_utc),
                    "o": c.open, "h": c.high, "l": c.low, "c": c.close, "v": c.volume,
                    "complete": c.complete,
                }
                for c in self.recent_candles
            ],
            "indicator_series": {k: list(v) for k, v in self.indicator_series.items()},
            "candle_integrity": self.candle_integrity,
            "market_state": self.market_state.to_dict() if self.market_state else None,
        }


@dataclass(frozen=True)
class PairChart:
    pair: str
    views: tuple[ChartView, ...]
    long_score: float
    short_score: float
    dominant_regime: str
    chart_story: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    session: SessionContext | None = None  # killzone / Judas / Silver Bullet / JP holiday
    confluence: dict[str, object] = field(default_factory=dict)
    technical_candle_integrity: dict[str, object] = field(default_factory=dict)
    market_state_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "long_score": round(self.long_score, 4),
            "short_score": round(self.short_score, 4),
            "dominant_regime": self.dominant_regime,
            "chart_story": self.chart_story,
            "warnings": list(self.warnings),
            "views": [view.to_dict() for view in self.views],
            "session": _session_to_dict(self.session),
            "confluence": self.confluence,
            "technical_candle_integrity": self.technical_candle_integrity,
            "market_state_summary": self.market_state_summary,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pair_chart(
    pair: str,
    *,
    client: OandaReadOnlyClient,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    candles_by_tf: Mapping[str, Iterable[Candle]] | None = None,
    count: int = 200,
) -> PairChart:
    """Build a multi-timeframe view for a pair.

    `candles_by_tf` lets tests inject prebuilt series. Production FX fetches
    one MBA packet per timeframe: MID remains the indicator input while BID
    and ASK prove that a rollover spread did not distort the closed bar.
    """

    pair_key = pair.upper()
    views: list[ChartView] = []
    warnings: list[str] = []
    candles_by_tf_resolved: dict[str, tuple[Candle, ...]] = {}
    integrity_by_tf: dict[str, dict[str, object]] = {}
    for tf in timeframes:
        if candles_by_tf is not None and tf in candles_by_tf:
            candles = tuple(candles_by_tf[tf])
            candle_integrity = _not_evaluated_candle_integrity(
                pair=pair_key,
                granularity=tf,
                source="INJECTED",
                clean_count=len(candles),
                reason="caller-supplied candles have no broker bid/ask provenance",
            )
        else:
            try:
                if pair_key in NORMAL_SPREAD_PIPS:
                    batch = fetch_technical_candles_via_client(client, pair_key, tf, count=count)
                    candle_integrity = batch.integrity
                    clean_tail_count = candle_integrity.get("recent_clean_tail_count")
                    expected_warmup_min = (
                        TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
                        if tf in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
                        else 1
                    )
                    if (
                        isinstance(clean_tail_count, int)
                        and not isinstance(clean_tail_count, bool)
                        and 0 <= clean_tail_count <= len(batch.candles)
                        and candle_integrity.get("indicator_warmup_min_clean_count")
                        == expected_warmup_min
                    ):
                        candles = (
                            batch.candles[-clean_tail_count:]
                            if clean_tail_count > 0
                            else ()
                        )
                    else:
                        # The parser is the trusted producer, but keep this
                        # consumer fail-closed if a future receipt shape drifts.
                        candles = ()
                else:
                    # Context-only assets do not have an FX normal-spread
                    # policy. Preserve MID without inventing a threshold.
                    candles = fetch_candles_via_client(client, pair_key, tf, count=count)
                    candle_integrity = _not_evaluated_candle_integrity(
                        pair=pair_key,
                        granularity=tf,
                        source="OANDA_MID",
                        clean_count=len(candles),
                        reason="canonical NORMAL_SPREAD_PIPS policy unavailable for context asset",
                    )
            except Exception as exc:  # pragma: no cover - network path
                failure_reason = str(exc)[:256]
                warnings.append(f"{tf} fetch failed: {failure_reason}")
                integrity_by_tf[tf] = _failed_candle_integrity(
                    pair=pair_key,
                    granularity=tf,
                    source="OANDA_MBA" if pair_key in NORMAL_SPREAD_PIPS else "OANDA_MID",
                    reason=failure_reason,
                )
                continue
        integrity_by_tf[tf] = candle_integrity
        if candle_integrity.get("forecast_blocking") is True:
            blocking_codes = candle_integrity.get("blocking_codes") or []
            warnings.append(f"{tf} technical candle integrity BLOCKED: {','.join(map(str, blocking_codes))}")
        candles_by_tf_resolved[tf] = candles
        # Only the latest uninterrupted clean MID run reaches indicators and
        # structure.  This prevents a known quarantined bar from being silently
        # bridged by older history. Genuine no-tick gaps remain inside this run
        # because the MBA parser accepts whole-cadence gaps without resetting
        # ``recent_clean_tail_count``.
        indicators = compute_indicators(pair_key, tf, candles)
        regime, long_bias, short_bias = _classify(indicators)
        structure = analyze_structure(
            candles,
            pivot_strength=3,
            impulse_atr_mult=1.0,
            fvg_lookback=60,
            eq_tolerance_pips=1.5,
            pip_size=indicators.pip_size,
        ) if len(candles) >= 30 else None
        regime_reading, family_scores, stat_filters, smc_reading = _build_reading_layer(
            candles=candles, indicators=indicators, granularity=tf
        )
        market_state = classify_market_state(
            indicators=indicators,
            regime_reading=regime_reading,
            family_scores=family_scores,
            structure=structure,
            smc=smc_reading,
            legacy_regime=regime,
        )
        # Indicator series for divergence detection. Empty when there
        # aren't enough bars to seed the indicator.
        closes_for_series = tuple(c.close for c in candles)
        series_map = {
            "rsi_14": compute_rsi_series(closes_for_series, period=14, count=RECENT_CANDLES_PUBLISH),
            "macd_hist": compute_macd_hist_series(closes_for_series, count=RECENT_CANDLES_PUBLISH),
        }
        views.append(
            ChartView(
                granularity=tf,
                indicators=indicators,
                regime=regime,
                long_bias=long_bias,
                short_bias=short_bias,
                structure=structure,
                regime_reading=regime_reading,
                family_scores=family_scores,
                stat_filters=stat_filters,
                smc=smc_reading,
                recent_candles=tuple(candles[-RECENT_CANDLES_PUBLISH:]) if candles else tuple(),
                indicator_series=series_map,
                candle_integrity=candle_integrity,
                market_state=market_state,
            )
        )

    long_score, short_score, dominant = _aggregate(views)
    chart_story = _build_chart_story(pair_key, views, dominant)
    confluence = _build_confluence(views, long_score, short_score, dominant)
    # 2026-05-13 emergency precision additions (B/C/D for trader_brain
    # and attack_advisor consumption). All metrics are market-derived
    # from the candles already fetched; no JPY/pip literals introduced.
    # `confluence` is the canonical place to publish; readers
    # (trader_brain, attack_advisor) pull from this dict so adding new
    # keys does not force a schema migration.
    confluence.update(_build_extended_confluence(views, candles_by_tf_resolved))
    # Session context: derived from the M5 candles regardless of how they were
    # obtained (caller-supplied or fetched).
    session = _build_session_for_pair(timeframes, candles_by_tf_resolved, views)
    technical_candle_integrity = _aggregate_technical_candle_integrity(
        pair=pair_key,
        requested_timeframes=timeframes,
        integrity_by_tf=integrity_by_tf,
    )
    market_state_summary = summarize_market_states(views)
    return PairChart(
        pair=pair_key,
        views=tuple(views),
        long_score=long_score,
        short_score=short_score,
        dominant_regime=dominant,
        chart_story=chart_story,
        warnings=tuple(warnings),
        session=session,
        confluence=confluence,
        technical_candle_integrity=technical_candle_integrity,
        market_state_summary=market_state_summary,
    )


def _not_evaluated_candle_integrity(
    *,
    pair: str,
    granularity: str,
    source: str,
    clean_count: int,
    reason: str,
) -> dict[str, object]:
    return {
        "schema": TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": pair,
        "granularity": granularity,
        "source": source,
        "spread_evaluation_mode": _candle_spread_evaluation_mode(granularity),
        "evaluation_status": "NOT_EVALUATED",
        "evaluation_reason": reason,
        "clean_count": clean_count,
        "contaminated_count": 0,
        "malformed_count": 0,
        "provenance_complete": False,
        "forecast_blocking": False,
        "codes": [],
        "blocking_codes": [],
    }


def _failed_candle_integrity(
    *, pair: str, granularity: str, source: str, reason: str
) -> dict[str, object]:
    return {
        "schema": TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": pair,
        "granularity": granularity,
        "source": source,
        "spread_evaluation_mode": _candle_spread_evaluation_mode(granularity),
        "evaluation_status": "BLOCKED",
        "evaluation_reason": reason[:256],
        "clean_count": 0,
        "contaminated_count": 0,
        "malformed_count": 1,
        "provenance_complete": False,
        "forecast_blocking": True,
        "codes": [TECHNICAL_CANDLE_PROVENANCE_INVALID],
        "blocking_codes": [TECHNICAL_CANDLE_PROVENANCE_INVALID],
    }


def _candle_spread_evaluation_mode(granularity: str) -> str:
    return (
        TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE
        if granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
        else TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE
    )


def _aggregate_technical_candle_integrity(
    *,
    pair: str,
    requested_timeframes: tuple[str, ...],
    integrity_by_tf: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    sources: list[str] = []
    codes: list[str] = []
    blocking_codes: list[str] = []
    evaluated_count = 0
    for tf in requested_timeframes:
        item = integrity_by_tf.get(tf)
        if not isinstance(item, dict):
            codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
            blocking_codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
            continue
        source = item.get("source")
        if isinstance(source, str) and source:
            sources.append(source)
        if item.get("evaluation_status") != "NOT_EVALUATED":
            evaluated_count += 1
        for code in item.get("codes") or []:
            if isinstance(code, str) and code:
                codes.append(code)
        if item.get("forecast_blocking") is True:
            for code in item.get("blocking_codes") or []:
                if isinstance(code, str) and code:
                    blocking_codes.append(code)
            if not item.get("blocking_codes"):
                codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
                blocking_codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)

    codes = list(dict.fromkeys(codes))
    blocking_codes = list(dict.fromkeys(blocking_codes))
    forecast_blocking = bool(blocking_codes)
    if forecast_blocking:
        status = "BLOCKED"
    elif evaluated_count == 0:
        status = "NOT_EVALUATED"
    elif codes:
        status = "DEGRADED"
    else:
        status = "PASS"
    unique_sources = list(dict.fromkeys(sources))
    source = unique_sources[0] if len(unique_sources) == 1 else "MIXED"
    return {
        "schema": PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": pair,
        "source": source,
        "evaluation_status": status,
        "forecast_blocking": forecast_blocking,
        "codes": codes,
        "blocking_codes": blocking_codes,
        "requested_timeframes": list(requested_timeframes),
        "evaluated_timeframe_count": evaluated_count,
        "timeframes": {tf: integrity_by_tf[tf] for tf in requested_timeframes if tf in integrity_by_tf},
    }


def _build_extended_confluence(
    views: list[ChartView],
    candles_by_tf: dict[str, tuple],
) -> dict[str, object]:
    """B/C/D market-derived extras (2026-05-13).

    Returned dict is merged into PairChart.confluence so existing
    consumers stay untouched. All values are floats or None — None
    when the underlying data is missing rather than substituting a
    JPY/pip literal (AGENT_CONTRACT §3.5).

    Keys produced:
      - `price_percentile_24h` (0.0-1.0): where the current close sits
        in the last-24h H1 close distribution.
      - `price_range_24h_low/high`: the bounds used for the 24h percentile,
        so pending-entry gates can evaluate the entry price instead of the
        current price.
      - `price_percentile_7d` (0.0-1.0): same against the last 7d H4
        close distribution (42 H4 bars).
      - `price_range_7d_low/high`: the close-distribution bounds used for
        the 7d percentile.
      - `atr_percentile_24h`: H1 atr_pips percentile vs trailing
        distribution (read from the H1 indicators if available).
      - `range_24h_sigma_multiple`: (H1 24h high − H1 24h low) divided
        by the median H1 ATR over the same window. >1 means the pair
        has already covered more than one ATR span — buying tops or
        selling bottoms after this point is statistically expensive.
      - `tf_agreement_score` (0.0-1.0): fraction of M15/M30/H1
        agreeing on regime direction (TREND_UP / TREND_DOWN / RANGE /
        UNCLEAR). 1.0 means all three agree, 0.33 means each TF
        prints a different label.
    """
    out: dict[str, object] = {
        "price_percentile_24h": None,
        "price_range_24h_low": None,
        "price_range_24h_high": None,
        "price_percentile_7d": None,
        "price_range_7d_low": None,
        "price_range_7d_high": None,
        "atr_percentile_24h": None,
        "range_24h_sigma_multiple": None,
        "tf_agreement_score": None,
    }

    h1_candles = candles_by_tf.get("H1")
    if h1_candles and len(h1_candles) >= 24:
        last_24 = h1_candles[-24:]
        highs = [c.high for c in last_24]
        lows = [c.low for c in last_24]
        closes_24 = [c.close for c in last_24]
        current = closes_24[-1]
        lo = min(lows)
        hi = max(highs)
        if hi > lo:
            out["price_range_24h_low"] = round(lo, 5)
            out["price_range_24h_high"] = round(hi, 5)
            out["price_percentile_24h"] = round(
                max(0.0, min(1.0, (current - lo) / (hi - lo))), 4
            )
        # Range vs typical hourly ATR over the same window. We use the
        # median per-bar high-low to avoid one wick blowing the metric.
        per_bar_ranges = sorted(c.high - c.low for c in last_24 if c.high > c.low)
        if per_bar_ranges:
            median_range = per_bar_ranges[len(per_bar_ranges) // 2]
            if median_range > 0:
                out["range_24h_sigma_multiple"] = round((hi - lo) / median_range, 3)

    h4_candles = candles_by_tf.get("H4")
    if h4_candles and len(h4_candles) >= 42:
        last_42 = h4_candles[-42:]
        closes_7d = [c.close for c in last_42]
        current = closes_7d[-1]
        lo = min(closes_7d)
        hi = max(closes_7d)
        if hi > lo:
            out["price_range_7d_low"] = round(lo, 5)
            out["price_range_7d_high"] = round(hi, 5)
            out["price_percentile_7d"] = round(
                max(0.0, min(1.0, (current - lo) / (hi - lo))), 4
            )

    # ATR percentile preferentially from H1 indicators (already
    # percentile-of-trailing on the IndicatorSet).
    for view in views:
        if view.granularity == "H1" and view.indicators is not None:
            pct = getattr(view.indicators, "atr_percentile_100", None)
            if pct is not None:
                out["atr_percentile_24h"] = round(float(pct), 4)
            break

    # tf_agreement: M15 / M30 / H1 regime labels, fraction in majority.
    target_tfs = {"M15", "M30", "H1"}
    regimes: list[str] = [
        view.regime for view in views if view.granularity in target_tfs and view.regime
    ]
    if len(regimes) == 3:
        from collections import Counter

        counts = Counter(regimes)
        top_count = max(counts.values())
        out["tf_agreement_score"] = round(top_count / 3.0, 4)

    return out


def _build_reading_layer(
    *, candles: tuple[Candle, ...], indicators: IndicatorSet, granularity: str
) -> tuple[RegimeReading | None, FamilyScores | None, StatFilterReading | None, SMCReading | None]:
    """Compute the four per-view reading-layer artifacts.

    Each is fail-soft: returns None when input data is insufficient. Per
    AGENT_CONTRACT §3.5 we never silently substitute literal fallbacks — the
    trader either gets a real reading or sees None and acts accordingly.
    """

    if not candles:
        return None, None, None, None
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    regime_reading: RegimeReading | None = None
    try:
        regime_reading = classify_regime(
            closes=closes,
            highs=highs,
            lows=lows,
            bars_per_year=_bars_per_year_for(granularity),
            hurst_window=_hurst_window_for(granularity, len(closes)),
        )
        if regime_reading.state == "UNKNOWN":
            indicator_reading = _indicator_regime_reading(indicators, granularity=granularity)
            if indicator_reading.state != "UNKNOWN":
                regime_reading = indicator_reading
    except Exception:
        regime_reading = None

    family_scores: FamilyScores | None = None
    try:
        family_scores = compute_family_scores(indicators)
    except Exception:
        family_scores = None

    stat_filters: StatFilterReading | None = None
    try:
        atr_for_flat = indicators.atr_pips * indicators.pip_size if indicators.atr_pips else None
        stat_filters = compute_stat_filters(
            closes,
            atr_for_flat=atr_for_flat,
            **_stat_filter_kwargs_for(granularity, len(closes)),
        )
    except Exception:
        stat_filters = None

    smc_reading: SMCReading | None = None
    try:
        if len(candles) >= 50:
            smc_reading = analyze_smc(candles)
    except Exception:
        smc_reading = None

    return regime_reading, family_scores, stat_filters, smc_reading


def _bars_per_year_for(granularity: str) -> int:
    return _BARS_PER_YEAR_BY_TIMEFRAME.get(granularity.upper(), _BARS_PER_YEAR_BY_TIMEFRAME["M5"])


def _hurst_window_for(granularity: str, candle_count: int) -> int:
    base = _STAT_FILTER_WINDOWS_BY_TIMEFRAME.get(granularity.upper(), _STAT_FILTER_WINDOWS_BY_TIMEFRAME["M5"])["dfa"]
    # DFA requires window+1 closes. Clamp to available history so the reading can
    # use the fetched packet when possible; classify_regime still reports the
    # effective lookback in RegimeReading.lookback_bars.
    return max(32, min(base, max(candle_count - 1, 32)))


def _stat_filter_kwargs_for(granularity: str, candle_count: int) -> dict[str, int]:
    base = _STAT_FILTER_WINDOWS_BY_TIMEFRAME.get(granularity.upper(), _STAT_FILTER_WINDOWS_BY_TIMEFRAME["M5"])
    # Keep windows inside available history so M1/H4/D default packets return
    # meaningful filters instead of all-None, while underfilled estimators still
    # fail loudly through None fields.
    max_returns = max(candle_count - 1, 0)
    return {
        "autocorr_window": max(8, min(base["autocorr"], max_returns)),
        "moment_window": max(16, min(base["moment"], max_returns)),
        "jump_window": max(4, min(base["jump"], max_returns)),
        "dfa_window": max(32, min(base["dfa"], max_returns)),
    }


def _indicator_regime_reading(indicators: IndicatorSet, *, granularity: str) -> RegimeReading:
    """Build a regime reading from IndicatorSet values when annual lookback is underfilled.

    The pair-charts command fetches a bounded recent candle window every cycle.
    For that window, IndicatorSet already computes Hurst(available history),
    ADX(14), Choppiness(14), and ATR percentile over the fetched window. Using
    those values keeps the reading layer live and transparent; missing values
    still produce UNKNOWN and the `source` field discloses the shorter evidence
    window.
    """
    atr_pct = _indicator_percentile_to_100(indicators.atr_percentile_100)
    return classify_regime_from_values(
        hurst=indicators.hurst_100,
        adx=indicators.adx_14,
        choppiness=indicators.choppiness_14,
        atr_percentile=atr_pct,
        source=f"indicator_set_{granularity}",
        lookback_bars=indicators.candles_count,
    )


def _indicator_percentile_to_100(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _build_session_for_pair(
    timeframes: tuple[str, ...],
    candles_by_tf: Mapping[str, tuple[Candle, ...]] | None,
    views: list[ChartView],
) -> SessionContext | None:
    """Pull session context from the M5 candle stream when available."""

    m5_candles: tuple[Candle, ...] | None = None
    if candles_by_tf is not None and "M5" in candles_by_tf:
        m5_candles = tuple(candles_by_tf["M5"])
    if m5_candles is None or not m5_candles:
        return None
    try:
        return build_session_context(m5_candles)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reading-layer to_dict helpers (small, pure)
# ---------------------------------------------------------------------------


def _regime_reading_to_dict(r: RegimeReading | None) -> dict[str, object] | None:
    if r is None:
        return None
    return {
        "state": r.state,
        "hurst": r.hurst,
        "adx": r.adx,
        "choppiness": r.choppiness,
        "atr_percentile": r.atr_percentile,
        "confidence": r.confidence,
        "source": r.source,
        "lookback_bars": r.lookback_bars,
    }


def _family_scores_to_dict(f: FamilyScores | None) -> dict[str, object] | None:
    if f is None:
        return None
    return {
        "trend_score": round(f.trend_score, 4),
        "mean_rev_score": round(f.mean_rev_score, 4),
        "breakout_score": round(f.breakout_score, 4),
        "disagreement": round(f.disagreement, 4),
        "trend_components": {k: round(v, 4) for k, v in f.trend_components.items()},
        "mean_rev_components": {k: round(v, 4) for k, v in f.mean_rev_components.items()},
        "breakout_components": {k: round(v, 4) for k, v in f.breakout_components.items()},
    }


def _stat_filters_to_dict(s: StatFilterReading | None) -> dict[str, object] | None:
    if s is None:
        return None
    return {
        "lag1_autocorr": s.lag1_autocorr,
        "abs_return_acf_lag1": s.abs_return_acf_lag1,
        "abs_return_acf_decay": s.abs_return_acf_decay,
        "rolling_kurtosis": s.rolling_kurtosis,
        "rolling_skewness": s.rolling_skewness,
        "lee_mykland_jumps": list(s.lee_mykland_jumps),
        "last_jump_bars_ago": s.last_jump_bars_ago,
        "bipower_jump_share": s.bipower_jump_share,
        "flat_spot_count": s.flat_spot_count,
        "hurst_returns": s.hurst_returns,
        "variance_ratio_2": s.variance_ratio_2,
        "variance_ratio_4": s.variance_ratio_4,
    }


def _smc_to_dict(s: SMCReading | None) -> dict[str, object] | None:
    if s is None:
        return None
    # Try the dataclass's own to_dict if present, else build a shallow summary.
    if hasattr(s, "to_dict"):
        try:
            return s.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass
    # Fall back to summary fields the trader cares about most.
    summary: dict[str, object] = {
        "swing_count": len(getattr(s, "swings", ()) or ()),
        "structure_event_count": len(getattr(s, "structure_events", ()) or ()),
        "order_block_count": len(getattr(s, "order_blocks", ()) or ()),
        "fvg_count": len(getattr(s, "fair_value_gaps", ()) or ()),
        "liquidity_count": len(getattr(s, "liquidity", ()) or ()),
        "sweep_count": len(getattr(s, "sweeps", ()) or ()),
        "breaker_count": len(getattr(s, "breakers", ()) or ()),
        "mitigation_count": len(getattr(s, "mitigations", ()) or ()),
        "inversion_fvg_count": len(getattr(s, "inversion_fvgs", ()) or ()),
        "displacement_count": len(getattr(s, "displacements", ()) or ()),
    }
    dr = getattr(s, "dealing_range", None)
    if dr is not None:
        summary["dealing_range"] = {
            "swing_high": dr.swing_high.price if hasattr(dr.swing_high, "price") else None,
            "swing_low": dr.swing_low.price if hasattr(dr.swing_low, "price") else None,
            "equilibrium": getattr(dr, "equilibrium", None),
            "ote_sweet_spot": getattr(dr, "ote_sweet_spot", None),
        }
    return summary


def _session_to_dict(s: SessionContext | None) -> dict[str, object] | None:
    if s is None:
        return None
    current = s.current
    return {
        "current_tag": current.tag.value if hasattr(current.tag, "value") else str(current.tag),
        "ny_local_hour": current.ny_local_hour,
        "jp_holiday": current.jp_holiday,
        "holiday_name": current.holiday_name,
        "ny_midnight_open_utc": current.timestamp_utc.isoformat() if False else (
            s.ny_midnight_open_utc.isoformat() if s.ny_midnight_open_utc else None
        ),
        "ny_midnight_open_price": s.ny_midnight_open_price,
        "asian_range": list(s.asian_range) if s.asian_range else None,
        "london_range": list(s.london_range) if s.london_range else None,
        "ny_am_range": list(s.ny_am_range) if s.ny_am_range else None,
        "judas_armed": s.judas_armed,
        "next_killzone": s.next_killzone.value if (s.next_killzone and hasattr(s.next_killzone, "value")) else None,
        "minutes_to_next_killzone": s.minutes_to_next_killzone,
    }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _classify(ind: IndicatorSet) -> tuple[str, float, float]:
    """Return (regime_tag, long_bias, short_bias) from a single indicator set.

    The classification follows a small ruleset:

    - ADX >= 25 and EMA12 > EMA26 → TREND_UP
    - ADX >= 25 and EMA12 < EMA26 → TREND_DOWN
    - BB span very wide and price beyond upper/lower → IMPULSE_UP/DOWN
    - Donchian width small and ADX < 18 → RANGE
    - Otherwise UNCLEAR

    Bias is built from indicator agreement: each "long-leaning" reading adds to
    long_bias, each "short-leaning" reading adds to short_bias. We normalize to
    0..1 by dividing by the number of evaluated checks.
    """

    if not ind.valid:
        return "UNCLEAR", 0.0, 0.0

    long_signals = 0
    short_signals = 0
    total = 0

    def vote(cond_long: bool, cond_short: bool) -> None:
        nonlocal long_signals, short_signals, total
        total += 1
        if cond_long:
            long_signals += 1
        if cond_short:
            short_signals += 1

    if ind.ema_12 is not None and ind.ema_50 is not None:
        vote(ind.ema_12 > ind.ema_50, ind.ema_12 < ind.ema_50)

    if ind.ema_slope_5 is not None:
        vote(ind.ema_slope_5 > 0, ind.ema_slope_5 < 0)

    if ind.macd_hist is not None:
        vote(ind.macd_hist > 0, ind.macd_hist < 0)

    if ind.rsi_14 is not None:
        vote(ind.rsi_14 > 55, ind.rsi_14 < 45)

    if ind.plus_di_14 is not None and ind.minus_di_14 is not None:
        vote(ind.plus_di_14 > ind.minus_di_14, ind.minus_di_14 > ind.plus_di_14)

    if ind.ichimoku_cloud_pos:
        vote(ind.ichimoku_cloud_pos > 0, ind.ichimoku_cloud_pos < 0)

    if ind.vwap_gap_pips is not None:
        vote(ind.vwap_gap_pips > 0, ind.vwap_gap_pips < 0)

    if ind.roc_10 is not None:
        vote(ind.roc_10 > 0, ind.roc_10 < 0)

    long_bias = long_signals / total if total else 0.0
    short_bias = short_signals / total if total else 0.0

    adx = ind.adx_14 or 0.0
    bb_pips = ind.bb_span_pips or 0.0
    donch_pips = ind.donchian_width_pips or 0.0

    regime = "UNCLEAR"
    if adx >= 25:
        regime = "TREND_UP" if long_bias > short_bias else "TREND_DOWN"
    elif bb_pips and ind.atr_pips and bb_pips > ind.atr_pips * 6 and abs((ind.vwap_gap_pips or 0.0)) > (ind.atr_pips or 0.0) * 1.5:
        regime = "IMPULSE_UP" if long_bias > short_bias else "IMPULSE_DOWN"
    elif adx < 18 and donch_pips and ind.atr_pips and donch_pips < ind.atr_pips * 5:
        regime = "RANGE"
    elif _is_failure_risk(ind):
        regime = "FAILURE_RISK"

    return regime, long_bias, short_bias


def _is_failure_risk(ind: IndicatorSet) -> bool:
    """Detect a fading-momentum / failed-break setup.

    Heuristic: price near a swing extreme but RSI/MACD diverging the wrong
    way relative to direction, ADX rolling under 22, ROC10 contradicting ROC5.
    """

    if ind.rsi_14 is None or ind.adx_14 is None or ind.roc_5 is None or ind.roc_10 is None:
        return False
    fading_adx = ind.adx_14 < 22
    momentum_split = (ind.roc_5 > 0 and ind.roc_10 < 0) or (ind.roc_5 < 0 and ind.roc_10 > 0)
    overextended = (ind.rsi_14 > 70) or (ind.rsi_14 < 30)
    return fading_adx and (momentum_split or overextended)


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation + chart story
# ---------------------------------------------------------------------------


def _aggregate(views: list[ChartView]) -> tuple[float, float, str]:
    """Aggregate per-timeframe biases. Higher timeframes weigh more."""

    if not views:
        return 0.0, 0.0, "UNCLEAR"
    weight_map = {"M5": 1.0, "M15": 1.5, "H1": 2.5, "M30": 1.7, "H4": 3.0, "D": 4.0, "M1": 0.7}
    total_w = 0.0
    long_w = 0.0
    short_w = 0.0
    regime_votes: dict[str, float] = {}
    for view in views:
        w = weight_map.get(view.granularity, 1.0)
        total_w += w
        long_w += view.long_bias * w
        short_w += view.short_bias * w
        regime_votes[view.regime] = regime_votes.get(view.regime, 0.0) + w
    long_score = long_w / total_w if total_w else 0.0
    short_score = short_w / total_w if total_w else 0.0
    dominant = max(regime_votes.items(), key=lambda kv: kv[1])[0]
    return long_score, short_score, dominant


def _build_confluence(
    views: list[ChartView], long_score: float, short_score: float, dominant: str
) -> dict[str, object]:
    """Surface the dominant-regime / higher-TF / score-balance signals
    that the trader chronically under-weights (2026-05-11 incident:
    SHORT into M5 TREND_DOWN while H4 TREND_UP and entry TFs all read
    TREND_WEAK + RSI extremes).

    No thresholds — just exposes existing fields under one key so the
    Confluence Audit in `30_entry_decision.md` can be answered without
    re-reading every view.
    """

    highest: ChartView | None = None
    for preferred in ("D", "H4", "H1"):
        for view in views:
            if view.granularity == preferred:
                highest = view
                break
        if highest is not None:
            break

    higher_regime = highest.regime if highest else "UNKNOWN"
    higher_reading = None
    if highest is not None and highest.regime_reading is not None:
        higher_reading = {
            "state": highest.regime_reading.state,
            "confidence": round(highest.regime_reading.confidence, 4),
        }

    # Score-balance signal: raw gap exposed, trader interprets. The label
    # only flips between TIED / LONG_LEAN / SHORT_LEAN when the gap exceeds
    # the natural spread between bias votes; we surface both so the prompt
    # can apply judgement without a frozen JPY/NAV threshold.
    score_gap = round(long_score - short_score, 4)
    if abs(score_gap) <= 0.05:
        score_balance = "TIED"
    elif score_gap > 0:
        score_balance = "LONG_LEAN"
    else:
        score_balance = "SHORT_LEAN"

    # Direction alignment between aggregate score lean and highest TF
    # regime. Purely categorical — derives from existing regime labels
    # and the score lean above.
    higher_alignment = "MIXED"
    if score_balance == "LONG_LEAN" and higher_regime.startswith("TREND_UP"):
        higher_alignment = "ALIGNED"
    elif score_balance == "SHORT_LEAN" and higher_regime.startswith("TREND_DOWN"):
        higher_alignment = "ALIGNED"
    elif score_balance == "LONG_LEAN" and higher_regime.startswith("TREND_DOWN"):
        higher_alignment = "OPPOSED"
    elif score_balance == "SHORT_LEAN" and higher_regime.startswith("TREND_UP"):
        higher_alignment = "OPPOSED"
    elif higher_regime in {"RANGE", "UNCLEAR"} or score_balance == "TIED":
        higher_alignment = "NEUTRAL"

    return {
        "score_gap": score_gap,
        "score_balance": score_balance,
        "dominant_regime": dominant,
        "higher_tf": highest.granularity if highest else None,
        "higher_tf_regime": higher_regime,
        "higher_tf_reading": higher_reading,
        "higher_tf_alignment": higher_alignment,
    }


def _build_chart_story(pair: str, views: list[ChartView], dominant: str) -> str:
    """Compact chart_story citing ADX/RSI/ATR/BB/Williams/MFI/Aroon/Choppiness/Supertrend
    plus the most recent structure event (BOS / CHoCH) and quantile regime tag.

    The trader cites this string in `chart_story` of the decision receipt; the
    contract (§3.5) requires that decision numbers be sourced from this output
    rather than invented.
    """
    if not views:
        return f"{pair}: no candles available"
    fragments: list[str] = [f"{pair} {dominant}"]
    for view in views:
        ind = view.indicators
        bits = []
        if ind.adx_14 is not None:
            bits.append(f"ADX={ind.adx_14:.1f}")
        if ind.rsi_14 is not None:
            bits.append(f"RSI={ind.rsi_14:.1f}")
        if ind.atr_pips is not None:
            bits.append(f"ATR={ind.atr_pips:.1f}p")
        if ind.bb_span_pips is not None:
            bits.append(f"BB={ind.bb_span_pips:.1f}p")
        if ind.williams_r_14 is not None:
            bits.append(f"%R={ind.williams_r_14:.1f}")
        if ind.mfi_14 is not None:
            bits.append(f"MFI={ind.mfi_14:.1f}")
        if ind.aroon_osc_14 is not None:
            bits.append(f"AroonOsc={ind.aroon_osc_14:.0f}")
        if ind.choppiness_14 is not None:
            bits.append(f"Chop={ind.choppiness_14:.0f}")
        if ind.supertrend_dir is not None:
            bits.append(f"ST={'+' if ind.supertrend_dir > 0 else '-'}")
        if view.regime_reading is not None:
            bits.append(f"Read={view.regime_reading.state}:{view.regime_reading.confidence:.2f}")
        if view.market_state is not None:
            bits.append(f"Phase={view.market_state.phase}/{view.market_state.readiness}")
            bits.append(
                f"Life={view.market_state.trend_maturity} "
                f"DirQ={view.market_state.direction_quality} "
                f"Trigger={view.market_state.trigger} "
                f"Lq={view.market_state.liquidity}"
            )
        if ind.regime_quantile is not None:
            bits.append(f"q={ind.regime_quantile}")
        if ind.ichimoku_cloud_pos:
            cloud = {1: "above", -1: "below", 0: "in"}.get(ind.ichimoku_cloud_pos, "?")
            bits.append(f"cloud={cloud}")
        if view.structure and view.structure.last_event:
            ev = view.structure.last_event
            # `:wick` suffix flags a wick-only break (new swing pivot's
            # candle closed BACK inside the prior range). gpt_trader's
            # CLOSE Gate A ignores wick-only events; trader_brain still
            # sees them as advisory swing information.
            wick_tag = "" if ev.close_confirmed else ":wick"
            bits.append(f"struct={ev.kind}@{ev.broken_pivot_price:.4f}{wick_tag}")
        fragments.append(f"{view.granularity}({view.regime}, {' '.join(bits)})")
    return "; ".join(fragments)
