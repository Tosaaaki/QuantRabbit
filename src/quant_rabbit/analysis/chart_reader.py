"""Pair-level chart view that lets the trader rank pairs by edge.

For each pair, fetch candles at M5/M15/H1, run the indicator stack, and emit:

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

from quant_rabbit.analysis.candles import Candle, fetch_candles_via_client
from quant_rabbit.analysis.indicators import IndicatorSet, compute_indicators
from quant_rabbit.analysis.structure import StructureReading, analyze_structure
from quant_rabbit.broker.oanda import OandaReadOnlyClient

# Reading-layer additions (phases 1-3 per docs/research/ + AGENT_CONTRACT §6).
# These are fail-soft: when underlying data is insufficient the readings are
# emitted as None, never silently faked.
from quant_rabbit.analysis.regime import RegimeReading, classify_regime, classify_regime_from_values
from quant_rabbit.analysis.families import FamilyScores, compute_family_scores
from quant_rabbit.analysis.statfilters import StatFilterReading, compute_stat_filters
from quant_rabbit.analysis.sessions import SessionContext, build_session_context
from quant_rabbit.analysis.smc import SMCReading, analyze_smc


DEFAULT_TIMEFRAMES: tuple[str, ...] = ("M5", "M15", "H1")


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

    `candles_by_tf` lets tests inject prebuilt series; in production we go
    through `client.get_json` via `fetch_candles_via_client`.
    """

    views: list[ChartView] = []
    warnings: list[str] = []
    candles_by_tf_resolved: dict[str, tuple[Candle, ...]] = {}
    for tf in timeframes:
        if candles_by_tf is not None and tf in candles_by_tf:
            candles = tuple(candles_by_tf[tf])
        else:
            try:
                candles = fetch_candles_via_client(client, pair, tf, count=count)
            except Exception as exc:  # pragma: no cover - network path
                warnings.append(f"{tf} fetch failed: {exc}")
                continue
        candles_by_tf_resolved[tf] = candles
        indicators = compute_indicators(pair, tf, candles)
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
            )
        )

    long_score, short_score, dominant = _aggregate(views)
    chart_story = _build_chart_story(pair, views, dominant)
    # Session context: derived from the M5 candles regardless of how they were
    # obtained (caller-supplied or fetched).
    session = _build_session_for_pair(timeframes, candles_by_tf_resolved, views)
    return PairChart(
        pair=pair,
        views=tuple(views),
        long_score=long_score,
        short_score=short_score,
        dominant_regime=dominant,
        chart_story=chart_story,
        warnings=tuple(warnings),
        session=session,
    )


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
        regime_reading = classify_regime(closes=closes, highs=highs, lows=lows)
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
        stat_filters = compute_stat_filters(closes, atr_for_flat=atr_for_flat)
    except Exception:
        stat_filters = None

    smc_reading: SMCReading | None = None
    try:
        if len(candles) >= 50:
            smc_reading = analyze_smc(candles)
    except Exception:
        smc_reading = None

    return regime_reading, family_scores, stat_filters, smc_reading


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
        if ind.regime_quantile is not None:
            bits.append(f"q={ind.regime_quantile}")
        if ind.ichimoku_cloud_pos:
            cloud = {1: "above", -1: "below", 0: "in"}.get(ind.ichimoku_cloud_pos, "?")
            bits.append(f"cloud={cloud}")
        if view.structure and view.structure.last_event:
            ev = view.structure.last_event
            bits.append(f"struct={ev.kind}@{ev.broken_pivot_price:.4f}")
        fragments.append(f"{view.granularity}({view.regime}, {' '.join(bits)})")
    return "; ".join(fragments)
