"""Three composite scores grouped by indicator family.

Per `docs/research/03-quant-ensemble-regime.md` §2, the naive failure mode
is treating 15 indicators as 15 votes when RSI/Stoch/CCI/Williams%R/MFI/ROC
all measure the same momentum factor (rank correlation 0.75-0.92 on FX M5).

The fix is to (1) group by family — Trend / MeanRev / Breakout — and
average **within family first**, so the 6-momentum bloc cannot dominate
the 1-trend reading; (2) Z-score / percentile-rank normalize so absolute
threshold habits (RSI 70 = overbought) don't leak in; (3) emit a
`disagreement` metric so the trader can see "TrendScore=+1, MeanRev=-1 →
no-trade" instead of averaging them to zero.

The regime gate (`regime.py`) decides which score is read: TrendScore in
TREND regimes, MeanRevScore in RANGE, BreakoutScore in BREAKOUT_PENDING.

Pure-stdlib, no numpy/pandas. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid runtime import cycle with indicators.py
    from quant_rabbit.analysis.indicators import IndicatorSet


# ---------------------------------------------------------------------------
# Single-shot scaling constants — per AGENT_CONTRACT §3.5 (a)/(b)/(c)
# ---------------------------------------------------------------------------
#
# These constants exist because `IndicatorSet` exposes single-bar values
# (latest RSI, latest MACD-hist, etc.), not full series. A proper rolling
# Z-score requires history — `normalize.py` handles that case for callers
# that have it. When only a single value is available, we fall back to
# domain-knowledge scaling: dividing by the natural spread of each
# indicator so each contribution lands in roughly [-1, +1].
#
# RSI_HALF_RANGE = 14.5
#   (a) Market reality: RSI is bounded [0, 100] with mean 50 in a noise
#       process. The 1-stdev half-range on FX intraday RSI(14) is ≈14-15
#       — values near 35 or 65 are "1-stdev moves" without being
#       extreme. The midpoint of that empirical range, 14.5, scales
#       `(RSI - 50) / 14.5` to roughly Z-score units.
#   (b) Constant rather than market-derived: this is the *fallback* when a
#       full series isn't supplied. Callers with history pass the
#       normalized value directly via `normalize.py` and bypass the
#       fallback. The scaling is indicator-specification, not market data.
#   (c) Replace via: pass pre-normalized indicator values (preferred), or
#       update this constant if backtest evidence shows a different
#       empirical RSI dispersion on a specific pair/TF. The override
#       seam is to feed normalized inputs through `IndicatorSet`-like
#       extension fields once the chart-reader plumbs `normalize.py` end
#       to end.
RSI_HALF_RANGE: float = 14.5

# STOCH_HALF_RANGE = 14.5
#   (a) Market reality: Stoch %K is bounded [0, 100] with mean 50; on FX
#       intraday Stoch(14,3,3) the 1-stdev half-range is ≈14-15, same
#       footprint as RSI(14) (which is why the source research brief calls
#       them collinear).
#   (b) Constant rather than market-derived: same reason as RSI_HALF_RANGE.
#   (c) Replace via: same as RSI_HALF_RANGE.
STOCH_HALF_RANGE: float = 14.5

# AROON_RANGE = 100.0
#   (a) Market reality: Aroon Up and Aroon Down are each bounded [0, 100],
#       so Aroon-spread ∈ [-100, +100]. Dividing by 100 maps the spread
#       into [-1, +1] without further calibration — appropriate because
#       Aroon is already a "% of period" metric, not a price-distance
#       metric.
#   (b) Constant rather than market-derived: this is the indicator's
#       definition (period count), not market data.
#   (c) Replace via: not expected to change; if you swap to a different
#       Aroon period the bound stays [0, 100].
AROON_RANGE: float = 100.0

# BB_PCTB_HALF_RANGE = 0.5
#   (a) Market reality: BB %B is conventionally [0, 1] (with extension
#       outside the band). %B - 0.5 ∈ [-0.5, +0.5] inside the band;
#       dividing by 0.5 normalizes to [-1, +1] at the band edges.
#   (b) Constant rather than market-derived: definition of the indicator.
#   (c) Replace via: not expected to change.
BB_PCTB_HALF_RANGE: float = 0.5

# VWAP_DIST_ATR_FLOOR = 1.0
#   (a) Market reality: VWAP-distance only carries information once it
#       exceeds the local noise floor (≈1 ATR). Below 1 ATR, "price near
#       VWAP" carries no mean-rev signal. Scaling by max(ATR, floor) keeps
#       the score bounded and noise-aware; the floor of 1.0 (in price
#       units) is a guard for the degenerate ATR=0 case during ultra-quiet
#       sessions.
#   (b) Constant rather than market-derived: this is a numerical guard,
#       not a market value — its only job is to prevent divide-by-zero
#       when ATR has not yet seeded.
#   (c) Replace via: not expected to change; the guard is rarely hit in
#       practice once ATR(14) has accumulated 14 bars.
VWAP_DIST_ATR_FLOOR: float = 1.0


@dataclass(frozen=True)
class FamilyScores:
    """Three composite scores plus per-component breakdown and disagreement.

    Attributes:
        trend_score: Signed; positive = bullish trend strength.
        mean_rev_score: Signed; positive = mean-revert long bias (price
            below mean / oversold).
        breakout_score: Signed magnitude of breakout / squeeze pressure.
            Positive = compressed and ready to expand; the directional
            sign comes from a Donchian/BOS event when present.
        trend_components: Per-indicator contributions averaged into trend.
        mean_rev_components: Per-indicator contributions averaged into mean-rev.
        breakout_components: Per-indicator contributions averaged into breakout.
        disagreement: Population stdev across the three family signs.
            Low + strong sign = clean setup; high = mixed market = stand
            aside (don't average to zero).
    """

    trend_score: float
    mean_rev_score: float
    breakout_score: float
    trend_components: dict[str, float] = field(default_factory=dict)
    mean_rev_components: dict[str, float] = field(default_factory=dict)
    breakout_components: dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _stdev_pop(values: list[float]) -> float:
    """Population stdev. Returns 0.0 for n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return var ** 0.5


def _signed_pct_to_unit(p: float | None) -> float | None:
    """Map a percentile rank (0..100) to a centered [-1, +1] score.

    A percentile of 50 → 0; 100 → +1; 0 → -1. None propagates.
    """
    if p is None:
        return None
    return (p - 50.0) / 50.0


def compute_family_scores(
    indicators: "IndicatorSet",
    *,
    atr_percentile: float | None = None,
    bb_width_percentile: float | None = None,
) -> FamilyScores:
    """Compute Trend / MeanRev / Breakout family scores from an IndicatorSet.

    Per `docs/research/03-quant-ensemble-regime.md` §2 step 3:

        TrendScore = mean( z(price - EMA50)/ATR, z(MACD_hist),
                           sign(SuperTrend), z(Aroon_up - Aroon_down) )
        MeanRevScore = mean( z(BB %B - 0.5), z(RSI - 50)/14.5,
                             z(Stoch - 50)/14.5, z(VWAP_dist) )
            * (-1 if extended)
        BreakoutScore = mean( z(BB_width pct rank), z(ATR pct rank),
                              1{Donchian_break}, 1{recent_BOS} )

    For values not present in `IndicatorSet` (or None), the component is
    skipped — this preserves the family average over whatever the
    indicator pipeline did supply, rather than silently filling zeros.

    Args:
        indicators: Output of `compute_indicators(...)`.
        atr_percentile: Optional pre-computed ATR percentile (0..100). If
            None, falls back to `indicators.atr_percentile_100` scaled to
            0..100 (the IndicatorSet field is 0..1).
        bb_width_percentile: Same convention for BB width.

    Returns:
        FamilyScores. Every component used appears in the corresponding
        `*_components` dict for transparency.
    """
    trend_components: dict[str, float] = {}
    mean_rev_components: dict[str, float] = {}
    breakout_components: dict[str, float] = {}

    pip_size = getattr(indicators, "pip_size", 0.0001) or 0.0001

    # ---- TrendScore -------------------------------------------------------
    # (1) (price - EMA50) / ATR  — vol-normalized trend distance.
    close = getattr(indicators, "close", None)
    ema50 = getattr(indicators, "ema_50", None)
    atr = getattr(indicators, "atr_14", None)
    if close is not None and ema50 is not None and atr is not None and atr > 0:
        # Cap at +/-3 to avoid one-bar outliers dominating the family mean.
        # 3 stdev is the conventional outlier ceiling; values beyond that
        # are pinned but their sign is preserved.
        v = (close - ema50) / atr
        if v > 3.0:
            v = 3.0
        elif v < -3.0:
            v = -3.0
        trend_components["price_vs_ema50_per_atr"] = v

    # (2) MACD histogram — directional trend acceleration. Use the raw
    # value scaled by pip_size so cross-pair magnitudes are comparable.
    macd_hist = getattr(indicators, "macd_hist", None)
    if macd_hist is not None and pip_size > 0:
        # Scale: hist / pip_size lands roughly in [-10, +10] on FX intraday;
        # divide by 5 to put it in roughly [-2, +2] for the family mean.
        v = (macd_hist / pip_size) / 5.0
        if v > 3.0:
            v = 3.0
        elif v < -3.0:
            v = -3.0
        trend_components["macd_hist_scaled"] = v

    # (3) SuperTrend direction — discrete +1 / -1 / 0.
    st_dir = getattr(indicators, "supertrend_dir", None)
    if st_dir is not None:
        trend_components["supertrend_dir"] = float(st_dir)

    # (4) Aroon spread — already on [-100, +100], normalize to [-1, +1].
    aroon_up = getattr(indicators, "aroon_up_14", None)
    aroon_down = getattr(indicators, "aroon_down_14", None)
    if aroon_up is not None and aroon_down is not None:
        trend_components["aroon_spread"] = (aroon_up - aroon_down) / AROON_RANGE

    trend_score = _safe_mean(list(trend_components.values()))

    # ---- MeanRevScore -----------------------------------------------------
    # (1) BB %B - 0.5 — location within the band. Positive %B-0.5 = upper
    # half (mean-rev SHORT bias = negative score). Flip sign accordingly.
    bb_upper = getattr(indicators, "bb_upper", None)
    bb_lower = getattr(indicators, "bb_lower", None)
    if (close is not None and bb_upper is not None and bb_lower is not None
            and bb_upper > bb_lower):
        pct_b = (close - bb_lower) / (bb_upper - bb_lower)
        # %B-0.5 ∈ [-0.5,+0.5]; flip so positive = mean-rev LONG bias.
        mean_rev_components["bb_pct_b_inverted"] = -((pct_b - 0.5) / BB_PCTB_HALF_RANGE)

    # (2) (RSI - 50) / 14.5 — momentum extreme. RSI > 50 → mean-rev SHORT
    # bias (negative score).
    rsi = getattr(indicators, "rsi_14", None)
    if rsi is not None:
        mean_rev_components["rsi_inverted"] = -((rsi - 50.0) / RSI_HALF_RANGE)

    # (3) Stoch %K (or stoch_rsi as a proxy on this codebase) — same logic.
    stoch = getattr(indicators, "stoch_rsi", None)
    if stoch is not None:
        # stoch_rsi is on [0, 100] in this codebase (see indicators.py).
        mean_rev_components["stoch_inverted"] = -((stoch - 50.0) / STOCH_HALF_RANGE)

    # (4) VWAP distance — extended above VWAP → mean-rev SHORT bias.
    vwap = getattr(indicators, "vwap", None)
    if (close is not None and vwap is not None and atr is not None
            and atr > 0):
        # Distance in ATR units, sign-flipped for mean-rev convention.
        denom = max(atr, VWAP_DIST_ATR_FLOOR * pip_size)
        v = -((close - vwap) / denom)
        if v > 3.0:
            v = 3.0
        elif v < -3.0:
            v = -3.0
        mean_rev_components["vwap_dist_inverted"] = v

    mean_rev_score = _safe_mean(list(mean_rev_components.values()))

    # ---- BreakoutScore ----------------------------------------------------
    # (1) BB width percentile (caller-supplied or from IndicatorSet).
    bbw_pct = bb_width_percentile
    if bbw_pct is None:
        raw = getattr(indicators, "bb_width_percentile_100", None)
        if raw is not None:
            bbw_pct = raw * 100.0  # IndicatorSet stores 0..1
    if bbw_pct is not None:
        # Compressed BB (low percentile) → high breakout pressure.
        # Map percentile 0 → +1 (squeeze), 100 → -1 (already expanded).
        breakout_components["bb_width_squeeze"] = -((bbw_pct - 50.0) / 50.0)

    # (2) ATR percentile.
    atr_pct = atr_percentile
    if atr_pct is None:
        raw = getattr(indicators, "atr_percentile_100", None)
        if raw is not None:
            atr_pct = raw * 100.0
    if atr_pct is not None:
        breakout_components["atr_squeeze"] = -((atr_pct - 50.0) / 50.0)

    # (3) Donchian break — close outside the prior Donchian band.
    donchian_high = getattr(indicators, "donchian_high", None)
    donchian_low = getattr(indicators, "donchian_low", None)
    if (close is not None and donchian_high is not None
            and donchian_low is not None):
        if close >= donchian_high:
            breakout_components["donchian_break"] = 1.0
        elif close <= donchian_low:
            breakout_components["donchian_break"] = -1.0
        else:
            breakout_components["donchian_break"] = 0.0

    # (4) BB squeeze flag — proxy for "recent BOS" pre-condition since the
    # IndicatorSet doesn't currently surface BOS/CHOCH directly.
    bb_squeeze = getattr(indicators, "bb_squeeze", None)
    if bb_squeeze is not None:
        breakout_components["bb_squeeze_flag"] = float(bb_squeeze)

    breakout_score = _safe_mean(list(breakout_components.values()))

    # ---- Disagreement -----------------------------------------------------
    # Population stdev of the three family-score signs. Per the source
    # research §2 step 4: low stdev + strong sign = clean setup; high
    # stdev = mixed market = stand aside.
    disagreement = _stdev_pop([
        _sign(trend_score),
        _sign(mean_rev_score),
        _sign(breakout_score),
    ])

    return FamilyScores(
        trend_score=trend_score,
        mean_rev_score=mean_rev_score,
        breakout_score=breakout_score,
        trend_components=trend_components,
        mean_rev_components=mean_rev_components,
        breakout_components=breakout_components,
        disagreement=disagreement,
    )
