"""Statistical filters on the price-stream itself.

Reference: ``docs/research/03-quant-ensemble-regime.md`` §3
("Statistical filters on the price stream itself").

These filters operate on the *return series* derived from a sequence of
closes — independent of the indicator panel — and are designed to detect
states that a smoothed-indicator stack systematically misses:

- jumps and news spikes (Lee-Mykland test, bipower variation),
- volatility clustering (autocorrelation of |r|),
- distributional regime change (rolling kurtosis / skewness),
- microstructure / serial dependence (lag-1 autocorr, variance ratio),
- thin-liquidity flat spots (count of ATR-scaled near-zero bars),
- long-memory persistence (Hurst exponent of returns via DFA).

Pure functions, ``stdlib + math`` only. No ``scipy`` / ``pandas`` /
``numpy``. Every numeric default is documented inline per
``docs/AGENT_CONTRACT.md`` §3.5: (a) what market reality it represents,
(b) why it is constant rather than market-derived, (c) what should
replace it if it ever needs to be changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, pi, sqrt
from typing import Sequence


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatFilterReading:
    """Snapshot of every statistical filter computed from a closes series.

    Any field is ``None`` when the underlying window is underfilled or
    the input is degenerate (e.g. zero variance). Per AGENT_CONTRACT
    §3.5, callers must treat ``None`` as a missing input — never silently
    fall back to a literal.
    """

    lag1_autocorr: float | None
    """Pearson autocorrelation of returns at lag 1 over rolling window."""

    abs_return_acf_lag1: float | None
    """Autocorrelation of |r_t| at lag 1 — volatility clustering proxy."""

    abs_return_acf_decay: float | None
    """Mean ACF of |r_t| over lags 1..10."""

    rolling_kurtosis: float | None
    """Excess kurtosis of returns over the moment window (Gaussian = 0)."""

    rolling_skewness: float | None
    """Skewness of returns over the moment window."""

    lee_mykland_jumps: tuple[int, ...]
    """Bar indices (into the input ``closes``) flagged as jumps in the last window."""

    last_jump_bars_ago: int | None
    """Bars elapsed since the most recent flagged jump (None if none)."""

    bipower_jump_share: float | None
    """``max(0, RV - BV) / RV`` — fraction of variance attributable to jumps."""

    flat_spot_count: int
    """Count of bars in the last window with |r| < ATR-scaled epsilon."""

    hurst_returns: float | None
    """DFA Hurst exponent of the returns series (0.5 = random walk)."""

    variance_ratio_2: float | None
    """Lo-MacKinlay variance ratio at q=2."""

    variance_ratio_4: float | None
    """Lo-MacKinlay variance ratio at q=4."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns(closes: Sequence[float]) -> list[float]:
    """Simple arithmetic returns r_t = (P_t - P_{t-1}) / P_{t-1}.

    Uses arithmetic returns rather than log returns so an ATR-scaled
    epsilon (in price units) can be compared directly to |r| × P. The
    distinction is negligible at FX scales but documented per §3.5.
    """
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0 or not isfinite(prev) or not isfinite(closes[i]):
            out.append(0.0)
            continue
        out.append((closes[i] - prev) / prev)
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: Sequence[float], *, ddof: int = 0) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


# ---------------------------------------------------------------------------
# Lag-1 autocorrelation
# ---------------------------------------------------------------------------


def lag1_autocorr(returns: Sequence[float], *, window: int = 100) -> float | None:
    """Sample Pearson autocorrelation of ``returns`` at lag 1.

    §3.5 docstring on the default:
    - (a) ``window=100`` represents "roughly the last 8 hours of M5
      bars" — long enough to estimate a one-parameter correlation but
      short enough to track regime drift.
    - (b) constant rather than ATR-derived because the estimator's
      variance shrinks like ``1/N``; tying it to ATR gives no benefit.
    - (c) replace with a per-timeframe override (e.g. 200 for M1, 50
      for H1) if the cycle ever needs different memory length.
    """
    if window < 8:
        return None
    n = len(returns)
    if n < window + 1:
        return None
    sample = list(returns[-window:])
    m = _mean(sample)
    num = 0.0
    den = 0.0
    for i in range(1, len(sample)):
        num += (sample[i] - m) * (sample[i - 1] - m)
    for x in sample:
        den += (x - m) ** 2
    if den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# ACF of |returns|
# ---------------------------------------------------------------------------


def abs_return_acf(returns: Sequence[float], *, max_lag: int = 10) -> tuple[float, ...]:
    """Autocorrelation of ``|r_t|`` at lags 1..max_lag.

    §3.5 docstring on the default:
    - (a) ``max_lag=10`` represents "look ten bars back for vol
      clustering" — typical ARCH/GARCH effects decay within ~10 lags
      at intraday FX cadence.
    - (b) constant because longer lags add noise without improving the
      cluster signal.
    - (c) replace with 20 if running on tick or sub-minute data where
      memory persists longer.
    """
    if max_lag < 1:
        return tuple()
    abs_r = [abs(x) for x in returns]
    n = len(abs_r)
    if n < max_lag + 2:
        return tuple()
    m = _mean(abs_r)
    den = sum((x - m) ** 2 for x in abs_r)
    if den <= 0:
        return tuple(0.0 for _ in range(max_lag))
    out: list[float] = []
    for lag in range(1, max_lag + 1):
        num = 0.0
        for i in range(lag, n):
            num += (abs_r[i] - m) * (abs_r[i - lag] - m)
        out.append(num / den)
    return tuple(out)


# ---------------------------------------------------------------------------
# Rolling moments
# ---------------------------------------------------------------------------


def rolling_moment(
    returns: Sequence[float],
    *,
    window: int = 200,
    kind: str = "kurtosis",
) -> float | None:
    """Excess kurtosis or skewness of the last ``window`` returns.

    §3.5 docstring on the default:
    - (a) ``window=200`` represents "about a trading day of M5 bars" —
      enough samples for a stable 4th moment estimate while still
      adapting day-to-day.
    - (b) constant rather than market-derived because kurtosis has
      ~1/sqrt(N) noise; below ~150 samples the estimate is unusable.
    - (c) replace with 500 for daily-bar work, 100 for tick.

    ``kind`` must be ``"kurtosis"`` (Fisher excess) or ``"skewness"``.
    """
    if kind not in ("kurtosis", "skewness"):
        raise ValueError(f"unknown kind={kind!r}")
    if window < 16:
        return None
    n = len(returns)
    if n < window:
        return None
    sample = list(returns[-window:])
    m = _mean(sample)
    var = _variance(sample, ddof=0)
    if var <= 0:
        return None
    sd = sqrt(var)
    if kind == "kurtosis":
        # Fisher's excess kurtosis: E[((x-m)/sd)^4] - 3
        s = sum(((x - m) / sd) ** 4 for x in sample) / len(sample)
        return s - 3.0
    # skewness
    s = sum(((x - m) / sd) ** 3 for x in sample) / len(sample)
    return s


# ---------------------------------------------------------------------------
# Bipower variation
# ---------------------------------------------------------------------------


def bipower_variation(
    returns: Sequence[float],
    *,
    window: int = 50,
) -> tuple[float, float]:
    """Realized variance and bipower variation over the last ``window`` returns.

    BV = (π/2) × Σ |r_t| × |r_{t-1}|  (Barndorff-Nielsen & Shephard)
    RV = Σ r_t²

    BV is robust to jumps; RV is not. RV - BV measures jump variance.

    §3.5 docstring on the default:
    - (a) ``window=50`` represents "last ~4 hours of M5 bars" — long
      enough to suppress single-bar noise, short enough to localize
      a jump regime.
    - (b) constant because BV's bias scales as 1/window; below ~30
      bars the estimator is too noisy to trust.
    - (c) replace with 100 for tick data, 20 for hourly bars.

    Returns ``(RV, BV)``. Both are 0.0 if ``window`` is underfilled.
    """
    if window < 4:
        return (0.0, 0.0)
    n = len(returns)
    if n < window + 1:
        return (0.0, 0.0)
    sample = list(returns[-window:])
    rv = sum(r * r for r in sample)
    bv = 0.0
    for i in range(1, len(sample)):
        bv += abs(sample[i]) * abs(sample[i - 1])
    bv *= (pi / 2.0)
    return (rv, bv)


# ---------------------------------------------------------------------------
# Lee-Mykland test
# ---------------------------------------------------------------------------


def _gumbel_threshold(window: int, alpha: float) -> float:
    """Gumbel-quantile critical value for the Lee-Mykland statistic.

    From Lee & Mykland (RFS 2008): under the null of no jumps, the
    rescaled max statistic
        ξ = (max|L_t| - C_n) / S_n
    follows a Gumbel(0,1) distribution, where
        C_n = sqrt(2 ln n) - (ln(π) + ln(ln n)) / (2 sqrt(2 ln n))
        S_n = 1 / sqrt(2 ln n)
    Inverse Gumbel CDF: q(α) = -ln(-ln α). Threshold on |L_t| is then
    ``C_n + S_n × q(α)``.

    §3.5 docstring on the default ``alpha=0.999``:
    - (a) represents "1-in-1000 false-positive tolerance per bar" —
      Lee-Mykland's recommended cutoff for intraday jump detection.
    - (b) constant because the Gumbel tail decays fast; loosening to
      0.99 gives ~10× more false jumps and the filter loses meaning.
    - (c) replace with 0.9999 for very high-frequency data where the
      effective number of tests is much larger.
    """
    if window < 4:
        return float("inf")
    ln_n = log(window)
    sqrt_2lnn = sqrt(2.0 * ln_n)
    c_n = sqrt_2lnn - (log(pi) + log(ln_n)) / (2.0 * sqrt_2lnn)
    s_n = 1.0 / sqrt_2lnn
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    q = -log(-log(alpha))
    return c_n + s_n * q


def lee_mykland_test(
    returns: Sequence[float],
    *,
    window: int = 50,
    alpha: float = 0.999,
) -> tuple[int, ...]:
    """Indices of bars flagged as jumps by the Lee-Mykland (2008) test.

    For each bar t in the last ``window`` bars, σ̂_t is estimated from
    bipower variation over the preceding ``window`` returns and
    L_t = r_t / σ̂_t is compared to a Gumbel critical value at
    confidence ``alpha``.

    §3.5 docstring on the defaults:
    - ``window=50``: see :func:`bipower_variation` — same trade-off.
    - ``alpha=0.999``: see :func:`_gumbel_threshold` — Lee-Mykland's
      recommended intraday cutoff.

    Indices returned are positions in the ORIGINAL ``returns`` array
    (i.e. ``returns[i]`` was flagged), not positions in the window.
    """
    if window < 4:
        return tuple()
    n = len(returns)
    if n < window + 2:
        return tuple()
    threshold = _gumbel_threshold(window, alpha)
    jumps: list[int] = []
    # Scan the last `window` bars; for each, compute BV from preceding window.
    start = max(window, n - window)
    for t in range(start, n):
        prev = returns[t - window:t]
        if len(prev) < window:
            continue
        bv = 0.0
        for i in range(1, len(prev)):
            bv += abs(prev[i]) * abs(prev[i - 1])
        bv *= (pi / 2.0)
        # σ̂ per bar = sqrt(BV / window)
        if bv <= 0:
            continue
        sigma_hat = sqrt(bv / window)
        if sigma_hat <= 0:
            continue
        l_stat = abs(returns[t]) / sigma_hat
        if l_stat > threshold:
            jumps.append(t)
    return tuple(jumps)


# ---------------------------------------------------------------------------
# Detrended Fluctuation Analysis (Hurst exponent)
# ---------------------------------------------------------------------------


def detrended_fluctuation_hurst(
    values: Sequence[float],
    *,
    min_box: int = 4,
    max_box: int | None = None,
) -> float | None:
    """Hurst exponent of ``values`` via Detrended Fluctuation Analysis.

    Algorithm:
      1. Build the integrated profile y_k = Σ (values_i - mean).
      2. For each box size s in [min_box, max_box], split into
         non-overlapping boxes, fit a least-squares line within each
         box, take RMS of detrended residuals → F(s).
      3. Hurst = slope of log F(s) vs log s.

    §3.5 docstring on the defaults:
    - (a) ``min_box=4`` represents "smallest box where a linear fit is
      meaningful". Boxes < 4 underdetermine the regression.
    - (b) constant because the lower bound is purely statistical, not
      market-derived.
    - (c) replace with 8 if the input series is heavily smoothed.

    - (a) ``max_box=None`` defaults to ``len(values) // 4`` — the
      standard DFA convention so the largest box still has 4 samples
      worth of variance estimate.
    - (b) constant ratio because beyond N/4 the slope estimate becomes
      dominated by a single box.
    - (c) replace with N/2 only for very long series (>10000 samples).
    """
    n = len(values)
    if n < 32:
        return None
    if max_box is None:
        max_box = n // 4
    if max_box <= min_box:
        return None

    m = _mean(values)
    profile = [0.0] * n
    acc = 0.0
    for i in range(n):
        acc += values[i] - m
        profile[i] = acc

    # Box sizes: geometric spacing, base 2
    sizes: list[int] = []
    s = min_box
    while s <= max_box:
        sizes.append(s)
        s = max(s + 1, int(s * 1.5))
    if len(sizes) < 4:
        return None

    log_s: list[float] = []
    log_f: list[float] = []
    for size in sizes:
        n_boxes = n // size
        if n_boxes < 1:
            continue
        sq_sum = 0.0
        total = 0
        for b in range(n_boxes):
            seg = profile[b * size:(b + 1) * size]
            # Fit y = a*x + c on x = 0..size-1
            sx = sum(range(size))
            sxx = sum(i * i for i in range(size))
            sy = sum(seg)
            sxy = sum(i * seg[i] for i in range(size))
            denom = size * sxx - sx * sx
            if denom == 0:
                continue
            a = (size * sxy - sx * sy) / denom
            c = (sy - a * sx) / size
            for i in range(size):
                resid = seg[i] - (a * i + c)
                sq_sum += resid * resid
                total += 1
        if total == 0 or sq_sum <= 0:
            continue
        f_s = sqrt(sq_sum / total)
        if f_s <= 0:
            continue
        log_s.append(log(size))
        log_f.append(log(f_s))
    if len(log_s) < 4:
        return None

    # Linear regression slope
    nn = len(log_s)
    mx = sum(log_s) / nn
    my = sum(log_f) / nn
    num = sum((log_s[i] - mx) * (log_f[i] - my) for i in range(nn))
    den = sum((log_s[i] - mx) ** 2 for i in range(nn))
    if den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Variance Ratio (Lo-MacKinlay)
# ---------------------------------------------------------------------------


def variance_ratio(returns: Sequence[float], *, q: int = 2) -> float | None:
    """Lo-MacKinlay variance ratio at horizon ``q``.

    VR(q) = Var(r_t + r_{t+1} + ... + r_{t+q-1}) / (q × Var(r_t))

    Interpretation:
      - VR(q) ≈ 1  → random walk (no serial dependence at horizon q)
      - VR(q) > 1  → trending / positive autocorrelation
      - VR(q) < 1  → mean-reverting / negative autocorrelation

    §3.5 docstring on the default:
    - (a) ``q=2`` represents "is there persistence between adjacent
      bars" — the most common Lo-MacKinlay test horizon.
    - (b) constant because VR is a function of q itself, not a
      market-derived parameter.
    - (c) callers wanting a horizon scan compute it for q=2, 4, 8, etc.
    """
    if q < 2:
        return None
    n = len(returns)
    if n < q * 4:
        return None
    var_1 = _variance(returns, ddof=1)
    if var_1 <= 0:
        return None
    # q-period overlapping returns
    q_returns = []
    for i in range(n - q + 1):
        q_returns.append(sum(returns[i:i + q]))
    var_q = _variance(q_returns, ddof=1)
    if var_q <= 0:
        return None
    return var_q / (q * var_1)


# ---------------------------------------------------------------------------
# End-to-end aggregator
# ---------------------------------------------------------------------------


def compute_stat_filters(
    closes: Sequence[float],
    *,
    autocorr_window: int = 100,
    moment_window: int = 200,
    jump_window: int = 50,
    jump_alpha: float = 0.999,
    flat_spot_epsilon_atr_ratio: float = 0.05,
    atr_for_flat: float | None = None,
    dfa_window: int = 200,
) -> StatFilterReading:
    """Compute every statistical filter from a closes series.

    §3.5 docstring on each default:
    - ``autocorr_window=100``: see :func:`lag1_autocorr`.
    - ``moment_window=200``: see :func:`rolling_moment`.
    - ``jump_window=50`` / ``jump_alpha=0.999``: see
      :func:`lee_mykland_test`.
    - ``flat_spot_epsilon_atr_ratio=0.05``: a bar is "flat" when
      ``|ΔP| < 0.05 × ATR``. (a) Represents "less than 1/20 of normal
      bar range" — robust threshold for thin liquidity / quote-frozen
      conditions per §3.5. (b) Constant ratio rather than absolute
      because epsilon is then automatically ATR-derived (per AGENT
      CONTRACT §3.5: spread / geometry must be ATR-derived). (c)
      Replace with 0.10 if false positives in normal markets become
      a problem; 0.02 for tick data.
    - ``atr_for_flat=None``: caller MUST supply the current ATR in
      price units. If ``None``, the flat-spot count falls through to
      0 (no silent literal — the operator sees zeros and knows ATR is
      missing). Per AGENT_CONTRACT §3.5 we never invent a JPY/pip
      number here.
    - ``dfa_window=200``: tail of the closes series used for the DFA
      Hurst calc. Same rationale as ``moment_window``.
    """
    rets = _returns(closes)
    n_rets = len(rets)

    # Lag-1 autocorr
    ac1 = lag1_autocorr(rets, window=autocorr_window)

    # |r| ACF
    acf = abs_return_acf(rets, max_lag=10)
    abs_acf_lag1 = acf[0] if acf else None
    abs_acf_decay = (sum(acf) / len(acf)) if acf else None

    # Rolling moments
    kurt = rolling_moment(rets, window=moment_window, kind="kurtosis")
    skew = rolling_moment(rets, window=moment_window, kind="skewness")

    # Lee-Mykland jumps
    jumps = lee_mykland_test(rets, window=jump_window, alpha=jump_alpha)
    last_bars_ago: int | None = None
    if jumps:
        # `jumps` are indices into the returns array; bars-ago is from end of returns.
        last_bars_ago = (n_rets - 1) - jumps[-1]

    # Bipower jump share
    rv, bv = bipower_variation(rets, window=jump_window)
    if rv > 0:
        jump_share: float | None = max(0.0, rv - bv) / rv
    else:
        jump_share = None

    # Flat-spot count over the last `jump_window` bars
    flat_count = 0
    if atr_for_flat is not None and atr_for_flat > 0 and closes:
        # epsilon in PRICE units (ATR is price-unit). Compare to |ΔP|, not |r|.
        epsilon = atr_for_flat * flat_spot_epsilon_atr_ratio
        # Use absolute price differences over the last jump_window+1 closes.
        tail_closes = list(closes[-(jump_window + 1):])
        for i in range(1, len(tail_closes)):
            if abs(tail_closes[i] - tail_closes[i - 1]) < epsilon:
                flat_count += 1

    # DFA Hurst on the tail of the returns series
    dfa_input = list(rets[-dfa_window:]) if n_rets >= dfa_window else list(rets)
    hurst = detrended_fluctuation_hurst(dfa_input)

    # Variance ratios
    vr2 = variance_ratio(rets, q=2)
    vr4 = variance_ratio(rets, q=4)

    return StatFilterReading(
        lag1_autocorr=ac1,
        abs_return_acf_lag1=abs_acf_lag1,
        abs_return_acf_decay=abs_acf_decay,
        rolling_kurtosis=kurt,
        rolling_skewness=skew,
        lee_mykland_jumps=jumps,
        last_jump_bars_ago=last_bars_ago,
        bipower_jump_share=jump_share,
        flat_spot_count=flat_count,
        hurst_returns=hurst,
        variance_ratio_2=vr2,
        variance_ratio_4=vr4,
    )
