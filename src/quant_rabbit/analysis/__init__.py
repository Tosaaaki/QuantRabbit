"""Indicator and chart-reading toolkit for the discretionary trader.

The trader is the decision maker; this package builds the objective material
the trader reads. Pure-Python, no third-party dependencies.

Layers (per docs/research/ + AGENT_CONTRACT §3.5 / §6):

1. Indicators (indicators.py) — classical technical panel.
2. Normalization (normalize.py) — Z-score / percentile rank per pair/TF.
3. Regime gate (regime.py) — Hurst + ADX + Choppiness + ATR_pct → 4-state.
4. Family scoring (families.py) — Trend/MeanRev/Breakout composites.
5. Structure / SMC (structure.py, smc.py) — swings, BOS/CHOCH, OB, FVG,
   Sweep, Breaker, Mitigation, iFVG, Displacement, Dealing Range, OTE.
6. Levels (levels.py) — pivots, PDH/PDL/PWH/PWL, session ranges.
7. Sessions (sessions.py) — killzones, Judas window, Silver Bullet,
   NY midnight open, JP holiday flag.
8. TPO profile (profile.py) — POC, VAH/VAL, IB, Day Type, Open Type, NPOC.
9. Statistical filters (statfilters.py) — Lee-Mykland jumps, autocorr,
   ACF, kurtosis, Hurst-on-returns, variance ratio.
10. Cross-asset / strength / flow / options (cross_asset.py, strength.py,
    flow.py, options.py) — intermarket and microstructure proxies.
11. Calendar (calendar.py) — ForexFactory feed; (calendar_tier.py) — tier
    S/A/B/C scoring with minutes-to-event.
12. COT (cot.py) — CFTC fetch/parse; (cot_signals.py) — net %, z-score,
    week delta, commercial-extreme contrarian gate.
13. Macro / risk score (macro.py) — FRED yields/vol/credit, USD-credibility
    regime check, single-scalar risk score composite.
"""

from quant_rabbit.analysis.candles import Candle, fetch_candles, fetch_candles_via_client
from quant_rabbit.analysis.indicators import IndicatorSet, compute_indicators
from quant_rabbit.analysis.chart_reader import ChartView, PairChart, build_pair_chart

# Phase 1 — normalization, regime gate, family scoring
from quant_rabbit.analysis.normalize import (
    NormalizedValue,
    rolling_z,
    rolling_percentile_rank,
    normalize_indicator,
)
from quant_rabbit.analysis.regime import RegimeReading, classify_regime
from quant_rabbit.analysis.families import FamilyScores, compute_family_scores

# Phase 2 — SMC primitives, sessions, levels (existing)
from quant_rabbit.analysis.smc import (
    LiquiditySweep,
    BreakerBlock,
    MitigationBlock,
    InversionFVG,
    DisplacementCandle,
    DealingRange,
    SMCReading,
    detect_sweeps,
    detect_breakers,
    detect_mitigations,
    detect_inversion_fvgs,
    detect_displacement,
    compute_dealing_range,
    compute_premium_discount,
    analyze_smc,
)
from quant_rabbit.analysis.sessions import (
    SessionTag,
    SessionMarker,
    SessionContext,
    tag_bar,
    build_session_context,
    jp_holiday_calendar_for,
    ny_midnight_open,
)

# Phase 3 — TPO profile, statistical filters
from quant_rabbit.analysis.profile import (
    TPOBracket,
    TPOProfile,
    build_tpo_profile,
    naked_pocs,
)
from quant_rabbit.analysis.statfilters import (
    StatFilterReading,
    lag1_autocorr,
    abs_return_acf,
    rolling_moment,
    lee_mykland_test,
    bipower_variation,
    detrended_fluctuation_hurst,
    variance_ratio,
    compute_stat_filters,
)

# Phase 4 — calendar tier scoring, COT signals, macro / risk score
from quant_rabbit.analysis.calendar_tier import (
    event_tier,
    build_pair_windows as build_tiered_pair_windows,
)
from quant_rabbit.analysis.cot_signals import (
    cot_net_pct,
    cot_z_score,
    cot_week_delta,
    cot_commercial_extreme,
)
from quant_rabbit.analysis.macro import (
    MacroReading,
    FredClient,
    build_macro_reading,
)

__all__ = [
    "Candle",
    "fetch_candles",
    "fetch_candles_via_client",
    "IndicatorSet",
    "compute_indicators",
    "ChartView",
    "PairChart",
    "build_pair_chart",
    # Phase 1
    "NormalizedValue",
    "rolling_z",
    "rolling_percentile_rank",
    "normalize_indicator",
    "RegimeReading",
    "classify_regime",
    "FamilyScores",
    "compute_family_scores",
    # Phase 2 SMC
    "LiquiditySweep",
    "BreakerBlock",
    "MitigationBlock",
    "InversionFVG",
    "DisplacementCandle",
    "DealingRange",
    "SMCReading",
    "detect_sweeps",
    "detect_breakers",
    "detect_mitigations",
    "detect_inversion_fvgs",
    "detect_displacement",
    "compute_dealing_range",
    "compute_premium_discount",
    "analyze_smc",
    # Phase 2 sessions
    "SessionTag",
    "SessionMarker",
    "SessionContext",
    "tag_bar",
    "build_session_context",
    "jp_holiday_calendar_for",
    "ny_midnight_open",
    # Phase 3 TPO
    "TPOBracket",
    "TPOProfile",
    "build_tpo_profile",
    "naked_pocs",
    # Phase 3 statfilters
    "StatFilterReading",
    "lag1_autocorr",
    "abs_return_acf",
    "rolling_moment",
    "lee_mykland_test",
    "bipower_variation",
    "detrended_fluctuation_hurst",
    "variance_ratio",
    "compute_stat_filters",
    # Phase 4 calendar tier
    "event_tier",
    "build_tiered_pair_windows",
    # Phase 4 COT signals
    "cot_net_pct",
    "cot_z_score",
    "cot_week_delta",
    "cot_commercial_extreme",
    # Phase 4 macro
    "MacroReading",
    "FredClient",
    "build_macro_reading",
]
