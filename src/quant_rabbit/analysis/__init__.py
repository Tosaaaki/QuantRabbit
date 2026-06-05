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

from importlib import import_module
from typing import Any

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

_EXPORTS: dict[str, tuple[str, str]] = {
    "Candle": ("quant_rabbit.analysis.candles", "Candle"),
    "fetch_candles": ("quant_rabbit.analysis.candles", "fetch_candles"),
    "fetch_candles_via_client": ("quant_rabbit.analysis.candles", "fetch_candles_via_client"),
    "IndicatorSet": ("quant_rabbit.analysis.indicators", "IndicatorSet"),
    "compute_indicators": ("quant_rabbit.analysis.indicators", "compute_indicators"),
    "ChartView": ("quant_rabbit.analysis.chart_reader", "ChartView"),
    "PairChart": ("quant_rabbit.analysis.chart_reader", "PairChart"),
    "build_pair_chart": ("quant_rabbit.analysis.chart_reader", "build_pair_chart"),
    "NormalizedValue": ("quant_rabbit.analysis.normalize", "NormalizedValue"),
    "rolling_z": ("quant_rabbit.analysis.normalize", "rolling_z"),
    "rolling_percentile_rank": ("quant_rabbit.analysis.normalize", "rolling_percentile_rank"),
    "normalize_indicator": ("quant_rabbit.analysis.normalize", "normalize_indicator"),
    "RegimeReading": ("quant_rabbit.analysis.regime", "RegimeReading"),
    "classify_regime": ("quant_rabbit.analysis.regime", "classify_regime"),
    "FamilyScores": ("quant_rabbit.analysis.families", "FamilyScores"),
    "compute_family_scores": ("quant_rabbit.analysis.families", "compute_family_scores"),
    "LiquiditySweep": ("quant_rabbit.analysis.smc", "LiquiditySweep"),
    "BreakerBlock": ("quant_rabbit.analysis.smc", "BreakerBlock"),
    "MitigationBlock": ("quant_rabbit.analysis.smc", "MitigationBlock"),
    "InversionFVG": ("quant_rabbit.analysis.smc", "InversionFVG"),
    "DisplacementCandle": ("quant_rabbit.analysis.smc", "DisplacementCandle"),
    "DealingRange": ("quant_rabbit.analysis.smc", "DealingRange"),
    "SMCReading": ("quant_rabbit.analysis.smc", "SMCReading"),
    "detect_sweeps": ("quant_rabbit.analysis.smc", "detect_sweeps"),
    "detect_breakers": ("quant_rabbit.analysis.smc", "detect_breakers"),
    "detect_mitigations": ("quant_rabbit.analysis.smc", "detect_mitigations"),
    "detect_inversion_fvgs": ("quant_rabbit.analysis.smc", "detect_inversion_fvgs"),
    "detect_displacement": ("quant_rabbit.analysis.smc", "detect_displacement"),
    "compute_dealing_range": ("quant_rabbit.analysis.smc", "compute_dealing_range"),
    "compute_premium_discount": ("quant_rabbit.analysis.smc", "compute_premium_discount"),
    "analyze_smc": ("quant_rabbit.analysis.smc", "analyze_smc"),
    "SessionTag": ("quant_rabbit.analysis.sessions", "SessionTag"),
    "SessionMarker": ("quant_rabbit.analysis.sessions", "SessionMarker"),
    "SessionContext": ("quant_rabbit.analysis.sessions", "SessionContext"),
    "tag_bar": ("quant_rabbit.analysis.sessions", "tag_bar"),
    "build_session_context": ("quant_rabbit.analysis.sessions", "build_session_context"),
    "jp_holiday_calendar_for": ("quant_rabbit.analysis.sessions", "jp_holiday_calendar_for"),
    "ny_midnight_open": ("quant_rabbit.analysis.sessions", "ny_midnight_open"),
    "TPOBracket": ("quant_rabbit.analysis.profile", "TPOBracket"),
    "TPOProfile": ("quant_rabbit.analysis.profile", "TPOProfile"),
    "build_tpo_profile": ("quant_rabbit.analysis.profile", "build_tpo_profile"),
    "naked_pocs": ("quant_rabbit.analysis.profile", "naked_pocs"),
    "StatFilterReading": ("quant_rabbit.analysis.statfilters", "StatFilterReading"),
    "lag1_autocorr": ("quant_rabbit.analysis.statfilters", "lag1_autocorr"),
    "abs_return_acf": ("quant_rabbit.analysis.statfilters", "abs_return_acf"),
    "rolling_moment": ("quant_rabbit.analysis.statfilters", "rolling_moment"),
    "lee_mykland_test": ("quant_rabbit.analysis.statfilters", "lee_mykland_test"),
    "bipower_variation": ("quant_rabbit.analysis.statfilters", "bipower_variation"),
    "detrended_fluctuation_hurst": ("quant_rabbit.analysis.statfilters", "detrended_fluctuation_hurst"),
    "variance_ratio": ("quant_rabbit.analysis.statfilters", "variance_ratio"),
    "compute_stat_filters": ("quant_rabbit.analysis.statfilters", "compute_stat_filters"),
    "event_tier": ("quant_rabbit.analysis.calendar_tier", "event_tier"),
    "build_tiered_pair_windows": ("quant_rabbit.analysis.calendar_tier", "build_pair_windows"),
    "cot_net_pct": ("quant_rabbit.analysis.cot_signals", "cot_net_pct"),
    "cot_z_score": ("quant_rabbit.analysis.cot_signals", "cot_z_score"),
    "cot_week_delta": ("quant_rabbit.analysis.cot_signals", "cot_week_delta"),
    "cot_commercial_extreme": ("quant_rabbit.analysis.cot_signals", "cot_commercial_extreme"),
    "MacroReading": ("quant_rabbit.analysis.macro", "MacroReading"),
    "FredClient": ("quant_rabbit.analysis.macro", "FredClient"),
    "build_macro_reading": ("quant_rabbit.analysis.macro", "build_macro_reading"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
