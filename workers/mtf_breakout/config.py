"""Default configuration for the multi-timeframe breakout worker."""

from __future__ import annotations

ENV_PREFIX = "MTF_BREAKOUT"

DEFAULT_CONFIG: dict[str, object] = {
    "id": "mtf_breakout_m5_h1",
    # Symbols are provided by the caller. In QR ops this maps to USD_JPY only,
    # but keeping the list for future multi-instrument support.
    "universe": [],
    "timeframe_entry": "M5",
    "timeframe_trend": "H1",
    # M5 candles used to define the breakout box.
    "breakout_lookback": 20,
    # Minimum bars since the local extreme to qualify as a pullback.
    "pullback_min_bars": 2,
    # Avoid re-entering too soon after signalling.
    "cooldown_bars": 3,
    # Minimum number of H1 bars required before the worker starts evaluating signals.
    "trend_min_bars": 48,
    # Minimum combined trend strength (0..1) before triggering.
    "edge_threshold": 0.25,
    # Safety limits for concurrent positions when the worker is wired to live orders.
    "max_concurrent": 3,
    # Budget in basis points when translating to order size; caller can override.
    "budget_bps": 25,
    # Start in dry-run mode. Flip to True once the signal output looks sensible.
    "place_orders": False,
    # Optional exit hints; currently informational only because the core ExitManager
    # does not expose attach hooks. Left here for forward compatibility.
    "exit": {
        "stop_atr": 1.8,
        "tp_atr": 3.0,
        "trail_mult": None,
    },
    # Lightweight data-quality filters.
    "filters": {
        "min_vol_rank_pct": 30,
        "max_spread_bp": 5,
    },
}
