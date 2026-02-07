ENV_PREFIX = "STOP_RUN_REVERSAL"

DEFAULT_CONFIG = {
    "id": "stop_run_reversal",
    "universe": [],
    "timeframe": "5m",
    "wick_ratio": 0.6,         # upper/lower wick proportion threshold
    "min_range_mult": 1.8,     # candle range >= k * median(range, N)
    "confirm_bars": 2,         # failure to extend within M bars
    "exit": {"stop_atr": 1.2, "tp_atr": 2.0},
    "place_orders": False,
    "budget_bps": 25
}
