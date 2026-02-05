DEFAULT_CONFIG = {
    "id": "stop_run_reversal",
    "universe": [],
    "timeframe": "5m",
    "wick_ratio": 0.6,         # upper/lower wick proportion threshold
    "min_range_mult": 1.8,     # candle range >= k * median(range, N)
    "confirm_bars": 2,         # failure to extend within M bars
    "atr_len": 14,
    "max_spread_pips": 0.7,    # strict spread gate (pips)
    "max_spread_bp": 0.0,      # optional spread gate (bps)
    "session_sweep_required": True,
    "session_sweep_pips": 0.4,  # required sweep beyond prior session high/low
    "sessions": [
        {"name": "Tokyo", "tz": "Asia/Tokyo", "start": "09:00", "duration_minutes": 540},
        {"name": "London", "tz": "Europe/London", "start": "08:00", "duration_minutes": 540},
        {"name": "NewYork", "tz": "America/New_York", "start": "08:00", "duration_minutes": 540},
    ],
    "exit": {"stop_atr": 1.2, "tp_atr": 2.0},
    "place_orders": False,
    "budget_bps": 25
}
