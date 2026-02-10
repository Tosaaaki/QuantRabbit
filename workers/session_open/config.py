ENV_PREFIX = "SESSION_OPEN"

DEFAULT_CONFIG = {
    "id": "session_open_breakout",
    "universe": [],                 # e.g. ["BTCUSDT", "ETHUSDT"]
    "timeframe_entry": "1m",
    "sessions": [                   # local session windows (HH:MM, tz is IANA string)
        {"tz": "UTC", "start": "07:00", "build_minutes": 15, "hold_minutes": 120},
        {"tz": "UTC", "start": "12:30", "build_minutes": 15, "hold_minutes": 120},  # e.g. NY open
    ],
    "pad_bp": 2.0,                  # breakout padding in bps (avoid micro fakeouts)
    "cooldown_bars": 3,             # M bars to wait after an entry
    "edge_threshold": 0.0,          # optional: accept-all once breakout happens
    "place_orders": False,          # dry-run first; set True after validation
    "exit": {"stop_atr": 1.5, "tp_atr": 2.4, "trail_mult": None, "breakeven_mult": 0.6},
    "filters": {"max_spread_bp": 8, "min_bars": 60, "min_atr": 0.015}
}
