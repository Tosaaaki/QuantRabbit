DEFAULT_CONFIG = {
    "id": "mm_lite",
    "universe": [],
    "tick_size": 0.0,              # if 0.0 -> infer from symbol meta or skip tick rounding
    "base_spread_bp": 2.0,         # minimum quoted spread (bps)
    "spread_k_atr": 0.6,           # add-on spread as k * ATR% (approx by (ATR/price)*1e4)
    "atr_len": 14,
    "inventory_r": 1.0,            # +/- 1R per day inventory cap
    "rebalance_to_vwap": True,
    "disable_on_event": True,      # requires external event flag provider (optional hook)
    "size_bps": 10,                # per-quote notional
    "place_orders": False,
    "refresh_sec": 15.0,           # minimum seconds between cancel/replace
    "replace_bp": 0.4,             # replace only if quote moved by >= bps
    "max_quote_age_sec": 90.0      # force refresh after this many seconds
}
