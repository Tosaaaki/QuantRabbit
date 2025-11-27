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
    "place_orders": False
}
