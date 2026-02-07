ENV_PREFIX = "VOL_SQUEEZE"

DEFAULT_CONFIG = {
    "id": "vol_squeeze_breakout",
    "universe": [],
    "timeframe": "5m",
    "ema_len": 20,
    "atr_len": 14,
    "bb_len": 20,
    "squeeze_pctile": 0.2,      # current BB width <= 20% 分位（過去bb_len*5で算出）
    "keltner_mult": 1.5,        # Keltner = EMA +/− k*ATR
    "exit": {"stop_atr": 1.5, "tp_atr": 2.2, "breakeven_mult": 0.0},
    "ema_slope_min": 0.0,       # breakout direction must align with EMA slope (abs)
    "allow_long": True,
    "allow_short": True,
    "cooldown_bars": 2,
    "edge_threshold": 0.0,
    "place_orders": False,
    "budget_bps": 30
}
