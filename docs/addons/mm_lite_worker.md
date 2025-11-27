# Market‑Making Lite Worker
- Symmetric quotes around mid with spread = `base_spread_bp` + `spread_k_atr` * ATR%.
- Inventory bounded by `inventory_r` (per day); optional event kill switch.
- Requires a broker with limit order / cancel support.
