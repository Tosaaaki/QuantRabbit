# Stop‑Run Reversal Worker
- Detect outsized candle with a long wick (likely stop‑run), then wait for failure to extend.
- Enter reversal on break of the trigger candle's opposite extreme.
- Parameters: `wick_ratio`, `min_range_mult`, `confirm_bars`.
