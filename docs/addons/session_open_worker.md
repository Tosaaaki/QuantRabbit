# Session‑Open Breakout Worker
- Build the initial range during the first `build_minutes` after each configured session start.
- Trade a breakout of that range with a small padding in bps.
- Start with `place_orders=false`, verify signals, then enable real orders.
