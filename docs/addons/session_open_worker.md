# Session‑Open Breakout Worker
- Build the initial range during the first `build_minutes` after each configured session start.
- Trade a breakout of that range with a small padding in bps.
- Optional: flip to mean-reversion if the breakout fails quickly (`fail_window_bars`, `fail_reentry_bp`).
- Start with `place_orders=false`, verify signals, then enable real orders.
