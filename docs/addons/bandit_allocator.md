# Bandit Allocator (Thompson Sampling)
- Inputs per worker: `wins`, `trades`, `ev`, `hit`, `sharpe`, `max_dd`.
- Samples Beta for success rate and blends with a quality score.
- Outputs budgets in bps, honoring `cap_bps` and `floor_bps`.
