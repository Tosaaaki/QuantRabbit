# QuantRabbit Addons (Workers + Allocator)

Included components:
- `workers/session_open`: Initial-range breakout around session opens.
- `workers/vol_squeeze`: Squeeze detection with Keltner breakout.
- `workers/stop_run_reversal`: Stop-run wick reversal.
- `workers/mm_lite`: Simple inventory-constrained market maker.
- `allocator/bandit.py`: Thompson-sampling based budget allocator.

## Integration
- Add imports in your `workers/__init__.py` to expose new classes if your system uses discovery by import.
- Instantiate with your `Broker` and `DataFeed` implementations.
- Start with `place_orders=False` to dry-run, then enable live orders.

## Notes
- Exit/TP/SL are attached via `execution.exit_manager.ExitManager` if present in your repo.
- Time windows and thresholds are intentionally conservative; tune after replay/backtest.
