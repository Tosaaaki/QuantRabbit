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
- 本リポの `execution.exit_manager` は互換スタブ（自動EXITなし）。各ワーカーで SL/TP / exit_worker を持つ前提で統合する。
- 互換的に attach() を持つ ExitManager が必要な場合は `workers/common/exit_adapter.build_exit_manager()` を経由する。`EXIT_MANAGER_DISABLED=1` が既定で、設定時のみ attach を試みる。
- Time windows and thresholds are intentionally conservative; tune after replay/backtest.
