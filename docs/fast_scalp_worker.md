# FastScalp Worker Plan

This note captures the design for the ultra–short term “FastScalp” worker that will
run alongside the existing 60 s logic loop. The worker consumes tick data in
~250 ms cadence, evaluates simple momentum / mean‑reversion heuristics, and
submits tiny scalp orders that target +1.0 pip (plus spread buffer) while leaving
wide emergency stops (≈30 pips). The component co‑exists with the current
`scalp` pocket and stays within OANDA API limits.

## High‑level Flow

1. **Tick feed** — Reuse the existing `market_data.tick_window` cache. The worker
   samples the most recent ticks (1 s / 4 s windows) to compute direction,
   velocity, and spread. No extra WebSocket connection is opened.
2. **Spread gate** — Every iteration the worker checks `spread_monitor.is_blocked()`
   and enforces an additional `FAST_SCALP_MAX_SPREAD` guard
   (default 0.35 pips, configurable via env). When the gate is active the worker
   idles and logs `[SCALP-TICK] skip reason=spread`.
3. **Signal evaluation** — The worker derives a compact signal:
   - Momentum slope between the latest mid price and the mean of N recent ticks.
   - Micro range (high/low over the past few seconds).
   - If `abs(momentum)` ≥ `entry_threshold` (≈0.6 pip) and range width ≥
     `range_floor` (≈0.8 pip) the worker proposes a directional trade.
   - When both long and short conditions are weak the worker stays flat.
4. **Risk & sizing** — Lot sizing uses the existing `risk_guard.allowed_lot`.
   - Shared context delivers the latest `account_equity`, `margin_available`,
     `margin_rate`, and last `weight_scalp`.
   - The worker clamps exposure per trade to `FAST_SCALP_MAX_LOT` (default 0.05
     lot) while enforcing a **minimum** of `FAST_SCALP_MIN_UNITS` (default
     10 k units ≒ 0.1 lot) and a single concurrent FastScalp position.
5. **Order path** — The worker submits orders through `execution.order_manager.market_order`
   with `pocket="scalp_fast"`, `client_order_id` prefix `qr-fast-`, and attaches
   `takeProfitOnFill` / `stopLossOnFill`.
   - `tp_pips = 1.0 + max(spread, 0.2)`
   - `sl_pips = 30.0`
   - Min unit guard: abort if computed units < 10 k.
6. **Rate limiting** — A dedicated limiter enforces:
   - `max_orders_per_minute = 24`
   - `min_order_spacing = 2.5 s`
   - Exponential backoff when an order fails (0.3 s, 0.9 s, 2.7 s).
   Trade intents violating the limit are skipped with a log entry.
7. **Position tracking** — The worker keeps an in‑memory registry keyed by
   trade id for orders it opened. Every `FAST_SCALP_SYNC_INTERVAL` (≈45 s) it
   reconciles against `PositionManager.get_open_positions()` to avoid drift.
   - If external trades remain under `pocket=scalp_fast`, the worker hands
     them to the standard `ExitManager` for consistency.
8. **Exit discipline** — Primary exit is the order TP. Additional safety nets:
   - Time stop: if position survives > 60 s without hitting TP and unrealised
     gain < 0.4 pip, the worker requests a market close.
   - Drawdown stop: if unrealised loss exceeds 5 pips before TP, instruct
     `ExitManager` to close (prevents deepening losses while SL is wide).
9. **Metrics & logging** — All logs carry `[SCALP-TICK]` prefix. Metrics tagged
   with `pocket=scalp_fast`, `strategy=FastScalp`. `logs/trades.db` stores
   `pocket=scalp_fast` via order manager tag. An additional counter records
   skipped opportunities because of spread, rate limiting, or cooldown.

## Shared State

- A lightweight `FastScalpState` singleton holds:
  ```
  weight_scalp: float
  account_equity: float
  margin_available: float
  margin_rate: float
  risk_pct_override: float
  focus_tag: str
  updated_at: datetime
  ```
- `logic_loop` updates the state each 60 s cycle after fetching account metrics
  and GPT weights.
- The worker snapshots the structure every iteration without expensive REST
  calls.

## Module & Code Changes

1. **New package** `workers/fast_scalp/`
   - `state.py` — shared dataclass with thread/async safe accessors.
   - `rate_limiter.py` — sliding window limiter.
   - `worker.py` — async loop implementing the behaviour above.
   - `signal.py` — helper functions for momentum/range calculations.
   - `config.py` — constants sourced from env (thresholds, cadence).
2. **`main.py`**
   - Instantiate shared state object and pass into both `logic_loop` and
     `fast_scalp_worker`.
   - Update GPT / account snapshot section to call
     `fast_scalp_state.update_from_main(...)`.
   - Start worker task under `supervised_runner("fast_scalp", fast_scalp_worker(...))`.
   - Tag log lines `[SCALP-MAIN]` for existing scalp signals.
3. **`execution` adjustments**
   - Extend `risk_guard` pocket maps to include `"scalp_fast"` and set DD cap
     (e.g. 0.02) and `POCKET_MAX_RATIOS["scalp_fast"] = 0.12`.
   - Update `stage_tracker` to allow additional pocket strings and use
     `sqlite3.connect(..., check_same_thread=False)` for multi-task access.
   - Permit `order_manager` to accept the new pocket value and emit logs with
     `pocket=scalp_fast`.
4. **Logging / metrics**
   - Add new metric keys (e.g. `fast_scalp_signal`, `fast_scalp_skip_reason`).
   - Ensure `plan_partial_reductions` treats `scalp_fast` as scalp class with
     tighter default thresholds when range mode toggles.

## Safeguards

- Worker honours `RiskGuard.can_trade("scalp")` and new `can_trade("scalp_fast")`.
- Spread guard + time‑of‑day suppression (JST 03:00–05:30) to avoid illiquid
  intervals.
- When drawdown guard trips (per pocket or global) the worker logs and idles
  until unlocked.

This design keeps the initial delivery scoped: one worker, one basic
momentum strategy, hard coded TP/SL, and shared risk infrastructure.
Iterative tuning (strategy ensembles, knapsack allocation, advanced exits)
can layer on later without reworking the skeleton.
