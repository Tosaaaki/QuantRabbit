# London / NY Momentum & VWAP Strategy Expansion

_Date: 2025-11-11_

## 1. Data Recap (2025-11-10)
- Trades: 25 (micro 22, scalp 3). Macro pocket = 0 entries because snapshot stale.
- London session (07:00–12:00 UTC) executed 7 trades (all BB_RSI), net −¥314. NY序盤は4件のみ。
- Notable untraded swings (M1 candles, 20min windows, move >= 0.15 JPY):
  | Window UTC | Move | Direction | Context |
  |------------|------|-----------|---------|
  | 07:16–07:35 | 15.2 pips | down | London fix sellers。macro閉鎖＋micro cooldownで参加ゼロ。
  | 10:32–10:55 | 11.8 pips | up | NY先物open。News/VWAP系無し。
  | 13:05–13:30 | 9.4 pips  | down | 東京午後の調整。spread<1p でも stage tracker 冷却。

## 2. Strategy Requirements
### 2.1 London Momentum (LMO)
- **Time window**: 06:45–11:30 UTC (Asia close→London AM)。
- **Signal**: M5 momentum + H1 trend alignment。
  - `trend_bias = ema20_h1 - ema50_h1` > 0 → only long; <0 → only short。
  - M1 Z-score / slope > threshold; ATR(5m) >= 5p。
- **Entry**: MARKET using best bid/ask (existing helper)。TP=8p, SL=5p base (spread-aware clamp)。
- **Guards**: spread <= 1.4p, event window skip, StageTracker per direction with 120s cooldown。
- **Sizing**: use macro pocket if macro bias strong, otherwise micro pocket fallback (0.15–0.3 lot, scaling with ADX)。

### 2.2 NY VWAP Reversion (NVW)
- **Time window**: 12:30–20:00 UTC (pre-NY data→NY午前)。
- **Signal**: price deviation from rolling VWAP(90m) + RSI divergence。
  - Entry when `abs(price - vwap) >= 12p` AND RSI extremes (<=25 for long, >=75 for short)。
  - Confirm with order-book spread (<=1.6p) + tick density。
- **Exit**: TP=6p, SL=5p, trailing reduce if reversion >4p。
- **Sessions**: disable during scheduled events (CPI/NFP list) via MacroState events。

## 3. Implementation Plan
1. **Worker scaffolding**
   - Create `workers/london_momentum/` and `workers/ny_vwap/` with `config.py`, `worker.py`, `__init__.py` mirroring current scalp workers。
   - Provide env toggles `LONDON_MOMO_ENABLED`, `NY_VWAP_ENABLED` default False。
2. **Common helpers**
   - Extend `indicators/vwap_cache.py` (or new helper) to supply VWAP/M5 factors。
   - Add `session_gate.py` for time-window checks reused by both workers。
3. **Sizing / risk**
   - Hook into `execution/risk_guard.allowed_lot` with a new `risk_pct_override` per strategy (e.g., 0.015 for LMO, 0.012 for NVW)。
   - Write pocket affinity logic: London uses `macro` if bias strong else `micro`; NY VWAP uses `micro`.
4. **Integration**
   - Update `main.py` to start new workers when env flags true (similar to existing scalp workers list)。
   - Register strategies in `STRATEGIES` map if PocketPlan needs referencing later。
5. **Backtest / replay**
   - Extend `scripts/replay_workers.py` to accept `--worker london_momentum` / `ny_vwap`。
   - Provide sample commands with existing `tmp/ticks_*.jsonl`.

## 4. Macro Pocket Rebalance
- **Status (11/11 01:30 UTC)**: macro snapshot stale continues even after auto-refresh deploy; need to confirm new build stops `[MACRO] Snapshot stale` spam. If not, add watchdog/cron to rebuild every 5 min.
- **Weight strategy**: once macro resumes, cap weight based on rolling win rate: `weight_cap = clamp(0.18 + 0.04*(win_rate_7d-0.5), 0.18, 0.35)`; ensure PocketPlan respects this cap.
- **Lot scaling**: when `spread <= 1.0p` and `macro_bias >= 0.4`, allow up to 1.3x base lot; shrink to 0.8x if spread >1.3p or ATR <5p。
- **Monitoring**: add CLI helper to print snapshot `asof`, `macro_bias`, computed cap/lot multiplier for quick verification.

## 5. BB_RSI Cooldown Adjustments
- Current entry gating in `workers/micro_core/worker.py` uses StageTracker with 120s cooldown + global spread gate。
- Observed issue: after 2連敗, StageTracker blocks even when ATR/spread favorable。
- Plan:
  1. Introduce volatility-aware cooldown: `cooldown = base * max(0.5, min(1.5, atr/6))`。
  2. Add dynamic gate to skip entries only if RSI slope hasn’t reversed (avoid unnecessary wait when signal resets)。
  3. Log Stage resets with reason for easier debugging。
  4. Implementation steps:
     - Extend `StageTracker.set_cooldown` to accept `override_seconds` so worker can pass ATR-based value。
     - Cache last RSI slope + bounce timestamp; reset cooldown early when slope sign flips and spread <=1.1p。
     - Emit `METRIC bb_rsi_cooldown_skips` for monitoring。

## 6. Next Steps
1. Implement helpers + scaffolding for LMO/NVW workers。
2. Wire new env flags + session gates。
3. Apply BB_RSI cooldown changes and validate via replay。
4. Monitor macro pocket after restart to confirm `weight_macro` >0.18 entries resume。
