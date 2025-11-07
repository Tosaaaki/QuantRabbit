# Known Issues (2025-11-07)

## 1. Micro pocket exits at near-zero P/L
- **Symptom**: `BB_RSI` entries (micro_core) are closed within seconds at -0.2~-0.5 pips (e.g., trade ID 3549 @ 05:55Z).
- **Cause**: `scalp_exit` / micro exit policy has `allow_negative_exit=True` via `scalp_lock_release` and very low timeouts. Even small pullbacks trigger `MARKET_ORDER_TRADE_CLOSE`.
- **Next steps**:
  - Relax micro exit policy (`_be_profile_for('micro')`, `scalp_exit` thresholds) so the trade is allowed to develop.
  - Tighten micro entry filters (RSI extremes, spread gate) to reduce false signals.

## 2. Macro planning blocked by stale snapshot
- **Symptom**: `[MACRO] snapshot stale (age > 900s)` repeatedly logged, macro_core emits no entries.
- **Cause**: `fixtures/macro_snapshots/latest.json` is not refreshed frequently in VM; macro gate keeps `range_mode=TRUE`.
- **Next steps**:
  - Add cron/systemd timer to run `analysis/macro_snapshot_builder.py` every ~10 minutes or trigger refresh when age>600s.

## 3. Scalp strategy availability throttled by long cooldowns
- **Symptom**: `[IMP-RETEST-S5] cooling down for 1800s` etc, leaving only one strategy active.
- **Cause**: StageTracker pocket-level cooldowns + strategy cooldown default (360s) were too long.
- **Next steps**:
  - Confirm `SCALP_LOSS_COOLDOWN_SEC` env override applied (150s) and monitor `strategy_cooldown` table; adjust per strategy if needed.

## 4. Pullback runner still fragile to instant reversals
- **Symptom**: Even after widening SL, entry still occurs exactly at reversal points, causing quick SL hits.
- **Cause**: Z-score thresholds and trend alignment still allow entries against immediate momentum; no passive price/limit order.
- **Next steps**:
  - Increase FAST_Z_MIN / tighten trend gate; consider limit order or require price confirmation before sending market order.
