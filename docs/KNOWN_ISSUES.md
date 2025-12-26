# Known Issues (2025-11-07)

## 1. Micro pocket exits at near-zero P/L（旧 common exit, 廃止済み）
- **Symptom (legacy)**: `BB_RSI` entries (旧 micro_core) were closed within seconds at -0.2~-0.5 pips (e.g., trade ID 3549 @ 05:55Z)。micro_core は廃止済みで、専用 micro_* EXIT に置き換え済み。
- **Cause**: 共通 EXIT ワーカー（scalp_exit/micro_exit）が `allow_negative_exit=True` かつ極端なタイムアウトで早期クローズしていたため。
- **Status**: 共通 EXIT ワーカーを廃止し、各ワーカー専用 EXIT に統一。

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
