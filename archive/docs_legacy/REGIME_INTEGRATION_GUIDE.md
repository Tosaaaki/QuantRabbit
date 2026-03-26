# レジーム判定・アダプティブSL/TP 統合ガイド (2026-03-17)

## 概要

`analysis/market_regime.py` と `analysis/adaptive_sl_tp.py` を各 strategy worker に統合する手順。
AGENTS.md の「共通レイヤは強制的に戦略を選別しない」に従い、各 worker が strategy-local で呼び出す。

## 統合パターン（全 worker 共通）

```python
# --- worker.py の先頭に追加 ---
from analysis.market_regime import classify_regime, MarketRegime
from analysis.adaptive_sl_tp import compute_adaptive_sl_tp, map_strategy_tag_to_type

# --- エントリー判定の直前に追加 ---
regime = classify_regime()  # factor_cache から自動取得

# 1. レジームによるエントリー可否判定
if not regime.is_tradeable():
    logging.info("[%s] entry blocked: regime=%s", strategy_tag, regime.regime.value)
    continue  # or return

# 2. 方向の適合性チェック（トレンド逆張り防止）
from analysis.market_regime import should_enter
allowed, reason = should_enter(regime, strategy_type, side)
if not allowed:
    logging.info("[%s] entry blocked: %s", strategy_tag, reason)
    continue

# 3. アダプティブ SL/TP の取得
strategy_type = map_strategy_tag_to_type(strategy_tag)
adaptive = compute_adaptive_sl_tp(
    regime=regime,
    atr_pips=atr_pips,
    spread_pips=spread_pips,
    strategy_type=strategy_type,
)
if adaptive is None:
    logging.info("[%s] entry blocked by adaptive_sl_tp: regime=%s", strategy_tag, regime.regime.value)
    continue

sl_pips = adaptive["sl_pips"]
tp_pips = adaptive["tp_pips"]

# 4. entry_thesis にレジーム情報を記録（監査用）
entry_thesis["regime"] = regime.regime.value
entry_thesis["regime_confidence"] = regime.confidence
entry_thesis["adaptive_sl_pips"] = sl_pips
entry_thesis["adaptive_tp_pips"] = tp_pips
entry_thesis["adaptive_rr"] = adaptive["rr"]
```

## 各 worker の対応表

| Worker | strategy_type | ファイル |
|--------|---------------|----------|
| scalp_extrema_reversal | scalp_reversal | workers/scalp_extrema_reversal/worker.py |
| scalp_wick_reversal_blend (DroughtRevert/PrecisionLowVol/VwapRevertS/WickReversalBlend) | scalp_reversal | workers/scalp_wick_reversal_blend/worker.py |
| scalp_m1scalper | scalp_breakout | workers/scalp_m1scalper/worker.py |
| scalp_ping_5s (B/C/D/Flow) | ping_5s | workers/scalp_ping_5s/worker.py |
| micro_multistrat (MomentumBurst) | momentum | workers/micro_multistrat/worker.py |
| micro_multistrat (MicroLevelReactor) | micro_level | workers/micro_levelreactor/worker.py |
| micro_multistrat (MicroTrendRetest) | trend_follow | workers/micro_multistrat/worker.py |
| scalp_rangefader | range_fade | workers/scalp_rangefader/worker.py |
| session_open | scalp_breakout | workers/session_open/worker.py |

## 注意事項

- AGENTS.md: 各 worker が自分で `classify_regime()` を呼ぶ。shared gate には追加しない。
- AGENTS.md: entry_thesis に regime 情報を残す（監査トレーサビリティ）。
- AGENTS.md: fail-open 方針。regime == UNKNOWN の場合はエントリーを通す。
- `_normalize_sl_tp_rr` の SL下限 (SL_FLOOR_PIPS=2.0) はバックストップとして機能する。
  worker 側で adaptive SL/TP を使わなくても、最低限 2.0pip は保証される。
