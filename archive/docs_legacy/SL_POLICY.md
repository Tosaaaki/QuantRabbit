# SL Policy Map

SL（損切り）は「どこで決まるか」が複数レイヤに分かれています。  
このドキュメントは、USD/JPY 本番運用での優先順位と確認ポイントを整理したものです。

## 1. 生成順序（実運用の実体）
1. Worker が `entry_thesis.sl_pips` / `hard_stop_pips` / `tp_pips` を生成
2. `execution/order_manager.py` でエントリー前に再計算
   - dynamic SL (`ORDER_DYNAMIC_SL_*`)
   - min RR 調整 (`ORDER_MIN_RR_*`)
   - TP cap (`ORDER_TP_CAP_*`)
   - entry hard SL (`ORDER_ENTRY_HARD_STOP_PIPS*`)
3. OANDA 送信時の `stopLossOnFill` は `execution/order_manager.py` の
   `_entry_sl_disabled_for_strategy()` と `_allow_stop_loss_on_fill()` で最終判定
   - baseline: `ORDER_FIXED_SL_MODE`
   - strategy override: `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>`
   - family override: `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_[B|C|D]`
   - hard-stop disable: `disable_entry_hard_stop` / `ORDER_DISABLE_ENTRY_HARD_STOP_*`
4. fill 後の保護更新（`on_fill_protection`）
5. それでも SL が無い建玉は `order_manager.py` 側での保護更新処理が優先

## 1.5 固定SL baseline（全体既定）
- `ORDER_FIXED_SL_MODE=1` で全 pocket の baseline は ON。
- `ORDER_FIXED_SL_MODE=0` で全 pocket の baseline は OFF。
- 未設定は OFF 扱い（運用では `1`/`0` を明示する）。

## 1.6 戦略単位 override（baseline上書き）
- `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>=1/0` で strategy 単位に
  broker SL attach を明示上書きできる。
- ping family は互換のため `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B/C/D`
  でも上書き可能。
- 旧キー
  `ORDER_DISABLE_STOP_LOSS_*` / `ORDER_ENABLE_STOP_LOSS_*` /
  `ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_*`
  は運用対象外。

## 2. 重要な分岐（見落としやすい点）
- `ORDER_ENTRY_HARD_STOP_PIPS*` を設定しても、entry SL が無効の pocket では
  `stopLossOnFill` は付与されない（virtual SL としてのみ扱われる）。
- `quant-hard-stop-backfill.service` は現在の運用構成から除外。  
  SL 無し建玉の後付け処理はワーカー依存の緊急運用仕様ではなく、`order_manager.py` の通常フローを起点に再確認する。

## 2.5 SL関連キー（現行実装での役割）
- `ORDER_FIXED_SL_MODE`: broker SL attach の baseline ON/OFF
- `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>`: strategy 単位 attach override
- `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_[B|C|D]`: ping family attach override
- `ORDER_ENTRY_HARD_STOP_PIPS*`, `ORDER_ENTRY_MAX_SL_PIPS*`: エントリー側で使う hard stop 距離の上限/基準（仮想SL系）
- `ORDER_ENTRY_LOSS_CAP_*`, `ORDER_ENTRY_LOSS_CAP_BUFFER_PIPS`: 1約定あたりの損失上限でサイズを間接制御
- `ORDER_DYNAMIC_SL_*`: `trade`/`tick`情報から virtual SLを再計算するための入力
- `ORDER_MIN_RR_*`: エントリーサイズに対する最小RR監査の閾値
- `ORDER_TP_CAP_*`: TP上限（間接的にSL/TPの実効範囲に影響）
- `ORDER_ROLLOVER_SL_STRIP_*`: ロールオーバー時間帯で既存SLの後処理を行う機能
- `ORDER_DISABLE_ENTRY_HARD_STOP_TAGS`: 戦略単位で hard stop 無効化（実データ送信の ON/OFF ではない）

## 3. 現行の推奨（精度優先 + 通常SL）
- `ops/env/entry_precision_hardening.env` を適用
  - 鮮度/密度ゲート強化（factor stale と microstructure）
  - `ORDER_FIXED_SL_MODE=1`
- 反映コマンド:
  - `scripts/vm_apply_entry_precision_hardening.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t`

## 4. 反映後の確認SQL（VM）
`orders.db` で `submit_attempt` の OANDA payload に `stopLossOnFill` が入っているか確認:

```sql
SELECT ts,pocket,status,
       json_extract(request_json,'$.oanda.order.stopLossOnFill.price') AS sl_on_fill,
       json_extract(request_json,'$.oanda.order.takeProfitOnFill.price') AS tp_on_fill
FROM orders
WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S','now','-2 hours')
  AND status='submit_attempt'
  AND pocket IN ('micro','scalp','macro')
ORDER BY ts DESC
LIMIT 30;
```

## 5. 障害時の切り分け
- SL が付かない:
  - `ops/env/quant-v2-runtime.env` で baseline `ORDER_FIXED_SL_MODE` を確認
  - `ops/env/quant-order-manager.env` の
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>` /
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_*` を確認
  - `systemctl show -p Environment quant-order-manager.service`
  - `orders.db` の `submit_attempt` payload に `stopLossOnFill` があるか
- SL が広すぎる:
  - `orders.db` での実送信 `stopLossOnFill` 価格と `TP` の乖離
  - `entry_precision_hardening` 前提の `ORDER_ENTRY_HARD_STOP_PIPS*` / `ORDER_DYNAMIC_SL_*` 設定
