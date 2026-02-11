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
3. OANDA 送信時に `stopLossOnFill` を付与するか判定
   - `stop_loss_disabled_for_pocket()` と `_allow_stop_loss_on_fill()` の両方を満たす必要あり
4. fill 後の保護更新（`on_fill_protection`）
5. それでも SL が無い建玉は `quant-hard-stop-backfill.service` が後付け

## 2. 重要な分岐（見落としやすい点）
- `EXIT_NO_NEGATIVE_CLOSE=1` の場合、デフォルトでは broker SL は抑制される。  
  ただし `ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_{POCKET}=1` で pocket 単位で許可可能。
- `ORDER_DISABLE_STOP_LOSS` はデフォルト true 扱い。  
  本番では `ORDER_ENABLE_STOP_LOSS_{POCKET}=1` を明示して有効化する。
- `quant-hard-stop-backfill.service` は tighten-only（既存SLを広げない）。  
  ただし SL 無し建玉には fallback 値を付けるため、レンジ設定が過大だと実効RRを悪化させる。

## 3. 現行の推奨（精度優先 + 通常SL）
- `ops/env/entry_precision_hardening.env` を適用
  - 鮮度/密度ゲート強化（factor stale と microstructure）
  - `ORDER_ENABLE_STOP_LOSS_{MICRO,SCALP,MACRO}=1`
  - `ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_{MICRO,SCALP,MACRO}=1`
- 反映コマンド:
  - `scripts/vm_apply_entry_precision_hardening.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t`

## 4. 反映後の確認SQL（VM）
`orders.db` で `submit_attempt` の OANDA payload に `stopLossOnFill` が入っているか確認:

```sql
SELECT ts,pocket,status,
       json_extract(request_payload,'$.oanda.order.stopLossOnFill.price') AS sl_on_fill,
       json_extract(request_payload,'$.oanda.order.takeProfitOnFill.price') AS tp_on_fill
FROM orders
WHERE ts >= datetime('now','-2 hours')
  AND status='submit_attempt'
  AND pocket IN ('micro','scalp','macro')
ORDER BY ts DESC
LIMIT 30;
```

## 5. 障害時の切り分け
- SL が付かない:
  - `/etc/quantrabbit.env` に pocket 別 `ORDER_ENABLE_STOP_LOSS_*` と `ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_*` があるか
  - `systemctl show -p Environment quantrabbit.service`
  - `orders.db` の `submit_attempt` payload に `stopLossOnFill` があるか
- SL が広すぎる:
  - `quant-hard-stop-backfill.service` の `--min/--max/--fallback`
  - `scripts.attach_hard_stop_sl --max-sl-to-tp-ratio` の有無
