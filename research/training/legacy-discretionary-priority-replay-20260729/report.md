# 旧・裁量戦略ワーカー高速リプレイ

- 結論: **保護付きでは儲かっていない（no-SL対照のみ黒字）**
- 判定: **採用なし**
- 評価済み/試行済み: **37 / 82**
- 残り: **45**
- authority: `NONE` / live: `false`

## 優先family結果

| family | entry | best protected exit | AI | Net JPY | PF | Expectancy | DD | trades | 判定 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| momentum_burst | 367 | volatility_trail | OFF | -1923.88 | 0.760 | -5.24 | 2326.55 | 367 | 不採用 |
| scalp_macd_rsi_div | 1 | fixed_sl | OFF | 31.00 | n/a | 31.00 | 0.00 | 1 | 証拠不足 |
| scalp_macd_rsi_div_b | 5 | fixed_sl | OFF | 142.97 | n/a | 28.59 | 0.00 | 5 | 証拠不足 |
| scalp_ping_5s | 1716 | time_stop | OFF | -15432.32 | 0.426 | -8.99 | 15489.31 | 1716 | 不採用 |
| scalp_ping_5s_b | 26686 | time_stop | OFF | -205309.76 | 0.545 | -7.69 | 205910.54 | 26686 | 不採用 |
| scalp_ping_5s_c | 82641 | fixed_sl | ON | -744190.64 | 0.396 | -9.01 | 744231.64 | 82641 | 不採用 |
| scalp_ping_5s_d | 63272 | fixed_sl | ON | -579263.14 | 0.284 | -9.16 | 579346.14 | 63272 | 不採用 |
| scalp_ping_5s_flow | 68719 | fixed_sl | ON | -631634.81 | 0.281 | -9.19 | 631710.81 | 68719 | 不採用 |
| scalp_wick_reversal_blend | 1 | fixed_sl | ON | -3.00 | 0.000 | -3.00 | 3.00 | 1 | 証拠不足 |
| scalp_wick_reversal_pro | 0 | fixed_sl | OFF | 0.00 | n/a | n/a | 0.00 | 0 | 証拠不足 |

## 判定上の制約

- 2024–2026H1 corpusは既に反復利用済みのため、今回の5窓は `LINEAGE_UNSEEN_DIAGNOSTIC` であり、未使用holdoutとは表現しない。
- AI ONは外部modelを呼ばない凍結済み因果inventory rule。model call 0、execution AI cost 0円。モデル推論費用を含む実AIの経済性は別途未確定。
- spreadは記録bid/ask、slippageは全fill、financingは保有時間按分、期末open positionは実行可能sideでMTM。
- financingは比較用の保守的固定debit 10円/1万通貨/日。実brokerのside別swap実績ではない。
- wick blendのprojectionはarchive内の外部予測依存を除外し、past-only gateを通過したsignalに対するneutral passthrough。完全忠実性ではなくadapter診断。

## 安全性

- Paperは停止していない。broker/order mutationは実行していない。
- archive codeはread-only import、variantはprocess隔離。
