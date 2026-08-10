# MONTHLY_3X_GROWTH_ENGINE_V1 verdict

## 結論

`WORKING_PROFIT_CORE_FOUND__MONTHLY_3X_NOT_YET_PROVEN`

「全システムが使えない」は誤りです。修正済み金融正本でも64日系VALIDATIONは101件、費用後Net `+11,706.0523 JPY`、PF `1.4469` です。月3倍へ届かない主因はedgeがゼロだからではなく、利益の出るvehicleが少なく、既存AIが勝ち取引をSKIPし、利益vehicleと損失の大きいmarket-close vehicleを同じ集計へ混ぜていたことです。

## できる理由

1. 費用後プラスの土台が実在します。0から利益を発明する課題ではありません。
2. ALL_TRADESを残してprice-action Ridgeをサイズ配分へ使うと、20万円・cohort証拠金上限75%の64日系VALIDATIONでbaseline比 `+2,493.46 JPY` の点増分が出ました。LCBは `-19.27 JPY/件` なので、配分レバーは動いたがまだ安定化が必要です。
3. EUR/USD SHORTは64日系TRAIN 50件 `+7,500.0122 JPY`、VALIDATION 5件 `+2,173.1265 JPY`。VALIDATIONは5勝0敗、bootstrap LCB `+171.67134 JPY/件` でした。
4. 既存の正確な `EUR_USD SHORT BREAKOUT_FAILURE LIMIT + attached TP` はS5 bid/ask replayで4件4勝、Net `+3,255.0938 JPY`、expectancy `+813.7734 JPY/件` です。これはsample不足・fill時刻lag未解消ですが、利益vehicleの実在を示します。

## 何が間違っていたか

- 予測器をサイズ・順位ではなくSKIPに使い、ALL_TRADESの勝者を捨てていました。
- `BREAKOUT_FAILURE`全体で、attached-TP利益とMARKET_ORDER_TRADE_CLOSE損失を混ぜ、利益vehicle固有の再現性を消していました。
- missing margin/pathを「edgeなし」と混同していました。
- genericな5戦略族と固定時間exitを増やしても、TRAIN接続plateauは0でした。
- 固定BE、1ATR bracket、ATR trail、SMA劣化、structure breakは保守的M5 first-touchで初期SLへ偏り、固定時間exitを改善しませんでした。出口の種類ではなく、entry/vehicle/TP geometryの一致が必要です。

## 月3倍への距離

20万円から60万円は利益40万円です。

- 月200取引: `2,000 JPY/件`
- 月400取引: `1,000 JPY/件`
- 月800取引: `500 JPY/件`

EUR/USD SHORT VALIDATIONの観測expectancy `434.6253 JPY/件` を固定円で使うと、200取引ではedgeまたはsizeが `4.60x`、400取引では `2.30x`、800取引では `1.15x` 必要です。これは保証値ではなく、機会数・edge・資本効率を別々に増やす設計値です。

## 採用する成長設計

- baselineは常時TRADE候補として残す。
- hard safetyだけがblockし、technical/MLはrankとsizeを変える。
- 最優先vehicleを `EUR_USD SHORT BREAKOUT_FAILURE LIMIT + attached technical TP` に固定する。
- generic indicator entryを増やさず、既存の正確なfailed-break/retest/limit geometryをdecision-timeで再生成する。
- MARKET_ORDER_TRADE_CLOSE損失をTP vehicleの証拠へ混ぜない。出口はvehicle別に評価する。
- exact vehicleの機会を増やした後、同じ形を他pairへ移す。pairを増やす前にEUR/USDでfill・TP・financing・margin lineageを揃える。
- price-action RidgeはSKIPに使わず、0.5〜1.5の配分modifierとして再検証する。

## 境界

月3倍はまだ証明されていません。今回証明したのは、費用後プラスの核、動作する配分レバー、正のEUR/USD SHORT vehicle、そして次に増やすべき正確な形です。holdout、live、Paper、broker order、deployは未使用です。
