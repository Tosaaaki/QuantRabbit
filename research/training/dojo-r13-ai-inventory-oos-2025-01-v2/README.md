# r13 2025-01 OHLC AI在庫・multi-TF監督 v2

## 結論

2025-01は最終モデル検証ではなく、mechanism discovery / integration /
calibration monthとした。January内のsealed OOSを一度だけ評価した結果、
AI費用控除後に`net > 0`、`PF > 1`、DD非悪化、margin-call/ruin非増加を
BASE/STRESSの両方で満たしたのは、無介入を選んだ
`prev_day_extreme_fade`だけだった。

- `prev_day_extreme_fade`: BASE `+9,048.98 JPY`、STRESS
  `+7,262.22 JPY`。A/B/Cは同一で、guard-onlyの正解は`HOLD`だった。
  January短期ゲート上は`CONDITIONAL`。ACTIVE判定ではない。
- `mean_revert_24h`: actual CはBASEでA比`+2,891.63 JPY`、STRESSで
  `+4,896.00 JPY`、DDもそれぞれ`4.17%→2.73%`、
  `5.80%→3.44%`へ改善した。しかし費用後netは
  `-4,002.53 / -5,646.62 JPY`、PFも`0.673 / 0.561`であり、
  プラス化ゲートを通らない。
- `spike_fade`: inventory-only deterministic screenのBはBASE/STRESSとも
  約`+250 JPY`改善した。actual CはSTRESSでA比`+247.46 JPY`、
  DD`1.10%→0.94%`へ改善したが、BASEはAI費用分だけ悪化した。
  trade sampleは各cost 1件で、`DORMANT_OR_INSUFFICIENT`。
- `round_number_fade`: BASE/STRESSともnetはプラスで約`+126 JPY`改善したが、
  PFが`0.979 / 0.865`で1未満のためhard gate不通過。
- `burst`、`pullback_limit`はプラス化せず、`REJECT`ではなく
  `JANUARY_OBSERVED_FAILURE_NOT_REJECT`としてregistryに残す。
- actual forecast 4件はdirection accuracy `0%`、Brier `1.103754`、
  log loss `1.726689`、confidence calibration MAE `0.6925`、
  高信頼誤予測率`100%`。方向予測でリスクを増やす根拠は得られなかった。

したがって、AI介入メカニズムとしてBASE/STRESS両方のプラス化を確認した親familyは
0件。strategy factoryとmulti-strategy portfolio探索は起動しない。
Januaryを見た同versionの再調整も行わない。

## 正本・入力・権限

- immutable baseline:
  `/Users/tossaki/App/QuantRabbit-live/logs/dojo-historical/g2-parallel-rooms-20260726-r13`
- job:
  `81cec5d3b8f5fa371058aa2e42d213e239a163892295cfc234b85ef4c7e9be68`
- `month=2025-01`, `intrabar=OHLC`,
  `source=M5_EXACT28_2020_2026H1`, 12/12 COMPLETE, 0 failed
- immutable v1 prepared input:
  `logs/dojo-ai-inventory/r13-2025-01-oos-v1`
- v2r1 output:
  `logs/dojo-ai-inventory/r13-2025-01-oos-v2r1`
- prepared study SHA-256:
  `80b1403f0dff9a482538dce2ab1ae2f7cd03e52b18ec143d460843e5c9250415`
- calibration SHA-256:
  `9b79fad435e73860a63dc37188e087406cbc2025e9cbd9838f592c24e3205452`
- OOS result SHA-256:
  `9fac3d6e41255b1aefac4ca6a59ec0efcb5ec1a89ce98150941700c82702a775`
- `paper/replay only`, `live=false`, `broker mutation=false`,
  `order authority=NONE`
- baseline jobとv1成果物は再実行・編集・削除していない。
- `source_quote_coverage_proved=false`。結果は同一の不完全source上の
  experimental paired differenceであり、coverage修復後に再検証する。

先行して作成した
`logs/dojo-ai-inventory/r13-2025-01-oos-v2`は、deterministic calibration
screenへnotional AI費用を誤課金していたため診断用invalidated rootとした。
削除・上書きせず、修正版を別root`v2r1`へ生成した。

## multi-TF regime cacheと効率化

各decision timestampでclosed済みbarだけからD1/H4/H1/M5を計算した。
形成中HTF candle、未来quote、期間終値、append wall-clockは含めない。

- feature version: `QR_MTF_CLOSED_BAR_FEATURES_V2`
- decision timestamps: 6,336
- pairs: 28
- immutable cache rows: 177,408
- cache SHA-256:
  `693aa2db4cb65e8b4885a048b00d06696ba78e7f20d9ff6b94786cc46a96b913`
- calibration regime matrix: 712 rows
- OOS regime matrix: 722 rows
- missing regime state trades: 0
- future-append invarianceとforming-HTF exclusionをunit testで確認

full Cartesian brute-forceは使わず、cached regime matrixとsuccessive halvingを
使用した。288 cell-equivalent中120セルを評価し、deterministic screenを
`58.33%`削減した。actual Workerはfull schedule相当96 callsに対し13 callsで、
`86.46%`削減した。

実測処理時間はcache build `33.38s`、calibration matrix `4.95s`、
calibration/successive halving `140.75s`、OOS matrix `5.00s`、
OOS aggregate `6.18s`、合計`190.26s`にWorker応答待ち時間を加えた値。

## calibrationでsealしたpolicy / cadence

| Family | B/C policy | cadence | calibration sample |
|---|---|---|---|
| burst | INVENTORY_PROTECTIVE_V2 | FIXED_15M | eligible |
| mean_revert_24h | INVENTORY_BALANCED_V2 | ADAPTIVE | eligible |
| prev_day_extreme_fade | GUARD_ONLY_V2 | ADAPTIVE | eligible |
| pullback_limit | INVENTORY_PROTECTIVE_V2 | FIXED_60M | eligible |
| round_number_fade | MTF_CONFLICT_GUARD_V2 | FIXED_15M | eligible |
| spike_fade | GUARD_ONLY_V2 | ADAPTIVE | dormant/insufficient |

OOSを見た後にpolicy、cadence、thresholdを変更していない。

## 12-coordinate A/B/C比較

各セルは`net after all costs / PF / max DD`。Cのactual Worker費用は
160 JPY/USDで控除済み。rule-boundary deterministic B/Cセルはactual model
ではないためAI費用を課金していない。

| Family | Cost | A bot-only | B inventory only | C forecast+inventory | C calls/fallback/cost | C hard gate |
|---|---|---|---|---|---|---|
| burst | BASE | ¥-29,115 / 0.369 / 14.91% | ¥-29,115 / 0.369 / 14.91% | ¥-29,115 / 0.369 / 14.91% | 4/1/¥0.00 | FAIL |
| burst | STRESS | ¥-32,976 / 0.306 / 16.70% | ¥-33,127 / 0.305 / 16.75% | ¥-33,127 / 0.305 / 16.75% | 4/1/¥0.00 | FAIL |
| mean_revert_24h | BASE | ¥-6,894 / 0.741 / 4.17% | ¥-6,897 / 0.741 / 4.17% | ¥-4,003 / 0.673 / 2.73% | 4/4/¥10.48 | FAIL |
| mean_revert_24h | STRESS | ¥-10,543 / 0.625 / 5.80% | ¥-10,545 / 0.625 / 5.80% | ¥-5,647 / 0.561 / 3.44% | 3/3/¥8.88 | FAIL |
| prev_day_extreme_fade | BASE | ¥9,049 / 1.724 / 1.35% | ¥9,049 / 1.724 / 1.35% | ¥9,049 / 1.724 / 1.35% | 4/1/¥0.00 | PASS |
| prev_day_extreme_fade | STRESS | ¥7,262 / 1.470 / 1.62% | ¥7,262 / 1.470 / 1.62% | ¥7,262 / 1.470 / 1.62% | 4/1/¥0.00 | PASS |
| pullback_limit | BASE | ¥-3,569 / 0.825 / 3.25% | ¥-3,569 / 0.825 / 3.25% | ¥-3,569 / 0.825 / 3.25% | 4/1/¥0.00 | FAIL |
| pullback_limit | STRESS | ¥-7,169 / 0.665 / 4.69% | ¥-7,169 / 0.665 / 4.69% | ¥-7,169 / 0.665 / 4.69% | 4/1/¥0.00 | FAIL |
| round_number_fade | BASE | ¥2,513 / 0.921 / 1.06% | ¥2,639 / 0.979 / 1.06% | ¥2,639 / 0.979 / 1.06% | 4/1/¥0.00 | FAIL |
| round_number_fade | STRESS | ¥2,220 / 0.811 / 1.10% | ¥2,347 / 0.865 / 1.10% | ¥2,347 / 0.865 / 1.10% | 4/1/¥0.00 | FAIL |
| spike_fade | BASE | ¥-478 / 0.000 / 1.10% | ¥-229 / 0.000 / 0.94% | ¥-487 / 0.000 / 1.10% | 3/3/¥8.69 | FAIL |
| spike_fade | STRESS | ¥-519 / 0.000 / 1.10% | ¥-269 / 0.000 / 0.95% | ¥-272 / 0.000 / 0.94% | 3/3/¥8.78 | FAIL |

## Cの追加経済指標

`TP retained`はbaseline TP profitを分母にした比率で、追加保持益がある場合は
100%を超え得る。tokensはrule-boundaryセルではpacket/response見積であり、
actual model usageではない。

| Family | Cost | WR | Expectancy JPY | max margin | MC/ruin | TP retained | loss avoided | missed upside | turnover JPY | tokens in/out | equity multiple |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | BASE | 32.53% | -116.93 | 13.83% | 0/0 | 35.56% | +0.00 | +0.00 | 79,113,994 | 4494/2616 | 0.854423 |
| burst | STRESS | 30.12% | -133.04 | 13.26% | 0/0 | 34.00% | +0.00 | +0.00 | 74,526,254 | 4492/2619 | 0.834367 |
| mean_revert_24h | BASE | 44.30% | -20.90 | 14.10% | 0/0 | 147.43% | +14,425.77 | +11,523.66 | 27,383,557 | 4708/2797 | 0.979987 |
| mean_revert_24h | STRESS | 43.04% | -29.52 | 13.55% | 0/0 | 45.68% | +15,275.79 | +10,370.91 | 26,228,485 | 3867/2411 | 0.971767 |
| prev_day_extreme_fade | BASE | 57.14% | +143.63 | 7.97% | 0/0 | 54.32% | +0.00 | +0.00 | 25,486,655 | 4620/2622 | 1.045245 |
| prev_day_extreme_fade | STRESS | 55.56% | +115.27 | 7.91% | 0/0 | 52.39% | +0.00 | +0.00 | 25,193,368 | 4622/2622 | 1.036311 |
| pullback_limit | BASE | 45.95% | -24.12 | 7.85% | 0/0 | 35.30% | +0.00 | +0.00 | 57,446,237 | 4619/2620 | 0.982154 |
| pullback_limit | STRESS | 42.57% | -48.44 | 7.68% | 0/0 | 32.54% | +0.00 | +0.00 | 55,655,962 | 4630/2620 | 0.964155 |
| round_number_fade | BASE | 60.00% | +263.92 | 8.17% | 0/0 | 40.38% | +0.00 | +0.00 | 4,108,723 | 4626/2618 | 1.013196 |
| round_number_fade | STRESS | 60.00% | +234.69 | 8.15% | 0/0 | 38.40% | +126.62 | +0.00 | 4,098,530 | 4643/2619 | 1.011735 |
| spike_fade | BASE | 0.00% | -477.83 | 8.03% | 0/0 | 0.00% | +0.00 | +0.00 | 400,042 | 3863/2332 | 0.997567 |
| spike_fade | STRESS | 0.00% | -263.23 | 8.01% | 0/0 | 0.00% | +256.24 | +0.00 | 400,023 | 3842/2378 | 0.998640 |

## actual Main / Worker監査

Phase Aは全セルをdeterministicにscreenし、top/boundary/high-riskのC 4セルだけを
fresh Worker境界へ渡した。各packetはcutoff timestamp、packet hash、
prompt/policy version、response hashを保存し、未来quote、terminal result、
他arm結果、append wall-clockを含まない。

| Family | Cost | calls | accepted | schema invalid | call-cap fallback | accepted action | tokens in/out | AI cost |
|---|---|---:|---:|---:|---:|---|---:|---:|
| mean_revert_24h | BASE | 4 | 1 | 3 | 1 | REDUCE_SHORT 50%, NO_NEW_SHORTS | 4708/2797 | ¥10.48 |
| mean_revert_24h | STRESS | 3 | 1 | 2 | 1 | HOLD, NO_NEW_SHORTS | 3867/2411 | ¥8.88 |
| spike_fade | BASE | 3 | 1 | 2 | 1 | HOLD | 3863/2332 | ¥8.69 |
| spike_fade | STRESS | 3 | 1 | 2 | 1 | PARTIAL_CLOSE 50%, NO_NEW_SHORTS | 3842/2378 | ¥8.78 |

actual totalsは13 calls、accepted 4、schema invalid 9、input/output
`16,280 / 9,918 tokens`、notional AI cost`36.8272 JPY`。
invalid responseは利益都合で修復・再試行せず、fail-closed `HOLD`にした。
各セルのcall cap到達後も、別の追加callは行わなかった。

この高いschema失敗率はv2の主要な統合失敗である。実測後、今後のWorker
envelopeへMain validatorと一致するexact response schemaを同梱するようにした。
この修正を使って同じJanuary結果を再評価してはいない。

## forecast精度

- scored: 4
- direction accuracy: `0%`
- 30分: `0/2`、Brier `0.862408`、log loss `1.366684`
- 120分: `0/2`、Brier `1.345100`、log loss `2.086694`
- overall Brier: `1.103754`
- overall log loss: `1.726689`
- confidence calibration MAE: `0.6925`
- confidence 0.50–0.75: 3件、実精度0%、平均confidence `0.6633`
- confidence 0.75–1.00: 1件、実精度0%、平均confidence `0.78`
- wrong high-confidence forecast rate: `100%`

forecast採点はposthocのみでacting inputへ戻していない。低精度の方向予測を
売買リスク増加に使用しないというv2方針を維持する。

## 状態分類とfalse-negative防止

| Family | January status | 理由 |
|---|---|---|
| burst | JANUARY_OBSERVED_FAILURE_NOT_REJECT | 十分な1月sampleはあるが単月だけでREJECTしない |
| mean_revert_24h | JANUARY_OBSERVED_FAILURE_NOT_REJECT | DD/損失縮小は再現、net/PFは負 |
| prev_day_extreme_fade | CONDITIONAL_JANUARY_SHORT_GATE_PASS | BASE/STRESSで正、guard-only HOLD |
| pullback_limit | JANUARY_OBSERVED_FAILURE_NOT_REJECT | 1月では負、別regime未検証 |
| round_number_fade | JANUARY_OBSERVED_FAILURE_NOT_REJECT | net正だがPF gate不通過 |
| spike_fade | DORMANT_OR_INSUFFICIENT | calibration sampleなし、OOS各1 trade |

既存12 coordinatesは削除せずchampion/challenger registryへ残す。REJECTは、
十分なeligible observationと複数未使用期間の対応regimeで負、またはrisk
gate違反を確認した場合だけに限定する。

## 3x stretch target

最大monthly equity multipleは`prev_day_extreme_fade/BASE`の`1.045245`。
月3倍までの絶対gapは`1.954755x`、必要な追加return gapは`195.4755%`。
安全制約、全費用、現行資本のままJanuaryで3xを達成した候補は0。
単月3x、過剰なstrategy重複、martingale、無制限ナンピン、隠れレバレッジを
成果主張に使わない。

複数月のmedian/worst、positive-month hit rate、3x hit rate、連敗、回復期間、
return-DD-ruin Paretoはwalk-forward未実施のため未算出。

## walk-forward / strategy factory / portfolio

walk-forward contract SHA-256:
`0f13b25439770d952fc3f5a42cc461c36e1f28bbc963927a4fc6a00c2a2edf9e`。
同versionはJanuaryへ戻らず、時系列で最低8 non-overlap OOS blocks、
可能なら12か月以上へ一方向に進める。policy/schema/cadence/thresholdを変えたら
新versionとし、それ以降の完全未使用期間だけで評価する。

`prev_day_extreme_fade`はCONDITIONAL guard-only候補としてwalk-forwardへ進め得る。
ただしAI介入によるmarginal improvementは0であり、mechanism sibling factoryの
親にはしない。factory statusは
`NOT_STARTED_NO_JANUARY_BASE_STRESS_CHAMPION`、eligible parentは空。

multi-strategy portfolio contractには次をsealした。

- 同時刻PnL covariance/correlation、共通通貨・方向・session・regime exposure
- simultaneous loss、inventory overlap、capital/margin competition
- combined equity curveの実再構築
- marginal net、marginal DD、diversification benefit、追加cost/turnover
- sealed OOSでmarginal netが正、PFまたはrisk-adjusted return改善、
  DD/ruin非悪化の場合だけ追加

今回factoryを起動していないため、組み合わせ採用0、相関理由の不採用0。
単体プラスを単純合算したportfolio収益は主張しない。

## 次条件

1. January v2r1はseal済み。同versionをJanuary結果へ合わせて再調整しない。
2. `prev_day_extreme_fade` guard-onlyを、別月の未使用multi-TF regimeへ
   one-way walk-forwardする。
3. `mean_revert_24h`の有効メカニズム候補はcountertrend shortの50%縮小と
   NO_NEW_SHORTS。別月でnet/PFプラスへ越えるかを、同じthresholdのまま検証する。
4. `spike_fade`はsample不足。別月の同regimeと逆regimeでeligible tradeを蓄積し、
   DORMANTからCONDITIONALへ移せるか確認する。
5. Worker schemaをenvelope内で固定し、次の完全未使用期間でinvalid率を測る。
   過去Januaryの改善主張へは遡及しない。
6. coverage修復後、同じhash/cutoff契約で最終再検証する。
7. hard gateを通るAI intervention mechanismが現れるまでstrategy factoryと
   portfolio追加探索を開始しない。

D perfect-hindsight oracleはacting、candidate selection、成果主張に使用して
いない。live runtime、OANDA、12 paper rooms、automationは変更していない。
