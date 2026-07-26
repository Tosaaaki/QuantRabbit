# r13 2025-01 OHLC AI在庫・ナラティブ監督 OOS比較

## 結論

採用候補は0/6 strategy family。Phase B actual Workerでは、C
（forecast + adaptive inventory）が12 coordinates中2 coordinatesでA/Bより
改善したが、どちらも同一familyのBASE/STRESS両方を通過しなかった。
したがって全familyを`REJECT`とし、次段階のJanuary OLHC候補は出さない。

- `mean_revert_24h / BASE`: CはA/B比 +6,550.01 JPY、DD 4.1721%から
  0.1777%へ改善。初回`PAUSE_NEW_ENTRIES`で191予定取引中190件をskip。
  `STRESS`は差分0のためfamily不採用。
- `spike_fade / STRESS`: CはA/B比 +419.08 JPY、DD 1.0971%から
  0.8233%へ改善。初回`REDUCE_SHORT 25%`。`BASE`は差分0のためfamily不採用。
- Bは全12 coordinatesでAと同一経済結果。actual Workerの受理7件は全て
  `HOLD`、5件はschema不正でfail-closed `HOLD`。
- C forecastは12/12採点可能、direction accuracy 33.33%、30分40.00%、
  60分28.57%、Brier 0.757322、log loss 1.221414、confidence calibration
  MAE 0.543333。confidence 0.50–0.75帯の平均0.585に対し実精度0.3333で、
  過信傾向。confidence 0.70以上は0件。

## 入力・分割・権限

- immutable baseline:
  `/Users/tossaki/App/QuantRabbit-live/logs/dojo-historical/g2-parallel-rooms-20260726-r13`
- job:
  `81cec5d3b8f5fa371058aa2e42d213e239a163892295cfc234b85ef4c7e9be68`
- `month=2025-01`, `intrabar=OHLC`,
  `source=M5_EXACT28_2020_2026H1`, 12/12 COMPLETE, 0 failed
- prepared study SHA-256:
  `80b1403f0dff9a482538dce2ab1ae2f7cd03e52b18ec143d460843e5c9250415`
- immutable job result SHA-256:
  `0a79a279de2be8d541d14513b07fecd90fb5cd43dbd01d57d239a606faf8bca8`
- calibration: `[2025-01-01T22:00:00Z, 2025-01-18T00:00:00Z)`
- held-out OOS: `[2025-01-18T00:00:00Z, 2025-01-31T21:55:01Z)`
- boundary policy: partition開始前にopenしたtradeをpurge
- prepared market frames: 25,344 OHLC coordinate frames
- initial capital: 各partition 200,000 JPY
- `paper/replay only`, `live=false`, `broker mutation=false`,
  `order authority=NONE`
- `source_quote_coverage_proved=false`。同一不完全source上のpaired差分であり、
  official evidenceではない。coverage修復後に同じ契約で再検証する。

full-month A derivativeのending equityは、immutable baselineの12 coordinatesと
最大絶対誤差 `1.4551915228366852e-10 JPY` で一致した。baseline job resultと
準備時記録のSHA-256も一致した。baseline artifactは再実行・編集・削除していない。

## Main / Worker境界

Mainはmarket clockを進め、cadence/eventを発火し、cutoffまでのpacketだけを
fresh Workerへ渡した。Workerは対象packet以外のDOJO履歴、未来quote、将来損益、
終端、他群結果を読まない。Mainはexact schema、packet SHA-256、observed_at、
authorityをfail-closed検証してからactionを適用した。

各audit rowは`packet_sha256`、`cutoff_epoch`、`prompt_version`、
`policy_version`、`attempted_worker_response_sha256`、受理後
`response_sha256`、trigger、action、fallback/failure classを保持する。
Worker結果を見ての再試行は禁止し、各セルのactual AI call capは事前登録した1回。
cap到達またはinvalid response時は`HOLD`、その後はsupervisionを停止した。

Phase B operational totals:

| Arm | cells | actual calls | accepted | schema invalid | call-cap fallback | fallback decisions | accepted actions | tokens in/out | notional USD |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B inventory only | 12 | 12 | 7 | 5 | 10 | 15 | HOLD 7 | 17,896 / 9,535 | 0.232505 |
| C forecast + inventory | 12 | 12 | 12 | 0 | 10 | 10 | HOLD 10, PAUSE 1, REDUCE_SHORT 1 | 20,361 / 15,767 | 0.338310 |

notional costは`$5/M input + $15/M output`仮定であり、実請求額ではない。
取引の`net_after_all_costs_jpy`にはbaselineのspread/slippage/financing等を含むが、
このnotional AI USD費用はJPY換算せず別掲した。family採用はAI費用控除前でも0件
なので、費用控除後に採用へ反転しない。

## 12-coordinate OOS比較

`A/B/C net`はJPY、DDはfraction。B/Cのpolicy/cadenceはcalibrationだけで選択し、
OOS結果を選択へ戻していない。

| Family | Cost | B policy/cadence | C policy/cadence | A net | B net | C net | B-A | C-A | A DD | C DD | C gate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| burst | BASE | PROTECTIVE/FIXED30 | PATIENT/FIXED5 | -29,115.41 | -29,115.41 | -29,115.41 | 0.00 | 0.00 | 0.149097 | 0.149097 | FAIL |
| burst | STRESS | PROTECTIVE/FIXED5 | PATIENT/FIXED5 | -32,975.92 | -32,975.92 | -32,975.92 | 0.00 | 0.00 | 0.166984 | 0.166984 | FAIL |
| mean_revert_24h | BASE | PROTECTIVE/FIXED30 | PATIENT/EVENT | -6,894.15 | -6,894.15 | -344.14 | 0.00 | +6,550.01 | 0.041721 | 0.001777 | PASS coordinate |
| mean_revert_24h | STRESS | PROTECTIVE/FIXED30 | PATIENT/EVENT | -10,542.62 | -10,542.62 | -10,542.62 | 0.00 | 0.00 | 0.057959 | 0.057959 | FAIL |
| prev_day_extreme_fade | BASE | BALANCED/FIXED30 | PROTECTIVE/FIXED60 | +9,048.98 | +9,048.98 | +9,048.98 | 0.00 | 0.00 | 0.013455 | 0.013455 | FAIL |
| prev_day_extreme_fade | STRESS | BALANCED/FIXED30 | PROTECTIVE/FIXED60 | +7,262.22 | +7,262.22 | +7,262.22 | 0.00 | 0.00 | 0.016233 | 0.016233 | FAIL |
| pullback_limit | BASE | PATIENT/FIXED5 | PATIENT/FIXED5 | -3,569.22 | -3,569.22 | -3,569.22 | 0.00 | 0.00 | 0.032497 | 0.032497 | FAIL |
| pullback_limit | STRESS | PATIENT/FIXED5 | PATIENT/FIXED5 | -7,168.95 | -7,168.95 | -7,168.95 | 0.00 | 0.00 | 0.046852 | 0.046852 | FAIL |
| round_number_fade | BASE | BALANCED/EVENT | PROTECTIVE/FIXED60* | +2,513.46 | +2,513.46 | +2,513.46 | 0.00 | 0.00 | 0.010568 | 0.010568 | FAIL |
| round_number_fade | STRESS | BALANCED/EVENT | PROTECTIVE/FIXED60* | +2,220.28 | +2,220.28 | +2,220.28 | 0.00 | 0.00 | 0.011006 | 0.011006 | FAIL |
| spike_fade | BASE | PROTECTIVE/FIXED60* | PROTECTIVE/FIXED60* | -477.83 | -477.83 | -477.83 | 0.00 | 0.00 | 0.010956 | 0.010956 | FAIL |
| spike_fade | STRESS | PROTECTIVE/FIXED60* | PROTECTIVE/FIXED60* | -519.47 | -519.47 | -100.39 | 0.00 | +419.08 | 0.010971 | 0.008233 | PASS coordinate |

`*` calibrationでadmissible cellがなく、最大netのdiagnostic cellを選択したもの。
これは採用候補を意味しない。

## Aのcoordinate経済指標

Bは全12 coordinatesでAと経済指標が同一。`PF`、`WR`、`DD`、`MU`、
`TP retained`はそれぞれprofit factor、win rate、max drawdown、最大margin
utilization、TP profit retained fraction。

| Family | Cost | Net JPY | PF | WR | Expectancy JPY | DD | MU | MC/Ruin | TP retained | loss avoided | missed upside | Turnover JPY |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | BASE | -29,115.41 | 0.368858 | 32.53% | -116.93 | 14.9097% | 13.8347% | 0/0 | 35.56% | 0.00 | 0.00 | 79,113,994 |
| burst | STRESS | -32,975.92 | 0.306385 | 30.12% | -132.43 | 16.6984% | 13.2477% | 0/0 | 34.00% | 0.00 | 0.00 | 74,526,254 |
| mean_revert_24h | BASE | -6,894.15 | 0.741219 | 42.41% | -36.10 | 4.1721% | 14.3134% | 0/0 | 91.39% | 0.00 | 0.00 | 66,783,889 |
| mean_revert_24h | STRESS | -10,542.62 | 0.625209 | 41.88% | -55.20 | 5.7959% | 13.9077% | 0/0 | 50.16% | 0.00 | 0.00 | 64,200,222 |
| prev_day_extreme_fade | BASE | +9,048.98 | 1.723808 | 57.14% | +143.63 | 1.3455% | 7.9712% | 0/0 | 54.32% | 0.00 | 0.00 | 25,486,655 |
| prev_day_extreme_fade | STRESS | +7,262.22 | 1.469746 | 55.56% | +115.27 | 1.6233% | 7.9122% | 0/0 | 52.39% | 0.00 | 0.00 | 25,193,368 |
| pullback_limit | BASE | -3,569.22 | 0.825391 | 45.95% | -24.12 | 3.2497% | 7.8539% | 0/0 | 35.30% | 0.00 | 0.00 | 57,446,237 |
| pullback_limit | STRESS | -7,168.95 | 0.665165 | 42.57% | -48.44 | 4.6852% | 7.6799% | 0/0 | 32.54% | 0.00 | 0.00 | 55,655,962 |
| round_number_fade | BASE | +2,513.46 | 0.920540 | 60.00% | +251.35 | 1.0568% | 8.1712% | 0/0 | 39.27% | 0.00 | 0.00 | 4,108,723 |
| round_number_fade | STRESS | +2,220.28 | 0.810546 | 50.00% | +222.03 | 1.1006% | 8.1573% | 0/0 | 37.50% | 0.00 | 0.00 | 4,098,530 |
| spike_fade | BASE | -477.83 | 0.000000 | 0.00% | -477.83 | 1.0956% | 8.0280% | 0/0 | 0.00% | 0.00 | 0.00 | 400,042 |
| spike_fade | STRESS | -519.47 | 0.000000 | 0.00% | -519.47 | 1.0971% | 8.0289% | 0/0 | 0.00% | 0.00 | 0.00 | 400,023 |

Cは10 coordinatesで上表Aと全経済指標が同一。異なる2 coordinatesは次の通り。

| Family | Cost | Net JPY | PF | WR | Expectancy JPY | DD | MU | MC/Ruin | TP retained | loss avoided | missed upside | Turnover JPY |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean_revert_24h | BASE | -344.14 | 0.000000 | 0.00% | -1.80 | 0.1777% | 7.1636% | 0/0 | 0.00% | 26,286.50 | 19,736.49 | 357,244 |
| spike_fade | STRESS | -100.39 | 0.000000 | 0.00% | -100.39 | 0.8233% | 7.9383% | 0/0 | 0.00% | 419.08 | 0.00 | 400,023 |

## AI resource detail

各セルactual callは1回。tokensはinput/output、costはnotional USD。

| Family | Cost | B tokens | B cost | C tokens | C cost |
|---|---|---:|---:|---:|---:|
| burst | BASE | 1,601/871 | 0.021070 | 1,967/1,578 | 0.033505 |
| burst | STRESS | 1,568/839 | 0.020425 | 1,775/1,243 | 0.027520 |
| mean_revert_24h | BASE | 1,546/791 | 0.019595 | 2,027/1,632 | 0.034615 |
| mean_revert_24h | STRESS | 1,883/1,200 | 0.027415 | 1,968/1,539 | 0.032925 |
| prev_day_extreme_fade | BASE | 1,684/871 | 0.021485 | 1,943/1,543 | 0.032860 |
| prev_day_extreme_fade | STRESS | 1,636/815 | 0.020405 | 1,825/1,228 | 0.027545 |
| pullback_limit | BASE | 1,609/810 | 0.020195 | 1,820/1,230 | 0.027550 |
| pullback_limit | STRESS | 1,657/871 | 0.021350 | 1,747/1,200 | 0.026735 |
| round_number_fade | BASE | 1,602/781 | 0.019725 | 2,018/1,559 | 0.033475 |
| round_number_fade | STRESS | 1,634/818 | 0.020440 | 1,793/1,178 | 0.026635 |
| spike_fade | BASE | 738/434 | 0.010200 | 739/778 | 0.015365 |
| spike_fade | STRESS | 738/434 | 0.010200 | 739/1,059 | 0.019580 |

## 採用判断と次条件

採用ゲートはheld-out OOSでAよりnet改善、DD非悪化、margin call/ruin非増加、
CはBにも同条件で優越、かつBASE/STRESS方向一致。6 family全て不通過。
training/calibrationの適合や決定論Phase Aの改善は成果主張に使用しない。

次の反復条件:

1. source quote coverageを修復し、同じpacket cutoff/hash契約で再検証する。
2. Phase B call cap 1を増やす前に、schema-constrained structured outputを固定し、
   Bで発生した5/12 invalidを0へする。
3. `mean_revert_24h`はBASEの全停止が効いた一方STRESSへ転移しなかったため、
   全停止ではなく方向制限・loss/TP event後の段階縮小を事前登録して再校正する。
4. `spike_fade`はSTRESSの25%縮小だけが効いたため、BASE/STRESS共通の
   TP-progress/giveback条件を校正し、forecast方向そのものへの依存を下げる。
5. forecastは過信・低精度なので、confidenceをaction sizingへ使う前に
   calibrationを改善し、horizon 120分は未評価のため追加する。
6. 改善がBASE/STRESS両方で再現するまでpaper automationへ移植しない。

## 正本成果物

- output root:
  `logs/dojo-ai-inventory/r13-2025-01-oos-v1`
- calibration SHA-256:
  `a69ef78f9fba5f9d8bbb55bc05368a629fbea289cdfc402d452065e59c60a2e0`
- Phase B OOS result SHA-256:
  `942b5116dc2e1e6f137c5939b38320e8589d56c8ccf2802139b72807d4ef8e8a`
- machine result: `phase-b-oos-result.json`
- intervention audit: 各
  `phase-b-sessions/{B_INVENTORY_ONLY,C_FORECAST_INVENTORY}/{coordinate_id}.json`
- D perfect-hindsight oracleは実装・acting・成果主張に使用していない。
- 次月bot-only、live/paper runtime、12 paper rooms、automationは変更していない。
