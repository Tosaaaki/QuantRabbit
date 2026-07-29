# QuantRabbit｜旧・裁量戦略ワーカー高速リプレイ

## 最上段結論

- **保護付きでは儲かっていない。**
- **採用なし。**
- `scalp_ping_5s_b` のno-SL対照だけは +79,128.30円 / PF 1.308だったが、DD 131,473.51円で、time-stopを含む保護付きarmは全て赤字。無制限損失対照なので採用根拠にしない。
- `scalp_macd_rsi_div`、同B、wick blendは1 / 5 / 1 entry、wick proは0 entryで**証拠不足**。
- 優先10 familyのうち、十分なentryがあった6 familyは全て**不採用**、4 familyは**証拠不足**。

## Inventory進捗

| 区分 | family |
|---|---:|
| 正規化inventory | 82 |
| 評価・試行済み（今回前） | 27 |
| 今回の優先cluster | 10 |
| 評価・試行済み合計 | 37 |
| 未評価合計 | 45 |
| 未評価・実装回収可能 | 13 |
| 未評価・証拠のみ | 32 |

82 familyの各レコード、現在status、wrapper target、今回metricsは `normalized_inventory_status.json` に保持した。評価済み/試行済みと未評価を同一件数へ混ぜていない。

### 残る実装回収可能13 family

`h1_momentum`, `impulse_retrace`, `micro_adaptive_revert`, `momentum_stack`, `range_revert_lite`, `scalp_drought_revert`, `scalp_extrema_reversal`, `scalp_false_break_fade`, `scalp_level_reject`, `scalp_precision_lowvol`, `scalp_tick_imbalance`, `scalp_vwap_revert`, `trend_reclaim`

### 証拠のみ32 family

`basic`, `fast_scalp`, `london_momentum`, `ma_rsi_macd`, `macro_core`, `macro_tech_fusion`, `manual_spike`, `manual_swing`, `micro_core`, `micro_multistrat`, `micro_pullback_fib`, `micro_range_revert_lite`, `mirror_spike`, `mirror_spike_s5`, `mirror_spike_tight`, `mm_lite`, `mtf_breakout`, `onepip_maker_s1`, `pullback_runner_s5`, `pullback_scalp`, `range_bounce`, `range_compression_break`, `scalp_core`, `scalp_multistrat`, `scalp_precision`, `scalp_reversal_nwave`, `spike_reversal`, `squeeze_break_s5`, `tech_fusion`, `trend_pullback`, `vol_spike_rider`, `vol_squeeze`

## 今回の優先10 family

各family内で同じentry cohort、同じ記録bid/ask、同じ費用を使い、no-SL、固定SL、ATR SL、volatility trail、time-stopを比較した。下表はno-SLを除く保護付きarmのうち、Bot/AIそれぞれでNetが最も高いarm。

| family | entry | Bot exit | Bot Net | PF | Exp | DD | AI exit | AI Net | PF | Exp | DD | 判定 |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| momentum_burst | 367 | vol trail | -1,923.88 | 0.760 | -5.24 | 2,326.55 | fixed SL | -3,247.46 | 0.474 | -8.85 | 3,356.39 | 不採用 |
| scalp_macd_rsi_div | 1 | time-stop | +31.00 | n/a | +31.00 | 0.00 | time-stop | -8.00 | 0.000 | -8.00 | 8.00 | 証拠不足 |
| scalp_macd_rsi_div_b | 5 | fixed SL | +142.97 | n/a | +28.59 | 0.00 | time-stop | -44.00 | 0.397 | -8.80 | 73.00 | 証拠不足 |
| scalp_ping_5s | 1,716 | time-stop | -15,432.32 | 0.426 | -8.99 | 15,489.31 | vol trail | -16,733.15 | 0.288 | -9.75 | 16,733.15 | 不採用 |
| scalp_ping_5s_b | 26,686 | time-stop | -205,309.76 | 0.545 | -7.69 | 205,910.54 | time-stop | -229,437.28 | 0.419 | -8.60 | 229,472.28 | 不採用 |
| scalp_ping_5s_c | 82,641 | fixed SL | -748,673.47 | 0.474 | -9.06 | 748,714.47 | fixed SL | -744,190.64 | 0.396 | -9.01 | 744,231.64 | 不採用 |
| scalp_ping_5s_d | 63,272 | fixed SL | -580,145.23 | 0.329 | -9.17 | 580,228.22 | fixed SL | -579,263.14 | 0.284 | -9.16 | 579,346.14 | 不採用 |
| scalp_ping_5s_flow | 68,719 | vol trail | -632,720.69 | 0.098 | -9.21 | 632,720.69 | fixed SL | -631,634.81 | 0.281 | -9.19 | 631,710.81 | 不採用 |
| scalp_wick_reversal_blend | 1 | vol trail | -43.00 | 0.000 | -43.00 | 43.00 | time-stop | -3.00 | 0.000 | -3.00 | 3.00 | 証拠不足 |
| scalp_wick_reversal_pro | 0 | time-stop | 0.00 | n/a | n/a | 0.00 | time-stop | 0.00 | n/a | n/a | 0.00 | 証拠不足 |

AI ONは外部modelを呼ばない凍結済みのcausal inventory rule。model call 0、execution AI cost 0円。高頻度ping C/D/flowでは固定SL AI ONがBotより損失をわずかに減らす場合があったが、Net/PF/expectancyは明確な不採用域。

## 既評価結果の統合

### 旧Bot対AI inventory A/B

| family | Bot Net / PF / Exp / DD / trades | AI Net / PF / Exp / DD / trades | 結論 |
|---|---|---|---|
| TrendMA | +31,107.84 / 1.358 / +384.05 / 14,460.00 / 81 | -8,585.72 / 0.528 / -106.00 / 10,047.49 / 81 | Bot historical診断のみ。AI policy不採用 |
| PulseBreak | +2,053.53 / 1.079 / +78.98 / 10,428.37 / 26 | +2,530.95 / 1.225 / +97.34 / 5,007.92 / 26 | AI費用不明のため経済性未確定。Paper継続のみ |
| M1Scalper（旧簡略） | -352,351.19 / 0.807 / -229.54 / 364,030.90 / 1,535 | -57,206.60 / 0.508 / -37.27 / 57,526.60 / 1,535 | 不採用 |
| RangeFader | sample 6 | 未比較 | 証拠不足 |

このA/BのAI costはplatformから取得できず、3 familyともAI込み最終経済性を確定していない。

### M1Scalper faithful port

5つのlineage-unseen診断窓を合算すると、Botは Net -4,713.90円 / PF約0.629 / expectancy約-10.18円 / 463 trades。凍結local AIは Net -89.97円 / PF約0.439 / expectancy約-11.25円 / 8 trades / 2,872 decisions / model call 0。旧簡略結果の「AI provisional +51円 / 6 trades」は後続のfaithful portで再現せず、採用しない。

### その他の既試行family

`impulse_break_s5`, `impulse_retest_s5`, `impulse_momentum_s5`, `pullback_s5`, `vwap_magnet_s5` はholdout赤字、`stop_run_reversal` は0 trade。`trend_breakout` はBot -582.57円、AI -168.57円の各1 trade、`session_open` は+461.44円だが1 trade、他のzero-trade familyも証拠不足。既存27 familyを今回10 familyへ重複加算していない。

## 完全inventoryの証拠

- archive/Git/worker実装/systemd/既存台帳を正規化した82 familyを継承。
- VM識別子は4: `fx-trader-vm`, `fx-trader-rescue`, `qr-ssh-rescue`, `quantrabbit`。
- 直接worker対応が取れたVMは `fx-trader-vm` の3 family: `basic`, `failed_break_reverse`, `scalp_false_break_fade`。rescue名を推測でfamilyへ紐づけていない。
- systemd unit痕跡は218。
- thin wrapper関係は19。ping B/C/D/flow→ping base、MACD/RSI B→base、micro runtime系11、scalp precision系3。wrapper実体と正規化family数を混同していない。
- 現GCPは照会しておらず、削除済みVMについてはstatic artifactの範囲だけを証拠とした。

## Replay設計と制約

- corpus: 2026-01-23 / 25 / 26 / 27 / 28、合計414,780 ticks。
- evidence class: `LINEAGE_UNSEEN_DIAGNOSTIC`。2024–2026H1 corpusは既に反復利用済みで、グローバルな未使用holdoutとは表現しない。
- entry生成は現在tickまたは確定済みM1/M5/H1/H4 candleだけを使用。未来データは禁止。
- spreadは記録bid/ask。slippageは0.05 pip/fill。financingは保守的固定debit 10円/1万通貨/日を保有時間按分。実brokerのside別swap実績ではない。
- 期末open positionは実行可能sideでMTM。
- wick blendのarchive内外部projection依存はneutral passthrough。past-only wick/range/tick gateは維持したが、完全忠実portではなくadapter診断。
- 同時独立positionとして全entryを全armで評価したため、exit差によるcohort欠落はない。一方、live accountの同時position capを再現するportfolio backtestではない。

## 速度・安全性

- 確定run: workers=2、603.750秒、10/10 family成功、error 0。
- 観測中のchild CPUは概ね77–88%、RSSは約0.27–0.50GB/process。既存Paper/shadow processを停止せず完走。
- 現行Paper共存下で実測した安全上限は2。4並列はPaper保護のため未実測で、上限として主張しない。
- `live=false`、`authority=NONE`、broker/order mutationなし。archiveはread-only import、variantはprocess隔離。

## 採否と次cluster

- 採用: 0
- 不採用: `momentum_burst`, `scalp_ping_5s`, `scalp_ping_5s_b`, `scalp_ping_5s_c`, `scalp_ping_5s_d`, `scalp_ping_5s_flow`
- 証拠不足: `scalp_macd_rsi_div`, `scalp_macd_rsi_div_b`, `scalp_wick_reversal_blend`, `scalp_wick_reversal_pro`
- 次cluster: `scalp_drought_revert`, `scalp_extrema_reversal`, `scalp_false_break_fade`, `scalp_level_reject`, `scalp_precision_lowvol`, `scalp_tick_imbalance`, `scalp_vwap_revert`
- その後: `h1_momentum`, `impulse_retrace`, `micro_adaptive_revert`, `momentum_stack`, `range_revert_lite`, `trend_reclaim`

## 機械成果物

- `priority_replay.json`: 10 family×5 exit×AI OFF/ON、全metrics、source/tick SHA、bounded sample。
- `normalized_inventory_status.json`: 82 familyの現行partition、評価status、wrapper/VM/systemd証拠。
- `report.md`: 優先runの短い自動生成summary。
