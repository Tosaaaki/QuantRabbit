# DOJO capital-hold opportunity V1 audit

監査日: 2026-07-19

## 判定

`+¥166,991` は、旧DOJO replay台帳が記録した実現損益 `+¥166,990.89` の四捨五入表示として
来歴と内部計算を確認できた。ただし、この監査で証明できるのは保存済み旧TRAIN replayの
自己整合までであり、未使用相場でのedge、1か月実績、連続mark-to-market成績、または
bit-for-bit再実行可能性ではない。

| 証明対象 | 判定 |
|---|---|
| 保存された旧replayの開始・終了残高 | `VERIFIED` |
| 台帳hash chainと保存artifactのbyte同一性 | `VERIFIED` |
| corpusに対する保存約定・決済quote | `VERIFIED` |
| 保存eventからのslippage・financing・P/L再計算 | `VERIFIED` |
| exact source closureによる再実行 | `UNPROVABLE_MISSING_SOURCE_BYTES` |
| 60分解放の実市場忠実性 | `FAILED_ARCHIVE_EXECUTED_AT_61_MINUTES` |
| 注文時点でmarketableな指値の即時約定 | `FAILED_1_OF_305_ORDERS_PER_PATH_DELAYED` |
| 連続MTM drawdown・margin | `UNPROVEN_NO_ACCOUNT_MARKS` |
| 未使用holdout / prospective再現性 | `UNPROVEN_WORN_TRAIN` |
| 月次3倍 | `UNPROVEN_NOT_A_MONTH_AND_EXTRAPOLATED` |

## 対象

- batch: `CAPITAL_HOLD_OPPORTUNITY_V1`
- candidate: `full_time_release`
- artifact root: `/Users/tossaki/App/QuantRabbit_archives/DOJO_20260719/codex-capital-hold-opportunity-train-v1`
- window: `[2025-03-03T00:00:00Z, 2025-03-14T21:59:00Z]`（11日21時間59分）
- initial balance: `¥200,000`
- pairs: `USD_JPY`, `CAD_JPY`, `EUR_USD`, `GBP_USD`
- strategy: spike-fade、20x notional、25x broker leverage、TP 3 ATR、SL 25 pips、60分解放
- costs: 0.3 pip/fill、0.8 pip/day
- conservative path: synthetic M1 `OLHC`

## 再検証結果

- 682 ledger recordsのSHA-256 chainとterminal tipを全行検証した。
- 68 fills、68 resolved exits、51 winsを確認した。
- 開始 `¥200,000.00`、終了 `¥366,990.89`、差額 `+¥166,990.89` を確認した。
- 2桁丸め済みexit P/Lの合計は `+¥166,990.86`。非丸め口座残高との差は `¥0.03`。
- 136 fill/exit quotesはすべて封印corpusのOLHC座標と一致した。
- 54 conversion source quotesはすべてcorpusと一致した。
- 68 tradesのfill、TP/SL/CLOSE、slippage、financing、P/L独立再計算は不一致0だった。
- 4 corpus shardsの現在bytesは封印size/SHA-256と一致した。
- `QUOTE_BATCH_BEGIN`、`ACCOUNT_MARK`、terminal MTM markはすべて0件だった。
- CAD_JPYの `+¥110,084.08` が悲観側利益の65.9%を占めた。
- 同じ設定のOHLCは `+¥283,770.99` で、悲観OLHCとの差が大きくM1内経路依存も強い。
- 旧botはresting fillの60分時計を約定時ではなく次のOでの発見時から開始していた。
  OLHCでは68取引中11件、OHLCでは71取引中10件のtimeout CLOSEが実約定から
  `3,660秒` 後だった。両経路とも`3,600秒`ちょうどのtimeout CLOSEは0件だった。
- この旧artifactで週末を跨ぐ長時間overholdはなく、観測できる直接影響は21取引の
  1分延長である。ただし価格差を含む修正後損益はexact旧sourceがないため再実行不能である。
- OLHCの11 timeout取引を、同一entry/unitsのまま正しい60分Oへ直接repricingすると、合計は
  `-¥38,103.74` から約 `-¥39,219.52` となり、旧headlineは直接差分で約 `¥1,115.78`
  過大だった。ただし早い資金解放が後続の受付・sizeを変えるため、これはexact rerunではない。
- 旧VirtualBrokerは注文時のOですでにmarketableなLIMIT/STOPも即時処理せず、後続H/Lまで
  待たせる挙動だった。decision epochと封印corpusの次Oから、両経路とも305注文中1件、
  USD_JPY SHORT limit 149.647が21:20 O bid 149.649ですでにmarketableだったことを確認した。
- 悲観OLHCは後続Lでも149.647約定だったため、この1件の直接fill価格利益は0だった。
  OHLCは後続Hの149.795まで待って14.8 pips有利に約定し、同じtimeout CLOSEでの孤立差分は
  約 `+¥7,977.72`。したがってOHLC側 `+¥283,770.99` もその分を含む過大表示である。

主要digest:

- scoreboard raw: `dc00742fe594ae66539f7246d06cb73e14bb352dec45a87d2cd70b30505eda2a`
- scoreboard canonical: `faf61fe2cdd42cb094041bbec9b5057e2f55dbe0f250e5342e49e3259ec6fd93`
- target ledger raw: `27ba890bd9e55aa83a4d31cb23e624da6b5555bdf2db3913f32c1c3493d2f248`
- target ledger tip: `5dbb34e21c8f2090af4ea30ea70561afca91276542f5971f1fd0f70fa4412dd8`

## 証明境界

60分解放は、すでに観測したTRAIN baselineの勝敗時間から選択された。期間も12日弱であり、
scoreboardの30日換算 `x4.61010409` は機械的な複利外挿にすぎない。realized-event DD
`9.30253%` は連続MTM DDではない。

また、旧台帳の算術が正しいことと実市場メカニクスが正しいことは別である。保存台帳上の
実現損益は再計算一致したが、60分解放は実際には61分で、注文時点ですでにmarketableな
指値・逆指値を次のH/Lまで遅延させ得た。よって `+¥166,990.89` は旧simulator規則の下での
自己整合TRAIN損益であり、「60分解放を備えた実市場忠実な損益」としては証明されない。

さらに実行manifestの `git_head=30c047f8d8fc47b41f806a63e33ad3cfefc57184` に対し、
実行時はdirty intermediate sourceを使っていた。`virtual_broker.py` のexact bytesは後のcommit
`7f0df0fa2c6bff23c876c72a731f38bad7dde298` から回収できるが、次の2本は全branch、worktree、
archive、reachable/unreachable Git objectを探索してもhashだけでbytesが残っていない。

- `bots/lab_bot.py`: `a3c27b93816bac4452941744683b08188fabb685adeafa060ac62f1fd2bd9713`
- `src/quant_rabbit/dojo_lab_provenance.py`: `d51e972ab0cc40448cee4e212fcb7d67c0f435fc17fc732ec67c03420a06be23`

したがって旧runのbit-for-bit再実行は不可能で、現行strict scorerにも昇格できない。
同じ戦略仮説は、現行の封印済みsource closure、連続MTM runner、独立corpus scorerで新しく
再走する。月次3倍の判定は、その後の固定28〜30日窓と未到来forward paperを別に要求する。
