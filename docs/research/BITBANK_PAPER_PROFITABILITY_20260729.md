# QuantRabbit｜bitbank Paper黒字化調査・隔離比較

## 結論

2026-07-29 11:04 JST時点で、既存10レーンの確定損益は
`-1,639.46 JPY`、費用後PFは`0.0468`、期待値は
`-1.0344 JPY/trade`だった。板フェード2系統4レーンだけで
`-1,309.72 JPY`、全損失の`79.9%`を占める。板フェード停止は有力な
停止候補だが、除外後も`-329.74 JPY`、PF`0.1295`であり、黒字化案としては
採用できない。

maker出口、エントリー方向・閾値、regime選択、動的サイズ、6秒途中離脱も
費用後PFと期待値のゲートを通らなかった。ユーザー判断後、2026-07-29に
`ORDER_BOOK_FADE`と`ORDER_BOOK_FADE_COOLDOWN_5S`のSpot/Margin 4レーンだけを
新規エントリー`QUARANTINE`へ移した。既存成績・台帳・未決済Paperポジションの
現行risk contractは保持し、Paper全体と他レーンは継続する。

## 安全境界

- `NO_EXECUTE=true`
- `CRYPTO_LIVE_READY=false`
- `WITHDRAWAL_ENABLED=false`
- `CRYPTO_ORDER_AUTHORITY=NONE`
- 注文・取消・決済・出金API呼び出しなし
- 実口座変更なし
- Paper全体・他レーン・記録・改善ループは継続
- 板フェード4レーンは永続台帳から建玉を復元する短い再起動だけを行い、
  新規`ENTER`のみ遮断
- 未決済Paperポジションの強制決済なし
- OANDAから再利用したものは安全契約、append-only台帳、評価指標、
  未使用窓ゲートであり、戦略ロジックは移植していない

## 実測

operation_id:
`599a300ebc3ffbf0186ad1e40bfeb8900e86a3e21a5ab1ba53d2c40aa7bc12d4`

| 対象 | 取引数 | Gross JPY | 費用 JPY | Net JPY | PF | 期待値 JPY | DD JPY |
|---|---:|---:|---:|---:|---:|---:|---:|
| 既存10レーン | 1,585 | -57.73 | 1,581.66 | -1,639.46 | 0.0468 | -1.0344 | 1,640.66 |
| 板フェード2系統 | 1,310 | -54.41 | 1,255.25 | -1,309.72 | 0.0235 | -0.9998 | 1,309.72 |
| 板フェード除外 | 275 | -3.32 | 326.41 | -329.74 | 0.1295 | -1.1991 | 330.95 |

主因は次の3点である。

1. `FEE_DRAG_DOMINATES_GROSS_EDGE`: Gross自体が負で、さらに
   `1,581.66 JPY`の費用が加わった。
2. `PARTIAL_FILL_CHURN`: maker部分約定が大量に発生し、同一ポジションで
   Paper fillが反復した。
3. `MAKER_ENTRY_TAKER_EXIT_OVERTRADING`: 強制終了でtaker費用が反復した。

## 1カテゴリ隔離比較

| カテゴリ | 取引数 | Net JPY | 費用 JPY | PF | 期待値 JPY | DD JPY | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 板フェード停止候補 | 275 | -329.74 | 326.41 | 0.1295 | -1.1991 | 330.95 | 停止候補、未適用 |
| maker/taker整合 | 12 | -6.71 | 4.64 | 0.2395 | -0.5591 | 7.72 | 不採用、1前向き窓のみ |
| 方向・閾値 | 200 | -271.04 | 267.05 | 0.1151 | -1.3552 | 272.53 | 不採用 |
| regime別戦略 | 275 | -329.74 | 326.41 | 0.1295 | -1.1991 | 330.95 | 不採用 |
| 動的サイズ | 1,585 | -982.47 | 945.91 | 0.0531 | -0.6199 | 983.82 | 不採用 |
| 6秒途中離脱 | 1,585 | -1,614.02 | 1,581.66 | 0.0482 | -1.0183 | 1,617.02 | 不採用 |

maker出口だけが既存baselineと同時並行の前向きPaperで、それ以外の12〜13時間窓は
後付けの因果スクリーニングである。後付け区間を「未使用窓」と数えず、採用には
別途3以上の前向き未使用窓と各レーン30件以上を要求する。

途中離脱は、各ポジションの6秒以後で最初に記録されたquote-state PnLを用い、
実現済み費用を全額残す保守的近似である。queue-awareな約定再生ではないため、
不採用判定の補助証拠に限定する。

## 板フェード4レーンの新規エントリー隔離

制御正本は`config/crypto_paper_entry_control_v1.json`である。各Shadow processは
同ファイルを継続readbackし、次を満たさない設定はフラット時の新規entryだけを
fail-closedする。

- schema: `QR_CRYPTO_ENTRY_CONTROL_V1`
- status: `QUARANTINE_NEW_ENTRIES`
- existing position policy: `RISK_CONTRACT`
- authority: `NONE`
- live permission: `false`

変更前には4レーンすべてでBTC/JPY・ETH/JPYのPaper建玉が残っていた。したがって
process停止による状態初期化は使わず、append-only `PAPER_STATE`から同じ建玉、
平均取得価格、費用、DD、round tripを復元した。フラット時は
`ENTRY_QUARANTINED_NEW_ENTRIES`、建玉ありでは従来のTP/SL、MAX_HOLD、
signal invalidation、Paper margin contractを評価し続ける。

各ledgerにはpolicy SHA、mode、strategy、`new_entries_allowed=false`、
`existing_position_policy=RISK_CONTRACT`を
`ENTRY_CONTROL_READBACK`として重複なしで記録する。

`QUEUE_FLOW_MICROPRICE_MAKER`は`FORWARD_PAPER_ONLY`のままであり、
circuitが`NONE`の新規未使用窓以外では開始しない。費用後期待値`>0`、PF`>1`、
DD非悪化が複数の独立窓で再現するまで採用しない。

### 変更後readback

2026-07-29 11:44 JST、4レーンは制御正本SHA
`0868e66deca18ecec47c535a9835758923378f3f9805d7377e2c7af5a91c5c32`を
readbackし、全て`RUNNING`へ復帰した。全Shadow serviceは14 processのままで、
隔離対象外10 processのPIDは変わっていない。

| レーン | open Paper | 確定取引 | 費用 JPY | Net JPY | DD JPY | control後OPEN |
|---|---:|---:|---:|---:|---:|---:|
| Fade Spot | 2 | 198 | 164.08 | -169.48 | 1,542.46 | 0 |
| Fade Margin | 2 | 522 | 511.63 | -533.26 | 2,985.46 | 0 |
| Cooldown Spot | 2 | 80 | 67.31 | -73.94 | 2,544.48 | 0 |
| Cooldown Margin | 2 | 510 | 510.61 | -526.81 | 3,296.05 | 0 |

control event以後の`PAPER_ORDER`と`PAPER_FILL`をledger sequenceで照合し、
`position_effect=OPEN`は4レーンとも0件だった。一方、既存の微小残存建玉には
`CLOSE`だけが継続している。強制決済、確定取引の増加、費用の増加、DD更新はなく、
`forced_liquidation_count=0`を維持した。安全readbackも4レーン全てで
`NO_EXECUTE=true`、`CRYPTO_LIVE_READY=false`、`WITHDRAWAL_ENABLED=false`、
broker mutation `false`、order authority `NONE`だった。

起動時のappend-only ledger検証とtrade outbox復旧は、検証済みcheckpointと
outbox末尾sequence以後だけを読むようにした。これにより大規模ledgerでも
既存成績・台帳を変更せず短時間でrisk contractを再開できる。outboxの欠損時は
従来どおりledgerから復旧し、不正なsequenceはfail-closedする。

## bitbank固有仕様の確認

公式Public StreamはSocket.IO 4.xで、`ticker_{pair}`、
`transactions_{pair}`、`depth_whole_{pair}`、`depth_diff_{pair}`を提供する。
diff数量は絶対値で、0は価格レベル削除を表す。sequenceは単調増加だが連番とは
限らない。ローカル板はwholeとdiffを同時購読し、diffをbufferし、wholeの
`sequenceId`より大きいdiffだけを昇順適用する必要がある。diffはbest付近
約200価格レベルに限られるため、wholeでの置換を省略できない。

公式`GET /spot/pairs`の2026-07-29 11:04 JST観測値は次のとおり。
手数料や注文制約は固定値として埋め込まず、各Paper開始時に同APIから取得する。

| Pair | Spot maker | Spot taker | Margin maker | Margin taker | 日次金利 | 最小数量 | 指値最大 | 成行最大 | price/amount桁 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BTC/JPY | 0 | 0.10% | 0 | 0.10% | 0.04% | 0.0001 | 1,000 | 10 | 0 / 4 |
| XRP/JPY | -0.02% | 0.12% | -0.02% | 0.12% | 0.04% | 0.0001 | 40,000,000 | 400,000 | 3 / 4 |

取引所statusは`NORMAL/BUSY/VERY_BUSY/HALT`を取り、混雑時は最小数量が上がり得る。
PostOnlyはlimitだけに指定でき、circuit breaker modeが`NONE`でない場合は
`false`扱いとなる。したがってmaker前提の期待費用は、通常モード確認と
PostOnly成立をPaperモデルに含めるまで過信しない。

## 一次研究からの候補

- Queue imbalance: one-tick先の方向予測力は報告されているが、NASDAQ株式の
  結果であり、bitbankで再検証が必要。
- Order-flow imbalance: 短期価格変化との関係はdepthで変わる。transactionsと
  whole+diffを同じ時点までで集計し、同一区間の価格変化を特徴量へ混ぜない。
- Microprice: midより短期推定に有用な可能性がある。spreadと板不均衡から計算し、
  将来whole/diffを参照しない。
- Inventory-aware quoting/adaptive spread: 理論上は有用だが、Poisson約定や
  理想的queueを仮定すると利益を過大評価する。現在の固定maker部分約定モデルでは
  採用せず、queue-aware Paper完成後に再評価する。
- Cross-pair/lead-lag: clock同期、多重検定、同時受信遅延の監査が先であり、
  現時点では保留する。

次の独立Paper優先候補は`QUEUE_FLOW_MICROPRICE_MAKER`とする。
板不均衡、microprice、transactionsのsigned flowが同方向のときだけ、
maker entryとmaker-first exitを許す。失敗した板フェードの逆張り方向を
そのまま反転するのではなく、bitbank固有の配信を用いた新規ロジックとして実装する。

同候補をBTC/JPY・ETH/JPY、Spot/Margin、同時20秒の独立Public Stream Paperで
確認した。Spotは25イベント、decision p95 `8.708 µs`、Marginは24イベント、
decision p95 `11.042 µs`だった。両ペアとも公式circuit modeが`RESUMPTION`で、
Guardianは`HALT`、取引0、費用0、損益0、DD 0、PF未定義となった。これは利益証拠
ではなく、安全境界が働いた証拠である。全選択ペアが`NONE`へ戻るまで再開しない。

2026-07-29 11:45 JST、両ペアのcircuit modeが`NONE`へ戻ったことを同じPublic
Streamで確認し、新規未使用窓1をSpot/Margin同時45秒で開始した。

| Mode | events | completed trades | 費用 JPY | Net JPY | PF | DD JPY | decision p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spot | 352 | 0 | 0 | 0 | 未定義 | 0 | 338.333 µs |
| Margin | 352 | 0 | -0.25 | +0.2497 MTM | 未定義 | 0 | 336.209 µs |

Spotは全てWAIT、Marginはmaker部分約定1件で終了時に未決済Paper shortが1件残った。
確定取引が0のため、費用後期待値もPFも判定不能である。Guardianはclock skew観測で
`RESTRICT`だったため、窓1は利益再現性の証拠に数えず`FORWARD_PAPER_ONLY`を維持する。
別の未使用窓でも費用後期待値`>0`、PF`>1`、DD非悪化を満たすまでは採用しない。

## OSS比較・ライセンス監査

| OSS | License | 参考にする資産 | bitbank適合 |
|---|---|---|---|
| Hummingbot | Apache-2.0 | MM/arbitrage構成、connector分離 | native adapter未確認 |
| Cryptofeed | project独自BSD-4-Clause相当 | L2/trades正規化、feed callback | native adapter未確認 |
| NautilusTrader | LGPL-3.0 | deterministic replay、queue-aware fill | native adapter未確認 |
| Freqtrade | GPL-3.0 | dry-run、walk-forward、過剰適合対策 | CCXT依存、native適合未確認 |
| sstoikov/microprice sample | visible licenseなし | 論文式の確認のみ | コード利用禁止 |

外部コードはclone、install、import、実行、コピーしていない。アイデアもbitbankの
仕様と自前Paper台帳で独立に検証し、ライセンス適合だけで採用しない。

## 参照先

- bitbank公式API: https://github.com/bitbankinc/bitbank-api-docs
- bitbank Public Stream:
  https://github.com/bitbankinc/bitbank-api-docs/blob/master/public-stream.md
- bitbank REST:
  https://github.com/bitbankinc/bitbank-api-docs/blob/master/rest-api.md
- Queue Imbalance: https://arxiv.org/abs/1512.03492
- Order-flow Imbalance: https://arxiv.org/abs/1011.6402
- Microprice: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
- Avellaneda–Stoikov:
  https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf
- Weakly Consistent LOB: https://arxiv.org/abs/1903.07222
- Hummingbot: https://github.com/hummingbot/hummingbot
- Cryptofeed: https://github.com/bmoscon/cryptofeed
- NautilusTrader: https://github.com/nautechsystems/nautilus_trader
- Freqtrade: https://github.com/freqtrade/freqtrade

## ロールバック

新規entryを再開するには、制御正本の対象strategyを`ACTIVE`へ変える別の明示承認と
再検証が必要である。コードをrevertするだけでは運用判断を解除しない。
既存ledgerは削除・巻き戻しせず、`ENTRY_CONTROL_READBACK`を含めて履歴として
保持する。生成済み分析だけを破棄する場合は
`data/crypto/profitability-study-20260729/`を対象とし、runtime ledgerへは触れない。
