# QuantRabbit Crypto Strategy Lab｜収益性実測

観測時刻: 2026-07-28 19:33 JST

市場: bitbank Public Stream

資金: Spot/Margin各lane 10,000 JPY

権限: `NONE`（実注文・取消・決済・出金なし）

## 結論

- 稼働は継続しているが、収益性は未達。
- 常設baselineはSpot/Marginとも完了取引0、純損益0 JPY。
- Strategy Labの既存8 laneは合計269完了取引、純損益 -251.50 JPY。
  利益を出せた戦略はない。
- cooldownだけを変更した姉妹Paper 2 laneを加えた全10 laneは、
  314完了取引、純損益 -299.58 JPY、turnover 631,331.51 JPY。
- 現行戦略または候補をliveへ昇格しない。

## 常設baseline

| 戦略 | Mode | 完了取引 | 純損益 | Equity | DD | Guardian |
|---|---|---:|---:|---:|---:|---|
| FAST_MICROSTRUCTURE | Spot | 0 | 0 JPY | 10,000 JPY | 0 JPY | GREEN |
| FAST_MICROSTRUCTURE | Margin | 0 | 0 JPY | 10,000 JPY | 0 JPY | GREEN |

baselineの0円はStrategy Labの損益を表さない。安全gateを維持した比較基準である。

## Strategy Lab既存8 lane

| 戦略 | Mode | 完了取引 | 純損益 | PF | 期待値/取引 | 完了取引DD | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| RANGE_MAKER_REVERSION | Spot | 17 | -17.35 JPY | 0.0788 | -1.021 JPY | 18.09 JPY | 40,036.14 JPY |
| RANGE_MAKER_REVERSION | Margin | 33 | -30.14 JPY | 0.0959 | -0.913 JPY | 30.59 JPY | 77,503.22 JPY |
| BREAKOUT_CONFIRMATION | Spot | 0 | 0 JPY | N/A | N/A | 0 JPY | 0 JPY |
| BREAKOUT_CONFIRMATION | Margin | 0 | 0 JPY | N/A | N/A | 0 JPY | 0 JPY |
| TREND_PULLBACK_MAKER | Spot | 0 | 0 JPY | N/A | N/A | 0 JPY | 0 JPY |
| TREND_PULLBACK_MAKER | Margin | 0 | 0 JPY | N/A | N/A | 0 JPY | 0 JPY |
| ORDER_BOOK_FADE | Spot | 115 | -105.25 JPY | 0.0145 | -0.915 JPY | 105.25 JPY | 224,022.14 JPY |
| ORDER_BOOK_FADE | Margin | 104 | -98.76 JPY | 0.0054 | -0.950 JPY | 98.76 JPY | 198,271.57 JPY |

## 原因

1. 手数料負け。Order-book FadeはSpotで -105.25 JPY中104.00 JPY、
   Marginで -98.76 JPY中95.39 JPYが手数料。gross edgeが費用を吸収できていない。
2. 部分約定の細切れ。部分約定比率はRange Spot 85.7%、Range Margin
   97.5%、Fade Spot 95.8%、Fade Margin 96.7%。約定回数とturnoverが増え、
   最終的なTaker退出費用を回収できていない。
3. exit/regime不適合。Makerエントリー後の全完了取引が、MAX_HOLD、
   SIGNAL_INVALIDATED、STOP_LOSSによるTaker退出。Breakout/Trendは現在の
   相場窓で有効シグナル0件であり、閾値過剰または市場不適合を分離して検証する。

記録済みspread/adverse costが小さいことは、実コストがないことを意味しない。
現モデルで観測できない分を含むため、採用判定では過小評価リスクとして扱う。

## 一カテゴリ変更の実験

`ORDER_BOOK_FADE_COOLDOWN_5S` は元戦略からcooldownだけを750msから5秒へ
変更し、次の未使用実相場窓で並走した。

| Mode | 比較対象 | 完了取引 | 純損益 | PF | 期待値/取引 | 手数料/取引 | DD |
|---|---|---:|---:|---:|---:|---:|---:|
| Spot | 元戦略 | 18 | -18.56 JPY | 0.0113 | -1.031 JPY | 1.035 JPY | 18.56 JPY |
| Spot | cooldown 5秒 | 18 | -18.72 JPY | 0.00007 | -1.040 JPY | 0.931 JPY | 18.72 JPY |
| Margin | 元戦略 | 32 | -37.45 JPY | 0 | -1.170 JPY | 1.100 JPY | 37.45 JPY |
| Margin | cooldown 5秒 | 27 | -29.36 JPY | 0.00004 | -1.087 JPY | 1.030 JPY | 29.36 JPY |

判定: **不採用**。Spotは証拠収集を継続、Marginは早期棄却。手数料/取引は
下がったが、Spot/MarginともPFが1未満、期待値が負で採用条件を満たさない。
未来データは使用していない。

## 指標契約

- `completed_trade_count`: 1つのポジションが完全決済され、trade outboxへ記録された回数。
- `fill_count`: 部分約定を含む累積約定イベント数。
- 旧`trade_count`: `fill_count`と同義だったためdeprecated。
- `epoch_events_processed`: 現在の接続epochだけの市場イベント数。
- `service_events_processed_total`: 再接続・再起動をまたいだ累積市場イベント数。

従来`trade_count > events_processed`に見えたのは、累積fillと単一epochの市場イベントを
比較していたためで、完了取引数が市場イベント数を超えた意味ではない。

## 継続条件

- 既存baselineとStrategy Labを変更せず並走する。
- 変更は一度に1カテゴリだけとし、未使用の実相場窓で比較する。
- 採用には費用後PF>1、期待値>0、DD非悪化を3窓で再現する必要がある。
- 1トレード1行のローカルoutboxと定期集計を継続する。
- Notion route、Sheets、Slack connectorが確認できるまで外部送信だけfail closedする。
- `NO_EXECUTE=true`、`CRYPTO_LIVE_READY=false`、
  `WITHDRAWAL_ENABLED=false`、order authority `NONE`を維持する。
