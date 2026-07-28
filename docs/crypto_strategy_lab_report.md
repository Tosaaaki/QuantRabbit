# QuantRabbit Crypto Strategy Lab｜初期実測

観測時刻: 2026-07-28 18:54 JST

市場: bitbank Public Stream

資金: Spot/Margin各戦略10,000 JPY

権限: `NONE`（実注文・取消・決済・出金なし）

## 初期結果

| 戦略 | Spot完了取引 | Spot純損益 | Spot PF | Margin完了取引 | Margin純損益 | Margin PF |
|---|---:|---:|---:|---:|---:|---:|
| RANGE_MAKER_REVERSION | 3 | -2.19 JPY | 0.25 | 5 | -3.51 JPY | 0.17 |
| BREAKOUT_CONFIRMATION | 0 | 0 JPY | N/A | 0 | 0 JPY | N/A |
| TREND_PULLBACK_MAKER | 0 | 0 JPY | N/A | 0 | 0 JPY | N/A |
| ORDER_BOOK_FADE | 9 | -3.80 JPY | 0.12 | 3 | -0.21 JPY | 0.0005 |

全戦略とも採用条件（費用後PF>1、期待値>0、DD非悪化、3つの未使用相場窓で再現）を満たしていない。

## 判定

- Range MakerとOrder-book Fadeは取引機会を生成したが、費用後期待値が負。
- BreakoutとTrend Pullbackは初期窓で取引機会を生成していない。
- Maker部分約定が完了しない問題は、期限切れ・損切り・シグナル無効化時のTaker決済へ切り替え、完了取引として評価する。
- 現行FAST_MICROSTRUCTUREは変更せず、各戦略を独立台帳・独立プロセスで比較する。
- 未来データ、Private API、取引所注文経路は使用しない。

## 継続条件

- 各Spot/Margin戦略を独立launchd agentで継続実行する。
- 5分ごとにローカルRCAを生成し、外部connector障害は取引Botへ波及させない。
- 各lane 30完了取引、3つの独立相場窓を満たすまで採用しない。
- 変更する戦略カテゴリは一度に1つとし、現行baselineを保持する。
