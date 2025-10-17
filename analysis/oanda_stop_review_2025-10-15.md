# 2025-10-15 Loss Review

## Data Sources
- OANDA transactions via REST idrange (09 pages, 1891 fills)
- M1 candles 2025-10-14T23:00Z – 2025-10-16T00:00Z
- Local logs/trades.db (incomplete after 09:24Z)

## Reconciliation
- OANDA ORDER_FILL P/L on 2025-10-15: -6,500.07 JPY across 572 losing fills.
- Local trades.db reports only -1,650.64 JPY (102 losing trades) before 09:24Z.
- Desync indicates trade_sync stopped; remainder of day missing from sqlite.

## Stop-Loss Behaviour
- Average stop distance: 3.02 pips; average 15m rebound: 7.06 pips (x2.3).
- 15m recovery rate (price revisited entry or better): 99.5% ; 30m: 99.7%.
- Within 5m, price exceeded the stop distance again 51.4% of the time.
- Overall, 72.4% of stops would have reached break-even with a stop at least 1× wider; 53% exceeded 2×.

### By Strategy
strategy|count|avg_stop|avg_future15|avg_loss
BB_RSI|2|8.199999999999363|1.650000000000773|8.199999999999363
MicroTrendPullback|6|7.050000000000978|5.016666666666936|7.050000000000978
ScalpMeanRevert|558|2.9338709677419375|7.131541218638076|2.9338709677419375
TrendMA|6|5.450000000000443|4.483333333332951|5.450000000000443

### By Close Reason
reason|count|avg_future15|avg_future30|avg_loss
MARKET_ORDER|2|1.300000000000523|1.300000000000523|3.100000000000591
MARKET_ORDER_MARGIN_CLOSEOUT|35|9.834285714285329|13.062857142857151|0.7771428571428325
MARKET_ORDER_TRADE_CLOSE|5|4.819999999999709|7.439999999999145|5.4200000000003
STOP_LOSS_ORDER|530|6.922264150943511|9.43377358490568|3.147169811320767

### Dense Stop-Out Clusters
````markdown
count | window_utc | strategies | avg_stop | avg_future15
--- | --- | --- | --- | ---
7 | 00:42:13–00:46:04 | BB_RSI,ScalpMeanRevert | 3.63 | 4.71
8 | 01:07:16–01:16:26 | ScalpMeanRevert | 2.80 | 9.96
10 | 02:00:03–02:09:52 | ScalpMeanRevert,MicroTrendPullback | 2.07 | 11.19
10 | 04:18:24–04:27:30 | ScalpMeanRevert | 1.99 | 4.77
10 | 06:03:04–06:11:33 | ScalpMeanRevert,MicroTrendPullback | 4.14 | 12.40
````

## Notable Issues
1. Margin closeouts (35 fills) happened with <1 pip loss but 15m rebound >9.8 pips; exposure size triggered emergency exits at local extremes.
2. ScalpMeanRevert generated 558 / 572 losses with mean stop 2.93 pips; minute candles regularly oscillated >10 pips afterwards.
3. Trade logging stalled at 07:24:49Z (desync_auto_close entries); requires fix before further tuning.
