# Capture Economics Report

- Generated at UTC: `2026-07-05T18:08:38.642697+00:00`
- Status: `NEGATIVE_EXPECTANCY`
- Trades (trader-attributed, realized): `229`
- Win rate: `59.8%`
- Avg win / avg loss: `418.0` / `1063.9` JPY
- Payoff ratio: `0.393` (breakeven at win rate: `0.672`)
- Expectancy: `-177.4` JPY/trade, net `-40616.9` JPY

## Repair Summary

- Dominant loss exit: `MARKET_ORDER_TRADE_CLOSE` net `-74151.8` JPY
- Strongest positive exit: `TAKE_PROFIT_ORDER` net `48804.8` JPY
- Payoff gap to breakeven: `0.279`

## Segment Repair Priorities

| pair | side | method | priority | n | TP n/gap | market-close net | net |
|---|---|---|---|---|---|---|---|
| `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK` | 30 | 20/0 | -15091.7 | -4056.9 |
| `GBP_USD` | `LONG` | `BREAKOUT_FAILURE` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 23 | 10/10 | -22478.7 | -17245.2 |
| `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 36 | 17/3 | -7636.3 | 3705.2 |
| `EUR_USD` | `LONG` | `TREND_CONTINUATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 4 | 2/18 | -3307.4 | -2768.4 |
| `AUD_JPY` | `LONG` | `BREAKOUT_FAILURE` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 13 | 2/18 | -3077.3 | -3240.2 |
| `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 12 | 6/14 | -3016.3 | -1605.4 |
| `EUR_USD` | `LONG` | `RANGE_ROTATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 6 | 3/17 | -2633.6 | -1923.2 |
| `EUR_USD` | `SHORT` | `RANGE_ROTATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 15 | 10/10 | -2151.0 | 2086.6 |
| `EUR_USD` | `SHORT` | `TREND_CONTINUATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 7 | 5/15 | -1202.1 | 211.3 |
| `EUR_JPY` | `LONG` | `RANGE_ROTATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 2 | 1/19 | -1071.9 | -416.7 |
| `USD_CAD` | `LONG` | `TREND_CONTINUATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 3 | 1/19 | -540.9 | -83.8 |
| `AUD_JPY` | `SHORT` | `TREND_CONTINUATION` | `COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK` | 2 | 1/19 | -146.8 | 241.2 |

## Action Items

- repair exit payoff asymmetry before treating the daily target as arithmetically reachable: payoff_ratio=0.393 breakeven=0.672
- contain MARKET_ORDER_TRADE_CLOSE drag (-74151.8 JPY): prefer attached TP, TP-rebalance, profit-side TAKE_PROFIT_MARKET, and require hard Gate A/B evidence for loss-side CLOSE
- preserve profitable TAKE_PROFIT_ORDER behavior while repairing negative exit buckets

## By exit reason

| exit_reason | n | win% | avg win | avg loss | net |
|---|---|---|---|---|---|
| `MARKET_ORDER_MARGIN_CLOSEOUT` | 4 | 0% | 0.0 | 2358.4 | -9433.7 |
| `MARKET_ORDER_TRADE_CLOSE` | 98 | 24% | 222.6 | 1074.2 | -74151.8 |
| `STOP_LOSS_ORDER` | 31 | 55% | 183.2 | 639.3 | -5836.2 |
| `TAKE_PROFIT_ORDER` | 96 | 100% | 508.4 | 0.0 | 48804.8 |

## By ISO week

| week | n | win% | payoff | net |
|---|---|---|---|---|
| `2026-W19` | 48 | 48% | 1.041 | -532.2 |
| `2026-W20` | 63 | 56% | 0.238 | -35463.3 |
| `2026-W21` | 20 | 95% | 0.65 | 7950.3 |
| `2026-W22` | 12 | 92% | 0.073 | -656.7 |
| `2026-W23` | 28 | 68% | 0.24 | -6571.9 |
| `2026-W24` | 26 | 46% | 1.285 | 555.1 |
| `2026-W25` | 18 | 61% | 0.413 | -1604.5 |
| `2026-W26` | 11 | 64% | 0.52 | -294.8 |
| `2026-W27` | 3 | 0% | 0.0 | -3998.9 |
