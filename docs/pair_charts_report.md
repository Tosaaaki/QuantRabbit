# Pair Charts Report

- Generated at UTC: `2026-05-04T06:08:26.190735+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `GBP_USD` | `LONG` | `0.795` | `0.134` | `UNCLEAR` | GBP_USD UNCLEAR; M5(UNCLEAR, ADX=22.3 RSI=59.6 ATR=3.1p BB=10.2p cloud=above); M15(UNCLEAR, ADX=20.3 RSI=56.4 ATR=6.0p BB=17.8p cloud=above); H1(UNCLEAR, ADX=23.7 RSI=51.4 ATR=15.1p BB=72.6p) |
| `EUR_USD` | `LONG` | `0.675` | `0.211` | `RANGE` | EUR_USD RANGE; M5(RANGE, ADX=13.8 RSI=61.2 ATR=2.3p BB=6.5p cloud=above); M15(RANGE, ADX=16.7 RSI=54.9 ATR=4.6p BB=13.8p); H1(UNCLEAR, ADX=18.5 RSI=50.6 ATR=11.8p BB=55.0p) |
| `EUR_JPY` | `SHORT` | `0.234` | `0.657` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_UP, ADX=39.0 RSI=55.7 ATR=11.2p BB=21.3p cloud=above); M15(TREND_DOWN, ADX=42.9 RSI=49.4 ATR=19.4p BB=106.6p cloud=above); H1(TREND_DOWN, ADX=48.2 RSI=47.5 ATR=36.5p BB=81.1p) |
| `GBP_JPY` | `SHORT` | `0.234` | `0.657` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_UP, ADX=37.9 RSI=56.1 ATR=12.9p BB=30.2p cloud=above); M15(TREND_DOWN, ADX=38.1 RSI=50.1 ATR=22.7p BB=124.8p cloud=above); H1(TREND_DOWN, ADX=44.3 RSI=48.1 ATR=42.6p BB=93.8p) |
| `AUD_JPY` | `SHORT` | `0.237` | `0.637` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_UP, ADX=40.8 RSI=51.2 ATR=7.1p BB=6.1p cloud=above); M15(TREND_DOWN, ADX=36.9 RSI=45.8 ATR=12.7p BB=68.2p cloud=above); H1(TREND_DOWN, ADX=31.7 RSI=49.1 ATR=23.7p BB=66.8p cloud=above) |
| `USD_JPY` | `SHORT` | `0.234` | `0.632` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(TREND_UP, ADX=35.8 RSI=51.8 ATR=11.1p BB=16.1p cloud=above); M15(TREND_DOWN, ADX=33.2 RSI=47.5 ATR=19.8p BB=106.0p cloud=above); H1(TREND_DOWN, ADX=45.2 RSI=47.6 ATR=37.8p BB=89.4p) |
| `AUD_USD` | `SHORT` | `0.373` | `0.493` | `RANGE` | AUD_USD RANGE; M5(RANGE, ADX=10.2 RSI=47.5 ATR=2.1p BB=6.1p cloud=below); M15(RANGE, ADX=15.6 RSI=47.3 ATR=4.2p BB=10.3p cloud=below); H1(UNCLEAR, ADX=24.5 RSI=53.2 ATR=9.2p BB=21.3p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
