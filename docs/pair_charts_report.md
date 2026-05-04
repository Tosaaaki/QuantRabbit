# Pair Charts Report

- Generated at UTC: `2026-05-04T04:32:12.346328+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `EUR_JPY` | `SHORT` | `0.062` | `0.938` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=38.6 RSI=44.0 ATR=17.2p BB=177.7p cloud=above); M15(TREND_DOWN, ADX=35.1 RSI=40.3 ATR=20.1p BB=102.9p cloud=above); H1(TREND_DOWN, ADX=46.5 RSI=42.6 ATR=38.6p BB=89.1p) |
| `GBP_JPY` | `SHORT` | `0.062` | `0.938` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=39.1 RSI=43.4 ATR=19.9p BB=204.5p cloud=above); M15(TREND_DOWN, ADX=30.3 RSI=40.7 ATR=23.8p BB=122.7p cloud=above); H1(TREND_DOWN, ADX=42.1 RSI=42.8 ATR=44.8p BB=103.2p) |
| `USD_JPY` | `SHORT` | `0.134` | `0.866` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=36.0 RSI=44.8 ATR=16.8p BB=170.8p cloud=above); M15(UNCLEAR, ADX=24.1 RSI=42.2 ATR=20.5p BB=102.3p cloud=above); H1(TREND_DOWN, ADX=43.9 RSI=44.8 ATR=40.0p BB=94.1p) |
| `AUD_JPY` | `SHORT` | `0.163` | `0.775` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=36.4 RSI=43.6 ATR=11.1p BB=108.3p cloud=above); M15(UNCLEAR, ADX=24.5 RSI=40.9 ATR=13.6p BB=63.8p cloud=above); H1(TREND_DOWN, ADX=28.7 RSI=46.6 ATR=25.3p BB=90.7p cloud=above) |
| `AUD_USD` | `LONG` | `0.675` | `0.263` | `TREND_UP` | AUD_USD TREND_UP; M5(RANGE, ADX=17.6 RSI=48.5 ATR=2.5p BB=10.1p cloud=below); M15(UNCLEAR, ADX=21.5 RSI=50.0 ATR=4.5p BB=10.9p cloud=below); H1(TREND_UP, ADX=26.5 RSI=55.4 ATR=9.7p BB=33.9p cloud=above) |
| `EUR_USD` | `LONG` | `0.548` | `0.318` | `UNCLEAR` | EUR_USD UNCLEAR; M5(UNCLEAR, ADX=22.1 RSI=49.1 ATR=2.6p BB=15.4p cloud=below); M15(UNCLEAR, ADX=21.6 RSI=49.9 ATR=5.0p BB=14.1p cloud=below); H1(UNCLEAR, ADX=19.5 RSI=48.7 ATR=12.8p BB=55.1p) |
| `GBP_USD` | `LONG` | `0.511` | `0.355` | `FAILURE_RISK` | GBP_USD FAILURE_RISK; M5(FAILURE_RISK, ADX=19.2 RSI=47.7 ATR=3.5p BB=18.8p cloud=below); M15(FAILURE_RISK, ADX=18.8 RSI=50.0 ATR=6.5p BB=18.5p cloud=below); H1(TREND_UP, ADX=25.2 RSI=48.9 ATR=16.4p BB=72.8p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
