# Pair Charts Report

- Generated at UTC: `2026-05-04T08:06:51.610146+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `GBP_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=45.4 RSI=35.7 ATR=8.6p BB=48.1p); M15(TREND_DOWN, ADX=43.1 RSI=39.3 ATR=18.8p BB=103.7p); H1(TREND_DOWN, ADX=45.6 RSI=43.3 ATR=40.6p BB=102.0p) |
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=46.5 RSI=38.4 ATR=5.7p BB=23.9p); M15(TREND_DOWN, ADX=46.6 RSI=36.7 ATR=10.9p BB=58.6p cloud=below); H1(TREND_DOWN, ADX=34.0 RSI=43.5 ATR=23.0p BB=61.1p) |
| `EUR_JPY` | `SHORT` | `0.109` | `0.891` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=40.7 RSI=38.1 ATR=7.0p BB=34.0p); M15(TREND_DOWN, ADX=45.0 RSI=42.0 ATR=15.6p BB=86.1p cloud=above); H1(TREND_DOWN, ADX=48.8 RSI=44.9 ATR=34.5p BB=84.7p) |
| `GBP_USD` | `SHORT` | `0.143` | `0.857` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=38.0 RSI=23.1 ATR=4.5p BB=33.6p cloud=below); M15(UNCLEAR, ADX=22.4 RSI=29.0 ATR=7.4p BB=41.3p cloud=below); H1(UNCLEAR, ADX=21.2 RSI=39.6 ATR=15.8p BB=73.5p) |
| `AUD_USD` | `SHORT` | `0.180` | `0.820` | `TREND_DOWN` | AUD_USD TREND_DOWN; M5(TREND_DOWN, ADX=41.7 RSI=28.7 ATR=3.0p BB=17.3p cloud=below); M15(TREND_DOWN, ADX=25.2 RSI=27.7 ATR=4.8p BB=31.2p cloud=below); H1(UNCLEAR, ADX=22.4 RSI=39.2 ATR=9.6p BB=31.2p) |
| `EUR_USD` | `SHORT` | `0.188` | `0.812` | `UNCLEAR` | EUR_USD UNCLEAR; M5(TREND_DOWN, ADX=30.5 RSI=25.1 ATR=3.4p BB=22.7p cloud=below); M15(UNCLEAR, ADX=19.8 RSI=31.9 ATR=5.7p BB=23.9p cloud=below); H1(UNCLEAR, ADX=16.9 RSI=40.3 ATR=12.3p BB=55.4p cloud=below) |
| `USD_JPY` | `LONG` | `0.552` | `0.339` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=20.3 RSI=57.2 ATR=6.2p BB=14.7p cloud=above); M15(TREND_UP, ADX=34.2 RSI=52.7 ATR=15.3p BB=85.3p cloud=above); H1(TREND_DOWN, ADX=45.1 RSI=50.0 ATR=34.9p BB=85.1p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
