# Pair Charts Report

- Generated at UTC: `2026-05-04T08:14:18.523799+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `GBP_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=46.3 RSI=36.6 ATR=8.3p BB=45.9p); M15(TREND_DOWN, ADX=43.1 RSI=39.4 ATR=18.9p BB=103.6p); H1(TREND_DOWN, ADX=45.6 RSI=43.3 ATR=40.7p BB=101.9p) |
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=46.4 RSI=41.9 ATR=5.5p BB=21.7p); M15(TREND_DOWN, ADX=46.6 RSI=38.6 ATR=10.9p BB=58.1p cloud=below); H1(TREND_DOWN, ADX=34.0 RSI=44.4 ATR=23.1p BB=60.1p) |
| `GBP_USD` | `SHORT` | `0.143` | `0.857` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=37.6 RSI=27.8 ATR=4.6p BB=29.8p cloud=below); M15(UNCLEAR, ADX=22.4 RSI=29.6 ATR=7.4p BB=40.8p cloud=below); H1(UNCLEAR, ADX=21.2 RSI=40.0 ATR=15.8p BB=73.1p) |
| `EUR_JPY` | `SHORT` | `0.109` | `0.820` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=41.3 RSI=42.3 ATR=6.8p BB=33.7p); M15(TREND_DOWN, ADX=45.0 RSI=43.5 ATR=15.7p BB=85.9p cloud=above); H1(TREND_DOWN, ADX=48.8 RSI=45.5 ATR=34.5p BB=84.2p) |
| `EUR_USD` | `SHORT` | `0.188` | `0.812` | `UNCLEAR` | EUR_USD UNCLEAR; M5(TREND_DOWN, ADX=30.8 RSI=32.9 ATR=3.4p BB=21.1p cloud=below); M15(UNCLEAR, ADX=19.8 RSI=34.1 ATR=5.7p BB=22.7p cloud=below); H1(UNCLEAR, ADX=16.9 RSI=41.5 ATR=12.3p BB=54.7p cloud=below) |
| `AUD_USD` | `SHORT` | `0.205` | `0.795` | `TREND_DOWN` | AUD_USD TREND_DOWN; M5(TREND_DOWN, ADX=41.0 RSI=34.7 ATR=3.1p BB=14.4p cloud=below); M15(TREND_DOWN, ADX=25.2 RSI=29.9 ATR=4.8p BB=30.3p cloud=below); H1(UNCLEAR, ADX=22.4 RSI=40.4 ATR=9.6p BB=30.3p) |
| `USD_JPY` | `LONG` | `0.552` | `0.339` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=19.3 RSI=57.1 ATR=6.0p BB=13.1p cloud=above); M15(TREND_UP, ADX=34.2 RSI=52.7 ATR=15.3p BB=85.3p cloud=above); H1(TREND_DOWN, ADX=45.1 RSI=50.0 ATR=34.9p BB=85.1p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
