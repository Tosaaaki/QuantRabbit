# Pair Charts Report

- Generated at UTC: `2026-05-04T07:59:44.602194+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=46.1 RSI=41.0 ATR=5.9p BB=27.9p cloud=below); M15(TREND_DOWN, ADX=45.3 RSI=37.8 ATR=11.3p BB=64.8p cloud=below); H1(TREND_DOWN, ADX=32.7 RSI=44.2 ATR=24.4p BB=61.7p) |
| `GBP_JPY` | `SHORT` | `0.037` | `0.963` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=43.4 RSI=38.4 ATR=8.9p BB=50.2p); M15(TREND_DOWN, ADX=42.2 RSI=40.4 ATR=19.7p BB=113.0p cloud=above); H1(TREND_DOWN, ADX=44.8 RSI=43.7 ATR=43.1p BB=99.0p) |
| `GBP_USD` | `SHORT` | `0.143` | `0.857` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=38.3 RSI=33.3 ATR=4.7p BB=38.0p cloud=below); M15(UNCLEAR, ADX=21.3 RSI=34.5 ATR=7.4p BB=34.8p cloud=below); H1(UNCLEAR, ADX=21.8 RSI=42.0 ATR=16.5p BB=74.8p) |
| `AUD_USD` | `SHORT` | `0.180` | `0.820` | `UNCLEAR` | AUD_USD UNCLEAR; M5(TREND_DOWN, ADX=41.9 RSI=36.8 ATR=3.2p BB=22.0p cloud=below); M15(UNCLEAR, ADX=23.8 RSI=32.8 ATR=4.9p BB=27.8p cloud=below); H1(UNCLEAR, ADX=23.1 RSI=41.6 ATR=10.1p BB=25.5p) |
| `EUR_JPY` | `SHORT` | `0.109` | `0.820` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=38.1 RSI=41.7 ATR=7.4p BB=30.6p); M15(TREND_DOWN, ADX=44.4 RSI=43.7 ATR=16.4p BB=94.4p cloud=above); H1(TREND_DOWN, ADX=48.3 RSI=45.6 ATR=36.7p BB=82.6p) |
| `EUR_USD` | `SHORT` | `0.188` | `0.812` | `UNCLEAR` | EUR_USD UNCLEAR; M5(TREND_DOWN, ADX=28.5 RSI=36.2 ATR=3.4p BB=21.8p cloud=below); M15(UNCLEAR, ADX=18.4 RSI=38.4 ATR=5.6p BB=19.2p cloud=below); H1(UNCLEAR, ADX=18.0 RSI=43.4 ATR=12.8p BB=55.7p cloud=below) |
| `USD_JPY` | `SHORT` | `0.368` | `0.498` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=22.4 RSI=52.1 ATR=6.5p BB=18.0p cloud=above); M15(TREND_UP, ADX=34.4 RSI=50.4 ATR=16.0p BB=91.7p cloud=above); H1(TREND_DOWN, ADX=45.0 RSI=49.1 ATR=37.1p BB=87.5p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
