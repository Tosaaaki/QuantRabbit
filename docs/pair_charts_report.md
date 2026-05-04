# Pair Charts Report

- Generated at UTC: `2026-05-04T07:46:51.278179+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=46.4 RSI=37.5 ATR=5.8p BB=31.1p cloud=below); M15(TREND_DOWN, ADX=45.3 RSI=37.1 ATR=11.1p BB=65.0p cloud=below); H1(TREND_DOWN, ADX=32.7 RSI=43.9 ATR=24.4p BB=62.0p) |
| `GBP_JPY` | `SHORT` | `0.037` | `0.963` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=40.6 RSI=35.8 ATR=9.3p BB=49.5p); M15(TREND_DOWN, ADX=42.2 RSI=40.2 ATR=19.3p BB=113.1p cloud=above); H1(TREND_DOWN, ADX=44.8 RSI=43.6 ATR=43.1p BB=99.2p) |
| `GBP_USD` | `SHORT` | `0.143` | `0.857` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=38.0 RSI=28.6 ATR=4.3p BB=40.8p cloud=below); M15(UNCLEAR, ADX=21.2 RSI=33.7 ATR=6.9p BB=35.0p cloud=below); H1(UNCLEAR, ADX=21.8 RSI=41.8 ATR=16.4p BB=75.0p) |
| `AUD_USD` | `SHORT` | `0.180` | `0.820` | `UNCLEAR` | AUD_USD UNCLEAR; M5(TREND_DOWN, ADX=42.1 RSI=32.2 ATR=2.9p BB=25.3p cloud=below); M15(UNCLEAR, ADX=24.3 RSI=31.6 ATR=4.7p BB=28.0p cloud=below); H1(UNCLEAR, ADX=23.1 RSI=41.2 ATR=10.1p BB=25.8p) |
| `EUR_JPY` | `SHORT` | `0.109` | `0.820` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=35.7 RSI=41.1 ATR=7.8p BB=26.5p); M15(TREND_DOWN, ADX=44.4 RSI=44.0 ATR=16.2p BB=94.3p cloud=above); H1(TREND_DOWN, ADX=48.3 RSI=45.7 ATR=36.7p BB=82.4p) |
| `EUR_USD` | `SHORT` | `0.188` | `0.812` | `UNCLEAR` | EUR_USD UNCLEAR; M5(TREND_DOWN, ADX=26.9 RSI=34.9 ATR=3.3p BB=21.9p cloud=below); M15(RANGE, ADX=17.9 RSI=39.2 ATR=5.3p BB=19.0p cloud=below); H1(UNCLEAR, ADX=18.2 RSI=43.7 ATR=12.6p BB=55.6p cloud=below) |
| `USD_JPY` | `SHORT` | `0.368` | `0.498` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=23.0 RSI=52.0 ATR=6.7p BB=19.4p cloud=above); M15(TREND_UP, ADX=34.5 RSI=50.3 ATR=15.6p BB=91.7p cloud=above); H1(TREND_DOWN, ADX=45.0 RSI=49.1 ATR=37.1p BB=87.5p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
