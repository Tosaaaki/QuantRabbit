# Pair Charts Report

- Generated at UTC: `2026-05-04T07:53:40.724659+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `GBP_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=41.9 RSI=34.5 ATR=8.9p BB=50.6p); M15(TREND_DOWN, ADX=42.2 RSI=39.6 ATR=19.4p BB=113.3p); H1(TREND_DOWN, ADX=44.8 RSI=43.4 ATR=43.1p BB=99.6p) |
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=45.9 RSI=38.6 ATR=5.8p BB=29.5p cloud=below); M15(TREND_DOWN, ADX=45.3 RSI=37.2 ATR=11.3p BB=65.0p cloud=below); H1(TREND_DOWN, ADX=32.7 RSI=43.9 ATR=24.4p BB=62.0p) |
| `GBP_USD` | `SHORT` | `0.143` | `0.857` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=39.1 RSI=24.0 ATR=4.4p BB=40.9p cloud=below); M15(UNCLEAR, ADX=21.3 RSI=30.9 ATR=7.2p BB=36.6p cloud=below); H1(UNCLEAR, ADX=21.8 RSI=40.6 ATR=16.5p BB=76.1p) |
| `AUD_USD` | `SHORT` | `0.180` | `0.820` | `UNCLEAR` | AUD_USD UNCLEAR; M5(TREND_DOWN, ADX=42.3 RSI=31.0 ATR=3.0p BB=24.2p cloud=below); M15(UNCLEAR, ADX=24.3 RSI=27.7 ATR=4.8p BB=28.7p cloud=below); H1(UNCLEAR, ADX=23.1 RSI=40.0 ATR=10.1p BB=26.8p) |
| `EUR_JPY` | `SHORT` | `0.109` | `0.820` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=36.7 RSI=39.8 ATR=7.4p BB=28.6p); M15(TREND_DOWN, ADX=44.4 RSI=43.4 ATR=16.3p BB=94.4p cloud=above); H1(TREND_DOWN, ADX=48.3 RSI=45.5 ATR=36.7p BB=82.7p) |
| `EUR_USD` | `SHORT` | `0.188` | `0.812` | `UNCLEAR` | EUR_USD UNCLEAR; M5(TREND_DOWN, ADX=28.2 RSI=28.9 ATR=3.2p BB=23.1p cloud=below); M15(UNCLEAR, ADX=18.2 RSI=35.0 ATR=5.4p BB=20.5p cloud=below); H1(UNCLEAR, ADX=18.0 RSI=42.0 ATR=12.7p BB=56.4p cloud=below) |
| `USD_JPY` | `SHORT` | `0.393` | `0.498` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=22.4 RSI=56.9 ATR=6.6p BB=19.1p cloud=above); M15(TREND_UP, ADX=34.4 RSI=52.3 ATR=15.8p BB=91.9p cloud=above); H1(TREND_DOWN, ADX=45.0 RSI=49.8 ATR=37.1p BB=87.7p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
