# Pair Charts Report

- Generated at UTC: `2026-05-04T07:29:21.025881+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `AUD_JPY` | `SHORT` | `0.000` | `1.000` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_DOWN, ADX=43.3 RSI=36.9 ATR=6.2p BB=23.2p cloud=below); M15(TREND_DOWN, ADX=42.4 RSI=36.8 ATR=11.8p BB=67.8p cloud=below); H1(TREND_DOWN, ADX=32.5 RSI=44.4 ATR=23.9p BB=61.4p) |
| `GBP_JPY` | `SHORT` | `0.109` | `0.891` | `TREND_DOWN` | GBP_JPY TREND_DOWN; M5(TREND_DOWN, ADX=36.4 RSI=37.8 ATR=9.9p BB=32.2p); M15(TREND_DOWN, ADX=40.1 RSI=41.7 ATR=20.5p BB=119.3p cloud=above); H1(TREND_DOWN, ADX=44.6 RSI=44.5 ATR=41.9p BB=97.6p) |
| `GBP_USD` | `SHORT` | `0.180` | `0.820` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_DOWN, ADX=35.2 RSI=25.5 ATR=4.5p BB=36.7p cloud=below); M15(UNCLEAR, ADX=19.7 RSI=33.9 ATR=7.1p BB=26.3p cloud=below); H1(UNCLEAR, ADX=21.5 RSI=42.2 ATR=16.1p BB=74.7p) |
| `AUD_USD` | `SHORT` | `0.180` | `0.820` | `UNCLEAR` | AUD_USD UNCLEAR; M5(TREND_DOWN, ADX=36.4 RSI=26.3 ATR=3.0p BB=24.6p cloud=below); M15(FAILURE_RISK, ADX=20.5 RSI=28.7 ATR=4.9p BB=22.0p cloud=below); H1(UNCLEAR, ADX=22.9 RSI=40.8 ATR=9.9p BB=26.2p) |
| `EUR_USD` | `SHORT` | `0.214` | `0.786` | `UNCLEAR` | EUR_USD UNCLEAR; M5(UNCLEAR, ADX=23.6 RSI=32.2 ATR=3.4p BB=17.6p cloud=below); M15(RANGE, ADX=16.6 RSI=39.1 ATR=5.5p BB=15.2p cloud=below); H1(UNCLEAR, ADX=18.4 RSI=44.0 ATR=12.4p BB=55.4p) |
| `EUR_JPY` | `SHORT` | `0.250` | `0.641` | `TREND_DOWN` | EUR_JPY TREND_DOWN; M5(TREND_DOWN, ADX=31.8 RSI=44.5 ATR=8.2p BB=20.4p); M15(TREND_UP, ADX=43.2 RSI=46.1 ATR=17.2p BB=101.8p cloud=above); H1(TREND_DOWN, ADX=48.1 RSI=46.7 ATR=35.7p BB=81.5p) |
| `USD_JPY` | `LONG` | `0.468` | `0.423` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(UNCLEAR, ADX=24.2 RSI=56.6 ATR=7.4p BB=26.2p cloud=above); M15(TREND_UP, ADX=34.2 RSI=52.1 ATR=17.1p BB=99.4p cloud=above); H1(TREND_DOWN, ADX=45.0 RSI=49.8 ATR=36.8p BB=87.7p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
