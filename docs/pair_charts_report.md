# Pair Charts Report

- Generated at UTC: `2026-05-04T06:28:25.558676+00:00`
- Timeframes: `M5,M15,H1`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Regime | Story |
|---|---|---|---|---|---|
| `GBP_USD` | `LONG` | `0.795` | `0.134` | `UNCLEAR` | GBP_USD UNCLEAR; M5(TREND_UP, ADX=28.7 RSI=65.8 ATR=3.3p BB=14.5p cloud=above); M15(FAILURE_RISK, ADX=21.7 RSI=59.9 ATR=6.0p BB=18.6p cloud=above); H1(UNCLEAR, ADX=23.9 RSI=52.8 ATR=15.3p BB=72.5p) |
| `EUR_USD` | `LONG` | `0.757` | `0.171` | `UNCLEAR` | EUR_USD UNCLEAR; M5(UNCLEAR, ADX=19.8 RSI=68.2 ATR=2.4p BB=12.0p cloud=above); M15(RANGE, ADX=17.8 RSI=58.8 ATR=4.6p BB=15.1p cloud=above); H1(UNCLEAR, ADX=18.7 RSI=52.4 ATR=12.1p BB=54.8p) |
| `AUD_USD` | `LONG` | `0.664` | `0.270` | `RANGE` | AUD_USD RANGE; M5(RANGE, ADX=9.9 RSI=54.2 ATR=2.1p BB=6.2p); M15(RANGE, ADX=14.4 RSI=50.5 ATR=4.1p BB=9.2p cloud=below); H1(UNCLEAR, ADX=24.5 RSI=55.0 ATR=9.2p BB=21.1p cloud=above) |
| `USD_JPY` | `SHORT` | `0.246` | `0.620` | `TREND_DOWN` | USD_JPY TREND_DOWN; M5(TREND_UP, ADX=36.4 RSI=49.1 ATR=9.9p BB=13.6p cloud=above); M15(TREND_DOWN, ADX=34.1 RSI=46.7 ATR=19.3p BB=105.9p cloud=above); H1(TREND_DOWN, ADX=45.2 RSI=47.2 ATR=38.0p BB=89.4p) |
| `AUD_JPY` | `SHORT` | `0.275` | `0.600` | `TREND_DOWN` | AUD_JPY TREND_DOWN; M5(TREND_UP, ADX=39.1 RSI=50.7 ATR=6.6p BB=10.6p cloud=above); M15(TREND_DOWN, ADX=37.3 RSI=46.3 ATR=12.6p BB=68.2p cloud=above); H1(TREND_DOWN, ADX=31.4 RSI=49.2 ATR=24.1p BB=66.8p cloud=above) |
| `EUR_JPY` | `SHORT` | `0.371` | `0.520` | `TREND_UP` | EUR_JPY TREND_UP; M5(TREND_UP, ADX=35.7 RSI=55.0 ATR=10.4p BB=26.9p cloud=above); M15(TREND_UP, ADX=42.5 RSI=50.0 ATR=19.2p BB=106.5p cloud=above); H1(TREND_DOWN, ADX=47.9 RSI=48.0 ATR=37.1p BB=80.9p) |
| `GBP_JPY` | `SHORT` | `0.371` | `0.520` | `TREND_UP` | GBP_JPY TREND_UP; M5(TREND_UP, ADX=35.5 RSI=55.0 ATR=12.2p BB=33.0p cloud=above); M15(TREND_UP, ADX=37.9 RSI=50.6 ATR=22.5p BB=124.8p cloud=above); H1(TREND_DOWN, ADX=43.9 RSI=48.4 ATR=43.3p BB=93.7p) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
