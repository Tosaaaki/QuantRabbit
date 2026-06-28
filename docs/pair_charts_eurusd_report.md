# Pair Charts Report

- Generated at UTC: `2026-05-29T11:07:26.395922+00:00`
- Timeframes: `M1,M5,M15,M30,H1,H4,D`
- Candles per timeframe: `200`

## Pair Score Table

| Pair | Side | Long | Short | Momentum | Regime | Story |
|---|---|---|---|---|---|---|
| `EUR_USD` | `SHORT` | `0.438` | `0.452` | `-` | `RANGE` | EUR_USD RANGE; M1(RANGE, ADX=13.9 RSI=55.7 ATR=1.1p BB=4.5p %R=-22.5 MFI=73.8 AroonOsc=93 Chop=56 ST=+ Read=TRANSITION:0.25 q=NORMAL cloud=above struct=BOS_DOWN@1.1641:wick); M5(RANGE, ADX=17.4 RSI=58.8 ATR=2.6p BB=11.7p %R=-15.3 MFI=66.3 AroonOsc=14 Chop=60 ST=+ Read=TREND_WEAK:0.33 q=QUIET cloud=above struct=CHOCH_UP@1.1639); M15(UNCLEAR, ADX=22.7 RSI=53.1 ATR=5.0p BB=22.4p %R=-5.4 MFI=60.8 AroonOsc=36 Chop=52 ST=- Read=TRANSITION:0.25 q=NORMAL cloud=above struct=CHOCH_DOWN@1.1634); M30(RANGE, ADX=15.5 RSI=51.8 ATR=6.9p BB=19.7p %R=-35.0 MFI=60.4 AroonOsc=-29 Chop=49 ST=- Read=TRANSITION:0.25 q=QUIET cloud=above struct=BOS_DOWN@1.1634); H1(RANGE, ADX=14.7 RSI=52.8 ATR=9.8p BB=26.2p %R=-39.9 MFI=39.1 AroonOsc=-50 Chop=51 ST=+ Read=TRANSITION:0.25 q=NORMAL cloud=above struct=BOS_DOWN@1.1634); H4(RANGE, ADX=12.0 RSI=53.7 ATR=20.7p BB=56.1p %R=-23.3 MFI=57.8 AroonOsc=-29 Chop=55 ST=+ Read=TREND_WEAK:0.33 q=QUIET cloud=above struct=CHOCH_DOWN@1.1616); D(RANGE, ADX=17.8 RSI=46.8 ATR=56.3p BB=224.9p %R=-67.9 MFI=23.5 AroonOsc=-57 Chop=47 ST=+ Read=TREND_WEAK:0.33 q=QUIET cloud=below struct=BOS_DOWN@1.1655) |

## How To Read

- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (D>H4>H1>M30>M15>M5>M1).
- Momentum is the previous-snapshot slope of long-short score_gap; UP means the chart's evidence is rotating toward long, even if the current score still leans short.
- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.
- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).
- Pairs are sorted by max(long, short); the top entries are where edges line up.
