# Legacy strategy profit tuning — in progress

Generated: 2026-07-29 JST

## Safety boundary

- Project: `qr-trading` / QuantRabbit
- Repository: `/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval`
- Branch: `codex/dojo-dual-eval`
- Environment: archived replay and isolated DOJO Paper only
- Authority: `live_permission=false`, external broker mutation forbidden,
  `order_authority=NONE`
- Existing four-room experiment and existing legacy A/B Paper rooms were not
  stopped or modified.

## Direct answer

The previous run did **not** test all 82 normalized strategy families. It tested
four previously selected families and mechanically replayed four additional
families. This continuation prioritizes every family for which an executable
runner can be recovered. Evidence-only families remain unevaluated rather than
being replaced with an invented strategy that merely shares the old name.

## M1Scalper constrained customization

Training used 2026-01-25 and 2026-01-26. The rule was frozen before evaluating
2026-01-27. A conservative 1.1-pip round-trip cost was deducted from every
trade.

The train-only fitter selected:

- entry suppression: UTC hour 23, long only
- inventory cap: one concurrent position
- cooldown: three minutes
- no lot increase

| window | arm | net JPY | PF | expectancy JPY | max DD JPY | trades |
|---|---|---:|---:|---:|---:|---:|
| 2026-01-25 train | original Bot | +525.60 | 1.1375 | +8.76 | 1,558.80 | 60 |
| 2026-01-25 train | tuned AI inventory gate | +462.60 | 1.6465 | +35.58 | 347.40 | 13 |
| 2026-01-26 train | original Bot | -10,639.50 | 0.7250 | -12.40 | 10,930.50 | 858 |
| 2026-01-26 train | tuned AI inventory gate | +97.00 | 2.3108 | +13.86 | 52.00 | 7 |
| 2026-01-27 holdout | original Bot | -14,205.50 | 0.6320 | -16.95 | 16,091.50 | 838 |
| 2026-01-27 holdout | tuned AI inventory gate | **+51.00** | **1.2589** | **+8.50** | **197.00** | **6** |
| 2026-01-23 final data check | original Bot | -927.90 | 0.6831 | -1.29 | 967.50 | 717 |
| 2026-01-23 final data check | tuned AI inventory gate | N/A | N/A | N/A | N/A | 0 |

The 2026-01-23 capture ended before the selected UTC hour, so it is an
insufficient final holdout rather than a win. The 2026-01-27 holdout is positive,
but six trades are too few for live adoption. This candidate is limited to a
future isolated Paper A/B once its archived signal implementation is ported
faithfully; it is not permitted to affect the existing rooms.

The two worst completed holdout losses were inspected by fresh AI only after the
mechanical replay. One fixed 60-second checkpoint proposed an early exit that
would have saved an estimated 33 JPY; the other remained a hold. Because the
loss windows were outcome-selected, this result stays shadow-only and is not
included in the +51 JPY holdout result.

## Other recovered runners

`BB_RSI` had no hour/direction bucket that was profitable in both training
windows with at least ten trades per window, so no tuned rule was promoted.

The archived new-strategy runner produced:

- `LiquiditySweep`: train 2026-01-26 PF 0.857; holdout 2026-01-27 PF 1.286;
  final 2026-01-23 PF 0.643 — unstable, reject.
- `MicroCompressionRevert`: train PF 0.833; no holdout trades; final PF 0.622 —
  reject/insufficient.
- `MicroTrendRetest`: one losing train trade, two holdout trades at PF 0.953,
  one losing final trade — reject.

The executable worker screens are recorded under
`baseline/mechanical-screen`. Completed realistic exit results from the short
capture are under `baseline/realistic-exit-short-capture`. A strategy with zero
trades is treated as insufficient, never as infinite PF.

### Five-second mechanical screen with 1.1-pip cost

The train columns combine 2026-01-23 and 2026-01-26. The holdout is
2026-01-27. The short Sunday capture on 2026-01-25 is retained as an additional
stress check, not merged into the full-day holdout.

| strategy | train net JPY | train PF | train trades | holdout net JPY | holdout PF | holdout DD JPY | holdout trades | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `impulse_break_s5` | -6,090.00 | 0.5499 | 74 | -7,460.00 | 0.3025 | 8,490.00 | 42 | reject |
| `impulse_retest_s5` | -2,045.00 | 0.1771 | 6 | -1,270.00 | 0.2784 | 1,270.00 | 5 | reject |
| `impulse_momentum_s5` | -41,810.00 | 0.2370 | 236 | -27,260.00 | 0.1757 | 27,360.00 | 126 | reject |
| `pullback_s5` | -1,243.20 | 0.6011 | 34 | -1,054.20 | 0.5173 | 1,054.20 | 19 | reject |
| `vwap_magnet_s5` | +217.50 | 1.2644 | 8 | -95.00 | 0.8288 | 475.00 | 7 | reject/insufficient |
| `stop_run_reversal` | N/A | N/A | 0 | N/A | N/A | N/A | 0 | insufficient |

No hour/direction bucket stayed profitable in both training windows at the
minimum sample threshold for these families. Consequently, no holdout-sensitive
custom rule was promoted. The realistic exit replay that did finish on the
short 2026-01-25 capture also reversed `vwap_magnet_s5` from a nominal gross
gain to -407.97 JPY, confirming that its small apparent edge is cost/exit
sensitive.

## Current decision

- **Positive provisional candidate:** `M1Scalper` with the frozen
  session/inventory gate. It changed the 2026-01-27 costed result from
  -14,205.50 JPY to +51.00 JPY and capped drawdown at 197.00 JPY.
- **Paper/live adoption:** not yet. Six holdout trades and no final-window
  observations are not enough to claim a durable edge.
- **Next safe action:** faithfully port the archived M1 signal into a new,
  isolated Bot-only/AI-inventory Paper pair, then require a larger forward
  sample before economic adoption. Existing Paper rooms remain unchanged.

## Artifacts

- `scripts/tune_session_direction_gate.py`: train-only session/direction and
  inventory-cap fitter
- `tuned/m1_scalper_session_inventory_gate.json`: frozen fitted rule and metrics
- `fresh_ai_review.json`: worst-loss, fixed-checkpoint shadow review
- `baseline/system/*.json`: common-runner trades
- `baseline/new-strategies/*.json`: LiquiditySweep/Compression/TrendRetest
- `baseline/mechanical-screen`: completed five-second worker screens
- `baseline/realistic-exit-short-capture`: completed realistic exit replay for
  the short capture

## External reporting gap

Notion and Slack connectors were unavailable in this execution. Per the active
fail-closed contract, no Slack API message is sent until the current Notion
Slack route, identity, mode, and destination thread can be directly obtained.
