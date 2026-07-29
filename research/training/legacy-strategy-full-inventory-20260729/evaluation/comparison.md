# Legacy strategy replay + fresh AI shadow comparison

- Authority: NONE; live_permission=false
- Window: 2026-01-27 full-day archive, USD_JPY, 5-second mechanical replay
- Cost: identical realistic next-tick/spread/slippage/latency assumptions
- Fresh AI: worst-loss window only; Paper/shadow counterfactual; no economic application

| strategy | Bot net | AI net | PF Bot/AI | Exp Bot/AI | DD Bot/AI | Giveback Bot/AI | trades | AI decisions | AI cost | decision |
|---|---:|---:|---|---|---|---|---:|---:|---|---|
| `trend_breakout` | -582.57 | -168.57 | 0.0/0.0 | -582.57/-168.57 | 582.57/168.57 | N/A/N/A | 1 | 1 | not_metered_in_codex_session | reject_economic_ai_shadow_only |
| `pullback_continuation` | N/A | N/A | N/A/N/A | N/A/N/A | N/A/N/A | N/A/N/A | 0 | 0 | not_metered_in_codex_session | insufficient_samples |
| `failed_break_reverse` | N/A | N/A | N/A/N/A | N/A/N/A | N/A/N/A | N/A/N/A | 0 | 0 | not_metered_in_codex_session | insufficient_samples |
| `session_open` | 461.44 | 461.44 | Infinity/Infinity | 461.44/461.44 | 0.0/0.0 | 0.0/0.0 | 1 | 0 | not_metered_in_codex_session | provisional_promising_more_samples_required |

## Interpretation

- SessionOpen earned 461.44 JPY, but one trade is not enough to promote it beyond provisional observation.
- TrendBreakout lost 582.57 JPY. Fresh AI exit at the 60-second checkpoint would have reduced the loss to 168.57 JPY (+414.00 JPY), but the result remains negative.
- PullbackContinuation and FailedBreakReverse had no trades, so PF/expectancy/DD are N/A rather than zero or infinity.
- No new continuous Paper room was launched from this replay because no candidate met a minimum evidence threshold.
