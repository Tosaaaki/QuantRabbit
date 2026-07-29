# Legacy strategy rediscovery — interim result

Generated: 2026-07-29 JST

## Outcome

- Normalized strategy families found: **82**
- Previously evaluated: **4** (`trend_ma`, `pulse_break`, `m1_scalper`, `range_fader`)
- Newly replayed with trades: **2** (`trend_breakout`, `session_open`)
- Replay attempted but no trades: **2** (`pullback_continuation`, `failed_break_reverse`)
- Not yet replayed: **74**
- Provisionally promising: **1** (`session_open`, one trade only)
- Confirmed for promotion: **0**

Aliases were merged into a single `duplicate_family`. Every row in `inventory.json`
contains implementation paths/commits, runtime evidence, pair/timeframe, entry/exit
description, cost features, historical metrics where available, reproducibility, and
replay priority.

## Profit protection result

| strategy | Bot net JPY | AI shadow net JPY | AI delta | PF Bot/AI | Expectancy Bot/AI | Max DD Bot/AI | trades | decision |
|---|---:|---:|---:|---|---|---|---:|---|
| `trend_breakout` | -582.57 | -168.57 | +414.00 | 0/0 | -582.57/-168.57 | 582.57/168.57 | 1 | AI reduced the loss, but economic application rejected |
| `session_open` | +461.44 | +461.44 | 0.00 | ∞/∞ | 461.44/461.44 | 0/0 | 1 | provisional; more samples required |
| `pullback_continuation` | N/A | N/A | N/A | N/A | N/A | N/A | 0 | insufficient |
| `failed_break_reverse` | N/A | N/A | N/A | N/A | N/A | N/A | 0 | insufficient |

Fresh AI was used only for the sole worst-loss window. At 60 seconds after the
TrendBreakout entry, the move was -2.106 pips with only +0.294 pip MFE and
-2.806 pips MAE. The shadow decision exited all inventory using the same replay
spread/slippage assumptions, reducing the eventual loss by 414 JPY. No lookahead
was used at the decision checkpoint. AI cost was not metered in this Codex session;
external AI API calls were zero and the judgment count was one.

Profit giveback is N/A for TrendBreakout because it never produced gross profit.
SessionOpen's giveback was 0%, but one trade is not enough evidence for adoption.
Zero-trade candidates are reported as N/A, not as PF infinity.

## Replay conditions

- Market: archived USD_JPY ticks, 2026-01-27 full day
- Mechanical replay first; 5-second resampling for fast completion
- Identical A/B entry stream and worker-defined units
- Realistic next-tick fill, 180 ms latency, spread/ATR/latency slippage
- Hard SL disabled; end-of-replay liquidation excluded
- `authority=NONE`, `live_permission=false`
- GCP Secret Manager disabled; dummy practice identifiers used only to satisfy
  legacy import-time configuration
- No broker call, cloud mutation, live order, or current GCP query

## Deleted GCP VM trace recovery

Recovered VM/host identifiers from local static artifacts:

- `fx-trader-vm`
- `fx-trader-rescue`
- `qr-ssh-rescue`
- `quantrabbit`

Direct same-file VM-to-worker evidence was recovered for `fx-trader-vm` and the
families `basic`, `failed_break_reverse`, and `scalp_false_break_fade`. The rescue
identifiers were recovered from archived gcloud logs, but no defensible strategy
mapping was present in the same artifacts, so they remain unmapped rather than
guessed.

The static scan also recovered 218 systemd unit identifiers and four secret names.
Only names and evidence paths are stored. Values, command bodies, authentication
material, and current cloud state are excluded. No disk/snapshot/image identifier
was found in the searched local artifacts.

## Runtime safety

The four current dojo rooms and four legacy comparison rooms remained detached and
running after the scan/replay:

- `qr-dojo-range-base-bot`
- `qr-dojo-range-base-gate`
- `qr-dojo-range-stress-bot`
- `qr-dojo-range-stress-gate`
- `qr-legacy-trendma-bot-only`
- `qr-legacy-trendma-ai-shadow`
- `qr-legacy-pulsebreak-bot-only`
- `qr-legacy-pulsebreak-ai-inventory`

No new continuous Paper room was launched: the only positive new candidate had one
trade, below the evidence threshold. Existing Paper was not stopped or modified.

## Remaining source gap

Notion and Slack connectors were unavailable in this execution. Their canonical,
archived, attachment, and historical-message corpora were therefore not counted as
searched. Per fail-closed policy, no Slack API report was posted without directly
re-reading the current Notion Slack route. Local/Git/archive discovery and replay
are complete; Notion/Slack enrichment and deduplicated thread reporting remain.
