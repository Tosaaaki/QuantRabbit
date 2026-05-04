# Research Archive — Trader Reading Layer

Source-of-truth research that grounds the trader's "market reading" upgrade. Each file is a self-contained briefing produced by a research agent on 2026-05-05.

These are **point-in-time snapshots**. Verify against current code before citing as fact. Memory tag `feedback_market_state_in_code` applies.

## Index

| File | Topic | Length |
|---|---|---|
| [01-smc-ict.md](01-smc-ict.md) | SMC / ICT primitives, multi-TF top-down, killzones, honest critique | ~3,000 words |
| [02-volume-profile-tpo.md](02-volume-profile-tpo.md) | Volume Profile, Market Profile (TPO), AMT, order flow proxies on retail FX | ~2,500 words |
| [03-quant-ensemble-regime.md](03-quant-ensemble-regime.md) | Regime classification, indicator confluence scoring, statistical filters, MTF ensembling, validation rigor | ~3,000 words |
| [04-intermarket-macro.md](04-intermarket-macro.md) | Pair-by-pair intermarket map, rate diffs, risk-on/off composite, COT, calendar tiers, JST holidays | ~3,500 words |

## Synthesis (4-layer model that operationalizes the research)

1. **Normalization** — Z-score / percentile rank per pair/TF/indicator (kills "RSI=37 → SHORT" naive citation).
2. **Family scoring** — Trend/MeanRev/Breakout composites (kills 6-momentum-vote false agreement).
3. **Regime gate** — Hurst + ADX + Choppiness + ATR-percentile → 4-state (decides which composite to read).
4. **Narrative primitives** — PDH/PDL, OB/FVG, Killzones, NPOC (kills indicator-dump masquerading as chart_story).

See `AGENT_CONTRACT.md` §3.5 / §6 and `SKILL_trader.md` for normative rules and the runtime checklist that consume this layer.
