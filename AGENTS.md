# QuantRabbit vNext Agent Instructions

## Package Manager
- None. Use system Python with `PYTHONPATH=src`.
- Test with `PYTHONPATH=src python3 -m unittest discover -s tests -v`.

## Commit Attribution
- AI commits MUST include:
```
Co-Authored-By: Codex <noreply@openai.com>
```

## Prime Directive
- Build from broker truth outward.
- Do not optimize prompts or strategy prose before broker gateway and risk contracts are enforced in code.
- Codex acts as a discretionary trader, but autonomy must be expressed as executable receipts, not vague market prose.

## Autonomous Trader Contract
- `autotrade-cycle --send` is the live automation entrypoint.
- Required loop: `BrokerSnapshot` -> `IntentGenerator` -> `TraderBrain` -> `LiveOrderGateway` when flat.
- Existing positions or pending orders block fresh entries.
- Required exposure loop: `BrokerSnapshot` -> `TraderBrain` -> `PositionManager` -> `PositionProtectionGateway`.
- Trader decisions must compare mined history, market story, campaign role, narrative risk, current broker state, risk geometry, and live exposure.
- Do not reduce entry selection to a single score threshold; every executable lane needs thesis, method, narrative, chart story, invalidation, TP, SL, and units.

## Live Trading
- Default mode is dry-run.
- Real entry sends require `QR_LIVE_ENABLED=1`, `--send`, fresh broker truth, explicit lane id, `RiskEngine.validate(..., for_live_send=True)`, and strategy-profile validation.
- OANDA entry writes go only through `LiveOrderGateway`.
- OANDA position writes go only through `PositionProtectionGateway`.
- Do not add independent OANDA write helpers, direct order scripts, or prompt-only workarounds.
- Do not read legacy `config/env.toml` into vNext. Use explicit `QR_OANDA_TOKEN`, `QR_OANDA_ACCOUNT_ID`, and `QR_OANDA_BASE_URL`.
- Any manual, tagless, or broker-synced exposure blocks new entries until adopted or closed.

## Position Protection
- Missing TP/SL is a repair requirement.
- Profitable protected positions can tighten SL to break-even or better.
- Contradicted trader-owned positions can be closed.
- Existing SL cannot be widened.
- Existing TP is not moved by the protection gateway.

## Legacy Knowledge
- Treat `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z` as read-only evidence.
- Use `PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy` before strategy work.
- Do not copy old schedulers, automation prompts, or order helpers into vNext wholesale.

## Current Commands
```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
QR_LIVE_ENABLED=1 PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --send
```

## Acceptance Bar
- Add a failing regression test for known legacy failure modes.
- Add a passing test for the new behavior.
- Preserve dry-run receipt paths for new execution features.
- No live side effect unless explicitly enabled.
- Never promise guaranteed returns; convert campaign pressure into bounded, testable execution behavior.
