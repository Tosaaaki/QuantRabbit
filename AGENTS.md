# QuantRabbit vNext Agent Instructions

Build from broker truth outward.

## Prime Directive

Do not optimize prompts or strategy prose before the broker gateway and risk contract are enforced in code.

## Live Trading

- Default mode is dry-run.
- Do not add OANDA write methods unless the call path first passes `RiskEngine.validate(..., for_live_send=True)`.
- Do not read legacy `config/env.toml` into vNext. Use explicit environment variables for broker credentials.
- Any manual, tagless, or broker-synced exposure blocks new entries until adopted or closed.

## Legacy Knowledge

- Treat `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z` as read-only evidence.
- Use `qr-vnext import-legacy` before strategy work.
- Do not copy old schedulers, automation prompts, or order helpers into vNext wholesale.

## Acceptance Bar

Each new execution feature needs:

- a failing regression test for a known legacy failure mode;
- a passing test for the new behavior;
- a dry-run receipt with risk metrics;
- no live side effect unless explicitly enabled.
