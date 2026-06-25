# Trader Decision Draft Report

- Action: `none`
- Selected lane: `none`
- Selected basket lanes: `none`
- Draft blockers: `not run yet`
- Verifier precheck: `not run yet`

## Verification Issues

- none

## Contract

- Draft generation is read-only and writes only the decision response JSON/report.
- It does not call model APIs, send orders, cancel orders, close positions, or change launchd state.
- `gpt-trader-decision` and `LiveOrderGateway` remain the execution gates.
