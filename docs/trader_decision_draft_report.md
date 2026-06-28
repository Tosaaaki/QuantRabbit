# Trader Decision Draft Report

- Action: `CANCEL_PENDING`
- Selected lane: `None`
- Selected basket lanes: `none`
- Cancel order ids: `472871`
- Draft blockers: `NO_LIVE_READY_LANES`
- Verifier precheck: `ACCEPTED`

## Verification Issues

- none

## Contract

- Draft generation is read-only and writes only the decision response JSON/report.
- It does not call model APIs, send orders, cancel orders, close positions, or change launchd state.
- `gpt-trader-decision` and `LiveOrderGateway` remain the execution gates.
