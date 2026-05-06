# Verify And Execute

## Use When

- `data/codex_trader_decision_response.json` has been written.
- The next step is verifier acceptance and a single gateway cycle.

## Verify

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

Verifier acceptance is required before gateway handoff.

## Execute One Gateway Cycle

```bash
./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## Execution Rules

- `--reuse-market-artifacts` pins verifier evidence to the packet cited by the receipt.
- `LiveOrderGateway` still fetches fresh broker truth before staging or sending.
- Real send also requires `QR_LIVE_ENABLED=1`, `--send`, and all live gates.
- Without live gates, the cycle stays dry-run.
- Do not rerun after a rejected / blocked / no-trade result to force a fill.

## Report

- `docs/gpt_trader_decision_report.md`
- `docs/autotrade_cycle_report.md`
- `docs/live_order_stage_report.md` or `docs/position_execution_report.md`
- Selected lane id(s), sent flag, verifier status, gateway status, blockers.
