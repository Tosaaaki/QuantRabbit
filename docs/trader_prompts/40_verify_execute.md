# Verify And Execute

## Use When

- `data/codex_trader_decision_response.json` has been written.
- The next step is verifier acceptance and a single gateway cycle.

## Verify

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

Verifier acceptance is required before gateway handoff.

If the verifier returns `TP_REBALANCE_REQUIRED`, the receipt is not an accepted
WAIT. Go to `position_management` and run `tp-rebalance` before writing another
WAIT/REQUEST_EVIDENCE receipt.

If the verifier rejects the receipt, or if the router reports that the receipt predates a refreshed broker snapshot / repriced intent packet, do not stop the trader because the old receipt exists. Return to the routed decision branch, write one current receipt, and continue through the normal verifier -> gateway path.

If the verifier accepts a `TRADE`, do not run refresh, analysis, TP rebalance,
projection, or thesis sidecar commands before the gateway. The accepted receipt
is tied to the current broker snapshot and order-intent packet; inserting
extra work between acceptance and gateway execution makes the receipt stale and
can turn a tradeable cycle into a no-send cycle.

## Execute One Gateway Cycle

Run the gateway only for an accepted receipt that actually has broker work
(`TRADE`, verified `CANCEL_PENDING`, or an accepted position-management action).
If the verifier accepts `WAIT`, do not run a no-op gateway just to create a
cycle boundary; continue to the protection sidecars in `docs/SKILL_trader.md`.

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
- `autotrade-cycle` syncs `data/execution_ledger.db` before and after gateway work and records gateway receipts in the ledger.
- Real send also requires `QR_LIVE_ENABLED=1`, `--send`, and all live gates.
- Without live gates, the cycle stays dry-run.
- Do not rerun after a rejected / blocked / no-trade result to force a fill. A later scheduled cycle may refresh broker truth and make a new decision; that is normal continuation, not a workaround send.
- Dynamic TP rebalance is mandatory after verifier acceptance even when the
  accepted action is WAIT. For TRADE receipts, run it only after the gateway
  handoff and after refreshing broker truth so newly filled trades are visible.

## Report

- `docs/gpt_trader_decision_report.md`
- `docs/autotrade_cycle_report.md`
- `docs/live_order_stage_report.md` or `docs/position_execution_report.md`
- `docs/execution_ledger_report.md`
- Selected lane id(s), sent flag, verifier status, gateway status, blockers.
