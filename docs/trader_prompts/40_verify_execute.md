# Verify And Execute

## Use When

- `data/codex_trader_decision_response.json` has been written.
- The next step is verifier completion and a single wrapper cycle.

## Verify

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

Verifier acceptance is required before fresh broker work. A rejected receipt
still goes through the wrapper once so existing-position maintenance and
post-cycle sidecars are not skipped; the gateway blocks new risk from the
rejected receipt.

If the verifier returns `TP_REBALANCE_REQUIRED`, the receipt is not an accepted
WAIT. Go to `position_management` and run `tp-rebalance` before writing another
WAIT/REQUEST_EVIDENCE receipt.

If the verifier rejects the receipt, or if the router reports that the receipt
predates a refreshed broker snapshot / repriced intent packet, do not stop the
trader because the old receipt exists. Return to the routed decision branch,
write one current receipt, and continue through the normal verifier -> wrapper
path.

After any completed verifier result, do not run refresh, analysis, TP
rebalance, projection, or thesis sidecar commands before the wrapper. The
receipt is tied to the current broker snapshot and order-intent packet;
inserting extra work between verification and wrapper execution makes the
receipt stale and can leave profit-side position maintenance stale.

## Execute One Gateway Cycle

Run the wrapper once after every completed verifier result, including
`REJECTED`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`,
`CANCEL_PENDING`, and `TRADE`. This is not a workaround send: it is the
contracted live path that refreshes broker truth, lets the gateway enforce
fresh-entry blocks, and then runs the consolidated protection sidecars while
the live lock is held.

```bash
QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh \
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
- Dynamic TP rebalance and position execution are mandatory after verifier
  completion even when the action is WAIT or the receipt is rejected.
  `run-autotrade-live.sh` calls `cycle-sidecars`, which refreshes broker truth
  first so newly filled or existing trades are visible, then runs TP/profit
  protection and the remaining sidecars in one process. Do not run the sidecar
  steps individually.

## Report

- `docs/gpt_trader_decision_report.md`
- `docs/autotrade_cycle_report.md`
- `docs/live_order_stage_report.md` or `docs/position_execution_report.md`
- `docs/execution_ledger_report.md`
- Selected lane id(s), sent flag, verifier status, gateway status, blockers.
