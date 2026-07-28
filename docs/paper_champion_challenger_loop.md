# Paper champion/challenger profitability loop

## Responsibility boundary

The existing hourly Paper AI supervisor remains the inventory/story decision
owner. Its deterministic gap detector may create a durable strategy-lab
request only when minimum evidence, a new data hash, cooldown, dedupe, and
candidate budget all pass. Strategy lab produces declarative Paper candidate
specifications, not runtime code. A separate shared-feed Paper executor fans
out each causal market event to champion, AI-inventory and challengers.

Every lane has isolated virtual capital, inventory, orders, P&L, ledger, and
risk budget. All lanes consume the same sequenced quote/bar event bytes; a lane
may not poll its own feed. The current independently polling Paper rooms are
diagnostic only and do not yet prove this shared-feed condition.

DOJO and all DOJO Automations are outside this loop. DOJO is neither a gate nor
an auxiliary verifier. No component has live authority.

## Hourly flow

1. Seal the latest market-story/regime/inventory snapshot and 24h/7d metrics.
2. The AI supervisor emits only shadow `HOLD`, `REDUCE`, `PAUSE`, `CLOSE`,
   `DIRECTION_LIMIT`, `RESUME`, or Bot-selection advice.
3. The deterministic gap detector measures PF, expectancy, DD, WAIT/opportunity
   loss, coverage, cost and data sufficiency. Missing fills are not success.
4. If evidence is fresh and sufficient, seal one strategy-lab request. Otherwise
   save `NO_NEW_EVIDENCE` and spend no strategy-generation call.
5. Admit at most two challengers with the policy in
   `config/paper_champion_challenger_policy_v1.json`.
6. Continue, reduce, kill or expire each challenger from its isolated ledger.
   Never rewrite history; rollback is disabling future Paper events.

## Minimum implementation plan

The safe first mutation is control-plane only: policy validation, candidate/data
hash dedupe, cooldown, budget and fail-closed checkpointing. The data-plane
shared-feed executor must be introduced as a new process and new ledgers; it
must not retrofit the currently running rooms in place.

Schedule and expected incremental cost:

- every hour: deterministic gap calculation, negligible model cost;
- existing supervisor: one bounded shadow decision using its current budget;
- strategy lab: normally zero calls, maximum one fresh task per 24 hours;
- Paper execution: at most champion + AI inventory + two challengers from one
  feed read; no broker writes;
- retention review: deterministic hourly, AI escalation only for ambiguous
  evidence.

Completion requires identical feed event-chain hashes across lanes, complete
account/ledger isolation, kill/expiry/rollback tests, restart idempotency, no
duplicate candidate/data hash, and a minimum of 30 settlements in at least two
regimes before any Paper-only continuation finding. A candidate is retained
only if cost-adjusted PF is above 1, expectancy is positive, DD is not worse,
BASE/STRESS move in the same direction, and sample requirements pass.

Current status: policy and responsibility boundary are sealed, but automatic
candidate admission and a shared-feed executor are not enabled. No Automation
was changed because the required Notion/API routing and inventory-sync
transaction is unavailable in this turn. Existing Paper runs continue
unchanged.
