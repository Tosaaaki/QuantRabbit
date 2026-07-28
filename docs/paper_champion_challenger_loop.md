# Paper champion/challenger profitability loop

## Responsibility boundary

The hourly Paper AI supervisor is the intended inventory/story decision owner.
Its deterministic gap detector may create a durable strategy-lab
request only when minimum evidence, a new data hash, cooldown, dedupe, and
candidate budget all pass. Strategy lab produces declarative Paper candidate
specifications, not runtime code. A separate shared-feed Paper executor fans
out each causal market event to champion, AI-inventory and challengers.

Every lane has isolated virtual capital, inventory, orders, P&L, ledger, and
risk budget. All lanes consume the same sequenced quote/bar event bytes; a lane
may not poll its own feed. `scripts/run-paper-champion-challenger.py` polls
read-only OANDA once, maintains one hash-chained feed, and fans each batch out
to six isolated lanes.

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

## Implemented pilot

The first candidate was selected from completed Paper evidence whose leading
cause was `COUNTERTREND_SHORT_CONCENTRATION`: -JPY 852.88 over 49 BASE/STRESS
settlements. The cost arms overlap, so they are not 49 independent
observations. The reviewed sibling template is `pullback_limit`, which enters a
pullback only in the completed 24-hour trend direction. This is a forward
experiment, not an adoption or a profitability claim.

The six lanes are champion / direction-limited inventory / pullback challenger,
each in BASE and STRESS cost arms. Each starts with JPY 50,000, maximum one
position, 1.0x per-position leverage and isolated account, inventory, orders,
ledger and risk budget. Only challengers are automatically killed at 5% DD.
The window expires after 14 days; the champion is retained.

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

Current status: candidate
`44322c7db119a4b4411cee59ca1f2eb64cd92f1e84bb89ecca1b88fdc3c8e332`
was admitted with idempotency hash
`039fd79e58b79bc73f15d1343da1d174a521e019d056dbcff8c5e4bbbe1e6d14`.
The shared-feed Paper process is running under experiment
`paper-pullback-sibling-20260728-v1`. No Automation or DOJO state was changed.
The hourly model-driven supervisor/strategy-lab scheduler is still not enabled;
the present strategy-lab decision is deterministic and bounded to reviewed
templates.
