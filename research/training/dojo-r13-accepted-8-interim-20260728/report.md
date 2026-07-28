# DOJO R13 accepted 1–8 interim report

Status: stopped at 8/84 by operator decision. `qr-dojo-fresh-model-executor-v1`
remains `PAUSED`. Cell 9 was not evaluated. No accepted response, queue event,
ledger entry, Automation, Paper room, or broker state was changed.

## Evidence boundary

The eight accepted responses are content-addressed but have no approved
provider signature. Their queue state remains
`NOT_YET_APPLIED_PIPELINE_PROOF_ONLY`. The archived economic results used below
have `provider_model_call_count=0`; they are the earlier deterministic policy
results, not an application of the accepted provider responses.

Each accepted response happens to select the same action as the deterministic
policy at its source checkpoint. The economic numbers are therefore reported
only as an **action-matched directional proxy**. They are not reported as
measured AI lift. Applying the provider responses now would require a new
economic checkpoint and would violate the operator's no-new-decision/no-resume
boundary for this stopped study.

## Accepted coverage

- Accepted cells: 8/84 (9.52%).
- Coordinates: 2/12 (16.67%).
- Coordinate `2b92…` (`mean_revert_24h`, BASE): all seven cadences; all seven
  actions are `PAUSE_NEW_ENTRIES`.
- Coordinate `4b7f…` (`round_number_fade`, STRESS): only
  `ADAPTIVE_60M_15M_EVENT`; action is `HOLD`.
- Ten coordinates and 76 cells are unobserved.
- The same portfolio is reused by seven cadence cells, so the seven rows must
  not be summed. Action, family, coordinate, and cost scenario are confounded.

## Action-matched directional economics

| accepted subset | family / cost | action | Bot-only net | proxy net | directional lift | Bot/proxy DD | Bot/proxy peak margin | Bot/proxy cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cells 1–7 | mean_revert_24h / BASE | PAUSE | -¥28,245.65 | -¥14,541.15 | +¥13,704.50 | 14.819% / 7.271% | 8.067% / 8.059% | ¥31,042.34 / ¥8,895.31 |
| cell 8 | round_number_fade / STRESS | HOLD | +¥4,783.72 | +¥4,102.73 | -¥680.99 | 1.087% / 1.012% | 8.059% / 8.080% | ¥3,879.08 / ¥3,910.56 |

For cells 1–7, the proxy reduced trades from 414 to 114 and reduced transaction
cost by ¥22,147.03. Net and DD improved, but executed-trade expectancy worsened
from -¥68.23 to approximately -¥127.55. This supports “stop trading a broken
coordinate” as a loss-containment direction; it does not show that surviving
entries became better.

For cell 8, the proxy remained profitable and has PF 2.756 on its available
policy-close scope, but net fell by ¥680.99. This is classified as sacrificed
upside, not prevented loss. The baseline trade-level PF is unavailable.

Both coordinates recorded zero forced-margin closeouts and zero ruin events.
That supports only “no observed forced loss in these two coordinates.” Ordinary
loss, TP retention/giveback, Bot-only PF, and comparable gross decomposition
are not available from this partial checkpoint. They are not estimated.

## What can and cannot be concluded

What can be said:

- PAUSE is directionally consistent with containing a losing BASE
  `mean_revert_24h` coordinate. The proxy improvement comes with a 72% reduction
  in trade count and a 71% reduction in transaction cost.
- HOLD is directionally consistent with avoiding an unnecessary intervention
  at the sampled STRESS `round_number_fade` checkpoint, although the existing
  policy path still sacrificed ¥680.99 versus Bot-only.
- No margin closeout or ruin occurred in the two observed coordinates.

What cannot be said:

- The accepted AI responses produced the economic lift.
- AI improves final P&L across BASE/STRESS, strategies, regimes, or cadences.
- The seven PAUSE answers are seven independent economic successes.
- TP giveback, normal versus forced loss, or Bot-only PF improved.
- The selected early cells are representative. Queue order creates selection
  bias, and the stopping decision leaves most coordinates untouched.

Decision: do not adopt or promote from this eight-cell sample. Preserve it as
pipeline/directional evidence and replace the costly 84-cell sequence with the
bounded event-review v2 pilot design. Paper champion/challenger work remains
independent of DOJO.
