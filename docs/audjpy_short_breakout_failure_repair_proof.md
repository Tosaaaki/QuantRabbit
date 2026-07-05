# AUD_JPY SHORT BREAKOUT_FAILURE Repair Proof

- Generated: `2026-07-05T18:15:59Z`
- Status: `REPAIR_REQUIRED`
- Standalone 4x path exists: `False`
- Preferred repair shape: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Can create live permission: `False`

| lane | order | daily % | standalone gap % | fresh S5 | proof pack | permission |
|---|---|---:|---:|---|---|---|
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `STOP-ENTRY` | 3.8818 | 1.524 | `EVIDENCE_GAP_UNDER_SAMPLED` | `True` | `False` |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `LIMIT` | 3.8818 | 1.524 | `EVIDENCE_GAP_UNDER_SAMPLED` | `True` | `False` |

## Missing Repair Requirements

- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE`: attached_tp_proof, forecast_executable_proof, fresh_s5_bidask_live_grade_support, geometry_proof, gpt_verifier_pass, live_order_gateway_pass, no_guardian_operator_review_blocker, portfolio_pairing_or_higher_frequency_required_for_4x, risk_engine_pass, s5_bidask_spread_included_replay
- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`: attached_tp_proof, forecast_executable_proof, fresh_s5_bidask_live_grade_support, geometry_proof, gpt_verifier_pass, live_order_gateway_pass, no_guardian_operator_review_blocker, portfolio_pairing_or_higher_frequency_required_for_4x, risk_engine_pass, s5_bidask_spread_included_replay
