# GPT Trader Decision Report

- Generated at UTC: `2026-06-26T02:32:26.136975+00:00`
- Status: `ACCEPTED`
- Action: `CANCEL_PENDING`
- Selected lane: `None`
- Selected basket lanes: `none`
- Cancel order ids: `472871`
- Confidence: `HIGH`
- 20m plan: `{'counterargument': 'Campaign pressure argues for taking LIVE_READY lanes, but named blockers outrank discretionary urgency.', 'entry_or_hold_trigger': 'Hold new entries until cycle-refresh produces clean LIVE_READY evidence and news-health is non-blocked.', 'evidence_refs': ['broker:snapshot', 'target:daily', 'self_improvement:audit', 'self_improvement:execution_quality'], 'failure_path': 'A blocker remains after refresh, so the next cycle must continue repair/evidence work.', 'horizon_minutes': 20, 'invalidation_or_cancel_trigger': 'Reconsider only after the named blocker code disappears from the refreshed packet.', 'next_cycle_check': 'First re-check broker snapshot, order_intents, ai_attack_advice, news-health, and selected lane refs.', 'primary_path': 'Refresh the named blocker(s) and keep broker-truth maintenance active before new risk.'}`
- Specialist reviews: `0`
- Operator summary: Autonomous trader draft will clear stale pending exposure through the verified CANCEL_PENDING gateway path before any new trade decision.

## Verification Issues

- none

## Close Gate Evidence

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires known `LIVE_READY` lane(s); pending entries are counted by gateway basket validation.
- `TRADE`/`CANCEL_PENDING` cancel ids must be current trader-owned pending entry orders from broker truth.
- Current `ai_attack_advice` recommendations make generic WAIT invalid while the daily target is open, but never grant live permission.
- Learning may only rank already-live-ready lanes. Any learning-influenced selected lane must be covered by a non-blocked `learning_audit` packet and cite `learning:audit` plus `learning:lane:<lane_id>`.
- `TRADE` must cite current chart evidence plus `news:health` and `news:items` or `news:current`; blocked news-health is a no-trade gate.
- `TRADE`, `WAIT`, and `REQUEST_EVIDENCE` receipts must include `twenty_minute_plan`: the next-20-minute primary path, failure path, trigger, invalidation/cancel trigger, strongest counterargument, next-cycle check, and known packet refs. This is a receipt-depth gate, not a new market-risk gate.
- `market_status` is deterministic calendar/session evidence only; broker truth still decides prices, positions, and tradability.
- A deterministic `tp-rebalance` sidecar requirement makes WAIT / REQUEST_EVIDENCE invalid until the sidecar is run.
- A deterministic entry-thesis blocker makes TRADE / WAIT invalid until the unverifiable active position is repaired or reviewed.
- Any self-improvement P0 blocks new `TRADE` receipts until the named blocker is repaired or the trader route explicitly justifies the exception.
- The 2025 operator precedent is advisory only. A `TRADE` that cites `operator:precedent` must also cite `manual:market_context`, at least one selected lane must match the current operator-precedent aligned lane set, and that selected lane must not conflict with the bounded manual technical replay buckets; otherwise the receipt must use current deterministic edge instead of precedent-based aggression.
- Evidence refs must come from the input packet; invented refs reject the decision.
- For H4 standing authorization, `close-confirmed` is not enough by itself: the exact structured event timestamp must postdate the matching broker-open / entry-thesis anchor and must not be future-dated versus the chart or broker snapshot. Pre-entry, missing-time, mismatched, future-dated, and unanchored H4 evidence is soft Gate A and still needs explicit Gate B.
- A `thesis_evolution` BROKEN/RECOMMEND_CLOSE label carries standing authorization only when its rationale records buffered price invalidation plus technical confirmation. A structural phrase inside that rationale is not market evidence; structural authority must come from the timestamped H4 or structural position-management paths.
- `CLOSE` requires Gate A plus the applicable Gate B. Hard Gate A (timestamped post-entry H4 close-confirmed BOS/CHOCH against side, buffered invalidation_price hit with technical confirmation, fresh thesis_evolution BROKEN/RECOMMEND_CLOSE whose rationale contains the canonical buffered hit and technical confirmation against the same position side, structural position_management / position_guardian_management REVIEW_EXIT, or position_thesis invalidation-hit/structural-break evidence with multi-TF confirmation) carries standing loss-cut authorization only when it has not been downgraded by fresh same-direction HOLD/EXTEND sidecars. Structural prose inside thesis-evolution is soft. Forecast flip, adverse drift, confidence/regime decay, and THESIS_EXPIRED are soft Gate A by themselves. M15 structure is Gate A evidence but not unattended hard Gate B unless H4 / recorded invalidation / hard sidecar also confirms; M15 internal structure or receipt-level `invalidation_price` cannot harden a matching soft entry-buffer / unrecorded-invalidation sidecar. `protection_sidecars.position_close_recommendations[].blocks_non_close_actions=false` means the sidecar is advisory for entry routing: do not write CLOSE merely to test the verifier; evaluate current LIVE_READY entries unless a current hard close sidecar separately blocks non-close actions. Softer Gate A still needs `QR_OPERATOR_CLOSE_OVERRIDE=1` or a fresh `data/.operator_close_token` when the trader chooses CLOSE, but operator Gate B does not override fresh same-direction HOLD/EXTEND support. If the same-direction market stack still supports the open position, treat it as TP rebalance / HOLD / profit-side partial / ADD geometry, not loss-side CLOSE plus same-direction re-entry. `TRADE` must not include `close_trade_ids`; automation ends the close cycle, then the next scheduled cycle must refresh broker truth, reprice intents, and require a separate verified `TRADE` receipt. The receipt's `operator_close_authorized` field is advisory only. See AGENT_CONTRACT §10.
