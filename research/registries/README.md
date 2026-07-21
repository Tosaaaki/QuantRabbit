# DOJO registry index

- Current goal-board input: `dojo_goal_board_input_20260719.json`
- Current content-addressed board:
  `dojo_goal_board_20260719_11c2ac8c81d4f716a08f2ee1cdadad062b632c39b82f09ffe4678827e4525ba6.json`
- Board result: `HYPOTHESIS / 3X_NOT_REACHABLE`, externally verified independent clusters `0`,
  `proof_admission.promotion_possible=false`
- Superseded non-content-addressed boards are excluded from the active
  registry. Keeping two internally valid `QR_DOJO_GOAL_BOARD_V2` objects here
  would permit semantic rollback even if prose labeled one "current".
- Prompt experiment preregistration: `dojo_prompt_experiment_v1.json`
- Prompt phase V1/V2, the first incomplete V3 startup, and candle-only V3R1 are
  superseded. V3R1 stopped before source acquisition at `0/90` because it
  repeated the known-null structure-only input class.
- Superseded AI forward evidence: `../forward/dojo-ai-forward-phase1-v3r1/precommit.json`,
  `start.json`, `supersession.json`, and `validity/events/*.json`
  - precommit canonical SHA-256 `d1c8e7d6c1c39b655154da21a05f9d4d7503c62cdc55b08d30addd43347251a3`
  - start receipt SHA-256 `b1bb63e67fce62168bb4f0db487c0a1a984df8f69eff4b45d944406c0e9dd198`
- Invalidated AI exit diagnostic: `../training/dojo-ai-exit-train-v1/evidence.json`
  - the decision M5 bar's high/low/close leaked at that bar's open
- Invalidated AI exit V2 diagnostic: `../training/dojo-ai-exit-train-v2/evidence.json`
  - decision-context lookahead was removed, but X05 failed corrected M5 path continuity
  - X06--X08 then shifted cohort identity; the preserved eight-cell totals cannot rank policies
- AI capital recycle diagnostic: `../training/dojo-ai-capital-recycle-train-v1/evidence.json`
  - existing-direction gate 6/6, next-direction 3/6
  - AI allocation `-26.65 capacity-pips` versus full HOLD `-19.30`; rejected
  - the post-hoc full-HOLD-until-invalid rule is a hypothesis, not a result
- AI capital recycle v2 diagnostic: `../training/dojo-ai-capital-recycle-train-v2/evidence.json`
  - one judgment per tool-free context; all six responses sealed before truth
  - hierarchical policy `-27.4`, full HOLD `-33.9`, cut-to-reserve `-26.8` capacity-pips
  - better than HOLD but worse than reserve; rejected and not proof
- AI V3R1 successor-evidence correction:
  `../corrections/dojo-ai-forward-v3r1-supersession-v1.json`
  - invalidates the stale `+0.6 pips` exit rationale for same-bar lookahead
  - binds the original supersession seal and the replacement V2 evidence bytes
- AI-supervised worker tuning: `../training/dojo-worker-ai-tuning-20260719/evidence.json`
  - six TRAIN survivors: two spike-fade tailguards, one tiny pullback hypothesis,
    and three capital-policy diagnostics
  - proof survivors `0`; old four-type diversity batch invalidated
  - current-byte cap-fixed diversity v2 rerun enforced pair/global limits but all four types lost
  - round-number TP 2→3 ATR follow-up also lost on both paths and was rejected
  - on the capital-policy replay, 60-minute release beat full HOLD and split reserve
  - capital-policy replay was JPY 200,000 over 11d 21h 59m; its 30-day multiple is extrapolated, not observed
- Separate-regime worker replay: `../training/dojo-worker-survivor-regime-replay-train-v1/evidence.json`
  - pullback A2 lost on both paths; both tailguards failed closed on stale conversion data
  - TRAIN survivors `0`; proof survivors `0`
- Historical holdout burn registry: `dojo-historical-holdout-burn-v1/events/`
  - 5 hash-chained events: genesis + 4 conservative legacy burns
  - latest event SHA-256 `90f0d3ba8a771ea43b01a2bc2f11a51e61ecec9f69ad5e3432905be103effdc9`
  - broad M5 price-path, exact S5, W54, and W55 windows are burned
  - `MARKET_PRICE_PATH` also blocks relabeling the same window as entry/exit
  - local discipline only; proof/promotion/live permission remain false
- Worker forward smoke spec: `dojo_worker_forward_smoke_v2_spec.json`
- Current worker forward evidence: `../forward/dojo-worker-forward-smoke-v2/precommit.json` and `start.json`
  - state `STARTED`, sealed days `0/14`
  - precommit canonical SHA-256 `b5240ca36ce84dd00945d8d307bf65ceb2f2b0688849b0d102680436fb22b06a`
  - start receipt SHA-256 `36fc12db99e81a7a3f6a3cb0b539ee9b55e4bd87c2d4ae58b537ac1a72591a2e`
- New-strategy research queue V1:
  `dojo_strategy_research_queue_v1-bcaed95081ed609fa9faba1f23b134fa25a3777855c1e0b1ab9aa7f07430aa6f.json`
  - three unexecuted designs: Asia sweep/reclaim + BE, H1 Donchian break + ATR trail,
    and G8 relative-strength risk-budget allocation
  - room-01/02/03 isolate strategy, thesis, input class, trainer lineage, search budget,
    parameters/results, and artifact root; only evaluator/cost/risk/source contracts are shared
  - cross-room result or parameter reuse creates a new hypothesis and debits the destination
    room; a four-capital-slot common TRAIN sparring arena is separate from unopened holdout
    examination and prospective-forward arenas
  - exact G2 baseline binding rejects the existing six families and any resealed duplicate
  - the queue is currently a planning catalog, not an operational reservation authority:
    its V1 transition state has no authenticated receipt chain or lock-backed compare-and-swap
  - the external AI trainer must not reserve or complete a design until that control plane exists;
    it may prepare an unreserved proposal only after a changed terminal/material
    cost+MTM+margin+LOPO TRAIN result, while unchanged results remain a semantic no-op
  - holdout/prospective opening, monthly-target backsolving, proof, promotion, live permission,
    broker mutation, and QuantRabbit model API calls remain forbidden

No registry in this directory grants live permission. A filename, self-declared status, or public SHA alone is not
trusted evidence.
