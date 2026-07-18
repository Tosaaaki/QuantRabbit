# DOJO registry index

- Current goal-board input: `dojo_goal_board_input_20260719.json`
- Current content-addressed board:
  `dojo_goal_board_20260719_f962517572391880bd2f25e0841aaefd206339e3e20b1e44e2defb268352fa93.json`
- Board result: `HYPOTHESIS / 3X_NOT_REACHABLE`, externally verified independent clusters `0`,
  `proof_admission.promotion_possible=false`
- Superseded non-content-addressed boards are excluded from the active
  registry. Keeping two internally valid `QR_DOJO_GOAL_BOARD_V2` objects here
  would permit semantic rollback even if prose labeled one "current".
- Prompt experiment preregistration: `dojo_prompt_experiment_v1.json`
- Prompt phase V1 and started V2 are superseded in the goal-board lifecycle. V3 is
  `IMPLEMENTATION_ONLY` with no precommit/start artifact and no response or score evidence.
- Prompt phase denominator remains 30 unique blind days × 3 variants = 90 exact cells; current V3 responses `0/90`
- Worker forward smoke spec: `dojo_worker_forward_smoke_v2_spec.json`
- Current worker forward evidence: `../forward/dojo-worker-forward-smoke-v2/precommit.json` and `start.json`
  - state `STARTED`, sealed days `0/14`
  - precommit canonical SHA-256 `b5240ca36ce84dd00945d8d307bf65ceb2f2b0688849b0d102680436fb22b06a`
  - start receipt SHA-256 `36fc12db99e81a7a3f6a3cb0b539ee9b55e4bd87c2d4ae58b537ac1a72591a2e`

No registry in this directory grants live permission. A filename, self-declared status, or public SHA alone is not
trusted evidence.
