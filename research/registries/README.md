# DOJO registry index

- Current goal-board input: `dojo_goal_board_input_20260719.json`
- Current content-addressed board:
  `dojo_goal_board_20260719_62dec2849c3dcf62c5358471a54762b8d8e21423ac2caefa3c048a70dba6938d.json`
- Board result: `HYPOTHESIS / 3X_NOT_REACHABLE`, externally verified independent clusters `0`,
  `proof_admission.promotion_possible=false`
- Superseded non-content-addressed boards are excluded from the active
  registry. Keeping two internally valid `QR_DOJO_GOAL_BOARD_V2` objects here
  would permit semantic rollback even if prose labeled one "current".
- Prompt experiment preregistration: `dojo_prompt_experiment_v1.json`
- Prompt phase-1 denominator: 30 unique blind days × 3 variants = 90 exact cells; current new responses `0/90`
- Worker forward smoke spec: `dojo_worker_forward_smoke_v1_spec.json`
- Worker forward evidence: `../forward/dojo-worker-forward-smoke-v1/precommit.json` and `start.json`
  - state `STARTED`, sealed days `0/14`
  - precommit canonical SHA-256 `7d849052082049721c9f83d658190d163dc7aa00e7c17c25d7ec50e487d71c5d`
  - start receipt SHA-256 `4b8fadbb452e304ae1e4f9721db3a8da93b212c08354b938abf1cda50a5523b9`

No registry in this directory grants live permission. A filename, self-declared status, or public SHA alone is not
trusted evidence.
