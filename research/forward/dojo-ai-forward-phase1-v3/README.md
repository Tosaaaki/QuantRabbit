# AI forward phase-1 V3 startup record

This run is `SUPERSEDED_STARTUP_INCOMPLETE` and must not receive day, model,
truth, or score artifacts.

- The V3 precommit was durably created before the first cutoff.
- The start receipt was durably created, but the command then exited non-zero
  while initializing the validity genesis.
- Cause: a relative `--run-dir` was prefixed twice when artifact paths were
  registered.
- No model process or market request was launched.
- The absent validity genesis makes this run unusable; do not reconstruct it
  after changing its source-bound implementation.

Identities:

- precommit canonical SHA-256:
  `1cb8626d3f64372a65b5f725c7848952d50064515fe72e4f5239c98a43c76e91`
- start receipt canonical SHA-256:
  `f9dc08e3cb556cf85c9ed95d1e7890fc902639654180f49cf868909903862a04`
- precommit file SHA-256:
  `9ce06974a50794ec23fe45b1a2a2dcce0683b36be4e96839dde303a5202363d6`
- start file SHA-256:
  `28764ccb8b39399340ea91902f4597a3809bc52843e2d7d0e6c8c41f74b3fae4`
