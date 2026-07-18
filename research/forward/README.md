# DOJO forward evidence

`dojo-worker-forward-smoke-v1/` は `[2026-07-20T00:00:00Z, 2026-08-03T00:00:00Z)` の
append-only prospective worker smokeである。

- state: `STARTED`
- sealed days: `0/14`
- fixed candidates: 12（3 signal families × 4 exits）
- intrabar: `OHLC` と `OLHC` の両方
- evidence tier: `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC`
- proof/live/promotion eligible: `false`

14暦日smokeは挙動と証拠鎖を反証するための短期試験で、worker proofに必要な60 active daysを満たさない。
日次receiptとfinal resultは既存ファイルを置換せず、このrun-dirへ追加する。
