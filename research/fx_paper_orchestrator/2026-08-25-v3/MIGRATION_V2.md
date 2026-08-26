# Paper FX orchestrator v2 migration record

## Authority

This directory is local/paper research only. It has no live, broker-account,
credential, order-endpoint, deploy, or external-configuration authority.

## Handoff seal

The imported baseline was read twice from the old active, dirty, detached
worktree without modifying it. The adopted definition runs from the source
root:

```sh
find . -type f ! -path './__pycache__/*' ! -path '*/__pycache__/*' -print0 \
  | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256
```

- aggregate SHA-256: `72834f633eb66845811165967dcb5ef42df564b621d446d28c492dba363882fa`
- regular files: `186`
- bytes: `830151581`
- source latest mtime ns: `1787741826559821154`

The earlier `20ae6fdff4a43473bc5d4004e9674e460011302f2a69fc62f4b02ed7ce49e245`
was calculated one directory higher and therefore hashed `shasum` lines with a
`research/fx_paper_orchestrator/2026-08-25-v3/` prefix. It is a stream-path
definition difference, not a content change.

Seven very large legacy ledgers are retained in this local handoff but ignored
at their exact paths for Git transport. They remain covered by the baseline
aggregate seal. Executable code, preregistrations, tests, result summaries,
legacy registry/seals, v2 state, and official V25 evidence are versioned.

## Legacy evidence

V4-V24 result files and seals are read-only. v2 checks the V1 registry hash and
uses older schemas as migration evidence; it does not rewrite historical
results. The known V4 nanosecond timestamp fixture failure is not repaired by
changing a historical seal and remains a separate migration item.

## V25 diagnostic boundary

The stopped diagnostic replay (500 signals, 80 effective days, walk-forward
RAW `1.004741490752261`, BASE `0.9969664235923138`, ADVERSE
`0.9907639221545058`) is recorded only as diagnostic context. It is not copied
into the official seal. V25 is first registered in the v2 registry and then
executed once through the v2 coordinator.

## Gate separation

System acceptance proves reproducibility, fail-closed behavior, paper-only
authority, source/signal/cost/inventory/terminal constraints, restart safety,
and an unopened holdout label. Strategy adoption remains a separate gate:
JPY 200,000 must reach at least 2.0x in every full comparable month in both
normal and adverse arms and reproduce on a previously unopened holdout; 3.0x
is stretch. A system pass is not profitability proof.
