# DOJO worktree整理台帳 — 2026-07-19

この台帳は削除リストではなく、DOJO統合時点の所有権と保全状態を固定するもの。dirty件数は
`git status --porcelain` の行数であり、内容の重要度を表さない。owner不明のworktreeはcleanでも
自動削除しない。

## DOJO正本

| path | branch / HEAD | 状態 | 方針 |
|---|---|---|---|
| `/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval` | `codex/dojo-dual-eval` / `dc3179af4`起点 | 統合実装中 | KEEP。DOJO研究コードとtracked registryの唯一の編集面 |
| `/private/tmp/QuantRabbit-episode-outcome` | `codex/episode-s5-outcome` / `dc3179af4` | 4件dirty、worktree lock済み | ARCHIVE CARRIER。未追跡4.4GBの参照移行・push完了まで削除禁止 |
| `/Users/tossaki/App/QuantRabbit_archives/DOJO_20260719` | Git外archive | code bundle + research data + Fable handoff + repaired worker rerun | KEEP。元data 463 files / 4,690,853,483 bytesをexact mirror manifestで全SHA照合済み |

episode branchからDOJO branchまでは線形で、元コードはbundle
`dojo-code-dc3179af4.bundle`（SHA-256
`d873cc5db9774993f39fad0414bd828f0e50d9782d2792947b78688f7fde60f4`）へ退避済み。

## 操作禁止面

| path | snapshot dirty count | 理由 |
|---|---:|---|
| `/Users/tossaki/App/QuantRabbit` | 85 | 別branch/別作業が進行中。DOJOの整理対象外 |
| `/Users/tossaki/App/QuantRabbit-live` | 107 | live runtime。DOJOから変更・reset・clean禁止 |
| `/Users/tossaki/App/QuantRabbit-worktrees/main` | 95 | branch ref外部fast-forwardによりindexが旧HEADへ残るstale-index事故。実WIPを保全 |
| `/Users/tossaki/App/QuantRabbit-worktrees/orchestrator-evidence-loop` | 26 | owner WIP |
| `/Users/tossaki/App/QuantRabbit-worktrees/ai-supervised-fast-executor` | 2 | owner WIP |
| `/Users/tossaki/App/QuantRabbit-worktrees/forecast-trade-loop` | 2 | owner WIP |
| `.claude/worktrees/*` | 0〜9 | Claude/Fable所有。DOJOからremove/pruneしない |

`sync-live-runtime.sh` は、checkout中target branchをrefだけfast-forwardしてindex/filesを置き去りに
する経路を修理した。以後はclean target worktree内の`merge --ff-only`で三者を同時更新し、dirty、
stale index、複数checkout、欠損worktreeをref更新前にblockする。

## cleanだが削除を保留した候補

- `/Users/tossaki/App/QuantRabbit-worktrees/gpt-tuning-close-contract`
- `/Users/tossaki/App/QuantRabbit-worktrees/technical-forecast-live-scout`
- `.claude/worktrees/awesome-johnson-5626f3`
- `.claude/worktrees/musing-wilson-19ce2c`
- `.claude/worktrees/youthful-murdock-4de769`

cleanは「不要」を意味しない。branchのremote到達、merge状態、実行process、owner確認、recoverable
backupを満たしてから別タスクでremoveする。

## 今回行った整理

1. 元DOJO worktreeを理由付きlock。
2. main、episode、dojo-dual-eval refsを含むverified Git bundleを作成。
3. 未追跡research dataをarchiveへコピーし、元/archive双方の全file SHAと件数・bytesを一致確認。
4. Fable scratchpadから必要なAI packet/calls/keys/tasksだけを1.1MBのhandoffへ退避。
5. DOJOの編集を専用branch/worktreeへ集約し、root/main/liveのWIPへ触れない運用に変更。
6. CLI help確認時に誤生成された96KBの診断runは、削除せずarchive内
   `accidental-cli-help-runs`へ移動。CLIは`argparse --help`でrunを始めないよう修理。
7. 修理後worker 12設定をOHLC/OLHCの計24 trialで再審判。TRAIN通過0。1.9GBのrunを
   `codex-worker-rerun-v1`へ保存し、scoreboard SHA-256を
   `dbe2f5e4f441e6c723e44cee6d447955941d8d093e345e820b01f6dd949328fd` として固定。

## 解錠・削除条件

次をすべて満たすまではepisode worktreeをunlock/removeしない。

- `codex/dojo-dual-eval` がcommitされremoteへpush済み。
- tracked registryとarchive pointerが最終commitから参照可能。
- `research-data-manifest-v1.json` のcanonical manifest SHA
  `70955036e6e43b0469d1f53a0cd62a127ae2c97f2b9f4a9d4814d55a1686b944` とsource/mirror
  relative-inventory SHA `daf8ebfc654c9e06105d5b80e2e51c5bf9f03c05ecbd26975ceddbd200b7b7da`
  が再検証可能。
- 旧positive artifactを読むconsumerがvalidity registryへ移行済み。
- 元worktree固有の未追跡差分4件をownerが分類済み。
