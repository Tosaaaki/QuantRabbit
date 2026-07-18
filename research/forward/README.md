# DOJO forward evidence

現行のprospective runは次の2本である。どちらもappend-onlyで、成績はまだ存在しない。

| run | state | 現在値 | 主要な未充足境界 |
|---|---|---|---|
| `dojo-worker-forward-smoke-v2/` | `STARTED / HYPOTHESIS` | 0/14 source days、0/24 execution cells | 経済event replay、外部単調witness、60 active days |
| `dojo-ai-forward-phase1-v2/` | `COLLECTING_SOURCE / HYPOTHESIS` | 0/30 days、0/90 responses、0 truth scores | model executor、provider identity/fresh-context attestation、外部単調witness、推移的invalidity registry |

worker v2は `[2026-07-20T00:00:00Z, 2026-08-03T00:00:00Z)`、12候補
（3 signal families × 4 exits）を `OHLC` と `OLHC` の両方で固定した。source receiptからexact corpusを組み、
固定runner/VirtualBrokerの24セルを一度だけ実行してledgerと結果を導出する。caller supplied result manifestは使わない。
ただし現行ledgerだけでは経済event replayを外部検証できないため、resultは必ず
`INVALID_UNSCOREABLE_TRIAL`、proof/live/promotion eligibleは`false`である。

AI v2は30個の月〜木15:00 UTC cutoff、A/B/Cのexact 90 cell、gpt-5.5/high policyを固定した。
最初のcutoffは `2026-07-22T15:00:00Z`。response封印後だけ24時間のOANDA M5/BA truthを取得し、
固定コストで90分母のNAVとpaired contrastを導出する。市場truth scorerは実装済みだが、モデルを呼びprovider identityと
fresh contextを証明するexecutorは未実装なので、caller JSONを封印してもAI試験実行済みとは数えない。

`dojo-worker-forward-smoke-v1/` はsource producerを事前束縛していなかったため
`SUPERSEDED / DIAGNOSTIC`。追跡bytesは履歴として保持し、v2へ混ぜない。

14暦日のworker smokeと30日のAI phase-1は挙動と証拠鎖を反証するpilotで、月次3倍、worker 60 active days、
AI 90 active daysの証明閾値を満たさない。全runは `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC`、
portfolio判定は `HYPOTHESIS / 3X_NOT_REACHABLE` のままである。
