# DOJO（道場）評価プログラム

DOJOは、戦略ワーカーとAI裁量を同じ実市場メカニクスで鍛え、再現可能な証拠を持つ候補だけを
次の前向き段階へ進める研究環境である。月次3倍は研究目標であって保証ではなく、DOJOの成績は
実弾権限を一切与えない。ライブ運用の権限境界と最上位KPIは `docs/AGENT_CONTRACT.md` が優先する。

## 2026-07-19時点の結論

| 対象 | 状態 | 結論 |
|---|---|---|
| W_FADE / W_SPIKE_FADE / W_ROUND / W_LADDERという戦略案 | `HYPOTHESIS` | アイデアは再試験可能だが、live昇段可能な生存者は0 |
| W46〜W53の旧worker成績証拠 | `INVALIDATED` | 誤帰属、同一建玉の複数bot所有、出口コスト欠落、摩耗holdoutがある |
| W37 AI方向読み | `UNESTABLISHED` | 40場面を4読者へ分割した結果で、4×40の独立試行ではない。的中41.18%、Wilson 95% CI 26.4–57.8% |
| W39 AI出口読み | `INVALIDATED` | 判定時点と同じM5バーの高値・安値・終値を入力していた。点推定+0.562 pipsのcluster bootstrap CIも -1.699〜+2.453でゼロを跨ぐ |
| W54/W55 AI日次読み | `INVALIDATED` | 後続日の履歴から先行日の答えを読めるpacket lookahead。単ペア89/90日、複数ペア79/80日が露出 |
| W54 legacy clean再試行 | `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC` | 40日中19応答のみ。旧packetは日付を保持し、応答schema・prompt/model/scorer/key隔離receiptも新契約を満たさない。旧計算は15 commit、8勝、NAV 0.975293、月換算0.971832だが再現可能な最終判定には使わない |
| worker前向きsmoke v1 | `STARTED / HYPOTHESIS` | `[2026-07-20, 2026-08-03)` の12候補×OHLC/OLHCを事前封印。0/14日採取。実OANDA M1/BA bytesからmanifestとday sealを自動生成するcollectorは実装済みだが、将来日次証拠はまだ0 |
| AI prompt phase-1 | `REGISTERED / HYPOTHESIS` | 旧90-cell一括manifestの前向き矛盾を検出。exact cutoff/cell/contextを先に固定し、日次OANDA M5/BA応答からA/B/Cを同時封印するV2 source/request/response/index lifecycleは実装済み。V2実artifact・model response・採点はまだ0/90 |
| 月次3倍 | `3X_NOT_REACHABLE` | 現在の有効証拠からは到達不能。サイズ逆算による帳尻合わせは禁止 |

修理後workerの実データ再審判も実施した。12設定をTRAINのOHLC/OLHC両経路、slippage 0.3 pips、
financing 0.8 pips/dayで計24試行した結果、通過0、VAL/FINAL進出0だった。trial内訳は
`INVALID_ZERO_TRADES` 6、`INVALID_TERMINAL_EXPOSURE` 10、
`FAIL_NON_POSITIVE_RESOLVED_BALANCE` 6、`FAIL_MARGIN_CLOSEOUT` 2。
scoreboardはarchiveの `codex-worker-rerun-v1/runs/20260718T175451.057612Z-3f168d34/scoreboard.json`
（SHA-256 `dbe2f5e4f441e6c723e44cee6d447955941d8d093e345e820b01f6dd949328fd`）。
過去窓を使うため陰性診断であり、新しい前向き証拠ではない。

既存の正値JSONやレポートは、後段で訂正文があるだけでは再利用事故を防げない。単調な無効化
レジストリと証拠bytesを再検証する機械判定が実装されるまでは、本文の無効化一覧を人間向け正本、
goal boardを安全側の診断器として扱う。無効な親から派生した採点・集計も無効である。

現在のcontent-addressed goal boardは
`research/registries/dojo_goal_board_20260719_62dec2849c3dcf62c5358471a54762b8d8e21423ac2caefa3c048a70dba6938d.json`。
判定は `HYPOTHESIS / 3X_NOT_REACHABLE`、外部検証済み独立clusterは0、
`proof_admission.promotion_possible=false` である。同じ日付の非content-addressed旧出力は途中診断であり、
現行判定として読まない。

前向きworkerの `precommit.json` はcommit、runner、bot、依存module、scorer、broker、12候補、両intrabar、
コスト、期間末owner決済を固定した。precommit canonical SHA-256は
`7d849052082049721c9f83d658190d163dc7aa00e7c17c25d7ec50e487d71c5d`、start receipt SHA-256は
`4b8fadbb452e304ae1e4f9721db3a8da93b212c08354b938abf1cda50a5523b9`。ただしこれは開始証拠であって
成績証拠ではない。この14暦日smokeはminimum 10 open-market daysを要求するが、proofの60 active daysを
構造的に満たさず、成功時も `promotion_eligible=false`、`live_permission=false` である。

AI phase evaluatorは登録済み90セル以外の部分集合を受理せず、欠測・schema failureを残り分母のFLATとして
集計し、A/B/Cを別NAVで評価する。応答はanswer keyを読む前に封印する。ただし新規cellはまだ作成しておらず、
market-derived answer key、provider能力隔離、外部attestationが未接続なので、これらを接続しないまま得た結果は
`SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC` を越えない。

旧 `QR_DOJO_PROMPT_PHASE_MANIFEST_V1` は全30将来日のsource/packet/request hashを応答前に要求していたため、
真のprospective runでは作成不能だった。V2はこれを事前scheduleと日次chainへ分離する。precommitが30個の
月〜木15:00 UTC cutoff、90 cell/context、model policy、prompt/scorer/code bindingを先に固定し、各cutoff後は
OANDAのread-only M5/BA応答からcutoff以前にcloseしたallowlist candleだけをsourceへ変換する。同日A/B/Cの
3 requestは一つの `QR_DOJO_AI_DAY_SOURCE_SEAL_V2` へ同時封印し、欠測日は3 cellともsynthetic FLAT failureとして
残す。各responseは固定schemaでdeadline前にanswer key未開封のまま個別封印し、response欠測もdeadline後に
synthetic FLATとして固定する。全truth horizon成熟後の90-cell phase indexはこのchainから導出し、事前固定証拠とは
呼ばない。V2 codeが存在するだけでは証拠に
ならず、precommit/start/day/response/truthの実artifactが必要である。

## 前向き日次の実行境界

workerの実運用source経路は `scripts/collect-dojo-worker-day.py` だけを使用する。旧
`run-dojo-worker-forward.py seal-day --source-manifest ...` はschema診断用であり、手書きmanifestを前向き証拠へ
使用しない。collectorはprecommitからUTC日を導出し、日終了2分後から12時間以内だけOANDAの
`USD_JPY / M1 / BA` read endpointへ接続する。全complete candleの時刻、重複、OHLC、bid/ask、固定coverage・最大gap・
両端制約を検証し、欠損slotをhashで保持する。content-addressed deterministic gzip、厳格parse後のcanonical response JSON、first-write capture、取得receipt、
source manifest、day sealを同じlock下でappend-only保存する。HTTP raw body・status・Date/RequestID headerはclient境界で
取得できず外部attestationでもないため、receiptの限界として残す。市場時間正本はOANDA公式
`https://www.oanda.com/us-en/trading/hours-of-operation/` のNew York時間「日曜17:05〜金曜16:59、毎日
16:59〜17:05の6分休止」と、OANDA aligned candleがclose境界およびreopenで終わるbucketを返す規則を版付きで固定する。
今回のordinal 1..14の許容M1 slot数は
`1436,1436,1436,1436,1260,0,176,1436,1436,1436,1436,1260,0,176` で、完全休場は6日目と13日目だけである。
OANDA実APIの直近14日preflightでは、open平日のcoverageは98.47〜100%、日曜部分日は85.71〜98.86%で、全slot必須は
正しい応答を拒否すると判明した。このため事前固定floorは通常日98%、部分日80%、最大連続gapと両端許容は各15分とする。
不足を補間・捏造せず、floor未満はその日を封印しない。

```bash
PYTHONPATH=src python3 scripts/collect-dojo-worker-day.py collect \
  --run-dir research/forward/dojo-worker-forward-smoke-v1 --ordinal 1
PYTHONPATH=src python3 scripts/collect-dojo-worker-day.py verify \
  --run-dir research/forward/dojo-worker-forward-smoke-v1 --ordinal 1
```

Day 1のsource windowは `[2026-07-20T00:00:00Z, 2026-07-21T00:00:00Z)`、安全な実行窓は
`[2026-07-21T00:02:00Z, 2026-07-21T12:00:00Z]`。取得失敗やopen日のzero rowをmarket closureへ読み替えない。
collectorはbroker write methodを持たず、credential/account idをartifactへ保存しない。既存precommit後に追加した
source producerなので、receiptには `SOURCE_PRODUCER_NOT_PRESENT_IN_ORIGINAL_PRECOMMIT` を永続的に残し、成功しても
diagnostic tierを越えない。

AI V2 operatorは `scripts/run-dojo-ai-forward.py`。`precommit` と `start` を最初のcutoff前に一度だけ作成し、各日
`collect-day` をsource-not-beforeから30分deadlineまでに実行する。CLIは時刻とstrict ordinalをnetwork接続前に
検証し、pair/range/granularity/priceをcallerから受け取らない。deadlineを失った日は後日sourceを差し替えず、
`seal-missing-day` で3 cell failureを固定する。`seal-response` は一つの固定schema responseをdeadline前にappend-only封印し、
`seal-missing-responses` は期限切れcellを上書き不能なFLATへ変える。30日・90 terminal・最終truth horizonが揃った後だけ
`seal-phase-index` が固定分母indexを導出する。全artifact writeはfsync済み一時inodeからhard-link publishする。
モデル実行とmarket-derived answer keyは別の未実装境界であり、このCLI自体はモデルを呼ばずanswer keyも生成しない。

## 共通の昇段語彙

- `INVALIDATED`: 未来参照、帰属、コスト、証拠鎖、または評価分割が壊れている。
- `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC`: 実行者自身の申告や公開checksumだけで、外部attestation、
  単調registry、market-derived key、能力隔離を証明できない診断。正負どちらのedge証拠にも使わない。
- `HYPOTHESIS`: 過去データで探索された候補。正のバックテストでもここを越えない。
- `EDGE_PROVEN`: 事前固定された独立holdoutで、全コスト後の正エッジと不確実性を満たす。
- `GOAL_COMPATIBLE`: `EDGE_PROVEN`に加え、破産・drawdown・証拠金制約込み分布が目標倍率と両立する。
- `FORWARD_PROVEN`: 未開封の前向き期間で、固定候補・固定評価器の必要標本を満たす。
- `LIVE_ELIGIBLE`: 別途レビューされた昇段契約を満たす。DOJOだけではこの状態を作れない。

`TAIL_ONLY`（分布の右尾でだけ3倍）を `GOAL_COMPATIBLE` と呼んではならない。同じモデルlineageの
複数応答、同一日、同一episode、同一通貨・同一因子は独立標本として水増ししない。

## 戦略ワーカーの評価契約

1. TRAINは仮説生成だけに使い、候補・パラメータ・実行vehicle・コスト・評価器を固定する。
2. VAL、最終holdout、prospective forwardは時間非重複とする。見た窓を名前変更して再利用しない。
3. replayは `epoch → intrabar phase → pair` の同期順とし、同時刻の別ペアの未来気配を見せない。
4. OHLCとOLHCを同格で走らせる。良い経路だけを次段へ送らない。
5. entry、TP、SL、手動close、margin closeoutの全fill pathで不利な執行を明示する。market/stop/SL/
   manual/closeoutには固定stress slippageを課し、価格保護されたLIMIT/TPは改善を削っても指値を越えて
   悪化約定させない。financingは強制決済まで含める。
6. botは自分のstrategy/owner tagに一致する建玉だけを管理する。他botの同一ペア建玉を採用しない。
7. 期間末の未決済建玉をmark-to-marketし、実現益だけのscoreboardを禁止する。
8. zero trade、欠損換算気配、欠損shard、破壊されたtrial provenanceは不合格とする。
9. 各sessionはcommit、bot/config/corpus SHA、期間、pairs、intrabar、granularity、cost、換算quote watermarkを封印する。
10. trial成果物はappend-only/content-addressedとし、再試行で旧trialを削除しない。

## AI裁量とプロンプト工学の評価契約

1. 一試行は一日・一fresh context。packetにはcutoff以前の観測だけを入れ、日付を匿名化する。
2. 答えkeyは応答を封印するまで物理的にmountしない。モデルはfilesystem、network、会話履歴を持たない。
3. prompt、variant、model/version/lineage、capability、packet、response、scorerをSHA-256で相互束縛する。
4. 出力schemaは `LONG|SHORT|FLAT`、pair、size、confidence、evidence refs、target、invalidation、反証、abstain理由に固定する。
5. prompt variantは実行前に登録し、同一の固定cohortへ割り付ける。結果を見て文章を直したrunは新experimentとする。
6. 評価者と生成者を分離し、採点器は封印済みanswer keyと実bid/ask・全コストだけを読む。
7. 同一model lineageの反復は一つの独立clusterとして扱う。多数決の人数を独立性の証拠にしない。
8. 汚染親、改変response、stale positive artifactを検出したら、答えkeyを開かず子孫まで無効化する。
9. incomplete runは診断値だけを出し、陽性・陰性の最終証拠に昇格させない。
10. prompt選定後は未開封のprospective cohortで一回だけ確認する。

## 月次3倍ゴールの判定

元本を30暦日で3倍にするには日次複利+3.7299%、22取引日なら+5.1205%が必要になる。判定は
単一点の倍率でなく、全コスト後のlog-return分布、最大drawdown、loss month、margin peak、ruin、
cluster依存性を同時に出す。

旧W46〜W53の無効化済みworker ledgerをfeasibility診断として再計算すると、55日で約x1.0668、
月換算約x1.0358、最大実現drawdown 11.21%、
最悪日 -6.32%、証拠金peak 53.53%だった。平均利益+350.77円、平均損失-1,002.54円、
profit factor約1.091で、約0.40 pipの追加コストが期待値を消す。92% marginまで単純拡大しても
月約x1.063に留まり、3倍へ逆算した約31倍sizingは試算でruin約74.6%だったため棄却する。
これは昇段証拠ではなく、目標と現状の桁差を測るためだけの診断である。

月次3倍へ進む条件は、サイズ拡大ではなく、相関の異なる複数の `EDGE_PROVEN` lane、前向きに
較正されたAIのabstain/selection edge、実執行コスト後のportfolio proofである。到達できないrunは
失敗ではなく、`3X_NOT_REACHABLE` として次の最小反証可能な実験を返す。

## 正本と保管

- 研究実装worktree: `/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval`
- 研究branch: `codex/dojo-dual-eval`
- 元DOJO worktree: `/private/tmp/QuantRabbit-episode-outcome`（未追跡証拠保全のためlock中）
- 検証済みarchive: `/Users/tossaki/App/QuantRabbit_archives/DOJO_20260719`
  - code bundle: `dojo-code-dc3179af4.bundle`
  - bundle SHA-256: `d873cc5db9774993f39fad0414bd828f0e50d9782d2792947b78688f7fde60f4`
  - research data: 463 files / 4,690,853,483 bytes（元とarchiveの全file SHA一致）
  - exact mirror manifest: `research-data-manifest-v1.json`
  - canonical manifest SHA-256: `70955036e6e43b0469d1f53a0cd62a127ae2c97f2b9f4a9d4814d55a1686b944`
  - manifest file-bytes SHA-256: `2ca175ed5399bfe18d72a196abaa777f5f16c8941a349df67c6504dbed4acf90`
  - source/mirror relative-inventory SHA-256: `daf8ebfc654c9e06105d5b80e2e51c5bf9f03c05ecbd26975ceddbd200b7b7da`
  - Fable AI handoff carrier: 122 regular files + archive外を指していた58 symlink
  - materialized Fable handoff: `fable-ai-discretion-materialized-v1`（全180 pathをregular file化し、元実体とSHA一致）
  - repaired worker rerun: `codex-worker-rerun-v1`（1.9GB、24 trials、TRAIN survivor 0）
  - supplemental archive manifest: `supplemental-evidence-manifest-v1.json`（上記materialized handoff、worker rerun、
    accidental runの計256 files / 2,057,789,055 bytes）
  - supplemental relative-inventory SHA-256: `7ecc9116bf80b66a954f8b30df2257e56c8b7a2d10d53e9d5d5bcb4e30faf18a`
  - supplemental manifest canonical SHA-256: `0374dbaddc5759490b7a43d1e235e76b755e1bb43cc14376adccee36925deb98`
  - supplemental manifest file-bytes SHA-256: `5262f2ff6e5d9b414733c2b8291e452a922b36d642935ea3ed0fe57ac15503f8`
- 仕組みの操作書: `docs/virtual_market_environment.md`
- 発見・訂正の時系列: `docs/design_weakness_ledger_20260718.md`
- Notion過去作業記録（履歴参照のみ。現行証拠/SSOTではない）:
  `https://app.notion.com/p/39cf1c8e53a781569171cc4d1de7ac2b`

元worktreeやarchiveを削除する条件は、追跡branchのpush、manifest検証、参照先の移行、owner確認が
すべて終わった後である。main/live/orchestratorのdirty worktreeはDOJO整理の名目で触らない。

## 次の実験順

1. worker smokeの各UTC日を順番どおり、当日終了後12時間以内にsource manifestごとappend-only封印する。
2. AI phase-1の30 blind-day assignment/packet manifestを応答前に固定し、90セルを一日一fresh contextで採取する。
3. response封印後にだけmarket-derived answer keyを接続し、欠測を落とさずA/B/C別NAVとpaired contrastを採点する。
4. `[2026-07-20, 2026-08-03)` 終了後にworker smokeを全12候補×両intrabarの固定分母でfinalizeする。
   この14日窓はsmoke/pilotであり、3ヶ月・worker 60 active days・AI 90 active daysの証明閾値を満たさない。
5. 同じ固定候補を事前宣言した継続窓で必要日数まで採取し、途中結果で候補や評価器を変えない。
6. 外部検証済み独立laneが増えるまでportfolioは `3X_NOT_REACHABLE` を維持し、実弾昇段を行わない。
