# DOJO（道場）評価プログラム

DOJOは、戦略ワーカーとAI裁量を同じ実市場メカニクスで鍛え、再現可能な証拠を持つ候補だけを
次の前向き段階へ進める研究環境である。月次3倍は研究目標であって保証ではなく、DOJOの成績は
実弾権限を一切与えない。ライブ運用の権限境界と最上位KPIは `docs/AGENT_CONTRACT.md` が優先する。

## DOJOの三段階

DOJOの本体は、過去のローソクとbid/askを時刻順に一段ずつ開示し、その時点で利用可能だった情報だけで
注文・約定・決済を再生する仮想市場である。未来の足、同じ足の未到来high/low/close、答えkeyを先読みさせない。

1. `稽古（TRAIN）`: Fableを含む過去市場replayを大量に回し、戦略・prompt・失敗条件を改善する。結果は仮説生成用であり、
   同じ期間へ適合した利益をedge証明に数えない。
2. `審査（HISTORICAL HOLDOUT）`: 稽古で一度も見ていない過去区間を封印し、固定した候補を一回だけreplayする。
   全コスト後の利益、drawdown、破産、両intrabar経路、欠測を固定分母で判定する。
3. `昇段確認（PROSPECTIVE）`: 審査通過候補だけを未到来期間で確認する。これはDOJOの最終確認であり、過去replayの代替ではない。

三段階の期間・prompt・parameter・scoreは混ぜない。現在は稽古の陽性候補と陰性結果があり、審査通過者は0、昇段確認は開始証拠のみである。
2024〜2026H1のM1/M5/S5は累計300超の探索と後続監査で摩耗しており、`GLOBAL_UNTOUCHED` と証明できる
過去窓は残っていない。既存履歴はTRAINに使えるが、裸の「未使用holdout」と呼ばない。新しいlineageに対する
`LINEAGE_UNSEEN_DIAGNOSTIC` と、事前固定後に新規取得する `PROSPECTIVE_SEALED` を区別する。
`research/registries/dojo-historical-holdout-burn-v1/events/` はこの区別をfirst-write SHA chainで記録する。
初期移行はgenesis + 4 legacy burn、tip SHA-256
`90f0d3ba8a771ea43b01a2bc2f11a51e61ecec9f69ad5e3432905be103effdc9`。
M5全価格pathを見た区間は、同じ時刻をENTRY/EXITや別時間足へ改名して予約できない。

## 2026-07-19時点の結論

| 対象 | 状態 | 結論 |
|---|---|---|
| W_FADE / W_SPIKE_FADE / W_ROUND / W_LADDERという戦略案 | `HYPOTHESIS` | アイデアは再試験可能だが、live昇段可能な生存者は0 |
| W46〜W53の旧worker成績証拠 | `INVALIDATED` | 誤帰属、同一建玉の複数bot所有、出口コスト欠落、摩耗holdoutがある |
| 複数通貨worker追加稽古 | `TRAIN / NO_SURVIVOR` | 3系統・4候補群・7ペアを新規replay。momentum 113戦 `-¥38,280`、spike-fade portfolio 90戦 `-¥20,871`、pullback 43戦 `-¥302`、burst control 336戦 `-¥6,622`。全候補群がコスト後負 |
| AIトレーナーによるworker再調整 | `TRAIN / 3 SURVIVORS` | 負け台帳からtail guard・低レバ化・短期化・低頻度pullbackを再設計。spike-fade 2候補とpullback 1候補が両intrabarで正。ただし既使用TRAINでproof生存者は0 |
| 型付きbot factory / 共通出口overlay | `IMPLEMENTED / RANKING BLOCKED` | AIが任意Pythonを書かず、審査済みbot族と範囲内parameterだけを提案するcatalogを追加。`FIXED / BREAKEVEN / ATR_TRAILING`を同じentryに後付け比較できるが、連続MTM mark契約の実装前は全候補をunrankedにするため、利益改善はまだ実測していない |
| 新規bot 2族 | `IMPLEMENTED / UNTRAINED` | London開始レンジ・ブレイクと週末ギャップ・リカバリーを実装。完成足、次足執行、完全D1 ATR14、NY17:00のDST対応週末境界、原子的TP/SL、spread、owner capをfail closed化。動的stop上限をconfigで封印するまではrank対象外 |
| 追加4 bot種 | `TRAIN / NO SURVIVOR` | 初回runは約定時上限バグで無効化。現行依存bytesで再実行した修正版8 runは上限違反0だが、悲観側はcompression `-¥846`、round-number `-¥412`、ladder `-¥2,905`、trailing `-¥5,557`で全棄却。round-numberのTPだけを2→3 ATRへ広げた別窓追試も両経路で負 |
| HOLD資金配分worker | `TRAIN / 3 POLICY SURVIVORS` | 元本20万円、11日21時間59分の同じ4通貨spike-fadeで480分full HOLD、5x分割、20x/60分解放を比較。悲観OLHCは順に `+¥90,672 / +¥43,704 / +¥166,991`。1か月実績ではなく、60分解放も摩耗TRAIN候補に留まる |
| W37 AI方向読み | `UNESTABLISHED` | 40場面を4読者へ分割した結果で、4×40の独立試行ではない。的中41.18%、Wilson 95% CI 26.4–57.8% |
| W39 AI出口読み | `INVALIDATED` | 判定時点と同じM5バーの高値・安値・終値を入力していた。点推定+0.562 pipsのcluster bootstrap CIも -1.699〜+2.453でゼロを跨ぐ |
| W54/W55 AI日次読み | `INVALIDATED` | 後続日の履歴から先行日の答えを読めるpacket lookahead。単ペア89/90日、複数ペア79/80日が露出 |
| W54 legacy clean再試行 | `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC` | 40日中19応答のみ。旧packetは日付を保持し、応答schema・prompt/model/scorer/key隔離receiptも新契約を満たさない。旧計算は15 commit、8勝、NAV 0.975293、月換算0.971832だが再現可能な最終判定には使わない |
| worker前向きsmoke v1 | `SUPERSEDED / DIAGNOSTIC` | source producerを事前束縛していなかった旧開始証拠。bytesは履歴として保持するが、現行runへ混ぜない |
| worker前向きsmoke v2 | `STARTED / HYPOTHESIS` | `[2026-07-20, 2026-08-03)` の12候補×OHLC/OLHC、collector、実行runtime、依存closureを事前封印。0/14日採取、0/24セル実行。将来日次証拠はまだ0 |
| AI prompt phase-1 v2 | `SUPERSEDED / DIAGNOSTIC` | 任意JSONをresponseとして搬入でき、実モデルのfirst-attempt実行証拠を固定できないため現行runへ混ぜない |
| AI prompt phase-1 v3 | `SUPERSEDED_STARTUP_INCOMPLETE / INVALID` | precommit/startは保存されたが、相対run-dir二重連結でvalidity genesis初期化に失敗。モデル・市場取得なし、上書きせず履歴保持 |
| AI prompt phase-1 v3r1 | `SUPERSEDED_BEFORE_SOURCE / INVALID` | 0/90、source取得0のまま停止。入力がW55と同じローソク構造だけで、既知のnullを再測定するため。precommit/start/supersession/validity chainは削除せず保持 |
| AI出口判断 TRAIN v1 | `INVALIDATED` | 1判断=1 fresh contextでも、判定バーの未到来high/low/closeを入力していた。記録値は保存するが比較に使わない |
| AI出口判断 TRAIN v2 | `INVALIDATED_SOURCE_GAP_CONTINUITY` | 判定barの先読みは除去したが、entryからHOLD終了までのM5連続性を再検査するとX05が欠測で失格し、X06〜X08のcohort IDも繰り上がった。AI -48.4 / CUT -63.8 / HOLD -29.6は保存値だけでpolicy比較に使わない |
| AI方向gate＋資金再配分 TRAIN v1 | `TRAIN / REJECTED ALLOCATION` | 6 fresh contextで既存方向の継続/無効は6/6、新規候補方向は3/6。ただしAI配分 -26.65 capacity-pipsはfull HOLD -19.30に劣後。事後gate rule -9.70は次の未検証仮説に限定 |
| AI方向gate＋資金再配分 TRAIN v2 | `TRAIN / REJECTED` | 2026H1の6 fresh contextを1判断=1コンテキストで固定評価。階層配分は -27.4 capacity-pipsでfull HOLDより+6.5改善したが、現金退避 -26.8にも0.6劣後。既存方向2/6、新規方向3/6、同時正解1/6 |
| 旧worker候補の別相場再稽古 | `TRAIN / NO SURVIVOR` | 2026-04の別相場で3候補×両intrabarを固定。pullbackは `-¥187 / -¥196`、tailguard 2候補は換算quote 120秒staleでfail closed。生存0 |
| 月次3倍 | `3X_NOT_REACHABLE` | 現在の有効証拠からは到達不能。サイズ逆算による帳尻合わせは禁止 |

修理後workerの実データ再審判も実施した。12設定をTRAINのOHLC/OLHC両経路、slippage 0.3 pips、
financing 0.8 pips/dayで計24試行した結果、通過0、VAL/FINAL進出0だった。trial内訳は
`INVALID_ZERO_TRADES` 6、`INVALID_TERMINAL_EXPOSURE` 10、
`FAIL_NON_POSITIVE_RESOLVED_BALANCE` 6、`FAIL_MARGIN_CLOSEOUT` 2。
scoreboardはarchiveの `codex-worker-rerun-v1/runs/20260718T175451.057612Z-3f168d34/scoreboard.json`
（SHA-256 `dbe2f5e4f441e6c723e44cee6d447955941d8d093e345e820b01f6dd949328fd`）。
過去窓を使うため陰性診断であり、新しい前向き証拠ではない。

追加の複数通貨稽古は `research/training/dojo-worker-multipair-train-20260719/evidence.json` に索引した。
対象は7ペア、モメンタム・spike fade・breakout pullback・低レバburst controlの4候補群。各runは
実M1 bid/ask、OHLC/OLHC、0.3 pip/fill、0.8 pip/day、owner isolation、期末全決済を使った。
全4候補群が保守経路で負となり、生存者は0。長期multi-pair runで換算quoteがstaleになったものは
補間せずfail closed artifactとして保持した。

AIトレーナーはこの負け台帳を入力にして、別TRAIN窓で実際の再売買まで行った。索引は
`research/training/dojo-worker-ai-tuning-20260719/evidence.json`。spike-fadeへ25-pip hard SLを追加した
候補は悲観側105戦 `+¥37,174.62`、30日換算1.5360、実現DD 3.38%、設定上ピーク証拠金68.8%。
同じSLに240分ceiling・per-position leverage 2を組み合わせた候補は106戦 `+¥17,351.15`、
30日換算1.2330、実現DD 1.58%、設定上ピーク証拠金32%。4つのsealed ledgerを再走査し、各pair最大1、
全体最大4の違反0を確認した。低頻度pullback A2も悲観側93戦 `+¥170.24`、30日換算1.000912で
小さなTRAIN仮説として残った。momentum調整2候補は負のまま棄却した。従ってTRAIN生存者は3だが、
摩耗済み履歴なのでproof生存者・昇段者は0である。

新しいbot factoryでは、AIの役割を「審査済みcatalog内の候補提案」までに限定する。
candidate設定、prompt、入力、raw response、model claim、source closure、市場窓、コストを結果開封前に
SHA-256で封印し、決定論的scorerだけが固定4セル `OHLC/OLHC × BASE/STRESS` を比較する。
TRAIN内の順位は仮説優先度であり、proof、昇段、live注文権限のいずれも生成しない。
出口はentryと分離し、同じsignal列に `FIXED / BREAKEVEN / ATR_TRAILING` を後付けする。BEは
entry ATRでtriggerして建値＋固定offsetへ一度だけ移動、ATR trailingは完成済みbarのpeak/troughから追随し、
どちらも既存TPを保ちSLを絶対に緩めない。指値がbar途中でfillした初回barのhigh/lowは使わない。

trainerはmainと全LOPOの台帳をseal時・評価時に再読込して再採点し、相対path、file SHA/size、
hash-chain終端、metrics、corpusを照合する。全feed pairのM1 epoch集合が一致しない部分欠損、
換算feedを落とすLOPO、caller自己申告metrics、symlink/traversal、途中書換えは失敗にする。
repo所有risk policyはposition 5x、pair 2枠、global 4枠、gross 20xをhard上限とし、
28日未満、STRESS 0.3 pip/fill・0.8 pip/day未満、緩いDD/margin/reject/集中度閾値をsealしない。
strict TRAIN replayは全atomic quote batchの座標とbatch-chain終端を結果開封前に封印し、各phaseの
bot action後に `ACCOUNT_MARK` を残す。scorerは封印済みgzip corpusからquoteを逐次再生し、必須の
指値約定・SL/TP・margin closeout、注文、position、残高、financing、hedge-net証拠金を独立再構築する。
欠落event、H/L/Cへ移したbot action、自己整合する偽account、corpusと異なるquoteは失敗にする。
legacy replayは連続MTM証拠へ昇格せず、`mtm_complete=false` のまま扱う。

本番候補の運用は、固定bot runtimeとtrainerを分離する。監視はレジーム、コスト、DD、reject、
集中度が実質変化した時のevent-driven、再TRAINは週次を目安、昇段審査は月次以下の低頻度とする。
live中の自己書き換えと自動昇段は行わず、変更候補は次のTRAINとshadow/forward窓からやり直す。

追加した4 bot種の初回runでは、compression breakとfade ladderが同一バー約定時に設定上限を超えた。
`VirtualBroker` の実約定境界でowner別pair/global上限を再検査し、超過注文を台帳付きで取消・拒否するよう修正した。
旧diversity scoreboardは `INVALIDATED_FILL_TIME_CONCURRENCY_CAP_NOT_ENFORCED` とした。修正版は別artifact
`codex-worker-diversity-capsfixed-train-v2` で4種×OHLC/OLHCを再売買し、全8 runでpair最大1、全体最大4、
上限違反0、owner不一致0を確認した。超過待機注文はpair capで213件取消され、global cap拒否は0。
悲観側はcompression break 29戦 `-¥845.73`、round-number fade 81戦 `-¥411.63`、fade ladder 471戦
`-¥2,904.78`、trailing burst 271戦 `-¥5,557.26`で、追加4種のTRAIN生存者は0だった。
同じ依存bytesでのv2 scoreboard raw SHA-256は
`690af01b4beb19a4e4c40982638b1c7a14f495e5ada34b5a3be3effd46d7a075`。最小損失のround-numberだけを
選び、TPを2.0 ATRから3.0 ATRへ広げた1仮説を別TRAIN窓で追試したが、OHLC 134戦 `-¥1,719.70`、
OLHC 136戦 `-¥1,244.79`で棄却した。勝幅拡大だけでは勝率低下を補えなかった。

HOLDの機会損失は、同じ4通貨spike-fade signal列を三つの資金政策へ通して実売買した。元本は20万円、
実測期間は2025-03-03 00:00から2025-03-14 21:59 UTCまでの11日21時間59分であり、1か月ではない。artifactは
`codex-capital-hold-opportunity-train-v1`、scoreboard raw SHA-256は
`dc00742fe594ae66539f7246d06cb73e14bb352dec45a87d2cd70b30505eda2a`。悲観OLHCで480分full HOLDは
60戦 `+¥90,671.91`、実現DD 10.47%、証拠金不足拒否48。5x分割+余力予約は105戦 `+¥43,704.45`、
DD 4.25%、拒否0。20xのまま60分で古い建玉を解放すると68戦 `+¥166,990.89`、DD 9.30%、拒否43となり、
両intrabarで `60分解放 > full HOLD > 分割` だった。方向勝率は75–77%で近く、差は方向当否より資金回転から
生じた。ただし約80%というmargin補助値はentry-notional/realized-NAV event時点で、連続MTM使用率ではない。
終了残高は悲観OLHCで36万6,990.89円だった。30日換算4.6101もこの摩耗TRAIN短期窓を機械的に
複利外挿した値で、観測された1か月成績でも月次3倍のproofでもない。DD 9.30%はrealized exit event基準で、
連続mark-to-market DDではない。
独立来歴監査は `research/audits/dojo-capital-hold-opportunity-v1-audit-20260719.md`。保存台帳の
`+¥166,990.89` と価格・費用計算は再確認したが、実行時source bytes 2本と連続MTM markが欠けるため、
exact再実行、未使用相場edge、月次3倍の証明には昇格しない。

同時刻candidateを資金効率順に記録する `dojo_portfolio_allocator.py` V4は、open/pending/candidateのgross margin、
gross stop-loss risk、通貨shadow、owner別同時保有上限を相殺なしでfail closed検査する。JPY建てペアのみ許可し、
非JPY建ては独立検証可能な市場換算receiptができるまで拒否する。`dojo_allocation_execution.py` V1はcontent-addressed
receiptのうち既存建玉を解放しない `HOLD_FULL` planだけをin-process `VirtualBroker` へ一回限り提出できる。
broker state・cost設定・owner cap・nonceを再検査し、同一allocation+intentの再実行をledgerで拒否する。
判断quoteのUTC秒・ローソク内phase・batch watermarkを全open pairとselected pairで完全一致させるが、
fill時のportfolio全体SL再評価とrelease+entryの原子性は未実装である。外部broker/live権限はなく、proofにも使わない。

既存の正値JSONやレポートは、後段で訂正文があるだけでは再利用事故を防げない。AI V3にはfirst-write
artifact registryと推移的無効化を実装したが、新V3 genesis以後のローカル証拠だけが対象であり、旧artifactを
遡及的に昇格させない。外部の単調witnessはなお存在しないため、本文の無効化一覧を人間向け正本、
goal boardを安全側の診断器として扱う。無効な親から派生した採点・集計も無効である。

現在のcontent-addressed goal boardは
`research/registries/dojo_goal_board_20260719_11c2ac8c81d4f716a08f2ee1cdadad062b632c39b82f09ffe4678827e4525ba6.json`。
判定は `HYPOTHESIS / 3X_NOT_REACHABLE`、外部検証済み独立clusterは0、
`proof_admission.promotion_possible=false` である。同じ日付の非content-addressed旧出力は途中診断であり、
現行判定として読まない。

前向きworker v2の `precommit.json` はcommit
`107e57497dbe70e718827a90df48967749a8b775`、runner、bot、collector、source/calendar/OANDA、
lab scorer、VirtualBrokerを含む依存closure、12候補、両intrabar、コスト、期間末owner決済を固定した。
precommit canonical SHA-256は
`b5240ca36ce84dd00945d8d307bf65ceb2f2b0688849b0d102680436fb22b06a`、start receipt SHA-256は
`36fc12db99e81a7a3f6a3cb0b539ee9b55e4bd87c2d4ae58b537ac1a72591a2e`。runtimeは解決済みPython path・
executable SHA・versionまで封印する。旧v1はsource producerを事前束縛していないため
`SUPERSEDED / DIAGNOSTIC` とし、追跡bytesだけを保持する。ただしv2も開始証拠であって成績証拠ではない。
この14暦日smokeはminimum 10 open-market daysを要求するが、proofの60 active daysを構造的に満たさず、
成功時も `promotion_eligible=false`、`live_permission=false` である。

workerの成績値はcaller supplied result manifestから受け取らない。14個のsource receiptとexact corpusを再検証し、
隔離runtimeで12候補×OHLC/OLHCの24セルを固定runner/VirtualBrokerへ一度だけ通し、corpusの実行前後hash、
attempt body、ledgerを封印して結果を導出する。中断時の再開は完全にscoreableな既存ledgerだけを採用し、
別attemptから良い方を選べない。それでも現段階のledgerは経済event replayと外部witnessを持たないため、
全result rowを `INVALID_UNSCOREABLE_TRIAL` とし、smoke gateを通過させない。これは偽の陽性昇段を防ぐ意図的な境界である。

AI phase evaluatorは登録済み90セル以外の部分集合を受理せず、欠測・schema failureを固定分母のFLATとして
集計し、A/B/Cを別NAVとpaired contrastで評価する。応答封印後だけ、同一cutoffから24時間のOANDA M5/BA truthを
取得し、方向・size、0.3 pip/fill、0.8 pip/day financing、固定1倍notionalで採点する。answer keyはpacketとresponseへ
束縛され、phase rowからday/cell/variantを再構成して改変を拒否する。v2 precommit canonical SHA-256は
`6e02b6ac43329afb50d793e40658cb8dc11700e1122d39f3408f387b4ceef431`、start receipt SHA-256は
`fbcf8d55d2665ac95b490fd27835cd780635b8d6682e33c6df6d81206c2aee6a`。最初のcutoffは
`2026-07-22T15:00:00Z`、source取得開始は同日`15:02:00Z`である。まだ実response/truth/scoreは0であり、
CLIへcaller JSONを渡す経路はprovider model identity、fresh-context、能力隔離を証明しない。モデルexecutor、
provider attestation、外部単調witness、V2の推移的invalidity registryを接続するまで、結果は
`SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC` を越えない。V3は実行request/receipt、raw stdout/stderr、
success/failure terminalを同じfirst-write registryへ登録し、任意importを採点対象から除外するが、
provider-signed model snapshotと外部単調witnessがない限界は残る。

停止したAI V3R1のprecommit canonical SHA-256は
`d1c8e7d6c1c39b655154da21a05f9d4d7503c62cdc55b08d30addd43347251a3`、start receipt SHA-256は
`b1bb63e67fce62168bb4f0db487c0a1a984df8f69eff4b45d944406c0e9dd198`である。状態は
`SUPERSEDED_BEFORE_SOURCE`、固定分母90、実行0、source取得0。停止理由は
`REDUNDANT_CANDLE_ONLY_INPUT_CLASS` で、supersession canonical SHA-256は
`d665e0ea61d24ad59fc304b60ec8c65a109b9ad3849c861e88bc8eadd80eebd3`。
失敗した最初のV3 startupは別run directoryへappend-only保存し、V3R1へ混入させない。
このsupersession内の旧「出口判断+0.6 pips」は後にsame-bar H/L/C lookaheadと判明したため、successor選定根拠として
失効した。元bytesは変更せず、`research/corrections/dojo-ai-forward-v3r1-supersession-v1.json`
（raw SHA-256 `ca4d7b1662c129a5cf598210d3d20b453db6368a97b7f14e61db6bf88ef673bf`）で訂正した。
V3R1を0/90で停止した状態とローソク-only nullの停止理由は、この旧正値に依存しない。

最初の出口判断TRAIN v1は `research/training/dojo-ai-exit-train-v1/evidence.json` に保存したが、
判定をbar openで行うのに同じM5 barのhigh/low/closeを見せる先読みがあり無効化した。修正版V2は
`research/training/dojo-ai-exit-train-v2/evidence.json`。判定barを丸ごと除外し、直前に完了したM5 36本と
H1 close 120本だけを1判断=1 fresh contextへ渡した。保存値はAI `-48.4 pips`、常時CUT `-63.8 pips`、
常時HOLD `-29.6 pips`。しかしgeneratorへentry/decision bar内TP touchとentryからHOLD終了までの連続M5を
必須化すると、元X05はsource gapで失格し、元X06〜X08は後続IDへ繰り上がった。完全8-cell cohortが変わったため、
この合計とHOLD順位を `INVALIDATED_SOURCE_GAP_CONTINUITY` としpolicy比較から外す。修正版40件はarchiveの
`codex-ai-exit-scenarios-corrected-v2` に固定し、元bytesと訂正後bytesを両方保持する。

続くAI資金再配分v1は `research/training/dojo-ai-capital-recycle-train-v1/evidence.json` に索引した。
6つの独立contextで既存方向の継続/無効gateは6/6、新規候補方向は3/6だったが、確信度に応じて既存建玉を
25–75%へ縮小した配分は合計 `-26.65 capacity-pips` で、full HOLD `-19.30`に7.35劣後した。唯一既存方向を
`FLAT` としたcellでは乗換により `-20.8 → -11.2`へ改善した。従って次promptは方向gateと配分を分離し、
既存方向が有効なら原則full HOLD、無効時だけrecycleする。これを同じ6件へ当てた `-9.70` は結果開封後の
仮説なので、別TRAIN窓で事前登録するまで成績に数えない。

別の2026H1 worn-TRAIN 6件でこの階層policyを先に固定したv2は
`research/training/dojo-ai-capital-recycle-train-v2/evidence.json`。全cellを1判断=1 fresh context、toolsなしで封印し、
全6応答のmanifest確定後にだけanswer keyを生成した。結果は階層policy `-27.4`、full HOLD `-33.9`、常時rotate
`-47.2`、cut-to-reserve `-26.8` capacity-pips。HOLDより6.5改善したがreserveにも0.6負け、既存方向2/6、
新規方向3/6、同時正解1/6のため棄却した。方向gateの小改善を利益edgeと読み替えない。
このcross-pair raw-pip合計は正規化した方向policy比較であり、pair別pip価値・証拠金・financing・相関を含む
円建て資金損益ではない。

旧TRAIN生存3候補の別相場固定再売買は
`research/training/dojo-worker-survivor-regime-replay-train-v1/evidence.json`。pullback A2は両経路で負、tailguard 2候補は
USD/JPY換算quoteの120秒staleでfail closedとなり、別相場生存者は0だった。失敗4cellを損益0や成功へ数えない。

read-only OANDA取得条件は、実APIのM1 14窓・M5 8窓を用いたpost-hoc calibrationでも検査した。
`research/calibration/dojo_oanda_coverage_20260719_v1.json` のcanonical SHA-256は
`2d61fff0b93d44215e8a4cbba56dd62e06c066505b7989ecdf90d7674db31e1d` で、全coverage gateは通過した。
これは取得契約の較正だけであり、前向き成績、edge、昇段、live権限の証拠ではない。

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
  --run-dir research/forward/dojo-worker-forward-smoke-v2 --ordinal 1
PYTHONPATH=src python3 scripts/collect-dojo-worker-day.py verify \
  --run-dir research/forward/dojo-worker-forward-smoke-v2 --ordinal 1
# 14日すべてのsource sealが揃った後だけ実行する
PYTHONPATH=src python3 scripts/run-dojo-worker-forward.py evaluate-derived \
  --run-dir research/forward/dojo-worker-forward-smoke-v2
PYTHONPATH=src python3 scripts/run-dojo-worker-forward.py verify-derived \
  --run-dir research/forward/dojo-worker-forward-smoke-v2
```

Day 1のsource windowは `[2026-07-20T00:00:00Z, 2026-07-21T00:00:00Z)`、安全な実行窓は
`[2026-07-21T00:02:00Z, 2026-07-21T12:00:00Z]`。取得失敗やopen日のzero rowをmarket closureへ読み替えない。
collectorはbroker write methodを持たず、credential/account idをartifactへ保存しない。v2はcollectorから
OANDA read clientまでproducer closureをprecommitへ束縛し、取得時にはcurrent bytesとpinned commit bytesを両方検証する。
HTTP raw body・status・Date/RequestIDの外部attestationがない限界は残るため、成功してもdiagnostic tierを越えない。

AI V3 operatorは `scripts/run-dojo-ai-forward.py`。V3 `precommit` と `start` を最初のcutoff前に一度だけ作成し、各日
`collect-day` をsource-not-beforeから30分deadlineまでに実行する。CLIは時刻とstrict ordinalをnetwork接続前に
検証し、pair/range/granularity/priceをcallerから受け取らない。deadlineを失った日は後日sourceを差し替えず、
`seal-missing-day` で3 cell failureを固定する。外部terminal/schedulerから `tools/run-dojo-ai-model-cell.py` を
一セル一回だけ起動する。runnerは空cwd・空HOME・allowlist環境・ChatGPT auth・tool無効・ephemeral processを要求し、
最初のraw transcript以前のlaunch intentも封印する。timeout、schema違反、tool event、複数final、process crashは再試行せず
execution failure terminalとして固定する。`import-diagnostic-response` は任意JSONを診断搬入できるだけで採点対象にならない。
`seal-missing-responses` は期限切れcellを上書き不能なFLATへ変える。30日・90 terminal・最終truth horizonが揃った後だけ
`seal-phase-index` が固定分母indexを導出する。全artifact writeはfsync済み一時inodeからhard-link publishする。
各response deadline後かつ24時間horizon成熟後に `collect-truth-day` を実行し、`seal-phase-score` はexact 90 terminalと
90 truth scoreから固定分母phase scoreを導出する。QuantRabbitのOpenAI API経路は使用しない。provider identity、backend
snapshot、provider側fresh state、base prompt/full requestはclientから完全には観測できないため、V3実行も外部attestation
なしには `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC` を越えない。

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
  - materialization receipt: `fable-ai-discretion-materialization-receipt-v2.json`
  - receipt canonical SHA-256: `ec07873896c254fb94b132d24223f634646afeb0ee4381beab59eef7e0bc51cf`
  - receipt file-bytes SHA-256: `bd8484b5a3ba39f6af98ed74bbac0d3fd340ca1300db1c5623108991d5c43eb5`
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

1. 全quote batchの期待座標commitment、bot action後の連続 `ACCOUNT_MARK`、corpus bytesからの独立再構築は実装済み。次は型付きbot factoryの初回TRAINとして、固定entryへの `FIXED / BREAKEVEN / ATR_TRAILING` を別グリッドで固定する。正式開始前に全candidate、窓、コスト、source、prompt/model/input/raw response、探索分母を封印する。
2. round-number TP 2→3 ATRの別TRAIN追試は両経路で負となったため凍結し、同familyの後追いparameter探索を止める。
3. 資金占有3案の比較は完了。次は5x基礎分割へ信頼度順に未使用余力を動的上乗せし、60分stale解放を組み合わせる。全candidate intent、pending影余力、通貨因子、拒否signalを順序非依存allocatorで記録する。
4. `KEEPならfull HOLD / FLATならrecycle` の固定TRAIN v2もreserveに劣後したため棄却する。次のAI裁量はローソク方向当てを増やさず、公開時刻を封印できるニュース/カレンダー/フローか、出口判断の独立前向き課題へ限定する。
5. 旧TRAIN生存3候補の別regime固定追試は生存0で完了。pullbackは凍結し、tailguardは換算source同期の契約修理後も同じ固定設定・固定分母でだけ再試行する。
6. worker v2 smokeの各UTC日を順番どおり、当日終了後12時間以内にsource receiptとday sealをappend-only封印する。
7. AI entryの前向き版は、ニュース・経済カレンダー・フローの公開時刻とcutoff時点bytesを封印できる因果的multi-source契約を作ってから別experimentとして開始する。
8. 各response封印後かつhorizon成熟後にmarket truthを採取し、欠測を落とさず固定分母でpaired contrastを採点する。
9. `[2026-07-20, 2026-08-03)` 終了後にworker v2を全12候補×両intrabarの固定分母でderived executionする。
   この14日窓はsmoke/pilotであり、3ヶ月・worker 60 active days・AI 90 active daysの証明閾値を満たさない。
10. worker経済event replayをsource coverage・pinned bot actionへ接続し、両laneへ外部monotonic witnessを追加する。
11. 同じ固定候補を事前宣言した継続窓で必要日数まで採取し、途中結果で候補や評価器を変えない。
12. 外部検証済み独立laneが増えるまでportfolioは `3X_NOT_REACHABLE` を維持し、実弾昇段を行わない。
