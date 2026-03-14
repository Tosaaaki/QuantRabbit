# QuantRabbit 開発履歴議事録（再構成版）

## 0. この文書の位置づけ

- この文書は実会議の録音や transcript ではなく、`git log`、主要仕様書、運用 docs から再構成した「開発判断の議事録」です。
- 対象期間は `2025-06-21` から `2026-03-14` までです。
- 対象 commit 数は `2453` 件です。
- 主な参照元は `git log`、`AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/WORKER_REFACTOR_LOG.md`、`docs/CURRENT_MECHANISMS.md` です。
- 読み方は「その期間に何を議題にし、何を決め、何を次へ持ち越したか」を追う想定です。

## 1. 作成時点の作業前メモ（2026-03-14 JST）

- USD/JPY 現在値は `159.733` 近辺、`bid=159.727`、`ask=159.739`。
- スプレッドは `1.2 pips`。
- `M1 ATR14=1.3 pips`、直近 `15分レンジ=3.1 pips`、直近 `60分レンジ=10.9 pips`。
- 直近 6 時間の `orders.db` は `filled=155`、`close_ok=11`、`reject_like=533` で、reject の大半は `entry_probability_reject`。
- `2026-03-14 08:06` から `08:08` JST に OANDA pricing stream の `503` 再接続が連発したが、`08:08:36` JST に `200 OK` へ復旧し、`09:50` JST 以降の candle seed と stream 再接続も正常。
- 文書化タスク自体は継続可能と判断。

## 2. 全体タイムライン

| 期間 | commit 数 | 主な編集面 | その時期の主題 |
| --- | ---: | --- | --- |
| 2025-06 | 20 | `analysis`, `cloudrun`, `infra`, `market_data` | 初期 import、指標計算、最初の戦略、GCP/Cloud Run 前提の立ち上げ |
| 2025-07 | 43 | `analysis`, `market_data`, `execution`, `strategies` | secrets、tick/candle 安定化、履歴取得、運用下地の整備 |
| 2025-08 | 6 | `infra`, `analysis`, `cloudrun` | news pipeline 修復と継続判断 |
| 2025-10 | 171 | `execution`, `main.py`, `scripts`, `analysis`, `workers` | autotune、GPT、scalp/macro 拡張、range/bias 強化 |
| 2025-11 | 140 | `workers`, `execution`, `main.py`, `scripts` | worker suite 導入、addon worker、plan executor 化 |
| 2025-12 | 316 | `workers`, `execution`, `systemd`, `strategies` | worker-only への転換、共通 exit 廃止、news 導線撤去 |
| 2026-01 | 457 | `workers`, `systemd`, `scripts`, `config` | precision scalp 拡張、reentry/exit hardening、local-only 判定化 |
| 2026-02 | 883 | `workers`, `docs`, `ops`, `systemd`, `tests` | V2 split、entry intent 契約、feedback/pattern/forecast loop、ping5s 集中調整 |
| 2026-03-01〜14 | 417 | `docs`, `tests`, `workers`, `ops`, `scripts` | local-v2 専用運用、setup-scoped 改善、trade findings 規律化 |

補足:

- `2025-09` は現存履歴上の commit が確認できませんでした。
- 履歴全体として、設計の重心は `cloud/news/GPT` から `worker-only/local-v2/local_decider` へ段階的に移っています。

## 3. 2025-06-21〜2025-08-31
### 初期構築と cloud/news 実験

- 期間 commit 数: `69`
- 主な議題:
  - リポジトリの初期 import とディレクトリ整理
  - tick/candle/factor の基本パイプライン
  - 最初の strategy と order/risk の骨格
  - GCP/Terraform/Cloud Run と news 要約導線の導入
  - AGENTS と secret 管理の初期ルール
- 決定事項:
  - OANDA の USD/JPY を単一対象にした async データ取得とテクニカル判定を基盤にする。
  - 当初は `GCP + Cloud Run + news summarizer` を運用経路に含める。
  - エージェント運用ルールを `AGENTS.md` へ早期に固定する。
- 議論メモ:
  - `2025-06-21` `fc535130` で初期 commit。
  - `2025-06-23` `76bc5ba7` で repo root へ import。
  - `2025-06-23` `17f4c3b8` で async candle/tick fetcher を追加。
  - `2025-06-23` `57e5ee81` で `IndicatorEngine` と factor cache を実装。
  - `2025-06-23` `3dd89342` で market order と risk 計算を追加。
  - `2025-06-23` `48ef93bd` で `Donchian55` / `BB_RSI` を投入。
  - `2025-06-23` `eb73c380` で Terraform による GCP 基盤を追加。
  - `2025-06-23` `5e776c30` で初代 `AGENTS.md` を追加。
  - `2025-06-26` `57776dd6` と `83f0c2e4` で Cloud Run/news fetch 導線を拡張。
  - `2025-07-13` `f25bf5ed` で Secret Manager を導入。
  - `2025-07-13` `e8f9245a` で env file からの secret load に対応。
  - `2025-07` 後半は WebSocket/tick まわり、historical candle fetch、lint/format CI を強化。
  - `2025-08` は news pipeline 修復が主題で、`ニュース要約でイベント時刻とインパクトを保持`、`fix(news): ニュースパイプラインの修復とリファクタリング` が残る。
- 持ち越し:
  - news 導線は動くが壊れやすく、後年まで「残すか外すか」の論点を引きずる。
  - cloud 依存とローカル判定の役割分担がまだ未整理。

## 4. 2025-10-01〜2025-10-31
### Autotune・GPT・scalp/macro 拡張

- 期間 commit 数: `171`
- 主な議題:
  - autotune UI と BigQuery/Cloud Run 周辺の整備
  - GPT decider、advisor、news worker の接続
  - scalp pocket と range-mode の拡張
  - stage tracker、exit telemetry、replay/backtest の導入
  - H1/macro bias と opportunistic macro の追加
- 決定事項:
  - main loop の上で複数 pocket と GPT 補助判断を動かす方向へ広げる。
  - autotune とレビュー UI を運用の一部に組み込む。
  - range でも bias があるなら macro/scalp を通す設計へ寄せる。
- 議論メモ:
  - `2025-10-22` は stage lot 丸め、range mode、ATR-linked TP/SL、risk 1% 化など execution 基盤の再調整が集中。
  - `2025-10-23` `abb0a9d0` で autotune backtesting pipeline、`5f5311d9` で autotune dashboard の Cloud Run 配備を追加。
  - `2025-10-23` `64c2f4a8` と `d7529235` で partial close と trade schema を強化。
  - `2025-10-23` `7ca594d2` / `4dc94ba1` で projection ベースの entry/exit を backtester と戦略へ統合。
  - `2025-10-24` `355f48ba` で spread guard baseline、`84f991a0` で thesis-aware exit + MFE guard を導入。
  - `2025-10-25` は GPT worker queue、diagnostic logging、model fallback、news worker service の commit が連続し、LLM 利用が濃くなる。
  - `2025-10-29` `f9fbacea` で deploy/legacy guide を更新。
  - `2025-10-30` `cad55557` / `ade81965` / `eada4500` で fast scalp pocket を本格統合。
  - `2025-10-31` `5673fae7` と `f0d1eead` で range-bias / range-macro を許可。
  - `2025-10-31` `3734c0fb` 以降で opportunistic macro probe を追加。
- 持ち越し:
  - GPT と news を足したぶん、システム複雑度と障害面積が大きくなる。
  - main loop 集約のままでは戦略数増加に対する分離が弱い。

## 5. 2025-11-01〜2025-11-30
### Worker suite 導入と plan executor 化

- 期間 commit 数: `140`
- 主な議題:
  - worker suite と deploy doctor の導入
  - addon worker の追加
  - policy bus / micro core / plan executor の整備
  - macro snapshot と adaptive gating/logging
  - 新戦略の追加と spread-aware protection
- 決定事項:
  - monolithic main から worker 群へ責務を分離し始める。
  - macro/scalp/micro を pocket 単位で plan/executor 化する。
  - 取引量を止めるのではなく、gating/logging を厚くしながら改善する。
- 議論メモ:
  - `2025-11-02` `8203e584` で worker suite、tuner pipeline、deploy doctor を投入。
  - `2025-11-03` `812aa8d2` で s5 workers の rollout を完了。
  - `2025-11-04` `9fc464c5` で `mtf_breakout`、`session_open`、`stop_run_reversal`、`vol_squeeze` など addon workers を追加。
  - `2025-11-06` `545fc8a0` / `bd13a31c` で policy bus と micro_core を導入。
  - `2025-11-07` `d3ed7ecd` で macro/scalp plan executors と main delegation gating を追加。
  - `2025-11-11` `b1d32e7f` で micro dynamic cooldown、`66262380` で session strategy roadmap を文書化。
  - `2025-11-18` `b0de381e` で micro pocket adaptive gating と logging。
  - `2025-11-21` `40183185` で `ImpulseRetrace` / `MomentumBurst` を追加。
  - `2025-11-24`〜`11-25` は spread-aware protection、macro snapshot freshening、entry unstick、micro gate 緩和が連続。
  - `2025-11-27` `60a5b5cb` で pattern stats refresh guard を追加。
- 持ち越し:
  - worker 化は進んだが、entry/exit の共通ロジックがまだ厚く、戦略ごとの差分が埋もれやすい。
  - VM 上の運用・deploy 作業がまだ強く残る。

## 6. 2025-12-01〜2025-12-31
### Worker-only への転換と news 導線の撤去

- 期間 commit 数: `316`
- 主な議題:
  - exit 設計の戦略別分離
  - worker-only mode への移行
  - news 導線の停止と purge
  - hedge balancer と exposure guard
  - exit を entry meta に寄せる整合
- 決定事項:
  - 共通 exit worker を段階的に廃止し、strategy ごとの dedicated exit へ寄せる。
  - main trading path を主役から降ろし、worker-only を正方向にする。
  - news strategy / news pipeline は撤去方向で整理する。
- 議論メモ:
  - `2025-12-02` `463c3543` で news strategies を停止。
  - `2025-12-09` は exit/alloc tuning と strategy score snapshot が中心。
  - `2025-12-12` は dynamic exit、tech composite、M1Scalper guard、lot/exit scaling の密度が高い。
  - `2025-12-23` `4521233f` と `59e9ec20` で per-strategy exit workers と technical exits を追加。
  - `2025-12-24` は複数戦略で「entry meta を exit に持ち込む」整合作業が集中。
  - `2025-12-26` `f5db9a51` で worker-only mode を導入し main trading を guard。
  - `2025-12-26` `4cdef5ad` で common exit workers を削除。
  - `2025-12-27` `8e4b664c` で core workers を外し per-strategy exits へ進める。
  - `2025-12-29` `19c39bbf` で common exit manager を既定無効化、`a4d80e33` で news remnants を purge。
  - `2025-12-30` `d6f0024c` で hedge balancer と side/gross exposure guard を追加。
  - `2025-12-31` は free margin が薄くても net-reducing hedge を通す方向へ調整。
- 持ち越し:
  - main loop 由来の設定や legacy 経路がなお散在。
  - worker-only にしたぶん、position/order のサービス化が次の論点になる。

## 7. 2026-01-01〜2026-01-31
### Reentry / Exit hardening と local-only decision への収束

- 期間 commit 数: `457`
- 主な議題:
  - precision scalp と strategy 別 exit の拡張
  - reentry / negative exit / tech-loss exit の精緻化
  - autotune・dashboard・ops policy の強化
  - strategy-aware risk scaling と hedge 管理
  - 月末の `LLM dependencies removal`
- 決定事項:
  - trade throughput は保ちながら、negative exit と reentry は strategy-local 条件で細かく制御する。
  - entry/exit の整合は `entry_thesis` を中心に寄せる。
  - LLM は主系から外し、local-only decision を本線にする。
- 議論メモ:
  - `2026-01-09` 前後で MR overlay exits、M1Scalper adverse exits、macro trend exits を追加。
  - `2026-01-13` `bb7129d1` で technique-based entry/exit gating、`e5f557a8` で pullback touch counter を追加。
  - `2026-01-18` `b907ac13` / `a68ee565` / `19e0fc4c` / `2045847d` / `345da706` で reentry open-stack limits、distance override、MTF gate、loss negative exit、tech-loss exits をまとめて強化。
  - `2026-01-22` は margin caps、entry 改善、TP cap、negative-close broadening、exit context snapshot、M1Scalper autotune まで一気に整備。
  - `2026-01-23` は strict negative exit guards と LLM ops policy diff generation が同居し、方針がまだ揺れている。
  - `2026-01-24`〜`01-29` は M1Scalper、Mirror 系、VWAP 系の entry/exit tighten が続く。
  - `2026-01-27` `67c61423` で entry gating と worker/exit workflow を再編。
  - `2026-01-31` `3745d0c8` で `Remove LLM dependencies and keep local-only decisions` を明示。
  - `2026-01-31` `e34a354a` で AGENTS から legacy LLM 参照を除去。
- 持ち越し:
  - local-only 判定へ寄せた一方、forecast や pattern、feedback の補助層をどう足すかが次月の中心課題になる。

## 8. 2026-02-01〜2026-02-12
### Precision/Predictive/Ping5s 拡張と replay 基盤

- 期間 commit 数: `前半だけでも非常に多く、月全体 883 件の主因`
- 主な議題:
  - replay workflow と precision scalp modes
  - hard SL / pro stop / margin relief
  - forecast bundle と predictive gating
  - scalp_ping_5s 常設化と force-exit 制御
  - pattern book / pattern gate / kata docs
- 決定事項:
  - precision scalp と ping5s を live throughput の主戦場にする。
  - replay と forecast を live 改善の前提データとして整備する。
  - pattern gate は共通強制ではなく strategy opt-in で運用する。
- 議論メモ:
  - `2026-02-03` `545a018c` で standard replay workflow と precision drought/lowvol modes を追加。
  - `2026-02-04` `84c41bc8` で per-pocket hard SL と loss-cut を追加。
  - `2026-02-05`〜`02-06` は `LiquiditySweep`、`CompressionRevert`、`TrendRetest` 追加、`pro_stop` と margin relief exit worker 導入、LLM brain gate 試行が集中。
  - `2026-02-10` `f56434da` で sklearn multi-horizon forecast bundle を導入。
  - `2026-02-11` は forecast gate、FastScalp pattern 学習、aggressive env、strategy-aware predictive gating、`scalp_ping_5s` worker 導入が同日に密集。
  - `2026-02-12` は `scalp_ping_5s` の fixed TP/no-negative/MTF bias/entry caps/tick density fallback が連続し、事実上の live tuning day になる。
  - `2026-02-12` `cbe39199` で strategy opt-in pattern gate、`7e111d7a` / `847d6463` / `43ac2f86` で kata docs を整備。
- 持ち越し:
  - ping5s の throughput と risk 制御が trade-off 化し、翌週以降も連続調整対象になる。
  - V2 split 前のため、service 境界と entry intent 契約はまだ固まり切っていない。

## 9. 2026-02-13〜2026-02-16
### V2 split、entry intent 契約、order/position manager サービス化

- 主な議題:
  - worker split docs の確定
  - order_manager / position_manager の service 化
  - strategy-local `entry_probability` / `entry_units_intent` 契約
  - pre-coordination wrapper と blackboard coordination
  - strategy feedback service の導入
- 決定事項:
  - V2 では strategy worker が `entry_thesis` 契約値を必ず持つ。
  - `order_manager` は preserve-intent を軸にし、方向意図の再採点をしない。
  - blackboard 協調は `strategy_entry` 側で行い、service 層は risk/final reject のみを担う。
- 議論メモ:
  - `2026-02-14` `a5ef2997` で strategy control を order preflight へ統合。
  - `2026-02-14` `f969e75d` で V2 用の order/position manager workers を追加。
  - `2026-02-14` `8e8e9bd0` で strategy-local entry intent fields を必須化。
  - `2026-02-14` `598ce959` で `session_open` 経路にも entry intent を forward。
  - `2026-02-14` `87bcbf39` で strategy worker ごとの service env files へ分割。
  - `2026-02-15` `0720afce` で pre-coordination wrapper を追加。
  - `2026-02-15` `a9e70772` / `d0156d44` / `aefe3ef1` で blackboard coordination を restore しつつ strategy intent preserve を徹底。
  - `2026-02-15` `7d380073` と `fb4281c5` で strategy feedback worker を導入し、feedback を `entry_thesis` へ流し込む。
  - `2026-02-16` は position-manager の sqlite/closed-db 回復や scalp 5s observability の補修が続く。
- 持ち越し:
  - V2 の骨格は固まったが、live での throughput 低下と ping5s の調整負荷は残る。

## 10. 2026-02-17〜2026-02-29
### Live throughput 調整、forecast guard、ping5s 集中チューニング

- 主な議題:
  - forecast の session-bias / EV guard
  - ping5s B/C/D の entry gate、min units rescue、perf guard の調整
  - negative exit protection と de-risk sentinel 正規化
  - sqlite lock / summary path / strategy guard stalls の耐障害化
- 決定事項:
  - loser lane を広域停止するより、guard と sizing の局所調整で throughput を回復させる。
  - forecast は戦略ごと・時間軸ごとに weight map を持たせる。
  - ping5s は wrapper ごとに perf bypass や min-unit rescue を分けて扱う。
- 議論メモ:
  - `2026-02-17` は session-bias tuning が中心。
  - `2026-02-24` は oversized stop-loss、reject floor、entry throughput 回復に集中。
  - `2026-02-25` は strategy-level EV guards、global hard-block checks、timeout tuning。
  - `2026-02-26` は現存履歴でも最も密度が高い日の一つで、`101` commit。`honor no-side-filter opt-in`、`min-units rescue`、`sqlite lock stalls` 対応、`Reduce units_below_min drops` が並ぶ。
  - `2026-02-27` も `80` commit と高密度で、ping C の setup perf guard、lot floor、loss hard block 調整が続く。
- 持ち越し:
  - 月末時点でも live 調整は継続中で、次月に local-v2 専用運用と PDCA 規律化へ接続する。

## 11. 2026-03-01〜2026-03-14
### Local-v2 専用運用、setup-scoped 改善、anti-loop 規律

- 期間 commit 数: `417`
- 主な議題:
  - local-only / no-VM ルールの固定
  - launchd autorecover と local feedback/watchdog
  - Ollama safe canary と Brain autotune
  - setup-scoped participation / feedback / dynamic alloc
  - `TRADE_FINDINGS` を中心にした preflight / lint / index / anti-loop 規律
- 決定事項:
  - `2026-03-04` 以降の現行運用は local-v2 専用とし、VM/GCP/Cloud Run を実務から外す。
  - 改善は strategy-wide blanket trim ではなく、setup / lane / flow regime 単位で行う。
  - 変更前 review と change diary を必須化し、同じ lane の pending tweak をループさせない。
- 議論メモ:
  - `2026-03-01` `a4318171` で `v2 runtime route` を強化。
  - `2026-03-02` `21b94507` で market check before execution を docs 化。
  - `2026-03-03` `35f483b5` で Ollama brain backend、`c5393280` で separated local-llm lane repo bridge を追加。
  - `2026-03-04` `129f3a03` で local-only ops violation を記録し no-VM rule を強制、`ba4e9f34` で cloud runbooks を archive。
  - `2026-03-05` は local_v2 autorecover、agent whiteboard、brain autotune、pattern gate revive、trade_min/trade_all の再整理が集中。
  - `2026-03-07` `b50c8124` / `a9a78558` で local read-only observers を追加。
  - `2026-03-09` は `61` commit の高密度日で、winner concentration、safe brain canary 維持、micro loser lane 切り分け、rangefader/momentumburst 回復が中心。
  - `2026-03-10` は reverse-entry loser lane の strategy-local quality guard、adaptive entry improvement loop、forecast service の local-v2 接続が進む。
  - `2026-03-11` `3853c37c` で `TRADE_FINDINGS` change diary を formalize。
  - `2026-03-11` 以降は `strategy_feedback`、`participation_alloc`、`dynamic_alloc` を strategy-wide ではなく live setup に scope する方向へ連続修正。
  - `2026-03-12` `4380e890` で `docs/CURRENT_MECHANISMS.md` を追加し、「今ある仕組み」の棚卸しを定例化。
  - `2026-03-13` `9b522bca` / `542a9f32` / `10c40646` で change preflight、preflight guard、lint/index を導入。
  - `2026-03-14` `fb89cbac` / `e719f7ad` で anti-loop 改善規律を docs と review/lint へ反映。
- 持ち越し:
  - 現行の運用論点は「winner setup の participation を落とさず、loser lane だけを setup-local に削る」ことへほぼ収束。
  - 改善速度は上がったが、記録と review の規律を守らないと same-lane loop に戻る危険がある。

## 12. 履歴全体から読み取れる主要転換点

- `Cloud/news/GPT` 主導の時代から、`local_decider + worker-only + local-v2` 主導へ完全に重心が移った。
- 共通ロジックで戦略全体を押さえる方向から、strategy-local / setup-local に loser lane を切る方向へ変化した。
- `entry_thesis` は単なるメタ情報から、entry intent 契約、feedback、forecast、coordination の接続面へ昇格した。
- exit は共通 manager ではなく dedicated exit worker へ分解され、利益保護や negative close も戦略ごとに扱う設計になった。
- 運用 docs は単なる補助資料から、preflight/hook/lint/index まで含む強制規律へ変わった。

## 13. 今後の運用メモ

- この文書は「過去を全部遡った初回版」として置き、以後は月次か major deploy 単位で追記するのが現実的。
- 実運用の改善履歴は引き続き `docs/TRADE_FINDINGS.md` に集約し、この文書は「長期の流れ」と「大きな設計転換」の記録に使う。
- 週次の中間粒度は `docs/REPO_HISTORY_WEEKLY_ANNEX.md` を参照。
- 日付単位の raw ledger は `docs/REPO_HISTORY_DAILY_ANNEX.md` を参照。
- 将来的にさらに細かい版が必要なら、次段階として `日次 annex` を `git log --date=short` から自動生成する余地がある。
