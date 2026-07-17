# 弱点台帳 + 構造利益化の設計図 (2026-07-18, Codex実行用)

`design_profit_path_4x_20260718.md` (レーンA-C) と `design_profit_structure_repair_20260718.md`
(欠陥1-5, S1-S6) を包含する完全版の弱点台帳。番号 W1-W20。
各弱点に修理方向と担当を付す。不変条件 (fail-closed / shadow-only / AI_ORDER_AUTHORITY=NONE /
live昇格はオペレーター明示承認) は全修理に適用。

## A. シグナル/アルファの弱点

- **W1 出口が非対称でない** (=旧欠陥3): 入口は構造判定、出口は固定倍率TP/SLのみ。勝ち玉の右尾を切断。
  → structure-break exit arm family V3 (freeze時にclosed candleから**事前確定**したswing levelを封印、
  時間内はそのlevel割れ/戻り超えまで保有。TP無し・time-stop併用)。既存182候補と identity 分離。
- **W2 指標セット固定・チューニング語彙が粗い** (=旧欠陥W2): GO/CAUTION/STOP はペア単位のみ。
  regime→family選択圧をかけられない。→ `QR_AI_REGIME_SUPERVISION_V2`: ペア×候補family単位の
  GO/CAUTION/STOP + regime宣言 (TREND/RANGE/SQUEEZE/EVENT) を追加。ボットは宣言regimeに
  不適合のfamilyを自動CAUTION化。
- **W3 エントリーが足確定境界のみ**: 決定は4h/closed-M1境界。オペレーターはブレイク/リテストの
  現在進行イベントで入っていた。→ M5内のtick非依存な決定論トリガー (直近closed swing level への
  touch-and-reject を次のS5 openで判定) を fast bot admission に追加。lookaheadなし。
- **W4 イベント/ニュース盲目**: news pipeline は trader consumer-only。ボットは経済指標カレンダーを
  見ない。→ pre-entry event gate: 高インパクト指標の前後N分は新規admission停止 (これは既存
  calendar データで実装可)。イベント後モメンタムはW2のEVENT regime familyとして別途候補化。
- **W5 セッション構造の不在**: Tokyo/London/NY の挙動差が階層に無い (SITUATION_WEIGHTS の思想が
  botに未移植)。→ session を candidate family の第一級次元にする (同一ロジック×session別 GO)。
- **W6 レンジ回転レーンの不在**: 歴史的に利益源だった range/SQUEEZE 回転 (LIMIT受動往復) が
  bot撲滅時に消えたまま。FX時間の過半はレンジ。→ レーンF: range-rail 回転 family
  (レール事前確定・受動LIMIT・両側、S5 exact評価)。`range_rail_geometry_repair` の既存成果を流用。
- **W7 オペレーター実績の未信号化** (=旧欠陥4): レーンE (precedent replication) で対応。

## B. 執行の弱点

- **W8 受動LIMIT+TTL90秒の逆選択**: fillするのは価格が戻って来た時だけ=不利な状況で選択的に
  約定し、トレンドでは不成立。→ momentum文脈 (W2のTREND宣言時) は STOP entry arm を主用、
  受動LIMITはRANGE宣言時に限定。4分位受動armのpaired比較で逆選択コストを定量化して封印。
- **W9 部分利確/分割が無い**: 全量in/全量out。スキャルプ回転の実績と乖離。→ V3 family に
  half-at-1R + runner (structure-break) の2段出口armを追加。
- **W10 スプレッド時間帯を回避しない**: コストゲートは個別拒否のみで、スケジューラは
  rollover (21-22UTC) や金曜後半に平然と決定を出し続ける。→ 低コスト時間帯マスクを
  admission に追加 (実測spread分布から事前宣言)。
- **W11 スリッページ/約定拒否の実測データ皆無** (=旧欠陥5と連動): shadow-onlyの限り永遠に
  取れない。→ T1 micro-live (下記W16) が唯一の解。

## C. ポートフォリオ/リスクの弱点

- **W12 相関盲目**: 28ペアを独立扱い。USDテーマで20ペア同時動作時の実効リスクが
  per-pair capの数倍になる。逆に「テーマ一致3ペア同時攻め」(実績あり) も表現不能。
  → 通貨エクスポージャ集計 (8通貨netポジション) を admission の一級制約に。
  同一テーマの意図的バスケットは conviction ladder の客観条件として定義。
- **W13 pips単位の選抜歪み**: scorecard集計がpips建てでペア間のpip価値差を無視 (NAV%原則違反)。
  → 全scorecardに quote-currency換算のNAV%列を並記し、選抜キーをNAV%側へ移行。
- **W14 equity-curve制御が無い**: 日次損失停止・連敗縮小・好調時の逓増が未実装。
  → conviction ladder (S4) に equity-curve条項を統合: 日次 -3% NAV 停止、
  3連敗でサイズ半減、週次+5%超で次週ベース+25% (全て事前宣言)。

## D. 学習ループの弱点

- **W15 適応タイムスケールが日〜週**: 6hチューニング+新cohort要件で、レジーム変化への追従が
  構造的に遅い。→ W2のfamily粒度GO/STOPは**既存証拠の再利用でなく配分変更**なので
  即時適用可と契約に明記 (新エッジ主張には使えない)。
- **W16 中間資本ティアの欠落** (=旧欠陥5): T0 shadow → T1 micro-live (NAV2% risk pool、
  執行データ収集目的、オペレーター承認必須) → T2 scaled。昇格契約モジュールを実装する。
- **W17 監督自身が採点されない**: GO/CAUTION/STOPの的中率を誰も測っていない。
  → supervision outcome ledger: 各監督行を24h後の実現ボラ/方向と突合し、
  supervisor精度をmonthly封印。精度が閾値未満のfamilyは監督を自動CAUTION化。
- **W18 棄却候補の死因分類が無い**: permanent shadowは結果を貯めるが「なぜ死んだか」の
  taxonomy が無く、次のfamily設計に還流しない。→ 棄却時に死因コード
  (COST/DIRECTION/EXIT_TIMING/REGIME_MISMATCH/...) を必須化し、四半期でtaxonomy集計。

## E. 運用/メタの弱点

- **W19 契約肥大とゲート負債**: incidentごとにゲートが増え、退役プロセスが無い。714行の契約は
  それ自体がスループットリスク (=欠陥2の根本原因)。→ 四半期 gate-debt review:
  全ゲートに (防いだ損失の実績 / 殺したエントリー数) を記録し、比率最低のゲートを
  統合・退役候補としてオペレーターに提示。
- **W20 直列の将来テストで証明速度が遅い**: 2週間に1本のprospective testでは PROVEN 行が
  増えない。→ 並列事前登録: 複数候補それぞれに独立の将来窓lockを許可 (family横断のWRC分母は
  登録簿で管理)。登録簿 `research/data/prospective_registry_v1.json` を新設。

## 実装状況 (2026-07-18 Claude、テスト付き・全suite green)

| 弱点 | モジュール | 状態 |
|---|---|---|
| W16 | `micro_live_promotion_contract.py` | 実装済み (T1契約+承認束縛検証) |
| S1/欠陥1 | `close_distance_gate.py` | 実装済み (pre-entry close-distance + max_admissible_hold) |
| W12 | `currency_exposure_guard.py` | 実装済み (8通貨net NAV%キャップ、ヘッジ相殺対応) |
| W2/W8/W15 | `regime_supervision_v2.py` | 実装済み (regime宣言×family行、不整合GO自動降格、6h TTL→UNSUPERVISED) |
| S2/欠陥2 | `gate_throughput_slo.py` | 実装済み (ファネル封印、floor割れでkiller gate命名+P0フラグ) |
| W14/S4 | `conviction_ladder.py` | 実装済み (0.25%基底、条件数で2x/4x、連敗半減、日次-3%停止) |
| W20 | `prospective_registry.py` | 実装済み (hash鎖・事前登録・family分母・成熟前評価拒否) |
| W17 | `supervision_outcome_scorer.py` | 実装済み (GO/CAUTION/STOP採点、精度floor割れfamilyの自動CAUTION指名) |
| W18 | `rejection_taxonomy.py` | 実装済み (固定死因コード必須化+family別集計) |
| W1/W9 | `asymmetric_exit_shadow.py` | 実装済み (TP無しstructure-break exitの独立shadow評価器。封印カタログ非接触のpaired再採点) |
| W13 | `nav_normalization.py` | 実装済み (pips→NAV%変換、明示レート必須) |
| W10 | `cost_window_mask.py` | 実装済み (事前宣言UTC高コスト窓マスク、デフォルトrollover 21-22) |

追加完了 (2026-07-18 第2弾):
- **未来テストexecutor実装済み**: `final-test` subcommand + `evaluate_locked_final_test`。
  8/3成熟前の実行・研究manifestの再利用・束縛改竄を全拒否。事前宣言判定
  (stressed net>0 && stressed PF>=1.05) 封印済み。残るのは壁時計のみ。
- **shadow層のruntime配線完了**: `build_fast_bot_shadow_admission_binding_v1` が
  close-distance / cost-window / 通貨エクスポージャの3ゲート判定を毎サイクル封印し、
  将来GO契約の必須提示物 (`future_go_contract_must_bind_this_artifact=true`) にbind。

コードでは完成させられない残余 (時間・証拠・境界に依存し、先回りは契約違反):
- レーンA未来テストの**実行** (8/3以降の新規取得後、コマンド1回)
- live trader runtime (本体repo・ルーチン所有) への同ゲート接続 = Codex S1-S6
- W1/W9のカタログ昇格 = shadow paired証拠が正になった時のみ (証拠先行の禁止は不変条件)

## 証明の階段 (narrative-first discipline、2026-07-18 オペレーター方針で明文化)

「過去ローソクをナラティブに分析して、リプレイ検証する」を全familyの必須手順にする。
ただし各段が証明できる範囲を混同しない:

1. **ナラティブ**: 過去ローソクから「誰がなぜ負けて、その反対側をどう取るか」の機構仮説を書く。
   機構なきパターンは登録不可。(例: 週末クローズ跨ぎの流動性欠落が exit を滑らせる →
   事前に跨がなければ損失族が消える)
2. **因果ルール化**: ナラティブを entry 前情報だけで計算できる規則に変換する。
3. **TRAINリプレイ**: 規則がナラティブの予測どおり動くかを exact bid/ask + receipt で検証。
   ここで「効く形」を1つに固定する (grid事前宣言・選抜はTRAINのみ)。
4. **外部窓リプレイ (VALIDATION)**: ナラティブ形成に使っていない窓で無変更複製。
   予測が再現しなければナラティブ棄却 (死因コード必須)。
5. **未来窓**: 戦略凍結時に存在しなかったデータでの1回評価。**ここで初めて「証明」と呼ぶ**。
   1-4は「反証されなかった」まで。今セッションの偽+689.4pは、精密リプレイが
   見えない仮定1つ (execution-gap事後削除) で崩れる実例。

実績: この階段を通った例が既に2つ — 週末跨ぎナラティブ (TRAIN反転→VAL +969.7p複製、8/3判定待ち)、
出血日ナラティブ「負け込んだ日に入り続ける取引は負の期待値」(50p日中ストップ、VAL最悪日-110p改善+net+29%)。

## 不参加は改善ではない (2026-07-18 オペレーター原則)

「トレードしなければ損失が出ない」型の改善は前進しない理論として**禁止**する。
スロットル (SKIP/HALF_SIZE/day-skip) は損害上限の道具であって、エッジの改善に数えない。
負けた日は入り口と出口が悪かった日であり、義務は診断と修理:

1. 負の日ごとに entry/exit fingerprint を封印する (`run-losing-day-entry-exit-diagnosis.py`)。
2. fingerprint から修理仮説 (ナラティブ) を作り、証明の階段に載せる。
3. スロットル系の採否判定には機会損失の実測 (抑制した取引の合計pips) を必須とし、
   正なら格下げ。day-skip 1p は VALIDATION で +1,658.5p を逃して棄却済み (実例)。

既知の出口証拠 (封印済みTRAIN): hold 6h化 net -185.9 / cadence 1h化 -712.4 / rank1化 +403.8。
=> この時間軸の負け日は「出口が遅い」問題ではない。エントリー品質 (シグナル強度・時刻・regime) の
診断が主戦場。

## W21 全天候要件 (2026-07-18 オペレーター指示で追加)

「レンジでもトレンドでも高低ボラ両方でとってこれないといけない」= regime×vol の 2×2 全セルに
検収済みfamilyを持つこと。実測の現在地:

| セル | 担当family | 状態 |
|---|---|---|
| トレンド×高vol | S5 survivor (RETURN_PIPS DIRECT 8h/12h) | VALIDATION複製済み・未証明 (8/3判定) |
| トレンド×低vol | 同上でカバー可能か要day分解 | 未測定 (day-regime分解が次の実験) |
| レンジ×(高/低)vol | **既存192では構造的に不可能**: INVERSE 96本全て TRAIN負 (最良-891.5p)。
  4-24h軸の向き反転はレンジ捕捉にならない | レーンF (短周期レール回転・受動LIMIT両側・
  range_rail_geometry_repair流用) を新規設計。M5取得後にTRAIN/VAL/WRC |
| イベント×高vol | W2のEVENT regime + W4 event gate (回避が先、捕捉はその後) | ゲート実装済み・捕捉family未設計 |

規律: セルごとに独立family・独立検収。1つのfamilyで全セルを取ろうとしない
(それが「何をやっても利益が出ない」単一エンジン設計の再演)。regime判定は
`regime_supervision_v2` の宣言regime + 因果的day分解で行い、セル間の資本配分は
family粒度GO/CAUTIONで切り替える。

## 実行順 (Codex)

P0: W16 (T1契約モジュール=Claude実装済みを検収) → W8+W2 (regime×order-type整合) → W4 (event gate) → W12 (通貨エクスポージャ制約)
P1: W1+W9 (V3出口family) → W6 (range-railレーンF) → W13 (NAV%化) → W20 (並列登録簿) → W3 (touch-reject admission)
P2: W5 (session次元) → W14 (equity条項) → W17 (監督採点) → W18 (死因taxonomy) → W19 (gate-debt review) → W10 (時間帯マスク)

検収の共通原則: 各修理は「殺すエントリー数」と「守る損失」の見積りを添付し、
gate throughput SLO (S2) の floor を割る修理は設計に差し戻す。
