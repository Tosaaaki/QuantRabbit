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

残り (コードとして今ここでは完成させられないもの):
- **runtime配線**: 上記12モジュールをfast bot admission / trader cycle / scorecard生成へ接続 (= Codex、実行順は下表)
- **データ待ち**: レーンA未来テスト (8/3以降に取得→評価)、レーンB/E/F研究 (M5取得後)、W17/W18の実データ運転
- **W1/W9のカタログ本体統合**: shadow評価器でのpaired証拠が正になった場合のみ、V3 familyとして
  identity繰り上げ+独立audit付きで昇格 (証拠より先にカタログへ入れない)

## 実行順 (Codex)

P0: W16 (T1契約モジュール=Claude実装済みを検収) → W8+W2 (regime×order-type整合) → W4 (event gate) → W12 (通貨エクスポージャ制約)
P1: W1+W9 (V3出口family) → W6 (range-railレーンF) → W13 (NAV%化) → W20 (並列登録簿) → W3 (touch-reject admission)
P2: W5 (session次元) → W14 (equity条項) → W17 (監督採点) → W18 (死因taxonomy) → W19 (gate-debt review) → W10 (時間帯マスク)

検収の共通原則: 各修理は「殺すエントリー数」と「守る損失」の見積りを添付し、
gate throughput SLO (S2) の floor を割る修理は設計に差し戻す。
