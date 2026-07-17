# 設計書: 利益を殺している構造の修理 (2026-07-18, Codex実行用)

前提の転換: オペレーター (ゆうき) は2025年に裁量で4xを実証済み。エッジの存在は既知。
問題は「エッジ探し」ではなく **システム構造がエッジを利益に変換できない** こと。
目標は月3xで可 (= +3.73%/日、25x・12同時で **179.2 p/日**。feasibility board `target 3.0` 行参照)。
本書は `design_profit_path_4x_20260718.md` の上位互換の診断であり、レーンA/B/C の作業指示は有効のまま。

---

## 1. 構造欠陥の診断 (システム自身の記録で裏付け)

### 欠陥1: クローズ跨ぎリーク族が TP エッジを支配 [証拠: 本体 `remaining_profitability_p0_decomposition.json` の `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`]
- S5研究でも同一族を独立確認済み: 週末exit遅延60件で **-1,438.8p** (TRAIN全損失の約2倍相当)。
  事前週末ゲート1つで TRAIN が -747.6p → +1,089.0p に反転した。
- **これは「エッジが無い」のではなく「保有時間の設計がクローズ構造を無視している」**。
- 修理: 全レーン (ライブ trader 含む) に pre-entry close-distance gate を義務化。
  `compute_market_status` の `minutes_to_next_close > hold + margin` を entry 条件に。
  ライブ側は thesis horizon (12h) とクローズまでの距離の min を hold 上限にする。

### 欠陥2: ゲート積層がスループットを殺す [証拠: 6/11 entry freeze catch-22 postmortem、guardian queue 48+8 設計]
- 個々に正当な fail-closed ゲートの合成が「事実上エントリー0」を作る。6/11に4層修理した前科。
- **利益 = 期待値 × 回数**。回数が0なら期待値の議論は無意味。
- 修理: **gate throughput SLO** を一級メトリクスにする。
  `data/gate_throughput_slo.json`: 直近7日の (シグナル発生数 → 各ゲート通過数 → 発注数) の
  ファネルを毎サイクル記録し、通過率が floor (例: 発生の10%) を割ったら
  「どのゲートが何件殺したか」を P0 で self-improvement queue に自動起票。
  ゲート追加 PR は throughput 予測影響の記載を必須にする (契約追記)。

### 欠陥3: 均一マイクロサイズ + 対称TP/SLグリッド ≠ オペレーターの勝ち方
- 2025年の4xは: 確信度Sで集中サイズ (10-15k)、SL-free/thesis-hold、テーマ一致で複数ペア同時、
  伸びる玉は構造が壊れるまで保有 (非対称)。
- 現システム: 全シグナル同サイズ、TP は entry 距離の固定倍率でキャップ、勝ち玉の上限が構造的に固定。
  → 分布の右尾を自分で切断している。`ai_test_bot` 229分母→18JPY/trade の罠 (5/6メモ) と同型。
- 修理 (レーンD: **非対称出口 family**): fast bot / S5 系の候補 family に
  「TP無し・structure-break exit (直近スイング割れ) ・time-stop のみ」の arm を追加し、
  同一凍結証拠で対称TPと paired 比較する。選抜は既存 fail-closed 手続きのまま。
- 修理 (サイジング): conviction ladder を**事前宣言**で導入。
  ベース 0.25% NAV risk/trade。事前定義した客観条件 (regime整合 + セッション + イベント無風 +
  レーン合意数) を満たす時のみ 2x/4x 段階増し。事後裁量での増しは禁止。日次 -3% NAV で当日停止。

### 欠陥4: オペレーターの実証済みアルファが信号化されていない
- 2025年の4x取引履歴は `data/manual_history_2025_mining.json` / precedent audit に採掘済みなのに、
  bot の候補 family は汎用モメンタムだけ。**実証済みエッジを検証対象にしていないのが最大の穴**。
- 修理 (レーンE: **operator precedent replication**):
  1. 2025年の勝ちトレードを状況タプル (session × regime × 直前構造イベント × 保有時間 × 方向) に符号化。
  2. 頻出上位の状況タプル ≤10個を固定 family として事前宣言。
  3. 長期 M5 (P0-2/P0-3 と同じ取得データ) で TRAIN/VAL/TEST 評価。WRC は10候補の分母で。
  - 検収: タプル定義は manual_history から機械的に導出し、価格結果を見て手直ししない
    (survivorship 禁止の対象)。導出スクリプトと入力の sha を family 宣言に封印。

### 欠陥5: 二値昇格 (shadow → full live) で中間資本が無い
- 現契約は shadow-only → (未実装の) 昇格契約の二値。証拠が揃うまで資本ゼロ、揃ったら一気。
  これは学習速度も遅い (実約定の執行データが貯まらない)。
- 修理: **3層資本ティア**を昇格契約として明文化:
  T0 shadow (現状) → T1 micro-live (専用予算: NAV の 2% を risk pool、最小 units、
  実行データ収集が主目的) → T2 scaled (feasibility board の PROVEN 行のみ)。
  T1 昇格基準: T0 で ≥100 fills / PF(stressed) ≥ 1.05 / 週次 LB > -0.5σ (T2 より緩い) 。
  T1→T2 は現契約の promotion gate (PF≥1.25 等) を維持。
  T1 は `LiveOrderGateway` / `RiskEngine` / receipt 検証を完全経由 (新しい壁は作らない)。

---

## 2. Codex 実行順 (構造修理分)

| # | 作業 | 依存 | 検収 |
|---|---|---|---|
| S1 | pre-entry close-distance gate を trader runtime + fast bot admission に実装 | なし | close跨ぎ entry が unit test で0件。既存週末ゲートと二重化しない |
| S2 | gate throughput SLO artifact + self-improvement 自動起票 | なし | 7日ファネルが毎サイクル更新、floor割れで P0 行が立つ |
| S3 | 非対称出口 arm (structure-break/time-stop) を grid family に追加 | なし | 対称TPと paired・同一凍結証拠・identity 分離 (policy V3) |
| S4 | conviction ladder + 日次 stop を sizing 契約として文書化・実装 | S2 | 事後増し不可がコードで強制される |
| S5 | operator precedent family 導出スクリプト (タプル化・封印) | M5取得(P0-2) | 価格結果非参照で導出、sha封印 |
| S6 | T1 micro-live 昇格契約の草案 (実装はオペレーター承認後) | S1-S4 | 契約文書のみ。live 権限は本書では一切付与しない |

不変条件は 4x 設計書 §5 と同一。**live 昇格の実行はオペレーター明示承認が必要** (S6 は草案まで)。

---

## 3. 3x 目標の数理 (feasibility board 対応済み)

- 3x/30日 = +3.73%/日 = **179.2 p/日** (25x・12同時)。現 survivor 額面 97 p/日 → 不足 82.2 p/日。
- 欠陥1修理 (クローズリーク除去) は既に survivor に織込み済み。残り 82 p/日は
  レーンB (目標80-120p) か レーンD/E のどちらか1本が通れば額面上は届く。
  ただし全て PROVEN になるまで目標達成は主張しない。
