# 設計書: 月次4x資産成長への利益経路 (2026-07-18, Codex実行用)

作成: Claude (branch `codex/episode-s5-outcome`, base commit `8fde5bf64`)。
本書は **目標を証明済みとは主張しない**。目標と現在の証拠の差分を数理で固定し、
差分を埋める作業を fail-closed の検収基準つきで Codex に渡すための設計書である。
`AI_ORDER_AUTHORITY=NONE` / shadow-only / 本番昇格禁止は全作業で不変。

---

## 0. 目標の数理分解 (固定)

- 月4倍 = 30暦日で `4^(1/30)` = **+4.73%/日** の複利。22取引日なら +6.50%/日。
- 証拠の通貨単位は pips/日 (ポートフォリオ合算) × サイジング(レバレッジ) で NAV%/日 に変換する。
  USDクォート近似で pip value = units × 0.0001。国内FXのレバレッジ上限 25x を境界条件とする。

### 現在の最良証拠 (未確定・額面) との差分

| 項目 | 値 |
|---|---|
| survivor `XS-55e9245625c2766af41e` locked VALIDATION | 969.7p / 10 active days = **97.0 p/日**, 19.2 trades/日, 5.05 p/trade |
| 25x満載・12同時ポジション均等割での日次リターン | **+2.02%/日 → 30日で 1.82x** |
| 4x に必要な日次 | +4.73%/日 → **227.0 p/日** (25x・12同時、board計算値。不足 130.0 p/日) または 58x レバレッジ (不可) |
| 統計的地位 | Holm補正後 p=1.0 (有意性なし)、VALIDATION は非独立 (M1近似の事前閲覧)、真の判定は Jul 20–Aug 3 未来テスト |

**結論 (正直な現在地)**: 額面ですら単一戦略では 4x に **2.6倍の edge密度が不足**。かつその額面自体が未証明。
したがって経路は (a) 未来テストによる既存 survivor の確定、(b) 独立レーンの追加による pips/日 の加算、
(c) 高頻度レーン (fast bot) の昇格、の3本を並列で進めるしかない。ドローダウン下の複利は
`positive_day_rate=0.4` の現実の下で破綻し得るため、サイジング設計 (§4) を利益設計と同格で扱う。

---

## 1. 穴 (既知の欠陥・リスク) — 監査で実証済みのもの

### 1.1 データ/検証系
- **[H-M5-1] M5 validator は年内内部欠落を50%まで素通し** (実証済み)。取得ゲートとしては
  前年比/ペア間 row_count 相対クロスチェックの追加が必要。
- **[H-M5-2] OANDA `to` 境界キャンドルのクランプ未実測**。全シャード全滅リスク。
  本取得前に 1ペア×1年のパイロット実測が必須。
- **[H-M5-3] fail-closed 経路の大半がテスト未演習** (raise 40+ に対しテスト9本だった。
  2026-07-18 に一部追加済み、残りは §5 P0-2)。
- **[H-SCRIPT] `scripts/oanda_history_fetch.py` の sha が S5/M5 両方の receipt 検証で
  「現在のファイル」と比較される**。fetch script を1行でも変更すると既存 exact-28 S5 root の
  検証が恒久的に壊れ、TRAIN anchor 再現が失敗する。**fetch script は凍結扱い**。
  `to` 境界対応が必要になった場合は validator 側 quarantine で吸収するか、
  受領済み sha の内容アドレス許可リスト (era registry) を新設計する。変更前に必ず本項を読むこと。
- **[H-DATA-REGIME] TRAIN 24 active days は単一レジーム**。どの結論も 2026-05/06 の
  一般化しか主張できない。長期 M5 (2020–2026) が唯一の広域証拠。

### 1.2 統計系
- **[H-STAT-1] 有意性ゼロ**: survivor は経済的 robustness 選抜のみ。192候補 Holm で p=1.0。
  日次正規近似は重複ホールド (12h hold / 4h cadence) の自己相関を無視しており甘い。
  長期系では week block bootstrap + White Reality Check を必須とする (§3)。
- **[H-STAT-2] VALIDATION 非独立**: M1近似の事前閲覧により `independent_validation_claim_allowed=false`。
  独立な確定は新規取得の未来データのみ。
- **[H-STAT-3] LONG側が弱い**: TRAIN stressed LONG +11.3p vs SHORT +845.7p。
  方向バイアスの恒久ルール化は禁止 (feedback_no_direction_bias_rules) だが、
  レジーム条件付きの非対称性としては長期データで検証する価値がある。

### 1.3 実行系
- **[H-EXEC-1] スリッページ/約定拒否は bid/ask open 実測以外モデル化していない**。
  25x満載サイジングでは 1 pip の執行劣化が日次 -0.25% に相当し複利を直撃する。
- **[H-EXEC-2] 同時12ポジション×28ペアの証拠金/相関制御は shadow orchestrator に未接続**
  (fast_bot_shadow_orchestrator の before-open risk capacity は H08 no-trade のまま)。
- **[H-EXEC-3] 週末ゲートで金曜後半の決定が落ちる** → 実効取引日は週約4.5日。
  月次換算は取引日ベース (+6.5%/日必要) で見るのが正しい。

---

## 2. 改善経路の全体像 (3レーン並列)

```
レーンA: 既存 survivor の確定      → Jul 20–Aug 3 未来テスト (§3.A)
レーンB: 広域 M5 長期研究で加算    → 8通貨 strength 12候補 (§3.B)
レーンC: 高頻度 fast bot 昇格      → 受動LIMIT スプレッド捕捉 (§3.C)
横串:   サイジング/複利シミュレータ → feasibility board (§4)
```

pips/日 の目標配分 (額面ベースの設計値であり約束ではない):
レーンA 97p + レーンB 目標80–120p + レーンC 目標50–100p ≈ 230–320p/日 → 25x で +4.6〜6.4%/日。
各レーンが独立に fail-closed 検収を通らない限り加算しない。

---

## 3. Codex への作業指示 (優先順)

### P0-1: 未来テスト executor (レーンA) — 最優先・期日あり
- `[2026-07-20T00:00Z, 2026-08-03T00:00Z)` の exact S5 を **2026-08-03 以降に新規取得**し、
  新 manifest を封印 → `evaluate_locked_spec` を **無変更のロック**
  (`research/data/adaptive_exact_s5_train_lock_v1.json`, spec `XS-55e9245625c2766af41e`) で1回だけ評価。
- 実装: `scripts/run-adaptive-exact-s5-profit-research.py` に `final-test` サブコマンドを追加。
  要件: (a) prospective lock (`adaptive_exact_s5_prospective_final_lock_v1.json`) の digest 検証と
  window 一致強制、(b) 2026-08-03T00:00Z 以前の実行拒否 (取得時刻 receipt で証明)、
  (c) research/lock/validation と同じ digest chain、(d) 出力は clean 必須・stale 拒否、
  (e) 評価は1回。失敗時の再実行は取得 receipt が同一の場合のみ。
- **検収**: 上記 (a)–(e) の各 fail-closed 経路に単体テスト。8/3 までは実データ実行禁止。
- 判定基準 (事前宣言): stressed net > 0 かつ PF(stressed) ≥ 1.05 で「初の独立プラス証拠」。
  それ未満は survivor 棄却し、レーンBへ全振り。**どちらでも 4x の証明にはならない** —
  これは「プラスを未来データで再現する」ゲートである。

### P0-2: M5 validator 残修理 (レーンB前提)
1. 内部欠落クロスチェック: manifest に per-pair/per-year `usable_rows` の
   前年同ペア比・同年ペア間中央値比を記録し、比率 < 0.70 を `COVERAGE_ANOMALY` で fail-closed。
2. fail-closed テスト演習の完遂 (未演習リスト: 重複/非単調タイムスタンプ、年境界シャード外行、
   gzip切断、空ファイル、カバレッジ下限、境界ギャップ>7日、receipt鎖破断、orphan receipt、
   fetch script 内容ドリフト、シンボリックリンク、summary 不一致、期間バリデーション)。
3. `to` 境界パイロット: **fetch script を変更せず** EUR_USD × 2020年のみ取得し、
   シャード終端ちょうどに開く行が返るかを実測。返る場合は validator 側で
   「シャード終端ちょうどの行を quarantine (行数 receipt に記録)」する設計変更を先に実装。
- **検収**: パイロット root が validator green になるまで 28ペア本取得は開始しない。

### P0-3: 8通貨 strength 長期研究 (レーンB本体)
- handoff 済みの設計: 28ペアを独立投票せず **8通貨 strength node へ least-squares 分解**。
  lookback {8h, 24h, 5d} × hold {4h, 8h} × non-overlapping strongest-vs-weakest matching = **12固定候補のみ**。
- TRAIN/VALIDATION/TEST を年単位で事前宣言 (例: TRAIN 2020–2023, VAL 2024–2025, TEST 2026)。
  有意性は week block bootstrap + day/basket/currency cluster + **White Reality Check** (12候補全体)。
- 週末ゲートは S5 survivor と同じ pre-entry 因果判定を再利用 (`compute_market_status`)。
- **検収**: 候補は12から増やさない (増やすなら新 family として再宣言し WRC の分母に含める)。
  survivor 0 本なら 0 本と報告する。

### P1-1: fast bot 昇格前提整備 (レーンC)
- 契約の昇格ゲート (≥100 fills / ≥10 filled days / PF≥1.25 / filled-day mean の95%下限>0) は既設。
  必要なのは shadow 蓄積の運転継続と、`fast_bot_shadow_orchestrator` の
  causal weekend admission + 通貨/basket 相関上限 + 証拠金容量を将来 GO 契約に bind する実装
  (handoff の未完了項目)。GO=0 のまま shape を完成させる。
- **検収**: H08 no-trade のまま 5,096 cells の admission/capacity 判定が単体テストで固定される。

### P1-2: サイジング/複利設計 (§4 の board を運転に接続)
- per-trade risk を NAV% で宣言し (feedback_use_nav_percent)、日次ドローダウンで
  サイズを幾何的に縮小する複利制御 (例: 日次 -3% で当日停止) を設計書として固定。
  25x満載固定は [H-EXEC-1] により禁止。Kelly比の 1/4 を上限とする。

### P2: 執行劣化の実測モデル (レーンC と共通)
- live ledger (`data/execution_ledger.db`) の実約定 vs 想定価格の差分分布を月次で封印し、
  feasibility board の stress 入力にする。

---

## 4. 4x feasibility board (本セッションで実装済み)

- `scripts/build-profit-path-feasibility-board.py` が lock/validation/prospective の
  digest を検証した上で、レバレッジ×レーン加算×ドローダウン仮定の格子で
  30日乗数を計算し `research/data/profit_path_feasibility_v1.json` に封印する。
- board は **証拠ステータス (PROVEN/UNPROVEN) を各行に強制**し、UNPROVEN 行の乗数を
  目標達成の根拠として引用することを contract 文字列で禁止する。
- Codex は各レーンの検収通過のたびに board を再生成し、`4x_gap_pips_per_day` の縮小を追跡する。

---

## 5. 不変条件 (全作業共通)

1. 事後情報での取引削除・選抜は一切禁止 (survivorship)。ambiguous は悲観計上。
2. TRAIN で選抜 → 凍結 → 後段データは無変更評価1回。分割境界の再宣言は新 family 扱い。
3. 価格合成禁止。receipt 鎖と digest chain を全成果物に。stale 出力拒否。
4. `AI_ORDER_AUTHORITY=NONE`、shadow-only、live 昇格は別契約が明文化されるまで禁止。
5. 「絶対勝てる」「毎月4倍達成」を証明済みとして書かない。board の PROVEN 行のみ引用可。
6. `scripts/oanda_history_fetch.py` は凍結 ([H-SCRIPT])。
