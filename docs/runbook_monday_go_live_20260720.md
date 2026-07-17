# Go-Live Runbook: 2026-07-20 (月) オープンで利益機会を最大化する手順

オペレーター (ゆうき) の明示指示 (2026-07-18):「すぐ利益出すところまでやって。別ブランチも何もかも許可する」。
市場は土日クローズのため、物理的な最速収益タイミングは **7/20 (月) 6:00 JST 頃のシドニーオープン**。
本runbookはそこまでに完了させる作業を、実行者と順序つきで固定する。

## 事実の確認 (先に読むこと)

- 唯一の正の証拠 (survivor: VALIDATION +969.7p / PF 1.42) は**未証明** (非独立・有意性なし)。
  7/20-8/3 がその判定窓であり、**7/20 は「証明開始日」であって「証明済み開始日」ではない**。
- 現ライブ系の記録は `NEGATIVE_EXPECTANCY_ACTIVE` + `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`。
  **リーク修理なしで再開すれば月曜は損失側に賭けるのと同じ**。ゲート配線が先。
- 月曜から発生しうる利益の現実的レンジ: 25x上限・修理済みリークで survivor 額面が出た場合
  日次 +2%前後。負の日も40-60%の確率で来る (positive_day_rate 0.4-0.54)。日次 -3% で停止。

## Phase 0 (土日中・Codex または新セッションClaude): branch統合

1. `codex/episode-s5-outcome` (@ e1fdf3ff0) を dev へmerge。全成果はcommit済み:
   ガード14 module + final-test executor + shadow admission binding + 設計書/台帳/runbook。
   コンフリクトは docs/* runtime drift のみのはず。merge後 full pytest (5196 passed基準)。
2. 本体repoのdirty branch (`codex/qr-trader-run-watchdog`) は**触らずcommit完了をCodexに要求**。
   mainへのFFは両branchのtest green後。

## Phase 1 (土日中・Codex): ライブtrader cycleへのゲート配線 (S1/W10/W12/S4)

配線先は trader runtime の entry admission 経路 (`docs/SKILL_trader.md` のcycle内)。全て既存module呼び出しのみ:

1. `close_distance_gate.evaluate_close_distance_gate(decision, hold_minutes=thesis_horizon)`
   を entry 判定の先頭に。拒否時は `max_admissible_hold_minutes` で horizon短縮を先に試す。
2. `cost_window_mask.evaluate_cost_window(decision)` を同位置に (rollover 21-22 UTC遮断)。
3. `currency_exposure_guard.evaluate_currency_exposure(open_positions, candidate, currency_cap_fraction=0.5)`
   を発注前最終チェックに。テーマ集中はconviction ladder条件を満たす時のみ cap 0.75 へ引上げ可。
4. `conviction_ladder.allowed_risk_fraction(...)` でサイズ決定。日次 -3% NAV で当日新規停止。
5. 配線後 `gate_throughput_slo` のファネル記録をcycleに追加 (7日でfloor検証)。
- 検収: 各ゲートの単体テストはbranchに既存。配線のsmoke = dry-run cycle 1周 +
  金曜20:00 UTC想定のシミュレーションで entry 0 件になること。

## Phase 2 (日曜夜・オペレーター本人): 発注権限の決定

月曜に新規エントリーを出せる主体は現契約では存在しない (7/16 cutoverでAI発注停止、fast botはshadow-only)。
選択肢は2つ。**どちらを選ぶかはゆうきさん本人の1行指示が必要** (これはコード作業ではなく権限決定):

- **案A (推奨): trader routine の裁量エントリーを再開** — 既存の発注機構・Guardian・災害ストップを
  そのまま使う。Phase 1 のゲートが今回の修理点。cutover文書に「2026-07-20 オペレーター指示で
  entry authority復帰、ゲートv2適用」の1段落を追記するだけで機構は動く。
- **案B: T1 micro-live** — `micro_live_promotion_contract` は承認artifact検証まで実装済みだが、
  fast bot→LiveOrderGateway の発注コードが未実装のため**月曜には間に合わない**。今週の実装項目。

案A採用時のオペレーター作業 (1コマンド、日曜夜):
Codexが用意する `operator-approve-entry-resume` スクリプトに承認文を打鍵 (Claude/Codexによる代行禁止)。

## Phase 3 (月曜 6:00 JST〜): 運転

1. routine 再開後の最初の3 cycleは entry size をladder基底 (0.25% risk) に固定。
2. `gate_throughput_slo` で「ゲートがエントリーを殺しすぎていないか」を12:00 JSTに点検。
3. 日次 -3% NAV 到達で当日停止 (ladder が自動で size 0 を返す)。
4. 7/20-8/3 は survivor の判定窓と重なるため、**S5系レーンのデータは汚さない**:
   routine の裁量エントリーと research窓は口座が同じでも evaluation は別系 (取得data基準) なので干渉しない。
5. 8/3 に `final-test` を実行 (新規取得→1回評価)。verdictが正なら survivor をレーンとして
  追加検討、負なら死因コードを記録してレーンB/E/Fへ。

## 私 (Claude) がやらないこと・できないこと

- 発注・発注コードの起動・live permission の付与・オペレーター承認の代行 (本人打鍵が必要)。
- 「月曜に必ず利益」の約束。上のレンジと停止規律が正直な全て。
