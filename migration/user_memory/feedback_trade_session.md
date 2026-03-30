---
name: trade_session_rules
description: 「トレード開始」で投資会社レベルのマルチエージェント裁量トレード体制を即起動するルール
type: feedback
---

「トレード開始」でトレードセッションを即開始する。

**Why:** ユーザーはClaudeの裁量スキャルプを主戦略として採用。一言で投資会社レベルの体制が立ち上がる必要がある。

---

## セッション起動手順

### Phase 0: 安全確認 (30秒)
1. `ps aux | grep workers` でボットプロセスを確認 → 動いていたら即kill
2. 全openTrades確認 → 不明なポジションはクローズ
3. 全pendingOrders確認 → 不明なオーダーはキャンセル
4. 口座状況表示 (NAV, Balance, MarginAvailable)

### Phase 1: マルチエージェント起動 (並列)

**Agent 1: マーケットアナリスト** (background)
- H1/M5/M1のMTF分析
- N波動検出 (`analysis/patterns/n_wave.py` を参照)
- レジーム分類 (TRENDING/RANGE/CHOPPY)
- M5サポート/レジスタンス特定
- 3層スコアリング (H1 Context + M5 Setup + M1 Trigger)

**Agent 2: ニュース・マクロリサーチ** (background)
- WebSearchで当日のFX関連ニュース (BOJ, Fed, 経済指標)
- USD/JPYに影響するイベントスケジュール
- 市場センチメント判断

**Agent 3: 過去データ分析** (background)
- OANDA transaction historyから直近の勝率/PF算出
- pattern_book_deep.jsonの高勝率パターン確認
- 当日の時間帯別パフォーマンス分析

### Phase 2: 統合判断 → エントリー (メインClaude)
- 3エージェントの結果を統合
- 裁量でエントリー判断 (スコア-3以下 or +3以上でのみ)
- OANDA REST API直接注文 (ボット経由禁止)
- ファイルログ開始 (`logs/live_trade_log.txt` + `docs/TRADE_LOG_YYYYMMDD.md`)

### Phase 3: スケジュールタスクによる継続トレード

**スケジュールタスク `scalp-trader` が5分ごとにClaudeセッションを自動起動。**
- 毎回フルの分析→判断→(必要なら)エントリー/決済
- 常駐スクリプトではない（毎回起動→終了）
- 判断は毎回言語化される
- TP/SLはエントリー時に必ず設定 → OANDA側で自動決済
- タスク設定: `/Users/tossaki/.claude/scheduled-tasks/scalp-trader/SKILL.md`

**手動セッション（「トレード開始」）との使い分け:**
- スケジュールタスク: 継続的な自動裁量トレード
- 手動セッション: ユーザーが対話しながら判断したいとき

**NGパターン (変わらず):**
- `while True` + `time.sleep()` の常駐スクリプト起動
- Bash `run_in_background` で監視ループを放置
- 判断を言語化せずに自動でクローズ/エントリー

---

## OANDA API Reference (quick)
- Base: `https://api-fxtrade.oanda.com`
- Creds: `config/env.toml` → oanda_token, oanda_account_id
- Order: `POST /v3/accounts/{acct}/orders`
- Close: `PUT /v3/accounts/{acct}/trades/{id}/close`
- Modify: `PUT /v3/accounts/{acct}/trades/{id}/orders`
- Margin rate=0.04 (1:25)

---

## データドリブン原則 (2026-03-18分析)

**TP到達 = 勝率100%, +2,863 JPY** → TPを設定して任せるのが最強
**手動クローズ = 勝率30%, -820 JPY** → 焦って切るのが最大の負け原因
**SL = 勝率25%, -3,527 JPY** → SLが近すぎる

→ **SLは広め (2x ATR, 最低10pip)。TPは確実に設定。手動クローズは最終手段。**

---

## 絶対ルール
- **トレードを止めない** — 連敗・イベント・低WRでも続ける。ロット縮小+SL拡大で対応
- ボットプロセス起動禁止。`workers/` のコードは参照のみ
- RSI 70超えでロング追加禁止
- H1方向に逆らわない
- M5 RSI 25以下でショート追いかけ禁止
- SL設定済みポジションは手動クローズ禁止 (TP到達を待つ)
- 注文は必ずOANDA REST API直接 (urllib)
- 毎回のエントリー/決済を `logs/live_trade_log.txt` にファイル記録
