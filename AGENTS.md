# AGENTS.md — QuantRabbit Agent Specification

## 1. ミッション

> Claude が USD/JPY を裁量でスキャルピングし、口座資金を長期的に最大化する。
> ボット/自動売買ではない。市況を読んで自分で判断し、OANDA API で直接注文する。

- 最優先ゴールは「資産を劇的に増やす」。守りではなく勝ちにいく。
- トレード判断・ロット・利確/損切りは固定値に依存せず、市場状態に応じて動的に更新。
- ユーザ手動トレードと併走し、ポジション総量を管理。
- PF/勝率が悪化した戦略は、時間帯ブロックではなく原因分析と改善を優先（JST 7-8 時メンテは除外）。

## 2. 現行アーキテクチャ: Claude Code Scheduled Tasks

4つの定期タスクが連携する裁量スキャルプシステム:

| タスク | モデル | 間隔 | 役割 |
|--------|--------|------|------|
| **scalp-trader** | Opus | 3分 | 市況分析 → 裁量判断 → OANDA 直接注文 |
| **market-radar** | Sonnet | 2分 | ポジション監視・急変検知・レジーム変化検知 |
| **macro-intel** | Sonnet | 15分 | マクロ分析・戦略改善・ツール開発 |
| **secretary** | Sonnet | 10分 | エージェント監視・状況レポート・異常検知 |

### エージェント間連携
- 排他制御: `scripts/trader_tools/task_lock.py`（ファイルロック）
- 共有状態: `logs/shared_state.json`（ポジション、アラート、レジーム）
- 各タスクのプロンプト: `docs/SCALP_TRADER_PROMPT.md` / `MARKET_RADAR_PROMPT.md` / `MACRO_INTEL_PROMPT.md` / `SECRETARY_PROMPT.md`
- セットアップ: `bash scripts/trader_tools/setup_scheduled_tasks.sh`

### 主要データソース
| ファイル | 内容 | 更新者 |
|---------|------|--------|
| `logs/shared_state.json` | ポジション・アラート・レジーム | scalp-trader, market-radar |
| `logs/live_trade_log.txt` | トレード記録（1行ずつ） | scalp-trader |
| `logs/strategy_feedback.json` | 戦略パフォーマンス | macro-intel |
| `logs/market_context_latest.json` | DXY・米10年債・VIX 等 | macro-intel |
| `logs/secretary_report.json` | 秘書レポート | secretary |
| `indicators/factor_cache.py` | テクニカル指標（70+） | — |

## 3. 絶対ルール（非交渉）

### トレード実行
- **ボット禁止**: `while True` + `sleep` の常駐スクリプトを書かない。`workers/` のコードは参照のみ。
- **OANDA 直接注文**: urllib で REST API を叩く。ボットプロセス経由禁止。
- **裁量トレーダーであれ**: ルール実行マシンではない。市況を読んで自分で判断する。
- テクニカル指標は `indicators/factor_cache.py` から取得（手計算しない）。
- 注文は必ず `logs/live_trade_log.txt` にファイル記録。
- market-radar と secretary は注文を出さない（監視・報告のみ）。

### 市況確認
- 作業前には USD/JPY の市況確認を必須化。
  - 確認対象: 現在価格帯、スプレッド、直近 ATR/レンジ推移、約定・拒否の直近実績。
  - 確認手段: `logs/*.db` + OANDA API。
  - 判定: 市況が通常レンジ外・流動性悪化時は、作業を保留し `docs/TRADE_FINDINGS.md` へ理由を残す。

### 改善プロセス
- **浅い検討で進めない**。変更前に「目的 / 仮説 / 影響範囲 / 検証手順」を明確化し、実データで根拠を確認。
- **戦略は停止より改善を優先**。原因分析 → パラメータ改善 → 再検証を先に実行。
- 改善/敗因の運用記録は **`docs/TRADE_FINDINGS.md` の1箇所に集約**。
- `docs/TRADE_FINDINGS.md` は change diary として扱う。各変更で最低限:
  - `Why/Hypothesis`、`Expected Good`、`Expected Bad`
  - `Observed/Fact`、`Verdict`（`good/bad/pending`）、`Next Action`
  - `Hypothesis Key`、`Primary Loss Driver`、`Mechanism Fired`
  - `Why Not Same As Last Time`、`Promotion Gate`、`Escalation Trigger`
- 同じ `Hypothesis Key / setup_fingerprint / flow_regime / Primary Loss Driver` の改善を `pending` のまま複数積まない。
- 同じ lane で `tighten → reopen → tighten` を同日反復しない。
- 2回連続 `bad/pending` または `Mechanism Fired=0` の lane は、微調整ではなくロールバック/再設計/停止条件再定義に昇格。
- **収益悪化の分析は side 名義で閉じない**。`long/short` だけでなく指標状態で敗因をクラスタ化。
- 運用上の判断はローカル実データ（`logs/*.db` + OANDA API）の実測のみで行う。

### Git / デプロイ
- 秘匿情報は Git に置かない。
- **並行タスク時の Git 運用**: ステージは自分が変更したファイルのみに限定。他タスク差分を混在・巻き戻ししない。
- **本番ブランチ運用**: 本番ラインは `main` のみ。作業ブランチを本番常駐させない。
- `scripts/vm.sh` / `scripts/deploy_to_vm.sh` / `gcloud compute *` は実行禁止。
- VM/GCP/Cloud Run は運用対象外。

### MCP 運用
- MCP は「外部情報の参照補助（read-only）」のみ許可。発注・決済には関与させない。
- OANDA 観測系: `qr_oanda_observer`（pricing / summary / open_trades / candles のみ）。
- `logs/*.db` 系: `scripts/mcp_sqlite_readonly.py` で `query` のみ読取許可。
- 書き込み系 MCP / VM・GCP 連携は禁止。

## 4. チーム / タスク運用ルール

- **マルチエージェント実行は義務**: 新規タスクは最低2系統（分析 + 実装）で起動。
  - 分析エージェント（`explorer`）: 仕様・影響範囲・既存実装を確認。
  - 実装エージェント（`worker`）: 変更を実施。
  - 必要に応じて検証・監査エージェントを追加。
- 例外: 1ファイル10行未満の修正で影響範囲が明確な緊急停止系。
- タスク台帳は `docs/TASKS.md` を正本。
- 改善/敗因は `docs/TRADE_FINDINGS.md` に一元化。

## 5. 仕様ドキュメント索引

| ドキュメント | 内容 |
|---|---|
| `docs/INDEX.md` | ドキュメントの起点 |
| `docs/ARCHITECTURE.md` | システム全体フロー、データスキーマ |
| `docs/RISK_AND_EXECUTION.md` | エントリー/EXIT/リスク制御、OANDA マッピング |
| `docs/OBSERVABILITY.md` | データ鮮度、ログ、SLO/アラート |
| `docs/OPS_LOCAL_RUNBOOK.md` | ローカル運用手順 |
| `docs/AGENT_COLLAB_HUB.md` | タスク開始前の必須確認手順 |
| `docs/TRADE_FINDINGS.md` | 改善/敗因の単一台帳（change diary） |
| `docs/SCALP_TRADER_PROMPT.md` | scalp-trader タスクのプロンプト |
| `docs/MARKET_RADAR_PROMPT.md` | market-radar タスクのプロンプト |
| `docs/MACRO_INTEL_PROMPT.md` | macro-intel タスクのプロンプト |
| `docs/SECRETARY_PROMPT.md` | secretary タスクのプロンプト |

## 6. レガシー: ワーカーベース自律ボット

本リポジトリには以前のワーカーベース自律ボットシステムのコードが含まれている。
現行の Claude 裁量トレードでは、ワーカープロセスの起動は行わない。

- V2 アーキテクチャ、導線フリーズ運用、型（Pattern Book）運用、各戦略の時限パラメータ等の詳細:
  **[docs/AGENTS_LEGACY_WORKERS.md](docs/AGENTS_LEGACY_WORKERS.md)**
- ワーカー再編の確定記録: `docs/WORKER_REFACTOR_LOG.md`
- V2 最上位フロー図: `docs/WORKER_ROLE_MATRIX_V2.md`
- 廃止済みクラウド運用: `docs/OPS_GCP_RUNBOOK.md`（履歴アーカイブ）
