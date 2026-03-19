# QuantRabbit — Claude裁量FXトレードシステム

このリポジトリはClaudeによる裁量FXトレーディングの運用基盤。
ボット/自動売買ではない。Claudeが自分の頭で判断してOANDA APIで直接注文する。
テクニカル計算・機械的ポジション管理はPythonスクリプトが担う。

## アーキテクチャ (v4)

### Python層（LLMコストゼロ、30秒間隔）
- `scripts/trader_tools/live_monitor.py` — launchdで30秒ごとに実行
  - データ: pricing, S5/M1/M5指標(divergence, Ichimoku, VWAP含む), H1/H4バイアス
  - ペアプロファイル: pair別のspread gate, SL/TP範囲, ADX閾値, セッション適性, ペア性格
  - 機械的ポジ管理: `logs/trade_registry.json` のルールに従いBE移動/ATR追従SL/trail/partial/close実行
  - リスク: margin使用率, ドローダウン, サーキットブレーカー
  - 出力: `logs/live_monitor.json`, `logs/live_monitor_summary.json`（軽量版）

### Claude層（3エージェント裁量体制）

**核心原則: Claudeはプロの裁量トレーダー。スコアに頼らず市場を自分で読む。**

| タスク | モデル | 間隔 | ロック | 役割 |
|--------|--------|------|--------|------|
| trader | Opus | 2-3分 | なし | 一人のプロトレーダー。市場を読み、スキャルプ(2-5pip)もスウィング(10-50pip)も判断 |
| analyst | Sonnet | 10分 | global | マクロ分析・クロスペアフロー・パフォーマンス解析・ツール開発 |
| secretary | Sonnet | 11分 | なし | ヘルスチェック・クリティカルアラートのみ。読み取り専用 |

- 排他制御: `scripts/trader_tools/task_lock.py` でグローバルロック（`global_agent`）。traderとsecretaryはロック不要
- エージェント間連携: `logs/shared_state.json`
- ポジション所有権: `logs/trade_registry.json`
- タスク定義: `~/.claude/scheduled-tasks/{trader,analyst,secretary}/SKILL.md`

### v3→v4の変更点（2026-03-20）
- scalp-fast + swing-trader → **trader**（一人のトレーダーがスキャルプもスウィングも判断）
- market-radar + macro-intel → **analyst**（分析を一人に統合）
- secretary → 簡素化（官僚的監査をやめ、ヘルスチェックに集中）
- live_monitor_summary.json → スコア(long_score/short_score)ではなく生データ(H1バイアス、DI、ATR、regime)を提供
- プロンプト → 手順書型チェックリスト → トレーダーの心得・原則ベース

### 自己改善ループ
```
trader → REVIEW/REFLECTION/PATTERN CHECK → live_trade_log.txt
analyst → reads trade_log → performance分析 → shared_state更新 / prompt改善
secretary → health監視 → critical alerts
```

## 絶対ルール

- **ボット禁止**: `while True` + `sleep` の常駐スクリプトを書かない。`workers/` のコードは参照のみ
- **Pythonツールは可**: テクニカル計算・機械的ポジ管理のスクリプトは可。トレード判断はClaude裁量
- **OANDA直接注文**: urllib で REST API を叩く。ボットプロセス経由禁止
- **裁量トレーダーであれ**: スコアの言いなりにならない。市場を読んで自分で判断する
- テクニカル指標は `logs/live_monitor_summary.json`（軽量）or `logs/live_monitor.json`（フル）から読む（手計算しない）
- 注文は必ず `logs/live_trade_log.txt` にファイル記録
- エントリー後は `logs/trade_registry.json` に登録（Python管理ルール適用のため）

## 変更時の必須ルール

**プロンプト・タスク・スクリプト・アーキテクチャを変更したら、必ず以下を実行:**

1. **CLAUDE.md更新**: アーキテクチャやタスク構成が変わった場合はこのファイルを更新。**バージョンが変わったら随時更新**
2. **メモリ更新**: 該当するメモリファイル（`~/.claude/projects/.../memory/*.md`）を更新。なければ新規作成してMEMORY.mdにインデックス追加
3. **変更ログ追記**: `docs/CHANGELOG.md` に日時と変更内容を1行で追記
4. **mainにマージ**: ワークツリーで編集した場合は**必ずmainにマージ**する。traderタスクの作業ディレクトリはmainリポジトリ(`/Users/tossaki/App/QuantRabbit`)のため、マージしないとタスクに変更が見えない

これを怠ると次のセッションのClaudeが旧構造で動き、障害の原因になる。

## ドキュメントマップ

### 必読（タスク実行前に読むもの）

| ファイル | 読む人 | 内容 |
|----------|--------|------|
| `CLAUDE.md` (このファイル) | 全員 | アーキテクチャ全体像、絶対ルール、変更時の必須ルール |
| `docs/TRADER_PROMPT.md` | trader | 裁量トレーダーの心得・エントリー・利確・振り返り |
| `docs/ANALYST_PROMPT.md` | analyst | マクロ分析・クロスペア・パフォーマンス・ツール開発 |
| `docs/SECRETARY_PROMPT.md` | secretary | ヘルスチェック・クリティカルアラート |

### 運用ドキュメント（必要時に参照）

| ファイル | 内容 |
|----------|------|
| `docs/CHANGELOG.md` | 全変更の時系列ログ。**変更時は必ず追記** |
| `docs/TRADE_LOG_*.md` | 日次トレード記録・振り返り |
| `docs/CURRENT_MECHANISMS.md` | 戦略・シグナル・ゲートの一覧 |
| `docs/hedge_plan.md` | ヘッジ/両建ての設計と注意点 |
| `docs/SL_POLICY.md` | SLの設計方針 |

### レガシー（読まなくていい、参考のみ）

`docs/` 内の以下はv3以前の設計書。現在のv4体制では不使用:
- v3プロンプト: `SCALP_FAST_PROMPT.md`, `SWING_TRADER_PROMPT.md`, `MARKET_RADAR_PROMPT.md`, `MACRO_INTEL_PROMPT.md`, `SCALP_TRADER_PROMPT.md`
- 旧workers/VM: `ARCHITECTURE.md`, `WORKER_*.md`, `VM_*.md`, `GCP_*.md`, `DEPLOYMENT.md`
- その他旧設計: `OPS_*.md`, `KATA_*.md`, `ONLINE_TUNER.md`, `REPLAY_STANDARD.md`, `REPO_HISTORY_*.md`

### ランタイムファイル（logs/）

| ファイル | 誰が書く | 誰が読む | 内容 |
|----------|----------|----------|------|
| `logs/live_monitor.json` | live_monitor.py | 全タスク | フルモニター画面（30秒更新） |
| `logs/live_monitor_summary.json` | live_monitor.py | trader | 軽量サマリー（~2KB、traderはこちらを優先） |
| `logs/trade_registry.json` | trader(entry時) | live_monitor.py | ポジション所有権と管理ルール |
| `logs/shared_state.json` | analyst/secretary | 全タスク | エージェント間連携（macro_bias, alerts, one_thing_now） |
| `logs/live_trade_log.txt` | trader + monitor | 全タスク | トレード実行ログ（時系列） |
| `logs/technicals_*.json` | refresh_factor_cache | live_monitor.py | H1/H4テクニカル指標 |
| `logs/strategy_feedback.json` | trade_performance.py | analyst | パフォーマンス統計 |
| `logs/secretary_report.json` | secretary | 全タスク | 秘書レポート |

### スクリプト

| ファイル | 実行方法 | 内容 |
|----------|----------|------|
| `scripts/trader_tools/live_monitor.py` | launchd 30秒 | データ収集+機械的ポジ管理 |
| `scripts/trader_tools/refresh_factor_cache.py` | analystが呼ぶ | H1/H4テクニカル指標の更新 |
| `scripts/trader_tools/task_lock.py` | analyst | グローバルロック排他制御 |
| `scripts/trader_tools/trade_performance.py` | analystが呼ぶ | パフォーマンス集計（live_trade_log.txt解析） |
| `scripts/trader_tools/setup_live_monitor.sh` | 手動(初回) | launchdへのmonitor登録 |
| `scripts/trader_tools/setup_scheduled_tasks.sh` | 手動(初回) | 全タスクの登録 |

## 主要ディレクトリ

- `docs/` — プロンプト、変更ログ、戦略ドキュメント
- `scripts/trader_tools/` — live_monitor, 分析ツール、ロック機構
- `indicators/` — テクニカル指標計算エンジン (IndicatorEngine)
- `logs/` — 共有状態、トレードログ、モニター出力、trade_registry
- `config/env.toml` — OANDA APIキー等(gitignore対象)

## ユーザーコマンド

- 「トレード開始」→ マルチエージェント裁量トレードセッション起動
- 「秘書」→ トレーディング秘書モード(状況確認・指示伝達)
