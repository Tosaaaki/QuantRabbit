# AGENTS.md – QuantRabbit Agent Specification（整理版）

## 1. ミッション / 役割
> 狙い: USD/JPY で 資産の10%増を狙う 24/7 無裁量トレーディング・エージェント。  
> 凄腕のプロトレーダーのトレードを再現するシステム。  
> 境界: 発注・リスクは機械的、曖昧判断はローカルルール（LLMは任意のゲートのみ）。

- VM 上で常時ログ・オーダーを監視。
- トレード判断・ロット・利確/損切り・保有調整は固定値運用を避け、市場状態に応じて常時動的に更新する。
- 手動玉を含めたエクスポージャを高水準で維持。
- PF/勝率が悪化した戦略・時間帯を自動ブロック。
- マージン拒否やタグ欠損を検知したら即パラメータ更新＆デプロイ。
- ユーザ手動トレードと併走し、ポジション総量を管理。
- 最優先ゴールは「資産を劇的に増やす」。必要なリスクテイクと調整を即断・即実行。

## 2. 非交渉ルール（必ず守る）
- ニュース連動パイプラインは撤去済み（`news_fetcher` / `summary_ingestor` / NewsSpike は無効）。
- LLM（Vertex）は **任意の Brainゲート** に限定して使用可。メインの判定は `analysis/local_decider.py` のローカル判定のみ。
- Brainゲート: `workers/common/brain.py` を `execution/order_manager.py` の preflight に適用し、**許可/縮小/拒否**を返す（default: disabled）。
- Brain 有効化は `BRAIN_ENABLED=1` と Vertex 認証（`VERTEX_PROJECT_ID` / `VERTEX_LOCATION` 等）が必須。
- 現行デフォルト: `WORKER_ONLY_MODE=true` / `MAIN_TRADING_ENABLED=0`。共通 `exit_manager` はスタブ化され、エントリー/EXIT は各戦略ワーカー＋専用 `exit_worker` が担当。
- 発注経路はワーカーが直接 OANDA に送信するのが既定（`SIGNAL_GATE_ENABLED=0` / `ORDER_FORWARD_TO_SIGNAL_GATE=0`）。共通ゲートを使う場合のみ両フラグを 1 にする。
- 共通エントリー/テックゲート（`entry_guard` / `entry_tech`）は廃止・使用禁止。
- 型ゲート（Pattern Gate）は `workers/common/pattern_gate.py` を `execution/order_manager.py` preflight に適用する。**ただし全戦略一律強制はしない**（デフォルトは戦略ワーカーの opt-in）。
- 運用方針は「全て動的トレード」。静的な固定パラメータに依存せず、戦略ごとのローカル判定とリスク制御で都度更新する。
- **重要**: 本番稼働は VM。運用上の指摘・報告・判断は必ず VM（ログ/DB/プロセス）または OANDA API を確認して行い、ローカルの `logs/*.db` やスナップショット/コード差分だけで断定しない。
- 変更は必ず `git commit` → `git push` → VM 反映（`scripts/vm.sh ... deploy -i -t` 推奨）で行う。未コミット状態やローカル差し替えでの運用は不可。
- **本番ブランチ運用**: 本番 VM は原則 `main` のみを稼働ブランチにする。`codex/*` など作業ブランチを本番常駐させない。
- **本番反映の固定手順**: `main` へ統合（merge/rebase）→ `git push origin main` → `scripts/vm.sh ... deploy -b main -i --restart quantrabbit.service -t` を必須化する（`pull` のみ禁止）。
- **反映確認の必須チェック**: デプロイ後に VM で `git rev-parse HEAD` と `git rev-parse origin/main` の一致を確認し、さらに `journalctl -u quantrabbit.service` の最新 `Application started!` がデプロイ後であることを確認する。`git pull` 後に再起動が無い場合は「未反映」と見なす。
- VM 削除禁止。再起動やブランチ切替で代替し、`gcloud compute instances delete` 等には触れない。

## 3. 時限情報（必ず最新を参照）
- 2025-12 の攻め設定、mask 済み unit などは `docs/OPS_CURRENT.md` を参照。

## 4. 仕様ドキュメント索引
- `docs/INDEX.md`: ドキュメントの起点。
- `docs/ARCHITECTURE.md`: システム全体フロー、データスキーマ。
- `docs/RISK_AND_EXECUTION.md`: エントリー/EXIT/リスク制御、OANDA マッピング。
- `docs/OBSERVABILITY.md`: データ鮮度、ログ、SLO/アラート、検証パイプライン。
- `docs/RANGE_MODE.md`: レンジモード強化とオンラインチューニング運用。
- `docs/OPS_GCP_RUNBOOK.md`: GCP/VM 運用ランブック。
- `docs/OPS_SKILLS.md`: 日次運用スキル運用。
- `docs/KATA_SCALP_PING_5S.md`: 5秒スキャ（`scalp_ping_5s`）の型（Kata）設計・運用。
- `docs/KATA_SCALP_M1SCALPER.md`: M1スキャ（`scalp_m1scalper`）の型（Kata）設計・運用。
- `docs/KATA_PROGRESS.md`: 型（Kata）の進捗ログ（VMスナップショット/展開計画）。

## 5. チーム / タスク運用ルール（要点）
- 1 ファイル = 1 PR、Squash Merge、CI green。
- 秘匿情報は Git に置かない。
- タスク台帳は `docs/TASKS.md` を正本とし、Open→進行→Archive の流れで更新。
- オンラインチューニング ToDo は `docs/autotune_taskboard.md` に集約。

## 6. 型（Pattern Book）運用ルール
- 目的: トレード履歴から「勝てる型 / 避ける型」を継続学習し、エントリー時の `block/reduce/boost` 判断に使う。
- 収集ジョブ: `scripts/pattern_book_worker.py`。systemd は `quant-pattern-book.service` + `quant-pattern-book.timer`（5分周期）。
- 実行Python: `quant-pattern-book.service` は必ず `/home/tossaki/QuantRabbit/.venv/bin/python` を使う（system python禁止）。
- 主な出力:
  - DB: `/home/tossaki/QuantRabbit/logs/patterns.db`
  - JSON: `/home/tossaki/QuantRabbit/config/pattern_book.json`
  - deep JSON: `/home/tossaki/QuantRabbit/config/pattern_book_deep.json`
- 主なテーブル:
  - `pattern_trade_features`（トレード特徴）
  - `pattern_stats` / `pattern_actions`（基本集計とアクション）
  - `pattern_scores` / `pattern_drift` / `pattern_clusters`（深掘り分析）
- 深掘り分析（`analysis/pattern_deep.py`）は `numpy/pandas/scipy/sklearn` を使い、統計検定・ドリフト・クラスタリングを更新する。
- エントリー連携:
  - `execution/order_manager.py` preflight で `workers/common/pattern_gate.py` を評価。
  - `quality=avoid` かつ十分サンプルで `pattern_block`。
  - `suggested_multiplier` と `drift` でロットを縮小/拡大（下限未満は `pattern_scale_below_min`）。
- 重要: デフォルトは戦略opt-in。`ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0` を維持し、各戦略ワーカーの `entry_thesis` に `pattern_gate_opt_in=true` を明示したものだけ適用する。
- 既定opt-in戦略: `scalp_ping_5s`（`SCALP_PING_5S_PATTERN_GATE_OPT_IN=1`）。
  - 追加予定: `scalp_m1scalper`（`SCALP_M1SCALPER_PATTERN_GATE_OPT_IN=1`）。
- 運用判断は必ずVM実データで行う。`patterns.db` / `pattern_book*.json` の時刻・件数・quality分布を確認してから閾値調整する。
