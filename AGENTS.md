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
- **後付けの一律EXIT判定は作らない**。exit判断は各戦略ワーカー/専用 `exit_worker` のみが行う。`quant-strategy-control` は `entry/exit/global_lock` のガードのみで、全戦略に対する共通ロジックの事後的拒否/抑止を追加しない。
- 各戦略は `entry_thesis` に「`entry_probability`」と「`entry_units_intent`」を必須で渡して、`order_manager` はここを受けるのみとする。`session_open` を含む `AddonLiveBroker` 経路でも、order 送出時にこの2値を確実に注入する。確率閾値・サイズ設計は戦略ローカルで行い、共通レイヤは強制的に戦略を選別しない（ガード・リスク系の拒否のみ）。
- **黒板協調（意図調整）は `execution/strategy_entry.py` 経由で実装**する。  
  - `market_order` / `limit_order` は、`/order/coordinate_entry_intent` を呼び、`instrument`/`pocket`/`strategy_tag`/`side`/`raw_units`/`entry_probability` を共有 DB（`entry_intent_board`）で照合して、同時衝突時に戦略意図を縮小または拒否する。  
  - `order_manager` は `order_manager` 側で黒板上書きをせず、ガード・リスク（margin/損失上限など）による最終拒否・縮小だけを行う。  
  - `entry_probability` が極端に低い、あるいは逆方向意図が優勢な場合は拒否または縮小し、最終ユニットは `strategy_entry` で反映した上でのみ送出する。  
  - `manual` や `strategy_tag` 解決不可、`min_units` 未満の縮小結果は `order_manager` 経路に流さず、協調拒否として扱う。
- 発注経路はワーカーが直接 OANDA に送信するのが既定（`SIGNAL_GATE_ENABLED=0` / `ORDER_FORWARD_TO_SIGNAL_GATE=0`）。共通ゲートを使う場合のみ両フラグを 1 にする。
- 共通エントリー/テックゲート（`entry_guard` / `entry_tech`）は廃止・使用禁止。
- 型ゲート（Pattern Gate）は `workers/common/pattern_gate.py` を `execution/order_manager.py` preflight に適用する。**ただし全戦略一律強制はしない**（デフォルトは戦略ワーカーの opt-in）。
- 運用方針は「全て動的トレード」。静的な固定パラメータに依存せず、戦略ごとのローカル判定とリスク制御で都度更新する。
- **重要**: 本番稼働は VM。運用上の指摘・報告・判断は必ず VM（ログ/DB/プロセス）または OANDA API を確認して行い、ローカルの `logs/*.db` やスナップショット/コード差分だけで断定しない。
- 変更は必ず `git commit` → `git push` → VM 反映（`scripts/vm.sh ... deploy -i -t` 推奨）で行う。未コミット状態やローカル差し替えでの運用は不可。
- 変更点は必ず AGENTS と実装仕様側へ同時記録すること。少なくとも `docs/WORKER_REFACTOR_LOG.md` と関連仕様（`docs/WORKER_ROLE_MATRIX_V2.md`/`docs/ARCHITECTURE.md` 等）へ追記し、追跡可能な監査ログを残す。
- 並行作業により「エージェントが触っていない未コミット差分」が作業ツリーに残っていることがある。その場合でも **自分が変更したファイルだけ** をステージして commit/push→VM反映すればよい（他の差分は混ぜない・勝手に戻さない）。
- **本番ブランチ運用**: 本番 VM は原則 `main` のみを稼働ブランチにする。`codex/*` など作業ブランチを本番常駐させない。
- **本番反映の固定手順**: `main` へ統合（merge/rebase）→ `git push origin main` → `scripts/vm.sh ... deploy -b main -i --restart <target-unit> -t` を必須化する。`<target-unit>` は起動中の systemd ユニット（最低でもデータ/制御を含む `quant-market-data-feed.service` 等）を明示する（`pull` のみ禁止）。
- **反映確認の必須チェック**: デプロイ後に VM で `git rev-parse HEAD` と `git rev-parse origin/main` の一致を確認し、対象ユニットの `journalctl -u <target-unit>` で直近 `Application started!` がデプロイ後であることを確認する。`git pull` 後に再起動が無い場合は「未反映」と見なす。
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
- `docs/KATA_MICRO_RANGEBREAK.md`: micro（`MicroRangeBreak`）の型（Kata）設計・運用。
- `docs/KATA_PROGRESS.md`: 型（Kata）の進捗ログ（VMスナップショット/展開計画）。
- `docs/WORKER_REFACTOR_LOG.md`: ワーカー再編（データ供給・制御・ENTRY/EXIT分離）の確定記録。

## 5. ワーカー再編（V2）—役割を完全分離

### 方針（固定）
- 詳細は `docs/WORKER_ROLE_MATRIX_V2.md`（最上位フロー図・運用制約）を正規版として参照。

- **データ面**: OANDA から tick/足を受けるのは `quant-market-data-feed` のみ。`tick_window` と `factor_cache` を更新することを唯一の責務とする。
- **制御面**: `quant-strategy-control` のみが `entry/exit/global_lock` を持つ。  
  各戦略ワーカーは自分の `ENTRY/EXIT` の判定にこれを参照する。
- **戦略面**: 各戦略は必ず `ENTRY ワーカー` と `EXIT ワーカー` を `1:1` で運用。  
  `strategy module が複数戦略を内部に持つ` 形は不可。
- 各戦略のENTRY判断は strategy ロジック側で完結。`entry_probability` / `entry_units_intent` を `entry_thesis` へ付与し、`order_manager` はその意図を参照して制約内で実行する。
- **オーダー面**: `execution/order_manager.py` の処理は **新規ワーカー分離対象**。  
  `quant-order-manager` へ移設済み。戦略群は本ワーカーを介してのみ注文実行を実施する。
  - 戦略の意図協調（黒板）は `execution/strategy_entry.py` の同一意図判定呼び出しにより、最終ロットは pre-order で決定してから `order_manager` へ渡す。
- **ポジ面**: `execution/position_manager.py` も **新規ワーカー分離対象**。  
  `quant-position-manager` へ移設済み。戦略群は本ワーカーを介してのみ保有状態を参照する。
- **分析・監視面**: `quant-pattern-book`, `quant-range-metrics`, `quant-dynamic-alloc`,
  `quant-ops-policy` 等は「データ分析/状態分析」ロールに固定し、戦略実行ロジックへ混在させない。

### V2 で保有すべき固定サービス群（実行時）
- `quant-market-data-feed`
- `quant-strategy-control`
- `quant-order-manager`
- `quant-position-manager`
- 戦略 ENTRY/EXIT 一対一サービス（scalp / micro / s5）
- 補助: `quant-pattern-book`, `quant-range-metrics`, `quant-ops-policy`, `quant-dynamic-alloc`, `quant-policy-guard`

### V2 移行目標（debt）
- `quantrabbit.service`（monolithicエントリ）を廃止。`main.py` は開発・分析用途としてのみ扱い、本番起動用 unit から外す。
- オーダー処理とポジ管理は `V2` で「別サービス化」済み。

```mermaid
flowchart LR
  Tick[OANDA Tick/Candle] --> MF["quant-market-data-feed"]
  MF --> TW["tick_window"]
  MF --> FC["factor_cache"]
  FC --> SW["strategy workers (ENTRY)"]
  TW --> SW
  SC["quant-strategy-control"] --> SW
  SC --> EW["strategy workers (EXIT)"]
  SW --> OM["quant-order-manager"]
  OM --> OANDA[OANDA Order API]
  PM["quant-position-manager"] --> EW
  PM --> PMDB[(trades.db / positions)]
  AF["analysis services"] --> SW
```

## 6. チーム / タスク運用ルール（要点）
- 1 ファイル = 1 PR、Squash Merge、CI green。
- 秘匿情報は Git に置かない。
- タスク台帳は `docs/TASKS.md` を正本とし、Open→進行→Archive の流れで更新。
- オンラインチューニング ToDo は `docs/autotune_taskboard.md` に集約。

## 7. 型（Pattern Book）運用ルール
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
- opt-in 戦略（main 実装済み）:
- `scalp_ping_5s`: `SCALP_PING_5S_PATTERN_GATE_OPT_IN=1`
- `scalp_m1scalper`: `SCALP_M1SCALPER_PATTERN_GATE_OPT_IN=1`
- `TickImbalance`（`workers/scalp_precision`）: `TICK_IMB_PATTERN_GATE_OPT_IN=1`（+必要なら `TICK_IMB_PATTERN_GATE_ALLOW_GENERIC=1`）
- `MicroRangeBreak`（`workers/micro_multistrat`）: `MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN=1`（+必要なら `MICRO_RANGEBREAK_PATTERN_GATE_ALLOW_GENERIC=1`）
- 運用判断は必ずVM実データで行う。`patterns.db` / `pattern_book*.json` の時刻・件数・quality分布を確認してから閾値調整する。

## 8. V2導線フリーズ運用（env/systemd監査）

- 方針: 発注導線・EXIT導線は V2 分離構成を変えず固定し、後付けの一律判断を導入しない。
- 不変条件
  - エントリーは strategy worker → `execution.order_manager` → 共通 preflight のみ。
  - `order_manager` が戦略の選別ロジックを上書きしない。許可/縮小/拒否は preflight ガード・risk 側条件のみに限定。
  - close/exit 判断は各 strategy の exit_worker と該当ワーカー側ルールを主とし、`quant-order-manager`/`quant-position-manager` は通路と参照窓口として扱う。
  - `entry_probability` と `entry_units_intent` を `entry_thesis` で必須維持。
  - `quantrabbit.service` を本番主導線にしない。
- 実VM監査（実行）
  - running 状態（V2固定群）
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "systemctl list-units --type=service --state=running --no-pager | grep -E 'quant-(market-data-feed|strategy-control|order-manager|position-manager|scalp|micro|pattern|range|dynamic|policy)'"
  ```
  - unit enable 状態（主要群）
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "systemctl list-unit-files --type=service | grep -E 'quant-market-data-feed|quant-order-manager|quant-position-manager|quant-strategy-control|quant-scalp-.*|quant-micro-.*'"
  ```
  - runtime env `ops/env/quant-v2-runtime.env` の監査キー
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "for k in WORKER_ONLY_MODE MAIN_TRADING_ENABLED SIGNAL_GATE_ENABLED ORDER_FORWARD_TO_SIGNAL_GATE EXIT_MANAGER_DISABLED ORDER_MANAGER_SERVICE_ENABLED ORDER_MANAGER_SERVICE_FALLBACK_LOCAL POSITION_MANAGER_SERVICE_ENABLED POSITION_MANAGER_SERVICE_FALLBACK_LOCAL ORDER_PATTERN_GATE_ENABLED ORDER_PATTERN_GATE_GLOBAL_OPT_IN BRAIN_ENABLED ORDER_MANAGER_BRAIN_GATE_ENABLED ORDER_MANAGER_FORECAST_GATE_ENABLED POLICY_HEURISTIC_PERF_BLOCK_ENABLED ENTRY_GUARD_ENABLED ENTRY_TECH_ENABLED; do v=$(grep -E \"^${k}=\\\"?\" /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env | head -n 1); echo \"${k}=${v:-<MISSING>}\"; done"
  ```
  - 主要 unit の env/起動引数（照合）
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "for u in quant-order-manager.service quant-position-manager.service quant-strategy-control.service quant-market-data-feed.service quant-scalp-ping-5s.service quant-scalp-ping-5s-exit.service quant-micro-multi.service quant-micro-multi-exit.service quant-scalp-macd-rsi-div.service quant-scalp-macd-rsi-div-exit.service; do if systemctl cat \"$u\" >/dev/null 2>&1; then echo \"---$u---\"; systemctl cat \"$u\" | sed -n '1,130p'; fi; done"
  ```
  - `main` 側自動監査（service化）:
  ```bash
  sudo bash scripts/install_trading_services.sh --repo /home/tossaki/QuantRabbit --units quant-v2-audit.service quant-v2-audit.timer
  sudo systemctl enable --now quant-v2-audit.timer
  ```
  - 監査報告:
  - `logs/ops_v2_audit_latest.json`
  - `journalctl -u quant-v2-audit.service -n 200 --no-pager`
  - 戦略workerの運用ルール（監査用）
  - `systemctl cat` で `EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env` と
    `EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-<worker>.env` の両方が存在することを確認する。
    - 例: `quant-scalp-ping-5s*.service` は上記2つ＋戦略オーバーライドenv (`scalp_ping_5s*.env`) を持つ。
    - `quant-order-manager.service` / `quant-position-manager.service` は
      `ops/env/quant-order-manager.env` / `ops/env/quant-position-manager.env` を持つことを確認する。
- 判定のゴール
  - 上位導線群が active/running し、`quantrabbit.service` が主導線になっていない。
  - 上記監査キーがフリーズ方針に一致し、service 側の `EnvironmentFile` が一本化されている。
