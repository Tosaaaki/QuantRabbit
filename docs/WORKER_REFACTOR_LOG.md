# ワーカー再編の確定ログ（2026-02-13）

## 方針（最終確定）

- 各戦略は `ENTRY/EXIT` を1対1で持つ。
- `precision` 系のサービス名は廃止し、サービス名は `quant-scalp-*` へ切り分ける。
- `quant-hard-stop-backfill` / `quant-realtime-metrics` は削除対象。
- データ供給は `quant-market-data-feed`、制御配信は `quant-strategy-control` に分離。
- 補助的運用ワーカーは本体管理マップから除外。

## 補足（戦略判断責務の明確化）

- **方針確定**: 各戦略ワーカーは「ENTRY/EXIT判定の脳」を保持し、ロジックの主判断は各ワーカー固有で行う。
- `quant-strategy-control` は「最終実行可否」を左右する制御入力を配信するのみ（`entry_enabled` / `exit_enabled` / `global_lock` / メモ）。
- したがって、`strategy-control` が戦略ロジックを代行しているわけではなく、**各戦略の意思決定を中断/再開するガードレイヤー**として機能する。
- UI の戦略ON/OFFや緊急ロックはこのガードレイヤーを介して、並行中の戦略群へ即時反映する。

## 追加（実装済み）

- `systemd/quant-market-data-feed.service`
- `systemd/quant-strategy-control.service`
- `apps/autotune_ui.py`
  - `summary`/`ops` ビューに「戦略制御」セクションを追加
  - `POST /api/strategy-control`
  - `POST /ops/strategy-control`
- `systemd/quant-scalp-ping-5s-exit.service`
- `systemd/quant-scalp-macd-rsi-div-exit.service`
- `systemd/quant-scalp-tick-imbalance.service`
- `systemd/quant-scalp-tick-imbalance-exit.service`
- `systemd/quant-scalp-squeeze-pulse-break.service`
- `systemd/quant-scalp-squeeze-pulse-break-exit.service`
- `systemd/quant-scalp-wick-reversal-blend.service`
- `systemd/quant-scalp-wick-reversal-blend-exit.service`
- `systemd/quant-scalp-wick-reversal-pro.service`
- `systemd/quant-scalp-wick-reversal-pro-exit.service`
- `workers/market_data_feed/worker.py`
- `workers/strategy_control/worker.py`
- `workers/scalp_ping_5s/exit_worker.py`
- `workers/scalp_macd_rsi_div/exit_worker.py`
- `systemd/quant-order-manager.service`
- `systemd/quant-position-manager.service`
- `workers/order_manager/__init__.py`
- `workers/order_manager/worker.py`
- `workers/position_manager/__init__.py`
- `workers/position_manager/worker.py`
- `execution/order_manager.py`（service-first 経路追加、`_ORDER_MANAGER_SERVICE_*` 利用）
- `execution/position_manager.py`（service-first 経路追加、`_POSITION_MANAGER_SERVICE_*` 利用）
- `config/env.example.toml`（order/position service URL/enable 設定追加）
- `main.py`
  - `WORKER_SERVICES` に `market_data_feed` / `strategy_control` を追加。
  - `initialize_history("USD_JPY")` を `worker_only_loop` から撤去（初期シードを market-data-feed worker に移譲）。

## 削除（実装済み）

- `systemd/quant-hard-stop-backfill.service`
- `systemd/quant-realtime-metrics.service`
- `systemd/quant-m1scalper-trend-long.service`
- `systemd/quant-scalp-precision-*.service`（まとめて削除）
- `systemd/quant-hedge-balancer.service`
- `systemd/quant-hedge-balancer-exit.service`
- `systemd/quant-trend-reclaim-long.service`
- `systemd/quant-trend-reclaim-long-exit.service`
- `systemd/quant-margin-relief-exit.service`
- `systemd/quant-realtime-metrics.timer`

## 補足

- `scalp_ping_5s` と `scalp_macd_rsi_div` は現状、ENTRY/EXIT を1対1で分離済み。
- `quant-scalp-precision-*` は削除済みで、置換された `quant-scalp-*` サービス群が戦略別で単独起動される。
- `strategy_control` はフラグ配信の母体で、`execution/order_manager.py` の事前チェックで
  `strategy_control.can_enter/can_exit` を参照することで、ENTRY/EXIT 可否を実行時に即時反映。

## V2 追加（完了: 2026-02-14）

- `ops/systemd/quantrabbit.service` は monolithic エントリとして廃止対象へ昇格（本番起動から排除）。
- 本設計では「データ / 制御 / 戦略 / 分析 / 注文 / ポジ管理」を別プロセス境界で扱う。
- `execution/order_manager.py` / `execution/position_manager.py` は service-first ルートを持ち、
  strategy worker は基本的に HTTP で各サービスを経由する運用へ。

## 運用反映（2026-02-14 直近）

- `fx-trader-vm` にて `main` 基点の全再インストールを実施し、`quant-market-data-feed` / `quant-strategy-control` を含む
  V2サービス群を再有効化。`quantrabbit.service` は再起動済み。
- `quant-order-manager.service` / `quant-position-manager.service` はサービス側で再有効化したが、`main` 上に
  `workers/order_manager` / `workers/position_manager` が未収録のため、現時点では起動が `ModuleNotFoundError` で継続リトライ。
- 今回の状態は次のデプロイでワーカー実装を main に反映して解消する必要がある。
- `scripts/install_trading_services.sh` を改善し、`enable --now` の起動失敗でスクリプト全体が止まらないように
  `enable` と `start` を分離。これにより、起動時点で `enabled` 指定されたサービス群は有効化された状態を維持し、
  VM再起動時の自動起動対象から漏れにくくする運用を確立。

## 2026-02-14 組織図更新運用（V2）

- `docs/WORKER_ROLE_MATRIX_V2.md` の「現在の状態」「図」「運用制約」を、V2構成変更時に毎回更新する運用ルールを明文化。
- WORKER関連の変更点は、`WORKER_REFACTOR_LOG` と `WORKER_ROLE_MATRIX_V2` を同一コミットで同期更新する運用を追加。
- `main` 反映後の VM 監査時に、構成図と実サービス状態の齟齬がないかを確認する（監査ログの追記対象）。

## 2026-02-16 VM再投入後整備（V2固定）

- `fx-trader-vm` で再監査し、V2外のレガシー戦略・monolithic系ユニットを停止/無効化しました。対象:
  - `quantrabbit.service`
  - `quant-impulse-retest-s5*`
  - `quant-hard-stop-backfill.service`
  - `quant-margin-relief-exit*`
  - `quant-trend-reclaim-long*`
  - `quant-micro-adaptive-revert*`
  - `quant-scalp-precision-*`（旧系）
  - `quant-realtime-metrics.service/timer`（分析補助タイマーも除外）
- VM上の V2実行群は `quant-market-data-feed` / `quant-strategy-control` / 各ENTRY-EXITペア + `quant-order-manager` / `quant-position-manager` のみ有効稼働を維持。
- `systemctl list-unit-files --state=enabled --all` / `systemctl list-units --state=active` で再確認済み。

### 2026-02-16（追加）V2のENTRY/EXIT責務固定

- 戦略実行の意思決定入力を統一:
  - `scalp_ping_5s`
  - `scalp_m1scalper`
  - `micro_multistrat`
  - `scalp_macd_rsi_div`
  - `scalp_precision`（`scalp_squeeze_pulse_break` / `scalp_tick_imbalance` / `scalp_wick_reversal_*` のラッパー含む）
- 各戦略の `entry_thesis` を拡張し、`entry_probability` と `entry_units_intent` を付与する実装を反映:
  - `workers/scalp_ping_5s/worker.py`
  - `workers/scalp_m1scalper/worker.py`
  - `workers/micro_multistrat/worker.py`
  - `workers/scalp_macd_rsi_div/worker.py`
  - `workers/scalp_precision/worker.py`
- `order_manager` 側の役割を「ガード＋リスク検査」に限定:
  - `quant-strategy-control` の可否フラグ（entry/exit/global）を参照するだけのフローに合わせる
  - 戦略横断の強制的な勝率採点/順位付けや「代替戦略選別」ロジックは追加しない方針を維持。
- `WORKER_ROLE_MATRIX_V2.md` を今回内容に合わせて同一コミットで更新（責務・禁止ルール・実行図の注記）。

### 2026-02-16（追記）ping-5s 配布整合性

- `workers/scalp_ping_5s/worker.py` の `entry_thesis` 生成部に起きていた
  `IndentationError: unexpected indent (line 4253)` を修正。
- 同コミットで `entry_probability` と `entry_units_intent` を付与したロジックは維持しつつ、`entry_thesis` の
  インデントを `WORKER_ROLE_MATRIX_V2.md` の責務定義に準拠する形へ整形。
- `main` (`1b7f6c56`) を VM へ反映し、`quant-scalp-ping-5s.service` は `active (running)` を確認済み。

### 2026-02-16（追記）session_openの意図受け渡しをaddon_live経路へ統一

- `workers/session_open/worker.py`
  - `projection_probability` を `entry_probability` として `order` へ付与。
  - `size_mult` 由来の意図ロットを `entry_units_intent` として `order` へ付与。
- `workers/common/addon_live.py`
  - `order` から `entry_probability` / `entry_units_intent` を抽出し、`entry_thesis` に確実に反映するように統一。
  - `intent` 側の同名値もフォールバックで受ける運用に変更。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` 側の責務文言は、
  `AddonLive` 経路でも `session_open` を含む各戦略で意図値を `entry_thesis` へ保持する運用へ揃えた。

### 2026-02-16（追記）runtime env 参照の `ops/env/quant-v2-runtime.env` へ移行

- `systemd/*.service` の `EnvironmentFile` を `ops/env/quant-v2-runtime.env` に統一。
- `quant-v2-runtime.env` へ V2に必要なキーのみを収束（OANDA, V2ガード制御, order/position service, pattern/brain/forecast gate, tuner）。
- scalp系調整系スクリプト（`vm_apply_scalp_ping_5s_*`）の環境適用先を
  `ops/env/scalp_ping_5s.env` 系へ移行。
- `startup_script.sh` と `scripts/deploy_via_metadata.sh`/`scripts/vm_apply_entry_precision_hardening.sh` で
  legacy 環境ファイル依存を撤去し、`ops/env/quant-v2-runtime.env` をデフォルト注入先に変更。
- 併せて AGENTS/VM/GCP/監査ドキュメントの監査対象コマンドを新環境ファイル参照へ更新。
