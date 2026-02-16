# ワーカー再編の確定ログ（2026-02-13）

## 方針（最終確定）

- 各戦略は `ENTRY/EXIT` を1対1で持つ。
- `precision` 系のサービス名は廃止し、サービス名は `quant-scalp-*` へ切り分ける。
- `quant-hard-stop-backfill` / `quant-realtime-metrics` は削除対象。
- データ供給は `quant-market-data-feed`、制御配信は `quant-strategy-control` に分離。
- 補助的運用ワーカーは本体管理マップから除外。

### 2026-02-16（追記）scalp_ping_5s_b ラッパー env_prefix 混在回避を根本修正

- `workers/scalp_ping_5s_b/worker.py` の `_apply_alt_env()` を修正し、`SCALP_PING_5S_*` の掃除時に
  `SCALP_PING_5S_B_*` を残してから base-prefix へ再投入する順序へ変更。
- これにより、`SCALP_PING_5S_B_ENABLED` や `SCALP_PING_5S_B_*` の必須値が誤って削除され、
  子プロセス `workers.scalp_ping_5s.worker` が意図せず `SCALP_PING_5S_ENABLED=0` になる問題を解消。
- `SCALP_PING_5S_B` 側の設定を `SCALP_PING_5S_*` へ正規射影し、`ENV_PREFIX` を `SCALP_PING_5S_B` へ固定した
  上で `enabled/strategy_tag/log_prefix` も一貫注入する経路を明文化。
- 対象は B版起動時の env 混在・取り残し（`no_signal` の母集団原因に直結する無効化）を避けるための
  根本対応として記録。

### 2026-02-16（追記）env_prefix 共通化フェイルセーフ（entry→order_manager）

- `execution/strategy_entry.py` の `env_prefix` 取りまとめを強化し、`entry_thesis` / `meta` / `strategy_tag` から
  正規化 (`strip` + 大文字化) したうえで一貫して注入するよう変更。
- `execution/order_manager.py` 側でも `env_prefix` を同様に正規化し、`strategy_tag` からのフォールバック推定を追加。
  これにより、`SCALP_PING_5S_B` と `SCALP_PING_5S` が混在した場合の
  ガード参照ずれを抑え、`no_signal` の母集団に寄与しうる「意図外の設定混線」を低減。

### 2026-02-16（追記）scalp_precision 実行委譲の完全停止（戦略別独立実行）

- `workers/scalp_tick_imbalance`, `workers/scalp_squeeze_pulse_break`, `workers/scalp_wick_reversal_blend`, `workers/scalp_wick_reversal_pro` の `entry`/`exit` を `workers.scalp_precision` の子プロセス実行から切り離し、各パッケージ内のローカル `worker.py` / `exit_worker.py` を直接実行する構成へ変更。
- `workers/scalp_macd_rsi_div` と `workers/scalp_ping_5s` / `workers/scalp_ping_5s_b` の `exit_worker.py` も `python -m workers.scalp_precision.exit_worker` 委譲を停止し、同一プロセス内で `exit` 判定を完結。
- 戦略独立実行のため、上記5戦略に対応する `config.py` / `common.py`（該当戦略）をローカル配備し、戦略起動時の実行経路から `scalp_precision` 参照を外した。

### 2026-02-16（追記）scalp_precision からの最終依存排除

- `workers/scalp_tick_imbalance/`, `workers/scalp_squeeze_pulse_break/`, `workers/scalp_wick_reversal_blend/`, `workers/scalp_wick_reversal_pro/` の
  `strategy_tag` fallback を `"scalp_precision"` から各戦略名へ変更し、`client_order_id`/`market_order` の strategy tag 注入でも戦略別値を使用。
- `workers/scalp_macd_rsi_div/`, `workers/scalp_ping_5s/`, `workers/scalp_ping_5s_b/` の `exit_worker` も戦略別エントリ名へ変更し、同名 main 呼び出しに揃えた。

### 2026-02-16（追記）`scalp_precision` 依存 wrapper の切断（戦略別ワーカー単位）

- `workers/scalp_tick_imbalance` / `scalp_squeeze_pulse_break` / `scalp_wick_reversal_*` / `scalp_macd_rsi_div` / `scalp_ping_5s_b` の
  `entry` / `exit_worker` から `workers.scalp_precision` のインポート依存を除去し、各戦略ワーカー側を subprocess 起動に切替え。
- `python -m workers.<strategy>.worker` / `python -m workers.<strategy>.exit_worker` を直接起動し、
  ワーカー本体の起動責務は戦略別ランナー（entry/exit）として完結させた。
- テスト `tests/test_spread_ok_tick_cache_fallback.py` の `scalp_precision.common` 参照を削除し、共有 `spread_ok` の依存を排除。

### 2026-02-16（追記）起動監査ログの統一

- 各戦略 `entry/exit` の起動直後に `Application started!` を明示ログ化。
  対象は `scalp_tick_imbalance` / `scalp_squeeze_pulse_break` / `scalp_wick_reversal_blend` / `scalp_wick_reversal_pro` / `scalp_macd_rsi_div` / `scalp_ping_5s` / `scalp_ping_5s_b` の `entry`/`exit`。
- VM 上の反映確認を、`journalctl` の `Application started!` 検索に統一し、起動実在性を監査しやすくした。

### 2026-02-16（追記）strategy_entry の意図注入を完全化

- `execution/strategy_entry.py` に `entry_thesis` / `meta` の `env_prefix` 双方向注入を追加し、`coordinate_entry_intent` まで `env_prefix` を確実反映。
- `workers/scalp_ping_5s/worker.py` の `no_signal` 理由正規化を拡張し、`insufficient_signal_rows_fallback` を判別可能にして監査集計の粒度を固定化。

### 2026-02-16（追記）`PositionManager.close()` の共有DB保護

- `execution/position_manager.py` の `PositionManager.close()` に共有サービスモード保護を追加。
- `POSITION_MANAGER_SERVICE_ENABLED=1` かつ `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0` の運用では、クライアント側からの
  `close()` 呼び出しをリモート `/position/close` に転送せず、共有 `trades.db` を意図せず閉じないガードを実装。
- 直近 VM ログで観測される大量の `Cannot operate on a closed database` は、この close 過剰呼び出し由来の再発を抑止する対象。
- `workers/position_manager/worker.py` は `PositionManager` をローカルモードで起動するため、サービス停止時の
  正規クローズ経路は維持。

### 2026-02-16（追記）`entry_probability` の高確率側サイズ増強を全戦略共通化

- `execution/order_manager.py` の `order_manager` 共通プリフライトで `entry_probability` スケーリングを拡張し、
  `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY` 以上の高確率時のみ、サイズを `>1` へ拡張可能に変更。
- 同時に、`ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE` を追加して上振れ上限を制御可能化。
- 低確率側（`<= reject_under`）の拒否/縮小は既存ルールを維持し、縮小判断の主軸は従来どおり `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE`
  / `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` に寄せたまま。
- 追加運用キー（例値）:
  - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY=0.80`
  - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE=1.25`
- 変更は `order_manager` 経由の全戦略に共通適用され、戦略側ロジック選別は現行どおり保持。

### 2026-02-16（追記）`install_trading_services` でログ自動軽量化を標準化

- `scripts/install_trading_services.sh` を更新し、`--all` / `--units` 指定に関わらず `quant-cleanup-qr-logs.service` と
  `quant-cleanup-qr-logs.timer` が常設でインストールされるようにしました。
- `systemd/cleanup-qr-logs.timer` は 1日2回（07:30/19:30）起動で、実運用ノードでもディスククリーンアップを自動化する前提へ統一。

### 2026-02-16（追記）非5秒エントリー再開のための実行条件調整

- `market_data/tick_fetcher.py` の `callback` 発火経路を `_dispatch_tick_callback` に一本化し、`tick_fetcher reconnect` 側の
  `NoneType can't be used in 'await' expression` ループ再接続原因の対処を反映。
- `ops/env/quant-micro-multi.env` に `MICRO_MULTI_ENABLED=1` を追加し、`quant-micro-multi` の ENTRY 側を起動状態に寄せる。
- `ops/env/quant-m1scalper.env` の `M1SCALP_ALLOWED_REGIMES` を `trend` 固定から `trend,range,mixed` に変更し、市況レジーム偏在時の過度な阻害を回避。

### 2026-02-22（追記）M1Scalper 戦略ラベルの復元ルールを明文化

- `execution/position_manager.py` の戦略タグ正規化に、`client_order_id` の
  `m1scalpe*` 系文字列を `M1Scalper-M1` に復元するエントリを追加し、`strategy` / `strategy_tag` の表示整合を向上。
- 同ファイルに `M1Scalper` 系の戦略名を戦略-ポケット推定 (`_STRATEGY_POCKET_MAP`) へ追加。
- `scripts/backfill_strategy_tags.py` の `client_id` 由来補完に、`qr-<ts>-scalp-m1scalpe*` 系を `M1Scalper-M1` として扱うロジックを追加。

## 補足（戦略判断責務の明確化）

- **方針確定**: 各戦略ワーカーは「ENTRY/EXIT判定の脳」を保持し、ロジックの主判断は各ワーカー固有で行う。
- `quant-strategy-control` は「最終実行可否」を左右する制御入力を配信するのみ（`entry_enabled` / `exit_enabled` / `global_lock` / メモ）。
- したがって、`strategy-control` が戦略ロジックを代行しているわけではなく、**各戦略の意思決定を中断/再開するガードレイヤー**として機能する。
- UI の戦略ON/OFFや緊急ロックはこのガードレイヤーを介して、並行中の戦略群へ即時反映する。

### 2026-02-16（追記）5秒スキャ（scalp_ping_5s）運用通過率の即時改善

- `ops/env/scalp_ping_5s.env` を更新し、5秒戦略向け `entry_probability` スケーリング後の
  最小ロット拒否を回避するため、戦略別最小ロットを引き下げた。
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S=20`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_LIVE=20`
- 同ファイルで `SCALP_PING_5S_PERF_GUARD_*` を追加し、5秒戦略単体で `perf_guard` の実運用しきい値を事実上緩和。
  - `PERF_GUARD_MIN_TRADES=99999` / `PF_MIN=0.0` / `WIN_MIN=0.0`
  - `PERF_GUARD_FAILFAST_MIN_TRADES=99999` / `FAILFAST_PF=0.0` / `FAILFAST_WIN=0.0`
- 変更対象は 5秒エントリー運用のみに限定（`SCALP_PING_5S_*` プレフィックス）。
- 反映後、VM 上で `orders.db` の `strategy=scalp_ping_5s_live` の `entry_probability_reject` と `perf_block`
  比率低下、`filled`/`submit_attempt` 増加を優先監視する。

### 2026-02-16（追記）5秒スキャ no_signal 原因可視化の追加

- `workers/scalp_ping_5s/worker.py` の `_build_tick_signal()` を
  `reason` 付き返却に変更し、`entry-skip` の `no_signal` に
  `insufficient_*`/`stale_tick`/`momentum_tail_failed` などの原因を
  `detail` として残す監査を追加。
- `ops/env/scalp_ping_5s.env` で 5秒戦略の入口閾値を追加緩和。
  - `SCALP_PING_5S_SIGNAL_WINDOW_SEC=0.35`
  - `SCALP_PING_5S_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_MOMENTUM_SPREAD_MULT=0.12`
  - `SCALP_PING_5S_ENTRY_BID_ASK_EDGE_PIPS=0.0`
- 本修正は 5秒戦略のみ対象。`SCALP_PING_5S` 系サービスの入場通過率回復を優先し、
  `entry_probability_reject` が減り `filled` が増えるかを次回監視で確認する。

### 2026-02-16（追記）env_prefix 混在対策と no_signal 算出粒度の追加

- `execution/order_manager.py` の `env_prefix` 解決を `entry_thesis` 優先へ固定し、
  `meta` 側と値が混在した場合も戦略側の意図が上書きされないようにした。
  - 混在時は `pocket` / `strategy` / `meta` / `entry` の値を debug ログへ出力し、原因追跡を可能化。
- `workers/scalp_ping_5s/worker.py` の `no_signal` 原因集計で
  `insufficient_mid_rows` / `insufficient_rows` / `invalid_latest_epoch` / `stale_tick`
  を追加サブ分類化し、`insufficient_signal_rows` 系と同様に監査しやすくした。

### 2026-02-16（追記）M1スキャ (`scalp_m1scalper`) の最小ロットソフトガードを調整

- `workers/scalp_m1scalper/worker.py` の `_resolve_min_units_guard` にて、
  `min_units_soft_raise` 時に `config.MIN_UNITS` へサイズを引き上げる処理を廃止し、
  ソフト判定後も `abs_units` を維持して通過させるよう変更。
- `ops/env/quant-m1scalper.env` に `M1SCALP_MIN_UNITS_SOFT_FAIL_RATIO=0.05` を追加し、
  M1スキャの小型シグナル（例: `-57`）でも過度な拒否になりにくい状態へ寄せた。

### 2026-02-16（追記）M1スキャ戦略の実行実態観測結果に基づく最小ロット再調整

- `OPS` 実行環境（`quant-v2-runtime.env`）で `ORDER_MIN_UNITS_STRATEGY_M1SCALPER_M1=700` を
  `ORDER_MIN_UNITS_STRATEGY_M1SCALPER_M1=300` に更新し、同時に `quant-m1scalper.env` に
  `M1SCALP_MIN_UNITS=350` と `M1SCALP_MIN_UNITS_SOFT_FAIL_RATIO=0.62` を明示。
- 本日ログ上の `OPEN_SKIP` で、`entry_probability_below_min_units` が実質的な主因（約90%超）であったため、
  `ORDER_MIN_UNITS` と worker 側 min_units threshold を引き下げ、最低サイズ起因の取り逃しを抑える方向へ変更。
- 適用後は同日再集計で `OPEN_SKIP` 内訳（`entry_probability_below_min_units`）の件数が低下するかを
  監視対象に設定。

### 2026-02-16（追記）5秒B no_signal 可観測性の精緻化

- `workers/scalp_ping_5s/worker.py`
  - `entry-skip` ログの `no_signal` 集計を `no_signal:<detail_head>` まで分割してカウントし、
    `insufficient_*` 系と `revert_*` 系の内訳を見える化。
  - `revert_long/revert_short` を含む `no_signal` に対して方向推定を行い、サイド別内訳の精度を向上。
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0` を明示し、5秒B起動時にも
    lookaheadゲートが確実に `off` になる状態を固定。

### 2026-02-16（追記）5秒B閾値を5秒A運用帯へ寄せて通過率を改善

- `ops/env/scalp_ping_5s_b.env` を、`scalp_ping_5s`（A）側で稼働している
  エントリー通過優先値に寄せる形で緩和。
  - `SCALP_PING_5S_B_MIN_TICKS=4`
  - `SCALP_PING_5S_B_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE=0.50`
  - `SCALP_PING_5S_B_MIN_TICK_RATE=0.50`
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_SEC=0.35`
  - `SCALP_PING_5S_B_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_B_MOMENTUM_SPREAD_MULT=0.12`
  - `SCALP_PING_5S_B_ENTRY_BID_ASK_EDGE_PIPS=0.0`
- 目的: B版の `no_signal`（特に `insufficient_signal_rows`）を減らし、`scalp_ping_5s_b_live` の実入場頻度を
  主要な5秒戦略レベルまで引き上げる。

### 2026-02-16（追記）5秒Bの再取り込み改善（`insufficient_signal_rows` / `revert_not_enabled` 対策）

- `ops/env/scalp_ping_5s_b.env` に追加・更新:
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_SEC=0.55`
  - `SCALP_PING_5S_B_MIN_TICK_RATE=0.70`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE=0.45`
  - `SCALP_PING_5S_B_REVERT_MIN_TICK_RATE=1.2`
- `revert_not_enabled` は実質的な `no_signal` 側ボトルネックだったため、
  `revert` 判定閾値とシグナル窓を緩和し、同時に短側閾値もわずかに緩めることで
  `revert_not_enabled` 集計を抑える方針に変更。

### 2026-02-16（追記）`no_signal` のフォールバックを窓不足向けに限定

- `workers/scalp_ping_5s/worker.py`
  - `insufficient_signal_rows` 時のみ、`SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` で指定した上限内で
    追加窓を1回だけ再評価するように変更（不足時は `insufficient_signal_rows` の理由へ
    `window/fallback` を記録）。
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_SEC=2.00` を追加。
- `TickSignal` に `signal_window_sec` を付与し、実際に使ったシグナル窓を監査しやすくした。

### 2026-02-24（追記）5秒B no_signal 監査の実装事故修正

- `workers/scalp_ping_5s/config.py` に `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` を
  読み込む `SIGNAL_WINDOW_FALLBACK_SEC` 定数を追加し、VM起動時の
  `AttributeError: module ... has no attribute 'SIGNAL_WINDOW_FALLBACK_SEC'` を解消。
- `no_signal` のフォールバックは従来どおり `insufficient_signal_rows` 時のみ行い、データ不足以外の
  退避導線は従来ロジックを維持して副作用増加を回避。

### 2026-02-16（追記）fallback窓展開を「明示設定のみ」に固定

- `workers/scalp_ping_5s/worker.py`
  - `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` が `0` の場合、`WINDOW_SEC` へ無条件で拡張しないよう制御を変更。
  - fallback が実施されなかった/失敗した場合も `insufficient_signal_rows` 理由に
    `fallback_used=attempted` を残して、監査時に「窓を広げたが不足継続」を判別できるようにした。

### 2026-02-16（追記）5秒Bの意図設定プレフィックスを明示化

- `workers/scalp_ping_5s_b/worker.py` の環境コピー処理で
  `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S_B` を明示的に上書きするよう追加。
- `ops/env/scalp_ping_5s.env` / `config/env.example.toml` に `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S` を明示し、
  A/Bのプレフィックス判定を設定側で明確化。
- `ops/env/scalp_ping_5s_b.env` へ
  `SCALP_PING_5S_B_ENV_PREFIX=SCALP_PING_5S_B` を追加。
- B版の `PERF_GUARD` / `entry_thesis` 参照で、A/B混在時に B 固有キーを誤適用しないようにした。
- `workers/scalp_ping_5s/config.py` で `ENV_PREFIX` を `SCALP_PING_5S_ENV_PREFIX` 参照に切替し、
  B版 `SCALP_PING_5S_B_ENV_PREFIX` 経由で `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S_B` が有効化されるよう明示化。
- 追跡指標は `perf_block` / `entry_probability_reject` の内訳で次回検証し、必要であれば
  `SCALP_PING_5S_B_PERF_GUARD_*` の調整へ接続する。

### 2026-02-16（追記）5秒B no_signal 原因粒度の崩れを最小修正

- `workers/scalp_ping_5s/worker.py`
  - `insufficient_signal_rows` の詳細文字列を `detail_parts` で組み立てる方式へ変更し、
    `fallback_min_rate_window` 付与時に `fallback_used` が欠落していた組版バグを是正。
  - これにより `entry-skip` 集計で `insufficient_signal_rows_fallback` /
    `insufficient_signal_rows_fallback_exhausted` の振り分けが崩れないようにした。

### 2026-02-24（追記）quant-scalp-ping-5s-b の env_prefix 混在排除

- `workers/scalp_ping_5s_b/worker.py` を追加修正し、`SCALP_PING_5S_B_*` を
  `SCALP_PING_5S_*` に写像する前に、実行時環境内の既存 `SCALP_PING_5S_*` を一度削除してから再注入するように変更。
- B サービスは `scalp_ping_5s.env` と `scalp_ping_5s_b.env` を併読するため、A 側残存値で上書きされる混在リスクを除去。
- 追加監査として、`env_prefix` 起因が疑われる `revert_not_enabled` / `momentum_tail_failed` が
  再発しないかを `no_signal` ログで追跡する。
- 同時に、`systemd/quant-scalp-ping-5s-b.service` から `scalp_ping_5s.env` の同読を外し、Bサービスを
  `scalp_ping_5s_b.env` と本体ランタイム env のみで起動する構成へ寄せた。

### 2026-02-16（追記）PositionManager 停止再開耐障害化

- `execution/position_manager.py`:
  - `trades.db` 接続再確立時の再試行回数/待ち時間を追加し、連続再接続失敗時はインターバルで抑止するよう調整。
  - `close()` で接続をクローズ後 `self.con = None` へセットし、閉じた接続の再利用を防止。
  - closed DB 例外時の再接続制御を `_ensure_connection_open` / `_commit_with_retry` / `_executemany_with_retry` で統一。
- `workers/position_manager/worker.py`:
  - 起動時 `PositionManager` 初期化をリトライ化。失敗時は明示エラー付きで起動継続し、サービス再起動ループを回避。

### 2026-02-16（追記）PositionManager service 呼び出しタイムアウトの整合化

- `execution/position_manager.py` は `POSITION_MANAGER_HTTP_TIMEOUT` と
  `POSITION_MANAGER_SERVICE_TIMEOUT` のギャップを起動時に検査し、サービス側タイムアウトが短すぎる構成を
  自動補正するロジックを前提として、運用環境設定の同期を追加。
- `ops/env/quant-v2-runtime.env` を次の値へ更新:
  - `POSITION_MANAGER_SERVICE_TIMEOUT=9.0`
  - `POSITION_MANAGER_HTTP_TIMEOUT=8.0`
- 目的: service経路で `sync_trades/get_open_positions` のタイムアウト再試行を抑制し、
  直近データ取得の安定性を上げる。

### 2026-02-16（追記）5秒Bの `client_order_id` と strategy_tag 復元を固定

- `workers/scalp_ping_5s/worker.py` の `_client_order_id()` は `config.STRATEGY_TAG` を 24 文字まで保持し、
  `qr-<ts>-<strategy>-<side><digest>` を生成するよう変更。これにより `scalp_ping_5s` と
  `scalp_ping_5s_b` の識別崩れ（先頭 12 文字トランケート）を解消。
- `execution/position_manager.py` の `_infer_strategy_tag()` は `qr-<ts>-<strategy>-...` 系の
  戦略タグ推定を強化し、`qr-<timestamp>` を strategy_tag と誤認する既知不具合を抑止。
- 目的は 5 秒B 側の `open_trades` 管理漏れを潰し、取り残し残高の監視・追跡を確実化すること。

### 2026-02-16（追記）scalp_precision と scalp_ping_5s の `position_manager` 失敗時ガード

- `workers/scalp_precision/exit_worker.py`
  - `PositionManager.get_open_positions()` を非同期タイムアウト付きの安全取得へ変更し、`position_manager_error` / `position_manager_timeout` 時は同サイクルをスキップ。
  - `_filter_trades` は `entry_thesis` 欠損時に `client_id/client_order_id` ベースの戦略タグ推定を追加し、タグ欠落による戦略漏れを抑制。
  - 取得失敗の連続警報を間引き。
- `workers/scalp_ping_5s/worker.py`
  - `_safe_get_open_positions()` 後、エラー時に `continue` で当該ループを明示保留に変更し、空 `pocket_info` での判定を防止。
  - `forced_exit`/`profit_bank` 再取得でも同様にスキップへ統一。
- 運用意図: フォールバックは「空データ埋め込み」ではなく、取引判断の実行抑止（Fail-safe）に限定。

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

### 2026-02-14（追記）market_order 入口の entry_thesis 補完を追加

- `execution/order_manager.py` に `market_order()` 入口ガード `_ensure_entry_intent_payload()` を追加。
- 戦略側 `entry_thesis` が欠けるケースに対し、`entry_units_intent` と `entry_probability` を実行時に補完。
- 併せて `strategy_tag` が未入力時は `client_order_id` から補完して `entry_thesis` に反映するようにし、V2各戦略の `market_order` 呼び出し互換性を維持。
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

### 2026-02-18（追記）黒板協調判定要件を仕様化

- `execution/strategy_entry.py` は `/order/coordinate_entry_intent` を経由して
  `execution/order_manager.py` の `entry_intent_board` 判定と整合した運用へ固定。
- 判定の固定要件を明文化:
  - `own_score = abs(raw_units) * normalized(entry_probability)`  
  - `dominance = opposite_score / max(own_score,1.0)` を算出し監査記録するが、方向意図は `raw_units` を基本維持して通す
  - 最終 `abs(final_units) < min_units_for_strategy(strategy_tag, pocket)` なら reject
  - `reason` と `status` は `entry_intent_board` に永続化し、`final_units=0` は `order_manager` 経路に流さない運用をログ追跡対象化。
  - `status`: `intent_accepted/intent_scaled/intent_rejected`
  - `reason`: `scale_to_zero/below_min_units_after_scale/coordination_load_error` 等
- `opposite_domination` は廃止し、逆方向優勢でも方向意図を否定しない運用へ更新。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` を同一ブランチ変更で更新し、監査対象文言を同期済みにする運用へ反映。

### 2026-02-19（追記）戦略ENTRYで技術判定コンテキストを付与

- `execution/strategy_entry.py` の `market_order` / `limit_order` に、
  `analysis.technique_engine.evaluate_entry_techniques` を使って
  `N波`, `フィボ`, `ローソク` を含む入場技術スコア算出を追加。
- エントリー価格は `entry_thesis` の価格情報優先、未提供時は直近ティックから補完し算出。
- 算出結果は `entry_thesis["technical_context"]` に保存され、各戦略からの
  エントリー意図として `ENTRY/EXIT` の判断（ロギング/追跡含む）に利用可能。
- 機能スイッチは `ENTRY_TECH_CONTEXT_ENABLED`（未設定時 true）とし、必要時は
  `execution/strategy_entry.py` の既定動作から外せるようにした。

### 2026-02-15（追記）strategy_entry の戦略キー正規化経路を関数化

- `execution/strategy_entry.py` の `_NORMALIZED_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を
  集約生成する処理を `_normalize_strategy_requirements()` に集約し、`_strategy_key` 参照順序依存を
  回避する形に変更。
- これにより、`quant-strategy-entry` 起動時の `NameError` リスク（`_strategy_key` 未定義）を回避しやすくした。

### 2026-02-14（追記）戦略ENTRYに戦略別技術断面を常設

- `execution/strategy_entry.py` の `_inject_entry_technical_context()` を拡張し、
  `entry_thesis["technical_context"]` に `evaluate_entry_techniques`（N波/フィボ/ローソク）結果だけでなく、
  `entry_price` の有無に関わらず `D1/H4/H1/M5/M1` の技術指標スナップショットを保存するようにした。
- 主要保存項目には `ma10/ma20/ema12/ema20/ema24/rsi/atr/atr_pips/adx/bbw/macd/...` を含む
  `indicators` が入り、`ENTRY_TECH_DEFAULT_TFS` で参照TFの優先順を上書き可能。
- 戦略側が必要とする指標を限定したい場合、`technical_context_tfs` / `technical_context_fields` を
  `entry_thesis` に付与して保存範囲を絞り込める仕様を同時に導入。
- 既存の `ENTRY_TECH_CONTEXT_ENABLED` スイッチは維持し、無効時は `entry_thesis` への技術注入を抑制。

### 2026-02-20（追記）戦略別の必要データ契約を明文化

- `execution/strategy_entry.py` で、戦略が必要とする技術入力は `entry_thesis` を通じて受領する運用を明示。
  - `technical_context_tfs`: 収集する指標TF順（例: `["H1", "M5", "M1"]`）
  - `technical_context_fields`: `indicators` に保存するフィールド名（未指定は全件）
  - `technical_context_ticks`: エントリー時に参照する最新ティック名（例: `latest_bid` / `latest_ask` / `spread_pips`）
  - `technical_context_candle_counts`: `{"H1": 120, "M5": 80}` のような TF 別ローソク本数指定
- `entry_thesis["technical_context"]` には、上記要求を反映した `indicators`（TF毎）と `ticks`、要求内容を保存し、技術判定結果（`result`）とセットで持つ。
- `analysis/technique_engine.evaluate_entry_techniques` は `technical_context_tfs` / `technical_context_candle_counts` を解釈して、
  TF 及びローソク取得本数を戦略側要求へ寄せる処理を追加（`common` 既定は維持）。
  - `ENTRY_TECH_CONTEXT_GATE_MODE` は `off/soft/hard` を明示し、`hard` 時は `technical_context.result.allowed=False` を最終拒否条件に反映する運用を確認（当時の共通評価仕様）。
  - 同時に `session_open` 経路（`addon_live -> strategy_entry`）向け契約も追加し、`technical_context_tfs`/`fields`/`ticks`/`candle_counts` を明示。
  （当時の仕様では N波/フィボ/ローソク必須寄りだったため、現在は戦略側の `tech_policy` 明示に委譲）

### 2026-02-21（追記）技術判定の共通計算を strategy_entry から分離

- `execution/strategy_entry.py` で共通技術判定を実行するフローをデフォルト停止し、`ENTRY_TECH_CONTEXT_COMMON_EVAL=0`（既定）時は
  `analysis.technique_engine.evaluate_entry_techniques` を呼ばないようにした。
- `entry_thesis["technical_context"]` への保存は維持し、`indicators`（TF別）・`ticks`・要求パラメータは戦略のローカル計算入力として保全。
- `technical_context["result"]` は共通側未評価時は `allowed=True` の監査用結果を持たせ、評価結果不在での注文拒否を発生させないようにする。
- サイズ決定・方向決定は引き続き戦略側（各strategyワーカー）に委譲。`strategy_entry` 側では共通スコアに基づく拒否/縮小は行わない。
- AGENTS/運用側の方針に合わせ、各戦略内で N波/フィボ/ローソク判定を含む必要なテクニカルを計算して `entry_thesis`/`technical_context` の形で整合させる前提へ一本化。

### 2026-02-15（追記）共通評価設定キーを環境定義から整理

- `config/env.example.toml` から `ENTRY_TECH_CONTEXT_GATE_MODE` / `ENTRY_TECH_CONTEXT_COMMON_EVAL` / `ENTRY_TECH_CONTEXT_APPLY_SIZE_MULT` を削除し、
  `ENTRY_TECH_CONTEXT_ENABLED` のみ残して `technical_context` 注入（保存）を明文化。
- `WORKER_ROLE_MATRIX_V2.md` の技術要件章を更新し、`strategy_entry` は各戦略のローカル判断を代替しない構成へ統一。
- `strategy_entry` は `technical_context.result` を上書きしない（保存専有）運用を前提に文言を整合。

### 2026-02-22（追記）N波/フィボ/ローソクの必須条件を共通契約から外す

- 現行運用として、`strategy_entry` 側の共通注入は `technical_context` 取得範囲までに限定し、
  `require_fib` / `require_nwave` / `require_candle` の既定強制を廃止。
- `execution/strategy_entry.py` の契約辞書から `tech_policy` の既定付与を事実上無効化し、各戦略ワーカー側の
  `evaluate_entry_techniques(..., tech_policy=...)` 呼び出しで戦略個別要件を持つ運用へ戻す。
- これにより、戦略ごとの意図（許容するテクニカル条件）が壊れず維持される設計に再整合。

### 2026-02-22（追記）tech_fusion を mode 別 tech_policy 明示へ収束

- `workers/tech_fusion/worker.py` の `tech_policy` を `range` / `trend` モードで明示分岐化。
- `range` モードは `require_nwave=True` を明示し、`trend` モードは `require_*` を `False` のまま
  戦略ローカルで明示定義する形へ統一。
- `technical_context_tfs/fields/ticks/candle_counts` は現行定義を保持しつつ、`evaluate_entry_techniques` への
  要件注入を戦略側で完結する形へ更新。  
  同時に監査観点の `tech_fusion` `require_*` 未最適化状態を解消。

### 2026-02-15（追記）戦略内テクニカル評価の統一ルートを拡張

- `workers/hedge_balancer/worker.py` と `workers/manual_swing/worker.py` を
  `evaluate_entry_techniques` のローカル呼び出し＋`technical_context_*` 明示へ統一し、ローカル判定後に
  `entry_probability` / `entry_units_intent` を `entry_thesis` に反映する形へ拡張。
- `workers/macro_tech_fusion/worker.py` / `workers/micro_pullback_fib/worker.py` /
  `workers/range_compression_break/worker.py` / `workers/scalp_reversal_nwave/worker.py` について、
  `tech_tfs` を維持しつつ `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts`
  を明示化し、監査観点での入力要求の一貫性を揃えた。
- `docs/strategy_entry_technical_context_audit_2026_02_15.md` を更新し、現時点の集計を `evaluate_entry_techniques` 実装 37、
  `technical_context` 一式 37 に反映。

### 2026-02-24（追記）戦略別分析係ワーカーを追加

- `analysis/strategy_feedback_worker.py` を新規追加し、`quant-strategy-feedback.service` / `quant-strategy-feedback.timer` で
  定期実行する設計を追加。
- ワーカーは以下を自動反映し、戦略追加・停止・再開へ追従する運用を実装:
  - `systemd` からの戦略ワーカー検出（`quant-*.service`）
  - `workers.common.strategy_control` の有効状態
  - `logs/trades.db` の直近実績
- 主要出力先は `logs/strategy_feedback.json`（既存 `analysis.strategy_feedback.current_advice` の入力と互換）へ更新。
- 事故回避として、停止中戦略については最近のクローズ履歴なしなら出力抑止する `keep_inactive` 条件を導入。
- 追加改良: エントリー専用ワーカーが稼働中かを優先基準にし、`EXIT` ワーカーのみ残存するケースでの誤適用を防止。
- `ops/env/quant-strategy-feedback.env` を追加し、`STRATEGY_FEEDBACK_*` の運用キー（lookback/min_trades/保存先/探索範囲）を明文化。
- `docs/WORKER_REFACTOR_LOG.md` と `docs/WORKER_ROLE_MATRIX_V2.md` へ同時反映（監査トレースを維持）。

### 2026-02-16（追記）5秒スキャの最小ロット下限制御を運用値へ追従

- `workers/scalp_ping_5s/config.py` の `MIN_UNITS` が `max(100, ...)` で固定されていたため、`SCALP_PING_5S_MIN_UNITS=50` が設定されても
  実戦ロジックでは `units_below_min` が発生してエントリーを通過しづらい不整合があった。
- `SCALP_PING_5S_MIN_UNITS` の下限固定を `max(1, ...)` に変更し、環境変数ベースで 50 以上での運用実験を可能にした。
- 併せて、`ORDER_MIN_UNITS_SCALP_FAST=50` との整合を前提に、5秒スキャの再現監査で `units_below_min` の抑制を優先監視項目に加える運用を明示した。

### 2026-02-15（追記）35戦略の `evaluate_entry_techniques` 組み込みを構文修正

- 42戦略監査の残作業として一括追加した新規 `evaluate_entry_techniques` 呼び出しで、
  一部 `market_order/limit_order(` の引数開始直後にブロック混入が発生し、構文エラーを起こしていた箇所を補正。
- `entry_thesis_ctx` 前処理を注文呼び出しの外側へ移動し、`market_order/limit_order` 呼び出しの構文を復元。
- 補正対象は 30 戦略（`workers/*/worker.py` のうち `market_order/limit_order` 直呼び + `entry_thesis_ctx = None` 直下パターン）。
- 対象ファイルの `docs/strategy_entry_technical_context_audit_2026_02_15.md` を再生成し、`evaluate_entry_techniques` と
  `technical_context_*`/`tech_policy` 要件の監査ビューを最新化。

### 2026-02-15（追記）戦略別 technical_context 要件監査を実施

- `42` 戦略の監査対象（ユーザー指定）について、`evaluate_entry_techniques` と `technical_context_*` キーの実装有無を再集計。
- 監査結果は `docs/strategy_entry_technical_context_audit_2026_02_15.md` に保存。
- まとめ:
  - `evaluate_entry_techniques` 未実装かつ `technical_context` 明示なしが多数（ユーザー指定42件の主要対象）
  - `market_order` 非検出の wrapper/非entry系を除くと、実装未完了対象が 31 件
  - `tech_policy` の `require_*` 監査対象 5戦略について、`require_*` 値の確認を同時実施済み（`tech_fusion` は `range`/`trend` で分岐定義、`range` は `False/F/F/F`、`trend` は `False/F/F/F`）
- 次アクションとして、未実装戦略へ `technical_context_*` 要求明示とローカルテック評価の呼び出し導線を順次付与する運用を開始。

### 2026-02-15（追記）戦略側監査対象を戦略ワーカーに限定

- `order_manager` は注文 API 経路のインフラ層であり、戦略ローカル判断の監査対象から除外。
- `strategy` 側の条件（`workers/*/worker.py`）として再集計し、`market_order/limit_order` 直呼び 37 件が
  `evaluate_entry_techniques` と `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts` を
  すべて明示する状態を確認。
- ここで未完了だった `hedge_balancer` / `manual_swing` / `macro_tech_fusion` / `micro_pullback_fib` /
  `range_compression_break` / `scalp_reversal_nwave` を含む戦略群を最新定義に揃える対応を完了。

- 追記サマリ（戦略側監査完了）:
  - `workers/*/worker.py` 中の戦略ワーカー `market_order/limit_order` 直呼び: 37
  - `evaluate_entry_techniques` 未呼び: 0
  - `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts` 未明示: 0
  - `order_manager` は監査外（インフラ/API 入口）

### 2026-02-15（追記）technical_context の自動契約注入を明示要件時に限定

- `execution/strategy_entry.py` に `ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS` を追加（既定 `false`）。
- `strategy_entry` は、戦略タグ由来の自動補完ではなく、`technical_context_*` を戦略側で明示している場合のみ
  `technical_context` の取得・注入を実施する方針へ変更（共通で全戦略へ前提を押し付けない）。
- `technical_context_tfs` / `technical_context_fields` / `technical_context_ticks` / `technical_context_candle_counts` の
  明示がない戦略は、既定値フォールバックでの自動注入を行わない。
- `workers/tech_fusion/worker.py` について、`evaluate_entry_techniques` 呼び出しに
  `tech_tfs` / `tech_policy`（`require_fib` / `require_nwave` / `require_candle` 含む）と
  `technical_context_*` の要求定義を追加し、戦略ローカルでの評価前提を明示。

### 2026-02-20（追記）派生タグの戦略別技術契約を明示

- `execution/strategy_entry.py` の `strategy_tag` 解決を拡張し、サフィックス付き戦略名でも明示契約で解決できるようにした。
- 新規に明示化した主な `strategy_tag`（規約化キー）:
  - `tech_fusion`, `macro_tech_fusion`
  - `MicroPullbackFib`, `RangeCompressionBreak`
  - `ScalpReversalNWave`, `TrendReclaimLong`, `VolSpikeRider`
  - `MacroTechFusion`, `MacroH1Momentum`, `trend_h1`, `LondonMomentum`, `H1MomentumSwing`
- `entry_thesis` の受け渡し時に `technical_context_ticks` / `technical_context_tfs` / `technical_context_candle_counts` を
  明示し、`tech_policy` による要件定義を戦略側で扱う方針へ移行する布石とした（当時は `true` 前提を記録していた）。
- `SESSION_OPEN` を含む既存フローは維持しつつ、suffix 付き `scalp`/`macro`/`micro` タグでも
  pocket 非依存で解決可能なフォールバックを追加。  
  これにより N波/フィボ/ローソク要件の適用経路がより安定化した。

### 2026-02-14（追記）戦略別技術契約の運用名寄せ

- `execution/strategy_entry.py` に `_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を追加し、戦略ごとの既定テック要件を明文化。
- 自動注入されるキー:
  - `technical_context_tfs`（取得TF順）
  - `technical_context_fields`（保存指標）
  - `technical_context_ticks`（参照tick）
  - `technical_context_candle_counts`（TF別ローソク本数）
- 戦略側 `entry_thesis` がこれらを未設定の場合、上位契約で補完される。  
  補完後に `analysis.technique_engine.evaluate_entry_techniques` へ渡され、  
  `technical_context["result"]` と合わせて `entry_thesis["technical_context"]` へ格納する運用を統一。
- 対象は `SCALP_PING_5S`, `SCALP_PING_5S_B`, `SCALP_M1SCALPER`, `SCALP_MACD_RSI_DIV`,
  `SCALP_TICK_IMBALANCE`, `SCALP_SQUEEZE_PULSE_BREAK`,
  `SCALP_WICK_REVERSAL_BLEND`, `SCALP_WICK_REVERSAL_PRO`,
  `MICRO_ADAPTIVE_REVERT`, `MICRO_MULTISTRAT`。

### 2026-02-14（追記）戦略要件の絶対化（N波/フィボ/ローソク）

- `execution/strategy_entry.py` の
  `_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を更新し、以下を標準化:
  - `technical_context_ticks` の戦略別明示（`latest_bid/ask/mid/spread_pips` を前提に、`tick_imbalance` 系で `tick_rate` 追加）
  - `technical_context_candle_counts` の戦略別明示（N本取得本数を戦略単位で定義）
  - `tech_policy` を戦略契約に追加し、`require_fib` / `require_nwave` / `require_candle` を `true` 固定
  - `tech_policy_locked` を追加し、`TECH_*` 環境上書きによる要件破壊を抑制
- 対象戦略/サブタグを契約化:
  - `scalp_ping_5s`, `scalp_m1scalper`, `scalp_macd_rsi_div`, `scalp_squeeze_pulse_break`
  - `tick_imbalance`, `tick_imbalance_rrplus`
  - `level_reject`, `level_reject_plus`
  - `tick_wick_reversal`, `wick_reversal`, `wick_reversal_blend`, `wick_reversal_hf`, `wick_reversal_pro`
  - `micro_multistrat` の代表として `micro_rangebreak`, `micro_vwapbound`, `micro_vwaprevert` 等の主要マイクロサブタグ
- `analysis/technique_engine.py` に `tech_policy_locked` を反映し、ロック時は `TECH_` 系環境変数での上書きをスキップする挙動を追加。
- 補足: `entry_thesis` が既に `tech_policy` を持つ場合も、`tech_policy_locked=True` を契約側で維持するためのマージ規則を追加。

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

### 2026-02-16（追記）戦略ENTRY/EXIT workerのenv分離

- V2戦略ENTRY/EXIT群（`scalp*`, `micro*`, `session_open`, `impulse_retest_s5`）の `systemd/*.service` から
  戦略固有 `Environment=` を切り出し、各サービス対応の `ops/env/quant-<service>.env` を新設。
  - 追加/更新対象 `systemd`:
    - `quant-m1scalper*.service`
    - `quant-micro-adaptive-revert*.service`
    - `quant-micro-multi*.service`
    - `quant-scalp-macd-rsi-div*.service`
    - `quant-scalp-ping-5s*.service`
    - `quant-scalp-squeeze-pulse-break*.service`
    - `quant-scalp-tick-imbalance*.service`
    - `quant-scalp-wick-reversal-blend*.service`
    - `quant-scalp-wick-reversal-pro*.service`
    - `quant-session-open*.service`
    - `quant-impulse-retest-s5*.service`
  - 追加/更新対象 `ops/env/*`:
    - `ops/env/quant-m1scalper*.env`
    - `ops/env/quant-micro-adaptive-revert*.env`
    - `ops/env/quant-micro-multi*.env`
    - `ops/env/quant-scalp-macd-rsi-div*.env`
    - `ops/env/quant-scalp-ping-5s*.env`
    - `ops/env/quant-scalp-squeeze-pulse-break*.env`
    - `ops/env/quant-scalp-tick-imbalance*.env`
    - `ops/env/quant-scalp-wick-reversal-blend*.env`
    - `ops/env/quant-scalp-wick-reversal-pro*.env`
    - `ops/env/quant-session-open*.env`
    - `ops/env/quant-impulse-retest-s5*.env`
- `quant-scalp-ping-5s` 系は既存の戦略上書きenv（`scalp_ping_5s.env`, `scalp_ping_5s_b.env`）を維持し、`ops/env/quant-scalp-ping-5s*.env` を基本設定用として分離。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` を同一コミットで更新し、監査時に `EnvironmentFile` の二段構造
  (`quant-v2-runtime.env` + `quant-<service>.env`) をチェック対象化。

### 2026-02-17（追記）position_manager 呼び出しのHTTPメソッド齟齬解消

- VM監査で `quant-position-manager` 側の `POST /position/open_positions` が `405` となるログを確認。
- 原因は `execution/position_manager.py` の `open_positions` 呼び出しが固定 `POST` だったため、ワーカー定義
  (`workers/position_manager/worker.py`) が `GET /position/open_positions` を公開していることとの不一致。
- 修正: `execution/position_manager.py` の `_position_manager_service_call()` を `path == "/position/open_positions"` 時に
  `GET` + query params (`include_unknown`) へ分岐するよう変更し、サービス経路の整合を復旧。
- 変更反映後、`quant-order-manager` を再起動して該当 405 検知率の改善を確認する。

### 2026-02-17（追記）open_positions 405 の下位互換対策

- 運用側の呼び出しに POST が混在しているケースを想定し、`workers/position_manager/worker.py` に
  `POST /position/open_positions` を追加受け口として実装。
- `execution/position_manager.py` では `path` の末尾スラッシュを除去して正規化し、`/position/open_positions` 系を
  `GET + params` へ固定振り分けする分岐を堅牢化。
- 既存の GET 経路は維持しつつ、POST 混在時の `405 Method Not Allowed` を回避。

### 2026-02-14（追記）order_manager の戦略意図保全（市場/リミット両方）

- `execution/order_manager.py` の `market_order()` と `limit_order()` 入口で、`entry_probability` と `entry_units_intent` を
  `entry_thesis` へ必須注入・補完する仕組みを統一し、`entry_probability` に応じたロット縮小／リジェクトのみを
  `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1` 時の実装として明確化。
  - reduce_only/manual 除外時のみ `preserve` を有効化。
- `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1` かつ pocket/manual 以外では、戦略側意図が示すSL/TPやサイズ方針を
  order_manager が一方的に再設計しないよう、以下は `not preserve` 条件へ追従:
  - Brain / Forecast / Pattern gate
  - entry-quality / microstructure gate
  - spread block / dynamic SL / min-RR 調整 / TP&SLキャップ / hard stop / normalize / loss cap / direction cap
- ただし、`entry_probability` による「許容上限超えでの縮小」や「超低確率での拒否」は risk 側許容範囲として維持。
- リミット注文側も同様に `entry_probability` 注入・同条件ガードを追加し、`order_manager_service` 経路に同じ意図を引き継ぐように統一。

### 2026-02-14（追記）戦略横断意図協調（entry_intent_board）基盤整備
- `execution/order_manager.py` に `entry_intent_board` / `intent_coordination` の基盤（スキーマ、DB、preflight、worker endpoint）を追加。
- 当時の方針整理で `strategy_entry` 側連携は一旦抑止し、`order_manager` 側に上書き的な再設計を残さない構成を優先した。

### 2026-02-18（追記）意図協調をstrategy_entry経由で復帰
- `execution/strategy_entry.py` の `market_order` / `limit_order` で
  `entry_probability` と `entry_units_intent` を維持したまま `strategy_tag` 解決し、
  `/order/coordinate_entry_intent` を経由してから `order_manager` へ転送する形へ戻す。
- `workers/order_manager/worker.py` の `POST /order/coordinate_entry_intent` を有効のまま維持し、
  各戦略が自戦略意図を保持したまま黒板協調の結果を反映できる運用へ復元。

### 2026-02-14（追記）黒板協調・最小ロット判定を strategy 固定化
- `execution/order_manager.py` の `entry_intent_board` 集約キーを `strategy_tag` 前提へ更新。
- `_coordinate_entry_intent` が `pocket` ではなく `strategy_tag + instrument` の組で照合するよう変更。
- `min_units_for_strategy(strategy_tag, pocket)` を新設し、`strategy_tag` 指定時は `ORDER_MIN_UNITS_STRATEGY_<strategy>` を優先適用。
- `execution/strategy_entry.py` の戦略側協調前チェックも `min_units_for_strategy` を利用するよう更新。

### 2026-02-17（追記）order/position worker の自己service呼び出しガード

- `quant-position-manager.service` と `quant-order-manager.service` の環境衝突（`quant-v2-runtime.env` が
  `*_SERVICE_ENABLED=1` を上書きしていた）を受け、専用 env を新設してサービス責務を明確化。
  - 追加: `ops/env/quant-position-manager.env`
  - 追加: `ops/env/quant-order-manager.env`
  - 更新: `systemd/quant-position-manager.service`

### 2026-02-17（追記）order_manager 入力確率の欠損耐性を強化

- `execution/order_manager.py` の `market_order` 入口で、`entry_probability` の正規化候補を拡張。
- `entry_probability` が `None` / 非数値 / `NaN` / `Inf` いずれでも
  `entry_thesis["confidence"]` / `confidence` 引数（優先順）を用いて補完し、意図値を欠損に依存させない実装を追加。
- `entry_probability` が不正値でも有効な `confidence` があれば上書き補完する挙動に変更し、`entry_units_intent` 同様に
  実行経路の安定性を維持。
- 本変更は品質低下を避けるため、ロジック上の選別を追加せず、既存のガード/リスク条件の枠内でのみ運用される。
  - 更新: `systemd/quant-order-manager.service`
- 両ワーカー側にも明示ガードを入れ、`execution/*_manager.py` の service-first 経路を有効化しつつ、
  各ワーカー実体が self-call（自分自身のHTTP経路を再コール）しない安全策を追加。
- `main` 経由の再監査で `POSITION_MANAGER` 側の 127.0.0.1:8301 での read timeout 連鎖が解消することを確認済み（次アクションとして
  デプロイ後の VM 監査結果を添付）。

### 2026-02-14（追記）market-data-feed の履歴取得を差分化

- `market_data/candle_fetcher.py`
  - `fetch_historical_candles()` が `count` 取得時と同時に `from` / `to` 範囲取得（RFC3339）も扱えるように拡張。
  - `initialize_history()` を「既存キャッシュ終端から次バー境界まで」を起点にした差分再取得へ変更。
    - 既存 `factor_cache` の最終足時刻を参照し、その `+TF幅` から `now` までを取得。
    - 取得時に重複時刻を除外して append し、既存件数に加えて 20本条件を満たせば成功扱い。
  - 運用中再シード時に固定リトライで同一履歴を上書きし続ける問題を低減。
- `WORKER_ROLE_MATRIX_V2.md` のデータ面に、シード時の差分補完方針（前回キャッシュ終端ベース）を追記。

### 2026-02-18（追記）V2監査（VM自動実行）追加

- `scripts/ops_v2_audit.py` を追加し、V2導線（データ/制御/ORDER/POSITION/戦略）監査を1回の実行で集約。
- 追加: `scripts/ops_v2_audit.sh`（systemd起動ラッパ）
- 追加: `systemd/quant-v2-audit.service` / `systemd/quant-v2-audit.timer`
- 監査対象:
  - `quant-market-data-feed`, `quant-strategy-control`, `quant-order-manager`, `quant-position-manager` の active
  - 戦略ENTRY/EXITの主要ペア active 状態
  - `EnvironmentFile` 構成（`quant-v2-runtime.env` + `quant-<service>.env`）
  - `quant-v2-runtime.env` 制御キー値（主要フラグ）
  - `position/open_positions` の `405` 監視（order/position worker 呼び出し相当）
  - `quantrabbit.service` 等 legacy active の有無
- `systemd install`/`timer` 導入は `install_trading_services.sh --units` 経由で統一し、運用側は `logs/ops_v2_audit_latest.json` を監査ログとして参照。

### 2026-02-17（追記）V2監査の誤検知を抑制するための修正

- `quant-v2-audit.service` が誤検知していた `position/open_positions` の `405` 集計は、journal 行中の
  タイムスタンプ文字列（`...:405`）が `405` 検知に拾われる副作用が原因だったため、`scripts/ops_v2_audit.py` を修正。
  - メソッド不一致判定は `Method Not Allowed` と `... /position/open_positions ... 405` の実リクエスト行のみを対象化。
- `install_trading_services.sh --all` が V2監査で禁止とするレガシーサービスを誤って再有効化しないよう、除外対象を明示。
  - 除外: `quant-impulse-retest-s5*`, `quant-micro-adaptive-revert*`（`--all` ではインストールせず、明示 `--units` 指定時のみ許容）
- `install_trading_services --all` 再実行時も V2監査の disallow ルールを壊しにくい状態に更新。

### 2026-02-14（追記）legacy残存時の監査許容ルールを追加

- `scripts/ops_v2_audit.py` に、`OPS_V2_ALLOWED_LEGACY_SERVICES` で legacy サービスを明示許可する設定を追加。
- `ops/env/quant-v2-runtime.env` に `OPS_V2_ALLOWED_LEGACY_SERVICES=...` を設定し、
  `quant-impulse-retest-s5*` と `quant-micro-adaptive-revert*` の active 判定を
  `critical` ではなく `warn` へトレードオフし、当面の運用継続を担保。

### 2026-02-14（追記）install_trading_services.sh の起動待機ハング対策

- `scripts/install_trading_services.sh --all` 実行時に、`quant-strategy-optimizer.service` の
  oneshot長時間処理起動でスクリプト全体が待機し続ける現象を確認。
- 対応として `scripts/install_trading_services.sh` の `enable_unit()` に
  `NO_BLOCK_START_UNITS` を追加し、`quant-strategy-optimizer.service` を
  `systemctl start --no-block` で起動要求するよう変更。

### 2026-02-14（追記）本番トレード制御フラグ有効化

- `ops/env/quant-v2-runtime.env` の `MAIN_TRADING_ENABLED` を `0` から `1` に変更し、
  V2運用の「戦略ワーカー→order/position manager経路」実行を本番可で許可。
- 併せて VM 運用環境の `ops/env/quant-v2-runtime.env` も同値で更新し、core 監査（`quant-v2-audit`）後に
  リアルタイム取引許可状態の整合を確認。
- `ops_v2_audit` の実運用期待値を `MAIN_TRADING_ENABLED=1` に更新し、取引許可状態を監査基準へ反映。
- このため `--all` 実行時の完了待機を回避しつつ、監査ジョブ（`quant-v2-audit`）の定期実行を維持。

### 2026-02-24（追記）戦略分析係に戦略固有パラメータ参照を追加

- `analysis/strategy_feedback_worker.py` を更新し、`systemd` 上の戦略サービス `EnvironmentFile` から
  戦略ごとに一致する環境パラメータを抽出して `strategy_params` として保持するようにした。
- 取得した `strategy_params` は `strategy_feedback` 生成時に各戦略の `strategy_params.configured_params` として
  JSON 出力へ同梱し、`entry_probability_multiplier` / `entry_units_multiplier` /
  `tp_distance_multiplier` / `sl_distance_multiplier` の根拠追跡性を強化。
- 戦略追加・停止時の観測に対しても `systemd` 検知を優先し、停止戦略は
  `LAST_CLOSED` の古さ条件を満たさない限り `strategy_feedback` 出力対象外にして誤適用を回避。
- 追加で、実行中 `quant-*.service` の `FragmentPath` を systemd から直接読み取り、リポジトリ上の
  unit ファイルに未同期の戦略追加にも即座に追従するようにした。

### 2026-02-15（追記）analysis_feedback の最終受け渡しを明示化

- `analysis/strategy_feedback.py` で `strategy_params` 内の `configured_params` を分離して
  `advice["configured_params"]` として明示的に出力し、ノイズ対策しない形で戦略固有パラメータを保持。
- `execution/strategy_entry.py` で分析結果を `entry_thesis["analysis_feedback"]` に格納し、
  既存利用者互換として `analysis_advice` も併記して戦略別改善値の監査性を維持。

### 2026-02-15（追記）戦略側 technical_context 要件の最終穴埋め

- `workers/hedge_balancer/worker.py` と `workers/manual_swing/worker.py` の
  `evaluate_entry_techniques` 呼び出し前に `tech_policy` を明示化し、
  `require_fib/require_median/require_nwave/require_candle` を全戦略側で揃えた。
- 同時に監査資料 `docs/strategy_entry_technical_context_audit_2026_02_15.md` の
  該当行（`hedge_balancer`, `manual_swing`）も同値に更新。

### 2026-02-15（追記）tick imbalance ラッパーを独立戦略モード固定化

- `workers/scalp_tick_imbalance/worker.py` を `scalp_precision_worker` 直接 import 依存から
  起動時に `SCALP_PRECISION_*` を明示設定して mode 固定起動する構成へ更新。
- `SCALP_PRECISION_MODE=tick_imbalance`、`ALLOWLIST`、`MODE_FILTER_ALLOWLIST`、`UNIT_ALLOWLIST`、
  `LOG_PREFIX` を `__main__` 起動時に再設定し、`workers/scalp_precision/worker.py` の
  `evaluate_entry_techniques` 入口を再利用しつつ、戦略名を `scalp_tick_imbalance` に固定。
- 監査資料 `docs/strategy_entry_technical_context_audit_2026_02_15.md` の
  `scalp_tick_imbalance` 行を `evaluate_entry_techniques`/`technical_context_*` 実装済み扱いに更新。

### 2026-02-15（追記）V2実用起動候補（8戦略ペア）構成監査

- 対象エントリー/EXITペア:
  - `quant-scalp-ping-5s` / `quant-scalp-ping-5s-exit`
  - `quant-micro-multi` / `quant-micro-multi-exit`
  - `quant-scalp-macd-rsi-div` / `quant-scalp-macd-rsi-div-exit`
  - `quant-scalp-tick-imbalance` / `quant-scalp-tick-imbalance-exit`
  - `quant-scalp-squeeze-pulse-break` / `quant-scalp-squeeze-pulse-break-exit`
  - `quant-scalp-wick-reversal-blend` / `quant-scalp-wick-reversal-blend-exit`
  - `quant-scalp-wick-reversal-pro` / `quant-scalp-wick-reversal-pro-exit`
  - `quant-m1scalper` / `quant-m1scalper-exit`
- `/systemd/*.service` 上で 16本全てが `ExecStart` を `workers.<strategy>.worker` / `.exit_worker` に正しく固定し、
  `EnvironmentFile` も `quant-v2-runtime.env` + 各戦略 env の2行で定義される構成を確認。
- `quant-scalp-ping-5s.service` の追加 `EnvironmentFile` として
  `ops/env/scalp_ping_5s.env` を明示的に用意し、参照行の未作成問題を解消。
- `scalp_wick_reversal_blend` 側 env は `SCALP_PRECISION_ENABLED=1` に更新し、実運用起動前提を揃えた。

### 2026-02-15（追記）WickReversalBlend 起動停止原因の除去

- `workers/scalp_precision/worker.py` の `_place_order` 内で、
  `not tech_decision.allowed` の分岐と `_tech_units <= 0` の分岐が
  `continue` になっており、関数内制御として不正だったため
  `SyntaxError: 'continue' not properly in loop` を発生させていた点を修正。
- 2箇所を `return None` に変更し、`quant-scalp-wick-reversal-blend.service` の起動停止要因を解消する
  形にした。

### 2026-02-15（追記）wick/squeeze/tick wrapper の独立起動化

- `workers/scalp_squeeze_pulse_break/worker.py` / `workers/scalp_wick_reversal_blend/worker.py` /
  `workers/scalp_wick_reversal_pro/worker.py` を `SCALP_PRECISION_MODE` 固定起動形式へ揃え、
  `SCALP_PRECISION_ENABLED/ALLOWLIST/UNIT_ALLOWLIST/LOG_PREFIX` を `__main__` で明示設定するように修正。
- これにより、環境差し戻し時でもそれぞれの戦略名で `scalp_precision` のローカル評価を実行し、
  V2実用8戦略（`scalp_tick_imbalance`含む）へ同一方針で意図固定を維持できる状態を整備。

### 2026-02-16（追記）M1 Scalp の N-Wave アライメント停止要因を可変化

- `strategies/scalping/m1_scalper.py` に `M1SCALP_NWAVE_ALIGN_*` 環境変数を追加し、
  N-Wave 連続検知時の `skip_nwave_*_alignment` を運用側で可変化。
- `ops/env/quant-m1scalper.env` に `M1SCALP_NWAVE_ALIGN_ENABLED=0` を追加して
  アライメントガードを一時無効化（必要に応じて再有効化可能）し、5秒以外の戦略通過率改善を優先。

### 2026-02-16（追記）micro_multi の因子劣化時軽量化 + PositionManager サービス耐障害化

- `workers/micro_multistrat/worker.py`
  - M1 因子 `age_m1` が `MAX_FACTOR_AGE_SEC` を超過した場合、`FRESH_TICKS_STALE_SCALE_MIN` と
    `FRESH_TICKS_STALE_SCALE_HARD_SEC` を使って `stale_scale` を算出し、`units` に適用して
    スケールアウト時に過大エントリーを抑える「自動軽量化」を追加。
  - `workers/micro_multistrat/config.py` に上記2変数を追加し、運用時の調整余地を確保。
- `execution/position_manager.py`
  - サービス障害時の再試行を指数バックオフ（`POSITION_MANAGER_SERVICE_FAIL_BACKOFF_*`）で抑止し、
    短時間の連打エラーを軽減。
- `ops/env/quant-v2-runtime.env`
  - `ORDER_MANAGER_SERVICE_FALLBACK_LOCAL` と `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL` を `1` に変更し、
    サービス面不具合時はローカルフォールバック経路へ短時間で移行する運用を追加。

### 2026-02-16（追記）session_open（AddonLive）経路で `env_prefix` を固定注入

- `workers/common/addon_live.py` の `AddonLiveBroker.send()` で、`order/intent/meta` から
  `env_prefix`（または `ENV_PREFIX`）を受け取り、`entry_thesis` と `meta_payload` に
  同一値を注入するよう変更。
- `execution/strategy_entry` 側の `coordinate_entry_intent` で `env_prefix` が混在しにくい状態を保つための
  追加対策として、session_open 系の発注にも意図値（`entry_units_intent` / `entry_probability`）と同様に
  `env_prefix` の明示注入を追加。

- 2026-02-16: `scalp_m1scalper` エントリー抑制対策として `ops/env/quant-m1scalper.env` に `M1SCALP_ALLOW_REVERSION=1` を追加し、レンジ条件許可 (`M1SCALP_REVERSION_REQUIRE_STRONG_RANGE=1`, `M1SCALP_REVERSION_ALLOWED_RANGE_MODES=range`) と閾値を緩和 (`M1SCALP_REVERSION_MIN_RANGE_SCORE=0.72`, `M1SCALP_REVERSION_MAX_ADX=20`) へ変更。M1の`buy-dip / sell-rally` 系シグナル（リバーション）を通過しやすくし、エントリー減少の改善を試験的に実施。
- 2026-02-16: `M1SCALP_REVERSION_ALLOWED_RANGE_MODES` を `range,mixed` に広げ、`M1SCALP_REVERSION_MIN_RANGE_SCORE` を `0.55`、`M1SCALP_REVERSION_MAX_ADX` を `30` に変更。`ALLOW_REVERSION=1` を維持しつつ、リバーション通過条件の厳しさを中立寄りに再調整。
