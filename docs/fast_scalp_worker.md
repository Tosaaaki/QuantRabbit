# FastScalp Worker Snapshot (2025-12-26)

Tick ベースの超短期スキャル (`workers/fast_scalp/worker.py`) と、PnL>0 だけを捌く専用 EXIT (`workers/fast_scalp/exit_worker.py`) の現行仕様メモ。共通 ExitManager はスタブ化済みで自動EXITは発火しない。

## 現状サマリ
- ループ間隔: 0.25s 既定。`tick_window` を優先し、深さ不足時は OANDA HTTP pricing snapshot で補完する。
- ゲート: `MAX_SPREAD_PIPS`(既定0.45)、JST 03:00–05:00 クールダウン、stale tick (>=3s) でスキップ。low-vol 連続検知で短いクールダウンを付与。
- シグナル: `evaluate_signal` が tick モメンタム/レンジ/インパルス + ATR/RSI で方向を返す。ATR/スプレッドに応じてエントリー閾値を動的調整し、`consolidation_ok` が外れると見送り。任意で `PATTERN_MODEL_PATH` によるパターンスコアゲートを追加できる。
- サイジング: `allowed_lot` を使い pocket `scalp_fast` で上限チェックし、`MAX_LOT`/`MIN_UNITS`/`MAX_ACTIVE_TRADES`/`MAX_PER_DIRECTION` と StageTracker クールダウンを併用。既定は物理SL無効で TP だけを付け、必要に応じて `FAST_SCALP_USE_SL=1`（stop_loss_policy 側で `scalp_fast` の SL が有効な場合）で実SLを送る。
- オーダー: `market_order` で `client_order_id=qr-fast-...`、エントリーメタにシグナル/ATR/RSI/パターン/timeout情報を詰める。fill 後は `set_trade_protections` で TP/SL を再設定。
- EXIT（worker内）: `TimeoutController` が elapsed/pips/tick_rate/latency から close を提案。RSI フェード/ATR spike/ドローダウンで強制 EXIT あり。`MIN_HOLD_SEC` 未満は close 無効。`NO_LOSS_CLOSE`=true 既定で含み損ではクローズしない（drawdown/health は例外）。
- EXIT（専用 exit_worker）: 最低保有 10s、PnL>0 のみクローズ。range_mode 時は trail/take/lock をタイト化。`client_order_id` が無いポジションは処理しない。
- 監視: ログ接頭辞 `[SCALP-TICK]` / `[EXIT-fast_scalp]`。metrics は `fast_scalp_signal` / `fast_scalp_skip` / timeout_summary 等。

## 運用メモ
- `EXIT_MANAGER_DISABLED=1` が前提（スタブのまま）。fast_scalp を動かすときは `workers/fast_scalp/exit_worker.py` を一緒に起動し、損失クローズが欲しい場合は `FAST_SCALP_NO_LOSS_CLOSE=0` にする。
- pocket 設定は risk_guard/order_manager に組み込み済み。`FAST_SCALP_ENABLED=0` で worker 全体を無効化できる。
- 物理 SL を使わない運用では `max_drawdown_close_pips` と exit_worker の trail/backoff を定期的に見直す（強制損切り経路が exit_worker にも存在しないため）。
