# Technical EXIT 強化タスクリスト（アップデート 2025-12-26）

- 下記 17 ワーカーはすべて専用 exit_worker 実装済みで、PnL>0 決済・最低保有・range タイト化を備える。共通 ExitManager はスタブ化済み。
  - scalp/fast 系: fast_scalp / pullback_s5 / vwap_magnet_s5 / impulse_momentum_s5 / impulse_retest_s5 / pullback_runner_s5 / scalp_multistrat
  - micro 系: micro_multistrat / trendma micro variants / pullback_ema / level_reactor / range_break / vwap_bound / momentum_burst / micro_trendmomentum
  - macro 系: trendma / donchian55 / h1momentum / london_momentum / manual_swing
- 新しい EXIT 調整は `tasks/EXIT_REWORK.md` 側で管理する（min_hold/微益トレイル/タグ必須の検証など）。本ファイルの旧リストは完了として扱う。
- 追加でテクニカルを EXIT に持ち込む場合は、各 exit_worker に指標を渡しても PnL>0 前提を崩さないことを確認する。
