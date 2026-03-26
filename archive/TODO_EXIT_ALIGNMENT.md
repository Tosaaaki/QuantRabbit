## EXIT 整合性チェックメモ（2025-12-26）

### 共通方針
- main は `WORKER_ONLY_MODE=true` / `MAIN_TRADING_ENABLED=0` が既定。ポケット共通 EXIT は廃止し、各ワーカーがエントリー〜EXIT まで完結させる。
- `execution/exit_manager.py` は互換スタブ（自動EXITなし）。`plan_closures` は常に空、attach() は `exit_disabled=True` を付けるのみ。
- エントリー/クローズとも `client_order_id` 必須。欠損ポジションは PositionManager でスキップし、exit_worker でもタグ/ID欠損は拒否。
- 最低保有と「損失ではクローズしない」ルールを徹底する: scalp>=10s、micro>=15–20s、macro>=15m。PnL<=0 は原則ホールド（強制ドローダウン/ヘルスのみ例外）。

### 現在の実装状況
- **scalp/fast系**: impulse_*/pullback_*/mirror/squeeze/vwap_magnet/fast_scalp/scalp_m1scalper/scalp_multistrat すべて exit_worker 持ち。PnL>0 決済のみ、range_mode で trail/take をタイト化、client_id 必須。
- **micro系**: momentum_burst/stack/pullback_ema/level_reactor/range_break/vwap_bound/micro_multistrat/trendmomentum いずれも専用 exit_worker で min_hold>=15–20s + PnL>0。range 判定は ADX/BBW/ATR 共有。
- **macro系**: trendma/h1momentum/donchian55/london/manual_swing が専用 exit_worker を持ち、min_hold>=15m + PnL>0、range 中はタイト化。H1/H4 のテクニカルから trail/take をスケール。
- **仕組み**: micro_core/macro_core/scalp_core と core_executor/systemd ユニットは削除済み。共通 exit ワーカーも起動しない。

### 残タスク / チェック項目
- exit_worker ごとの min_hold / trail / lock / RSI take が期待どおりログに出ているか短時間リプレイで再確認する。
- fast_scalp の `NO_LOSS_CLOSE` 運用と exit_worker (PnL>0 専用) の併走で早期決済が失われていないか確認。
- PositionManager 経由でタグ欠損ポジションが渡ってこないか監視し、見つけたらログ＋手動クローズ手順を残す。
- `EXIT_MANAGER_DISABLED=1` を維持すること（addon 実行時に誤って attach を期待しないよう警告を追加済みか確認）。

### 検証の進め方
1. 対象ワーカーのエントリー条件と exit_worker の閾値を洗い出し（指標/時間/レンジ/RSI/トレイル）。
2. PnL<=0 でクローズしないこと、min_hold を守ること、client_id/タグ必須が効いていることをログ/metrics で確認。
3. リプレイ or 短時間の実行で「エントリー→最低保有→微益/TP/トレイル決済」の流れをチェックし、閾値の整合/上書き（entry メタ tp_pips/sl_pips など）を調整。
4. 回帰後は docs/tasks/EXIT_REWORK.md へ結果を追記し、systemd 再起動で exit_worker が自動起動することを確認する。
