# EXIT 個別化リファクタ タスク

## 目的
- 共通 ExitManager 依存を廃止し、戦略ごとに専用 EXIT ロジックを実装する。
- 最短保有時間を厳格化し、PnL<0 ではクローズしない（原則 TP 待ち）。
- client_order_id 空のクローズを禁止し、追跡不能な約定をゼロにする。

## 方針
- スキャ系 → ミクロ系 → マクロ系の順に分離・実装。
- 最低保有: scalp ≥10s, micro ≥15–20s, macro ≥15m。
- クローズ条件: PnL>0 のみ（タイムアウトでも BE 未満は原則ホールド、緊急フェイルセーフのみ例外）。
- スプレッド/滑り判定は最短保有経過後のみ適用。
- client_order_id 必須（発注・クローズとも）。欠損時は拒否＋アラート。

## タスク
1) スキャ系 exit_worker 分離
   - impulse/m1scalper/fast_scalp/scalp_multi/micro_multi の exit_worker から ExitManager 呼び出しを外し、専用ロジックを実装。
   - 最短保有・PnL>0のみクローズ・client_id必須をコードに埋め込む。
2) ミクロ系分離
   - TrendMomentum/BBrsi など micro 用 exit_worker を専用化。
   - 最短保有 15–20s、負けクローズ禁止、スプレッドチェックは経過後のみ。
   - micro_core から各戦略（MomentumBurst/MomentumStack/PullbackEMA/LevelReactor/RangeBreak/VWAPBound/TrendMomentumMicro）を分離し、個別 exit_worker を配置。
3) マクロ系分離
   - trendma/h1momentum/donchian55 の exit_worker を専用化。
   - 最短保有 15m 以上、短期逆行では決済しない。RR固定 TP/SL に寄せる。
4) ログ/監視
- 日次で全約定CSVを自動出力（ID付与済み）し、空ID/即クローズ件数を監視・アラート。
5) デプロイ/検証
 - 小ロットで順次再開し、PnL勾配と決済タイミングを確認。問題があれば即停止。

## 現在の状態（2025-12-26）
- 共通 ExitManager は互換スタブのみ（`plan_closures` は常に空、attach で `exit_disabled` を付与するだけ）。自動EXIT経路は完全停止。
- main は `WORKER_ONLY_MODE=true` / `MAIN_TRADING_ENABLED=0` がデフォルトで、エントリー/EXIT はワーカー専用に一本化している。
- 発注側は client_order_id 欠損を拒否するパッチ済み。クローズ側も client_id 欠損を拒否（metric + reject）。EXIT から close_trade へ client_id 伝搬済み。
- 専用 EXIT 完了: impulse_break_s5 / impulse_momentum_s5 / impulse_retest_s5 / scalp_m1scalper / scalp_multistrat / fast_scalp / micro_multistrat / macro_trendma / macro_h1momentum / macro_donchian55 / micro_trendmomentum（TrendMomentumMicro / MicroMomentumStack） / micro_pullbackema / micro_levelreactor / micro_rangebreak / micro_vwapbound / micro_momentumburst（いずれも最低保有 + PnL>0 決済のみ、ExitManager 非依存）。
- 共通コア（micro_core / macro_core / scalp_core）と core_executor を削除。scalp_core 用 systemd ユニットも削除済み。
- 未着手: exit_worker 回帰リプレイと再起動動作の確認。fast_scalp の `NO_LOSS_CLOSE` 運用を含む保有時間・微益トレイルのログ確認。

## EXIT誤爆防止・タグ運用（必須事項）
- 発注時は必ず `strategy_tag` と `client_order_id` をセット。欠損なら発注を拒否する。
- PositionManager は `strategy_tag` 欠損ポジションを EXIT へ渡さない（agent pocketのみ）。タグ欠損が出たらログで警告。
- EXITワーカーは ALLOWED_TAGS のみを処理し、タグ欠損ポジはスキップする。ポケット共通 EXIT と専用 EXIT を併用しない。
- manual ポケットは自動EXIT対象外。手動玉は手動でのみ決済。

## デプロイ手順メモ（再発防止用）
- 通常（IAP + OS Login鍵、依存更新、サービス再起動）  
  `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t -k ~/.ssh/gcp_oslogin_quantrabbit deploy -i --restart quantrabbit.service`
- gcloudアカウントを切り替える場合（例: other@example.com）  
  `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -A other@example.com -t -k ~/.ssh/gcp_oslogin_quantrabbit deploy -i --restart quantrabbit.service`
- ブランチ指定  
  `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t -k ~/.ssh/gcp_oslogin_quantrabbit deploy -b main -i --restart quantrabbit.service`
- vm.sh が失敗した場合の直接実行（2段階）  
  1) コード更新 + venv更新  
     `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && git fetch --all -q || true && git checkout -q main || git checkout -b main origin/main || true && git pull --ff-only && if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi'"`
  2) systemd 再起動  
     `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"`
