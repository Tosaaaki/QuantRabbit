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
- 共通 ExitManager はまだ生存。スキャ/マクロ系ワーカーは停止中（再開前に専用EXIT実装が必要）。
- 発注側は client_order_id 欠損を拒否するパッチ済み。クローズ側も client_id 欠損を拒否（metric + reject）。EXIT から close_trade へ client_id 伝搬済み。
- 専用 EXIT 完了: impulse_break_s5 / impulse_momentum_s5 / impulse_retest_s5 / scalp_m1scalper / scalp_multistrat / fast_scalp / micro_multistrat / macro_trendma / macro_h1momentum / macro_donchian55 / micro_trendmomentum（TrendMomentumMicro / MicroMomentumStack） / micro_pullbackema / micro_levelreactor / micro_rangebreak / micro_vwapbound / micro_momentumburst（いずれも最低保有 + PnL>0 決済のみ、ExitManager 非依存）。
- 共通コア（micro_core / macro_core / scalp_core）と core_executor を削除。scalp_core 用 systemd ユニットも削除済み。
- 未着手: 回帰テスト/再起動確認。
