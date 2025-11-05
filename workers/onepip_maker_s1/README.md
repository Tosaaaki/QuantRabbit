# onepip_maker_s1 Shadow Worker

- 目的: 板情報を用いた 1pip メイカー戦略のためのシャドウ検証ワーカー。初期状態は `ONEPIP_MAKER_S1_ENABLED=false` / `SHADOW_MODE=true`。
- シャドウログ: `logs/onepip_maker_s1_shadow.jsonl` に条件成立時のスナップショットを追記する。`ONEPIP_MAKER_S1_SHADOW_LOG_ALL=true` で全ループを記録。
- 必要データ: `market_data.orderbook_state.update_snapshot()` で L2 スナップショットが投入されていること。最新スプレッドは `spread_monitor` を併用。
- コストガード: `analytics.cost_guard` が OANDA のトランザクションログから `c` を推定し、`ONEPIP_MAKER_S1_MAX_COST_PIPS` を超える場合はエントリーを抑止。
- レンジ条件: 既定では `RANGE_ONLY=true` で `detect_range_mode` がアクティブな時間帯のみ集計。必要に応じて env で解除/閾値調整する。
- live モード: `SHADOW_MODE=false` かつ `ONEPIP_MAKER_S1_ENABLED=true` にすると、`execution.order_manager.limit_order` を用いて post-only 指値を送信し、TTL 超過で自動キャンセルする（TTL<1s 時は内部で cancel タスクを起動）。SL/TP/lot は `ONEPIP_MAKER_S1_*` の env で調整できる。

## Go-live Checklist
1. **L2 フィード**: `orderbook_state` に対して LP 由来のベスト気配＋深度が更新されていること（`MIN_TOP_LIQUIDITY`・`DEPTH_LEVELS` 満足を確認）。
2. **コストサンプル**: `analytics.cost_guard.snapshot()` の `count` が `ONEPIP_MAKER_S1_MIN_COST_SAMPLES` 以上で推移し、平均コストが許容範囲に収まっていること。
3. **リスク連携**: `risk_guard` が口座 NAV から許容ロットを算出し、`can_trade(pocket)` が true を返す状態であること（ポケット DD がリセットされているか確認）。
4. **メトリクス監視**: `logs/metrics.db` へ `onepip_maker_*` メトリクスが記録されていること（TTL キャンセル／order submitted／account nav）。
5. **サンドボックス検証**: シャドウで 1 営業日以上ログを取り、シグナル頻度・条件成立回数・排除理由（`onepip_maker_skip`）をレビュー。
6. **デプロイ**: 上記を満たしたら `ONEPIP_MAKER_S1_ENABLED=true`, `ONEPIP_MAKER_S1_SHADOW_MODE=false` を設定し、最初は小ロットでライブ挙動を確認する。

## Shadow Replay Tips
- `scripts/replay_orderbook_support.py <tick_jsonl>` を使うと、`logs/replay/` に記録したティック（`bids`/`asks` 含む）から `orderbook_state` へ疑似 L2 スナップショットを流し込めます。シャドウワーカーをオフライン検証する際は、別プロセスでこのスクリプトを回しつつ `ONEPIP_MAKER_S1_ENABLED=true`, `SHADOW_MODE=true` で動かすと、live 循環に近い条件でログが取得できます。
- ログの俯瞰: `scripts/analyse_onepip_shadow.py logs/onepip_maker_s1_shadow.jsonl` を実行すると、件数や平均スプレッド、スキップ理由や live 結果の分布を JSON で確認できます。
