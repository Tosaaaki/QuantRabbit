# 両建て運用メモ（2025-12-30）

目的: 証拠金をネット額で最大限活用し、片方向の圧迫時でも反対方向エントリーで余力を回復しながらフルに張る。

## 現状の挙動
- OANDA は同一ペアで netting。必要証拠金は「ネット units × 価格 × 証拠金率(≒4%)」で計算される。
- `ALLOW_HEDGE_ON_LOW_MARGIN=1`（デフォルトON）: free margin が閾値未満でもヘッジ方向のエントリーをスキップしない。
- `allowed_lot` でロット計算時に open long/short を参照し、エントリー後の net margin をシミュレートしてサイズ決定（`workers/fast_scalp` / `workers/micro_multistrat` で適用済み）。
- cap は `MAX_MARGIN_USAGE` を 0.92 にクランプし、`MARGIN_SAFETY_FACTOR` でさらに少し絞る（デフォルト0.9）。

## 具体的なイメージ
- NAV 50万円、証拠金率4%の場合、片側だけなら約50万×0.92/（価格×0.04）で ~7.3万 units ほど。ロングとショートを組み合わせれば net が小さくなるため、総建玉は倍近くまで持てる。
- 例: 45kショート保有時に50kロングを入れると net +5k → 必要証拠金は ~3.1万円まで低下し、余力が回復する。

## 推奨設定
- 確保: `ALLOW_HEDGE_ON_LOW_MARGIN=1`（既定）。
- 必要なら緩和: `MIN_FREE_MARGIN_RATIO` を 0.005–0.008 に下げてヘッジ方向を通しやすくする。
- スプレッドが原因で止まる場合: `FAST_SCALP_MAX_SPREAD_PIPS` を 1.1–1.2p 目安に調整して検証。
- stale 緩和: `FAST_SCALP_MAX_SIGNAL_AGE_MS=6000`（遅い tick feed でもエントリー可能）。

## 運用フロー
1) ポジション取得: `/v3/accounts/{id}/openPositions` から long/short units を取得。
2) ロット計算: `allowed_lot(..., side, open_long_units, open_short_units)` でエントリー方向と net 後の証拠金を考慮。
3) ゲート確認: spread / stale / pattern / cooldown を通過できるよう閾値を調整。ログで `fast_scalp_skip` / `signal_stale` を監視。
4) 継続監視: `orders.db` で filled が動くか確認。`metrics` に `account.margin_usage_ratio` / `free_margin_ratio` を記録済み。

## 注意点
- DIR_CAP_RATIO（同方向キャップ）は既定のまま。 gross 全体を抑えたいときだけ調整。
- cap=0.92+安全係数で「総裁量が過大になる」ケースは抑制されるが、急変時のマージンコールに注意。

## 実装済みヘッジワーカー (HedgeBalancer)
- 役割: マージン使用率が高まったときに逆方向の reduce-only シグナルを `signal_bus` 経由で main 関所へ送り、ネットエクスポージャを軽くする。ファイル: `workers/hedge_balancer/worker.py`。
- トリガー: `margin_usage >= 0.88` または `free_margin_ratio <= 0.08` かつ net_units≥15k。ターゲット使用率は 0.82、ネット削減量は net の 55% 以内で最小 10k、最大 90k units。
- シグナル内容: pocket=`macro`、confidence=90、sl/tp=5pips（reduce_only 前提で sl/tp は添付されない）、proposed_units を明示して reduce_only+reduce_cap で送信。
- 連続発火ガード: 20s クールダウン。価格は直近 tick mid（5s 窓）を使用、価格が MIN_PRICE(<90) ならスキップ。
- 環境変数: `HEDGE_BALANCER_ENABLED` / `HEDGE_TRIGGER_MARGIN_USAGE` / `HEDGE_TARGET_MARGIN_USAGE` / `HEDGE_TRIGGER_FREE_MARGIN_RATIO` / `HEDGE_MIN_NET_UNITS` / `HEDGE_MIN_HEDGE_UNITS` / `HEDGE_MAX_HEDGE_UNITS` / `HEDGE_MAX_REDUCTION_FRACTION` / `HEDGE_COOLDOWN_SEC` / `HEDGE_POCKET` / `HEDGE_CONFIDENCE` / `HEDGE_SL_PIPS` / `HEDGE_TP_PIPS` / `HEDGE_MIN_PRICE`。
- 起動例: `python -m workers.hedge_balancer.worker`（既定で有効。ログに `[HEDGE] enqueue dir=...` が出れば発火）。
