# 両建て運用メモ（2025-12-30）

目的: 証拠金をネット額で最大限活用し、片方向の圧迫時でも反対方向エントリーで余力を回復しながらフルに張る。

## 現状の挙動
- OANDA は同一ペアで netting。必要証拠金は「ネット units × 価格 × 証拠金率(≒4%)」で計算される。
- `ALLOW_HEDGE_ON_LOW_MARGIN=1`（デフォルトON）: free margin が閾値未満でもヘッジ方向のエントリーをスキップしない。
- `allowed_lot` でロット計算時に open long/short を参照し、エントリー後の net margin をシミュレートしてサイズ決定（`workers/fast_scalp` / `workers/micro_multistrat` で適用済み）。
- cap は `MAX_MARGIN_USAGE` を 0.92 にクランプし、`MARGIN_SAFETY_FACTOR` でさらに少し絞る（デフォルト0.9）。

## 含み損の常態化（手動ポジ除外 / 2026-01-18 VM確認）
- OANDA openTrades（agentのみ: client_id=qr-/qs-）: 31件（含み損30 / 含み益1）。
- 含み損合計: -23,712.18 JPY / 含み益合計: +189.5 JPY。
- 保有時間: 24h超 31件、72h超 7件、最長 84.4h。
- エクスポージャ: net +28,160 units / gross 39,224 units（net が小さくても含み損は残る）。
- 含み損寄与（戦略）: M1Scalper -9,824 / TrendMAbu -5,759 / VolCompre -4,915 / Donchian5 -3,176。
- pocket別合計: scalp -9,824 / macro -8,973 / micro -4,725。
- 判断: 基本はプラス決済だが、実装には複数の例外条件があり、発火条件が限定的なため含み損が取り残されやすい。

## EXIT条件の実装概要（現行）
- 共通: `execution/exit_manager.py` はスタブで不使用。各 `workers/*/exit_worker.py` が主担当。
- 共通: `execution/section_axis.py` の section_exit（fib/median/left_behind など）で負け決済が許可される場合がある。
- 共通: `analysis/technique_engine.py` の tech_exit が allow_negative を返すと負け決済が許可される。
- 共通: `workers/common/exit_emergency.py` により health_buffer が閾値以下なら負け決済が許可される。
- 方針: EXITロジックは共通化せず、ワーカー別に最適化する（共通化はタグ整合・観測のみ）。
- fast_scalp: `workers/fast_scalp/exit_worker.py` は基本プラス決済だが、`workers/fast_scalp/worker.py` に RSI fade / ATR spike / drawdown / timeout / health_exit などの強制クローズがある（`NO_LOSS_CLOSE` の例外あり）。
- pullback系: `workers/pullback_scalp/exit_worker.py` / `workers/pullback_s5/exit_worker.py` / `workers/vwap_magnet_s5/exit_worker.py` は stop_loss / time_cut / rsi_fade / vwap_cut / structure_break 等で負け決済が発生し得る。
- micro_multistrat: `workers/micro_multistrat/exit_worker.py` で reversion_failure/structure_break などの負け決済があり、tech_exit の allow_negative が必要。
- macro_trendma: `workers/macro_trendma/exit_worker.py` で trend_failure（MA崩れ＋逆行幅）時に tech_exit 連動で負け決済。
- micro_levelreactor / trend_h1 などは原則プラス決済中心だが section_exit 経由の例外はあり得る。

## EXITが発火しにくい原因（実データ＋コード）
- orders.db の entry_thesis ではタグが派生形（例: `M1Scalper-trend-long`, `TrendMA-bull`, `Donchian55-breakout-up`, `VolCompressionBreak-long`）。`_filter_trades` が完全一致のみだと拾えず、正規化に失敗すると exit_worker が対象外になる。
- M1Scalper: `workers/scalp_m1scalper/exit_worker.py` は max_hold/max_adverse で負け決済できるが、タグ一致が前提。派生タグのままだとスキップされる。
- TrendMA: `workers/macro_trendma/exit_worker.py` の負け決済は trend_failure + tech_exit の連動のみ。時間経過での損切りが無く、含み損が残りやすい。
- Donchian55: `workers/macro_donchian55/exit_worker.py` も trend_failure + tech_exit 依存。ブレイク後の逆行が「構造崩れ」と判定されない限り残存しやすい。
- VolCompressionBreak: micro_multistrat の負け決済は reversion_failure 対象外。section_exit が主ルートなので、基準未達だと長期化する。
- データ依存: exit_worker は `tick_window` / `all_factors` から mid/ADX/RSI を引く。データ欠損や stale で `_latest_mid()` が None だと判定自体がスキップされる。
- 対応: exit_worker のタグ判定を base tag / `strategy_tag_raw` / `trade.strategy_tag` まで拡張し、派生タグの取り残しを抑える（ワーカー別ルールは維持）。
- 対応: M1Scalper/TrendMA/Donchian55/VolCompressionBreak に加え、MomentumBurst/MicroRangeBreak/MicroPullbackEMA/TrendMomentumMicro/MicroLevelReactor/ScalpMulti(非reversion) にテクニック系の負け決済を追加（RSI fade / VWAP cut / 構造崩れ / ATR spike / レンジ復帰 など）。
- 追加ガード: テクニック負け決済は `*_EXIT_TECH_NEG_MIN_PIPS` + `*_EXIT_TECH_NEG_HOLD_SEC` + スコア閾値で早すぎを抑制し、`*_EXIT_TECH_NEG_HARD_PIPS` で過大マイナスを上限化。

## 対策方針（ヘッジと損切りの両輪）
- 時間×深さで損切り: open>24h かつ PnL<0 なら reduce-only、>72h なら強制縮小/クローズ。
- 含み損スタック制限: 同一戦略の含み損ポジが N 件以上なら新規停止（reduce-only は許可）。
- net/gross 圧縮ルール: net が小さくても gross が大きい状態は、順次縮小してコストを抑える。
- pocket/タグ整合性: client_id と pocket tag の不一致は exit_worker の対象外になり得るため、分類ルールの統一を優先。
- 監視指標の追加: open_trades_neg_count / neg_sum / age_p95 を metrics に記録し、常時含み損化を検知。

## 具体的なイメージ
- NAV 50万円、証拠金率4%の場合、片側だけなら約50万×0.92/（価格×0.04）で ~7.3万 units ほど。ロングとショートを組み合わせれば net が小さくなるため、総建玉は倍近くまで持てる。
- 例: 45kショート保有時に50kロングを入れると net +5k → 必要証拠金は ~3.1万円まで低下し、余力が回復する。

## 推奨設定
- 確保: `ALLOW_HEDGE_ON_LOW_MARGIN=1`（既定）。
- 必要なら緩和: `MIN_FREE_MARGIN_RATIO` を 0.005–0.008 に下げてヘッジ方向を通しやすくする。
- スプレッドが原因で止まる場合: `FAST_SCALP_MAX_SPREAD_PIPS` を 1.1–1.2p 目安に調整して検証。
- stale 緩和: `FAST_SCALP_MAX_SIGNAL_AGE_MS=6000`（遅い tick feed でもエントリー可能）。

## 追加案: 押し目/戻りカウンター（ロング/ショート両対応）
- ロング時は「押し目」、ショート時は「戻り」を同一ロジックでカウント。
- 定義例: 直近高値/安値からの逆行幅が `PULLBACK_PIPS` 以上、かつ直近の順行幅が `TREND_CONFIRM_PIPS` 以上で 1 カウント。
- 窓: `PULLBACK_WINDOW_SEC` 内でのみ加算し、時間超過で減衰またはリセット。
- リセット条件: 新高値/新安値更新、または `PULLBACK_RESET_SEC` 経過。
- 使い道: `counter >= N` でヘッジ方向の新規許可、`counter` 上昇でヘッジロットを段階減衰、急増時は reduce-only を優先。

## 追加であると良い候補
- 逆行連続カウンター: 逆行が連続したら新規ヘッジ停止/縮小（過剰両建て回避）。
- ヘッジ・サイズの段階係数: `free_margin_ratio` だけでなく net_units/ATR も加味して縮小。
- 時間帯フィルタ: 流動性が低い時間はヘッジ禁止 or サイズ半減。
- reduce-only 優先モード: マージン逼迫時は新規ではなく縮小のみ許可。
- spread spike guard: 直近平均比でスプレッドが広いときはヘッジ中止。
- hedge TTL: 一定時間でヘッジ建玉を必ず縮小/クローズ。
- 同方向取り残しガード: 同一ワーカーの同方向オープン数/含み損が閾値超なら新規停止。
- 距離が十分離れた場合の例外: `stack_reentry_pips` 超の逆行なら soft cap を一時的に許可（hard cap は維持）。
- 妥当性チェック: `stack_reentry_require_tech=true` の場合、MTF テクニック（fib/nwave）で OK のときだけ例外を許可。

## 運用フロー
1) ポジション取得: `/v3/accounts/{id}/openPositions` から long/short units を取得。
2) ロット計算: `allowed_lot(..., side, open_long_units, open_short_units)` でエントリー方向と net 後の証拠金を考慮。
3) ゲート確認: spread / stale / pattern / cooldown を通過できるよう閾値を調整。ログで `fast_scalp_skip` / `signal_stale` を監視。
4) 継続監視: `orders.db` で filled が動くか確認。`metrics` に `account.margin_usage_ratio` / `free_margin_ratio` を記録済み。

## 注意点
- DIR_CAP_RATIO（同方向キャップ）は既定のまま。 gross 全体を抑えたいときだけ調整。
- cap=0.92+安全係数で「総裁量が過大になる」ケースは抑制されるが、急変時のマージンコールに注意。

## 追い過ぎ防止（entry guard / MTF）検証メモ
- 発注前の preflight で `evaluate_entry_guard` を必ず実行（reduce_only / manual を除く）。
- MTF は `ENTRY_GUARD_TFS` で複数 TF を評価、`ENTRY_GUARD_MTF_MIN_BLOCKS` でブロック判定（M1/H1/H4 の整合数=align_count）。
- ガードは range/fib + trend/pullback + soft allow を併用。ADX が閾値以上ならバイパス可。
- 直近 200 件クローズ検証の傾向: H1+H4 一致の平均 +3.79p、H1 ADX>=20 の平均 +3.50p（<20 は +1.48p）、H1 MA20 距離は勝ち 0.92 ATR/負け 1.36 ATR。
- 期待値改善ゲート案（戦略別）: Donchian5=align>=2、M1Scalper=align>=1 & 過熱除外、TrendMAbu=MA20 距離<=0.8 ATR、MomentumB=align>=2、ImpulseRe=align>=1 & ADX 25-35、MomentumP は原則ブロック。
- 追加ゲート（env）: `ENTRY_GUARD_ALIGN_MIN_*` / `ENTRY_GUARD_OVERHEAT_BLOCK_*` / `ENTRY_GUARD_ADX_RANGE_*` / `ENTRY_GUARD_MA20_GAP_ATR_MAX_*` を戦略別に設定。debug に align_count/RSI/ADX/MA20 gap を出す。
- ブロック集計: `python scripts/report_entry_guard.py --days 7 --group-base`
- VMログで解析する場合: `python scripts/report_entry_guard.py --orders-db remote_logs/orders.db`
- レポート出力: 戦略別/ポケット別の block rate は `--min-total` で小サンプルを除外。
- 追加出力: reason 別の p25/median/p75（edge_pips / distance_pips / rsi / adx / gap_atr）。
- 推奨レンジ: reason 別の候補閾値（25/50/75%のブロック許容）も出力。
- さらに: ブロックが多い戦略トップに対して、strategy別の推奨レンジも出力。
- VM集計メモ（2026-01-17）: guard_total=1463、blocked_rate=60.1%、極端ブロックは extreme_long が 96%。
- Top block 例: TrendMA-bull / Donchian55-breakout-up / MomentumBurst-open_long / MicroPullbackEMA。trend bypass の ADX_MIN を個別に下げて緩和（env 追記済み）。
- VM集計メモ（2026-01-17, 直近14日）:
  - M1/H4 指標取得率: 965/966。MTF は実運用で評価済み。
  - M1Scalper: RSI 70/30 ブロックで avg=4.24p（n=333）、65/35 で avg=4.15p（n=267）。
  - TrendMA: MA20 gap<=1.2ATR で avg=9.63p（n=56）、<=0.8ATR で avg=12.23p（n=29）。
  - TrendMAbu: MA20 gap<=1.2ATR で avg=6.10p（n=13）。
  - ImpulseRe: base avg=-12.71p → H4 ADX>=20 で avg=+1.05p（n=2）。
  - MomentumPulse: PF=0.52 / win=0.42（n=12）で期待値悪化。perf_guard で自動ブロック対象。
- 反映方針:
  - `ENTRY_GUARD_ADX_RANGE_MIN_IMPULSERE=20` / `ENTRY_GUARD_ADX_TF_IMPULSERE=H4`。
  - `ENTRY_GUARD_MA20_GAP_ATR_MAX_TRENDMA=1.2` / `ENTRY_GUARD_MA20_GAP_TF_TRENDMA=M1`。
  - `ENTRY_GUARD_MA20_GAP_ATR_MAX_TRENDMABU=1.2` / `ENTRY_GUARD_MA20_GAP_TF_TRENDMABU=M1`。
- `PERF_GUARD_GLOBAL_ENABLED=1` + `PERF_GUARD_LOOKBACK_DAYS=14` + `PERF_GUARD_MIN_TRADES=10`（MomentumPulse のみブロック）。

## マイナス決済の許可方針（戻り期待値ベース）
- exit_worker が理由を出した場合、`pnl <= 0` でも close を許可（`allow_negative` 経由）。
- 戻り期待が低いときは `section_exit` / `tech_exit` / `reversion_failure` で負けでも切る。
- 損切り後は `reentry_guard` で同一ワーカーの再入を抑制し、別条件の再エントリーを優先。

## 実装済みヘッジワーカー (HedgeBalancer)
- 役割: マージン使用率が高まったときに逆方向の reduce-only シグナルを `signal_bus` 経由で main 関所へ送り、ネットエクスポージャを軽くする。ファイル: `workers/hedge_balancer/worker.py`。
- トリガー: `margin_usage >= 0.88` または `free_margin_ratio <= 0.08` かつ net_units≥15k。ターゲット使用率は 0.82、ネット削減量は net の 55% 以内で最小 10k、最大 90k units。
- シグナル内容: pocket=`macro`、confidence=90、sl/tp=5pips（reduce_only 前提で sl/tp は添付されない）、proposed_units を明示して reduce_only+reduce_cap で送信。
- 連続発火ガード: 20s クールダウン。価格は直近 tick mid（5s 窓）を使用、価格が MIN_PRICE(<90) ならスキップ。
- 環境変数: `HEDGE_BALANCER_ENABLED` / `HEDGE_TRIGGER_MARGIN_USAGE` / `HEDGE_TARGET_MARGIN_USAGE` / `HEDGE_TRIGGER_FREE_MARGIN_RATIO` / `HEDGE_MIN_NET_UNITS` / `HEDGE_MIN_HEDGE_UNITS` / `HEDGE_MAX_HEDGE_UNITS` / `HEDGE_MAX_REDUCTION_FRACTION` / `HEDGE_COOLDOWN_SEC` / `HEDGE_POCKET` / `HEDGE_CONFIDENCE` / `HEDGE_SL_PIPS` / `HEDGE_TP_PIPS` / `HEDGE_MIN_PRICE`。
- 起動例: `python -m workers.hedge_balancer.worker`（既定で有効。ログに `[HEDGE] enqueue dir=...` が出れば発火）。
