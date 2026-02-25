# 型（Kata）設計: `scalp_ping_5s`（5秒スキャ）

このドキュメントは、`workers/scalp_ping_5s` の「型（Kata）」を **再現可能に作る**ための設計書です。
型は、過去トレードから抽出したパターンを DB/JSON に継続学習し、次のエントリーで `block / reduce / boost`（拒否/縮小/増強）に接続します。

## ゴール
- 5秒スキャ特有の「追いかけ過ぎ（高値掴み/底値掴み）」「薄いエッジ」「逆行しやすい地合い」を、履歴から定量化して回避・縮小する。
- 人間の裁量に依存せず、同じ条件なら同じ判断をする（再現性）。
- 型は戦略ごとに最適化する（全戦略共通の一律型にしない）。

## 1. 全体フロー（VM常時運用）
1. `logs/trades.db` のクローズ済みトレードを材料にする
2. `scripts/pattern_book_worker.py` が `logs/patterns.db` を更新
3. 同時に deep 分析（`analysis/pattern_deep.py`）で `pattern_scores / drift / cluster` を更新
4. `execution/order_manager.py` の preflight で `workers/common/pattern_gate.py` を評価
5. opt-in のエントリーに限り、型に応じて `block/scale` する

systemd（VM）:
- `systemd/quant-pattern-book.service`
- `systemd/quant-pattern-book.timer`（5分周期）

重要:
- `quant-pattern-book.service` は必ず `/home/tossaki/QuantRabbit/.venv/bin/python` を使う（`numpy/pandas/scipy/sklearn` が必要）。

## 2. 何を「型」と呼ぶか（pattern_id）
型の最小単位は `pattern_id`（文字列）です。トレードは必ずどれか 1 つの `pattern_id` に割り当てられます。

`pattern_id` は `analysis/pattern_book.py:build_pattern_id()` で生成され、次のトークンを `|` 区切りで連結します。

- `st:` strategy tag（戦略名）
- `pk:` pocket（資金/口座内の区分）
- `sd:` direction（long/short）
- `sg:` signal mode（5秒スキャの「モード」）
- `mtf:` MTF gate（マルチタイムフレームの地合いゲート）
- `hz:` horizon gate（複数ホライズン整合ゲート）
- `ex:` extrema gate reason（高値/安値寄り回避ゲート）
- `rg:` range bucket（レンジ内の位置; 後述）
- `pt:` pattern tag（戦略が任意で付ける追加タグ）

この `pattern_id` は次の 2 箇所で必ず一致させます。
- パターン収集: `scripts/pattern_book_worker.py`（`trades.db` -> `patterns.db`）
- エントリー制御: `workers/common/pattern_gate.py`（発注前に同じ `pattern_id` を作る）

## 3. `scalp_ping_5s` の「型」を構成する情報源（entry_thesis）
5秒スキャは、エントリー直前に `entry_thesis` を組み立てて `market_order(..., entry_thesis=...)` に渡します。

起点:
- `workers/scalp_ping_5s/worker.py`（`entry_thesis = {...}`）

`pattern_id` に使われる（=型を分岐させる）主要キー（5秒スキャ由来）:
- `signal_mode`（例: `momentum`, `revert`, `momentum_cont`, `revert_hz` など）
- `mtf_regime_gate`（例: `mtf_continuation_follow`, `mtf_reversion_to_trend` など）
- `horizon_gate`（例: `horizon_align`, `horizon_counter_scaled` など）
- `extrema_gate_reason`（例: `ok`, `long_top_soft`, `short_bottom_soft`, `short_bottom_soft_balanced` など）

`pattern_trade_features` に保存され deep 分析に使われるキー（抜粋）:
- `confidence` / `spread_pips` / `tp_pips` / `sl_pips`
- `hold_sec`（open/close から算出）
- `pl_pips`（勝敗/収益性）

## 4. DB/JSON 出力（VM）
主な出力は次の通りです。

- DB: `/home/tossaki/QuantRabbit/logs/patterns.db`
- JSON: `/home/tossaki/QuantRabbit/config/pattern_book.json`（基本集計 + top edges）
- deep JSON: `/home/tossaki/QuantRabbit/config/pattern_book_deep.json`（deep 結果のダイジェスト）

主要テーブル:
- `pattern_trade_features`: 取引1件ごとの特徴（pattern_id と特徴量）
- `pattern_stats` / `pattern_actions`: ざっくり集計（deep が未稼働でも動く）
- `pattern_scores`: deep の主表（品質/スコア/倍率/統計）
- `pattern_drift`: 直近劣化（ドリフト）検出
- `pattern_clusters` / `pattern_cluster_summary`: 類似型のクラスタ

## 5. deep 分析の中身（5秒スキャの「型」を強くする）
deep（`analysis/pattern_deep.py`）は以下の目的で動きます。

- 少サンプルを過信しない（ベイズ的に縮める）
- 「偶然の当たり」を弾く（統計検定 + 重み）
- 直近劣化を捕まえる（ドリフト）
- 似た型をまとめて俯瞰する（クラスタ）

主な指標:
- `shrink_avg_pips`: 小標本を縮めた平均pips（過信防止）
- `z_edge`: pocket+direction の基準平均との差を標準化したエッジ
- `p_value`: pocket+direction 基準の勝率との差（binomtest）
- `robust_score`: サンプル数/有意性/追いかけ罰則を統合したスコア
- `suggested_multiplier`: `robust_score` を S 字に変換した推奨倍率（0.65-1.35 の範囲）
- `quality`: `robust/candidate/neutral/weak/avoid/learn_only`
- `boot_ci_low/high`: 平均pipsの bootstrap CI（粗いが「型の不確実性」を見る）

ドリフト:
- 直近 `recent_days` と、それ以前 `baseline_days` の分布差（Mann-Whitney U）
- `deterioration/soft_deterioration` を Pattern Gate の縮小に反映可能

## 6. エントリー制御（Pattern Gate）の仕様
実装:
- `workers/common/pattern_gate.py`
- 呼び出し元: `execution/order_manager.py`（preflight）

重要: **opt-in したエントリーだけに適用**
- グローバル強制: `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=1`（通常は 0 を維持）
- `scalp_ping_5s` 側: `entry_thesis["pattern_gate_opt_in"]=...` を明示（デフォルト true）

ブロック:
- `quality in ORDER_PATTERN_GATE_BLOCK_QUALITIES`（既定: `avoid`）
- `trades >= ORDER_PATTERN_GATE_BLOCK_MIN_TRADES`（既定: 90）
- `robust_score <= ORDER_PATTERN_GATE_BLOCK_MAX_SCORE`（既定: -0.9）
- `p_value <= ORDER_PATTERN_GATE_BLOCK_MAX_PVALUE`（既定: 0.35）

スケール:
- `suggested_multiplier` をベースに `ORDER_PATTERN_GATE_SCALE_MIN/MAX` へクリップ
- `weak` などの品質では「縮小寄り」に倒す（`ORDER_PATTERN_GATE_REDUCE_FALLBACK_SCALE`）
- ドリフト悪化時は縮小上限を下げる（`ORDER_PATTERN_GATE_DRIFT_*`）

無効化（即時ロールバック）:
- `ORDER_PATTERN_GATE_ENABLED=0` または `SCALP_PING_5S_PATTERN_GATE_OPT_IN=0`
- 既存ポジを勝手に決済することはない（エントリー前ゲートのみ）

## 7. 5秒スキャの型を「完璧」に近づける設計ルール
5秒スキャは microstructure（スプレッド/ティック密度/遅延）に支配されやすく、型の設計が雑だと次のどちらかになります。
- 型が細かすぎる: パターンが分裂して永遠に `learn_only`
- 型が粗すぎる: 「高値掴み/底値掴み」など重要な差が混ざって平均化される

基本方針（推奨）:
- `pattern_id` は **低カーディナリティ**（各トークンは 10-20 値程度が上限）
- 連続値（pips や score）の生値を `pattern_id` に入れない（入れるなら必ずバケット化）
- 5秒スキャで効く軸を優先する:
  - `signal_mode`（momentum vs revert）
  - `mtf_regime_gate`（継続/反転、順張り/逆張り）
  - `horizon_gate`（整合の強さ）
  - `extrema_gate_reason`（高値/安値寄りの回避）

「高値掴み/底値掴み」を型に落とす（最重要）:
- `analysis/pattern_deep.py` の `chase_risk_rate` は `rg:`（range bucket）を使う。
- `scalp_ping_5s` を完璧にするなら、`entry_thesis` に **`entry_range_bucket` を埋める**（例: `bot/low/mid/high/top`）。
  - 既存の `extrema_m1_pos / extrema_m5_pos`（0-1 の位置）を 5 分割して `entry_range_bucket` に入れるのが手堅い。
  - これが無いと `rg:na` になり、追いかけ判定が学習できない。

追加で効く可能性が高い（ただし増やしすぎ注意）:
- `pt:`（pattern_tag）に 1 つだけ追加タグを載せる
  - 例: `lookahead_reason` を縮約したタグ（`edge_ok/edge_thin_scale` など）
  - 例: direction bias の整合/逆行だけ（`bias_align/bias_counter`）

## 8. 運用チェック（VM）
pattern book が効いているかを確認する時は、必ず VM の実データで見る。

systemd:
- `systemctl status quant-pattern-book.timer`
- `journalctl -u quant-pattern-book.service -n 200 --no-pager`

DB:
- `sqlite3 /home/tossaki/QuantRabbit/logs/patterns.db "..."`
- `pattern_scores` で `strategy_tag like 'scalp_ping_5s%'` を絞って `avoid/weak` を確認

発注ログ:
- `execution/order_manager.py` が `pattern_block` / `pattern_scale_below_min` などのステータスを入れる（orders.db の status/skip_reason 側で追える）。

## 9. 変更ルール（重要）
- 型の設計変更（`pattern_id` の構成要素変更）は **互換性が壊れる**（古い pattern_id と別物になる）。
  - 変更するなら「なぜ」「何を」「どうバケット化するか」を必ず記録し、反映直後は `learn_only` が増える前提で運用する。
- 本番反映は `main` 統合 -> `git push origin main` -> `scripts/vm.sh ... deploy -b main -i --restart quantrabbit.service -t` の手順を守る。

## 10. 2026-02-17 更新（無印導線削除）

- 5秒スキャは `scalp_ping_5s_b_live` のみ運用対象。
- 無印向けの rapid mode / autotune スクリプトと専用envは削除済み。
- 現在の運用パラメータは `ops/env/scalp_ping_5s_b.env` を正本として管理する。

## 11. 2026-02-17 更新（下落継続ショート取り逃し対策）

- `workers/scalp_ping_5s` の extrema 合意判定を side別に分離。
  - `SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_LONG`
  - `SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT`
- B版（`ENV_PREFIX=SCALP_PING_5S_B`）では short 側の既定を
  `SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=true` に設定。
- B運用env（`ops/env/scalp_ping_5s_b.env`）は以下を追加。
  - `SCALP_PING_5S_B_MIN_UNITS=300`
- 意図:
  - `short_bottom_m1m5` の M1 単独 block を抑え、下落継続時のショート再エントリーを回復する。
  - `units_below_min` での空振りを減らし、`entry_intent` が order_manager へ到達する率を引き上げる。

## 12. 2026-02-17 更新（極値反転ルーティング追加）

- 目的:
  - エントリー件数を落とさずに、底/天井付近の同方向積み上げを減らす。
  - `extrema` が強く出た時に block ではなく反転 side へルーティングして、方向精度を上げる。
- 実装:
  - `workers/scalp_ping_5s/config.py`
    - `EXTREMA_REVERSAL_*` を追加。
    - B版（`ENV_PREFIX=SCALP_PING_5S_B`）は `EXTREMA_REVERSAL_ENABLED` を既定ON。
  - `workers/scalp_ping_5s/worker.py`
    - `_extrema_reversal_route()` を追加。
    - `short_bottom_*` / `long_top_*`（+ `short_h4_low`）で、
      M1/M5/H4の位置、M1の `RSI/EMA`、`MTF heat`、`horizon` を点数化し、
      閾値到達時は `signal.side` を反転して継続。
    - `entry_thesis` に `extrema_reversal_applied`, `extrema_reversal_score` を追加し、
      実運用の追跡を可能化。
- 運用意図:
  - 「極値で止める」よりも「極値で方向転換する」設計へ寄せることで、
    発注本数を維持しつつ bottom/top 近傍の逆行エントリーを抑える。

## 13. 2026-02-18 更新（signal window 可変化 + shadow 計測）

- 追加設定（`workers/scalp_ping_5s/config.py`）:
  - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_ENABLED`
  - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED`
  - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC`
  - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES`
  - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS`
- 実装:
  - `_build_tick_signal(...)` に window override を追加し、候補窓を同一ティック列で評価可能化。
  - `trades.db` の `entry_thesis.signal_window_sec` を戦略タグ単位で集計し、
    side/mode と窓幅近傍で期待値を推定。
  - `adaptive=off` 時は既存窓を維持しつつ shadow 比較のみログ出力。
  - `adaptive=on` 時のみ、十分サンプル + マージン超過時に窓を切替。
- 監査キー（`entry_thesis`）:
  - `signal_window_adaptive_applied`
  - `signal_window_adaptive_live_sec`
  - `signal_window_adaptive_selected_sec`
  - `signal_window_adaptive_best_sec`
  - `signal_window_adaptive_best_sample`
- 運用方針:
  - 先に shadow で分布を確認し、`best_sample` と `best_score` の安定後に `ADAPTIVE_ENABLED=1` を段階適用する。

## 14. 2026-02-18 更新（direction_flip de-risk の close reason 正規化）

- 対象:
  - `workers/scalp_ping_5s/exit_worker.py`
  - `workers/scalp_ping_5s_b/exit_worker.py`
  - `config/strategy_exit_protections.yaml`
- 変更:
  - de-risk 内部判定の sentinel `__de_risk__` は内部制御専用とし、
    部分クローズ失敗時の full close は `direction_flip.reason`（既定: `m1_structure_break`）にフォールバック。
  - `scalp_ping_5s*` の negative close 許可理由に
    `m1_structure_break` / `risk_reduce` を追加。
- 背景:
  - VM で `reason=__de_risk__` の close 失敗連発が発生し、
    負け玉の整理が遅延するケースを確認。
- 運用意図:
  - internal sentinel の外部流出を防ぎ、strict negative gate 下でも
    direction flip 系の損失縮小が実行される状態を維持する。

## 15. 2026-02-19 更新（確率帯ロット再配分）

- 問題:
  - `scalp_ping_5s_b_live` で `entry_probability >= 0.90` 帯の期待値が劣後しているのに、
    ロットが高確率帯へ偏っていた。
  - 逆に `entry_probability < 0.70` 帯は相対優位でもロット不足になり、
    「勝ちで小、負けで大」の逆配分が発生していた。
- 実装:
  - `workers/scalp_ping_5s/worker.py`
    - `EntryProbabilityBandMetrics` を追加。
    - `_load_entry_probability_band_metrics()` で side別に
      `mean_pips/win_rate/sl_rate` を帯別集計（lookbackベース）。
    - `_entry_probability_band_units_multiplier()` を追加し、
      帯間ギャップ（pips/win/sl）とサンプル強度で
      `high縮小 / low増量` を決定。
    - side別 `SL hit` と `MARKET close +` の比率を `side_mult` に反映。
    - 最終ロット計算に `probability_band_units_mult` を適用。
  - `workers/scalp_ping_5s/config.py`
    - `ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
  - `ops/env/scalp_ping_5s_b.env`
    - B本番用の `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
- 監査キー（`entry_thesis`）:
  - `entry_probability_band_units_mult`
  - `entry_probability_band_allocation.{reason,bucket,band_mult,side_mult,units_mult,...}`
- 運用意図:
  - エントリー件数を維持したまま、損失帯へのロット集中を抑える。
  - 有利帯へロットを寄せ、同一頻度でのPL効率を改善する。

## 16. 2026-02-20 更新（方向転換の反応速度を優先）

- 背景（VM実測）:
  - `2026-02-20 01:39 JST` 時点の直近クローズで
    - long: `STOP_LOSS_ORDER` 2件 `-25.5`
    - short: `MARKET_ORDER_TRADE_CLOSE` 5件 `+12.9`
  - 高頻度帯で「方向転換の遅れ」と「逆方向ロット偏重」が再発。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - 確率帯配分の反応を短期化:
    - `ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES=120`
    - `...MIN_TRADES_PER_BAND=14`
    - `...HIGH_REDUCE_MAX=0.78`
    - `...LOW_BOOST_MAX=0.50`
    - `...SAMPLE_STRONG_TRADES=30`
  - side成績配分を即応化:
    - `...SIDE_METRICS_LOOKBACK_TRADES=36`
    - `...SIDE_METRICS_GAIN=1.35`
    - `...SIDE_METRICS_MIN_MULT=0.40`
  - flip発火を前倒し:
    - `FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS=0.08`
    - `FAST_DIRECTION_FLIP_COOLDOWN_SEC=0.6`
    - `SL_STREAK_DIRECTION_FLIP_MIN_STREAK=1`
    - `SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP=1`
    - `SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS=1`
    - `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=2`
    - `SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM=0`
    - `SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.48`
    - `SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.30`
  - 極値由来のショート遅れを緩和:
    - `EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=0`
    - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`
- 運用意図:
  - エントリー頻度は維持しつつ、逆方向の過大ロットを早期に絞る。
  - SL発生後の side 反転までの待ち時間を短縮し、取り返しの遅れを減らす。

## 17. 2026-02-20 更新（side実績フリップ導入）

- 目的:
  - 「特定sideでSLが連続しているのに反転が遅い」局面を、連敗数だけでなく
    side実績そのもの（SL率・成り行き利確率）で検知して即時反転する。
- 実装:
  - `workers/scalp_ping_5s/worker.py`
    - `_maybe_side_metrics_direction_flip()` を追加。
    - 条件:
      - 現在sideの `SL率` が閾値以上
      - 現在sideと反対sideの `SL率` 差が閾値以上
      - 反対sideの `MARKET_ORDER_TRADE_CLOSE(+PL)` 率が優位
      - 最低サンプル数を満たす
    - 条件成立時は `signal.side` を opposite side へリターゲットし、
      `mode` に `_smflip` を付与して追跡可能化。
  - `workers/scalp_ping_5s/config.py`
    - `SIDE_METRICS_DIRECTION_FLIP_*` の設定群を追加。
  - `ops/env/scalp_ping_5s_b.env`
    - B運用値を反映（lookback 36, min trades 8/6, min SL rate 0.58 等）。
- 監査:
  - `entry_thesis.side_metrics_direction_flip_*`
  - `tech_route_reasons` に `side_metrics_flip`
- 運用意図:
  - 頻度を落とさず、負けsideへの連続エントリーを短時間で打ち切る。

## 18. 2026-02-20 更新（逆行スタック時のロット圧縮）

- 背景:
  - 同方向の建玉が短時間に積み上がる局面で、反転前に `STOP_LOSS_ORDER` が
    クラスター化して損失が拡大するケースが残った。
- 実装:
  - `workers/scalp_ping_5s/worker.py`
    - `_side_adverse_stack_units_multiplier()` を追加。
    - 以下を同時に満たすときに、エントリーは維持したままロットだけ圧縮:
      - 現在sideの `SL率` が高く、反対sideより劣後
      - 反対sideの `MARKET_ORDER_TRADE_CLOSE(+PL)` 率が優位
      - 同方向の open trades が閾値以上
      - 同方向の含み損DD（pips）が閾値を超過
    - `units` 計算チェーンに `side_adverse_stack_units_mult` を追加。
  - `workers/scalp_ping_5s/config.py`
    - `SIDE_ADVERSE_STACK_UNITS_*` / `SIDE_ADVERSE_STACK_DD_*` /
      `SIDE_ADVERSE_STACK_LOG_INTERVAL_SEC` を追加。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_*` を有効化。
- 監査キー（`entry_thesis`）:
  - `side_adverse_stack_units_mult`
  - `side_adverse_stack_reason`
  - `side_adverse_stack_metrics_adverse`
  - `side_adverse_stack_side_mult`
  - `side_adverse_stack_dd_mult`
  - `side_adverse_stack_dd_pips`
  - `side_adverse_stack_active_same_side`
  - `side_adverse_stack_current_sl_rate` / `...target_sl_rate`
- 運用意図:
  - エントリー頻度を落とさず、逆行局面の同方向ロット過多だけを抑える。

## 19. 2026-02-24 更新（リプレイ stale 判定修正 + ルートWFO導線）

- 背景:
  - replay 実行時、`_build_tick_signal()` が `time.time()`（実時間）で tick age を判定するため、
    過去日付ティックが `stale_tick` になり `trades=0` へ落ちる事象があった。
- 修正:
  - `scripts/replay_exit_workers.py`
    - `sim_clock` を ping worker/exit module の `time.time` と `time.monotonic` へ注入する
      パッチを追加（`_patch_module_clock`, `_patch_ping_runtime_clock`）。
    - replay main 起動時に patch を適用し、過去ティックでも signal 判定が再現可能になるよう補正。
- 追加導線:
  - `scripts/replay_regime_router_wfo.py` を追加し、
    `regime_route`（trend/breakout/range/mixed/event/unknown）ごとに
    C/D の walk-forward マッピングを算出可能化。
  - replay trade には `macro_regime` / `micro_regime` / `regime_route` を保持して
    ルート別集計に接続。
- 運用メモ:
  - replay 実行時は `SCALP_REPLAY_MODE=scalp_ping_5s_[c|d]` と
    `SCALP_REPLAY_ALLOWLIST=scalp_ping_5s_[c|d]` を明示し、
    `spread_revert` 既定に落ちないようにする。

## 20. 2026-02-24 更新（D narrow worker: short + allow_jst_hours）

- 背景（実リプレイ, `--sp-live-entry --exclude-end-of-replay`）:
  - `scalp_ping_5s_d` の全時間 short-only は依然マイナスだったが、
    時間帯を `allow_jst_hours=1,10` へ絞ると day23/day26 ともプラスを確認。
- 変更:
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_ALLOW_HOURS_JST` を追加。
  - `workers/scalp_ping_5s/worker.py`
    - `ALLOW_HOURS_JST` を entry gate に追加。
    - `outside_allow_hour_jst` で許可時間外を skip する判定を実装。
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_SIDE_FILTER=short`
    - `SCALP_PING_5S_D_ALLOW_HOURS_JST=10`
    - `SCALP_PING_5S_D_BLOCK_HOURS_JST=`
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=9000`
    - `SCALP_PING_5S_D_MAX_UNITS=9000`
    - `SCALP_PING_5S_D_MAX_ACTIVE_TRADES=1`
    - `SCALP_PING_5S_D_MAX_PER_DIRECTION=1`
  - `scripts/replay_exit_workers.py`
    - `SCALP_REPLAY_ALLOW_JST_HOURS` / `SCALP_REPLAY_BLOCK_JST_HOURS` 未指定時は、
      Dプレフィックス側（`SCALP_PING_5S_D_ALLOW_HOURS_JST` /
      `SCALP_PING_5S_D_BLOCK_HOURS_JST`）をフォールバックして
      replay と live の時間帯条件を一致させる。
- 検証:
  - `allow=10, units=9000`（day26）:
    - `+2104.70 JPY` / `42 trades`
    - `jpy_per_hour(active)=+2079.16`
    - `max_drawdown_jpy=2589.66`
- 注意:
  - C/D ルーター混在では C 側寄り配分で悪化するため、
    D narrow の評価は D 単独導線で実施する。

### 20.1 2026-02-25 更新（当日ティック再最適化）

- 対象:
  - `logs/replay/USD_JPY/USD_JPY_ticks_20260225.jsonl`（VM実ログを取得）
- 条件:
  - `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`
- 単体窓の結果（D）:
  - `short_allow9`: `-1974.52 JPY`
  - `short_allow10`: `-2809.12 JPY`
  - `short_allow11`: `+153.00 JPY`
  - `long_allow10`: `-1168.54 JPY`
  - `both_allow11`: `+190.12 JPY`（最良）
- 反映方針:
  - `SCALP_PING_5S_D_ALLOW_HOURS_JST=11`
  - `SCALP_PING_5S_D_SIDE_FILTER=`（両方向）
  - `10時(JST)` は当日損失源として除外。

### 20.2 2026-02-25 更新（11時窓での units sweep + C停止）

- 追加検証（同日ティック）:
  - `allow=11`, `side=both`, `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`
  - `base/max=9000`: `+190.12 JPY`
  - `base/max=12000`: `+253.49 JPY`
  - `base/max=15000`: `+316.87 JPY`
- 採用:
  - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=15000`
  - `SCALP_PING_5S_D_MAX_UNITS=15000`
- 併行措置:
  - `scalp_ping_5s_c_live` は直近90分で `-671.68 JPY` と悪化していたため、
    `SCALP_PING_5S_C_ENABLED=0` で entry を停止。

### 20.3 2026-02-25 更新（Dのorder_manager二重縮小を解除）

- 症状:
  - 実運用ログで `entry_probability_below_min_units` → `order_manager_none` が発生し、
    Dのエントリーがほぼ通らない状態が継続。
- 対応:
  - D専用の preserve-intent 係数を固定化（追加縮小なし）:
    - `REJECT_UNDER=0.35`, `MIN_SCALE=1.00`, `MAX_SCALE=1.00`
  - D専用 `ORDER_MIN_UNITS` を `30` へ引下げ。
  - `scalp_ping_5s_d.env` と `quant-order-manager.env` の両方に同値を設定し、
    worker fallback 経路と service 経路の判定差を解消。
- 再生検証:
  - `allow=11`, `side=both`, `units=15000` で `+316.87 JPY` を維持（悪化なし）。

### 20.4 2026-02-25 更新（Dの詰まり解消: min_units + perf_guard mode）

- 背景（VM実績, 直近1h）:
  - `scalp_ping_5s_c_live`: `35 trades`, `-671.7 JPY`
  - `scalp_ping_5s_d_live`: `3 trades`, `-28.5 JPY`
  - D は `units_below_min` / `perf_block` で通過件数が不足。
- 検証（当日ティック replay, allow=11, side=both, units=15000）:
  - baseline: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
  - 候補A（`MIN_UNITS=30`, `PERF_GUARD_MODE=reduce`）: `+316.87 JPY`（同値）
  - 候補B（上記 + `MIN_TICKS=3`, `MIN_SIGNAL_TICKS=2`, `SHORT_MIN_SIGNAL_TICKS=3`）:
    `+181.87 JPY`（悪化）→ 不採用
- 採用:
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_MIN_UNITS=30`
    - `SCALP_PING_5S_D_PERF_GUARD_MODE=reduce`
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_D_PERF_GUARD_MODE=reduce`
- 意図:
  - replay の優位性を維持しつつ、live での `units_below_min` と
    `perf_block` 起因の取りこぼしを減らす。

### 20.5 2026-02-25 更新（WFO 3日比較 + units 20%増）

- WFO比較（`train2/test1` 相当, 2/13+2/19 -> 2/25）:
  - `h11_both`: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
  - `h11_short`: `+255.00 JPY`（`3 trades`）
  - `h11_long`: `-1547.86 JPY`
  - `h10,11_both`: `-5522.09 JPY`
  - `h9,11_both`: `-7999.04 JPY`
  - `h10,11_short`: `-4426.87 JPY`
- 判定:
  - 時間帯拡張（10時/9時追加）は大幅悪化のため不採用。
  - 方向は `both` を維持（`short-only` より総益が高い）。
- units スイープ（2/25, `h11_both`）:
  - `15000`: `+316.87 JPY`
  - `18000`: `+380.24 JPY`
  - `22000`: `+464.74 JPY`
  - `30000`: `+633.73 JPY`
- 採用:
  - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=18000`
  - `SCALP_PING_5S_D_MAX_UNITS=18000`
- 意図:
  - 条件（11時/both）は固定し、過去再生で悪化を出さずに
    期待値を拡張できる最小増分として +20% を適用。

### 20.6 2026-02-25 更新（Dバーンイン自動判定ガード）

- 目的:
  - `u=18000` 運用中の 1-2h バーンイン判定を定型化し、
    `promote / hold / rollback` を機械的に返す。
- 実装:
  - `scripts/ping5s_d_canary_guard.py`
  - 入力:
    - `logs/trades.db` / `logs/orders.db`
    - `ops/env/scalp_ping_5s_d.env`
  - 判定指標（既定）:
    - `jpy_per_hour > 0`
    - `trades_per_hour >= 6`
    - `margin_reject_count == 0`
  - 分岐:
    - `promote` -> `target_units=22000`
    - `hold` -> 現行維持
    - `rollback` -> `target_units=15000`
- 例:
```bash
.venv/bin/python scripts/ping5s_d_canary_guard.py \
  --window-minutes 120 \
  --out logs/ping5s_d_canary_guard_latest.json
```
