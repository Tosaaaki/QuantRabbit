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
