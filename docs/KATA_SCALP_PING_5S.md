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
- `extrema_gate_reason`（例: `ok`, `long_top_soft`, `short_bottom_soft` など）

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

## 10. 2026-02-13 追加実装（scalp_ping_5s）: 可変コアリスク化（BASE/SAFE 即時切替）

### 10.1 方針
- 固定値の長所・短所を固定しないため、`base/safe` を切り替える2系統の環境として管理する。
- 運用中に「即時切替」対象にするキーは次を含む:
  - 取引制御: `ORDER_FIXED_SL_MODE`
  - 取りこぼし抑制: `SCALP_PING_5S_ENTRY_COOLDOWN_SEC`, `SCALP_PING_5S_MIN_ORDER_SPACING_SEC`, `SCALP_PING_5S_MAX_ORDERS_PER_MINUTE`, `SCALP_PING_5S_MAX_ACTIVE_TRADES`, `SCALP_PING_5S_MAX_PER_DIRECTION`
- `SCALP_PING_5S` の実運用上の判断は「実績ベースの自動切替」で行う前提にし、短時間統計が悪化したら `SAFE`、改善したら `BASE` に戻す。

### 10.2 反映ファイル
- `scripts/vm_apply_scalp_ping_5s_rapid_mode.sh`
- `ops/env/scalp_ping_5s_rapid_mode_base_20260213.env`
- `ops/env/scalp_ping_5s_rapid_mode_safe_20260213.env`

### 10.3 BASE / SAFE のパラメータ差分（要点）
- BASE
  - `ORDER_FIXED_SL_MODE=0`
  - `SCALP_PING_5S_MAX_ORDERS_PER_MINUTE=96`
  - `SCALP_PING_5S_MAX_ACTIVE_TRADES=40`
  - `SCALP_PING_5S_MAX_PER_DIRECTION=24`
  - `SCALP_PING_5S_ENTRY_COOLDOWN_SEC=0.18`
  - `SCALP_PING_5S_MIN_ORDER_SPACING_SEC=0.10`
- SAFE
  - `ORDER_FIXED_SL_MODE=0`
  - `SCALP_PING_5S_MAX_ORDERS_PER_MINUTE=60`
  - `SCALP_PING_5S_MAX_ACTIVE_TRADES=24`
  - `SCALP_PING_5S_MAX_PER_DIRECTION=12`
  - `SCALP_PING_5S_ENTRY_COOLDOWN_SEC=0.22`
  - `SCALP_PING_5S_MIN_ORDER_SPACING_SEC=0.12`

### 10.4 `--mode auto` の切替ロジック（現行）
- `scripts/vm_apply_scalp_ping_5s_rapid_mode.sh` の `--auto` 判定は、直近 15分（`--window-min` で変更可）の統計を参照。
- 参照指標:
  - `short_sl_rate` / `short_avg`
  - `long_sl_rate` / `long_avg`
  - `overall_sl_rate` / `overall_avg`
  - `trade_count`（対象期間のクローズ数）
- 判定方針:
  - BASE中は短期SL率や平均pips悪化（短・長）で SAFE へ。
  - SAFE中は短・長・全体の指標が戻り基準を満たした場合にストリークを積み、一定連続数（現在2）で BASE へ復帰。

### 10.5 運用チェック（反映確認）
- 適用コマンド:
  - `bash scripts/vm_apply_scalp_ping_5s_rapid_mode.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t --mode base|safe|auto [--window-min 15]`
- VM反映確認:
  - `/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s.env` の `SCALP_PING_5S_RAPID_MODE`
  - 上記コアキー（`ORDER_FIXED_SL_MODE`）が意図どおり入替わっていること
- 重要: 既存ポジションの即時終了は本仕様には含めない（既存ポジ維持前提）。ただし SAFE/BASE 変更の副作用は新規エントリーの抑制・許可条件に反映される。
