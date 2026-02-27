# Trade Findings Ledger (Single Source of Truth)

このファイルは、QuantRabbit の「改善記録」と「敗因記録」の単一台帳です。
以後、同種の記録は必ずここに追記し、他の分散ファイルは作らないこと。

## Rules (Read First)
- 記録先はこのファイルのみ（`docs/TRADE_FINDINGS.md` 固定）。
- 新規の改善/敗因分析を行ったら、必ず 1 エントリ以上追記する。
- 追記順は「新しいものを上」に統一する。
- 事実は VM/OANDA 実測を優先し、日時は UTC と JST を明記する。
- 推測は `Hypothesis` と明示し、事実 (`Fact`) と混在させない。
- 最低限の記載項目:
  - `Period`（集計期間）
  - `Fact`（数値）
  - `Failure Cause`（敗因）
  - `Improvement`（改善施策）
  - `Verification`（確認方法/判定基準）
  - `Status`（open/in_progress/done）

## Entry Template
```
## YYYY-MM-DD HH:MM UTC / YYYY-MM-DD HH:MM JST - <short title>
Period:
- ...

Fact:
- ...

Failure Cause:
1. ...
2. ...

Improvement:
1. ...
2. ...

Verification:
1. ...
2. ...

Status:
- open | in_progress | done
```

## 2026-02-26 12:19 UTC / 2026-02-26 21:19 JST - PDCA深掘り（`perf_block` 固定化 + `orders.db` ロック + SLO劣化の重畳）
Period:
- 監査期間: 直近24h（`2026-02-25 12:11` ～ `2026-02-26 12:19` UTC）
- ロック集中窓: `2026-02-26 11:26` ～ `11:46` UTC（`20:26` ～ `20:46` JST）
- Source: VM `journalctl -u quant-order-manager.service`, `journalctl -u quant-bq-sync.service`, `journalctl -u quant-position-manager.service`, `/home/tossaki/QuantRabbit/logs/{orders,metrics}.db`, `lsof`, `systemctl cat quant-bq-sync.service`

Fact:
- `orders.db` 24h集計（`rows=44862`）:
  - `preflight_start=17121`, `perf_block=16389`, `probability_scaled=7348`, `entry_probability_reject=594`
  - `submit_attempt=545`, `filled=541`（filled/submit `=99.3%`）
  - 戦略別 `perf_block`: `scalp_ping_5s_c_live=7032`, `scalp_ping_5s_b_live=4577`, `scalp_ping_5s_flow_live=3577`, `M1Scalper-M1=577`
- `perf_block` の実ログ内訳（`[ORDER][OPEN_REJECT]` 行）:
  - 合計 `191`
  - `scalp_ping_5s_b_live=157`, `M1Scalper-M1=34`
  - 主因ノート: `perf_block:hard:hour9:failfast:pf=0.56`（88件）, `hour11:failfast:pf=0.15`（36件）, `failfast:pf=0.38`（34件）
- `entry_probability` skip は `RangeFader` 系中心（sell/buy/neutral 合計74件）、`scalp_ping_5s_c_live=5`, `scalp_ping_5s_b_live=2`。
- `database is locked` は 24hで `83` 件。発生分は `35` 分に集中し、最多は `11:31/11:34`（各5件）。
- lock集中分（例 `11:26-11:42`）は `submit_attempt/filled=0` かつ `manual_margin_pressure` 併発分が多く、発注可否判定の再試行だけが増える形になっていた。
- 同時点で `lsof /home/tossaki/QuantRabbit/logs/orders.db` は PID `3400`（`run_sync_pipeline.py --interval 60 --bq-interval 300 --limit 1200`）が `40+` FD を保持。加えて戦略ワーカー PID `682/706` が保持。
- `orders.db` 自体は `6.7G` / `921,528 rows`（`2025-12-29` ～ `2026-02-26`）まで増大。
- `quant-bq-sync.service` は 60秒周期で常時 `sync_trades start`。env は `POSITION_MANAGER_SERVICE_ENABLED=1`, `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1`, `PIPELINE_DB_READ_TIMEOUT_SEC=2.0`。
- `quant-position-manager.service` は 24hで timeout/busy 警告が複数（`fetch_recent_trades timeout`, `sync_trades timeout`, `position manager busy`）。
- `metrics.db` 24h:
  - `data_lag_ms`: `p50=768.9`, `p90=1958.7`, `p95=3603.5`, `p99=13325.1`, `max=225164.5`、閾値超過（`>1500ms`）`16.21%`
  - `decision_latency_ms`: `p50=24.4`, `p90=99.4`, `p95=678.8`, `p99=11947.6`, `max=37865.4`、閾値超過（`>1200ms`）`3.98%`
- `manual_margin_pressure=36` 件（24h）で、`scalp_ping_5s_b_live=29`, `scalp_ping_5s_c_live=7`。サンプルに `manual_net_units=-8500`, `margin_available_jpy=4677.279` を確認。

Failure Cause:
1. `perf_block` は一時的ノイズではなく、`scalp_ping_5s_b_live` と `M1Scalper-M1` の failfast 判定が時間帯別に固定化している。
2. `orders.db` への高頻度書き込み（order-manager）と、高頻度読み取り（bq-sync + position-manager fallback local）が重なり、ロック競合を増幅している。
3. `data_lag_ms` のテール悪化と `manual_margin_pressure` が同時に存在し、通るべき注文の密度を下げて改善ループが遅れる。

Hypothesis:
- `run_sync_pipeline.py` の orders 参照導線（毎分複数クエリ）と `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1` の併用で、`orders.db` ハンドル保持が増えやすく、ロック競合を誘発している可能性が高い（A/B確認が必要）。
- `preflight_start` 行の `request_json` が `{"note":"preflight_start",...}` のみで戦略情報を持たず、ボトルネック時間帯の戦略別分解を難化させている。

Improvement:
1. P0: `quant-bq-sync` の `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL` を無効化し、service timeout 時は stale キャッシュ優先でローカルDBフォールバックを抑制する。
2. P0: `orders.db` へ `preflight_start/perf_block` 記録時の `strategy_tag` と `reject_note` 永続化を必須化する（監査盲点の解消）。
3. P1: `run_sync_pipeline.py` の SQLite 読み取りを明示 close に統一し、read-only URI（`mode=ro`）・短timeout・接続数上限を導入する。
4. P1: `replay_quality_gate` / `trade_counterfactual` の worker 対象を `TrendMA/BB_RSI` 以外（`scalp_ping_5s_b_live`, `M1Scalper-M1` など）へ拡張し、failfast固定化へ直接フィードバックする。
5. P1: `manual_margin_pressure` 発火時の段階的建玉縮退（manual併走含む）を優先し、`margin_available_jpy` の下限回復を先に実行する。

Verification:
1. `journalctl -u quant-order-manager.service --since "1 hour ago" | grep -c "database is locked"` が `<=5`。
2. `lsof /home/tossaki/QuantRabbit/logs/orders.db` で PID `3400` の FD 本数が `<=10` に低下。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 1200` を継続達成。
4. `perf_block` 主因（`hour9/hour11 failfast`）の件数が日次で逓減し、`submit_attempt`/`filled` 密度が回復する。
5. `manual_margin_pressure` が 24h 連続で `0`（または明確な逓減）になる。

Status:
- in_progress

## 2026-02-27 01:00 UTC / 2026-02-27 10:00 JST - `StageTracker` 起動時ロック再発をテーブル存在確認で抑止
Period:
- VM実測: `2026-02-27 00:46-00:50 UTC`
- Source: `journalctl -u quant-scalp-wick-reversal-blend.service`, `logs/stage_state.db`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に
  `sqlite3.OperationalError: database is locked` で連続停止。
- 例外位置は `execution/stage_tracker.py` の初期DDL（`CREATE TABLE IF NOT EXISTS ...`）。
- 同時に `stage_state.db` は複数 worker (`order_manager`, `scalp_ping_5s`, `tick_imbalance` 等) が共有。

Failure Cause:
1. `StageTracker.__init__` が毎回スキーマDDLを書き込み実行し、共有DBのスキーマロック競合時に起動失敗する。
2. 既存テーブルでも DDL/ALTER を実行するため、起動時ロック競合の露出面が広い。

Improvement:
1. `execution/stage_tracker.py` に `_table_exists` / `_column_exists` を追加。
2. DDLは `_ensure_table` / `_ensure_column` 経由で「不足時のみ」実行へ変更。
3. 既存テーブル環境では起動時DDLを書き込まないため、スキーマロック競合を回避。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が `3 passed`。
3. VM反映後に `quant-scalp-wick-reversal-blend.service` の `Application started!` と連続稼働を確認する。

Status:
- in_progress

## 2026-02-27 00:48 UTC / 2026-02-27 09:48 JST - `WickReversalBlend` が `stage_tracker` ロックで停止する障害を修正
Period:
- Incident window: `2026-02-27 00:46` ～ `00:48` UTC
- Source: VM `journalctl -u quant-scalp-wick-reversal-blend.service`, repo `execution/stage_tracker.py`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に `failed` へ遷移。
- 直近ログで `execution/stage_tracker.py` 初期化中に
  `sqlite3.OperationalError: database is locked` を確認。
- 同時に B/C ワーカーは稼働継続しており、勝ち筋戦略だけが停止していた。

Failure Cause:
1. `StageTracker.__init__` の schema 作成が単発 `execute` で、ロック競合時に即例外終了していた。
2. `stage_state.db` に `busy_timeout` / `WAL` / retry の耐性が不足していた。

Improvement:
1. `execution/stage_tracker.py`
   - `STAGE_DB_BUSY_TIMEOUT_MS` / `STAGE_DB_LOCK_RETRY` /
     `STAGE_DB_LOCK_RETRY_SLEEP_SEC` を追加。
   - 接続を `busy_timeout + WAL + autocommit` で初期化。
   - schema作成 SQL を lock retry 付き `_execute_with_lock_retry()` 経由へ変更。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が pass（`3 passed`）。
3. 反映後 `quant-scalp-wick-reversal-blend.service` が `active` を維持すること。

Status:
- in_progress

## 2026-02-27 00:33 UTC / 2026-02-27 09:33 JST - 勝ち筋寄せ再配分（WickBlend増量 + B/C縮小）
Period:
- Snapshot window: `2026-02-27 00:30` ～ `00:33` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 自動戦略24h（strategy_tag あり）:
  - `WickReversalBlend`: `3 trades / +114.7 JPY`
  - `scalp_ping_5s_b_live`: `292 trades / -540.2 JPY`
  - `scalp_ping_5s_c_live`: `466 trades / -3403.0 JPY`
- 直近15分（自動のみ）は `-91.9 JPY` で、依然マイナス。

Failure Cause:
1. 損失寄与の大きい B/C の約定量が、勝ち筋に対して過大。
2. WickBlend は勝っているが発火頻度と配分が低く、寄与不足。

Improvement:
1. `ops/env/quant-scalp-wick-reversal-blend.env`
   - `SCALP_PRECISION_UNIT_BASE_UNITS=9500`（from `7000`）
   - `SCALP_PRECISION_UNIT_CAP_MAX=0.65`（from `0.55`）
   - `SCALP_PRECISION_COOLDOWN_SEC=8`（from `12`）
   - `SCALP_PRECISION_MAX_OPEN_TRADES=2`（from `1`）
   - `WICK_BLEND_RANGE_SCORE_MIN=0.40`（from `0.45`）
   - `WICK_BLEND_ADX_MIN/MAX=14/28`（from `16/24`）
   - `WICK_BLEND_BB_TOUCH_RATIO=0.18`（from `0.22`）
   - `WICK_BLEND_TICK_MIN_STRENGTH=0.30`（from `0.40`）
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=600`（from `720`）
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=5`（from `6`）
3. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=220`（from `260`）
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）

Verification:
1. 反映後10分で `WickReversalBlend` の `filled` 件数が増えること。
2. 反映後15分で自動損益（strategy_tagあり）が 0 以上へ改善すること。
3. 同窓で B/C の損失寄与（JPY）が縮小すること。

Status:
- in_progress

## 2026-02-27 01:05 UTC / 2026-02-27 10:05 JST - `margin_usage_projected_cap` 誤拒否（side cap と net-reducing の不整合）
Period:
- VM `journalctl -u quant-order-manager.service`（00:11〜00:20 UTC）
- VM `/home/tossaki/QuantRabbit/logs/orders.db`（`margin_usage_projected_cap` 行）

Fact:
- `quant-order-manager` で `margin_usage_projected_cap` が連発し、B/C の `sell` シグナルが約定前に落ちていた。
- 同時刻ログに `projected margin scale ... usage=0.921~0.946` が記録され、cap 付近として扱われていた。
- 一方で口座スナップショット（同VM実測）は `usage_total` が低位（例: 約 `0.03` 台）で、総ネット余力とは乖離していた。
- 乖離は「総ネット使用率」は低いが「同方向 side 使用率」だけが高い局面で顕在化した。

Failure Cause:
1. `MARGIN_SIDE_CAP_ENABLED=1` 経路で `usage/projected_usage` を side ベースへ上書きした後、net-reducing 例外も同じ side 値で判定していた。
2. そのため「総ネット使用率を下げる注文」でも `projected_usage < usage` 条件を満たせず、`margin_usage_projected_cap` として誤拒否されていた。
3. 拒否ログに total/side の両指標が十分残っておらず、現場切り分けコストが高かった。

Improvement:
1. `execution/order_manager.py` に `_is_net_reducing_usage(...)` を追加し、例外判定を常に total usage（netting）基準へ固定。
2. side-cap 経路で `usage/projected_usage` を side 用に使っても、拒否可否は `usage_total` と `projected_usage_total` で評価するよう修正（market/limit 両経路）。
3. `margin_usage_projected_cap` ログ payload に `projected_usage_total / margin_usage_total / side_usage / side_projected` を追加し、再発時の即時判別を可能化。

Verification:
1. ローカル回帰:
   - `pytest -q tests/execution/test_order_manager_preflight.py tests/execution/test_order_manager_log_retry.py`
   - `35 passed`
2. 新規テスト:
   - `test_is_net_reducing_usage_*`（純粋判定）
   - `test_limit_order_allows_net_reducing_under_side_cap`（side cap 高負荷でも net-reducing を許可）
3. VM検証（次段）:
   - 反映後 `margin_usage_projected_cap` の連発が収束し、同条件シグナルで `filled/submitted` が再開することを確認。

Status:
- implemented_local_verified

## 2026-02-27 00:25 UTC / 2026-02-27 09:25 JST - 自動損益の実測確認と `scalp_ping_5s_b_live` 即効デリスク
Period:
- Snapshot window: `2026-02-27 00:22` ～ `00:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 直近15分:
  - 全体: `17 trades / +1311.9 JPY`
  - 自動のみ（`strategy_tag != null`）: `16 trades / -65.1 JPY`
  - `scalp_ping_5s_b_live + scalp_ping_5s_c_live`: `14 trades / +6.9 JPY`
- JST当日（`2026-02-27 00:00 JST` 以降）:
  - 全体: `455 trades / +731.4 JPY`
  - 自動のみ: `454 trades / -645.6 JPY`
  - B/C 内訳:
    - `scalp_ping_5s_b_live`: `244 trades / -428.1 JPY`
    - `scalp_ping_5s_c_live`: `204 trades / -97.4 JPY`
- 全体プラスの主因は `strategy_tag=null` の単発決済
  （ticket `400470`, `+1377.0 JPY`, `MARKET_ORDER_TRADE_CLOSE`）。
- 直近2時間の B は long 側損失偏重:
  - long `34 trades / -41.1 JPY`
  - short `3 trades / -6.2 JPY`

Failure Cause:
1. 「全体損益」は手動/タグ欠損の単発利益に引っ張られ、自動戦略の実態が見えにくい。
2. B は long 側で低品質エントリーが残り、stop 系損失が先行している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env` を即時デリスク:
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=720`（from `900`）
   - `SCALP_PING_5S_B_CONF_FLOOR=75`（from `72`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.60`（from `0.54`）
2. 停止なし方針を維持しつつ、B の低確度 long 発火を抑制して損失勾配を圧縮する。

Verification:
1. 反映後30分で `scalp_ping_5s_b_live` の `realized_pl` がゼロ超へ改善すること。
2. `orders.db` で B の `probability_scaled` / `rejected` 比率が低下すること。
3. JST当日の自動損益（`strategy_tag != null`）がマイナス幅縮小に転じること。

Status:
- in_progress

## 2026-02-26 13:30 UTC / 2026-02-26 22:30 JST - `SIDE_FILTER=none` が wrapper で `sell` 強制され、B/C entry が詰まる問題を修正
Period:
- Analysis/patch window: `2026-02-26 13:12` ～ `13:30` UTC
- Source: `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`, `tests/workers/test_scalp_ping_5s_b_worker_env.py`, `ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `ops/env/scalp_ping_5s_b.env` は `SCALP_PING_5S_B_SIDE_FILTER=none` だったが、
  wrapper 側の fail-closed 実装により不正値扱いで `sell` に上書きされていた。
- `ops/env/scalp_ping_5s_c.env` の `SCALP_PING_5S_C_SIDE_FILTER=none` と
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` も同様に `sell` へ上書きされていた。
- このため `SIDE_FILTER=none` を設定しても実効せず、`side_filter_block` が残る状態だった。

Failure Cause:
1. wrapper で `ALLOW_NO_SIDE_FILTER` が実装されておらず、`none` が常に invalid 扱いだった。
2. env の意図値と実効値が乖離し、skip要因の切り分けを難しくしていた。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`
   - `ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER in {"", "none", "off", "disabled"}` のとき、
     `SCALP_PING_5S_SIDE_FILTER=""` を許可する正規化を追加。
   - 上記以外の未設定/不正値は従来どおり `sell` へ fail-closed。
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` を追加。
3. `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - B/C の `ALLOW_NO_SIDE_FILTER=1` で空 side filter が通る検証を追加/更新。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py -k "side_filter"` → `8 passed`
2. `python3 -m py_compile workers/scalp_ping_5s_b/worker.py workers/scalp_ping_5s_c/worker.py` → pass
3. VM 反映後に `entry-skip summary` の `side_filter_block` 比率が低下することを確認する。

Status:
- in_progress

## 2026-02-26 13:05 UTC / 2026-02-26 22:05 JST - 方向精度再崩れの根本対策（C no-side-filter封鎖 + side-filter復元）
Period:
- Analysis/patch window: `2026-02-26 12:40` ～ `13:05` UTC
- Source: repository `workers/scalp_ping_5s*`, `ops/env/scalp_ping_5s_c.env`, unit tests

Fact:
- `workers/scalp_ping_5s_c/worker.py` には `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER=none` で
  side filter を未設定扱いにできる分岐が存在した。
- `ops/env/scalp_ping_5s_c.env` は実際に `SCALP_PING_5S_C_SIDE_FILTER=none`,
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` だった。
- `workers/scalp_ping_5s/worker.py` は初段で side_filter を通した後に
  ルーティングで side が反転した場合、最終 `side_filter_final_block` で no-entry になる設計だった。

Failure Cause:
1. C に no-side-filter 例外があり、方向固定の前提が運用設定で破れる。
2. side_filter を通過したシグナルでも、後段flipで反転すると発注前に消失し、エントリー密度が不安定化する。

Improvement:
1. C ラッパーで no-side-filter 例外を廃止し、invalid/missing を常に `sell` へ fail-closed。
2. 本体ワーカーへ `_resolve_final_signal_for_side_filter()` を追加し、
   後段反転時は初段の side-filter 適合シグナルへ復元して発注経路を維持。
3. 運用envを `SCALP_PING_5S_C_SIDE_FILTER=sell`, `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=0` に固定。
4. ロット計算は途中丸めを廃止して最終丸めへ統一し、
   `units_below_min` による 0 化でシグナルが消える経路を縮小。
5. `MIN_UNITS_RESCUE`（確率/信頼度/リスクcap条件付き）を導入し、
   最終段だけ 1unit 救済して実約定導線を維持。

Verification:
1. 対象テスト（10件）:
   - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - `tests/workers/test_scalp_ping_5s_worker.py -k resolve_final_signal_for_side_filter`
2. 結果: `10 passed`

Status:
- in_progress

## 2026-02-26 12:35 UTC / 2026-02-26 21:35 JST - quote 問題の実測再監査と執行層ハードニング
Period:
- 直近24h / 7d（VM実DB）
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 24h `orders` 合計: `50365`
- 24h status 上位:
  - `preflight_start=18629`
  - `perf_block=16971`
  - `probability_scaled=7719`
  - `entry_probability_reject=1101`
- 24h error:
  - `TRADE_DOESNT_EXIST=8`、`OFF_QUOTES/PRICE_*` は 0
- 7d `quant-order-manager` journal でも `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` は 0

Failure Cause:
1. 現在の損失・機会損失の主因は quote 不足ではなく、`perf_block` と確率/余力ガード側。
2. ただし再クオート要求が急増する局面では既定 `FETCH=2 + RETRY=1` が薄く、将来的な取り逃しリスクは残る。

Improvement:
1. `ops/env/quant-order-manager.env` で quote 再取得耐性を強化。
2. `ORDER_SUBMIT_MAX_ATTEMPTS=1` は維持し、quote 専用リトライだけ増やして戦略判定を汚さない。

Verification:
1. 反映後24hで `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` の件数を再計測。
2. 同期間で `filled / submit_attempt` 比率の悪化がないことを確認。
3. `perf_block` が依然主因なら quote ではなく戦略側改善を優先継続。

Status:
- in_progress

## 2026-02-26 12:25 UTC / 2026-02-26 21:25 JST - no-stop維持で「無約定化」を解消する再配線
Period:
- Incident window: `2026-02-26 11:40` ～ `12:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, `journalctl -u quant-scalp-ping-5s-{b,c}.service`

Fact:
- `orders.db` の `orders` は直近30分で `0 rows`（`datetime(substr(ts,1,19)) >= now-30m`）。
- `quant-order-manager` は `coordinate_entry_intent` 受信を継続しているが、`preflight_start` 以降の新規発注イベントが停止。
- `entry_intent_board` の直近45分は `1件` のみで、`scalp_extrema_reversal_live` が `below_min_units_after_scale`（`raw=45`, `min_units=1000`）で reject。
- B/C ワーカーは稼働継続しているが、`entry-skip summary` は
  `no_signal:revert_not_found` が最多、次点で `no_signal:side_filter_block` / `units_below_min` が継続。

Failure Cause:
1. no-stop方針の下で `SIDE_FILTER=sell` と逆風ドリフト縮小が重なり、B/C が local 判定段階で枯渇。
2. 共通 `POLICY_HEURISTIC_PERF_BLOCK` が有効のままで、他戦略側の再起動余地も狭い。
3. 最小ロット閾値が小口シグナルの通過率を削り、intent が `order_manager` まで届かない局面が残る。

Improvement:
1. 共通 hard reject の解除:
   - `ops/env/quant-v2-runtime.env`
   - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0`（from `1`）
2. B の通過率回復:
   - `SCALP_PING_5S_B_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=1`（from `5`）
   - `SHORT_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.58`（from `0.42`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
3. C の無約定解消（停止ではなく両方向縮小運転）:
   - `SCALP_PING_5S_C_SIDE_FILTER=none`（from `sell`）
   - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
   - `SCALP_PING_5S_C_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`（from `5`）
   - `SHORT/LONG_MOMENTUM_TRIGGER_PIPS=0.08/0.18`（from `0.10/0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.62`（from `0.45`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
4. C の reject/rate-limit 緩和:
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=3`（from `1`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.68`（from `0.76`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`（from `0.70`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.38`（from `0.24`）
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.52`（from `0.62`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
   - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.46`（from `0.52`）
   - `ops/env/quant-order-manager.env` の C上書き値を同値同期
     - `REJECT_UNDER=0.46`, `MIN/MAX_SCALE=0.40/0.85`, `BOOST_PROBABILITY=0.85`
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
     - `SCALP_PING_5S(_C)_PERF_GUARD_MODE=off`, `SCALP_PING_5S(_C)_PERF_GUARD_ENABLED=0`
   - `scalp_extrema_reversal_live` の協調拒否解消:
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`（runtime/order-manager 両env）

Verification:
1. 反映後 30分で `orders.db` の `preflight_start` と `filled` が再出現すること。
2. `entry-skip summary` の `units_below_min` 比率が低下すること。
3. 反映後 60分で `trades.db` の `realized_pl` 増分が `scalp_ping_5s_c_live` 単独で急悪化しないこと（損失勾配監視）。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - `scalp_ping_5s_b_live/c_live` 方向劣化の再発防止（side filter fail-closed）
Period:
- Direction audit window: `datetime(close_time) >= now - 24 hours`
- Runtime env check: `quant-scalp-ping-5s-b.service` MainPID/child PID
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, systemd process env

Fact:
- 24h集計（`trades.db`）で `scalp_ping_5s_b_live` の buy 側が劣化:
  - buy `72 trades`, win rate `20.83%`, avg `-0.936 pips`
  - `entry_probability>=0.75` の buy でも win rate `18.84%`, avg `-1.051 pips`
- 実行中プロセス（起動 `2026-02-26 12:10:55 UTC`）では
  `SCALP_PING_5S_B_SIDE_FILTER=sell` と
  `SCALP_PING_5S_SIDE_FILTER=sell` が有効。
- ただし過去履歴には buy 発注が残るため、env 欠落/不正時に再発しうる。

Failure Cause:
1. B/C variant の方向制御が env 設定依存で、設定欠落時に fail-open になり得る。
2. 高確率帯 buy の実績悪化と整合しない方向シグナルが通過した履歴が存在する。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py` と `workers/scalp_ping_5s_c/worker.py` で side filter を fail-closed 化。
2. `SCALP_PING_5S_SIDE_FILTER` が未設定/不正値なら `sell` を強制。
3. 起動ログへ `side_filter` を明示出力し、監査を容易化。
4. `tests/workers/test_scalp_ping_5s_b_worker_env.py` に
   B/C それぞれの `missing/invalid/valid` ケースを追加。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py` が全緑（14 passed）。
2. VMで `quant-scalp-ping-5s-b` の子プロセス環境に
   `SCALP_PING_5S_SIDE_FILTER=sell` が存在することを確認。

Status:
- in_progress

## 2026-02-26 12:09 UTC / 2026-02-26 21:09 JST - B側の `units_below_min` 残りを削るため最小ロット閾値を再調整
Period:
- Snapshot: `2026-02-26 12:06:59` ～ `12:08:10` UTC（`21:06:59` ～ `21:08:10` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/ops/env/{scalp_ping_5s_b.env,quant-order-manager.env}`

Fact:
- 12:06:59 UTC 再起動後、`C` は `units_below_min=0` まで低下。
- 同条件で `B` は `entry-skip summary side=short total=4 units_below_min=4` が残存。
- `RISK multiplier` は引き続き `mult=0.40` で、縮小局面の最終ユニットが閾値を割り込みやすい。

Failure Cause:
1. Bは strategy 側と order-manager 側の min units が `20` で揃っており、縮小ロットの通過余地が不足していた。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`
2. `ops/env/quant-order-manager.env`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`

Verification:
1. 反映後15分で `journalctl -u quant-scalp-ping-5s-b.service` の `units_below_min` が 0 になること。
2. `orders.db` で `scalp_ping_5s_b_live` の `submit_attempt` / `filled` が再出現すること。
3. `manual_margin_pressure` / `slo_block` が再増加しないこと。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - short-only後の無約定ボトルネック（revert_not_found + short units_below_min）
Period:
- Observation window: `2026-02-26 12:00:00` 以降（VM journal）
- Source: `journalctl -u quant-scalp-ping-5s-b.service`, `journalctl -u quant-scalp-ping-5s-c.service`, `journalctl -u quant-order-manager.service`

Fact:
- `SCALP_PING_5S_B/C_SIDE_FILTER=sell` は機能し、long は `no_signal:side_filter_block` / `side_filter_final_block` で継続遮断。
- 一方で約定が止まり、`quant-order-manager` の `OPEN_REJECT/OPEN_FILLED` は同窓で実質発生なし。
- B/C の skip 内訳は `no_signal:revert_not_found` が最大（各30秒集計で概ね `40-94` 件）。
- short 側は `units_below_min` が継続（B: `3-17`, C: `1-6` / 30秒集計）。

Failure Cause:
1. short-only化後、短期反転検知（revert）が成立せず `revert_not_found` に集中。
2. 成立した short シグナルも、動的縮小後ユニットが最小ロット未満になり通過不能。
3. long遮断は効いているが、short化の再配線が不足し、取引密度が0近傍に落ちた。

Improvement:
1. B/C の `revert` 閾値を同時緩和（`REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO`, `REVERT_SHORT_WINDOW`）。
2. short 最小通過ロットを引き下げ（`SCALP_PING_5S_{B,C}_MIN_UNITS`, `ORDER_MIN_UNITS_STRATEGY_*` を `5`）。
3. C は short 発火側を追加緩和（`SHORT_MIN_TICKS`, `SHORT_MIN_SIGNAL_TICKS`）。
4. long→short 変換を有効化（B/C `EXTREMA_GATE_ENABLED=1`, `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`, `LONG_TO_SHORT_MIN_SCORE` 緩和）。
5. C は flip系を再稼働（`SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`）し、short側への再配線を強化。

Verification:
1. 反映後30分で `entry-skip summary side=short` の `units_below_min` 比率が低下すること。
2. `orders.db` で `filled` が再発し、`strategy in {scalp_ping_5s_b_live, scalp_ping_5s_c_live}` の short約定が出ること。
3. `trades.db` で B/C の新規closeにおける `realized_pl` の負勾配が反転または鈍化すること。

Status:
- in_progress

## 2026-02-26 12:06 UTC / 2026-02-26 21:06 JST - 損失主因の再監査（執行品質より制御異常が支配）
Period:
- 24h監査: `orders.db` / `trades.db` / `metrics.db`（`datetime(ts/close_time) >= now - 24 hours`）
- 事故窓監査: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`11:20:16` ～ `18:13:14` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `oanda_*_live.json`

Fact:
- V2主導線は稼働中（`quant-market-data-feed/order-manager/position-manager/strategy-control` は active/running）。
- 直近24hの `orders`:
  - `preflight_start=18629`, `perf_block=16971`, `probability_scaled=7719`, `submit_attempt=1221`, `filled=1202`
  - `rejected=18`, `slo_block=12`, `margin_usage_projected_cap=190`, `manual_margin_pressure=36`
- 直近24hの `trades`: `1283件`, `-4417.54 JPY`。
  - close reason: `STOP_LOSS_ORDER=390件/-5232.48 JPY`, `MARKET_ORDER_MARGIN_CLOSEOUT=3件/-2163.76 JPY`
  - 戦略別下位: `scalp_ping_5s_c_live=-7025.83 JPY`, `scalp_ping_5s_b_live=-3144.31 JPY`
- 執行品質（`analyze_entry_precision.py --limit 1200`）:
  - `slip p95=0.300 pips`, `cost_vs_mid p95=0.700 pips`, `submit latency p95=276.9 ms`, `missing quote=0`
- レイテンシ/SLO:
  - `data_lag_ms p95=4485.9`, `decision_latency_ms p95=8835.9`
  - 閾値超過率: `data_lag_ms>1500ms = 22.8%`, `decision_latency_ms>2000ms = 10.71%`
  - `journalctl quant-order-manager` に `slo_block:data_lag_p95_exceeded` を確認（2026-02-26 11:42～11:45 UTC）
- 事故窓の確定事実:
  - `strategy_control_exit_disabled=10277`（2026-02-24 JST日単位）
  - `micro-micropul*` 4注文で各 `2044～2045` 回の exit disabled 後、`MARKET_ORDER_MARGIN_CLOSEOUT`
  - 上記4件合計 `-16837.43 JPY`
- 直近口座スナップショット（2026-02-26 12:00:35 UTC）:
  - `nav=56970.299`, `margin_used=53060.740`, `margin_available=3943.559`, `health_buffer=0.06918`
  - `USD_JPY short_units=8500`
- `entry_thesis` 必須欠損:
  - 直近24hで `entry_probability/entry_units_intent` 欠損 `80/1283` 件（主に `scalp_ping_5s_*` 派生タグ行）

Failure Cause:
1. 大損の一次原因は、戦略予測精度より `exit不能（strategy_control_exit_disabled）` と `margin closeout tail` のシステム異常。
2. manual併走を含む高マージン使用状態で、`margin_*` ガードと closeout が発火しやすい口座状態が継続。
3. `data_lag_ms` スパイクにより `slo_block` が増え、良い局面のエントリーが欠落して収益回復を阻害。
4. `entry_thesis` 欠損により、意図協調/監査の完全性が崩れ、異常時の切り分けと制御精度を落としている。
5. B/Cは執行コスト劣化より `STOP_LOSS_ORDER` と手仕舞い設計由来の期待値悪化が支配的。

Improvement:
1. `strategy_control` ガード強化:
   - `entry=1 & exit=0` を検知即時で自動修正（`entry=0 & exit=1`）し、監査ログとアラートを必須化。
2. exit不能の再発防止:
   - `strategy_control_exit_disabled` が同一 `client_order_id` で閾値超過したら強制エスカレーション（exit優先モード）。
3. マージン防衛:
   - manual玉込みで `health_buffer` / `margin_available` 連動の新規抑制を強化し、先に建玉縮退を実行。
4. SLO復旧:
   - `data_lag_ms` スパイク時間帯の `market-data-feed` / DB遅延要因を切り分け、`slo_block` は hard reject 連打ではなく段階縮小中心へ。
5. `entry_thesis` スキーマ強制:
   - 非manual注文は `entry_probability` と `entry_units_intent` 欠損時に reject（`status=missing_entry_thesis_fields`）。
6. 収益回復の優先順:
   - B/Cは方向・手仕舞いロジックを再調整し、`STOP_LOSS_ORDER` 依存を減らす。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h連続 `0`。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 `0`。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 2000` を継続達成。
4. `orders.db` で `slo_block` の連続発火（同一時間帯クラスタ）が解消。
5. `trades.db` で `entry_thesis` 必須2項目の欠損が `0`。
6. B/Cの 24h `jpy` と `PF` が改善方向（少なくとも連続悪化が停止）。

Status:
- in_progress

## 2026-02-26 12:02 UTC / 2026-02-26 21:02 JST - B/C sell限定運用で `units_below_min` が主因化し、発注ゼロ化したため最小ロット閾値を緩和
Period:
- Snapshot: `2026-02-26 11:59:57` ～ `12:01:39` UTC（`20:59:57` ～ `21:01:39` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` は `11:59:57 UTC` に再起動済み（active/running）。
- しかし `orders.db` は `ts >= 2026-02-26T11:59:57Z` で `0件`（新規発注フロー未到達）。
- ワーカーログで共通して以下を確認:
  - `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell` により long 候補が `side_filter_block`。
  - short 候補は `units_below_min` が発生（B: `10件`, C: `8件`）。
  - `RISK multiplier` は `mult=0.40` で推移し、縮小後ユニットが閾値未満へ落ちやすい。

Failure Cause:
1. long を遮断する side filter 自体は意図どおりだが、short 側の最終ユニットが `min_units=30` を下回り、注文生成まで到達できない。
2. strategy 側 min units と order 側 min units の閾値差（Cは 30 固定）が、縮小運転時の失効を助長している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 30 -> 20`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_MIN_UNITS: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C: 30 -> 20`

Verification:
1. 反映後30分で `orders.db` に `submit_attempt` / `filled` が B/C で再出現すること。
2. `journalctl -u quant-scalp-ping-5s-{b,c}.service` の `units_below_min` 件数が減少すること。
3. `orders.db` で `manual_margin_pressure` / `slo_block` の再悪化がないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - PDCA導線の実運用監査（稼働中だが改善ループに断点あり）
Period:
- Snapshot: `2026-02-26 11:39` ～ `11:50` UTC（`20:39` ～ `20:50` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `/home/tossaki/QuantRabbit/logs/*_latest.json`, OANDA account summary/open positions

Fact:
- V2主導線は稼働中:
  - `quant-market-data-feed`, `quant-strategy-control`, `quant-order-manager`, `quant-position-manager`, `quant-forecast` は `active(running)`。
  - `quantrabbit.service` は存在せず、VMリポジトリは `HEAD == origin/main == 0c0caae2c05295cbccd7113454fb24cb7f8afda3`。
- 分析/改善タイマーは起動:
  - `quant-pattern-book`, `quant-dynamic-alloc`, `quant-policy-guard`, `quant-replay-quality-gate`, `quant-trade-counterfactual`, `quant-forecast-improvement-audit` が schedule され実行履歴あり。
- ただし改善ループは市場オープン中に停止:
  - `quant-replay-quality-gate.service` は `skipped: market_open` を連続出力。
  - `quant-trade-counterfactual.service` も `skipped: market_open` を連続出力。
- 発注導線の品質劣化:
  - `quant-order-manager` 直近1時間で `database is locked` が `67` 回。
- 24hの orders 状態:
  - `preflight_start=18629`, `submit_attempt=1221`, `filled=1202`, `perf_block=16971`, `entry_probability_reject=1101`。
- 監査ログ差分:
  - `logs/ops_v2_audit_latest.json` で `POLICY_HEURISTIC_PERF_BLOCK_ENABLED expected=0 actual=1` の `warn`。

Failure Cause:
1. `replay` と `counterfactual` が market open 時に止まるため、改善施策の入力が日中に更新されない。
2. replay quality gate の対象が `TrendMA/BB_RSI` 中心で、現行の主要 scalp/micro 群を十分に監査できていない。
3. auto-improve が `block_jst_hours` を `worker_reentry.yaml` へ自動反映する設定で、方針「停止より改善優先」と衝突しやすい。
4. `orders.db` lock 競合が残り、preflight->submit の実効通過率を毀損している。
5. pattern gate の実効キー (`ORDER_MANAGER_PATTERN_GATE_ENABLED`) と運用キー (`ORDER_PATTERN_GATE_ENABLED`) が分岐し、設定意図と実動作がずれる余地がある。

Improvement:
1. `orders.db` 競合の再抑制を最優先（busy timeout/retry/lock制御と write 経路の再点検）。
2. `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=0` と `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=0` へ変更し、改善ループを 24/7 化。
3. `config/replay_quality_gate_main.yaml` の `workers` を現行稼働戦略へ拡張し、PDCA対象を本番導線へ合わせる。
4. `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_REENTRY=0` にして、自動時間帯ブロック反映を停止（提案のみ運用）。
5. pattern gate の env キーを一本化し、order-manager 側の参照名に統一。
6. `POLICY_HEURISTIC_PERF_BLOCK_ENABLED` の期待値と実運用値を監査基準に合わせる。

Verification:
1. `journalctl -u quant-order-manager` の `database is locked` が 1h あたり 0 件へ収束。
2. `replay_quality_gate_latest.json` と `trade_counterfactual_latest.json` の `generated_at` が市場オープン中も更新される。
3. replay report の対象戦略に、稼働中 scalp/micro 群が含まれる。
4. `worker_reentry.yaml` への自動 `block_jst_hours` 書き換えが停止し、改善提案だけが残る。
5. `orders.db` の `preflight_start -> submit_attempt -> filled` の通過率が改善する。

Status:
- in_progress

## 2026-02-26 10:12 UTC / 2026-02-26 19:12 JST - 停止なし・時間帯停止なし運用へ再構成
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:12` UTC（`18:52` ～ `19:12` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `config/worker_reentry.yaml`, OANDA `openTrades`

Fact:
- 24h は `1120 trades`, `-11510.6 JPY`, `PF=0.455`（主損失: `scalp_ping_5s_c_live`, `scalp_ping_5s_b_live`, `M1Scalper-M1`）。
- 直近監査で `entry=1 & exit=0` は `0` 件、`strategy_control_exit_disabled` も直近1hで `0` 件。
- ただし運用設定には「停止相当」の制約が残存:
  - `scalp_ping_5s_{b,c}` の `ALLOW_HOURS_JST`（時間帯限定）
  - `worker_reentry` の `block_jst_hours`（`M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live`）

Failure Cause:
1. 収益改善を「停止/時間帯限定」に寄せると、運用方針（常時動的トレード）と矛盾し、再現性が崩れる。
2. `scalp_ping_5s_c_live` は `close_reject_no_negative` が残ると損失玉の解放が遅れやすい。

Improvement:
1. `ops/env/scalp_ping_5s_{b,c}.env`:
   - `ALLOW_HOURS_JST=`（時間帯停止を撤去）
   - `PERF_GUARD_MODE=reduce`（停止ではなく縮小）
2. `config/worker_reentry.yaml`:
   - `M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live` の `block_jst_hours` を空配列化。
3. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_{c,c_live}.neg_exit` に `strict_no_negative: false`, `deny_reasons: []` を追加し、EXIT詰まりを解消。
4. `strategy_control_flags` は `entry=1 & exit=1` を基本状態として維持し、停止は緊急安全時のみ許可。

Verification:
1. `orders.db` で `strategy_control_entry_disabled` / `strategy_control_exit_disabled` が発生しないこと。
2. `orders.db` で `close_reject_no_negative` が `scalp_ping_5s_c_live` で減少すること。
3. 24hで `B/C` の `jpy PF` が改善方向（`>=1.0` へ接近）すること。
4. `worker_reentry` 由来の時間帯ブロック理由で待機しないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - `slo_block(data_lag_p95_exceeded)` を緩和し、M1配分を引き上げ
Period:
- 30m: `datetime(ts) >= now - 30 minutes`
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 30m `orders.db`:
  - `preflight_start=110`, `probability_scaled=33`
  - `slo_block=10`（latest `2026-02-26T11:45:09Z`）
  - `manual_margin_pressure=25`（ただし latest `2026-02-26T11:34:06Z` で新規発生停止）
- `slo_block` の reason は全件 `data_lag_p95_exceeded`。
  - `data_lag_p95_ms` は `~7152ms`、現行閾値 `5000ms` を超過。
- 24h損益:
  - `M1Scalper-M1` は直近で `-3.1 JPY`（ほぼ横ばい、下振れ縮小）
  - B/C は side_filter=sell 反映後、B/C の `buy` 注文は 0 件。

Failure Cause:
1. strategy-control 側の data lag p95 スパイクが SLO 閾値を超え、scalp_fast が連続 reject。
2. M1 は止めずに残しているが、配分が低く利益寄与の伸びが不足。

Improvement:
1. `ops/env/quant-order-manager.env`
   - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS: 5000 -> 9000`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路）
   - `ORDER_SLO_GUARD_*` を明示追加し、`ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000`
3. `ops/env/quant-m1scalper.env`
   - `M1SCALP_BASE_UNITS: 3000 -> 4500`

Verification:
1. 反映後 30m で `slo_block` 最新時刻が更新されないこと（再発停止）。
2. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移比率が改善すること。
3. `M1Scalper-M1` の 6h / 24h `realized_pl` が正方向へ改善すること。

Status:
- in_progress

## 2026-02-26 11:45 UTC / 2026-02-26 20:45 JST - B/C の side 偏りを実損で是正（long 逆風遮断）
Period:
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- `scalp_ping_5s_c_live`:
  - long `347 trades / -5380.5 JPY / -353.4 pips`
  - short `75 trades / -88.8 JPY / -52.2 pips`
- `scalp_ping_5s_b_live`:
  - long `128 trades / -1300.8 JPY / -141.3 pips`
  - short `26 trades / -7.7 JPY / -0.5 pips`
- 直近損失の主因は B/C ともに long 側へ偏在。

Failure Cause:
1. no-stop 継続を優先して `SIDE_FILTER` を空に戻したことで、逆風側（long）の負けが継続拡大。
2. B/C の短期ローカル判定は機能しているが、方向バイアスの切替が遅く実損に追随できていない。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_SIDE_FILTER=sell`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_SIDE_FILTER=sell`

Verification:
1. 反映後 30m で `trades.db` の B/C long 新規クローズ件数が 0 であること。
2. B/C の short 側 `avg_lose_jpy` と `total_jpy` が long 全開時より改善すること。
3. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移を継続確認し、`perf_block` 再発がないこと。

Status:
- in_progress

## 2026-02-26 11:30 UTC / 2026-02-26 20:30 JST - no-stop阻害点を `perf_block` + `manual_margin_pressure` に限定して除去
Period:
- 15m: `datetime(ts) >= now - 15 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, live account snapshot via `execution.order_manager.get_account_snapshot()`

Fact:
- 直近15分の status 集計:
  - `preflight_start=68`
  - `perf_block=52`
  - `probability_scaled=33`
  - `manual_margin_pressure=8`
  - `entry_probability_reject=4`
  - `slo_block=1`
- strategy別（同窓）:
  - `M1Scalper-M1 | perf_block=25`
  - `scalp_ping_5s_b_live | perf_block=24`
  - `scalp_ping_5s_b_live | manual_margin_pressure=10`
- 口座スナップショット:
  - `nav=57,556.799`, `margin_used=53,037.280`, `margin_available=4,553.519`, `health_buffer=0.07907`
  - manual 建玉: `-8500 units`, `1 trade`

Failure Cause:
1. no-stop向けに failfast を緩めても、`perf_guard` の hard reason 判定で B/M1 が preflight reject を継続。
2. manual 併走時の `manual_margin_guard` が小ロット再開局面でも `manual_margin_pressure` を発火させ、B の通過を削る。

Improvement:
1. `ops/env/quant-order-manager.env`:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路へ実効反映）:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
3. 戦略env側も整合:
   - `ops/env/scalp_ping_5s_b.env`: `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `ops/env/quant-m1scalper.env`: `M1SCALP_PERF_GUARD_ENABLED=0`

Verification:
1. `orders.db` 15分窓で `status in ('perf_block','manual_margin_pressure')` が B/M1 で減少すること。
2. 同窓で `filled` と `probability_scaled` の増加が確認できること。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が増えないこと（24h監査継続）。

Status:
- in_progress

## 2026-02-26 11:02 UTC / 2026-02-26 20:02 JST - `manual_margin_pressure` が B エントリー再開の最終ボトルネック
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, OANDA account snapshot/openTrades
- Window: `datetime(ts) >= now - 15 minutes`（UTC）

Fact:
- 全体（15分）: `entry_probability_reject=13`, `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `scalp_ping_5s_b_live`（15分）: `preflight_start=6`, `probability_scaled=3`, `manual_margin_pressure=3`
- `manual_margin_pressure` 3件はすべて B の `scalp_fast` エントリー（`units=139/140/181`）
- 口座実測（2026-02-26 11:01 UTC）:
  - `NAV=57,930.80`, `margin_used=53,022.32`, `margin_available=4,942.48`, `health_buffer=0.0853`
  - open trade: `USD_JPY -8500`（`id=400470`, `TP/SLなし`）

Failure Cause:
1. failfast/forecast/leading-profile を緩和した後、B の最終拒否が `manual_margin_pressure` に収束。
2. guard閾値（`free_ratio>=0.05`, `health_buffer>=0.07`, `available>=3000`）が、手動玉併走時の小ロット意図まで遮断。
3. その結果、B の `probability_scaled` 後の通過意図が実発注に到達しない。

Improvement:
1. `ops/env/quant-order-manager.env` の manual margin guard を no-stop 方針向けに再調整:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO: 0.05 -> 0.01`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER: 0.07 -> 0.02`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY: 3000 -> 500`
2. guard 自体は維持（`ORDER_MANUAL_MARGIN_GUARD_ENABLED=1`）し、極端な near-closeout だけを継続遮断。

Verification:
1. `quant-order-manager` 再起動後に process env で3閾値の反映を確認する。
2. 反映後 15 分で `manual_margin_pressure` 件数が減少し、`submit_attempt/filled` が増えることを確認する。
3. `margin_usage_projected_cap` と `MARKET_ORDER_MARGIN_CLOSEOUT` が増加しないことを同時監視する。

Status:
- in_progress

## 2026-02-26 11:12 UTC / 2026-02-26 20:12 JST - no-stop 維持のまま「負け源圧縮 + 勝ち源増量」へ再配分
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`
- Window:
  - 拒否分析: `datetime(ts) >= now - 30 minutes`
  - 損益分析: `datetime(close_time) >= now - 7 days`（補助で 6h）

Fact:
- 直近30分の拒否内訳:
  - `entry_probability_reject=21`（全件 `rangefader`）
  - `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `rangefader` 拒否理由は `entry_probability_below_min_units` に収束。
  - 直近サンプルの `entry_probability` は約 `0.40`
  - 確率スケール後ユニットが pocket 最小ユニット未満で落ちる状態。
- 7日損益（strategy別、主なもの）:
  - `MomentumBurst`: `+1613.7 JPY`（n=7）
  - `MicroRangeBreak`: `+662.3 JPY`（n=32, PF=3.05）
  - `scalp_ping_5s_b_live`: `-9475.8 JPY`（n=2422, PF=0.43）
  - `scalp_ping_5s_c_live`: `-2735.5 JPY`（n=894, PF=0.86）
  - `M1Scalper-M1`: `-1627.3 JPY`（n=284, PF=0.64）
- 直近6hでも `scalp_ping_5s_c_live` は `-1859.9 JPY`（n=125）。

Failure Cause:
1. 発注経路は稼働しているが、`rangefader` が最小ユニット条件で連続 reject され、約定機会が失われた。
2. B/C と M1 が数量面で重く、no-stop 運用時に損失寄与が勝ち寄与を上回る配分になっていた。

Improvement:
1. `RangeFader` の通過回復（停止ではなく通過条件調整）:
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=300` を `quant-order-manager` に追加。
2. 負け源の即時圧縮（B/C/M1 を継続稼働のまま減速）:
   - B: `BASE 1800->900`, `MAX 3600->1800`, `MAX_ORDERS_PER_MINUTE 6->4`, `CONF_FLOOR 74->78`
   - C: `BASE 400->220`, `MAX 900->500`, `MAX_ORDERS_PER_MINUTE 2->1`, `CONF_FLOOR 82->86`
   - M1: `BASE 10000->6000->3000`, `MAX_OPEN_TRADES 2->1`
   - B 追加ホットフィックス: `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE) 30->20`
     （縮小後の確率スケールで `below_min_units` 連発したため）
   - B/M1 追加ホットフィックス: `hard:failfast` 連続拒否を避けるため
     `PERF_GUARD_FAILFAST_*` を soft 警告側へ再設定（Bは `FAILFAST_PF=0.10`, `HARD_PF=0.00`。
     M1は `MODE=reduce`, `FAILFAST_PF/WIN=0.30/0.35`, `HARD_PF=0.20`）。
3. 勝ち源の増量:
   - `MicroRangeBreak` と `MomentumBurstMicro` の `MICRO_MULTI_BASE_UNITS 42000->52000`
   - 上記2戦略の breakout 発火閾値を緩和:
     - `MIN_ADX 20.0->16.0`, `MIN_RANGE_SCORE 0.42->0.34`, `MIN_ATR 1.2->0.9`
     - `LOOP_INTERVAL_SEC 4.0->3.0`
   - `RangeFader` は過大化を避けるため `RANGEFADER_BASE_UNITS 13000->11000`

Verification:
1. 反映後30分で `rangefader` の `entry_probability_below_min_units` が減少し、`submit_attempt/filled` が発生すること。
2. 反映後2時間で B/C/M1 の `realized_pl` ドローダウン勾配が低下すること。
3. 同時に `MicroRangeBreak` / `MomentumBurst` の filled 数と `realized_pl` を増分監視すること。

Status:
- in_progress

## 2026-02-26 10:31 UTC / 2026-02-26 19:31 JST - Bがhard failfastで全面停止して約定不足
Period:
- `datetime(ts) >= now - 30 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 直近30分の `orders.db`:
  - `scalp_ping_5s_b_live`: `perf_block=32`
  - `scalp_ping_5s_c_live`: `perf_block=4`
  - `RangeFader-sell-fade`: `entry_probability_reject=23`
- `quant-order-manager` 拒否ログ（B）:
  - `perf_block:hard:hour10:failfast:pf=0.62 win=0.29 n=191` が連続発生
- 口座状態:
  - open trades はフラット（新規約定不足）

Failure Cause:
1. `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF/WIN` が現状実績より高く、`reduce` 設定でも hard 理由で新規を全面拒否。
2. `SCALP_PING_5S_B_SIDE_FILTER=buy` / `SCALP_PING_5S_C_SIDE_FILTER=buy` で方向固定が残り、相場方向とのズレ時に機会損失が拡大。

Improvement:
1. Bの hard failfast 閾値を実績連動に下げる:
   - `quant-order-manager` 内では `ORDER_MANAGER_SERVICE_ENABLED=0` に固定（自己HTTP再入を停止）
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF: 0.88 -> 0.58`
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN: 0.48 -> 0.27`
   - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX: 0.52 -> 0.75`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER(B/C): 0.64/0.76 -> 0.48/0.62`
   - 反映先: `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
2. B/C の方向固定を解除:
   - `SCALP_PING_5S_B_SIDE_FILTER=`
   - `SCALP_PING_5S_C_SIDE_FILTER=`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
3. strategy_entry の forecast 逆行一律拒否を解除:
   - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=0`
   - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED=0`
   - 反映先: `ops/env/quant-v2-runtime.env`
4. B/C の strategy_entry leading profile 拒否を解除:
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=0`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=0`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Verification:
1. `orders.db` 30分窓で `scalp_ping_5s_b_live` の `perf_block` 比率が低下すること。
2. 同窓で `submit_attempt` / `filled` が再出現すること。
3. `trades.db` の当日 `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` 実現JPYが改善方向に転じること。

Status:
- in_progress

## 2026-02-26 10:06 UTC / 2026-02-26 19:06 JST - 即日止血（C停止 + strategy_control 衝突解消）
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:06` UTC（`18:52` ～ `19:06` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, OANDA `openTrades`

Fact:
- 24h: `1120 trades`, `-1009.3 pips`, `-11510.6 JPY`, `PF=0.455`
- 7d: `4189 trades`, `-2127.9 pips`, `-25067.5 JPY`, `PF=0.656`
- 24h 主損失は `scalp_ping_5s_c_live=-5633.2 JPY`, `scalp_ping_5s_b_live=-3102.6 JPY`, `M1Scalper-M1=-1265.8 JPY`
- `entry=1 & exit=0` の残骸は是正前に複数残存（`micropullbackema*`, `microtrendretest*`, `scalp_ping_5s_flow_live`）
- `openTradeCount=0`（OANDA、手動玉含めフラット）

Failure Cause:
1. C/B が負EVのまま高回転し、日次損失の主因を継続している。
2. `strategy_control` の stale flag（`entry=1 & exit=0`）が再発時の closeout テール要因になりうる。
3. 直近は `close_reject_no_negative` も C 側で多発し、EXIT遅延を誘発していた。

Improvement:
1. VM運用ガードを即時適用:
   - `strategy_control_flags` の `entry=1 & exit=0` を全解消（`entry=0 & exit=1` に補正）。
2. 即日止血:
   - `scalp_ping_5s_c` を `entry=0 & exit=1` として新規停止（EXITは許可）。
3. 収益機会の残し方:
   - `microtrendretest` / `microtrendretest-long` は `entry=1 & exit=1` へ再有効化。
4. 高リスク再発系は停止維持:
   - `micropullbackema*` と `scalp_ping_5s_flow*` は `entry=0 & exit=1` を維持。

Verification:
1. `strategy_control.db` 現在値: `entry=1 & exit=0` が `0` 件。
2. 直近1h: `orders.db` の `strategy_control_exit_disabled=0`、`close_reject_no_negative=0`。
3. OANDA `openTrades=[]` を確認（含み損ポジション持越しなし）。

Status:
- in_progress

## 2026-02-26 09:36 UTC / 2026-02-26 18:36 JST - 改善策統合プラン（P0/P1/P2）
Period:
- Synthesis window: `2026-02-24` ～ `2026-02-26`（既存3エントリ統合）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account/openTrades

Fact:
- 24h: `1211 trades`, `-5155.7 JPY`, `PF=0.773`
- 7d: `4277 trades`, `-25750.3 JPY`, `PF=0.650`
- 7d 主損失は `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY` と `STOP_LOSS_ORDER=-18393.1 JPY`
- `strategy_control_exit_disabled=10277`（`2026-02-24` 単日）
- 24h 主損失戦略: `scalp_ping_5s_c_live=-6094.2 JPY`, `scalp_ping_5s_b_live=-3141.3 JPY`
- core unit churn: `quant-order-manager starts=62`, `quant-market-data-feed starts=22`, `quant-scalp-ping-5s-c starts=30`（24h）

Failure Cause:
1. `entry=1 & exit=0` と broker `stopLossOnFill` 欠損が重なると、EXIT封鎖から closeout テールへ直結する。
2. `scalp_ping_5s_b/c` は勝率寄り運用で、`avg_win/avg_loss` と `jpy PF` が負のまま件数が先行している。
3. 手動玉込み余力監視と 1-trade loss cap の運用が弱く、大損1発の耐性が不足している。
4. core unit の再起動多発で、執行品質（reject/timeout/latency）の再現性が落ちる。

Improvement:
1. P0（当日）安全不変条件:
   - `entry=1 & exit=0` を自動補正で禁止（検知時は `entry=0` へ強制 + alert）。
   - 非manual新規注文は broker `stopLossOnFill` 必須化（欠損は `missing_broker_sl` reject）。
   - 手動玉込み余力で新規抑制を先行適用し、`MARKET_ORDER_MARGIN_CLOSEOUT` を即時遮断。
2. P1（24-48h）収益反転:
   - `scalp_ping_5s_c_live` は停止ではなく「低頻度・低サイズ運用」に固定し、`jpy PF>=1` まで上限解除しない。
   - `scalp_ping_5s_b_live` も同一KPI（`jpy PF`, `avg_win/avg_loss`, `avg_loss_jpy`）でロット再配分。
   - `dynamic_alloc` は pips優先を避け、`realized_jpy_per_1k_units` と `jpy_downside_share` 主導に固定。
3. P1（24-48h）EXIT詰まり解消:
   - `close_reject_no_negative` を strategy tag × exit_reason で棚卸しし、拒否条件の過剰部分を除去。
   - 負け玉の平均保有時間を短縮するため、reject系ステータスの即時監査を追加。
4. P2（72h）運用品質:
   - core unit ごとの restart budget を設定し、閾値超過時は新規改善より先に安定化を実施。
   - 日次で `orders/trades/strategy_control` を定型監査し、本台帳へ同フォーマット追記。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` が 0 件、`orders.db` で `strategy_control_exit_disabled` が 24h 連続 0 件。
2. 非manual filled の broker `stopLossOnFill` 設定率 `>= 99%`。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続で件数 0 / JPY損失 0。
4. `scalp_ping_5s_b/c` の 24h `jpy PF >= 1.00` かつ `avg_win/abs(avg_loss) >= 1.00`。
5. core unit の start/stop が各 24h で `<= 5`。
6. 全体 `all_24h JPY` が 3日連続でプラス。

Status:
- in_progress

## 2026-02-26 09:25 UTC / 2026-02-26 18:25 JST - 回転点まとめ（資産曲線を反転させる条件）
Period:
- 24h/7d の実績再集計: `datetime(close_time) >= now - 1 day / 7 day`
- Exit封鎖イベント確認: `2026-02-24`（`orders.db` / `strategy_control.db`）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account summary

Fact:
- 24h 合計: `1211 trades`, `+525.6 pips`, `-5155.7 JPY`, `PF=0.773`
- 7d 合計: `4277 trades`, `-2287.7 pips`, `-25750.3 JPY`, `PF=0.650`
- 24h の主損失戦略:
  - `scalp_ping_5s_c_live`: `545 trades`, `-6094.2 JPY`（勝率 `53.0%` でも `avg_win=+20.5 JPY < avg_loss=-48.1 JPY`）
  - `scalp_ping_5s_b_live`: `480 trades`, `-3141.3 JPY`
- 24h の反事実:
  - `all_24h = -5155.7 JPY`
  - `exclude C = +938.5 JPY`
  - `exclude B/C = +4079.7 JPY`
  - `exclude margin_closeout = -2991.9 JPY`
- 7d の損失内訳:
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY`（負け総額の `51.0%`）
  - `STOP_LOSS_ORDER=-18393.1 JPY`（同 `49.0%`）
- `2026-02-24` 単日に `strategy_control_exit_disabled=10277`。同日 `micropullbackema` で `margin closeout 4件 = -16837.4 JPY`。
- 現在も `strategy_control.db` に `exit_enabled=0` が残存:
  - `micropullbackema`, `micropullbackema-short`, `microtrendretest`, `microtrendretest-long`, `scalp_ping_5s_flow_live`
- 24h の unit churn（journal）:
  - `quant-order-manager starts=62 / stops=30`
  - `quant-market-data-feed starts=22 / stops=20`
  - `quant-scalp-ping-5s-c starts=30 / stops=28`

Failure Cause:
1. C/B の期待値が負のまま件数が多く、日次損失のベースを作っている。
2. Exit封鎖（`entry=1 & exit=0`）が実ポジに刺さると、損失玉が閉じられず closeout テールが発生する。
3. 再起動多発で order/position 連携が不安定化し、執行品質とリスク制御の再現性が落ちる。
4. `perf_block` と `probability_scaled` が大きく、良い局面の約定密度が不足する。

Improvement:
1. **回転点A（資産反転の最低条件）**: `entry=1 & exit=0` を本番で禁止（自動修正 + アラート）。
2. **回転点B（収益反転の主条件）**: `scalp_ping_5s_c_live` の EV を 0 以上にするまで、件数/サイズを強制的に抑える。
3. **回転点C（ドローダウン抑制）**: `MARKET_ORDER_MARGIN_CLOSEOUT` を連続 0 件に固定（manual玉込み余力ガード）。
4. **回転点D（実運用品質）**: core unit の restart/stop を抑制し、連続運転で検証する。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h 連続で 0 件。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 0 件。
3. `scalp_ping_5s_c_live` の 24h `ev_jpy >= 0` かつ `avg_loss_jpy` の絶対値縮小。
4. 日次反事実で `exclude C` が不要な状態（実績 `all_24h` が単独でプラス）へ移行。

Status:
- in_progress

## 2026-02-26 12:58 UTC / 2026-02-26 21:58 JST - `stage_state.db` ロックで market_order が失敗しエントリー停止
Period:
- Window: `2026-02-26 11:42` ～ `12:56` UTC
- Source: VM `journalctl -u quant-order-manager.service`, `logs/orders.db`, `logs/trades.db`

Fact:
- `quant-order-manager` は active だが、`[ORDER_MANAGER_WORKER] request failed: database is locked` が連続発生。
- `orders.db` 直近2h: `preflight_start=237`, `probability_scaled=99`, `perf_block=78`, `entry_probability_reject=19`, `slo_block=11`。
- `trades.db` 直近2h の新規 `entry_time` は `0` 件、open trades も `0` 件。
- 失敗直前ログは `OPEN_SCALE` / `projected margin scale` まで進み、その後に `database is locked` で abort している。

Failure Cause:
1. `execution/strategy_guard.py` が `logs/stage_state.db` に対してロック耐性なし（busy_timeout/retryなし）でアクセス。
2. 同DBを `stage_tracker` 系と共有するため、短時間の書き込み競合で `sqlite OperationalError: database is locked` が発生。
3. 例外が `market_order` 経路まで伝播し、preflight通過後でも発注まで到達しない。

Improvement:
1. `strategy_guard` に busy_timeout + WAL + lock retry を追加。
2. lock時は fail-open（`is_blocked=False`）で返し、エントリーを DB 競合で止めない。
3. `set_block` / `clear_expired` / expired削除も lock耐性化して例外伝播を防止。

Verification:
1. `pytest -q tests/test_stage_tracker.py` pass（3 passed）。
2. デプロイ後、`journalctl -u quant-order-manager.service` で `request failed: database is locked` が再発しないこと。
3. `orders.db` で `preflight_start` のみ増える状態が解消し、`filled` が復帰すること。

Status:
- in_progress

## 2026-02-26 13:10 UTC / 2026-02-26 22:10 JST - B/C エントリー枯渇に対する no-signal 緩和（revert依存解除 + side filter開放）
Period:
- Window: `2026-02-26 13:08` ～ `13:10` UTC
- Source: VM `journalctl -u quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service`

Fact:
- `quant-order-manager` の lockエラーは再発していないが、B/C ワーカーは `entry-skip` が継続。
- 主因は `no_signal:revert_not_found` と `units_below_min`（B/C とも short 側で 8〜15件/集計周期）。
- `side_filter_block` も残り、sell固定のままでは通過率が伸びにくい。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SIDE_FILTER=none`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=900`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=65`
   - `CONF_FLOOR=72`
2. `ops/env/scalp_ping_5s_c.env`
   - `SIDE_FILTER=none`, `ALLOW_NO_SIDE_FILTER=1`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=260`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=70`
   - `CONF_FLOOR=74`

Verification:
1. デプロイ後 10〜20 分で `orders.db` の `preflight_start` だけでなく `filled` が増えること。
2. `entry-skip summary` の `revert_not_found` 比率が低下すること。
3. `units_below_min` が `min_units_rescue` に置換され、0件へ収束すること。

Status:
- in_progress

## 2026-02-26 13:20 UTC / 2026-02-26 22:20 JST - B/C side filter を sell 再固定（精度優先）
Period:
- Analysis/retune window: `2026-02-26 13:10` ～ `13:20` UTC
- Source: repository `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Fact:
- 最新 `main` では B/C とも `SIDE_FILTER=none` + `ALLOW_NO_SIDE_FILTER=1` になっていた。
- 同時に `MIN_UNITS_RESCUE` は導入済みで、no-entry 側は rescue で緩和可能な状態だった。

Failure Cause:
1. side filter 開放により、方向精度劣化の再発リスク（buy 再流入）を残していた。

Improvement:
1. B/C を `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0` へ再固定。
2. `MIN_UNITS_RESCUE` は維持しつつ閾値を再引き上げ
   - B: `prob>=0.58`, `conf>=78`
   - C: `prob>=0.60`, `conf>=82`

Verification:
1. デプロイ後ログで `env mapped ... side_filter=sell` を確認。
2. restart 後の B/C journal で `side_filter_fallback:long->short` が継続して出ることを確認。

Status:
- in_progress

## 2026-02-26 09:15 UTC / 2026-02-26 18:15 JST - EXIT封鎖とSL未実装の複合で margin closeout が連鎖
Period:
- Incident window: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`2026-02-24 11:20:16` ～ `18:13:14` JST）
- P/L window: `datetime(close_time) >= now - 14 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, OANDA account/openTrades

Fact:
- `strategy_control_flags` に `entry=1 & exit=0` が残存（`note=manual_hold_current_positions_20260224`）:
  - `micropullbackema`, `micropullbackema-short`
  - `microtrendretest`, `microtrendretest-long`
  - `scalp_ping_5s_flow_live`
- `orders.db` の `strategy_control_exit_disabled` は合計 `10,277` 件。
  - 戦略別: `MicroPullbackEMA-short=8,177`, `MicroTrendRetest-long=2,068`, `scalp_ping_5s_flow_live=32`
- `MicroPullbackEMA` の margin closeout 4件（`384420/384425/384430/384435`）は、
  各 ticket で `strategy_control_exit_disabled` が `2,044～2,045` 回連続した後に強制クローズ。
  - 合計 `-16,837.4 JPY` / `-582.6 pips`
- 上記4件の filled 注文は `takeProfitOnFill` はある一方、`stopLossOnFill` が未設定（broker SLなし）。
- 14日合計: `-93,081.6 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-49,370.5 JPY`
  - 非closeoutでも `-43,711.0 JPY`
- 直近口座状態（2026-02-26 09:04 UTC）:
  - `NAV=57,731.21`, `margin_used=53,020.96`, `margin_available=4,744.25`
  - open trade に `TP/SL なし` の `USD_JPY -8500` が残存

Failure Cause:
1. `entry=1 & exit=0` の危険状態を作る運用が残り、保有調整を詰まらせた。
2. close経路が `strategy_control` により拒否され続け、損失玉が放置された。
3. broker側 `stopLossOnFill` 未設定の玉は、ローカルEXIT依存となり封鎖時に脆弱。
4. margin closeout を除いても `scalp_ping_5s_b/c` を中心に期待値が負で、資産減少が継続。

Improvement:
1. 運用ルール固定: `entry=1 & exit=0` を禁止し、保有維持が必要なときは `entry=0 & exit=1` のみ許可。
2. 自動ガード追加: `strategy_control` に `entry=1 & exit=0` が入った時点でアラート + 自動修正（`entry=0` へ強制）。
3. 発注必須化: 非manualの新規注文は `stopLossOnFill` 必須。欠損時は reject して `status=missing_broker_sl` を記録。
4. 既存残骸の解消: `manual_hold_current_positions_20260224` の stale flag を全解除し、原因ノート付きで再設定履歴を残す。
5. 平常時改善: `scalp_ping_5s_b/c` は `win_rate` ではなく `jpy PF` と `avg_win/avg_loss` を主指標に再設計。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` 行が 0 件。
2. `orders.db` の `strategy_control_exit_disabled` が 24h で 0 件。
3. 7日連続で `MARKET_ORDER_MARGIN_CLOSEOUT` の件数 0 / JPY損失 0。
4. 非manual filled の `stopLossOnFill` 設定率が `>= 95%`（戦略別に監査）。
5. `scalp_ping_5s_b/c` の14日 `jpy PF >= 1.0` かつ `avg_loss_jpy` の絶対値縮小を確認。

Status:
- in_progress

## 2026-02-26 08:40 UTC / 2026-02-26 17:40 JST - Margin Closeout Tail が資産毀損の主因
Period:
- 24h: `datetime(close_time) >= now - 1 day`
- 7d: `datetime(close_time) >= now - 7 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 24h 合計: `1213 trades`, `+555.4 pips`, `-4652 JPY`
- 7d 合計: `4283 trades`, `-2281.8 pips`, `-25242 JPY`
- 7d close reason:
  - `STOP_LOSS_ORDER`: `1752`, `-3980.5 pips`, `-18436 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT`: `17`, `-602.5 pips`, `-17415 JPY`
  - `MARKET_ORDER_TRADE_CLOSE`: `2217`, `+1805.6 pips`, `+315 JPY`
- 7d 戦略別（下位）:
  - `MicroPullbackEMA`: `46`, `-520.3 pips`, `-15527 JPY`（`margin_closeout 4件 = -16837 JPY`）
  - `scalp_ping_5s_c_live`: `855`, `-1006.7 pips`, `-10554 JPY`
  - `scalp_ping_5s_b_live`: `2093`, `-1879.5 pips`, `-9448 JPY`
- 口座スナップショット（VM実測）:
  - `nav=56813.21`, `margin_used=53057.68`, `margin_available=3789.53`, `health_buffer=0.06666`
  - open positions: `manual short -8500 units`
- 稼働確認:
  - `quant-scalp-ping-5s-b.service`: running, `SCALP_PING_5S_B_ENABLED=1`
  - `quant-scalp-ping-5s-c.service`: running, `SCALP_PING_5S_C_ENABLED=1`
  - `quant-scalp-ping-5s-d.service`: running, `SCALP_PING_5S_D_ENABLED=0`

Failure Cause:
1. `Margin closeout` を許す残余証拠金状態で稼働を継続し、少数の巨大損失が累積。
2. `scalp_ping_5s_c_live` は勝率が一定でもペイオフ負け（平均損失が平均利益を上回る）。
3. `close_reject_no_negative` が多発し、負け玉解放が遅れる経路が残る。
4. DD判定は存在するが、エントリー停止ガードへの直接連携が弱く、資産保全の最終防衛線になっていない。
5. 1トレード損失上限（loss cap）機能はあるが、運用値未設定で未活用。

Improvement:
1. 最優先は `margin closeout 回避`:
   - 余力逼迫時の新規抑制条件を先に適用（手動玉込みで判定）。
2. 次に `1トレード損失上限` を明示有効化:
   - `ORDER_ENTRY_LOSS_CAP_*` 系を strategy/pocket 単位で設定。
3. `close_reject_no_negative` の原因別棚卸し:
   - `exit_reason` と `neg_exit policy` の不整合を strategy tag 単位で解消。
4. `scalp_ping_5s_b/c` は勝率ではなく `avg_win/avg_loss` と `jpy PF` を第一指標に調整。
5. 24hごとに同一フォーマットで本台帳へ追記し、改善の効き/副作用を継続監査。

Verification:
1. `MARKET_ORDER_MARGIN_CLOSEOUT` の 24h 件数・JPY損失が連続減少。
2. 24h 合計で `pips` と `JPY` の符号乖離（`+pips / -JPY`）が解消。
3. `close_reject_no_negative` の件数が減少し、負け玉の平均保有時間が短縮。
4. `scalp_ping_5s_b/c` の `jpy PF` と `avg_loss_jpy` が改善。

Status:
- in_progress
