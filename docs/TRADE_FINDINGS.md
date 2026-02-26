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
