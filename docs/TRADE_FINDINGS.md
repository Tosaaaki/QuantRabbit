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
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF: 0.88 -> 0.58`
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN: 0.48 -> 0.27`
   - 反映先: `ops/env/quant-order-manager.env` と `ops/env/scalp_ping_5s_b.env`
2. B/C の方向固定を解除:
   - `SCALP_PING_5S_B_SIDE_FILTER=`
   - `SCALP_PING_5S_C_SIDE_FILTER=`
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
