# Ops Current (2026-02-11 JST)

## 0-13. 2026-02-27 UTC `scalp_ping_5s_c` 第9ラウンド（約定再開後のロット底上げ）
- 背景（VM実測, UTC 2026-02-27 14:00-14:02）:
  - Round8b 後、`perf_block` は解消し `orders.db` で `filled` が再開。
  - ただし再開直後の C 約定は `buy 5 fills` で `avg_units=1.3`（min=1, max=2）と小口に偏った。
  - `entry_thesis.entry_units_intent` も `1-2` が中心で、long 露出が不足していた。
- 対応（`ops/env/scalp_ping_5s_c.env`）:
  - `BASE_ENTRY_UNITS: 80 -> 140`
  - `MIN_UNITS: 1 -> 5`
  - `MAX_UNITS: 160 -> 260`
  - `ALLOW_HOURS_OUTSIDE_UNITS_MULT: 0.55 -> 0.70`
  - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT: 0.58 -> 0.72`
  - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT: 0.85 -> 1.00`
- 意図:
  - 通過再開を維持したまま、long の実効ロット下限を引き上げる。
  - 低ロット連打による収益立ち上がり遅延を縮める。

## 0-12. 2026-02-27 UTC `scalp_ping_5s_c` 第8ラウンド（order-manager env同期で hard perf block を解除）
- 背景（VM実測, UTC 2026-02-27 13:47-13:52）:
  - Round7反映後の `orders.db`（`scalp_ping_5s_c_live`）は
    `perf_block=39`, `entry_probability_reject=11`, `probability_scaled=9`, `filled=0`。
  - `quant-scalp-ping-5s-c` の reject は `reason=perf_block` が連続し、
    送信フェーズへ到達しない状態だった。
  - VM上で `perf_guard.is_allowed(...)` を実測すると:
    - `quant-order-manager.env` + `quant-v2-runtime.env` 読込:  
      `allowed=False, reason='hard:hour13:failfast:pf=0.32 win=0.36 n=22'`
    - そこへ `scalp_ping_5s_c.env` を追加読込:  
      `allowed=True, reason='warn:margin_closeout_soft...'`
  - つまり order-manager サービスが読む env と worker 側 env が乖離し、
    C の failfast が order-manager 側だけ過緊縮で hard block 化していた。
- 対応（`ops/env/quant-order-manager.env`）:
  - C preserve-intent を worker 側運用値へ同期:
    - `REJECT_UNDER: 0.76 -> 0.74`
    - `MIN_SCALE: 0.24 -> 0.34`
    - `MAX_SCALE: 0.50 -> 0.56`
  - C failfast（+fallback）を worker 側運用値へ同期:
    - `SCALP_PING_5S[_C]_PERF_GUARD_FAILFAST_MIN_TRADES: 8 -> 30`
    - `..._FAILFAST_PF: 0.90 -> 0.20`
    - `..._FAILFAST_WIN: 0.48 -> 0.20`
  - 再起動後に `hard:sl_loss_rate=0.50 pf=0.32 n=22` が主因化したため、
    C `SL_LOSS_RATE` ガードも warmup 寄りへ同期:
    - `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_MIN_TRADES: 16 -> 30`
    - `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_MAX: 0.55/0.50 -> 0.68`
- 意図:
  - order-manager preflight の hard failfast を解除し、long の送信・約定再開を優先する。
  - 閾値ソースを worker と揃えて、同一戦略が経路差で別挙動になる状態を解消する。

## 0-11. 2026-02-27 UTC `scalp_ping_5s_b/c` long-side RR改善 + lot圧縮緩和
- 背景（VM実測, UTC 2026-02-27 09:20 集計）:
  - 24h side集計で `long` は `755 trades / -730.1 JPY / avg_units=185.8`、
    `short` は `277 trades / +1448.6 JPY / avg_units=338.9`。
  - `scalp_ping_5s_b_live long` は `avg_sl=2.03 pips` に対し `avg_tp=0.99 pips`
    （`tp/sl=0.49`）で、`-565.8 JPY`。
  - `scalp_ping_5s_c_live long` は `avg_sl=1.30 pips` に対し `avg_tp=0.90 pips`
    （`tp/sl=0.69`）で、`-117.8 JPY`。
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - long RR補正: `SL_BASE 1.6 -> 1.35`, `SL_MAX 2.4 -> 2.0`,
      `TP_BASE 0.35 -> 0.55`, `TP_MAX 1.4 -> 1.9`, `TP_NET_MIN 0.35 -> 0.45`
    - short据え置き: `SHORT_TP_BASE/SHORT_TP_MAX` を明示
    - lot圧縮緩和: `BASE_ENTRY_UNITS 220 -> 260`, `MAX_UNITS 750 -> 900`,
      `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.80 -> 0.95`,
      `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.32 -> 0.42`
  - `ops/env/scalp_ping_5s_c.env`
    - long RR補正: `SL_BASE 1.3 -> 1.15`, `SL_MIN=0.85`, `SL_MAX=1.9`,
      `TP_BASE 0.20 -> 0.45`, `TP_MAX 1.0 -> 1.5`, `TP_NET_MIN 0.25 -> 0.40`
    - short据え置き: `SHORT_TP_*` / `SHORT_SL_*` を明示
    - lot圧縮緩和: `BASE_ENTRY_UNITS 70 -> 95`, `MAX_UNITS 160 -> 220`,
      `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.75 -> 0.90`,
      `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.50 -> 0.62`
- 意図:
  - long側の「SL広い/利幅小さい」を是正して `avg_win/avg_loss` を改善しつつ、
    過度に潰れていた units を小幅復元する。

## 0-10. 2026-02-26 UTC `scalp_ping_5s_c_live` 時間帯停止を解除（実証不足のため）
- 背景（VM実測, UTC 2026-02-26 07:16 集計）:
  - 直近14日 `scalp_ping_5s_c_live` のJST時間帯別はばらつきが大きく、
    許可時間帯にしていた `18` 時は `34 trades / -698.8 JPY / -280.9 pips` と悪化。
  - 許可バケット（`18/19/22`）合算は `181 trades / -325.3 JPY` だが、
    これは観測集計であり、時間帯ブロック自体の因果効果（A/B）を示すものではない。
  - `journalctl -u quant-scalp-ping-5s-c.service` では
    `entry-skip ... outside_allow_hour_jst` が継続し、
    時間帯フィルタでエントリーを止める状態が常態化していた。
- 対応:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`
    - 時間帯ブロックを解除し、戦略ローカル判定（forecast/perf/risk）で常時運転へ戻す。
- 意図:
  - 実証不足のフィルタでエントリー機会を機械的に落とさず、
    効果が定量確認できるガード（確率・flip・risk）に責務を戻す。

## 0-9. 2026-02-26 UTC `scalp_ping_5s_c_live` 方向転換デリスク（SL連発帯の過発火抑制）
- 背景（VM実測, UTC 2026-02-26 07:05 前後）:
  - `ui_state` 直近60件で `flip_any=19`（`fast=14 / sl=2 / side=3`）、
    かつ全て `short->long` 偏重だった。
  - 同期間の平均 `pl_pips` は `flipあり=-3.12` / `flipなし=-0.84` で、
    反転経路の損失寄与が上回っていた。
  - 直近12件では `STOP_LOSS_ORDER=11` と SL 連発帯が継続。
- 対応（`ops/env/scalp_ping_5s_c.env`）:
  - 発火の絞り込み:
    - `FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN: 0.50 -> 0.58`
    - `FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN: 0.30 -> 0.38`
    - `FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN: 2 -> 3`
    - `FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN: 0.76 -> 0.82`
    - `FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS: 0.08 -> 0.16`
    - `FAST_DIRECTION_FLIP_COOLDOWN_SEC: 0.6 -> 1.2`
    - `FAST_DIRECTION_FLIP_CONFIDENCE_ADD: 3 -> 1`
  - `SL streak` 反転の過剰許可を抑制:
    - `MIN_SIDE_SL_HITS: 2 -> 3`
    - `MIN_TARGET_MARKET_PLUS: 1 -> 2`
    - `FORCE_STREAK: 3 -> 4`
    - `METRICS_OVERRIDE_ENABLED: 1 -> 0`
    - `DIRECTION_SCORE_MIN: 0.48 -> 0.55`
    - `HORIZON_SCORE_MIN: 0.30 -> 0.42`
  - `side_metrics` 反転を停止:
    - `SIDE_METRICS_DIRECTION_FLIP_ENABLED: 1 -> 0`
  - 同時にロット/通過も圧縮:
    - `MAX_ORDERS_PER_MINUTE: 6 -> 4`
    - `BASE_ENTRY_UNITS: 900 -> 700`
    - `MAX_UNITS: 1800 -> 1200`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...: 0.55 -> 0.60`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE...: 0.55 -> 0.40`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE...: 1.00 -> 0.85`
- 意図:
  - 方向転換ロジック自体は維持しつつ、SL連発局面での「反転起点の過大ロット連打」を先に止める。

## 0-8. 2026-02-26 UTC EV反転ホットフィックス（`scalp_ping_5s_c_live` + `dynamic_alloc`）
- 背景（VM実測, UTC 2026-02-26 06:46 前後）:
  - 直近24hの戦略別で `scalp_ping_5s_c_live` が `559 trades / -729.7 pips / -6,882 JPY`。
  - 直近7dでは `scalp_ping_5s_c_live` が `+918.7 pips` でも `-2,592 JPY` となり、
    pips基準配分が実損益と乖離していた。
  - `config/dynamic_alloc.json` で同戦略に `lot_multiplier=1.201` が配分され、
    実損益悪化時にもサイズが維持される状態だった。
- 対応:
  - `scripts/dynamic_alloc_worker.py`
    - `realized_pl` と `units` を集計対象に追加。
    - `realized_jpy_per_1k_units` / `jpy_pf` / `jpy_downside_share` を導入し、
      スコアと lot multiplier に反映。
    - `sum_realized_jpy` が一定以上マイナスの戦略へ段階的 cap
      （`0.88 -> 0.70 -> 0.55`）を適用。
  - `ops/env/scalp_ping_5s_c.env`
    - `SIDE_FILTER=buy`
    - `MAX_ACTIVE_TRADES=2`, `MAX_PER_DIRECTION=2`
    - `MAX_ORDERS_PER_MINUTE=6`
    - `BASE_ENTRY_UNITS=1200`, `MAX_UNITS=1800`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]=0.60`
    - `MIN_SCALE=0.40`, `MAX_SCALE=0.85`
- 意図:
  - 「pipsは勝っているがJPYで負ける」逆配分を止める。
  - C戦略の同方向クラスターと低確率通過を先に削り、ドローダウンの傾きを下げる。

## 0-7. 2026-02-25 UTC `order_manager_none` + `orders.db stale` 事象（原因確定）
- 症状（VM実測, UTC 2026-02-25 09:24-09:47）:
  - strategy worker で `order_manager service call failed ... Read timed out (read timeout=20.0)` と
    `order_manager_none` が断続。
  - `orders.db` が `MAX(ts)=2026-02-25T04:17:20+00:00` で停止したように見え、
    `orders.db-wal` が `8.6GB` まで肥大。
  - `quant-order-manager` で `failed to persist orders log: database is locked` が多発。
- 原因:
  - 発注導線の lock競合:
    - service/fallback/aux worker が `orders.db` へ同時書き込みし、
      `SQLITE_BUSY` が連鎖。
    - `entry_intent_board` の不要 write（参照前 delete）も競合を増幅。
  - 運用側プロセスの残留:
    - `sqlite3 VACUUM` と `cron` の `tar` が `orders.db(-wal/-shm)` を長時間保持し、
      checkpoint/truncate が進まず stale 化を助長。
  - 可観測性の欠陥:
    - `ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE=0` の経路では
      local reject が DB に残らず、strategy 側で `order_manager_none` と見えていた。
- 恒久対応（main反映済み）:
  - `execution/order_manager.py`
    - `orders.db.lock` の `flock` で write を直列化。
    - `entry_intent_board` の purge/record に rollback/reset を追加。
    - `coordinate_entry_intent` の重複削除 write を廃止。
    - `ORDER_STATUS_CACHE_TTL_SEC` を追加し、DB未記録経路でも
      `entry_probability_reject*` を返せるようにした。
  - env:
    - `ORDER_DB_FILE_LOCK_*` / `ORDER_STATUS_CACHE_TTL_SEC` を
      `quant-order-manager.env` と `quant-v2-runtime.env` に追加。
- 復旧手順（実施済み）:
  - 残留 `sqlite3` / `tar` を停止して FD を解放。
  - `PRAGMA wal_checkpoint(TRUNCATE)` で WAL を縮退（`8.6GB -> 0` 確認）。
  - `orders.db` 最新時刻追従を再確認。
- 監視の即時チェック:
  - `ls -lh /home/tossaki/QuantRabbit/logs/orders.db*`
  - `sqlite3 ... 'SELECT id,ts,status FROM orders ORDER BY id DESC LIMIT 5;'`
  - `journalctl ... | grep -E 'order_manager service call failed|database is locked|order_manager_none'`
  - `sudo lsof /home/tossaki/QuantRabbit/logs/orders.db*`

## 0-6. 2026-02-24 UTC `order_manager` の発注遅延ホットフィックス（orders.db lock待機短縮）
- 背景（VM実測）:
  - `preflight_start -> submit_attempt` が `23s / 64s / 91s / 167s` の遅延を記録。
  - 同時間帯に `quant-order-manager` で `failed to persist orders log: database is locked` が連発。
  - `ops/env` で `ORDER_DB_BUSY_TIMEOUT_MS=5000` かつ `RETRY_ATTEMPTS=8` が有効で、
    1回のログ書き込みが長時間ブロックし得る設定になっていた。
- 反映:
  - `ops/env/quant-v2-runtime.env`
  - `ops/env/quant-order-manager.env`
  - `ORDER_DB_BUSY_TIMEOUT_MS=250`
  - `ORDER_DB_LOG_RETRY_ATTEMPTS=3`
  - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.02`
  - `ORDER_DB_LOG_RETRY_BACKOFF=1.5`
  - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.10`
- 目的:
  - ロック発生時の待機時間上限を縮めて、発注経路の詰まりを先に解消する。
  - 監査ログ欠落より、約定遅延と取り残しの抑制を優先する。

## 0-5. 2026-02-19 UTC `scalp_ping_5s_b_live` 方向転換遅れ + 注文ログ詰まりの同時是正
- 背景（VM実測）:
  - `sl_streak_direction_flip_reason` が `below_min_streak / streak_stale` に偏り、SL連敗後の反転が不発。
  - `quant-order-manager` と `quant-scalp-ping-5s-b` で `failed to persist orders log: database is locked` が多発し、
    `coordinate_entry_intent` timeout と `order_manager_none` を誘発。
- 実装反映:
  - `workers/scalp_ping_5s/worker.py`
    - `SL streak flip` に `metrics override` を追加。
      - 連続SL streak が不足/失効でも、side別 recent統計（`SL hits / side trades / opposite MARKET+`）が
        閾値を満たす場合は方向反転を許可。
      - 監査 reason に `slrate` と `m_ovr`（metrics override）を追加。
  - `workers/scalp_ping_5s/config.py`
    - 追加パラメータ:
      - `SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED`
      - `SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_TRADES_MIN`
      - `SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_SL_RATE_MIN`
  - `execution/order_manager.py`
    - `ORDER_DB_BUSY_TIMEOUT_MS` の既定を `5000 -> 250` ms に短縮
      （orders.db lock待機で発注経路が詰まるのを回避）。
  - `ops/env/scalp_ping_5s_b.env` / `ops/env/quant-order-manager.env`
    - `SL_STREAK_DIRECTION_FLIP_METRICS_*` をBへ明示設定。
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE: 0.45 -> 0.42`
      （方向補正後probの通過率を少し戻し、件数低下を抑制）。
    - `ORDER_DB_BUSY_TIMEOUT_MS=250` を反映。
- 目的:
  - 「SLを何回か踏んだ側」を早く切り替える。
  - DBロック由来の timeout/reject で反転シグナル自体が失われる問題を抑える。

## 0-4. 2026-02-19 UTC `scalp_ping_5s_b_live` 緊急デリスク（連続マイナス拡大の抑制）
- 背景（VM実測）:
  - `scalp_ping_5s_b_live` 直近2hで `-104.8 pips`。
  - 損失主因は `long + STOP_LOSS_ORDER`（`155件 / -373.8 pips`）。
  - `SL streak flip` が `streak_stale` で不発になる比率が高かった。
  - 同方向の同時建玉が増えやすく、SLクラスターで損失が拡大。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - 建玉クラスター抑制:
    - `MAX_ACTIVE_TRADES: 40 -> 20`
    - `MAX_PER_DIRECTION: 24 -> 12`
  - 方向転換の有効期限を延長:
    - `SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC: 180 -> 480`
    - `SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES: 24 -> 36`
  - side成績連動のロット減衰を強化:
    - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN: 0.65 -> 0.90`
    - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT: 0.70 -> 0.60`
  - 片側シグナルの取りこぼし低減:
    - `MIN_UNITS: 150 -> 100`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 150 -> 100`
- 目的:
  - エントリー頻度を極端に落とさず、同方向クラスター起因のSL連鎖を抑える。
  - 連続SL時の方向転換を「失効しにくい」状態にする。

## 0-3. 2026-02-19 UTC `scalp_ping_5s_b_live` 逆配分・反転遅れの根本補正
- 背景（VM実績, 直近600 close）:
  - 高確率帯 (`entry_probability>=0.90`) が劣後継続。
    - long: `n=268`, `avg=-0.485 pips`, `SL率=45.5%`
    - short: `n=100`, `avg=-1.532 pips`, `SL率=77.0%`
  - ロット逆配分が残存（高確率帯ほど平均unitsが大きい）。
- 実施:
  - `workers/scalp_ping_5s/worker.py`
    - `entry_probability` の floor 適用を厳格化。
      - 逆方向優勢（`support < counter`）または counter過大時は floor を不適用。
    - `SL streak flip` に強制反転バイパスを追加。
      - 連敗が `force_streak` 以上で `SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM=1` の場合、
        tech confirm 未充足でも方向反転を許可。
    - `confidence` ロット増幅を `CONFIDENCE_SCALE_MIN/MAX_MULT` で可変化し、
      Bでは過大増幅を圧縮。
  - `workers/scalp_ping_5s/config.py`
    - 追加:
      - `CONFIDENCE_SCALE_MIN_MULT`, `CONFIDENCE_SCALE_MAX_MULT`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER`
      - `SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM`
  - `ops/env/scalp_ping_5s_b.env`
    - 反映:
      - `CONF_SCALE_MIN/MAX_MULT` を追加
      - `ENTRY_PROBABILITY_BAND_ALLOC_*` の high削減/side減衰を強化
      - `FAST_DIRECTION_FLIP_*` を前倒し
      - `SL_STREAK_DIRECTION_FLIP_*` を前倒し + `FORCE_WITHOUT_TECH_CONFIRM=1`
- 目的:
  - エントリー頻度を落とさず、`高確率=高ロット` の過信を抑制。
  - 連続SL局面での方向転換を早め、損失テールを短縮する。

## 0-2. 2026-02-19 UTC `scalp_macd_rsi_div_b_live` 精度優先チューニング
- 背景:
  - VM `trades.db`（UTC 2026-02-18 02:22〜2026-02-19 01:33）で
    `scalp_macd_rsi_div_b_live` が `4 trades / PF=0.046 / -32.9 pips`。
  - `tick_entry_validate`（ticket `365759`）で
    `TP_touch<=600s = 0/1` を確認し、逆行局面での誤発火を優先是正。
- 対応（`ops/env/quant-scalp-macd-rsi-div-b.env`）:
  - レンジ限定:
    - `SCALP_MACD_RSI_DIV_B_REQUIRE_RANGE_ACTIVE=1`
    - `SCALP_MACD_RSI_DIV_B_RANGE_MIN_SCORE=0.35`
    - `SCALP_MACD_RSI_DIV_B_MAX_ADX=30`
  - シグナル品質の引き締め:
    - `SCALP_MACD_RSI_DIV_B_MIN_DIV_SCORE=0.08`
    - `SCALP_MACD_RSI_DIV_B_MIN_DIV_STRENGTH=0.12`
    - `SCALP_MACD_RSI_DIV_B_MAX_DIV_AGE_BARS=24`
    - `SCALP_MACD_RSI_DIV_B_RSI_LONG_ENTRY=36`
    - `SCALP_MACD_RSI_DIV_B_RSI_SHORT_ENTRY=62`
  - エクスポージャ抑制:
    - `SCALP_MACD_RSI_DIV_B_MAX_OPEN_TRADES=1`
    - `SCALP_MACD_RSI_DIV_B_COOLDOWN_SEC=45`
    - `SCALP_MACD_RSI_DIV_B_BASE_ENTRY_UNITS=5000`
  - fail-open 経路を停止:
    - `SCALP_MACD_RSI_DIV_B_TECH_FAILOPEN=0`

## 0-1. 2026-02-18 UTC MicroCompressionRevert デリスク（専用調整）
- 背景:
  - 直近24hで `MicroCompressionRevert-short` が `PF<1`（特に同時多発エントリー後の損失クラスター）を確認。
- 対応（`ops/env/quant-micro-compressionrevert.env`）:
  - サイズ縮小:
    - `MICRO_MULTI_BASE_UNITS=14000`（従来 28000）
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroCompressionRevert:0.45`
  - 同時多発抑制:
    - `MICRO_MULTI_MAX_SIGNALS_PER_CYCLE=1`
    - `MICRO_MULTI_MULTI_SIGNAL_MIN_SCALE=0.55`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=120`
  - 低成績ブロックを前倒し:
    - `MICRO_MULTI_HIST_MIN_TRADES=8`
    - `MICRO_MULTI_HIST_SKIP_SCORE=0.55`
    - `MICRO_MULTI_DYN_ALLOC_MIN_TRADES=8`
    - `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.45`
- EXIT調整（`config/strategy_exit_protections.yaml` `MicroCompressionRevert.exit_profile`）:
  - `range_max_hold_sec=900`
  - `loss_cut_soft_sl_mult=0.95`
  - `loss_cut_hard_sl_mult=1.20`（従来 1.60）
  - `loss_cut_max_hold_sec=900`（従来 2400）
  - `loss_cut_cooldown_sec=4`
  - 追加: `profit/trail/lock` の明示値（`profit_pips=1.1`, `trail_start_pips=1.5` など）

## 0. 2026-02-17 UTC 5秒スキャをB専用へ固定
- 無印5秒スキャ（`scalp_ping_5s_live`）の運用導線を削除。
  - 削除: `quant-scalp-ping-5s.service`, `quant-scalp-ping-5s-exit.service`
  - 削除: `ops/env/quant-scalp-ping-5s.env`, `ops/env/quant-scalp-ping-5s-exit.env`, `ops/env/scalp_ping_5s.env`
- 5秒スキャは B版（`scalp_ping_5s_b_live`）のみ稼働。
  - 使用env: `ops/env/quant-scalp-ping-5s-b.env`, `ops/env/quant-scalp-ping-5s-b-exit.env`, `ops/env/scalp_ping_5s_b.env`
- 2026-02-17 UTC 追加: 現況追従のエントリー頻度回復チューニング
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_B_REVERT_CONFIRM_TICKS=1`
    - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=1`
    - `SCALP_PING_5S_B_MIN_UNITS=150`
    - `SCALP_PING_5S_B_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=1`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_BLOCK_POS=0.10`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_POS=0.18`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=150`
  - `scalp_ping_5s_flow`
    - `SCALP_PING_5S_FLOW_MIN_TICKS=4`
    - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS=3`
    - `SCALP_PING_5S_FLOW_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_TICKS=3`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_SIGNAL_TICKS=3`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_IMBALANCE_MIN=0.50`
    - `SCALP_PING_5S_FLOW_MOMENTUM_TRIGGER_PIPS=0.10`
    - `SCALP_PING_5S_FLOW_SHORT_MOMENTUM_TRIGGER_PIPS=0.10`
    - `SCALP_PING_5S_FLOW_MOMENTUM_SPREAD_MULT=0.12`
    - `SCALP_PING_5S_FLOW_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=1`
    - `SCALP_PING_5S_FLOW_LOOKAHEAD_GATE_ENABLED=0`
    - `SCALP_PING_5S_FLOW_REVERT_WINDOW_SEC=1.60`
    - `SCALP_PING_5S_FLOW_REVERT_SHORT_WINDOW_SEC=0.55`
    - `SCALP_PING_5S_FLOW_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_FLOW_REVERT_CONFIRM_TICKS=1`
    - `SCALP_PING_5S_FLOW_REVERT_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_REVERT_RANGE_MIN_PIPS=0.35`
    - `SCALP_PING_5S_FLOW_REVERT_SWEEP_MIN_PIPS=0.22`
    - `SCALP_PING_5S_FLOW_REVERT_BOUNCE_MIN_PIPS=0.08`
    - `SCALP_PING_5S_FLOW_REVERT_CONFIRM_RATIO_MIN=0.50`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_PIPS=0.15`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_TICKS=3`
- 2026-02-17 UTC 追加: 極値反転ルーティング（件数維持 + 方向補正）
  - `workers/scalp_ping_5s` に `EXTREMA_REVERSAL_*` を追加。
  - `SCALP_PING_5S_B` は既定で `EXTREMA_REVERSAL_ENABLED=1`。
  - `short_bottom_*` / `long_top_*` で反転根拠が揃う場合、
    block ではなく side 反転 (`*_extrev`) で注文継続。
  - `entry_thesis` 監査項目:
    - `extrema_reversal_applied`
    - `extrema_reversal_score`
- 2026-02-19 UTC 追加: ショート偏重クラスタ抑制（方向補正の非対称化）
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT=0.42`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT=0.30`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE=2.10`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=3`
- 2026-02-19 UTC 追加: flip過発火の緊急デリスク（方向上書き厳格化）
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.52`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.32`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN=3`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.82`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS=0.18`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_CONFIDENCE_ADD=2`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_COOLDOWN_SEC=1.2`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE=0.60`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES=10`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS=3`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS=2`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=5`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.55`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.42`
- 2026-02-19 UTC 追加: `quant-order-manager` 側の `perf_block` 是正（件数維持）
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=warn`
  - 目的:
    - `OPEN_REJECT note=perf_block:margin_closeout_n=...` を防ぎ、
      scalp_ping_5s_b の方向改善ロジックを発注欠落なく評価する。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` 利伸ばし設定（exit最適化）
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` を個別exit_profile化
    - `profit_pips=2.0`
    - `trail_start_pips=2.3`
    - `trail_backoff_pips=0.95`
    - `lock_buffer_pips=0.70`
    - `lock_floor_min_hold_sec=45`
    - `range_profit_pips=1.6`
    - `range_trail_start_pips=2.0`
    - `range_trail_backoff_pips=0.80`
    - `range_lock_buffer_pips=0.55`
  - 目的:
    - `lock_floor` での早取り（平均 +0.6p）を減らし、
      `take_profit` 側での利伸ばし比率を上げる。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` ロット逆転是正（確率スケール平坦化）
  - 観測（直近24h, VM）:
    - `win3` 平均ロット `620.6` に対し、`loss24` 平均ロット `864.7`
    - `ep>=0.90` は `avg_units=1464.4` なのに `avg_pips=-0.95`
  - 反映:
    - `ops/env/quant-order-manager.env`
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.45`
      - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
    - `ops/env/scalp_ping_5s_b.env`
      - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
  - 目的:
    - 高確率時の過大ロットを抑えつつ、低確率側の過小ロットを補正する。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` 高確率遅行補正（方向整合prob再校正）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - `ep>=0.90`: `367件`, `avg_pips=-0.59`
    - ショート `ep>=0.90`: `123件`, `avg_pips=-1.07`, `SL率=63.4%`
    - 発注遅延は主要因でなく `preflight->fill ≒ 381ms`
  - 反映:
    - `workers/scalp_ping_5s/config.py`
      - `ENTRY_PROBABILITY_ALIGN_*`（direction/horizon/m1 加重、penalty/boost、units follow）
    - `workers/scalp_ping_5s/worker.py`
      - `entry_probability` を `confidence` 直結から整合再校正へ変更
      - `probability_units_mult` をサイズ計算に追加
      - `entry_thesis` に `entry_probability_raw` と `entry_probability_alignment.*` を追加
    - `ops/env/scalp_ping_5s_b.env`
      - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_*` を追加
  - 目的:
    - 高確率の過大評価による逆行ロット集中を抑え、
      エントリー頻度を大きく落とさずに損失振れ幅を下げる。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` short反転撤退の高速化（サイド別EXIT）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - `short + MARKET_ORDER_TRADE_CLOSE`: `n=831`, `avg=-2.666p`, `avg_hold=625s`
    - `900s+` 保有の short MARKET close: `n=278`, `PL=-12,537.1 JPY`（主損失塊）
  - 反映:
    - `workers/scalp_ping_5s/exit_worker.py`
      - `non_range_max_hold_sec_<side>` と `direction_flip.<side>_*` オーバーライドを追加
    - `config/strategy_exit_protections.yaml`（`scalp_ping_5s_b_live`）
      - `non_range_max_hold_sec_short=300`
      - `direction_flip.short_*` を追加
        - `min_hold_sec=45`
        - `min_adverse_pips=1.0`
        - `score_threshold=0.56`
        - `release_threshold=0.42`
        - `confirm_hits=2`
        - `confirm_window_sec=18`
        - `de_risk_threshold=0.50`
        - `forecast_weight=0.45`
  - 目的:
    - short逆行を長時間抱える tail を圧縮し、反転時の微益/小損撤退を前倒しする。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` 確率帯ロット再配分（逆配分是正）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - 高確率帯（`ep>=0.90`）の成績劣後に対してロットが重く、
      低確率帯（`ep<0.70`）の優位局面でロットが不足していた。
  - 反映:
    - `workers/scalp_ping_5s/config.py`
      - `ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
    - `workers/scalp_ping_5s/worker.py`
      - `trades.db` の帯別統計（`<0.70` / `>=0.90`）で
        `probability_band_units_mult` を算出し、最終ロットへ反映。
      - side別 `SL hit` と `MARKET_ORDER_TRADE_CLOSE +` 件数を `side_mult` に反映。
      - `entry_thesis` に `entry_probability_band_units_mult` と
        `entry_probability_band_allocation.*` を記録。
    - `ops/env/scalp_ping_5s_b.env`
      - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
  - 目的:
    - エントリー頻度は維持しつつ、負け筋への過大ロット集中を抑え、
      勝ち筋への配分を増やす。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` 利小損大是正（EXIT非対称の強化）
  - 背景（VM, 直近12h）:
    - 平均勝ち `+2.021p` に対し平均負け `-3.634p`。
    - `MARKET_ORDER_TRADE_CLOSE` の負け平均が `-6.343p` と大きく、
      特に short 側で長時間保有後の深いマイナスが残っていた。
  - 反映（`config/strategy_exit_protections.yaml`）:
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` の exit profile を再調整
      - 利益側: `profit_pips/trail_start` を引き上げ（利を伸ばす）
      - 損失側: `loss_cut_hard_pips=6.0`, `loss_cut_hard_cap_pips=6.2`,
        `loss_cut_max_hold_sec=900` に圧縮
      - `non_range_max_hold_sec_short=240` を `b_live` にも明示
      - `direction_flip` の short 閾値を前倒し（早期 de-risk/close）
  - 目的:
    - 「勝ちをやや伸ばす」よりも先に「負けを深くしない」を強化し、
      1トレード当たりの損益非対称を改善する。

## 1. 2026-02-12 JST 追加チューニング（稼働戦略のみ）
- `TickImbalance` / `LevelReject` / `M1Scalper` だけを対象に EXIT の time-stop を短縮。
  - `TickImbalance`: `range_max_hold_sec=600`, `loss_cut_max_hold_sec=600`
  - `LevelReject`: `range_max_hold_sec=1200`, `loss_cut_max_hold_sec=1200`
  - `M1Scalper`: `range_max_hold_sec=300`, `loss_cut_max_hold_sec=300`
- `M1Scalper` は本番上書きでサイズを半減。
  - `config/vm_env_overrides_aggressive.env`: `M1SCALP_BASE_UNITS=7500`, `M1SCALP_EXIT_MAX_HOLD_SEC=300`
- `micro_multistrat` に戦略別サイズ倍率を追加。
  - 新規 env: `MICRO_MULTI_STRATEGY_UNITS_MULT`（例: `TickImbalance:0.70,LevelReject:0.70`）

## 2. 運用モード（2025-12 攻め設定）
- マージン活用を 85–92% 目安に引き上げ。
- ロット上限を拡大（`RISK_MAX_LOT` 既定 10.0 lot）。
- 手動ポジションを含めた総エクスポージャでガード。
- PF/勝率の悪い戦略は自動ブロック。
- 必要に応じて `PERF_GUARD_GLOBAL_ENABLED=0` で解除。
- 2026-02-09 以降、`env_prefix` を渡す worker の設定解決は「`<PREFIX>_UNIT_*` → `<PREFIX>_*` のみ」。グローバル `*` へのフォールバックは無効。

## 3. 2026-02-06 JST 時点の `fx-trader-vm` mask 済みユニット
```
quant-scalp-impulseretrace.service
quant-scalp-impulseretrace-exit.service
quant-scalp-multi.service
quant-scalp-multi-exit.service
quant-pullback-s5.service
quant-pullback-s5-exit.service
quant-pullback-runner-s5.service
quant-pullback-runner-s5-exit.service
quant-range-comp-break.service
quant-range-comp-break-exit.service
quant-scalp-reversal-nwave.service
quant-scalp-reversal-nwave-exit.service
quant-vol-spike-rider.service
quant-vol-spike-rider-exit.service
quant-tech-fusion.service
quant-tech-fusion-exit.service
quant-macro-tech-fusion.service
quant-macro-tech-fusion-exit.service
quant-micro-pullback-fib.service
quant-micro-pullback-fib-exit.service
quant-manual-swing.service
quant-manual-swing-exit.service
```

- 2026-02-06 JST: `quant-m1scalper*.service` は一度 VM からアンインストール（ユニット削除）。
- 2026-02-11 JST: `quant-m1scalper.service` は再導入され、`enabled/active` を確認。

## 0-12. 2026-02-27 UTC `scalp_ping_5s_b/c` 無約定化対策（revert + rate-limit 緩和）
- 背景（VM実測, UTC 2026-02-27 10:50 前後）:
  - 反映後ログで `entry-skip` 主因が `no_signal:revert_not_found` と `rate_limited` に集中。
  - 直近300行集計:
    - B: `summary_total=757`, `revert_not_found=278`, `rate_limited=123`
    - C: `summary_total=724`, `revert_not_found=289`, `rate_limited=92`
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE: 4 -> 6`
    - `REVERT_*` を緩和（`MIN_TICKS=1`, `RANGE_MIN=0.08`, `SWEEP_MIN=0.04`, `BOUNCE_MIN=0.01`, `CONFIRM_RATIO_MIN=0.22`）
    - long圧縮緩和: `SIDE_METRICS_MIN_MULT: 0.30 -> 0.42`, `MAX_MULT: 0.82 -> 0.95`,
      `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE: 0.24 -> 0.30`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 4 -> 6`
    - `REVERT_*` を緩和（`MIN_TICKS=1`, `RANGE_MIN=0.08`, `SWEEP_MIN=0.04`, `BOUNCE_MIN=0.01`, `CONFIRM_RATIO_MIN=0.22`）
    - long圧縮緩和: `SIDE_METRICS_MIN_MULT: 0.25 -> 0.38`, `MAX_MULT: 0.88 -> 1.00`,
      `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE: 0.28 -> 0.34`
- 意図:
  - `signal不成立` と `rate-limit` 起因の無約定を減らし、long側の実効units回復を優先する。

## 0-13. 2026-02-27 UTC `scalp_ping_5s_b/c` RR再補正 + longロット押上げ（第3ラウンド）
- 目的:
  - `SLが大きい / TPが小さい` 非対称を縮め、long側の実効ロット不足を解消する。
- 仮説（VM, UTC 2026-02-27 13:05 前後）:
  - Round2 で `rate_limited` / `revert_not_found` は沈静化し、現ボトルネックは
    `entry_leading_profile_reject` と long側 `TP/SL<1` に移った。
  - 直近6h（`orders.db` filled）:
    - B long: `avg_units=78.8`, `avg_sl=1.84 pips`, `avg_tp=0.99 pips`, `tp/sl=0.54`
    - B short: `avg_units=130.4`
    - C long: `avg_units=31.5`, `avg_sl=1.31 pips`, `avg_tp=0.96 pips`, `tp/sl=0.74`
    - C short: `avg_units=37.2`
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - `BASE_ENTRY_UNITS: 260 -> 300`, `MAX_UNITS: 900 -> 1000`
    - `TP_BASE/MAX: 0.55/1.9 -> 0.75/2.2`
    - `SL_BASE/MAX: 1.35/2.0 -> 1.20/1.8`
    - `TP_NET_MIN: 0.45 -> 0.65`, `TP_TIME_MULT_MIN: 0.55 -> 0.72`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE: 0.30 -> 0.34`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.70 -> 0.68`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN/MAX: 0.64/0.95 -> 0.70/1.00`
  - `ops/env/scalp_ping_5s_c.env`
    - `BASE_ENTRY_UNITS: 95 -> 120`, `MAX_UNITS: 220 -> 260`
    - `TP_BASE/MAX: 0.45/1.5 -> 0.60/1.8`
    - `SL_BASE/MAX: 1.15/1.9 -> 1.05/1.7`
    - `TP_NET_MIN: 0.40 -> 0.55`, `TP_TIME_MULT_MIN: 0.55 -> 0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE: 0.34 -> 0.38`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.70 -> 0.68`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN/MAX: 0.62/0.90 -> 0.68/0.95`
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル判定とロット算出のみ。
  - V2 共通導線（`quant-order-manager` / `quant-position-manager` / `quant-strategy-control`）の責務分離は不変。
- 検証:
  1. 反映後2h/24hで B/C long の `avg_tp/avg_sl` が上昇し、`tp/sl` が改善すること。
  2. 反映後2h/24hで `avg_units(long)` が増加し、`entry_leading_profile_reject` 比率が低下すること。
  3. 24hで `scalp_ping_5s_b/c long` の `sum(realized_pl)` が改善方向へ向かうこと。

## 0-14. 2026-02-27 UTC `scalp_ping_5s_b/c` 通過不足の再解消（第4ラウンド）
- 目的:
  - long の実効ロット不足を、`rate_limited` / `revert_not_found` / `perf_block` の同時多発を抑えて解消する。
- 仮説（VM, UTC 2026-02-27 13:21-13:26）:
  - Round3 後も `entry-skip summary` の上位が `rate_limited` と `no_signal:revert_not_found`。
  - C は反映後の約定がほぼ出ず、long 側の通過率改善が不足している。
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE: 6 -> 10`
    - `REVERT_*` 緩和（`RANGE_MIN 0.08->0.05`, `SWEEP_MIN 0.04->0.02`, `BOUNCE_MIN 0.01->0.008`, `CONFIRM_RATIO_MIN 0.22->0.18`）
    - long通過緩和: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`
    - longロット下限押上げ: `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT 0.70->0.76`
    - preserve-intent 緩和: `REJECT_UNDER 0.78->0.74`, `MIN_SCALE 0.34->0.40`
    - `MIN_UNITS_RESCUE` 緩和: `prob 0.58->0.54`, `conf 78->75`
    - perf setup の過剰早期ブロック抑制: `HOURLY_MIN_TRADES 6->10`, `SETUP_MIN_TRADES 6->10`, `SETUP_PF_MIN 0.92->0.88`, `SETUP_WIN_MIN 0.48->0.44`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 6 -> 10`
    - `REVERT_*` 緩和（B と同値）
    - long通過緩和: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`
    - longロット下限押上げ: `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT 0.68->0.74`
    - preserve-intent 緩和: `REJECT_UNDER 0.76->0.72`, `MIN_SCALE 0.38->0.44`
    - `MIN_UNITS_RESCUE` 緩和: `prob 0.60->0.56`, `conf 82->78`
    - perf setup 緩和（`SCALP_PING_5S_C_*` と fallback の `SCALP_PING_5S_*` を両方更新）
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 判定・ロット算出のみ。
  - V2 共通導線（`quant-order-manager` / `quant-position-manager` / `quant-strategy-control`）は非変更。
- 検証:
  1. 反映後30分/2hで `entry-skip summary` の `rate_limited` と `revert_not_found` 比率が低下すること。
  2. C の `filled` が再開し、B/C long の `avg_units` が上昇すること。
  3. `perf_block` が急増せず、`tp/sl` 改善方向を維持すること。

## 0-15. 2026-02-27 UTC `scalp_ping_5s_c` spread guard 上限緩和（第5ラウンド）
- 目的:
  - C の `spread_blocked` 連発を解消して、約定通過率を回復する。
- 仮説（VM, UTC 2026-02-27 13:30 前後）:
  - C の skip 主因が `spread_guard` に集中し、`entry-skip summary total=143` 中 `spread_blocked=134` を観測。
  - ガード理由は `spread_med ... >= limit 1.00p` で、実勢 `p95=1.16p` が現行閾値を上回る。
- 対応:
  - `ops/env/scalp_ping_5s_c.env` に C 専用 `spread_guard_*` を追加:
    - `spread_guard_max_pips=1.30`
    - `spread_guard_release_pips=1.05`
    - `spread_guard_hot_trigger_pips=1.50`
    - `spread_guard_hot_cooldown_sec=6`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の spread guard 判定のみ（C 戦略ローカル）。
  - B と V2 共通導線（order_manager/position_manager/strategy_control）は非変更。
- 検証:
  1. 反映後30分で C の `entry-skip summary` に占める `spread_blocked` 比率が低下すること。
  2. C の `filled` 再開と `avg_units(long)` 回復を確認すること。
  3. `hot_spread_now` の発生頻度が抑制されること。

## 0-16. 2026-02-27 UTC `scalp_ping_5s_c` 通過率再補正（第6ラウンド）
- 目的:
  - 第5ラウンド後に残った `rate_limited` / `perf_block` を低減し、C の約定再開を優先する。
- 仮説（VM, UTC 2026-02-27 13:40 前後）:
  - `spread_blocked` は解消したが、C の skip 主因が `rate_limited` と `revert_not_found` に移行。
  - `orders.db` は `perf_block` と `probability_scaled` のみで `filled=0` が継続。
- 対応:
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 6 -> 10`
    - perf guard の過早ブロック抑制:
      - `SCALP_PING_5S_C_PERF_GUARD_HOURLY_MIN_TRADES: 10 -> 16`
      - `SCALP_PING_5S_C_PERF_GUARD_SETUP_MIN_TRADES: 10 -> 16`
      - fallback `SCALP_PING_5S_PERF_GUARD_HOURLY_MIN_TRADES: 10 -> 16`
      - fallback `SCALP_PING_5S_PERF_GUARD_SETUP_MIN_TRADES: 10 -> 16`
    - 通過ゲート緩和:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.82 -> 0.78`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.74 -> 0.70`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 判定のみ。
  - B および V2 共通導線は非変更。
- 検証:
  1. 反映後30分/2hで C の `rate_limited` 比率が低下すること。
  2. `orders.db` で C の `filled` が再開すること。
  3. `perf_block` が優位理由でなくなること。

## 0-17. 2026-02-27 UTC `scalp_ping_5s_c` レート制限主因への追加対応（第7ラウンド）
- 目的:
  - C の `rate_limited` 優位を下げ、実発注まで到達させる。
- 仮説（VM, UTC 2026-02-27 13:43-13:45）:
  - 第6ラウンド後も `entry-skip summary` で `rate_limited` が最大（例: `total=107, rate_limited=65`）。
  - `spread_blocked` は解消済みで、次ボトルネックはレート上限と通過閾値。
- 対応:
  - `ops/env/scalp_ping_5s_c.env`
    - `ENTRY_COOLDOWN_SEC: 1.6 -> 1.2`
    - `MAX_ORDERS_PER_MINUTE: 10 -> 16`
    - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY: 0.60 -> 0.56`
    - `MIN_UNITS_RESCUE_MIN_CONFIDENCE: 82 -> 78`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.78 -> 0.74`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.70 -> 0.66`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 判定・通過率のみ。
  - B および V2 共通導線は非変更。
- 検証:
  1. 反映後30分/2hで C の `rate_limited` 比率が低下すること。
  2. `orders.db` で C の `filled` が再開すること。
  3. `entry_probability_reject` / `entry_leading_profile_reject` が過度に増えないこと。
