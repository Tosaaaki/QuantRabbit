# Risk & Execution

## 1. エントリー/EXIT/リスク制御

### Strategy フロー
- Focus/Local decision → `ranked_strategies` 順に Strategy Plugin を呼び、`StrategyDecision` または None を返す。
- `None` はノートレード。

### Confidence スケーリング
- `confidence`(0–100) を pocket 割当 lot に掛け、最小 0.2〜最大 1.0 の段階的エントリー。
- `STAGE_RATIOS` に従い `_stage_conditions_met` を通過したステージのみ追撃。

### Brainゲート（任意）
- `execution/order_manager.py` で LLM 判断を実行し、**ALLOW/REDUCE/BLOCK** を返す。
- `REDUCE` は units 縮小のみ（増加は禁止）。

### Exit
- 各戦略の `exit_worker` が最低保有時間とテクニカル/レンジ判定を踏まえ、PnL>0 決済が原則。
- 例外は強制 DD / ヘルス / マージン使用率 / 余力 / 未実現DDの総合判定のみ。
- 共通 `execution/exit_manager.py` は常に空を返す互換スタブ。
- `execution/stage_tracker` がクールダウンと方向別ブロックを管理。

### scalp_ping_5s_b 運用補足（取り残し抑制）
- `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE` を
  `quant-order-manager` の環境で運用し、低確率シグナルを `order_manager` 側で reject する。
- `scalp_ping_5s_b*` は entry 時の SL 欠損を防ぐため、
  `order_manager` 側で `stopLossOnFill` を戦略別に許可する。
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B=1`
  - `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_B=0`
  - Bワーカー env は `SCALP_PING_5S_B_USE_SL=1` / `SCALP_PING_5S_B_DISABLE_ENTRY_HARD_STOP=0` を維持する。
  - `workers/scalp_ping_5s_b.worker` は fail-safe として上記2値を起動時に強制補正する
    （`SCALP_PING_5S_B_ALLOW_UNPROTECTED_ENTRY=1` のときのみ無効化）。
- `scalp_ping_5s_b*` は方向転換遅延による逆張り連発を抑えるため、
  `direction_bias` と `horizon_bias` が同方向で強く一致した場合に
  エントリーをブロックせず `fast_direction_flip` で side を即時リルートする。
  - 既定は B で有効（`SCALP_PING_5S_B_FAST_DIRECTION_FLIP_ENABLED=1`）。
  - `horizon=neutral` でも `direction_bias` が十分強ければ反転を許可する
    （`SCALP_PING_5S_B_FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN`）。
  - 頻度維持のため、reject ではなく side リライト＋confidence 加算のみ行う。
- `scalp_ping_5s_b*` は「同方向の `STOP_LOSS_ORDER` 連発」を方向ミスマッチとして扱い、
  直近クローズ履歴に同方向SLが規定回数（既定2連）続いた場合、
  次エントリーの side を `sl_streak_direction_flip` で反転する。
  - 既定は B で有効（`SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED=1`）。
  - 発注拒否ではなく side リライトで頻度を維持し、`entry_thesis` に
    `sl_streak_*`（side/count/age/applied/reason）を記録する。
  - 過去トレード参照は `strategy_tag + pocket` のクローズ履歴に限定し、
    `SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC` を超える古い連敗は反転対象外にする。
  - 2026-02-19 追記:
    - `SL回数` と `MARKET_ORDER_TRADE_CLOSE でのプラス回数` を追加条件にした。
      - `SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS`（既定2）
      - `SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS`（既定1）
      - 集計窓は `SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES`（既定24）
    - さらに `direction_bias/horizon` のテクニカル一致がない場合は反転しない
      （`SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM=1`）。
    - `fast_direction_flip` が同ループで発火した場合は `fast_flip` を優先し、
      `sl_streak` 側の逆上書きを禁止する
      （`SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP=0` 既定）。
    - 連敗数が `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK` 以上のときは、
      `MIN_TARGET_MARKET_PLUS` 条件のみをバイパス可能にして反転遅延を抑える
      （`direction_bias/horizon` の tech 一致要件は維持）。
- `scalp_ping_5s_b*` の extrema は 2026-02-19 以降、ショート側のみ非対称チューニング。
  - `short_bottom_soft` は専用倍率
    `EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT`（B既定 0.42）で縮小。
  - `mtf_balanced` かつ short非優勢では
    `EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT`（B既定 0.30）へ追加縮小し、
    `entry_thesis.extrema_gate_reason=short_bottom_soft_balanced` で監査する。
  - extrema reversal は B既定で `long -> short` を無効化
    （`EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`）し、
    `long_top_soft_reverse` クラスタの誤反転を抑制する。
- `RANGEFADER_EXIT_NEW_POLICY_START_TS` を `quant-scalp-ping-5s-b-exit` の環境で固定し、
  service再起動時も既存建玉が legacy 扱いで loss-cut 系ルールから外れないようにする。
  - `workers/scalp_ping_5s_b.exit_worker` は同キーを float として読むため、
    値は ISO ではなく Unix秒（例: `1771286400`）で指定する。
- `config/strategy_exit_protections.yaml` では
  `scalp_ping_5s_b_live: *SCALP_PING_5S_EXIT_PROFILE` を維持し、
  `ALLOWED_TAGS=scalp_ping_5s_b_live` の建玉が default EXITプロファイルへ落ちないようにする。

### scalp_ping_5s_flow 運用補足（stale closeout 抑止）
- `quant-order-manager` 環境で
  `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を運用する。
- service timeout 時の local fallback でも同条件を維持するため、
  `quant-scalp-ping-5s-flow` worker 環境
  （`ops/env/quant-scalp-ping-5s-flow.env` と `ops/env/scalp_ping_5s_flow.env`）
  にも同じ `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を設定する。
- `margin_closeout_n>0` の緊急ブロック条件自体は維持し、
  「直近1日」の closeout のみで block 判定する。
- これにより、古い closeout（2日以上前）で `OPEN_REJECT perf_block:margin_closeout_n=*`
  が継続し、flow 戦略の新規が長時間停止する経路を避ける。
- `strategy_entry` / `order_manager` の `env_prefix` 推論は
  `scalp_ping_5s_flow_* -> SCALP_PING_5S_FLOW` を優先し、
  `SCALP_PING_5S` への丸め込みをしない。
  - flowタグが `SCALP_PING_5S` に正規化されると、
    flow専用 `PERF_GUARD_*` が効かず、意図せず stale closeout block が継続するため。

### orders.db ログ運用補足（lock耐性）
- `execution/order_manager.py` の orders logger は lock 検知時に
  `ORDER_DB_LOG_RETRY_*` の短時間 backoff 再試行を行う。
- 既定運用値:
  - `ORDER_DB_BUSY_TIMEOUT_MS=1500`
  - `ORDER_DB_LOG_RETRY_ATTEMPTS=3`
  - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.03`
  - `ORDER_DB_LOG_RETRY_BACKOFF=2.0`
  - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.20`
- `quant-order-manager.service` は
  `ops/env/quant-v2-runtime.env` に加えて
  `ops/env/quant-order-manager.env` も読むため、
  両ファイルで `ORDER_DB_*` を同値に揃える。
  `quant-order-manager.env` 側の古い `ORDER_DB_BUSY_TIMEOUT_MS=250` が残ると、
  runtime 設定を後段で上書きして lock 警告が再発する。
- 目的は「発注判断を変えずに」orders 監査ログ欠損を減らすこと。
  発注可否ロジック（perf/risk/policy/coordination）には影響しない。

### scalp_macd_rsi_div 運用補足（legacy tag 互換）
- `quant-scalp-macd-rsi-div-exit` は `SCALP_PRECISION_EXIT_TAGS=scalp_macd_rsi_div_live` で運用する。
- `workers/scalp_macd_rsi_div.exit_worker` では
  `scalpmacdrsi*`（例: `scalpmacdrsic7c3e9c1`）を `scalp_macd_rsi_div_live` に正規化し、
  旧 client_id 由来タグでも EXIT 監視から外れないようにする。
- `config/strategy_exit_protections.yaml` に
  `scalp_macd_rsi_div_live` の `exit_profile` を必ず定義し、
  default（`loss_cut_enabled=false`）フォールバックで負け玉が残留しないようにする。

### scalp_macd_rsi_div_b 運用補足（精度優先プロファイル）
- `ops/env/quant-scalp-macd-rsi-div-b.env` は 2026-02-19 以降、
  looser mode ではなく precision-biased を既定とする。
- 主要ゲートは次を維持する:
  - `SCALP_MACD_RSI_DIV_B_REQUIRE_RANGE_ACTIVE=1`
  - `SCALP_MACD_RSI_DIV_B_RANGE_MIN_SCORE=0.35`
  - `SCALP_MACD_RSI_DIV_B_MAX_ADX=30`
  - `SCALP_MACD_RSI_DIV_B_MIN_DIV_SCORE=0.08`
  - `SCALP_MACD_RSI_DIV_B_MIN_DIV_STRENGTH=0.12`
  - `SCALP_MACD_RSI_DIV_B_MAX_DIV_AGE_BARS=24`
  - `SCALP_MACD_RSI_DIV_B_TECH_FAILOPEN=0`
- `workers/scalp_macd_rsi_div.worker` は `MIN_ENTRY_CONF` を実効評価し、
  `confidence < MIN_ENTRY_CONF` のエントリーを reject する
  （`gate_block confidence` ログで監査）。

### dynamic_alloc 運用補足（サイズ過多抑止）
- `scripts/dynamic_alloc_worker.py` は PF ガードを持ち、
  `pf < 1.0` の戦略を `lot_multiplier <= 0.95`、
  `pf < 0.7` を `lot_multiplier <= 0.90` に制限する。
- `trades < min_trades` の戦略は `lot_multiplier <= 1.00` とし、
  サンプル不足での過剰増量を防ぐ。
- `quant-dynamic-alloc.service` の `--target-use` は `0.88` を基準とし、
  `account.margin_usage_ratio` が高止まりする局面での margin block 連発を抑える。

### Release gate
- PF>1.1、勝率>52%、最大 DD<5% を 2 週間連続で満たすと実弾へ昇格。

## 2. リスク計算とロット
- `pocket_equity = account_equity * pocket_ratio`
- `POCKET_MAX_RATIOS` は macro/micro/scalp/scalp_fast すべて 0.85 を起点に ATR・PF・free_margin で動的スケールし、最終値を 0.92〜1.0 にクランプ（scalp_fast は scalp から 0.35 割合で分岐）
- `risk_pct = 0.02`、`risk_amount = pocket_equity * risk_pct`
- 1 lot の pip 価値は 1000 JPY
- `lot = min(MAX_LOT, round(risk_amount / (sl_pips * 1000), 3))` → `units = int(round(lot * 100000))`
- `abs(units) < 1000` は発注しない
- 最小ロット下限: macro 0.1, micro 0.0, scalp 0.05（env で上書き可）
- `clamp_sl_tp(price, sl, tp, side)` で 0.001 丸め、SL/TP 逆転時は 0.1 バッファ

## 3. OANDA API マッピング

| Strategy action | REST 注文 | units 符号 | SL/TP | 備考 |
|-----------------|-----------|------------|-------|------|
| `OPEN_LONG` | `MARKET` | `+abs(units)` | `stopLossOnFill`, `takeProfitOnFill` | `timeInForce=FOK`, `positionFill=DEFAULT` |
| `OPEN_SHORT` | `MARKET` | `-abs(units)` | 同上 | ask/bid 逆転チェック後送信 |
| `CLOSE` | `MARKET` | 既存ポジの反対売買 | 指定なし | `OrderCancelReplace` で逆指値削除 |
| `HOLD` | 送信なし | 0 | なし | Strategy ループ継続 |

## 4. 発注冪等性とリトライ
- `client_order_id` は 90 日ユニーク。
- OANDA 429/5xx/timeout は 0.5s→1.5s→3.5s（+100ms ジッター）で最大 3 回リトライ。
- 同一 ID を再利用。
- WebSocket 停止検知時は `halt_reason="stream_disconnected"` を残して停止。

## 5. 安全装置と状態遷移

### 安全装置
- Pocket DD: micro 5% / macro 15% / scalp 3% / scalp_fast 2% で該当 pocket 新規停止。
- Global DD 20% でプロセス終了。
- Event モード（指標 ±30min）は micro 新規禁止。
- Timeout: OANDA REST 5s 再試行。
- Healthbeat は `main.py` から 5min ping。

### 状態遷移

| 状態 | 遷移条件 | 動作 |
|------|----------|------|
| `NORMAL` | 初期 | 全 pocket 取引許可 |
| `EVENT_LOCK` | 経済指標 ±30min | `micro` 新規停止、建玉縮小ロジック発動 |
| `MICRO_STOP` | `micro` pocket DD ≥5% | `micro` 決済のみ、`macro` 継続 |
| `GLOBAL_STOP` | Global DD ≥20% または `healthbeat` 欠損>10min | 全取引停止、プロセス終了 |
| `RECOVERY` | DD が閾値の 80% 未満、24h 経過 | 新規建玉再開前に `main.py` ドライラン |
