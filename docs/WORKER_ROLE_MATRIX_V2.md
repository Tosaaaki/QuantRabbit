# WORKER ROLE MATRIX (V2)

## 目的

本番プロセスでの役割混線をなくすため、責務を固定する。  
「集約責務の重複」「main からの横断起動」「複数戦略を 1 つのワーカーサービスで処理」を全て排除する。

## V1 問題点（混線）

- `main.py` と `quantrabbit.service` が生存していたことで、サービス起動制御・実行制御・一部戦略処理が混在した。
- `order_manager` / `position_manager` が多数の戦略ワーカーと EXIT ワーカーから直接参照され、独立面が不明瞭になった。
- 補助ワーカーの有効化/無効化が一部残存し、起動状態の把握が難しかった。

## V2 役割定義（最終方針）

### 1) データ面（単独）

- `quant-market-data-feed.service`
  - OANDA から tick を受ける
  - `tick_window` 更新
  - M1/M5/H1/H4/D1 集約
  - `factor_cache` 更新
  - `factor_cache` 連携は `on_candle(timeframe, candle)` 契約を維持し、購読側に渡すハンドラは
    timeframe を束縛した `Callable[[Candle], ...]` で実行する（`on_candle(candle)` 形式を禁止）
  - 起動時と定期再起動時のシードは、既存キャッシュ終端（最後の確定足）から「境界を1本進めた時点→now」へレンジ補完する。

### 2) 制御面（単独）

- `quant-strategy-control.service`
  - `global_entry_enabled` / `global_exit_enabled` / `global_lock` の集約
  - strategy slug 単位の `entry_enabled` / `exit_enabled` 配信
  - 各戦略のロジックには介入せず、実行可否のガード（`entry_enabled` / `exit_enabled` / `global_lock`）としてのみ働く

### 3) 戦略実行面（完全1:1）

- 戦略の ENTRY / EXIT はセットで1:1運用
- `quant-scalp-ping-5s-b` + `quant-scalp-ping-5s-b-exit`
- `quant-scalp-macd-rsi-div` + `quant-scalp-macd-rsi-div-exit`
- `quant-scalp-macd-rsi-div-b` + `quant-scalp-macd-rsi-div-b-exit`
- `quant-scalp-tick-imbalance` + `quant-scalp-tick-imbalance-exit`
- `quant-scalp-squeeze-pulse-break` + `quant-scalp-squeeze-pulse-break-exit`
- `quant-scalp-wick-reversal-blend` + `quant-scalp-wick-reversal-blend-exit`
- `quant-scalp-wick-reversal-pro` + `quant-scalp-wick-reversal-pro-exit`
- `quant-m1scalper` + `quant-m1scalper-exit`
- `quant-micro-rangebreak` + `quant-micro-rangebreak-exit`
- `quant-micro-levelreactor` + `quant-micro-levelreactor-exit`
- `quant-micro-vwapbound` + `quant-micro-vwapbound-exit`
- `quant-micro-vwaprevert` + `quant-micro-vwaprevert-exit`
- `quant-micro-momentumburst` + `quant-micro-momentumburst-exit`
- `quant-micro-momentumstack` + `quant-micro-momentumstack-exit`
- `quant-micro-pullbackema` + `quant-micro-pullbackema-exit`
- `quant-micro-trendmomentum` + `quant-micro-trendmomentum-exit`
- `quant-micro-trendretest` + `quant-micro-trendretest-exit`
- `quant-micro-compressionrevert` + `quant-micro-compressionrevert-exit`
- `quant-micro-momentumpulse` + `quant-micro-momentumpulse-exit`
- `quant-session-open` + `quant-session-open-exit`（該当期間のみ）
- 補助戦略の追加は、ENTRY/EXIT を追加してから有効化
- 共通ルール:
- 各戦略ENTRYは `entry_thesis` に `entry_probability` と `entry_units_intent` を必須で付与する。
  - `entry_probability` は戦略ローカルの「どれだけ入るべきか」判断、`entry_units_intent` は戦略ローカルの希望ロットを表す。
  - `AddonLiveBroker` 経路（`session_open` など）でも上記2値を `entry_thesis` に渡し、order manager はそれを前提にガード/リスク判定のみを行う。
  - `order_manager` は strategy 側意図の受け取りとガード/リスク検査のみで、戦略横断の採点・再選別は行わない。
  - 補足: `execution/order_manager.py` 側で `market_order()` / `limit_order()` 呼び出し時に当該2値の欠落補完を行うフェールセーフは実装済み。通常は戦略側での注入を優先し、欠損時のみ補完。
- 各戦略ENTRYでは `entry_thesis["technical_context"]` に技術入力断面（`indicators`/`ticks`/要求TF）を保持する。  
- N波/フィボ/ローソクを含む技術判定は、各戦略ワーカー側のローカルロジックで実施する。  
- 各戦略ENTRYは、戦略タグ別の `forecast_profile`（`timeframe`/`step_bars`）を `entry_thesis` へ持ち、
  ローカル予測（テクニカル由来の方向・期待pips）を算出して `tp_mult` / `size_mult` の補正に使う。  
- TP再計算とロット補正は戦略ワーカー内で完結させ、`order_manager` 側で戦略意図を再採点しない。  
- `technical_context.result` は保存用の監査フィールド。戦略側が独自評価を入れる場合のみ埋める。  
- 補足（技術要件契約）:
  - 技術要件は `execution/strategy_entry.py` の契約辞書で戦略タグ単位に定義し、`entry_thesis` 未指定項目を自動補完する。
  - `entry_thesis["technical_context_tfs"]` / `technical_context_fields` / `technical_context_ticks` / `technical_context_candle_counts` を各戦略別に規定。
  - 各戦略が必要なら `entry_thesis["tech_policy"]` で
    `require_fib` / `require_nwave` / `require_candle` を明示する。
  - `strategy_entry` 側は `technical_context` の保存補完のみで、上記要求を強制的に追加しない。
- 現行マッピング（`_STRATEGY_TECH_CONTEXT_REQUIREMENTS`）:
  - Scalp系
    - `scalp_ping_5s_b`
    - `scalp_m1scalper`
    - `scalp_tick_imbalance`/`scalp_tick_imbalance_rrplus` 系
    - `scalp_tick_wick_reversal`, `scalp_wick_reversal`, `scalp_wick_reversal_pro`, `scalp_wick_reversal_hf`, `scalp_tick_wick_reversal_hf`
    - `scalp_level_reject`, `scalp_level_reject_plus`
    - `scalp_squeeze_pulse_break`
    - `scalp_macd_rsi_div`
    - `ScalpReversalNWave`（`-reversal` suffix を受ける）
    - `TrendReclaimLong`
    - `VolSpikeRider`
  - Micro系
    - `MicroRangeBreak`（`workers/micro_rangebreak` 経由）
    - `MicroRangeBreak`（`workers/micro_rangebreak` 経由）
    - `MicroLevelReactor`（`workers/micro_levelreactor`）
    - `MicroVWAPBound`（`workers/micro_vwapbound`）
    - `MicroVWAPRevert`（`workers/micro_vwaprevert`）
    - `MomentumBurstMicro`（`workers/micro_momentumburst`）
    - `MicroMomentumStack`（`workers/micro_momentumstack`）
    - `MicroPullbackEMA`（`workers/micro_pullbackema`）
    - `TrendMomentumMicro`（`workers/micro_trendmomentum`）
    - `MicroTrendRetest`（`workers/micro_trendretest`）
    - `MicroCompressionRevert`（`workers/micro_compressionrevert`）
    - `MomentumPulse`（`workers/micro_momentumpulse`）
    - `micro_adaptive_revert`（レガシー想定）
    - `MicroPullbackFib`（`-pullback` suffix を受ける）
    - `RangeCompressionBreak`（`-break` suffix を受ける）
  - Macro系
    - `MacroTechFusion`（`-trend` suffix を受ける）
    - `TechFusion`
    - `MacroH1Momentum`
    - `trend_h1`
    - `LondonMomentum`
    - `H1MomentumSwing`
  - `session_open`
  - 技術コンテキスト共通の取得項目
    - `technical_context_ticks`: 原則 `["latest_bid", "latest_ask", "latest_mid", "spread_pips"]`
      - 例外: `tick_imbalance` 系で `tick_rate` 追加
    - `technical_context_candle_counts`: 戦略別に個別上限（例: Scalp系 `M1/H1/M5/H4` 系、Micro系 `M5/M1/H1` 系）
- 仕様上の役割分離は維持:
  - 共通 `strategy_entry.py` は指標入力契約の補完・保存を担い、評価ロジックの主体は各戦略ワーカーへ移す。
  - 最終的な受け入れ/サイズ拡大縮小は `order_manager` 側で再選別しない（意図受け渡し + ガード/リスクのみ）。
- 戦略側は `entry_thesis` により要求仕様（`technical_context_tfs` / `technical_context_fields` /
  `technical_context_ticks` / `technical_context_candle_counts`）を明示できる。  
- `strategy_entry` 側は `strategy_tag` を受け、該当戦略の契約要件を既定補完する。  
  補完優先順位は `entry_thesis` 指定値 > 戦略契約既定値。
- `ENTRY_TECH_DEFAULT_TFS` で初期取得TF順を切替え、必要なら
  `entry_thesis["technical_context_tfs"]` / `entry_thesis["technical_context_fields"]` で戦略側制限を付与可能。  
- order-manager では `technical_context` を上書きしない（保存専有）。

※ `quant-micro-adaptive-revert*` と `quant-impulse-retest-s5*` は V2再整備で VM から停止対象へ移行予定の legacy。  
  現行では `OPS_V2_ALLOWED_LEGACY_SERVICES` に明示登録することで監査を `critical` でなく `warn` 運用にできる（監査ログ上で明示追跡）。

### 4) 予測面（独立）

- `quant-forecast.service` + `workers/forecast/worker.py` + `workers/common/forecast_gate.py`
- 目的: `order_manager` の forecast gate は `forecast_decide` API を経由して `allow/reduce/block` を取得。
- `FORECAST_SERVICE_ENABLED=1` と `FORECAST_SERVICE_URL` が有効な場合、`forecast_gate` 決定をワーカー越しで取得して
  `order_manager` に反映。
- `order_manager` 側ではサービス障害時のみローカル fallback を許容し、判定仕様を維持。
- 予測決定は `expected_pips` に加えて `anchor_price` / `target_price` / `tp_pips_hint` / `sl_pips_cap` / `rr_floor`
  を `forecast_context` として各経路へ伝播し、`order_manager` と `entry_intent_board` の監査へ反映する。
- 戦略ワーカー側では forecast gate とは独立に、短中期（例: `M1x1`, `M1x5`, `M5x2`）のローカル予測を
  エントリー時に都度計算し、`entry_thesis.tech_tp_mult` / `tech_score` / `entry_units_intent` へ反映する。
- ローカル予測の過去再現評価は `scripts/eval_local_forecast.py` を正規手順として使い、
  `baseline` 比で `hit_rate` と `MAE(pips)` を同時監査する。
- 短期 horizon（`1m`/`5m`/`10m`）は `forecast_gate` 内で `M1` 基準に正規化して計算する。
  戦略が `M5x2` のような指定を渡した場合も、短期は `M1` へ換算（例: `M5x2 -> M1x10`）し、
  足更新遅延による stale/欠損の影響を最小化する。
- さらに `factor_cache` の `M1` が stale のときは `logs/oanda/candles_M1_latest.json` を短期予測入力の
  フォールバックとして使い、短期出力の欠落を避ける。
- `forecast` 系は `order_manager` の判定処理から切り離し、`execution` 側の責務分離として
  専用 service を通した決定供給を行う。

### 5) オーダー面（分離済み）

- `execution/order_manager.py` の注文経路は `quant-order-manager.service` 経由。
- 目的: 戦略は「注文意図」を投げ、実API送信は order-manager が担当。
- 受け渡しは `entry_probability` / `entry_units_intent` を前提にし、`order_manager` 側は意図の再選別をせず、ガード・リスク検査のみ実施。
- `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1`（既定）運用では、戦略側が意図した
  `entry_probability` / `entry_units_intent` / SL/TP 設計を、`order_manager` 側で上書きしない方針へ統一。  
  例外として、`ORDER_MANAGER_PRESERVE_INTENT_UNIT_ADJUST_ENABLED=1`（strategy 固有キー
  `ORDER_MANAGER_PRESERVE_INTENT_UNIT_ADJUST_ENABLED_STRATEGY_<TAG>`）を明示した場合のみ、
  `entry_probability` によるサイズ調整・リジェクトを有効化する。
- `entry_probability` / `entry_units_intent` をもとに、`execution/strategy_entry.py` が
  `execution/order_manager.py` の `/order/coordinate_entry_intent` を呼んで黒板協調を行った後に
  注文を `order-manager` へ転送する。
- `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1` 方針は維持し、`order_manager` は
  戦略意図を上書きしない前提で、ガード/リスク判定と必要最小限の縮小・拒否のみに留める。
- `execution/strategy_entry.py` 経由の協調判定は以下を固定ルール化する。  
  - 判定対象は `ORDER_INTENT_COORDINATION_ENABLED=true` かつ `pocket != manual` のみ。  
  - 同一 `strategy_tag` + `instrument` + `window`（既定 2 秒）内の未期限 board を集約し、`own_score=abs(raw_units)*prob` とする。  
  - `opposite_score` が 0 のときは協調受理。  
  - `opposite_score > 0` のときも方向意図は原則維持し、`raw_units` をそのまま通す。`dominance` は監査記録用に保持する。  
  - `abs(final_units) < min_units_for_strategy(strategy_tag, pocket)` は拒否（優先解釈は戦略別設定）。  
  - `reason` は `order_manager` の `entry_intent_board` へ記録し、`strategy_entry` は 0 なら注文を出さない。  

### 6) ポジ管理面（分離済み）

- `execution/position_manager.py` は `quant-position-manager.service` 経由。
- 目的: 保有集計・sync/trades の集約責任を独立し、各戦略が状態管理を持たない。
- `get_open_positions` はホットパスでの `orders.db` 参照を最小化し、`orders.db` は read-only/短timeoutで参照する。
  writer 競合時は fail-fast + 既存キャッシュ返却を優先し、strategy worker 側 timeout を増幅させない。

### 7) 分析・監視面（データ管理）

- `quant-pattern-book`, `quant-dynamic-alloc`, `quant-ops-policy`, `quant-policy-guard`, `quant-range-metrics`, `quant-strategy-feedback` は分析/監視へ固定
- 分析系が戦略判断本体と混ざる構造を禁ずる

## 禁止ルール（V2）

1. 1つの戦略ワーカーが複数戦略ロジックを実体レベルで担当しないこと
2. market_data / control / strategy / order / position の責務混在を許可しないこと
3. `quantrabbit.service` の本番起動を許可しないこと
4. `main.py` を systemd 本番エントリとして扱わないこと
5. order-manager / strategy-control が戦略の意思決定を上書きしないこと
6. entry/exit の戦略ワーカーは `scalp_precision` の共通実行器を経由しないこと（完全に戦略ロゴ独立実行）

## 現在の状態（2026-02-17 時点）

- runtime env は `ops/env/quant-v2-runtime.env` を基準参照とし、共通設定を持つ。  
  `order_manager` / `position_manager` は各ワーカー固定実行のため、service-mode の制御は専用 env でオーバーライド:
  - `ops/env/quant-order-manager.env`
  - `ops/env/quant-position-manager.env`
- strategy 固有の追加設定は `ops/env/scalp_ping_5s_b.env` などの既存上書き env に加え、各ENTRY/EXIT戦略の基本設定を
  `ops/env/quant-<service>.env` へ集約する。

- この図は V2 運用で構成が変わるたびに更新する（組織図更新の必須運用）。  
  `docs/WORKER_REFACTOR_LOG.md` と同一コミットで差分が並走すること。

- 実装済み（運用へ反映）
  - `quant-market-data-feed`
  - `quant-strategy-control`
  - `quant-scalp-*`/`quant-micro-*` の ENTRY+EXIT 1:1化
  - 補助的冗長ワーカー群の縮小
- 実装済み（2026-02-14 時点）
- `quant-order-manager.service` / `quant-position-manager.service` 追加
- `execution/order_manager.py`, `execution/position_manager.py` の service-first 経路化
- API 契約（/order/*, /position/*）を基準化
- 注記: 直近の運用レビューでは、データ記録系 DB と分析系成果物の更新は確認済み（VM側状態監査前提）。
- 予測判定専用 `quant-forecast.service` を追加し、`ORDER_MANAGER_FORECAST_GATE_ENABLED=1` で
  `order_manager` からサービス経由で `forecast_decide` を取得する導線を実装。
- 運用整備（2026-02-16 追加）
  - 戦略ENTRYの出力に `entry_probability` / `entry_units_intent` を必須化し、V2本体戦略から `order_manager` への意図受け渡しを統一。
  - `WORKER_REFACTOR_LOG.md` の同時追記を行い、実装・図面の変更差分を同一コミットへ反映。
- 5秒スキャBでは `SCALP_PING_5S_B_MIN_UNITS` を `50` まで下げる運用を許容するため、`workers/scalp_ping_5s/config.py` の
  ローカル最小ロット下限を固定100から可変化（`max(1, ...)`）して、`ORDER_MIN_UNITS_SCALP_FAST`（50）との整合を担保。
- 運用整備（2026-02-16）
  - VM側で `quantrabbit.service` を除去し、レガシー戦略・補助ユニット（`quant-impulse-retest-s5*`, `quant-micro-adaptive-revert*`, `quant-trend-reclaim-long*`, `quant-margin-relief-exit*`, `quant-hard-stop-backfill*`, `quant-realtime-metrics*`, precision 系）を停止・無効化。
  - `systemctl list-unit-files --state=enabled --all` で V2実行群（`market-data-feed`, `strategy-control`, 各ENTRY/EXIT, `order-manager`, `position-manager`）のみが実行系として起動対象であることを確認。
- 運用整備（2026-02-17）
  - `quant-order-manager.service` / `quant-position-manager.service` へ専用 env を追加し、共通 runtime env でサービス自体を
    ON にしない形へ分離。  
  - worker起動時に service-mode の誤自己参照を抑止するガードを追加。
- `micro_multistrat` は共通Runnerとしての運用を打ち切り、レンジ時の順張り/押し目判定運用は各独立 micro ワーカー側へ移行したため、同種の範囲制御は
  各専用ワーカーの設定で管理している。
- 運用整備（2026-02-24）
  - `analysis/strategy_feedback_worker.py` を追加し、`quant-strategy-feedback.service` / `quant-strategy-feedback.timer` で
    `logs/trades.db` と strategy list を再解析して `logs/strategy_feedback.json` を更新する分析係ワーカーを導入。
  - 戦略の追加・停止（systemd/service状態）に追従して指標を更新し、停止/追加時の事故条件を避ける `keep_inactive` 制約を明記。
  - エントリーワーカー稼働を優先判定に変更し、EXITワーカーのみ残存するケースでは `strategy_feedback` の更新適用を抑止。
  - `ops/env/quant-strategy-feedback.env` に `STRATEGY_FEEDBACK_*` を追加し、lookback / min_trades / systemd_path を運用制御可能化。

## 監査用更新プロトコル（毎回）

- 変更を加えるたびに実行:
1. `docs/WORKER_REFACTOR_LOG.md` に変更内容を追記
2. `docs/WORKER_ROLE_MATRIX_V2.md` の「現在の状態」を同一コミットで更新
3. `docs/INDEX.md` が必要なら参照を同期
4. `main` 統合 → `git commit` → `git push` → VM 反映

### 自動監査（VM監査ユニット）

- 追加サービス（導線監査の自動化）: `quant-v2-audit.service` + `quant-v2-audit.timer`
- 実施内容:
  - `quant-market-data-feed` / `quant-strategy-control` / `quant-order-manager` / `quant-position-manager` の実行状態
  - strategy 主要ENTRY/EXITユニットの稼働監査
  - V2 runtime env と service/env 分離状態の監査
  - `position/open_positions` 周りの 405 / Method Not Allowed 監査
    - 405 判定は `POST`/`GET` の該当リクエスト行または `Method Not Allowed` 文字列のみを対象化
    - journal タイムスタンプ中の `:405` を誤判定しない
  - `quant-v2-runtime.env` の主要制御キー値差分監査
  - `quantrabbit.service` 等 legacy active 判定
- 出力: `logs/ops_v2_audit_latest.json`
- デフォルト実行: 10分間隔（`OnCalendar=*:00/10`）

### 2026-02-17（追記）install/trade 監査連携

- `scripts/install_trading_services.sh --all` の対象外にして V2外のレガシー戦略を勝手に再有効化しない運用を追加。
  - 除外対象: `quant-impulse-retest-s5*`, `quant-micro-adaptive-revert*`
- V2運用中の監査対象外扱いを維持するため、`--all` 実行では上記を明示的にスキップし、
  レガシーは必要時の `--units` 指定時にのみ再導入可能とする。

### 2026-02-14（追記）install_trading_services の実行安定化

- `scripts/install_trading_services.sh --all` 実行中に、`quant-strategy-optimizer.service` の
  oneshot 長時間起動が原因で `systemctl start` がブロックし、後続ユニット有効化が停滞する問題を対策。
- `NO_BLOCK_START_UNITS` を導入し、`quant-strategy-optimizer.service` は
  `systemctl start --no-block` で起動要求する運用へ変更。
- `--all` 実行の完了性を担保し、V2監査/運用サービスの再起動ループを維持する状態に更新。

## V2 反映図（最上位・並行）

```mermaid
flowchart LR
  OANDA["OANDA API"] --> MD["quant-market-data-feed<br>/workers/market_data_feed/worker.py"]
  MD --> TW["market_data.tick_window.record"]
  MD --> FC["indicators.factor_cache.on_candle / on_candle_live"]

  subgraph "戦略サイド（並行）"
    SCW[ "strategy ENTRY workers<br>/workers/*/worker.py" ]
    SWX[ "strategy EXIT workers<br>/workers/*/exit_worker.py" ]
  end

  SC["quant-strategy-control<br>/workers/strategy_control/worker.py<br>/workers/common/strategy_control.py"] -->|entry/exit flags| SCW
  SC -->|entry/exit flags| SWX
  TW --> SCW
  FC --> SCW
  TW --> SWX
  FC --> SWX

  SCW --> OM["order intents<br>entry_probability / entry_units_intent"]
  SWX --> OM
  OM --> OWM["quant-order-manager<br>/workers/order_manager/worker.py"]
  OWM --> OEX["execution/order_manager.py"]
  OEX -->|forecast_decide| FSG["quant-forecast<br>/workers/forecast/worker.py"]
  OEX --> OANDA_ORDER["OANDA Order API"]

  PM["quant-position-manager<br>/workers/position_manager/worker.py"] -->|sync/positions| SWX
  PM -->|position snapshot| SCW
  OEX --> PM
  PM --> DB["logs/trades.db, logs/orders.db"]

  ANAL["analysis/services<br>quant-pattern-book, quant-range-metrics, quant-strategy-feedback, etc."] -->|feedback params / labels| SCW
  ANAL -->|feedback params / labels| SWX

  UI["QuantRabbit UI<br>apps/autotune_ui.py"] --> SC
  UI -->|strategy on/off / global lock| SC
```

備考
- 既存ワーカーは「並行実行」に前提を置く。ENTRY/EXIT は原則 1:1 で個別 `systemd` ユニット化。
- `strategy control` は「可否フラグ配信」だけを担当。実行判断と注文/保有更新は各ワーカーと専用サービスで実施。


## 監査用の記載先

- AGENTS は規約面、`docs/WORKER_REFACTOR_LOG.md` は実装履歴、`docs/INDEX.md` には本ドキュメント参照を追加済み
