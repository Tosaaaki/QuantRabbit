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
  - 起動時と定期再起動時のシードは、既存キャッシュ終端（最後の確定足）から「境界を1本進めた時点→now」へレンジ補完する。

### 2) 制御面（単独）

- `quant-strategy-control.service`
  - `global_entry_enabled` / `global_exit_enabled` / `global_lock` の集約
  - strategy slug 単位の `entry_enabled` / `exit_enabled` 配信
  - 各戦略のロジックには介入せず、実行可否のガード（`entry_enabled` / `exit_enabled` / `global_lock`）としてのみ働く

### 3) 戦略実行面（完全1:1）

- 戦略の ENTRY / EXIT はセットで1:1運用
- `quant-scalp-ping-5s` + `quant-scalp-ping-5s-exit`
- `quant-scalp-macd-rsi-div` + `quant-scalp-macd-rsi-div-exit`
- `quant-scalp-tick-imbalance` + `quant-scalp-tick-imbalance-exit`
- `quant-scalp-squeeze-pulse-break` + `quant-scalp-squeeze-pulse-break-exit`
- `quant-scalp-wick-reversal-blend` + `quant-scalp-wick-reversal-blend-exit`
- `quant-scalp-wick-reversal-pro` + `quant-scalp-wick-reversal-pro-exit`
- `quant-m1scalper` + `quant-m1scalper-exit`
- `quant-micro-multi` + `quant-micro-multi-exit`
- `quant-session-open` + `quant-session-open-exit`（該当期間のみ）
- 補助戦略の追加は、ENTRY/EXIT を追加してから有効化
- 共通ルール:
- 各戦略ENTRYは `entry_thesis` に `entry_probability` と `entry_units_intent` を必須で付与する。
  - `entry_probability` は戦略ローカルの「どれだけ入るべきか」判断、`entry_units_intent` は戦略ローカルの希望ロットを表す。
  - `AddonLiveBroker` 経路（`session_open` など）でも上記2値を `entry_thesis` に渡し、order manager はそれを前提にガード/リスク判定のみを行う。
  - `order_manager` は strategy 側意図の受け取りとガード/リスク検査のみで、戦略横断の採点・再選別は行わない。

※ `quant-micro-adaptive-revert*` と `quant-impulse-retest-s5*` は V2再整備で VM から停止対象へ移行済み（legacy）。

### 4) オーダー面（分離済み）

- `execution/order_manager.py` の注文経路は `quant-order-manager.service` 経由。
- 目的: 戦略は「注文意図」だけを投げ、実API送信は order-manager が担当。
- 受け渡しは `entry_probability` / `entry_units_intent` を前提にし、`order_manager` 側は意図を縮小・拒否（ガード）するのみ。

### 5) ポジ管理面（分離済み）

- `execution/position_manager.py` は `quant-position-manager.service` 経由。
- 目的: 保有集計・sync/trades の集約責任を独立し、各戦略が状態管理を持たない。

### 6) 分析・監視面（データ管理）

- `quant-pattern-book`, `quant-dynamic-alloc`, `quant-ops-policy`, `quant-policy-guard`, `quant-range-metrics` は分析/監視へ固定
- 分析系が戦略判断本体と混ざる構造を禁ずる

## 禁止ルール（V2）

1. 1つの戦略ワーカーが複数戦略ロジックを実体レベルで担当しないこと
2. market_data / control / strategy / order / position の責務混在を許可しないこと
3. `quantrabbit.service` の本番起動を許可しないこと
4. `main.py` を systemd 本番エントリとして扱わないこと
5. order-manager / strategy-control が戦略の意思決定を上書きしないこと

## 現在の状態（2026-02-17 時点）

- runtime env は `ops/env/quant-v2-runtime.env` を基準参照とし、共通設定を持つ。  
  `order_manager` / `position_manager` は各ワーカー固定実行のため、service-mode の制御は専用 env でオーバーライド:
  - `ops/env/quant-order-manager.env`
  - `ops/env/quant-position-manager.env`
- strategy 固有の追加設定は `ops/env/scalp_ping_5s.env` などの既存上書き env に加え、各ENTRY/EXIT戦略の基本設定を
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
- 運用整備（2026-02-16 追加）
  - 戦略ENTRYの出力に `entry_probability` / `entry_units_intent` を必須化し、V2本体戦略から `order_manager` への意図受け渡しを統一。
  - `WORKER_REFACTOR_LOG.md` の同時追記を行い、実装・図面の変更差分を同一コミットへ反映。
- 運用整備（2026-02-16）
  - VM側で `quantrabbit.service` を除去し、レガシー戦略・補助ユニット（`quant-impulse-retest-s5*`, `quant-micro-adaptive-revert*`, `quant-trend-reclaim-long*`, `quant-margin-relief-exit*`, `quant-hard-stop-backfill*`, `quant-realtime-metrics*`, precision 系）を停止・無効化。
  - `systemctl list-unit-files --state=enabled --all` で V2実行群（`market-data-feed`, `strategy-control`, 各ENTRY/EXIT, `order-manager`, `position-manager`）のみが実行系として起動対象であることを確認。
- 運用整備（2026-02-17）
  - `quant-order-manager.service` / `quant-position-manager.service` へ専用 env を追加し、共通 runtime env でサービス自体を
    ON にしない形へ分離。  
  - worker起動時に service-mode の誤自己参照を抑止するガードを追加。

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

## V2 反映図（最上位・並行）

```mermaid
flowchart LR
  OANDA["OANDA API"] --> MD["quant-market-data-feed<br>/workers/market_data_feed/worker.py"]
  MD --> TW["market_data.tick_window.record"]
  MD --> FC["indicators.factor_cache.on_candle_live"]

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
  OEX --> OANDA_ORDER["OANDA Order API"]

  PM["quant-position-manager<br>/workers/position_manager/worker.py"] -->|sync/positions| SWX
  PM -->|position snapshot| SCW
  OEX --> PM
  PM --> DB["logs/trades.db, logs/orders.db"]

  ANAL["analysis/services<br>quant-pattern-book, quant-range-metrics, etc."] -->|feedback params / labels| SCW
  ANAL -->|feedback params / labels| SWX

  UI["QuantRabbit UI<br>apps/autotune_ui.py"] --> SC
  UI -->|strategy on/off / global lock| SC
```

備考
- 既存ワーカーは「並行実行」に前提を置く。ENTRY/EXIT は原則 1:1 で個別 `systemd` ユニット化。
- `strategy control` は「可否フラグ配信」だけを担当。実行判断と注文/保有更新は各ワーカーと専用サービスで実施。


## 監査用の記載先

- AGENTS は規約面、`docs/WORKER_REFACTOR_LOG.md` は実装履歴、`docs/INDEX.md` には本ドキュメント参照を追加済み
