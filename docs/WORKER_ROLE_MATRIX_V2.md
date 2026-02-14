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
- `quant-micro-adaptive-revert` + `quant-micro-adaptive-revert-exit`
- `quant-impulse-retest-s5` + `quant-impulse-retest-s5-exit`
- `quant-session-open` + `quant-session-open-exit`（該当期間のみ）
- 補助戦略の追加は、ENTRY/EXIT を追加してから有効化

### 4) オーダー面（分離済み）

- `execution/order_manager.py` の注文経路は `quant-order-manager.service` 経由。
- 目的: 戦略は「注文意図」だけを投げ、実API送信は order-manager が担当。

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

## 現在の状態（2026-02-14 時点）

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

## 監査用更新プロトコル（毎回）

- 変更を加えるたびに実行:
  1. `docs/WORKER_REFACTOR_LOG.md` に変更内容を追記
  2. `docs/WORKER_ROLE_MATRIX_V2.md` の「現在の状態」を同一コミットで更新
  3. `docs/INDEX.md` が必要なら参照を同期
  4. `main` 統合 → `git commit` → `git push` → VM 反映

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

  SCW --> OM["order intents"]
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
