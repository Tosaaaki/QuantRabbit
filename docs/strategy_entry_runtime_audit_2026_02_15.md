# 戦略エントリー系サービス運用監査（V2実用8戦略）

- 実施日時: 2026-02-15 20:xx (JST)
- 対象サービス: V2で起動運用中の実用8戦略の `entry + exit` 16ユニット

## 1. 起動状態監査

- すべて `active` を確認
  - `quant-scalp-ping-5s.service`
  - `quant-scalp-ping-5s-exit.service`
  - `quant-micro-multi.service`
  - `quant-micro-multi-exit.service`
  - `quant-scalp-macd-rsi-div.service`
  - `quant-scalp-macd-rsi-div-exit.service`
  - `quant-scalp-tick-imbalance.service`
  - `quant-scalp-tick-imbalance-exit.service`
  - `quant-scalp-squeeze-pulse-break.service`
  - `quant-scalp-squeeze-pulse-break-exit.service`
  - `quant-scalp-wick-reversal-blend.service`
  - `quant-scalp-wick-reversal-blend-exit.service`
  - `quant-scalp-wick-reversal-pro.service`
  - `quant-scalp-wick-reversal-pro-exit.service`
  - `quant-m1scalper.service`
  - `quant-m1scalper-exit.service`

## 2. systemd定義監査（抜粋）

- 以下が各サービスに設定されていることを確認
  - `EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env`
  - 各サービス固有の `ops/env/quant-*.env`
  - 対応するワーカー `ExecStart` パス
- 例:
  - `quant-scalp-tick-imbalance.service` -> `python -m workers.scalp_tick_imbalance.worker`
  - `quant-scalp-tick-imbalance-exit.service` -> `python -m workers.scalp_tick_imbalance.exit_worker`

## 3. ランタイムエラーチェック

- `journalctl -n 120` で `error|failed|traceback|exception` を検索。
- 本監査時点で16ユニットとも当該キーワードは検出されず（重大エラー通知なし）。

## 4. 運用状態

- `scalp_precision` 系のサブ戦略（tick imbalance, squeeze/pulse break, wick reversal系）は、
  `SCALP_PRECISION_*` 設定を各サービスenvで確認し、想定モードで `entry/exit` が起動していることを確認。
- 主要実用8戦略の起動・exit経路は想定どおり整合。
