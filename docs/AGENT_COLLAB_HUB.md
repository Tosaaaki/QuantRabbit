# AGENT_COLLAB_HUB

## 目的
本日付のタスク開始前に、実行方針と市況前提を固定し、ローカル主導の運用規律を守るための最小チェックを集約する。

## 運用手順（本体）

### 1) 着手前必須確認
- AGENTS へ準拠するため、まず以下を実行:
  ```bash
  sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md
  sed -n '/^## 運用原則/,/^## /p' docs/OPS_LOCAL_RUNBOOK.md
  ```
- AGENTS 本体の本日運用方針（`GCP/VM廃止前提`）を再確認する。

### 2) 市況確認（USD/JPY）
- 例外要否の判断:
  - 価格レンジ
  - spread
  - ATR/レンジ推移
  - 直近の約定・拒否実績
  - OANDA API 応答品質
- 市況悪化時は作業保留し、`docs/TRADE_FINDINGS.md` と運用ログへ理由を残す。

### 3) ローカル導線起動・監視
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
- `scripts/collect_local_health.sh`
- `scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200`
- 必要なら `screen` や `runtime_ui` で実行状態を確認

### 4) ローカルPDCA
- 変更後は `scripts/local_v2_stack.sh` で主要サービスの稼働を確認。
- 失敗ケース・検証結果は `docs/TRADE_FINDINGS.md` に1箇所集約。

## 旧VM/GCP資料（実行対象外）
- `docs/OPS_GCP_RUNBOOK.md` / `docs/VM_OPERATIONS.md` / `docs/VM_BOOTSTRAP.md` は履歴アーカイブ。
- 現行運用では `scripts/vm.sh` / `gcloud compute *` を実行しない。

## ログ/台帳
- 運用監査ログの書込み先:
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/INDEX.md` の参照同期
