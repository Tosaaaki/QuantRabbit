# AGENT_COLLAB_HUB

## 目的
本日付のタスク開始前に、実行方針と市況前提を固定し、ローカル主導の運用規律を守るための最小チェックを集約する。

## 運用手順（本体）

### 1) 着手前必須確認
- AGENTS へ準拠するため、まず以下を実行:
  ```bash
  sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md
  sed -n '/^## 0\\. 運用原則/,/^## /p' docs/OPS_LOCAL_RUNBOOK.md
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
- `scripts/collect_local_health.sh`（標準ヘルスチェック導線）
  - 互換ラッパー: 内部で `scripts/run_health_snapshot.sh` を実行
  - 既定: `HEALTH_UPLOAD_DISABLE=1`（ローカル保存のみ）
  - 出力: `logs/health_snapshot.json`
- `scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200`
- 必要なら `screen` や `runtime_ui` で実行状態を確認

### 4) ローカルPDCA
- 変更後は `scripts/local_v2_stack.sh` で主要サービスの稼働を確認。
- 収益悪化の RCA は `long/short` の片側説明で止めず、`pattern_tag / RSI / ADX / MA gap / trend_snapshot / divergence / 連続バー偏り` を軸に cluster して、両方向に効く strategy-local quality guard へ落とす。
- 失敗ケース・検証結果は `docs/TRADE_FINDINGS.md` に1箇所集約。
- `docs/TRADE_FINDINGS.md` は変更日記として使い、各変更で `Why/Hypothesis / Expected Good / Expected Bad / Observed/Fact / Verdict / Next Action` を最低限残す。
- 収益/リスク/ENTRY/EXIT 改善の前には必ず `python3 scripts/trade_findings_review.py --query "<strategy_tag or hypothesis_key or close_reason>"` を実行し、同系改善の `Verdict / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless` を確認する。
- 市況が通常帯なのに
  `fills_15m=0`
  または
  `fills_30m<=1`
  になった場合は、
  「低稼働」とみなして 15 分以内に fast triage を行う。
  先に
  `logs/orders.db`
  と
  `logs/local_v2_stack/*.log`
  から dominant block family
  （例:
  `entry_probability_reject`,
  `lookahead_block`,
  `cluster cooldown`,
  `strategy_cooldown:loss_streak`,
  `no_signal:revert_not_found`
  ）
  を 1 つに絞り、
  広域緩和ではなく
  strategy-local
  の 1 変更で PDCA を回す。

## 旧VM/GCP資料（実行対象外）
- `docs/OPS_GCP_RUNBOOK.md` / `docs/VM_OPERATIONS.md` / `docs/VM_BOOTSTRAP.md` は履歴アーカイブ。
- 現行運用では `scripts/vm.sh` / `gcloud compute *` を実行しない。

## ログ/台帳
- 運用監査ログの書込み先:
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/INDEX.md` の参照同期

## 共有ホワイトボード（MVP / local-only）
- 保存先DB: `logs/agent_whiteboard.db`（SQLite）
- CLI: `python3 scripts/agent_whiteboard.py`
- サポート操作: `post` / `list` / `watch` / `resolve` / `archive-task` / `purge-task` / `note` / `event` / `events` / `auto-session`
- 標準運用（ユーザー操作不要）:
  - 各AIエージェントは作業コマンドを `auto-session` で実行し、開始/進捗/完了を自動記録する。
  - 成功時は `resolve + archive` を自動実行。失敗時は `open` のまま `error` event を残す。
- 例:
  ```bash
  python3 scripts/agent_whiteboard.py post --task "worker実装開始" --body "担当: worker" --author "$USER"
  python3 scripts/agent_whiteboard.py list --status open --limit 20
  python3 scripts/agent_whiteboard.py watch --status all --interval-sec 2
  python3 scripts/agent_whiteboard.py note 1 --body "検証ケースを追加" --author "$USER"
  python3 scripts/agent_whiteboard.py events --task-id 1 --limit 50
  python3 scripts/agent_whiteboard.py auto-session --task "worker実装" --author "$USER" -- python3 -m pytest tests/workers/test_scalp_ping_5s_worker.py
  python3 scripts/agent_whiteboard.py resolve 1
  python3 scripts/agent_whiteboard.py archive-task 1
  python3 scripts/agent_whiteboard.py purge-task 1 --yes
  ```
