---
name: Memory & changelog discipline
description: アーキテクチャ/プロンプト/タスク変更時は必ずメモリ更新+変更ログ記録。変更を追跡可能にする
type: feedback
---

変更したらメモリは必ず更新する。変更ログも残す。

**Why:** アーキテクチャやプロンプトを変更しても、メモリが古いままだと次のセッションのClaudeが旧構造で動く。変更が追跡できないと問題の原因特定もできない。

**How to apply:**
1. **プロンプト変更** → 対応するメモリファイルを更新。何をなぜ変えたか記録
2. **タスク構成変更** (新規作成/無効化/間隔変更) → `project_architecture_v2.md` を更新
3. **live_monitor.py変更** (新機能追加/ルール変更) → 同上 + 変更ログ
4. **全変更** → `docs/CHANGELOG.md` に日時と変更内容を1行で追記
5. **ユーザーからのフィードバック** → 即座にfeedback_*.mdとして保存
6. **ワークツリーでの編集後は必ずmainにマージする** → traderタスクの作業ディレクトリはmainリポジトリ(`/Users/tossaki/App/QuantRabbit`)。ワークツリーブランチのままだとtraderに変更が見えない。docs/やscripts/の変更は特に注意

変更ログは `docs/CHANGELOG.md` に簡潔に:
```
2026-03-19 10:30 v3アーキテクチャ移行: live_monitor.py機械的ポジ管理追加、scalp-fast/swing-trader分離
```
