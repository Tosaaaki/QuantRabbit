# TASKS – Repository Task Board

本リポジトリの全タスクを一元管理する台帳。タスクが発生したら本ファイルへ逐次追記し、作業中は本ファイルを参照しながら進め、完了後はアーカイブへ移してください。詳細ポリシーは `AGENTS.md` の「11. タスク運用ルール（共通）」を参照。

## 運用ルール（要約）

- 発生: 本ファイル「Open Tasks」にテンプレートで追加。
- 進行: ステータス/Plan/Notes を更新し、PR/コミットには `[Task:<ID>]` を含める。
- 完了: 「Archive」へ移動し、完了日・PR/コミットリンク・要約を追記。
- 例外: オンライン自動チューニング関連は `docs/autotune_taskboard.md` を主に使用（必要に応じて相互リンク）。

## テンプレート

以下をコピーして使用:

```md
- [ ] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: todo | in-progress | review | done
  Priority: P1 | P2 | P3
  Owner: <担当>
  Scope/Paths: <例> AGENTS.md, docs/TASKS.md
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <受入条件1>
    - <受入条件2>
  Plan:
    - <主要ステップ1>
    - <主要ステップ2>
  Notes:
    - <補足>
```

## Open Tasks

（ここに新規タスクを追加）

> 現在、オープンなタスクはありません。

## Archive

完了タスクを以下形式で移動・記録:

```md
- [x] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: done
  Priority: P2
  Owner: <担当>
  Scope/Paths: <対象>
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <満たした条件1>
  Completed: YYYY-MM-DD
  PR: <PRリンク or コミットSHA>
  Summary: <1〜3 行で要約>
```

---

- [x] ID: T-20251102-001
  Title: gcloud doctor/installer とデプロイ前診断導入
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/install_gcloud.sh, scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, README.md
  Context: デプロイ不能・gcloud 未導入の再発防止（IAP/OS Login を含む）
  Acceptance:
    - gcloud 未導入時に install スクリプトで導入可能
    - doctor で project/API/OS Login/インスタンス/SSH を検査できる
    - deploy スクリプトが前提不備で早期失敗・誘導
    - セットアップ手順ドキュメントがある
  Completed: 2025-11-02
  PR: <pending>
  Summary: gcloud インストーラ・プリフライト（doctor）とドキュメントを追加し、deploy スクリプトに前提チェックを組み込みました。

- [x] ID: T-20251102-002
  Title: ヘッドレス運用（サービスアカウント）対応
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, AGENTS.md
  Context: アクティブアカウント不在でも GCP/VM を操作できるようにする
  Acceptance:
    - SA キー指定で doctor/deploy が動作（-K）
    - 必要時 SA インパーソネートにも対応（-A）
    - ドキュメントに手順とロール要件を明記
  Completed: 2025-11-02
  PR: <pending>
  Summary: doctor/deploy に SA キー自動有効化とインパーソネートを追加し、ヘッドレス運用手順を docs に追記しました。

- [x] ID: T-20251102-003
  Title: AGENTS.md に quantrabbit クイックコマンドを追加
  Status: done
  Priority: P3
  Owner: maint
  Scope/Paths: AGENTS.md
  Context: 実運用の即時参照用に、Doctor/Deploy/IAP/SA の実値例を集約
  Acceptance:
    - AGENTS.md 10.3 に quantrabbit 固定のコマンド例がある
    - 旧来の冗長/重複手順は削除済み
  Completed: 2025-11-02
  PR: <pending>
  Summary: プロジェクト/ゾーン/インスタンス実値のクイックコマンドを AGENTS.md に追記。
