# Firestore 移行タスクリスト

QuantRabbit のトレードログを Firestore 中心に切り替えるための運用／開発タスクを整理した計画書です。

## 1. Cloud Run から Firestore への読み取り権限付与

### 1-1. 必要なロールの確認と付与
- 対象サービスアカウント: `fx-trader` / `fx-exit-manager` などトレード関連 Cloud Run サービスが利用するサービスアカウント。
- 付与する IAM ロール（最小構成）:
  - `roles/datastore.user`（Firestore への読み書き）
  - `roles/logging.logWriter`（監査ログ記録を確実にするため既存ロールを維持）
- 付与コマンド例:
  ```bash
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/datastore.user"
  ```

### 1-2. Cloud Run 環境変数の設定
新しいリポジトリ層で参照する環境変数を Cloud Run に注入します。

```bash
gcloud run services update fx-trader \
  --project=$PROJECT_ID \
  --region=asia-northeast1 \
  --update-env-vars=\
TRADE_REPOSITORY_PROJECT=$PROJECT_ID,\
TRADE_REPOSITORY_COLLECTION=trades,\
TRADE_REPOSITORY_CACHE_TTL=45,\
TRADE_REPOSITORY_MAX_FETCH=2000

gcloud run services update fx-exit-manager \
  --project=$PROJECT_ID \
  --region=asia-northeast1 \
  --update-env-vars=TRADE_REPOSITORY_PROJECT=$PROJECT_ID
```

> MEMO: `TRADE_REPOSITORY_SQLITE` はフォールバック用なので設定不要。Workload Identity を利用している場合は `GOOGLE_APPLICATION_CREDENTIALS` を設定する必要はありません。

### 1-3. 実トラフィックでのクールダウン挙動確認
1. Cloud Run `fx-trader` / `fx-exit-manager` を再デプロイ後、`run.googleapis.com%2Fstderr` ログで `[RISK] Global drawdown …` などが出力されることを確認。
2. Firestore が未同期の場合は `Global drawdown unknown (trade data unavailable); blocking trades` が出るので、同期フロー（PositionManager など）を整備するまでは一時的に `TRADE_REPOSITORY_DISABLE_FIRESTORE=true` でロールバック可能。
3. 代表的なクールダウン（micro recent loss / hard cooldown）が Firestore 参照で動作しているか、特定時間帯の発注停止ログが期待どおりかをチェック。

## 2. Firestore 側のインデックスとクエリ監視

### 2-1. 必須コンポジットインデックスの作成
`state`, `pocket`, `close_time` を使ったクエリを高速化するため、以下のコンポジットインデックスを作成します。Firestore ネイティブモードを前提とします。

```bash
gcloud firestore indexes composite create \
  --collection-group=trades \
  --field-config="fieldPath=state,order=ASCENDING" \
  --field-config="fieldPath=pocket,order=ASCENDING" \
  --field-config="fieldPath=close_time,order=DESCENDING" \
  --project=$PROJECT_ID
```

必要に応じてオープントレード用（`state=OPEN`）のクエリ最適化として `state ASC, pocket ASC, entry_time DESC` のインデックスも作成する。

### 2-2. クエリ統計の監視
- GCP Console の **Firestore → Usage → Query Statistics** を有効化し、上記クエリ（`trades` collection group）のレイテンシが 50ms を大幅に超えないか監視。
- 週次で `gcloud firestore indexes composite list --project=$PROJECT_ID` を取得し、不要インデックスの整理や必要なフィールドの追加をレビュー。
- Cloud Logging で `[trade_repo] Firestore fetch failed` が継続していないかアラートを設定。

## 3. 書き込み経路を Firestore に統合する計画

### 3-1. 現状把握
- `execution/position_manager.py`, `execution/scalp_engine.py`, `analysis/*` がそれぞれ SQLite に書き込み。
- Cloud Run の複数リビジョンで同期が取れていないため、Firestore への単一化が必須。

### 3-2. 実装方針
1. **トレード書き込みインターフェースの定義**
   - `utils/trade_repository.py` に Firestore への upsert 用メソッド（例: `upsert_trades`, `update_trade_state`）を追加。
   - Firestore Document ID は `ticket_id` もしくは OANDA 取引 ID をそのまま利用。
2. **PositionManager の書き込み置き換え**
   - `PositionManager.sync_trades()` で取得した OANDA トランザクションを Firestore に upsert。
   - SQLite への書き込みはオプション化し、ローカルバックアップが必要な場合のみ `TRADE_REPOSITORY_DISABLE_FIRESTORE=true` を設定してフォールバック。
3. **Scalp エンジンのロギング**
   - スカルプ用の `scalp_trades` テーブル相当のメタデータを Firestore サブコレクション（例: `/trades/{ticket_id}/scalp_logs/{id}`）に保存するか、BigQuery / GCS ログバケットへ移行する。
4. **分析ツールの対応**
   - `analysis/perf_monitor.py` やレポート系スクリプトは Firestore から pandas 経由でデータを取得するよう改修。
   - レポート用途でローカル SQLite が必要な場合は、`scripts/trades_report.py --sync-sqlite` を使って Firestore → SQLite の片方向同期を実行。

### 3-3. デプロイと段階的移行
1. 段階 0: 読み取りのみ Firestore（現在）。
2. 段階 1: PositionManager を Firestore 書き込み → Cloud Run 本番で二重書き込み（Firestore + SQLite）を 1 週間運用。
3. 段階 2: リスクガード／マイクロガードが Firestore の最新データで安定していることを確認後、SQLite 依存コードを削除。
4. 段階 3: Firestore データの定期バックアップ（`gcloud firestore export` → GCS → `infra/terraform` でスケジュール化）。

### 3-4. テストとロールバック戦略
- ステージング環境で OANDA テストアカウントを使ってトレード→Firestore 書き込み→クールダウン挙動までの E2E テストを実施。
- Firestore 障害時のフォールバックとして `TRADE_REPOSITORY_DISABLE_FIRESTORE=true` で SQLite モードに戻せるよう環境変数を残す。

---

### 付録: トラブルシューティング
- **Firestore の権限エラー (`403 PERMISSION_DENIED`)**: Service Account に `roles/datastore.user` が付与されているか確認。Workload Identity を使う場合は `gcloud beta run services describe` でバインド先をチェック。
- **Composite Index 未作成エラー**: Firestore ログに `FAILED_PRECONDITION: The query requires an index` が出た場合、エラーメッセージの `View Index` リンクから定義を確認し、`gcloud firestore indexes composite create` を実行。
- **レイテンシが高い**: キャッシュ TTL を延ばす（デフォルト 30 秒 → 45–60 秒）か、クエリ対象期間を短縮。必要なら Cloud Memorystore や Vertex Cache の検討。

---

## 4. 「脱 Firestore」ロードマップ（2025-10-21 草案）

BigQuery／Pub/Sub を中心とした新アーキテクチャが整い始めたため、Firestore 依存を徐々に解消する方針を記録する。

### 4-1. 現状の代替コンポーネント
- **リアルタイム統計**: `analysis/bq_stats.py` が BigQuery `trades_raw` を 7日／30日窓で集計し、`risk_multiplier` へフォールバック指標を提供。
- **リスクモデル**: `analytics/risk_model.py` + Cloud Run サービスが BigQuery ML → Pub/Sub 更新を完結。Firestore の学習スコアに頼らなくてもリスク係数が決まる状態。
- **ローカル同期**: `scripts/export_trades_to_bq.py` で SQLite → BigQuery を全件同期済み。以降は差分エクスポートを自動化（10 分間隔）。

### 4-2. Firestore が担っている残タスク
1. Cloud Run `trader_service` のステータス記録 (`/status/trader`)・設定ドキュメント `config/params` の取得。
2. ニュース／学習スコアの共有 (`status.trader.learning_scores`, `status.trader.news_*`)。
3. スカルプ設定 (`analysis/scalp_config.py`) のオーバーライド。
4. 一部運用ツール（`scripts/trades_report.py`）が Firestore → SQLite の再同期に依存。

### 4-3. 段階的撤廃ステップ
| Phase | 目標 | 対応タスク | 目安 |
|-------|------|------------|------|
| P0 | Firestore 依存箇所の棚卸しとモニタリング継続 | `cloudrun/trader_service.py` のログ監視、Firestore 障害時のフォールバック動作確認 (`TRADE_REPOSITORY_DISABLE_FIRESTORE`) | 実施済み |
| P1 | ステータス／設定のモダン化 | Firestore 書き込みを BigQuery テーブル or Cloud Logging へ切り替え。設定値は `config/env` + BigQuery `config` テーブルへ集約。 | 2025-11 |
| P2 | ニュース・学習スコアの移行 | Pub/Sub / BigQuery ビューに切り替え。`analysis/risk_feed.py` の JSON にニュースハイライトを含め、Firestore `status.trader` 依存を除去。 | 2025-12 |
| P3 | スカルプ設定の再設計 | `analysis/scalp_config.py` を BigQuery / GCS YAML 参照へ移行。Firestore ドキュメントは参照のみとして段階的に廃止。 | 2026-01 |
| P4 | Firestore 完全撤退 | Firestore でのみ管理していたコレクションを BigQuery / GCS に移し、ローカルバックアップ → BigQuery 一元化を完了。Firestore プロジェクトからデータ削除。 | 2026-02 |

### 4-4. 補足メモ
- Firestore を段階的に無効化する前に、Cloud Console のアラート（Firestore 利用量／エラー率）をセットし、影響範囲を把握する。
- BigQuery 側に `status_trader` テーブルを用意する場合は、`STRUCT` で JSON を丸ごと保存し、Looker Studio などから参照できる形を目指す。
- スケジュールは暫定。実装ロードマップの進捗に応じて適宜アップデートすること。
