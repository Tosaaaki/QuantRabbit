# AGENT.me  –  QuantRabbit Agent Specification

## 1. ミッション
> **狙い** : USD/JPY で 1 日 +100 pips を実現する、 24/7 無裁量トレーディング・エージェント。  
> **境界** : 発注・リスクは機械的、曖昧判断とニュース解釈は GPT‑4o‑mini に委譲。

---

## 2. コンポーネント間の契約

| レイヤ | 担当 | 期待する入出力 |
|--------|------|----------------|
| **DataFetcher** | `market_data/*` | + Tick JSON<br>+ Candle dict<br>+ raw News JSON (→ GCS) |
| **IndicatorEngine** | `indicators/*` | ← Candle deque<br>→ factors dict {ma10, rsi, …} |
| **Regime & Focus** | `analysis/regime_classifier.py` / `focus_decider.py` | ← factors<br>→ macro/micro レジーム・weight_macro |
| **GPT Decider** | `analysis/gpt_decider.py` | ← focus + perf + news<br>→ JSON {focus_tag, weight_macro, ranked_strategies} |
| **Strategy Plugin** | `strategies/*` | ← factors<br>→ dict {action, sl_pips, tp_pips} or None |
| **Risk Guard** | `execution/risk_guard.py` | ← lot, SL/TP, pocket<br>→ bool (可否)・調整値 |
| **Order Manager** | `execution/order_manager.py` | ← units, sl, tp, tag<br>→ OANDA ticket ID |
| **Logger** | `logs/*.db` | 全コンポーネントが INSERT |

---

## 3. ライフサイクル

1. **Startup (`main.py`)**
   1. env.toml 読込 → Secrets 確認
   2. WebSocket 接続確立
2. **Every 60 s**
   1. 新ローソク → factors 更新  
   2. regime + focus → GPT decision  
   3. pocket lot 配分 → Strategy loop  
   4. Risk guard → order_manager 発注  
   5. trades.db / news.db にログ
3. **Background Jobs**
   - `news_fetcher` RSS → GCS raw/  
   - Cloud Run `news‑summarizer`  raw → summary/  
   - `summary_ingestor` summary/ → news.db  
   - `execution.exit_manager.exit_loop` VM 本体で 30 s 間隔のポジション監視（SL/TP 調整・強制決済）
   - nightly `backup_to_gcs.sh` logs/ → backup bucket

> **VM 常駐サービス**: `quantrabbit.service`（`main.py` + exit manager）と `scripts/oanda_sync_loop.py`（PositionManager 補完）を systemd で管理する。Cloud Run 側にはニュース要約系のみを残す。

---

## 4. 環境変数 / Secret 一覧

| Key | 説明 |
|-----|------|
| `OPENAI_API_KEY` | GPT‑4o‑mini 用 |
| `OANDA_TOKEN` / `OANDA_ACCOUNT` | REST / Stream |
| `GCP_PROJECT` / `GOOGLE_APPLICATION_CREDENTIALS` | GCS・Pub/Sub |
| `GCS_BACKUP_BUCKET` | logs バックアップ先 |

---

## 5. トークン & コストガード

* `.cache/token_usage.json` に月累計。  
* `openai.max_month_tokens` (env.toml) で上限設定。  
* 超過時：`news_fetcher` は継続、`gpt_decider` はフォールバック JSON を返す。

---

## 6. 安全装置

* **Pocket DD** : micro 5 %、macro 15 % → 取引停止  
* **Global DD** : 20 % → プロセス自動終了 (`risk_guard`)  
* **Event モード** : 指標 ±30 min → micro 新規禁止  
* **Timeout** : GPT 7 s、OANDA REST 5 s → 再試行 / フォールバック  
* **Healthbeat** : `main.py` が 5 min ping を Cloud Logging に残す

---

## 7. デプロイ手順 (要約)

```bash
gcloud builds submit --tag gcr.io/$PROJ/news-summarizer
cd infra/terraform && terraform init && terraform apply
gcloud compute ssh fx-trader-vm --command "git pull && ./startup.sh"

8. チーム運用ルール
	1.	1 ファイル = 1 PR、Squash Merge、CI green 必須
	2.	コード規約：black / ruff / mypy (optional)
	3.	秘匿情報は 絶対に Git に push しない
	4.	不具合・改善は GitHub Issue で管理（ラベル: bug/feat/doc/ops）
	5.	コードを変更した場合は必ず本番デプロイまで完遂し、ローカル環境での運用・検証は実施しない
	6.	全ての運用判断とスケジュールは JST（日本標準時）基準で徹底する

⸻

9. 参考ドキュメント
	•	README.md             – 🍵 ユーザ向け概観
	•	パッチ適用の推奨シーケンス.pdf – 開発手順ガイド
	•	全体仕様まとめ（最終版）.pdf – アーキテクチャ詳細
	•	OFL.txt + ZenOldMincho-*.ttf – 付属フォントライセンス

---

## 10. 運用メモ（2025-10-20 更新）

- **2025-10-21 コストガード修正**: Cloud Run/VM 向けに `utils.cost_guard` を破損耐性付きに更新。`token_usage.json` が壊れても JSONDecodeError で止まらず、自動バックアップ＋リセットする。VM は Terraform のスタートアップスクリプトで同ファイルを上書きするよう調整済み。
- **2025-10-21 VM 作り直し**: e2-micro のリソース不足により `asia-northeast1-a` で失敗したため、Terraform を `asia-northeast1-b` へ切り替え。`dpkg --configure -a` を追加し、起動時の `dpkg` ロックで失敗しないようにした。
- **2025-10-21 デプロイ記録**: `gcloud builds submit --tag gcr.io/quantrabbit/news-summarizer` と `terraform apply` を実行し、Cloud Run / GCE 構成を更新。`gcloud compute ssh fx-trader-vm --zone asia-northeast1-a --command "cd /opt/quantrabbit && git pull && ./startup.sh"` は OS Login 公開鍵拒否（Permission denied, publickey）で失敗したため、鍵登録またはサービスアカウント権限の確認が必要。
- **ローカルテスト非実施**: ユーザ指示により、デプロイ前後で `pytest` などローカルテストは走らせない。検証はリモート環境や実運用ログで行う。
- **GCE 依存パッケージ**: `fx-trader-vm` のスタートアップでは `pip install -r requirements.txt` が完了しないと `quantrabbit.service` や `/etc/quantrabbit.env` が作成されない。外部で配布が止まったパッケージ（例: pandas-ta）は requirements から除外し、必要ならベンダリングすること。
- **フォールバック禁止**: GPT 決裁・エグジット・戦略補完のフォールバックは全停止。`ALLOW_*_FALLBACK` 系 env は `false` が既定値で、障害時はスキップ/ログのみで運用する。
- **ロット設定**: VM 環境で `RISK_PCT=0.06`, `RISK_MAX_LOT=5.0`, `SCALP_BASE_LOT=0.012` を既定にし、macro/micro/scalp が常時エントリーできるよう基準を引き上げている。
- **指示履歴の記録**: 新たな運用ルールや重要指示は本節へ追記し、エージェント全体で共有する。
- **トレード履歴取得**: ローカル `logs/` 配下の CSV/DB は遅延するため、分析やレポート作成時は GCP（Firestore / BigQuery / GCS バックアップ）から最新データを取得すること。
- **DB ログ確保**: Cloud Run 側で《`utils.trade_logger.log_trade_snapshot`》が全トレードを `logs/trades.db` に即時書き込みする。追加で `scripts/oanda_sync_loop.py` を VM で常駐させ、OANDA 取引と SQLite/Firestore の整合性を 45–60 秒間隔で保つこと。
