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
| **GPT Decider** | `analysis/gpt_decider.py` | ← focus + perf + news<br>→ JSON {focus_tag, weight_macro, ranked_strategies, strategy_directives} |
| **Strategy Plugin** | `strategies/*` | ← factors<br>→ dict {action, sl_pips, tp_pips} or None |
| **GPT Exit Advisor** | `analysis/gpt_exit_advisor.py` / `execution/exit_manager.py` | ← open trades + factors<br>→ JSON {close_now, target_tp_pips, target_sl_pips, confidence} |
| **News Summarizer** | `cloudrun/news_summarizer_runner.py` | ← RSS/Body<br>→ summary JSON (GPT-5 nano by default) |
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
   - nightly `backup_to_gcs.sh` logs/ → backup bucket

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

⸻

9. 参考ドキュメント
	•	README.md             – 🍵 ユーザ向け概観
	•	パッチ適用の推奨シーケンス.pdf – 開発手順ガイド
	•	全体仕様まとめ（最終版）.pdf – アーキテクチャ詳細
	•	OFL.txt + ZenOldMincho-*.ttf – 付属フォントライセンス
