# OANDA API接続

## 認証
設定ファイル: `config/env.toml`（gitignore対象）
```python
token = open('config/env.toml').read()
token = [l.split('=')[1].strip().strip('"') for l in token.split('\n') if l.startswith('oanda_token')][0]
acct = [l.split('=')[1].strip().strip('"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_account_id')][0]
base = 'https://api-fxtrade.oanda.com'  # 本番環境
```

## 注文は urllib で REST API 直叩き
- 注文後は必ず `logs/live_trade_log.txt` にファイル記録
- **成行だけ使うな**: 指値(LIMIT)・TP・SL・トレーリングストップを活用しろ → 詳細はSKILL.mdの「指値・TP・SL・トレールを使え」参照

## 注文タイプ
| タイプ | 用途 | エンドポイント |
|--------|------|---------------|
| MARKET | 即時約定 | POST /v3/accounts/{acct}/orders |
| LIMIT | 指値（価格待ち伏せ） | POST /v3/accounts/{acct}/orders |
| TP/SL追加 | 既存ポジに利確/損切り | PUT /v3/accounts/{acct}/trades/{id}/orders |
| Trailing Stop | 利益追従型損切り | PUT /v3/accounts/{acct}/trades/{id}/orders |
| 未約定注文確認 | pending orders一覧 | GET /v3/accounts/{acct}/pendingOrders |
| 注文キャンセル | 指値取消 | PUT /v3/accounts/{acct}/orders/{orderID}/cancel |

## 決済（ヘッジ口座ミス防止）
```bash
python3 tools/close_trade.py {tradeID}          # 全決済
python3 tools/close_trade.py {tradeID} {units}   # 部分決済
```
**POST /orders で反対注文を出すな。新規ポジが開く。** 必ず `close_trade.py`（PUT /trades/{id}/close）を使え。

## データ取得ツール
| ツール | 用途 |
|--------|------|
| `python3 tools/refresh_factor_cache.py --all --quiet` | H1/H4テクニカル更新 |
| `python3 collab_trade/indicators/quick_calc.py {PAIR} {TF} {BARS}` | 共同トレード時のテクニカル計算 |
| `python3 tools/trade_performance.py` | パフォーマンス集計 |
| `cat logs/technicals_{PAIR}.json` | キャッシュ済みテクニカル参照 |

## 監視ペア
USD_JPY, EUR_USD, GBP_USD, AUD_USD, EUR_JPY, GBP_JPY, AUD_JPY
