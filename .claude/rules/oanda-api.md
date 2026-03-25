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

## データ取得ツール
| ツール | 用途 |
|--------|------|
| `python3 scripts/trader_tools/refresh_factor_cache.py --all --quiet` | H1/H4テクニカル更新 |
| `python3 collab_trade/indicators/quick_calc.py {PAIR} {TF} {BARS}` | 共同トレード時のテクニカル計算 |
| `python3 scripts/trader_tools/trade_performance.py` | パフォーマンス集計 |
| `cat logs/technicals_{PAIR}.json` | キャッシュ済みテクニカル参照 |

## 監視ペア
USD_JPY, EUR_USD, GBP_USD, AUD_USD, EUR_JPY, GBP_JPY, AUD_JPY
