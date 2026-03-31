# OANDA API Connection

## Authentication
Config file: `config/env.toml` (gitignored)
```python
token = open('config/env.toml').read()
token = [l.split('=')[1].strip().strip('"') for l in token.split('\n') if l.startswith('oanda_token')][0]
acct = [l.split('=')[1].strip().strip('"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_account_id')][0]
base = 'https://api-fxtrade.oanda.com'  # production environment
```

## Hit the REST API directly with urllib
- After every order, log to `logs/live_trade_log.txt` without fail
- **Don't use market orders only**: Use LIMIT, TP, SL, and trailing stops → see "Use limit/TP/SL/trail" in SKILL.md for details

## Order Types
| Type | Purpose | Endpoint |
|--------|------|---------------|
| MARKET | Immediate execution | POST /v3/accounts/{acct}/orders |
| LIMIT | Limit order (price ambush) | POST /v3/accounts/{acct}/orders |
| TP/SL add | Add TP/SL to existing position | PUT /v3/accounts/{acct}/trades/{id}/orders |
| Trailing Stop | Profit-trailing stop loss | PUT /v3/accounts/{acct}/trades/{id}/orders |
| Pending order check | List pending orders | GET /v3/accounts/{acct}/pendingOrders |
| Order cancel | Cancel limit order | PUT /v3/accounts/{acct}/orders/{orderID}/cancel |

## Closing Positions (hedge account mistake prevention)
```bash
python3 tools/close_trade.py {tradeID}          # full close
python3 tools/close_trade.py {tradeID} {units}   # partial close
```
**Never place a counter order via POST /orders. It opens a new position.** Always use `close_trade.py` (PUT /trades/{id}/close).

## Data Retrieval Tools
| Tool | Purpose |
|--------|------|
| `python3 tools/refresh_factor_cache.py --all --quiet` | Update H1/H4 technicals |
| `python3 collab_trade/indicators/quick_calc.py {PAIR} {TF} {BARS}` | Technical calculation for collab trade |
| `python3 tools/trade_performance.py` | Performance summary |
| `cat logs/technicals_{PAIR}.json` | Read cached technicals |

## Monitored Pairs
USD_JPY, EUR_USD, GBP_USD, AUD_USD, EUR_JPY, GBP_JPY, AUD_JPY
