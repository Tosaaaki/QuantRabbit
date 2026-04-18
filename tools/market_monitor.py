#!/usr/bin/env python3
"""
常駐市場監視スクリプト — Claudeの代わりに市場を監視し続ける。
Claudeは logs/market_snapshot.json を読んで判断するだけ。

Usage:
    python3 tools/market_monitor.py

出力: logs/market_snapshot.json (30秒ごとに更新)
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config_loader import get_oanda_config

# `market_monitor.py` lives in `<repo>/tools/`, so the repo root is one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL_PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
SNAPSHOT_PATH = ROOT / "logs" / "market_snapshot.json"
HANDOFF_REQ = ROOT / "logs" / ".trader_handoff_request"
LOCK_FILE = ROOT / "logs" / ".trader_lock"

TECHNICALS_INTERVAL = 600  # 10分
PRICE_INTERVAL = 30         # 30秒


def load_config():
    cfg = get_oanda_config()
    return cfg["oanda_token"], cfg["oanda_account_id"], cfg["oanda_base_url"]


def api_get(token, base_url, path):
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def fetch_prices(token, account_id, base_url):
    instruments = ",".join(ALL_PAIRS)
    data = api_get(token, base_url, f"/v3/accounts/{account_id}/pricing?instruments={instruments}")
    if "error" in data:
        return {"error": data["error"]}

    prices = {}
    for p in data.get("prices", []):
        inst = p["instrument"]
        bid = float(p["bids"][0]["price"]) if p.get("bids") else None
        ask = float(p["asks"][0]["price"]) if p.get("asks") else None
        if bid and ask:
            prices[inst] = {
                "bid": bid,
                "ask": ask,
                "mid": round((bid + ask) / 2, 5),
                "spread": round((ask - bid) * (100 if "JPY" in inst else 10000), 1),
            }
    return prices


def fetch_account(token, account_id, base_url):
    data = api_get(token, base_url, f"/v3/accounts/{account_id}/summary")
    if "error" in data:
        return {"error": data["error"]}

    acct = data.get("account", {})
    return {
        "balance": float(acct.get("balance", 0)),
        "nav": float(acct.get("NAV", 0)),
        "unrealizedPL": float(acct.get("unrealizedPL", 0)),
        "marginUsed": float(acct.get("marginUsed", 0)),
        "marginAvailable": float(acct.get("marginAvailable", 0)),
        "openTradeCount": int(acct.get("openTradeCount", 0)),
    }


def fetch_open_trades(token, account_id, base_url):
    data = api_get(token, base_url, f"/v3/accounts/{account_id}/openTrades")
    if "error" in data:
        return {"error": data["error"]}

    trades = []
    for t in data.get("trades", []):
        trades.append({
            "id": t["id"],
            "instrument": t["instrument"],
            "units": int(t["currentUnits"]),
            "price": float(t["price"]),
            "unrealizedPL": float(t.get("unrealizedPL", 0)),
            "stopLoss": float(t["stopLossOrder"]["price"]) if t.get("stopLossOrder") else None,
            "takeProfit": float(t["takeProfitOrder"]["price"]) if t.get("takeProfitOrder") else None,
            "trailingStop": float(t["trailingStopLossOrder"]["distance"]) if t.get("trailingStopLossOrder") else None,
        })
    return trades


def check_handoff():
    return HANDOFF_REQ.exists()


def update_lock():
    LOCK_FILE.write_text(str(int(time.time())))


def load_technicals():
    """既存のtechnicals JSONを読む"""
    result = {}
    for pair in ALL_PAIRS:
        path = ROOT / "logs" / f"technicals_{pair}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                result[pair] = data.get("timeframes", {})
            except Exception:
                pass
    return result


def run_technicals():
    """refresh_factor_cache.pyを実行"""
    import subprocess
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "trader_tools" / "refresh_factor_cache.py"), "--all", "--quiet"],
            timeout=60,
            capture_output=True,
        )
        return True
    except Exception as e:
        print(f"Technicals error: {e}", file=sys.stderr)
        return False


def write_snapshot(snapshot):
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2)
    tmp.rename(SNAPSHOT_PATH)


def main():
    token, account_id, base_url = load_config()
    last_technicals = 0
    cycle = 0

    print(f"Market monitor started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Snapshot: {SNAPSHOT_PATH}")
    print(f"Price interval: {PRICE_INTERVAL}s, Technicals interval: {TECHNICALS_INTERVAL}s")

    while True:
        cycle += 1
        now = time.time()

        # テクニカル更新（10分ごと）
        if now - last_technicals > TECHNICALS_INTERVAL:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing technicals...")
            run_technicals()
            last_technicals = now

        # データ取得
        prices = fetch_prices(token, account_id, base_url)
        account = fetch_account(token, account_id, base_url)
        trades = fetch_open_trades(token, account_id, base_url)
        technicals = load_technicals()
        handoff = check_handoff()

        # ロック更新（生存証明）
        update_lock()

        # スナップショット書き出し
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "handoff_requested": handoff,
            "account": account,
            "open_trades": trades,
            "prices": prices,
            "technicals_summary": {},
        }

        # テクニカルサマリー（H1のみ、コンパクト）
        for pair, tfs in technicals.items():
            h1 = tfs.get("H1", {})
            if h1:
                snapshot["technicals_summary"][pair] = {
                    "rsi": h1.get("rsi"),
                    "adx": h1.get("adx"),
                    "di+": h1.get("plus_di"),
                    "di-": h1.get("minus_di"),
                    "macd_hist": h1.get("macd_hist"),
                    "stoch_rsi": h1.get("stoch_rsi"),
                    "bb_mid": h1.get("bb_mid"),
                    "regime": h1.get("regime"),
                }

        write_snapshot(snapshot)

        ts = datetime.now().strftime('%H:%M:%S')
        trade_count = len(trades) if isinstance(trades, list) else 0
        print(f"[{ts}] Cycle {cycle} | Trades: {trade_count} | Handoff: {handoff}")

        if handoff:
            print("Handoff requested — monitor continues but flagging for Claude")

        time.sleep(PRICE_INTERVAL)


if __name__ == "__main__":
    main()
