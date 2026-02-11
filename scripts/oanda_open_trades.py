#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys

import requests

# Allow running as `python3 scripts/...py` without needing PYTHONPATH.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.secrets import get_secret


def main() -> int:
    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")
    except Exception as exc:
        print(f"failed to load secrets: {exc}")
        return 1

    host = "https://api-fxtrade.oanda.com"
    try:
        if get_secret("oanda_practice").lower() == "true":
            host = "https://api-fxpractice.oanda.com"
    except Exception:
        pass

    url = f"{host}/v3/accounts/{account}/openTrades"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"failed to fetch openTrades: {exc}")
        try:
            print(f"status={resp.status_code}")
            print(resp.text[:400])
        except Exception:
            pass
        return 2

    trades = payload.get("trades") or []
    out = []
    for t in trades:
        tp = t.get("takeProfitOrder") or {}
        sl = t.get("stopLossOrder") or {}
        tsl = t.get("trailingStopLossOrder") or {}
        out.append(
            {
                "id": t.get("id"),
                "instrument": t.get("instrument"),
                "currentUnits": t.get("currentUnits"),
                "price": t.get("price"),
                "openTime": t.get("openTime"),
                "unrealizedPL": t.get("unrealizedPL"),
                "realizedPL": t.get("realizedPL"),
                "marginUsed": t.get("marginUsed"),
                "clientExtensions": t.get("clientExtensions") or {},
                "takeProfit": {"id": tp.get("id"), "price": tp.get("price")} if tp else None,
                "stopLoss": {"id": sl.get("id"), "price": sl.get("price")} if sl else None,
                "trailingStopLoss": {"id": tsl.get("id"), "distance": tsl.get("distance")} if tsl else None,
            }
        )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

