#!/usr/bin/env python3
from __future__ import annotations
import sys
import json
import requests
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

    url = f"{host}/v3/accounts/{account}/summary"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"failed to fetch summary: {exc}")
        try:
            print(f"status={resp.status_code}")
            print(resp.text[:400])
        except Exception:
            pass
        return 2

    summary = resp.json().get("account", {})
    print(json.dumps({
        "account": account,
        "hedgingEnabled": summary.get("hedgingEnabled"),
        "marginRate": summary.get("marginRate"),
        "openTradeCount": summary.get("openTradeCount"),
        "pl": summary.get("pl"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

