#!/usr/bin/env python3
"""Read-only OANDA auth smoke test using the shared config loader."""

from __future__ import annotations

import argparse
import json
import urllib.request

from config_loader import get_oanda_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OANDA credentials from config/env.toml")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    args = parser.parse_args()

    cfg = get_oanda_config()
    account_id = cfg["oanda_account_id"]
    token = cfg["oanda_token"]
    base_url = cfg["oanda_base_url"]

    req = urllib.request.Request(
        f"{base_url}/v3/accounts/{account_id}/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    account = json.loads(urllib.request.urlopen(req, timeout=20).read())["account"]
    payload = {
        "status": "ok",
        "account_id": account_id,
        "practice": cfg["oanda_practice"],
        "alias": account.get("alias"),
        "currency": account.get("currency"),
        "nav": account.get("NAV"),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        env_name = "practice" if cfg["oanda_practice"] else "live"
        print(f"OANDA auth OK ({env_name})")
        print(f"account_id={payload['account_id']}")
        print(f"alias={payload['alias']}")
        print(f"currency={payload['currency']}")
        print(f"nav={payload['nav']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
