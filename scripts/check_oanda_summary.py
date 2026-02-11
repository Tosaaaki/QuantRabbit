#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys

import requests

# Re-exec into the repo venv when available so scripts work on the VM (deps + Secret Manager).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
try:
    if VENV_PY.exists() and pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
        os.execv(str(VENV_PY), [str(VENV_PY), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])
except Exception:
    # Best-effort; continue on current interpreter.
    pass

# Allow running as `python3 scripts/...py` without needing PYTHONPATH.
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
    print(
        json.dumps(
            {
                "account": account,
                "hedgingEnabled": summary.get("hedgingEnabled"),
                "marginRate": summary.get("marginRate"),
                "openTradeCount": summary.get("openTradeCount"),
                "pl": summary.get("pl"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

