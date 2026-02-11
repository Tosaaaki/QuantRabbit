#!/usr/bin/env python3
from __future__ import annotations
import os
import pathlib
import sys
import json
import requests

# Allow running as `python3 scripts/...py` without needing PYTHONPATH.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_env_file(path: str = "/etc/quantrabbit.env") -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except Exception:
        return env
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
            v = v[1:-1]
        if k:
            env[k] = v
    return env


def _get_secret(key: str) -> str:
    # Prefer production code path (Secret Manager / env.toml / env vars) when available.
    try:
        from utils.secrets import get_secret  # type: ignore

        return get_secret(key)
    except Exception:
        pass

    # Fallback: /etc/quantrabbit.env (common on VM), then process env.
    env_map = {
        "oanda_token": "OANDA_TOKEN",
        "oanda_account_id": "OANDA_ACCOUNT",
        "oanda_practice": "OANDA_PRACTICE",
    }
    env_key = env_map.get(key)
    if not env_key:
        raise KeyError(f"unknown secret key: {key}")
    if os.environ.get(env_key):
        return str(os.environ[env_key])

    env = _load_env_file()
    # Populate os.environ for downstream users (best-effort).
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if os.environ.get(env_key):
        return str(os.environ[env_key])
    if env.get(env_key):
        return str(env[env_key])
    raise KeyError(f"secret '{key}' not found in env or /etc/quantrabbit.env")


def main() -> int:
    try:
        token = _get_secret("oanda_token")
        account = _get_secret("oanda_account_id")
    except Exception as exc:
        print(f"failed to load secrets: {exc}")
        return 1

    host = "https://api-fxtrade.oanda.com"
    try:
        if _get_secret("oanda_practice").lower() == "true":
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
