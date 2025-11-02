"""
execution.account_info
~~~~~~~~~~~~~~~~~~~~~~
Fetches OANDA account summary to derive equity and available margin.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

from utils.secrets import get_secret

TOKEN = get_secret("oanda_token")
try:
    PRACTICE = get_secret("oanda_practice").lower() == "true"
except Exception:
    PRACTICE = True
ACCOUNT = get_secret("oanda_account_id")

REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACTICE else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

_DEFAULT_EQUITY = float(os.getenv("REFERENCE_EQUITY", "10000") or 10000.0)
_DEFAULT_MARGIN_RATE = float(os.getenv("DEFAULT_MARGIN_RATE", "0.05") or 0.05)
_STATE_LOCK: asyncio.Lock = asyncio.Lock()
_ACCOUNT_CACHE: Dict[str, Any] = {
    "equity": _DEFAULT_EQUITY,
    "NAV": _DEFAULT_EQUITY,
    "balance": _DEFAULT_EQUITY,
    "marginAvailable": _DEFAULT_EQUITY,
    "marginRate": _DEFAULT_MARGIN_RATE,
    "currency": os.getenv("ACCOUNT_CURRENCY", "USD"),
    "updated_at": datetime.min.replace(tzinfo=timezone.utc),
    "last_error": None,
}


async def get_account_summary() -> dict:
    """Return OANDA account summary as dict.

    Keys of interest: NAV, balance, marginAvailable, marginRate (floats).
    """
    url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/summary"
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json().get("account", {})
    # Normalize numeric fields to float
    def f(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    out = {
        "NAV": f(data.get("NAV")),
        "balance": f(data.get("balance")),
        "marginAvailable": f(data.get("marginAvailable")),
        "marginRate": f(data.get("marginRate")),
        "currency": data.get("currency"),
    }
    return out


async def get_account_state(ttl: float = 60.0) -> Dict[str, Any]:
    """Return cached account snapshot, refreshing from REST when stale."""

    now = datetime.now(timezone.utc)
    async with _STATE_LOCK:
        age = (
            (now - _ACCOUNT_CACHE["updated_at"]).total_seconds()
            if isinstance(_ACCOUNT_CACHE.get("updated_at"), datetime)
            else float("inf")
        )
        if age <= ttl:
            return dict(_ACCOUNT_CACHE)

        try:
            summary = await get_account_summary()
            _ACCOUNT_CACHE["equity"] = summary.get("NAV") or summary.get("balance") or _ACCOUNT_CACHE["equity"]
            _ACCOUNT_CACHE["NAV"] = summary.get("NAV", _ACCOUNT_CACHE["equity"])
            _ACCOUNT_CACHE["balance"] = summary.get("balance", _ACCOUNT_CACHE["equity"])
            _ACCOUNT_CACHE["marginAvailable"] = summary.get(
                "marginAvailable", _ACCOUNT_CACHE["marginAvailable"]
            )
            margin_rate = summary.get("marginRate")
            if margin_rate:
                _ACCOUNT_CACHE["marginRate"] = margin_rate
            _ACCOUNT_CACHE["currency"] = summary.get("currency") or _ACCOUNT_CACHE.get("currency")
            _ACCOUNT_CACHE["last_error"] = None
        except Exception as exc:
            logging.warning("[ACCOUNT] summary refresh failed: %s", exc)
            _ACCOUNT_CACHE["last_error"] = str(exc)
        finally:
            _ACCOUNT_CACHE["updated_at"] = now

        return dict(_ACCOUNT_CACHE)


def account_state_snapshot() -> Dict[str, Any]:
    """Return the latest cached snapshot without refreshing."""

    return dict(_ACCOUNT_CACHE)
