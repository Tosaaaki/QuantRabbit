"""
execution.account_info
~~~~~~~~~~~~~~~~~~~~~~
Fetches OANDA account summary to derive equity and available margin.
"""

from __future__ import annotations
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

