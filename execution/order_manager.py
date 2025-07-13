"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations
import requests
from typing import Literal

from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
PRACT = False  # OANDAのpracticeフラグは常にFalse (本番環境用)

REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def market_order(
    instrument: str,
    units: int,
    sl_price: float,
    tp_price: float,
    pocket: Literal["micro", "macro"],
) -> str:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id
    """
    url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/orders"

    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price:.3f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.3f}"},
            "clientExtensions": {"tag": f"pocket={pocket}"},
        }
    }

    r = requests.post(url, headers=HEADERS, json=body, timeout=5)
    r.raise_for_status()
    data = r.json()
    trade_id = (
        data.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
    )
    return trade_id
