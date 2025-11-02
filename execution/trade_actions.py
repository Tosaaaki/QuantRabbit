from __future__ import annotations

import requests
from utils.secrets import get_secret

try:
    PRACT = get_secret("oanda_practice").lower() == "true"
except Exception:
    PRACT = True

TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def close_trade(trade_id: str, *, units: int | None = None) -> bool:
    """Close a trade by trade ID. Returns True if request accepted.

    units: 指定時はその数量のみクローズ（正の整数、通貨単位）。未指定で全量。
    """
    url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}/close"
    try:
        payload = {"units": str(units)} if units is not None else {"units": "ALL"}
        r = requests.put(url, headers=HEADERS, timeout=5, json=payload)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[trade_actions] close_trade error for {trade_id}: {e}")
        return False


def update_trade_orders(
    trade_id: str,
    *,
    sl_price: float | None = None,
    tp_price: float | None = None,
    trailing_distance: float | None = None,
) -> bool:
    """Update SL/TP/Trailing for a trade.

    - sl_price/tp_price: absolute price
    - trailing_distance: absolute price distance (e.g., 0.20 for 20 pips on JPY pairs)
    Returns True on 2xx.
    """
    url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}/orders"
    body: dict[str, dict] = {}
    if tp_price is not None:
        body["takeProfit"] = {"price": f"{tp_price:.3f}", "timeInForce": "GTC"}
    if sl_price is not None:
        body["stopLoss"] = {"price": f"{sl_price:.3f}", "timeInForce": "GTC"}
    if trailing_distance is not None:
        # distance is in price units
        body["trailingStopLoss"] = {"distance": f"{trailing_distance:.3f}", "timeInForce": "GTC"}
    if not body:
        return True
    try:
        r = requests.put(url, headers=HEADERS, json=body, timeout=5)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[trade_actions] update_trade_orders error for {trade_id}: {e}")
        return False
