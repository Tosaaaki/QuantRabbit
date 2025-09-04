"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations
from typing import Literal

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCreate

from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACTICE_FLAG = get_secret("oanda_practice").lower() == "true"
except Exception:
    PRACTICE_FLAG = True  # デフォルトは practice を安全側に

ENVIRONMENT = "practice" if PRACTICE_FLAG else "live"

api = API(access_token=TOKEN, environment=ENVIRONMENT)


async def market_order(
    instrument: str,
    units: int,
    sl_price: float,
    tp_price: float,
    pocket: Literal["micro", "macro"],
    *,
    strategy: str | None = None,
    macro_regime: str | None = None,
    micro_regime: str | None = None,
) -> str:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id
    """
    # Encode strategy/regime metadata for later learning (PositionManagerが回収)
    comment_parts = []
    if strategy:
        comment_parts.append(f"strategy={strategy}")
    if macro_regime:
        comment_parts.append(f"macro={macro_regime}")
    if micro_regime:
        comment_parts.append(f"micro={micro_regime}")
    comment = "|".join(comment_parts) if comment_parts else ""

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price:.3f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.3f}"},
            "clientExtensions": {"tag": f"pocket={pocket}", "comment": comment},
        }
    }

    r = OrderCreate(accountID=ACCOUNT, data=order_data)
    try:
        api.request(r)
        response = r.response
        trade_id = (
            response.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
        )
        return trade_id
    except V20Error as e:
        print(f"OANDA API Error in market_order: {e}")
        return None
