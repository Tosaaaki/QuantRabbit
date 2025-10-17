"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Any, Dict

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCreate

from utils.secrets import get_secret

_api: API | None = None
_account_id: str | None = None
_SYSTEM_VERSION = None


def _get_system_version() -> str:
    global _SYSTEM_VERSION
    if _SYSTEM_VERSION is not None:
        return _SYSTEM_VERSION
    version = os.environ.get("SYSTEM_VERSION")
    if not version:
        try:
            version = get_secret("system_version")
        except Exception:
            version = "V2"
    _SYSTEM_VERSION = version
    return _SYSTEM_VERSION


def _ensure_api() -> tuple[API | None, str | None]:
    """API クライアントを遅延初期化。未設定なら (None, None)。"""
    global _api, _account_id
    if _api is not None and _account_id is not None:
        return _api, _account_id
    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")
        try:
            practice = get_secret("oanda_practice").lower() == "true"
        except Exception:
            practice = True
        env = "practice" if practice else "live"
        _api = API(access_token=token, environment=env)
        _account_id = account
        return _api, _account_id
    except Exception as exc:
        logging.error("[order_manager] Failed to initialize OANDA API: %s", exc)
        return None, None


Pocket = Literal["micro", "macro", "scalp"]


async def market_order(
    instrument: str,
    units: int,
    sl_price: float,
    tp_price: float,
    pocket: Pocket,
    *,
    strategy: str | None = None,
    macro_regime: str | None = None,
    micro_regime: str | None = None,
) -> Dict[str, Any]:
    """Submit a market order and return rich result dict."""
    comment_parts = []
    if strategy:
        comment_parts.append(f"strategy={strategy}")
    if macro_regime:
        comment_parts.append(f"macro={macro_regime}")
    if micro_regime:
        comment_parts.append(f"micro={micro_regime}")
    version = _get_system_version()
    if version:
        comment_parts.append(f"ver={version}")
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

    api, account = _ensure_api()
    if not api or not account:
        logging.error("[order_manager] OANDA credentials missing; order skipped")
        return {"success": False, "trade_id": None, "error": "missing_credentials"}

    r = OrderCreate(accountID=account, data=order_data)
    try:
        api.request(r)
        response = r.response
        fill_tx = response.get("orderFillTransaction", {}) or {}
        trade_opened = fill_tx.get("tradeOpened") or {}
        trade_id = (
            trade_opened.get("tradeID")
            or trade_opened.get("id")
            or fill_tx.get("tradeID")
            or fill_tx.get("id")
        )
        order_id = (
            fill_tx.get("orderID")
            or response.get("orderCreateTransaction", {}).get("id")
            or fill_tx.get("id")
        )
        success = bool(fill_tx) or bool(trade_id)
        logging.info(
            "[ORDER_SUCCESS] instrument=%s units=%s trade_id=%s order_id=%s",
            instrument,
            units,
            trade_id,
            order_id,
        )
        return {
            "success": success,
            "trade_id": trade_id,
            "order_id": order_id,
            "raw": response,
        }
    except V20Error as e:
        err_payload = getattr(e, "response", None)
        logging.error(
            "[ORDER_ERROR] code=%s message=%s", getattr(e, "code", None), getattr(e, "msg", e)
        )
        if err_payload:
            logging.error("[ORDER_ERROR_PAYLOAD] %s", err_payload)
        return {
            "success": False,
            "trade_id": None,
            "error": str(e),
            "error_code": getattr(e, "code", None),
            "raw": err_payload,
        }
    except Exception as exc:
        logging.exception("[ORDER_EXCEPTION] %s", exc)
        return {"success": False, "trade_id": None, "error": str(exc)}
