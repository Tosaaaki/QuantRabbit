"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations
from typing import Literal, Optional

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCreate

import logging

from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACTICE_FLAG = get_secret("oanda_practice").lower() == "true"
except KeyError:
    PRACTICE_FLAG = False  # デフォルトは本番環境

try:
    HEDGING_ENABLED = get_secret("oanda_hedging_enabled").lower() == "true"
except KeyError:
    HEDGING_ENABLED = False

ENVIRONMENT = "practice" if PRACTICE_FLAG else "live"
POSITION_FILL = "OPEN_ONLY" if HEDGING_ENABLED else "DEFAULT"

api = API(access_token=TOKEN, environment=ENVIRONMENT)

if HEDGING_ENABLED:
    logging.info("[ORDER] Hedging mode enabled (positionFill=OPEN_ONLY).")


def _extract_trade_id(response: dict) -> Optional[str]:
    """
    OANDA の orderFillTransaction から tradeID を抽出する。
    tradeOpened が無い場合でも tradeReduced / tradesClosed を拾って決済扱いにする。
    """
    fill = response.get("orderFillTransaction") or {}

    opened = fill.get("tradeOpened")
    if opened and opened.get("tradeID"):
        return str(opened["tradeID"])

    reduced = fill.get("tradeReduced")
    if reduced and reduced.get("tradeID"):
        return str(reduced["tradeID"])

    for closed in fill.get("tradesClosed") or []:
        trade_id = closed.get("tradeID")
        if trade_id:
            return str(trade_id)

    # tradeReduced が複数のケース（現状 API 仕様では単一辞書）に備えて念のため走査
    for reduced_item in fill.get("tradesReduced") or []:
        trade_id = reduced_item.get("tradeID")
        if trade_id:
            return str(trade_id)

    return None


async def market_order(
    instrument: str,
    units: int,
    sl_price: float,
    tp_price: float,
    pocket: Literal["micro", "macro", "scalp"],
) -> Optional[str]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id（決済のみの fill でも tradeID を返却）
    """
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": POSITION_FILL,
            "stopLossOnFill": {"price": f"{sl_price:.3f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.3f}"},
            "clientExtensions": {"tag": f"pocket={pocket}"},
            "tradeClientExtensions": {"tag": f"pocket={pocket}"},
        }
    }

    units_to_send = units
    for attempt in range(2):
        payload = order_data.copy()
        payload["order"] = dict(order_data["order"], units=str(units_to_send))
        r = OrderCreate(accountID=ACCOUNT, data=payload)
        try:
            api.request(r)
            response = r.response
            reject = response.get("orderRejectTransaction") or response.get(
                "orderCancelTransaction"
            )
            if reject:
                reason = reject.get("rejectReason") or reject.get("reason")
                logging.error(
                    "OANDA order rejected (attempt %d) pocket=%s units=%s reason=%s",
                    attempt + 1,
                    pocket,
                    units_to_send,
                    reason,
                )
                logging.error("Reject payload: %s", reject)
                if attempt == 0 and abs(units_to_send) >= 2000:
                    units_to_send = int(units_to_send * 0.5)
                    if units_to_send == 0:
                        break
                    logging.info(
                        "Retrying order with reduced units=%s (half).", units_to_send
                    )
                    continue
                return None

            trade_id = _extract_trade_id(response)
            if trade_id:
                fill_type = response.get("orderFillTransaction", {}).get(
                    "tradeOpened"
                )
                if not fill_type:
                    logging.info(
                        "OANDA order filled by adjusting existing trade(s): %s", trade_id
                    )
                return trade_id

            logging.error(
                "OANDA order fill lacked trade identifiers (attempt %d): %s",
                attempt + 1,
                response,
            )
            return None
        except V20Error as e:
            logging.error(
                "OANDA API Error (attempt %d) pocket=%s units=%s: %s",
                attempt + 1,
                pocket,
                units_to_send,
                e,
            )
            resp = getattr(e, "response", None)
            if resp:
                logging.error("OANDA response: %s", resp)

            if attempt == 0 and abs(units_to_send) >= 2000:
                units_to_send = int(units_to_send * 0.5)
                if units_to_send == 0:
                    break
                logging.info(
                    "Retrying order with reduced units=%s (half).", units_to_send
                )
                continue
            return None
        except Exception as exc:
            logging.exception(
                "Unexpected error submitting order (attempt %d): %s",
                attempt + 1,
                exc,
            )
            return None
    return None
