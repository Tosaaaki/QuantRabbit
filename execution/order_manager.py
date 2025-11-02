"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行注文を発注。SL/TP はリアルタイム価格に合わせて補正する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.pricing import PricingInfo

from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACTICE_FLAG = get_secret("oanda_practice").lower() == "true"
except KeyError:
    PRACTICE_FLAG = False  # デフォルトは本番環境

ENVIRONMENT = "practice" if PRACTICE_FLAG else "live"

api = API(access_token=TOKEN, environment=ENVIRONMENT)

_MIN_BUFFER = 0.001  # 0.1 pip safety margin


@dataclass
class _TopOfBook:
    bid: float
    ask: float
    timestamp: Optional[str]


def _fetch_top_of_book(instrument: str) -> _TopOfBook:
    """Return the latest closeout bid/ask used for SL/TP anchoring."""
    r = PricingInfo(accountID=ACCOUNT, params={"instruments": instrument})
    api.request(r)
    prices = r.response.get("prices", [])
    if not prices:
        raise RuntimeError("OANDA pricing response returned no prices")
    price = prices[0]
    return _TopOfBook(
        bid=float(price["closeoutBid"]),
        ask=float(price["closeoutAsk"]),
        timestamp=price.get("time"),
    )


def _compose_client_extensions(
    pocket: str,
    *,
    strategy: Optional[str] = None,
    macro_regime: Optional[str] = None,
    micro_regime: Optional[str] = None,
    focus_tag: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    extensions: Dict[str, str] = {"tag": f"pocket={pocket}"}
    parts: list[str] = []
    if strategy:
        parts.append(f"strategy={strategy}")
    if macro_regime:
        parts.append(f"macro={macro_regime}")
    if micro_regime:
        parts.append(f"micro={micro_regime}")
    if focus_tag:
        parts.append(f"focus={focus_tag}")
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
    if parts:
        comment = "|".join(str(p) for p in parts)
        # OANDA client comment max length is 128 chars.
        extensions["comment"] = comment[:128]
    return extensions


def _adjust_levels_for_entry(
    *,
    is_buy: bool,
    entry_price: float,
    sl_price: float,
    tp_price: float,
) -> tuple[float, float]:
    """Ensure SL/TP remain on the correct side of entry with a small safety buffer."""
    buffer = _MIN_BUFFER
    if is_buy:
        if sl_price >= entry_price - buffer:
            sl_price = entry_price - buffer
        if tp_price <= entry_price + buffer:
            tp_price = entry_price + buffer
    else:
        if sl_price <= entry_price + buffer:
            sl_price = entry_price + buffer
        if tp_price >= entry_price - buffer:
            tp_price = entry_price - buffer
    return round(sl_price, 3), round(tp_price, 3)


def _anchor_levels_to_entry(
    *,
    price_anchor: Optional[float],
    entry_price: float,
    sl_price: float,
    tp_price: float,
) -> tuple[float, float]:
    if price_anchor is None:
        return sl_price, tp_price

    sl_offset = sl_price - price_anchor
    tp_offset = tp_price - price_anchor
    anchored_sl = entry_price + sl_offset
    anchored_tp = entry_price + tp_offset
    return anchored_sl, anchored_tp


async def market_order(
    instrument: str,
    units: int,
    sl_price: float,
    tp_price: float,
    pocket: Literal["micro", "macro", "scalp"],
    *,
    price_anchor: Optional[float] = None,
    strategy: Optional[str] = None,
    macro_regime: Optional[str] = None,
    micro_regime: Optional[str] = None,
    focus_tag: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Submit a market order and attach SL/TP.

    The strategy-provided SL/TP values can be anchored to the latest bid/ask so they
    remain on the correct side of the market even if price moved after the signal.
    """
    is_buy = units > 0
    result: Dict[str, Any] = {
        "success": False,
        "trade_id": None,
        "order_id": None,
    }

    try:
        tob = _fetch_top_of_book(instrument)
    except Exception as exc:
        result["error"] = f"pricing_fetch_failed: {exc}"
        return result

    entry_price = tob.ask if is_buy else tob.bid

    # Anchor SL/TP around the actual entry quote to avoid immediate cancellations.
    anchored_sl, anchored_tp = _anchor_levels_to_entry(
        price_anchor=price_anchor,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
    )
    adj_sl, adj_tp = _adjust_levels_for_entry(
        is_buy=is_buy,
        entry_price=entry_price,
        sl_price=anchored_sl,
        tp_price=anchored_tp,
    )

    client_extensions = _compose_client_extensions(
        pocket,
        strategy=strategy,
        macro_regime=macro_regime,
        micro_regime=micro_regime,
        focus_tag=focus_tag,
        metadata=metadata,
    )

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": client_extensions,
            "stopLossOnFill": {
                "price": f"{adj_sl:.3f}",
                "timeInForce": "GTC",
            },
            "takeProfitOnFill": {
                "price": f"{adj_tp:.3f}",
                "timeInForce": "GTC",
            },
        }
    }

    r = OrderCreate(accountID=ACCOUNT, data=order_data)
    try:
        api.request(r)
    except V20Error as exc:
        result.update(
            {
                "error": f"OANDA API error: {exc}",
                "entry_price": entry_price,
                "sl": adj_sl,
                "tp": adj_tp,
                "bid": tob.bid,
                "ask": tob.ask,
            }
        )
        return result

    response = r.response or {}
    order_create_tx = response.get("orderCreateTransaction", {})
    order_fill_tx = response.get("orderFillTransaction")
    order_cancel_tx = response.get("orderCancelTransaction")
    order_reject_tx = response.get("orderRejectTransaction")

    order_id = order_create_tx.get("id")
    trade_id = None
    if order_fill_tx:
        trade_opened = order_fill_tx.get("tradeOpened", {})
        trade_id = trade_opened.get("tradeID") or order_fill_tx.get("id")

    result.update(
        {
            "order_id": order_id,
            "trade_id": trade_id,
            "entry_price": entry_price,
            "sl": adj_sl,
            "tp": adj_tp,
            "bid": tob.bid,
            "ask": tob.ask,
            "tob_time": tob.timestamp,
            "response": response,
        }
    )

    if order_fill_tx and trade_id:
        result["success"] = True
        return result

    # Capture cancellation/rejection reason for diagnostics.
    reason = None
    if order_cancel_tx:
        reason = order_cancel_tx.get("reason")
    elif order_reject_tx:
        reason = order_reject_tx.get("rejectReason")
    else:
        reason = "unknown_outcome"

    result["error"] = reason
    return result
