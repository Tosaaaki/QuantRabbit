from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterable

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side


class OandaReadOnlyClient:
    """Read-only OANDA client.

    vNext deliberately starts without write methods. Live execution must be added
    behind the risk gateway, not by reviving old helper scripts.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.token = token or os.environ.get("QR_OANDA_TOKEN") or os.environ.get("OANDA_TOKEN")
        self.account_id = account_id or os.environ.get("QR_OANDA_ACCOUNT_ID") or os.environ.get("OANDA_ACCOUNT_ID")
        self.base_url = (base_url or os.environ.get("QR_OANDA_BASE_URL") or "https://api-fxtrade.oanda.com").rstrip("/")
        if not self.token or not self.account_id:
            raise RuntimeError("OANDA read requires QR_OANDA_TOKEN and QR_OANDA_ACCOUNT_ID")

    def get_json(self, path: str, query: dict[str, str] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def snapshot(self, pairs: Iterable[str]) -> BrokerSnapshot:
        fetched_at = datetime.now(timezone.utc)
        positions = tuple(self._open_positions())
        orders = tuple(self._pending_orders())
        quotes = self._quotes(tuple(pairs))
        return BrokerSnapshot(fetched_at_utc=fetched_at, positions=positions, orders=orders, quotes=quotes)

    def _open_positions(self) -> list[BrokerPosition]:
        payload = self.get_json(f"/v3/accounts/{self.account_id}/openTrades")
        positions: list[BrokerPosition] = []
        for trade in payload.get("trades", []) or []:
            units_signed = int(float(trade.get("currentUnits") or 0))
            if units_signed == 0:
                continue
            positions.append(
                BrokerPosition(
                    trade_id=str(trade.get("id") or ""),
                    pair=str(trade.get("instrument") or ""),
                    side=Side.LONG if units_signed > 0 else Side.SHORT,
                    units=abs(units_signed),
                    entry_price=float(trade.get("price") or 0.0),
                    unrealized_pl_jpy=float(trade.get("unrealizedPL") or 0.0),
                    take_profit=_nested_price(trade.get("takeProfitOrder")),
                    stop_loss=_nested_price(trade.get("stopLossOrder")),
                    owner=_owner_from_trade(trade),
                    raw=trade,
                )
            )
        return positions

    def _pending_orders(self) -> list[BrokerOrder]:
        payload = self.get_json(f"/v3/accounts/{self.account_id}/pendingOrders")
        orders: list[BrokerOrder] = []
        for order in payload.get("orders", []) or []:
            orders.append(
                BrokerOrder(
                    order_id=str(order.get("id") or ""),
                    pair=order.get("instrument"),
                    order_type=str(order.get("type") or ""),
                    trade_id=order.get("tradeID"),
                    price=_optional_float(order.get("price")),
                    state=order.get("state"),
                    units=_optional_int(order.get("units")),
                    owner=_owner_from_trade(order),
                    raw=order,
                )
            )
        return orders

    def _quotes(self, pairs: tuple[str, ...]) -> dict[str, Quote]:
        if not pairs:
            return {}
        payload = self.get_json(
            f"/v3/accounts/{self.account_id}/pricing",
            {"instruments": ",".join(sorted(set(pairs)))},
        )
        quotes: dict[str, Quote] = {}
        for price in payload.get("prices", []) or []:
            pair = str(price.get("instrument") or "")
            bids = price.get("bids") or []
            asks = price.get("asks") or []
            if not pair or not bids or not asks:
                continue
            ts = _parse_oanda_time(price.get("time")) or datetime.now(timezone.utc)
            quotes[pair] = Quote(pair=pair, bid=float(bids[0]["price"]), ask=float(asks[0]["price"]), timestamp_utc=ts)
        return quotes


def _nested_price(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    return _optional_float(payload.get("price"))


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _owner_from_trade(trade: dict) -> Owner:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = trade.get(key) or {}
        tag = str(ext.get("tag") or "").strip().lower()
        if tag == Owner.TRADER.value:
            return Owner.TRADER
        if tag == Owner.MANUAL.value:
            return Owner.MANUAL
        if tag:
            return Owner.EXTERNAL
    return Owner.UNKNOWN


def _parse_oanda_time(value: object) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text = f"{core}+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


class OandaExecutionClient(OandaReadOnlyClient):
    """Narrow OANDA execution client used only behind the risk gateway."""

    def post_order_json(self, order_request: dict) -> dict:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        body = json.dumps({"order": order_request}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def cancel_order(self, order_id: str) -> dict:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{urllib.parse.quote(order_id)}/cancel"
        req = urllib.request.Request(
            url,
            data=b"",
            method="PUT",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
