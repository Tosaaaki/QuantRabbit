from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.paths import DEFAULT_ENV_LOCAL


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
        env_file: Path | None = None,
    ) -> None:
        env_values = _load_env_file(env_file or _configured_env_file())
        self.token = token or os.environ.get("QR_OANDA_TOKEN") or env_values.get("QR_OANDA_TOKEN")
        self.account_id = account_id or os.environ.get("QR_OANDA_ACCOUNT_ID") or env_values.get("QR_OANDA_ACCOUNT_ID")
        self.base_url = (
            base_url
            or os.environ.get("QR_OANDA_BASE_URL")
            or env_values.get("QR_OANDA_BASE_URL")
            or "https://api-fxtrade.oanda.com"
        ).rstrip("/")
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
        account = self._account_summary_safe(fetched_at)
        return BrokerSnapshot(
            fetched_at_utc=fetched_at,
            positions=positions,
            orders=orders,
            quotes=quotes,
            account=account,
        )

    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        payload = self.get_json(f"/v3/accounts/{self.account_id}/summary")
        return _account_summary_from_payload(payload, now_utc=now_utc or datetime.now(timezone.utc))

    def _account_summary_safe(self, now_utc: datetime) -> AccountSummary | None:
        try:
            return self.account_summary(now_utc=now_utc)
        except Exception:
            # Account summary is best-effort: snapshot must still return positions/orders/quotes
            # even if the summary call fails (e.g. transient network error). Downstream code
            # treats `account is None` as "fall back to legacy semantics".
            return None

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


def _account_summary_from_payload(payload: dict, *, now_utc: datetime) -> AccountSummary:
    account = payload.get("account") or payload
    return AccountSummary(
        nav_jpy=float(account.get("NAV") or account.get("nav") or 0.0),
        balance_jpy=float(account.get("balance") or 0.0),
        unrealized_pl_jpy=float(account.get("unrealizedPL") or 0.0),
        margin_used_jpy=float(account.get("marginUsed") or 0.0),
        margin_available_jpy=float(account.get("marginAvailable") or 0.0),
        pl_jpy=float(account.get("pl") or 0.0),
        financing_jpy=float(account.get("financing") or 0.0),
        last_transaction_id=str(account.get("lastTransactionID") or ""),
        fetched_at_utc=now_utc,
    )


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


def _configured_env_file() -> Path:
    override = os.environ.get("QR_OANDA_ENV_FILE")
    if override:
        return Path(override)
    return DEFAULT_ENV_LOCAL


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"QR_OANDA_TOKEN", "QR_OANDA_ACCOUNT_ID", "QR_OANDA_BASE_URL"}:
            continue
        values[key] = _clean_env_value(value)
    return values


def _clean_env_value(value: str) -> str:
    text = value.strip()
    if "#" in text and not (text.startswith('"') or text.startswith("'")):
        text = text.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


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

    def replace_trade_dependent_orders(self, trade_id: str, order_request: dict) -> dict:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{urllib.parse.quote(trade_id)}/orders"
        body = json.dumps(order_request).encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="PUT",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{urllib.parse.quote(trade_id)}/close"
        body = json.dumps({"units": units}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="PUT",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
