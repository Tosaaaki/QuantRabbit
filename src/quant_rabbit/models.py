from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @classmethod
    def parse(cls, value: str) -> "Side":
        upper = value.strip().upper()
        if upper not in cls.__members__:
            raise ValueError(f"side must be LONG or SHORT, got {value!r}")
        return cls[upper]


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_ENTRY = "STOP-ENTRY"

    @classmethod
    def parse(cls, value: str) -> "OrderType":
        upper = value.strip().upper()
        if upper == "STOP":
            upper = "STOP-ENTRY"
        for item in cls:
            if item.value == upper:
                return item
        raise ValueError(f"unsupported order type: {value!r}")


class TradeMethod(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    RANGE_ROTATION = "RANGE_ROTATION"
    BREAKOUT_FAILURE = "BREAKOUT_FAILURE"
    EVENT_RISK = "EVENT_RISK"
    POSITION_MANAGEMENT = "POSITION_MANAGEMENT"

    @classmethod
    def parse(cls, value: str) -> "TradeMethod":
        upper = value.strip().upper().replace("-", "_").replace(" ", "_")
        for item in cls:
            if item.value == upper:
                return item
        raise ValueError(f"unsupported trade method: {value!r}")


class Owner(str, Enum):
    TRADER = "trader"
    MANUAL = "manual"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Quote:
    pair: str
    bid: float
    ask: float
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class BrokerPosition:
    trade_id: str
    pair: str
    side: Side
    units: int
    entry_price: float
    unrealized_pl_jpy: float = 0.0
    take_profit: float | None = None
    stop_loss: float | None = None
    owner: Owner = Owner.UNKNOWN
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    pair: str | None
    order_type: str
    trade_id: str | None = None
    price: float | None = None
    state: str | None = None
    units: int | None = None
    owner: Owner = Owner.UNKNOWN
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountSummary:
    """OANDA `/v3/accounts/{id}/summary` snapshot in JPY-denominated account currency.

    `nav_jpy` is the canonical equity-after-PL value used to compute current equity and
    today's start balance. `balance_jpy` is cash before unrealized PnL; on a new campaign
    day with zero realized PnL it is also today's start balance.
    """

    nav_jpy: float
    balance_jpy: float
    unrealized_pl_jpy: float = 0.0
    margin_used_jpy: float = 0.0
    margin_available_jpy: float = 0.0
    pl_jpy: float = 0.0
    financing_jpy: float = 0.0
    last_transaction_id: str = ""
    # OANDA v20 account-level hedge mode. When true, opposite-side same-pair
    # orders can open a separate trade instead of reducing the existing side.
    hedging_enabled: bool = False
    fetched_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class BrokerSnapshot:
    fetched_at_utc: datetime
    positions: tuple[BrokerPosition, ...] = ()
    orders: tuple[BrokerOrder, ...] = ()
    quotes: dict[str, Quote] = field(default_factory=dict)
    account: AccountSummary | None = None
    # Account-home conversion factors keyed by quote currency, e.g. {"USD": 157.0}
    # from OANDA pricing `homeConversions`. These are broker-provided factors
    # for converting quote-currency P/L into the account home currency and avoid
    # treating an unchanged USD_JPY quote as stale conversion evidence.
    home_conversions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketContext:
    regime: str
    narrative: str
    chart_story: str
    method: TradeMethod
    invalidation: str
    event_risk: str = ""
    session: str = ""


@dataclass(frozen=True)
class OrderIntent:
    pair: str
    side: Side
    order_type: OrderType
    units: int
    tp: float
    sl: float
    thesis: str
    owner: Owner = Owner.TRADER
    entry: float | None = None
    reason: str = ""
    market_context: MarketContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskIssue:
    code: str
    message: str
    severity: str = "BLOCK"


@dataclass(frozen=True)
class RiskMetrics:
    entry_price: float
    loss_pips: float
    reward_pips: float
    risk_jpy: float
    reward_jpy: float
    reward_risk: float
    spread_pips: float
    jpy_per_pip: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    metrics: RiskMetrics | None
    issues: tuple[RiskIssue, ...]

    @property
    def block_reasons(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == "BLOCK"]
