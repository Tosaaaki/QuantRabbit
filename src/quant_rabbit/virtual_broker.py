"""Virtual broker: OANDA-mechanics paper account priced by REAL quotes.

Operator directive (2026-07-19): a virtual environment identical to
reality where the duty agent can trade at will.  This module is the
broker core; feeds (live polling or historical replay) supply quotes.

Honesty properties:
  * Fills happen ONLY at quotes actually supplied by the feed — market
    orders fill at the current real ask/bid; limit/TP/SL fill when a
    supplied quote touches the level, at the level (or the quote when
    it gapped past, whichever is worse for the trader).  No synthesis.
  * Accounting mirrors OANDA: hedge netting (margin on the larger side
    per instrument), leverage cap, margin closeout at 100% usage
    liquidating everything at current quotes.
  * Every action and fill is written to a hash-chained append-only
    ledger with the exact quote that caused it.
  * The broker never talks to the real broker.  It cannot place real
    orders by construction.

Non-JPY-quote pairs convert P&L at the latest USD_JPY mid supplied to
the broker (declared approximation, logged per fill).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

UTC = timezone.utc

LEVERAGE_DEFAULT = 25.0
CLOSEOUT_USAGE = 1.0
MAX_CONVERSION_QUOTE_AGE_S = 90.0
SNAPSHOT_SCHEMA = "QR_VIRTUAL_BROKER_SNAPSHOT_V2"
_SNAPSHOT_KEYS = {
    "schema",
    "balance_jpy",
    "seq",
    "positions",
    "orders",
    "quote_seq",
    "last_quotes",
    "last_quote_sequences",
    "last_quote_watermarks",
    "quote_history",
    "feed_cursor",
    "ledger_tip_sha",
}


class VirtualBrokerError(ValueError):
    """Contract violation; callers must fail closed."""


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _round_price(pair: str, price: float) -> float:
    """Instrument price precision (JPY quotes: 3 dp, others: 5 dp) —
    mirrors broker tick precision and kills float-epsilon artifacts."""

    return round(price, 3 if pair.endswith("JPY") else 5)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise VirtualBrokerError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _finite_number(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise VirtualBrokerError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VirtualBrokerError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise VirtualBrokerError(f"{name} must be finite")
    if positive and number <= 0:
        raise VirtualBrokerError(f"{name} must be positive")
    if non_negative and number < 0:
        raise VirtualBrokerError(f"{name} must be non-negative")
    return number


def _validate_finite_tree(value: Any, path: str = "payload") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise VirtualBrokerError(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")
        return
    raise VirtualBrokerError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _validate_pair(pair: str) -> None:
    parts = pair.split("_")
    if len(parts) != 2 or any(
        len(part) != 3 or not part.isalpha() or not part.isupper() for part in parts
    ):
        raise VirtualBrokerError(f"invalid pair: {pair}")


@dataclass
class VBPosition:
    trade_id: str
    pair: str
    side: str  # LONG / SHORT
    units: float
    entry_price: float
    opened_ts: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


@dataclass
class VBOrder:
    order_id: str
    pair: str
    side: str
    units: float
    limit_price: float
    tp_pips: Optional[float] = None
    sl_pips: Optional[float] = None
    kind: str = "LIMIT"  # LIMIT (at level or better) / STOP (breakout trigger)


@dataclass
class VirtualBroker:
    ledger_path: Path
    balance_jpy: float = 200_000.0
    fast_ledger: bool = False  # flush without fsync (lab runs)
    slippage_pips: float = 0.0  # stress: extra pips against the trader per fill
    financing_pips_per_day: float = 0.0  # holding cost, pro-rata vs opened_ts
    leverage: float = LEVERAGE_DEFAULT
    positions: dict[str, VBPosition] = field(default_factory=dict)
    orders: dict[str, VBOrder] = field(default_factory=dict)
    last_quotes: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    _last_quote_sequences: dict[str, int] = field(default_factory=dict, repr=False)
    _last_quote_watermarks: dict[str, int] = field(default_factory=dict, repr=False)
    _quote_history: dict[str, list[tuple[float, float, str, int]]] = field(
        default_factory=dict, repr=False
    )
    _quote_seq: int = 0
    feed_cursor: Optional[dict[str, Any]] = field(default=None, repr=False)
    _entry_admission: Optional[
        Callable[[str, str, Optional[str]], Optional[dict[str, Any]]]
    ] = field(default=None, init=False, repr=False)
    _seq: int = 0
    _prev_sha: str = "0" * 64

    def __post_init__(self) -> None:
        self.balance_jpy = _finite_number("balance_jpy", self.balance_jpy)
        self.slippage_pips = _finite_number(
            "slippage_pips", self.slippage_pips, non_negative=True
        )
        self.financing_pips_per_day = _finite_number(
            "financing_pips_per_day", self.financing_pips_per_day, non_negative=True
        )
        self.leverage = _finite_number("leverage", self.leverage, positive=True)
        if self.ledger_path.exists():
            expected_prev = "0" * 64
            with self.ledger_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise VirtualBrokerError(
                            f"blank ledger record at line {line_number}"
                        )
                    try:
                        record = _strict_json_loads(line)
                    except (json.JSONDecodeError, VirtualBrokerError) as exc:
                        raise VirtualBrokerError(
                            f"invalid ledger JSON at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise VirtualBrokerError(
                            f"invalid ledger record at line {line_number}"
                        )
                    supplied_sha = record.get("sha")
                    body = {key: value for key, value in record.items() if key != "sha"}
                    if set(body) != {"ts_utc", "event", "payload", "prev_sha"}:
                        raise VirtualBrokerError(
                            f"invalid ledger schema at line {line_number}"
                        )
                    _validate_finite_tree(body, f"ledger[{line_number}]")
                    if body["prev_sha"] != expected_prev:
                        raise VirtualBrokerError(
                            f"ledger prev_sha mismatch at line {line_number}"
                        )
                    if not isinstance(supplied_sha, str) or supplied_sha != _sha(body):
                        raise VirtualBrokerError(
                            f"ledger sha mismatch at line {line_number}"
                        )
                    expected_prev = supplied_sha
            self._prev_sha = expected_prev
        self._handle = self.ledger_path.open("a", encoding="utf-8")

    # ---- ledger ----------------------------------------------------------
    def _log(self, event: str, payload: dict[str, Any]) -> None:
        _validate_finite_tree(payload)
        body = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "prev_sha": self._prev_sha,
        }
        record = {**body, "sha": _sha(body)}
        self._handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self._handle.flush()
        if not self.fast_ledger:
            os.fsync(self._handle.fileno())
        self._prev_sha = record["sha"]

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq:06d}"

    # ---- conversion / accounting ----------------------------------------
    @staticmethod
    def _quote_phase(ts: str) -> Optional[str]:
        if "#" not in ts:
            return None
        phase = ts.rsplit("#", 1)[1]
        return phase or None

    def _quote_as_of(
        self, pair: str, as_of_sequence: Optional[int] = None
    ) -> tuple[float, float, str, int]:
        """Return the newest observed quote no later than ``as_of_sequence``.

        Fill conversion is point-in-time evidence.  A later quote from another
        pair, even in the same replay epoch, must never reprice an earlier fill.
        The bounded history covers asynchronous live polling and all four
        replay phases without allowing a future quote to leak backwards.
        """

        history = self._quote_history.get(pair, [])
        if as_of_sequence is None:
            if history:
                return history[-1]
        else:
            for quote in reversed(history):
                if quote[3] <= as_of_sequence:
                    return quote
        if pair not in self.last_quotes:
            raise VirtualBrokerError(f"quote required for conversion pair {pair}")
        last_sequence = self._last_quote_sequences.get(pair)
        if last_sequence is not None and (
            as_of_sequence is None or last_sequence <= as_of_sequence
        ):
            bid, ask, ts = self.last_quotes[pair]
            return bid, ask, ts, last_sequence
        raise VirtualBrokerError(f"no {pair} conversion quote at or before fill quote")

    def _conversion_evidence(
        self,
        pair: str,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> dict[str, Any]:
        """Bind the exact quote(s) used to convert quote-currency P/L to JPY."""

        _validate_pair(pair)
        parts = pair.split("_")
        quote_ccy = parts[1]
        sources: list[tuple[str, tuple[float, float, str, int]]] = []
        if quote_ccy == "JPY":
            rate = 1.0
        elif quote_ccy == "USD":
            usd_jpy = self._quote_as_of("USD_JPY", as_of_sequence)
            sources.append(("USD_JPY", usd_jpy))
            rate = (usd_jpy[0] + usd_jpy[1]) / 2.0
        else:
            direct_pair = f"{quote_ccy}_JPY"
            try:
                direct = self._quote_as_of(direct_pair, as_of_sequence)
            except VirtualBrokerError:
                direct = None
            if direct is not None:
                sources.append((direct_pair, direct))
                rate = (direct[0] + direct[1]) / 2.0
            else:
                usd_jpy = self._quote_as_of("USD_JPY", as_of_sequence)
                via_pair = f"USD_{quote_ccy}"
                via_usd = self._quote_as_of(via_pair, as_of_sequence)
                via_mid = (via_usd[0] + via_usd[1]) / 2.0
                if via_mid <= 0:
                    raise VirtualBrokerError(f"invalid conversion quote for {via_pair}")
                sources.extend((("USD_JPY", usd_jpy), (via_pair, via_usd)))
                rate = ((usd_jpy[0] + usd_jpy[1]) / 2.0) / via_mid
        reference_epoch = (
            self._ts_epoch(reference_ts) if reference_ts is not None else None
        )
        for source_pair, source_quote in sources:
            source_ts = source_quote[2]
            if source_ts == reference_ts:
                continue
            source_epoch = self._ts_epoch(source_ts)
            if reference_ts is not None and (
                reference_epoch is None or source_epoch is None
            ):
                raise VirtualBrokerError(
                    f"unparseable conversion freshness timestamp for {source_pair}"
                )
            if reference_epoch is not None and source_epoch is not None:
                age_s = reference_epoch - source_epoch
                if abs(age_s) > MAX_CONVERSION_QUOTE_AGE_S:
                    raise VirtualBrokerError(
                        f"stale conversion quote for {source_pair}: {age_s:.3f}s"
                    )
        rate = _finite_number("conversion rate", rate, positive=True)
        return {
            "quote_currency": quote_ccy,
            "rate_jpy_per_quote_unit": rate,
            "as_of_quote_sequence": as_of_sequence,
            "source_quote_sequences": [quote[3] for _, quote in sources],
            "source_quotes": [
                {
                    "pair": source_pair,
                    "bid": quote[0],
                    "ask": quote[1],
                    "ts": quote[2],
                    "phase": self._quote_phase(quote[2]),
                }
                for source_pair, quote in sources
            ],
        }

    def _jpy_per_quote_unit(
        self,
        pair: str,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> float:
        """JPY value derived only from quotes observed by the requested time."""

        return float(
            self._conversion_evidence(pair, as_of_sequence, reference_ts)[
                "rate_jpy_per_quote_unit"
            ]
        )

    @staticmethod
    def _ts_epoch(ts: str) -> Optional[float]:
        try:
            return datetime.fromisoformat(ts.split("#")[0]).timestamp()
        except Exception:
            return None

    def _financing_jpy(
        self, pos: VBPosition, exit_ts: str, conversion_rate: Optional[float] = None
    ) -> float:
        if self.financing_pips_per_day <= 0:
            return 0.0
        t0 = self._ts_epoch(pos.opened_ts)
        t1 = self._ts_epoch(exit_ts)
        if t0 is None or t1 is None:
            raise VirtualBrokerError("financing requires parseable position timestamps")
        if t1 < t0:
            raise VirtualBrokerError("financing timestamp precedes position open")
        if t1 == t0:
            return 0.0
        days = (t1 - t0) / 86400.0
        rate = (
            self._jpy_per_quote_unit(pos.pair)
            if conversion_rate is None
            else conversion_rate
        )
        return self.financing_pips_per_day * _pip(pos.pair) * pos.units * rate * days

    def _position_pl_jpy(
        self,
        pos: VBPosition,
        bid: float,
        ask: float,
        *,
        as_of_sequence: Optional[int],
        reference_ts: str,
    ) -> tuple[float, float]:
        mark = bid if pos.side == "LONG" else ask
        diff = (
            (mark - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - mark)
        )
        rate = self._jpy_per_quote_unit(pos.pair, as_of_sequence, reference_ts)
        return diff * pos.units * rate, rate

    def _exposure_jpy(
        self,
        pair: str,
        units: float,
        price: float,
        *,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> float:
        if pair.endswith("JPY"):
            return units * price
        # base-currency exposure valued via quote conversion
        return (
            units * price * self._jpy_per_quote_unit(pair, as_of_sequence, reference_ts)
        )

    def _adverse_exit_price(self, pair: str, side: str, price: float) -> float:
        slip = self.slippage_pips * _pip(pair)
        exit_price = _round_price(
            pair, price - slip if side == "LONG" else price + slip
        )
        if exit_price <= 0:
            raise VirtualBrokerError("slippage produced a non-positive exit price")
        return exit_price

    def account(self) -> dict[str, Any]:
        equity = self.balance_jpy
        margin = 0.0
        accrued_financing = 0.0
        by_pair: dict[str, dict[str, float]] = {}
        for pos in self.positions.values():
            q = self.last_quotes.get(pos.pair)
            if q is None:
                raise VirtualBrokerError(f"no quote for open position pair {pos.pair}")
            watermark = self._last_quote_watermarks.get(pos.pair)
            if watermark is None:
                raise VirtualBrokerError(
                    f"no accounting watermark for open position pair {pos.pair}"
                )
            unrealized, conversion_rate = self._position_pl_jpy(
                pos,
                q[0],
                q[1],
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            financing = self._financing_jpy(pos, q[2], conversion_rate)
            equity += unrealized - financing
            accrued_financing += financing
            side_units = by_pair.setdefault(pos.pair, {"LONG": 0.0, "SHORT": 0.0})
            side_units[pos.side] += pos.units
        for pair, sides in by_pair.items():
            q = self.last_quotes[pair]
            mid = (q[0] + q[1]) / 2.0
            margin += (
                self._exposure_jpy(
                    pair,
                    max(sides["LONG"], sides["SHORT"]),
                    mid,
                    as_of_sequence=self._last_quote_watermarks[pair],
                    reference_ts=q[2],
                )
                / self.leverage
            )
        usage = margin / equity if equity > 0 else 999.0
        return {
            "balance_jpy": round(self.balance_jpy, 2),
            "equity_jpy": round(equity, 2),
            "margin_used_jpy": round(margin, 2),
            "margin_usage": round(usage, 6),
            "accrued_financing_jpy": round(accrued_financing, 2),
            "open_positions": len(self.positions),
            "resting_orders": len(self.orders),
        }

    def _margin_headroom_ok(self, pair: str, side: str, units: float) -> bool:
        """OANDA-faithful: refuse orders whose margin would not fit."""

        acct = self.account()
        q = self.last_quotes.get(pair)
        if q is None:
            return False
        mid = (q[0] + q[1]) / 2.0
        long_u = sum(
            p.units
            for p in self.positions.values()
            if p.pair == pair and p.side == "LONG"
        )
        short_u = sum(
            p.units
            for p in self.positions.values()
            if p.pair == pair and p.side == "SHORT"
        )
        if side == "LONG":
            long_u += units
        else:
            short_u += units
        watermark = self._last_quote_watermarks.get(pair)
        if watermark is None:
            raise VirtualBrokerError(f"no accounting watermark for {pair}")
        new_pair_margin = (
            self._exposure_jpy(
                pair,
                max(long_u, short_u),
                mid,
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            / self.leverage
        )
        old_pair_margin = (
            self._exposure_jpy(
                pair,
                max(
                    long_u - (units if side == "LONG" else 0),
                    short_u - (units if side == "SHORT" else 0),
                ),
                mid,
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            / self.leverage
        )
        new_total = acct["margin_used_jpy"] - old_pair_margin + new_pair_margin
        return new_total <= acct["equity_jpy"]

    def _entry_admission_rejection(
        self, pair: str, side: str, order_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Run an installed DOJO admission policy at the actual fill boundary.

        The base virtual broker has no strategy policy.  A provenance owner
        registry may install one, and a malformed decision fails closed before
        either a market or resting order can become a position.
        """

        if self._entry_admission is None:
            return None
        rejection = self._entry_admission(pair, side, order_id)
        if rejection is None:
            return None
        expected_keys = {
            "scope",
            "reason",
            "active_pair_positions",
            "max_concurrent_per_pair",
            "active_global_positions",
            "global_max_concurrent",
        }
        if not isinstance(rejection, dict) or set(rejection) != expected_keys:
            raise VirtualBrokerError(
                "entry admission policy returned malformed evidence"
            )
        if rejection["scope"] not in {"PAIR", "GLOBAL"} or rejection["reason"] not in {
            "OWNER_PAIR_CONCURRENCY_CAP_REACHED",
            "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED",
        }:
            raise VirtualBrokerError(
                "entry admission policy returned an invalid reason"
            )
        for key in ("active_pair_positions", "active_global_positions"):
            value = rejection[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VirtualBrokerError(
                    "entry admission policy returned an invalid count"
                )
        for key in ("max_concurrent_per_pair", "global_max_concurrent"):
            value = rejection[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VirtualBrokerError(
                    "entry admission policy returned an invalid cap"
                )
        return rejection

    # ---- agent actions ---------------------------------------------------
    def market_order(
        self,
        pair: str,
        side: str,
        units: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        q = self.last_quotes.get(pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pair}; cannot fill")
        rejection = self._entry_admission_rejection(pair, side)
        if rejection is not None:
            self._log(
                "ORDER_REJECTED_CONCURRENCY_CAP",
                {
                    "pair": pair,
                    "side": side,
                    "units": units,
                    "admission": rejection,
                },
            )
            raise VirtualBrokerError("owner concurrency cap reached for market order")
        if not self._margin_headroom_ok(pair, side, units):
            self._log(
                "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                {"pair": pair, "side": side, "units": units},
            )
            raise VirtualBrokerError("insufficient margin for market order")
        bid, ask, ts = q
        quote_sequence = self._last_quote_watermarks.get(pair)
        if quote_sequence is None:
            raise VirtualBrokerError(f"no accounting watermark for {pair}")
        conversion = self._conversion_evidence(pair, quote_sequence, ts)
        pip = _pip(pair)
        slip = self.slippage_pips * pip
        entry = (ask + slip) if side == "LONG" else (bid - slip)
        entry = _round_price(pair, entry)
        if entry <= 0:
            raise VirtualBrokerError("slippage produced a non-positive entry price")
        tp = (
            _round_price(
                pair, entry + tp_pips * pip if side == "LONG" else entry - tp_pips * pip
            )
            if tp_pips
            else None
        )
        sl = (
            _round_price(
                pair, entry - sl_pips * pip if side == "LONG" else entry + sl_pips * pip
            )
            if sl_pips
            else None
        )
        trade_id = self._next_id("T")
        self.positions[trade_id] = VBPosition(
            trade_id=trade_id,
            pair=pair,
            side=side,
            units=units,
            entry_price=entry,
            opened_ts=ts,
            tp_price=tp,
            sl_price=sl,
        )
        self._log(
            "FILL_MARKET",
            {
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "units": units,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            },
        )
        self._enforce_margin_after_action()
        return trade_id

    def limit_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        price = _finite_number("price", price, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        order_id = self._next_id("O")
        self.orders[order_id] = VBOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            units=units,
            limit_price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
        self._log(
            "ORDER_LIMIT",
            {
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "units": units,
                "price": price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
            },
        )
        return order_id

    def stop_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        """Breakout entry: LONG fills once the real ask reaches price (at
        the level or WORSE when gapped); SHORT once the real bid does."""

        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        price = _finite_number("price", price, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        order_id = self._next_id("O")
        self.orders[order_id] = VBOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            units=units,
            limit_price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            kind="STOP",
        )
        self._log(
            "ORDER_STOP",
            {
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "units": units,
                "price": price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
            },
        )
        return order_id

    def cancel_order(self, order_id: str) -> None:
        if order_id not in self.orders:
            raise VirtualBrokerError(f"unknown order: {order_id}")
        del self.orders[order_id]
        self._log("ORDER_CANCEL", {"order_id": order_id})

    def close_trade(self, trade_id: str, units: Optional[float] = None) -> float:
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        q = self.last_quotes.get(pos.pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pos.pair}; cannot close")
        bid, ask, ts = q
        quote_sequence = self._last_quote_watermarks.get(pos.pair)
        if quote_sequence is None:
            raise VirtualBrokerError(f"no accounting watermark for {pos.pair}")
        conversion = self._conversion_evidence(pos.pair, quote_sequence, ts)
        conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
        requested_units = (
            pos.units
            if units is None
            else _finite_number("close units", units, positive=True)
        )
        close_units = min(requested_units, pos.units)
        price = self._adverse_exit_price(
            pos.pair, pos.side, bid if pos.side == "LONG" else ask
        )
        diff = (
            (price - pos.entry_price)
            if pos.side == "LONG"
            else (pos.entry_price - price)
        )
        gross_pl = diff * close_units * conversion_rate
        financing = self._financing_jpy(pos, ts, conversion_rate) * (
            close_units / pos.units
        )
        pl = gross_pl - financing
        self.balance_jpy += pl
        if close_units >= pos.units:
            del self.positions[trade_id]
        else:
            pos.units -= close_units
        self._log(
            "CLOSE",
            {
                "trade_id": trade_id,
                "units": close_units,
                "price": price,
                "pl_jpy": round(pl, 2),
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "gross_pl_jpy": round(gross_pl, 2),
                "financing_jpy": round(financing, 2),
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            },
        )
        return pl

    def set_exit(
        self,
        trade_id: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> None:
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        if tp_price is not None:
            tp_price = _finite_number("tp_price", tp_price, positive=True)
        if sl_price is not None:
            sl_price = _finite_number("sl_price", sl_price, positive=True)
        pos.tp_price = tp_price
        pos.sl_price = sl_price
        self._log("SET_EXIT", {"trade_id": trade_id, "tp": tp_price, "sl": sl_price})

    # ---- feed ------------------------------------------------------------
    def on_quote(
        self, pair: str, bid: float, ask: float, ts: str
    ) -> list[dict[str, Any]]:
        """Process one real quote: resting orders, TP/SL, margin. Returns events."""

        bid, ask = self._validate_quote(pair, bid, ask, ts)
        self._record_quote(pair, bid, ask, ts)
        self._last_quote_watermarks[pair] = self._quote_seq
        return self._process_current_quote(pair, bid, ask, ts, self._quote_seq)

    def on_quote_batch(
        self, quotes: list[tuple[str, float, float, str]]
    ) -> list[dict[str, Any]]:
        """Atomically stage a simultaneous feed phase before processing fills.

        Conversion rates and margin marks then use one batch watermark rather
        than depending on lexicographic pair delivery order.
        """

        if not quotes:
            raise VirtualBrokerError("quote batch must not be empty")
        normalized: list[tuple[str, float, float, str]] = []
        seen: set[str] = set()
        for pair, bid, ask, ts in quotes:
            if pair in seen:
                raise VirtualBrokerError(f"duplicate pair in quote batch: {pair}")
            seen.add(pair)
            clean_bid, clean_ask = self._validate_quote(pair, bid, ask, ts)
            normalized.append((pair, clean_bid, clean_ask, ts))
        for pair, bid, ask, ts in normalized:
            self._record_quote(pair, bid, ask, ts)
        batch_watermark = self._quote_seq
        for pair, _, _, _ in normalized:
            self._last_quote_watermarks[pair] = batch_watermark
        events: list[dict[str, Any]] = []
        for pair, bid, ask, ts in normalized:
            events.extend(
                self._process_current_quote(pair, bid, ask, ts, batch_watermark)
            )
        return events

    @staticmethod
    def _validate_quote(
        pair: str, bid: float, ask: float, ts: str
    ) -> tuple[float, float]:
        _validate_pair(pair)
        bid = _finite_number("bid", bid, positive=True)
        ask = _finite_number("ask", ask, positive=True)
        if ask < bid:
            raise VirtualBrokerError(f"invalid quote {pair} {bid}/{ask}")
        if not isinstance(ts, str) or not ts:
            raise VirtualBrokerError("quote timestamp must be a non-empty string")
        return bid, ask

    def _record_quote(self, pair: str, bid: float, ask: float, ts: str) -> None:
        self._quote_seq += 1
        self.last_quotes[pair] = (bid, ask, ts)
        self._last_quote_sequences[pair] = self._quote_seq
        history = self._quote_history.setdefault(pair, [])
        history.append((bid, ask, ts, self._quote_seq))
        # Four replay phases plus asynchronous cross-pair polling need only a
        # short tail.  Keep a wider fixed bound so long sessions cannot grow
        # without limit while conversion remains point-in-time reproducible.
        if len(history) > 128:
            del history[:-128]

    def _process_current_quote(
        self, pair: str, bid: float, ask: float, ts: str, quote_sequence: int
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        # resting limit fills (worse-of level/quote when gapped)
        for order_id in list(self.orders):
            order = self.orders[order_id]
            if order.pair != pair:
                continue
            filled_price = None
            if order.kind == "LIMIT":
                if order.side == "LONG" and ask <= order.limit_price:
                    filled_price = min(order.limit_price, ask)
                elif order.side == "SHORT" and bid >= order.limit_price:
                    filled_price = max(order.limit_price, bid)
            else:  # STOP: triggers at the level, fills at level or worse
                if order.side == "LONG" and ask >= order.limit_price:
                    filled_price = max(order.limit_price, ask)
                elif order.side == "SHORT" and bid <= order.limit_price:
                    filled_price = min(order.limit_price, bid)
            if filled_price is None:
                continue
            rejection = self._entry_admission_rejection(pair, order.side, order_id)
            if rejection is not None:
                del self.orders[order_id]
                event = {
                    "event": "ORDER_CANCEL_CONCURRENCY_CAP",
                    "order_id": order_id,
                    "pair": pair,
                    "side": order.side,
                    "units": order.units,
                    "quote": {"bid": bid, "ask": ask, "ts": ts},
                    "admission": rejection,
                }
                self._log("ORDER_CANCEL_CONCURRENCY_CAP", event)
                events.append(event)
                continue
            applied_slippage_pips = 0.0
            if self.slippage_pips > 0:
                slip = self.slippage_pips * _pip(pair)
                stressed_price = _round_price(
                    pair,
                    filled_price + slip
                    if order.side == "LONG"
                    else filled_price - slip,
                )
                if order.kind == "LIMIT":
                    # A limit order may receive less improvement under stress,
                    # but can never execute through its protected price.
                    filled_price = (
                        min(order.limit_price, stressed_price)
                        if order.side == "LONG"
                        else max(order.limit_price, stressed_price)
                    )
                else:
                    filled_price = stressed_price
                if filled_price <= 0:
                    raise VirtualBrokerError(
                        "slippage produced a non-positive entry price"
                    )
                observed_price = ask if order.side == "LONG" else bid
                applied_slippage_pips = abs(filled_price - observed_price) / _pip(pair)
            if not self._margin_headroom_ok(pair, order.side, order.units):
                del self.orders[order_id]
                self._log(
                    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                    {"order_id": order_id, "pair": pair},
                )
                continue
            pip = _pip(pair)
            tp = (
                _round_price(
                    pair,
                    filled_price + order.tp_pips * pip
                    if order.side == "LONG"
                    else filled_price - order.tp_pips * pip,
                )
                if order.tp_pips
                else None
            )
            sl = (
                _round_price(
                    pair,
                    filled_price - order.sl_pips * pip
                    if order.side == "LONG"
                    else filled_price + order.sl_pips * pip,
                )
                if order.sl_pips
                else None
            )
            trade_id = self._next_id("T")
            conversion = self._conversion_evidence(pair, quote_sequence, ts)
            self.positions[trade_id] = VBPosition(
                trade_id=trade_id,
                pair=pair,
                side=order.side,
                units=order.units,
                entry_price=filled_price,
                opened_ts=ts,
                tp_price=tp,
                sl_price=sl,
            )
            del self.orders[order_id]
            event = {
                "event": "FILL_LIMIT",
                "order_id": order_id,
                "trade_id": trade_id,
                "pair": pair,
                "side": order.side,
                "units": order.units,
                "price": filled_price,
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
                "applied_slippage_pips": round(applied_slippage_pips, 8),
                "price_protection": order.kind == "LIMIT",
            }
            self._log("FILL_LIMIT", event)
            events.append(event)

        # TP/SL: SL first when both touch on the same quote (pessimistic)
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            if pos.pair != pair:
                continue
            exit_price = None
            reason = None
            if pos.side == "LONG":
                if pos.sl_price is not None and bid <= pos.sl_price:
                    exit_price, reason = min(pos.sl_price, bid), "SL"
                elif pos.tp_price is not None and bid >= pos.tp_price:
                    exit_price, reason = pos.tp_price, "TP"
            else:
                if pos.sl_price is not None and ask >= pos.sl_price:
                    exit_price, reason = max(pos.sl_price, ask), "SL"
                elif pos.tp_price is not None and ask <= pos.tp_price:
                    exit_price, reason = pos.tp_price, "TP"
            if exit_price is None:
                continue
            applied_slippage_pips = 0.0
            if reason == "SL":
                stressed_exit = self._adverse_exit_price(pair, pos.side, exit_price)
                applied_slippage_pips = abs(stressed_exit - exit_price) / _pip(pair)
                exit_price = stressed_exit
            else:
                # TP is a price-protected limit exit.  Fixed stress slippage
                # must not fabricate an execution through its protected price.
                exit_price = _round_price(pair, exit_price)
            conversion = self._conversion_evidence(pos.pair, quote_sequence, ts)
            conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
            diff = (
                (exit_price - pos.entry_price)
                if pos.side == "LONG"
                else (pos.entry_price - exit_price)
            )
            gross_pl = diff * pos.units * conversion_rate
            financing = self._financing_jpy(pos, ts, conversion_rate)
            pl = gross_pl - financing
            self.balance_jpy += pl
            del self.positions[trade_id]
            event = {
                "event": f"EXIT_{reason}",
                "trade_id": trade_id,
                "price": exit_price,
                "pl_jpy": round(pl, 2),
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "gross_pl_jpy": round(gross_pl, 2),
                "financing_jpy": round(financing, 2),
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
                "applied_slippage_pips": round(applied_slippage_pips, 8),
                "price_protection": reason == "TP",
            }
            self._log(f"EXIT_{reason}", event)
            events.append(event)

        events.extend(self._enforce_margin_after_action())
        return events

    def _enforce_margin_after_action(self) -> list[dict[str, Any]]:
        acct = self.account()
        if acct["margin_usage"] < CLOSEOUT_USAGE or not self.positions:
            return []
        events = []
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            q = self.last_quotes[pos.pair]
            quote_sequence = self._last_quote_watermarks.get(pos.pair)
            if quote_sequence is None:
                raise VirtualBrokerError(
                    f"no accounting watermark for open position pair {pos.pair}"
                )
            conversion = self._conversion_evidence(pos.pair, quote_sequence, q[2])
            conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
            price = self._adverse_exit_price(
                pos.pair, pos.side, q[0] if pos.side == "LONG" else q[1]
            )
            diff = (
                (price - pos.entry_price)
                if pos.side == "LONG"
                else (pos.entry_price - price)
            )
            gross_pl = diff * pos.units * conversion_rate
            financing = self._financing_jpy(pos, q[2], conversion_rate)
            pl = gross_pl - financing
            self.balance_jpy += pl
            del self.positions[trade_id]
            event = {
                "event": "MARGIN_CLOSEOUT",
                "trade_id": trade_id,
                "price": price,
                "pl_jpy": round(pl, 2),
                "gross_pl_jpy": round(gross_pl, 2),
                "financing_jpy": round(financing, 2),
                "quote": {"bid": q[0], "ask": q[1], "ts": q[2]},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            }
            self._log("MARGIN_CLOSEOUT", event)
            events.append(event)
        return events

    # ---- persistence -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        snap = {
            "schema": SNAPSHOT_SCHEMA,
            "balance_jpy": self.balance_jpy,
            "seq": self._seq,
            "positions": [vars(p) for p in self.positions.values()],
            "orders": [vars(o) for o in self.orders.values()],
            "quote_seq": self._quote_seq,
            "last_quotes": {
                pair: {"bid": q[0], "ask": q[1], "ts": q[2]}
                for pair, q in self.last_quotes.items()
            },
            "last_quote_sequences": dict(self._last_quote_sequences),
            "last_quote_watermarks": dict(self._last_quote_watermarks),
            "quote_history": {
                pair: [
                    {"bid": q[0], "ask": q[1], "ts": q[2], "sequence": q[3]}
                    for q in history
                ]
                for pair, history in self._quote_history.items()
            },
            "feed_cursor": self.feed_cursor,
            "ledger_tip_sha": self._prev_sha,
        }
        _validate_finite_tree(snap, "snapshot")
        return snap

    def restore(self, snap: dict[str, Any]) -> None:
        if not isinstance(snap, dict):
            raise VirtualBrokerError("broker snapshot must be an object")
        if set(snap) != _SNAPSHOT_KEYS or snap.get("schema") != SNAPSHOT_SCHEMA:
            raise VirtualBrokerError("broker snapshot schema mismatch")
        _validate_finite_tree(snap, "snapshot")
        balance_jpy = _finite_number("snapshot balance_jpy", snap["balance_jpy"])
        seq_value = _finite_number("snapshot seq", snap["seq"], non_negative=True)
        if not seq_value.is_integer():
            raise VirtualBrokerError("snapshot seq must be an integer")
        sequence_counter = int(seq_value)

        positions: dict[str, VBPosition] = {}
        raw_positions = snap.get("positions")
        if not isinstance(raw_positions, list):
            raise VirtualBrokerError("snapshot positions must be a list")
        for raw in raw_positions:
            if not isinstance(raw, dict):
                raise VirtualBrokerError("snapshot position must be an object")
            try:
                pos = VBPosition(**raw)
            except (KeyError, TypeError) as exc:
                raise VirtualBrokerError("invalid snapshot position schema") from exc
            _validate_pair(pos.pair)
            if pos.side not in {"LONG", "SHORT"}:
                raise VirtualBrokerError("invalid snapshot position side")
            pos.units = _finite_number(
                "snapshot position units", pos.units, positive=True
            )
            pos.entry_price = _finite_number(
                "snapshot position entry_price", pos.entry_price, positive=True
            )
            if pos.tp_price is not None:
                pos.tp_price = _finite_number(
                    "snapshot position tp_price", pos.tp_price, positive=True
                )
            if pos.sl_price is not None:
                pos.sl_price = _finite_number(
                    "snapshot position sl_price", pos.sl_price, positive=True
                )
            if not pos.trade_id or pos.trade_id in positions:
                raise VirtualBrokerError("duplicate or empty snapshot trade_id")
            if not isinstance(pos.opened_ts, str) or not pos.opened_ts:
                raise VirtualBrokerError("snapshot position opened_ts is required")
            positions[pos.trade_id] = pos

        orders: dict[str, VBOrder] = {}
        raw_orders = snap.get("orders")
        if not isinstance(raw_orders, list):
            raise VirtualBrokerError("snapshot orders must be a list")
        for raw in raw_orders:
            if not isinstance(raw, dict):
                raise VirtualBrokerError("snapshot order must be an object")
            try:
                order = VBOrder(**raw)
            except (KeyError, TypeError) as exc:
                raise VirtualBrokerError("invalid snapshot order schema") from exc
            _validate_pair(order.pair)
            if order.side not in {"LONG", "SHORT"} or order.kind not in {
                "LIMIT",
                "STOP",
            }:
                raise VirtualBrokerError("invalid snapshot order side/kind")
            order.units = _finite_number(
                "snapshot order units", order.units, positive=True
            )
            order.limit_price = _finite_number(
                "snapshot order price", order.limit_price, positive=True
            )
            if order.tp_pips is not None:
                order.tp_pips = _finite_number(
                    "snapshot order tp_pips", order.tp_pips, positive=True
                )
            if order.sl_pips is not None:
                order.sl_pips = _finite_number(
                    "snapshot order sl_pips", order.sl_pips, positive=True
                )
            if not order.order_id or order.order_id in orders:
                raise VirtualBrokerError("duplicate or empty snapshot order_id")
            orders[order.order_id] = order

        raw_last_quotes = snap.get("last_quotes", {})
        raw_sequences = snap.get("last_quote_sequences", {})
        raw_watermarks = snap.get("last_quote_watermarks", {})
        raw_history = snap.get("quote_history", {})
        if not all(
            isinstance(value, dict)
            for value in (raw_last_quotes, raw_sequences, raw_watermarks, raw_history)
        ):
            raise VirtualBrokerError("invalid snapshot quote state")
        quote_seq_value = _finite_number(
            "snapshot quote_seq", snap.get("quote_seq", 0), non_negative=True
        )
        if not quote_seq_value.is_integer():
            raise VirtualBrokerError("snapshot quote_seq must be an integer")
        quote_seq = int(quote_seq_value)
        last_quotes: dict[str, tuple[float, float, str]] = {}
        last_sequences: dict[str, int] = {}
        last_watermarks: dict[str, int] = {}
        quote_history: dict[str, list[tuple[float, float, str, int]]] = {}
        for pair, raw_quote in raw_last_quotes.items():
            if not isinstance(raw_quote, dict):
                raise VirtualBrokerError("invalid snapshot last quote")
            bid, ask = self._validate_quote(
                pair, raw_quote.get("bid"), raw_quote.get("ask"), raw_quote.get("ts")
            )
            sequence_value = _finite_number(
                "snapshot quote sequence", raw_sequences.get(pair), positive=True
            )
            watermark_value = _finite_number(
                "snapshot quote watermark", raw_watermarks.get(pair), positive=True
            )
            if not sequence_value.is_integer() or not watermark_value.is_integer():
                raise VirtualBrokerError("snapshot quote sequence must be an integer")
            sequence = int(sequence_value)
            watermark = int(watermark_value)
            if sequence > watermark or watermark > quote_seq:
                raise VirtualBrokerError("snapshot quote watermark is inconsistent")
            last_quotes[pair] = (bid, ask, raw_quote["ts"])
            last_sequences[pair] = sequence
            last_watermarks[pair] = watermark
            raw_pair_history = raw_history.get(pair)
            if not isinstance(raw_pair_history, list) or not raw_pair_history:
                raise VirtualBrokerError("snapshot quote history is missing")
            parsed_history: list[tuple[float, float, str, int]] = []
            previous_sequence = 0
            for raw_item in raw_pair_history:
                if not isinstance(raw_item, dict):
                    raise VirtualBrokerError("invalid snapshot quote history")
                hist_bid, hist_ask = self._validate_quote(
                    pair,
                    raw_item.get("bid"),
                    raw_item.get("ask"),
                    raw_item.get("ts"),
                )
                hist_sequence_value = _finite_number(
                    "snapshot history sequence",
                    raw_item.get("sequence"),
                    positive=True,
                )
                if not hist_sequence_value.is_integer():
                    raise VirtualBrokerError(
                        "snapshot history sequence must be an integer"
                    )
                hist_sequence = int(hist_sequence_value)
                if hist_sequence <= previous_sequence or hist_sequence > quote_seq:
                    raise VirtualBrokerError("snapshot quote history is not monotonic")
                previous_sequence = hist_sequence
                parsed_history.append(
                    (hist_bid, hist_ask, raw_item["ts"], hist_sequence)
                )
            if (
                parsed_history[-1][:3] != last_quotes[pair]
                or parsed_history[-1][3] != sequence
            ):
                raise VirtualBrokerError("snapshot last quote/history mismatch")
            quote_history[pair] = parsed_history
        if set(raw_sequences) != set(last_quotes) or set(raw_watermarks) != set(
            last_quotes
        ):
            raise VirtualBrokerError("snapshot quote maps disagree")
        if set(raw_history) != set(last_quotes):
            raise VirtualBrokerError("snapshot quote history pairs disagree")

        ledger_tip = snap.get("ledger_tip_sha")
        if (
            not isinstance(ledger_tip, str)
            or len(ledger_tip) != 64
            or any(char not in "0123456789abcdef" for char in ledger_tip)
            or ledger_tip != self._prev_sha
        ):
            raise VirtualBrokerError("snapshot ledger tip does not match ledger")
        feed_cursor = snap.get("feed_cursor")
        if feed_cursor is not None and not isinstance(feed_cursor, dict):
            raise VirtualBrokerError("snapshot feed_cursor must be an object")

        generated_ids = [*positions, *orders]
        generated_sequences: list[int] = []
        for identity in generated_ids:
            if (
                len(identity) != 7
                or identity[0] not in {"T", "O"}
                or not identity[1:].isdigit()
            ):
                raise VirtualBrokerError("snapshot contains an invalid generated id")
            generated_sequences.append(int(identity[1:]))
        if generated_sequences and sequence_counter < max(generated_sequences):
            raise VirtualBrokerError("snapshot sequence would reuse an existing id")

        self.balance_jpy = balance_jpy
        self._seq = sequence_counter
        self.positions = positions
        self.orders = orders
        self.last_quotes = last_quotes
        self._last_quote_sequences = last_sequences
        self._last_quote_watermarks = last_watermarks
        self._quote_history = quote_history
        self._quote_seq = quote_seq
        self.feed_cursor = feed_cursor
