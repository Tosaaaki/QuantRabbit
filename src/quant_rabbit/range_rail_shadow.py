"""Range-rail mean-reversion shadow evaluator (Lane F, completeness gap #5).

The all-weather subtraction/addition proof (W28) showed the empty RANGE
cells cannot be made positive by cutting them — only a family that actually
wins in range can.  This is that family's mechanism: a passive LIMIT at a
pre-committed rail, reverting to mid.  SHORT sells the upper rail as price
rises into it and buys back at mid; LONG buys the lower rail as price falls
into it and sells at mid.  Fills and exits use only real S5 bid/ask opens.

Causality: rails are pre-committed from closed candles before the decision
(caller seals provenance).  A passive LIMIT fills only when the executable
open reaches the rail.  Any TP/SL touch in the same S5 candle as the fill
is temporally ambiguous and charged the full stop.  A statistical proof of
edge still awaits the planned M5 corpus; this module is the mechanism, not
the proof.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

from quant_rabbit.adaptive_exact_s5_profit_engine import ExactS5Series

POLICY = "PASSIVE_RAIL_LIMIT_REVERT_TO_MID_S5_PESSIMISTIC_AMBIGUITY_V1"
CONTRACT = "QR_RANGE_RAIL_SHADOW_OUTCOME_V1"
_UTC = timezone.utc


class RangeRailError(ValueError):
    """Raised when range-rail inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _price(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RangeRailError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RangeRailError(f"{label} must be a positive finite price")
    return number


def resolve_range_rail_rotation(
    series: ExactS5Series,
    *,
    side: str,
    rail_price: float,
    mid_price: float,
    stop_price: float,
    rail_provenance_sha256: str,
    decision_utc: datetime,
    entry_ttl_seconds: int,
    horizon_seconds: int,
    pip_factor: float,
) -> dict[str, Any]:
    """Score one passive rail LIMIT reverting to mid, on exact S5 bid/ask."""

    name = str(side).upper()
    if name not in {"LONG", "SHORT"}:
        raise RangeRailError("side must be LONG or SHORT")
    rail = _price(rail_price, "rail_price")
    mid = _price(mid_price, "mid_price")
    stop = _price(stop_price, "stop_price")
    factor = _price(pip_factor, "pip_factor")
    if decision_utc.tzinfo is None:
        raise RangeRailError("decision clock must be timezone-aware")
    if not isinstance(entry_ttl_seconds, int) or entry_ttl_seconds <= 0:
        raise RangeRailError("entry_ttl_seconds must be positive")
    if not isinstance(horizon_seconds, int) or horizon_seconds <= entry_ttl_seconds:
        raise RangeRailError("horizon must exceed entry TTL")
    provenance = str(rail_provenance_sha256 or "")
    if len(provenance) != 64 or any(c not in "0123456789abcdef" for c in provenance):
        raise RangeRailError("rail provenance must be a lowercase sha256")
    # Geometry: SHORT sells high (rail > mid > stop-below? no) — the rail is
    # the extreme, mid is the revert target, stop is beyond the rail.
    if name == "SHORT":
        if not (stop > rail > mid):
            raise RangeRailError("SHORT rails require stop > rail > mid")
    else:
        if not (stop < rail < mid):
            raise RangeRailError("LONG rails require stop < rail < mid")

    decision = decision_utc.astimezone(_UTC)
    decision_epoch = int(decision.timestamp())
    ttl_epoch = decision_epoch + entry_ttl_seconds
    horizon_epoch = decision_epoch + horizon_seconds

    epochs = series.s5_epochs
    start = bisect.bisect_left(epochs, decision_epoch)

    fill_epoch: int | None = None
    fill_index: int | None = None
    for index in range(start, len(epochs)):
        epoch = int(epochs[index])
        if epoch > ttl_epoch:
            break
        bid_open = float(series.bid_opens[index])
        ask_open = float(series.ask_opens[index])
        # Passive fill: SHORT sells when bid reaches up to the rail; LONG buys
        # when ask reaches down to the rail.
        if name == "SHORT" and bid_open >= rail:
            fill_epoch, fill_index = epoch, index
            break
        if name == "LONG" and ask_open <= rail:
            fill_epoch, fill_index = epoch, index
            break

    if fill_index is None:
        body: dict[str, Any] = {
            "status": "NO_FILL_WITHIN_TTL",
            "filled": False,
            "realized_pips": None,
            "result_available": False,
        }
    else:
        entry = rail
        realized: float | None = None
        exit_reason = None
        ambiguous = False
        for index in range(fill_index, len(epochs)):
            epoch = int(epochs[index])
            if epoch >= horizon_epoch:
                bid_open = float(series.bid_opens[index])
                ask_open = float(series.ask_opens[index])
                # Closing a SHORT buys back at the ASK; closing a LONG sells
                # at the BID — the exit convention, not the entry one.
                exit_price = ask_open if name == "SHORT" else bid_open
                realized = (
                    (entry - exit_price) * factor
                    if name == "SHORT"
                    else (exit_price - entry) * factor
                )
                exit_reason = "EXECUTABLE_TIME_CLOSE"
                break
            bid_open = float(series.bid_opens[index])
            ask_open = float(series.ask_opens[index])
            if name == "SHORT":
                tp_hit = ask_open <= mid  # buy back at/below mid
                sl_hit = ask_open >= stop
            else:
                tp_hit = bid_open >= mid  # sell at/above mid
                sl_hit = bid_open <= stop
            is_fill_candle = epoch == fill_epoch
            if is_fill_candle and (tp_hit or sl_hit):
                # Same-S5 as fill: temporally ambiguous, charge full stop.
                realized = (
                    (entry - stop) * factor
                    if name == "SHORT"
                    else (stop - entry) * factor
                )
                exit_reason = "STOP_LOSS_AMBIGUOUS_FILL_S5"
                ambiguous = True
                break
            if sl_hit:
                exit_price = ask_open if name == "SHORT" else bid_open
                realized = (
                    (entry - max(exit_price, stop)) * factor
                    if name == "SHORT"
                    else (min(exit_price, stop) - entry) * factor
                )
                exit_reason = "STOP_LOSS"
                break
            if tp_hit:
                realized = (
                    (entry - mid) * factor
                    if name == "SHORT"
                    else (mid - entry) * factor
                )
                exit_reason = "MEAN_REVERT_TO_MID"
                break
        if realized is None:
            body = {
                "status": "UNRESOLVED_INSUFFICIENT_COVERAGE",
                "filled": True,
                "realized_pips": None,
                "result_available": False,
                "pessimistic_realized_pips": round(
                    (entry - stop) * factor if name == "SHORT" else (stop - entry) * factor,
                    9,
                ),
            }
        else:
            body = {
                "status": "RESOLVED",
                "filled": True,
                "fill_at_utc": datetime.fromtimestamp(fill_epoch, tz=_UTC).isoformat(),
                "exit_reason": exit_reason,
                "ambiguous_same_s5": ambiguous,
                "realized_pips": round(realized, 9),
                "result_available": True,
            }

    body.update(
        {
            "contract": CONTRACT,
            "policy": POLICY,
            "side": name,
            "rail_price": rail,
            "mid_price": mid,
            "stop_price": stop,
            "rail_provenance_sha256": provenance,
            "decision_utc": decision.isoformat(),
            "proof_awaits_m5_corpus": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
    )
    return {**body, "outcome_sha256": _canonical_sha(body)}
