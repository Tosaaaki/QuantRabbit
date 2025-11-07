"""Helpers for massaging open-position snapshots."""

from __future__ import annotations

import os
from typing import Dict

_IGNORE_MANUAL_TRADES = os.getenv("IGNORE_MANUAL_TRADES", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

__all__ = ["filter_bot_managed_positions", "IGNORE_MANUAL_TRADES"]


IGNORE_MANUAL_TRADES = _IGNORE_MANUAL_TRADES


def filter_bot_managed_positions(
    open_positions: Dict,
    *,
    ignore_manual_trades: bool | None = None,
) -> Dict:
    """Return a copy with only QuantRabbit-managed trades (client_id starts with qr-)."""

    if ignore_manual_trades is None:
        ignore_manual_trades = IGNORE_MANUAL_TRADES
    if not ignore_manual_trades:
        return dict(open_positions or {})
    managed: Dict = {}
    for pocket, info in (open_positions or {}).items():
        if pocket == "__net__":
            continue
        trades = list((info or {}).get("open_trades") or [])
        kept = [tr for tr in trades if str(tr.get("client_id") or "").startswith("qr-")]
        if not kept:
            continue
        long_units = short_units = 0
        long_weighted = short_weighted = 0.0
        total_units = 0
        weighted_price = 0.0
        for tr in kept:
            units = int(tr.get("units", 0) or 0)
            price = float(tr.get("price", 0.0) or 0.0)
            total_units += units
            weighted_price += price * units
            if units > 0:
                long_units += units
                long_weighted += price * units
            elif units < 0:
                abs_units = abs(units)
                short_units += abs_units
                short_weighted += price * abs_units
        avg_price = (weighted_price / total_units) if total_units else info.get("avg_price", 0.0)
        long_avg = (long_weighted / long_units) if long_units else 0.0
        short_avg = (short_weighted / short_units) if short_units else 0.0
        managed[pocket] = {
            "units": total_units,
            "avg_price": avg_price,
            "trades": len(kept),
            "long_units": long_units,
            "long_avg_price": long_avg,
            "short_units": short_units,
            "short_avg_price": short_avg,
            "open_trades": kept,
            "unrealized_pl": (info or {}).get("unrealized_pl", 0.0),
            "unrealized_pl_pips": (info or {}).get("unrealized_pl_pips", 0.0),
        }
    return managed

