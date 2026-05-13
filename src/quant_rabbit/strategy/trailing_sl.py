"""Broker-side trailing stop-loss updates for new entries (2026-05-13).

The operator demanded a structural fix after the 2026-05-12T15:33 UTC
mass-close incident: every new entry must carry a broker-side SL on
entry (handled by `_oanda_order_request` under `QR_NEW_ENTRY_INITIAL_SL=1`),
AND the SL must trail as adverse M15 structure prints so a panic-close
trader cannot override the broker-side protection.

This module handles the trailing-update half. It is invoked from
`autotrade-cycle` (or the `trailing-sl-update` CLI for manual / smoke
use). The function NEVER touches a position that lacks an existing
broker SL — positions in SL-free mode (including the 9 trades that
were open at the 2026-05-12T15:33 incident and any future SL-free
positions) are skipped by construction, so this code path cannot
modify any "absolute protection" trade.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from quant_rabbit.models import BrokerSnapshot, Owner, Side


# Buffer applied to the BOS price when computing the new SL. A small
# multiple of current spread keeps the SL out of routine round-trip
# fill jitter without giving the move room to retest. Per
# AGENT_CONTRACT §3.5 this is a market-derived buffer (spread × N) and
# the override env knob is documented.
TRAILING_SL_SPREAD_BUFFER_MULT = 2.0


@dataclass(frozen=True)
class TrailingUpdate:
    trade_id: str
    pair: str
    side: str
    old_sl: float
    new_sl: float
    bos_tf: str
    bos_price: float
    reason: str
    applied: bool


# Regex matches the chart_reader chart_story format:
#   "M15(REGIME, ADX=… struct=BOS_UP@1.1234)"
# Captures (timeframe, event_type, direction, price).
_STRUCT_RE = re.compile(
    r"\b(M5|M15|M30|H1|H4|D)\([^)]*?struct=(BOS|CHOCH)_(UP|DOWN)@([0-9]+\.?[0-9]*)"
)

# Only these timeframes can drive a trailing update. M5/M1 print BOS
# events on routine noise; H4/D fire too rarely to be useful for
# trailing a fresh entry. M15 / M30 / H1 is the operator-relevant band.
TRAILING_TIMEFRAMES = ("M15", "M30", "H1")


def _parse_struct_events(chart_story: str) -> dict[str, tuple[str, str, float]]:
    out: dict[str, tuple[str, str, float]] = {}
    if not chart_story:
        return out
    for tf, event_type, direction, price_str in _STRUCT_RE.findall(chart_story):
        try:
            out[tf] = (event_type, direction, float(price_str))
        except (TypeError, ValueError):
            continue
    return out


def _spread_pips(snapshot: BrokerSnapshot, pair: str) -> float | None:
    quote = snapshot.quotes.get(pair) if snapshot.quotes else None
    if quote is None or quote.bid is None or quote.ask is None:
        return None
    pip_factor = 100.0 if pair.endswith("_JPY") else 10000.0
    diff = abs(quote.ask - quote.bid)
    return diff * pip_factor


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _round_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _compute_new_sl(
    *,
    side: str,
    current_sl: float,
    bos_price: float,
    spread_pips: float | None,
    pair: str,
) -> float | None:
    """Return a tighter SL anchored on the BOS price + spread buffer.

    For LONG the BOS price prints DOWN (adverse). The new SL sits
    JUST BELOW it by `TRAILING_SL_SPREAD_BUFFER_MULT × spread`, so a
    routine wick into the BOS does not trigger SL. The SL is only
    moved if the candidate is CLOSER to the current price than the
    existing SL — broker SL must never be widened (AGENT_CONTRACT
    §10: "Existing SL cannot be widened").
    """
    if spread_pips is None:
        return None
    buffer = spread_pips * TRAILING_SL_SPREAD_BUFFER_MULT * _pip_size(pair)
    side_upper = side.upper()
    if side_upper == "LONG":
        # LONG SL is below price; tighter means HIGHER (closer to price).
        candidate = bos_price - buffer
        if candidate > current_sl:
            return _round_price(pair, candidate)
    elif side_upper == "SHORT":
        # SHORT SL is above price; tighter means LOWER.
        candidate = bos_price + buffer
        if candidate < current_sl:
            return _round_price(pair, candidate)
    return None


def _new_entry_threshold_trade_id() -> int | None:
    raw = os.environ.get("QR_TRAILING_SL_FROM_TRADE_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _iter_eligible_positions(
    snapshot: BrokerSnapshot,
    *,
    from_trade_id: int | None,
) -> Iterable[Any]:
    """Yield trader-owned positions eligible for trailing SL update.

    Eligibility:
      - Owner is `trader` (manual/unknown positions are operator-owned
        and never touched).
      - Position has a broker-side `stop_loss` already set. Positions
        in SL-free mode (no SL) are EXPLICITLY skipped — this is the
        invariant that protects every existing 2026-05-12 trade and
        any future operator-preferred SL-free entry.
      - `trade_id > QR_TRAILING_SL_FROM_TRADE_ID` if that env is set.
        This is the operator's "trailing applies to trades opened on
        or after this id" gate; absence means apply to all trader-owned
        positions with SL.
    """
    threshold = from_trade_id
    for pos in snapshot.positions:
        if pos.owner != Owner.TRADER:
            continue
        if pos.stop_loss is None:
            continue
        if threshold is not None:
            try:
                tid = int(pos.trade_id) if pos.trade_id is not None else None
            except (TypeError, ValueError):
                continue
            if tid is None or tid <= threshold:
                continue
        yield pos


def apply_trailing_sls(
    *,
    snapshot: BrokerSnapshot,
    pair_charts_payload: dict[str, Any],
    broker_client: Any,
    dry_run: bool = False,
) -> list[TrailingUpdate]:
    """Tighten SL on eligible trader-owned positions when M15/M30/H1
    structure prints against the position side.

    Returns the list of updates considered. `applied=False` means the
    update was identified but skipped (e.g. dry_run, or new SL was
    wider than current).
    """
    from_id = _new_entry_threshold_trade_id()
    chart_by_pair = {
        c.get("pair"): c
        for c in (pair_charts_payload.get("charts") or [])
        if isinstance(c, dict)
    }

    results: list[TrailingUpdate] = []
    for pos in _iter_eligible_positions(snapshot, from_trade_id=from_id):
        pair = pos.pair
        side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
        chart = chart_by_pair.get(pair) or {}
        chart_story = str(chart.get("chart_story") or "")
        structs = _parse_struct_events(chart_story)

        counter_dir = "DOWN" if side.upper() == "LONG" else "UP"
        bos_event = None
        bos_tf = None
        for tf in TRAILING_TIMEFRAMES:
            event = structs.get(tf)
            if event and event[1] == counter_dir:
                bos_event = event
                bos_tf = tf
                break
        if bos_event is None:
            continue

        spread_pips = _spread_pips(snapshot, pair)
        bos_price = bos_event[2]
        new_sl = _compute_new_sl(
            side=side,
            current_sl=pos.stop_loss,
            bos_price=bos_price,
            spread_pips=spread_pips,
            pair=pair,
        )
        if new_sl is None:
            continue

        reason = (
            f"{bos_tf} {bos_event[0]}_{bos_event[1]}@{bos_event[2]:g} prints "
            f"against {side.upper()} thesis; tighten SL "
            f"{pos.stop_loss:g} → {new_sl:g}"
        )

        applied = False
        if not dry_run:
            order_request = {
                "stopLoss": {
                    "price": f"{new_sl:.{3 if pair.endswith('_JPY') else 5}f}",
                    "timeInForce": "GTC",
                },
            }
            try:
                broker_client.replace_trade_dependent_orders(str(pos.trade_id), order_request)
                applied = True
            except Exception:  # broker-side errors must not stop the loop
                applied = False
        results.append(
            TrailingUpdate(
                trade_id=str(pos.trade_id),
                pair=pair,
                side=side,
                old_sl=float(pos.stop_loss),
                new_sl=float(new_sl),
                bos_tf=bos_tf or "",
                bos_price=float(bos_price),
                reason=reason,
                applied=applied,
            )
        )
    return results
