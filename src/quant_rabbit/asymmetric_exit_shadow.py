"""Asymmetric structure-break exit shadow evaluator (weakness ledger W1/W9).

Symmetric TP grids cut off the right tail that funded the operator's 2025
run.  This evaluator scores the operator-style exit on exact S5 bid/ask:
no take-profit, hold until either the pre-committed structure level breaks
or the time boundary arrives.  It never touches the sealed 182-candidate
catalog: it re-scores already-filled entries under the alternative exit so
the two policies can be compared pairwise on identical fills.

Causality rules: the structure level must be pre-committed (from closed
candles before activation — the caller seals its provenance); any structure
touch in the fill candle itself is temporally ambiguous and charged as an
immediate structure exit; a gap through the level exits at the executable
open beyond it; missing coverage yields an explicit unresolved status that
still carries a pessimistic full-adverse value so cohort aggregation can
stay complete instead of deleting the trade.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle

POLICY = "NO_TP_STRUCTURE_BREAK_OR_TIME_EXIT_PESSIMISTIC_AMBIGUITY_V1"
CONTRACT = "QR_ASYMMETRIC_EXIT_SHADOW_OUTCOME_V1"
_UTC = timezone.utc


class AsymmetricExitError(ValueError):
    """Raised when evaluation inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_price(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AsymmetricExitError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise AsymmetricExitError(f"{label} must be a positive finite price")
    return number


def resolve_structure_break_exit(
    *,
    side: str,
    fill_price: float,
    fill_at_utc: datetime,
    structure_level: float,
    structure_provenance_sha256: str,
    time_boundary_utc: datetime,
    candles: Sequence[S5BidAskCandle],
    pip_factor: float,
) -> dict[str, Any]:
    """Score one filled entry under the asymmetric exit on exact S5 data."""

    side_name = str(side).upper()
    if side_name not in {"LONG", "SHORT"}:
        raise AsymmetricExitError("side must be LONG or SHORT")
    fill = _finite_price(fill_price, "fill_price")
    structure = _finite_price(structure_level, "structure_level")
    factor = _finite_price(pip_factor, "pip_factor")
    provenance = str(structure_provenance_sha256 or "")
    if len(provenance) != 64 or any(
        c not in "0123456789abcdef" for c in provenance
    ):
        raise AsymmetricExitError(
            "structure provenance must be a lowercase sha256"
        )
    if fill_at_utc.tzinfo is None or time_boundary_utc.tzinfo is None:
        raise AsymmetricExitError("clocks must be timezone-aware")
    fill_at = fill_at_utc.astimezone(_UTC)
    boundary = time_boundary_utc.astimezone(_UTC)
    if boundary <= fill_at:
        raise AsymmetricExitError("time boundary must be after the fill")
    if side_name == "LONG" and structure >= fill:
        raise AsymmetricExitError("LONG structure level must sit below the fill")
    if side_name == "SHORT" and structure <= fill:
        raise AsymmetricExitError("SHORT structure level must sit above the fill")

    previous: datetime | None = None
    exit_price: float | None = None
    exit_at: datetime | None = None
    exit_reason: str | None = None
    ambiguous = False
    for candle in candles:
        if candle.__class__ is not S5BidAskCandle:
            raise AsymmetricExitError("candles must be exact S5 bid/ask rows")
        stamp = candle.timestamp_utc.astimezone(_UTC)
        if previous is not None and stamp <= previous:
            raise AsymmetricExitError("candles must be chronological and unique")
        previous = stamp
        if stamp < fill_at.replace(second=(fill_at.second // 5) * 5, microsecond=0):
            continue
        is_fill_candle = stamp <= fill_at
        if stamp >= boundary:
            exit_price = float(candle.bid_o if side_name == "LONG" else candle.ask_o)
            exit_at = stamp
            exit_reason = "EXECUTABLE_TIME_CLOSE"
            break
        if side_name == "LONG":
            gap = float(candle.bid_o) < structure
            touched = float(candle.bid_l) <= structure
            executable_open = float(candle.bid_o)
        else:
            gap = float(candle.ask_o) > structure
            touched = float(candle.ask_h) >= structure
            executable_open = float(candle.ask_o)
        if touched:
            exit_price = executable_open if gap else structure
            exit_at = stamp
            exit_reason = (
                "STRUCTURE_BREAK_GAP" if gap else "STRUCTURE_BREAK"
            )
            ambiguous = is_fill_candle
            if is_fill_candle:
                exit_reason += "_AMBIGUOUS_FILL_S5"
            break

    pessimistic = (
        (structure - fill) * factor
        if side_name == "LONG"
        else (fill - structure) * factor
    )
    if exit_price is None:
        body: dict[str, Any] = {
            "contract": CONTRACT,
            "policy": POLICY,
            "status": "UNRESOLVED_INSUFFICIENT_COVERAGE",
            "result_available": False,
            "pessimistic_realized_pips": round(pessimistic, 9),
            "cohort_must_use_pessimistic_value": True,
        }
    else:
        realized = (
            (exit_price - fill) * factor
            if side_name == "LONG"
            else (fill - exit_price) * factor
        )
        body = {
            "contract": CONTRACT,
            "policy": POLICY,
            "status": "RESOLVED",
            "result_available": True,
            "exit_at_utc": exit_at.isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "ambiguous_same_s5": ambiguous,
            "realized_pips": round(realized, 9),
            "pessimistic_realized_pips": round(pessimistic, 9),
            "cohort_must_use_pessimistic_value": False,
        }
    body.update(
        {
            "side": side_name,
            "fill_price": fill,
            "fill_at_utc": fill_at.isoformat(),
            "structure_level": structure,
            "structure_provenance_sha256": provenance,
            "time_boundary_utc": boundary.isoformat(),
            "take_profit_exists": False,
            "order_authority": "NONE",
            "live_permission": False,
        }
    )
    return {**body, "outcome_sha256": _canonical_sha(body)}
