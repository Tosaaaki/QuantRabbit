#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from execution.order_manager import set_trade_protections
from execution.position_manager import PositionManager, agent_client_prefixes

PIP = 0.01


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_sl_pips(entry_thesis: object) -> Optional[float]:
    if not isinstance(entry_thesis, dict):
        return None
    for key in ("hard_stop_pips", "sl_pips", "loss_guard_pips", "loss_guard"):
        val = _as_float(entry_thesis.get(key))
        if val is not None and val > 0:
            return float(val)
    return None


def _existing_sl_price(trade: dict) -> Optional[float]:
    raw = trade.get("stop_loss") or {}
    if isinstance(raw, dict):
        sl = _as_float(raw.get("price"))
        if sl is not None and sl > 0:
            return float(sl)
    return None


def _desired_sl_price(entry: float, units: int, sl_pips: float) -> Optional[float]:
    if entry <= 0 or units == 0 or sl_pips <= 0:
        return None
    offset = sl_pips * PIP
    if units > 0:
        return round(entry - offset, 3)
    return round(entry + offset, 3)


def _tighten_sl(existing: Optional[float], desired: float, units: int) -> Optional[float]:
    if units == 0:
        return None
    if existing is None:
        return desired
    # Never loosen (widen) stops: only tighten.
    if units > 0:  # long: higher SL is tighter
        return max(existing, desired)
    # short: lower SL is tighter
    return min(existing, desired)


async def _run(args: argparse.Namespace) -> int:
    pockets_filter = {p.strip() for p in (args.pockets or "").split(",") if p.strip()}
    apply = bool(args.apply)
    min_pips = float(args.min_sl_pips) if args.min_sl_pips is not None else None
    max_pips = float(args.max_sl_pips) if args.max_sl_pips is not None else None
    fallback_pips = float(args.fallback_sl_pips) if args.fallback_sl_pips is not None else None

    pm = PositionManager()
    try:
        positions = pm.get_open_positions()
    finally:
        try:
            pm.close()
        except Exception:
            pass

    candidates: list[tuple[str, dict]] = []
    for pocket, info in positions.items():
        if pocket.startswith("__"):
            continue
        if pockets_filter and pocket not in pockets_filter:
            continue
        trades = info.get("open_trades") if isinstance(info, dict) else None
        if not trades:
            continue
        for tr in trades:
            client_id = str(tr.get("client_id") or "")
            if not client_id.startswith(agent_client_prefixes):
                continue
            candidates.append((pocket, tr))

    if args.limit is not None and args.limit > 0:
        candidates = candidates[: int(args.limit)]

    updated = 0
    skipped = 0
    missing = 0

    for pocket, tr in candidates:
        trade_id = str(tr.get("trade_id") or "")
        units = int(tr.get("units", 0) or 0)
        entry = _as_float(tr.get("price")) or 0.0
        if not trade_id or units == 0 or entry <= 0:
            skipped += 1
            continue

        thesis = tr.get("entry_thesis") or {}
        sl_pips = _pick_sl_pips(thesis)
        if sl_pips is None:
            if fallback_pips is None or fallback_pips <= 0:
                missing += 1
                continue
            sl_pips = fallback_pips

        if min_pips is not None and sl_pips < min_pips:
            sl_pips = min_pips
        if max_pips is not None and sl_pips > max_pips:
            sl_pips = max_pips

        desired = _desired_sl_price(entry, units, sl_pips)
        if desired is None:
            skipped += 1
            continue

        existing = _existing_sl_price(tr)
        target = _tighten_sl(existing, desired, units)
        if target is None:
            skipped += 1
            continue

        if existing is not None and abs(existing - target) < 1e-9:
            skipped += 1
            continue

        side = "long" if units > 0 else "short"
        msg = (
            f"trade={trade_id} pocket={pocket} side={side} entry={entry:.3f} "
            f"sl_pips={sl_pips:.2f} existing_sl={existing if existing is not None else '-'} target_sl={target:.3f}"
        )
        if not apply:
            print(f"[DRYRUN] {msg}")
            updated += 1
            continue

        ok = await set_trade_protections(trade_id, sl_price=target, tp_price=None)
        if ok:
            updated += 1
            logging.info("[APPLY] %s", msg)
        else:
            skipped += 1
            logging.warning("[APPLY] failed %s", msg)

    print(f"candidates={len(candidates)} updated={updated} skipped={skipped} missing_sl_pips={missing}")
    if not apply:
        print("Re-run with --apply to update trades.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach (or tighten) hard stop-loss prices for open trades based on entry_thesis."
    )
    parser.add_argument(
        "--pockets",
        default="",
        help="Comma-separated pockets to include (e.g. scalp,micro,macro). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max trades to process (0 = no limit).",
    )
    parser.add_argument(
        "--min-sl-pips",
        type=float,
        default=None,
        help="Clamp SL pips to at least this value (optional).",
    )
    parser.add_argument(
        "--max-sl-pips",
        type=float,
        default=None,
        help="Clamp SL pips to at most this value (optional).",
    )
    parser.add_argument(
        "--fallback-sl-pips",
        type=float,
        default=None,
        help="Fallback SL pips when thesis has no sl_pips/hard_stop_pips (optional).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually send TradeCRCDO updates (default: dry-run).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

