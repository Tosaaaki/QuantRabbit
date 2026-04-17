#!/usr/bin/env python3
"""
Thesis Gate — make bots respect trader's macro thesis.

Reads `collab_trade/state.md` for the trader's current "Best vehicle NOW:" line
and "Expressions to avoid:" line. Bots must align with the macro thesis or skip.

The trader sees H4 + macro + Currency Pulse + news. The bot sees M5 + H1 only.
When they disagree, the trader's read is more authoritative — and our P&L proves
it (today's bot SHORTs vs trader's LONG bias = -2,800 JPY).

Usage from bots:
    import thesis_gate
    blocked, reason = thesis_gate.check(pair, "BUY")
    if blocked:
        skip
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "collab_trade" / "state.md"
STATE_MAX_AGE_SEC = 25 * 60  # state.md older than 25 min = ignore

# Returns: { pair: {"long_ok": bool, "short_ok": bool, "reason": str} }
def parse_thesis() -> dict[str, dict]:
    if not STATE_PATH.exists():
        return {}
    try:
        text = STATE_PATH.read_text()
    except Exception:
        return {}

    # Check state.md freshness
    age = (datetime.now(timezone.utc).timestamp() - STATE_PATH.stat().st_mtime)
    if age > STATE_MAX_AGE_SEC:
        return {}  # stale → permissive

    out: dict[str, dict] = {}

    # Pattern: "Best vehicle NOW: AUD_USD LONG ..." or "Best vehicle: EUR_USD SHORT ..."
    # Pattern: "Second-best: GBP_USD LONG ..."
    best_lines = re.findall(
        r"(?:Best vehicle(?:\s+NOW)?|Second-best|Best expression NOW)\s*:\s*([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT|BUY|SELL)",
        text, re.IGNORECASE,
    )
    for pair, side in best_lines:
        side_norm = "LONG" if side.upper() in ("LONG", "BUY") else "SHORT"
        out.setdefault(pair, {"long_ok": True, "short_ok": True, "reason": ""})
        if side_norm == "LONG":
            out[pair] = {"long_ok": True, "short_ok": False, "reason": f"trader best vehicle = {pair} LONG"}
        else:
            out[pair] = {"long_ok": False, "short_ok": True, "reason": f"trader best vehicle = {pair} SHORT"}

    # Pattern: "Expressions to avoid: ANY SHORT on EUR/AUD" or "AUD_USD SHORT" or "EUR_USD LONG"
    # Look for "avoid:" line
    avoid_match = re.search(r"Expressions to avoid:\s*(.+?)(?:\n[A-Z]|\n##|\n\*\*|$)",
                            text, re.IGNORECASE | re.DOTALL)
    if avoid_match:
        avoid_text = avoid_match.group(1)
        # ANY SHORT on EUR/AUD/GBP — group blocks
        any_short = re.search(r"ANY\s+SHORT\s+on\s+([A-Z/_]+)", avoid_text, re.IGNORECASE)
        any_long = re.search(r"ANY\s+LONG\s+on\s+([A-Z/_]+)", avoid_text, re.IGNORECASE)
        currencies_short_blocked = set()
        currencies_long_blocked = set()
        if any_short:
            for c in re.findall(r"[A-Z]{3}", any_short.group(1)):
                currencies_short_blocked.add(c)
        if any_long:
            for c in re.findall(r"[A-Z]{3}", any_long.group(1)):
                currencies_long_blocked.add(c)
        for pair in ("USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY"):
            base, quote = pair.split("_")
            # SHORT on EUR_USD = selling EUR. If EUR is in short-block list, block SHORT.
            if base in currencies_short_blocked:
                cur = out.setdefault(pair, {"long_ok": True, "short_ok": True, "reason": ""})
                cur["short_ok"] = False
                cur["reason"] = (cur["reason"] + f" / avoid SHORT on {base}").strip(" /")
            if base in currencies_long_blocked:
                cur = out.setdefault(pair, {"long_ok": True, "short_ok": True, "reason": ""})
                cur["long_ok"] = False
                cur["reason"] = (cur["reason"] + f" / avoid LONG on {base}").strip(" /")

        # Pair-specific avoids: "USD_JPY SHORT" / "AUD_USD LONG"
        pair_avoids = re.findall(r"([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT|BUY|SELL)", avoid_text)
        for pair, side in pair_avoids:
            side_norm = "LONG" if side.upper() in ("LONG","BUY") else "SHORT"
            cur = out.setdefault(pair, {"long_ok": True, "short_ok": True, "reason": ""})
            if side_norm == "LONG":
                cur["long_ok"] = False
            else:
                cur["short_ok"] = False
            cur["reason"] = (cur["reason"] + f" / avoid {side_norm} on {pair}").strip(" /")

    return out


def check(pair: str, direction: str) -> tuple[bool, str]:
    """Return (blocked, reason). True = bot must skip this entry.

    direction can be BUY/SELL or LONG/SHORT.
    Permissive default: missing/stale state.md returns no-block.
    """
    side = (direction or "").upper()
    long_side = side in ("BUY", "LONG")
    thesis = parse_thesis()
    if not thesis:
        return False, ""
    pair_thesis = thesis.get(pair)
    if not pair_thesis:
        return False, ""  # no opinion on this pair
    if long_side and not pair_thesis.get("long_ok", True):
        return True, f"thesis_gate {pair_thesis.get('reason','contra-trader')}"
    if not long_side and not pair_thesis.get("short_ok", True):
        return True, f"thesis_gate {pair_thesis.get('reason','contra-trader')}"
    return False, ""


if __name__ == "__main__":
    # CLI debug: show parsed thesis
    import json
    print(json.dumps(parse_thesis(), indent=2))
