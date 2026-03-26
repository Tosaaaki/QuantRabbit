#!/usr/bin/env python3
"""Print current OANDA account snapshot for manual swing sizing."""

from __future__ import annotations

from datetime import datetime, timezone

from utils.oanda_account import get_account_snapshot


def main() -> None:
    snap = get_account_snapshot()
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] Account snapshot")
    print(f"  NAV                : {snap.nav:,.2f} JPY")
    print(f"  Balance            : {snap.balance:,.2f} JPY")
    print(f"  Margin Available   : {snap.margin_available:,.2f} JPY")
    print(f"  Margin Used        : {snap.margin_used:,.2f} JPY")
    print(f"  Margin Rate        : {snap.margin_rate:,.5f}")
    if snap.free_margin_ratio is not None:
        print(f"  Free Margin Ratio  : {snap.free_margin_ratio:.4f}")
    if snap.health_buffer is not None:
        print(f"  Health Buffer      : {snap.health_buffer:.4f}")
    print(f"  Unrealized P/L     : {snap.unrealized_pl:,.2f} JPY")


if __name__ == "__main__":
    main()
