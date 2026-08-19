#!/usr/bin/env python3
"""Pre-trade arithmetic for the frozen scalp rule. Read-only.

    tools/scalp_session.py                 # status + max size + stop prices
    tools/scalp_session.py --pair EUR_USD
    tools/scalp_session.py --target 0.03   # trades/day needed for 3%/day

The rule in `research/paper/scalp_preregistration_v1.json` is only worth
anything if it is followed at the moment of entry, which is exactly when
arithmetic gets skipped. This does the arithmetic beforehand: the largest size
the margin cap allows, the stop price for either direction, and the JPY at risk.

It also audits what is already open against the same rule, because the frozen
cap applies to the book, not just to the next trade.

Read-only: no order is ever constructed. Placing the trade and its stop is the
operator's action.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "research" / "paper" / "scalp_preregistration_v1.json"
ENV = Path("/Users/tossaki/App/QuantRabbit/.env.local")
LEVERAGE = 25.0


def creds():
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip()
    return env["QR_OANDA_TOKEN"], env["QR_OANDA_ACCOUNT_ID"], env.get(
        "QR_OANDA_BASE_URL", "https://api-fxtrade.oanda.com")


def get(base, token, path, query=None):
    url = f"{base}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


class Rates:
    """Convert base- and quote-currency amounts into JPY.

    A pip on EUR_USD is worth USD, not JPY; margin on EUR_USD is EUR, not JPY.
    Getting this wrong silently mis-sizes every non-JPY pair, so both legs are
    converted explicitly from live mid quotes.
    """

    def __init__(self, base, token, acct):
        self.base, self.token, self.acct = base, token, acct
        self._mid = {}

    def mid(self, pair: str) -> float | None:
        if pair in self._mid:
            return self._mid[pair]
        try:
            p = get(self.base, self.token, f"/v3/accounts/{self.acct}/pricing",
                    {"instruments": pair})["prices"][0]
            v = (float(p["bids"][0]["price"]) + float(p["asks"][0]["price"])) / 2
        except Exception:
            v = None
        self._mid[pair] = v
        return v

    def to_jpy(self, ccy: str) -> float | None:
        """JPY per one unit of ccy."""
        if ccy == "JPY":
            return 1.0
        v = self.mid(f"{ccy}_JPY")
        if v:
            return v
        v = self.mid(f"JPY_{ccy}")
        return 1.0 / v if v else None

    def pip_value_jpy(self, pair: str, units: float) -> float | None:
        """JPY moved per pip, for `units` of `pair`. Pip is in the quote currency."""
        q = pair.split("_")[1]
        conv = self.to_jpy(q)
        return abs(units) * pip_size(pair) * conv if conv else None

    def margin_jpy(self, pair: str, units: float, price: float) -> float | None:
        """Margin is notional in the BASE currency, converted to JPY."""
        b = pair.split("_")[0]
        conv = self.to_jpy(b)
        return abs(units) * conv / LEVERAGE if conv else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--target", type=float, default=None, help="daily return target, e.g. 0.03")
    args = ap.parse_args(argv)

    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    stop_pips = reg["rule"]["stop_distance_pips"]
    cap = reg["rule"]["max_margin_used_fraction"]
    token, acct, base = creds()
    rates = Rates(base, token, acct)

    summary = get(base, token, f"/v3/accounts/{acct}/summary")["account"]
    nav = float(summary["NAV"])
    used = float(summary["marginUsed"])
    avail = float(summary["marginAvailable"])
    closeout = float(summary.get("marginCloseoutPercent", 0))

    print(f"=== scalp session / stop -{stop_pips:.0f}p / margin cap {cap:.0%} ===")
    print(f"NAV {nav:,.0f}   margin used {used:,.0f} ({used/nav:.0%})   "
          f"available {avail:,.0f}   closeoutPct {closeout:.3f}")
    if used / nav > cap:
        print(f"  ** BOOK OVER THE CAP: {used/nav:.0%} > {cap:.0%}. "
              f"A new scalp under the rule cannot be opened until this comes down. **")

    print(f"\n--- open positions vs the rule ---")
    trades = get(base, token, f"/v3/accounts/{acct}/openTrades")["trades"]
    if not trades:
        print("  none")
    for t in trades:
        pair, u, px = t["instrument"], int(float(t["currentUnits"])), float(t["price"])
        pv = rates.pip_value_jpy(pair, u)
        marg = rates.margin_jpy(pair, u, px)
        upl = float(t["unrealizedPL"])
        pips = upl / pv if pv else float("nan")
        bad = []
        if not t.get("stopLossOrder"):
            bad.append("NO STOP")
        if marg and marg / nav > cap:
            bad.append(f"MARGIN {marg/nav:.0%}")
        room = avail / pv if pv else float("nan")
        print(f"  {pair} {u:+,} @ {px}  uPL {upl:+,.0f} ({pips:+.1f}p)  "
              f"margin {marg:,.0f} ({marg/nav:.0%})  liq ~{room:.0f}p away"
              + (f"   ** {' / '.join(bad)} **" if bad else "   ok"))

    pair = args.pair
    q = get(base, token, f"/v3/accounts/{acct}/pricing", {"instruments": pair})["prices"][0]
    bid, ask = float(q["bids"][0]["price"]), float(q["asks"][0]["price"])
    pip = pip_size(pair)
    print(f"\n--- {pair} now: bid {bid} / ask {ask} (spread {(ask-bid)/pip:.1f}p) ---")

    headroom = max(0.0, nav * cap - used)
    conv_b = rates.to_jpy(pair.split("_")[0])
    max_units = int(headroom * LEVERAGE / conv_b) if conv_b else 0
    max_units = (max_units // 1000) * 1000
    print(f"  margin headroom under the cap: {headroom:,.0f} JPY")
    print(f"  max size the rule allows now : {max_units:,} units")
    if max_units <= 0:
        print("  -> zero. Reduce the open book before the next scalp.")
        return 0

    pv = rates.pip_value_jpy(pair, max_units)
    print(f"  risk at -{stop_pips:.0f}p on that size: {pv*stop_pips:,.0f} JPY "
          f"({pv*stop_pips/nav:.2%} of NAV)")
    print(f"    LONG  entry {ask}  ->  stop {ask - stop_pips*pip:.3f}")
    print(f"    SHORT entry {bid}  ->  stop {bid + stop_pips*pip:.3f}")

    edge = reg["prior_estimate_do_not_reuse_as_evidence"]["mean_pips"]
    print(f"\n--- rate needed (at the FITTED {edge:+.2f}p/trade — not yet evidence) ---")
    for tgt in ([args.target] if args.target else [0.01, 0.03, 0.05]):
        need_jpy = nav * tgt
        per = pv * edge
        print(f"  {tgt:.0%}/day = {need_jpy:,.0f} JPY -> {need_jpy/per:.1f} trades/day at {max_units:,} units")
    print(f"\n  evidence still required: {reg['required_n']} clean trades "
          f"(run tools/scalp_verify.py). The rule is frozen; do not retune it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
