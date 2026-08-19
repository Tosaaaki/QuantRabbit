"""Non-price inputs for the operator model — the one axis still open.

Fifteen months of measurement closed every price-derived axis: 13,120 indicator
combinations reached +1.89 pips against a +3.59 requirement, a 60,000-cell frame
sweep came back indistinguishable from its null, and the surviving quiet gate
turned out to be a pooling artifact. What was never tested is information that
is not in the price at all — positioning, the event calendar, and news.

This module starts with **CFTC Commitments of Traders**, because it is free,
public, structured, weekly, and — unlike news — it has a complete history, so
its contribution can be measured against the 464 existing entries today instead
of waiting out a forward test.

COT is published Friday for the preceding Tuesday, so a report is only usable
from the Friday release onward. `as_of` enforces that: a lookup at time T
returns the most recent report whose release date is strictly before T. Getting
this wrong would leak three days of future positioning into every feature.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOCRATA = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CACHE = Path(__file__).resolve().parents[2] / "research" / "operator_model" / "cot_cache.json"

# The currency leg each contract speaks for. A pair's positioning feature is the
# base leg's skew minus the quote leg's, so USD_JPY reads as "dollar minus yen".
CONTRACTS = {
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    # renamed at the CFTC in 2022; the old "NEW ZEALAND DOLLAR" series stops
    # at 2022-02-01 and silently returns nothing
    "NZD": "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
}
# The dollar has no usable series: the ICE dollar-index contract also stops at
# 2022-02-01 in this dataset. But every contract above is quoted against USD, so
# being long yen futures IS being short dollars — the dollar's skew is the
# open-interest-weighted negative of the rest, which is how it is derived here
# rather than left as a zero that would silently blank every USD pair.
# Tuesday snapshot, released the following Friday at 15:30 ET (19:30 UTC).
RELEASE_LAG_DAYS = 3
RELEASE_HOUR_UTC = 20


def _fetch(name: str, since: str = "2025-01-01") -> list[dict]:
    query = urllib.parse.urlencode({
        "$limit": "600",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$where": f"market_and_exchange_names='{name}' "
                  f"AND report_date_as_yyyy_mm_dd > '{since}T00:00:00.000'",
    })
    with urllib.request.urlopen(f"{SOCRATA}?{query}", timeout=60) as resp:
        return json.loads(resp.read())


def load(refresh: bool = False) -> dict[str, list[dict]]:
    """Per-currency COT history: report date, release datetime, and the skew."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for ccy, name in CONTRACTS.items():
        rows = []
        for r in _fetch(name):
            try:
                long_ = float(r["noncomm_positions_long_all"])
                short = float(r["noncomm_positions_short_all"])
                oi = float(r["open_interest_all"])
            except (KeyError, TypeError, ValueError):
                continue
            if oi <= 0:
                continue
            report = datetime.fromisoformat(r["report_date_as_yyyy_mm_dd"]).replace(tzinfo=timezone.utc)
            rows.append({
                "report_utc": report.isoformat().replace("+00:00", "Z"),
                "release_utc": (report + timedelta(days=RELEASE_LAG_DAYS)
                                ).replace(hour=RELEASE_HOUR_UTC).isoformat().replace("+00:00", "Z"),
                # speculative skew, normalised by open interest so contracts of
                # very different size are comparable
                "skew": round((long_ - short) / oi, 6),
                "oi": oi,
            })
        if rows:
            out[ccy] = rows
    out["USD"] = _derive_usd(out)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


def _derive_usd(cot: dict[str, list[dict]]) -> list[dict]:
    """USD skew as the OI-weighted negative of every other currency."""
    by_report: dict[str, list[tuple[float, float]]] = {}
    release: dict[str, str] = {}
    for ccy, rows in cot.items():
        for r in rows:
            by_report.setdefault(r["report_utc"], []).append((r["skew"], r["oi"]))
            release[r["report_utc"]] = r["release_utc"]
    out = []
    for report in sorted(by_report):
        legs = by_report[report]
        total = sum(oi for _s, oi in legs)
        if total <= 0 or len(legs) < 3:
            continue
        out.append({"report_utc": report, "release_utc": release[report],
                    "skew": round(-sum(s * oi for s, oi in legs) / total, 6),
                    "oi": total})
    return out


def as_of(history: list[dict], at: datetime) -> dict | None:
    """The latest report actually released before `at`. Never leaks."""
    usable = [r for r in history
              if datetime.fromisoformat(r["release_utc"].replace("Z", "+00:00")) < at]
    return usable[-1] if usable else None


def features(pair: str, at: datetime, cot: dict | None = None) -> dict:
    """Positioning features for one pair at one moment.

    `cot_skew` is base-leg skew minus quote-leg skew: for USD_JPY, how long the
    speculative crowd is the dollar relative to the yen. `cot_z` is that against
    its own trailing year, so "stretched" means stretched for this pair rather
    than for the dataset. Missing legs return zeros with a `has_` flag rather
    than being dropped, so a pair without a contract is visibly absent instead
    of silently imputed.
    """
    cot = cot if cot is not None else load()
    base, quote = pair.split("_")
    out: dict[str, float] = {}
    legs = {}
    for role, ccy in (("base", base), ("quote", quote)):
        hist = cot.get(ccy)
        rec = as_of(hist, at) if hist else None
        legs[role] = rec
        out[f"cot_{role}_skew"] = rec["skew"] if rec else 0.0
        out[f"cot_has_{role}"] = 1.0 if rec else 0.0
    if legs["base"] and legs["quote"]:
        out["cot_skew"] = round(legs["base"]["skew"] - legs["quote"]["skew"], 6)
        # z-score of the differential against the trailing 52 reports
        hb, hq = cot[base], cot[quote]
        cut = at
        sb = [r["skew"] for r in hb
              if datetime.fromisoformat(r["release_utc"].replace("Z", "+00:00")) < cut][-52:]
        sq = [r["skew"] for r in hq
              if datetime.fromisoformat(r["release_utc"].replace("Z", "+00:00")) < cut][-52:]
        n = min(len(sb), len(sq))
        if n >= 20:
            diff = [a - b for a, b in zip(sb[-n:], sq[-n:])]
            mu = sum(diff) / n
            sd = (sum((d - mu) ** 2 for d in diff) / n) ** 0.5
            out["cot_z"] = round((out["cot_skew"] - mu) / sd, 4) if sd > 0 else 0.0
            # one-week change: is the crowd adding or unwinding
            out["cot_delta"] = round(diff[-1] - diff[-2], 6) if n >= 2 else 0.0
        else:
            out["cot_z"] = 0.0
            out["cot_delta"] = 0.0
    else:
        out["cot_skew"] = 0.0
        out["cot_z"] = 0.0
        out["cot_delta"] = 0.0
    return out
