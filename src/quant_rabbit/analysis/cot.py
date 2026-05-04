"""CFTC Commitments of Traders (COT) — currency futures positioning.

Pulls the weekly disaggregated COT report from the CFTC's public site and
extracts net positioning of leveraged funds + asset managers in each major
currency future. The trader uses this to spot stretched positioning that
often precedes reversals.

Source: CFTC publishes the report every Friday at 15:30 ET. The CSV is at
https://www.cftc.gov/dea/newcot/deacot{YYYY}.txt or — easier to parse —
https://publicreporting.cftc.gov/resource/72hh-3qpy.csv (Socrata API).

We default to the Socrata endpoint with a `LIMIT 50` filter to keep the
download small. If unreachable we emit `MISSING_CFTC_FEED`.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Traders in Financial Futures (TFF) — designed for currencies / equities /
# bonds and exposes the leveraged-funds + asset-managers split we want.
CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.csv"

# Map CFTC market name → FX currency code
CFTC_MARKET_MAP: dict[str, str] = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "DXY",
    "USD INDEX - ICE FUTURES U.S.": "DXY",
}


@dataclass(frozen=True)
class COTReport:
    market: str
    currency: str
    report_date: str
    leveraged_long: int | None
    leveraged_short: int | None
    leveraged_net: int | None
    asset_mgr_long: int | None
    asset_mgr_short: int | None
    asset_mgr_net: int | None
    open_interest: int | None
    week_change_leveraged_net: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "currency": self.currency,
            "report_date": self.report_date,
            "leveraged_long": self.leveraged_long,
            "leveraged_short": self.leveraged_short,
            "leveraged_net": self.leveraged_net,
            "asset_mgr_long": self.asset_mgr_long,
            "asset_mgr_short": self.asset_mgr_short,
            "asset_mgr_net": self.asset_mgr_net,
            "open_interest": self.open_interest,
            "week_change_leveraged_net": self.week_change_leveraged_net,
        }


@dataclass(frozen=True)
class COTSnapshot:
    generated_at_utc: str
    source_url: str
    reports: tuple[COTReport, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "source_url": self.source_url,
            "reports": [r.to_dict() for r in self.reports],
            "issues": list(self.issues),
        }


def fetch_cot_csv(*, url: str = CFTC_URL, limit: int = 400, timeout: int = 30) -> bytes:
    """Pull the most recent rows from the TFF Socrata endpoint.

    A higher default limit (400) ensures we capture both the latest and prior
    week for every G10 currency future plus DXY — enough to compute the
    week-on-week change in leveraged-funds net positioning.
    """
    query = urlencode({"$limit": str(limit), "$order": "report_date_as_yyyy_mm_dd DESC"})
    full_url = f"{url}?{query}"
    req = Request(full_url, headers={"User-Agent": "QuantRabbit/1.0 (research)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_cot_csv(payload: bytes) -> tuple[COTReport, ...]:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    out: list[COTReport] = []
    seen_markets: set[str] = set()
    prev_net_by_market: dict[str, int] = {}
    rows = list(reader)
    # Group by market_and_exchange_names; rows are sorted desc by date.
    # The most recent row per market is the current report; the second is for week-on-week change.
    by_market: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        market = row.get("market_and_exchange_names") or row.get("market_and_exchange_name") or ""
        if market not in CFTC_MARKET_MAP:
            continue
        by_market.setdefault(market, []).append(row)

    for market, market_rows in by_market.items():
        market_rows.sort(key=lambda r: r.get("report_date_as_yyyy_mm_dd") or "", reverse=True)
        if not market_rows:
            continue
        latest = market_rows[0]
        prev = market_rows[1] if len(market_rows) > 1 else None

        def _i(row: dict[str, str], key: str) -> int | None:
            v = row.get(key)
            try:
                return int(float(v)) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None

        # TFF schema: lev_money_positions_long / lev_money_positions_short
        lev_long = _i(latest, "lev_money_positions_long")
        lev_short = _i(latest, "lev_money_positions_short")
        lev_net = (lev_long - lev_short) if (lev_long is not None and lev_short is not None) else None
        am_long = _i(latest, "asset_mgr_positions_long")
        am_short = _i(latest, "asset_mgr_positions_short")
        am_net = (am_long - am_short) if (am_long is not None and am_short is not None) else None
        oi = _i(latest, "open_interest_all")
        prev_lev_net = None
        if prev is not None:
            pll = _i(prev, "lev_money_positions_long")
            pls = _i(prev, "lev_money_positions_short")
            if pll is not None and pls is not None:
                prev_lev_net = pll - pls
        week_change = (lev_net - prev_lev_net) if (lev_net is not None and prev_lev_net is not None) else None

        out.append(COTReport(
            market=market,
            currency=CFTC_MARKET_MAP[market],
            report_date=latest.get("report_date_as_yyyy_mm_dd") or "",
            leveraged_long=lev_long,
            leveraged_short=lev_short,
            leveraged_net=lev_net,
            asset_mgr_long=am_long,
            asset_mgr_short=am_short,
            asset_mgr_net=am_net,
            open_interest=oi,
            week_change_leveraged_net=week_change,
        ))
    return tuple(out)


def build_cot_snapshot(*, fetch: bool = True, url: str = CFTC_URL) -> COTSnapshot:
    issues: list[str] = []
    reports: tuple[COTReport, ...] = tuple()
    if fetch:
        try:
            payload = fetch_cot_csv(url=url)
            reports = parse_cot_csv(payload)
        except (URLError, OSError, ValueError) as exc:
            issues.append(f"MISSING_CFTC_FEED: {exc}")
    return COTSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_url=url,
        reports=reports,
        issues=tuple(issues),
    )
