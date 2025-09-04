from __future__ import annotations

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

import httpx

from utils.secrets import get_secret
import logging


def _oanda_host() -> str:
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except Exception:
        practice = True
    return "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"

def _alt_host(current: str) -> str:
    return "https://api-fxtrade.oanda.com" if "practice" in current else "https://api-fxpractice.oanda.com"


def _oanda_headers() -> Dict[str, str]:
    token = get_secret("oanda_token")
    return {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }


def _oanda_account() -> str:
    return get_secret("oanda_account_id")


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_transactions_since(start: datetime) -> List[Dict[str, Any]]:
    """Fetch OANDA transactions since the given UTC datetime (inclusive).

    Returns a list of transaction dicts.
    """
    host = _oanda_host()
    account = _oanda_account()
    headers = _oanda_headers()
    params = {"from": _iso_utc(start)}

    url = f"{host}/v3/accounts/{account}/transactions"
    with httpx.Client(timeout=15.0) as client:
        for h in (host, _alt_host(host)):
            try:
                r = client.get(f"{h}/v3/accounts/{account}/transactions", params={"from": "1"}, headers=headers)
                r.raise_for_status()
                meta = r.json() or {}
                pages = meta.get("pages", [])
                if not pages:
                    continue
                # Iterate from newest to older until we fall before start
                # Focus on the most recent 6 pages first (fast path)
                candidate_pages = list(reversed(pages))[:6]
                out: List[Dict[str, Any]] = []
                scanned = 0
                for purl in candidate_pages:
                    try:
                        rp = client.get(purl, headers=headers)
                        rp.raise_for_status()
                        data = rp.json() or {}
                        txs = data.get("transactions", [])
                        if not txs:
                            continue
                        # if whole page newer than start, keep all; if older, keep those >= start and break
                        first_time = _parse_time(txs[0].get("time"))
                        last_time = _parse_time(txs[-1].get("time"))
                        for t in txs:
                            tt = _parse_time(t.get("time"))
                            if tt >= start:
                                out.append(t)
                        if last_time < start:
                            break
                    except Exception as e:
                        logging.debug(f"idrange fetch failed: {e}")
                        continue
                if out:
                    # sort by time
                    out.sort(key=lambda t: _parse_time(t.get("time")))
                    return out
            except Exception as e:
                logging.debug(f"meta pages fetch failed: {e}")
                continue
        return []


def _parse_time(s: str | None) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        # Normalize RFC3339 with up to nanoseconds to python-compatible microseconds
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        if '.' in s:
            head, rest = s.split('.', 1)
            if '+' in rest:
                frac, tz = rest.split('+', 1)
                frac = (frac + '000000')[:6]
                s = f"{head}.{frac}+{tz}"
            elif '-' in rest:
                frac, tz = rest.split('-', 1)
                frac = (frac + '000000')[:6]
                s = f"{head}.{frac}-{tz}"
            else:
                # no timezone, unlikely from OANDA
                frac = (rest + '000000')[:6]
                s = f"{head}.{frac}"
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def compute_daily_pnl(day: datetime | None = None) -> Dict[str, Any]:
    """Compute realized daily PnL from OANDA transactions (UTC day).

    Returns dict with realized components and simple counts.
    """
    if day is None:
        day = datetime.utcnow().date()
        day = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    elif isinstance(day, datetime) and day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)

    start = day
    end = day + timedelta(days=1)
    txs = fetch_transactions_since(start)

    realized_pl = 0.0
    financing = 0.0
    commission = 0.0
    fees = 0.0
    fills = 0
    win = 0
    lose = 0
    pl_pos = 0.0
    pl_neg = 0.0

    for t in txs:
        # Filter to the target date (UTC) and relevant types
        t_dt = _parse_time(t.get("time"))
        if t_dt is None or not (start <= t_dt < end):
            continue

        typ = t.get("type", "")
        if typ not in ("ORDER_FILL", "TRADE_CLOSE", "TRADE_SETTLEMENT"):
            continue

        # Sum realized metrics (fields may be strings)
        def f(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        pl = f(t.get("pl"))
        fin = f(t.get("financing"))
        com = f(t.get("commission"))
        gef = f(t.get("guaranteedExecutionFee"))

        realized_pl += pl
        if pl > 0:
            pl_pos += pl
        elif pl < 0:
            pl_neg += pl
        financing += fin
        commission += com
        fees += gef

        if typ == "ORDER_FILL":
            fills += 1
            if pl > 0:
                win += 1
            elif pl < 0:
                lose += 1

    net = realized_pl + financing - abs(commission) - abs(fees)
    return {
        "date": start.date().isoformat(),
        "realized_pl": round(realized_pl, 2),
        "financing": round(financing, 2),
        "commission": round(commission, 2),
        "fees": round(fees, 2),
        "net_pl": round(net, 2),
        "fills": fills,
        "win": win,
        "lose": lose,
        "pl_pos": round(pl_pos, 2),
        "pl_neg": round(pl_neg, 2),
    }
