"""
Tools to (re)build the macro snapshot JSON consumed by MacroState.

The snapshot summarises lightweight macro context derived from:
 - Current technical factors (ATR, MA differentials) as a proxy for volatility / USD bias
 - Recent news summaries stored in ``logs/news.db`` for currency-specific sentiment
 - Upcoming high-impact events for gating

The generated file lives under ``fixtures/macro_snapshots/latest.json`` by default.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from analysis.macro_state import MacroSnapshot
from indicators.factor_cache import all_factors

NEWS_DB_PATH = Path("logs/news.db")
DEFAULT_SNAPSHOT_PATH = Path("fixtures/macro_snapshots/latest.json")
DEFAULT_WINDOW_HOURS = 24
DEFAULT_EVENT_LOOKAHEAD_HOURS = 12


def _normalise_pair_bias(pair_bias: str | None) -> Optional[Tuple[str, str, str]]:
    if not pair_bias:
        return None
    bias = pair_bias.strip().upper()
    if "_" in bias:
        pair, direction = bias.split("_", 1)
    else:
        pair, direction = bias[:6], bias[6:]
    pair = "".join(ch for ch in pair if ch.isalpha())
    direction = direction or "UP"
    if len(pair) != 6:
        return None
    return pair[:3], pair[3:], direction


def _aggregate_news_scores(
    db_path: Path,
    *,
    now: dt.datetime,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> Tuple[Dict[str, float], List[Tuple[str, str, str]]]:
    if not db_path.exists():
        return {}, []

    start_ts = (now - dt.timedelta(hours=window_hours)).isoformat()
    upcoming_limit = (now + dt.timedelta(hours=DEFAULT_EVENT_LOOKAHEAD_HOURS)).isoformat()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    scores: Dict[str, float] = {}
    cur.execute(
        """
        SELECT pair_bias, sentiment, impact
        FROM news
        WHERE ts_utc >= ?
        """,
        (start_ts,),
    )
    for row in cur.fetchall():
        parsed = _normalise_pair_bias(row["pair_bias"])
        if not parsed:
            continue
        base, quote, direction = parsed
        sentiment = row["sentiment"] or 0
        impact = row["impact"] or 1
        weight = float(sentiment)
        if abs(weight) < 1e-6:
            weight = 1.0 if direction.endswith("UP") else -1.0
        if direction.endswith("DOWN"):
            weight = -weight
        weight *= max(1.0, float(impact))
        scores[base] = scores.get(base, 0.0) + weight
        scores[quote] = scores.get(quote, 0.0) - weight

    # Normalise scores into [-1, 1]
    if scores:
        max_abs = max(abs(v) for v in scores.values()) or 1.0
        for key, val in list(scores.items()):
            scores[key] = round(max(-1.0, min(1.0, val / max_abs)), 4)

    # Upcoming events
    events: List[Tuple[str, str, str]] = []
    cur.execute(
        """
        SELECT event_time, summary, pair_bias, impact
        FROM news
        WHERE event_time IS NOT NULL
          AND event_time BETWEEN ? AND ?
        ORDER BY event_time ASC
        LIMIT 50
        """,
        (now.isoformat(), upcoming_limit),
    )
    for row in cur.fetchall():
        parsed = _normalise_pair_bias(row["pair_bias"])
        if not parsed:
            continue
        event_time = row["event_time"]
        summary = (row["summary"] or "").strip()
        summary = summary[:160] if summary else "Economic event"
        base, quote, _ = parsed
        for ccy in (base, quote):
            events.append((event_time, ccy, summary))

    conn.close()
    return scores, events


def _derive_vix_and_dxy(factors: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    fac_m1 = factors.get("M1") or {}
    fac_h4 = factors.get("H4") or {}
    atr_pips = float(fac_m1.get("atr_pips") or 10.0)
    close_m1 = float(fac_m1.get("close") or 0.0)
    ema20_m1 = float(fac_m1.get("ema20") or close_m1 or 0.0)
    close_h4 = float(fac_h4.get("close") or close_m1 or 0.0)
    ma20_h4 = float(fac_h4.get("ma20") or close_h4 or 0.0)

    # ATR proxy -> implied volatility
    if atr_pips <= 0 or close_m1 <= 0:
        vix = 18.0
    else:
        vix = max(10.0, min(45.0, 12.0 + atr_pips * 0.6))

    if close_h4 <= 0 or ma20_h4 <= 0:
        dxy = 0.0
    else:
        dxy = ((close_h4 - ma20_h4) / ma20_h4) * 100.0
    return round(vix, 3), round(dxy, 3)


def _derive_yields(scores: Dict[str, float], base_pairs: Iterable[str]) -> Dict[str, float]:
    yields: Dict[str, float] = {}
    for ccy in base_pairs:
        bias = scores.get(ccy, 0.0)
        yields[ccy] = round(0.5 + bias * 0.2, 4)
    return yields


def refresh_macro_snapshot(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    deadzone: float = 0.25,
    now: Optional[dt.datetime] = None,
    refresh_if_older_than_minutes: int = 10,
) -> Path:
    """
    Ensure the macro snapshot JSON exists and is reasonably recent.

    Returns the path to the refreshed snapshot.
    """
    snapshot_path = snapshot_path.expanduser().resolve()
    now = now or dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    needs_refresh = True
    if snapshot_path.exists():
        try:
            mtime = dt.datetime.fromtimestamp(snapshot_path.stat().st_mtime, tz=dt.timezone.utc)
            mtime_age_min = (now - mtime).total_seconds() / 60.0
        except Exception:
            mtime_age_min = refresh_if_older_than_minutes + 1
        asof_age_min = refresh_if_older_than_minutes + 1
        try:
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            asof_raw = existing.get("asof")
            if asof_raw:
                asof_dt = dt.datetime.fromisoformat(asof_raw.replace("Z", "+00:00"))
                asof_age_min = (now - asof_dt).total_seconds() / 60.0
        except Exception:
            asof_age_min = refresh_if_older_than_minutes + 1
        if (
            mtime_age_min <= refresh_if_older_than_minutes
            and asof_age_min <= refresh_if_older_than_minutes
        ):
            needs_refresh = False
    if not needs_refresh:
        return snapshot_path

    factors = all_factors()
    vix, dxy = _derive_vix_and_dxy(factors)

    news_scores, events = _aggregate_news_scores(NEWS_DB_PATH, now=now)
    base_currencies: set[str] = {"USD", "JPY"}
    base_currencies.update(news_scores.keys())
    yield_map = _derive_yields(news_scores, base_currencies)

    snapshot = MacroSnapshot(
        asof=now.isoformat(timespec="seconds"),
        vix=vix,
        dxy=dxy,
        yield2y=yield_map,
        news_ccy_score=news_scores,
        events=events,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
    return snapshot_path
