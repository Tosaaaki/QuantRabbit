#!/usr/bin/env python3
"""Quantify how often losing closes later trade back to breakeven or better.

This is a diagnostic for "got shaken out in chop" versus "the thesis was simply
wrong." It joins the local trades table with OANDA close timestamps, then checks
the best favorable move after each losing close using OANDA M1 mid candles.

Usage:
  python3 tools/post_close_regret.py --jst-date-from 2026-04-20
  python3 tools/post_close_regret.py --jst-date-from 2026-04-20 --hours 6 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean

from config_loader import get_oanda_config


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "collab_trade" / "memory" / "memory.db"
UTC = timezone.utc
JST = timezone(timedelta(hours=9))


@dataclass
class LossTrade:
    session_date: str
    trade_id: str
    pair: str
    direction: str
    units: int
    entry_price: float
    exit_price: float
    pl: float
    regime: str
    reason: str


@dataclass
class PretradeMeta:
    session_bucket: str
    regime_snapshot: str
    execution_style: str
    thesis_family: str
    thesis_key: str
    allocation_band: str
    live_tape_bias: str
    live_tape_state: str
    live_tape_bucket: str


def _parse_jst_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_oanda_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    if "." not in normalized:
        return datetime.fromisoformat(normalized)
    main, frac_and_tz = normalized.split(".", 1)
    frac = frac_and_tz
    tz = ""
    for marker in ("+", "-"):
        if marker in frac_and_tz:
            frac, tz = frac_and_tz.split(marker, 1)
            tz = marker + tz
            break
    frac = (frac[:6]).ljust(6, "0")
    return datetime.fromisoformat(f"{main}.{frac}{tz}")


def _load_loss_trades(session_date_from: str | None) -> dict[str, LossTrade]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT session_date, trade_id, pair, direction, units, entry_price, exit_price, pl,
               COALESCE(regime, '') AS regime,
               COALESCE(reason, '') AS reason
        FROM trades
        WHERE pl < 0
          AND trade_id IS NOT NULL
          AND exit_price IS NOT NULL
    """
    params: list[str] = []
    if session_date_from:
        sql += " AND session_date >= ?"
        params.append(session_date_from)
    sql += " ORDER BY session_date, id"
    rows = conn.execute(sql, tuple(params)).fetchall()
    return {
        str(row["trade_id"]): LossTrade(
            session_date=row["session_date"],
            trade_id=str(row["trade_id"]),
            pair=row["pair"],
            direction=row["direction"],
            units=int(row["units"] or 0),
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            pl=float(row["pl"]),
            regime=row["regime"],
            reason=row["reason"],
        )
        for row in rows
    }


def _load_pretrade_meta(session_date_from: str | None) -> dict[str, PretradeMeta]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT trade_id,
               COALESCE(session_bucket, '') AS session_bucket,
               COALESCE(regime_snapshot, '') AS regime_snapshot,
               COALESCE(execution_style, '') AS execution_style,
               COALESCE(thesis_family, '') AS thesis_family,
               COALESCE(thesis_key, '') AS thesis_key,
               COALESCE(allocation_band, '') AS allocation_band,
               COALESCE(live_tape_bias, '') AS live_tape_bias,
               COALESCE(live_tape_state, '') AS live_tape_state,
               COALESCE(live_tape_bucket, '') AS live_tape_bucket,
               id
        FROM pretrade_outcomes
        WHERE trade_id IS NOT NULL
    """
    params: list[str] = []
    if session_date_from:
        sql += " AND session_date >= ?"
        params.append(session_date_from)
    sql += " ORDER BY id DESC"
    rows = conn.execute(sql, tuple(params)).fetchall()
    meta: dict[str, PretradeMeta] = {}
    for row in rows:
        trade_id = str(row["trade_id"])
        if trade_id in meta:
            continue
        meta[trade_id] = PretradeMeta(
            session_bucket=row["session_bucket"],
            regime_snapshot=row["regime_snapshot"],
            execution_style=row["execution_style"],
            thesis_family=row["thesis_family"],
            thesis_key=row["thesis_key"],
            allocation_band=row["allocation_band"],
            live_tape_bias=row["live_tape_bias"],
            live_tape_state=row["live_tape_state"],
            live_tape_bucket=row["live_tape_bucket"],
        )
    return meta


def _oanda_request(path_or_url: str, cfg: dict[str, object]) -> dict:
    if path_or_url.startswith("http"):
        url = path_or_url
    else:
        url = f"{cfg['oanda_base_url']}{path_or_url}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def _fetch_close_times(
    trades: dict[str, LossTrade],
    cfg: dict[str, object],
    fetch_from_utc: datetime,
) -> dict[str, dict[str, str | float]]:
    acct = str(cfg["oanda_account_id"])
    params = urllib.parse.urlencode(
        {
            "from": fetch_from_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "type": "ORDER_FILL",
            "pageSize": 1000,
        }
    )
    start = _oanda_request(f"/v3/accounts/{acct}/transactions?{params}", cfg)
    transactions: list[dict] = []
    for page_url in start.get("pages", []):
        transactions.extend(_oanda_request(page_url, cfg).get("transactions", []))

    closes: dict[str, dict[str, str | float]] = {}
    for tx in transactions:
        if tx.get("type") != "ORDER_FILL":
            continue
        tx_time = tx.get("time", "")
        for closed in tx.get("tradesClosed", []) + tx.get("tradesReduced", []):
            tid = str(closed.get("tradeID", ""))
            if tid not in trades:
                continue
            previous = closes.get(tid)
            if previous is None or tx_time > str(previous["time"]):
                closes[tid] = {
                    "time": tx_time,
                    "reason": tx.get("reason", ""),
                    "price": float(tx.get("price", 0.0)),
                }
    return closes


def _fetch_m1_candles(
    pair: str,
    start_utc: datetime,
    end_utc: datetime,
    cfg: dict[str, object],
    cache: dict[tuple[str, str, str], list[dict]],
) -> list[dict]:
    if end_utc <= start_utc:
        return []
    key = (
        pair,
        start_utc.strftime("%Y%m%d%H%M%S"),
        end_utc.strftime("%Y%m%d%H%M%S"),
    )
    if key in cache:
        return cache[key]
    params = urllib.parse.urlencode(
        {
            "price": "M",
            "granularity": "M1",
            "from": start_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": end_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
    )
    data = _oanda_request(f"/v3/instruments/{pair}/candles?{params}", cfg)
    candles = []
    for candle in data.get("candles", []):
        mid = candle.get("mid") or {}
        if not mid:
            continue
        candles.append(
            {
                "time": candle["time"],
                "dt": _parse_oanda_time(candle["time"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
            }
        )
    cache[key] = candles
    return candles


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("JPY") else 10000


def _classify_collapse_component(reason: str, close_reason: str) -> str:
    text = " ".join((reason or "", close_reason or "")).lower()
    if any(token in text for token in ("spread", "slippage", "friction", "too wide")):
        return "vehicle_friction"
    if any(token in text for token in ("stale", "zombie", "recycle", "aged", "wish_distance")):
        return "stale_thesis"
    if any(token in text for token in ("acceptance_above_entry", "accept", "body break", "structure", "shelf broke")):
        return "structure_break"
    if any(
        token in text
        for token in (
            "m1_pulse_flip",
            "no_first_confirmation",
            "no_reclaim",
            "shelf_fail",
            "no_rebuy",
            "failed_break",
            "failed_floor",
            "retest",
            "reclaim",
        )
    ):
        return "trigger_timing_fail"
    if "stop_loss_order" in text:
        return "hard_invalidation"
    if "market_order_trade_close" in text:
        return "manual_discretionary_cut"
    return "unknown"


def _thesis_lens_label(meta: PretradeMeta | None) -> str:
    if meta is None:
        return "unknown"
    family = meta.thesis_family or ""
    parts = [part for part in family.split("|") if part]
    if len(parts) >= 6:
        archetype = parts[2]
        wave = parts[3].replace("wave_", "")
        session_bucket = parts[4].replace("session_", "")
        regime = parts[5].replace("regime_", "")
        return f"{archetype.lower()} | {wave} | {session_bucket}/{regime}"
    session_bucket = meta.session_bucket or "unknown"
    regime = meta.regime_snapshot or "unknown"
    style = meta.execution_style or "unknown"
    return f"{style.lower()} | {session_bucket}/{regime}"


def _favorable_pips(trade: LossTrade, candles: list[dict]) -> float:
    if not candles:
        return 0.0
    pip_factor = _pip_factor(trade.pair)
    if trade.direction == "LONG":
        return (max(c["high"] for c in candles) - trade.exit_price) * pip_factor
    return (trade.exit_price - min(c["low"] for c in candles)) * pip_factor


def _recovered_to_entry(trade: LossTrade, candles: list[dict]) -> tuple[bool, str | None]:
    for candle in candles:
        if trade.direction == "LONG" and candle["high"] >= trade.entry_price:
            return True, candle["time"]
        if trade.direction == "SHORT" and candle["low"] <= trade.entry_price:
            return True, candle["time"]
    return False, None


def _build_result_rows(
    trades: dict[str, LossTrade],
    closes: dict[str, dict[str, str | float]],
    pretrade_meta: dict[str, PretradeMeta],
    cfg: dict[str, object],
    *,
    hours: int,
    jst_date_from: date | None,
    limit: int | None,
) -> list[dict]:
    cache: dict[tuple[str, str, str], list[dict]] = {}
    results: list[dict] = []
    now_utc = datetime.now(UTC)

    for trade_id, trade in trades.items():
        close = closes.get(trade_id)
        if close is None:
            continue
        close_dt = _parse_oanda_time(str(close["time"]))
        close_jst = close_dt.astimezone(JST)
        if jst_date_from and close_jst.date() < jst_date_from:
            continue
        meta = pretrade_meta.get(trade_id)
        end_utc = min(close_dt + timedelta(hours=hours), now_utc)
        candles = _fetch_m1_candles(trade.pair, close_dt, end_utc, cfg, cache)
        favorable = _favorable_pips(trade, candles)
        recovered, recovered_at = _recovered_to_entry(trade, candles)
        loss_pips = abs(trade.entry_price - trade.exit_price) * _pip_factor(trade.pair)
        results.append(
            {
                "session_date": trade.session_date,
                "trade_id": trade.trade_id,
                "pair": trade.pair,
                "direction": trade.direction,
                "units": trade.units,
                "pl": round(trade.pl, 1),
                "regime": trade.regime or "unknown",
                "reason": trade.reason,
                "close_reason": close["reason"],
                "collapse_component": _classify_collapse_component(trade.reason, str(close["reason"])),
                "session_bucket": meta.session_bucket if meta else "unknown",
                "regime_snapshot": meta.regime_snapshot if meta else "unknown",
                "execution_style": meta.execution_style if meta else "unknown",
                "allocation_band": meta.allocation_band if meta else "unknown",
                "live_tape_bias": (meta.live_tape_bias if meta else "") or "unknown",
                "live_tape_state": (meta.live_tape_state if meta else "") or "unknown",
                "live_tape_bucket": (meta.live_tape_bucket if meta else "") or "unknown",
                "thesis_family": meta.thesis_family if meta else "",
                "thesis_lens": _thesis_lens_label(meta),
                "close_time_utc": close_dt.strftime("%Y-%m-%d %H:%M UTC"),
                "close_time_jst": close_jst.strftime("%Y-%m-%d %H:%M JST"),
                "loss_pips": round(loss_pips, 1),
                "fav_pips": round(favorable, 1),
                "recovered": recovered,
                "recovered_at": (
                    _parse_oanda_time(recovered_at)
                    .astimezone(JST)
                    .strftime("%Y-%m-%d %H:%M JST")
                    if recovered_at
                    else None
                ),
            }
        )

    results.sort(key=lambda row: row["close_time_jst"], reverse=True)
    if limit:
        return results[:limit]
    return results


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "count": 0,
            "recovered": 0,
            "recovery_rate": 0.0,
            "avg_loss_pips": 0.0,
            "avg_fav_pips": 0.0,
        }
    recovered = sum(1 for row in results if row["recovered"])
    return {
        "count": len(results),
        "recovered": recovered,
        "recovery_rate": round(recovered / len(results) * 100, 1),
        "avg_loss_pips": round(mean(row["loss_pips"] for row in results), 1),
        "avg_fav_pips": round(mean(row["fav_pips"] for row in results), 1),
    }


def _summarize_by_key(results: list[dict], key: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        buckets[str(row[key])].append(row)
    summary = []
    for bucket_key, rows in buckets.items():
        recovered = sum(1 for row in rows if row["recovered"])
        summary.append(
            {
                key: bucket_key,
                "count": len(rows),
                "recovered": recovered,
                "recovery_rate": round(recovered / len(rows) * 100, 1),
                "avg_loss_pips": round(mean(row["loss_pips"] for row in rows), 1),
                "avg_fav_pips": round(mean(row["fav_pips"] for row in rows), 1),
            }
        )
    return sorted(summary, key=lambda row: (-row["recovery_rate"], -row["count"], row[key]))


def _render_text(results: list[dict], summary: dict, by_regime: list[dict], hours: int) -> str:
    lines = []
    lines.append(
        f"Post-close regret ({hours}h window): {summary['recovered']}/{summary['count']} "
        f"recovered ({summary['recovery_rate']}%). avg loss {summary['avg_loss_pips']}pip | "
        f"avg later favorable {summary['avg_fav_pips']}pip."
    )
    if by_regime:
        lines.append("")
        lines.append("By regime:")
        for row in by_regime:
            lines.append(
                f"- {row['regime']}: {row['recovered']}/{row['count']} "
                f"({row['recovery_rate']}%) | avg loss {row['avg_loss_pips']}pip | "
                f"avg favorable {row['avg_fav_pips']}pip"
            )
    top_recovered = sorted(
        [row for row in results if row["recovered"]],
        key=lambda row: (row["fav_pips"] - row["loss_pips"], row["fav_pips"]),
        reverse=True,
    )[:8]
    top_failed = sorted(
        [row for row in results if not row["recovered"]],
        key=lambda row: (row["loss_pips"] - row["fav_pips"], row["loss_pips"]),
        reverse=True,
    )[:8]
    if top_recovered:
        lines.append("")
        lines.append("Recovered after close:")
        for row in top_recovered:
            lines.append(
                f"- {row['close_time_jst']} {row['pair']} {row['direction']} id={row['trade_id']} "
                f"loss {row['loss_pips']}pip -> later {row['fav_pips']}pip | "
                f"{row['collapse_component']} | recovered at {row['recovered_at']}"
            )
    if top_failed:
        lines.append("")
        lines.append("Still bad after close:")
        for row in top_failed:
            lines.append(
                f"- {row['close_time_jst']} {row['pair']} {row['direction']} id={row['trade_id']} "
                f"loss {row['loss_pips']}pip -> later {row['fav_pips']}pip | "
                f"{row['collapse_component']} | regime {row['regime']}"
            )
    return "\n".join(lines)


def build_regret_result_map(
    session_date_from: str | None = None,
    *,
    jst_date_from: str | None = None,
    hours: int = 6,
    limit: int | None = None,
) -> dict[str, dict]:
    trades = _load_loss_trades(session_date_from)
    if not trades:
        return {}

    pretrade_meta = _load_pretrade_meta(session_date_from)
    cfg = get_oanda_config()
    if jst_date_from:
        start_jst = _parse_jst_date(jst_date_from)
        fetch_from_utc = datetime.combine(start_jst, time.min, tzinfo=JST).astimezone(UTC) - timedelta(days=1)
        jst_filter = start_jst
    elif session_date_from:
        fetch_from_utc = datetime.strptime(session_date_from, "%Y-%m-%d").replace(tzinfo=UTC) - timedelta(days=1)
        jst_filter = None
    else:
        fetch_from_utc = datetime.now(UTC) - timedelta(days=7)
        jst_filter = None

    closes = _fetch_close_times(trades, cfg, fetch_from_utc)
    rows = _build_result_rows(
        trades,
        closes,
        pretrade_meta,
        cfg,
        hours=hours,
        jst_date_from=jst_filter,
        limit=limit,
    )
    return {
        str(row["trade_id"]): row
        for row in rows
        if row.get("trade_id")
    }


def build_regret_payload(
    session_date_from: str | None = None,
    *,
    jst_date_from: str | None = None,
    hours: int = 6,
    limit: int | None = None,
) -> dict:
    trades = _load_loss_trades(session_date_from)
    if not trades:
        return {
            "summary": _summarize([]),
            "by_regime": [],
            "by_collapse_component": [],
            "by_thesis_lens": [],
            "by_live_tape_bucket": [],
            "results": [],
        }

    pretrade_meta = _load_pretrade_meta(session_date_from)
    cfg = get_oanda_config()
    if jst_date_from:
        parsed_jst = _parse_jst_date(jst_date_from)
        fetch_from_utc = datetime.combine(parsed_jst, time.min, tzinfo=JST).astimezone(UTC) - timedelta(days=1)
        jst_filter = parsed_jst
    elif session_date_from:
        fetch_from_utc = datetime.strptime(session_date_from, "%Y-%m-%d").replace(tzinfo=UTC) - timedelta(days=1)
        jst_filter = None
    else:
        fetch_from_utc = datetime.now(UTC) - timedelta(days=7)
        jst_filter = None

    closes = _fetch_close_times(trades, cfg, fetch_from_utc)
    results = _build_result_rows(
        trades,
        closes,
        pretrade_meta,
        cfg,
        hours=hours,
        jst_date_from=jst_filter,
        limit=limit,
    )
    return {
        "summary": _summarize(results),
        "by_regime": _summarize_by_key(results, "regime"),
        "by_collapse_component": _summarize_by_key(results, "collapse_component"),
        "by_thesis_lens": _summarize_by_key(results, "thesis_lens"),
        "by_live_tape_bucket": _summarize_by_key(results, "live_tape_bucket"),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether losing closes later traded back to breakeven or better."
    )
    parser.add_argument("--session-date-from", help="Filter trades.session_date from this UTC date (YYYY-MM-DD).")
    parser.add_argument("--jst-date-from", help="Filter by close timestamp in JST from this date (YYYY-MM-DD).")
    parser.add_argument("--hours", type=int, default=6, help="Post-close lookahead window in hours. Default: 6.")
    parser.add_argument("--limit", type=int, help="Limit output rows after sorting by most recent close.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    payload = build_regret_payload(
        session_date_from=args.session_date_from,
        jst_date_from=args.jst_date_from,
        hours=args.hours,
        limit=args.limit,
    )
    results = payload["results"]
    if not results:
        print("No losing closed trades matched the filter.")
        return 0
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_render_text(results, payload["summary"], payload["by_regime"], args.hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
