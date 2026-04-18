#!/usr/bin/env python3
"""Sync and report S-hunt seat outcomes into memory.db."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "collab_trade" / "memory"))

from schema import fetchall_dict, get_conn, init_db  # type: ignore
from config_loader import get_oanda_config

LEDGER_PATH = ROOT / "logs" / "s_hunt_ledger.jsonl"
JST = timezone(timedelta(hours=9))
OANDA_PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}
HORIZON_ORDER = {"Short-term S": 0, "Medium-term S": 1, "Long-term S": 2}


def _pip_factor(pair: str) -> int:
    return 100 if "JPY" in pair else 10000


def _parse_state_updated(value: str | None) -> datetime | None:
    if not value:
        return None
    formats = (
        ("%Y-%m-%d %H:%M UTC", timezone.utc),
        ("%Y-%m-%d %H:%M:%S UTC", timezone.utc),
        ("%Y-%m-%d %H:%M JST", JST),
        ("%Y-%m-%d %H:%M:%S JST", JST),
    )
    for fmt, tzinfo in formats:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=tzinfo).astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _parse_oanda_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        base = value.replace("Z", "+00:00")
        if "." in base:
            base = base.split(".")[0] + "+00:00"
        return datetime.fromisoformat(base).astimezone(timezone.utc)
    except Exception:
        return None


def _review_day_bounds_utc(session_date: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(session_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _load_ledger_rows(session_date: str | None = None) -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    rows = []
    for raw in LEDGER_PATH.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if session_date and row.get("session_date") != session_date:
            continue
        rows.append(row)
    rows.sort(key=lambda row: row.get("state_last_updated") or "")
    return rows


def _load_ledger_dates() -> list[str]:
    rows = _load_ledger_rows()
    return sorted({row.get("session_date") for row in rows if row.get("session_date")})


def fetch_closed_trades(session_date: str) -> list[dict]:
    """Fetch closed trades for the specified UTC day from OANDA."""
    cfg = get_oanda_config()
    start_utc, end_utc = _review_day_bounds_utc(session_date)
    params = urllib.parse.urlencode(
        {
            "from": start_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": end_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "type": "ORDER_FILL",
            "pageSize": 1000,
        }
    )
    url = f"{cfg['oanda_base_url']}/v3/accounts/{cfg['oanda_account_id']}/transactions?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            id_range = json.loads(resp.read())
    except Exception:
        return []

    all_txns = []
    for page_url in id_range.get("pages", []):
        try:
            req = urllib.request.Request(
                page_url,
                headers={
                    "Authorization": f"Bearer {cfg['oanda_token']}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                all_txns.extend(data.get("transactions", []))
        except Exception:
            continue

    entries = {}
    closes = {}
    for txn in all_txns:
        if txn.get("type") != "ORDER_FILL":
            continue
        instrument = txn.get("instrument", "")
        if instrument not in OANDA_PAIRS:
            continue

        trade_opened = txn.get("tradeOpened")
        if trade_opened:
            tid = trade_opened.get("tradeID", "")
            units = int(float(trade_opened.get("units", 0)))
            entries[tid] = {
                "pair": instrument,
                "direction": "LONG" if units > 0 else "SHORT",
                "units": abs(units),
                "entry_time": txn.get("time", ""),
            }

        for tc in txn.get("tradesClosed", []) + txn.get("tradesReduced", []):
            tid = tc.get("tradeID", "")
            pl = float(tc.get("realizedPL", 0))
            units_signed = int(float(tc.get("units", 0)))
            close = closes.setdefault(
                tid,
                {
                    "pair": instrument,
                    "direction": "LONG" if units_signed > 0 else "SHORT" if units_signed < 0 else None,
                    "pl": 0.0,
                    "close_time": txn.get("time", ""),
                },
            )
            close["pl"] += pl

    trades = []
    for tid in sorted(set(entries) | set(closes)):
        close = closes.get(tid)
        if not close:
            continue
        entry = entries.get(tid)
        trades.append(
            {
                "trade_id": tid,
                "pair": (entry or close).get("pair"),
                "direction": (entry or close).get("direction"),
                "pl": float(close.get("pl", 0.0)),
                "entry_time": (entry or {}).get("entry_time"),
                "close_time": close.get("close_time"),
            }
        )
    return trades


def fetch_current_prices(pairs: set[str]) -> dict[str, float]:
    if not pairs:
        return {}
    cfg = get_oanda_config()
    pairs_str = ",".join(sorted(pairs))
    url = (
        f"{cfg['oanda_base_url']}/v3/accounts/{cfg['oanda_account_id']}/pricing"
        f"?instruments={pairs_str}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
    except Exception:
        return {}

    prices = {}
    for price in resp.get("prices", []):
        try:
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
        except Exception:
            continue
        prices[price["instrument"]] = (bid + ask) / 2
    return prices


def fetch_session_close_prices(session_date: str, pairs: set[str]) -> dict[str, float]:
    if not pairs:
        return {}

    cfg = get_oanda_config()
    _, end_utc = _review_day_bounds_utc(session_date)
    to_value = end_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    prices = {}

    for pair in sorted(pairs):
        params = urllib.parse.urlencode(
            {"price": "M", "granularity": "M5", "count": 1, "to": to_value}
        )
        url = f"{cfg['oanda_base_url']}/v3/instruments/{pair}/candles?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {cfg['oanda_token']}",
                "Content-Type": "application/json",
            },
        )
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        except Exception:
            continue
        candles = data.get("candles") or []
        if not candles:
            continue
        mid = candles[-1].get("mid") or {}
        close = mid.get("c")
        if close is None:
            continue
        try:
            prices[pair] = float(close)
        except Exception:
            continue
    return prices


def _matching_trades(
    pair: str,
    direction: str,
    state_ts: datetime | None,
    trade_lookup: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    matches = []
    for trade in trade_lookup.get((pair, direction), []):
        entry_ts = _parse_oanda_time(trade.get("entry_time"))
        close_ts = _parse_oanda_time(trade.get("close_time"))
        if state_ts:
            if close_ts and close_ts >= state_ts:
                matches.append(trade)
                continue
            if entry_ts and entry_ts >= state_ts:
                matches.append(trade)
                continue
            continue
        matches.append(trade)
    return matches


def _deployment_flags(orderability: str | None, deployment_result: str | None) -> dict[str, object]:
    orderability_text = " ".join((orderability or "").split())
    deployment_text = " ".join((deployment_result or "").split())
    combined = " ".join(filter(None, [orderability_text, deployment_text])).upper()

    has_id = "ID=" in combined
    is_dead = "DEAD THESIS" in combined
    armed = any(token in combined for token in ("ENTER NOW", "STOP-ENTRY", "LIMIT", "ARMED")) or has_id

    if is_dead and not has_id:
        orderable = 0
        deployment_status = "dead"
    elif has_id and "ENTERED" in combined:
        orderable = 1
        deployment_status = "entered"
    elif has_id:
        orderable = 1
        deployment_status = "armed"
    elif armed:
        orderable = 1
        deployment_status = "orderable"
    elif "STILL PASS" in combined:
        orderable = 0
        deployment_status = "pass"
    else:
        orderable = 0
        deployment_status = "unclear"

    return {
        "orderable": orderable,
        "deployed": 1 if has_id else 0,
        "deployment_status": deployment_status,
    }


def _upsert_seat_outcome(conn, record: dict) -> None:
    conn.execute(
        """
        INSERT INTO seat_outcomes (
            session_date, state_last_updated, source, horizon, pair, direction,
            setup_type, why, mtf_chain, payout_path, orderability, deployment_result,
            trigger, invalidation, reference_price, discovered, orderable, deployed,
            captured, missed, directionally_correct, deployment_status, outcome_status,
            matched_trade_count, matched_trade_ids, realized_pl, eval_price,
            eval_price_source, pip_move, notes, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            datetime('now', 'localtime')
        )
        ON CONFLICT(state_last_updated, horizon, pair, direction) DO UPDATE SET
            session_date=excluded.session_date,
            source=excluded.source,
            setup_type=excluded.setup_type,
            why=excluded.why,
            mtf_chain=excluded.mtf_chain,
            payout_path=excluded.payout_path,
            orderability=excluded.orderability,
            deployment_result=excluded.deployment_result,
            trigger=excluded.trigger,
            invalidation=excluded.invalidation,
            reference_price=excluded.reference_price,
            discovered=excluded.discovered,
            orderable=excluded.orderable,
            deployed=excluded.deployed,
            captured=excluded.captured,
            missed=excluded.missed,
            directionally_correct=excluded.directionally_correct,
            deployment_status=excluded.deployment_status,
            outcome_status=excluded.outcome_status,
            matched_trade_count=excluded.matched_trade_count,
            matched_trade_ids=excluded.matched_trade_ids,
            realized_pl=excluded.realized_pl,
            eval_price=excluded.eval_price,
            eval_price_source=excluded.eval_price_source,
            pip_move=excluded.pip_move,
            notes=excluded.notes,
            updated_at=datetime('now', 'localtime')
        """,
        (
            record["session_date"],
            record["state_last_updated"],
            record["source"],
            record["horizon"],
            record["pair"],
            record["direction"],
            record.get("setup_type"),
            record.get("why"),
            record.get("mtf_chain"),
            record.get("payout_path"),
            record.get("orderability"),
            record.get("deployment_result"),
            record.get("trigger"),
            record.get("invalidation"),
            record.get("reference_price"),
            record.get("discovered", 1),
            record.get("orderable", 0),
            record.get("deployed", 0),
            record.get("captured", 0),
            record.get("missed", 0),
            record.get("directionally_correct"),
            record.get("deployment_status"),
            record.get("outcome_status"),
            record.get("matched_trade_count", 0),
            record.get("matched_trade_ids"),
            record.get("realized_pl"),
            record.get("eval_price"),
            record.get("eval_price_source"),
            record.get("pip_move"),
            record.get("notes"),
        ),
    )


def sync_seat_outcomes(session_date: str | None = None, *, live: bool = False) -> dict:
    init_db()
    conn = get_conn()

    dates = [session_date] if session_date else _load_ledger_dates()
    upserted = 0

    for target_date in dates:
        if not target_date:
            continue
        rows = _load_ledger_rows(target_date)
        if not rows:
            continue

        pairs = {
            horizon.get("pair")
            for row in rows
            for horizon in row.get("horizons", [])
            if horizon.get("pair")
        }
        trades = fetch_closed_trades(target_date)
        trade_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for trade in trades:
            key = (trade.get("pair"), trade.get("direction"))
            trade_lookup[key].append(trade)
        for bucket in trade_lookup.values():
            bucket.sort(key=lambda item: item.get("entry_time") or item.get("close_time") or "")

        today_utc = datetime.now(timezone.utc).date().isoformat()
        if live or target_date == today_utc:
            eval_prices = fetch_current_prices({pair for pair in pairs if pair})
            eval_source = "live_mid"
        else:
            eval_prices = fetch_session_close_prices(target_date, {pair for pair in pairs if pair})
            eval_source = "utc_day_close"

        for row in rows:
            state_ts = _parse_state_updated(row.get("state_last_updated"))
            for horizon in row.get("horizons", []):
                pair = horizon.get("pair")
                direction = horizon.get("direction")
                if not pair or not direction:
                    continue

                flags = _deployment_flags(
                    horizon.get("orderability"),
                    horizon.get("deployment_result"),
                )
                matching = _matching_trades(pair, direction, state_ts, trade_lookup)
                deployed = max(int(flags["deployed"]), 1 if matching else 0)
                orderable = int(flags["orderable"])

                realized_pl = None
                matched_trade_ids = None
                matched_trade_count = len(matching)
                captured = 0
                missed = 0
                directionally_correct = None
                outcome_status = "pending"
                pip_move = None
                eval_price = eval_prices.get(pair)
                notes = None

                ref_price = horizon.get("reference_price")
                moved_right = None
                if eval_price is not None and ref_price not in (None, ""):
                    try:
                        ref_value = float(ref_price)
                        moved_right = (
                            direction == "LONG" and eval_price > ref_value
                        ) or (
                            direction == "SHORT" and eval_price < ref_value
                        )
                        pip_move = abs(eval_price - ref_value) * _pip_factor(pair)
                        directionally_correct = 1 if moved_right else 0
                    except Exception:
                        moved_right = None

                if matching:
                    realized_pl = float(sum(trade.get("pl", 0.0) for trade in matching))
                    matched_trade_ids = ",".join(
                        sorted(str(trade.get("trade_id")) for trade in matching if trade.get("trade_id"))
                    ) or None
                    if realized_pl > 0:
                        captured = 1
                        outcome_status = "captured"
                        directionally_correct = 1
                        notes = "Seat was deployed and realized positive P&L."
                    elif realized_pl < 0:
                        outcome_status = "failed"
                        directionally_correct = 0
                        notes = "Seat was deployed but failed to convert into positive P&L."
                    else:
                        outcome_status = "flat"
                        notes = "Seat was deployed but closed flat."
                elif deployed:
                    outcome_status = "pending_open"
                    notes = "Seat has a real deployment receipt but no closed trade yet."
                elif moved_right is True:
                    missed = 1
                    outcome_status = "missed"
                    notes = "Direction worked from the reference price without capture."
                elif moved_right is False:
                    outcome_status = "not_captured"
                    notes = "Direction did not work from the reference price."
                else:
                    outcome_status = "no_data"
                    notes = "Could not score direction from reference price."

                record = {
                    "session_date": target_date,
                    "state_last_updated": row.get("state_last_updated"),
                    "source": "s_hunt",
                    "horizon": horizon.get("horizon"),
                    "pair": pair,
                    "direction": direction,
                    "setup_type": horizon.get("type"),
                    "why": horizon.get("why"),
                    "mtf_chain": horizon.get("mtf_chain"),
                    "payout_path": horizon.get("payout_path"),
                    "orderability": horizon.get("orderability"),
                    "deployment_result": horizon.get("deployment_result"),
                    "trigger": horizon.get("trigger"),
                    "invalidation": horizon.get("invalidation"),
                    "reference_price": horizon.get("reference_price"),
                    "discovered": 1,
                    "orderable": orderable,
                    "deployed": deployed,
                    "captured": captured,
                    "missed": missed,
                    "directionally_correct": directionally_correct,
                    "deployment_status": flags["deployment_status"],
                    "outcome_status": outcome_status,
                    "matched_trade_count": matched_trade_count,
                    "matched_trade_ids": matched_trade_ids,
                    "realized_pl": realized_pl,
                    "eval_price": eval_price,
                    "eval_price_source": eval_source if eval_price is not None else None,
                    "pip_move": pip_move,
                    "notes": notes,
                }
                _upsert_seat_outcome(conn, record)
                upserted += 1

    return {"dates": dates, "upserted": upserted}


def review_lines(session_date: str) -> list[str]:
    init_db()
    conn = get_conn()
    rows = fetchall_dict(
        conn,
        """
        SELECT state_last_updated, horizon, pair, direction, setup_type, reference_price,
               deployment_result, matched_trade_count, realized_pl, outcome_status,
               directionally_correct, pip_move, discovered, orderable, deployed,
               captured, missed, notes
        FROM seat_outcomes
        WHERE session_date = ?
        ORDER BY state_last_updated,
                 CASE horizon
                   WHEN 'Short-term S' THEN 0
                   WHEN 'Medium-term S' THEN 1
                   WHEN 'Long-term S' THEN 2
                   ELSE 9
                 END,
                 pair, direction
        """,
        (session_date,),
    )
    if not rows:
        return []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["state_last_updated"]].append(row)

    lines = ["## S Hunt Capture Review", ""]
    horizon_stats: dict[str, dict[str, int]] = {}

    for header in sorted(grouped):
        lines.append(f"### {header}")
        lines.append("")
        for row in grouped[header]:
            label = row.get("horizon") or "Unknown"
            bucket = horizon_stats.setdefault(
                label,
                {
                    "discovered": 0,
                    "orderable": 0,
                    "deployed": 0,
                    "captured": 0,
                    "missed": 0,
                    "correct": 0,
                    "wrong": 0,
                },
            )
            bucket["discovered"] += int(row.get("discovered") or 0)
            bucket["orderable"] += int(row.get("orderable") or 0)
            bucket["deployed"] += int(row.get("deployed") or 0)
            bucket["captured"] += int(row.get("captured") or 0)
            bucket["missed"] += int(row.get("missed") or 0)
            if row.get("directionally_correct") == 1:
                bucket["correct"] += 1
            elif row.get("directionally_correct") == 0:
                bucket["wrong"] += 1

            type_value = row.get("setup_type") or "setup"
            ref_text = (
                f" ref={float(row['reference_price']):.5f}".rstrip("0").rstrip(".")
                if row.get("reference_price") not in (None, "")
                else ""
            )
            deployment = row.get("deployment_result") or "missing deployment receipt"

            if int(row.get("matched_trade_count") or 0) > 0:
                realized_pl = float(row.get("realized_pl") or 0.0)
                result = "WIN" if realized_pl > 0 else "LOSS" if realized_pl < 0 else "FLAT"
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    f"→ ENTERED ({int(row['matched_trade_count'])} trade{'s' if int(row['matched_trade_count']) != 1 else ''}) "
                    f"→ {result} {realized_pl:+,.0f} JPY"
                )
            elif row.get("outcome_status") == "missed":
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    f"→ MISSED → CORRECT ({float(row.get('pip_move') or 0.0):.1f}pip from reference)"
                )
            elif row.get("outcome_status") == "pending_open":
                direction_text = "favorable" if row.get("directionally_correct") == 1 else "adverse"
                pip_text = (
                    f" ({float(row.get('pip_move') or 0.0):.1f}pip {direction_text})"
                    if row.get("pip_move") is not None
                    else ""
                )
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    f"→ DEPLOYED / OPEN{pip_text}"
                )
            elif row.get("directionally_correct") == 1:
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    f"→ NOT_CAPTURED → CORRECT ({float(row.get('pip_move') or 0.0):.1f}pip from reference)"
                )
            elif row.get("directionally_correct") == 0:
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    f"→ NOT_CAPTURED → WRONG ({float(row.get('pip_move') or 0.0):.1f}pip from reference)"
                )
            else:
                lines.append(
                    f"  {label}: {row['pair']} {row['direction']} {type_value}{ref_text} | {deployment} "
                    "→ NOT_CAPTURED → no score"
                )
        lines.append("")

    lines.append("### Horizon Scoreboard")
    lines.append("")
    for label in sorted(horizon_stats, key=lambda item: HORIZON_ORDER.get(item, 9)):
        bucket = horizon_stats[label]
        total_scored = bucket["correct"] + bucket["wrong"]
        accuracy = bucket["correct"] / total_scored * 100 if total_scored else 0.0
        lines.append(
            f"  {label}: discovered={bucket['discovered']} orderable={bucket['orderable']} "
            f"deployed={bucket['deployed']} captured={bucket['captured']} missed={bucket['missed']} "
            f"| directional accuracy={bucket['correct']}/{total_scored} ({accuracy:.0f}%)"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync or summarize seat outcomes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sync_cmd = sub.add_parser("sync", help="Sync seat outcomes from s_hunt_ledger.jsonl into memory.db")
    sync_cmd.add_argument("--date", default=None, help="UTC session date to sync (default: all dates)")
    sync_cmd.add_argument("--live", action="store_true", help="Use live prices instead of UTC day-close prices")

    stats_cmd = sub.add_parser("stats", help="Print review-style seat-outcome summary")
    stats_cmd.add_argument("--date", required=True, help="UTC session date")

    args = parser.parse_args()

    if args.cmd == "sync":
        result = sync_seat_outcomes(args.date, live=args.live)
        dates = [date for date in result["dates"] if date]
        date_text = ",".join(dates) if dates else "none"
        print(f"SEAT_OUTCOMES_SYNC_OK dates={date_text} upserted={result['upserted']}")
        return 0

    if args.cmd == "stats":
        lines = review_lines(args.date)
        if not lines:
            print("SEAT_OUTCOMES_STATS_EMPTY")
            return 0
        print("\n".join(lines))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
