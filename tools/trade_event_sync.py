#!/usr/bin/env python3
"""Sync OANDA close fills into the trade log and Slack.

This catches TP/SL fills that happen at OANDA without passing through
close_trade.py. Close notifications are batched so #qr-trades stays readable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_loader import get_oanda_config, load_env_toml

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "live_trade_log.txt"
STATE_PATH = ROOT / "logs" / "trade_event_sync_state.json"
JST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class CloseEvent:
    txn_id: str
    trade_id: str
    time_utc: datetime
    pair: str
    side: str
    units: int
    price: str
    pl: float
    reason: str
    action: str
    spread_pips: float | None


def _request_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _parse_oanda_time(value: str) -> datetime:
    text = value
    if text.endswith("Z"):
        text = text[:-1]
        if "." in text:
            head, frac = text.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text = f"{text}+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("JPY") else 10000


def _spread_pips(txn: dict, pair: str) -> float | None:
    full_price = txn.get("fullPrice") or {}
    try:
        bids = full_price.get("bids") or []
        asks = full_price.get("asks") or []
        bid = float(bids[0]["price"]) if bids else float(full_price["closeoutBid"])
        ask = float(asks[0]["price"]) if asks else float(full_price["closeoutAsk"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return abs(ask - bid) * _pip_factor(pair)


def _side_from_close_units(units: str | int) -> str:
    signed = int(float(units))
    return "LONG" if signed < 0 else "SHORT"


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_transaction_id": 0, "synced_close_transaction_ids": []}
    try:
        data = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {"last_transaction_id": 0, "synced_close_transaction_ids": []}
    data.setdefault("last_transaction_id", 0)
    data.setdefault("synced_close_transaction_ids", [])
    return data


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids = [str(item) for item in state.get("synced_close_transaction_ids", [])]
    state["synced_close_transaction_ids"] = ids[-500:]
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _fetch_transactions(cfg: dict, *, since_id: int | None, lookback_hours: int) -> list[dict]:
    acct = str(cfg["oanda_account_id"])
    token = str(cfg["oanda_token"])
    base = str(cfg["oanda_base_url"])
    if since_id and since_id > 0:
        url = f"{base}/v3/accounts/{acct}/transactions/sinceid?id={since_id}"
        return _request_json(url, token).get("transactions", []) or []

    start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    from_time = start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    index = _request_json(f"{base}/v3/accounts/{acct}/transactions?from={from_time}", token)
    txns: list[dict] = []
    pages = index.get("pages") or []
    if not pages:
        return index.get("transactions", []) or []
    for page_url in pages:
        txns.extend(_request_json(page_url, token).get("transactions", []) or [])
    return txns


def _close_events_from_transaction(txn: dict) -> list[CloseEvent]:
    if txn.get("type") != "ORDER_FILL":
        return []
    pair = str(txn.get("instrument") or "")
    if not pair:
        return []
    close_units = txn.get("units")
    if close_units is None:
        return []

    legs: list[tuple[str, dict, str]] = []
    for item in txn.get("tradesClosed") or []:
        legs.append(("CLOSE", item, str(item.get("tradeID") or "")))
    reduced = txn.get("tradeReduced")
    if reduced:
        legs.append(("PARTIAL_CLOSE", reduced, str(reduced.get("tradeID") or "")))
    if not legs:
        return []

    events: list[CloseEvent] = []
    side = _side_from_close_units(close_units)
    spread = _spread_pips(txn, pair)
    time_utc = _parse_oanda_time(str(txn.get("time")))
    for action, leg, trade_id in legs:
        if not trade_id:
            continue
        units = abs(int(float(leg.get("units") or close_units)))
        price = str(leg.get("price") or txn.get("price") or "?")
        try:
            pl = float(leg.get("realizedPL", txn.get("pl", 0.0)) or 0.0)
        except (TypeError, ValueError):
            pl = 0.0
        events.append(
            CloseEvent(
                txn_id=str(txn.get("id")),
                trade_id=trade_id,
                time_utc=time_utc,
                pair=pair,
                side=side,
                units=units,
                price=price,
                pl=pl,
                reason=str(txn.get("reason") or "ORDER_FILL"),
                action=action,
                spread_pips=spread,
            )
        )
    return events


def _existing_log_index() -> tuple[set[str], set[tuple[str, str]]]:
    if not LOG_PATH.exists():
        return set(), set()
    text = LOG_PATH.read_text(errors="ignore")
    txn_ids = set(re.findall(r"\btxn=(\d+)\b", text))
    close_keys: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if "CLOSE" not in line:
            continue
        trade_match = re.search(r"\bid=(\d+)\b", line) or re.search(r"\btradeId=(\d+)\b", line)
        pl_match = re.search(r"\bP/L=([+-]?\d+(?:\.\d+)?)JPY\b", line) or re.search(
            r"\bPL=([+-]?\d+(?:\.\d+)?)JPY\b", line
        )
        if not trade_match or not pl_match:
            continue
        try:
            pl_key = f"{float(pl_match.group(1)):.4f}"
        except ValueError:
            continue
        close_keys.add((trade_match.group(1), pl_key))
    return txn_ids, close_keys


def _event_is_logged(event: CloseEvent, txn_ids: set[str], close_keys: set[tuple[str, str]]) -> bool:
    if event.txn_id in txn_ids:
        return True
    return (event.trade_id, f"{event.pl:.4f}") in close_keys


def _log_line(event: CloseEvent) -> str:
    stamp = event.time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    spread = f"{event.spread_pips:.1f}pip" if event.spread_pips is not None else "n/a"
    return (
        f"[{stamp}] {event.action} {event.pair} {event.side} {event.units}u @{event.price} "
        f"P/L={event.pl:+.4f}JPY Sp={spread} reason={event.reason} "
        f"id={event.trade_id} txn={event.txn_id}"
    )


def _append_missing_logs(events: list[CloseEvent], *, no_log: bool) -> list[CloseEvent]:
    txn_ids, close_keys = _existing_log_index()
    missing = [event for event in events if not _event_is_logged(event, txn_ids, close_keys)]
    if missing and not no_log:
        with LOG_PATH.open("a") as fh:
            for event in missing:
                fh.write(_log_line(event) + "\n")
    return missing


def _reason_label(reason: str) -> str:
    labels = {
        "TAKE_PROFIT_ORDER": "TP",
        "STOP_LOSS_ORDER": "SL",
        "MARKET_ORDER_TRADE_CLOSE": "MANUAL",
        "MARKET_ORDER_POSITION_CLOSEOUT": "CLOSEOUT",
    }
    return labels.get(reason, reason.replace("_", " ").lower())


def _today_realized_pl(cfg: dict) -> float | None:
    try:
        now_utc = datetime.now(timezone.utc)
        boundary = now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
        if now_utc.hour < 21:
            boundary -= timedelta(days=1)
        from_time = boundary.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        acct = str(cfg["oanda_account_id"])
        token = str(cfg["oanda_token"])
        base = str(cfg["oanda_base_url"])
        index = _request_json(f"{base}/v3/accounts/{acct}/transactions?from={from_time}", token)
        total = 0.0
        for page_url in index.get("pages", []) or []:
            data = _request_json(page_url, token)
            for txn in data.get("transactions", []) or []:
                if txn.get("type") == "ORDER_FILL":
                    total += float(txn.get("pl", 0.0) or 0.0)
        return total
    except Exception:
        return None


def _format_slack_message(events: list[CloseEvent], cfg: dict) -> str:
    now_jst = datetime.now(JST).strftime("%H:%M")
    total = sum(event.pl for event in events)
    lines = [f"CLOSE SYNC {len(events)} fill(s) [{now_jst} JST]"]
    for event in events:
        result = "WIN" if event.pl >= 0 else "LOSS"
        lines.append(
            f"- {result} {event.pair} {event.side} {event.units}u @{event.price} "
            f"{event.pl:+.1f}JPY {_reason_label(event.reason)} "
            f"trade={event.trade_id} txn={event.txn_id}"
        )
    daily = _today_realized_pl(cfg)
    summary = f"Batch total: {total:+.1f}JPY"
    if daily is not None:
        summary += f" | OANDA day realized: {daily:+.1f}JPY"
    lines.append(summary)
    return "\n".join(lines)


def _post_slack(text: str) -> str:
    raw_cfg = load_env_toml()
    token = str(raw_cfg["slack_bot_token"])
    channel = str(raw_cfg["slack_channel_trades"])
    payload = {"channel": channel, "text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    if not resp.get("ok"):
        raise RuntimeError(str(resp.get("error", "unknown Slack error")))
    return str(resp.get("ts", ""))


def _recent_slack_text(limit: int = 80) -> str:
    try:
        raw_cfg = load_env_toml()
        token = str(raw_cfg["slack_bot_token"])
        channel = str(raw_cfg["slack_channel_trades"])
        url = "https://slack.com/api/conversations.history?" + urllib.parse.urlencode(
            {"channel": channel, "limit": limit}
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        if not resp.get("ok"):
            return ""
        return "\n".join(str(msg.get("text") or "") for msg in resp.get("messages", []) or [])
    except Exception:
        return ""


def _compact_number(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def _event_seen_in_slack(event: CloseEvent, slack_text: str) -> bool:
    if not slack_text:
        return False
    if f"txn={event.txn_id}" in slack_text or f"txn {event.txn_id}" in slack_text:
        return True
    if f"trade={event.trade_id}" in slack_text:
        return True
    if event.pair not in slack_text or f"@{event.price}" not in slack_text:
        return False
    if f"{event.units}units" not in slack_text and f"{event.units}u" not in slack_text:
        return False
    pl_forms = {
        _compact_number(event.pl, 1),
        _compact_number(event.pl, 2),
        _compact_number(event.pl, 4),
        f"{event.pl:+.1f}",
        f"{event.pl:+.2f}",
        f"{event.pl:+.4f}",
    }
    return any(form and form in slack_text for form in pl_forms)


def _max_txn_id(transactions: list[dict], fallback: int) -> int:
    current = fallback
    for txn in transactions:
        try:
            current = max(current, int(txn.get("id", 0)))
        except (TypeError, ValueError):
            continue
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-hours", type=int, default=8)
    parser.add_argument("--since-id", type=int)
    parser.add_argument("--notify-slack", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--unlogged-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = get_oanda_config()
    state_exists = STATE_PATH.exists()
    state = _load_state()
    since_id = args.since_id if args.since_id is not None else int(state.get("last_transaction_id") or 0)
    transactions = _fetch_transactions(cfg, since_id=since_id, lookback_hours=args.lookback_hours)
    latest_id = _max_txn_id(transactions, since_id)

    events: list[CloseEvent] = []
    synced_ids = {str(item) for item in state.get("synced_close_transaction_ids", [])}
    for txn in transactions:
        for event in _close_events_from_transaction(txn):
            if event.txn_id not in synced_ids:
                events.append(event)

    missing = _append_missing_logs(events, no_log=args.no_log or args.dry_run)
    effective_unlogged_only = args.unlogged_only or (not state_exists and args.since_id is None)
    notify_events = missing if effective_unlogged_only else events
    slack_seen_ids: set[str] = set()
    if args.notify_slack and notify_events:
        slack_text = _recent_slack_text()
        filtered: list[CloseEvent] = []
        for event in notify_events:
            if _event_seen_in_slack(event, slack_text):
                slack_seen_ids.add(event.txn_id)
            else:
                filtered.append(event)
        notify_events = filtered

    if args.json:
        print(
            json.dumps(
                {
                    "latest_transaction_id": latest_id,
                    "events": [event.__dict__ | {"time_utc": event.time_utc.isoformat()} for event in events],
                    "missing_log_events": [
                        event.__dict__ | {"time_utc": event.time_utc.isoformat()} for event in missing
                    ],
                    "notify_count": len(notify_events),
                    "slack_seen_transaction_ids": sorted(slack_seen_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            f"TRADE_EVENT_SYNC events={len(events)} missing_log={len(missing)} "
            f"notify={len(notify_events)} latest_txn={latest_id}"
        )
        for event in notify_events:
            print(_log_line(event))

    if args.notify_slack and notify_events:
        message = _format_slack_message(notify_events, cfg)
        if args.dry_run:
            print("SLACK_DRY_RUN")
            print(message)
        else:
            ts = _post_slack(message)
            print(f"SLACK_POSTED ts={ts}")

    if not args.dry_run:
        state["last_transaction_id"] = latest_id
        merged_ids = list(synced_ids | slack_seen_ids | {event.txn_id for event in events})
        state["synced_close_transaction_ids"] = sorted(merged_ids, key=lambda x: int(x) if x.isdigit() else 0)
        _save_state(state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
