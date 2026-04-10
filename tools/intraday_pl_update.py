#!/usr/bin/env python3
"""Post intraday P&L update to #qr-daily.

Fetches realized P&L from OANDA transactions API (not log parsing)
and lists open positions with unrealized P&L.

Usage:
    python3 tools/intraday_pl_update.py          # post to Slack
    python3 tools/intraday_pl_update.py --dry-run # print only, don't post
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOD_NAV_FILE = ROOT / "logs" / "sod_nav.json"

JST = timezone(timedelta(hours=9))
UTC = timezone.utc


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_get(path, token):
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def get_account_summary(token, acct):
    data = oanda_get(f"/v3/accounts/{acct}/summary", token)["account"]
    return {
        "balance": float(data["balance"]),
        "nav": float(data["NAV"]),
        "unrealized_pl": float(data["unrealizedPL"]),
        "margin_used": float(data["marginUsed"]),
        "margin_available": float(data["marginAvailable"]),
        "open_trade_count": int(data["openTradeCount"]),
        "open_position_count": int(data["openPositionCount"]),
    }


def get_open_trades(token, acct):
    data = oanda_get(f"/v3/accounts/{acct}/openTrades", token)
    trades = []
    for t in data.get("trades", []):
        trades.append({
            "id": t["id"],
            "instrument": t["instrument"].replace("_", "/"),
            "instrument_raw": t["instrument"],
            "side": "LONG" if int(t["currentUnits"]) > 0 else "SHORT",
            "units": abs(int(t["currentUnits"])),
            "upl": float(t["unrealizedPL"]),
            "price": float(t["price"]),
        })
    return trades


def get_realized_pl_today(token, acct):
    """Fetch today's realized P&L from OANDA transactions API.

    'Today' is defined as the current JST date (00:00-23:59 JST).
    """
    now_jst = datetime.now(JST)
    today_start_jst = now_jst.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start_jst = today_start_jst + timedelta(days=1)

    # Convert to UTC for OANDA API
    from_utc = today_start_jst.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    to_utc = tomorrow_start_jst.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

    total_pl = 0.0
    close_count = 0

    # Step 1: Get page URLs from the transactions endpoint
    path = f"/v3/accounts/{acct}/transactions?from={from_utc}&to={to_utc}&type=ORDER_FILL"
    try:
        data = oanda_get(path, token)
    except Exception as e:
        print(f"WARN: transactions API error: {e}", file=sys.stderr)
        return total_pl, close_count

    # Step 2: Fetch each page URL (OANDA returns transactions in pages, not inline)
    pages = data.get("pages", [])
    for page_url in pages:
        try:
            req = urllib.request.Request(page_url, headers={"Authorization": f"Bearer {token}"})
            page_data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        except Exception as e:
            print(f"WARN: page fetch error: {e}", file=sys.stderr)
            continue

        for tx in page_data.get("transactions", []):
            if tx.get("type") != "ORDER_FILL":
                continue
            pl = float(tx.get("pl", "0"))
            if pl != 0:
                total_pl += pl
                close_count += 1

    return total_pl, close_count


def post_slack(text, channel_id, token):
    payload = {"channel": channel_id, "text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if not resp.get("ok"):
        print(f"ERROR: Slack {resp.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return resp


def load_sod_nav():
    """Load start-of-day NAV. Returns (nav, date_str) or (None, None)."""
    if not SOD_NAV_FILE.exists():
        return None, None
    try:
        data = json.loads(SOD_NAV_FILE.read_text())
        return data.get("nav"), data.get("date")
    except Exception:
        return None, None


def save_sod_nav(nav):
    """Save current NAV as start-of-day NAV (first call of the day wins)."""
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    saved_nav, saved_date = load_sod_nav()
    if saved_date == today_str:
        return saved_nav  # already recorded today
    data = {"nav": nav, "date": today_str}
    SOD_NAV_FILE.write_text(json.dumps(data))
    return nav


def build_message(acct_summary, open_trades, realized_pl, close_count, sod_nav):
    now_jst = datetime.now(JST)
    date_str = now_jst.strftime("%m/%d")
    time_str = now_jst.strftime("%H:%M")

    icon = "\U0001f4c8" if realized_pl >= 0 else "\U0001f4c9"
    nav = acct_summary["nav"]
    margin_used = acct_summary["margin_used"]
    margin_pct = (margin_used / nav * 100) if nav > 0 else 0
    upl = acct_summary["unrealized_pl"]
    trade_count = acct_summary["open_trade_count"]

    # Daily return %: actual NAV change from start of day
    if sod_nav and sod_nav > 0:
        daily_pct = (nav - sod_nav) / sod_nav * 100
    else:
        daily_pct = 0.0

    lines = []
    lines.append(f"{icon} *Intraday Update* {date_str} {time_str} JST")
    lines.append("")
    lines.append(f"*Realized P&L*: {realized_pl:+,.0f} JPY ({close_count} closes)")
    lines.append(f"*Unrealized P&L*: {upl:+,.0f} JPY ({trade_count} open trades)")
    lines.append(f"*NAV*: {nav:,.0f} JPY (*{daily_pct:+.2f}%*) | *Margin*: {margin_pct:.1f}%")

    if open_trades:
        lines.append("")
        for t in open_trades:
            lines.append(f"  {t['instrument_raw']} {t['side']} {t['units']}u UPL:{t['upl']:+,.0f} JPY")

    return "\n".join(lines)


def main():
    dry_run = "--dry-run" in sys.argv

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    acct_summary = get_account_summary(token, acct)
    open_trades = get_open_trades(token, acct)
    realized_pl, close_count = get_realized_pl_today(token, acct)

    # Record SOD NAV on first run of the day; use it for accurate daily %
    sod_nav = save_sod_nav(acct_summary["nav"])

    message = build_message(acct_summary, open_trades, realized_pl, close_count, sod_nav)

    if dry_run:
        print(message)
        print(f"\n--- dry run (not posted) ---")
        return

    channel = cfg.get("slack_channel_daily", cfg.get("slack_channel_id", ""))
    resp = post_slack(message, channel, cfg["slack_bot_token"])
    ts = resp.get("ts", "")
    print(f"OK: posted intraday update (ts={ts})")


if __name__ == "__main__":
    main()
