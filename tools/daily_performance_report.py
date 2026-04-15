#!/usr/bin/env python3
"""Daily Performance Report — aggregates OANDA realized P&L and posts to Slack.

Fetches all ORDER_FILL transactions since system start (2026-03-18),
aggregates by today / this week / all time, and posts to #qr-daily.

Usage:
    python3 tools/daily_performance_report.py           # post to Slack
    python3 tools/daily_performance_report.py --dry-run  # print only, no Slack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SYSTEM_START = "2026-03-18"
JST = timezone(timedelta(hours=9))
UTC = timezone.utc


def load_config():
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    cfg = {}
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def fetch_fills(token: str, acct: str, from_date: str) -> list[dict]:
    """Fetch all ORDER_FILL transactions with non-zero P&L."""
    base = "https://api-fxtrade.oanda.com"
    url = f"{base}/v3/accounts/{acct}/transactions?from={from_date}&type=ORDER_FILL"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())

    fills = []
    for page_url in data.get("pages", []):
        req2 = urllib.request.Request(page_url, headers={"Authorization": f"Bearer {token}"})
        page_data = json.loads(urllib.request.urlopen(req2, timeout=30).read())
        for tx in page_data.get("transactions", []):
            if tx.get("type") == "ORDER_FILL":
                pl = float(tx.get("pl", 0))
                if pl != 0:
                    fills.append({
                        "time": tx["time"][:19],
                        "pl": pl,
                    })
    return fills


def get_account_summary(cfg):
    token = cfg['oanda_token']
    acct = cfg['oanda_account_id']
    url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/summary"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())['account']
    return {
        'balance': int(float(data['balance'])),
        'nav': int(float(data['NAV'])),
        'unrealized_pl': int(float(data['unrealizedPL'])),
    }


def aggregate(fills: list[dict]) -> dict:
    """Aggregate fills into daily buckets (UTC date)."""
    daily = defaultdict(lambda: {"pl": 0, "closes": 0})
    for f in fills:
        day = f["time"][:10]
        daily[day]["pl"] += f["pl"]
        daily[day]["closes"] += 1
    # Round
    for d in daily.values():
        d["pl"] = int(d["pl"])
    return dict(daily)


def build_report(daily: dict, acct: dict) -> str:
    now_jst = datetime.now(JST)
    today_utc = datetime.now(UTC).strftime("%Y-%m-%d")

    # Today
    today_data = daily.get(today_utc, {"pl": 0, "closes": 0})

    # This week (Monday to today, UTC dates)
    now_utc = datetime.now(UTC)
    monday = now_utc - timedelta(days=now_utc.weekday())
    week_start = monday.strftime("%Y-%m-%d")
    week_days = []
    d = monday
    while d.strftime("%Y-%m-%d") <= today_utc:
        ds = d.strftime("%Y-%m-%d")
        if ds in daily:
            week_days.append((ds, daily[ds]))
        else:
            week_days.append((ds, {"pl": 0, "closes": 0}))
        d += timedelta(days=1)
    week_total = sum(wd[1]["pl"] for wd in week_days)

    # All time since system start
    all_days = sorted([(d, v) for d, v in daily.items() if d >= SYSTEM_START])
    cumulative = sum(v["pl"] for _, v in all_days)
    cumulative_icon = "\u2705" if cumulative >= 0 else "\u26a0\ufe0f"

    # Format
    lines = []
    lines.append(f"\U0001f4ca *QuantRabbit Performance Report* ({now_jst.strftime('%Y/%m/%d %H:%M')} JST)")
    lines.append("")

    # Today
    lines.append(f"*[Today {today_utc}]* {today_data['pl']:+,} JPY ({today_data['closes']} closes)")
    lines.append("")

    # This week
    lines.append(f"*[This week {week_start}~{today_utc}]* {week_total:+,} JPY")
    for i, (ds, wd) in enumerate(week_days):
        prefix = "\u2514" if i == len(week_days) - 1 else "\u251c"
        lines.append(f"{prefix} {ds}: {wd['pl']:+,} JPY ({wd['closes']} closes)")
    lines.append("")

    # All time — show daily with running cumulative
    lines.append(f"*[All time since {SYSTEM_START}]*")
    running = 0
    for ds, v in all_days:
        running += v["pl"]
        lines.append(f"{ds}: {v['pl']:+,} JPY (cum: {running:+,})")
    lines.append(f"*Cumulative: {cumulative:+,} JPY* {cumulative_icon}")
    lines.append("")

    # Account
    lines.append(f"*[Account status]*")
    lines.append(f"Balance: {acct['balance']:,} JPY | NAV: {acct['nav']:,} JPY | Unrealized P&L: {acct['unrealized_pl']:+,} JPY")

    return "\n".join(lines)


def post_slack(text: str, channel: str, token: str):
    payload = {"channel": channel, "text": text}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    if not resp.get('ok'):
        print(f"ERROR: Slack post failed: {resp.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return resp


def main():
    parser = argparse.ArgumentParser(description="Daily performance report")
    parser.add_argument("--dry-run", action="store_true", help="Print report without posting to Slack")
    args = parser.parse_args()

    cfg = load_config()

    # Fetch all fills since system start
    from_date = f"{SYSTEM_START}T00:00:00Z"
    fills = fetch_fills(cfg['oanda_token'], cfg['oanda_account_id'], from_date)

    daily = aggregate(fills)
    acct = get_account_summary(cfg)
    report = build_report(daily, acct)

    if args.dry_run:
        print(report)
    else:
        channel = cfg.get('slack_channel_daily', '')
        post_slack(report, channel, cfg['slack_bot_token'])
        print(f"Posted performance report to Slack")


if __name__ == '__main__':
    main()
