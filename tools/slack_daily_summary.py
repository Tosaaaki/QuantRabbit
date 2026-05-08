#!/usr/bin/env python3
"""Post the previous JST day's trade summary to Slack #qr-daily.

Day boundary = JST 00:00–23:59 (= UTC 15:00 prev day → UTC 14:59 same day).
Reads OANDA + Slack credentials from `.env.local` (vNext; no `config/env.toml`).

Usage:
    python3 tools/slack_daily_summary.py [--date YYYY-MM-DD] [--dry-run]
        --date     JST date to summarize (default: previous JST day)
        --dry-run  Print the message instead of posting
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse the Slack credential loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slack_post import load_slack_config, post_message  # noqa: E402

UTC = timezone.utc
JST = timezone(timedelta(hours=9))

OANDA_KEYS = {"QR_OANDA_TOKEN", "QR_OANDA_ACCOUNT_ID", "QR_OANDA_BASE_URL"}
DEFAULT_BASE_URL = "https://api-fxtrade.oanda.com"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_file() -> Path:
    override = os.environ.get("QR_OANDA_ENV_FILE")
    if override:
        return Path(override)
    return _repo_root() / ".env.local"


def _clean(value: str) -> str:
    text = value.strip()
    if "#" in text and not (text.startswith('"') or text.startswith("'")):
        text = text.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def load_oanda_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    path = _env_file()
    if path.exists():
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in OANDA_KEYS:
                cfg[key] = _clean(value)
    for key in OANDA_KEYS:
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


def oanda_get(base_url: str, path: str, token: str) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_account_summary(base_url: str, token: str, account_id: str) -> dict:
    data = oanda_get(base_url, f"/v3/accounts/{account_id}/summary", token)["account"]
    return {
        "balance": float(data["balance"]),
        "nav": float(data["NAV"]),
        "unrealized_pl": float(data["unrealizedPL"]),
        "open_positions": int(data["openPositionCount"]),
        "open_trades": int(data["openTradeCount"]),
    }


def jst_day_to_utc_range(date_str: str) -> tuple[str, str]:
    target = datetime.strptime(date_str, "%Y-%m-%d")
    start = target - timedelta(hours=9)
    end = target + timedelta(days=1) - timedelta(hours=9)
    fmt = "%Y-%m-%dT%H:%M:%S.000000000Z"
    return start.strftime(fmt), end.strftime(fmt)


def fetch_realized_pl(
    base_url: str, token: str, account_id: str, date_str: str
) -> tuple[float, int, int, float | None]:
    """Return (total_pl, entry_count, close_count, day_start_balance)."""
    from_utc, to_utc = jst_day_to_utc_range(date_str)
    path = f"/v3/accounts/{account_id}/transactions?from={from_utc}&to={to_utc}&type=ORDER_FILL"
    try:
        data = oanda_get(base_url, path, token)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"WARN: transactions API error: {exc}", file=sys.stderr)
        return 0.0, 0, 0, None

    total_pl = 0.0
    entry_count = 0
    close_count = 0
    day_start_balance: float | None = None

    for page_url in data.get("pages", []):
        try:
            req = urllib.request.Request(page_url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                page_data = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"WARN: page fetch error: {exc}", file=sys.stderr)
            continue

        for tx in page_data.get("transactions", []):
            if tx.get("type") != "ORDER_FILL":
                continue
            pl = float(tx.get("pl", "0"))
            if day_start_balance is None:
                acct_bal = float(tx.get("accountBalance", "0"))
                day_start_balance = acct_bal - pl
            if pl != 0:
                total_pl += pl
                close_count += 1
            else:
                entry_count += 1

    return total_pl, entry_count, close_count, day_start_balance


def build_message(
    target_date: str,
    realized_pl: float,
    entry_count: int,
    close_count: int,
    day_start_balance: float | None,
    acct: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"\U0001f4ca *Daily Summary: {target_date}*")
    lines.append("")

    pl_icon = "\U0001f7e2" if realized_pl >= 0 else "\U0001f534"
    if day_start_balance and day_start_balance > 0:
        pct_change = realized_pl / day_start_balance * 100
    else:
        pct_change = 0.0
    lines.append(f"{pl_icon} *Daily Realized P&L: {realized_pl:+,.0f}JPY ({pct_change:+.2f}%)*")

    if entry_count > 0 or close_count > 0:
        lines.append(f"\U0001f4dd Entries: {entry_count} | Closes: {close_count}")
    else:
        lines.append("\U0001f4dd No trade records")

    lines.append("")
    lines.append("*Account Status:*")
    lines.append(f"  Balance: {acct['balance']:,.0f}JPY")
    lines.append(f"  NAV: {acct['nav']:,.0f}JPY")
    lines.append(f"  Unrealized P&L: {acct['unrealized_pl']:+,.0f}JPY")
    lines.append(f"  Open: {acct['open_trades']} trades ({acct['open_positions']} pairs)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="JST date YYYY-MM-DD (default: previous JST day)")
    parser.add_argument("--dry-run", action="store_true", help="Print message instead of posting")
    args = parser.parse_args()

    target_date = args.date or (datetime.now(JST) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Dedup guard for auto-runs. Manual --date overrides skip the lock.
    log_dir = _repo_root() / "logs"
    lock_file = log_dir / "daily_summary_last.txt"
    if not args.date and not args.dry_run:
        try:
            if lock_file.exists() and lock_file.read_text().strip() == target_date:
                print(f"Already posted for {target_date}, skipping")
                return
        except OSError:
            pass

    oanda = load_oanda_config()
    token = oanda.get("QR_OANDA_TOKEN")
    account_id = oanda.get("QR_OANDA_ACCOUNT_ID")
    base_url = oanda.get("QR_OANDA_BASE_URL", DEFAULT_BASE_URL)
    if not token or not account_id:
        print("ERROR: QR_OANDA_TOKEN and QR_OANDA_ACCOUNT_ID required", file=sys.stderr)
        sys.exit(2)

    acct = get_account_summary(base_url, token, account_id)
    realized_pl, entry_count, close_count, day_start_balance = fetch_realized_pl(
        base_url, token, account_id, target_date
    )

    message = build_message(target_date, realized_pl, entry_count, close_count, day_start_balance, acct)

    if args.dry_run:
        print(message)
        return

    slack = load_slack_config()
    slack_token = slack.get("QR_SLACK_BOT_TOKEN")
    channel = slack.get("QR_SLACK_CHANNEL_DAILY") or slack.get("QR_SLACK_CHANNEL_ID")
    if not slack_token or not channel:
        print("ERROR: QR_SLACK_BOT_TOKEN and QR_SLACK_CHANNEL_DAILY (or _ID) required", file=sys.stderr)
        sys.exit(2)

    post_message(message, channel, slack_token)
    print(f"Posted daily summary for {target_date} to channel {channel}")

    if not args.date:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(target_date)
        except OSError as exc:
            print(f"WARN: could not write lock file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
