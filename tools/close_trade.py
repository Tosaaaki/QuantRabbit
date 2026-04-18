#!/usr/bin/env python3
"""Trade closer — PUT /trades/{id}/close でポジションを安全に決済する。

Usage:
    python3 tools/close_trade.py {tradeID} [units] [--reason REASON] [--auto-log] [--auto-slack] [--force-worker-close]

Examples:
    python3 tools/close_trade.py 465743                              # 全決済 (manual log/slack)
    python3 tools/close_trade.py 465743 1000                         # 部分決済
    python3 tools/close_trade.py 465743 --reason zombie_hold --auto-log --auto-slack  # 全自動
    python3 tools/close_trade.py 465743 1000 --reason half_tp --auto-log --auto-slack  # 部分+全自動
    python3 tools/close_trade.py 468302 --reason worker_emergency_override --auto-log --auto-slack --force-worker-close
"""

from __future__ import annotations

import argparse
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from oanda_trade_tags import attach_trade_extensions

ROOT = Path(__file__).resolve().parent.parent
WORKER_TAGS = {"range_bot", "range_bot_market", "trend_bot_market"}
BOT_OWNED_CLOSE_PREFIXES = ("bot_",)
FORCE_WORKER_CLOSE_EMERGENCY_REASONS = {
    "worker_emergency_override",
    "panic_margin",
    "deadlock_relief",
    "worker_policy_breach",
    "rollover_emergency",
}


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    cfg = load_config()
    acct = cfg["oanda_account_id"]
    url = f"https://api-fxtrade.oanda.com{path.format(account_id=acct)}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {cfg['oanda_token']}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_tag(payload: dict) -> str:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = payload.get(key, {}) or {}
        tag = ext.get("tag")
        if tag:
            return str(tag)
    return ""


def parse_oanda_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    normalized = ts
    if ts.endswith("Z"):
        core = ts[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            normalized = f"{head}.{frac[:6]}+00:00"
        else:
            normalized = f"{core}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def fetch_trade(trade_id: str) -> dict:
    trade = oanda_request(f"/v3/accounts/{{account_id}}/trades/{trade_id}").get("trade", {})
    if not trade:
        return trade
    return attach_trade_extensions(trade, oanda_request)


def enforce_worker_guard(trade: dict, reason: str, force_worker_close: bool) -> None:
    tag = get_tag(trade)
    if tag not in WORKER_TAGS:
        return

    normalized_reason = str(reason or "").strip()
    if normalized_reason.startswith(BOT_OWNED_CLOSE_PREFIXES):
        return
    if force_worker_close and reason in FORCE_WORKER_CLOSE_EMERGENCY_REASONS:
        return
    opened_at = parse_oanda_time(trade.get("openTime"))
    age_note = ""
    if opened_at is not None:
        age_min = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
        age_note = f" ({age_min:.1f}m old)"
    if force_worker_close and reason not in FORCE_WORKER_CLOSE_EMERGENCY_REASONS:
        pair = trade.get("instrument", "?")
        allowed = ", ".join(sorted(FORCE_WORKER_CLOSE_EMERGENCY_REASONS))
        raise RuntimeError(
            f"worker-close-guard: {pair} trade {trade.get('id', '?')}{age_note} is worker-owned. "
            f"--force-worker-close is limited to emergency reasons: {allowed}."
        )
    pair = trade.get("instrument", "?")
    allowed = ", ".join(sorted(FORCE_WORKER_CLOSE_EMERGENCY_REASONS))
    raise RuntimeError(
        f"worker-close-guard: {pair} trade {trade.get('id', '?')}{age_note} stays worker-owned. "
        f"Use policy steering, or override only with --force-worker-close and an emergency reason: {allowed}."
    )


def close_trade(
    trade_id: str,
    units: str | None = None,
    reason: str = "",
    auto_log: bool = False,
    auto_slack: bool = False,
    force_worker_close: bool = False,
):
    cfg = load_config()
    acct = cfg["oanda_account_id"]
    trade = fetch_trade(trade_id)
    enforce_worker_guard(trade, reason, force_worker_close)

    body = {}
    if units:
        body["units"] = units

    try:
        result = oanda_request(f"/v3/accounts/{{account_id}}/trades/{trade_id}/close", method="PUT", body=body or {})

        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            inst = fill.get("instrument", "?")
            pl = fill.get("pl", "0")
            price = fill.get("price", "?")
            closed_units = fill.get("units", "?")
            # Determine side from units sign
            u = int(closed_units) if closed_units != "?" else 0
            side = "LONG" if u < 0 else "SHORT"  # close fills have opposite sign
            abs_units = abs(u)
            is_partial = units is not None

            # Get spread from fill
            half_spread = float(fill.get("halfSpreadCost", "0"))
            # Estimate spread in pips
            pip_mult = 100 if "JPY" in inst else 10000
            spread_pips = abs(half_spread * 2 / abs_units * pip_mult) if abs_units > 0 else 0

            print(f"CLOSED: {inst} {closed_units}u @{price} P/L={pl} JPY")

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            reason_str = f" reason={reason}" if reason else ""
            action = "PARTIAL_CLOSE" if is_partial else "CLOSE"

            # Auto-log to live_trade_log.txt
            if auto_log:
                log_line = f"[{now_str}] {action} {inst} {side} {abs_units}u @{price} P/L={pl}JPY Sp={spread_pips:.1f}pip{reason_str} id={trade_id}\n"
                log_path = ROOT / "logs" / "live_trade_log.txt"
                with open(log_path, "a") as f:
                    f.write(log_line)
                print(f"  → logged to live_trade_log.txt")

            # Auto-slack notification
            if auto_slack:
                try:
                    import subprocess
                    if is_partial:
                        slack_cmd = [
                            sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
                            "modify",
                            "--pair", inst,
                            "--action", f"PARTIAL CLOSE ({reason or 'scale-out'})",
                            "--units", str(abs_units),
                            "--price", str(price),
                            "--pl", f"{pl}JPY",
                        ]
                    else:
                        slack_cmd = [
                            sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
                            "close",
                            "--pair", inst,
                            "--side", side,
                            "--units", str(abs_units),
                            "--price", str(price),
                            "--pl", f"{pl}JPY",
                        ]
                    subprocess.run(slack_cmd, capture_output=True, timeout=10)
                    print(f"  → Slack notified")
                except Exception as e:
                    print(f"  → Slack failed: {e}")

            return result
        else:
            print(json.dumps(result, indent=2))
            return result

    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        print(f"ERROR {e.code}: {err.get('errorMessage', err)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trade_id")
    parser.add_argument("units", nargs="?")
    parser.add_argument("--reason", default="")
    parser.add_argument("--auto-log", action="store_true")
    parser.add_argument("--auto-slack", action="store_true")
    parser.add_argument("--force-worker-close", action="store_true")
    args = parser.parse_args()

    close_trade(
        args.trade_id,
        args.units,
        args.reason,
        args.auto_log,
        args.auto_slack,
        args.force_worker_close,
    )
