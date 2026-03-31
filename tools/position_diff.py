#!/usr/bin/env python3
"""
position_diff.py — Position change detection & automatic recording

Called periodically by secretary. Compares current OANDA positions/trades with the previous
snapshot, and automatically records any changes to state.md and daily/trades.md.

Usage:
    python3 tools/position_diff.py [--collab]

    --collab: Collab trade mode. Writes to collab_trade/state.md and collab_trade/daily/

Output (stdout JSON):
    {
        "changes": [...],       # List of detected changes
        "open_positions": [...], # Current open positions
        "recorded": true/false   # Whether records were written
    }
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

# --- Config ---
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    """Load config from env.toml. Works with Python 3.10+ (no tomllib needed)."""
    cfg = {}
    with open(os.path.join(REPO_DIR, "config", "env.toml")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                cfg[key] = val
    return cfg

def oanda_get(path, cfg):
    base = "https://api-fxtrade.oanda.com/v3"
    url = f"{base}{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {cfg['oanda_token']}",
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

# --- Snapshot management ---
SNAPSHOT_PATH = os.path.join(REPO_DIR, "logs", "position_snapshot.json")

def load_snapshot():
    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    return {"trades": {}, "timestamp": None}

def save_snapshot(trades_dict):
    snap = {
        "trades": trades_dict,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snap, f, indent=2)

# --- OANDA fetch ---
def get_open_trades(cfg):
    acc = cfg["oanda_account_id"]
    data = oanda_get(f"/accounts/{acc}/openTrades", cfg)
    trades = {}
    for t in data.get("trades", []):
        trades[t["id"]] = {
            "id": t["id"],
            "instrument": t["instrument"].replace("_", "/"),
            "instrument_raw": t["instrument"],
            "units": int(t["currentUnits"]),
            "direction": "LONG" if int(t["currentUnits"]) > 0 else "SHORT",
            "price": float(t["price"]),
            "unrealizedPL": float(t.get("unrealizedPL", 0)),
            "openTime": t["openTime"],
        }
    return trades

def get_account_summary(cfg):
    acc = cfg["oanda_account_id"]
    data = oanda_get(f"/accounts/{acc}/summary", cfg)
    a = data["account"]
    return {
        "balance": float(a["balance"]),
        "nav": float(a["NAV"]),
        "unrealizedPL": float(a["unrealizedPL"]),
        "pl": float(a["pl"]),
    }

def get_recent_closed_trades(cfg, count=20):
    """Get recently closed trades to detect closes since last snapshot."""
    acc = cfg["oanda_account_id"]
    data = oanda_get(f"/accounts/{acc}/trades?state=CLOSED&count={count}", cfg)
    trades = []
    for t in data.get("trades", []):
        trades.append({
            "id": t["id"],
            "instrument": t["instrument"],
            "units": int(t.get("initialUnits", t.get("currentUnits", 0))),
            "direction": "LONG" if int(t.get("initialUnits", t.get("currentUnits", 0))) > 0 else "SHORT",
            "price": float(t["price"]),
            "realizedPL": float(t.get("realizedPL", 0)),
            "openTime": t.get("openTime", ""),
            "closeTime": t.get("closeTime", ""),
        })
    return trades

# --- Diff detection ---
def detect_changes(prev_trades, curr_trades, recent_closed):
    changes = []
    prev_ids = set(prev_trades.keys())
    curr_ids = set(curr_trades.keys())

    # New trades
    for tid in curr_ids - prev_ids:
        t = curr_trades[tid]
        changes.append({
            "type": "OPEN",
            "trade_id": tid,
            "instrument": t["instrument_raw"],
            "direction": t["direction"],
            "units": abs(t["units"]),
            "price": t["price"],
            "time": t["openTime"],
        })

    # Closed trades (were in prev, not in curr)
    closed_ids = prev_ids - curr_ids
    if closed_ids:
        # Match with recent closed to get realizedPL
        closed_map = {t["id"]: t for t in recent_closed}
        for tid in closed_ids:
            prev_t = prev_trades[tid]
            closed_info = closed_map.get(tid, {})
            changes.append({
                "type": "CLOSE",
                "trade_id": tid,
                "instrument": prev_t.get("instrument_raw", prev_t.get("instrument", "").replace("/", "_")),
                "direction": prev_t["direction"],
                "units": abs(prev_t["units"]),
                "open_price": prev_t["price"],
                "realizedPL": closed_info.get("realizedPL", 0),
                "close_time": closed_info.get("closeTime", ""),
            })

    return changes

# --- Recording ---
def now_utc():
    return datetime.now(timezone.utc)

def now_jst():
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(hours=9)

def record_to_state_md(curr_trades, account, collab_mode):
    """Update state.md with current positions. Only updates the Positions section."""
    if collab_mode:
        state_path = os.path.join(REPO_DIR, "collab_trade", "state.md")
    else:
        return  # auto mode doesn't use state.md

    if not os.path.exists(state_path):
        return

    now = now_utc().strftime("%Y-%m-%d %H:%M UTC")
    jst = now_jst().strftime("%H:%M JST")

    # Build position section
    lines = []
    lines.append(f"**Last Updated**: {now} ({jst}) [secretary auto-recorded]")
    lines.append("")
    lines.append("## Positions")
    lines.append("")

    if not curr_trades:
        lines.append("*No positions*")
    else:
        # Group by instrument + direction
        groups = {}
        for t in curr_trades.values():
            key = f"{t['instrument_raw']}_{t['direction']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(t)

        for key, trades in groups.items():
            inst = trades[0]["instrument_raw"]
            direction = trades[0]["direction"]
            total_units = sum(abs(t["units"]) for t in trades)
            total_upl = sum(t["unrealizedPL"] for t in trades)
            lines.append(f"### {inst} {direction} — {len(trades)} position(s)")
            for t in trades:
                upl_jpy = round(t["unrealizedPL"])
                lines.append(f"- Trade ID: {t['id']} | {t['price']} ({abs(t['units'])}u) → **{'+' if upl_jpy >= 0 else ''}{upl_jpy}JPY**")
            lines.append(f"- **Total**: {total_units}u, unrealized P&L {'+' if total_upl >= 0 else ''}{round(total_upl)}JPY")
            lines.append("")

    # Read existing state.md and replace the header + Positions section
    with open(state_path) as f:
        content = f.read()

    # Find where Positions section starts and the next ## section
    import re
    # Replace from "**Last Updated**" to the line before the next non-Positions "## " header
    # Strategy: keep everything from "## Today's Story" onwards
    pattern = r'\*\*最終更新\*\*.*?(?=## 今日の|## 確定|## 教訓|## メモ|\Z)'
    new_content = "\n".join(lines) + "\n\n"

    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_content, content, count=1, flags=re.DOTALL)
    else:
        # If pattern not found, append at the end
        content += "\n" + new_content

    with open(state_path, "w") as f:
        f.write(content)

def record_to_trades_md(changes, collab_mode):
    """Append trade changes to daily/YYYY-MM-DD/trades.md"""
    if not changes:
        return

    today = now_jst().strftime("%Y-%m-%d")
    if collab_mode:
        base = os.path.join(REPO_DIR, "collab_trade", "daily", today)
    else:
        base = os.path.join(REPO_DIR, "logs")  # auto mode

    os.makedirs(base, exist_ok=True)
    trades_path = os.path.join(base, "trades.md")

    now = now_utc().strftime("%H:%M:%S UTC")
    lines = []
    for c in changes:
        if c["type"] == "OPEN":
            lines.append(f"- [{now}] **OPEN** {c['instrument']} {c['direction']} {c['units']}u @ {c['price']} (TradeID: {c['trade_id']}) [secretary detected]")
        elif c["type"] == "CLOSE":
            pl = c.get("realizedPL", 0)
            pl_str = f"+{round(pl)}JPY" if pl >= 0 else f"{round(pl)}JPY"
            lines.append(f"- [{now}] **CLOSE** {c['instrument']} {c['direction']} {c['units']}u (TradeID: {c['trade_id']}) → {pl_str} [secretary detected]")

    if lines:
        with open(trades_path, "a") as f:
            f.write("\n".join(lines) + "\n")

def record_to_live_trade_log(changes):
    """Append to the main live_trade_log.txt for trade_performance.py to parse."""
    if not changes:
        return
    log_path = os.path.join(REPO_DIR, "logs", "live_trade_log.txt")
    now = now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for c in changes:
        if c["type"] == "OPEN":
            lines.append(f"[{now}] SECRETARY-DETECTED: OPEN {c['instrument']} {c['direction']} {c['units']}u @ {c['price']} tradeId={c['trade_id']}")
        elif c["type"] == "CLOSE":
            pl = c.get("realizedPL", 0)
            lines.append(f"[{now}] SECRETARY-DETECTED: CLOSE {c['instrument']} {c['direction']} {c['units']}u tradeId={c['trade_id']} PL={round(pl)}JPY")
    if lines:
        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

# --- Main ---
def main():
    collab_mode = "--collab" in sys.argv

    cfg = load_config()
    prev = load_snapshot()
    prev_trades = prev["trades"]

    # Fetch current state
    curr_trades = get_open_trades(cfg)
    account = get_account_summary(cfg)
    recent_closed = get_recent_closed_trades(cfg)

    # Detect changes
    changes = detect_changes(prev_trades, curr_trades, recent_closed)

    # Record if changes found
    recorded = False
    if changes:
        record_to_trades_md(changes, collab_mode)
        record_to_live_trade_log(changes)
        recorded = True

    # Always update state.md with current positions (keeps it fresh)
    if collab_mode:
        record_to_state_md(curr_trades, account, collab_mode)

    # Save new snapshot
    save_snapshot(curr_trades)

    # Output
    result = {
        "changes": changes,
        "open_positions": [
            {
                "instrument": t["instrument_raw"],
                "direction": t["direction"],
                "units": abs(t["units"]),
                "price": t["price"],
                "unrealizedPL": round(t["unrealizedPL"]),
            }
            for t in curr_trades.values()
        ],
        "account": account,
        "recorded": recorded,
        "collab_mode": collab_mode,
        "prev_snapshot_time": prev["timestamp"],
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
