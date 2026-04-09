"""
QuantRabbit — Daily Review Data Gatherer
Fact collection for daily review. No judgment — just lay out the facts.

Called by the daily-review scheduled task. Claude reads this output and
writes the review to strategy_memory.md.

Usage:
  python3 tools/daily_review.py --date 2026-03-27
  python3 tools/daily_review.py  # today
"""
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "collab_trade" / "memory"))
from schema import get_conn, init_db, fetchall_dict, fetchone_val

ENV_TOML = ROOT / "config" / "env.toml"
OANDA_PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}


def _load_oanda_config():
    text = ENV_TOML.read_text()
    cfg = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def fetch_closed_trades(session_date: str) -> list[dict]:
    """Fetch closed trades for the specified date from the OANDA API"""
    try:
        cfg = _load_oanda_config()
    except Exception:
        return []

    token = cfg.get('oanda_token', '')
    acct = cfg.get('oanda_account_id', '')
    base = 'https://api-fxtrade.oanda.com'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    since = f"{session_date}T00:00:00.000000000Z"
    dt = datetime.strptime(session_date, '%Y-%m-%d')
    to = (dt + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00.000000000Z')

    try:
        import urllib.parse
        params = urllib.parse.urlencode({
            'from': since, 'to': to, 'type': 'ORDER_FILL', 'pageSize': 1000,
        })
        url = f"{base}/v3/accounts/{acct}/transactions?{params}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            id_range = json.loads(resp.read())

        all_txns = []
        for page_url in id_range.get('pages', []):
            req = urllib.request.Request(page_url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                all_txns.extend(data.get('transactions', []))
    except Exception as e:
        print(f"OANDA API error: {e}")
        return []

    entries = {}
    closes = {}
    for txn in all_txns:
        if txn.get('type') != 'ORDER_FILL':
            continue
        instrument = txn.get('instrument', '')
        if instrument not in OANDA_PAIRS:
            continue

        trade_opened = txn.get('tradeOpened')
        if trade_opened:
            tid = trade_opened.get('tradeID', '')
            units = int(float(trade_opened.get('units', 0)))
            entries[tid] = {
                'price': float(txn.get('price', 0)),
                'units': abs(units),
                'direction': 'LONG' if units > 0 else 'SHORT',
                'pair': instrument,
                'time': txn.get('time', ''),
            }

        for tc in txn.get('tradesClosed', []) + txn.get('tradesReduced', []):
            tid = tc.get('tradeID', '')
            pl = float(tc.get('realizedPL', 0))
            units_c = abs(int(float(tc.get('units', 0))))
            close_time = txn.get('time', '')
            if tid not in closes:
                closes[tid] = {
                    'price': float(txn.get('price', 0)),
                    'pl': pl, 'units': units_c,
                    'instrument': instrument, 'time': close_time,
                }
            else:
                closes[tid]['pl'] += pl
                closes[tid]['units'] += units_c

    trades = []
    for tid in set(list(entries.keys()) + list(closes.keys())):
        entry = entries.get(tid)
        close = closes.get(tid)
        if not close:
            continue
        pair = entry['pair'] if entry else close.get('instrument', 'UNKNOWN')
        direction = entry['direction'] if entry else ('LONG' if close['pl'] >= 0 else 'SHORT')

        # Hold time calculation
        hold_min = None
        if entry and close:
            try:
                t1 = datetime.fromisoformat(entry['time'].replace('Z', '+00:00').split('.')[0] + '+00:00')
                t2 = datetime.fromisoformat(close['time'].replace('Z', '+00:00').split('.')[0] + '+00:00')
                hold_min = int((t2 - t1).total_seconds() / 60)
            except Exception:
                pass

        trades.append({
            'trade_id': tid,
            'pair': pair,
            'direction': direction,
            'units': entry['units'] if entry else close['units'],
            'entry_price': entry['price'] if entry else None,
            'exit_price': close['price'],
            'pl': close['pl'],
            'hold_minutes': hold_min,
        })

    return trades


def match_pretrade_outcomes(conn, session_date: str, oanda_trades: list[dict]):
    """Fill pl in the pretrade_outcomes table (linking predictions to results)"""
    # pretrade_outcomes for this day (where pl is still null)
    outcomes = fetchall_dict(conn,
        """SELECT id, pair, direction, pretrade_level
           FROM pretrade_outcomes
           WHERE session_date = ? AND pl IS NULL""",
        (session_date,))

    if not outcomes:
        return 0

    # Group by pair × direction
    from collections import defaultdict
    trade_by_key = defaultdict(list)
    for t in oanda_trades:
        trade_by_key[(t['pair'], t['direction'])].append(t)

    matched = 0
    for outcome in outcomes:
        key = (outcome['pair'], outcome['direction'])
        trades_for_key = trade_by_key.get(key, [])
        if not trades_for_key:
            continue

        # Link the first unmatched trade
        trade = trades_for_key.pop(0)
        conn.execute(
            "UPDATE pretrade_outcomes SET pl = ?, trade_id = ? WHERE id = ?",
            (trade['pl'], trade['trade_id'], outcome['id'])
        )
        matched += 1

    return matched


def parse_pretrade_from_trades_md(trades_md_path: Path) -> list[dict]:
    """Extract entries with pretrade results from trades.md"""
    if not trades_md_path.exists():
        return []

    text = trades_md_path.read_text()
    entries = []

    # Extract pretrade results from table rows
    for line in text.split('\n'):
        if 'pretrade' not in line.lower() and 'LOW' not in line and 'MEDIUM' not in line and 'HIGH' not in line:
            continue
        # Extract pair and direction
        pair_match = re.search(r'(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)', line)
        dir_match = re.search(r'(LONG|SHORT)', line)
        level_match = re.search(r'(?:pretrade|PRETRADE)[:\s]*(\w+)', line)
        if not level_match:
            level_match = re.search(r'(LOW|MEDIUM|HIGH)', line)

        if pair_match and dir_match and level_match:
            entries.append({
                'pair': pair_match.group(1),
                'direction': dir_match.group(1),
                'pretrade_level': level_match.group(1).upper(),
                'line': line.strip(),
            })

    return entries


def analyze_s_scan_outcomes(session_date: str, oanda_trades: list[dict]) -> list[str]:
    """Analyze S-scan detections from audit_history.jsonl.
    For each NOT_HELD detection, check:
    1. Did the trader eventually enter? (matched against oanda_trades)
    2. Was the direction call correct? (price moved in predicted direction)
    Returns lines for the report, or empty list if no data."""
    history_path = ROOT / "logs" / "audit_history.jsonl"
    if not history_path.exists():
        return []

    # Collect all NOT_HELD detections for this date
    detections = []  # {pair, direction, recipe, price, timestamp}
    try:
        for line in history_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            ts = entry.get("timestamp", "")
            if not ts.startswith(session_date):
                continue
            for sc in entry.get("s_scan", []):
                if isinstance(sc, dict) and sc.get("status") == "NOT_HELD":
                    detections.append({
                        "pair": sc.get("pair", "?"),
                        "direction": sc.get("direction", "?"),
                        "recipe": sc.get("recipe", "?"),
                        "price": sc.get("price_at_detection", 0),
                        "timestamp": ts,
                    })
    except Exception:
        return []

    if not detections:
        return []

    # Deduplicate: keep first detection per (pair, direction, recipe)
    seen = set()
    unique = []
    for d in detections:
        key = (d["pair"], d["direction"], d["recipe"])
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # Build closed-trade lookup: (pair, direction) → list of P&L
    trade_lookup = {}
    for t in oanda_trades:
        key = (t["pair"], "LONG" if t["units"] > 0 else "SHORT")
        trade_lookup.setdefault(key, []).append(t["pl"])

    # Fetch current prices for outcome check
    current_prices = {}
    try:
        cfg = _load_oanda_config()
        pairs_str = ",".join(set(d["pair"] for d in unique))
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{cfg['oanda_account_id']}/pricing?instruments={pairs_str}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json"
        })
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        for p in resp.get("prices", []):
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
            current_prices[p["instrument"]] = (bid + ask) / 2
    except Exception:
        pass

    # Analyze each detection
    lines = ["## S-Scan Outcome Analysis", ""]
    recipe_stats = {}  # recipe → {correct: N, wrong: N, entered: N, missed: N}

    for d in unique:
        pair = d["pair"]
        direction = d["direction"]
        recipe = d["recipe"]
        det_price = d["price"]

        if recipe not in recipe_stats:
            recipe_stats[recipe] = {"correct": 0, "wrong": 0, "entered": 0, "missed": 0}

        # Did trader enter this setup?
        key = (pair, direction)
        entered = key in trade_lookup
        if entered:
            recipe_stats[recipe]["entered"] += 1
            pl = sum(trade_lookup[key])
            result = "WIN" if pl > 0 else "LOSS"
            lines.append(f"  {pair} {direction} {recipe} @{det_price} → ENTERED → {result} {pl:+,.0f} JPY")
            if pl > 0:
                recipe_stats[recipe]["correct"] += 1
            else:
                recipe_stats[recipe]["wrong"] += 1
        else:
            recipe_stats[recipe]["missed"] += 1
            # Check if direction was correct using current price
            cur = current_prices.get(pair)
            if cur and det_price > 0:
                moved_right = (direction == "LONG" and cur > det_price) or \
                              (direction == "SHORT" and cur < det_price)
                pip_move = abs(cur - det_price) * (100 if "JPY" in pair else 10000)
                verdict = "CORRECT" if moved_right else "WRONG"
                lines.append(f"  {pair} {direction} {recipe} @{det_price} → MISSED → {verdict} ({pip_move:.1f}pip moved)")
                if moved_right:
                    recipe_stats[recipe]["correct"] += 1
                else:
                    recipe_stats[recipe]["wrong"] += 1
            else:
                lines.append(f"  {pair} {direction} {recipe} @{det_price} → MISSED → no price data")

    # Summary per recipe
    lines.append("")
    lines.append("### Recipe Accuracy Summary")
    lines.append("")
    for recipe, stats in sorted(recipe_stats.items()):
        total = stats["correct"] + stats["wrong"]
        acc = stats["correct"] / total * 100 if total > 0 else 0
        lines.append(
            f"  {recipe}: {acc:.0f}% accuracy ({stats['correct']}/{total}) "
            f"| entered: {stats['entered']} missed: {stats['missed']}"
        )
    lines.append("")

    return lines


def generate_report(session_date: str) -> str:
    """Generate a structured report for the daily review"""
    conn = get_conn()
    lines = []

    lines.append(f"# Daily Review Report: {session_date}")
    lines.append("")

    # 1. OANDA closed trades
    oanda_trades = fetch_closed_trades(session_date)
    lines.append(f"## Closed Trades ({len(oanda_trades)} trades)")
    lines.append("")

    if not oanda_trades:
        lines.append("No closed trades")
    else:
        total_pl = sum(t['pl'] for t in oanda_trades)
        wins = [t for t in oanda_trades if t['pl'] > 0]
        losses = [t for t in oanda_trades if t['pl'] <= 0]
        lines.append(f"Total P&L: {total_pl:+,.0f} JPY | Win: {len(wins)} Loss: {len(losses)} | Win rate: {len(wins)/len(oanda_trades)*100:.0f}%")
        lines.append("")

        # Per-pair summary
        from collections import defaultdict
        by_pair = defaultdict(list)
        for t in oanda_trades:
            by_pair[t['pair']].append(t)

        for pair, trades in sorted(by_pair.items()):
            pair_pl = sum(t['pl'] for t in trades)
            pair_wins = sum(1 for t in trades if t['pl'] > 0)
            lines.append(f"### {pair}: {pair_pl:+,.0f} JPY ({pair_wins}/{len(trades)} wins)")
            for t in trades:
                hold = f" ({t['hold_minutes']}min)" if t['hold_minutes'] is not None else ""
                lines.append(f"  {t['direction']} {t['units']}u → {t['pl']:+,.0f} JPY{hold}")
            lines.append("")

    # 2. pretrade_outcomes matching
    matched = match_pretrade_outcomes(conn, session_date, oanda_trades)
    if matched:
        lines.append(f"## Pretrade Outcomes: {matched} matched")
        lines.append("")

    # 3. pretrade results vs actual P&L (feedback for this day)
    outcomes = fetchall_dict(conn,
        """SELECT pair, direction, pretrade_level, pretrade_score, pl, pretrade_warnings
           FROM pretrade_outcomes
           WHERE session_date = ? AND pl IS NOT NULL
           ORDER BY pl""",
        (session_date,))

    if outcomes:
        lines.append("## Pretrade Prediction vs Result")
        lines.append("")
        for o in outcomes:
            result = "WIN" if o['pl'] > 0 else "LOSS"
            lines.append(f"  {o['pair']} {o['direction']} pretrade={o['pretrade_level']}(score={o['pretrade_score']}) → {result} {o['pl']:+,.0f} JPY")
        lines.append("")

        # Analysis of ignored LOW entries
        low_entries = [o for o in outcomes if o['pretrade_level'] == 'LOW']
        if low_entries:
            low_wins = sum(1 for o in low_entries if o['pl'] > 0)
            low_total_pl = sum(o['pl'] for o in low_entries)
            lines.append(f"### LOW-ignored entries: {len(low_entries)} trades")
            lines.append(f"  Win rate: {low_wins}/{len(low_entries)} | Total: {low_total_pl:+,.0f} JPY")
            lines.append("")

    # 4. Entries with pretrade notes from trades.md (also catches those not in DB)
    trades_md_path = ROOT / "collab_trade" / "daily" / session_date / "trades.md"
    md_entries = parse_pretrade_from_trades_md(trades_md_path)
    if md_entries:
        lines.append("## Pretrade-annotated entries in trades.md")
        lines.append("")
        for e in md_entries:
            lines.append(f"  {e['pair']} {e['direction']} pretrade={e['pretrade_level']}")
        lines.append("")

    # 5. Win patterns vs loss patterns
    if oanda_trades:
        lines.append("## Pattern Analysis")
        lines.append("")

        # Average hold time: wins vs losses
        win_holds = [t['hold_minutes'] for t in wins if t['hold_minutes'] is not None]
        loss_holds = [t['hold_minutes'] for t in losses if t['hold_minutes'] is not None]
        if win_holds:
            lines.append(f"  Avg hold (wins): {sum(win_holds)/len(win_holds):.0f} min")
        if loss_holds:
            lines.append(f"  Avg hold (losses): {sum(loss_holds)/len(loss_holds):.0f} min")

        # Best/worst trade
        if wins:
            best = max(wins, key=lambda t: t['pl'])
            lines.append(f"  Best win: {best['pair']} {best['direction']} {best['pl']:+,.0f} JPY")
        if losses:
            worst = min(losses, key=lambda t: t['pl'])
            lines.append(f"  Worst loss: {worst['pair']} {worst['direction']} {worst['pl']:+,.0f} JPY")

        lines.append("")

    # 6. S-scan outcome tracking (audit_history.jsonl)
    s_scan_lines = analyze_s_scan_outcomes(session_date, oanda_trades)
    if s_scan_lines:
        lines.extend(s_scan_lines)

    # 7. DB trades stats (comparison against all-time)
    db_stats = fetchall_dict(conn,
        """SELECT pair, direction,
           COUNT(*) as cnt,
           SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) as wins,
           AVG(pl) as avg_pl,
           SUM(pl) as total_pl
           FROM trades WHERE pl IS NOT NULL
           GROUP BY pair, direction
           ORDER BY total_pl""")

    if db_stats:
        lines.append("## All-time stats by pair × direction (reference)")
        lines.append("")
        for s in db_stats:
            wr = s['wins'] / s['cnt'] * 100 if s['cnt'] > 0 else 0
            lines.append(f"  {s['pair']} {s['direction']}: {s['wins']}/{s['cnt']} wins ({wr:.0f}%) avg {s['avg_pl']:+,.0f} JPY total {s['total_pl']:+,.0f} JPY")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    init_db()

    target_date = str(date.today())
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--date" and i < len(sys.argv) - 1:
            target_date = sys.argv[i + 1]

    report = generate_report(target_date)
    print(report)
