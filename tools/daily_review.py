"""
QuantRabbit — Daily Review Data Gatherer
Fact collection for daily review. No judgment — just lay out the facts.

Called by the daily-review scheduled task. Claude reads this output and
writes the review to strategy_memory.md.

Usage:
  python3 tools/daily_review.py --date 2026-03-27
  python3 tools/daily_review.py  # most recent completed UTC trading day
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "collab_trade" / "memory"))
from schema import get_conn, init_db, fetchall_dict, fetchone_val

ENV_TOML = ROOT / "config" / "env.toml"
OANDA_PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}
JST = timezone(timedelta(hours=9))
PRETRADE_LOOKBACK_DAYS = 3
PRETRADE_MATCH_HOURS = 72
RECURRING_TAG_PREFIXES = ("trader",)
LESSON_REGISTRY_JSON = ROOT / "collab_trade" / "memory" / "lesson_registry.json"
MAX_AUDIT_SIGNAL_GAP_HOURS = 12
HOSTILE_LIVE_TAPE_BUCKETS = {"friction", "spread_unstable", "two_way", "opposed", "unavailable"}


def payoff_metrics(pls: list[float]) -> dict:
    """Realized payoff quality for review output."""
    if not pls:
        return {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "rr_ratio": 0.0,
            "expectancy": 0.0,
            "break_even_win_rate": None,
        }

    wins = [pl for pl in pls if pl > 0]
    losses = [pl for pl in pls if pl < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss < 0:
        rr_ratio = avg_win / abs(avg_loss)
    elif avg_win > 0 and not losses:
        rr_ratio = float("inf")
    else:
        rr_ratio = 0.0

    break_even_win_rate = None
    if avg_win > 0 and avg_loss < 0:
        break_even_win_rate = abs(avg_loss) / (avg_win + abs(avg_loss))
    elif avg_win > 0 and not losses:
        break_even_win_rate = 0.0

    return {
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio,
        "expectancy": sum(pls) / len(pls),
        "break_even_win_rate": break_even_win_rate,
    }


def _load_oanda_config():
    text = ENV_TOML.read_text()
    cfg = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def completed_utc_review_day(now: datetime | None = None) -> str:
    """Return the most recent fully completed UTC trading day."""
    current = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    return (current.date() - timedelta(days=1)).isoformat()


def review_day_bounds_utc(session_date: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(session_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def review_window_label(session_date: str) -> str:
    start_utc, end_utc = review_day_bounds_utc(session_date)
    start_jst = start_utc.astimezone(JST)
    end_jst = end_utc.astimezone(JST)
    return (
        f"{start_utc.strftime('%Y-%m-%d %H:%M')}→{end_utc.strftime('%Y-%m-%d %H:%M')} UTC "
        f"({start_jst.strftime('%m/%d %H:%M')}→{end_jst.strftime('%m/%d %H:%M')} JST)"
    )


def load_trade_tags_from_log() -> dict[str, dict[str, str]]:
    """Recover trade tags/comments when the opening fill is outside the review window."""
    path = ROOT / "logs" / "live_trade_log.txt"
    if not path.exists():
        return {}

    tags: dict[str, dict[str, str]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        id_match = re.search(r"\bid=(\d+)\b", line)
        if not id_match:
            continue
        tag_match = re.search(r"\btag=([A-Za-z0-9_:-]+)", line)
        comment_match = re.search(r"\bcomment=([^\s|]+)", line)
        if not tag_match and not comment_match:
            continue
        tags[id_match.group(1)] = {
            "tag": tag_match.group(1) if tag_match else "",
            "comment": comment_match.group(1) if comment_match else "",
        }
    return tags


def load_daily_trade_id_evidence(session_date: str) -> dict[str, str]:
    """Recover ownership hints from the daily trade journal when tags are missing."""
    path = ROOT / "collab_trade" / "daily" / session_date / "trades.md"
    if not path.exists():
        return {}

    evidence: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        lower = line.lower()
        ids = set()
        ids.update(re.findall(r"\b(?:trade id|id)\s*(?:\||=|:)\s*(\d+)\b", line, flags=re.IGNORECASE))
        table_match = re.match(r"^\|\s*(\d{5,})\s*\|", line)
        if table_match:
            ids.add(table_match.group(1))
        if not ids:
            continue

        source = "daily_record"
        if "bot" in lower or "legacy" in lower:
            source = "bot_record"

        for tid in ids:
            if source == "bot_record":
                evidence[tid] = source
            else:
                evidence.setdefault(tid, source)
    return evidence


def is_recurring_trader_trade(tag: str | None, comment: str | None) -> bool:
    lowered_tag = (tag or "").lower()
    lowered_comment = (comment or "").lower()
    return lowered_tag.startswith(RECURRING_TAG_PREFIXES) or lowered_comment.startswith("qr-trader")


def classify_trade_ownership(
    trade_id: str,
    tag: str | None,
    comment: str | None,
    daily_evidence: dict[str, str],
) -> tuple[str, str]:
    if is_recurring_trader_trade(tag, comment):
        return "recurring_trader", "explicit_tag_or_comment"

    if daily_evidence.get(trade_id) == "daily_record":
        return "recurring_trader", "daily_trade_record"

    if tag or comment:
        return "other_execution", "explicit_non_recurring_tag"

    if daily_evidence.get(trade_id) == "bot_record":
        return "other_execution", "daily_bot_record"

    return "other_execution", "unresolved"


def summarize_trade_bucket(title: str, trades: list[dict], tag_coverage: tuple[int, int] | None = None) -> list[str]:
    lines = [f"### {title}: {len(trades)} trades"]
    if tag_coverage is not None:
        tagged, total = tag_coverage
        lines.append(f"  Tag coverage: {tagged}/{total} trades linked to an explicit tag/comment")
    if not trades:
        lines.append("  No closed trades in this bucket.")
        lines.append("")
        return lines

    total_pl = sum(t["pl"] for t in trades)
    metrics = payoff_metrics([t["pl"] for t in trades])
    wins = [t for t in trades if t["pl"] > 0]
    losses = [t for t in trades if t["pl"] <= 0]
    lines.append(
        f"  P&L: {total_pl:+,.0f} JPY | Win: {metrics['wins']} Loss: {metrics['losses']} "
        f"| Win rate: {metrics['win_rate']*100:.0f}% | EV {metrics['expectancy']:+,.0f}"
    )
    lines.append(
        f"  Avg win: {metrics['avg_win']:+,.0f} | Avg loss: {metrics['avg_loss']:+,.0f} | R:R {metrics['rr_ratio']:.2f}"
    )
    by_pair: dict[str, list[dict]] = {}
    for trade in trades:
        by_pair.setdefault(trade["pair"], []).append(trade)
    for pair, pair_trades in sorted(by_pair.items(), key=lambda item: sum(t["pl"] for t in item[1])):
        pair_pl = sum(t["pl"] for t in pair_trades)
        lines.append(f"  {pair}: {pair_pl:+,.0f} JPY across {len(pair_trades)} trades")
    if wins:
        best = max(wins, key=lambda t: t["pl"])
        lines.append(f"  Best win: {best['pair']} {best['direction']} {best['pl']:+,.0f} JPY")
    if losses:
        worst = min(losses, key=lambda t: t["pl"])
        lines.append(f"  Worst loss: {worst['pair']} {worst['direction']} {worst['pl']:+,.0f} JPY")
    lines.append("")
    return lines


def memory_promotion_gate(recurring_trades: list[dict], other_trades: list[dict]) -> list[str]:
    def summarize_pairs(trades: list[dict]) -> list[str]:
        buckets: dict[tuple[str, str], dict[str, float]] = {}
        for trade in trades:
            key = (trade["pair"], trade["direction"])
            bucket = buckets.setdefault(key, {"count": 0, "pl": 0.0})
            bucket["count"] += 1
            bucket["pl"] += trade["pl"]
        ordered = sorted(
            buckets.items(),
            key=lambda item: (abs(item[1]["pl"]), item[1]["count"]),
            reverse=True,
        )
        return [
            f"{pair} {direction} ({bucket['count']} trades, {bucket['pl']:+,.0f} JPY)"
            for (pair, direction), bucket in ordered[:5]
        ]

    lines = ["## Memory Promotion Gate", ""]
    if recurring_trades:
        lines.append("Promote into recurring trader memory only from this clean evidence:")
        for item in summarize_pairs(recurring_trades):
            lines.append(f"  - {item}")
    else:
        lines.append("Promote no pair/direction lesson today: there is no closed recurring-trader evidence.")

    lines.append("")
    if other_trades:
        lines.append("Quarantine this evidence from strategy_memory pair lessons:")
        for item in summarize_pairs(other_trades):
            lines.append(f"  - {item}")
        lines.append(
            "  - If a quarantine trade reveals a process/tooling issue, rewrite it as a hygiene lesson only after separating it from the non-recurring execution path."
        )
    else:
        lines.append("No quarantine execution detected.")
    lines.append("")
    return lines


def lesson_registry_state_suggestions() -> list[str]:
    if not LESSON_REGISTRY_JSON.exists():
        return []
    try:
        registry = json.loads(LESSON_REGISTRY_JSON.read_text())
    except Exception:
        return []

    queue = registry.get("review_queue") or []
    if not queue:
        return []

    lines = ["## Lesson State Suggestions", ""]
    lines.append("Apply these suggestions when rewriting strategy_memory.md. Registry state is not the source of truth; the markdown edit is.")
    lines.append("")

    def rank(state: str) -> int:
        return {
            "deprecated": 1,
            "candidate": 2,
            "watch": 3,
            "confirmed": 4,
        }.get(state, 0)

    promote = [item for item in queue if rank(item.get("suggested_state", "")) > rank(item.get("state", ""))]
    demote = [
        item for item in queue
        if rank(item.get("suggested_state", "")) < rank(item.get("state", ""))
        and item.get("suggested_state") != "deprecated"
    ]
    stale = [item for item in queue if item.get("suggested_state") == "deprecated"]

    def emit(title: str, items: list[dict]):
        if not items:
            return
        lines.append(f"### {title}")
        for item in items[:5]:
            reasons = "; ".join(item.get("suggestion_reasons") or []) or "state drift"
            pair = f"{item.get('pair')} " if item.get("pair") else ""
            lines.append(
                f"  {item.get('state')} -> {item.get('suggested_state')} | trust {item.get('trust_score')} | "
                f"{pair}{item.get('title')} | {reasons}"
            )
        lines.append("")

    emit("Promote", promote)
    emit("Demote", demote)
    emit("Stale / Drop", stale)

    if len(lines) <= 3:
        return []
    return lines


def bayesian_evidence_update(oanda_trades: list[dict]) -> list[str]:
    if not oanda_trades or not LESSON_REGISTRY_JSON.exists():
        return []
    try:
        registry = json.loads(LESSON_REGISTRY_JSON.read_text())
    except Exception:
        return []

    lessons = registry.get("lessons") or []
    if not lessons:
        return []

    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for trade in oanda_trades:
        if trade.get("bucket") != "recurring_trader":
            continue
        key = (trade["pair"], trade["direction"])
        bucket = buckets.setdefault(key, {"count": 0, "wins": 0, "pl": 0.0})
        bucket["count"] += 1
        bucket["pl"] += trade["pl"]
        if trade["pl"] > 0:
            bucket["wins"] += 1

    if not buckets:
        return []

    lines = ["## Bayesian Evidence Update", ""]
    added = 0
    for (pair, direction), bucket in sorted(
        buckets.items(),
        key=lambda item: (abs(item[1]["pl"]), item[1]["count"]),
        reverse=True,
    ):
        exact = sorted(
            [
                lesson for lesson in lessons
                if lesson.get("pair") == pair
                and lesson.get("direction") == direction
                and lesson.get("state") != "deprecated"
            ],
            key=lambda lesson: (
                int(lesson.get("trust_score", 0)),
                int(lesson.get("state_rank", 0)),
                str(lesson.get("lesson_date", "")),
            ),
            reverse=True,
        )
        pair_only = sorted(
            [
                lesson for lesson in lessons
                if lesson.get("pair") == pair
                and not lesson.get("direction")
                and lesson.get("state") != "deprecated"
            ],
            key=lambda lesson: (
                int(lesson.get("trust_score", 0)),
                int(lesson.get("state_rank", 0)),
                str(lesson.get("lesson_date", "")),
            ),
            reverse=True,
        )
        prior = exact[0] if exact else (pair_only[0] if pair_only else None)
        if not prior:
            continue

        count = int(bucket["count"])
        wins = int(bucket["wins"])
        pl = float(bucket["pl"])
        if pl > 0 and wins == count:
            evidence = "supports prior strongly"
        elif pl > 0:
            evidence = "supports prior"
        elif pl < 0 and wins == 0:
            evidence = "contradicts prior strongly"
        elif pl < 0:
            evidence = "contradicts prior"
        else:
            evidence = "mixed / inconclusive"

        lines.append(
            f"  {pair} {direction}: prior {prior.get('state')} trust {prior.get('trust_score')} "
            f"from `{prior.get('title')}` | today {wins}/{count} wins {pl:+,.0f} JPY → {evidence}"
        )
        added += 1
        if added >= 8:
            break

    if added == 0:
        return []
    lines.append("")
    return lines


def after_action_review_queue(oanda_trades: list[dict]) -> list[str]:
    if not oanda_trades:
        return []

    lines = ["## After Action Review Queue", ""]
    added = 0

    losses = [trade for trade in oanda_trades if trade["pl"] < 0]
    if losses:
        worst = min(losses, key=lambda trade: trade["pl"])
        hold = f"{worst['hold_minutes']}min" if worst.get("hold_minutes") is not None else "hold n/a"
        lines.append(f"### Biggest loss: {worst['pair']} {worst['direction']}")
        lines.append(f"  Facts: {worst['pl']:+,.0f} JPY | {hold} | {worst.get('units', '?')}u")
        lines.append("  Fill in AAR: Planned / Actual / Gap / Next hypothesis")
        lines.append("")
        added += 1

    wins = [trade for trade in oanda_trades if trade["pl"] > 0]
    if wins:
        best = max(wins, key=lambda trade: trade["pl"])
        hold = f"{best['hold_minutes']}min" if best.get("hold_minutes") is not None else "hold n/a"
        lines.append(f"### Best win: {best['pair']} {best['direction']}")
        lines.append(f"  Facts: {best['pl']:+,.0f} JPY | {hold} | {best.get('units', '?')}u")
        lines.append("  Fill in AAR: What exactly made this work, and should it size up next time?")
        lines.append("")
        added += 1

    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for trade in oanda_trades:
        key = (trade["pair"], trade["direction"])
        bucket = buckets.setdefault(key, {"count": 0, "pl": 0.0})
        bucket["count"] += 1
        bucket["pl"] += trade["pl"]
    repetitive = [
        (key, bucket) for key, bucket in buckets.items()
        if bucket["count"] >= 3
    ]
    repetitive.sort(key=lambda item: (item[1]["pl"], -item[1]["count"]))
    if repetitive:
        (pair, direction), bucket = repetitive[0]
        lines.append(f"### Repetition check: {pair} {direction}")
        lines.append(
            f"  Facts: {int(bucket['count'])} trades | total {float(bucket['pl']):+,.0f} JPY"
        )
        lines.append("  Fill in AAR: Was this repeated edge proof or idea recycling?")
        lines.append("")
        added += 1

    return lines if added else []


def fetch_closed_trades(session_date: str) -> list[dict]:
    """Fetch closed trades for the specified completed UTC trading day from the OANDA API."""
    try:
        cfg = _load_oanda_config()
    except Exception:
        return []

    token = cfg.get('oanda_token', '')
    acct = cfg.get('oanda_account_id', '')
    base = 'https://api-fxtrade.oanda.com'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    start_utc, end_utc = review_day_bounds_utc(session_date)
    since = start_utc.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
    to = end_utc.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
    log_tags = load_trade_tags_from_log()
    daily_evidence = load_daily_trade_id_evidence(session_date)

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
            client_ext = trade_opened.get("clientExtensions") or {}
            log_meta = log_tags.get(tid, {})
            entries[tid] = {
                'price': float(txn.get('price', 0)),
                'units': abs(units),
                'direction': 'LONG' if units > 0 else 'SHORT',
                'pair': instrument,
                'time': txn.get('time', ''),
                'tag': client_ext.get("tag") or log_meta.get("tag", ""),
                'comment': client_ext.get("comment") or log_meta.get("comment", ""),
            }

        for tc in txn.get('tradesClosed', []) + txn.get('tradesReduced', []):
            tid = tc.get('tradeID', '')
            pl = float(tc.get('realizedPL', 0))
            units_signed = int(float(tc.get('units', 0)))
            units_c = abs(units_signed)
            close_time = txn.get('time', '')
            if tid not in closes:
                closes[tid] = {
                    'price': float(txn.get('price', 0)),
                    'pl': pl, 'units': units_c,
                    'instrument': instrument,
                    'time': close_time,
                    'direction': 'LONG' if units_signed > 0 else 'SHORT' if units_signed < 0 else None,
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
        log_meta = log_tags.get(tid, {})
        pair = entry['pair'] if entry else close.get('instrument', 'UNKNOWN')
        direction = entry['direction'] if entry else (close.get('direction') or ('LONG' if close['pl'] >= 0 else 'SHORT'))
        tag = (entry or {}).get("tag") or log_meta.get("tag", "")
        comment = (entry or {}).get("comment") or log_meta.get("comment", "")
        bucket, ownership_source = classify_trade_ownership(tid, tag, comment, daily_evidence)

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
            'entry_time': entry['time'] if entry else None,
            'close_time': close['time'],
            'tag': tag,
            'comment': comment,
            'bucket': bucket,
            'ownership_source': ownership_source,
        })

    return trades


def _parse_oanda_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        base = value.replace('Z', '+00:00')
        if '.' in base:
            base = base.split('.')[0] + '+00:00'
        return datetime.fromisoformat(base).astimezone(JST)
    except Exception:
        return None


def _parse_local_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=JST)
    except Exception:
        return None


def _parse_state_updated(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).astimezone(JST)
        except Exception:
            continue
    return None


def _parse_audit_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    cleaned = value.strip()
    if cleaned.endswith("Z") and "T" in cleaned:
        try:
            return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).astimezone(JST)
        except Exception:
            pass

    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc).astimezone(JST)
        except Exception:
            continue
    return None


def _pip_factor(pair: str) -> int:
    return 100 if "JPY" in pair else 10000


def fetch_current_prices(pairs: set[str]) -> dict[str, float]:
    if not pairs:
        return {}

    try:
        cfg = _load_oanda_config()
        pairs_str = ",".join(sorted(pairs))
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{cfg['oanda_account_id']}/pricing?instruments={pairs_str}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json"
        })
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
    except Exception:
        return {}

    current_prices = {}
    for price in resp.get("prices", []):
        try:
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
        except Exception:
            continue
        current_prices[price["instrument"]] = (bid + ask) / 2
    return current_prices


def repair_pretrade_outcome_links(conn, trade_ids: list[str]) -> int:
    """Keep only the newest outcome linked to each trade_id.

    Older matching logic could attach the same closed trade to multiple pretrade rows.
    Those older probes should return to the unmatched pool so future reviews do not
    learn from duplicate labels.
    """
    if not trade_ids:
        return 0

    placeholders = ",".join("?" for _ in trade_ids)
    rows = fetchall_dict(
        conn,
        f"""SELECT id, trade_id, created_at
            FROM pretrade_outcomes
            WHERE trade_id IN ({placeholders})
            ORDER BY trade_id, created_at DESC, id DESC""",
        tuple(trade_ids),
    )

    seen = set()
    repaired = 0
    for row in rows:
        trade_id = row.get('trade_id')
        if not trade_id:
            continue
        if trade_id not in seen:
            seen.add(trade_id)
            continue
        conn.execute(
            "UPDATE pretrade_outcomes SET pl = NULL, trade_id = NULL WHERE id = ?",
            (row['id'],),
        )
        repaired += 1

    return repaired


def collapse_duplicate_pretrade_probes(conn) -> int:
    """Drop redundant unmatched probes with identical decision fields.

    Historical sessions logged the same pretrade check many times before a real
    entry decision existed. For evaluation, one exact unmatched probe per
    session/pair/direction/level/score/thesis is enough.
    """
    rows = fetchall_dict(
        conn,
        """SELECT session_date, pair, direction, pretrade_level, pretrade_score,
                  COALESCE(thesis, '') AS thesis_key, MAX(id) AS keep_id, COUNT(*) AS cnt
           FROM pretrade_outcomes
           WHERE pl IS NULL AND trade_id IS NULL
           GROUP BY session_date, pair, direction, pretrade_level, pretrade_score, thesis_key
           HAVING COUNT(*) > 1""",
    )
    deleted = 0
    for row in rows:
        duplicates = fetchall_dict(
            conn,
            """SELECT id
               FROM pretrade_outcomes
               WHERE session_date = ?
                 AND pair = ?
                 AND direction = ?
                 AND pretrade_level = ?
                 AND pretrade_score = ?
                 AND COALESCE(thesis, '') = ?
                 AND pl IS NULL
                 AND trade_id IS NULL
                 AND id != ?
               ORDER BY id""",
            (
                row["session_date"],
                row["pair"],
                row["direction"],
                row["pretrade_level"],
                row["pretrade_score"],
                row["thesis_key"],
                row["keep_id"],
            ),
        )
        for duplicate in duplicates:
            conn.execute("DELETE FROM pretrade_outcomes WHERE id = ?", (duplicate["id"],))
            deleted += 1
    return deleted


def _recent_pretrade_feedback_rows(
    conn,
    outcome: dict,
    *,
    exact_only: bool = False,
    family_only: bool = False,
    limit: int = 4,
) -> list[dict]:
    conditions = ["pair = ?", "direction = ?", "pl IS NOT NULL"]
    params: list[object] = [outcome["pair"], outcome["direction"]]

    thesis_key = outcome.get("thesis_key")
    thesis_family = outcome.get("thesis_family")
    thesis_trigger = outcome.get("thesis_trigger")
    thesis_vehicle = outcome.get("thesis_vehicle")
    if exact_only and thesis_key:
        conditions.append("COALESCE(thesis_key, '') = ?")
        params.append(thesis_key)
    elif family_only and thesis_family:
        conditions.append("COALESCE(thesis_family, '') = ?")
        params.append(thesis_family)
        if thesis_key:
            conditions.append("(thesis_key IS NULL OR thesis_key != ?)")
            params.append(thesis_key)
    else:
        scoped = []
        if thesis_trigger:
            scoped.append("COALESCE(thesis_trigger, '') = ?")
            params.append(thesis_trigger)
        if thesis_vehicle:
            scoped.append("COALESCE(thesis_vehicle, '') = ?")
            params.append(thesis_vehicle)
        if not scoped:
            conditions.append("pretrade_level = ?")
            params.append(outcome.get("pretrade_level"))
            if outcome.get("execution_style"):
                conditions.append("COALESCE(execution_style, '') = ?")
                params.append(outcome.get("execution_style"))
        else:
            conditions.append("(" + " OR ".join(scoped) + ")")

    params.append(limit)
    return fetchall_dict(
        conn,
        f"""SELECT id, session_date, pl, execution_style, lesson_from_review,
                  thesis_trigger, thesis_vehicle, collapse_layer
           FROM pretrade_outcomes
           WHERE {' AND '.join(conditions)}
           ORDER BY id DESC
           LIMIT ?""",
        tuple(params),
    )


def _loss_streak(rows: list[dict]) -> tuple[int, float]:
    streak = 0
    total = 0.0
    for row in rows:
        pl = float(row.get("pl") or 0.0)
        if pl >= 0:
            break
        streak += 1
        total += pl
    return streak, total


def _clip_feedback(text: str, limit: int = 280) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _live_tape_bucket_label(bucket: str | None) -> str:
    text = str(bucket or "unknown").strip().replace("_", " ")
    return text or "unknown"


def _live_tape_summary(row: dict) -> str | None:
    bucket = str(row.get("live_tape_bucket") or "").strip()
    bias = str(row.get("live_tape_bias") or "").strip()
    state = str(row.get("live_tape_state") or "").strip()
    if not bucket and not bias and not state:
        return None
    parts = []
    if bucket:
        parts.append(_live_tape_bucket_label(bucket))
    detail = " / ".join(part for part in (bias, state) if part)
    if detail:
        if parts:
            return f"{parts[0]} ({detail})"
        return detail
    return parts[0] if parts else None


def _loss_attribution_bucket(row: dict) -> str:
    collapse = str(row.get("collapse_layer") or "").strip().lower()
    style = str(row.get("execution_style") or "").strip().upper()
    live_bucket = str(row.get("live_tape_bucket") or "").strip().lower()

    if collapse in {"vehicle", "trigger"}:
        return "execution_tape"
    if style == "MARKET" and live_bucket in HOSTILE_LIVE_TAPE_BUCKETS:
        return "execution_tape"
    if collapse in {"market", "structure"}:
        return "market_structure"
    if collapse == "aging" or style == "PASS":
        return "stale_process"
    return "unclassified"


def _pretrade_failure_attribution_lines(outcomes: list[dict]) -> list[str]:
    losses = [row for row in outcomes if float(row.get("pl") or 0.0) < 0]
    if not losses:
        return []

    labels = {
        "execution_tape": "Execution / Tape",
        "market_structure": "Market / Structure",
        "stale_process": "Stale / Process",
        "unclassified": "Unclassified",
    }
    buckets: dict[str, list[dict]] = {key: [] for key in labels}
    tape_buckets: dict[str, list[dict]] = {}

    for row in losses:
        buckets[_loss_attribution_bucket(row)].append(row)
        tape_key = str(row.get("live_tape_bucket") or "unknown").strip() or "unknown"
        tape_buckets.setdefault(tape_key, []).append(row)

    hostile_market = [
        row for row in losses
        if str(row.get("execution_style") or "").upper() == "MARKET"
        and str(row.get("live_tape_bucket") or "").strip().lower() in HOSTILE_LIVE_TAPE_BUCKETS
    ]

    lines = ["## Failure Attribution (market-state vs execution/tape)", ""]
    for key in ("execution_tape", "market_structure", "stale_process", "unclassified"):
        rows = buckets[key]
        if not rows:
            continue
        total_pl = sum(float(row.get("pl") or 0.0) for row in rows)
        lines.append(f"  {labels[key]}: {len(rows)} losses | {total_pl:+,.0f} JPY")
    if hostile_market:
        hostile_pl = sum(float(row.get("pl") or 0.0) for row in hostile_market)
        lines.append(
            f"  MARKET into hostile tape: {len(hostile_market)} losses | {hostile_pl:+,.0f} JPY"
        )
    lines.append("")
    lines.append("### Losses By Entry Tape")
    for tape_key, rows in sorted(
        tape_buckets.items(),
        key=lambda item: (sum(float(row.get("pl") or 0.0) for row in item[1]), -len(item[1]), item[0]),
    ):
        total_pl = sum(float(row.get("pl") or 0.0) for row in rows)
        collapse_counts: dict[str, int] = {}
        for row in rows:
            collapse = str(row.get("collapse_layer") or "none").strip().lower() or "none"
            collapse_counts[collapse] = collapse_counts.get(collapse, 0) + 1
        dominant = ", ".join(
            f"{layer} {count}"
            for layer, count in sorted(collapse_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        )
        lines.append(
            f"  {_live_tape_bucket_label(tape_key)}: {len(rows)} losses | {total_pl:+,.0f} JPY | collapse {dominant}"
        )
    lines.append("")
    return lines


def _collapse_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"market": 0, "structure": 0, "trigger": 0, "vehicle": 0, "aging": 0}
    for row in rows:
        layer = str(row.get("collapse_layer") or "").strip().lower()
        if layer in counts:
            counts[layer] += 1
    return counts


def _load_regret_rows(session_date: str) -> dict[str, dict]:
    try:
        from post_close_regret import build_regret_result_map
    except Exception:
        return {}
    try:
        return build_regret_result_map(session_date_from=session_date, hours=6)
    except Exception:
        return {}


def _review_collapse_layer(
    outcome: dict,
    exact_rows: list[dict],
    family_rows: list[dict],
    regret_row: dict | None,
) -> tuple[str | None, str | None]:
    pl = float(outcome.get("pl") or 0.0)
    if pl >= 0:
        return None, None

    style = str(outcome.get("execution_style") or "").upper()
    exact_streak, _ = _loss_streak(exact_rows)
    family_streak, family_total = _loss_streak(family_rows)
    family_recent_wins = sum(1 for row in family_rows[:3] if float(row.get("pl") or 0.0) > 0)
    recovered = bool(regret_row and regret_row.get("recovered"))
    regret_component = str((regret_row or {}).get("collapse_component") or "")
    family_counts = _collapse_counts(family_rows)
    live_tape_bucket = str(outcome.get("live_tape_bucket") or "").strip().lower()
    live_tape_context = _live_tape_summary(outcome)

    if style == "PASS":
        return "aging", "This overrode a blocked/pass seat. The thesis age was the problem, not a fresh market read."
    if regret_component == "stale_thesis":
        return "aging", "The loss came from a stale recycled idea. Do not rewrite the market story from a stale seat."
    if exact_streak >= 2 or (family_streak >= 2 and family_recent_wins == 0 and family_counts["aging"] >= 1):
        return "aging", f"The family is recycling losses ({family_total:+,.0f} JPY). The idea aged out before the tape improved."
    if style == "MARKET" and live_tape_bucket in HOSTILE_LIVE_TAPE_BUCKETS:
        detail = f" ({live_tape_context})" if live_tape_context else ""
        return "vehicle", f"The entry paid market friction into hostile tape{detail}. Fix the vehicle before rewriting the direction."
    if recovered:
        if style == "MARKET" or regret_component == "vehicle_friction":
            return "vehicle", "The market/structure later recovered. The paid market vehicle was wrong; wait for a better vehicle, not a new direction."
        return "trigger", "The market/structure later recovered. The trigger or first wobble was misread as full thesis death."
    if regret_component == "structure_break":
        return "structure", "Price did not recover and the defended level failed. The structure layer, not just the trigger, was dead."
    if regret_component in {"trigger_timing_fail", "manual_discretionary_cut"}:
        return "trigger", "The exit treated a trigger wobble as thesis collapse. Keep the family alive but require a new print."
    if regret_component == "vehicle_friction":
        return "vehicle", "The execution vehicle paid friction without enough edge. Keep the story only if the next vehicle changes."
    if live_tape_bucket == "opposed":
        detail = f" ({live_tape_context})" if live_tape_context else ""
        return "trigger", f"The tape was paying the other side at entry{detail}. The trigger was early even if the broader story looked attractive."
    if family_streak >= 2 and family_recent_wins == 0 and (family_counts["market"] + family_counts["structure"] >= 2):
        return "market", "The same market-state family is failing at the directional layer. Require a new market-state read, not a better entry."
    if style in {"LIMIT", "STOP-ENTRY"}:
        return "structure", "A better price / trigger did not save the trade and it did not recover. Treat the defended structure as broken."
    return "market", "The trade did not recover and no better layer explanation survived. Treat the directional market-state read as wrong."


def _generate_pretrade_review_note(
    outcome: dict,
    exact_rows: list[dict],
    family_rows: list[dict],
    regret_row: dict | None,
) -> tuple[str | None, str | None, str | None]:
    pl = float(outcome.get("pl") or 0.0)
    style = str(outcome.get("execution_style") or "").upper()
    level = str(outcome.get("pretrade_level") or "")
    exact_streak, exact_total = _loss_streak(exact_rows)
    family_streak, family_total = _loss_streak(family_rows)
    family_recent_wins = sum(1 for row in family_rows[:3] if float(row.get("pl") or 0.0) > 0)
    collapse_layer, collapse_note = _review_collapse_layer(outcome, exact_rows, family_rows, regret_row)

    notes: list[str] = []
    if pl < 0:
        if style == "PASS":
            notes.append("Trade overrode a pretrade PASS. Keep this thesis blocked until the written contradiction is gone.")
        elif style == "MARKET":
            notes.append("Market execution paid friction before proof. Re-enter only after a materially new trigger or a better vehicle exists.")
        elif style == "STOP-ENTRY":
            notes.append("Trigger proof was still not enough. Do not re-arm the same trigger without a materially new print.")
        elif style == "LIMIT":
            notes.append("Price improvement alone did not rescue the thesis. Wait for a new state change, not the same limit again.")
        else:
            notes.append("This thesis/vehicle failed. Require a materially new state change before the next attempt.")

        if exact_streak >= 2:
            notes.append(f"Same thesis key is now {exact_streak} straight losses ({exact_total:+,.0f} JPY).")
        elif exact_rows:
            notes.append("The most recent exact thesis is negative, so the next identical attempt should stay blocked.")
        elif family_streak >= 2 and family_recent_wins == 0:
            notes.append(f"The wider thesis family is still negative ({family_total:+,.0f} JPY). Keep it in watch/B lane.")
        if collapse_layer and collapse_note:
            survivor = {
                "market": "If you retry, rewrite the market-state first.",
                "structure": "Do not keep the same defended level alive just because the direction is attractive.",
                "trigger": "Keep the market/structure story, but change the trigger print.",
                "vehicle": "Keep the story only if the vehicle changes first.",
                "aging": "Treat this as a stale idea, not a fresh read.",
            }.get(collapse_layer, "")
            notes.append(f"Collapse layer: {collapse_layer}. {collapse_note}")
            if survivor:
                notes.append(survivor)
    else:
        if style in {"LIMIT", "STOP-ENTRY"}:
            notes.append("Proof-first execution matched the tape. Keep this thesis in trigger/price-improvement form until it repeats cleanly.")
        elif style == "MARKET":
            notes.append("Immediate execution worked today, but it upgrades only after repeated clean evidence.")
        if level == "LOW":
            notes.append("LOW was conservative here. Review whether this regime deserves promotion instead of treating the win as a fluke.")

    if not notes:
        return None, collapse_layer, collapse_note
    return _clip_feedback(" ".join(notes)), collapse_layer, collapse_note


def backfill_pretrade_review_feedback(conn, session_date: str, oanda_trades: list[dict]) -> int:
    regret_rows = _load_regret_rows(session_date)
    trade_ids = [trade["trade_id"] for trade in oanda_trades if trade.get("trade_id")]
    if trade_ids:
        placeholders = ",".join("?" for _ in trade_ids)
        rows = fetchall_dict(
            conn,
            f"""SELECT id, session_date, pair, direction, pretrade_level, pl,
                      execution_style, thesis_key, thesis_family, thesis_trigger, thesis_vehicle,
                      lesson_from_review, collapse_layer, collapse_note, trade_id,
                      live_tape_bias, live_tape_state, live_tape_bucket
               FROM pretrade_outcomes
               WHERE trade_id IN ({placeholders})
                 AND pl IS NOT NULL
                 AND (
                    lesson_from_review IS NULL OR lesson_from_review = ''
                    OR collapse_layer IS NULL OR collapse_layer = ''
                    OR collapse_note IS NULL OR collapse_note = ''
                 )
               ORDER BY id DESC""",
            tuple(trade_ids),
        )
    else:
        rows = fetchall_dict(
            conn,
            """SELECT id, session_date, pair, direction, pretrade_level, pl,
                      execution_style, thesis_key, thesis_family, thesis_trigger, thesis_vehicle,
                      lesson_from_review, collapse_layer, collapse_note, trade_id,
                      live_tape_bias, live_tape_state, live_tape_bucket
               FROM pretrade_outcomes
               WHERE session_date = ?
                 AND pl IS NOT NULL
                 AND (
                    lesson_from_review IS NULL OR lesson_from_review = ''
                    OR collapse_layer IS NULL OR collapse_layer = ''
                    OR collapse_note IS NULL OR collapse_note = ''
                 )
               ORDER BY id DESC""",
            (session_date,),
        )

    updated = 0
    for row in rows:
        exact_rows = _recent_pretrade_feedback_rows(conn, row, exact_only=True)
        family_rows = _recent_pretrade_feedback_rows(conn, row, family_only=True)
        regret_row = regret_rows.get(str(row.get("trade_id") or ""))
        note, collapse_layer, collapse_note = _generate_pretrade_review_note(row, exact_rows, family_rows, regret_row)
        if not note and not collapse_layer and not collapse_note:
            continue
        conn.execute(
            """UPDATE pretrade_outcomes
               SET lesson_from_review = COALESCE(?, lesson_from_review),
                   collapse_layer = COALESCE(?, collapse_layer),
                   collapse_note = COALESCE(?, collapse_note)
               WHERE id = ?""",
            (note, collapse_layer, collapse_note, row["id"]),
        )
        updated += 1
    return updated


def match_pretrade_outcomes(conn, session_date: str, oanda_trades: list[dict]):
    """Fill pl in the pretrade_outcomes table (linking predictions to results)"""
    repair_pretrade_outcome_links(
        conn,
        [trade['trade_id'] for trade in oanda_trades if trade.get('trade_id')],
    )

    cutoff = (datetime.strptime(session_date, '%Y-%m-%d') - timedelta(days=PRETRADE_LOOKBACK_DAYS)).strftime('%Y-%m-%d')

    # All recent unmatched outcomes, not just today's. Trades often close the next day.
    outcomes = fetchall_dict(conn,
        """SELECT id, pair, direction, pretrade_level, created_at
           FROM pretrade_outcomes
           WHERE session_date >= ? AND pl IS NULL
           ORDER BY created_at, id""",
        (cutoff,))

    if not outcomes:
        return 0

    used_trade_ids = {
        row['trade_id']
        for row in fetchall_dict(
            conn,
            "SELECT trade_id FROM pretrade_outcomes WHERE trade_id IS NOT NULL"
        )
        if row.get('trade_id')
    }

    candidate_trades = []
    for trade in oanda_trades:
        trade_id = trade.get('trade_id')
        if not trade_id or trade_id in used_trade_ids:
            continue
        trade_time = _parse_oanda_time(trade.get('entry_time')) or _parse_oanda_time(trade.get('close_time'))
        if trade_time is None:
            continue
        candidate_trades.append((trade_time, trade))

    candidate_trades.sort(key=lambda item: item[0])

    matched = 0
    used_outcome_ids = set()
    for trade_time, trade in candidate_trades:
        matching = []
        for outcome in outcomes:
            if outcome['id'] in used_outcome_ids:
                continue
            if outcome['pair'] != trade['pair'] or outcome['direction'] != trade['direction']:
                continue
            created_at = _parse_local_created_at(outcome.get('created_at'))
            if created_at is None:
                continue
            delta = trade_time - created_at
            if delta.total_seconds() < -300:
                continue
            if delta > timedelta(hours=PRETRADE_MATCH_HOURS):
                continue
            matching.append((created_at, outcome))

        if not matching:
            continue

        # Nearest pretrade check before the entry/fill wins; older probes stay unmatched.
        _, outcome = max(matching, key=lambda item: item[0])
        conn.execute(
            "UPDATE pretrade_outcomes SET pl = ?, trade_id = ? WHERE id = ?",
            (trade['pl'], trade['trade_id'], outcome['id'])
        )
        used_trade_ids.add(trade['trade_id'])
        used_outcome_ids.add(outcome['id'])
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
    """Analyze scanner and narrative audit opportunities from audit_history.jsonl."""
    try:
        from seat_outcomes import fetch_window_mid_stats as fetch_seat_window_mid_stats
        from seat_outcomes import _score_window_move as score_window_move
    except Exception:
        fetch_seat_window_mid_stats = None
        score_window_move = None

    history_path = ROOT / "logs" / "audit_history.jsonl"
    if not history_path.exists():
        return []

    scanner_detections = []
    narrative_detections = []
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
                    scanner_detections.append({
                        "pair": sc.get("pair", "?"),
                        "direction": sc.get("direction", "?"),
                        "label": sc.get("recipe", "?"),
                        "price": sc.get("price_at_detection", 0),
                        "timestamp": ts,
                    })
            for pick in entry.get("narrative_picks", []):
                if not isinstance(pick, dict):
                    continue
                direction = pick.get("direction", "?")
                edge = pick.get("edge") or pick.get("conviction")
                allocation = pick.get("allocation") or edge
                held_status = pick.get("held_status")
                if direction not in {"LONG", "SHORT"} or edge not in {"S", "A"}:
                    continue
                if held_status not in (None, "", "NOT_HELD"):
                    continue
                narrative_detections.append({
                    "pair": pick.get("pair", "?"),
                    "direction": direction,
                    "label": f"Edge {edge} / Allocation {allocation}",
                    "price": pick.get("entry_price", 0) or 0,
                    "timestamp": ts,
                })

            strongest = entry.get("strongest_unheld")
            if isinstance(strongest, dict):
                direction = strongest.get("direction", "?")
                edge = strongest.get("edge") or strongest.get("conviction")
                allocation = strongest.get("allocation") or edge
                if direction in {"LONG", "SHORT"} and edge in {"S", "A"}:
                    pair = strongest.get("pair", "?")
                    if not any(
                        detection["pair"] == pair and detection["direction"] == direction
                        and detection["timestamp"] == ts
                        for detection in narrative_detections
                    ):
                        narrative_detections.append({
                            "pair": pair,
                            "direction": direction,
                            "label": f"Strongest unheld {edge}/{allocation}",
                            "price": strongest.get("entry_price", 0) or 0,
                            "timestamp": ts,
                        })
    except Exception:
        return []

    if not scanner_detections and not narrative_detections:
        return []

    def dedupe(detections: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for detection in detections:
            key = (
                detection["timestamp"],
                detection["pair"],
                detection["direction"],
                detection["label"],
                detection.get("price"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(detection)
        return unique

    scanner_unique = dedupe(scanner_detections)
    narrative_unique = dedupe(narrative_detections)
    _review_start_utc, review_end_utc = review_day_bounds_utc(session_date)
    review_end_jst = review_end_utc.astimezone(JST)

    trade_lookup: dict[tuple[str, str], list[tuple[datetime, dict]]] = {}
    for t in oanda_trades:
        trade_ts = _parse_oanda_time(t.get("entry_time")) or _parse_oanda_time(t.get("close_time"))
        if trade_ts is None:
            continue
        key = (t["pair"], t["direction"])
        trade_lookup.setdefault(key, []).append((trade_ts, t))
    for bucket in trade_lookup.values():
        bucket.sort(key=lambda item: item[0])

    current_prices = fetch_current_prices({
        d["pair"] for d in scanner_unique + narrative_unique
        if d["pair"] and d["pair"] != "?"
    })
    excursion_cache: dict[tuple[str, str, str], dict] = {}

    def attach_trade_windows(detections: list[dict]) -> list[dict]:
        enriched = []
        by_key: dict[tuple[str, str], list[dict]] = {}
        for detection in detections:
            payload = dict(detection)
            payload["parsed_timestamp"] = _parse_audit_timestamp(detection.get("timestamp"))
            payload["matched_trades"] = []
            enriched.append(payload)
            by_key.setdefault((payload["pair"], payload["direction"]), []).append(payload)

        for key, items in by_key.items():
            items.sort(key=lambda item: item.get("parsed_timestamp") or datetime.min.replace(tzinfo=JST))
            trades = trade_lookup.get(key, [])
            for idx, item in enumerate(items):
                start = item.get("parsed_timestamp")
                if start is None:
                    continue
                next_ts = None
                for later in items[idx + 1:]:
                    later_ts = later.get("parsed_timestamp")
                    if later_ts is not None:
                        next_ts = later_ts
                        break
                window_end = next_ts or (start + timedelta(hours=MAX_AUDIT_SIGNAL_GAP_HOURS))
                if window_end > review_end_jst:
                    window_end = review_end_jst
                matched = []
                for trade_ts, trade in trades:
                    if trade_ts < start:
                        continue
                    if trade_ts >= window_end:
                        continue
                    matched.append(trade)
                item["matched_trades"] = matched
                item["window_end"] = window_end
        return enriched

    def analyze_group(title: str, detections: list[dict]) -> list[str]:
        if not detections:
            return []

        lines = [f"### {title}", ""]
        stats = {}

        for detection in attach_trade_windows(detections):
            pair = detection["pair"]
            direction = detection["direction"]
            label = detection["label"]
            det_price = detection["price"]
            matched_trades = detection.get("matched_trades") or []

            if label not in stats:
                stats[label] = {"correct": 0, "premature": 0, "wrong": 0, "entered": 0, "missed": 0}

            cutoff_price = current_prices.get(pair)
            window_stats = None
            start = detection.get("parsed_timestamp")
            window_end = detection.get("window_end")
            if fetch_seat_window_mid_stats is not None and start is not None and window_end is not None:
                cache_key = (pair, start.isoformat(), window_end.isoformat())
                if cache_key not in excursion_cache:
                    excursion_cache[cache_key] = fetch_seat_window_mid_stats(
                        pair,
                        start.astimezone(timezone.utc),
                        window_end.astimezone(timezone.utc),
                    )
                window_stats = excursion_cache.get(cache_key) or {}
                if window_stats.get("last_close") is not None:
                    try:
                        cutoff_price = float(window_stats["last_close"])
                    except Exception:
                        pass

            moved_right = None
            pip_move = None
            if score_window_move is not None:
                scored = score_window_move(pair, direction, det_price, window_stats, cutoff_price)
                moved_right = scored.get("moved_right")
                pip_move = scored.get("pip_move")
            elif cutoff_price and det_price > 0:
                moved_right = (direction == "LONG" and cutoff_price > det_price) or (
                    direction == "SHORT" and cutoff_price < det_price
                )
                pip_move = abs(cutoff_price - det_price) * _pip_factor(pair)

            if matched_trades:
                stats[label]["entered"] += 1
                pl = sum(float(trade.get("pl") or 0.0) for trade in matched_trades)
                if pl > 0:
                    verdict = "CORRECT"
                    stats[label]["correct"] += 1
                elif moved_right:
                    verdict = "PREMATURE"
                    stats[label]["premature"] += 1
                else:
                    verdict = "WRONG"
                    stats[label]["wrong"] += 1
                price_str = f" @{det_price}" if det_price else ""
                if pip_move is not None and moved_right:
                    lines.append(
                        f"  {pair} {direction} {label}{price_str} → ENTERED → {verdict} "
                        f"({len(matched_trades)} trade{'s' if len(matched_trades) != 1 else ''}, {pl:+,.0f} JPY, best favorable excursion {pip_move:.1f}pip)"
                    )
                else:
                    lines.append(
                        f"  {pair} {direction} {label}{price_str} → ENTERED → {verdict} "
                        f"({len(matched_trades)} trade{'s' if len(matched_trades) != 1 else ''}, {pl:+,.0f} JPY)"
                    )
                continue

            stats[label]["missed"] += 1
            if pip_move is not None and moved_right is not None:
                verdict = "CORRECT" if moved_right else "WRONG"
                lines.append(
                    f"  {pair} {direction} {label} @{det_price} → MISSED → "
                    f"{verdict} (best favorable excursion {pip_move:.1f}pip)"
                )
                if moved_right:
                    stats[label]["correct"] += 1
                else:
                    stats[label]["wrong"] += 1
            else:
                price_str = f" @{det_price}" if det_price else ""
                lines.append(f"  {pair} {direction} {label}{price_str} → MISSED → no price data")

        lines.append("")
        lines.append("#### Accuracy Summary")
        lines.append("")
        for label, bucket in sorted(stats.items()):
            total = bucket["correct"] + bucket["premature"] + bucket["wrong"]
            directional = bucket["correct"] + bucket["premature"]
            acc = directional / total * 100 if total > 0 else 0
            lines.append(
                f"  {label}: {acc:.0f}% directional ({directional}/{total}) "
                f"| correct: {bucket['correct']} premature: {bucket['premature']} wrong: {bucket['wrong']} "
                f"| entered: {bucket['entered']} missed: {bucket['missed']}"
            )
        lines.append("")
        return lines

    lines = ["## Audit Opportunity Outcome Analysis", ""]
    lines.extend(analyze_group("S-Scan Outcomes", scanner_unique))
    lines.extend(analyze_group("Narrative A/S Outcomes", narrative_unique))
    return lines if len(lines) > 2 else []


def analyze_s_hunt_ledger(session_date: str, oanda_trades: list[dict]) -> list[str]:
    """Analyze horizon-by-horizon S Hunt capture quality from the formal seat_outcomes table."""
    try:
        from seat_outcomes import review_lines as seat_outcome_review_lines
        from seat_outcomes import sync_seat_outcomes
    except Exception:
        return []

    try:
        sync_seat_outcomes(session_date, live=False)
    except Exception:
        return []

    try:
        return seat_outcome_review_lines(session_date)
    except Exception:
        return []


def analyze_s_excavation(session_date: str, oanda_trades: list[dict]) -> list[str]:
    """Review near-S podium seats from the formal seat_outcomes table."""
    try:
        from seat_outcomes import excavation_review_lines
    except Exception:
        return []

    try:
        return excavation_review_lines(session_date)
    except Exception:
        return []


def generate_report(session_date: str) -> str:
    """Generate a structured report for the daily review"""
    conn = get_conn()
    lines = []

    lines.append(f"# Daily Review Report: {session_date} (completed UTC trading day)")
    lines.append("")
    lines.append(f"Window: {review_window_label(session_date)}")
    lines.append("")

    deduped = collapse_duplicate_pretrade_probes(conn)
    if deduped:
        lines.append(f"## Pretrade Probe Cleanup: removed {deduped} duplicate unmatched rows")
        lines.append("")

    # 1. OANDA closed trades
    oanda_trades = fetch_closed_trades(session_date)
    lines.append(f"## Closed Trades ({len(oanda_trades)} trades)")
    lines.append("")

    if not oanda_trades:
        lines.append("No closed trades")
    else:
        total_pl = sum(t['pl'] for t in oanda_trades)
        metrics = payoff_metrics([t['pl'] for t in oanda_trades])
        wins = [t for t in oanda_trades if t['pl'] > 0]
        losses = [t for t in oanda_trades if t['pl'] <= 0]
        be_wr = metrics.get("break_even_win_rate")
        be_text = f" | Break-even WR: {be_wr:.0%}" if be_wr is not None else ""
        lines.append(
            f"Total P&L: {total_pl:+,.0f} JPY | Win: {metrics['wins']} Loss: {metrics['losses']} "
            f"| Win rate: {metrics['win_rate']*100:.0f}% | Expectancy: {metrics['expectancy']:+,.0f} JPY/trade{be_text}"
        )
        lines.append(
            f"Avg win: {metrics['avg_win']:+,.0f} JPY | Avg loss: {metrics['avg_loss']:+,.0f} JPY "
            f"| R:R: {metrics['rr_ratio']:.2f}"
        )
        lines.append("")

        # Per-pair summary
        from collections import defaultdict
        by_pair = defaultdict(list)
        for t in oanda_trades:
            by_pair[t['pair']].append(t)

        for pair, trades in sorted(by_pair.items()):
            pair_pl = sum(t['pl'] for t in trades)
            metrics = payoff_metrics([t['pl'] for t in trades])
            lines.append(
                f"### {pair}: {pair_pl:+,.0f} JPY ({metrics['wins']}/{len(trades)} wins)"
                f" | EV {metrics['expectancy']:+,.0f} | R:R {metrics['rr_ratio']:.2f}"
            )
            for t in trades:
                hold = f" ({t['hold_minutes']}min)" if t['hold_minutes'] is not None else ""
                lines.append(f"  {t['direction']} {t['units']}u → {t['pl']:+,.0f} JPY{hold}")
            lines.append("")

        recurring_trades = [t for t in oanda_trades if t.get("bucket") == "recurring_trader"]
        other_trades = [t for t in oanda_trades if t.get("bucket") != "recurring_trader"]
        tagged_count = sum(1 for t in oanda_trades if t.get("tag") or t.get("comment"))
        recurring_tag_count = sum(1 for t in recurring_trades if t.get("ownership_source") == "explicit_tag_or_comment")
        recurring_daily_count = sum(1 for t in recurring_trades if t.get("ownership_source") == "daily_trade_record")
        unresolved_count = sum(1 for t in oanda_trades if t.get("ownership_source") == "unresolved")
        lines.append("## Execution Split")
        lines.append("")
        lines.append(
            "Read this before writing lessons. Recurring ownership uses explicit tags/comments first, then the daily trade journal as a fallback. "
            "If the damage came from manual / legacy / unresolved execution, "
            "do not teach that lesson to the recurring trader."
        )
        lines.append("")
        lines.append(
            f"Ownership evidence: explicit recurring={recurring_tag_count} | daily journal fallback={recurring_daily_count} "
            f"| explicit tagged non-recurring={tagged_count - recurring_tag_count} | unresolved={unresolved_count}"
        )
        lines.append("")
        lines.extend(summarize_trade_bucket(
            "Recurring trader (`tag=trader*` / `qr-trader` / daily trade record fallback)",
            recurring_trades,
            tag_coverage=(tagged_count, len(oanda_trades)),
        ))
        lines.extend(summarize_trade_bucket(
            "Other execution / unknown tag",
            other_trades,
        ))
        lines.extend(memory_promotion_gate(recurring_trades, other_trades))
        lines.extend(lesson_registry_state_suggestions())
        lines.extend(bayesian_evidence_update(oanda_trades))
        lines.extend(after_action_review_queue(oanda_trades))

    # 2. pretrade_outcomes matching
    matched = match_pretrade_outcomes(conn, session_date, oanda_trades)
    feedback_notes = backfill_pretrade_review_feedback(conn, session_date, oanda_trades)
    if oanda_trades:
        lines.append(f"## Pretrade Outcomes: {matched} newly matched")
        lines.append("")
    if feedback_notes:
        lines.append(f"## Pretrade Feedback Notes: wrote {feedback_notes} review notes back to pretrade_outcomes")
        lines.append("")

    # 3. pretrade results vs actual P&L (include carry-over trades that closed today)
    trade_ids = [t['trade_id'] for t in oanda_trades if t.get('trade_id')]
    if trade_ids:
        placeholders = ",".join("?" for _ in trade_ids)
        outcomes = fetchall_dict(
            conn,
            f"""SELECT session_date, pair, direction, pretrade_level, pretrade_score, pl, pretrade_warnings
                , execution_style, live_tape_bias, live_tape_state, live_tape_bucket
                , lesson_from_review, collapse_layer, collapse_note
                FROM pretrade_outcomes
                WHERE trade_id IN ({placeholders})
                ORDER BY pl""",
            tuple(trade_ids),
        )
    else:
        outcomes = fetchall_dict(conn,
            """SELECT session_date, pair, direction, pretrade_level, pretrade_score, pl, pretrade_warnings
               , execution_style, live_tape_bias, live_tape_state, live_tape_bucket
               , lesson_from_review, collapse_layer, collapse_note
               FROM pretrade_outcomes
               WHERE session_date = ? AND pl IS NOT NULL
               ORDER BY pl""",
            (session_date,))

    if outcomes:
        lines.append("## Pretrade Prediction vs Result")
        lines.append("")
        for o in outcomes:
            result = "WIN" if o['pl'] > 0 else "LOSS"
            carry = f" [checked {o['session_date']}]" if o.get('session_date') and o['session_date'] != session_date else ""
            style = str(o.get("execution_style") or "").upper()
            tape_summary = _live_tape_summary(o)
            style_bits = []
            if style:
                style_bits.append(f"style={style}")
            if tape_summary:
                style_bits.append(f"tape={tape_summary}")
            style_text = f" | {' | '.join(style_bits)}" if style_bits else ""
            lines.append(
                f"  {o['pair']} {o['direction']} pretrade={o['pretrade_level']}(score={o['pretrade_score']})"
                f"{carry} → {result} {o['pl']:+,.0f} JPY{style_text}"
            )
            if o.get("lesson_from_review"):
                lines.append(f"    → {o['lesson_from_review']}")
            if o.get("collapse_layer"):
                lines.append(f"    collapse={o['collapse_layer']}: {o.get('collapse_note') or 'no note'}")
        lines.append("")
        lines.extend(_pretrade_failure_attribution_lines(outcomes))

        # Analysis of ignored LOW entries
        low_entries = [o for o in outcomes if o['pretrade_level'] == 'LOW']
        if low_entries:
            low_wins = sum(1 for o in low_entries if o['pl'] > 0)
            low_total_pl = sum(o['pl'] for o in low_entries)
            low_metrics = payoff_metrics([o['pl'] for o in low_entries])
            lines.append(f"### LOW-ignored entries: {len(low_entries)} trades")
            lines.append(
                f"  Win rate: {low_wins}/{len(low_entries)} | Total: {low_total_pl:+,.0f} JPY "
                f"| EV {low_metrics['expectancy']:+,.0f} | R:R {low_metrics['rr_ratio']:.2f}"
            )
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

    # 6. Audit opportunity tracking (audit_history.jsonl)
    s_scan_lines = analyze_s_scan_outcomes(session_date, oanda_trades)
    if s_scan_lines:
        lines.extend(s_scan_lines)

    # 7. S Hunt capture tracking (s_hunt_ledger.jsonl)
    s_hunt_lines = analyze_s_hunt_ledger(session_date, oanda_trades)
    if s_hunt_lines:
        lines.extend(s_hunt_lines)

    # 7b. S Excavation podium review (near-S seats not promoted into S Hunt)
    excavation_lines = analyze_s_excavation(session_date, oanda_trades)
    if excavation_lines:
        lines.extend(excavation_lines)

    # 8. DB trades stats (comparison against all-time)
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
            lines.append(
                f"  {s['pair']} {s['direction']}: {s['wins']}/{s['cnt']} wins ({wr:.0f}%) "
                f"avg {s['avg_pl']:+,.0f} JPY/trade total {s['total_pl']:+,.0f} JPY"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    init_db()

    target_date = completed_utc_review_day()
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--date" and i < len(sys.argv) - 1:
            target_date = sys.argv[i + 1]

    report = generate_report(target_date)
    print(report)
