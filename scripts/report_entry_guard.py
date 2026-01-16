import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
ORDERS_DB = LOG_DIR / "orders.db"


def _parse_ts(raw: str):
    if not raw:
        return None
    try:
        text = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_json(raw: str):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _ratio(num: int, den: int) -> str:
    if den <= 0:
        return "-"
    return f"{(num / den) * 100:.1f}%"


def _percentile(values, pct: float):
    if not values:
        return None
    items = sorted(values)
    if len(items) == 1:
        return float(items[0])
    pct = max(0.0, min(100.0, pct))
    pos = (len(items) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(items) - 1)
    if lo == hi:
        return float(items[lo])
    weight = pos - lo
    return float(items[lo] * (1.0 - weight) + items[hi] * weight)


def _quantiles(values):
    if not values:
        return None, None, None
    return (
        _percentile(values, 25.0),
        _percentile(values, 50.0),
        _percentile(values, 75.0),
    )


def _fmt_num(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def _strategy_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    thesis = payload.get("entry_thesis")
    if isinstance(thesis, dict):
        for key in ("strategy_tag", "strategy", "tag"):
            value = thesis.get(key)
            if value:
                return str(value).strip()
    meta = payload.get("meta")
    if isinstance(meta, dict):
        for key in ("strategy_tag", "strategy", "tag"):
            value = meta.get(key)
            if value:
                return str(value).strip()
    return ""


def _strategy_from_client_id(client_order_id: str) -> str:
    if not client_order_id:
        return ""
    parts = str(client_order_id).split("-")
    if len(parts) < 4:
        return ""
    return "-".join(parts[3:]).strip()


def _base_strategy_tag(tag: str) -> str:
    if not tag:
        return "unknown"
    for suffix in ("-long", "-short", "-buy", "-sell"):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _resolve_strategy(payload: dict, client_order_id: str, group_base: bool) -> str:
    tag = _strategy_from_payload(payload)
    if not tag:
        tag = _strategy_from_client_id(client_order_id)
    if not tag:
        tag = "unknown"
    return _base_strategy_tag(tag) if group_base else tag


def _load_rows(days: int, limit: int, orders_db: Path):
    if not orders_db.exists():
        print("orders.db not found:", orders_db)
        return []
    con = sqlite3.connect(orders_db)
    con.row_factory = sqlite3.Row
    since = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    sql = """
        SELECT id, ts, pocket, side, units, client_order_id, status, attempt, request_json
        FROM orders
        WHERE status IN ('preflight_start', 'entry_guard_block', 'submit_attempt')
          AND ts >= ?
        ORDER BY id DESC
    """
    params = [since.isoformat()]
    if limit and limit > 0:
        sql = sql.strip() + " LIMIT ?"
        params.append(int(limit))
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows, since


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="max rows to scan (0 = no limit)")
    ap.add_argument("--top", type=int, default=8, help="top strategies by block count")
    ap.add_argument("--min-total", type=int, default=5, help="min total for block rate tables")
    ap.add_argument("--group-base", action="store_true", help="strip -long/-short/-buy/-sell")
    ap.add_argument("--orders-db", type=str, default=str(ORDERS_DB), help="path to orders.db")
    args = ap.parse_args()

    orders_db = Path(args.orders_db)
    rows, since = _load_rows(args.days, args.limit, orders_db)
    preflight_total = 0
    blocked_total = 0
    passed_total = 0
    preflight_by_pocket = Counter()
    passed_by_pocket = Counter()
    blocked_by_reason = Counter()
    blocked_by_pocket = Counter()
    blocked_by_strategy = Counter()
    blocked_by_reason_strategy = Counter()
    blocked_by_reason_pocket = Counter()
    blocked_by_reason_align = Counter()
    align_counts = Counter()
    passed_by_strategy = Counter()
    passed_by_pocket_strategy = Counter()
    blocked_by_pocket_strategy = Counter()
    seen_submit = set()
    seen_block = set()
    seen_preflight = set()
    metrics_by_reason = defaultdict(lambda: defaultdict(list))
    metrics_by_reason_strategy = defaultdict(lambda: defaultdict(list))

    def _push_metric(reason: str, key: str, value, strategy: str = ""):
        if value is None:
            return
        try:
            metrics_by_reason[reason][key].append(float(value))
            if strategy:
                metrics_by_reason_strategy[(reason, strategy)][key].append(float(value))
        except Exception:
            return

    for row in rows:
        ts = _parse_ts(row["ts"])
        if ts is None or ts < since:
            continue
        payload = _parse_json(row["request_json"])
        pocket = row["pocket"] or "unknown"
        status = row["status"] or ""
        client_order_id = row["client_order_id"] or ""
        if status == "preflight_start":
            reduce_only = False
            if isinstance(payload, dict):
                reduce_only = bool(payload.get("reduce_only") or False)
            if reduce_only:
                continue
            if client_order_id and client_order_id in seen_preflight:
                continue
            if client_order_id:
                seen_preflight.add(client_order_id)
            preflight_total += 1
            preflight_by_pocket[pocket] += 1
            continue

        if status == "submit_attempt":
            if not client_order_id or client_order_id in seen_submit:
                continue
            seen_submit.add(client_order_id)
            strategy = _resolve_strategy(payload, client_order_id, args.group_base)
            passed_total += 1
            passed_by_pocket[pocket] += 1
            passed_by_strategy[strategy] += 1
            passed_by_pocket_strategy[(pocket, strategy)] += 1
            continue

        if status != "entry_guard_block":
            continue
        if client_order_id and client_order_id in seen_block:
            continue
        if client_order_id:
            seen_block.add(client_order_id)
        blocked_total += 1
        reason = "unknown"
        if isinstance(payload, dict):
            reason = payload.get("reason") or reason
        blocked_by_reason[reason] += 1
        blocked_by_pocket[pocket] += 1

        strategy = _resolve_strategy(payload, client_order_id, args.group_base)
        blocked_by_strategy[strategy] += 1
        blocked_by_reason_strategy[(reason, strategy)] += 1
        blocked_by_pocket_strategy[(pocket, strategy)] += 1

        align_bucket = "na"
        entry_guard = payload.get("entry_guard") if isinstance(payload, dict) else None
        if isinstance(entry_guard, dict):
            mtf = entry_guard.get("mtf")
            if isinstance(mtf, dict):
                try:
                    align = int(mtf.get("align_count"))
                    align_counts[align] += 1
                    align_bucket = str(align)
                except Exception:
                    pass
            try:
                align_count = mtf.get("align_count") if isinstance(mtf, dict) else None
                _push_metric(reason, "align_count", align_count, strategy)
            except Exception:
                pass
            if reason.startswith("entry_guard_overheat"):
                overheat = entry_guard.get("overheat")
                if isinstance(overheat, dict):
                    _push_metric(reason, "rsi", overheat.get("rsi"), strategy)
            if reason.startswith("entry_guard_adx_"):
                adx_range = entry_guard.get("adx_range")
                if isinstance(adx_range, dict):
                    _push_metric(reason, "adx", adx_range.get("adx"), strategy)
            if reason == "entry_guard_ma20_gap":
                ma_gap = entry_guard.get("ma20_gap_atr")
                if isinstance(ma_gap, dict):
                    _push_metric(reason, "gap_atr", ma_gap.get("gap_atr"), strategy)
            if reason.startswith("entry_guard_extreme"):
                entry = entry_guard.get("entry")
                upper = entry_guard.get("upper")
                lower = entry_guard.get("lower")
                range_pips = entry_guard.get("range_pips")
                edge_pips = None
                try:
                    if entry is not None and upper is not None:
                        edge_pips = (float(entry) - float(upper)) / 0.01
                        _push_metric(reason, "edge_pips", edge_pips, strategy)
                    if entry is not None and lower is not None:
                        edge_pips = (float(lower) - float(entry)) / 0.01
                        _push_metric(reason, "edge_pips", edge_pips, strategy)
                except Exception:
                    pass
                _push_metric(reason, "range_pips", range_pips, strategy)
                if edge_pips is not None and range_pips:
                    try:
                        _push_metric(
                            reason,
                            "edge_frac",
                            float(edge_pips) / float(range_pips),
                            strategy,
                        )
                    except Exception:
                        pass
                _push_metric(reason, "adx", entry_guard.get("adx"), strategy)
            if reason.startswith("entry_guard_mid_far"):
                distance_pips = entry_guard.get("distance_pips")
                range_pips = entry_guard.get("range_pips")
                _push_metric(reason, "distance_pips", distance_pips, strategy)
                _push_metric(reason, "range_pips", range_pips, strategy)
                if distance_pips is not None and range_pips:
                    try:
                        _push_metric(
                            reason,
                            "distance_frac",
                            float(distance_pips) / float(range_pips),
                            strategy,
                        )
                    except Exception:
                        pass
                _push_metric(reason, "adx", entry_guard.get("adx"), strategy)
        blocked_by_reason_pocket[(reason, pocket)] += 1
        blocked_by_reason_align[(reason, align_bucket)] += 1

    print(f"=== Entry Guard Blocks (last {args.days} days) ===")
    guard_total = passed_total + blocked_total
    print(
        f"preflight_unique={preflight_total} guard_total={guard_total} blocked={blocked_total} blocked_rate={_ratio(blocked_total, guard_total)}"
    )
    print(f"passed={passed_total} pass_rate={_ratio(passed_total, guard_total)}")

    print("\n-- by reason --")
    for reason, count in blocked_by_reason.most_common():
        print(f"{reason:28s} {count:4d} ({_ratio(count, blocked_total)})")

    print("\n-- by pocket --")
    for pocket, count in blocked_by_pocket.most_common():
        total = count + passed_by_pocket.get(pocket, 0)
        rate = _ratio(count, total)
        print(f"{pocket:10s} {count:4d} ({_ratio(count, blocked_total)}) rate={rate}")

    print("\n-- by strategy (top) --")
    for tag, count in blocked_by_strategy.most_common(max(1, args.top)):
        print(f"{tag:24s} {count:4d} ({_ratio(count, blocked_total)})")

    print("\n-- by strategy (block rate) --")
    totals = {
        tag: blocked_by_strategy.get(tag, 0) + passed_by_strategy.get(tag, 0)
        for tag in set(blocked_by_strategy) | set(passed_by_strategy)
    }
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    shown = 0
    for tag, total in ranked:
        if total < max(1, args.min_total):
            continue
        blocked = blocked_by_strategy.get(tag, 0)
        passed = passed_by_strategy.get(tag, 0)
        print(
            f"{tag:24s} blocked={blocked:4d} passed={passed:4d} rate={_ratio(blocked, total)}"
        )
        shown += 1
        if shown >= max(1, args.top):
            break

    if align_counts:
        print("\n-- align_count distribution --")
        for key in sorted(align_counts):
            count = align_counts[key]
            print(f"{key:2d}: {count:4d} ({_ratio(count, blocked_total)})")

    if blocked_by_reason_strategy:
        print("\n-- top reason x strategy --")
        ranked = sorted(
            blocked_by_reason_strategy.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for (reason, tag), count in ranked[: max(1, args.top)]:
            print(f"{reason:20s} {tag:24s} {count:4d}")

    if blocked_by_reason_pocket:
        print("\n-- top reason x pocket --")
        ranked = sorted(
            blocked_by_reason_pocket.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for (reason, pocket), count in ranked[: max(1, args.top)]:
            print(f"{reason:20s} {pocket:10s} {count:4d}")

    if blocked_by_reason_align:
        print("\n-- top reason x align_count --")
        ranked = sorted(
            blocked_by_reason_align.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for (reason, align), count in ranked[: max(1, args.top)]:
            print(f"{reason:20s} {align:>3s} {count:4d}")

    if metrics_by_reason:
        print("\n-- reason metrics (p25/median/p75) --")
        for reason, _count in blocked_by_reason.most_common(max(1, args.top)):
            metrics = metrics_by_reason.get(reason)
            if not metrics:
                continue
            for key in sorted(metrics.keys()):
                values = metrics[key]
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} {key:12s} n={len(values):4d} p25={_fmt_num(p25)} med={_fmt_num(med)} p75={_fmt_num(p75)}"
                )

        print("\n-- suggested thresholds (allow 25/50/75% of blocked) --")
        for reason, _count in blocked_by_reason.most_common(max(1, args.top)):
            metrics = metrics_by_reason.get(reason)
            if not metrics:
                continue
            if reason.startswith("entry_guard_overheat"):
                values = metrics.get("rsi", [])
                p25, med, p75 = _quantiles(values)
                if reason.endswith("long"):
                    print(
                        f"{reason:20s} RSI_HIGH -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)"
                    )
                else:
                    print(
                        f"{reason:20s} RSI_LOW  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (lower=more lenient)"
                    )
            elif reason == "entry_guard_adx_low":
                values = metrics.get("adx", [])
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} ADX_MIN  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (lower=more lenient)"
                )
            elif reason == "entry_guard_adx_high":
                values = metrics.get("adx", [])
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} ADX_MAX  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)"
                )
            elif reason == "entry_guard_ma20_gap":
                values = metrics.get("gap_atr", [])
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} GAP_ATR  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)"
                )
            elif reason.startswith("entry_guard_mid_far"):
                values = metrics.get("distance_pips", [])
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} MID_PIPS -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)"
                )
            elif reason.startswith("entry_guard_extreme"):
                values = metrics.get("edge_frac", [])
                p25, med, p75 = _quantiles(values)
                print(
                    f"{reason:20s} FIB_DELTA-> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (subtract from current)"
                )

    if blocked_by_pocket_strategy or passed_by_pocket_strategy:
        print("\n-- by pocket x strategy (block rate) --")
        totals = {
            key: blocked_by_pocket_strategy.get(key, 0) + passed_by_pocket_strategy.get(key, 0)
            for key in set(blocked_by_pocket_strategy) | set(passed_by_pocket_strategy)
        }
        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        shown = 0
        for (pocket, tag), total in ranked:
            if total < max(1, args.min_total):
                continue
            blocked = blocked_by_pocket_strategy.get((pocket, tag), 0)
            passed = passed_by_pocket_strategy.get((pocket, tag), 0)
            print(
                f"{pocket:10s} {tag:22s} blocked={blocked:4d} passed={passed:4d} rate={_ratio(blocked, total)}"
            )
            shown += 1
            if shown >= max(1, args.top):
                break

    if metrics_by_reason_strategy and blocked_by_strategy:
        print("\n-- suggested thresholds by strategy (top) --")
        top_strategies = [tag for tag, _ in blocked_by_strategy.most_common(max(1, args.top))]
        for tag in top_strategies:
            printed = False
            for reason, _count in blocked_by_reason.most_common():
                metrics = metrics_by_reason_strategy.get((reason, tag))
                if not metrics:
                    continue
                if not printed:
                    print(f"[{tag}]")
                    printed = True
                if reason.startswith("entry_guard_overheat"):
                    values = metrics.get("rsi", [])
                    p25, med, p75 = _quantiles(values)
                    label = "RSI_HIGH" if reason.endswith("long") else "RSI_LOW "
                    hint = "higher=more lenient" if reason.endswith("long") else "lower=more lenient"
                    print(f"  {reason:18s} {label} -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} ({hint})")
                elif reason == "entry_guard_adx_low":
                    values = metrics.get("adx", [])
                    p25, med, p75 = _quantiles(values)
                    print(f"  {reason:18s} ADX_MIN  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (lower=more lenient)")
                elif reason == "entry_guard_adx_high":
                    values = metrics.get("adx", [])
                    p25, med, p75 = _quantiles(values)
                    print(f"  {reason:18s} ADX_MAX  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)")
                elif reason == "entry_guard_ma20_gap":
                    values = metrics.get("gap_atr", [])
                    p25, med, p75 = _quantiles(values)
                    print(f"  {reason:18s} GAP_ATR  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)")
                elif reason.startswith("entry_guard_mid_far"):
                    values = metrics.get("distance_pips", [])
                    p25, med, p75 = _quantiles(values)
                    print(f"  {reason:18s} MID_PIPS -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (higher=more lenient)")
                elif reason.startswith("entry_guard_extreme"):
                    values = metrics.get("edge_frac", [])
                    p25, med, p75 = _quantiles(values)
                    print(f"  {reason:18s} FIB_DELTA-> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (subtract from current)")
                    adx_vals = metrics.get("adx", [])
                    if adx_vals:
                        p25, med, p75 = _quantiles(adx_vals)
                        print(
                            f"  {reason:18s} ADX_MIN  -> {_fmt_num(p25)} / {_fmt_num(med)} / {_fmt_num(p75)} (lower=more bypass)"
                        )


if __name__ == "__main__":
    main()
