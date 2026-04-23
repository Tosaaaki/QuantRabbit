#!/usr/bin/env python3
"""Place discretionary trader orders directly in OANDA, auto-log them, and print a state receipt.

Usage:
  python3 tools/place_trader_order.py LIMIT EUR_USD LONG 3000 --entry 1.17815 --tp 1.17893 --sl 1.17758 --thesis first_defense_rearm
  python3 tools/place_trader_order.py STOP-ENTRY EUR_JPY LONG 3000 --entry 187.286 --tp 187.420 --sl 187.148 --thesis tokyo_reclaim_stop
  python3 tools/place_trader_order.py MARKET EUR_USD SHORT 3000 --tp 1.17710 --sl 1.17835 --thesis honest_break_now
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_loader import get_oanda_config
from pricing_probe import execution_guard, probe_market
from technicals_json import load_technicals_timeframes, timeframe_age_minutes
from trader_order_guard import (
    exact_allocation_band,
    exact_allocation_grade,
    exact_pretrade_advisories,
    exact_pretrade_issues,
    exact_pretrade_label,
    requested_style_from_order_type,
    run_exact_pretrade,
)
from validate_trader_state import STATE_PATH, validate_state_for_entry


ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_FILE = ROOT / "logs" / "live_trade_log.txt"
TRADER_TAG = "trader"
ORDER_TYPES = {"MARKET", "LIMIT", "STOP", "STOP-ENTRY"}
TECHNICAL_STALE_LIMITS = {
    "M1": 4.0,
    "M5": 10.0,
    "M15": 30.0,
    "H1": 120.0,
}
UNIT_BANDS = {
    "S": (8000, 10000),
    "A": (4000, 8000),
    "B+": (3000, 4500),
    "B0": (2000, 3000),
    "B-": (1000, 2000),
    "C": (0, 1000),
}
COUNTER_KEYWORDS = ("counter", "reversal", "mean_revert", "mean-revert")


def _normalize_order_type(value: str) -> str:
    upper = value.strip().upper()
    if upper not in ORDER_TYPES:
        raise ValueError(f"Unsupported order type: {value}")
    return "STOP" if upper == "STOP-ENTRY" else upper


def _normalize_side(value: str) -> str:
    upper = value.strip().upper()
    if upper not in {"LONG", "SHORT"}:
        raise ValueError("Side must be LONG or SHORT.")
    return upper


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("JPY") else 10000


def _format_price(price: float, pair: str) -> str:
    return f"{price:.3f}" if pair.endswith("JPY") else f"{price:.5f}"


def _slug_comment(*parts: str | None) -> str:
    raw = "_".join(part.strip().lower() for part in parts if part and part.strip())
    slug = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return (slug or "trader_order")[:120]


def _oanda_request(path: str, cfg: dict[str, object], *, method: str = "GET", body: dict | None = None) -> dict:
    url = f"{cfg['oanda_base_url']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _fetch_live_quote(pair: str, cfg: dict[str, object]) -> tuple[float | None, float | None, float | None]:
    acct = str(cfg["oanda_account_id"])
    try:
        data = _oanda_request(f"/v3/accounts/{acct}/pricing?instruments={pair}", cfg)
        prices = data.get("prices", [])
        if not prices:
            return None, None, None
        bid = float(prices[0]["bids"][0]["price"])
        ask = float(prices[0]["asks"][0]["price"])
        return bid, ask, abs(ask - bid) * _pip_factor(pair)
    except Exception:
        return None, None, None


def _spread_from_fill(fill: dict, pair: str) -> float | None:
    full_price = fill.get("fullPrice") or {}
    try:
        bids = full_price.get("bids") or []
        asks = full_price.get("asks") or []
        bid = float(bids[0]["price"]) if bids else float(full_price["closeoutBid"])
        ask = float(asks[0]["price"]) if asks else float(full_price["closeoutAsk"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return abs(ask - bid) * _pip_factor(pair)


def _build_gtd_time(hours: int, explicit_time: str | None) -> str:
    if explicit_time:
        return explicit_time
    expiry = datetime.now(timezone.utc) + timedelta(hours=hours)
    return expiry.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _append_log(line: str) -> None:
    with LOG_FILE.open("a") as fh:
        fh.write(line.rstrip() + "\n")


def _notify_slack(pair: str, side: str, units: int, price: str, sl: str, thesis: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "slack_trade_notify.py"),
        "entry",
        "--pair",
        pair,
        "--side",
        side,
        "--units",
        str(units),
        "--price",
        price,
        "--sl",
        sl,
        "--thesis",
        thesis,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=False, capture_output=True, text=True, timeout=15)


def _extract_effective_band(args: argparse.Namespace) -> str | None:
    explicit = str(getattr(args, "allocation_band", "") or "").strip().upper()
    if explicit in UNIT_BANDS:
        return explicit

    pretrade = str(args.pretrade or "")
    matches = re.findall(r"(B\+|B0|B-|A|S|C)", pretrade.upper())
    if matches:
        return matches[-1]

    allocation = str(args.allocation or "").strip().upper()
    if allocation in UNIT_BANDS:
        return allocation
    return None


def _is_counter_reversal(args: argparse.Namespace) -> bool:
    if getattr(args, "counter", False):
        return True
    blob = " ".join(
        part for part in (
            str(args.thesis or ""),
            str(args.reason or ""),
            str(args.pretrade or ""),
        )
        if part
    ).lower()
    return any(token in blob for token in COUNTER_KEYWORDS)


def _recent_best_winner_units(lookback_hours: int = 18) -> int | None:
    if not LOG_FILE.exists():
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    best_units: int | None = None
    line_re = re.compile(
        r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\] "
        r"(?:CLOSE|PARTIAL_CLOSE) .*? (?P<units>\d+)u @.*? P/L=(?P<pl>[+-]?\d+(?:\.\d+)?)JPY"
    )
    for line in LOG_FILE.read_text().splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        pl = float(m.group("pl"))
        if pl <= 0:
            continue
        units = int(m.group("units"))
        if best_units is None or units > best_units:
            best_units = units
    return best_units


def _validate_units(args: argparse.Namespace, effective_band: str | None, counter_reversal: bool) -> list[str]:
    problems: list[str] = []
    if effective_band:
        lower, upper = UNIT_BANDS[effective_band]
        if args.units < lower:
            problems.append(
                f"{effective_band} size floor is {lower}u but the order asks for {args.units}u"
            )
        if upper > 0 and args.units > upper:
            problems.append(
                f"{effective_band} size cap is {upper}u but the order asks for {args.units}u"
            )

    if counter_reversal:
        if args.units > UNIT_BANDS["B+"][1]:
            problems.append(
                f"counter/reversal seats are B-max until they pay; {args.units}u exceeds the {UNIT_BANDS['B+'][1]}u ceiling"
            )
        if args.order_type == "MARKET":
            problems.append("counter/reversal seats must use STOP-ENTRY or LIMIT, not MARKET")

    best_winner_units = _recent_best_winner_units()
    dirty_band = effective_band not in {"A", "S", "B+"}
    if best_winner_units and dirty_band and args.units > max(best_winner_units, 2000):
        label = effective_band or str(args.allocation or "unknown")
        problems.append(
            f"size asymmetry guard: dirty seat {label} at {args.units}u is larger than the recent paid winner size {best_winner_units}u"
        )

    return problems


def _build_requested_entry_text(args: argparse.Namespace) -> str:
    return _format_price(args.entry, args.pair) if args.entry is not None else "MARKET"


def _apply_exact_pretrade_metadata(args: argparse.Namespace, result: dict) -> None:
    args.pretrade = exact_pretrade_label(result)
    exact_allocation = exact_allocation_grade(result)
    exact_band = exact_allocation_band(result)
    if exact_allocation:
        args.allocation = exact_allocation
    if exact_band:
        args.allocation_band = exact_band


def _pair_technical_stale_reasons(pair: str) -> list[str]:
    timeframes = load_technicals_timeframes(ROOT / "logs" / f"technicals_{pair}.json")
    now_utc = datetime.now(timezone.utc)
    reasons = []
    for tf, limit in TECHNICAL_STALE_LIMITS.items():
        age = timeframe_age_minutes(timeframes, tf, now_utc)
        if age is None:
            reasons.append(f"{tf}:missing")
        elif age > limit:
            reasons.append(f"{tf}:{age:.0f}m>{limit:.0f}m")
    return reasons


def _ensure_pair_technicals_fresh(pair: str) -> str | None:
    reasons = _pair_technical_stale_reasons(pair)
    if not reasons:
        return None

    try:
        subprocess.run(
            [str(VENV_PYTHON), str(ROOT / "tools" / "refresh_factor_cache.py"), pair, "--quiet"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return f"{pair} technical refresh failed before order: {exc}"

    reasons = _pair_technical_stale_reasons(pair)
    if reasons:
        return f"{pair} technical cache is stale at order time ({', '.join(reasons)})"
    return None


def _tech_value(payload: dict, field: str) -> float:
    try:
        return float(payload.get(field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _infer_current_pair_regime(pair: str) -> str | None:
    timeframes = load_technicals_timeframes(ROOT / "logs" / f"technicals_{pair}.json")
    m5 = timeframes.get("M5") or {}
    if not m5:
        return None

    adx = _tech_value(m5, "adx")
    plus_di = _tech_value(m5, "plus_di")
    minus_di = _tech_value(m5, "minus_di")
    bbw = _tech_value(m5, "bbw")
    kc_width = _tech_value(m5, "kc_width")
    di_gap = abs(plus_di - minus_di)

    if kc_width > 0 and bbw > 0 and bbw < kc_width * 0.9:
        return "squeeze"
    if adx >= 25 and di_gap >= 8:
        return "trending"
    if adx < 18:
        return "range"
    if bbw < 0.0015:
        return "quiet"
    return "transition"


def _probe_live_tape_summary(
    cfg: dict[str, object],
    pair: str,
) -> tuple[dict | None, str | None]:
    try:
        probe = probe_market(
            cfg,
            pairs=[pair],
            samples=5,
            interval_sec=0.25,
            write_cache=True,
        )
    except Exception as exc:
        return None, f"pricing probe failed before exact pretrade: {exc}"
    summary = (probe.get("pairs") or {}).get(pair) or None
    return summary, None


def _planned_distance_pips(fill_price: float, target_price: float, pair: str) -> float:
    return abs(target_price - fill_price) * _pip_factor(pair)


def _fill_guard_reason(
    args: argparse.Namespace,
    *,
    live_spread_pips: float | None,
    fill_spread_pips: float | None,
    fill_price: float | None,
) -> str | None:
    if fill_spread_pips is None or fill_price is None:
        return None

    target_pips = _planned_distance_pips(fill_price, args.tp, args.pair)
    stop_pips = _planned_distance_pips(fill_price, args.sl, args.pair)
    live_ref = max(float(live_spread_pips or 0.0), 0.1)
    hard_abs = 6.0 if args.pair.endswith("JPY") else 3.0

    too_wide_vs_live = fill_spread_pips >= max(hard_abs, live_ref * 3.0)
    too_wide_vs_target = target_pips > 0 and fill_spread_pips >= target_pips * 0.6
    too_wide_vs_stop = stop_pips > 0 and fill_spread_pips >= stop_pips * 0.75

    if too_wide_vs_live and (too_wide_vs_target or too_wide_vs_stop):
        return (
            f"fill spread {fill_spread_pips:.1f}pip was too wide for the planned path "
            f"(live {live_ref:.1f}pip, target {target_pips:.1f}pip, stop {stop_pips:.1f}pip)"
        )
    return None


def _close_trade_immediately(cfg: dict[str, object], trade_id: str) -> dict:
    acct = str(cfg["oanda_account_id"])
    return _oanda_request(
        f"/v3/accounts/{acct}/trades/{trade_id}/close",
        cfg,
        method="PUT",
        body={"units": "ALL"},
    )


def _build_emergency_close_log(
    args: argparse.Namespace,
    trade_id: str,
    close_fill: dict,
    reason: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    price = close_fill.get("price", "?")
    pl = close_fill.get("pl", "0")
    spread = _spread_from_fill(close_fill, args.pair)
    spread_text = f"{spread:.1f}pip" if spread is not None else "n/a"
    return (
        f"[{now}] EMERGENCY_CLOSE {args.pair} {args.side} {args.units}u @{price} "
        f"P/L={pl}JPY Sp={spread_text} tag={TRADER_TAG} reason={reason} id={trade_id}"
    )


def _build_payload(args: argparse.Namespace, live_spread_pips: float | None = None) -> dict:
    signed_units = str(args.units if args.side == "LONG" else -args.units)
    comment = _slug_comment(args.reason, args.thesis)
    order: dict[str, object] = {
        "type": args.order_type,
        "instrument": args.pair,
        "units": signed_units,
        "takeProfitOnFill": {
            "price": _format_price(args.tp, args.pair),
            "timeInForce": "GTC",
        },
        "stopLossOnFill": {
            "price": _format_price(args.sl, args.pair),
            "timeInForce": "GTC",
        },
        "clientExtensions": {
            "tag": TRADER_TAG,
            "comment": comment,
        },
        "tradeClientExtensions": {
            "tag": TRADER_TAG,
            "comment": comment,
        },
    }
    if args.order_type == "MARKET":
        order["timeInForce"] = "FOK"
    else:
        order["price"] = _format_price(args.entry, args.pair)
        order["timeInForce"] = "GTD"
        order["gtdTime"] = _build_gtd_time(args.gtd_hours, args.gtd_time)
        if args.order_type == "STOP":
            order["priceBound"] = _stop_price_bound(args.entry, args.pair, args.side, live_spread_pips)
    return {"order": order}


def _stop_price_bound(entry_price: float, pair: str, side: str, live_spread_pips: float | None) -> str:
    bound_pips = max(float(live_spread_pips or 0.0) * 2.0, 4.0 if pair.endswith("JPY") else 1.5)
    delta = bound_pips / _pip_factor(pair)
    worst_price = entry_price + delta if side == "LONG" else entry_price - delta
    return _format_price(worst_price, pair)


def _build_log_and_receipt(
    args: argparse.Namespace,
    *,
    result: dict | None,
    spread_pips: float | None,
    dry_run: bool,
) -> tuple[str, str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    spread_text = f"{spread_pips:.1f}pip" if spread_pips is not None else "n/a"
    tp = _format_price(args.tp, args.pair)
    sl = _format_price(args.sl, args.pair)
    entry = _format_price(args.entry, args.pair) if args.entry is not None else "MARKET"
    pretrade = args.pretrade or "n/a"
    allocation = args.allocation or "n/a"
    thesis = args.thesis
    reason_text = f" reason={args.reason}" if args.reason else ""

    if dry_run:
        if args.order_type == "MARKET":
            return (
                f"[{now}] ENTRY {args.pair} {args.side} {args.units}u @DRY_RUN TP={tp} SL={sl} Sp={spread_text} | "
                f"thesis={thesis} | pretrade={pretrade} allocation={allocation} tag={TRADER_TAG}",
                "ENTER NOW already filled as trade id=`DRY_RUN`",
            )
        order_label = "ENTRY_ORDER" if args.order_type == "STOP" else "LIMIT"
        gtd = _build_gtd_time(args.gtd_hours, args.gtd_time)
        receipt = f"armed {'STOP' if args.order_type == 'STOP' else 'LIMIT'} id=`DRY_RUN`"
        return (
            f"[{now}] {order_label} {args.pair} {args.side} {args.units}u @{entry} "
            f"{'id=DRY_RUN ' if args.order_type == 'STOP' else ''}"
            f"TP={tp} SL={sl} GTD={gtd} {'id=DRY_RUN ' if args.order_type == 'LIMIT' else ''}"
            f"Sp={spread_text} pretrade={pretrade} allocation={allocation} tag={TRADER_TAG} thesis={thesis}{reason_text}".replace(
                "  ", " "
            ).strip(),
            receipt,
        )

    if args.order_type == "MARKET":
        fill = (result or {}).get("orderFillTransaction", {})
        fill_price = fill.get("price", entry)
        trade_id = (fill.get("tradeOpened") or {}).get("tradeID") or fill.get("id") or "?"
        log_line = (
            f"[{now}] ENTRY {args.pair} {args.side} {args.units}u @{fill_price} id={trade_id} "
            f"TP={tp} SL={sl} Sp={spread_text} | thesis={thesis} | pretrade={pretrade} allocation={allocation} tag={TRADER_TAG}"
        )
        return log_line, f"ENTER NOW already filled as trade id=`{trade_id}`"

    create = (result or {}).get("orderCreateTransaction", {})
    order_id = create.get("id", "?")
    gtd = create.get("gtdTime") or _build_gtd_time(args.gtd_hours, args.gtd_time)
    if args.order_type == "STOP":
        log_line = (
            f"[{now}] ENTRY_ORDER {args.pair} {args.side} {args.units}u @{entry} id={order_id} TP={tp} SL={sl} "
            f"GTD={gtd} Sp={spread_text} pretrade={pretrade} allocation={allocation} tag={TRADER_TAG} thesis={thesis}{reason_text}"
        )
        return log_line, f"armed STOP id=`{order_id}`"

    log_line = (
        f"[{now}] LIMIT {args.pair} {args.side} {args.units}u @{entry} TP={tp} SL={sl} GTD={gtd} id={order_id} "
        f"Sp={spread_text} pretrade={pretrade} thesis={thesis} allocation={allocation} tag={TRADER_TAG}{reason_text}"
    )
    return log_line, f"armed LIMIT id=`{order_id}`"


def _build_reject_log(args: argparse.Namespace, reason: str, spread_pips: float | None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    spread_text = f"{spread_pips:.1f}pip" if spread_pips is not None else "n/a"
    entry = _format_price(args.entry, args.pair) if args.entry is not None else "MARKET"
    return (
        f"[{now}] ORDER_REJECT {args.order_type} {args.pair} {args.side} {args.units}u @{entry} "
        f"TP={_format_price(args.tp, args.pair)} SL={_format_price(args.sl, args.pair)} Sp={spread_text} "
        f"tag={TRADER_TAG} reason={reason}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_type", help="MARKET / LIMIT / STOP-ENTRY")
    parser.add_argument("pair")
    parser.add_argument("side")
    parser.add_argument("units", type=int)
    parser.add_argument("--entry", type=float)
    parser.add_argument("--tp", type=float, required=True)
    parser.add_argument("--sl", type=float, required=True)
    parser.add_argument("--gtd-hours", type=int, default=6)
    parser.add_argument("--gtd-time")
    parser.add_argument("--thesis", required=True)
    parser.add_argument("--pretrade")
    parser.add_argument("--allocation")
    parser.add_argument("--allocation-band")
    parser.add_argument("--counter", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--auto-slack", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.order_type = _normalize_order_type(args.order_type)
    args.side = _normalize_side(args.side)
    args.pair = args.pair.upper()

    if args.order_type in {"LIMIT", "STOP"} and args.entry is None:
        parser.error("--entry is required for LIMIT / STOP-ENTRY orders.")
    if args.order_type == "MARKET" and args.entry is not None:
        parser.error("--entry is not used for MARKET orders.")

    state_errors = validate_state_for_entry(
        STATE_PATH.resolve(),
        verify_live_oanda=True,
        verify_live_book_coverage=True,
    )
    if state_errors:
        print("FRESH_RISK_BLOCKED", file=sys.stderr)
        print(
            "Resolve state/live-book drift before sending a new trader order.",
            file=sys.stderr,
        )
        for error in state_errors[:20]:
            print(f"- {error}", file=sys.stderr)
        return 2

    cfg = get_oanda_config()
    freshness_error = _ensure_pair_technicals_fresh(args.pair)
    if freshness_error:
        _append_log(_build_reject_log(args, freshness_error, None))
        print("ORDER_BLOCKED", file=sys.stderr)
        print(f"- {freshness_error}", file=sys.stderr)
        return 4

    live_bid, live_ask, spread_pips = _fetch_live_quote(args.pair, cfg)
    market_entry_proxy = None
    if args.order_type == "MARKET":
        market_entry_proxy = live_ask if args.side == "LONG" else live_bid
        if market_entry_proxy is None:
            reason = "live bid/ask quote is unavailable, so exact market-entry geometry cannot be verified"
            _append_log(_build_reject_log(args, reason, spread_pips))
            print("ORDER_BLOCKED", file=sys.stderr)
            print(f"- {reason}", file=sys.stderr)
            return 6
    current_regime = _infer_current_pair_regime(args.pair)
    live_tape_summary, live_tape_error = _probe_live_tape_summary(cfg, args.pair)

    exact_result = run_exact_pretrade(
        pair=args.pair,
        direction=args.side,
        entry_price=args.entry if args.entry is not None else market_entry_proxy,
        tp_price=args.tp,
        sl_price=args.sl,
        counter=bool(args.counter),
        regime=current_regime,
        spread_pips=spread_pips,
        live_tape=live_tape_summary,
    )
    exact_errors = exact_pretrade_issues(
        requested_style=requested_style_from_order_type(args.order_type),
        result=exact_result,
    )
    exact_advisories = exact_pretrade_advisories(
        requested_style=requested_style_from_order_type(args.order_type),
        result=exact_result,
    )
    _apply_exact_pretrade_metadata(args, exact_result)
    if exact_errors:
        reason = "; ".join(exact_errors)
        _append_log(_build_reject_log(args, reason, spread_pips))
        print("ORDER_BLOCKED", file=sys.stderr)
        for item in exact_errors:
            print(f"- {item}", file=sys.stderr)
        return 6

    if args.order_type == "MARKET":
        if live_tape_summary is None:
            reason = live_tape_error or "pricing probe failed before market order"
            _append_log(_build_reject_log(args, reason, None))
            print("ORDER_BLOCKED", file=sys.stderr)
            print(f"- {reason}", file=sys.stderr)
            return 5
        probe_error = execution_guard(live_tape_summary, order_type=args.order_type)
        if probe_error:
            _append_log(_build_reject_log(args, probe_error, None))
            print("ORDER_BLOCKED", file=sys.stderr)
            print(f"- {probe_error}", file=sys.stderr)
            return 5

    live_spread_before_order = spread_pips
    effective_band = _extract_effective_band(args)
    counter_reversal = _is_counter_reversal(args)
    size_errors = _validate_units(args, effective_band, counter_reversal)
    if size_errors:
        reason = "; ".join(size_errors)
        _append_log(_build_reject_log(args, reason, spread_pips))
        print("ORDER_BLOCKED", file=sys.stderr)
        for item in size_errors:
            print(f"- {item}", file=sys.stderr)
        return 3
    payload = _build_payload(args, live_spread_before_order)

    if args.dry_run:
        log_line, state_receipt = _build_log_and_receipt(args, result=None, spread_pips=spread_pips, dry_run=True)
        print("DRY_RUN")
        if effective_band:
            print(f"EFFECTIVE_BAND: {effective_band}")
        print(f"COUNTER_REVERSAL: {'YES' if counter_reversal else 'NO'}")
        print(f"STATE_RECEIPT: {state_receipt}")
        for item in exact_advisories:
            print(f"ADVISORY: {item}")
        print(f"LOG_LINE: {log_line}")
        if args.json:
            print(json.dumps(payload, indent=2))
        return 0

    for item in exact_advisories:
        print(f"ADVISORY: {item}", file=sys.stderr)

    acct = str(cfg["oanda_account_id"])
    try:
        result = _oanda_request(f"/v3/accounts/{acct}/orders", cfg, method="POST", body=payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if hasattr(exc, "read") else str(exc)
        _append_log(_build_reject_log(args, body.replace("\n", " "), spread_pips))
        print(body, file=sys.stderr)
        return 1

    fill_tx = result.get("orderFillTransaction", {}) or {}
    fill_spread_pips = _spread_from_fill(fill_tx, args.pair)
    if args.order_type == "MARKET":
        spread_pips = fill_spread_pips or spread_pips

    log_line, state_receipt = _build_log_and_receipt(args, result=result, spread_pips=spread_pips, dry_run=False)
    _append_log(log_line)

    trade_opened = fill_tx.get("tradeOpened") or {}
    trade_id = trade_opened.get("tradeID")
    fill_price = None
    try:
        fill_price = float(fill_tx.get("price")) if fill_tx.get("price") is not None else None
    except (TypeError, ValueError):
        fill_price = None
    fill_guard_reason = None
    if args.order_type in {"MARKET", "STOP"} and trade_id:
        fill_guard_reason = _fill_guard_reason(
            args,
            live_spread_pips=live_spread_before_order,
            fill_spread_pips=fill_spread_pips or spread_pips,
            fill_price=fill_price,
        )
    if fill_guard_reason and trade_id:
        close_result = _close_trade_immediately(cfg, str(trade_id))
        close_fill = close_result.get("orderFillTransaction", {}) or {}
        _append_log(_build_emergency_close_log(args, str(trade_id), close_fill, fill_guard_reason))
        print("FILL_GUARD_TRIPPED", file=sys.stderr)
        print(fill_guard_reason, file=sys.stderr)
        print(f"Closed trade id={trade_id} immediately.", file=sys.stderr)
        return 4

    if args.auto_slack:
        if args.order_type == "MARKET":
            fill_price = (result.get("orderFillTransaction", {}) or {}).get("price")
            _notify_slack(
                args.pair,
                args.side,
                args.units,
                str(fill_price or "MARKET"),
                _format_price(args.sl, args.pair),
                args.thesis,
            )
        else:
            price = _format_price(args.entry, args.pair)
            thesis = f"{args.order_type} armed | {args.thesis}"
            _notify_slack(args.pair, args.side, args.units, price, _format_price(args.sl, args.pair), thesis)

    print("ORDER_OK")
    print(f"STATE_RECEIPT: {state_receipt}")
    print(f"LOG_LINE: {log_line}")
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
