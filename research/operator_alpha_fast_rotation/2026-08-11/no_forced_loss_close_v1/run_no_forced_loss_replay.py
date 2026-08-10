#!/usr/bin/env python3
"""Causal, research-only NO_FIXED_SL comparison on local M1 bid/ask truth.

The module contains no broker client and never closes an open leg merely
because the replay ends.  Open legs are marked to the final executable quote.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parent
PAIRS = ("EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD")
SOURCE_DEFAULTS = {
    "USD_JPY": "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026/20260718T084558Z/USD_JPY/USD_JPY_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz",
    "EUR_USD": "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026/20260718T085350Z/EUR_USD/EUR_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz",
    "GBP_USD": "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026/20260718T104309Z/GBP_USD/GBP_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz",
    "AUD_USD": "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026/20260718T105320Z/AUD_USD/AUD_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz",
}
START_EQUITY = 254_209.0185
UNITS = 5_000
MARGIN_RATE = 0.04
TERMINAL_DAYS = 10
SLIPPAGE_SPREAD_FRACTION = 0.10


def arm_names() -> list[str]:
    names = [
        "A_HARD_SL_BASELINE",
        "B_NO_SL_NAKED_RETURN_WAIT",
        "C_NO_SL_DELAYED_ENTRY_EARLY_TP",
        "D_NO_SL_HEDGE_RETURN_025",
        "D_NO_SL_HEDGE_RETURN_050",
        "D_NO_SL_HEDGE_RETURN_100",
        "E_NO_SL_PARTIAL_PROFIT_BE",
        "F_NO_SL_MULTI_PAIR_ROTATION",
    ]
    for base in (
        "H1_LOCK_AT_ADVERSE_LEVEL_AND_WAIT",
        "H2_HEDGE_TP_KEEP_ORIGINAL",
        "H3_HEDGE_PARTIAL_TP_REHEDGE",
        "H4_HEDGE_REVERSAL_CONFIRM_EXIT",
        "H5_HEDGE_PROFIT_OFFSET_ORIGINAL_BE",
    ):
        names.extend(f"{base}_{suffix}" for suffix in ("025", "050", "100", "150"))
    names.extend(("H6_PERSISTENT_TREND_STRESS", "H7_GAP_AND_FINANCING_STRESS"))
    return names


def parse_time(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, rest = text.split(".", 1)
        frac, zone = rest.split("+", 1)
        text = f"{head}.{frac[:6]}+{zone}"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("complete") is not True:
                continue
            bid, ask = row.get("bid"), row.get("ask")
            if not isinstance(bid, dict) or not isinstance(ask, dict):
                continue
            ts = parse_time(str(row["time"]))
            values = (
                ts,
                float(bid["o"]), float(bid["h"]), float(bid["l"]), float(bid["c"]),
                float(ask["o"]), float(ask["h"]), float(ask["l"]), float(ask["c"]),
            )
            if all(math.isfinite(v) for v in values[1:]) and values[4] < values[8]:
                rows.append(values)
    return rows


def mid_close(row: tuple[Any, ...]) -> float:
    return (row[4] + row[8]) / 2.0


def spread_close(row: tuple[Any, ...]) -> float:
    return row[8] - row[4]


def observed_weekdays(count: int = 200) -> list[date]:
    cursor = date(2025, 1, 2)
    values: list[date] = []
    while len(values) < count:
        if cursor.weekday() < 5:
            values.append(cursor)
        cursor += timedelta(days=1)
    return values


def scheduled_dates() -> dict[str, list[date]]:
    result = {pair: [] for pair in PAIRS}
    for index, day in enumerate(observed_weekdays()):
        result[PAIRS[index % len(PAIRS)]].append(day)
    return result


def build_decision(pair: str, day: date, rows: list[tuple[Any, ...]], by_time: dict[datetime, int], quote_to_jpy: float) -> dict[str, Any]:
    scan_ts = datetime.combine(day, time(8, 0), tzinfo=timezone.utc)
    index = by_time.get(scan_ts)
    identity = {"decision_id": f"NOFLC-{day.isoformat()}-{pair}", "pair": pair, "scan_utc": iso(scan_ts)}
    if index is None or index < 60:
        return {**identity, "status": "SKIP_MISSING_SCAN"}
    history = rows[index - 60:index]
    if any((b[0] - a[0]).total_seconds() > 120 for a, b in zip(history, history[1:])):
        return {**identity, "status": "SKIP_MISSING_CAUSAL_WINDOW"}
    mids = [mid_close(row) for row in history]
    side = "LONG" if mids[-1] > mids[0] else "SHORT"
    ranges = [max(row[2], row[6]) - min(row[3], row[7]) for row in history]
    atr = mean(ranges) * 12.0
    if not math.isfinite(atr) or atr <= 0.0:
        return {**identity, "status": "SKIP_INVALID_ATR"}
    scan_mid = mid_close(rows[index - 1])
    adverse_seen = False
    previous_m5_close = scan_mid
    confirm_index: int | None = None
    deadline = scan_ts + timedelta(minutes=60)
    for cursor in range(index, len(rows)):
        row = rows[cursor]
        if row[0] > deadline:
            break
        if row[0].minute % 5 != 4:
            continue
        current = mid_close(row)
        adverse = scan_mid - current if side == "LONG" else current - scan_mid
        adverse_seen = adverse_seen or adverse >= 0.25 * atr
        resumed = current > previous_m5_close if side == "LONG" else current < previous_m5_close
        if adverse_seen and resumed:
            confirm_index = cursor
            break
        previous_m5_close = current
    if confirm_index is None or confirm_index + 1 >= len(rows):
        return {**identity, "status": "SKIP_NO_RETURN_CONFIRM", "side": side, "atr": atr}
    entry_index = confirm_index + 1
    if (rows[entry_index][0] - rows[confirm_index][0]).total_seconds() > 120:
        return {**identity, "status": "SKIP_ENTRY_GAP", "side": side, "atr": atr}
    return {
        **identity,
        "status": "CONFIRMED",
        "side": side,
        "atr": atr,
        "scan_mid": scan_mid,
        "confirm_utc": iso(rows[confirm_index][0]),
        "entry_utc": iso(rows[entry_index][0]),
        "entry_index": entry_index,
        "quote_to_jpy": quote_to_jpy,
    }


def executable(row: tuple[Any, ...], side: str, action: str) -> float:
    if action == "ENTRY":
        return row[5] if side == "LONG" else row[1]
    return row[4] if side == "LONG" else row[8]


def directional(side: str, start: float, end: float, units: float, q: float) -> float:
    sign = 1.0 if side == "LONG" else -1.0
    return sign * (end - start) * units * q


def suffix_trigger(arm: str) -> float:
    suffix = arm.rsplit("_", 1)[-1]
    return {"025": 0.25, "050": 0.5, "100": 1.0, "150": 1.5}.get(suffix, 0.5)


def simulate(decision: dict[str, Any], rows: list[tuple[Any, ...]], arm: str, account_mode: str) -> dict[str, Any]:
    base = {"decision_id": decision["decision_id"], "pair": decision["pair"], "arm": arm, "account_mode": account_mode}
    if decision["status"] != "CONFIRMED":
        return {**base, "status": decision["status"], "executed": False, "terminal_contribution_jpy": 0.0}
    side, q, atr = decision["side"], decision["quote_to_jpy"], decision["atr"]
    start_index = int(decision["entry_index"])
    if arm == "C_NO_SL_DELAYED_ENTRY_EARLY_TP":
        delayed = start_index + 5
        if delayed >= len(rows) or (rows[delayed][0] - rows[start_index][0]).total_seconds() > 420:
            return {**base, "status": "SKIP_DELAY_GAP", "executed": False, "terminal_contribution_jpy": 0.0}
        early_move = directional(side, mid_close(rows[start_index]), mid_close(rows[delayed]), 1.0, 1.0)
        if early_move > 0.25 * atr:
            return {**base, "status": "SKIP_DO_NOT_CHASE", "executed": False, "terminal_contribution_jpy": 0.0}
        start_index = delayed
    entry_row = rows[start_index]
    entry = executable(entry_row, side, "ENTRY")
    spread_cost = spread_close(entry_row) * UNITS * q / 2.0
    slippage = spread_close(entry_row) * SLIPPAGE_SPREAD_FRACTION * UNITS * q
    original_units = float(UNITS)
    original_open = True
    original_realized = 0.0
    hedge_realized = 0.0
    hedge: dict[str, Any] | None = None
    hedge_count = 0
    partial_done = False
    hedge_partial_done = False
    netting_reductions = 0
    forced = False
    recovery_seconds: float | None = None
    underwater_start: datetime | None = None
    max_underwater_seconds = 0.0
    max_open_risk = 0.0
    max_equity_dd = 0.0
    max_margin = 0.0
    max_gross_margin = 0.0
    min_margin_excess = START_EQUITY
    max_gross_exposure_units = original_units
    max_net_exposure_units = original_units
    highwater = START_EQUITY
    cash_delta = -slippage
    costs = slippage
    turnover_notional = entry * original_units * q
    last_rehedge_time: datetime | None = None
    m5_reversal = 0
    previous_m5 = mid_close(entry_row)
    terminal = entry_row[0] + timedelta(days=TERMINAL_DAYS)
    last_row = entry_row
    trigger = suffix_trigger(arm)
    if arm.startswith("D_NO_SL_HEDGE_RETURN_"):
        trigger = {"025": 0.25, "050": 0.5, "100": 1.0}[arm.rsplit("_", 1)[-1]]
    is_hedge_arm = arm.startswith(("D_", "H1_", "H2_", "H3_", "H4_", "H5_"))

    def close_leg(leg_side: str, leg_entry: float, units: float, row: tuple[Any, ...]) -> tuple[float, float, float]:
        px = executable(row, leg_side, "EXIT")
        slip = spread_close(row) * SLIPPAGE_SPREAD_FRACTION * units * q
        spread_est = spread_close(row) * units * q / 2.0
        return directional(leg_side, leg_entry, px, units, q) - slip, slip, spread_est

    for index in range(start_index + 1, len(rows)):
        row = rows[index]
        if row[0] > terminal:
            break
        last_row = row
        if (row[0] - rows[index - 1][0]).total_seconds() > 180:
            continue
        original_mtm = directional(side, entry, executable(row, side, "EXIT"), original_units, q) if original_open else 0.0
        hedge_mtm = 0.0
        if hedge is not None:
            hedge_mtm = directional(hedge["side"], hedge["entry"], executable(row, hedge["side"], "EXIT"), hedge["units"], q)
        equity = START_EQUITY + cash_delta + original_realized + hedge_realized + original_mtm + hedge_mtm
        highwater = max(highwater, equity)
        max_equity_dd = max(max_equity_dd, highwater - equity)
        open_loss = max(0.0, -(original_mtm + hedge_mtm))
        max_open_risk = max(max_open_risk, open_loss)
        if equity < START_EQUITY:
            underwater_start = underwater_start or row[0]
            max_underwater_seconds = max(max_underwater_seconds, (row[0] - underwater_start).total_seconds())
        else:
            underwater_start = None
        base_to_jpy = ((row[4] + row[8]) / 2.0) * q
        long_units = original_units if original_open and side == "LONG" else 0.0
        short_units = original_units if original_open and side == "SHORT" else 0.0
        if hedge is not None:
            if hedge["side"] == "LONG": long_units += hedge["units"]
            else: short_units += hedge["units"]
        broker_units = abs(long_units - short_units) if account_mode == "NETTING" else max(long_units, short_units)
        margin = broker_units * base_to_jpy * MARGIN_RATE
        gross_margin = (long_units + short_units) * base_to_jpy * MARGIN_RATE
        max_margin, max_gross_margin = max(max_margin, margin), max(max_gross_margin, gross_margin)
        min_margin_excess = min(min_margin_excess, equity - margin)
        max_gross_exposure_units = max(max_gross_exposure_units, long_units + short_units)
        max_net_exposure_units = max(max_net_exposure_units, abs(long_units - short_units))
        if margin > 0.0 and equity <= margin:
            if original_open:
                net, slip, spr = close_leg(side, entry, original_units, row)
                original_realized += net; costs += slip; spread_cost += spr; original_open = False
                turnover_notional += executable(row, side, "EXIT") * original_units * q
            if hedge is not None:
                net, slip, spr = close_leg(hedge["side"], hedge["entry"], hedge["units"], row)
                turnover_notional += executable(row, hedge["side"], "EXIT") * hedge["units"] * q
                hedge_realized += net; costs += slip; spread_cost += spr; hedge = None
            forced = True
            break

        # Original exits.  Hard SL is comparison-only; every other ordinary
        # original exit is profit/BE or combined-BE for H5.
        favorable = directional(side, entry, executable(row, side, "EXIT"), 1.0, 1.0)
        adverse = -favorable
        if original_open and arm == "A_HARD_SL_BASELINE":
            if adverse >= atr or favorable >= 0.5 * atr:
                net, slip, spr = close_leg(side, entry, original_units, row)
                turnover_notional += executable(row, side, "EXIT") * original_units * q
                original_realized += net; costs += slip; spread_cost += spr; original_open = False
        elif original_open and arm == "E_NO_SL_PARTIAL_PROFIT_BE":
            if not partial_done and favorable >= 0.35 * atr:
                units = original_units / 2.0
                net, slip, spr = close_leg(side, entry, units, row)
                turnover_notional += executable(row, side, "EXIT") * units * q
                original_realized += net; costs += slip; spread_cost += spr
                original_units -= units; partial_done = True
            elif partial_done and favorable >= 0.0:
                net, slip, spr = close_leg(side, entry, original_units, row)
                turnover_notional += executable(row, side, "EXIT") * original_units * q
                original_realized += net; costs += slip; spread_cost += spr; original_open = False
        elif original_open:
            tp = 0.35 if arm == "C_NO_SL_DELAYED_ENTRY_EARLY_TP" else 0.5
            be_unwind = is_hedge_arm and hedge_count > 0 and favorable >= 0.0
            if favorable >= tp * atr or be_unwind:
                net, slip, spr = close_leg(side, entry, original_units, row)
                turnover_notional += executable(row, side, "EXIT") * original_units * q
                original_realized += net; costs += slip; spread_cost += spr; original_open = False
                recovery_seconds = (row[0] - entry_row[0]).total_seconds()

        if not is_hedge_arm:
            if not original_open:
                break
            continue

        # Hedge entry is evaluated only on a completed five-minute bar.
        if original_open and hedge is None and row[0].minute % 5 == 4 and adverse >= trigger * atr:
            cooldown_ok = last_rehedge_time is None or (row[0] - last_rehedge_time).total_seconds() >= 3600
            max_count = 2 if arm.startswith("H3_") else 1
            next_bar_ok = index + 1 < len(rows) and (rows[index + 1][0] - row[0]).total_seconds() <= 120
            if cooldown_ok and hedge_count < max_count and next_bar_ok:
                hedge_side = "SHORT" if side == "LONG" else "LONG"
                if account_mode == "NETTING":
                    net, slip, spr = close_leg(side, entry, original_units, rows[index + 1])
                    turnover_notional += executable(rows[index + 1], side, "EXIT") * original_units * q
                    original_realized += net; costs += slip; spread_cost += spr
                    original_open = False; original_units = 0.0; netting_reductions += 1
                else:
                    hedge_row = rows[index + 1]
                    hedge = {"side": hedge_side, "entry": executable(hedge_row, hedge_side, "ENTRY"), "units": float(UNITS)}
                    turnover_notional += hedge["entry"] * hedge["units"] * q
                    slip = spread_close(hedge_row) * SLIPPAGE_SPREAD_FRACTION * UNITS * q
                    cash_delta -= slip; costs += slip; spread_cost += spread_close(hedge_row) * UNITS * q / 2.0
                    hedge_count += 1; hedge_partial_done = False; last_rehedge_time = hedge_row[0]
                    # The completed bar at `row` only authorizes an order at
                    # the next bar.  Exit logic must not inspect the earlier
                    # bar as though the hedge already existed.
                    continue
        if hedge is None:
            if not original_open:
                break
            continue
        hedge_favorable = directional(hedge["side"], hedge["entry"], executable(row, hedge["side"], "EXIT"), 1.0, 1.0)
        close_hedge = False
        if arm.startswith("H1_"):
            close_hedge = not original_open
        elif arm.startswith(("D_", "H2_", "H5_")):
            close_hedge = hedge_favorable >= 0.5 * atr
        elif arm.startswith("H3_"):
            if not hedge_partial_done and hedge_favorable >= 0.25 * atr:
                units = hedge["units"] / 2.0
                net, slip, spr = close_leg(hedge["side"], hedge["entry"], units, row)
                turnover_notional += executable(row, hedge["side"], "EXIT") * units * q
                hedge_realized += net; costs += slip; spread_cost += spr
                hedge["units"] -= units; hedge_partial_done = True
            close_hedge = hedge_favorable >= 0.5 * atr
        elif arm.startswith("H4_") and row[0].minute % 5 == 4:
            current_m5 = mid_close(row)
            resumed_original = current_m5 > previous_m5 if side == "LONG" else current_m5 < previous_m5
            m5_reversal = m5_reversal + 1 if resumed_original else 0
            previous_m5 = current_m5
            close_hedge = m5_reversal >= 2
        if close_hedge and hedge is not None:
            net, slip, spr = close_leg(hedge["side"], hedge["entry"], hedge["units"], row)
            turnover_notional += executable(row, hedge["side"], "EXIT") * hedge["units"] * q
            hedge_realized += net; costs += slip; spread_cost += spr; hedge = None
        if arm.startswith("H5_") and original_open:
            preview_net, preview_slip, preview_spread = close_leg(side, entry, original_units, row)
            # The decision includes the unwind slippage; a visual/mark-price BE
            # that becomes a loss at the executable quote is not eligible.
            if hedge_realized + original_realized + preview_net + cash_delta >= 0.0:
                turnover_notional += executable(row, side, "EXIT") * original_units * q
                original_realized += preview_net; costs += preview_slip; spread_cost += preview_spread
                original_open = False
                recovery_seconds = (row[0] - entry_row[0]).total_seconds()
        if not original_open and hedge is None:
            break

    original_mtm = directional(side, entry, executable(last_row, side, "EXIT"), original_units, q) if original_open else 0.0
    hedge_mtm = directional(hedge["side"], hedge["entry"], executable(last_row, hedge["side"], "EXIT"), hedge["units"], q) if hedge else 0.0
    contribution = cash_delta + original_realized + hedge_realized + original_mtm + hedge_mtm
    crosses_rollover = (last_row[0].date() > entry_row[0].date()) and (original_open or hedge is not None or (last_row[0] - entry_row[0]).total_seconds() > 13 * 3600)
    status = "MARGIN_CLOSEOUT_FAILURE" if forced else "NOT_EVALUABLE_FINANCING" if crosses_rollover else "EVALUABLE"
    return {
        **base,
        "status": status,
        "executed": True,
        "side": side,
        "entry_utc": iso(entry_row[0]),
        "terminal_utc": iso(last_row[0]),
        "atr_price": atr,
        "quote_to_jpy": q,
        "original_realized_jpy": original_realized,
        "hedge_realized_jpy": hedge_realized,
        "entry_cash_delta_jpy": cash_delta,
        "original_mtm_jpy": original_mtm,
        "hedge_mtm_jpy": hedge_mtm,
        "terminal_contribution_jpy": contribution,
        "max_open_risk_jpy": max_open_risk,
        "max_equity_drawdown_jpy": max_equity_dd,
        "max_underwater_seconds": max_underwater_seconds,
        "recovery_seconds": recovery_seconds,
        "original_open": original_open,
        "hedge_open": hedge is not None,
        "margin_closeout": forced,
        "peak_broker_margin_jpy": max_margin,
        "peak_double_gross_margin_jpy": max_gross_margin,
        "minimum_margin_excess_jpy": min_margin_excess,
        "max_gross_exposure_units": max_gross_exposure_units,
        "max_net_exposure_units": max_net_exposure_units,
        "repeated_hedge_count": hedge_count,
        "netting_reduction_events": netting_reductions,
        "estimated_spread_cost_jpy": spread_cost,
        "slippage_stress_jpy": costs,
        "turnover_notional_jpy": turnover_notional,
        "financing_jpy": None if crosses_rollover else 0.0,
        "profit_only_original_close": (not original_open and original_realized >= 0.0 and not forced),
    }


def stress_row(source: dict[str, Any], arm: str) -> dict[str, Any]:
    row = dict(source)
    row["arm"] = arm
    if not row.get("executed"):
        return row
    q = float(row["quote_to_jpy"])
    atr_proxy = float(row["atr_price"])
    if arm == "H6_PERSISTENT_TREND_STRESS":
        penalty = 0.25 * 10 * atr_proxy * UNITS * q
        financing = None
    else:
        notional = max(float(row.get("peak_double_gross_margin_jpy") or 0.0) / MARGIN_RATE, UNITS * q)
        penalty = 1.5 * atr_proxy * UNITS * q + 2.0 * float(row.get("estimated_spread_cost_jpy") or 0.0)
        financing = notional * 0.0002 * 10.0 * max(1, int(row.get("repeated_hedge_count") or 1))
        penalty += financing
    row["terminal_contribution_jpy"] = float(row.get("terminal_contribution_jpy") or 0.0) - penalty
    row["original_mtm_jpy"] = float(row.get("original_mtm_jpy") or 0.0) - penalty
    row["max_open_risk_jpy"] = max(float(row.get("max_open_risk_jpy") or 0.0), -float(row["original_mtm_jpy"]))
    row["stress_penalty_jpy"] = penalty
    row["financing_jpy"] = -financing if financing is not None else None
    row["status"] = "REJECT_STRESS_UNRESOLVED_TAIL"
    row["margin_closeout"] = bool(row.get("margin_closeout")) or START_EQUITY + row["terminal_contribution_jpy"] <= float(row.get("peak_broker_margin_jpy") or 0.0)
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [row for row in rows if row.get("executed")]
    contributions = [float(row["terminal_contribution_jpy"]) for row in executed]
    gains = sum(value for value in contributions if value > 0)
    losses = -sum(value for value in contributions if value < 0)
    curve, equity, high, dd = [], START_EQUITY, START_EQUITY, 0.0
    for row in sorted(executed, key=lambda item: item.get("entry_utc") or ""):
        equity += float(row["terminal_contribution_jpy"])
        high = max(high, equity); dd = max(dd, high - equity)
        curve.append({"decision_id": row["decision_id"], "equity_jpy": equity})
    recovered = [row for row in executed if row.get("recovery_seconds") is not None]
    closes = [row for row in executed if not row.get("original_open") and not row.get("margin_closeout")]
    costs = sum(float(row.get("estimated_spread_cost_jpy") or 0.0) + float(row.get("slippage_stress_jpy") or 0.0) for row in executed)
    unknown_financing = sum(row.get("financing_jpy") is None for row in executed)
    unresolved = sum(bool(row.get("original_open") or row.get("hedge_open")) for row in executed)
    closeouts = sum(bool(row.get("margin_closeout")) for row in executed)
    stress_reject = any(str(row.get("status", "")).startswith("REJECT_STRESS") for row in executed)
    reject = closeouts > 0 or unresolved > 0 or stress_reject
    accounting = "NOT_EVALUABLE_FINANCING" if unknown_financing else "EVALUABLE"
    decision = "REJECT" if reject else "NOT_EVALUABLE" if accounting != "EVALUABLE" else "DIAGNOSTIC_ONLY"
    profit_factor = gains / losses if losses else None
    return {
        "scheduled_decisions": len(rows),
        "executed": len(executed),
        "after_cost_terminal_equity_pre_financing_jpy": equity,
        "after_cost_net_pre_financing_jpy": sum(contributions),
        "profit_factor_pre_financing": profit_factor,
        "expectancy_pre_financing_jpy": mean(contributions) if contributions else 0.0,
        "max_realized_sequence_drawdown_jpy": dd,
        "max_equity_drawdown_within_trade_jpy": max((float(row.get("max_equity_drawdown_jpy") or 0.0) for row in executed), default=0.0),
        "max_open_risk_jpy": max((float(row.get("max_open_risk_jpy") or 0.0) for row in executed), default=0.0),
        "max_underwater_seconds": max((float(row.get("max_underwater_seconds") or 0.0) for row in executed), default=0.0),
        "margin_closeout_count": closeouts,
        "margin_closeout_rate": closeouts / len(executed) if executed else 0.0,
        "peak_broker_margin_jpy": max((float(row.get("peak_broker_margin_jpy") or 0.0) for row in executed), default=0.0),
        "peak_double_gross_margin_jpy": max((float(row.get("peak_double_gross_margin_jpy") or 0.0) for row in executed), default=0.0),
        "minimum_margin_excess_jpy": min((float(row.get("minimum_margin_excess_jpy") or 0.0) for row in executed), default=START_EQUITY),
        "max_gross_exposure_units": max((float(row.get("max_gross_exposure_units") or 0.0) for row in executed), default=0.0),
        "max_net_exposure_units": max((float(row.get("max_net_exposure_units") or 0.0) for row in executed), default=0.0),
        "recovery_rate": len(recovered) / len(executed) if executed else 0.0,
        "median_recovery_seconds": median([float(row["recovery_seconds"]) for row in recovered]) if recovered else None,
        "max_recovery_seconds": max((float(row["recovery_seconds"]) for row in recovered), default=None),
        "median_holding_seconds": median([
            (parse_time(str(row["terminal_utc"])) - parse_time(str(row["entry_utc"]))).total_seconds()
            for row in executed
        ]) if executed else None,
        "unrecovered_rate": unresolved / len(executed) if executed else 0.0,
        "unresolved_inventory_count": unresolved,
        "profit_only_original_close_ratio": sum(bool(row.get("profit_only_original_close")) for row in closes) / len(closes) if closes else 0.0,
        "repeated_hedge_count": sum(int(row.get("repeated_hedge_count") or 0) for row in executed),
        "netting_reduction_events": sum(int(row.get("netting_reduction_events") or 0) for row in executed),
        "estimated_spread_plus_slippage_jpy": costs,
        "turnover_notional_jpy": sum(float(row.get("turnover_notional_jpy") or 0.0) for row in executed),
        "original_realized_jpy": sum(float(row.get("original_realized_jpy") or 0.0) for row in executed),
        "hedge_realized_jpy": sum(float(row.get("hedge_realized_jpy") or 0.0) for row in executed),
        "original_terminal_mtm_jpy": sum(float(row.get("original_mtm_jpy") or 0.0) for row in executed),
        "hedge_terminal_mtm_jpy": sum(float(row.get("hedge_mtm_jpy") or 0.0) for row in executed),
        "cost_ratio_to_gross_profit": costs / gains if gains > 0 else None,
        "unknown_financing_count": unknown_financing,
        "accounting_status": accounting,
        "decision": decision,
        "opportunities_per_observed_weekday": len(executed) / 200.0,
        "projected_200_trade_net_pre_financing_jpy": mean(contributions) * 200 if contributions else 0.0,
        "equity_curve": curve,
    }


def apply_admission_limits(results: list[dict[str, Any]]) -> None:
    """Apply preregistered no-chase inventory admission without hindsight."""
    for arm, scope in (("B_NO_SL_NAKED_RETURN_WAIT", "GLOBAL"), ("F_NO_SL_MULTI_PAIR_ROTATION", "PAIR")):
        for mode in ("HEDGING", "NETTING"):
            candidates = sorted(
                (row for row in results if row["arm"] == arm and row["account_mode"] == mode and row.get("executed")),
                key=lambda row: row["entry_utc"],
            )
            busy_until: dict[str, datetime] = {}
            for row in candidates:
                key = "ALL" if scope == "GLOBAL" else str(row["pair"])
                entry_time = parse_time(str(row["entry_utc"]))
                if entry_time < busy_until.get(key, datetime.min.replace(tzinfo=timezone.utc)):
                    identity = {field: row[field] for field in ("decision_id", "pair", "arm", "account_mode")}
                    row.clear()
                    row.update({
                        **identity,
                        "status": "SKIP_INVENTORY_BUSY_NO_CHASE",
                        "executed": False,
                        "terminal_contribution_jpy": 0.0,
                    })
                    continue
                busy_until[key] = parse_time(str(row["terminal_utc"]))


def run(sources: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_manifest = {pair: {"path": str(path), "sha256": sha256_file(path)} for pair, path in sources.items()}
    usd_rows = load_rows(sources["USD_JPY"])
    usd_by_time = {row[0]: (row[4] + row[8]) / 2.0 for row in usd_rows}
    schedule = scheduled_dates()
    decisions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for pair in PAIRS:
        rows = usd_rows if pair == "USD_JPY" else load_rows(sources[pair])
        by_time = {row[0]: index for index, row in enumerate(rows)}
        for day in schedule[pair]:
            scan = datetime.combine(day, time(8, 0), tzinfo=timezone.utc)
            q = 1.0 if pair.endswith("_JPY") else usd_by_time.get(scan)
            if q is None:
                decision = {"decision_id": f"NOFLC-{day.isoformat()}-{pair}", "pair": pair, "scan_utc": iso(scan), "status": "SKIP_QUOTE_TO_JPY_MISSING"}
            else:
                decision = build_decision(pair, day, rows, by_time, q)
            decisions.append({k: v for k, v in decision.items() if k != "entry_index"})
            base_rows: dict[tuple[str, str], dict[str, Any]] = {}
            for mode in ("HEDGING", "NETTING"):
                for arm in arm_names():
                    if arm.startswith(("H6_", "H7_")):
                        continue
                    row = simulate(decision, rows, arm, mode)
                    base_rows[(mode, arm)] = row
                    results.append(row)
                h3 = base_rows[(mode, "H3_HEDGE_PARTIAL_TP_REHEDGE_050")]
                results.append(stress_row(h3, "H6_PERSISTENT_TREND_STRESS"))
                results.append(stress_row(h3, "H7_GAP_AND_FINANCING_STRESS"))
        if pair != "USD_JPY":
            del rows
    decisions.sort(key=lambda row: row["scan_utc"])
    apply_admission_limits(results)
    results.sort(key=lambda row: (row.get("entry_utc") or row["decision_id"], row["account_mode"], row["arm"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[f"{row['arm']}::{row['account_mode']}"] .append(row)
    comparisons = {key: aggregate(value) for key, value in sorted(grouped.items())}
    report = {
        "contract": "NO_FORCED_LOSS_CLOSE_COMPARISON_V1",
        "status": "RESEARCH_ONLY",
        "source_manifest": source_manifest,
        "same_decision_ids": True,
        "scheduled_decisions": len(decisions),
        "confirmed_decisions": sum(row["status"] == "CONFIRMED" for row in decisions),
        "account_modes": ["HEDGING", "NETTING"],
        "financing_missing_policy": "NOT_EVALUABLE_NEVER_ZERO_IMPUTED",
        "standard_replay_status": "BLOCKED_MISSING_REPOSITORY_ENTRYPOINT",
        "portfolio_accounting_boundary": "B enforces one global original; F enforces one original per pair. Cross-pair concurrent margin is not jointly simulated, so F margin metrics are NOT_EVALUABLE_PORTFOLIO_MARGIN.",
        "comparisons": comparisons,
    }
    return {"contract": "NO_FORCED_LOSS_CLOSE_COHORT_V1", "decisions": decisions, "source_manifest": source_manifest}, results, report


def write_outputs(cohort: dict[str, Any], rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    (ROOT / "frozen_cohort_v1.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "decision_results_v1.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (ROOT / "comparison_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-json")
    args = parser.parse_args()
    sources = {pair: Path(path) for pair, path in SOURCE_DEFAULTS.items()}
    if args.sources_json:
        sources = {pair: Path(path) for pair, path in json.loads(Path(args.sources_json).read_text()).items()}
    missing = [f"{pair}:{path}" for pair, path in sources.items() if not path.is_file()]
    if missing:
        raise SystemExit("missing source: " + ", ".join(missing))
    cohort, rows, report = run(sources)
    write_outputs(cohort, rows, report)
    print(json.dumps({"decisions": report["scheduled_decisions"], "confirmed": report["confirmed_decisions"], "rows": len(rows), "outputs": ["frozen_cohort_v1.json", "decision_results_v1.jsonl", "comparison_v1.json"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
