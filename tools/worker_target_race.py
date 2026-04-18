#!/usr/bin/env python3
"""Shared target-race plan helpers for worker trades.

Worker entries carry a compact runner plan in OANDA clientExtensions comment so:
- pending LIMIT fills can still hand the plan to bot_trade_manager later
- the manager can execute TP1 partials and promote the remainder to a runner

Mutable execution state (TP1 done / runner orders updated) lives in a local JSON file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "worker_target_race_state.json"
COMMENT_PREFIX = "qrtr1"
DEFAULT_PARTIAL_FRACTION = 0.50


def price_decimals(pair: str) -> int:
    return 3 if "JPY" in pair else 5


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def round_price(price: float, pair: str) -> float:
    return round(float(price), price_decimals(pair))


def format_price_token(price: float, pair: str) -> str:
    return f"{round_price(price, pair):.{price_decimals(pair)}f}"


def distance_pips(pair: str, a: float, b: float) -> float:
    return abs((float(b) - float(a)) / pip_size(pair))


def eta_bars(distance_pips_value: float, pace_pips_per_bar: float) -> float | None:
    pace = max(float(pace_pips_per_bar or 0.0), 0.1)
    if distance_pips_value <= 0:
        return 0.0
    return round(distance_pips_value / pace, 1)


def build_plan(
    *,
    style: str,
    pair: str,
    direction: str,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    hold_boundary: float,
    pace_fast_pips: float,
    pace_slow_pips: float,
    partial_fraction: float = DEFAULT_PARTIAL_FRACTION,
) -> dict:
    entry = round_price(entry, pair)
    stop = round_price(stop, pair)
    tp1 = round_price(tp1, pair)
    tp2 = round_price(tp2, pair)
    hold_boundary = round_price(hold_boundary, pair)
    partial_fraction = max(0.1, min(0.9, float(partial_fraction)))

    return {
        "version": 1,
        "style": str(style),
        "pair": pair,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "hold_boundary": hold_boundary,
        "partial_fraction": round(partial_fraction, 2),
        "tp1_pips": round(distance_pips(pair, entry, tp1), 1),
        "tp2_pips": round(distance_pips(pair, entry, tp2), 1),
        "hold_boundary_pips": round(distance_pips(pair, entry, hold_boundary), 1),
        "eta_fast_bars": eta_bars(distance_pips(pair, entry, tp1), pace_fast_pips),
        "eta_slow_bars": eta_bars(distance_pips(pair, entry, tp2), pace_slow_pips),
    }


def encode_comment(plan: dict) -> str:
    pair = plan["pair"]
    style_token = "r" if plan.get("style") == "range" else "t"
    fraction_pct = int(round(float(plan.get("partial_fraction", DEFAULT_PARTIAL_FRACTION)) * 100))
    return (
        f"{COMMENT_PREFIX}|s={style_token}|f={fraction_pct}"
        f"|t1={format_price_token(plan['tp1'], pair)}"
        f"|t2={format_price_token(plan['tp2'], pair)}"
        f"|h={format_price_token(plan['hold_boundary'], pair)}"
    )


def decode_comment(comment: str, pair: str, direction: str) -> dict | None:
    if not comment or not comment.startswith(f"{COMMENT_PREFIX}|"):
        return None
    parts = comment.split("|")
    values: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    try:
        style = "range" if values.get("s", "r") == "r" else "trend"
        partial_fraction = float(values.get("f", "50")) / 100.0
        tp1 = float(values["t1"])
        tp2 = float(values["t2"])
        hold_boundary = float(values["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "version": 1,
        "style": style,
        "pair": pair,
        "direction": direction,
        "tp1": tp1,
        "tp2": tp2,
        "hold_boundary": hold_boundary,
        "partial_fraction": round(max(0.1, min(0.9, partial_fraction)), 2),
    }


def get_comment(payload: dict) -> str:
    for key in ("tradeClientExtensions", "clientExtensions"):
        ext = payload.get(key, {}) or {}
        comment = ext.get("comment")
        if comment:
            return str(comment)
    return ""


def plan_from_payload(payload: dict, pair: str, direction: str) -> dict | None:
    return decode_comment(get_comment(payload), pair, direction)


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp_path.replace(STATE_FILE)


def prune_state(open_trade_ids: set[str]) -> None:
    state = load_state()
    filtered = {tid: payload for tid, payload in state.items() if tid in open_trade_ids}
    if filtered != state:
        save_state(filtered)


def get_trade_state(trade_id: str) -> dict:
    return load_state().get(str(trade_id), {})


def remember_trade_plan(trade_id: str, plan: dict, current_units: int | None = None) -> dict:
    state = load_state()
    trade_id = str(trade_id)
    entry = state.get(trade_id, {})
    merged = {
        "plan": dict(plan),
        "tp1_done": bool(entry.get("tp1_done", False)),
        "runner_orders_set": bool(entry.get("runner_orders_set", False)),
        "partial_units": int(entry.get("partial_units", 0) or 0),
        "current_units": int(current_units or entry.get("current_units", 0) or 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state[trade_id] = merged
    save_state(state)
    return merged


def update_trade_state(trade_id: str, **fields) -> dict:
    state = load_state()
    trade_id = str(trade_id)
    entry = dict(state.get(trade_id, {}))
    entry.update(fields)
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    state[trade_id] = entry
    save_state(state)
    return entry


def extract_trade_id_from_order_result(result: dict) -> str | None:
    fill = result.get("orderFillTransaction", {}) or {}
    trade_opened = fill.get("tradeOpened") or {}
    trade_id = trade_opened.get("tradeID")
    if trade_id:
        return str(trade_id)
    return None
