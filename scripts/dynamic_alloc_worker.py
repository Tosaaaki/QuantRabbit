#!/usr/bin/env python
"""
Compute simple strategy scores from recent trades and emit config/dynamic_alloc.json.

Intended to run periodically (cron/systemd timer) to feed score-driven allocation context
confidence trims and pocket caps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.strategy_tags import extract_strategy_tags, resolve_strategy_tag
from workers.common.setup_context import extract_setup_identity

BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_DB = BASE_DIR / "logs" / "trades.db"
OUTPUT_PATH = BASE_DIR / "config" / "dynamic_alloc.json"


def normalize_strategy_key(raw: str | None) -> str:
    key = resolve_strategy_tag(raw)
    return key or "unknown"


def _strategy_key_from_row(row: Tuple) -> str:
    raw_key = row[0] if len(row) > 0 else None
    if len(row) >= 10:
        raw_key, _canonical_key = extract_strategy_tags(
            strategy_tag=row[7],
            strategy=row[8],
            entry_thesis=row[9],
        )
        if not raw_key:
            raw_key = row[0] if len(row) > 0 else None
    return normalize_strategy_key(str(raw_key or "").strip())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _default_setup_min_trades(_min_trades: int) -> int:
    return 4


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_json_loads(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _setup_context_from_row(row: Tuple) -> dict[str, str]:
    entry_thesis = _safe_json_loads(row[9] if len(row) > 9 else None)
    units = int(_safe_float(row[6] if len(row) > 6 else 0.0, 0.0))
    return extract_setup_identity(entry_thesis, units=units)


def _setup_match_dimension(context: dict[str, str]) -> str:
    if str(context.get("setup_fingerprint") or "").strip():
        return "setup_fingerprint"
    if str(context.get("flow_regime") or "").strip() and str(
        context.get("microstructure_bucket") or ""
    ).strip():
        return "flow_micro"
    if str(context.get("flow_regime") or "").strip():
        return "flow_regime"
    if str(context.get("microstructure_bucket") or "").strip():
        return "microstructure_bucket"
    return ""


def _empty_perf_bucket() -> Dict[str, float]:
    return {
        "wins": 0,
        "losses": 0,
        "trades": 0,
        "sum_pips": 0.0,
        "sum_realized_jpy": 0.0,
        "win_pips": 0.0,
        "loss_pips": 0.0,
        "w_trades": 0.0,
        "w_wins": 0.0,
        "w_sum_pips": 0.0,
        "w_sum_realized_jpy": 0.0,
        "w_win_pips": 0.0,
        "w_loss_pips": 0.0,
        "w_win_realized_jpy": 0.0,
        "w_loss_realized_jpy": 0.0,
        "w_abs_units": 0.0,
        "w_sl_like": 0.0,
        "w_margin_closeout": 0.0,
        "w_market_close": 0.0,
        "w_market_close_loss": 0.0,
        "w_market_close_loss_pips": 0.0,
        "w_market_close_loss_realized_jpy": 0.0,
    }


def _update_perf_bucket(
    bucket: Dict[str, float],
    *,
    pl: float,
    weight: float,
    realized_jpy: float,
    abs_units: float,
    close_reason: str,
) -> None:
    sl_like = close_reason in {"STOP_LOSS_ORDER", "MARKET_ORDER_MARGIN_CLOSEOUT"}
    margin_closeout = close_reason == "MARKET_ORDER_MARGIN_CLOSEOUT"
    market_close = close_reason == "MARKET_ORDER_TRADE_CLOSE"

    bucket["sum_pips"] += pl
    bucket["sum_realized_jpy"] += realized_jpy
    bucket["trades"] += 1
    if pl > 0:
        bucket["wins"] += 1
        bucket["win_pips"] += pl
    elif pl < 0:
        bucket["losses"] += 1
        bucket["loss_pips"] += abs(pl)
    bucket["w_trades"] += weight
    bucket["w_sum_pips"] += pl * weight
    bucket["w_sum_realized_jpy"] += realized_jpy * weight
    bucket["w_abs_units"] += abs_units * weight
    if pl > 0:
        bucket["w_wins"] += weight
        bucket["w_win_pips"] += pl * weight
        if realized_jpy > 0.0:
            bucket["w_win_realized_jpy"] += realized_jpy * weight
    elif pl < 0:
        bucket["w_loss_pips"] += abs(pl) * weight
        if realized_jpy < 0.0:
            bucket["w_loss_realized_jpy"] += abs(realized_jpy) * weight
    if sl_like:
        bucket["w_sl_like"] += weight
    if margin_closeout:
        bucket["w_margin_closeout"] += weight
    if market_close:
        bucket["w_market_close"] += weight
        if pl < 0.0 or realized_jpy < 0.0:
            bucket["w_market_close_loss"] += weight
            if pl < 0.0:
                bucket["w_market_close_loss_pips"] += abs(pl) * weight
            if realized_jpy < 0.0:
                bucket["w_market_close_loss_realized_jpy"] += abs(realized_jpy) * weight


def _compute_perf_snapshot(stats: Dict[str, float], *, pf_cap: float) -> Dict[str, float]:
    wins = int(stats["wins"])
    trades = int(stats["trades"])
    sum_pips = float(stats["sum_pips"])
    sum_realized_jpy = float(stats["sum_realized_jpy"])
    w_trades = float(stats.get("w_trades", 0.0))
    w_wins = float(stats.get("w_wins", 0.0))
    w_sum_pips = float(stats.get("w_sum_pips", 0.0))
    w_sum_realized_jpy = float(stats.get("w_sum_realized_jpy", 0.0))
    w_win_pips = float(stats.get("w_win_pips", 0.0))
    w_loss_pips = float(stats.get("w_loss_pips", 0.0))
    w_win_realized_jpy = float(stats.get("w_win_realized_jpy", 0.0))
    w_loss_realized_jpy = float(stats.get("w_loss_realized_jpy", 0.0))
    w_abs_units = float(stats.get("w_abs_units", 0.0))
    w_sl_like = float(stats.get("w_sl_like", 0.0))
    w_margin_closeout = float(stats.get("w_margin_closeout", 0.0))
    w_market_close = float(stats.get("w_market_close", 0.0))
    w_market_close_loss = float(stats.get("w_market_close_loss", 0.0))
    w_market_close_loss_pips = float(stats.get("w_market_close_loss_pips", 0.0))
    w_market_close_loss_realized_jpy = float(stats.get("w_market_close_loss_realized_jpy", 0.0))

    wr = wins / max(1, trades)
    weighted_wr = w_wins / max(1e-9, w_trades)
    avg_pl = w_sum_pips / max(1e-9, w_trades)
    avg_realized_jpy = w_sum_realized_jpy / max(1e-9, w_trades)
    realized_jpy_per_1k_units = 1000.0 * w_sum_realized_jpy / max(1.0, w_abs_units)
    pf = w_win_pips / w_loss_pips if w_loss_pips > 0 else (w_win_pips if w_win_pips > 0 else 0.0)
    pf = min(pf, max(0.1, pf_cap))
    jpy_pf = (
        w_win_realized_jpy / w_loss_realized_jpy
        if w_loss_realized_jpy > 0
        else (w_win_realized_jpy if w_win_realized_jpy > 0 else 0.0)
    )
    jpy_pf = min(jpy_pf, max(0.1, pf_cap))
    downside_share = w_loss_pips / max(1e-9, w_win_pips + w_loss_pips)
    jpy_downside_share = w_loss_realized_jpy / max(1e-9, w_win_realized_jpy + w_loss_realized_jpy)
    sl_rate = w_sl_like / max(1e-9, w_trades)
    margin_closeout_rate = w_margin_closeout / max(1e-9, w_trades)
    market_close_rate = w_market_close / max(1e-9, w_trades)
    market_close_loss_rate = w_market_close_loss / max(1e-9, w_trades)
    market_close_loss_pips_share = w_market_close_loss_pips / max(1e-9, w_loss_pips)
    market_close_loss_jpy_share = w_market_close_loss_realized_jpy / max(1e-9, w_loss_realized_jpy)
    market_close_loss_share = max(market_close_loss_pips_share, market_close_loss_jpy_share)

    wr_norm = _clamp((weighted_wr - 0.42) / 0.26, 0.0, 1.0)
    pf_norm = _clamp((pf - 0.60) / max(0.10, (pf_cap - 0.60)), 0.0, 1.0)
    jpy_pf_norm = _clamp((jpy_pf - 0.60) / max(0.10, (pf_cap - 0.60)), 0.0, 1.0)
    avg_norm = _clamp((avg_pl + 1.80) / 4.20, 0.0, 1.0)
    jpy_avg_norm = _clamp((realized_jpy_per_1k_units + 8.0) / 16.0, 0.0, 1.0)
    downside_penalty = _clamp((downside_share - 0.50) / 0.35, 0.0, 1.0)
    jpy_downside_penalty = _clamp((jpy_downside_share - 0.50) / 0.35, 0.0, 1.0)
    sl_penalty = _clamp((sl_rate - 0.38) / 0.40, 0.0, 1.0)
    margin_closeout_penalty = _clamp((margin_closeout_rate - 0.01) / 0.10, 0.0, 1.0)
    market_close_penalty = _clamp((market_close_rate - 0.45) / 0.35, 0.0, 1.0)
    market_close_loss_penalty = _clamp((market_close_loss_share - 0.40) / 0.30, 0.0, 1.0)

    base_score = (
        0.28 * pf_norm
        + 0.25 * wr_norm
        + 0.16 * avg_norm
        + 0.14 * jpy_pf_norm
        + 0.17 * jpy_avg_norm
        - 0.18 * downside_penalty
        - 0.15 * jpy_downside_penalty
        - 0.16 * sl_penalty
        - 0.22 * margin_closeout_penalty
        - 0.08 * market_close_penalty
        - 0.18 * market_close_loss_penalty
    )
    return {
        "wins": wins,
        "trades": trades,
        "sum_pips": sum_pips,
        "sum_realized_jpy": sum_realized_jpy,
        "wr": wr,
        "weighted_wr": weighted_wr,
        "avg_pl": avg_pl,
        "avg_realized_jpy": avg_realized_jpy,
        "realized_jpy_per_1k_units": realized_jpy_per_1k_units,
        "pf": pf,
        "jpy_pf": jpy_pf,
        "downside_share": downside_share,
        "jpy_downside_share": jpy_downside_share,
        "sl_rate": sl_rate,
        "margin_closeout_rate": margin_closeout_rate,
        "market_close_rate": market_close_rate,
        "market_close_loss_rate": market_close_loss_rate,
        "market_close_loss_share": market_close_loss_share,
        "base_score": _clamp(base_score, 0.0, 1.0),
    }


def _compute_pocket_profiles(
    pocket_stats: Dict[str, Dict[str, float]],
    *,
    min_trades: int,
    pf_cap: float,
) -> Dict[str, Dict[str, float]]:
    profiles: Dict[str, Dict[str, float]] = {}
    for pocket, stats in pocket_stats.items():
        snapshot = _compute_perf_snapshot(stats, pf_cap=pf_cap)
        trades = int(snapshot["trades"])
        base_score = float(snapshot["base_score"])
        sample_scale = min(1.0, trades / max(1, min_trades))
        score = _clamp(base_score * (0.55 + 0.45 * sample_scale), 0.0, 1.0)
        pocket_mult = 0.82 + 0.46 * score

        if snapshot["pf"] < 1.0:
            pocket_mult = min(pocket_mult, 0.96)
        if snapshot["pf"] < 0.85:
            pocket_mult = min(pocket_mult, 0.88)
        if snapshot["pf"] < 0.70:
            pocket_mult = min(pocket_mult, 0.80)
        if snapshot["sum_realized_jpy"] <= -2500.0 and snapshot["pf"] < 0.80:
            pocket_mult = min(pocket_mult, 0.76)
        if snapshot["sum_realized_jpy"] <= -5000.0 and snapshot["pf"] < 0.65:
            pocket_mult = min(pocket_mult, 0.68)
        if (
            snapshot["sum_realized_jpy"] > 0.0
            and snapshot["pf"] >= 1.05
            and snapshot["realized_jpy_per_1k_units"] >= 1.0
        ):
            pocket_mult = max(pocket_mult, 1.05)
        if (
            snapshot["sum_realized_jpy"] >= 800.0
            and snapshot["pf"] >= 1.15
            and snapshot["realized_jpy_per_1k_units"] >= 2.0
        ):
            pocket_mult = max(pocket_mult, 1.14)
        if (
            snapshot["sum_realized_jpy"] >= 1500.0
            and snapshot["pf"] >= 1.25
            and snapshot["weighted_wr"] >= 0.52
        ):
            pocket_mult = max(pocket_mult, 1.20)

        profiles[pocket] = {
            "score": round(score, 3),
            "lot_multiplier": round(_clamp(pocket_mult, 0.60, 1.25), 3),
            "trades": trades,
            "pf": round(float(snapshot["pf"]), 3),
            "win_rate": round(float(snapshot["wr"]), 3),
            "weighted_win_rate": round(float(snapshot["weighted_wr"]), 3),
            "sum_realized_jpy": round(float(snapshot["sum_realized_jpy"]), 2),
            "sum_pips": round(float(snapshot["sum_pips"]), 2),
            "realized_jpy_per_1k_units": round(float(snapshot["realized_jpy_per_1k_units"]), 3),
            "market_close_loss_share": round(float(snapshot["market_close_loss_share"]), 3),
        }
    return profiles


def _score_snapshot_to_record(
    snapshot: Dict[str, float],
    *,
    pocket: str,
    pocket_profile: Dict[str, float],
    min_trades: int,
    min_lot_multiplier: float,
    max_lot_multiplier: float,
    allow_loser_block: bool,
    allow_winner_only: bool,
) -> Dict[str, Any]:
    wins = int(snapshot["wins"])
    trades = int(snapshot["trades"])
    sum_pips = float(snapshot["sum_pips"])
    sum_realized_jpy = float(snapshot["sum_realized_jpy"])
    wr = float(snapshot["wr"])
    weighted_wr = float(snapshot["weighted_wr"])
    avg_pl = float(snapshot["avg_pl"])
    avg_realized_jpy = float(snapshot["avg_realized_jpy"])
    realized_jpy_per_1k_units = float(snapshot["realized_jpy_per_1k_units"])
    pf = float(snapshot["pf"])
    jpy_pf = float(snapshot["jpy_pf"])
    downside_share = float(snapshot["downside_share"])
    jpy_downside_share = float(snapshot["jpy_downside_share"])
    sl_rate = float(snapshot["sl_rate"])
    margin_closeout_rate = float(snapshot["margin_closeout_rate"])
    market_close_rate = float(snapshot["market_close_rate"])
    market_close_loss_rate = float(snapshot["market_close_loss_rate"])
    market_close_loss_share = float(snapshot["market_close_loss_share"])
    base_score = float(snapshot["base_score"])
    sample_scale = min(1.0, trades / max(1, min_trades))
    score = _clamp(base_score * (0.55 + 0.45 * sample_scale), 0.0, 1.0)
    pocket_lot_multiplier = float(pocket_profile.get("lot_multiplier", 1.0) or 1.0)

    min_mult = _clamp(float(min_lot_multiplier), 0.10, 1.00)
    effective_min_mult = min_mult
    max_mult = max(min_mult, float(max_lot_multiplier))
    strategy_lot_multiplier = min_mult + (max_mult - min_mult) * score

    if pf < 1.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.95)
    if pf < 0.8:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.82)
    if pf < 0.7:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.74)
    if pf < 0.6:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.68)
    if trades >= max(12, min_trades) and avg_pl <= -1.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.72)
    if trades >= max(24, min_trades * 2) and sum_pips <= -80.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.66)
    if trades >= max(12, min_trades) and sl_rate >= 0.60:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.66)
    if trades >= max(8, min_trades // 2) and sum_realized_jpy < 0.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.88)
    if trades >= max(24, min_trades) and sum_realized_jpy <= -1500.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.70)
        effective_min_mult = min(effective_min_mult, 0.30)
    if trades >= max(24, min_trades) and sum_realized_jpy <= -2500.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.60)
        effective_min_mult = min(effective_min_mult, 0.24)
    if trades >= max(32, min_trades * 2) and sum_realized_jpy <= -4000.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.55)
        effective_min_mult = min(effective_min_mult, 0.22)
    if trades >= max(40, min_trades * 2) and sum_realized_jpy <= -5000.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.50)
        effective_min_mult = min(effective_min_mult, 0.20)
    if trades >= max(12, min_trades) and realized_jpy_per_1k_units <= -8.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.62)
        effective_min_mult = min(effective_min_mult, 0.25)
    if trades >= max(12, min_trades) and realized_jpy_per_1k_units <= -5.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.58)
        effective_min_mult = min(effective_min_mult, 0.24)
    if trades >= max(10, min_trades // 2) and margin_closeout_rate >= 0.05:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.58)
        effective_min_mult = min(effective_min_mult, 0.22)
    if trades >= max(20, min_trades) and margin_closeout_rate >= 0.10:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.50)
        effective_min_mult = min(effective_min_mult, 0.18)
    if trades >= max(24, min_trades * 2) and sum_realized_jpy <= -2500.0 and pf < 0.70:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.38)
        effective_min_mult = min(effective_min_mult, 0.18)
    if (
        trades >= max(24, min_trades * 2)
        and sum_realized_jpy <= -2500.0
        and realized_jpy_per_1k_units <= -8.0
    ):
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.32)
        effective_min_mult = min(effective_min_mult, 0.15)
    if trades >= max(32, min_trades * 2) and market_close_loss_share >= 0.55 and sum_realized_jpy < 0.0:
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.28)
        effective_min_mult = min(effective_min_mult, 0.14)
    if (
        trades >= max(32, min_trades * 2)
        and sum_realized_jpy <= -4000.0
        and market_close_loss_share >= 0.55
        and jpy_downside_share >= 0.60
    ):
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.18)
        effective_min_mult = min(effective_min_mult, 0.12)
    if (
        trades >= max(40, min_trades * 3)
        and sum_realized_jpy <= -5500.0
        and market_close_loss_share >= 0.75
        and realized_jpy_per_1k_units <= -7.0
    ):
        strategy_lot_multiplier = min(strategy_lot_multiplier, 0.12)
        effective_min_mult = min(effective_min_mult, 0.10)
    if trades >= max(16, min_trades) and sum_realized_jpy <= 0.0 and avg_realized_jpy < 0.0:
        if realized_jpy_per_1k_units <= -1.0:
            strategy_lot_multiplier = min(strategy_lot_multiplier, 0.62)
            effective_min_mult = min(effective_min_mult, 0.24)
        if realized_jpy_per_1k_units <= -2.0:
            strategy_lot_multiplier = min(strategy_lot_multiplier, 0.52)
            effective_min_mult = min(effective_min_mult, 0.22)
        if realized_jpy_per_1k_units <= -4.0:
            strategy_lot_multiplier = min(strategy_lot_multiplier, 0.42)
            effective_min_mult = min(effective_min_mult, 0.20)
    if (
        trades >= max(24, min_trades * 2)
        and score >= 0.70
        and pf >= 1.30
        and jpy_pf >= 1.20
        and sum_realized_jpy >= 1000.0
        and realized_jpy_per_1k_units >= 10.0
        and margin_closeout_rate <= 0.15
    ):
        strategy_lot_multiplier = max(strategy_lot_multiplier, 0.85)
    if trades < max(1, min_trades):
        strategy_lot_multiplier = min(strategy_lot_multiplier, 1.00)
    strategy_lot_multiplier = _clamp(strategy_lot_multiplier, effective_min_mult, max_mult)

    pocket_lot_multiplier_applied = pocket_lot_multiplier
    cash_profitable = (
        sum_realized_jpy >= 0.0 and avg_realized_jpy >= 0.0 and realized_jpy_per_1k_units >= 0.0
    )
    if (
        score >= 0.55
        or (pf >= 1.10 and sum_realized_jpy >= 0.0)
        or (weighted_wr >= 0.58 and sum_pips > 0.0)
    ):
        if cash_profitable:
            pocket_lot_multiplier_applied = max(1.0, pocket_lot_multiplier_applied)
        else:
            pocket_lot_multiplier_applied = min(1.0, pocket_lot_multiplier_applied)
    elif pf < 0.95 or sum_realized_jpy < 0.0:
        if pocket_lot_multiplier_applied > 1.0:
            pocket_lot_multiplier_applied = 1.0 + (pocket_lot_multiplier_applied - 1.0) * 0.35

    lot_multiplier = strategy_lot_multiplier * pocket_lot_multiplier_applied
    lot_multiplier = _clamp(lot_multiplier, effective_min_mult, max_mult)

    return {
        "pocket": pocket,
        "score": round(score, 3),
        "lot_multiplier": round(lot_multiplier, 3),
        "strategy_lot_multiplier": round(strategy_lot_multiplier, 3),
        "pocket_lot_multiplier": round(pocket_lot_multiplier, 3),
        "pocket_lot_multiplier_applied": round(pocket_lot_multiplier_applied, 3),
        "pocket_score": round(float(pocket_profile.get("score", 0.0) or 0.0), 3),
        "effective_min_lot_multiplier": round(effective_min_mult, 3),
        "trades": trades,
        "win_rate": round(wr, 3),
        "weighted_win_rate": round(weighted_wr, 3),
        "pf": round(pf, 3),
        "jpy_pf": round(jpy_pf, 3),
        "avg_pips": round(avg_pl, 3),
        "sum_pips": round(sum_pips, 2),
        "avg_realized_jpy": round(avg_realized_jpy, 3),
        "sum_realized_jpy": round(sum_realized_jpy, 2),
        "realized_jpy_per_1k_units": round(realized_jpy_per_1k_units, 3),
        "sl_rate": round(sl_rate, 3),
        "margin_closeout_rate": round(margin_closeout_rate, 3),
        "market_close_rate": round(market_close_rate, 3),
        "market_close_loss_rate": round(market_close_loss_rate, 3),
        "market_close_loss_share": round(market_close_loss_share, 3),
        "downside_share": round(downside_share, 3),
        "jpy_downside_share": round(jpy_downside_share, 3),
        "allow_loser_block": bool(allow_loser_block),
        "allow_winner_only": bool(allow_winner_only),
    }


def _build_setup_overrides(
    *,
    strategy_record: Dict[str, Any],
    setup_stats: Dict[tuple[str, str, str, str], Dict[str, float]],
    setup_pockets: Dict[tuple[str, str, str, str], Dict[str, int]],
    pocket_profiles: Dict[str, Dict[str, float]],
    min_trades: int,
    setup_min_trades: int,
    pf_cap: float,
    min_lot_multiplier: float,
    max_lot_multiplier: float,
    allow_loser_block: bool,
    allow_winner_only: bool,
) -> list[Dict[str, Any]]:
    overrides: list[Dict[str, Any]] = []
    resolved_setup_min_trades = max(
        1,
        int(setup_min_trades or _default_setup_min_trades(min_trades)),
    )
    parent_multiplier = float(strategy_record.get("lot_multiplier", 1.0) or 1.0)
    parent_cap = min(float(max_lot_multiplier), max(1.0, parent_multiplier * 1.25))
    for key, stats in setup_stats.items():
        snapshot = _compute_perf_snapshot(stats, pf_cap=pf_cap)
        trades = int(snapshot["trades"])
        severe_low_sample_loser = bool(
            0 < trades < resolved_setup_min_trades
            and float(snapshot.get("sum_realized_jpy", 0.0) or 0.0) <= -8.0
            and float(snapshot.get("weighted_win_rate", 0.0) or 0.0) <= 0.25
            and float(snapshot.get("jpy_pf", 0.0) or 0.0) <= 0.25
        )
        if trades < resolved_setup_min_trades and not severe_low_sample_loser:
            continue
        match_dimension, setup_fingerprint, flow_regime, microstructure_bucket = key
        pocket_counts = setup_pockets.get(key, {})
        pocket = max(pocket_counts, key=pocket_counts.get) if pocket_counts else str(
            strategy_record.get("pocket") or "unknown"
        )
        setup_record = _score_snapshot_to_record(
            snapshot,
            pocket=pocket,
            pocket_profile=pocket_profiles.get(pocket, {}),
            min_trades=resolved_setup_min_trades,
            min_lot_multiplier=min_lot_multiplier,
            max_lot_multiplier=max_lot_multiplier,
            allow_loser_block=allow_loser_block,
            allow_winner_only=allow_winner_only,
        )
        setup_multiplier = _clamp(
            float(setup_record.get("lot_multiplier", parent_multiplier) or parent_multiplier),
            0.10,
            max(0.10, parent_cap),
        )
        effective_min = min(
            setup_multiplier,
            float(setup_record.get("effective_min_lot_multiplier", setup_multiplier) or setup_multiplier),
        )
        override: Dict[str, Any] = {
            "match_dimension": match_dimension,
            "setup_fingerprint": setup_fingerprint or None,
            "flow_regime": flow_regime or None,
            "microstructure_bucket": microstructure_bucket or None,
            "pocket": pocket,
            "score": round(float(setup_record.get("score", 0.0) or 0.0), 3),
            "lot_multiplier": round(setup_multiplier, 3),
            "strategy_lot_multiplier": round(
                float(setup_record.get("strategy_lot_multiplier", setup_multiplier) or setup_multiplier),
                3,
            ),
            "effective_min_lot_multiplier": round(max(0.10, effective_min), 3),
            "trades": trades,
            "win_rate": round(float(setup_record.get("win_rate", 0.0) or 0.0), 3),
            "weighted_win_rate": round(float(setup_record.get("weighted_win_rate", 0.0) or 0.0), 3),
            "pf": round(float(setup_record.get("pf", 0.0) or 0.0), 3),
            "jpy_pf": round(float(setup_record.get("jpy_pf", 0.0) or 0.0), 3),
            "avg_pips": round(float(setup_record.get("avg_pips", 0.0) or 0.0), 3),
            "sum_pips": round(float(setup_record.get("sum_pips", 0.0) or 0.0), 2),
            "avg_realized_jpy": round(float(setup_record.get("avg_realized_jpy", 0.0) or 0.0), 3),
            "sum_realized_jpy": round(float(setup_record.get("sum_realized_jpy", 0.0) or 0.0), 2),
            "realized_jpy_per_1k_units": round(
                float(setup_record.get("realized_jpy_per_1k_units", 0.0) or 0.0),
                3,
            ),
            "sl_rate": round(float(setup_record.get("sl_rate", 0.0) or 0.0), 3),
            "margin_closeout_rate": round(float(setup_record.get("margin_closeout_rate", 0.0) or 0.0), 3),
            "market_close_rate": round(float(setup_record.get("market_close_rate", 0.0) or 0.0), 3),
            "market_close_loss_rate": round(
                float(setup_record.get("market_close_loss_rate", 0.0) or 0.0),
                3,
            ),
            "market_close_loss_share": round(
                float(setup_record.get("market_close_loss_share", 0.0) or 0.0),
                3,
            ),
        }
        overrides.append(override)
    specificity_rank = {
        "setup_fingerprint": 4,
        "flow_micro": 3,
        "flow_regime": 2,
        "microstructure_bucket": 1,
    }
    overrides.sort(
        key=lambda item: (
            specificity_rank.get(str(item.get("match_dimension") or ""), 0),
            int(item.get("trades") or 0),
        ),
        reverse=True,
    )
    return overrides[:8]


def _parse_utc_timestamp(raw: object) -> dt.datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _recency_weight(close_time_raw: object, *, now_utc: dt.datetime, half_life_hours: float) -> float:
    ts = _parse_utc_timestamp(close_time_raw)
    if ts is None:
        return 1.0
    age_hours = max(0.0, (now_utc - ts).total_seconds() / 3600.0)
    half_life = max(1.0, float(half_life_hours))
    return 0.5 ** (age_hours / half_life)


def fetch_trades(limit: int, lookback_days: int) -> List[Tuple]:
    if not TRADES_DB.exists():
        return []
    # 読み取り専用で接続し、ロック影響を最小化
    uri = f"file:{TRADES_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0, isolation_level=None)
    try:
        cur = conn.cursor()
        sql = """
            SELECT
              COALESCE(NULLIF(strategy_tag, ''), strategy) AS strategy_key,
              pocket,
              pl_pips,
              close_time,
              close_reason,
              realized_pl,
              units,
              strategy_tag,
              strategy,
              entry_thesis
            FROM trades
            WHERE close_time IS NOT NULL
              AND julianday(close_time) >= julianday('now', ?)
            ORDER BY close_time DESC
        """
        params = [f"-{int(lookback_days)} day"]
        if int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def compute_scores(
    rows: List[Tuple],
    *,
    min_trades: int,
    setup_min_trades: int = 4,
    pf_cap: float,
    min_lot_multiplier: float = 0.45,
    max_lot_multiplier: float = 1.65,
    half_life_hours: float = 36.0,
    allow_loser_block: bool = False,
    allow_winner_only: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    pocket_stats: Dict[str, Dict[str, float]] = {}
    pockets: Dict[str, Dict[str, int]] = {}
    setup_stats: Dict[str, Dict[tuple[str, str, str, str], Dict[str, float]]] = {}
    setup_pockets: Dict[str, Dict[tuple[str, str, str, str], Dict[str, int]]] = {}
    now_utc = dt.datetime.now(dt.timezone.utc)
    for row in rows:
        strat = _strategy_key_from_row(row)
        pocket = row[1] if len(row) > 1 else None
        pl_pips = row[2] if len(row) > 2 else 0.0
        close_time = row[3] if len(row) > 3 else None
        close_reason = str(row[4] if len(row) > 4 else "")
        pocket = pocket or "unknown"
        key = strat
        weight = _recency_weight(close_time, now_utc=now_utc, half_life_hours=half_life_hours)
        s = stats.setdefault(key, _empty_perf_bucket())
        ps = pocket_stats.setdefault(pocket, _empty_perf_bucket())
        pockets.setdefault(key, {})
        pockets[key][pocket] = pockets[key].get(pocket, 0) + 1
        pl = float(pl_pips or 0.0)
        realized_jpy = float(row[5] or 0.0) if len(row) > 5 else pl
        abs_units = abs(float(row[6] or 0.0)) if len(row) > 6 else 1000.0
        if abs_units <= 0.0:
            abs_units = 1000.0
        _update_perf_bucket(
            s,
            pl=pl,
            weight=weight,
            realized_jpy=realized_jpy,
            abs_units=abs_units,
            close_reason=close_reason,
        )
        _update_perf_bucket(
            ps,
            pl=pl,
            weight=weight,
            realized_jpy=realized_jpy,
            abs_units=abs_units,
            close_reason=close_reason,
        )
        setup_context = _setup_context_from_row(row)
        match_dimension = _setup_match_dimension(setup_context)
        if match_dimension:
            setup_key = (
                match_dimension,
                str(setup_context.get("setup_fingerprint") or "").strip(),
                str(setup_context.get("flow_regime") or "").strip(),
                str(setup_context.get("microstructure_bucket") or "").strip(),
            )
            strategy_setup_stats = setup_stats.setdefault(key, {})
            setup_bucket = strategy_setup_stats.setdefault(setup_key, _empty_perf_bucket())
            _update_perf_bucket(
                setup_bucket,
                pl=pl,
                weight=weight,
                realized_jpy=realized_jpy,
                abs_units=abs_units,
                close_reason=close_reason,
            )
            strategy_setup_pockets = setup_pockets.setdefault(key, {})
            setup_pocket_counts = strategy_setup_pockets.setdefault(setup_key, {})
            setup_pocket_counts[pocket] = setup_pocket_counts.get(pocket, 0) + 1

    pocket_profiles = _compute_pocket_profiles(
        pocket_stats,
        min_trades=min_trades,
        pf_cap=pf_cap,
    )

    scores: Dict[str, Dict[str, Any]] = {}
    for strat, s in stats.items():
        snapshot = _compute_perf_snapshot(s, pf_cap=pf_cap)
        pocket_counts = pockets.get(strat, {})
        pocket = max(pocket_counts, key=pocket_counts.get) if pocket_counts else "unknown"
        pocket_profile = pocket_profiles.get(pocket, {})
        score_record = _score_snapshot_to_record(
            snapshot,
            pocket=pocket,
            pocket_profile=pocket_profile,
            min_trades=min_trades,
            min_lot_multiplier=min_lot_multiplier,
            max_lot_multiplier=max_lot_multiplier,
            allow_loser_block=allow_loser_block,
            allow_winner_only=allow_winner_only,
        )
        overrides = _build_setup_overrides(
            strategy_record=score_record,
            setup_stats=setup_stats.get(strat, {}),
            setup_pockets=setup_pockets.get(strat, {}),
            pocket_profiles=pocket_profiles,
            min_trades=min_trades,
            setup_min_trades=setup_min_trades,
            pf_cap=pf_cap,
            min_lot_multiplier=min_lot_multiplier,
            max_lot_multiplier=max_lot_multiplier,
            allow_loser_block=allow_loser_block,
            allow_winner_only=allow_winner_only,
        )
        if overrides:
            score_record["setup_overrides"] = overrides
        scores[strat] = score_record
    return scores, pocket_profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of recent trades to use (0 to disable LIMIT and use full lookback window)",
    )
    ap.add_argument("--lookback-days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--min-trades", type=int, default=12, help="Min trades for full score weight")
    ap.add_argument("--setup-min-trades", type=int, default=4, help="Min trades for setup-scoped override emission")
    ap.add_argument("--pf-cap", type=float, default=2.0, help="Profit factor cap for normalization")
    ap.add_argument("--target-use", type=float, default=0.88, help="Target margin usage fraction")
    ap.add_argument("--half-life-hours", type=float, default=36.0, help="Recency half-life for scoring")
    ap.add_argument("--min-lot-multiplier", type=float, default=0.45, help="Lower bound of strategy lot multiplier")
    ap.add_argument("--max-lot-multiplier", type=float, default=1.65, help="Upper bound of strategy lot multiplier")
    ap.add_argument(
        "--soft-participation",
        type=int,
        choices=(0, 1),
        default=1,
        help="If 1, disable dyn alloc hard blocks and keep all strategies in reduced size",
    )
    ap.add_argument(
        "--allow-loser-block",
        type=int,
        choices=(0, 1),
        default=0,
        help="If 1, dyn alloc may hard-block low-score strategies",
    )
    ap.add_argument(
        "--allow-winner-only",
        type=int,
        choices=(0, 1),
        default=0,
        help="If 1, workers may route entries to winner-only subset",
    )
    args = ap.parse_args()

    rows = fetch_trades(args.limit, args.lookback_days)
    soft_participation = bool(int(args.soft_participation))
    allow_loser_block = bool(int(args.allow_loser_block)) and not soft_participation
    allow_winner_only = bool(int(args.allow_winner_only)) and not soft_participation
    scores, pocket_profiles = compute_scores(
        rows,
        min_trades=args.min_trades,
        setup_min_trades=max(1, int(args.setup_min_trades)),
        pf_cap=args.pf_cap,
        min_lot_multiplier=args.min_lot_multiplier,
        max_lot_multiplier=args.max_lot_multiplier,
        half_life_hours=args.half_life_hours,
        allow_loser_block=allow_loser_block,
        allow_winner_only=allow_winner_only,
    )

    pocket_cap_weights = {
        "macro": 1.0,
        "micro": float(pocket_profiles.get("micro", {}).get("lot_multiplier", 1.0) or 1.0),
        "scalp": float(pocket_profiles.get("scalp", {}).get("lot_multiplier", 1.0) or 1.0),
        "scalp_fast": float(pocket_profiles.get("scalp_fast", {}).get("lot_multiplier", 1.0) or 1.0),
    }
    cap_total = sum(max(0.10, value) for value in pocket_cap_weights.values())
    pocket_caps = {
        pocket: round(max(0.10, value) / max(1e-9, cap_total), 3)
        for pocket, value in pocket_cap_weights.items()
    }

    alloc = {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_days": int(args.lookback_days),
        "min_trades": int(args.min_trades),
        "setup_min_trades": max(1, int(args.setup_min_trades)),
        "pf_cap": float(args.pf_cap),
        "half_life_hours": float(args.half_life_hours),
        "target_use": args.target_use,
        "pocket_caps": pocket_caps,
        "pocket_profiles": pocket_profiles,
        "allocation_policy": {
            "mode": "soft_participation" if soft_participation else "classic",
            "soft_participation": soft_participation,
            "allow_loser_block": allow_loser_block,
            "allow_winner_only": allow_winner_only,
            "min_lot_multiplier": float(args.min_lot_multiplier),
            "max_lot_multiplier": float(args.max_lot_multiplier),
        },
        "strategies": scores,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(alloc, ensure_ascii=False, indent=2))
    print(f"[dynamic_alloc] wrote {OUTPUT_PATH} with {len(scores)} strategies")


if __name__ == "__main__":
    main()
