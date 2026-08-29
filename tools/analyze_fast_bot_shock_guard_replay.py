#!/usr/bin/env python3
"""Replay the preregistered EUR/USD shock guard and protective-SL grid.

Input is existing or freshly GET-only OANDA M1 bid/ask JSONL.gz truth.  The
tool has no broker client, writes no policy, and never claims a stop is
guaranteed.  Same-minute TP/SL is scored stop-first.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_shock_guard import load_config  # noqa: E402


def _epoch(value: str) -> int:
    parsed = datetime.fromisoformat(value[:19]).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _load(paths: list[Path]) -> dict[str, np.ndarray]:
    columns = {name: array("d") for name in ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")}
    times = array("q")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("complete") is not True or row.get("granularity") != "M1":
                    continue
                bid = row["bid"]
                ask = row["ask"]
                times.append(_epoch(str(row["time"])))
                for prefix, source in (("b", bid), ("a", ask)):
                    for suffix, key in (("o", "o"), ("h", "h"), ("l", "l"), ("c", "c")):
                        columns[prefix + suffix].append(float(source[key]))
    order = np.argsort(np.frombuffer(times, dtype=np.int64), kind="stable")
    result = {"t": np.frombuffer(times, dtype=np.int64)[order]}
    for name, values in columns.items():
        result[name] = np.frombuffer(values, dtype=np.float64)[order]
    if len(result["t"]):
        unique = np.r_[True, np.diff(result["t"]) != 0]
        result = {name: values[unique] for name, values in result.items()}
    return result


def _m5_atr(data: dict[str, np.ndarray]) -> np.ndarray:
    t = data["t"]
    mid_h = (data["bh"] + data["ah"]) / 2.0
    mid_l = (data["bl"] + data["al"]) / 2.0
    mid_c = (data["bc"] + data["ac"]) / 2.0
    bucket = t // 300
    _, starts = np.unique(bucket, return_index=True)
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    highs = np.maximum.reduceat(mid_h, starts)
    lows = np.minimum.reduceat(mid_l, starts)
    closes = mid_c[ends]
    complete = (t[ends] - t[starts] == 240) & (np.diff(np.r_[t[starts], t[ends[-1]] + 300])[: len(starts)] >= 0)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.r_[closes[0], closes[:-1]]), np.abs(lows - np.r_[closes[0], closes[:-1]]))) * 10_000.0
    cumulative = np.cumsum(np.r_[0.0, tr])
    atr = np.full(len(tr), np.nan)
    for index in range(14, len(tr) + 1):
        if complete[index - 14 : index].all():
            atr[index - 1] = (cumulative[index] - cumulative[index - 14]) / 14.0
    completed_at = t[ends] + 60
    location = np.searchsorted(completed_at, t, side="right") - 1
    mapped = np.full(len(t), np.nan)
    valid = location >= 0
    mapped[valid] = atr[location[valid]]
    return mapped


def _episodes(data: dict[str, np.ndarray], config: dict[str, Any]) -> list[int]:
    t = data["t"]
    mid = (data["bc"] + data["ac"]) / 2.0
    spread = (data["ac"] - data["bc"]) * 10_000.0
    detection = config["detection"]
    window = int(detection["window_minutes"])
    velocity_window = int(detection["short_velocity_window_minutes"])
    swing_lookback = int(detection["swing_lookback_minutes"])
    spread_lookback = int(detection["spread_lookback_minutes"])
    history = window + max(velocity_window * 2, swing_lookback, spread_lookback)
    result: list[int] = []
    lock_until = -1
    for index in range(history, len(t) - 61):
        if t[index] <= lock_until or np.any(np.diff(t[index - history : index + 61]) != 60):
            continue
        signed_impulse = (mid[index] - mid[index - window]) * 10_000.0
        direction = 1.0 if signed_impulse > 0.0 else -1.0 if signed_impulse < 0.0 else 0.0
        impulse = abs(signed_impulse)
        short_velocity = (
            (mid[index] - mid[index - velocity_window])
            * 10_000.0
            * direction
            / velocity_window
        )
        prior_velocity = (
            (mid[index - velocity_window] - mid[index - velocity_window * 2])
            * 10_000.0
            * direction
            / velocity_window
        )
        acceleration = (short_velocity - prior_velocity) / velocity_window
        prior = mid[index - window - swing_lookback : index - window]
        swing_break = bool(
            (direction > 0.0 and mid[index] > np.max(prior))
            or (direction < 0.0 and mid[index] < np.min(prior))
        )
        median_spread = float(np.median(spread[index - spread_lookback : index]))
        spread_ratio = spread[index] / median_spread if median_spread > 0.0 else 0.0
        confirmations = sum(
            (
                short_velocity + 1e-9
                >= float(detection["minimum_velocity_pips_per_minute"]),
                acceleration + 1e-9
                >= float(detection["minimum_acceleration_pips_per_minute2"]),
                spread[index] >= float(detection["minimum_spread_pips"])
                and spread_ratio >= float(detection["minimum_spread_ratio"]),
                swing_break,
            )
        )
        if (
            impulse >= float(detection["minimum_impulse_pips"])
            and confirmations >= int(detection["minimum_raw_confirmation_count"])
        ):
            result.append(index)
            lock_until = int(t[index] + 3600)
    return result


def _pf(values: list[float]) -> float | None:
    wins = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    return round(wins / losses, 6) if losses > 0.0 else None


def _max_loss_streak(values: list[float]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value < 0.0 else 0
        best = max(best, current)
    return best


def _metrics(values: list[float]) -> dict[str, Any]:
    losses = [value for value in values if value < 0.0]
    return {
        "trades": len(values),
        "net_pips": round(sum(values), 6),
        "profit_factor": _pf(values),
        "hit_rate": round(sum(value > 0.0 for value in values) / len(values), 6) if values else None,
        "maximum_loss_streak": _max_loss_streak(values),
        "p05_trade_pips": round(float(np.quantile(values, 0.05)), 6) if values else None,
        "average_loss_pips": round(sum(losses) / len(losses), 6) if losses else None,
    }


def _architecture_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["pnl"]) for row in rows]
    scaled = [
        float(row["pnl"]) * float(row["normalized_unit_fraction"]) for row in rows
    ]
    base = _metrics(values)
    held = [int(row["held"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    margin = [float(row["normalized_unit_fraction"]) * int(row["held"]) for row in rows]
    scaled_mae = [
        float(row["mae"]) * float(row["normalized_unit_fraction"]) for row in rows
    ]
    return {
        **base,
        "risk_scaled_net_pip_units": round(sum(scaled), 6),
        "risk_scaled_profit_factor": _pf(scaled),
        "risk_scaled_average_loss_pip_units": (
            round(sum(value for value in scaled if value < 0.0) / sum(value < 0.0 for value in scaled), 6)
            if any(value < 0.0 for value in scaled)
            else None
        ),
        "risk_scaled_p05_pip_units": round(float(np.quantile(scaled, 0.05)), 6)
        if scaled
        else None,
        "risk_scaled_maximum_adverse_pip_units": round(max(scaled_mae), 6)
        if scaled_mae
        else None,
        "maximum_adverse_excursion_pips": round(max(mae), 6) if mae else None,
        "p95_adverse_excursion_pips": round(float(np.quantile(mae, 0.95)), 6) if mae else None,
        "average_inventory_residence_minutes": round(sum(held) / len(held), 6) if held else None,
        "maximum_inventory_residence_minutes": max(held) if held else None,
        "normalized_margin_unit_minutes": round(sum(margin) / len(margin), 6) if margin else None,
        "exit_reason_counts": dict(sorted(Counter(str(row["reason"]) for row in rows).items())),
        "gap_slippage_pips": round(sum(float(row["gap_slippage"]) for row in rows), 6),
        "gap_exit_count": sum(str(row["reason"]).endswith("_GAP") for row in rows),
    }


def _baseline_trade(data: dict[str, np.ndarray], index: int, direction: int) -> float:
    horizon = index + 60
    if direction > 0:
        return (data["bc"][horizon] - data["ac"][index]) * 10_000.0
    return (data["bc"][index] - data["ac"][horizon]) * 10_000.0


def _sl_trade(
    data: dict[str, np.ndarray],
    index: int,
    direction: int,
    width: float,
    *,
    take_profit: float = 2.4,
) -> tuple[float, str, float, int | None]:
    entry = data["ac"][index] if direction > 0 else data["bc"][index]
    stop = entry - width / 10_000.0 if direction > 0 else entry + width / 10_000.0
    target = entry + take_profit / 10_000.0 if direction > 0 else entry - take_profit / 10_000.0
    for cursor in range(index + 1, index + 61):
        if direction > 0:
            open_px, high_px, low_px = data["bo"][cursor], data["bh"][cursor], data["bl"][cursor]
            if open_px <= stop:
                realized = (open_px - entry) * 10_000.0
                return realized, "STOP_GAP", max(0.0, -width - realized), cursor
            stop_hit = low_px <= stop
            target_hit = high_px >= target
        else:
            open_px, high_px, low_px = data["ao"][cursor], data["ah"][cursor], data["al"][cursor]
            if open_px >= stop:
                realized = (entry - open_px) * 10_000.0
                return realized, "STOP_GAP", max(0.0, -width - realized), cursor
            stop_hit = high_px >= stop
            target_hit = low_px <= target
        if stop_hit:
            return -width, "STOP", 0.0, cursor
        if target_hit:
            return take_profit, "TARGET", 0.0, None
    if direction > 0:
        return (data["bc"][index + 60] - entry) * 10_000.0, "HORIZON", 0.0, None
    return (entry - data["ac"][index + 60]) * 10_000.0, "HORIZON", 0.0, None


def _catastrophe_width(
    data: dict[str, np.ndarray], atr: float, index: int, direction: int, config: dict[str, Any]
) -> float:
    policy = config["protective_stop"]
    entry = data["ac"][index] if direction > 0 else data["bc"][index]
    spread = (data["ac"][index] - data["bc"][index]) * 10_000.0
    impulse = abs(
        ((data["bc"][index] + data["ac"][index]) - (data["bc"][index - 15] + data["ac"][index - 15]))
        * 5_000.0
    )
    if direction > 0:
        swing = float(np.min(data["bl"][index - 9 : index + 1]))
        swing_distance = (entry - swing) * 10_000.0
    else:
        swing = float(np.max(data["ah"][index - 9 : index + 1]))
        swing_distance = (swing - entry) * 10_000.0
    raw = max(
        float(policy["minimum_catastrophe_stop_pips"]),
        swing_distance + spread * float(policy["catastrophe_spread_buffer_multiple"]),
        impulse * float(policy["catastrophe_impulse_multiple"]),
    )
    atr_upper = max(
        float(policy["minimum_catastrophe_stop_pips"]),
        atr * float(policy["maximum_auxiliary_atr_multiple"]),
    )
    return min(raw, atr_upper, float(policy["maximum_catastrophe_stop_pips"]))


def _exit_architecture_trade(
    data: dict[str, np.ndarray],
    index: int,
    direction: int,
    *,
    catastrophe_width: float | None,
    config: dict[str, Any],
    structure_enabled: bool,
    disconnect_minutes: int = 0,
    take_profit: float = 2.4,
) -> dict[str, Any]:
    policy = config["structure_exit"]
    entry = float(data["ac"][index] if direction > 0 else data["bc"][index])
    stop = (
        entry - catastrophe_width / 10_000.0
        if direction > 0 and catastrophe_width is not None
        else entry + catastrophe_width / 10_000.0
        if catastrophe_width is not None
        else None
    )
    target = entry + take_profit / 10_000.0 if direction > 0 else entry - take_profit / 10_000.0
    mid = (data["bc"] + data["ac"]) / 2.0
    spread = (data["ac"] - data["bc"]) * 10_000.0
    initial_high = float(np.max(mid[index - 15 : index + 1]))
    initial_low = float(np.min(mid[index - 15 : index + 1]))
    mae = 0.0
    gap_slippage = 0.0

    def close_pnl(cursor: int) -> float:
        return (
            (float(data["bc"][cursor]) - entry) * 10_000.0
            if direction > 0
            else (entry - float(data["ac"][cursor])) * 10_000.0
        )

    for cursor in range(index + 1, index + int(policy["holding_time_cap_minutes"]) + 1):
        held = cursor - index
        if direction > 0:
            open_px, high_px, low_px = data["bo"][cursor], data["bh"][cursor], data["bl"][cursor]
            adverse = (entry - float(low_px)) * 10_000.0
            if stop is not None and open_px <= stop:
                pnl = (float(open_px) - entry) * 10_000.0
                gap_slippage = max(0.0, -float(catastrophe_width) - pnl)
                return {"pnl": pnl, "reason": "CATASTROPHE_STOP_GAP", "held": held, "mae": max(mae, adverse), "gap_slippage": gap_slippage}
            stop_hit = stop is not None and low_px <= stop
            target_hit = high_px >= target
        else:
            open_px, high_px, low_px = data["ao"][cursor], data["ah"][cursor], data["al"][cursor]
            adverse = (float(high_px) - entry) * 10_000.0
            if stop is not None and open_px >= stop:
                pnl = (entry - float(open_px)) * 10_000.0
                gap_slippage = max(0.0, -float(catastrophe_width) - pnl)
                return {"pnl": pnl, "reason": "CATASTROPHE_STOP_GAP", "held": held, "mae": max(mae, adverse), "gap_slippage": gap_slippage}
            stop_hit = stop is not None and high_px >= stop
            target_hit = low_px <= target
        mae = max(mae, adverse)
        if stop_hit:
            return {"pnl": -float(catastrophe_width), "reason": "CATASTROPHE_STOP", "held": held, "mae": mae, "gap_slippage": 0.0}
        if target_hit:
            return {"pnl": take_profit, "reason": "TARGET", "held": held, "mae": mae, "gap_slippage": 0.0}

        pnl = close_pnl(cursor)
        if catastrophe_width is None and pnl <= -float(policy["campaign_loss_cap_pips"]):
            return {"pnl": pnl, "reason": "CAMPAIGN_LOSS_CAP_SHADOW", "held": held, "mae": mae, "gap_slippage": 0.0}
        if structure_enabled and held > disconnect_minutes:
            reason = None
            if held >= 5:
                path = mid[index + 1 : cursor + 1]
                new_extreme = (
                    float(np.max(path)) > initial_high
                    if direction > 0
                    else float(np.min(path)) < initial_low
                )
                adverse5 = (
                    (initial_high - float(np.min(path))) * 10_000.0
                    if direction > 0
                    else (float(np.max(path)) - initial_low) * 10_000.0
                )
                if not new_extreme and adverse5 >= float(config["resolution"]["minimum_adverse_reversal_pips"]):
                    reason = "FAILED_CONTINUATION"
            swing = int(policy["swing_lookback_minutes"])
            if reason is None and held >= swing:
                adverse_break = (
                    float(mid[cursor]) < float(np.min(mid[cursor - swing : cursor]))
                    if direction > 0
                    else float(mid[cursor]) > float(np.max(mid[cursor - swing : cursor]))
                )
                if adverse_break:
                    reason = "ADVERSE_SWING_BREAK"
            window = int(policy["short_window_minutes"])
            if reason is None and cursor - window * 2 >= index:
                velocity = (mid[cursor] - mid[cursor - window]) * 10_000.0 * direction / window
                prior_velocity = (
                    (mid[cursor - window] - mid[cursor - window * 2])
                    * 10_000.0
                    * direction
                    / window
                )
                acceleration = (velocity - prior_velocity) / window
                if velocity <= -float(policy["velocity_reversal_pips_per_minute"]):
                    reason = "ADVERSE_VELOCITY"
                elif acceleration <= -float(policy["acceleration_reversal_pips_per_minute2"]):
                    reason = "ADVERSE_ACCELERATION"
            if reason is None:
                prior_spread = spread[max(index, cursor - 20) : cursor]
                median_spread = float(np.median(prior_spread)) if len(prior_spread) else 0.0
                if median_spread > 0.0 and spread[cursor] / median_spread >= float(policy["spread_expansion_ratio"]):
                    reason = "SPREAD_EXPANSION"
            if reason is None and held >= int(policy["time_stop_minutes"]):
                reason = "TIME_STOP"
            if reason is not None:
                return {"pnl": pnl, "reason": reason, "held": held, "mae": mae, "gap_slippage": 0.0}
    cursor = index + int(policy["holding_time_cap_minutes"])
    return {"pnl": close_pnl(cursor), "reason": "HOLDING_TIME_CAP", "held": cursor - index, "mae": mae, "gap_slippage": 0.0}


def _widths(
    data: dict[str, np.ndarray], atr: float, index: int, direction: int, config: dict[str, Any]
) -> dict[str, float]:
    entry = data["ac"][index] if direction > 0 else data["bc"][index]
    spread = (data["ac"][index] - data["bc"][index]) * 10_000.0
    if direction > 0:
        swing = float(np.min(data["bl"][max(0, index - 9) : index + 1]))
        swing_width = max(0.5, (entry - swing) * 10_000.0 + spread)
    else:
        swing = float(np.max(data["ah"][max(0, index - 9) : index + 1]))
        swing_width = max(0.5, (swing - entry) * 10_000.0 + spread)
    return {
        "FIXED_3_2": 3.2,
        "ATR_1_0": atr,
        "ATR_1_5": atr * 1.5,
        "ATR_2_0": atr * 2.0,
        "SWING_SPREAD_BUFFER": swing_width,
        "CONSERVATIVE_ATR_SWING": max(atr * 1.5, swing_width),
        "CONSERVATIVE_CATASTROPHE": _catastrophe_width(
            data, atr, index, direction, config
        ),
    }


def analyze(paths: list[Path], config_path: Path) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    data = _load(paths)
    atr = _m5_atr(data)
    raw_episodes = _episodes(data, config)
    episodes = [index for index in raw_episodes if math.isfinite(float(atr[index]))]
    mid = (data["bc"] + data["ac"]) / 2.0
    baseline: list[float] = []
    directions: list[int] = []
    failed = 0
    whipsaw = 0
    geometries: dict[str, dict[str, Any]] = {}
    raw_geometry: dict[str, list[float]] = {}
    stops: dict[str, list[str]] = {}
    gap_slippage: dict[str, list[float]] = {}
    reentry_loss: dict[str, list[float]] = {}
    architecture_rows: dict[str, list[dict[str, Any]]] = {
        "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT": [],
        "ATR_1_5_ONLY": [],
        "FIXED_3_2_ONLY": [],
        "NO_SL_STRUCTURE_EXIT_SHADOW_ONLY": [],
    }
    disconnect_stress: dict[str, list[dict[str, Any]]] = {
        "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT": [],
        "NO_SL_STRUCTURE_EXIT_SHADOW_ONLY": [],
    }
    episode_rows: list[dict[str, Any]] = []
    for index in episodes:
        direction = 1 if mid[index] > mid[index - 15] else -1
        directions.append(direction)
        initial = (mid[index] - mid[index - 15]) * 10_000.0 * direction
        path5 = (mid[index + 5] - mid[index]) * 10_000.0 * direction
        new_extreme = (
            np.max(mid[index + 1 : index + 6]) > mid[index]
            if direction > 0
            else np.min(mid[index + 1 : index + 6]) < mid[index]
        )
        failed_now = bool(
            not new_extreme
            and path5 <= -float(config["resolution"]["minimum_adverse_reversal_pips"])
        )
        if failed_now:
            failed += 1
        adverse30 = -(np.min((mid[index + 1 : index + 31] - mid[index]) * 10_000.0 * direction))
        if adverse30 >= 0.5 * initial:
            whipsaw += 1
        baseline.append(_baseline_trade(data, index, direction))
        episode_rows.append(
            {
                "at_utc": datetime.fromtimestamp(int(data["t"][index]), tz=timezone.utc).isoformat(),
                "direction": "UP" if direction > 0 else "DOWN",
                "impulse_15m_pips": round((mid[index] - mid[index - 15]) * 10_000.0, 6),
                "m5_atr_pips": round(float(atr[index]), 6),
                "baseline_60m_pips": round(baseline[-1], 6),
                "failed_continuation_5m": failed_now,
                "retraced_50pct_within_30m": bool(adverse30 >= 0.5 * initial),
            }
        )
        widths = _widths(data, float(atr[index]), index, direction, config)
        for name, width in widths.items():
            realized, reason, slippage, stopped_at = _sl_trade(data, index, direction, width)
            raw_geometry.setdefault(name, []).append(realized)
            stops.setdefault(name, []).append(reason)
            gap_slippage.setdefault(name, []).append(slippage)
            if stopped_at is not None and stopped_at + 60 < len(data["t"]):
                retry, _, _, _ = _sl_trade(data, stopped_at, direction, width)
                reentry_loss.setdefault(name, []).append(min(0.0, retry))
        architecture_specs = {
            "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT": (
                widths["CONSERVATIVE_CATASTROPHE"],
                True,
            ),
            "ATR_1_5_ONLY": (widths["ATR_1_5"], False),
            "FIXED_3_2_ONLY": (3.2, False),
            "NO_SL_STRUCTURE_EXIT_SHADOW_ONLY": (None, True),
        }
        for name, (width, structure_enabled) in architecture_specs.items():
            result = _exit_architecture_trade(
                data,
                index,
                direction,
                catastrophe_width=width,
                config=config,
                structure_enabled=structure_enabled,
            )
            risk_width = width or float(config["structure_exit"]["campaign_loss_cap_pips"])
            result["normalized_unit_fraction"] = min(1.0, 3.2 / float(risk_width))
            architecture_rows[name].append(result)
        for name, width in (
            (
                "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT",
                widths["CONSERVATIVE_CATASTROPHE"],
            ),
            ("NO_SL_STRUCTURE_EXIT_SHADOW_ONLY", None),
        ):
            stressed = _exit_architecture_trade(
                data,
                index,
                direction,
                catastrophe_width=width,
                config=config,
                structure_enabled=True,
                disconnect_minutes=5,
            )
            risk_width = width or float(config["structure_exit"]["campaign_loss_cap_pips"])
            stressed["normalized_unit_fraction"] = min(1.0, 3.2 / float(risk_width))
            disconnect_stress[name].append(stressed)
    for name, values in raw_geometry.items():
        base = _metrics(values)
        stop_reasons = stops[name]
        base.update(
            stop_hit_rate=round(sum(reason.startswith("STOP") for reason in stop_reasons) / len(values), 6) if values else None,
            stop_gap_count=sum(reason == "STOP_GAP" for reason in stop_reasons),
            gap_slippage_pips=round(sum(gap_slippage[name]), 6),
            post_stop_reentry_loss_pips=round(sum(reentry_loss.get(name, [])), 6),
            maximum_consecutive_stops=_max_loss_streak([-1.0 if reason.startswith("STOP") else 1.0 for reason in stop_reasons]),
            median_stop_width_pips=round(float(np.median([_widths(data, float(atr[index]), index, directions[pos], config)[name] for pos, index in enumerate(episodes)])), 6) if episodes else None,
        )
        geometries[name] = base
    baseline_metrics = _metrics(baseline)
    architecture_metrics = {
        name: {
            **_architecture_metrics(rows),
            "server_side_catastrophic_stop_required": name
            != "NO_SL_STRUCTURE_EXIT_SHADOW_ONLY",
            "live_candidate_eligible": name
            == "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT",
            "atr_used_for_onset_trigger": False,
            "automatic_reentry_during_shock_allowed": False,
            "comparison_controls": dict(config["structure_exit"]),
        }
        for name, rows in architecture_rows.items()
    }
    for name, rows in disconnect_stress.items():
        architecture_metrics[name]["five_minute_runtime_disconnect_stress"] = (
            _architecture_metrics(rows)
        )
    half_drain = [value * 0.5 for value in baseline]
    # New-entry freeze has no trade P/L by definition.  It is a loss-avoidance
    # arm, not a profitable reversal strategy.
    guard_metrics = _metrics([])
    guard_metrics.update(
        avoided_baseline_loss_pips=round(-sum(value for value in baseline if value < 0.0), 6),
        entry_rejection_rate_in_shock_band=1.0 if episodes else None,
        non_shock_entry_rejection_rate=0.0,
    )
    best = min(
        geometries,
        key=lambda name: (
            -float(
                geometries[name]["net_pips"]
                + geometries[name]["post_stop_reentry_loss_pips"]
            ),
            float(geometries[name]["maximum_consecutive_stops"]),
            -float(geometries[name]["p05_trade_pips"]),
        ),
    ) if geometries else None
    file_rows = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_rows.append({"path": str(path), "sha256": digest})
    return {
        "contract": "QR_FAST_BOT_SHOCK_GUARD_REPLAY_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "truth": {
            "files": file_rows,
            "rows": len(data["t"]),
            "from_utc": datetime.fromtimestamp(int(data["t"][0]), tz=timezone.utc).isoformat() if len(data["t"]) else None,
            "to_utc": datetime.fromtimestamp(int(data["t"][-1]), tz=timezone.utc).isoformat() if len(data["t"]) else None,
            "duplicate_timestamps": 0,
            "volume_available": True,
            "volume_used": False,
            "broker_http_methods_used": [],
        },
        "episodes": {
            "raw_detected_count": len(raw_episodes),
            "atr_auxiliary_unavailable_excluded_from_four_arm_comparison": len(raw_episodes)
            - len(episodes),
            "count": len(episodes),
            "up": sum(direction > 0 for direction in directions),
            "down": sum(direction < 0 for direction in directions),
            "failed_continuation_5m": failed,
            "whipsaw_30m_50pct": whipsaw,
            "august_28_2026": [
                row for row in episode_rows if row["at_utc"].startswith("2026-08-28")
            ],
        },
        "arms": {
            "baseline_immediate_continuation": baseline_metrics,
            "current_side_specific_quarantine": {
                "status": "NOT_IDENTIFIABLE_FROM_PRICE_ONLY_EPISODES",
                "reason": "RANGE_ROTATION strategy labels and proposal truth are not present in M1 BA candle files; no side proxy was fabricated.",
            },
            "new_shock_guard": guard_metrics,
            "new_shock_guard_plus_50pct_drain_proxy": {
                **_metrics(half_drain),
                "margin_release_proxy_fraction": 0.5,
                "manual_tagless_policy": "NO_TOUCH",
            },
        },
        "protective_stop_geometries": geometries,
        "exit_architecture_arms": architecture_metrics,
        "selected_operational_shadow_arm": "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT",
        "selected_arm_live_admission": False,
        "no_sl_arm_live_promotion_allowed": False,
        "atr_role": "AUXILIARY_NORMALIZATION_AND_UPPER_BOUND_ONLY",
        "selected_shadow_geometry_by_bounded_loss_tail_rule": best,
        "selection_rule": "MAXIMIZE_NET_PLUS_POST_STOP_REENTRY_PIPS_THEN_MINIMIZE_CONSECUTIVE_STOPS_THEN_P05_TAIL",
        "selection_is_live_admission": False,
        "positive_pf_required_for_live_promotion": True,
        "automatic_reversal_allowed": False,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "fast_bot_shock_guard_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.input, args.config)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
