#!/usr/bin/env python3
"""Causal opened-development replay for MTF_FX_CAUSAL_GEOMETRY_V3.

This module is deliberately offline.  It imports only the sealed V2 replay's
pure parsing/accounting helpers, reads immutable local BID/ASK candles, creates
cost-blind signals, and applies executable costs afterward to identical trade
lineage.  It has no network, credential, broker, order, or launchd surface.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "preregistration.json"
RESULT = ROOT / "result.json"
PACKET = ROOT / "evidence_packet.json"
V2_ROOT = ROOT.parents[1] / "m5_ema_challenger" / "2026-08-28-v2"
V2_SCRIPT = V2_ROOT / "replay_m5_interaction.py"

# Five minutes is the immutable source cadence. A different cadence changes
# decision/fill chronology and therefore requires a new candidate version.
M5_STEP = dt.timedelta(minutes=5)
# The three aggregate periods are UTC-aligned market structures, not tuned
# strategy parameters. A future non-clock aggregation requires a new family.
PERIOD_MINUTES = {"M15": 15, "H1": 60, "H4": 240}
# This tiny denominator only prevents division by zero in a flat ATR path. It
# should be replaced only if the numeric representation itself changes.
EPSILON = 1e-15
SCENARIOS = ("raw", "base", "adverse")


def _load_v2():
    spec = importlib.util.spec_from_file_location("sealed_v2_replay", V2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return V2.sha_file(path)


def parse_time(value):
    return V2.parse_time(value)


def iso_utc(value):
    return V2.iso_utc(value)


def midpoint(row, field):
    return V2.midpoint(row, field)


def pip_size(pair):
    return V2.pip_size(pair)


def path_pips(pair, side, entry, exit_):
    return V2.path_pips(pair, side, entry, exit_)


def quantile(values, probability):
    return V2.quantile(values, probability)


def close_time(row):
    return row["_time"] + M5_STEP


def utc_session(when):
    hour = when.hour
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 17:
        return "OVERLAP"
    if hour < 21:
        return "NY_LATE"
    return "ROLLOVER"


def opposite(side):
    return "SHORT" if side == "LONG" else "LONG"


def verify_v2_immutable(prereg):
    actual = {}
    for name, expected in prereg["v2_immutable_hashes"].items():
        digest = sha_file(V2_ROOT / name)
        if digest != expected:
            raise ValueError(f"V2 immutable hash mismatch: {name}")
        actual[name] = digest
    return actual


def validate_preregistered_contract(prereg):
    expected = {
        "C0": ("H4_H1_ALIGN_PULLBACK_RECLAIM", 4),
        "C1": ("H4_H1_ALIGN_PULLBACK_RECLAIM", 8),
        "C2": ("H4_H1_ALIGN_H1_COMPRESSION_BREAK_ACCEPTANCE", 4),
        "C3": ("H4_H1_ALIGN_H1_COMPRESSION_BREAK_ACCEPTANCE", 8),
        "C4": ("H4_DIRECTION_M15_OPPOSITE_RAIL_SWEEP_RECLAIM", 4),
        "C5": ("H4_DIRECTION_M15_OPPOSITE_RAIL_SWEEP_RECLAIM", 8),
        "C6": ("H4_EXTENSION_H1_DECELERATION_M15_SWEEP_RECLAIM", 8),
        "C7": ("USD_STAR_RESIDUAL_H1_RECOUPLING_M15_SWEEP_RECLAIM", 8),
    }
    actual = {
        key: (value["family"], value["max_age_hours"])
        for key, value in prereg["configs"].items()
    }
    split = prereg["inputs"]["split"]
    checks = {
        "candidate_id": prereg["candidate_id"] == "MTF_FX_CAUSAL_GEOMETRY_V3",
        "configs": actual == expected,
        "family_size": prereg["selection"]["family_size"] == 8,
        "family_alpha": prereg["selection"]["family_alpha"] == 0.05,
        "split": math.isclose(
            split["calibration_fraction"]
            + split["tuning_fraction"]
            + split["opened_development_fraction"],
            1.0,
        ),
        "m15_ema": prereg["features"]["m15_fast_ema_bars"] == 4,
        "break_window": prereg["features"]["break_reference_m15_bars"] == 8,
        "q_low": prereg["features"]["calibration_quantile_low"] == 0.35,
        "q_high": prereg["features"]["calibration_quantile_high"] == 0.65,
        "base_cost": prereg["costs"]["base_slippage_pips_per_side"] == 0.3,
        "adverse_cost": prereg["costs"]["adverse_slippage_pips_per_side"] == 0.9,
        "zero_fee": prereg["costs"]["fees_pips_per_side"] == 0.0,
        "units": prereg["portfolio"]["units"] == 1000,
        "one_position": prereg["chronology"]["max_positions_per_pair_config"] == 1,
        "same_bar_false": prereg["chronology"]["same_bar_fill"] is False,
        "cost_gate_false": prereg["execution_arms"]["cost_gate"] is False,
        "shadow_false": prereg["future_evidence"]["shadow_challenger_eligible_now"] is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError("preregistered contract mismatch: " + ",".join(failed))
    return checks


def load_inputs(prereg):
    return V2.load_inputs(prereg)


def _floor_time(value, minutes):
    epoch = int(value.timestamp())
    seconds = minutes * 60
    return dt.datetime.fromtimestamp(epoch - epoch % seconds, tz=dt.timezone.utc)


def aggregate_bars(rows, minutes):
    """Build exact UTC-aligned complete bars and reject incomplete buckets."""
    count = minutes // 5
    buckets = {}
    for index, row in enumerate(rows):
        start = _floor_time(row["_time"], minutes)
        buckets.setdefault(start, []).append((index, row))
    bars = []
    for start, members in sorted(buckets.items()):
        if len(members) != count:
            continue
        expected = [start + M5_STEP * position for position in range(count)]
        if [row["_time"] for _, row in members] != expected:
            continue
        first_index, first = members[0]
        last_index, last = members[-1]
        bars.append(
            {
                "start": start,
                "close_time": start + dt.timedelta(minutes=minutes),
                "source_start_index": first_index,
                "source_end_index": last_index,
                "o": midpoint(first, "o"),
                "h": max(midpoint(row, "h") for _, row in members),
                "l": min(midpoint(row, "l") for _, row in members),
                "c": midpoint(last, "c"),
            }
        )
    return bars


def ema_series(values, window):
    alpha = 2.0 / (window + 1.0)
    result = []
    current = None
    for value in values:
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        result.append(current)
    return result


def _contiguous(bars, start, end, minutes):
    if start < 0 or end >= len(bars) or start > end:
        return False
    step = dt.timedelta(minutes=minutes)
    return all(
        bars[index + 1]["start"] - bars[index]["start"] == step
        for index in range(start, end)
    )


def _latest_index(bars, decision_time):
    closes = [bar["close_time"] for bar in bars]
    return bisect.bisect_right(closes, decision_time) - 1


def compute_pair_states(pair, rows):
    m15 = aggregate_bars(rows, 15)
    h1 = aggregate_bars(rows, 60)
    h4 = aggregate_bars(rows, 240)
    m15_ema4 = ema_series([bar["c"] for bar in m15], 4)
    h1_fast = ema_series([bar["c"] for bar in h1], 3)
    h1_slow = ema_series([bar["c"] for bar in h1], 12)
    h4_fast = ema_series([bar["c"] for bar in h4], 3)
    h4_slow = ema_series([bar["c"] for bar in h4], 12)
    states = []

    for index in range(9, len(m15)):
        bar = m15[index]
        if not _contiguous(m15, index - 9, index, 15):
            continue
        decision_time = bar["close_time"]
        h1_index = _latest_index(h1, decision_time)
        h4_index = _latest_index(h4, decision_time)
        if h1_index < 2 or h4_index < 6:
            continue
        if h1_fast[h1_index] == h1_slow[h1_index] or h4_fast[h4_index] == h4_slow[h4_index]:
            continue
        h1_side = "LONG" if h1_fast[h1_index] > h1_slow[h1_index] else "SHORT"
        h4_side = "LONG" if h4_fast[h4_index] > h4_slow[h4_index] else "SHORT"
        previous = m15[index - 1]
        pullback_side = None
        if previous["c"] <= m15_ema4[index - 1] and bar["c"] > m15_ema4[index]:
            pullback_side = "LONG"
        if previous["c"] >= m15_ema4[index - 1] and bar["c"] < m15_ema4[index]:
            pullback_side = "SHORT" if pullback_side is None else None

        acceptance_rails = m15[index - 9 : index - 1]
        acceptance_upper = max(item["h"] for item in acceptance_rails)
        acceptance_lower = min(item["l"] for item in acceptance_rails)
        acceptance_candidates = []
        if previous["c"] > acceptance_upper and bar["c"] > acceptance_upper:
            acceptance_candidates.append("LONG")
        if previous["c"] < acceptance_lower and bar["c"] < acceptance_lower:
            acceptance_candidates.append("SHORT")
        acceptance_side = acceptance_candidates[0] if len(acceptance_candidates) == 1 else None

        sweep_rails = m15[index - 8 : index]
        sweep_upper = max(item["h"] for item in sweep_rails)
        sweep_lower = min(item["l"] for item in sweep_rails)
        sweep_candidates = []
        if bar["l"] < sweep_lower and bar["c"] > sweep_lower and bar["c"] > bar["o"]:
            sweep_candidates.append("LONG")
        if bar["h"] > sweep_upper and bar["c"] < sweep_upper and bar["c"] < bar["o"]:
            sweep_candidates.append("SHORT")
        sweep_side = sweep_candidates[0] if len(sweep_candidates) == 1 else None

        h1_bar = h1[h1_index]
        h1_previous = h1[h1_index - 1]
        h1_return = math.log(h1_bar["c"] / h1_previous["c"])
        h1_previous_return = math.log(h1_previous["c"] / h1[h1_index - 2]["c"])
        h1_range_ratio = (h1_bar["h"] - h1_bar["l"]) / h1_bar["c"]
        atr_sample = [item["h"] - item["l"] for item in h4[h4_index - 5 : h4_index + 1]]
        h4_atr6 = statistics.mean(atr_sample)
        h4_extension = abs(h4[h4_index]["c"] - h4_slow[h4_index]) / (h4_atr6 + EPSILON)
        four_bar_return = math.log(bar["c"] / m15[index - 4]["c"])
        states.append(
            {
                "pair": pair,
                "m15_index": index,
                "decision_time_dt": decision_time,
                "decision_time": iso_utc(decision_time),
                "decision_source_end_index": bar["source_end_index"],
                "session": utc_session(decision_time),
                "h4_side": h4_side,
                "h1_side": h1_side,
                "h1_range_ratio": h1_range_ratio,
                "h1_deceleration": abs(h1_return) < abs(h1_previous_return),
                "h4_extension": h4_extension,
                "pullback_side": pullback_side,
                "acceptance_side": acceptance_side,
                "sweep_side": sweep_side,
                "m15_o": bar["o"],
                "m15_h": bar["h"],
                "m15_l": bar["l"],
                "m15_c": bar["c"],
                "m15_ema4": m15_ema4[index],
                "four_bar_return": four_bar_return,
                "usd_star": None,
                "usd_residual": None,
                "graph_side": None,
            }
        )
    return {"M15": m15, "H1": h1, "H4": h4, "states": states}


def attach_usd_star(states_by_pair):
    """Attach simultaneous three-pair USD factor without forward filling."""
    by_time = {}
    for pair, states in states_by_pair.items():
        for state in states:
            by_time.setdefault(state["decision_time"], {})[pair] = state
    for members in by_time.values():
        if set(members) != {"EUR_USD", "AUD_USD", "USD_JPY"}:
            continue
        oriented = {
            pair: (state["four_bar_return"] if pair == "USD_JPY" else -state["four_bar_return"])
            for pair, state in members.items()
        }
        factor = statistics.mean(oriented.values())
        if factor == 0.0:
            continue
        for pair, state in members.items():
            state["usd_star"] = factor
            state["usd_residual"] = oriented[pair] - factor
            if pair == "USD_JPY":
                state["graph_side"] = "LONG" if factor > 0.0 else "SHORT"
            else:
                state["graph_side"] = "SHORT" if factor > 0.0 else "LONG"


def derive_thresholds(states_by_pair, calibration_ends, prereg):
    low = prereg["features"]["calibration_quantile_low"]
    high = prereg["features"]["calibration_quantile_high"]
    minimum = prereg["features"]["minimum_calibration_observations_per_pair"]
    thresholds = {}
    for pair, states in states_by_pair.items():
        sample = [
            state
            for state in states
            if state["decision_source_end_index"] < calibration_ends[pair]
        ]
        graph_sample = [
            abs(state["usd_residual"])
            for state in sample
            if state["usd_residual"] is not None
        ]
        if len(sample) < minimum or len(graph_sample) < minimum:
            raise ValueError(f"insufficient calibration states: {pair}")
        thresholds[pair] = {
            "rows": len(sample),
            "graph_rows": len(graph_sample),
            "h1_compression_q35": quantile([state["h1_range_ratio"] for state in sample], low),
            "h4_extension_q65": quantile([state["h4_extension"] for state in sample], high),
            "usd_residual_abs_q65": quantile(graph_sample, high),
        }
    return thresholds


def state_feature_hash(state):
    material = {
        key: value
        for key, value in state.items()
        if key not in {"decision_time_dt", "m15_index"}
    }
    return sha_bytes(canonical(material))


def make_source_signals(pair, rows, states, start, end, prereg_sha):
    times = [row["_time"] for row in rows]
    result = []
    for state in states:
        decision_index = state["decision_source_end_index"]
        if decision_index < start or decision_index >= end:
            continue
        # Strictly later excludes the M5 open exactly at the M15 close.
        fill_index = bisect.bisect_right(times, state["decision_time_dt"])
        if fill_index >= end:
            continue
        if rows[fill_index]["_time"] - state["decision_time_dt"] != M5_STEP:
            continue
        feature_hash = state_feature_hash(state)
        signal_id = sha_bytes(
            f"{prereg_sha}|{pair}|{state['decision_time']}|{feature_hash}".encode("utf-8")
        )
        result.append(
            {
                **state,
                "feature_hash": feature_hash,
                "signal_id": signal_id,
                "decision_index": decision_index,
                "fill_index": fill_index,
            }
        )
    return result


def gate_signal(config_id, signal, threshold):
    aligned = signal["h4_side"] == signal["h1_side"]
    if config_id in {"C0", "C1"}:
        return signal["h4_side"] if aligned and signal["pullback_side"] == signal["h4_side"] else None
    if config_id in {"C2", "C3"}:
        compressed = signal["h1_range_ratio"] <= threshold["h1_compression_q35"]
        return signal["h4_side"] if aligned and compressed and signal["acceptance_side"] == signal["h4_side"] else None
    if config_id in {"C4", "C5"}:
        return signal["h4_side"] if signal["sweep_side"] == signal["h4_side"] else None
    if config_id == "C6":
        extended = signal["h4_extension"] >= threshold["h4_extension_q65"]
        fade_side = opposite(signal["h4_side"])
        return fade_side if extended and signal["h1_deceleration"] and signal["sweep_side"] == fade_side else None
    if config_id == "C7":
        if signal["usd_residual"] is None or signal["graph_side"] is None:
            return None
        residual = abs(signal["usd_residual"]) >= threshold["usd_residual_abs_q65"]
        recoupled = signal["h1_side"] == signal["graph_side"]
        return signal["graph_side"] if residual and recoupled and signal["sweep_side"] == signal["graph_side"] else None
    raise ValueError(f"unknown config: {config_id}")


def is_consecutive(rows, start, end):
    return V2.is_consecutive(rows, start, end)


def _mtm(pair, rows, side, fill_index, exit_index, entry_mid, entry_exec):
    points = []
    mfe = -math.inf
    mae = math.inf
    for index in range(fill_index, exit_index + 1):
        row = rows[index]
        mid_close = midpoint(row, "c")
        executable_close = row["bid"]["c"] if side == "LONG" else row["ask"]["c"]
        raw = path_pips(pair, side, entry_mid, mid_close)
        observed = path_pips(pair, side, entry_exec, executable_close)
        favorable = midpoint(row, "h") if side == "LONG" else midpoint(row, "l")
        adverse = midpoint(row, "l") if side == "LONG" else midpoint(row, "h")
        mfe = max(mfe, path_pips(pair, side, entry_mid, favorable))
        mae = min(mae, path_pips(pair, side, entry_mid, adverse))
        points.append(
            {
                "time": iso_utc(close_time(row)),
                "raw": raw,
                "base": observed - 0.6,
                "adverse": observed - 1.8,
            }
        )
    return points, mfe, mae


def replay_config(pair, rows, source_signals, threshold, config_id, end, prereg):
    horizon_hours = prereg["configs"][config_id]["max_age_hours"]
    horizon_bars = horizon_hours * 12
    trades = []
    gated = []
    busy_until = -1
    collision_skips = 0
    gap_unscorable = 0
    for signal in source_signals:
        side = gate_signal(config_id, signal, threshold)
        if side is None:
            continue
        gated.append(dict(signal, config_id=config_id, side=side))
        fill_index = signal["fill_index"]
        if fill_index <= busy_until:
            collision_skips += 1
            continue
        planned_exit = fill_index + horizon_bars - 1
        exit_index = min(planned_exit, end - 1)
        if not is_consecutive(rows, signal["decision_index"], exit_index):
            gap_unscorable += 1
            continue
        entry = rows[fill_index]
        exit_row = rows[exit_index]
        entry_mid = midpoint(entry, "o")
        exit_mid = midpoint(exit_row, "c")
        entry_exec = entry["ask"]["o"] if side == "LONG" else entry["bid"]["o"]
        exit_exec = exit_row["bid"]["c"] if side == "LONG" else exit_row["ask"]["c"]
        raw = path_pips(pair, side, entry_mid, exit_mid)
        observed = path_pips(pair, side, entry_exec, exit_exec)
        mtm, mfe, mae = _mtm(pair, rows, side, fill_index, exit_index, entry_mid, entry_exec)
        terminal = planned_exit >= end
        trade_id = sha_bytes(f"{signal['signal_id']}|{config_id}|{side}".encode("utf-8"))
        trades.append(
            {
                "trade_id": trade_id,
                "signal_id": signal["signal_id"],
                "pair": pair,
                "config_id": config_id,
                "side": side,
                "session": signal["session"],
                "decision_time": signal["decision_time"],
                "decision_index": signal["decision_index"],
                "entry_index": fill_index,
                "entry_time": entry["time"],
                "exit_index": exit_index,
                "exit_time": iso_utc(close_time(exit_row)),
                "exit_reason": "SPLIT_BOUNDARY_LIQUIDATION" if terminal else f"FINITE_MAX_AGE_{horizon_hours}H",
                "terminal_liquidation": terminal,
                "age_bars": exit_index - fill_index + 1,
                "units": 1000,
                "entry_mid": entry_mid,
                "entry_executable": entry_exec,
                "exit_mid": exit_mid,
                "exit_executable": exit_exec,
                "raw_pips": raw,
                "base_pips": observed - 0.6,
                "adverse_pips": observed - 1.8,
                "roundtrip_spread_pips": raw - observed,
                "raw_mfe_pips": mfe,
                "raw_mae_pips": mae,
                "direction_correct": raw > 0.0,
                "mtm_points": mtm,
            }
        )
        busy_until = exit_index
    return {
        "source_signals": source_signals,
        "gated_signals": gated,
        "trades": trades,
        "collision_skips": collision_skips,
        "gap_unscorable": gap_unscorable,
    }


def lineage_hash(trades):
    payload = [
        {
            "trade_id": trade["trade_id"],
            "signal_id": trade["signal_id"],
            "pair": trade["pair"],
            "side": trade["side"],
            "decision_time": trade["decision_time"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
        }
        for trade in sorted(trades, key=lambda item: item["trade_id"])
    ]
    return sha_bytes(canonical(payload))


def _group_summary(trades, scenario, key):
    grouped = {}
    for trade in trades:
        grouped.setdefault(trade[key], []).append(trade[f"{scenario}_pips"])
    return {
        name: {
            "trades": len(values),
            "expectancy_pips": statistics.mean(values),
            "positive_rate": sum(value > 0.0 for value in values) / len(values),
        }
        for name, values in sorted(grouped.items())
    }


def cluster_summary(trades, scenario, selection):
    clusters = {}
    for trade in trades:
        when = parse_time(trade["decision_time"])
        cluster = f"{when.date().isoformat()}|H{(when.hour // 4) * 4:02d}|USD"
        clusters.setdefault(cluster, []).append(trade[f"{scenario}_pips"])
    means = [statistics.mean(values) for _, values in sorted(clusters.items())]
    z_value = statistics.NormalDist().inv_cdf(
        1.0 - selection["family_alpha"] / selection["family_size"]
    )
    mean = statistics.mean(means) if means else None
    standard_error = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else None
    lcb = mean - z_value * standard_error if standard_error is not None else None
    return {
        "cluster_definition": selection["cluster"],
        "n_eff_currency_time_clusters": len(means),
        "cluster_mean_pips": mean,
        "cluster_median_pips": statistics.median(means) if means else None,
        "cluster_standard_error_pips": standard_error,
        "family_critical_z": z_value,
        "family_adjusted_lcb_pips": lcb,
        "positive_cluster_rate": sum(value > 0.0 for value in means) / len(means) if means else 0.0,
    }


class JpyConverter(V2.JpyConverter):
    pass


def metrics(source_signals, gated_signals, trades, diagnostics, scenario, converter, selection):
    values = [trade[f"{scenario}_pips"] for trade in trades]
    pair_results = _group_summary(trades, scenario, "pair")
    session_results = _group_summary(trades, scenario, "session")
    cluster = cluster_summary(trades, scenario, selection)
    lineage = lineage_hash(trades)
    pnl = {
        trade["trade_id"]: converter.pnl(trade["pair"], trade[f"{scenario}_pips"], trade["exit_time"])
        for trade in trades
    }
    mark_events = {}
    exit_events = {}
    for trade in trades:
        exit_events.setdefault(trade["exit_time"], []).append(trade)
        for point in trade["mtm_points"]:
            mark_events.setdefault(point["time"], []).append(
                (trade["trade_id"], converter.pnl(trade["pair"], point[scenario], point["time"]))
            )
    initial = 200000.0
    realized = 0.0
    active = {}
    peak = initial
    max_drawdown = 0.0
    month_ends = {}
    for when in sorted(set(mark_events) | set(exit_events)):
        for trade_id, value in mark_events.get(when, []):
            active[trade_id] = value
        equity = initial + realized + sum(active.values())
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        for trade in exit_events.get(when, []):
            realized += pnl[trade["trade_id"]]
            active.pop(trade["trade_id"], None)
        equity = initial + realized + sum(active.values())
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        month_ends[when[:7]] = equity
    monthly = {}
    prior = initial
    for month, equity in sorted(month_ends.items()):
        monthly[month] = equity / prior if prior > 0.0 else None
        prior = equity
    raw_expectancy = statistics.mean(trade["raw_pips"] for trade in trades) if trades else 0.0
    expectancy = statistics.mean(values) if values else 0.0
    ages = [trade["age_bars"] for trade in trades]
    density = selection["density"]
    final_equity = initial + sum(pnl.values())
    source_days = {signal["decision_time"][:10] for signal in source_signals}
    return {
        "scenario": scenario,
        "source_signals": len(source_signals),
        "source_signals_per_day": len(source_signals) / len(source_days) if source_days else 0.0,
        "gated_signals": len(gated_signals),
        "trades": len(trades),
        "lineage_sha256": lineage,
        "collision_skips": diagnostics["collision_skips"],
        "data_gap_unscorable": diagnostics["gap_unscorable"],
        "direction_accuracy": sum(trade["direction_correct"] for trade in trades) / len(trades) if trades else 0.0,
        "expectancy_pips": expectancy,
        "gross_expectancy_pips": raw_expectancy,
        "break_even_roundtrip_cost_pips": raw_expectancy,
        "realized_cost_drag_pips": raw_expectancy - expectancy,
        **cluster,
        "pair_results": pair_results,
        "session_results": session_results,
        "pair_min_expectancy_pips": min((item["expectancy_pips"] for item in pair_results.values()), default=None),
        "session_min_expectancy_pips": min((item["expectancy_pips"] for item in session_results.values()), default=None),
        "pairs_with_positive_expectancy": sum(item["expectancy_pips"] > 0.0 for item in pair_results.values()),
        "sessions_with_positive_expectancy": sum(item["expectancy_pips"] > 0.0 for item in session_results.values()),
        "pairs_meeting_trade_floor": sum(item["trades"] >= density["per_pair_trades_gte"] for item in pair_results.values()),
        "mfe_mean_pips": statistics.mean(trade["raw_mfe_pips"] for trade in trades) if trades else 0.0,
        "mae_mean_pips": statistics.mean(trade["raw_mae_pips"] for trade in trades) if trades else 0.0,
        "inventory_age_q50_bars": quantile(ages, 0.50) if ages else 0.0,
        "inventory_age_q90_bars": quantile(ages, 0.90) if ages else 0.0,
        "inventory_age_q99_bars": quantile(ages, 0.99) if ages else 0.0,
        "terminal_liquidations": sum(trade["terminal_liquidation"] for trade in trades),
        "terminal_liquidation_pips": sum(trade[f"{scenario}_pips"] for trade in trades if trade["terminal_liquidation"]),
        "terminal_open_inventory": len(active),
        "turnover_units": 2000 * len(trades),
        "equity_multiple": final_equity / initial,
        "final_equity_jpy": final_equity,
        "max_drawdown": max_drawdown,
        "monthly_multiples": monthly,
        "monthly_2x_count": sum(value is not None and value >= 2.0 for value in monthly.values()),
    }


def split_result(data, signals_by_pair, thresholds, config_id, split_ends, converter, prereg):
    combined_source = []
    combined_gated = []
    combined_trades = []
    diagnostics = {"collision_skips": 0, "gap_unscorable": 0}
    for pair in sorted(data):
        replay = replay_config(
            pair, data[pair], signals_by_pair[pair], thresholds[pair], config_id, split_ends[pair], prereg
        )
        combined_source.extend(replay["source_signals"])
        combined_gated.extend(replay["gated_signals"])
        combined_trades.extend(replay["trades"])
        diagnostics["collision_skips"] += replay["collision_skips"]
        diagnostics["gap_unscorable"] += replay["gap_unscorable"]
    combined_trades.sort(key=lambda item: (item["exit_time"], item["pair"], item["trade_id"]))
    scenario_metrics = {
        scenario: metrics(
            combined_source,
            combined_gated,
            combined_trades,
            diagnostics,
            scenario,
            converter,
            prereg["selection"],
        )
        for scenario in SCENARIOS
    }
    hashes = {scenario: item["lineage_sha256"] for scenario, item in scenario_metrics.items()}
    if len(set(hashes.values())) != 1:
        raise AssertionError("RAW/BASE/ADVERSE lineage mismatch")
    return {
        "scenario_metrics": scenario_metrics,
        "shared_lineage_sha256": next(iter(hashes.values())),
        "same_trade_id_all_arms": True,
    }


def density_stability(metric, selection):
    density = selection["density"]
    stability = selection["stability"]
    gates = {
        "trades": metric["trades"] >= density["executed_trades_gte"],
        "clusters": metric["n_eff_currency_time_clusters"] >= density["currency_time_clusters_gte"],
        "pairs_meeting_floor": metric["pairs_meeting_trade_floor"] >= density["pairs_meeting_floor_gte"],
        "positive_pairs": metric["pairs_with_positive_expectancy"] >= stability["pairs_with_positive_expectancy_gte"],
        "positive_sessions": metric["sessions_with_positive_expectancy"] >= stability["sessions_with_positive_expectancy_gte"],
    }
    return gates, all(gates.values())


def _rank_key(config_id, configs):
    raw = configs[config_id]["tuning"]["scenario_metrics"]["raw"]
    def number(value):
        return value if value is not None else -math.inf
    return (
        -number(raw["family_adjusted_lcb_pips"]),
        -raw["expectancy_pips"],
        -number(raw["pair_min_expectancy_pips"]),
        -number(raw["session_min_expectancy_pips"]),
        -raw["n_eff_currency_time_clusters"],
        config_id,
    )


def select_config(configs, config_ids, selection):
    dense = []
    for config_id in config_ids:
        raw = configs[config_id]["tuning"]["scenario_metrics"]["raw"]
        if density_stability(raw, selection)[1]:
            dense.append(config_id)
    pool = dense or list(config_ids)
    return sorted(pool, key=lambda item: _rank_key(item, configs))[0], dense


def period_gate(metric, selection):
    diagnostic, stable = density_stability(metric, selection)
    gates = {
        **diagnostic,
        "positive_expectancy": metric["expectancy_pips"] > 0.0,
        "positive_family_lcb": metric["family_adjusted_lcb_pips"] is not None and metric["family_adjusted_lcb_pips"] > 0.0,
    }
    return gates, stable and gates["positive_expectancy"] and gates["positive_family_lcb"]


def classify(tuning, development, selection):
    tuning_gates, tuning_pass = period_gate(tuning["raw"], selection)
    dev_gates, dev_pass = period_gate(development["raw"], selection)
    gross = tuning_pass and dev_pass
    base = gross and development["base"]["expectancy_pips"] > 0.0 and development["base"]["family_adjusted_lcb_pips"] is not None and development["base"]["family_adjusted_lcb_pips"] > 0.0
    adverse = base and development["adverse"]["expectancy_pips"] > 0.0 and development["adverse"]["family_adjusted_lcb_pips"] is not None and development["adverse"]["family_adjusted_lcb_pips"] > 0.0
    if not gross:
        label = "REJECTED_NO_STABLE_GROSS_EDGE"
    elif not base:
        label = "GROSS_ONLY_COST_BOUND"
    elif not adverse:
        label = "BASE_EXECUTABLE_DEVELOPMENT_ONLY"
    else:
        label = "ADVERSE_ROBUST_DEVELOPMENT_ONLY"
    return {
        "classification": label,
        "tuning_gates": tuning_gates,
        "tuning_pass": tuning_pass,
        "opened_development_gates": dev_gates,
        "opened_development_pass": dev_pass,
        "gross_gate": gross,
        "base_gate": base,
        "adverse_gate": adverse,
    }


def main(write=True):
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    contract_checks = validate_preregistered_contract(prereg)
    prereg_sha = sha_file(PREREG)
    v2_hashes = verify_v2_immutable(prereg)
    data = load_inputs(prereg)
    structures = {pair: compute_pair_states(pair, rows) for pair, rows in data.items()}
    states_by_pair = {pair: value["states"] for pair, value in structures.items()}
    attach_usd_star(states_by_pair)
    split = prereg["inputs"]["split"]
    calibration_ends = {pair: int(len(rows) * split["calibration_fraction"]) for pair, rows in data.items()}
    tuning_ends = {
        pair: int(len(rows) * (split["calibration_fraction"] + split["tuning_fraction"]))
        for pair, rows in data.items()
    }
    thresholds = derive_thresholds(states_by_pair, calibration_ends, prereg)
    tuning_signals = {
        pair: make_source_signals(pair, data[pair], states_by_pair[pair], calibration_ends[pair], tuning_ends[pair], prereg_sha)
        for pair in data
    }
    development_signals = {
        pair: make_source_signals(pair, data[pair], states_by_pair[pair], tuning_ends[pair], len(data[pair]), prereg_sha)
        for pair in data
    }
    converter = JpyConverter(data["USD_JPY"])
    config_ids = [f"C{index}" for index in range(8)]
    configs = {}
    for config_id in config_ids:
        tuning = split_result(data, tuning_signals, thresholds, config_id, tuning_ends, converter, prereg)
        development = split_result(
            data,
            development_signals,
            thresholds,
            config_id,
            {pair: len(rows) for pair, rows in data.items()},
            converter,
            prereg,
        )
        configs[config_id] = {
            "definition": prereg["configs"][config_id],
            "tuning": tuning,
            "opened_development": development,
        }
    selected, stable_candidates = select_config(configs, config_ids, prereg["selection"])
    selected_tuning = configs[selected]["tuning"]["scenario_metrics"]
    selected_development = configs[selected]["opened_development"]["scenario_metrics"]
    classification = classify(selected_tuning, selected_development, prereg["selection"])
    result = {
        "schema_version": 1,
        "candidate_id": prereg["candidate_id"],
        "status": "UNADMITTED_OPENED_DEVELOPMENT_RESULT",
        "development_classification": classification["classification"],
        "prereg_sha256": prereg_sha,
        "script_sha256": sha_file(Path(__file__)),
        "contract_checks": contract_checks,
        "v2_immutable_hashes_verified": v2_hashes,
        "inputs": {
            pair: {"sha256": sha_file(Path(prereg["inputs"]["files"][pair]["path"])), "rows": len(rows), "first": rows[0]["time"], "last": rows[-1]["time"]}
            for pair, rows in data.items()
        },
        "aggregation_counts": {
            pair: {timeframe: len(structures[pair][timeframe]) for timeframe in ("M15", "H1", "H4")}
            for pair in data
        },
        "split_indices": {
            pair: {"calibration_end_exclusive": calibration_ends[pair], "tuning_end_exclusive": tuning_ends[pair], "development_end_exclusive": len(data[pair])}
            for pair in data
        },
        "calibration_thresholds": thresholds,
        "calibration_thresholds_sha256": sha_bytes(canonical(thresholds)),
        "source_signal_counts": {
            "tuning": {pair: len(value) for pair, value in tuning_signals.items()},
            "opened_development": {pair: len(value) for pair, value in development_signals.items()},
        },
        "configs": configs,
        "selected_config": selected,
        "selection_basis": "tuning_RAW_only_8_family_Bonferroni_USD_currency_4h_clusters",
        "stable_density_candidates": stable_candidates,
        "selected_tuning": selected_tuning,
        "selected_opened_development": selected_development,
        "gates": classification,
        "admission": False,
        "holdout": False,
        "opened_development_only": True,
        "profit_unproven": True,
        "shadow_challenger_eligible": False,
        "external_orders": 0,
        "network_attempts": 0,
        "credential_reads": 0,
        "live_authority": False,
    }
    if write:
        RESULT.write_bytes(json.dumps(result, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        packet = {
            "schema_version": 1,
            "candidate_id": prereg["candidate_id"],
            "status": result["status"],
            "development_classification": result["development_classification"],
            "selected_config": selected,
            "exact_config": prereg["configs"][selected],
            "selection_basis": result["selection_basis"],
            "selected_tuning": selected_tuning,
            "selected_opened_development": selected_development,
            "gates": classification,
            "admission": False,
            "holdout": False,
            "profit_unproven": True,
            "shadow_challenger_eligible": False,
            "external_orders": 0,
            "authority": prereg["authority"],
            "prereg_sha256": prereg_sha,
            "script_sha256": result["script_sha256"],
            "test_sha256": sha_file(ROOT / "test_multitf_geometry.py"),
            "readme_sha256": sha_file(ROOT / "README.md"),
            "result_sha256": sha_file(RESULT),
            "v2_immutable_hashes_verified": v2_hashes,
        }
        PACKET.write_bytes(json.dumps(packet, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()
    output = main(write=not arguments.no_write)
    print(
        json.dumps(
            {
                "selected_config": output["selected_config"],
                "development_classification": output["development_classification"],
                "tuning_raw": output["selected_tuning"]["raw"],
                "opened_development_raw": output["selected_opened_development"]["raw"],
                "opened_development_base": output["selected_opened_development"]["base"],
            },
            sort_keys=True,
        )
    )
