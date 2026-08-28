#!/usr/bin/env python3
"""Deterministic offline replay for M5_EMA_EXHAUSTION_INTERACTION_V2.

The module has no network, credential, broker, order, or runtime integration.
It deliberately produces cost-independent signals first, selects a config from
tuning RAW_SIGNAL results only, and applies execution costs afterward to the
same immutable trade lineage.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import datetime as dt
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "preregistration.json"
RESULT = ROOT / "result.json"
PACKET = ROOT / "evidence_packet.json"
V1_ROOT = ROOT.parent / "2026-08-28-v1"

# Five minutes is the fixed source-bar granularity. If this changes, a new
# preregistered candidate and data contract are required.
BAR_STEP = dt.timedelta(minutes=5)
# The tiny denominator prevents a zero division for a flat 12-return path. It
# is many orders below an FX log return and should be replaced only if numeric
# precision or the return representation changes.
PE_EPSILON = 1e-15
# These sessions are the preregistered decision window. Excluding the later
# sessions keeps every normal H48 exit no later than 20:55 UTC and avoids
# importing rollover/financing assumptions into this bounded experiment.
DECISION_SESSIONS = frozenset(("ASIA", "LONDON", "OVERLAP"))
SCENARIOS = ("raw", "base", "adverse")


def canonical(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value):
    # Python 3.10 accepts microseconds, while the immutable OANDA corpus stores
    # nanoseconds. Truncation changes representation only, not event ordering.
    body = value[:-1] if value.endswith("Z") else value
    if "." in body:
        head, fraction = body.split(".", 1)
        body = head + "." + fraction[:6]
    return dt.datetime.fromisoformat(body + "+00:00")


def pip_size(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


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


def close_time(row):
    """Return when every OHLC field of this OANDA M5 candle is available."""
    return row["_time"] + BAR_STEP


def iso_utc(when):
    return when.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def midpoint(row, field):
    return (row["bid"][field] + row["ask"][field]) / 2.0


def path_pips(pair, side, entry_price, exit_price):
    movement = exit_price - entry_price
    if side == "SHORT":
        movement = -movement
    return movement / pip_size(pair)


def load_inputs(prereg):
    result = {}
    for pair, spec in prereg["inputs"]["files"].items():
        path = Path(spec["path"])
        if sha_file(path) != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {pair}")
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    not row.get("complete")
                    or row.get("granularity") != "M5"
                    or row.get("price") != "BA"
                    or row.get("pair") != pair
                ):
                    raise ValueError(f"invalid completed M5 BID/ASK row: {pair}")
                row["_time"] = parse_time(row["time"])
                rows.append(row)
        if len(rows) != spec["rows"]:
            raise ValueError(f"input row-count mismatch: {pair}")
        if any(
            rows[index]["_time"] >= rows[index + 1]["_time"]
            for index in range(len(rows) - 1)
        ):
            raise ValueError(f"non-monotonic input chronology: {pair}")
        result[pair] = rows
    return result


def verify_v1_immutable(prereg):
    actual = {}
    for name, expected in prereg["v1_immutable_hashes"].items():
        path = V1_ROOT / name
        digest = sha_file(path)
        if digest != expected:
            raise ValueError(f"V1 immutable hash mismatch: {name}")
        actual[name] = digest
    return actual


def validate_preregistered_contract(prereg):
    """Fail closed if code-fixed mechanics disagree with preregistered values."""
    expected_configs = {
        "C0": ("all eligible source states", "d_t", 24),
        "C1": ("all eligible source states", "-d_t", 24),
        "C2": ("PE>=pair-session calibration Q67", "-d_t", 24),
        "C3": ("PE>=pair-session calibration Q67", "-d_t", 48),
        "C4": ("ASIA and PE>=Q67 and RV Q33<=RV<Q67", "-d_t", 48),
        "C5": ("LONDON and PE>=Q67 and RV Q33<=RV<Q67", "-d_t", 48),
        "C6": (
            "break rejection direction=-d_t and PE>=Q67 and RV Q33<=RV<Q67",
            "rejection direction",
            48,
        ),
        "C7": (
            "break acceptance direction=d_t and PE>=Q67 and RV>=Q67",
            "acceptance direction",
            48,
        ),
    }
    actual_configs = {
        config_id: (spec["gate"], spec["side"], spec["max_age_bars"])
        for config_id, spec in prereg["configs"].items()
    }
    gross = prereg["gross_edge_gate"]
    split = prereg["inputs"]["split"]
    checks = {
        "candidate_id": prereg["candidate_id"] == "M5_EMA_EXHAUSTION_INTERACTION_V2",
        "ema_fast": prereg["features"]["ema_fast_bars"] == 3,
        "ema_slow": prereg["features"]["ema_slow_bars"] == 12,
        "path_window": prereg["features"]["path_window_returns"] == 12,
        "break_reference": prereg["features"]["break_reference_bars"] == 12,
        "pe_epsilon": prereg["features"]["path_efficiency_epsilon"] == PE_EPSILON,
        "quantile_low": prereg["features"]["calibration_quantile_low"] == 0.33,
        "quantile_high": prereg["features"]["calibration_quantile_high"] == 0.67,
        "split_fractions": math.isclose(
            split["calibration_fraction"]
            + split["tuning_fraction"]
            + split["opened_development_fraction"],
            1.0,
        )
        and split["calibration_fraction"] > 0.0
        and split["tuning_fraction"] > 0.0
        and split["opened_development_fraction"] > 0.0,
        "decision_sessions": frozenset(prereg["features"]["decision_sessions"])
        == DECISION_SESSIONS,
        "configs": actual_configs == expected_configs,
        "initial_equity": prereg["portfolio"]["initial_equity_jpy"] == 200000,
        "units": prereg["portfolio"]["units"] == 1000,
        "one_position": prereg["portfolio"]["max_positions_per_pair_config"] == 1,
        "base_slippage": prereg["costs"]["base_slippage_pips_per_side"] == 0.3,
        "adverse_slippage": prereg["costs"]["adverse_slippage_pips_per_side"] == 0.9,
        "zero_fees": prereg["costs"]["fees_pips_per_side"] == 0,
        "family_size": prereg["selection"]["family_size"] == 8,
        "family_alpha": prereg["selection"]["family_alpha"] == 0.05,
        "density_trades": prereg["selection"]["density"]["executed_trades_gte"] == 120,
        "density_days": prereg["selection"]["density"]["utc_decision_day_clusters_gte"] == 20,
        "density_per_pair": prereg["selection"]["density"]["per_pair_trades_gte"] == 30,
        "density_pairs": prereg["selection"]["density"]["pairs_with_at_least_30_trades_gte"] == 2,
        "ranking": prereg["selection"]["ranking"]
        == [
            "maximum RAW family-adjusted LCB",
            "maximum RAW pooled expectancy",
            "maximum RAW UTC-day median",
            "maximum N_eff UTC-day clusters",
            "config_id lexical",
        ],
        "gross_periods": gross["required_periods"]
        == ["tuning", "opened_development"],
        "gross_density_required": gross["each_period"]["density_gate_required"]
        is True,
        "base_requires_gross": gross[
            "base_executable_candidate_opened_development"
        ]["requires_gross_edge_gate"]
        is True,
        "stress_requires_base": gross[
            "stress_robust_candidate_opened_development"
        ]["requires_base_executable_candidate"]
        is True,
        "classification_order": gross["classification_order"]
        == [
            "REJECTED_NO_GROSS_EDGE",
            "GROSS_ONLY_COST_BOUND",
            "BASE_EXECUTABLE_CANDIDATE",
            "STRESS_ROBUST_CANDIDATE",
        ],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("preregistered contract mismatch: " + ",".join(failed))
    return checks


def is_consecutive(rows, start, end):
    """Return true when all timestamps in the inclusive range are 5m apart."""
    if start < 0 or end >= len(rows) or start > end:
        return False
    return all(
        rows[index + 1]["_time"] - rows[index]["_time"] == BAR_STEP
        for index in range(start, end)
    )


def compute_states(rows):
    """Compute completed-bar-only feature and break state for every row."""
    closes = [midpoint(row, "c") for row in rows]
    highs = [midpoint(row, "h") for row in rows]
    lows = [midpoint(row, "l") for row in rows]
    states = [None] * len(rows)
    alpha_fast = 2.0 / 4.0
    alpha_slow = 2.0 / 13.0
    ema_fast = None
    ema_slow = None
    segment_start = 0

    for index, row in enumerate(rows):
        if index == 0 or row["_time"] - rows[index - 1]["_time"] != BAR_STEP:
            segment_start = index
            ema_fast = closes[index]
            ema_slow = closes[index]
        else:
            ema_fast = alpha_fast * closes[index] + (1.0 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * closes[index] + (1.0 - alpha_slow) * ema_slow

        # Fourteen contiguous bars are needed for t-13..t break context. This
        # also contains the 13 closes required for twelve log returns.
        if index - segment_start < 13 or ema_fast == ema_slow:
            continue
        returns = [
            math.log(closes[position] / closes[position - 1])
            for position in range(index - 11, index + 1)
        ]
        efficiency = abs(sum(returns)) / (
            sum(abs(value) for value in returns) + PE_EPSILON
        )
        realized_energy = math.sqrt(sum(value * value for value in returns))
        trend_side = "LONG" if ema_fast > ema_slow else "SHORT"

        upper = max(highs[index - 13 : index - 1])
        lower = min(lows[index - 13 : index - 1])
        previous = index - 1
        candidates = []
        if closes[previous] > upper and closes[index] > upper:
            candidates.append(("ACCEPTANCE", "LONG"))
        if closes[previous] < lower and closes[index] < lower:
            candidates.append(("ACCEPTANCE", "SHORT"))
        if (
            highs[previous] > upper
            and closes[previous] <= upper
            and closes[index] < closes[previous]
        ):
            candidates.append(("REJECTION", "SHORT"))
        if (
            lows[previous] < lower
            and closes[previous] >= lower
            and closes[index] > closes[previous]
        ):
            candidates.append(("REJECTION", "LONG"))

        if len(candidates) == 1:
            break_kind, break_side = candidates[0]
        elif not candidates:
            break_kind, break_side = "NONE", None
        else:
            break_kind, break_side = "UNKNOWN", None
        states[index] = {
            "index": index,
            "bar_open_time": row["time"],
            "decision_time": iso_utc(close_time(row)),
            "session": utc_session(close_time(row)),
            "trend_side": trend_side,
            "path_efficiency": efficiency,
            "realized_energy": realized_energy,
            "break_kind": break_kind,
            "break_side": break_side,
            "reference_upper": upper,
            "reference_lower": lower,
        }
    return states


def derive_thresholds(
    states_by_pair,
    calibration_ends,
    minimum_rows,
    quantile_low=0.33,
    quantile_high=0.67,
):
    """Freeze pair-session PE/RV quantiles from calibration prefixes only."""
    thresholds = {}
    sessions = ("ASIA", "LONDON", "OVERLAP", "NY_LATE", "ROLLOVER")
    for pair, states in states_by_pair.items():
        thresholds[pair] = {}
        calibration_states = [
            state
            for state in states[: calibration_ends[pair]]
            if state is not None
        ]
        for session in sessions:
            sample = [
                state for state in calibration_states if state["session"] == session
            ]
            if len(sample) < minimum_rows:
                raise ValueError(
                    f"insufficient calibration rows: {pair}/{session}={len(sample)}"
                )
            pe = [state["path_efficiency"] for state in sample]
            rv = [state["realized_energy"] for state in sample]
            thresholds[pair][session] = {
                "rows": len(sample),
                "pe_q33": quantile(pe, quantile_low),
                "pe_q67": quantile(pe, quantile_high),
                "rv_q33": quantile(rv, quantile_low),
                "rv_q67": quantile(rv, quantile_high),
            }
    return thresholds


def source_feature_hash(pair, row, state):
    # Signal identity intentionally excludes BID/ASK spread and any execution
    # cost. Mid geometry and completed-state values are the prediction input.
    material = {
        "pair": pair,
        "bar_open_time": row["time"],
        "decision_time": state["decision_time"],
        "mid_o": midpoint(row, "o"),
        "mid_h": midpoint(row, "h"),
        "mid_l": midpoint(row, "l"),
        "mid_c": midpoint(row, "c"),
        "trend_side": state["trend_side"],
        "path_efficiency": state["path_efficiency"],
        "realized_energy": state["realized_energy"],
        "session": state["session"],
        "break_kind": state["break_kind"],
        "break_side": state["break_side"],
        "reference_upper": state["reference_upper"],
        "reference_lower": state["reference_lower"],
    }
    return sha_bytes(canonical(material))


def make_source_signals(pair, rows, states, start, end, prereg_sha):
    signals = []
    first = max(start, 13)
    # A decision at end-1 has no next-bar open inside its owning split.
    for index in range(first, end - 1):
        state = states[index]
        if state is None or state["session"] not in DECISION_SESSIONS:
            continue
        feature_hash = source_feature_hash(pair, rows[index], state)
        signal_id = sha_bytes(
            (
                f"{prereg_sha}|{pair}|{state['decision_time']}|"
                f"{feature_hash}|{state['trend_side']}"
            ).encode("utf-8")
        )
        signals.append(
            {
                "signal_id": signal_id,
                "feature_hash": feature_hash,
                "pair": pair,
                "decision_index": index,
                "decision_time": state["decision_time"],
                "fill_index": index + 1,
                **state,
            }
        )
    return signals


def opposite(side):
    return "SHORT" if side == "LONG" else "LONG"


def gate_signal(config_id, signal, threshold):
    pe_high = signal["path_efficiency"] >= threshold["pe_q67"]
    rv_middle = (
        threshold["rv_q33"]
        <= signal["realized_energy"]
        < threshold["rv_q67"]
    )
    rv_high = signal["realized_energy"] >= threshold["rv_q67"]
    trend = signal["trend_side"]
    contra = opposite(trend)
    if config_id == "C0":
        return trend
    if config_id == "C1":
        return contra
    if config_id == "C2" and pe_high:
        return contra
    if config_id == "C3" and pe_high:
        return contra
    if config_id == "C4" and signal["session"] == "ASIA" and pe_high and rv_middle:
        return contra
    if (
        config_id == "C5"
        and signal["session"] == "LONDON"
        and pe_high
        and rv_middle
    ):
        return contra
    if (
        config_id == "C6"
        and signal["break_kind"] == "REJECTION"
        and signal["break_side"] == contra
        and pe_high
        and rv_middle
    ):
        return signal["break_side"]
    if (
        config_id == "C7"
        and signal["break_kind"] == "ACCEPTANCE"
        and signal["break_side"] == trend
        and pe_high
        and rv_high
    ):
        return signal["break_side"]
    return None


def _mtm_points(pair, rows, side, fill_index, exit_index, entry_mid, entry_exec):
    points = []
    raw_mfe = -math.inf
    raw_mae = math.inf
    for index in range(fill_index, exit_index + 1):
        row = rows[index]
        mid_close = midpoint(row, "c")
        exec_close = row["bid"]["c"] if side == "LONG" else row["ask"]["c"]
        raw_mark = path_pips(pair, side, entry_mid, mid_close)
        observed_mark = path_pips(pair, side, entry_exec, exec_close)
        mid_favorable = midpoint(row, "h") if side == "LONG" else midpoint(row, "l")
        mid_adverse = midpoint(row, "l") if side == "LONG" else midpoint(row, "h")
        raw_mfe = max(raw_mfe, path_pips(pair, side, entry_mid, mid_favorable))
        raw_mae = min(raw_mae, path_pips(pair, side, entry_mid, mid_adverse))
        points.append(
            {
                "time": iso_utc(close_time(row)),
                "raw": raw_mark,
                "base": observed_mark - 0.6,
                "adverse": observed_mark - 1.8,
            }
        )
    return points, raw_mfe, raw_mae


def replay_config(pair, rows, source_signals, thresholds, config_id, end):
    horizon = 24 if config_id in ("C0", "C1", "C2") else 48
    gated = []
    trades = []
    busy_until = -1
    collision_skips = 0
    gap_unscorable = 0
    gap_signal_ids = []

    for signal in source_signals:
        threshold = thresholds[pair][signal["session"]]
        side = gate_signal(config_id, signal, threshold)
        if side is None:
            continue
        gated_signal = dict(signal, config_id=config_id, side=side)
        gated.append(gated_signal)
        fill_index = signal["fill_index"]
        if fill_index <= busy_until:
            collision_skips += 1
            continue
        planned_exit = fill_index + horizon - 1
        exit_index = min(planned_exit, end - 1)
        if not is_consecutive(rows, signal["decision_index"], exit_index):
            gap_unscorable += 1
            gap_signal_ids.append(signal["signal_id"])
            continue
        entry = rows[fill_index]
        exit_row = rows[exit_index]
        entry_mid = midpoint(entry, "o")
        exit_mid = midpoint(exit_row, "c")
        entry_exec = entry["ask"]["o"] if side == "LONG" else entry["bid"]["o"]
        exit_exec = exit_row["bid"]["c"] if side == "LONG" else exit_row["ask"]["c"]
        raw_pips = path_pips(pair, side, entry_mid, exit_mid)
        observed_pips = path_pips(pair, side, entry_exec, exit_exec)
        mtm, raw_mfe, raw_mae = _mtm_points(
            pair,
            rows,
            side,
            fill_index,
            exit_index,
            entry_mid,
            entry_exec,
        )
        terminal = planned_exit >= end
        trade_id = sha_bytes(
            f"{signal['signal_id']}|{config_id}|{side}".encode("utf-8")
        )
        trades.append(
            {
                "trade_id": trade_id,
                "signal_id": signal["signal_id"],
                "pair": pair,
                "config_id": config_id,
                "side": side,
                "session": signal["session"],
                "break_kind": signal["break_kind"],
                "decision_index": signal["decision_index"],
                "decision_time": signal["decision_time"],
                "entry_index": fill_index,
                "entry_time": entry["time"],
                "exit_index": exit_index,
                "exit_time": iso_utc(close_time(exit_row)),
                "exit_reason": "TERMINAL_LIQUIDATION" if terminal else f"FIXED_H{horizon}_CLOSE",
                "terminal_liquidation": terminal,
                "age_bars": exit_index - fill_index + 1,
                "units": 1000,
                "entry_mid": entry_mid,
                "entry_executable": entry_exec,
                "exit_mid": exit_mid,
                "exit_executable": exit_exec,
                "raw_pips": raw_pips,
                "base_pips": observed_pips - 0.6,
                "adverse_pips": observed_pips - 1.8,
                "roundtrip_spread_pips": raw_pips - observed_pips,
                "raw_mfe_pips": raw_mfe,
                "raw_mae_pips": raw_mae,
                "direction_correct": raw_pips > 0.0,
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
        "gap_signal_ids": gap_signal_ids,
    }


def lineage_payload(trades):
    return [
        {
            "signal_id": trade["signal_id"],
            "trade_id": trade["trade_id"],
            "pair": trade["pair"],
            "side": trade["side"],
            "decision_time": trade["decision_time"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "entry_index": trade["entry_index"],
            "exit_index": trade["exit_index"],
        }
        for trade in sorted(
            trades,
            key=lambda item: (
                item["decision_time"],
                item["pair"],
                item["signal_id"],
            ),
        )
    ]


def signal_set_hash(signals):
    return sha_bytes(canonical(sorted(signal["signal_id"] for signal in signals)))


class JpyConverter:
    def __init__(self, usd_jpy_rows):
        # A candle close is causal only at row-open + 5m. Using the provider's
        # candle-open timestamp here would make the closing conversion rate
        # visible five minutes early.
        self.times = [close_time(row) for row in usd_jpy_rows]
        self.rates = [midpoint(row, "c") for row in usd_jpy_rows]
        self.cache = {}

    def pnl(self, pair, pips, when):
        quote_amount = 1000.0 * pips * pip_size(pair)
        if pair.endswith("_JPY"):
            return quote_amount
        if when not in self.cache:
            timestamp = parse_time(when)
            position = bisect.bisect_right(self.times, timestamp) - 1
            if position < 0:
                raise ValueError("no causal USD_JPY conversion rate")
            self.cache[when] = self.rates[position]
        return quote_amount * self.cache[when]


def _group_summary(trades, scenario, field):
    grouped = {}
    for trade in trades:
        grouped.setdefault(trade[field], []).append(trade[f"{scenario}_pips"])
    return {
        key: {
            "trades": len(values),
            "expectancy_pips": statistics.mean(values),
            "median_pips": statistics.median(values),
            "positive_rate": sum(value > 0.0 for value in values) / len(values),
        }
        for key, values in sorted(grouped.items())
    }


def cluster_summary(daily_values, selection):
    """Return the preregistered one-sided family-corrected daily LCB."""
    daily_means = [statistics.mean(values) for _, values in sorted(daily_values.items())]
    family_size = selection["family_size"]
    family_alpha = selection["family_alpha"]
    family_z = statistics.NormalDist().inv_cdf(
        1.0 - family_alpha / family_size
    )
    cluster_mean = statistics.mean(daily_means) if daily_means else None
    cluster_se = (
        statistics.stdev(daily_means) / math.sqrt(len(daily_means))
        if len(daily_means) > 1
        else None
    )
    lcb = (
        cluster_mean - family_z * cluster_se
        if cluster_mean is not None and cluster_se is not None
        else None
    )
    return {
        "daily_means": daily_means,
        "family_size": family_size,
        "family_alpha": family_alpha,
        "family_critical_z": family_z,
        "cluster_mean": cluster_mean,
        "cluster_standard_error": cluster_se,
        "family_adjusted_lcb": lcb,
    }


def metrics(
    source_signals,
    gated_signals,
    trades,
    diagnostics,
    scenario,
    converter,
    selection,
):
    values = [trade[f"{scenario}_pips"] for trade in trades]
    trade_count = len(trades)
    daily = {}
    for trade, value in zip(trades, values):
        daily.setdefault(trade["decision_time"][:10], []).append(value)
    cluster = cluster_summary(daily, selection)
    daily_means = cluster["daily_means"]
    family_z = cluster["family_critical_z"]
    cluster_mean = cluster["cluster_mean"]
    cluster_se = cluster["cluster_standard_error"]
    lcb = cluster["family_adjusted_lcb"]

    lineage = lineage_payload(trades)
    lineage_sha = sha_bytes(canonical(lineage))
    pnl_by_trade = {
        trade["trade_id"]: converter.pnl(
            trade["pair"], trade[f"{scenario}_pips"], trade["exit_time"]
        )
        for trade in trades
    }
    mark_events = {}
    exit_events = {}
    for trade in trades:
        exit_events.setdefault(trade["exit_time"], []).append(trade)
        for point in trade["mtm_points"]:
            mark_events.setdefault(point["time"], []).append(
                (
                    trade["trade_id"],
                    converter.pnl(trade["pair"], point[scenario], point["time"]),
                )
            )

    initial_equity = 200000.0
    realized = 0.0
    active = {}
    peak = initial_equity
    max_drawdown = 0.0
    ruin_time = None
    month_end_equity = {}
    monthly_realized = {}
    for when in sorted(set(mark_events) | set(exit_events)):
        for trade_id, mark in mark_events.get(when, []):
            active[trade_id] = mark
        marked_equity = initial_equity + realized + sum(active.values())
        peak = max(peak, marked_equity)
        max_drawdown = min(max_drawdown, marked_equity / peak - 1.0)
        if ruin_time is None and marked_equity <= 0.0:
            ruin_time = when
        for trade in exit_events.get(when, []):
            trade_pnl = pnl_by_trade[trade["trade_id"]]
            realized += trade_pnl
            active.pop(trade["trade_id"], None)
            month = when[:7]
            monthly_realized[month] = monthly_realized.get(month, 0.0) + trade_pnl
        settled_equity = initial_equity + realized + sum(active.values())
        peak = max(peak, settled_equity)
        max_drawdown = min(max_drawdown, settled_equity / peak - 1.0)
        if ruin_time is None and settled_equity <= 0.0:
            ruin_time = when
        month_end_equity[when[:7]] = settled_equity

    monthly_multiples = {}
    prior_equity = initial_equity
    for month, end_equity in sorted(month_end_equity.items()):
        monthly_multiples[month] = end_equity / prior_equity if prior_equity > 0 else None
        prior_equity = end_equity
    valid_months = [
        value for value in monthly_multiples.values() if value is not None
    ]
    sorted_values = sorted(values)
    tail_count = max(1, math.ceil(0.05 * trade_count)) if trade_count else 0
    cvar = statistics.mean(sorted_values[:tail_count]) if tail_count else None
    pair_summary = _group_summary(trades, scenario, "pair")
    session_summary = _group_summary(trades, scenario, "session")
    break_summary = _group_summary(trades, scenario, "break_kind")
    gated_days = {signal["decision_time"][:10] for signal in gated_signals}
    source_days = {signal["decision_time"][:10] for signal in source_signals}
    ages = [trade["age_bars"] for trade in trades]
    final_equity = initial_equity + sum(pnl_by_trade.values())
    raw_expectancy = (
        statistics.mean(trade["raw_pips"] for trade in trades)
        if trades
        else 0.0
    )
    expectancy = statistics.mean(values) if values else 0.0
    cost_drag = raw_expectancy - expectancy
    per_pair_trade_floor = selection["density"]["per_pair_trades_gte"]

    return {
        "scenario": scenario,
        "source_signals": len(source_signals),
        "source_signal_set_sha256": signal_set_hash(source_signals),
        "source_signals_per_utc_decision_day": (
            len(source_signals) / len(source_days) if source_days else 0.0
        ),
        "gated_signals": len(gated_signals),
        "gated_signal_set_sha256": signal_set_hash(gated_signals),
        "gated_signals_per_utc_decision_day": (
            len(gated_signals) / len(gated_days) if gated_days else 0.0
        ),
        "trades": trade_count,
        "lineage_sha256": lineage_sha,
        "collision_skips": diagnostics["collision_skips"],
        "data_gap_unscorable": diagnostics["gap_unscorable"],
        "gap_signal_ids_sha256": sha_bytes(
            canonical(sorted(diagnostics["gap_signal_ids"]))
        ),
        "direction_accuracy": (
            sum(trade["direction_correct"] for trade in trades) / trade_count
            if trade_count
            else 0.0
        ),
        "expectancy_pips": expectancy,
        "gross_expectancy_pips": raw_expectancy,
        "break_even_roundtrip_cost_pips": raw_expectancy,
        "realized_cost_drag_pips": cost_drag,
        "cost_coverage_ratio": (
            raw_expectancy / cost_drag if cost_drag > 0.0 else None
        ),
        "utc_day_cluster_mean_pips": cluster_mean,
        "utc_day_cluster_median_pips": (
            statistics.median(daily_means) if daily_means else None
        ),
        "family_adjusted_lcb_pips": lcb,
        "family_critical_z": family_z,
        "family_size": cluster["family_size"],
        "family_alpha": cluster["family_alpha"],
        "utc_day_cluster_standard_error_pips": cluster_se,
        "n_eff_utc_day_clusters": len(daily_means),
        "positive_utc_day_rate": (
            sum(value > 0.0 for value in daily_means) / len(daily_means)
            if daily_means
            else 0.0
        ),
        "mfe_mean_pips": (
            statistics.mean(trade["raw_mfe_pips"] for trade in trades)
            if trades
            else 0.0
        ),
        "mae_mean_pips": (
            statistics.mean(trade["raw_mae_pips"] for trade in trades)
            if trades
            else 0.0
        ),
        "trade_cvar_5pct_pips": cvar,
        "pair_results": pair_summary,
        "session_results": session_summary,
        "break_state_results": break_summary,
        "pairs_with_at_least_30_trades": sum(
            summary["trades"] >= 30 for summary in pair_summary.values()
        ),
        "per_pair_trade_floor": per_pair_trade_floor,
        "pairs_meeting_per_pair_trade_floor": sum(
            summary["trades"] >= per_pair_trade_floor
            for summary in pair_summary.values()
        ),
        "pairs_with_positive_expectancy": sum(
            summary["expectancy_pips"] > 0.0 for summary in pair_summary.values()
        ),
        "turnover_units": 2 * 1000 * trade_count,
        "inventory_age_q50_bars": quantile(ages, 0.50) if ages else 0.0,
        "inventory_age_q90_bars": quantile(ages, 0.90) if ages else 0.0,
        "inventory_age_q99_bars": quantile(ages, 0.99) if ages else 0.0,
        "inventory_age_max_bars": max(ages, default=0),
        "terminal_liquidations": sum(
            trade["terminal_liquidation"] for trade in trades
        ),
        "terminal_liquidation_pips": sum(
            trade[f"{scenario}_pips"]
            for trade in trades
            if trade["terminal_liquidation"]
        ),
        "terminal_open_inventory": len(active),
        "equity_multiple": final_equity / initial_equity,
        "return_on_initial_equity": final_equity / initial_equity - 1.0,
        "final_equity_jpy": final_equity,
        "max_drawdown": max_drawdown,
        "drawdown_basis": "completed_bar_portfolio_mtm_including_open_inventory",
        "equity_ruin": ruin_time is not None,
        "equity_ruin_time": ruin_time,
        "monthly_multiples": monthly_multiples,
        "monthly_realized_change_on_initial_equity": {
            month: 1.0 + pnl / initial_equity
            for month, pnl in sorted(monthly_realized.items())
        },
        "monthly_multiple_std": (
            statistics.pstdev(valid_months) if len(valid_months) > 1 else None
        ),
        "monthly_2x_count": sum(value >= 2.0 for value in valid_months),
    }


def split_result(
    data,
    signals_by_pair,
    thresholds,
    config_id,
    split_end,
    converter,
    selection,
):
    per_pair = {
        pair: replay_config(
            pair,
            data[pair],
            signals_by_pair[pair],
            thresholds,
            config_id,
            split_end[pair],
        )
        for pair in sorted(data)
    }
    source_signals = []
    gated_signals = []
    trades = []
    diagnostics = {
        "collision_skips": 0,
        "gap_unscorable": 0,
        "gap_signal_ids": [],
    }
    for pair in sorted(per_pair):
        replay = per_pair[pair]
        source_signals.extend(replay["source_signals"])
        gated_signals.extend(replay["gated_signals"])
        trades.extend(replay["trades"])
        diagnostics["collision_skips"] += replay["collision_skips"]
        diagnostics["gap_unscorable"] += replay["gap_unscorable"]
        diagnostics["gap_signal_ids"].extend(replay["gap_signal_ids"])
    trades.sort(key=lambda item: (item["exit_time"], item["pair"], item["trade_id"]))
    scenario_metrics = {
        scenario: metrics(
            source_signals,
            gated_signals,
            trades,
            diagnostics,
            scenario,
            converter,
            selection,
        )
        for scenario in SCENARIOS
    }
    lineage_hashes = {
        scenario: scenario_metrics[scenario]["lineage_sha256"]
        for scenario in SCENARIOS
    }
    if len(set(lineage_hashes.values())) != 1:
        raise AssertionError("execution arms do not share identical lineage")
    return {
        "scenario_metrics": scenario_metrics,
        "shared_lineage_sha256": next(iter(lineage_hashes.values())),
        "lineage_by_scenario": lineage_hashes,
        "same_signal_and_trade_path_all_scenarios": True,
    }


def density_gates(metric, selection):
    density = selection["density"]
    gates = {
        "executed_trades_gte_threshold": metric["trades"]
        >= density["executed_trades_gte"],
        "utc_decision_day_clusters_gte_threshold": metric[
            "n_eff_utc_day_clusters"
        ]
        >= density["utc_decision_day_clusters_gte"],
        "pairs_meeting_per_pair_trade_floor_gte_threshold": metric[
            "pairs_meeting_per_pair_trade_floor"
        ]
        >= density["pairs_with_at_least_30_trades_gte"],
    }
    return gates, all(gates.values())


def _ranking_key(config_id, configs):
    raw = configs[config_id]["tuning"]["scenario_metrics"]["raw"]
    lcb = raw["family_adjusted_lcb_pips"]
    median = raw["utc_day_cluster_median_pips"]
    return (
        -(lcb if lcb is not None else -math.inf),
        -raw["expectancy_pips"],
        -(median if median is not None else -math.inf),
        -raw["n_eff_utc_day_clusters"],
        config_id,
    )


def select_config(configs, config_ids, selection):
    """Select using tuning RAW only, with density fallback fixed by prereg."""
    density_candidates = []
    for config_id in config_ids:
        raw = configs[config_id]["tuning"]["scenario_metrics"]["raw"]
        _, passes = density_gates(raw, selection)
        if passes:
            density_candidates.append(config_id)
    ranking_pool = density_candidates or list(config_ids)
    selected = sorted(
        ranking_pool, key=lambda config_id: _ranking_key(config_id, configs)
    )[0]
    return selected, density_candidates


def gross_period_gate(metric, selection, gross_spec):
    density, density_pass = density_gates(metric, selection)
    required = gross_spec["each_period"]
    gates = {
        **density,
        "raw_family_adjusted_lcb_gt_threshold": (
            metric["family_adjusted_lcb_pips"] is not None
            and metric["family_adjusted_lcb_pips"]
            > required["raw_family_adjusted_lcb_pips_gt"]
        ),
        "raw_expectancy_gt_threshold": metric["expectancy_pips"]
        > required["raw_expectancy_pips_gt"],
        "raw_utc_day_median_gt_threshold": (
            metric["utc_day_cluster_median_pips"] is not None
            and metric["utc_day_cluster_median_pips"]
            > required["raw_utc_day_cluster_median_pips_gt"]
        ),
        "positive_pairs_gte_threshold": metric["pairs_with_positive_expectancy"]
        >= required["raw_pairs_with_positive_expectancy_gte"],
    }
    return gates, density_pass and all(gates.values())


def classify_candidate(tuning_metrics, development_metrics, selection, gross_spec):
    tuning_gates, tuning_pass = gross_period_gate(
        tuning_metrics["raw"], selection, gross_spec
    )
    development_gates, development_pass = gross_period_gate(
        development_metrics["raw"], selection, gross_spec
    )
    gross_gate = tuning_pass and development_pass
    base_spec = gross_spec["base_executable_candidate_opened_development"]
    base_candidate = (
        gross_gate
        and development_metrics["base"]["expectancy_pips"]
        > base_spec["base_expectancy_pips_gt"]
        and development_metrics["base"]["family_adjusted_lcb_pips"] is not None
        and development_metrics["base"]["family_adjusted_lcb_pips"]
        > base_spec["base_family_adjusted_lcb_pips_gt"]
    )
    stress_spec = gross_spec["stress_robust_candidate_opened_development"]
    stress_candidate = (
        base_candidate
        and development_metrics["adverse"]["expectancy_pips"]
        > stress_spec["adverse_expectancy_pips_gt"]
        and development_metrics["adverse"]["family_adjusted_lcb_pips"] is not None
        and development_metrics["adverse"]["family_adjusted_lcb_pips"]
        > stress_spec["adverse_family_adjusted_lcb_pips_gt"]
    )
    if not gross_gate:
        classification = "REJECTED_NO_GROSS_EDGE"
    elif not base_candidate:
        classification = "GROSS_ONLY_COST_BOUND"
    elif stress_candidate:
        classification = "STRESS_ROBUST_CANDIDATE"
    else:
        classification = "BASE_EXECUTABLE_CANDIDATE"
    return {
        "classification": classification,
        "tuning_gates": tuning_gates,
        "tuning_pass": tuning_pass,
        "development_gates": development_gates,
        "development_pass": development_pass,
        "gross_gate": gross_gate,
        "base_candidate": base_candidate,
        "stress_candidate": stress_candidate,
    }


def main(write=True):
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    contract_checks = validate_preregistered_contract(prereg)
    prereg_sha = sha_file(PREREG)
    v1_hashes = verify_v1_immutable(prereg)
    data = load_inputs(prereg)
    states = {pair: compute_states(rows) for pair, rows in data.items()}
    split_spec = prereg["inputs"]["split"]
    calibration_fraction = split_spec["calibration_fraction"]
    tuning_end_fraction = calibration_fraction + split_spec["tuning_fraction"]
    calibration_ends = {
        pair: int(len(rows) * calibration_fraction) for pair, rows in data.items()
    }
    tuning_ends = {
        pair: int(len(rows) * tuning_end_fraction) for pair, rows in data.items()
    }
    thresholds = derive_thresholds(
        states,
        calibration_ends,
        prereg["features"]["minimum_calibration_rows_per_pair_session"],
        prereg["features"]["calibration_quantile_low"],
        prereg["features"]["calibration_quantile_high"],
    )
    threshold_sha = sha_bytes(canonical(thresholds))
    tuning_signals = {
        pair: make_source_signals(
            pair,
            data[pair],
            states[pair],
            calibration_ends[pair],
            tuning_ends[pair],
            prereg_sha,
        )
        for pair in data
    }
    development_signals = {
        pair: make_source_signals(
            pair,
            data[pair],
            states[pair],
            tuning_ends[pair],
            len(data[pair]),
            prereg_sha,
        )
        for pair in data
    }
    converter = JpyConverter(data["USD_JPY"])
    configs = {}
    expected_ids = [f"C{index}" for index in range(8)]
    if sorted(prereg["configs"]) != expected_ids:
        raise ValueError("preregistered config family is not exactly C0-C7")
    for config_id in expected_ids:
        tuning = split_result(
            data,
            tuning_signals,
            thresholds,
            config_id,
            tuning_ends,
            converter,
            prereg["selection"],
        )
        development = split_result(
            data,
            development_signals,
            thresholds,
            config_id,
            {pair: len(rows) for pair, rows in data.items()},
            converter,
            prereg["selection"],
        )
        tune_density, tune_density_pass = density_gates(
            tuning["scenario_metrics"]["raw"], prereg["selection"]
        )
        dev_density, dev_density_pass = density_gates(
            development["scenario_metrics"]["raw"], prereg["selection"]
        )
        configs[config_id] = {
            "definition": prereg["configs"][config_id],
            "tuning": tuning,
            "opened_development": development,
            "density": {
                "tuning": tune_density,
                "tuning_pass": tune_density_pass,
                "opened_development": dev_density,
                "opened_development_pass": dev_density_pass,
            },
        }

    selected, density_candidates = select_config(
        configs, expected_ids, prereg["selection"]
    )
    selected_tuning = configs[selected]["tuning"]["scenario_metrics"]
    selected_development = configs[selected]["opened_development"][
        "scenario_metrics"
    ]
    classification_result = classify_candidate(
        selected_tuning,
        selected_development,
        prereg["selection"],
        prereg["gross_edge_gate"],
    )
    tuning_gates = classification_result["tuning_gates"]
    tuning_gross_pass = classification_result["tuning_pass"]
    development_gates = classification_result["development_gates"]
    development_gross_pass = classification_result["development_pass"]
    gross_gate = classification_result["gross_gate"]
    base_candidate = classification_result["base_candidate"]
    stress_candidate = classification_result["stress_candidate"]
    classification = classification_result["classification"]

    result = {
        "schema_version": 1,
        "candidate_id": prereg["candidate_id"],
        "status": "UNADMITTED_OPENED_DEVELOPMENT_RESULT",
        "development_classification": classification,
        "prereg_sha256": prereg_sha,
        "script_sha256": sha_file(Path(__file__)),
        "preregistered_contract_checks": contract_checks,
        "v1_immutable_hashes_verified": v1_hashes,
        "input_verification": {
            pair: {
                "sha256": sha_file(Path(prereg["inputs"]["files"][pair]["path"])),
                "rows": len(rows),
                "first": rows[0]["time"],
                "last": rows[-1]["time"],
            }
            for pair, rows in data.items()
        },
        "split_indices": {
            pair: {
                "calibration_end_exclusive": calibration_ends[pair],
                "tuning_end_exclusive": tuning_ends[pair],
                "development_end_exclusive": len(data[pair]),
            }
            for pair in data
        },
        "calibration_thresholds": thresholds,
        "calibration_thresholds_sha256": threshold_sha,
        "source_signal_counts": {
            "tuning": {pair: len(signals) for pair, signals in tuning_signals.items()},
            "opened_development": {
                pair: len(signals) for pair, signals in development_signals.items()
            },
        },
        "configs": configs,
        "selected_config": selected,
        "selection_basis": "tuning_RAW_SIGNAL_only_8_family_daily_cluster_Bonferroni",
        "selection_density_candidates": density_candidates,
        "selected_tuning": selected_tuning,
        "selected_opened_development": selected_development,
        "gross_edge_gates": {
            "tuning": tuning_gates,
            "tuning_pass": tuning_gross_pass,
            "opened_development": development_gates,
            "opened_development_pass": development_gross_pass,
            "both_periods_pass": gross_gate,
        },
        "base_executable_candidate": base_candidate,
        "stress_robust_candidate": stress_candidate,
        "admission": False,
        "holdout": False,
        "opened_development_only": True,
        "profit_unproven": True,
        "shadow_challenger_eligible": gross_gate,
        "external_orders": 0,
        "live_authority": False,
        "network_attempts": 0,
        "credential_reads": 0,
    }
    if write:
        RESULT.write_bytes(
            json.dumps(result, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        )
        packet = {
            "schema_version": 1,
            "candidate_id": prereg["candidate_id"],
            "status": "UNADMITTED_OPENED_DEVELOPMENT_RESULT",
            "development_classification": classification,
            "selected_config": selected,
            "exact_config": prereg["configs"][selected],
            "selection_basis": result["selection_basis"],
            "calibration_thresholds_sha256": threshold_sha,
            "selected_tuning": selected_tuning,
            "selected_opened_development": selected_development,
            "gross_edge_gates": result["gross_edge_gates"],
            "admission": False,
            "holdout": False,
            "opened_development_only": True,
            "profit_unproven": True,
            "shadow_challenger_eligible": gross_gate,
            "authority": prereg["authority"],
            "external_orders": 0,
            "prereg_sha256": prereg_sha,
            "script_sha256": result["script_sha256"],
            "test_sha256": sha_file(ROOT / "test_replay_m5_interaction.py"),
            "readme_sha256": sha_file(ROOT / "README.md"),
            "result_sha256": sha_file(RESULT),
            "v1_immutable_hashes_verified": v1_hashes,
            "inputs": prereg["inputs"]["files"],
        }
        PACKET.write_bytes(
            json.dumps(packet, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()
    output = main(not arguments.no_write)
    print(
        json.dumps(
            {
                "selected_config": output["selected_config"],
                "development_classification": output[
                    "development_classification"
                ],
                "admission": output["admission"],
                "opened_development": output["selected_opened_development"],
            },
            sort_keys=True,
        )
    )
