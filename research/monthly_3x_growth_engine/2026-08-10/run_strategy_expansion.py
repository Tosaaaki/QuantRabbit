#!/usr/bin/env python3
"""Bounded independent strategy-family replay on archived OANDA M5 bid/ask."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE / "strategy_expansion_preregister_v1.json"
SEED = 20260810


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def utc(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    head, sep, tail = value.partition(".")
    if sep:
        digits, offset = tail.split("+", 1)
        value = f"{head}.{digits[:6]}+{offset}"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def crosses_financing(entry: datetime, exit_: datetime) -> bool:
    day = entry.date()
    while day <= exit_.date():
        boundary = datetime(day.year, day.month, day.day, 21, tzinfo=timezone.utc)
        if entry < boundary <= exit_:
            return True
        day += timedelta(days=1)
    return False


def lcb(values: list[float], key: str) -> float | None:
    if not values:
        return None
    seed = SEED ^ int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(2000)])
    return float(np.quantile(means, 0.025))


def metrics(values: list[float], key: str) -> dict[str, Any]:
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(values),
        "net_jpy_per_1000u": sum(values),
        "expectancy_jpy_per_1000u": float(np.mean(values)) if values else None,
        "profit_factor": None if not values or (loss == 0 and gain == 0) else (math.inf if loss == 0 else gain / loss),
        "bootstrap_lcb_expectancy_jpy": lcb(values, key),
        "max_drawdown_jpy_per_1000u": drawdown,
        "win_rate": sum(value > 0 for value in values) / len(values) if values else None,
    }


def load_bars(path: Path, holdout_start: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            match = re.search(r'"time"\s*:\s*"([^"]+)"', line)
            if match is None:
                raise ValueError(f"M5 row has no time field: {path}")
            ts = utc(match.group(1))
            if ts >= holdout_start:
                break
            raw = json.loads(line)
            if raw.get("complete") is not True:
                continue
            rows.append(
                {
                    "time": ts,
                    "bid": raw["bid"],
                    "ask": raw["ask"],
                    "mid": {key: 0.5 * (float(raw["bid"][key]) + float(raw["ask"][key])) for key in "ohlc"},
                }
            )
    return rows


def in_session(ts: datetime, session: str) -> bool:
    if session == "ALL":
        return True
    start, end = (int(value) for value in session.split("_"))
    return start <= ts.hour < end


def consecutive(times: list[datetime], start: int, end: int) -> bool:
    return start >= 0 and end < len(times) and times[end] - times[start] == timedelta(minutes=5 * (end - start))


def signal(
    family: str,
    index: int,
    lookback: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
) -> str | None:
    prior_high = float(high[index - lookback:index].max())
    prior_low = float(low[index - lookback:index].min())
    current = float(close[index])
    if family == "BREAKOUT_CONTINUATION":
        if current > prior_high and sma20[index] > sma50[index]:
            return "LONG"
        if current < prior_low and sma20[index] < sma50[index]:
            return "SHORT"
    elif family == "LIQUIDITY_SWEEP_REVERSAL":
        if low[index] < prior_low and current > prior_low:
            return "LONG"
        if high[index] > prior_high and current < prior_high:
            return "SHORT"
    elif family == "RANGE_REVERSION":
        width = prior_high - prior_low
        if width <= 0:
            return None
        position = (current - prior_low) / width
        path = float(np.abs(np.diff(close[index - lookback:index + 1])).sum())
        efficiency = abs(current - close[index - lookback]) / path if path > 0 else 0.0
        if position <= 0.15 and efficiency <= 0.30:
            return "LONG"
        if position >= 0.85 and efficiency <= 0.30:
            return "SHORT"
    elif family == "COMPRESSION_BREAKOUT":
        compression_high = float(high[index - 6:index].max())
        compression_low = float(low[index - 6:index].min())
        broad = prior_high - prior_low
        compressed = compression_high - compression_low
        if broad <= 0 or compressed / broad > 0.35:
            return None
        if current > compression_high and sma20[index] > sma50[index]:
            return "LONG"
        if current < compression_low and sma20[index] < sma50[index]:
            return "SHORT"
    elif family == "SMA_PULLBACK_CONTINUATION":
        aligned_long = bool(np.all(sma20[index - lookback + 1:index + 1] > sma50[index - lookback + 1:index + 1]))
        aligned_short = bool(np.all(sma20[index - lookback + 1:index + 1] < sma50[index - lookback + 1:index + 1]))
        if aligned_long and close[index - 1] <= sma20[index - 1] and current > sma20[index] and current > close[index - 1]:
            return "LONG"
        if aligned_short and close[index - 1] >= sma20[index - 1] and current < sma20[index] and current < close[index - 1]:
            return "SHORT"
    return None


def replay_config(
    bars: list[dict[str, Any]],
    family: str,
    lookback: int,
    horizon: int,
    session: str,
    slip_multiple: float,
) -> list[dict[str, Any]]:
    times = [row["time"] for row in bars]
    close = np.asarray([row["mid"]["c"] for row in bars])
    high = np.asarray([row["mid"]["h"] for row in bars])
    low = np.asarray([row["mid"]["l"] for row in bars])
    sma20 = np.full(len(close), np.nan)
    sma50 = np.full(len(close), np.nan)
    sma20[19:] = np.convolve(close, np.ones(20) / 20, mode="valid")
    sma50[49:] = np.convolve(close, np.ones(50) / 50, mode="valid")
    trades: list[dict[str, Any]] = []
    next_free = 0
    minimum = max(50, lookback) + 1
    for index in range(minimum, len(bars) - horizon - 1):
        entry_index = index + 1
        exit_index = entry_index + horizon
        if entry_index < next_free or not in_session(times[entry_index], session):
            continue
        if not consecutive(times, index - max(50, lookback), exit_index):
            continue
        side = signal(family, index, lookback, close, high, low, sma20, sma50)
        if side is None or crosses_financing(times[entry_index], times[exit_index]):
            continue
        entry_bid = float(bars[entry_index]["bid"]["o"])
        entry_ask = float(bars[entry_index]["ask"]["o"])
        exit_bid = float(bars[exit_index]["bid"]["o"])
        exit_ask = float(bars[exit_index]["ask"]["o"])
        entry_extra = slip_multiple * (entry_ask - entry_bid)
        exit_extra = slip_multiple * (exit_ask - exit_bid)
        if side == "LONG":
            pnl = ((exit_bid - exit_extra) - (entry_ask + entry_extra)) * 1000.0
        else:
            pnl = ((entry_bid - entry_extra) - (exit_ask + exit_extra)) * 1000.0
        trades.append(
            {
                "signal_time": times[index],
                "entry_time": times[entry_index],
                "exit_time": times[exit_index],
                "side": side,
                "pnl_jpy_per_1000u": pnl,
            }
        )
        next_free = exit_index + 1
    return trades


def adjacent(a: tuple[int, int], b: tuple[int, int], lookbacks: list[int], horizons: list[int]) -> bool:
    return abs(lookbacks.index(a[0]) - lookbacks.index(b[0])) + abs(horizons.index(a[1]) - horizons.index(b[1])) == 1


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    holdout = utc(prereg["time_split"]["holdout_start_utc"])
    sources: dict[str, list[dict[str, Any]]] = {}
    for pair, source in prereg["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit(f"source SHA mismatch: {path}")
        sources[pair] = load_bars(path, holdout)

    families = list(prereg["families"])
    lookbacks = [int(value) for value in prereg["grid"]["lookback"]]
    horizons = [int(value) for value in prereg["grid"]["horizon_bars"]]
    sessions = list(prereg["grid"]["entry_session_utc"])
    slips = [float(value) for value in prereg["grid"]["slippage_spread_multiple"]]
    rows: list[dict[str, Any]] = []
    trade_cache: dict[tuple[str, str, int, int, str, float], list[dict[str, Any]]] = {}

    for pair, bars in sources.items():
        for family in families:
            for lookback in lookbacks:
                for horizon in horizons:
                    for session in sessions:
                        for slip in slips:
                            key = (pair, family, lookback, horizon, session, slip)
                            trade_cache[key] = replay_config(bars, family, lookback, horizon, session, slip)

    for days in prereg["time_split"]["windows_days"]:
        window = f"{days}D"
        start = holdout - timedelta(days=days)
        split_at = start + timedelta(days=days * 0.60)
        validation_start = split_at + timedelta(hours=1)
        for key, trades in trade_cache.items():
            pair, family, lookback, horizon, session, slip = key
            for split, left, right in (
                ("TRAIN", start, split_at),
                ("VALIDATION", validation_start, holdout),
            ):
                selected = [
                    trade for trade in trades
                    if left <= trade["entry_time"] and trade["exit_time"] < right
                ]
                values = [float(trade["pnl_jpy_per_1000u"]) for trade in selected]
                row = {
                    "window": window,
                    "split": split,
                    "pair": pair,
                    "family": family,
                    "lookback": lookback,
                    "horizon_bars": horizon,
                    "entry_session_utc": session,
                    "slippage_spread_multiple": slip,
                    "partition_from_utc": left.isoformat().replace("+00:00", "Z"),
                    "partition_to_utc": right.isoformat().replace("+00:00", "Z"),
                    **metrics(values, ":".join(map(str, (window, split, *key)))),
                }
                rows.append(row)

    primary = float(prereg["grid"]["selection_cost_scenario"])
    connected: set[tuple[str, str, str, int, int, str]] = set()
    for window in (f"{days}D" for days in prereg["time_split"]["windows_days"]):
        for pair in sources:
            for family in families:
                for session in sessions:
                    candidates = [
                        row for row in rows
                        if row["window"] == window and row["split"] == "TRAIN"
                        and row["pair"] == pair and row["family"] == family
                        and row["entry_session_utc"] == session
                        and row["slippage_spread_multiple"] == primary
                        and row["trades"] >= 20
                        and row["expectancy_jpy_per_1000u"] is not None
                        and row["expectancy_jpy_per_1000u"] > 0
                        and row["profit_factor"] is not None and row["profit_factor"] > 1
                        and row["bootstrap_lcb_expectancy_jpy"] is not None
                        and row["bootstrap_lcb_expectancy_jpy"] > 0
                    ]
                    points = [(row["lookback"], row["horizon_bars"]) for row in candidates]
                    for point in points:
                        if any(adjacent(point, other, lookbacks, horizons) for other in points if other != point):
                            connected.add((window, pair, family, point[0], point[1], session))

    for row in rows:
        key = (row["window"], row["pair"], row["family"], row["lookback"], row["horizon_bars"], row["entry_session_utc"])
        row["train_connected_plateau"] = key in connected and row["slippage_spread_multiple"] == primary
        row["validation_pass"] = bool(
            row["split"] == "VALIDATION"
            and row["train_connected_plateau"]
            and row["trades"] >= 10
            and row["expectancy_jpy_per_1000u"] is not None
            and row["expectancy_jpy_per_1000u"] > 0
            and row["profit_factor"] is not None and row["profit_factor"] > 1
            and row["bootstrap_lcb_expectancy_jpy"] is not None
            and row["bootstrap_lcb_expectancy_jpy"] > 0
        )

    accepted = [row for row in rows if row["validation_pass"]]
    family_signs: dict[tuple[str, str, str, int, int, str], dict[str, float]] = {}
    for row in accepted:
        key = (row["pair"], row["family"], row["lookback"], row["horizon_bars"], row["entry_session_utc"], row["slippage_spread_multiple"])
        family_signs.setdefault(key, {})[row["window"]] = row["expectancy_jpy_per_1000u"]
    stable_32_64 = [
        {"pair": key[0], "family": key[1], "lookback": key[2], "horizon_bars": key[3], "entry_session_utc": key[4], "slippage_spread_multiple": key[5], "expectancy_by_window": values}
        for key, values in family_signs.items()
        if values.get("32D", 0) > 0 and values.get("64D", 0) > 0
    ]

    report = {
        "contract": prereg["contract"],
        "preregister_sha256": digest(PREREG),
        "holdout_used": False,
        "source_hashes_verified": True,
        "source_rows_before_holdout": {pair: len(bars) for pair, bars in sources.items()},
        "grid_rows": len(rows),
        "train_connected_parameter_count": len(connected),
        "validation_pass_count": len(accepted),
        "stable_32d_64d_count": len(stable_32_64),
        "stable_32d_64d": stable_32_64,
        "accepted_rows": accepted,
        "conclusion": "INDEPENDENT_EDGE_FOUND" if stable_32_64 else "NO_STABLE_INDEPENDENT_EDGE_YET",
    }

    (HERE / "strategy_expansion_grid_v1.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    (HERE / "strategy_expansion_report_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    manifest = {
        "contract": prereg["contract"],
        "preregister_sha256": digest(PREREG),
        "outputs": {
            name: digest(HERE / name)
            for name in ("strategy_expansion_grid_v1.jsonl", "strategy_expansion_report_v1.json")
        },
    }
    (HERE / "strategy_expansion_manifest_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
