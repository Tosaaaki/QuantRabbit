#!/usr/bin/env python3
"""Evaluate local forecast profile formulas against historical M1 candles.

This script compares:
- `baseline`: legacy momentum-only expectation
- `improved`: strategy-local profile aware hybrid expectation

Data source defaults to local JSON snapshots under `logs/`.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Iterable, Optional

PIP = 0.01


@dataclass(frozen=True)
class EvalResult:
    step: int
    count: int
    hit_rate: float
    mae_pips: float


def _parse_ts(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        if "." not in text:
            return None
        head, tail = text.split(".", 1)
        sign_idx = max(tail.find("+"), tail.find("-"))
        if sign_idx >= 0:
            frac = tail[:sign_idx]
            zone = tail[sign_idx:]
            frac = (frac[:6]).ljust(6, "0")
            text2 = f"{head}.{frac}{zone}"
        else:
            frac = (tail[:6]).ljust(6, "0")
            text2 = f"{head}.{frac}"
        try:
            return datetime.fromisoformat(text2)
        except Exception:
            return None


def _to_close(candle: dict) -> Optional[float]:
    for key in ("close", "c"):
        value = candle.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    mid = candle.get("mid")
    if isinstance(mid, dict):
        value = mid.get("c")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _iter_candle_rows(paths: Iterable[str]) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        candles = payload.get("candles")
        if not isinstance(candles, list):
            continue
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            ts = _parse_ts(candle.get("time") or candle.get("timestamp"))
            if ts is None:
                continue
            close = _to_close(candle)
            if close is None:
                continue
            rows.append((ts, close))
    rows.sort(key=lambda x: x[0])
    dedup: dict[datetime, float] = {}
    for ts, close in rows:
        dedup[ts] = close
    return sorted(dedup.items(), key=lambda x: x[0])


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _baseline_expected(closes: list[float], idx: int, step: int) -> float:
    short_window = min(max(4, step + 1), idx)
    long_window = min(max(short_window + 1, step * 3), idx)
    short_drift = (closes[idx] - closes[idx - short_window]) / max(short_window, 1)
    long_drift = (closes[idx] - closes[idx - long_window]) / max(long_window, 1)
    expected_move = ((0.65 * short_drift) + (0.35 * long_drift)) * float(step)
    return expected_move / PIP


def _improved_expected(closes: list[float], idx: int, step: int) -> float:
    short_window = min(max(4, step + 1), idx)
    long_window = min(max(short_window + 1, step * 3), idx)
    short_drift = (closes[idx] - closes[idx - short_window]) / max(short_window, 1)
    long_drift = (closes[idx] - closes[idx - long_window]) / max(long_window, 1)
    close_deltas = [closes[k] - closes[k - 1] for k in range(1, idx + 1)]
    if not close_deltas:
        return _baseline_expected(closes, idx, step)
    close_vol_window = min(max(10, step * 8), len(close_deltas))
    recent_deltas = close_deltas[-close_vol_window:]
    close_vol = max(mean(abs(delta) for delta in recent_deltas), PIP * 0.12)

    if step <= 1:
        ma_window = min(max(8, step * 8), idx + 1)
        ma_price = mean(closes[idx - ma_window + 1 : idx + 1])
        deviation = closes[idx] - ma_price
        momentum_move = ((0.68 * short_drift) + (0.32 * long_drift)) * float(step)
        expected_move = momentum_move + (0.10 * close_deltas[-1]) - (0.05 * deviation)
        cap = 1.9 * close_vol * float(max(step, 1))
        expected_move = _clamp(expected_move, -cap, cap)
        return expected_move / PIP

    if step <= 5:
        ma_window = min(max(8, step * 10), idx + 1)
        ma_price = mean(closes[idx - ma_window + 1 : idx + 1])
        deviation = closes[idx] - ma_price
        signs = [1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0) for delta in recent_deltas]
        persistence = abs(sum(signs)) / float(len(signs)) if signs else 0.0
        momentum_move = ((0.70 * short_drift) + (0.30 * long_drift)) * float(step)
        reversion_move = (-deviation) * (1.0 - persistence) * 0.06
        expected_move = momentum_move + (0.12 * close_deltas[-1]) + reversion_move
        cap = 1.9 * close_vol * float(max(step, 1))
        expected_move = _clamp(expected_move, -cap, cap)
        return expected_move / PIP

    return _baseline_expected(closes, idx, step)


def _evaluate(closes: list[float], step: int, mode: str) -> EvalResult:
    hits: list[int] = []
    errors: list[float] = []
    for idx in range(15, len(closes) - step):
        if mode == "baseline":
            expected = _baseline_expected(closes, idx, step)
        else:
            expected = _improved_expected(closes, idx, step)
        realized = (closes[idx + step] - closes[idx]) / PIP
        hits.append(1 if expected * realized > 0 else 0)
        errors.append(abs(realized - expected))
    if not hits:
        return EvalResult(step=step, count=0, hit_rate=0.0, mae_pips=0.0)
    return EvalResult(
        step=step,
        count=len(hits),
        hit_rate=sum(hits) / float(len(hits)),
        mae_pips=sum(errors) / float(len(errors)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local forecast formulas.")
    parser.add_argument(
        "--patterns",
        default=(
            "logs/candles_M1*.json,"
            "logs/candles_USDJPY_M1*.json,"
            "logs/oanda/candles_M1_latest.json"
        ),
        help="Comma-separated glob patterns for M1 candle JSON files.",
    )
    parser.add_argument(
        "--steps",
        default="1,5,10",
        help="Comma-separated horizon steps in M1 bars.",
    )
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in [token.strip() for token in args.patterns.split(",") if token.strip()]:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    rows = _iter_candle_rows(paths)
    if not rows:
        print("no candle rows")
        return 1
    closes = [close for _, close in rows]
    print(f"bars={len(closes)} from={rows[0][0].isoformat()} to={rows[-1][0].isoformat()}")

    steps: list[int] = []
    for token in [token.strip() for token in args.steps.split(",") if token.strip()]:
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            steps.append(value)
    steps = sorted(set(steps))
    if not steps:
        print("no valid steps")
        return 1

    baseline_results = [_evaluate(closes, step, "baseline") for step in steps]
    improved_results = [_evaluate(closes, step, "improved") for step in steps]

    print("baseline:")
    for row in baseline_results:
        print(
            f" step={row.step} n={row.count} "
            f"hit_rate={row.hit_rate:.4f} mae_pips={row.mae_pips:.4f}"
        )
    print("improved:")
    for row in improved_results:
        print(
            f" step={row.step} n={row.count} "
            f"hit_rate={row.hit_rate:.4f} mae_pips={row.mae_pips:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

