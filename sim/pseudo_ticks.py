"""
Pseudo tick generator that densifies S5 candles into synthetic ticks suitable
for fast scalp replays.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .pseudo_cfg import SimCfg

PIP_VALUE = 0.01  # USD/JPY pip (0.01)


def session_of(ts: datetime) -> str:
    hour = ts.hour
    if 7 <= hour < 15:
        return "asia"
    if 15 <= hour < 20:
        return "london"
    return "ny"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 1
    limit = math.exp(-lam)
    k = 0
    prod = 1.0
    while prod > limit:
        k += 1
        prod *= rng.random()
    return max(1, k - 1)


def linspace(start: float, end: float, steps: int) -> List[float]:
    if steps <= 1:
        return [end]
    delta = (end - start) / float(steps)
    return [start + delta * i for i in range(1, steps + 1)]


def _make_skeleton_path(o: float, h: float, l: float, c: float, rng: random.Random) -> List[float]:
    if rng.random() < 0.5:
        return [o, h, l, c] if c >= o else [o, l, h, c]
    return [o, l, h, c] if c >= o else [o, h, l, c]


def generate_ticks_for_candle(
    candle: Dict[str, float],
    prev_close: float,
    atr_pips: float,
    cfg: SimCfg,
    rng: random.Random,
) -> List[Dict[str, float]]:
    sess = session_of(datetime.fromtimestamp(candle["ts"] / 1000, tz=timezone.utc))
    base_lambda = {
        "london": cfg.density.tpm_5s_london,
        "ny": cfg.density.tpm_5s_ny,
        "asia": cfg.density.tpm_5s_asia,
    }[sess]
    lam = base_lambda * (cfg.density.atr_k_high if atr_pips > 0.4 else cfg.density.atr_k_low)
    n_ticks = poisson(lam, rng)

    o, h, l, c = candle["o"], candle["h"], candle["l"], candle["c"]
    skeleton = _make_skeleton_path(o, h, l, c, rng)
    segments = len(skeleton) - 1
    samples: List[float] = []
    for i in range(segments):
        steps = max(1, n_ticks // segments)
        samples.extend(linspace(skeleton[i], skeleton[i + 1], steps))
    if not samples:
        samples = [c]

    if (
        rng.random() < cfg.shape.stall_prob
        and len(samples) >= cfg.shape.stall_ticks + 2
        and len(samples) > cfg.shape.stall_ticks + 2
    ):
        start = rng.randrange(1, len(samples) - cfg.shape.stall_ticks - 1)
        center = samples[start]
        for k in range(cfg.shape.stall_ticks):
            jitter = (rng.random() * 2 - 1) * cfg.shape.stall_range_pips * PIP_VALUE
            samples[start + k] = clamp(center + jitter, l, h)

    if (
        rng.random() < cfg.shape.impulse_prob
        and len(samples) >= cfg.shape.impulse_ticks + 2
        and len(samples) > cfg.shape.impulse_ticks + 2
    ):
        start = rng.randrange(1, len(samples) - cfg.shape.impulse_ticks - 1)
        direction = 1 if c >= o else -1
        step = cfg.shape.impulse_atr_k * atr_pips * PIP_VALUE / max(1, cfg.shape.impulse_ticks)
        for k in range(cfg.shape.impulse_ticks):
            samples[start + k] = clamp(samples[start + k] + direction * step, l, h)

    for idx in range(len(samples)):
        samples[idx] += (rng.random() * 2 - 1) * cfg.shape.noise_pips_sigma * PIP_VALUE
        samples[idx] = clamp(samples[idx], min(o, h, l, c), max(o, h, l, c))

    if samples:
        samples[0] = (samples[0] + o) / 2
        samples[-1] = (samples[-1] + c) / 2

    window_ms = 5000
    timestamps = []
    elapsed = 0
    for _ in samples:
        delta_ms = max(1, int(rng.expovariate(lam / max(1.0, window_ms / 1000.0)) * 1000))
        elapsed += delta_ms
        if elapsed >= window_ms:
            elapsed = window_ms - 1
        timestamps.append(int(candle["ts"]) + elapsed)

    spread_mean = {
        "london": cfg.spread.mean_pips_london,
        "ny": cfg.spread.mean_pips_ny,
        "asia": cfg.spread.mean_pips_asia,
    }[sess]
    spread = max(0.1, rng.gauss(spread_mean, cfg.spread.std_pips))
    if sess == "asia":
        spread *= cfg.spread.night_multiplier

    ticks: List[Dict[str, float]] = []
    for ts, mid in zip(timestamps, samples):
        bid = mid - spread * PIP_VALUE / 2
        ask = mid + spread * PIP_VALUE / 2
        ticks.append({"ts": ts, "bid": bid, "ask": ask})
    return ticks


def _load_candles(path: Path) -> List[Dict[str, float]]:
    payload = json.loads(path.read_text())
    candles = payload.get("candles")
    if candles is None:
        raise ValueError(f"{path} does not contain 'candles'")
    return candles


def synth_from_candles(
    candles_path: str, out_path: str, sim_cfg: SimCfg
) -> Tuple[Path, List[Dict[str, float]]]:
    rng = random.Random(sim_cfg.random_seed)
    candles = _load_candles(Path(candles_path))

    def _to_price(value: float | str) -> float:
        return float(value)

    first = candles[0]
    mid = first.get("mid", {})
    prev_close = _to_price(mid.get("o", mid.get("c", 0.0)))
    all_ticks: List[Dict[str, float]] = []
    for candle in candles:
        mid = candle.get("mid", {})
        o = _to_price(mid.get("o", prev_close))
        h = _to_price(mid.get("h", o))
        l = _to_price(mid.get("l", o))
        c = _to_price(mid.get("c", o))
        ts_iso = candle.get("time")
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        candle_payload = {
            "ts": int(dt.timestamp() * 1000),
            "o": o,
            "h": h,
            "l": l,
            "c": c,
        }
        atr = candle.get("atr_pips")
        if atr is None:
            atr = max(0.05, (h - l) / PIP_VALUE * 0.2)
        ticks = generate_ticks_for_candle(candle_payload, prev_close, atr, sim_cfg, rng)
        all_ticks.extend(ticks)
        prev_close = c

    density_info = density_summary(all_ticks, sim_cfg)

    out = Path(out_path)
    with out.open("w", encoding="utf-8") as fh:
        for tick in all_ticks:
            fh.write(json.dumps(tick) + "\n")
    return out, density_info


def scan_tick_density(
    ticks: Iterable[Dict[str, float]], window_sec: int, min_ticks: int
) -> Tuple[int, int, float]:
    tick_list = sorted(int(t["ts"]) for t in ticks)
    if not tick_list:
        return (0, 0, 0.0)
    window_ms = window_sec * 1000
    count = 0
    meet = 0
    left = 0
    for right in range(len(tick_list)):
        while tick_list[right] - tick_list[left] > window_ms:
            left += 1
        window_count = right - left + 1
        count += 1
        if window_count >= min_ticks:
            meet += 1
    coverage = meet / count if count else 0.0
    return count, meet, coverage


def density_summary(ticks: List[Dict[str, float]], cfg: SimCfg) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for window_sec, min_ticks in cfg.density.tickrate_checks:
        count, meet, coverage = scan_tick_density(ticks, window_sec, min_ticks)
        summary.append(
            {
                "window_sec": window_sec,
                "min_ticks": min_ticks,
                "samples": count,
                "meet": meet,
                "coverage": coverage,
            }
        )
    return summary
