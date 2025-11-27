"""
analytics.cost_guard
~~~~~~~~~~~~~~~~~~~~
Track realised round-trip costs (spread + commission + slippage) expressed in
pips.  Maker型ワーカーで 1 pip 利益を狙う際、コスト c の推定値が条件を満たす
タイミングのみエントリーするための補助を行う。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional
import json

PIP_VALUE = 0.01
_MAX_SAMPLES = 720  # ≒12h worth at 60s cadence


@dataclass(slots=True)
class CostSample:
    epoch_ts: float
    monotonic_ts: float
    units: int
    c_pips: float
    spread_pips: float
    commission_pips: float
    slippage_pips: float
    source: str = "unknown"


_SAMPLES: Deque[CostSample] = deque(maxlen=_MAX_SAMPLES)


def reset() -> None:
    _SAMPLES.clear()


def record(
    *,
    epoch_ts: float,
    units: int,
    spread_pips: float,
    commission_pips: float,
    slippage_pips: float,
    source: str = "unknown",
) -> CostSample:
    units = int(units)
    abs_units = abs(units)
    if abs_units == 0:
        raise ValueError("units must be non-zero")
    c_pips = spread_pips + commission_pips + abs(slippage_pips)
    sample = CostSample(
        epoch_ts=float(epoch_ts),
        monotonic_ts=time.monotonic(),
        units=units,
        c_pips=float(c_pips),
        spread_pips=float(spread_pips),
        commission_pips=float(commission_pips),
        slippage_pips=float(slippage_pips),
        source=source,
    )
    _SAMPLES.append(sample)
    return sample


def _window(window_sec: float) -> Iterable[CostSample]:
    if window_sec <= 0:
        return tuple(_SAMPLES)
    cutoff = time.monotonic() - window_sec
    return tuple(sample for sample in _SAMPLES if sample.monotonic_ts >= cutoff)


def mean_cost(window_sec: float = 600.0) -> Optional[float]:
    samples = list(_window(window_sec))
    if not samples:
        return None
    return sum(s.c_pips for s in samples) / len(samples)


def percentile_cost(pct: float, window_sec: float = 600.0) -> Optional[float]:
    samples = sorted(_window(window_sec), key=lambda s: s.c_pips)
    if not samples:
        return None
    pct = max(0.0, min(100.0, float(pct)))
    if len(samples) == 1 or pct == 100.0:
        return samples[-1].c_pips
    rank = pct / 100.0 * (len(samples) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(samples) - 1)
    frac = rank - lower
    lower_val = samples[lower].c_pips
    upper_val = samples[upper].c_pips
    return lower_val * (1.0 - frac) + upper_val * frac


def allow_entry(max_cost_pips: float, window_sec: float = 300.0) -> tuple[bool, str]:
    if max_cost_pips <= 0.0:
        return False, "max_cost<=0"
    estimate = mean_cost(window_sec)
    if estimate is None:
        return False, "no_cost_samples"
    if estimate <= max_cost_pips:
        return True, f"avg_cost={estimate:.2f}<=limit {max_cost_pips:.2f}"
    return False, f"avg_cost={estimate:.2f}>limit {max_cost_pips:.2f}"


def snapshot(window_sec: float = 600.0) -> Dict[str, float]:
    samples = list(_window(window_sec))
    if not samples:
        return {"count": 0}
    avg = sum(s.c_pips for s in samples) / len(samples)
    pct95 = percentile_cost(95.0, window_sec=window_sec) or avg
    return {
        "count": len(samples),
        "avg_cost_pips": avg,
        "p95_cost_pips": pct95,
        "max_cost_pips": max(s.c_pips for s in samples),
        "min_cost_pips": min(s.c_pips for s in samples),
    }


def extract_from_transaction(payload: Dict[str, object]) -> Optional[CostSample]:
    """
    Attempt to derive a CostSample from an OANDA transaction payload.
    """

    if (payload or {}).get("type") != "ORDER_FILL":
        return None
    epoch_ts = time.time()
    ts_raw = payload.get("time")
    if isinstance(ts_raw, str):
        try:
            iso = ts_raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            epoch_ts = dt.timestamp()
        except Exception:
            pass
    try:
        units = int(float(payload.get("units", 0)))
    except Exception:
        return None
    abs_units = abs(units)
    if abs_units == 0:
        return None
    half_spread_cost = float(payload.get("halfSpreadCost", 0.0))
    commission = float(payload.get("commission", 0.0))
    total_spread_cost = abs(half_spread_cost) * 2.0
    pip_value = abs_units * PIP_VALUE
    spread_pips = total_spread_cost / pip_value if pip_value else 0.0
    commission_pips = abs(commission) / pip_value if pip_value else 0.0
    price_exec = float(payload.get("price", 0.0))
    reference = price_exec
    full_price = payload.get("fullPrice") or {}
    if units > 0:
        # long: executed at ask, compare against best ask
        asks = full_price.get("asks") or []
        if asks:
            try:
                reference = float(asks[0].get("price", price_exec))
            except Exception:
                reference = price_exec
    else:
        bids = full_price.get("bids") or []
        if bids:
            try:
                reference = float(bids[0].get("price", price_exec))
            except Exception:
                reference = price_exec
    slippage_pips = (price_exec - reference) / PIP_VALUE if reference else 0.0
    return record(
        epoch_ts=epoch_ts,
        units=units,
        spread_pips=spread_pips,
        commission_pips=commission_pips,
        slippage_pips=slippage_pips,
        source="transaction",
    )


def bootstrap_from_logs(
    pattern: str,
    *,
    max_files: int = 4,
    max_lines: int = 500,
) -> int:
    """
    Load transaction entries from jsonl logs matching ``pattern``.

    Parameters
    ----------
    pattern:
        Glob pattern, e.g. ``logs/oanda/transactions_*.jsonl``.
    max_files:
        Number of newest files to inspect.
    max_lines:
        Number of trailing lines per file.

    Returns
    -------
    int
        Count of cost samples ingested.
    """

    if max_files <= 0 or max_lines <= 0:
        return 0
    loaded = 0
    paths = sorted(Path().glob(pattern), reverse=True)
    for path in paths[:max_files]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        slice_start = max(0, len(lines) - max_lines)
        for line in lines[slice_start:]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            sample = extract_from_transaction(payload)
            if sample:
                loaded += 1
    return loaded


def latest_sample_age_sec() -> Optional[float]:
    """Return seconds since the newest sample, or None when empty."""

    if not _SAMPLES:
        return None
    newest = max(_SAMPLES, key=lambda sample: sample.monotonic_ts)
    return max(0.0, time.time() - newest.epoch_ts)
