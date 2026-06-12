from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from quant_rabbit.analysis.candles import fetch_candles_via_client


# Projection candle truth is read-only broker evidence, but it runs inside the
# unattended 20-minute live cycle. Forty-five seconds allows several normal
# OANDA REST reads or three default socket timeouts, then yields partial truth
# so the cycle can continue and surface the degraded evidence. This is an
# infrastructure watchdog budget, not a market-derived trading threshold; tune
# it from live endpoint latency telemetry if that becomes available.
DEFAULT_PROJECTION_CANDLE_TRUTH_BUDGET_SECONDS = 45.0


@dataclass(frozen=True)
class ProjectionCandleTruth:
    candles_by_pair: dict[str, dict[str, list[Any]]] | None
    candle_counts: dict[str, int]
    candle_granularity_counts: dict[str, dict[str, int]]
    candle_errors: dict[str, str]
    deadline_exceeded: bool
    budget_seconds: float
    elapsed_seconds: float


def projection_candle_truth_budget_seconds() -> float:
    raw = os.environ.get("QR_PROJECTION_VERIFY_CANDLE_BUDGET_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_PROJECTION_CANDLE_TRUTH_BUDGET_SECONDS
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError("QR_PROJECTION_VERIFY_CANDLE_BUDGET_SECONDS must be a positive number of seconds") from exc
    if seconds <= 0:
        raise ValueError("QR_PROJECTION_VERIFY_CANDLE_BUDGET_SECONDS must be a positive number of seconds")
    return seconds


def load_projection_candle_truth(
    client: Any,
    verification_pairs: Iterable[str],
    *,
    m1_count: int,
    m5_count: int,
    budget_seconds: float | None = None,
    fetcher: Callable[..., Any] | None = None,
) -> ProjectionCandleTruth:
    budget = projection_candle_truth_budget_seconds() if budget_seconds is None else float(budget_seconds)
    if budget <= 0:
        raise ValueError("projection candle truth budget must be a positive number of seconds")
    fetch = fetcher or fetch_candles_via_client
    started = time.monotonic()
    deadline = started + budget
    candles_by_pair: dict[str, dict[str, list[Any]]] = {}
    candle_counts: dict[str, int] = {}
    candle_granularity_counts: dict[str, dict[str, int]] = {}
    candle_errors: dict[str, str] = {}
    deadline_exceeded = False

    for pair in sorted({str(item) for item in verification_pairs if str(item or "").strip()}):
        pair_candles: dict[str, list[Any]] = {}
        per_granularity: dict[str, int] = {}
        stop_pair = False
        for granularity, count in (("M1", int(m1_count)), ("M5", int(m5_count))):
            if count <= 0:
                continue
            now = time.monotonic()
            if now >= deadline:
                deadline_exceeded = True
                candle_errors["_deadline"] = _deadline_message(
                    pair=pair,
                    granularity=granularity,
                    budget_seconds=budget,
                    elapsed_seconds=now - started,
                )
                stop_pair = True
                break
            try:
                candles = list(fetch(client, pair, granularity, count=count))
            except Exception as exc:  # noqa: BLE001 - degraded truth is reported below.
                candle_errors[f"{pair}:{granularity}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            pair_candles[granularity] = candles
            per_granularity[granularity] = len(candles)
            after = time.monotonic()
            if after >= deadline:
                deadline_exceeded = True
                candle_errors["_deadline"] = _deadline_message(
                    pair=pair,
                    granularity=granularity,
                    budget_seconds=budget,
                    elapsed_seconds=after - started,
                )
                stop_pair = True
                break
        if pair_candles:
            candles_by_pair[pair] = pair_candles
            candle_counts[pair] = sum(per_granularity.values())
            candle_granularity_counts[pair] = per_granularity
        if stop_pair:
            break

    elapsed = round(time.monotonic() - started, 3)
    return ProjectionCandleTruth(
        candles_by_pair=candles_by_pair or None,
        candle_counts=candle_counts,
        candle_granularity_counts=candle_granularity_counts,
        candle_errors=candle_errors,
        deadline_exceeded=deadline_exceeded,
        budget_seconds=budget,
        elapsed_seconds=elapsed,
    )


def projection_candle_truth_summary(result: ProjectionCandleTruth) -> dict[str, Any]:
    return {
        "candle_counts": result.candle_counts,
        "candle_granularity_counts": result.candle_granularity_counts,
        "candle_errors": result.candle_errors,
        "candle_truth_deadline_exceeded": result.deadline_exceeded,
        "candle_truth_budget_seconds": result.budget_seconds,
        "candle_truth_elapsed_seconds": result.elapsed_seconds,
    }


def _deadline_message(*, pair: str, granularity: str, budget_seconds: float, elapsed_seconds: float) -> str:
    return (
        "projection candle truth budget exceeded "
        f"after {elapsed_seconds:.1f}s/{budget_seconds:.1f}s at {pair}:{granularity}; "
        "continuing with partial candle truth plus quote/ATR verification"
    )
