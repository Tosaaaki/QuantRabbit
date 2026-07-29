"""Causal adapter for the frozen legacy M1Scalper signal.

The strategy and indicator implementations are byte-for-byte copies from
commit ``d8f751afc`` except for two reviewed causal-port edits in the strategy:
the config path is pinned to a repository file and the session multiplier
reads the completed bar's UTC hour instead of the process wall clock.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from quant_rabbit.legacy_m1_frozen import (
    SOURCE_COMMIT,
    SOURCE_INDICATOR_SHA256,
    SOURCE_STRATEGY_SHA256,
)
from quant_rabbit.legacy_m1_frozen.calc_core_d8f751afc import IndicatorEngine
from quant_rabbit.legacy_m1_frozen.m1_scalper_d8f751afc import M1Scalper


SIGNAL_CONTRACT = "QR_DOJO_LEGACY_M1_SIGNAL_PORT_V1"
PORT_REPAIRS = (
    "PIN_CONFIG_PATH",
    "BAR_UTC_HOUR_REPLACES_PROCESS_WALL_CLOCK",
)


class LegacyM1SignalError(ValueError):
    """Raised when a bar could violate the causal signal contract."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise LegacyM1SignalError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LegacyM1SignalError(f"{name} must be finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise LegacyM1SignalError(f"{name} must be positive and finite")
    return number


def normalize_completed_bar(bar: dict[str, Any]) -> dict[str, Any]:
    """Return the MID OHLC row consumed by the historical indicator engine."""

    epoch = bar.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise LegacyM1SignalError("completed bar epoch must be a positive integer")
    bid = {key: _finite(f"bid_{key}", bar.get(f"bid_{key}")) for key in "ohlc"}
    ask = {key: _finite(f"ask_{key}", bar.get(f"ask_{key}")) for key in "ohlc"}
    if any(ask[key] < bid[key] for key in "ohlc"):
        raise LegacyM1SignalError("ask must not be below bid")
    row = {
        "epoch": epoch,
        "timestamp": datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
        **{name: (bid[key] + ask[key]) / 2.0 for name, key in zip(("open", "high", "low", "close"), "ohlc")},
    }
    if row["high"] < max(row["open"], row["close"]) or row["low"] > min(
        row["open"], row["close"]
    ):
        raise LegacyM1SignalError("invalid completed bar geometry")
    return row


def build_legacy_factors(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute only from the supplied completed prefix, never a future row."""

    candles = list(rows)
    if len(candles) < 20:
        raise LegacyM1SignalError("at least 20 completed bars are required")
    epochs = [row.get("epoch") for row in candles]
    if any(
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0
        for epoch in epochs
    ):
        raise LegacyM1SignalError("factor rows require positive integer epochs")
    if any(left >= right for left, right in zip(epochs, epochs[1:])):
        raise LegacyM1SignalError("factor rows must be strictly chronological")
    frame = pd.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "open": _finite("open", row["open"]),
                "high": _finite("high", row["high"]),
                "low": _finite("low", row["low"]),
                "close": _finite("close", row["close"]),
            }
            for row in candles
        ]
    )
    factors = IndicatorEngine.compute(frame)
    latest = candles[-1]
    factors.update(
        {
            "candles": [
                {
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }
                for row in candles
            ],
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "close": latest["close"],
            "timestamp": latest["timestamp"],
            "_legacy_utc_hour": datetime.fromtimestamp(
                int(latest["epoch"]), tz=timezone.utc
            ).hour,
        }
    )
    return factors


def signal_from_completed_prefix(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    factors = build_legacy_factors(rows)
    signal = M1Scalper.check(factors)
    return dict(signal) if signal else None


class CausalM1Signal:
    """Stateful forward adapter that rejects duplicate or rewound bars."""

    def __init__(self, *, max_bars: int = 2000) -> None:
        if max_bars < 120:
            raise LegacyM1SignalError("max_bars must preserve legacy warmup")
        self._rows: deque[dict[str, Any]] = deque(maxlen=max_bars)
        self._latest_factors: dict[str, Any] | None = None

    @property
    def completed_count(self) -> int:
        return len(self._rows)

    @property
    def last_epoch(self) -> int | None:
        return int(self._rows[-1]["epoch"]) if self._rows else None

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(float(row["close"]) for row in self._rows)

    @property
    def latest_atr_pips(self) -> float:
        if self._latest_factors is None:
            return 0.0
        return float(self._latest_factors.get("atr") or 0.0) / 0.01

    def add_completed_bar(
        self, bar: dict[str, Any], *, emit_signal: bool
    ) -> dict[str, Any] | None:
        row = normalize_completed_bar(bar)
        if self._rows and row["epoch"] <= self._rows[-1]["epoch"]:
            raise LegacyM1SignalError("duplicate or rewound completed bar")
        self._rows.append(row)
        if not emit_signal or len(self._rows) < 120:
            return None
        self._latest_factors = build_legacy_factors(self._rows)
        signal = M1Scalper.check(self._latest_factors)
        return dict(signal) if signal else None

    def seed_bar(self, bar: dict[str, Any]) -> None:
        self.add_completed_bar(bar, emit_signal=False)

    def on_bar_closed(self, bar: dict[str, Any]) -> dict[str, Any] | None:
        return self.add_completed_bar(bar, emit_signal=True)

    def manifest(self) -> dict[str, Any]:
        return {
            "contract": SIGNAL_CONTRACT,
            "source_commit": SOURCE_COMMIT,
            "source_strategy_sha256": SOURCE_STRATEGY_SHA256,
            "source_indicator_sha256": SOURCE_INDICATOR_SHA256,
            "port_repairs": list(PORT_REPAIRS),
            "completed_bars_only": True,
            "future_data_allowed": False,
        }
