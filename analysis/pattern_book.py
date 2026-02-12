"""Utilities for trade pattern bucketing and action scoring."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_token(value: Any, *, default: str = "na", max_len: int = 32) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    token = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if not token:
        return default
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    head = token[: max(8, max_len - 9)]
    return f"{head}_{digest}"


def _range_bucket(entry_thesis: Mapping[str, Any]) -> str:
    explicit = _norm_token(entry_thesis.get("entry_range_bucket"), default="")
    if explicit:
        return explicit
    axis = entry_thesis.get("section_axis")
    if not isinstance(axis, Mapping):
        return "na"
    high = _as_float(axis.get("high"))
    low = _as_float(axis.get("low"))
    entry_ref = _as_float(entry_thesis.get("entry_ref"))
    if entry_ref is None:
        for key in ("entry_price", "ideal_entry", "entry_mean"):
            entry_ref = _as_float(entry_thesis.get(key))
            if entry_ref is not None:
                break
    if high is None or low is None or entry_ref is None or high <= low:
        return "na"
    pos = (entry_ref - low) / (high - low)
    if pos <= 0.2:
        return "bot"
    if pos <= 0.4:
        return "low"
    if pos <= 0.6:
        return "mid"
    if pos <= 0.8:
        return "high"
    return "top"


def _strategy_tag(entry_thesis: Mapping[str, Any], fallback: str) -> str:
    for key in ("strategy_tag", "strategy", "tag"):
        token = _norm_token(entry_thesis.get(key), default="")
        if token:
            return token
    return _norm_token(fallback, default="unknown")


def _direction(entry_thesis: Mapping[str, Any], units: int) -> str:
    if units > 0:
        return "long"
    if units < 0:
        return "short"
    side = _norm_token(entry_thesis.get("side"), default="")
    if side in {"buy", "open_long"}:
        return "long"
    if side in {"sell", "open_short"}:
        return "short"
    return "unknown"


def build_pattern_id(
    *,
    entry_thesis: Mapping[str, Any] | None,
    units: int,
    pocket: str = "",
    strategy_tag_fallback: str = "",
) -> str:
    thesis = entry_thesis if isinstance(entry_thesis, Mapping) else {}
    strategy = _strategy_tag(thesis, strategy_tag_fallback)
    side = _direction(thesis, units)
    signal_mode = _norm_token(
        thesis.get("signal_mode") or thesis.get("entry_mode") or thesis.get("mode"),
    )
    mtf_gate = _norm_token(thesis.get("mtf_regime_gate") or thesis.get("mtf_gate"))
    horizon_gate = _norm_token(thesis.get("horizon_gate"))
    extrema_reason = _norm_token(thesis.get("extrema_gate_reason"))
    pattern_tag = _norm_token(thesis.get("pattern_tag"))
    range_bucket = _range_bucket(thesis)
    pocket_token = _norm_token(pocket, default="unknown")
    return "|".join(
        (
            f"st:{strategy}",
            f"pk:{pocket_token}",
            f"sd:{side}",
            f"sg:{signal_mode}",
            f"mtf:{mtf_gate}",
            f"hz:{horizon_gate}",
            f"ex:{extrema_reason}",
            f"rg:{range_bucket}",
            f"pt:{pattern_tag}",
        )
    )


@dataclass(slots=True)
class PatternAggregate:
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_pips: float
    total_pips: float
    gross_profit: float
    gross_loss: float
    profit_factor: float


@dataclass(slots=True)
class PatternAction:
    action: str
    lot_multiplier: float
    reason: str


def classify_pattern_action(
    aggregate: PatternAggregate,
    *,
    min_samples_soft: int = 30,
    min_samples_block: int = 120,
) -> PatternAction:
    if aggregate.trades < min_samples_soft:
        return PatternAction("learn_only", 1.0, "insufficient_samples")
    if (
        aggregate.trades >= min_samples_block
        and aggregate.avg_pips <= -0.12
        and aggregate.profit_factor < 0.90
        and aggregate.win_rate < 0.44
    ):
        return PatternAction("block", 0.0, "persistent_negative_expectancy")
    if aggregate.avg_pips <= -0.05 or aggregate.profit_factor < 0.95:
        return PatternAction("reduce", 0.75, "weak_expectancy")
    if (
        aggregate.avg_pips >= 0.18
        and aggregate.win_rate >= 0.64
        and aggregate.profit_factor >= 1.35
    ):
        return PatternAction("boost", 1.18, "strong_edge")
    if (
        aggregate.avg_pips >= 0.10
        and aggregate.win_rate >= 0.60
        and aggregate.profit_factor >= 1.20
    ):
        return PatternAction("boost", 1.12, "good_edge")
    if (
        aggregate.avg_pips >= 0.05
        and aggregate.win_rate >= 0.56
        and aggregate.profit_factor >= 1.05
    ):
        return PatternAction("boost", 1.06, "mild_edge")
    return PatternAction("neutral", 1.0, "no_clear_edge")
