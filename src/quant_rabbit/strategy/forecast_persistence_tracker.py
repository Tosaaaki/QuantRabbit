"""Forecast persistence tracker — close on PERSISTENT flip, not noise.

The kill switch (`QR_DISABLE_AUTO_CLOSE=1`) blocked all auto-closes
because individual REVIEW_EXIT triggers fired on noise. This module
replaces that blunt instrument with a sharper rule:

  - Record `DirectionalForecast` for each pair every cycle to
    `data/forecast_history.jsonl`.
  - For each open position, check the last N forecast records for
    that pair.
  - If ≥ FLIP_PERSISTENCE_CYCLES consecutive forecasts have direction
    OPPOSITE the position (e.g., LONG position + DOWN forecasts) →
    `RECOMMEND_CLOSE` with reason "persistent flip".
  - If ≥ RANGE_PERSISTENCE_CYCLES consecutive forecasts are RANGE or
    UNCLEAR (eroding edge) → `RECOMMEND_CLOSE` with reason "edge lost".
  - Otherwise → `HOLD`.

Single-cycle noise (one DOWN forecast followed immediately by UP)
will NOT trigger close because persistence requires ≥3 cycles
straight. This is the discretionary trader's mental model:
「3サイクル同じ反対方向 = 本当に間違えたっぽい、損切り」.

Output is read-only (written to `data/forecast_persistence_report.json`).
The trader / GPT trader uses a fresh `RECOMMEND_CLOSE` as Gate A
evidence. Auto-close still goes through gpt_trader Gate A/B, and this
module never provides Gate B operator authorization.

Kill switch: `QR_DISABLE_FORECAST_PERSISTENCE=1`.
"""

from __future__ import annotations

import json
import os
import fcntl
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN


HISTORY_FILENAME = "forecast_history.jsonl"
FLIP_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_FLIP_PERSISTENCE", "3"))
RANGE_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_RANGE_PERSISTENCE", "5"))
HISTORY_LOOKBACK_CYCLES = int(os.environ.get("QR_FORECAST_HISTORY_LOOKBACK", "10"))


@dataclass(frozen=True)
class _HistoryKeyCacheEntry:
    stat_key: tuple[int, int, int, int]
    cycle_pairs: frozenset[tuple[str, str]]


_FORECAST_HISTORY_PROCESS_LOCK = threading.RLock()
_FORECAST_HISTORY_KEY_CACHE: dict[str, _HistoryKeyCacheEntry] = {}


@dataclass(frozen=True)
class PersistenceVerdict:
    trade_id: str
    pair: str
    side: str  # position side
    last_n_directions: tuple[str, ...]
    last_n_confidences: tuple[float, ...]
    verdict: str  # "RECOMMEND_CLOSE" | "HOLD" | "EXTEND"
    reason: str

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "last_n_directions": list(self.last_n_directions),
            "last_n_confidences": [round(c, 3) for c in self.last_n_confidences],
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_FORECAST_PERSISTENCE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def record_forecast(
    forecast: Any,
    *,
    data_root: Path,
    now: Optional[datetime] = None,
    cycle_id: str | None = None,
) -> bool:
    """Append a `DirectionalForecast` to forecast_history.jsonl.

    A cycle-level forecast is a pair fact, not a lane fact. Several runtime
    branches may ask for the same pair forecast inside one cycle, so
    `cycle_id + pair` is idempotent to keep persistence statistics honest.
    """
    if _is_disabled():
        return False
    path = data_root / HISTORY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pair = getattr(forecast, "pair", "")
    now = now or datetime.now(timezone.utc)
    entry = {
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "cycle_id": cycle_id,
        "pair": pair,
        "direction": getattr(forecast, "direction", "UNCLEAR"),
        "confidence": float(getattr(forecast, "confidence", 0)),
        "current_price": getattr(forecast, "current_price", None),
        "invalidation_price": getattr(forecast, "invalidation_price", None),
        "target_price": getattr(forecast, "target_price", None),
        "horizon_min": int(getattr(forecast, "horizon_min", 0)),
        "raw_confidence": getattr(forecast, "raw_confidence", None),
        "calibration_multiplier": getattr(forecast, "calibration_multiplier", None),
        "up_score": getattr(forecast, "up_score", None),
        "down_score": getattr(forecast, "down_score", None),
        "range_score": getattr(forecast, "range_score", None),
        "drivers_for": list(getattr(forecast, "drivers_for", ()) or ()),
        "drivers_against": list(getattr(forecast, "drivers_against", ()) or ()),
        "rationale_summary": getattr(forecast, "rationale_summary", ""),
    }
    with _FORECAST_HISTORY_PROCESS_LOCK:
        with path.open("a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                (
                    existing_cycle_pairs,
                    removed_duplicates,
                    compacted_lines,
                ) = _history_cycle_pair_index_from_handle(path, f)
                cycle_pair_key = (str(cycle_id), str(pair)) if cycle_id and pair else None
                if cycle_pair_key and cycle_pair_key in existing_cycle_pairs:
                    if removed_duplicates and compacted_lines is not None:
                        _rewrite_history_handle(f, compacted_lines)
                        f.flush()
                    _cache_forecast_history_cycle_pairs(path, f, existing_cycle_pairs)
                    return False
                line = json.dumps(entry, ensure_ascii=False)
                if removed_duplicates and compacted_lines is not None:
                    _rewrite_history_handle(f, [*compacted_lines, line])
                else:
                    f.seek(0, os.SEEK_END)
                    f.write(line + "\n")
                if cycle_pair_key:
                    existing_cycle_pairs.add(cycle_pair_key)
                f.flush()
                _cache_forecast_history_cycle_pairs(path, f, existing_cycle_pairs)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return True


def _history_cycle_pair_index_from_handle(
    path: Path,
    handle: IO[str],
) -> tuple[set[tuple[str, str]], int, list[str] | None]:
    cache_id = _forecast_history_cache_id(path)
    stat_key = _forecast_history_stat_key(handle)
    cached = _FORECAST_HISTORY_KEY_CACHE.get(cache_id)
    if cached is not None and cached.stat_key == stat_key:
        return set(cached.cycle_pairs), 0, None
    handle.seek(0)
    compacted_lines, removed_duplicates, existing_cycle_pairs = _compact_history_lines(
        handle.read().splitlines()
    )
    return set(existing_cycle_pairs), removed_duplicates, compacted_lines


def _cache_forecast_history_cycle_pairs(
    path: Path,
    handle: IO[str],
    cycle_pairs: set[tuple[str, str]],
) -> None:
    _FORECAST_HISTORY_KEY_CACHE[_forecast_history_cache_id(path)] = _HistoryKeyCacheEntry(
        stat_key=_forecast_history_stat_key(handle),
        cycle_pairs=frozenset(cycle_pairs),
    )


def _forecast_history_cache_id(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _forecast_history_stat_key(handle: IO[str]) -> tuple[int, int, int, int]:
    stat = os.fstat(handle.fileno())
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    )


def _clear_forecast_history_key_cache() -> None:
    with _FORECAST_HISTORY_PROCESS_LOCK:
        _FORECAST_HISTORY_KEY_CACHE.clear()


def _compact_history_lines(lines: list[str]) -> tuple[list[str], int, set[tuple[str, str]]]:
    """Collapse historical cycle_id/pair duplicates while preserving latest fact."""
    compacted: list[str] = []
    cycle_pair_indexes: dict[tuple[str, str], int] = {}
    removed_duplicates = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        key: tuple[str, str] | None = None
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = None
        if isinstance(item, dict):
            cycle_id = item.get("cycle_id")
            pair = item.get("pair")
            if cycle_id and pair:
                key = (str(cycle_id), str(pair))
        if key is None:
            compacted.append(line)
            continue
        previous_index = cycle_pair_indexes.get(key)
        if previous_index is None:
            cycle_pair_indexes[key] = len(compacted)
            compacted.append(line)
            continue
        compacted[previous_index] = line
        removed_duplicates += 1
    return compacted, removed_duplicates, set(cycle_pair_indexes)


def _rewrite_history_handle(handle: IO[str], lines: list[str]) -> None:
    handle.seek(0)
    handle.truncate(0)
    if not lines:
        return
    handle.write("\n".join(lines) + "\n")


def _history_has_cycle_pair(path: Path, *, cycle_id: str, pair: str) -> bool:
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("cycle_id") == cycle_id and item.get("pair") == pair:
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _history_handle_has_cycle_pair(handle: IO[str], *, cycle_id: str, pair: str) -> bool:
    """Check idempotency under the same file lock used for append."""
    handle.seek(0)
    for line in handle:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("cycle_id") == cycle_id and item.get("pair") == pair:
            return True
    return False


def _load_recent(data_root: Path, pair: str, count: int) -> List[Dict[str, Any]]:
    """Return the last `count` forecast entries for `pair`."""
    path = data_root / HISTORY_FILENAME
    if not path.exists():
        return []
    # Forecast persistence is cycle-based. TraderBrain may score several lanes
    # for the same pair inside one automation cycle; those candidate-lane
    # passes must not count as multiple "persistent" flips.
    by_cycle: dict[str, Dict[str, Any]] = {}
    cycle_order: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("pair") == pair:
                key = str(d.get("cycle_id") or d.get("timestamp_utc") or len(cycle_order))
                if key not in by_cycle:
                    cycle_order.append(key)
                by_cycle[key] = d
    except OSError:
        return []
    return [by_cycle[key] for key in cycle_order][-count:]


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def assess_position(
    *,
    trade_id: str,
    pair: str,
    side: str,
    data_root: Path,
    fresh_after_utc: datetime | None = None,
) -> PersistenceVerdict:
    """Check forecast persistence for one position.

    Pulls last HISTORY_LOOKBACK_CYCLES forecasts for the pair.
    Returns verdict based on directional consistency:
    - LONG position + last N forecasts all DOWN → RECOMMEND_CLOSE
    - LONG position + last N forecasts all UP → EXTEND (forecast confirms)
    - Mixed or insufficient history → HOLD
    """
    if _is_disabled():
        return PersistenceVerdict(
            trade_id=trade_id, pair=pair, side=side,
            last_n_directions=(), last_n_confidences=(),
            verdict="HOLD", reason="persistence tracker disabled",
        )
    side_up = side.upper()
    aligned_dir = "UP" if side_up == "LONG" else "DOWN"
    opposite_dir = "DOWN" if side_up == "LONG" else "UP"

    recent = _load_recent(data_root, pair, HISTORY_LOOKBACK_CYCLES)
    if not recent:
        return PersistenceVerdict(
            trade_id=trade_id, pair=pair, side=side_up,
            last_n_directions=(), last_n_confidences=(),
            verdict="HOLD", reason="no forecast history yet",
        )

    latest_ts = _parse_timestamp(recent[-1].get("timestamp_utc"))
    if fresh_after_utc is not None:
        if fresh_after_utc.tzinfo is None:
            fresh_after = fresh_after_utc.replace(tzinfo=timezone.utc)
        else:
            fresh_after = fresh_after_utc.astimezone(timezone.utc)
        if latest_ts is None or latest_ts < fresh_after:
            latest_text = latest_ts.isoformat() if latest_ts is not None else "unknown"
            stale_directions = tuple(str(r.get("direction", "")) for r in recent)
            stale_confidences = tuple(float(r.get("confidence", 0)) for r in recent)
            return PersistenceVerdict(
                trade_id=trade_id, pair=pair, side=side_up,
                last_n_directions=stale_directions, last_n_confidences=stale_confidences,
                verdict="HOLD",
                reason=(
                    "stale forecast history: latest forecast "
                    f"{latest_text} predates broker snapshot {fresh_after.isoformat()}; "
                    "refresh trader-brain forecast before persistence close advice"
                ),
            )
    continuous_recent = _continuous_recent_window(recent)
    if len(continuous_recent) < len(recent):
        recent = continuous_recent
        if not recent:
            return PersistenceVerdict(
                trade_id=trade_id, pair=pair, side=side_up,
                last_n_directions=(), last_n_confidences=(),
                verdict="HOLD", reason="forecast history continuity reset — need fresh consecutive cycles",
            )

    directions = tuple(str(r.get("direction", "")) for r in recent)
    confidences = tuple(float(r.get("confidence", 0)) for r in recent)

    # Persistent flip check
    flip_run = 0
    for d, confidence in reversed(tuple(zip(directions, confidences))):
        if d == opposite_dir and confidence >= ENTRY_CONFIDENCE_MIN:
            flip_run += 1
        else:
            break
    if flip_run >= FLIP_PERSISTENCE_CYCLES:
        return PersistenceVerdict(
            trade_id=trade_id, pair=pair, side=side_up,
            last_n_directions=directions, last_n_confidences=confidences,
            verdict="RECOMMEND_CLOSE",
            reason=f"last {flip_run} forecasts flipped to {opposite_dir} (position is {side_up}) — direction prediction now AGAINST",
        )

    # Persistent RANGE / UNCLEAR (edge loss)
    edge_loss_run = 0
    for d in reversed(directions):
        if d in ("RANGE", "UNCLEAR"):
            edge_loss_run += 1
        else:
            break
    if edge_loss_run >= RANGE_PERSISTENCE_CYCLES:
        return PersistenceVerdict(
            trade_id=trade_id, pair=pair, side=side_up,
            last_n_directions=directions, last_n_confidences=confidences,
            verdict="RECOMMEND_CLOSE",
            reason=f"last {edge_loss_run} forecasts went RANGE/UNCLEAR — directional edge lost, recycle capital",
        )

    # Strong alignment → EXTEND
    aligned_run = 0
    for d in reversed(directions):
        if d == aligned_dir:
            aligned_run += 1
        else:
            break
    if aligned_run >= FLIP_PERSISTENCE_CYCLES:
        return PersistenceVerdict(
            trade_id=trade_id, pair=pair, side=side_up,
            last_n_directions=directions, last_n_confidences=confidences,
            verdict="EXTEND",
            reason=f"last {aligned_run} forecasts aligned {aligned_dir} (position {side_up}) — extend TP, hold",
        )

    return PersistenceVerdict(
        trade_id=trade_id, pair=pair, side=side_up,
        last_n_directions=directions, last_n_confidences=confidences,
        verdict="HOLD",
        reason=f"mixed forecast history — recent={list(directions[-3:])}",
    )


def _continuous_recent_window(recent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop forecasts before a time discontinuity in the persistence window.

    Persistence means consecutive cycle evidence. If the forecast loop was
    down for days, old RANGE / flip records must not be stitched to one fresh
    record and treated as an unbroken run. The continuity budget is derived
    from the observed cadence and forecast horizons in the same window rather
    than a wall-clock literal.
    """
    if len(recent) < 2:
        return recent
    timestamps = [_parse_timestamp(item.get("timestamp_utc")) for item in recent]
    if any(ts is None for ts in timestamps):
        return recent
    gaps = [
        (timestamps[idx] - timestamps[idx - 1]).total_seconds()
        for idx in range(1, len(timestamps))
        if timestamps[idx] is not None
        and timestamps[idx - 1] is not None
        and (timestamps[idx] - timestamps[idx - 1]).total_seconds() > 0
    ]
    if not gaps:
        return recent
    horizons = []
    for item in recent:
        try:
            horizon_seconds = float(item.get("horizon_min") or 0.0) * 60.0
        except (TypeError, ValueError):
            horizon_seconds = 0.0
        if horizon_seconds > 0:
            horizons.append(horizon_seconds)
    cadence_seconds = _median(gaps)
    horizon_seconds = _median(horizons) if horizons else 0.0
    continuity_budget_seconds = max(cadence_seconds, horizon_seconds) * max(HISTORY_LOOKBACK_CYCLES, 1)
    if continuity_budget_seconds <= 0:
        return recent
    start_index = 0
    for idx in range(1, len(timestamps)):
        gap_seconds = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
        if gap_seconds > continuity_budget_seconds:
            start_index = idx
    return recent[start_index:]


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def assess_all_positions(
    positions: List[Any],
    *,
    data_root: Path,
    fresh_after_utc: datetime | None = None,
) -> List[PersistenceVerdict]:
    out: List[PersistenceVerdict] = []
    for p in positions:
        owner = getattr(p, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        side = getattr(p, "side", None)
        side_val = side.value if hasattr(side, "value") else str(side or "")
        out.append(assess_position(
            trade_id=str(getattr(p, "trade_id", "")),
            pair=str(getattr(p, "pair", "")),
            side=side_val,
            data_root=data_root,
            fresh_after_utc=fresh_after_utc,
        ))
    return out
