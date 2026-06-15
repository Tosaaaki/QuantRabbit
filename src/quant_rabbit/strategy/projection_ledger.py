"""Projection ledger — record every forecast, verify outcomes, self-calibrate.

User directive 2026-05-14:「予測の精度を最大限高める。そうすれば間違えない」.

Every `ProjectionSignal` emitted by `forward_projection.py` is recorded
to `data/projection_ledger.jsonl` at emission time with:
- timestamp_emitted_utc, pair, signal_name, direction, lead_time_min
- entry_price (current price when prediction was made)
- predicted_target_price (for liquidity sweep / directional forecast) or None (for EITHER)
- predicted_invalidation_price for synthesized directional forecasts
- resolution_window_min = lead_time_min × 2 (give the move 2x slack), or a
  minimum immediate-followthrough scoring window when lead_time_min is 0
- resolution_status = "PENDING" initially

After the resolution window elapses, `verify_pending_projections()`
checks each PENDING entry against price truth. Prefer ordered candle truth
covering the emitted→expiry window, so a target that was reached and then
mean-reverted is still counted correctly. M1 is best; coarser M5 fallback is
acceptable when older M1 truth has rolled out of the broker's recent-candle
window:
- "UP" direction → did window high exceed entry by ≥ ATR_pips × 0.5?
- "DOWN" → did window low go below by ≥ same?
- "EITHER" → did the window range expand by the same ATR-based threshold?
- For liquidity_sweep: did price reach the named sweep target? The
  signal direction is the post-sweep fade / entry direction, so target
  side is derived from `signal_name` (`*_high` uses window high,
  `*_low` uses window low).
- For directional_forecast with both target and invalidation: whichever
  level touches first wins. Invalidation-first is MISS even if the target
  prints later; same-candle ambiguity is treated as MISS for calibration.

Resolved entries get tagged HIT / MISS / TIMEOUT. Rolling hit-rate per
`(signal_name, pair, regime, direction)` is then queryable by `confidence_calibration()`
which returns a multiplier on the raw confidence — when a detector
has a poor hit-rate (e.g., 30%), its confidence is dampened; when
strong (e.g., 80%), boosted. This creates a self-improving feedback
loop — the layer stops trusting detectors that don't pan out.

Storage: append-only JSONL so the ledger can be re-played and audited.
File location: `data/projection_ledger.jsonl` (gitignored).

The verifier is idempotent for scored outcomes — running it multiple times
resolves new PENDING entries and may repair truth-missing TIMEOUT entries when
historical candles become available, but it never re-scores HIT/MISS rows.
"""

from __future__ import annotations

import json
import math
import os
import fcntl
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Optional

from quant_rabbit.instruments import instrument_pip_factor


LEDGER_FILENAME = "projection_ledger.jsonl"
HIT_RATE_LOOKBACK = int(os.environ.get("QR_PROJECTION_HIT_RATE_LOOKBACK", "100"))
CONFIDENCE_MIN_SAMPLES = int(os.environ.get("QR_PROJECTION_CONFIDENCE_MIN_SAMPLES", "10"))
CONFIDENCE_DAMPING = float(os.environ.get("QR_PROJECTION_CONFIDENCE_DAMPING", "0.6"))
# Multiplier when a detector has 100% hit-rate
CONFIDENCE_MAX_MULTIPLIER = float(os.environ.get("QR_PROJECTION_CONFIDENCE_MAX_MULT", "1.5"))
# Multiplier when a detector has 0% hit-rate
CONFIDENCE_MIN_MULTIPLIER = float(os.environ.get("QR_PROJECTION_CONFIDENCE_MIN_MULT", "0.2"))
_PROJECTION_KEY_CACHE: dict[tuple[str, str, str], tuple[tuple[int, int], set[tuple]]] = {}
_HIT_RATE_CACHE: dict[
    tuple[str, int],
    tuple[tuple[int, int], Dict[str, Dict[str, Dict[str, float]]]],
] = {}
# (a) Immediate news/event follow-through signals fire after a catalyst is
#     already in the tape, so their lead time is correctly 0 minutes.
# (b) They still need a real observation window for calibration; one H1 bar is
#     the shortest common macro/technical window used by the chart packet and
#     prevents same-cycle telemetry from becoming instantly expired PENDING.
# (c) Replace with detector-specific windows once the projection ledger has
#     enough event/news-theme samples to calibrate post-catalyst half-life.
IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN = float(
    os.environ.get("QR_IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN", "60.0")
)
if IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN <= 0:
    raise ValueError("QR_IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN must be positive")
# RANGE forecast calibration checks whether the emitted box stayed intact.
# The tolerance is ATR-derived, matching the existing projection verifier's
# half-ATR movement threshold, so normal bar noise does not mislabel a held
# range as a breakout.
RANGE_FORECAST_BREAKOUT_TOLERANCE_ATR_MULT = float(
    os.environ.get("QR_RANGE_FORECAST_BREAKOUT_TOLERANCE_ATR_MULT", "0.5")
)
if RANGE_FORECAST_BREAKOUT_TOLERANCE_ATR_MULT < 0:
    raise ValueError("QR_RANGE_FORECAST_BREAKOUT_TOLERANCE_ATR_MULT must be non-negative")


@dataclass
class LedgerEntry:
    timestamp_emitted_utc: str
    pair: str
    signal_name: str
    direction: str
    lead_time_min: float
    confidence: float
    entry_price: Optional[float]
    predicted_target_price: Optional[float]
    resolution_window_min: float
    resolution_status: str  # "PENDING" | "HIT" | "MISS" | "TIMEOUT"
    predicted_invalidation_price: Optional[float] = None
    resolved_at_utc: Optional[str] = None
    resolution_evidence: str = ""
    pre_emission_range_pips: Optional[float] = None
    predicted_range_low_price: Optional[float] = None
    predicted_range_high_price: Optional[float] = None
    # 2026-05-14: regime tagging for segmented hit_rate calculation.
    # Detectors that fire in TREND regimes often perform very differently
    # from the same detector in RANGE regimes — bucketing the calibration
    # by regime makes the multiplier much more accurate.
    regime_at_emission: Optional[str] = None  # "TREND" | "RANGE" | "REVERSAL_RISK" | "UNCLEAR" | None
    cycle_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_emitted_utc": self.timestamp_emitted_utc,
            "pair": self.pair,
            "signal_name": self.signal_name,
            "direction": self.direction,
            "lead_time_min": self.lead_time_min,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "predicted_target_price": self.predicted_target_price,
            "predicted_invalidation_price": self.predicted_invalidation_price,
            "resolution_window_min": self.resolution_window_min,
            "resolution_status": self.resolution_status,
            "resolved_at_utc": self.resolved_at_utc,
            "resolution_evidence": self.resolution_evidence,
            "pre_emission_range_pips": self.pre_emission_range_pips,
            "predicted_range_low_price": self.predicted_range_low_price,
            "predicted_range_high_price": self.predicted_range_high_price,
            "regime_at_emission": self.regime_at_emission,
            "cycle_id": self.cycle_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        return cls(
            timestamp_emitted_utc=str(d.get("timestamp_emitted_utc", "")),
            pair=str(d.get("pair", "")),
            signal_name=str(d.get("signal_name", "")),
            direction=str(d.get("direction", "")),
            lead_time_min=float(d.get("lead_time_min", 0)),
            confidence=float(d.get("confidence", 0)),
            entry_price=d.get("entry_price"),
            predicted_target_price=d.get("predicted_target_price"),
            resolution_window_min=float(d.get("resolution_window_min", 0)),
            resolution_status=str(d.get("resolution_status", "PENDING")),
            predicted_invalidation_price=d.get("predicted_invalidation_price"),
            resolved_at_utc=d.get("resolved_at_utc"),
            resolution_evidence=str(d.get("resolution_evidence", "")),
            pre_emission_range_pips=d.get("pre_emission_range_pips"),
            predicted_range_low_price=d.get("predicted_range_low_price"),
            predicted_range_high_price=d.get("predicted_range_high_price"),
            regime_at_emission=d.get("regime_at_emission"),
            cycle_id=d.get("cycle_id"),
        )


def _ledger_path(data_root: Path) -> Path:
    return data_root / LEDGER_FILENAME


def record_projections(
    signals: List[Any],
    *,
    pair: str,
    current_price: Optional[float],
    data_root: Path,
    pre_emission_range_pips: Optional[float] = None,
    regime_at_emission: Optional[str] = None,
    cycle_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Append all signals to the ledger. Returns count written.

    Idempotency: when `cycle_id` is supplied, a signal with the same
    cycle/pair/name/direction/entry/target is written only once. The
    trader brain scores multiple candidate lanes per pair; without this
    key, one market prediction is counted repeatedly and corrupts hit-rate
    calibration.
    """
    if not signals:
        return 0
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat().replace("+00:00", "Z")
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        seen_keys = _existing_projection_keys_from_handle(f, cycle_id=cycle_id, pair=pair) if cycle_id else set()
        f.seek(0, os.SEEK_END)
        for s in signals:
            # Liquidity sweep signals carry an implied target price in
            # the rationale; capture it heuristically when present.
            target_price = _extract_target_price_from_rationale(getattr(s, "rationale", ""))
            key = _projection_key(
                cycle_id=cycle_id,
                pair=pair,
                signal_name=getattr(s, "name", "?"),
                direction=getattr(s, "direction", "?"),
                entry_price=current_price,
                target_price=target_price,
            )
            if cycle_id and key in seen_keys:
                continue
            lead_time_min = float(getattr(s, "lead_time_min", 0))
            entry = LedgerEntry(
                timestamp_emitted_utc=ts,
                pair=pair,
                signal_name=getattr(s, "name", "?"),
                direction=getattr(s, "direction", "?"),
                lead_time_min=lead_time_min,
                confidence=float(getattr(s, "confidence", 0)),
                entry_price=current_price,
                predicted_target_price=target_price,
                resolution_window_min=_projection_resolution_window_min(lead_time_min),
                resolution_status="PENDING",
                pre_emission_range_pips=pre_emission_range_pips,
                regime_at_emission=regime_at_emission,
                cycle_id=cycle_id,
            )
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            seen_keys.add(key)
            written += 1
        f.flush()
        _cache_projection_keys_for_handle(f, cycle_id=cycle_id, pair=pair, keys=seen_keys)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return written


def _projection_resolution_window_min(lead_time_min: float) -> float:
    if lead_time_min <= 0:
        return IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN
    return lead_time_min * 2.0


def record_directional_forecast(
    forecast: Any,
    *,
    pair: str,
    current_price: Optional[float],
    data_root: Path,
    regime_at_emission: Optional[str] = None,
    cycle_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Record the synthesized pair-level forecast in the same calibration ledger.

    Individual projection detectors already self-calibrate through this file, but
    the final `DirectionalForecast` used by trader_brain was previously written
    only to `forecast_history.jsonl`. That meant
    `confidence_calibration("directional_forecast", ...)` had no samples and
    always returned the neutral 1.0 multiplier. This entry closes that loop.
    """
    direction = str(getattr(forecast, "direction", "") or "").upper()
    if direction not in {"UP", "DOWN", "RANGE"}:
        return 0
    entry_price = current_price
    if entry_price is None:
        entry_price = getattr(forecast, "current_price", None)
    try:
        parsed_entry = float(entry_price) if entry_price is not None else None
    except (TypeError, ValueError):
        parsed_entry = None
    if parsed_entry is None or parsed_entry <= 0:
        return 0
    target_price = getattr(forecast, "target_price", None)
    try:
        parsed_target = float(target_price) if target_price is not None else None
    except (TypeError, ValueError):
        parsed_target = None
    invalidation_price = getattr(forecast, "invalidation_price", None)
    try:
        parsed_invalidation = float(invalidation_price) if invalidation_price is not None else None
    except (TypeError, ValueError):
        parsed_invalidation = None
    range_low_price = getattr(forecast, "range_low_price", None)
    range_high_price = getattr(forecast, "range_high_price", None)
    try:
        parsed_range_low = float(range_low_price) if range_low_price is not None else None
    except (TypeError, ValueError):
        parsed_range_low = None
    try:
        parsed_range_high = float(range_high_price) if range_high_price is not None else None
    except (TypeError, ValueError):
        parsed_range_high = None
    range_width_pips = getattr(forecast, "range_width_pips", None)
    try:
        parsed_range_width = float(range_width_pips) if range_width_pips is not None else None
    except (TypeError, ValueError):
        parsed_range_width = None
    if (
        (parsed_range_width is None or parsed_range_width <= 0)
        and parsed_range_low is not None
        and parsed_range_high is not None
        and parsed_range_high > parsed_range_low
    ):
        parsed_range_width = (parsed_range_high - parsed_range_low) * float(instrument_pip_factor(pair))
    try:
        horizon_min = float(getattr(forecast, "horizon_min", 0) or 0)
    except (TypeError, ValueError):
        horizon_min = 0.0
    if horizon_min <= 0:
        horizon_min = 60.0
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat().replace("+00:00", "Z")
    key = _projection_key(
        cycle_id=cycle_id,
        pair=pair,
        signal_name="directional_forecast",
        direction=direction,
        entry_price=parsed_entry,
        target_price=parsed_target,
    )
    entry = LedgerEntry(
        timestamp_emitted_utc=ts,
        pair=pair,
        signal_name="directional_forecast",
        direction=direction,
        lead_time_min=horizon_min,
        confidence=float(getattr(forecast, "confidence", 0.0) or 0.0),
        entry_price=parsed_entry,
        predicted_target_price=parsed_target if direction != "RANGE" else None,
        resolution_window_min=horizon_min,
        resolution_status="PENDING",
        predicted_invalidation_price=parsed_invalidation if direction != "RANGE" else None,
        pre_emission_range_pips=parsed_range_width if direction == "RANGE" else None,
        predicted_range_low_price=parsed_range_low if direction == "RANGE" else None,
        predicted_range_high_price=parsed_range_high if direction == "RANGE" else None,
        regime_at_emission=regime_at_emission,
        cycle_id=cycle_id,
    )
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        seen_keys = _existing_projection_keys_from_handle(f, cycle_id=cycle_id, pair=pair) if cycle_id else set()
        if cycle_id and key in seen_keys:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return 0
        f.seek(0, os.SEEK_END)
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        seen_keys.add(key)
        f.flush()
        _cache_projection_keys_for_handle(f, cycle_id=cycle_id, pair=pair, keys=seen_keys)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return 1


def _projection_key(
    *,
    cycle_id: Optional[str],
    pair: str,
    signal_name: str,
    direction: str,
    entry_price: Optional[float],
    target_price: Optional[float],
) -> tuple:
    return (
        cycle_id,
        pair,
        signal_name,
        direction,
        _round_key_price(entry_price),
        _round_key_price(target_price),
    )


def _round_key_price(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def _existing_projection_keys(data_root: Path, *, cycle_id: Optional[str], pair: str) -> set[tuple]:
    if not cycle_id:
        return set()
    keys: set[tuple] = set()
    for entry in load_ledger(data_root):
        if entry.cycle_id != cycle_id or entry.pair != pair:
            continue
        keys.add(
            _projection_key(
                cycle_id=entry.cycle_id,
                pair=entry.pair,
                signal_name=entry.signal_name,
                direction=entry.direction,
                entry_price=entry.entry_price,
                target_price=entry.predicted_target_price,
            )
        )
    return keys


def _existing_projection_keys_from_handle(
    handle: IO[str],
    *,
    cycle_id: Optional[str],
    pair: str,
) -> set[tuple]:
    if not cycle_id:
        return set()
    cache_id = _projection_key_cache_id(handle, cycle_id=cycle_id, pair=pair)
    stat_token = _projection_key_cache_stat(handle)
    cached = _PROJECTION_KEY_CACHE.get(cache_id)
    if cached is not None and cached[0] == stat_token:
        return set(cached[1])
    keys = _scan_existing_projection_keys_from_handle(handle, cycle_id=cycle_id, pair=pair)
    _PROJECTION_KEY_CACHE[cache_id] = (stat_token, set(keys))
    return keys


def _scan_existing_projection_keys_from_handle(
    handle: IO[str],
    *,
    cycle_id: Optional[str],
    pair: str,
) -> set[tuple]:
    if not cycle_id:
        return set()
    keys: set[tuple] = set()
    handle.seek(0)
    for line in handle:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(item.get("cycle_id") or "") != cycle_id or str(item.get("pair") or "") != pair:
            continue
        keys.add(
            _projection_key(
                cycle_id=cycle_id,
                pair=pair,
                signal_name=str(item.get("signal_name", "")),
                direction=str(item.get("direction", "")),
                entry_price=item.get("entry_price"),
                target_price=item.get("predicted_target_price"),
            )
        )
    return keys


def _cache_projection_keys_for_handle(
    handle: IO[str],
    *,
    cycle_id: Optional[str],
    pair: str,
    keys: set[tuple],
) -> None:
    if not cycle_id:
        return
    _PROJECTION_KEY_CACHE[_projection_key_cache_id(handle, cycle_id=cycle_id, pair=pair)] = (
        _projection_key_cache_stat(handle),
        set(keys),
    )


def _projection_key_cache_id(handle: IO[str], *, cycle_id: str, pair: str) -> tuple[str, str, str]:
    name = getattr(handle, "name", "")
    try:
        path = os.path.realpath(os.fspath(name))
    except (TypeError, ValueError):
        path = f"fd:{handle.fileno()}"
    return (path, cycle_id, pair)


def _projection_key_cache_stat(handle: IO[str]) -> tuple[int, int]:
    stat = os.fstat(handle.fileno())
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _extract_target_price_from_rationale(rationale: str) -> Optional[float]:
    """Pull a target price out of a liquidity-sweep rationale string."""
    import re
    m = re.search(r"at\s+([0-9.]+)\s*\(", rationale or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def projection_telemetry_market_open(now: Optional[datetime] = None) -> bool:
    """Return whether forward-projection telemetry can be calibrated.

    The projection ledger scores future price movement over a resolution window.
    Forecasts emitted while the FX market is closed do not have a valid tradable
    observation window, so callers should keep forecast history but skip adding
    HIT/MISS-calibrated projection rows. On status failures, stay permissive so
    telemetry collection is not disabled by an auxiliary calendar outage.
    """
    try:
        from quant_rabbit.analysis.market_status import compute_market_status

        return bool(compute_market_status(now).is_fx_open)
    except Exception:
        return True


def load_ledger(data_root: Path) -> List[LedgerEntry]:
    """Read full ledger into memory. Returns [] when file missing."""
    path = _ledger_path(data_root)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return _load_ledger_from_handle(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        return []


def _load_ledger_from_handle(handle: IO[str]) -> List[LedgerEntry]:
    out: List[LedgerEntry] = []
    handle.seek(0)
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(LedgerEntry.from_dict(d))
        except json.JSONDecodeError:
            continue
    return out


def write_ledger(entries: List[LedgerEntry], data_root: Path) -> None:
    """Overwrite ledger with the given list (used after resolution updates)."""
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            _write_ledger_to_handle(entries, f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _write_ledger_to_handle(entries: List[LedgerEntry], handle: IO[str]) -> None:
    handle.seek(0)
    handle.truncate(0)
    for e in entries:
        handle.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    handle.flush()
    _HIT_RATE_CACHE.clear()


def verify_pending(
    data_root: Path,
    *,
    quotes_by_pair: Optional[Dict[str, Dict[str, float]]] = None,
    atr_pips_by_pair: Optional[Dict[str, float]] = None,
    candles_by_pair: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _verify_pending_unlocked(
                data_root,
                quotes_by_pair=quotes_by_pair,
                atr_pips_by_pair=atr_pips_by_pair,
                candles_by_pair=candles_by_pair,
                now=now,
                ledger_handle=lock_handle,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _verify_pending_unlocked(
    data_root: Path,
    *,
    quotes_by_pair: Optional[Dict[str, Dict[str, float]]] = None,
    atr_pips_by_pair: Optional[Dict[str, float]] = None,
    candles_by_pair: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    ledger_handle: Optional[IO[str]] = None,
) -> Dict[str, int]:
    """Walk unresolved ledger entries; resolve those past their window.

    `candles_by_pair`: {pair: [Candle-like]} or
    {pair: {granularity: [Candle-like]}} is preferred and resolves against the
    high/low path inside the resolution window. Granularity buckets are tried
    from finest to coarsest so M5 only fills windows no longer covered by M1.
    Entries that previously timed out only because quote/candle truth was
    unavailable are retried when candle truth is supplied; already-scored
    HIT/MISS entries and closed-market TIMEOUT entries remain immutable.
    `quotes_by_pair`: {pair: {"bid": float, "ask": float}} is a legacy
    fallback when the caller has not supplied historical candle truth.
    `atr_pips_by_pair`: optional per-pair ATR pips for EITHER signal
    expansion check.

    Returns count summary {"HIT": n, "MISS": n, "TIMEOUT": n}.
    """
    now = now or datetime.now(timezone.utc)
    entries = _load_ledger_from_handle(ledger_handle) if ledger_handle is not None else load_ledger(data_root)
    if not entries:
        return {"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0}

    quotes_by_pair = quotes_by_pair or {}
    atr_pips_by_pair = atr_pips_by_pair or {}
    counts = {"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0}
    for e in entries:
        retrying_truth_timeout = _retryable_truth_timeout(e) and candles_by_pair is not None
        if e.resolution_status != "PENDING" and not retrying_truth_timeout:
            continue
        try:
            emitted_at = datetime.fromisoformat(e.timestamp_emitted_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if not projection_telemetry_market_open(emitted_at):
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "market closed at projection emission; excluded from calibration"
            counts["TIMEOUT"] += 1
            continue

        elapsed_min = (now - emitted_at).total_seconds() / 60.0
        if elapsed_min < e.resolution_window_min:
            counts["PENDING"] += 1
            continue

        expires_at = emitted_at + timedelta(minutes=e.resolution_window_min)
        price_path = _price_path_for_entry(e, emitted_at=emitted_at, expires_at=expires_at, candles_by_pair=candles_by_pair)
        if price_path is None:
            if candles_by_pair is not None:
                if e.resolution_status == "PENDING":
                    e.resolution_status = "TIMEOUT"
                    e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
                    e.resolution_evidence = "no candle truth for projection window"
                    counts["TIMEOUT"] += 1
                continue
            price_path = _quote_point_path(e, quotes_by_pair=quotes_by_pair)
        if price_path is None:
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "no quote/candle truth for pair at verification time"
            counts["TIMEOUT"] += 1
            continue

        entry_price = e.entry_price if e.entry_price is not None else price_path["entry"]
        if entry_price is None or entry_price <= 0:
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "missing entry_price"
            counts["TIMEOUT"] += 1
            continue

        pip_factor = float(instrument_pip_factor(e.pair))
        atr_pips = atr_pips_by_pair.get(e.pair)
        move_threshold_price = None
        needs_atr_threshold = (
            e.predicted_target_price is None
            and str(e.direction or "").upper() in {"UP", "DOWN", "EITHER"}
            and (atr_pips is None or atr_pips <= 0)
        )
        if needs_atr_threshold:
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "missing ATR threshold for projection verification"
            counts["TIMEOUT"] += 1
            continue
        if atr_pips is not None and atr_pips > 0:
            move_threshold_price = atr_pips * 0.5 / pip_factor

        if e.predicted_target_price is not None:
            # Liquidity-sweep signals store the sweep trigger price, while
            # `direction` is the executable fade direction. Verify target
            # touch by signal name so sweep_high + DOWN is valid.
            target = e.predicted_target_price
            signal_name = e.signal_name.lower()
            if signal_name == "directional_forecast" and e.predicted_invalidation_price is not None:
                ordered_outcome = _directional_target_invalidation_outcome(
                    e,
                    emitted_at=emitted_at,
                    expires_at=expires_at,
                    entry_price=entry_price,
                    candles_by_pair=candles_by_pair,
                )
                if ordered_outcome is not None:
                    status, evidence = ordered_outcome
                    e.resolution_status = status
                    e.resolution_evidence = evidence
                    counts[status] += 1
                elif _directional_invalidation_touched(e, price_path):
                    e.resolution_status = "MISS"
                    e.resolution_evidence = (
                        "target/invalidation ordering unavailable; invalidation also touched in aggregate window"
                    )
                    counts["MISS"] += 1
                elif e.direction == "UP" and price_path["high"] >= target:
                    e.resolution_status = "HIT"
                    e.resolution_evidence = f"window high {price_path['high']:.5f} reached target {target:.5f}"
                    counts["HIT"] += 1
                elif e.direction == "DOWN" and price_path["low"] <= target:
                    e.resolution_status = "HIT"
                    e.resolution_evidence = f"window low {price_path['low']:.5f} reached target {target:.5f}"
                    counts["HIT"] += 1
                else:
                    e.resolution_status = "MISS"
                    e.resolution_evidence = (
                        f"window high/low {price_path['high']:.5f}/{price_path['low']:.5f} "
                        f"did not reach target {target:.5f}"
                    )
                    counts["MISS"] += 1
            elif "liquidity_sweep_high" in signal_name:
                if price_path["high"] >= target:
                    e.resolution_status = "HIT"
                    e.resolution_evidence = f"window high {price_path['high']:.5f} reached sweep-high target {target:.5f}"
                    counts["HIT"] += 1
                else:
                    e.resolution_status = "MISS"
                    e.resolution_evidence = (
                        f"window high {price_path['high']:.5f} did not reach sweep-high target {target:.5f}"
                    )
                    counts["MISS"] += 1
            elif "liquidity_sweep_low" in signal_name:
                if price_path["low"] <= target:
                    e.resolution_status = "HIT"
                    e.resolution_evidence = f"window low {price_path['low']:.5f} reached sweep-low target {target:.5f}"
                    counts["HIT"] += 1
                else:
                    e.resolution_status = "MISS"
                    e.resolution_evidence = (
                        f"window low {price_path['low']:.5f} did not reach sweep-low target {target:.5f}"
                    )
                    counts["MISS"] += 1
            elif e.direction == "UP" and target <= entry_price:
                e.resolution_status = "MISS"
                e.resolution_evidence = f"invalid UP target geometry target {target:.5f} <= entry {entry_price:.5f}"
                counts["MISS"] += 1
            elif e.direction == "DOWN" and target >= entry_price:
                e.resolution_status = "MISS"
                e.resolution_evidence = f"invalid DOWN target geometry target {target:.5f} >= entry {entry_price:.5f}"
                counts["MISS"] += 1
            elif e.direction == "UP" and price_path["high"] >= target:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"window high {price_path['high']:.5f} reached target {target:.5f}"
                counts["HIT"] += 1
            elif e.direction == "DOWN" and price_path["low"] <= target:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"window low {price_path['low']:.5f} reached target {target:.5f}"
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = (
                    f"window high/low {price_path['high']:.5f}/{price_path['low']:.5f} "
                    f"did not reach {target:.5f}"
                )
                counts["MISS"] += 1
        elif e.signal_name == "directional_forecast" and str(e.direction or "").upper() == "RANGE":
            status, evidence = _range_forecast_outcome(
                e,
                price_path=price_path,
                atr_pips=atr_pips,
                pip_factor=pip_factor,
            )
            e.resolution_status = status
            e.resolution_evidence = evidence
            counts[status] += 1
        elif e.direction == "EITHER":
            if move_threshold_price is None or atr_pips is None:
                e.resolution_status = "TIMEOUT"
                e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
                e.resolution_evidence = "missing ATR threshold for projection verification"
                counts["TIMEOUT"] += 1
                continue
            # Volatility expansion — did the whole window expand?
            window_range = price_path["high"] - price_path["low"]
            if window_range >= move_threshold_price:
                e.resolution_status = "HIT"
                e.resolution_evidence = (
                    f"window range {window_range * pip_factor:.1f}pip ≥ "
                    f"{atr_pips * 0.5:.1f}pip threshold"
                )
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = (
                    f"window range {window_range * pip_factor:.1f}pip < threshold (no expansion)"
                )
                counts["MISS"] += 1
        else:
            if move_threshold_price is None:
                e.resolution_status = "TIMEOUT"
                e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
                e.resolution_evidence = "missing ATR threshold for projection verification"
                counts["TIMEOUT"] += 1
                continue
            # UP/DOWN directional — favorable excursion inside the window.
            move = price_path["high"] - entry_price if e.direction == "UP" else entry_price - price_path["low"]
            if e.direction == "UP":
                signed_close_move = price_path["close"] - entry_price
            else:
                signed_close_move = entry_price - price_path["close"]
            if move >= move_threshold_price:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"favorable excursion {move * pip_factor:.1f}pip toward {e.direction}"
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = (
                    f"favorable excursion {move * pip_factor:.1f}pip; "
                    f"close move {signed_close_move * pip_factor:+.1f}pip"
                )
                counts["MISS"] += 1
        e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
    if ledger_handle is not None:
        _write_ledger_to_handle(entries, ledger_handle)
    else:
        write_ledger(entries, data_root)
    return counts


def _range_forecast_outcome(
    entry: LedgerEntry,
    *,
    price_path: Dict[str, float],
    atr_pips: Optional[float],
    pip_factor: float,
) -> tuple[str, str]:
    tolerance_pips = 0.0
    if atr_pips is not None and atr_pips > 0:
        tolerance_pips = atr_pips * RANGE_FORECAST_BREAKOUT_TOLERANCE_ATR_MULT
    tolerance_price = tolerance_pips / pip_factor if pip_factor > 0 else 0.0
    range_low = entry.predicted_range_low_price
    range_high = entry.predicted_range_high_price
    try:
        low = float(range_low) if range_low is not None else None
        high = float(range_high) if range_high is not None else None
    except (TypeError, ValueError):
        low = None
        high = None
    if low is not None and high is not None and high > low:
        lower_bound = low - tolerance_price
        upper_bound = high + tolerance_price
        broke_low = price_path["low"] < lower_bound
        broke_high = price_path["high"] > upper_bound
        if broke_low or broke_high:
            return (
                "MISS",
                (
                    f"range broke: window high/low {price_path['high']:.5f}/{price_path['low']:.5f} "
                    f"exceeded box {low:.5f}-{high:.5f} plus {tolerance_pips:.1f}pip ATR tolerance"
                ),
            )
        return (
            "HIT",
            (
                f"range held: window high/low {price_path['high']:.5f}/{price_path['low']:.5f} "
                f"stayed inside box {low:.5f}-{high:.5f} plus {tolerance_pips:.1f}pip ATR tolerance"
            ),
        )

    try:
        emitted_width_pips = (
            float(entry.pre_emission_range_pips)
            if entry.pre_emission_range_pips is not None
            else None
        )
    except (TypeError, ValueError):
        emitted_width_pips = None
    if emitted_width_pips is None or emitted_width_pips <= 0:
        return "TIMEOUT", "missing emitted range bounds for RANGE forecast verification"
    window_range_pips = (price_path["high"] - price_path["low"]) * pip_factor
    allowed_pips = emitted_width_pips + tolerance_pips
    if window_range_pips <= allowed_pips:
        return (
            "HIT",
            (
                f"range held: window range {window_range_pips:.1f}pip <= "
                f"emitted range {emitted_width_pips:.1f}pip + {tolerance_pips:.1f}pip ATR tolerance"
            ),
        )
    return (
        "MISS",
        (
            f"range broke: window range {window_range_pips:.1f}pip > "
            f"emitted range {emitted_width_pips:.1f}pip + {tolerance_pips:.1f}pip ATR tolerance"
        ),
    )


def _retryable_truth_timeout(entry: LedgerEntry) -> bool:
    if str(entry.resolution_status or "").upper() != "TIMEOUT":
        return False
    evidence = str(entry.resolution_evidence or "").lower()
    return any(
        marker in evidence
        for marker in (
            "no m1 candle truth",
            "no candle truth",
            "no quote/candle truth",
        )
    )


def retryable_truth_timeout_pairs(entries: Iterable[LedgerEntry]) -> set[str]:
    return {
        str(getattr(entry, "pair", ""))
        for entry in entries
        if str(getattr(entry, "pair", "") or "").strip() and _retryable_truth_timeout(entry)
    }


def _price_path_for_entry(
    entry: LedgerEntry,
    *,
    emitted_at: datetime,
    expires_at: datetime,
    candles_by_pair: Optional[Dict[str, Any]],
) -> Optional[dict[str, float]]:
    if candles_by_pair is None:
        return None
    window = _normalised_candle_window(
        entry,
        emitted_at=emitted_at,
        expires_at=expires_at,
        candles_by_pair=candles_by_pair,
    )
    if not window:
        return None
    return {
        "entry": window[0]["close"],
        "high": max(item["high"] for item in window),
        "low": min(item["low"] for item in window),
        "close": window[-1]["close"],
    }


def _normalised_candle_window(
    entry: LedgerEntry,
    *,
    emitted_at: datetime,
    expires_at: datetime,
    candles_by_pair: Optional[Dict[str, Any]],
) -> list[dict[str, Any]]:
    if candles_by_pair is None:
        return []
    # Candles are timestamped at bar open. Include the candle that overlaps
    # the emission minute; coarser fallback candles are acceptable when older
    # M1 truth has rolled out of the broker's recent-candle window.
    overlap_start = emitted_at - timedelta(minutes=1)
    for _label, candles in _ordered_candle_sets(candles_by_pair.get(entry.pair)):
        window = []
        for c in candles:
            norm = _normalise_candle(c)
            if norm is None:
                continue
            ts = norm["timestamp"]
            if overlap_start <= ts <= expires_at:
                window.append(norm)
        window.sort(key=lambda item: item["timestamp"])
        if window:
            return window
    return []


def _ordered_candle_sets(value: Any) -> list[tuple[str, Iterable[Any]]]:
    if isinstance(value, dict):
        if any(key in value for key in ("timestamp_utc", "timestamp", "time")):
            return [("candles", [value])]
        ordered: list[tuple[str, Iterable[Any]]] = []
        seen: set[str] = set()
        for key in ("M1", "M5", "M15", "M30", "H1", "H4", "D"):
            candles = value.get(key)
            if candles is None:
                continue
            ordered.append((key, candles))
            seen.add(key)
        for key in sorted([k for k in value.keys() if str(k) not in seen], key=str):
            candles = value.get(key)
            if candles is not None:
                ordered.append((str(key), candles))
        return ordered
    if value is None:
        return []
    return [("candles", value)]


def _directional_target_invalidation_outcome(
    entry: LedgerEntry,
    *,
    emitted_at: datetime,
    expires_at: datetime,
    entry_price: float,
    candles_by_pair: Optional[Dict[str, Any]],
) -> Optional[tuple[str, str]]:
    """Resolve directional forecasts by first target/invalidation touch.

    Aggregate high/low verification inflated calibration because a target touch
    after invalidation still counted as HIT. Ordered candles are the minimum
    truth needed to learn whether the forecast was useful before it was wrong.
    """
    target = entry.predicted_target_price
    invalidation = entry.predicted_invalidation_price
    if target is None or invalidation is None:
        return None
    direction = str(entry.direction or "").upper()
    if direction == "UP":
        if target <= entry_price:
            return "MISS", f"invalid UP target geometry target {target:.5f} <= entry {entry_price:.5f}"
        if invalidation >= entry_price:
            return "MISS", f"invalid UP invalidation geometry invalidation {invalidation:.5f} >= entry {entry_price:.5f}"
    elif direction == "DOWN":
        if target >= entry_price:
            return "MISS", f"invalid DOWN target geometry target {target:.5f} >= entry {entry_price:.5f}"
        if invalidation <= entry_price:
            return "MISS", f"invalid DOWN invalidation geometry invalidation {invalidation:.5f} <= entry {entry_price:.5f}"
    else:
        return None

    window = _normalised_candle_window(
        entry,
        emitted_at=emitted_at,
        expires_at=expires_at,
        candles_by_pair=candles_by_pair,
    )
    if not window:
        return None

    for item in window:
        target_touched = (
            item["high"] >= target if direction == "UP" else item["low"] <= target
        )
        invalidation_touched = (
            item["low"] <= invalidation if direction == "UP" else item["high"] >= invalidation
        )
        ts = item["timestamp"].isoformat().replace("+00:00", "Z")
        if target_touched and invalidation_touched:
            return (
                "MISS",
                f"{ts} price candle touched target {target:.5f} and invalidation {invalidation:.5f}; "
                "ordering ambiguous, counted as MISS for calibration",
            )
        if invalidation_touched:
            return "MISS", f"{ts} invalidation {invalidation:.5f} touched before target {target:.5f}"
        if target_touched:
            return "HIT", f"{ts} target {target:.5f} touched before invalidation {invalidation:.5f}"

    return "MISS", f"target {target:.5f} not reached before invalidation {invalidation:.5f}"


def _directional_invalidation_touched(entry: LedgerEntry, price_path: dict[str, float]) -> bool:
    invalidation = entry.predicted_invalidation_price
    if invalidation is None:
        return False
    direction = str(entry.direction or "").upper()
    if direction == "UP":
        return price_path["low"] <= invalidation
    if direction == "DOWN":
        return price_path["high"] >= invalidation
    return False


def _quote_point_path(
    entry: LedgerEntry,
    *,
    quotes_by_pair: Dict[str, Dict[str, float]],
) -> Optional[dict[str, float]]:
    quote = quotes_by_pair.get(entry.pair)
    if not quote:
        return None
    try:
        current_price = (float(quote.get("bid", 0)) + float(quote.get("ask", 0))) / 2.0
    except (TypeError, ValueError):
        return None
    if current_price <= 0:
        return None
    return {
        "entry": entry.entry_price if entry.entry_price is not None else current_price,
        "high": current_price,
        "low": current_price,
        "close": current_price,
    }


def _normalise_candle(candle: Any) -> Optional[dict[str, Any]]:
    ts = getattr(candle, "timestamp_utc", None)
    high = getattr(candle, "high", None)
    low = getattr(candle, "low", None)
    close = getattr(candle, "close", None)
    if isinstance(candle, dict):
        ts = candle.get("timestamp_utc") or candle.get("timestamp") or candle.get("time")
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    try:
        high_f = float(high)
        low_f = float(low)
        close_f = float(close)
    except (TypeError, ValueError):
        return None
    if high_f <= 0 or low_f <= 0 or close_f <= 0:
        return None
    return {"timestamp": ts, "high": high_f, "low": low_f, "close": close_f}


def compute_hit_rates(
    data_root: Path,
    *,
    lookback: int = HIT_RATE_LOOKBACK,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per `(signal_name, pair, regime)` hit-rate from last `lookback`
    resolved entries.

    Returns:
        {
            "<signal_name>": {
                "<pair>:<regime>": {"hit_rate": 0.0-1.0, "samples": int},
                "<pair>:_all_regimes": {...},
                "_all_pairs:_all_regimes": {...},
            },
            ...
        }

    The hierarchical keys let `confidence_calibration` prefer the most
    specific bucket (pair × regime) but fall back to less specific
    buckets when there aren't enough samples in the granular one. Do not
    globally truncate before grouping: sparse pair/regime buckets can be
    pushed out by high-volume pairs, which makes bad local history fall back
    to broad all-pair calibration and overstate confidence.
    """
    path = _ledger_path(data_root)
    cache_id = (_projection_file_cache_id(path), int(lookback))
    stat_token = _projection_file_cache_stat(path)
    cached = _HIT_RATE_CACHE.get(cache_id)
    if cached is not None and cached[0] == stat_token:
        return _copy_hit_rates(cached[1])

    out = _compute_hit_rates_uncached(data_root, lookback=lookback)
    _HIT_RATE_CACHE[cache_id] = (stat_token, _copy_hit_rates(out))
    return out


def _compute_hit_rates_uncached(
    data_root: Path,
    *,
    lookback: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    entries = load_ledger(data_root)
    resolved = _deduped_calibration_entries([
        e for e in entries
        if e.resolution_status in ("HIT", "MISS") and _calibration_entry_eligible(e)
    ])
    grouped: Dict[str, Dict[str, List[bool]]] = {}
    for e in resolved:
        for signal_name in _calibration_signal_names(e):
            s = grouped.setdefault(signal_name, {})
            regime = e.regime_at_emission or "UNCLEAR"
            # 3-level bucketing: most specific → most general
            s.setdefault(f"{e.pair}:{regime}", []).append(e.resolution_status == "HIT")
            s.setdefault(f"{e.pair}:_all_regimes", []).append(e.resolution_status == "HIT")
            s.setdefault(f"_all_pairs:{regime}", []).append(e.resolution_status == "HIT")
            s.setdefault("_all_pairs:_all_regimes", []).append(e.resolution_status == "HIT")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sig, by_key in grouped.items():
        out[sig] = {}
        for key, results in by_key.items():
            n = min(len(results), lookback)
            recent = results[-n:]
            if not recent:
                continue
            hr = sum(1 for r in recent if r) / float(n)
            out[sig][key] = {"hit_rate": round(hr, 3), "samples": n}
    return out


def _copy_hit_rates(
    hit_rates: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        signal_name: {
            bucket: dict(metrics)
            for bucket, metrics in by_bucket.items()
        }
        for signal_name, by_bucket in hit_rates.items()
    }


def _projection_file_cache_id(path: Path) -> str:
    return os.path.realpath(os.fspath(path))


def _projection_file_cache_stat(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _deduped_calibration_entries(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Collapse historical duplicate projection writes for calibration only.

    The JSONL ledger stays append-only for audit. Confidence learning is a
    statistical consumer, so one cycle/pair/signal prediction must contribute
    one sample even if concurrent workers wrote the same prediction twice.
    """
    latest: dict[tuple[Any, ...], tuple[int, LedgerEntry]] = {}
    for index, entry in enumerate(entries):
        key = _calibration_dedupe_key(entry, fallback_index=index)
        latest[key] = (index, entry)
    return [entry for _index, entry in sorted(latest.values(), key=lambda item: item[0])]


def _calibration_dedupe_key(entry: LedgerEntry, *, fallback_index: int) -> tuple[Any, ...]:
    if not entry.cycle_id:
        return ("no-cycle", fallback_index)
    return _projection_key(
        cycle_id=entry.cycle_id,
        pair=entry.pair,
        signal_name=entry.signal_name,
        direction=entry.direction,
        entry_price=entry.entry_price,
        target_price=entry.predicted_target_price,
    )


def _calibration_entry_eligible(entry: LedgerEntry) -> bool:
    """Exclude known-ambiguous legacy forecast samples from confidence learning."""
    if (
        entry.signal_name == "directional_forecast"
        and entry.predicted_target_price is not None
        and entry.predicted_invalidation_price is None
    ):
        return False
    return True


def _calibration_signal_names(entry: LedgerEntry) -> tuple[str, ...]:
    names = [entry.signal_name]
    directional_name = directional_calibration_signal_name(entry.signal_name, entry.direction)
    if directional_name is not None and directional_name not in names:
        names.append(directional_name)
    return tuple(names)


def directional_calibration_signal_name(signal_name: str, direction: str) -> Optional[str]:
    """Return the direction-specific calibration alias for directional signals.

    The base bucket is still recorded for compatibility, but detectors such as
    liquidity sweeps can have opposite edge by direction. Keeping UP and DOWN
    aliases lets calibration dampen the failing side without starving the good
    side of the same detector.
    """
    base = str(signal_name or "").strip()
    direction_norm = str(direction or "").upper()
    if not base or direction_norm not in {"UP", "DOWN", "RANGE"}:
        return None
    suffix = direction_norm.lower()
    if base.endswith(f"_{suffix}"):
        return base
    return f"{base}_{suffix}"


def has_confidence_calibration_samples(
    signal_name: str,
    pair: str,
    *,
    hit_rates: Dict[str, Dict[str, Any]],
    regime: Optional[str] = None,
) -> bool:
    """Whether `confidence_calibration` has enough samples for this bucket."""
    by_key = hit_rates.get(signal_name) or {}
    for candidate in _calibration_candidates(by_key, pair=pair, regime=regime):
        if not isinstance(candidate, dict):
            continue
        try:
            samples = int(candidate.get("samples", 0) or 0)
        except (TypeError, ValueError):
            continue
        if samples >= CONFIDENCE_MIN_SAMPLES:
            return True
    return False


def select_calibration_signal_name(
    signal_name: str,
    direction: str,
    pair: str,
    *,
    hit_rates: Dict[str, Dict[str, Any]],
    regime: Optional[str] = None,
) -> str:
    """Prefer direction-specific calibration when it has enough evidence."""
    directional_name = directional_calibration_signal_name(signal_name, direction)
    if directional_name is not None and has_confidence_calibration_samples(
        directional_name,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    ):
        return directional_name
    return signal_name


def _calibration_candidates(
    by_key: Dict[str, Any],
    *,
    pair: str,
    regime: Optional[str] = None,
) -> List[Optional[Dict[str, Any]]]:
    candidates_in_order: List[Optional[Dict[str, Any]]] = []
    if regime is not None:
        candidates_in_order.append(by_key.get(f"{pair}:{regime}"))
    candidates_in_order.append(by_key.get(f"{pair}:_all_regimes"))
    if regime is not None:
        candidates_in_order.append(by_key.get(f"_all_pairs:{regime}"))
    candidates_in_order.append(by_key.get("_all_pairs:_all_regimes"))
    # Backward compatibility: previous schema used just "<pair>" / "_all_pairs".
    if pair in by_key and isinstance(by_key[pair], dict):
        candidates_in_order.append(by_key.get(pair))
    if "_all_pairs" in by_key and isinstance(by_key["_all_pairs"], dict):
        candidates_in_order.append(by_key.get("_all_pairs"))
    return candidates_in_order


def _bayesian_posterior_mean(hits: int, total: int, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> float:
    """Beta-Bernoulli posterior mean for hit-rate.

    Prior Beta(α, β) updated by `hits` Hs and `total - hits` Ms gives
    posterior Beta(α + hits, β + total - hits). The posterior mean is
    `(α + hits) / (α + β + total)`.

    Default uniform prior Beta(1, 1) treats no-prior-info as 50/50 but
    SHRINKS the estimate toward 50% as `total` decreases. This is the
    statistically correct way to handle low-sample buckets — much more
    robust than the linear hit-rate which trusted 4 HITs / 4 trials
    as "100% hit rate" when really it's "we don't know yet".
    """
    if total <= 0:
        return alpha_prior / (alpha_prior + beta_prior)
    return (alpha_prior + hits) / (alpha_prior + beta_prior + total)


def _bayesian_posterior_variance(hits: int, total: int, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> float:
    """Beta posterior variance — used to apply lower-confidence-bound
    pessimism on small samples (Bayesian UCB / Wilson interval style)."""
    a = alpha_prior + hits
    b = beta_prior + (total - hits)
    denom = (a + b) ** 2 * (a + b + 1)
    if denom <= 0:
        return 0.25  # max variance
    return (a * b) / denom


def confidence_calibration(
    signal_name: str,
    pair: str,
    *,
    hit_rates: Dict[str, Dict[str, Any]],
    regime: Optional[str] = None,
) -> float:
    """Return a multiplier ∈ [CONFIDENCE_MIN_MULTIPLIER, CONFIDENCE_MAX_MULTIPLIER]
    to apply to raw `confidence`, based on Bayesian Beta-Bernoulli posterior
    of past hit-rate.

    2026-05-14: Upgraded from linear interp to Beta-Bernoulli posterior:
    - Posterior mean = `(α + hits) / (α + β + total)` with α=β=1 prior
    - Posterior variance gives a pessimism shrink: we use the LOWER 90%
      bound (μ - 1.28 σ) to be conservative on small samples
    - This means "3 HITs / 3 trials" → posterior mean 0.8 (not 1.0),
      lower bound ~0.55 — much more reliable than the naive 100%
    - As sample count grows, the bound approaches the true rate
    - Self-pessimizing on uncertainty, self-confident on lots of data

    Preference order (most specific → most general):
    1. pair × regime  →  2. pair × all regimes  →  3. all pairs × regime
       →  4. all pairs × all regimes
    """
    import math
    by_key = hit_rates.get(signal_name) or {}

    chosen = None
    for c in _calibration_candidates(by_key, pair=pair, regime=regime):
        if c is None:
            continue
        # Need raw samples count; old "_all_pairs" / new buckets all have "samples".
        samples = c.get("samples", 0)
        if samples >= CONFIDENCE_MIN_SAMPLES:
            chosen = c
            break
    if chosen is None:
        return 1.0

    hit_rate_obs = chosen.get("hit_rate", 0.5)
    samples = chosen.get("samples", 0)
    hits = int(round(hit_rate_obs * samples))
    # Bayesian posterior mean + variance
    mu = _bayesian_posterior_mean(hits, samples)
    var = _bayesian_posterior_variance(hits, samples)
    sigma = math.sqrt(var)
    # Lower 90% bound (1.28σ) — pessimistic estimate. As samples grow,
    # σ shrinks toward 0, so bound approaches mean.
    pessimistic = max(0.0, min(1.0, mu - 1.28 * sigma))
    # Map [0, 1] hit-rate to [MIN_MULT, MAX_MULT] via linear interp
    # centered at 0.5 = 1.0.
    if pessimistic >= 0.5:
        mult = 1.0 + (pessimistic - 0.5) * 2.0 * (CONFIDENCE_MAX_MULTIPLIER - 1.0)
    else:
        mult = 1.0 - (0.5 - pessimistic) * 2.0 * (1.0 - CONFIDENCE_MIN_MULTIPLIER)
    return round(mult, 3)


def setup_grade(
    aligned_signal_count: int,
    has_news_block: bool,
    confluence_score: float,
) -> str:
    """A/B/C/D setup grade. Used for trader rationale + size sizing.

    A: ≥4 aligned signals AND no news block AND confluence ≥ 25
    B: ≥3 aligned signals AND confluence ≥ 15
    C: ≥2 aligned signals
    D: 0-1 aligned signals OR news block present
    """
    if has_news_block:
        return "D"
    if aligned_signal_count >= 4 and confluence_score >= 25:
        return "A"
    if aligned_signal_count >= 3 and confluence_score >= 15:
        return "B"
    if aligned_signal_count >= 2:
        return "C"
    return "D"
