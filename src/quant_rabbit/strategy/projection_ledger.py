"""Projection ledger — record every forecast, verify outcomes, self-calibrate.

User directive 2026-05-14:「予測の精度を最大限高める。そうすれば間違えない」.

Every `ProjectionSignal` emitted by `forward_projection.py` is recorded
to `data/projection_ledger.jsonl` at emission time with:
- timestamp_emitted_utc, pair, signal_name, direction, lead_time_min
- entry_price (current price when prediction was made)
- predicted_target_price (for liquidity sweep) or None (for EITHER)
- resolution_window_min = lead_time_min × 2 (give the move 2x slack)
- resolution_status = "PENDING" initially

After the resolution window elapses, `verify_pending_projections()`
checks each PENDING entry against price truth. Prefer the M1 candle path
covering the emitted→expiry window, so a target that was reached and then
mean-reverted is still counted correctly:
- "UP" direction → did window high exceed entry by ≥ ATR_pips × 0.5?
- "DOWN" → did window low go below by ≥ same?
- "EITHER" → did the window range expand by the same ATR-based threshold?
- For liquidity_sweep: did price reach the predicted target?

Resolved entries get tagged HIT / MISS / TIMEOUT. Rolling hit-rate per
`(signal_name, pair)` is then queryable by `confidence_calibration()`
which returns a multiplier on the raw confidence — when a detector
has a poor hit-rate (e.g., 30%), its confidence is dampened; when
strong (e.g., 80%), boosted. This creates a self-improving feedback
loop — the layer stops trusting detectors that don't pan out.

Storage: append-only JSONL so the ledger can be re-played and audited.
File location: `data/projection_ledger.jsonl` (gitignored).

The verifier is idempotent — running it multiple times only resolves
new PENDING entries, never re-resolves already-tagged ones.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_rabbit.instruments import instrument_pip_factor


LEDGER_FILENAME = "projection_ledger.jsonl"
HIT_RATE_LOOKBACK = int(os.environ.get("QR_PROJECTION_HIT_RATE_LOOKBACK", "100"))
CONFIDENCE_MIN_SAMPLES = int(os.environ.get("QR_PROJECTION_CONFIDENCE_MIN_SAMPLES", "10"))
CONFIDENCE_DAMPING = float(os.environ.get("QR_PROJECTION_CONFIDENCE_DAMPING", "0.6"))
# Multiplier when a detector has 100% hit-rate
CONFIDENCE_MAX_MULTIPLIER = float(os.environ.get("QR_PROJECTION_CONFIDENCE_MAX_MULT", "1.5"))
# Multiplier when a detector has 0% hit-rate
CONFIDENCE_MIN_MULTIPLIER = float(os.environ.get("QR_PROJECTION_CONFIDENCE_MIN_MULT", "0.2"))


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
    resolved_at_utc: Optional[str] = None
    resolution_evidence: str = ""
    pre_emission_range_pips: Optional[float] = None
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
            "resolution_window_min": self.resolution_window_min,
            "resolution_status": self.resolution_status,
            "resolved_at_utc": self.resolved_at_utc,
            "resolution_evidence": self.resolution_evidence,
            "pre_emission_range_pips": self.pre_emission_range_pips,
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
            resolved_at_utc=d.get("resolved_at_utc"),
            resolution_evidence=str(d.get("resolution_evidence", "")),
            pre_emission_range_pips=d.get("pre_emission_range_pips"),
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
    seen_keys = _existing_projection_keys(data_root, cycle_id=cycle_id, pair=pair) if cycle_id else set()
    with path.open("a", encoding="utf-8") as f:
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
            entry = LedgerEntry(
                timestamp_emitted_utc=ts,
                pair=pair,
                signal_name=getattr(s, "name", "?"),
                direction=getattr(s, "direction", "?"),
                lead_time_min=float(getattr(s, "lead_time_min", 0)),
                confidence=float(getattr(s, "confidence", 0)),
                entry_price=current_price,
                predicted_target_price=target_price,
                resolution_window_min=float(getattr(s, "lead_time_min", 0)) * 2.0,
                resolution_status="PENDING",
                pre_emission_range_pips=pre_emission_range_pips,
                regime_at_emission=regime_at_emission,
                cycle_id=cycle_id,
            )
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            seen_keys.add(key)
            written += 1
    return written


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


def load_ledger(data_root: Path) -> List[LedgerEntry]:
    """Read full ledger into memory. Returns [] when file missing."""
    path = _ledger_path(data_root)
    if not path.exists():
        return []
    out: List[LedgerEntry] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(LedgerEntry.from_dict(d))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def write_ledger(entries: List[LedgerEntry], data_root: Path) -> None:
    """Overwrite ledger with the given list (used after resolution updates)."""
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def verify_pending(
    data_root: Path,
    *,
    quotes_by_pair: Optional[Dict[str, Dict[str, float]]] = None,
    atr_pips_by_pair: Optional[Dict[str, float]] = None,
    candles_by_pair: Optional[Dict[str, List[Any]]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """Walk PENDING ledger entries; resolve those past their window.

    `candles_by_pair`: {pair: [M1 Candle-like]} is preferred and resolves
    against the high/low path inside the resolution window.
    `quotes_by_pair`: {pair: {"bid": float, "ask": float}} is a legacy
    fallback when the caller has not supplied historical candle truth.
    `atr_pips_by_pair`: optional per-pair ATR pips for EITHER signal
    expansion check.

    Returns count summary {"HIT": n, "MISS": n, "TIMEOUT": n}.
    """
    now = now or datetime.now(timezone.utc)
    entries = load_ledger(data_root)
    if not entries:
        return {"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0}

    quotes_by_pair = quotes_by_pair or {}
    atr_pips_by_pair = atr_pips_by_pair or {}
    counts = {"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0}
    for e in entries:
        if e.resolution_status != "PENDING":
            continue
        try:
            emitted_at = datetime.fromisoformat(e.timestamp_emitted_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        elapsed_min = (now - emitted_at).total_seconds() / 60.0
        if elapsed_min < e.resolution_window_min:
            counts["PENDING"] += 1
            continue

        expires_at = emitted_at + timedelta(minutes=e.resolution_window_min)
        price_path = _price_path_for_entry(e, emitted_at=emitted_at, expires_at=expires_at, candles_by_pair=candles_by_pair)
        if price_path is None:
            if candles_by_pair is not None:
                e.resolution_status = "TIMEOUT"
                e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
                e.resolution_evidence = "no M1 candle truth for projection window"
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
        if e.predicted_target_price is None and (atr_pips is None or atr_pips <= 0):
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "missing ATR threshold for projection verification"
            counts["TIMEOUT"] += 1
            continue
        if atr_pips is not None and atr_pips > 0:
            move_threshold_price = atr_pips * 0.5 / pip_factor

        if e.predicted_target_price is not None:
            # Liquidity sweep — did price reach the target?
            target = e.predicted_target_price
            if e.direction == "UP" and target <= entry_price:
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
    write_ledger(entries, data_root)
    return counts


def _price_path_for_entry(
    entry: LedgerEntry,
    *,
    emitted_at: datetime,
    expires_at: datetime,
    candles_by_pair: Optional[Dict[str, List[Any]]],
) -> Optional[dict[str, float]]:
    if candles_by_pair is None:
        return None
    candles = candles_by_pair.get(entry.pair) or []
    window = []
    # M1 candles are timestamped at bar open. Include the candle that overlaps
    # the emission minute, accepting minute-level precision for verification.
    overlap_start = emitted_at - timedelta(minutes=1)
    for c in candles:
        norm = _normalise_candle(c)
        if norm is None:
            continue
        ts = norm["timestamp"]
        if overlap_start <= ts <= expires_at:
            window.append(norm)
    if not window:
        return None
    window.sort(key=lambda item: item["timestamp"])
    return {
        "entry": window[0]["close"],
        "high": max(item["high"] for item in window),
        "low": min(item["low"] for item in window),
        "close": window[-1]["close"],
    }


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
    buckets when there aren't enough samples in the granular one.
    """
    entries = load_ledger(data_root)
    resolved = [e for e in entries if e.resolution_status in ("HIT", "MISS")]
    resolved = resolved[-lookback * 10:]  # rough bound on size
    grouped: Dict[str, Dict[str, List[bool]]] = {}
    for e in resolved:
        s = grouped.setdefault(e.signal_name, {})
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
    candidates_in_order: List[Optional[Dict[str, Any]]] = []
    if regime is not None:
        candidates_in_order.append(by_key.get(f"{pair}:{regime}"))
    candidates_in_order.append(by_key.get(f"{pair}:_all_regimes"))
    if regime is not None:
        candidates_in_order.append(by_key.get(f"_all_pairs:{regime}"))
    candidates_in_order.append(by_key.get("_all_pairs:_all_regimes"))
    # Backward compatibility: previous schema used just "<pair>" / "_all_pairs"
    if pair in by_key and isinstance(by_key[pair], dict):
        candidates_in_order.append(by_key.get(pair))
    if "_all_pairs" in by_key and isinstance(by_key["_all_pairs"], dict):
        candidates_in_order.append(by_key.get("_all_pairs"))

    chosen = None
    for c in candidates_in_order:
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
