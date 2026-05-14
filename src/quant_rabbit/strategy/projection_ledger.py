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
checks each PENDING entry against price truth (current OANDA bid/ask):
- "UP" direction → did high since emission exceed entry by ≥ ATR_pips × 0.5?
- "DOWN" → did low go below by ≥ same?
- "EITHER" → did volatility expand (range ≥ 1.5× pre-emission range)?
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    now: Optional[datetime] = None,
) -> int:
    """Append all signals to the ledger. Returns count written.

    Idempotency: this is APPEND-ONLY. Duplicate writes are tolerated
    because verifier picks up by timestamp + signal_name + pair. To
    keep file size bounded, callers can periodically truncate to last
    N days externally.
    """
    if not signals:
        return 0
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat().replace("+00:00", "Z")
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for s in signals:
            # Liquidity sweep signals carry an implied target price in
            # the rationale; capture it heuristically when present.
            target_price = _extract_target_price_from_rationale(getattr(s, "rationale", ""))
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
            )
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


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
    quotes_by_pair: Dict[str, Dict[str, float]],
    atr_pips_by_pair: Optional[Dict[str, float]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """Walk PENDING ledger entries; resolve those past their window.

    `quotes_by_pair`: {pair: {"bid": float, "ask": float}}.
    `atr_pips_by_pair`: optional per-pair ATR pips for EITHER signal
    expansion check.

    Returns count summary {"HIT": n, "MISS": n, "TIMEOUT": n}.
    """
    now = now or datetime.now(timezone.utc)
    entries = load_ledger(data_root)
    if not entries:
        return {"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0}

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
        # Past the window — try to resolve
        quote = quotes_by_pair.get(e.pair)
        if not quote:
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "no quote for pair at verification time"
            counts["TIMEOUT"] += 1
            continue
        current_price = (float(quote.get("bid", 0)) + float(quote.get("ask", 0))) / 2.0
        if e.entry_price is None or current_price <= 0:
            e.resolution_status = "TIMEOUT"
            e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
            e.resolution_evidence = "missing entry_price or current_price"
            counts["TIMEOUT"] += 1
            continue

        pip_factor = 100.0 if e.pair.endswith("_JPY") else 10000.0
        atr_pips = atr_pips_by_pair.get(e.pair) or 10.0
        move_threshold_price = atr_pips * 0.5 / pip_factor

        if e.predicted_target_price is not None:
            # Liquidity sweep — did price reach the target?
            target = e.predicted_target_price
            if e.direction == "UP" and current_price >= target:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"price {current_price:.5f} reached target {target:.5f}"
                counts["HIT"] += 1
            elif e.direction == "DOWN" and current_price <= target:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"price {current_price:.5f} reached target {target:.5f}"
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = f"price {current_price:.5f} did not reach {target:.5f}"
                counts["MISS"] += 1
        elif e.direction == "EITHER":
            # Volatility expansion — did absolute move exceed threshold?
            move = abs(current_price - e.entry_price)
            if move >= move_threshold_price:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"abs move {move * pip_factor:.1f}pip ≥ {atr_pips * 0.5:.1f}pip threshold"
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = f"abs move {move * pip_factor:.1f}pip < threshold (no expansion)"
                counts["MISS"] += 1
        else:
            # UP/DOWN directional — current price vs entry
            move = current_price - e.entry_price
            if e.direction == "UP":
                hit = move >= move_threshold_price
            else:
                hit = (-move) >= move_threshold_price
            if hit:
                e.resolution_status = "HIT"
                e.resolution_evidence = f"price moved {move * pip_factor:+.1f}pip toward {e.direction}"
                counts["HIT"] += 1
            else:
                e.resolution_status = "MISS"
                e.resolution_evidence = f"price moved {move * pip_factor:+.1f}pip (not enough toward {e.direction})"
                counts["MISS"] += 1
        e.resolved_at_utc = now.isoformat().replace("+00:00", "Z")
    write_ledger(entries, data_root)
    return counts


def compute_hit_rates(
    data_root: Path,
    *,
    lookback: int = HIT_RATE_LOOKBACK,
) -> Dict[str, Dict[str, float]]:
    """Per `(signal_name, pair)` hit-rate from last `lookback` resolved entries.

    Returns:
        {
            "<signal_name>": {
                "<pair>": {"hit_rate": 0.0-1.0, "samples": int},
                "_all_pairs": {"hit_rate": 0.0-1.0, "samples": int},
            },
            ...
        }
    """
    entries = load_ledger(data_root)
    resolved = [e for e in entries if e.resolution_status in ("HIT", "MISS")]
    resolved = resolved[-lookback * 5:]  # rough bound on size
    # Group
    grouped: Dict[str, Dict[str, List[bool]]] = {}
    for e in resolved:
        s = grouped.setdefault(e.signal_name, {})
        s.setdefault(e.pair, []).append(e.resolution_status == "HIT")
        s.setdefault("_all_pairs", []).append(e.resolution_status == "HIT")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sig, by_pair in grouped.items():
        out[sig] = {}
        for pair, results in by_pair.items():
            n = min(len(results), lookback)
            recent = results[-n:]
            if not recent:
                continue
            hr = sum(1 for r in recent if r) / float(n)
            out[sig][pair] = {"hit_rate": round(hr, 3), "samples": n}
    return out


def confidence_calibration(
    signal_name: str,
    pair: str,
    *,
    hit_rates: Dict[str, Dict[str, Any]],
) -> float:
    """Return a multiplier ∈ [CONFIDENCE_MIN_MULTIPLIER, CONFIDENCE_MAX_MULTIPLIER]
    to apply to raw `confidence`, based on past hit-rate.

    Falls back to 1.0 (no adjustment) when there aren't enough samples
    to be confident in the calibration (< CONFIDENCE_MIN_SAMPLES).

    Preference order: per-pair hit-rate → all-pairs hit-rate → 1.0.
    """
    by_pair = hit_rates.get(signal_name) or {}
    per_pair = by_pair.get(pair)
    candidate = per_pair if (per_pair and per_pair.get("samples", 0) >= CONFIDENCE_MIN_SAMPLES) else None
    if candidate is None:
        all_pairs = by_pair.get("_all_pairs")
        candidate = all_pairs if (all_pairs and all_pairs.get("samples", 0) >= CONFIDENCE_MIN_SAMPLES) else None
    if candidate is None:
        return 1.0
    hit_rate = candidate.get("hit_rate", 0.5)
    # Linear interp: 0.0 → MIN_MULT, 0.5 → 1.0, 1.0 → MAX_MULT
    if hit_rate >= 0.5:
        mult = 1.0 + (hit_rate - 0.5) * 2.0 * (CONFIDENCE_MAX_MULTIPLIER - 1.0)
    else:
        mult = 1.0 - (0.5 - hit_rate) * 2.0 * (1.0 - CONFIDENCE_MIN_MULTIPLIER)
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
