"""Entry thesis ledger — record WHY each position was entered, and
re-evaluate vs current market state every cycle.

User directive 2026-05-15:
  「どの視点でエントリーしたのか、時間がたって今のポジ状況は
   エントリーしたときと市況が変わってないか。そうしたらどうすべきか」.

Without an entry-time thesis record, the trader has no way to answer
"is the original reason still valid?" The 17-layer prediction stack
provides a CURRENT snapshot but doesn't compare against entry.

This module:

1. **On entry**, captures the entry thesis:
   - timestamp_utc, trade_id, pair, side, entry_price
   - directional_forecast at entry: direction + confidence
   - regime at entry
   - key drivers (top-3 detector signals supporting the entry)
   - invalidation_price (from forecast)
   - target_price (from forecast)
   Writes to `data/entry_thesis_ledger.jsonl` (append-only).

2. **Every cycle**, for each open trader-owned position:
   - Load entry thesis from ledger
   - Synthesize current forecast
   - Compare:
     * direction: still same? flipped? went UNCLEAR?
     * confidence: still strong? decayed?
     * regime: same (TREND→TREND OK)? shifted (TREND→RANGE = caution)?
     * key drivers: still active? broken?
   - Emit `ThesisEvolution`:
     * status: STILL_VALID / WEAKENED / BROKEN
     * verdict: HOLD / EXTEND / RECOMMEND_CLOSE
     * rationale: what changed
   Writes to `data/thesis_evolution_report.json` per cycle.

The trader/operator reads thesis_evolution_report.json and:
- STILL_VALID + EXTEND → hold, let winners run
- WEAKENED → caution, smaller TP if any
- BROKEN → close (manual or via Gate A/B)

No auto-close from this module — INFORMATION only. The kill switch
(QR_DISABLE_AUTO_CLOSE) stays on; close decisions go through gpt_trader.

Kill switch: `QR_DISABLE_ENTRY_THESIS_LEDGER=1`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LEDGER_FILENAME = "entry_thesis_ledger.jsonl"
FORECAST_HISTORY_FILENAME = "forecast_history.jsonl"


@dataclass
class EntryThesis:
    timestamp_utc: str
    trade_id: str
    pair: str
    side: str
    entry_price: float
    forecast_direction: str
    forecast_confidence: float
    regime: Optional[str]
    invalidation_price: Optional[float]
    target_price: Optional[float]
    key_drivers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp_utc": self.timestamp_utc,
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "entry_price": self.entry_price,
            "forecast_direction": self.forecast_direction,
            "forecast_confidence": self.forecast_confidence,
            "regime": self.regime,
            "invalidation_price": self.invalidation_price,
            "target_price": self.target_price,
            "key_drivers": self.key_drivers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntryThesis":
        return cls(
            timestamp_utc=str(d.get("timestamp_utc", "")),
            trade_id=str(d.get("trade_id", "")),
            pair=str(d.get("pair", "")),
            side=str(d.get("side", "")),
            entry_price=float(d.get("entry_price", 0)),
            forecast_direction=str(d.get("forecast_direction", "UNCLEAR")),
            forecast_confidence=float(d.get("forecast_confidence", 0)),
            regime=d.get("regime"),
            invalidation_price=d.get("invalidation_price"),
            target_price=d.get("target_price"),
            key_drivers=list(d.get("key_drivers", [])),
        )


@dataclass(frozen=True)
class ThesisEvolution:
    trade_id: str
    pair: str
    side: str
    age_hours: float
    entry_forecast: str
    current_forecast: str
    entry_confidence: float
    current_confidence: float
    entry_regime: Optional[str]
    current_regime: Optional[str]
    status: str  # "STILL_VALID" | "WEAKENED" | "BROKEN"
    verdict: str  # "HOLD" | "EXTEND" | "RECOMMEND_CLOSE"
    rationale: str

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "age_hours": round(self.age_hours, 2),
            "entry_forecast": self.entry_forecast,
            "current_forecast": self.current_forecast,
            "entry_confidence": round(self.entry_confidence, 3),
            "current_confidence": round(self.current_confidence, 3),
            "entry_regime": self.entry_regime,
            "current_regime": self.current_regime,
            "status": self.status,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_ENTRY_THESIS_LEDGER", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def record_entry_thesis(thesis: EntryThesis, data_root: Path) -> None:
    """Append entry thesis to ledger."""
    if _is_disabled():
        return
    path = data_root / LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(thesis.to_dict(), ensure_ascii=False) + "\n")


def load_entry_thesis(trade_id: str, data_root: Path) -> Optional[EntryThesis]:
    """Load thesis for a trade_id. Returns None when not found."""
    path = data_root / LEDGER_FILENAME
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("trade_id", "")) == trade_id:
                return EntryThesis.from_dict(d)
    except OSError:
        return None
    return None


def load_latest_forecast(pair: str, data_root: Path) -> Optional[Dict[str, Any]]:
    """Read the most recent `forecast_history.jsonl` entry for `pair`.

    Used by record_entry_thesis_from_response() at fill time so the
    entry thesis carries the exact forecast that motivated entry. The
    forecast was emitted earlier in the cycle by trader_brain._score_lane
    via forecast_persistence_tracker.record_forecast().
    """
    path = data_root / FORECAST_HISTORY_FILENAME
    if not path.exists():
        return None
    latest: Optional[Dict[str, Any]] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("pair", "")) == pair:
                latest = d
    except OSError:
        return None
    return latest


def _response_trade_id(response: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract tradeID from OANDA order response (mirrors
    execution_ledger._response_trade_id without importing it)."""
    if not response or not isinstance(response, dict):
        return None
    fill = response.get("orderFillTransaction")
    if isinstance(fill, dict):
        opened = fill.get("tradeOpened")
        if isinstance(opened, dict) and opened.get("tradeID") is not None:
            return str(opened["tradeID"])
    return None


def _response_fill_price(response: Optional[Dict[str, Any]]) -> Optional[float]:
    if not response or not isinstance(response, dict):
        return None
    fill = response.get("orderFillTransaction")
    if isinstance(fill, dict):
        for key in ("price", "fullPrice"):
            v = fill.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                # OANDA fullPrice carries closeoutBid/Ask; fall back to bid.
                for sub in ("price", "closeoutBid", "closeoutAsk"):
                    sv = v.get(sub)
                    if isinstance(sv, (int, float)):
                        return float(sv)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    continue
    return None


def record_entry_thesis_from_response(
    *,
    response: Optional[Dict[str, Any]],
    intent: Any,
    data_root: Path,
    now: Optional[datetime] = None,
) -> Optional[EntryThesis]:
    """Build an `EntryThesis` from a SENT OANDA response + intent metadata.

    Called from execution.LiveOrderGateway right after a successful
    `post_order_json` returns. Reads the canonical forecast that
    trader_brain just wrote for this pair, snapshots side/entry_price,
    and appends to `data/entry_thesis_ledger.jsonl`. Returns the
    written thesis (or None when disabled / unable to identify trade).

    Failure here MUST NOT raise — execution path is the hot path.
    """
    if _is_disabled():
        return None
    try:
        trade_id = _response_trade_id(response)
        if not trade_id:
            return None
        pair = str(getattr(intent, "pair", ""))
        side_attr = getattr(intent, "side", None)
        side_val = getattr(side_attr, "value", None) or str(side_attr or "")
        side_up = side_val.upper()
        if not pair or side_up not in ("LONG", "SHORT"):
            return None
        fill_price = _response_fill_price(response) or float(getattr(intent, "entry", 0) or 0)

        forecast = load_latest_forecast(pair, data_root) or {}
        metadata = dict(getattr(intent, "metadata", {}) or {})
        # Extract top-3 drivers from intent metadata + forecast
        key_drivers: List[str] = []
        if forecast.get("direction"):
            key_drivers.append(
                f"forecast={forecast.get('direction')}@conf={float(forecast.get('confidence', 0)):.2f}"
            )
        for k in ("desk", "campaign_role", "regime_state", "target_reward_risk"):
            v = metadata.get(k)
            if v not in (None, ""):
                key_drivers.append(f"{k}={v}")
        thesis_text = str(getattr(intent, "thesis", "") or "")
        if thesis_text:
            key_drivers.append(thesis_text[:120])

        regime = forecast.get("regime") or metadata.get("regime_state") or None

        now = now or datetime.now(timezone.utc)
        entry_thesis = EntryThesis(
            timestamp_utc=now.isoformat().replace("+00:00", "Z"),
            trade_id=trade_id,
            pair=pair,
            side=side_up,
            entry_price=float(fill_price),
            forecast_direction=str(forecast.get("direction") or "UNCLEAR"),
            forecast_confidence=float(forecast.get("confidence") or 0.0),
            regime=str(regime) if regime else None,
            invalidation_price=forecast.get("invalidation_price"),
            target_price=forecast.get("target_price"),
            key_drivers=key_drivers[:6],
        )
        record_entry_thesis(entry_thesis, data_root)
        return entry_thesis
    except Exception:
        # Never break the live order path — thesis recording is purely
        # informational/auxiliary.
        return None


def evaluate_thesis_evolution(
    *,
    trade_id: str,
    pair: str,
    side: str,
    open_time_utc: Optional[str],
    current_forecast: Any,
    current_regime: Optional[str],
    data_root: Path,
    now: Optional[datetime] = None,
) -> Optional[ThesisEvolution]:
    """Compare current forecast/regime vs entry-time thesis.

    Returns None when no entry thesis is found (legacy position
    opened before this module).
    """
    if _is_disabled():
        return None
    thesis = load_entry_thesis(trade_id, data_root)
    if thesis is None:
        return None
    now = now or datetime.now(timezone.utc)
    age_hours = 0.0
    if open_time_utc:
        try:
            opened = datetime.fromisoformat(open_time_utc.replace("Z", "+00:00").split(".")[0] + "+00:00")
            age_hours = (now - opened).total_seconds() / 3600
        except (TypeError, ValueError):
            pass
    elif thesis.timestamp_utc:
        try:
            opened = datetime.fromisoformat(thesis.timestamp_utc.replace("Z", "+00:00"))
            age_hours = (now - opened).total_seconds() / 3600
        except (TypeError, ValueError):
            pass

    current_dir = getattr(current_forecast, "direction", "UNCLEAR")
    current_conf = float(getattr(current_forecast, "confidence", 0))

    # Status classification
    entry_dir = thesis.forecast_direction
    side_up = side.upper()
    aligned_dir = "UP" if side_up == "LONG" else "DOWN"

    reasons: List[str] = []
    if current_dir == entry_dir and current_dir == aligned_dir:
        # Both entry and current align with position
        if current_conf >= thesis.forecast_confidence * 0.8:
            status = "STILL_VALID"
            verdict = "EXTEND" if current_conf > thesis.forecast_confidence else "HOLD"
            reasons.append(
                f"entry forecast {entry_dir} conf={thesis.forecast_confidence:.2f} → "
                f"current {current_dir} conf={current_conf:.2f}: thesis intact"
            )
        else:
            status = "WEAKENED"
            verdict = "HOLD"
            reasons.append(
                f"forecast direction unchanged but confidence decayed "
                f"({thesis.forecast_confidence:.2f}→{current_conf:.2f})"
            )
    elif current_dir in ("RANGE", "UNCLEAR"):
        status = "WEAKENED"
        verdict = "HOLD"
        reasons.append(
            f"entry was {entry_dir} but current forecast is {current_dir} — directional edge lost"
        )
    elif current_dir != aligned_dir and current_dir != "UNCLEAR":
        # Current forecast is OPPOSITE the position direction
        status = "BROKEN"
        verdict = "RECOMMEND_CLOSE"
        reasons.append(
            f"FORECAST FLIPPED: entry {entry_dir} → current {current_dir} (position {side_up})"
        )
    else:
        status = "WEAKENED"
        verdict = "HOLD"
        reasons.append(f"forecast {current_dir} ambiguous vs entry {entry_dir}")

    # Regime shift detection
    if thesis.regime and current_regime and thesis.regime != current_regime:
        reasons.append(
            f"regime SHIFTED: {thesis.regime} (entry) → {current_regime} (now)"
        )
        if status == "STILL_VALID":
            status = "WEAKENED"

    return ThesisEvolution(
        trade_id=trade_id, pair=pair, side=side_up,
        age_hours=age_hours,
        entry_forecast=entry_dir,
        current_forecast=current_dir,
        entry_confidence=thesis.forecast_confidence,
        current_confidence=current_conf,
        entry_regime=thesis.regime,
        current_regime=current_regime,
        status=status, verdict=verdict,
        rationale="; ".join(reasons),
    )


def evaluate_all_open_positions(
    positions: List[Any],
    *,
    current_forecasts_by_pair: Dict[str, Any],
    current_regimes_by_pair: Dict[str, Optional[str]],
    data_root: Path,
    now: Optional[datetime] = None,
) -> List[ThesisEvolution]:
    """For every trader-owned open position, compute ThesisEvolution.

    `current_forecasts_by_pair` and `current_regimes_by_pair` are
    keyed by OANDA pair string (e.g. "EUR_JPY"). Forecasts can be
    `DirectionalForecast` instances or anything with `.direction` and
    `.confidence`. Positions without an entry thesis record (legacy
    pre-2026-05-15) are skipped silently.
    """
    if _is_disabled():
        return []
    out: List[ThesisEvolution] = []
    for p in positions:
        owner = getattr(p, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        pair = str(getattr(p, "pair", ""))
        side = getattr(p, "side", None)
        side_val = side.value if hasattr(side, "value") else str(side or "")
        side_up = side_val.upper()
        trade_id = str(getattr(p, "trade_id", ""))
        if not (pair and trade_id and side_up in ("LONG", "SHORT")):
            continue
        forecast = current_forecasts_by_pair.get(pair)
        if forecast is None:
            continue
        regime = current_regimes_by_pair.get(pair)
        open_time = getattr(p, "opened_at_utc", None) or getattr(p, "open_time_utc", None)
        ev = evaluate_thesis_evolution(
            trade_id=trade_id,
            pair=pair,
            side=side_up,
            open_time_utc=open_time,
            current_forecast=forecast,
            current_regime=regime,
            data_root=data_root,
            now=now,
        )
        if ev is not None:
            out.append(ev)
    return out


def write_thesis_evolution_report(
    evolutions: List[ThesisEvolution],
    *,
    data_root: Path,
    now: Optional[datetime] = None,
) -> Path:
    """Write the per-cycle `thesis_evolution_report.json`."""
    report_path = data_root / "thesis_evolution_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    payload = {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "count": len(evolutions),
        "by_status": {
            "STILL_VALID": sum(1 for e in evolutions if e.status == "STILL_VALID"),
            "WEAKENED": sum(1 for e in evolutions if e.status == "WEAKENED"),
            "BROKEN": sum(1 for e in evolutions if e.status == "BROKEN"),
        },
        "evolutions": [e.to_dict() for e in evolutions],
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path
