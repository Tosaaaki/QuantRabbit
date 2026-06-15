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
   - invalidation_price (from forecast, or actual broker/intent SL)
   - target_price (from forecast, or actual broker/intent TP)
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
     * status: STILL_VALID / WEAKENED / BROKEN / UNVERIFIABLE
     * verdict: HOLD / EXTEND / RECOMMEND_CLOSE / REQUIRE_THESIS_REPAIR
     * rationale: what changed
   Writes to `data/thesis_evolution_report.json` per cycle.

The trader/operator reads thesis_evolution_report.json and:
- STILL_VALID + EXTEND → hold, let winners run
- WEAKENED → caution, smaller TP if any
- BROKEN → hard Gate A evidence for close; standing structural loss-cut
  authorization applies
- Missing entry-thesis rows are surfaced as UNVERIFIABLE /
  REQUIRE_THESIS_REPAIR hard management blockers. They are not standalone
  close evidence, but they block normal WAIT/new-risk/TP-expansion paths until
  position_thesis / forecast_persistence supplies machine-checkable evidence.

No auto-close from this module. The kill switch (QR_DISABLE_AUTO_CLOSE)
stays on; close decisions go through gpt_trader and still need a verified
CLOSE receipt.

Kill switch: `QR_DISABLE_ENTRY_THESIS_LEDGER=1`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN

# A thesis inherits its time scope from the forecast that justified the entry:
# the pair forecast's own declared `horizon_min`. The multiplier mirrors
# forecast_persistence_tracker.FLIP_PERSISTENCE_CYCLES (= 3): an entry thesis
# that has reached neither target nor invalidation after three of its own
# forecast horizons has exhausted the predictive scope it was built on. This
# is a thesis-declared structural bound, not a tuned market literal (§3.5).
THESIS_HORIZON_FORECAST_MULT = float(os.environ.get("QR_THESIS_HORIZON_FORECAST_MULT", "3.0"))

# Durable per-cycle archive of thesis evolution states. The per-cycle report
# (thesis_evolution_report.json) is overwritten every cycle, which made the
# WEAKENED -> BROKEN -> close latency unmeasurable for any trade older than
# the current cycle (2026-06-11 exit-leak audit).
THESIS_EVOLUTION_HISTORY_FILENAME = "thesis_evolution_history.jsonl"

# Same-cycle dedup floor for the consecutive-WEAKENED check: an archived row
# younger than half the 20-minute scheduler cadence is the current cycle's own
# write (multiple evaluations can run inside one cycle), not a prior check.
THESIS_EXPIRY_PRIOR_CHECK_MIN_AGE_SECONDS = 600.0
RANGE_ROTATION_FAIL_TARGET_FRACTION = float(os.environ.get("QR_RANGE_ROTATION_FAIL_TARGET_FRACTION", "0.35"))


def _thesis_horizon_hours_from_forecast(forecast: Dict[str, Any]) -> Optional[float]:
    """Derive the thesis horizon from the entry forecast's declared horizon."""
    try:
        horizon_min = float(forecast.get("horizon_min") or 0)
    except (TypeError, ValueError):
        return None
    if horizon_min <= 0:
        return None
    return horizon_min / 60.0 * THESIS_HORIZON_FORECAST_MULT


def _last_archived_evolution_row(trade_id: str, data_root: Path) -> Optional[dict]:
    path = data_root / THESIS_EVOLUTION_HISTORY_FILENAME
    if not path.exists():
        return None
    last: Optional[dict] = None
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
                last = d
    except OSError:
        return None
    return last


def _has_prior_weakened_or_broken_cycle(*, trade_id: str, data_root: Path, now: datetime) -> bool:
    prior = _last_archived_evolution_row(trade_id, data_root)
    prior_status = str((prior or {}).get("status") or "")
    prior_at = _parse_utc_timestamp((prior or {}).get("generated_at_utc"))
    prior_is_previous_cycle = (
        prior_at is not None
        and (now - prior_at).total_seconds() >= THESIS_EXPIRY_PRIOR_CHECK_MIN_AGE_SECONDS
    )
    return prior_status in ("WEAKENED", "BROKEN") and prior_is_previous_cycle


LEDGER_FILENAME = "entry_thesis_ledger.jsonl"
PENDING_LEDGER_FILENAME = "pending_entry_thesis_ledger.jsonl"
FORECAST_HISTORY_FILENAME = "forecast_history.jsonl"
PENDING_ORDER_TYPES = {"LIMIT_ORDER", "STOP_ORDER", "MARKET_IF_TOUCHED_ORDER"}
DEFAULT_THESIS_INVALIDATION_BUFFER_PIPS = 2.0
TECHNICAL_INVALIDATION_TFS = ("M5", "M15", "M30", "H1")
UNVERIFIABLE_STATUS = "UNVERIFIABLE"
REQUIRE_THESIS_REPAIR_VERDICT = "REQUIRE_THESIS_REPAIR"
_CONTEXT_REF_RE = re.compile(r"\b(?:matrix|context_asset|cross|news|calendar|cot|flow|level|currency_strength):[A-Za-z0-9_./:-]+\b")
_CONTEXT_ASSET_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,8}_[A-Z]{3}\b")
_NEWS_CONTEXT_TOKENS = ("news", "headline", "calendar", "macro", "event", "catalyst")
_CONTEXT_LIST_LIMIT = 8
_CONTEXT_TEXT_LIMIT = 240
_FX_CONTEXT_CURRENCIES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}


def thesis_invalidation_buffer_pips(buffer_pips: Optional[float] = None) -> float:
    if buffer_pips is not None:
        return max(0.0, float(buffer_pips))
    try:
        return max(0.0, float(os.environ.get(
            "QR_THESIS_INVALIDATION_BUFFER_PIPS",
            str(DEFAULT_THESIS_INVALIDATION_BUFFER_PIPS),
        )))
    except ValueError:
        return DEFAULT_THESIS_INVALIDATION_BUFFER_PIPS


def invalidation_pip_size(pair: str) -> float:
    return 0.01 if str(pair or "").upper().endswith("_JPY") else 0.0001


def invalidation_buffer_price(pair: str, buffer_pips: Optional[float] = None) -> float:
    return thesis_invalidation_buffer_pips(buffer_pips) * invalidation_pip_size(pair)


def invalidation_price_hit_reason(
    *,
    pair: str,
    side: str,
    current_price: Optional[float],
    invalidation_price: Optional[float],
    price_label: Optional[str] = None,
    buffer_pips: Optional[float] = None,
) -> Optional[str]:
    """Return a reason only after price clears the anti-wick buffer."""

    if current_price is None:
        return None
    try:
        price = float(current_price)
        invalidation = float(invalidation_price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or invalidation <= 0.0:
        return None

    side_up = str(side or "").upper()
    label = (price_label or ("bid" if side_up == "LONG" else "ask")).strip()
    buffer_pips_value = thesis_invalidation_buffer_pips(buffer_pips)
    buffer_price = invalidation_buffer_price(pair, buffer_pips_value)
    if side_up == "LONG":
        trigger = invalidation - buffer_price
        if price <= trigger:
            return (
                f"invalidation hit: current {label} {price:.5f} <= buffered invalidation "
                f"{trigger:.5f} (raw {invalidation:.5f}, buffer {buffer_pips_value:.1f}p)"
            )
    if side_up == "SHORT":
        trigger = invalidation + buffer_price
        if price >= trigger:
            return (
                f"invalidation hit: current {label} {price:.5f} >= buffered invalidation "
                f"{trigger:.5f} (raw {invalidation:.5f}, buffer {buffer_pips_value:.1f}p)"
            )
    return None


def _technical_invalidation_min_tfs() -> int:
    try:
        return max(1, int(os.environ.get("QR_THESIS_INVALIDATION_MIN_TECH_TFS", "2")))
    except ValueError:
        return 2


def _technical_invalidation_min_signals() -> int:
    try:
        return max(1, int(os.environ.get("QR_THESIS_INVALIDATION_MIN_TECH_SIGNALS", "4")))
    except ValueError:
        return 4


def _view_tf(view: Dict[str, Any]) -> str:
    return str(view.get("timeframe") or view.get("tf") or view.get("granularity") or "").upper()


def _float_signal(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_move(event: Dict[str, Any]) -> Optional[str]:
    kind = str(event.get("kind") or "").upper()
    if kind.endswith("_UP"):
        return "UP"
    if kind.endswith("_DOWN"):
        return "DOWN"
    return None


def technical_invalidation_confirmation_reason(
    pair_chart: Optional[Dict[str, Any]],
    *,
    side: str,
) -> Optional[str]:
    """Confirm invalidation with chart shape and technicals, not price alone."""

    if not pair_chart:
        return None
    side_up = str(side or "").upper()
    if side_up not in {"LONG", "SHORT"}:
        return None
    adverse_move = "DOWN" if side_up == "LONG" else "UP"
    raw_views = pair_chart.get("views") or []
    if isinstance(raw_views, dict):
        views = [
            {"granularity": key, **value}
            for key, value in raw_views.items()
            if isinstance(value, dict)
        ]
    elif isinstance(raw_views, list):
        views = raw_views
    else:
        return None

    tf_hits: dict[str, list[str]] = {}
    for view in views:
        if not isinstance(view, dict):
            continue
        tf = _view_tf(view)
        if tf not in TECHNICAL_INVALIDATION_TFS:
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else view
        signals: list[str] = []

        regime = str(view.get("regime") or view.get("regime_state") or "").upper()
        if adverse_move == "UP" and ("TREND_UP" in regime or "IMPULSE_UP" in regime):
            signals.append(f"{tf} regime={regime}")
        elif adverse_move == "DOWN" and ("TREND_DOWN" in regime or "IMPULSE_DOWN" in regime):
            signals.append(f"{tf} regime={regime}")

        rsi = _float_signal((indicators or {}).get("rsi_14") or (indicators or {}).get("rsi"))
        if rsi is not None:
            if adverse_move == "UP" and rsi >= 55.0:
                signals.append(f"{tf} RSI={rsi:.1f}")
            elif adverse_move == "DOWN" and rsi <= 45.0:
                signals.append(f"{tf} RSI={rsi:.1f}")

        macd_hist = _float_signal((indicators or {}).get("macd_hist"))
        if macd_hist is not None:
            if adverse_move == "UP" and macd_hist > 0:
                signals.append(f"{tf} MACD+")
            elif adverse_move == "DOWN" and macd_hist < 0:
                signals.append(f"{tf} MACD-")

        st = _float_signal((indicators or {}).get("supertrend_dir"))
        if st is not None:
            if adverse_move == "UP" and st > 0:
                signals.append(f"{tf} ST+")
            elif adverse_move == "DOWN" and st < 0:
                signals.append(f"{tf} ST-")

        cloud = _float_signal((indicators or {}).get("ichimoku_cloud_pos"))
        if cloud is not None:
            if adverse_move == "UP" and cloud > 0:
                signals.append(f"{tf} cloud+")
            elif adverse_move == "DOWN" and cloud < 0:
                signals.append(f"{tf} cloud-")

        plus_di = _float_signal((indicators or {}).get("plus_di_14") or (indicators or {}).get("plus_di"))
        minus_di = _float_signal((indicators or {}).get("minus_di_14") or (indicators or {}).get("minus_di"))
        if plus_di is not None and minus_di is not None:
            if adverse_move == "UP" and plus_di > minus_di:
                signals.append(f"{tf} +DI>{minus_di:.1f}")
            elif adverse_move == "DOWN" and minus_di > plus_di:
                signals.append(f"{tf} -DI>{plus_di:.1f}")

        structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
        last_event = structure.get("last_event") if isinstance(structure.get("last_event"), dict) else {}
        if last_event and bool(last_event.get("close_confirmed")):
            move = _event_move(last_event)
            if move == adverse_move:
                signals.append(f"{tf} {last_event.get('kind')}")

        if signals:
            tf_hits[tf] = signals

    signal_count = sum(len(items) for items in tf_hits.values())
    if len(tf_hits) < _technical_invalidation_min_tfs() or signal_count < _technical_invalidation_min_signals():
        return None

    details = "; ".join(
        f"{tf}: {', '.join(items[:3])}" for tf, items in sorted(tf_hits.items())
    )
    return f"technical invalidation confirmed against {side_up}: {details}"


def same_direction_chart_support_reason(
    pair_chart: Optional[Dict[str, Any]],
    *,
    side: str,
) -> Optional[str]:
    """Detect categorical chart support that keeps an aged thesis from hard-closing.

    This deliberately reuses chart_reader's published confluence label instead
    of adding a new numeric threshold. A timed-out thesis with current same-side
    confluence is a geometry/reprice problem, not unattended loss-cut evidence.
    """

    if not isinstance(pair_chart, dict):
        return None
    side_up = str(side or "").upper()
    if side_up not in {"LONG", "SHORT"}:
        return None
    confluence = pair_chart.get("confluence")
    if not isinstance(confluence, dict):
        return None
    score_balance = str(confluence.get("score_balance") or "").upper()
    expected = "LONG_LEAN" if side_up == "LONG" else "SHORT_LEAN"
    if score_balance != expected:
        return None
    score_gap = confluence.get("score_gap")
    regime = confluence.get("dominant_regime")
    higher_alignment = confluence.get("higher_tf_alignment")
    details = [f"chart confluence {score_balance}"]
    if score_gap is not None:
        details.append(f"score_gap={score_gap}")
    if regime:
        details.append(f"dominant_regime={regime}")
    if higher_alignment:
        details.append(f"higher_tf_alignment={higher_alignment}")
    return f"current {' '.join(details)} still supports {side_up}"


def same_direction_forecast_support_reason(
    *,
    current_direction: str,
    current_confidence: float,
    side: str,
) -> Optional[str]:
    """Return same-side forecast support for HOLD/reprice decisions.

    This is intentionally softer than ENTRY_CONFIDENCE_MIN: a sub-entry
    confidence forecast cannot open new risk, but when it points with an
    existing position it is evidence that a RANGE thesis should be repriced or
    TP-managed instead of converted into unattended hard loss-close evidence.
    """

    side_up = str(side or "").upper()
    if side_up not in {"LONG", "SHORT"}:
        return None
    direction = str(current_direction or "").upper()
    aligned = "UP" if side_up == "LONG" else "DOWN"
    if direction != aligned:
        return None
    try:
        confidence = float(current_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return f"current forecast {direction} conf={confidence:.2f} supports {side_up}"


def thesis_invalidation_hit_reason(
    thesis: "EntryThesis",
    *,
    side: str,
    current_price: Optional[float],
    price_label: Optional[str] = None,
    buffer_pips: Optional[float] = None,
) -> Optional[str]:
    """Return a reproducible reason when broker truth crosses invalidation."""

    return invalidation_price_hit_reason(
        pair=thesis.pair,
        side=side or thesis.side,
        current_price=current_price,
        invalidation_price=thesis.invalidation_price,
        price_label=price_label,
        buffer_pips=buffer_pips,
    )


def _is_range_rotation_thesis(thesis: "EntryThesis") -> bool:
    entry_dir = str(thesis.forecast_direction or "").upper()
    regime = str(thesis.regime or "").upper()
    evidence = thesis.context_evidence or {}
    method = str(evidence.get("market_context_method") or "").upper()
    drivers = " ".join(str(driver).upper() for driver in thesis.key_drivers)
    return (
        entry_dir == "RANGE"
        and (
            "RANGE" in regime
            or "RANGE_ROTATION" in method
            or "RANGE_ROTATION" in drivers
        )
    )


def _range_rotation_adverse_reason(
    thesis: "EntryThesis",
    *,
    side: str,
    current_price: Optional[float],
    price_label: Optional[str],
    buffer_pips: Optional[float] = None,
) -> Optional[str]:
    """Detect a range-rotation entry drifting against its entry before disaster SL.

    Range entries do not carry a directional forecast ("RANGE"), so waiting only
    for a forecast flip or a wide broker SL can turn a failed rotation into an
    unmanaged loss. This check uses the declared target distance as the trade's
    local geometry and requires the caller to confirm consecutive WEAKENED
    cycles before promoting it to BROKEN.
    """

    if not _is_range_rotation_thesis(thesis):
        return None
    try:
        price = float(current_price) if current_price is not None else 0.0
        entry = float(thesis.entry_price)
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or entry <= 0.0:
        return None

    side_up = str(side or thesis.side or "").upper()
    target = _safe_float(thesis.target_price)
    target_distance = 0.0
    if target is not None:
        if side_up == "LONG" and target > entry:
            target_distance = target - entry
        elif side_up == "SHORT" and target < entry:
            target_distance = entry - target
    buffer_pips_value = thesis_invalidation_buffer_pips(buffer_pips)
    min_distance = invalidation_buffer_price(thesis.pair, buffer_pips_value)
    fail_distance = max(min_distance, target_distance * max(0.0, RANGE_ROTATION_FAIL_TARGET_FRACTION))
    if fail_distance <= 0.0:
        return None

    label = (price_label or ("bid" if side_up == "LONG" else "ask")).strip()
    fail_pips = fail_distance / invalidation_pip_size(thesis.pair)
    if side_up == "LONG":
        trigger = entry - fail_distance
        if price <= trigger:
            return (
                f"RANGE_ROTATION_FAILED: current {label} {price:.5f} <= adverse entry trigger "
                f"{trigger:.5f} (entry {entry:.5f}, fail_distance {fail_pips:.1f}p, "
                f"target_fraction {RANGE_ROTATION_FAIL_TARGET_FRACTION:.2f})"
            )
    elif side_up == "SHORT":
        trigger = entry + fail_distance
        if price >= trigger:
            return (
                f"RANGE_ROTATION_FAILED: current {label} {price:.5f} >= adverse entry trigger "
                f"{trigger:.5f} (entry {entry:.5f}, fail_distance {fail_pips:.1f}p, "
                f"target_fraction {RANGE_ROTATION_FAIL_TARGET_FRACTION:.2f})"
            )
    return None


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
    context_evidence: Dict[str, Any] = field(default_factory=dict)
    # Structural time scope inherited from the entry forecast's horizon_min.
    # None for legacy rows recorded before 2026-06-12; those never expire.
    horizon_hours: Optional[float] = None

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
            "context_evidence": self.context_evidence,
            "horizon_hours": self.horizon_hours,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntryThesis":
        side = str(d.get("side", ""))
        entry_price = float(d.get("entry_price", 0))
        return cls(
            timestamp_utc=str(d.get("timestamp_utc", "")),
            trade_id=str(d.get("trade_id", "")),
            pair=str(d.get("pair", "")),
            side=side,
            entry_price=entry_price,
            forecast_direction=str(d.get("forecast_direction", "UNCLEAR")),
            forecast_confidence=float(d.get("forecast_confidence", 0)),
            regime=d.get("regime"),
            invalidation_price=_first_directional_price(
                side=side,
                entry_price=entry_price,
                role="INVALIDATION",
                values=(d.get("invalidation_price"),),
            ),
            target_price=_first_directional_price(
                side=side,
                entry_price=entry_price,
                role="TARGET",
                values=(d.get("target_price"),),
            ),
            key_drivers=list(d.get("key_drivers", [])),
            context_evidence=dict(d.get("context_evidence") or {}) if isinstance(d.get("context_evidence"), dict) else {},
            horizon_hours=_optional_positive_float(d.get("horizon_hours")),
        )


def _optional_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


@dataclass
class PendingEntryThesis:
    timestamp_utc: str
    order_id: str
    pair: str
    side: str
    entry_price: float
    forecast_direction: str
    forecast_confidence: float
    regime: Optional[str]
    invalidation_price: Optional[float]
    target_price: Optional[float]
    key_drivers: List[str] = field(default_factory=list)
    lane_id: Optional[str] = None
    context_evidence: Dict[str, Any] = field(default_factory=dict)
    horizon_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_utc": self.timestamp_utc,
            "order_id": self.order_id,
            "pair": self.pair,
            "side": self.side,
            "entry_price": self.entry_price,
            "forecast_direction": self.forecast_direction,
            "forecast_confidence": self.forecast_confidence,
            "regime": self.regime,
            "invalidation_price": self.invalidation_price,
            "target_price": self.target_price,
            "key_drivers": self.key_drivers,
            "lane_id": self.lane_id,
            "context_evidence": self.context_evidence,
            "horizon_hours": self.horizon_hours,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PendingEntryThesis":
        side = str(d.get("side", ""))
        entry_price = float(d.get("entry_price", 0))
        return cls(
            timestamp_utc=str(d.get("timestamp_utc", "")),
            order_id=str(d.get("order_id", "")),
            pair=str(d.get("pair", "")),
            side=side,
            entry_price=entry_price,
            forecast_direction=str(d.get("forecast_direction", "UNCLEAR")),
            forecast_confidence=float(d.get("forecast_confidence", 0)),
            regime=d.get("regime"),
            invalidation_price=_first_directional_price(
                side=side,
                entry_price=entry_price,
                role="INVALIDATION",
                values=(d.get("invalidation_price"),),
            ),
            target_price=_first_directional_price(
                side=side,
                entry_price=entry_price,
                role="TARGET",
                values=(d.get("target_price"),),
            ),
            key_drivers=list(d.get("key_drivers", [])),
            lane_id=d.get("lane_id"),
            context_evidence=dict(d.get("context_evidence") or {}) if isinstance(d.get("context_evidence"), dict) else {},
            horizon_hours=_optional_positive_float(d.get("horizon_hours")),
        )


@dataclass(frozen=True)
class EntryThesisRecordResult:
    status: str
    trade_id: Optional[str] = None
    order_id: Optional[str] = None
    issue: Optional[str] = None
    thesis: Optional[EntryThesis] = None
    pending: Optional[PendingEntryThesis] = None

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {"status": self.status}
        if self.trade_id:
            payload["trade_id"] = self.trade_id
        if self.order_id:
            payload["order_id"] = self.order_id
        if self.issue:
            payload["issue"] = self.issue
        if self.thesis is not None:
            payload["thesis"] = self.thesis.to_dict()
        if self.pending is not None:
            payload["pending"] = self.pending.to_dict()
        return payload


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
    status: str  # "STILL_VALID" | "WEAKENED" | "BROKEN" | "UNVERIFIABLE"
    verdict: str  # "HOLD" | "EXTEND" | "RECOMMEND_CLOSE" | "REQUIRE_THESIS_REPAIR"
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


def _parse_utc_timestamp(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # OANDA timestamps may carry nanoseconds. Python's stdlib accepts only
        # microseconds, so truncate the fractional part for age/audit math.
        import re

        match = re.match(r"^(.*\.\d{6})\d+([+-]\d{2}:\d{2})$", text)
        if not match:
            return None
        try:
            parsed = datetime.fromisoformat(match.group(1) + match.group(2))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def record_entry_thesis(thesis: EntryThesis, data_root: Path) -> None:
    """Append entry thesis to ledger."""
    if _is_disabled():
        return
    path = data_root / LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(thesis.to_dict(), ensure_ascii=False) + "\n")


def record_pending_entry_thesis(thesis: PendingEntryThesis, data_root: Path) -> None:
    """Append pending-order thesis keyed by OANDA order id.

    STOP/LIMIT orders receive a trade id only when OANDA later emits an
    ORDER_FILL transaction. This sidecar preserves the original entry reason so
    the fill sync can promote it to `entry_thesis_ledger.jsonl`.
    """
    if _is_disabled():
        return
    if load_pending_entry_thesis(thesis.order_id, data_root) is not None:
        return
    path = data_root / PENDING_LEDGER_FILENAME
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


def load_pending_entry_thesis(order_id: str, data_root: Path) -> Optional[PendingEntryThesis]:
    """Load pending thesis for an OANDA order id."""
    path = data_root / PENDING_LEDGER_FILENAME
    if not path.exists():
        return None
    latest: Optional[PendingEntryThesis] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("order_id", "")) == str(order_id):
                latest = PendingEntryThesis.from_dict(d)
    except OSError:
        return None
    return latest


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


def _response_order_create(response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not response or not isinstance(response, dict):
        return None
    create = response.get("orderCreateTransaction")
    return create if isinstance(create, dict) else None


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


def _response_pending_order_id(response: Optional[Dict[str, Any]]) -> Optional[str]:
    create = _response_order_create(response)
    if not create:
        return None
    if str(create.get("type") or "").upper() not in PENDING_ORDER_TYPES:
        return None
    order_id = create.get("id") or create.get("orderID")
    return str(order_id) if order_id is not None else None


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


def _intent_side(intent: Any) -> str:
    side_attr = getattr(intent, "side", None)
    return (getattr(side_attr, "value", None) or str(side_attr or "")).upper()


def _build_key_drivers(*, forecast: Dict[str, Any], metadata: Dict[str, Any], thesis_text: str) -> List[str]:
    key_drivers: List[str] = []
    if forecast.get("direction"):
        key_drivers.append(
            f"forecast={forecast.get('direction')}@conf={float(forecast.get('confidence', 0)):.2f}"
        )
    for k in ("desk", "campaign_role", "regime_state", "target_reward_risk"):
        v = metadata.get(k)
        if v not in (None, ""):
            key_drivers.append(f"{k}={v}")
    lane_id = metadata.get("parent_lane_id") or metadata.get("lane_id")
    if lane_id:
        key_drivers.append(f"lane_id={lane_id}")
    if thesis_text:
        key_drivers.append(thesis_text[:120])
    return key_drivers[:6]


def _context_text(value: Any, *, limit: int = _CONTEXT_TEXT_LIMIT) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:limit]


def _context_text_list(value: Any, *, limit: int = _CONTEXT_LIST_LIMIT) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    elif value in (None, ""):
        raw_items = []
    else:
        raw_items = [value]
    items: list[str] = []
    for item in raw_items:
        text = _context_text(item)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _jsonable_context_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable_context_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_context_value(item) for item in value]
    return str(value)


def _context_refs_from_texts(texts: list[str]) -> list[str]:
    refs: list[str] = []
    for text in texts:
        for match in _CONTEXT_REF_RE.findall(text):
            if match not in refs:
                refs.append(match)
    return refs[:_CONTEXT_LIST_LIMIT]


def _asset_symbols_from_texts(texts: list[str], *, traded_pair: str = "") -> list[str]:
    symbols: list[str] = []
    traded = traded_pair.upper().strip()
    for text in texts:
        for match in _CONTEXT_ASSET_RE.findall(text):
            if match == traded or _is_plain_fx_pair_symbol(match):
                continue
            if match not in symbols:
                symbols.append(match)
    return symbols[:_CONTEXT_LIST_LIMIT]


def _is_plain_fx_pair_symbol(symbol: str) -> bool:
    base, sep, quote = symbol.partition("_")
    return bool(sep and base in _FX_CONTEXT_CURRENCIES and quote in _FX_CONTEXT_CURRENCIES)


def _market_context_field(intent: Any, name: str) -> str:
    market_context = getattr(intent, "market_context", None)
    if market_context is None:
        return ""
    return _context_text(getattr(market_context, name, ""))


def _news_context_from_metadata(metadata: Dict[str, Any]) -> list[str]:
    news_context: list[str] = []
    for key, value in metadata.items():
        key_text = str(key).lower()
        value_text = json.dumps(_jsonable_context_value(value), ensure_ascii=False, sort_keys=True)
        haystack = f"{key_text} {value_text.lower()}"
        if not any(token in haystack for token in _NEWS_CONTEXT_TOKENS):
            continue
        text = _context_text(f"{key}={value_text}")
        if text and text not in news_context:
            news_context.append(text)
        if len(news_context) >= _CONTEXT_LIST_LIMIT:
            break
    return news_context


def _build_context_evidence(*, intent: Any, metadata: Dict[str, Any], forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot advisory context that motivated entry for later P/L attribution."""

    evidence: Dict[str, Any] = {}
    matrix_ref = _context_text(metadata.get("market_context_matrix_ref"))
    if matrix_ref:
        evidence["market_context_matrix_ref"] = matrix_ref

    for key in (
        "matrix_support_count",
        "matrix_reject_count",
        "matrix_warning_count",
    ):
        if metadata.get(key) not in (None, ""):
            evidence[key] = _jsonable_context_value(metadata.get(key))

    for key in (
        "matrix_support_layers",
        "matrix_reject_layers",
        "matrix_warning_layers",
    ):
        values = _context_text_list(metadata.get(key))
        if values:
            evidence[key] = values

    explicit_context_refs: list[str] = []
    for key in (
        "matrix_context_refs",
        "matrix_support_refs",
        "matrix_reject_refs",
        "matrix_warning_refs",
    ):
        values = _context_text_list(metadata.get(key))
        if values:
            evidence[key] = values
            explicit_context_refs.extend(values)

    context_texts: list[str] = []
    for key in (
        "matrix_support_context",
        "matrix_reject_context",
        "matrix_warning_context",
    ):
        values = _context_text_list(metadata.get(key))
        if values:
            evidence[key] = values
            context_texts.extend(values)

    for key in (
        "strongest_matrix_support",
        "strongest_matrix_reject",
        "strongest_matrix_warning",
    ):
        text = _context_text(metadata.get(key))
        if text:
            evidence[key] = text
            context_texts.append(text)

    chart_story = _market_context_field(intent, "chart_story")
    if chart_story:
        evidence["chart_story_excerpt"] = chart_story
        context_texts.append(chart_story)
    for key in ("regime", "method", "session", "event_risk"):
        text = _market_context_field(intent, key)
        if text:
            evidence[f"market_context_{key}"] = text

    news_context = _news_context_from_metadata(metadata)
    if news_context:
        evidence["news_context"] = news_context
        context_texts.extend(news_context)

    forecast_news_context = _news_context_from_metadata(forecast)
    if forecast_news_context:
        evidence["forecast_news_context"] = forecast_news_context
        context_texts.extend(forecast_news_context)

    refs = _context_refs_from_texts(([matrix_ref] if matrix_ref else []) + explicit_context_refs + context_texts)
    if refs:
        evidence["evidence_refs"] = refs
        evidence["context_asset_refs"] = [
            ref for ref in refs if ref.startswith("context_asset:") or ref.startswith("cross:")
        ][:_CONTEXT_LIST_LIMIT]

    asset_symbols = _asset_symbols_from_texts(context_texts, traded_pair=str(getattr(intent, "pair", "") or ""))
    if asset_symbols:
        evidence["context_asset_symbols"] = asset_symbols

    return evidence


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_price(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None and parsed > 0.0:
            return parsed
    return None


def _first_directional_price(
    *,
    side: str,
    entry_price: Any,
    role: str,
    values: tuple[Any, ...],
) -> Optional[float]:
    """Return the first price that is on the correct side of entry.

    Forecast geometry is advisory at thesis-record time; broker/intent
    protection prices are the executable truth. If a forecast target or
    invalidation is on the wrong side of the actual entry, recording it as the
    entry thesis poisons later thesis-evolution checks. Keep the record
    directional, or leave the field empty when no valid candidate exists.
    """
    side_up = str(side or "").upper()
    if side_up not in {"LONG", "SHORT"}:
        return _first_price(*values)
    entry = _safe_float(entry_price)
    if entry is None or entry <= 0.0:
        return _first_price(*values)
    role_up = str(role or "").upper()
    for value in values:
        parsed = _safe_float(value)
        if parsed is None or parsed <= 0.0:
            continue
        if role_up == "TARGET":
            if (side_up == "LONG" and parsed > entry) or (side_up == "SHORT" and parsed < entry):
                return parsed
        elif role_up == "INVALIDATION":
            if (side_up == "LONG" and parsed < entry) or (side_up == "SHORT" and parsed > entry):
                return parsed
    return None


def _nested_price(payload: Any, key: str) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    nested = payload.get(key)
    if isinstance(nested, dict):
        return _first_price(nested.get("price"))
    return None


def _response_protection_price(response: Optional[Dict[str, Any]], kind: str) -> Optional[float]:
    if not isinstance(response, dict):
        return None
    if kind == "tp":
        nested_key = "takeProfitOnFill"
        transaction_keys = ("takeProfitOrderTransaction",)
    elif kind == "sl":
        nested_key = "stopLossOnFill"
        transaction_keys = ("stopLossOrderTransaction",)
    else:
        return None

    candidates: list[Any] = []
    for key in ("orderCreateTransaction", "orderFillTransaction"):
        payload = response.get(key)
        candidates.append(_nested_price(payload, nested_key))
    for key in transaction_keys:
        payload = response.get(key)
        if isinstance(payload, dict):
            candidates.append(payload.get("price"))
    candidates.append(_nested_price(response, nested_key))
    return _first_price(*candidates)


def _transaction_protection_price(transaction: Dict[str, Any], kind: str) -> Optional[float]:
    if kind == "tp":
        return _nested_price(transaction, "takeProfitOnFill")
    if kind == "sl":
        return _nested_price(transaction, "stopLossOnFill")
    return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _side_from_units(units: Optional[int]) -> Optional[str]:
    if units is None or units == 0:
        return None
    return "LONG" if units > 0 else "SHORT"


def record_entry_thesis_from_response_result(
    *,
    response: Optional[Dict[str, Any]],
    intent: Any,
    data_root: Path,
    now: Optional[datetime] = None,
) -> EntryThesisRecordResult:
    """Build and verify entry-thesis sidecars from a SENT OANDA response.

    Called from execution.LiveOrderGateway right after a successful
    `post_order_json` returns. Reads the canonical forecast that
    trader_brain just wrote for this pair, snapshots side/entry_price,
    and appends to `data/entry_thesis_ledger.jsonl` or the pending-order
    sidecar. Failure here does not raise, but the result is explicit so the
    live gateway can surface a verification gap.
    """
    if _is_disabled():
        return EntryThesisRecordResult(
            status="DISABLED",
            issue="QR_DISABLE_ENTRY_THESIS_LEDGER is active",
        )
    try:
        trade_id = _response_trade_id(response)
        if not trade_id:
            pending_order_id = _response_pending_order_id(response)
            if not pending_order_id:
                return EntryThesisRecordResult(status="NOT_APPLICABLE")
            pair = str(getattr(intent, "pair", ""))
            side_up = _intent_side(intent)
            if not pair or side_up not in ("LONG", "SHORT"):
                return EntryThesisRecordResult(
                    status="FAILED",
                    order_id=pending_order_id,
                    issue="pending order response has no recordable pair/side",
                )
            forecast = load_latest_forecast(pair, data_root) or {}
            metadata = dict(getattr(intent, "metadata", {}) or {})
            context_evidence = _build_context_evidence(
                intent=intent,
                metadata=metadata,
                forecast=forecast,
            )
            now = now or datetime.now(timezone.utc)
            pending = PendingEntryThesis(
                timestamp_utc=now.isoformat().replace("+00:00", "Z"),
                order_id=pending_order_id,
                pair=pair,
                side=side_up,
                entry_price=float(getattr(intent, "entry", 0) or 0),
                forecast_direction=str(forecast.get("direction") or "UNCLEAR"),
                forecast_confidence=float(forecast.get("confidence") or 0.0),
                horizon_hours=_thesis_horizon_hours_from_forecast(forecast),
                regime=str(forecast.get("regime") or metadata.get("regime_state") or "") or None,
                invalidation_price=_first_directional_price(
                    side=side_up,
                    entry_price=getattr(intent, "entry", None),
                    role="INVALIDATION",
                    values=(
                        forecast.get("invalidation_price"),
                        _response_protection_price(response, "sl"),
                        getattr(intent, "sl", None),
                    ),
                ),
                target_price=_first_directional_price(
                    side=side_up,
                    entry_price=getattr(intent, "entry", None),
                    role="TARGET",
                    values=(
                        forecast.get("target_price"),
                        _response_protection_price(response, "tp"),
                        getattr(intent, "tp", None),
                    ),
                ),
                key_drivers=_build_key_drivers(
                    forecast=forecast,
                    metadata=metadata,
                    thesis_text=str(getattr(intent, "thesis", "") or ""),
                ),
                lane_id=str(metadata.get("parent_lane_id") or metadata.get("lane_id") or ""),
                context_evidence=context_evidence,
            )
            record_pending_entry_thesis(
                pending,
                data_root,
            )
            loaded_pending = load_pending_entry_thesis(pending_order_id, data_root)
            if loaded_pending is None:
                return EntryThesisRecordResult(
                    status="FAILED",
                    order_id=pending_order_id,
                    issue="pending entry thesis write could not be verified",
                )
            return EntryThesisRecordResult(
                status="PENDING_RECORDED",
                order_id=pending_order_id,
                pending=loaded_pending,
            )
        pair = str(getattr(intent, "pair", ""))
        side_up = _intent_side(intent)
        if not pair or side_up not in ("LONG", "SHORT"):
            return EntryThesisRecordResult(
                status="FAILED",
                trade_id=trade_id,
                issue="fill response has no recordable pair/side",
            )
        fill_price = _response_fill_price(response) or float(getattr(intent, "entry", 0) or 0)

        forecast = load_latest_forecast(pair, data_root) or {}
        metadata = dict(getattr(intent, "metadata", {}) or {})
        context_evidence = _build_context_evidence(
            intent=intent,
            metadata=metadata,
            forecast=forecast,
        )

        regime = forecast.get("regime") or metadata.get("regime_state") or None

        now = now or datetime.now(timezone.utc)
        existing = load_entry_thesis(trade_id, data_root)
        if existing is not None:
            return EntryThesisRecordResult(status="RECORDED", trade_id=trade_id, thesis=existing)
        entry_thesis = EntryThesis(
            timestamp_utc=now.isoformat().replace("+00:00", "Z"),
            trade_id=trade_id,
            pair=pair,
            side=side_up,
            entry_price=float(fill_price),
            forecast_direction=str(forecast.get("direction") or "UNCLEAR"),
            forecast_confidence=float(forecast.get("confidence") or 0.0),
            horizon_hours=_thesis_horizon_hours_from_forecast(forecast),
            regime=str(regime) if regime else None,
            invalidation_price=_first_directional_price(
                side=side_up,
                entry_price=fill_price,
                role="INVALIDATION",
                values=(
                    forecast.get("invalidation_price"),
                    _response_protection_price(response, "sl"),
                    getattr(intent, "sl", None),
                ),
            ),
            target_price=_first_directional_price(
                side=side_up,
                entry_price=fill_price,
                role="TARGET",
                values=(
                    forecast.get("target_price"),
                    _response_protection_price(response, "tp"),
                    getattr(intent, "tp", None),
                ),
            ),
            key_drivers=_build_key_drivers(
                forecast=forecast,
                metadata=metadata,
                thesis_text=str(getattr(intent, "thesis", "") or ""),
            ),
            context_evidence=context_evidence,
        )
        record_entry_thesis(entry_thesis, data_root)
        loaded = load_entry_thesis(trade_id, data_root)
        if loaded is None:
            return EntryThesisRecordResult(
                status="FAILED",
                trade_id=trade_id,
                issue="entry thesis write could not be verified",
            )
        return EntryThesisRecordResult(status="RECORDED", trade_id=trade_id, thesis=loaded)
    except Exception as exc:
        return EntryThesisRecordResult(status="FAILED", issue=str(exc))


def record_entry_thesis_from_response(
    *,
    response: Optional[Dict[str, Any]],
    intent: Any,
    data_root: Path,
    now: Optional[datetime] = None,
) -> Optional[EntryThesis]:
    """Backward-compatible wrapper returning only immediate-fill thesis."""
    return record_entry_thesis_from_response_result(
        response=response,
        intent=intent,
        data_root=data_root,
        now=now,
    ).thesis


def record_entry_thesis_from_order_fill(
    *,
    transaction: Dict[str, Any],
    data_root: Path,
) -> Optional[EntryThesis]:
    """Promote a pending-order thesis when OANDA later emits ORDER_FILL.

    Pending entry orders are accepted before they have a trade id. The live
    gateway stores the motivating thesis under the OANDA order id; this function
    is called by execution-ledger sync when broker truth reports the fill.
    """
    if _is_disabled():
        return None
    try:
        opened = transaction.get("tradeOpened")
        if not isinstance(opened, dict):
            return None
        trade_id = opened.get("tradeID")
        order_id = transaction.get("orderID")
        if trade_id is None or order_id is None:
            return None
        existing = load_entry_thesis(str(trade_id), data_root)
        if existing is not None:
            return existing
        pending = load_pending_entry_thesis(str(order_id), data_root)
        if pending is None:
            return None
        units = _safe_int(opened.get("units")) or _safe_int(transaction.get("units"))
        side = _side_from_units(units) or pending.side
        if side not in ("LONG", "SHORT"):
            return None
        price = _safe_float(opened.get("price")) or _safe_float(transaction.get("price")) or pending.entry_price
        if price is None:
            return None
        entry_thesis = EntryThesis(
            timestamp_utc=str(transaction.get("time") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            trade_id=str(trade_id),
            pair=str(transaction.get("instrument") or pending.pair),
            side=side,
            entry_price=float(price),
            forecast_direction=pending.forecast_direction,
            forecast_confidence=pending.forecast_confidence,
            horizon_hours=pending.horizon_hours,
            regime=pending.regime,
            invalidation_price=_first_directional_price(
                side=side,
                entry_price=price,
                role="INVALIDATION",
                values=(
                    pending.invalidation_price,
                    _transaction_protection_price(transaction, "sl"),
                ),
            ),
            target_price=_first_directional_price(
                side=side,
                entry_price=price,
                role="TARGET",
                values=(
                    pending.target_price,
                    _transaction_protection_price(transaction, "tp"),
                ),
            ),
            key_drivers=list(pending.key_drivers),
            context_evidence=dict(pending.context_evidence),
        )
        record_entry_thesis(entry_thesis, data_root)
        return entry_thesis
    except Exception:
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
    current_price: Optional[float] = None,
    current_price_label: Optional[str] = None,
    invalidation_buffer_pips: Optional[float] = None,
    pair_chart: Optional[Dict[str, Any]] = None,
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
    opened = _parse_utc_timestamp(open_time_utc) or _parse_utc_timestamp(thesis.timestamp_utc)
    if opened is not None:
        age_hours = (now - opened).total_seconds() / 3600

    current_dir = getattr(current_forecast, "direction", "UNCLEAR")
    current_conf = float(getattr(current_forecast, "confidence", 0))

    # Status classification
    entry_dir = thesis.forecast_direction
    side_up = side.upper()
    aligned_dir = "UP" if side_up == "LONG" else "DOWN"
    forecast_support_reason = same_direction_forecast_support_reason(
        current_direction=current_dir,
        current_confidence=current_conf,
        side=side_up,
    )

    reasons: List[str] = []
    invalidation_reason = thesis_invalidation_hit_reason(
        thesis,
        side=side_up,
        current_price=current_price,
        price_label=current_price_label,
        buffer_pips=invalidation_buffer_pips,
    )
    if invalidation_reason:
        technical_reason = technical_invalidation_confirmation_reason(pair_chart, side=side_up)
        if not technical_reason:
            reasons.append(
                f"{invalidation_reason}; waiting for chart/technical confirmation"
            )
        elif forecast_support_reason:
            reasons.append(
                f"{invalidation_reason}; {technical_reason}; {forecast_support_reason}, so "
                "the invalidation hit is HOLD/reprice/TP rebalance evidence until the "
                "same-direction recovery edge disappears or higher-timeframe structure breaks"
            )
        else:
            return ThesisEvolution(
                trade_id=trade_id, pair=pair, side=side_up,
                age_hours=age_hours,
                entry_forecast=entry_dir,
                current_forecast=current_dir,
                entry_confidence=thesis.forecast_confidence,
                current_confidence=current_conf,
                entry_regime=thesis.regime,
                current_regime=current_regime,
                status="BROKEN",
                verdict="RECOMMEND_CLOSE",
                rationale=f"{invalidation_reason}; {technical_reason}",
            )

    if current_dir == entry_dir and current_dir == aligned_dir:
        if invalidation_reason:
            reasons.append("price invalidation is buffered, but chart confirmation is not complete")
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
        if current_conf < ENTRY_CONFIDENCE_MIN:
            status = "WEAKENED"
            verdict = "HOLD"
            reasons.append(
                f"forecast direction flipped to {current_dir}, but confidence "
                f"{current_conf:.2f} < ENTRY_CONFIDENCE_MIN={ENTRY_CONFIDENCE_MIN:.2f}; "
                "do not convert a weak forecast into Gate A close evidence"
            )
        else:
            return ThesisEvolution(
                trade_id=trade_id, pair=pair, side=side_up,
                age_hours=age_hours,
                entry_forecast=entry_dir,
                current_forecast=current_dir,
                entry_confidence=thesis.forecast_confidence,
                current_confidence=current_conf,
                entry_regime=thesis.regime,
                current_regime=current_regime,
                status="BROKEN",
                verdict="RECOMMEND_CLOSE",
                rationale=(
                    f"FORECAST FLIPPED: entry {entry_dir} → current {current_dir} "
                    f"(position {side_up})"
                ),
            )
    elif forecast_support_reason:
        status = "WEAKENED"
        verdict = "HOLD"
        reasons.append(
            f"{forecast_support_reason}, but entry thesis was {entry_dir}; "
            "treat the position as HOLD/reprice/TP rebalance until hard invalidation "
            "or an opposing high-confidence forecast appears"
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

    # RANGE_ROTATION fail-fast: a RANGE entry has no directional thesis to
    # recover back to. If it is repeatedly WEAKENED and has already moved
    # against entry by a meaningful fraction of its own TP geometry, do not
    # wait for the wider disaster SL or full forecast-horizon expiry.
    range_adverse_reason = _range_rotation_adverse_reason(
        thesis,
        side=side_up,
        current_price=current_price,
        price_label=current_price_label,
        buffer_pips=invalidation_buffer_pips,
    )
    if status == "WEAKENED" and range_adverse_reason:
        if forecast_support_reason:
            reasons.append(
                f"{range_adverse_reason}; {forecast_support_reason}, so adverse range drift "
                "is a HOLD/reprice/TP-rebalance problem, not hard Gate A close evidence"
            )
        elif _has_prior_weakened_or_broken_cycle(trade_id=trade_id, data_root=data_root, now=now):
            status = "BROKEN"
            verdict = "RECOMMEND_CLOSE"
            reasons.append(
                f"{range_adverse_reason}; thesis was WEAKENED across consecutive checks "
                "without strong current directional support"
            )
        else:
            reasons.append(
                f"{range_adverse_reason}; waiting for consecutive WEAKENED check before Gate A"
            )

    # THESIS_EXPIRED escalation (2026-06-12, AGENT_CONTRACT §10): a WEAKENED
    # thesis that has outlived its declared horizon without reaching target or
    # invalidation is no longer the entry thesis — holding it is an unpriced
    # new position. Requires the PREVIOUS archived check (older than the
    # same-cycle dedup floor) to also be WEAKENED or worse — the same
    # smallest-repetition defense as the §8 loss-streak gate — so a single
    # transient RANGE/UNCLEAR forecast cycle cannot flush a position.
    # Ledger evidence 2026-06-11: market closes held past 12h were 22/22
    # losses averaging -2,310 JPY while TP exits stayed profitable in every
    # hold bucket; the leak is specifically decayed theses held past scope.
    # STILL_VALID positions never expire on the clock alone.
    if (
        status == "WEAKENED"
        and thesis.horizon_hours is not None
        and thesis.horizon_hours > 0
        and age_hours > thesis.horizon_hours
    ):
        if _has_prior_weakened_or_broken_cycle(trade_id=trade_id, data_root=data_root, now=now):
            chart_support_reason = same_direction_chart_support_reason(pair_chart, side=side_up)
            if chart_support_reason or forecast_support_reason:
                support_reason = chart_support_reason or forecast_support_reason
                reasons.append(
                    f"THESIS_EXPIRED_SOFT: age {age_hours:.1f}h exceeds declared horizon "
                    f"{thesis.horizon_hours:.1f}h across consecutive WEAKENED checks, "
                    f"but {support_reason}; treat as HOLD/reprice/TP rebalance instead "
                    "of unattended loss-side CLOSE"
                )
            else:
                status = "BROKEN"
                verdict = "RECOMMEND_CLOSE"
                reasons.append(
                    f"THESIS_EXPIRED: age {age_hours:.1f}h exceeds declared horizon "
                    f"{thesis.horizon_hours:.1f}h with neither target nor invalidation reached, "
                    "and the thesis has been WEAKENED across consecutive checks without "
                    "current same-direction chart support; the entry thesis no longer exists "
                    "— close or re-justify via a fresh receipt"
                )

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


def evaluate_missing_entry_thesis_position(
    *,
    trade_id: str,
    pair: str,
    side: str,
    open_time_utc: Optional[str],
    current_forecast: Any,
    current_regime: Optional[str],
    now: Optional[datetime] = None,
) -> ThesisEvolution:
    """Surface a trader-owned open position that has no entry thesis row.

    The row is a hard management blocker, not standalone Gate A close evidence.
    No-ledger close evidence still comes from position_thesis_report with
    entry-buffer and multi-TF confirmation.
    """

    now = now or datetime.now(timezone.utc)
    opened = _parse_utc_timestamp(open_time_utc)
    age_hours = (now - opened).total_seconds() / 3600 if opened is not None else 0.0
    side_up = side.upper()
    current_dir = getattr(current_forecast, "direction", "UNCLEAR")
    current_conf = float(getattr(current_forecast, "confidence", 0))
    aligned_dir = "UP" if side_up == "LONG" else "DOWN"
    support_word = "supports" if current_dir == aligned_dir else "does not confirm"
    return ThesisEvolution(
        trade_id=trade_id,
        pair=pair,
        side=side_up,
        age_hours=age_hours,
        entry_forecast="MISSING_ENTRY_THESIS",
        current_forecast=current_dir,
        entry_confidence=0.0,
        current_confidence=current_conf,
        entry_regime=None,
        current_regime=current_regime,
        status=UNVERIFIABLE_STATUS,
        verdict=REQUIRE_THESIS_REPAIR_VERDICT,
        rationale=(
            "missing entry_thesis_ledger row; original entry reason cannot be "
            "machine-verified, so thesis_evolution will not authorize Gate A; "
            f"current forecast {current_dir} conf={current_conf:.2f} {support_word} {side_up}; "
            "hard management blocker: do not expand TP, do not add new risk, "
            "and use position_thesis no-ledger fallback or forecast_persistence "
            "for machine-checkable close/repair evidence"
        ),
    )


def evaluate_all_open_positions(
    positions: List[Any],
    *,
    current_forecasts_by_pair: Dict[str, Any],
    current_regimes_by_pair: Dict[str, Optional[str]],
    data_root: Path,
    quotes_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_charts_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> List[ThesisEvolution]:
    """For every trader-owned open position, compute ThesisEvolution.

    `current_forecasts_by_pair` and `current_regimes_by_pair` are
    keyed by OANDA pair string (e.g. "EUR_JPY"). Forecasts can be
    `DirectionalForecast` instances or anything with `.direction` and
    `.confidence`. Positions without an entry thesis record (legacy
    pre-2026-05-15 or failed fill promotion) are reported as UNVERIFIABLE /
    REQUIRE_THESIS_REPAIR hard management blockers instead of being skipped
    silently.
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
        quote = (quotes_by_pair or {}).get(pair) or {}
        current_price = None
        current_price_label = None
        try:
            if side_up == "LONG":
                current_price = float(quote.get("bid"))
                current_price_label = "bid"
            elif side_up == "SHORT":
                current_price = float(quote.get("ask"))
                current_price_label = "ask"
        except (TypeError, ValueError):
            current_price = None
            current_price_label = None
        ev = evaluate_thesis_evolution(
            trade_id=trade_id,
            pair=pair,
            side=side_up,
            open_time_utc=open_time,
            current_forecast=forecast,
            current_regime=regime,
            data_root=data_root,
            current_price=current_price,
            current_price_label=current_price_label,
            pair_chart=(pair_charts_by_pair or {}).get(pair),
            now=now,
        )
        if ev is None:
            ev = evaluate_missing_entry_thesis_position(
                trade_id=trade_id,
                pair=pair,
                side=side_up,
                open_time_utc=open_time,
                current_forecast=forecast,
                current_regime=regime,
                now=now,
            )
        out.append(ev)
    return out


def write_thesis_evolution_report(
    evolutions: List[ThesisEvolution],
    *,
    data_root: Path,
    now: Optional[datetime] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Write the per-cycle `thesis_evolution_report.json`."""
    report_path = output_path or data_root / "thesis_evolution_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    missing_thesis = [e.trade_id for e in evolutions if e.entry_forecast == "MISSING_ENTRY_THESIS"]
    blocking_thesis = [
        e.trade_id
        for e in evolutions
        if e.status == UNVERIFIABLE_STATUS or e.verdict == REQUIRE_THESIS_REPAIR_VERDICT
    ]
    payload = {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "count": len(evolutions),
        "by_status": {
            "STILL_VALID": sum(1 for e in evolutions if e.status == "STILL_VALID"),
            "WEAKENED": sum(1 for e in evolutions if e.status == "WEAKENED"),
            "BROKEN": sum(1 for e in evolutions if e.status == "BROKEN"),
            UNVERIFIABLE_STATUS: sum(1 for e in evolutions if e.status == UNVERIFIABLE_STATUS),
        },
        "entry_thesis_coverage": {
            "recorded": len(evolutions) - len(missing_thesis),
            "missing": len(missing_thesis),
            "missing_trade_ids": missing_thesis,
            "blocking": bool(blocking_thesis),
            "blocking_trade_ids": blocking_thesis,
        },
        "evolutions": [e.to_dict() for e in evolutions],
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Durable append-only history: the report above is overwritten each cycle,
    # so per-trade WEAKENED/BROKEN latency was unmeasurable beyond the current
    # cycle and the THESIS_EXPIRED consecutive-check needs the prior state.
    # Best-effort: history is an audit trail, not a gate.
    history_path = data_root / THESIS_EVOLUTION_HISTORY_FILENAME
    try:
        with history_path.open("a", encoding="utf-8") as fh:
            for e in evolutions:
                row = e.to_dict()
                row["generated_at_utc"] = payload["generated_at_utc"]
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass
    return report_path
