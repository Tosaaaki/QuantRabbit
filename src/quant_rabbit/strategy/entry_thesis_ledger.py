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
- BROKEN → fresh Gate A evidence for close; Gate B still required

No auto-close from this module. The kill switch (QR_DISABLE_AUTO_CLOSE)
stays on; close decisions go through gpt_trader and still need Gate B.

Kill switch: `QR_DISABLE_ENTRY_THESIS_LEDGER=1`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN


LEDGER_FILENAME = "entry_thesis_ledger.jsonl"
PENDING_LEDGER_FILENAME = "pending_entry_thesis_ledger.jsonl"
FORECAST_HISTORY_FILENAME = "forecast_history.jsonl"
PENDING_ORDER_TYPES = {"LIMIT_ORDER", "STOP_ORDER", "MARKET_IF_TOUCHED_ORDER"}
DEFAULT_THESIS_INVALIDATION_BUFFER_PIPS = 2.0
TECHNICAL_INVALIDATION_TFS = ("M5", "M15", "M30", "H1")


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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PendingEntryThesis":
        return cls(
            timestamp_utc=str(d.get("timestamp_utc", "")),
            order_id=str(d.get("order_id", "")),
            pair=str(d.get("pair", "")),
            side=str(d.get("side", "")),
            entry_price=float(d.get("entry_price", 0)),
            forecast_direction=str(d.get("forecast_direction", "UNCLEAR")),
            forecast_confidence=float(d.get("forecast_confidence", 0)),
            regime=d.get("regime"),
            invalidation_price=d.get("invalidation_price"),
            target_price=d.get("target_price"),
            key_drivers=list(d.get("key_drivers", [])),
            lane_id=d.get("lane_id"),
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


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
            pending_order_id = _response_pending_order_id(response)
            if not pending_order_id:
                return None
            pair = str(getattr(intent, "pair", ""))
            side_up = _intent_side(intent)
            if not pair or side_up not in ("LONG", "SHORT"):
                return None
            forecast = load_latest_forecast(pair, data_root) or {}
            metadata = dict(getattr(intent, "metadata", {}) or {})
            now = now or datetime.now(timezone.utc)
            record_pending_entry_thesis(
                PendingEntryThesis(
                    timestamp_utc=now.isoformat().replace("+00:00", "Z"),
                    order_id=pending_order_id,
                    pair=pair,
                    side=side_up,
                    entry_price=float(getattr(intent, "entry", 0) or 0),
                    forecast_direction=str(forecast.get("direction") or "UNCLEAR"),
                    forecast_confidence=float(forecast.get("confidence") or 0.0),
                    regime=str(forecast.get("regime") or metadata.get("regime_state") or "") or None,
                    invalidation_price=forecast.get("invalidation_price") or getattr(intent, "sl", None),
                    target_price=forecast.get("target_price") or getattr(intent, "tp", None),
                    key_drivers=_build_key_drivers(
                        forecast=forecast,
                        metadata=metadata,
                        thesis_text=str(getattr(intent, "thesis", "") or ""),
                    ),
                    lane_id=str(metadata.get("parent_lane_id") or metadata.get("lane_id") or ""),
                ),
                data_root,
            )
            return None
        pair = str(getattr(intent, "pair", ""))
        side_up = _intent_side(intent)
        if not pair or side_up not in ("LONG", "SHORT"):
            return None
        fill_price = _response_fill_price(response) or float(getattr(intent, "entry", 0) or 0)

        forecast = load_latest_forecast(pair, data_root) or {}
        metadata = dict(getattr(intent, "metadata", {}) or {})

        regime = forecast.get("regime") or metadata.get("regime_state") or None

        now = now or datetime.now(timezone.utc)
        if load_entry_thesis(trade_id, data_root) is not None:
            return load_entry_thesis(trade_id, data_root)
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
            key_drivers=_build_key_drivers(
                forecast=forecast,
                metadata=metadata,
                thesis_text=str(getattr(intent, "thesis", "") or ""),
            ),
        )
        record_entry_thesis(entry_thesis, data_root)
        return entry_thesis
    except Exception:
        # Never break the live order path — thesis recording is purely
        # informational/auxiliary.
        return None


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
            regime=pending.regime,
            invalidation_price=pending.invalidation_price,
            target_price=pending.target_price,
            key_drivers=list(pending.key_drivers),
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
    quotes_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_charts_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
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
        if ev is not None:
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
