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

Output is INFORMATIONAL (written to `data/forecast_persistence_report.json`).
The trader / GPT trader uses it as evidence. Auto-close still goes
through gpt_trader Gate A/B; this module provides the structural
evidence the Gate A requires.

Kill switch: `QR_DISABLE_FORECAST_PERSISTENCE=1`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


HISTORY_FILENAME = "forecast_history.jsonl"
FLIP_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_FLIP_PERSISTENCE", "3"))
RANGE_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_RANGE_PERSISTENCE", "5"))
HISTORY_LOOKBACK_CYCLES = int(os.environ.get("QR_FORECAST_HISTORY_LOOKBACK", "10"))


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
) -> None:
    """Append a `DirectionalForecast` to forecast_history.jsonl."""
    if _is_disabled():
        return
    path = data_root / HISTORY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    entry = {
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "pair": getattr(forecast, "pair", ""),
        "direction": getattr(forecast, "direction", "UNCLEAR"),
        "confidence": float(getattr(forecast, "confidence", 0)),
        "invalidation_price": getattr(forecast, "invalidation_price", None),
        "target_price": getattr(forecast, "target_price", None),
        "horizon_min": int(getattr(forecast, "horizon_min", 0)),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_recent(data_root: Path, pair: str, count: int) -> List[Dict[str, Any]]:
    """Return the last `count` forecast entries for `pair`."""
    path = data_root / HISTORY_FILENAME
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
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
                out.append(d)
    except OSError:
        return []
    return out[-count:]


def assess_position(
    *,
    trade_id: str,
    pair: str,
    side: str,
    data_root: Path,
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

    directions = tuple(str(r.get("direction", "")) for r in recent)
    confidences = tuple(float(r.get("confidence", 0)) for r in recent)

    # Persistent flip check
    flip_run = 0
    for d in reversed(directions):
        if d == opposite_dir:
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


def assess_all_positions(
    positions: List[Any],
    *,
    data_root: Path,
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
        ))
    return out
