"""Gate throughput SLO (weakness ledger S2 / gate-stack collapse).

Profit is expectancy times frequency; a gate stack that admits nothing makes
expectancy irrelevant (the 2026-06-11 entry-freeze catch-22).  This module
turns per-cycle admission funnels into a sealed SLO artifact: per-gate kill
counts, the end-to-end pass rate, and an explicit breach flag when the pass
rate falls below the declared floor.  A breach names the dominant killer
gates so the repair queue targets the right one.  Measurement only: it
never relaxes, bypasses, or reorders any gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

CONTRACT = "QR_GATE_THROUGHPUT_SLO_V1"
DEFAULT_PASS_RATE_FLOOR = 0.10


class GateThroughputError(ValueError):
    """Raised when funnel records are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateThroughputError(f"{label} must be a non-negative integer")
    return value


def build_gate_throughput_slo(
    cycles: Sequence[Mapping[str, Any]],
    *,
    window_label: str,
    pass_rate_floor: float = DEFAULT_PASS_RATE_FLOOR,
) -> dict[str, Any]:
    """Aggregate per-cycle gate funnels into one sealed SLO artifact.

    Each cycle carries ``signals_generated`` and an ordered ``gates`` list of
    ``{gate_id, evaluated, admitted}``.  Within a cycle the funnel must be
    monotone: a gate cannot evaluate more than the previous gate admitted.
    """

    if not isinstance(pass_rate_floor, (int, float)) or isinstance(
        pass_rate_floor, bool
    ):
        raise GateThroughputError("pass_rate_floor must be a number")
    floor = float(pass_rate_floor)
    if not 0.0 < floor < 1.0:
        raise GateThroughputError("pass_rate_floor must be inside (0, 1)")
    if not cycles:
        raise GateThroughputError("at least one funnel cycle is required")

    total_signals = 0
    total_admitted = 0
    kills: dict[str, int] = {}
    evaluated_totals: dict[str, int] = {}
    gate_order: list[str] = []
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, Mapping):
            raise GateThroughputError(f"cycle {index} must be an object")
        upstream = _count(cycle.get("signals_generated"), "signals_generated")
        total_signals += upstream
        gates = cycle.get("gates")
        if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
            raise GateThroughputError(f"cycle {index} gates must be a list")
        for gate in gates:
            gate_id = str(gate.get("gate_id") or "")
            if not gate_id:
                raise GateThroughputError(f"cycle {index} gate_id is required")
            evaluated = _count(gate.get("evaluated"), f"{gate_id}.evaluated")
            admitted = _count(gate.get("admitted"), f"{gate_id}.admitted")
            if admitted > evaluated or evaluated > upstream:
                raise GateThroughputError(
                    f"cycle {index} funnel is not monotone at {gate_id}"
                )
            if gate_id not in kills:
                kills[gate_id] = 0
                evaluated_totals[gate_id] = 0
                gate_order.append(gate_id)
            kills[gate_id] += evaluated - admitted
            evaluated_totals[gate_id] += evaluated
            upstream = admitted
        total_admitted += upstream

    pass_rate = (total_admitted / total_signals) if total_signals else 0.0
    ranking = sorted(
        (
            {
                "gate_id": gate_id,
                "evaluated": evaluated_totals[gate_id],
                "killed": kills[gate_id],
                "kill_fraction_of_signals": round(
                    kills[gate_id] / total_signals, 9
                )
                if total_signals
                else 0.0,
            }
            for gate_id in gate_order
        ),
        key=lambda row: (-row["killed"], row["gate_id"]),
    )
    breached = pass_rate < floor
    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "window_label": str(window_label),
        "cycle_count": len(cycles),
        "signals_generated": total_signals,
        "orders_admitted": total_admitted,
        "end_to_end_pass_rate": round(pass_rate, 9),
        "pass_rate_floor": floor,
        "floor_breached": breached,
        "dominant_killer_gates": [row["gate_id"] for row in ranking[:3]]
        if breached
        else [],
        "gate_ranking": ranking,
        "self_improvement_p0_required": breached,
        "measurement_only": True,
        "relaxes_any_gate": False,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "slo_sha256": _canonical_sha(body)}
