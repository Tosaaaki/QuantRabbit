from __future__ import annotations

import pytest

from quant_rabbit.gate_throughput_slo import (
    GateThroughputError,
    _canonical_sha,
    build_gate_throughput_slo,
)


def _cycle(signals: int, *gates: tuple[str, int, int]) -> dict:
    return {
        "signals_generated": signals,
        "gates": [
            {"gate_id": gate_id, "evaluated": evaluated, "admitted": admitted}
            for gate_id, evaluated, admitted in gates
        ],
    }


def test_breach_names_dominant_killers_and_seals() -> None:
    cycles = [
        _cycle(
            100,
            ("FRESHNESS", 100, 90),
            ("SPREAD_COST", 90, 40),
            ("RECEIPT", 40, 2),
        ),
        _cycle(
            50,
            ("FRESHNESS", 50, 48),
            ("SPREAD_COST", 48, 30),
            ("RECEIPT", 30, 1),
        ),
    ]

    slo = build_gate_throughput_slo(cycles, window_label="7d")

    assert slo["signals_generated"] == 150
    assert slo["orders_admitted"] == 3
    assert slo["floor_breached"] is True
    assert slo["self_improvement_p0_required"] is True
    # RECEIPT killed 67, SPREAD_COST killed 68 -> SPREAD_COST ranks first.
    assert slo["dominant_killer_gates"][0] == "SPREAD_COST"
    assert slo["dominant_killer_gates"][1] == "RECEIPT"
    body = {k: v for k, v in slo.items() if k != "slo_sha256"}
    assert slo["slo_sha256"] == _canonical_sha(body)
    assert slo["relaxes_any_gate"] is False


def test_healthy_window_does_not_flag() -> None:
    slo = build_gate_throughput_slo(
        [_cycle(100, ("FRESHNESS", 100, 95), ("SPREAD_COST", 95, 40))],
        window_label="7d",
    )
    assert slo["floor_breached"] is False
    assert slo["dominant_killer_gates"] == []


def test_non_monotone_funnel_is_refused() -> None:
    with pytest.raises(GateThroughputError, match="not monotone"):
        build_gate_throughput_slo(
            [_cycle(10, ("A", 10, 5), ("B", 7, 3))], window_label="7d"
        )
    with pytest.raises(GateThroughputError, match="non-negative"):
        build_gate_throughput_slo(
            [_cycle(10, ("A", 10, -1))], window_label="7d"
        )
    with pytest.raises(GateThroughputError, match="at least one"):
        build_gate_throughput_slo([], window_label="7d")
