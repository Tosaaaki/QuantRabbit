from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s import worker as scalp_worker


def test_short_probe_rescue_skips_when_risk_cap_below_min(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "MIN_UNITS", 2000)
    monkeypatch.setattr(scalp_worker.config, "SHORT_PROBE_RESCUE_ENABLED", True)

    units, status = scalp_worker._maybe_rescue_short_probe(
        units=500,
        units_risk=1200,
        side="short",
    )

    assert units == 500
    assert status == "risk_cap_below_min"


def test_short_probe_rescue_keeps_short_probe_for_viable_risk_cap(monkeypatch) -> None:
    monkeypatch.setattr(scalp_worker.config, "MIN_UNITS", 2000)
    monkeypatch.setattr(scalp_worker.config, "SHORT_PROBE_RESCUE_ENABLED", True)

    units, status = scalp_worker._maybe_rescue_short_probe(
        units=500,
        units_risk=2500,
        side="short",
    )

    assert units == 2000
    assert status == "short_probe_rescued"
