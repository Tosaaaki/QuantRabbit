from __future__ import annotations

import os


# Keep unit tests fast/offline even if optional GCP Secret Manager is configured elsewhere.
os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def test_tick_wick_reversal_is_range_style():
    from workers.common.air_state import _strategy_style

    assert _strategy_style("TickWickReversal") == "range"

