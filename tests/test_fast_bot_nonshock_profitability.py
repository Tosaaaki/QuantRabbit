from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_fast_bot_nonshock_profitability.py"
SPEC = importlib.util.spec_from_file_location("nonshock_profitability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def base() -> dict:
    return {"return_60_pips": 12.0, "return_240_pips": 20.0, "efficiency_60": 0.3}


def test_trend_family_is_mirror_symmetric() -> None:
    assert MODULE._family(base(), "H1_H4_TREND") == 1
    mirrored = {key: -value if key.startswith("return") else value for key, value in base().items()}
    assert MODULE._family(mirrored, "H1_H4_TREND") == -1


def test_pullback_resume_uses_h4_side() -> None:
    row = {"return_60_pips": -7.0, "return_240_pips": 15.0, "efficiency_60": 0.2}
    assert MODULE._family(row, "H4_PULLBACK_RESUME") == 1
    assert MODULE._family({**row, "return_60_pips": 7.0}, "H4_PULLBACK_RESUME") is None


def test_range_fade_requires_bounded_h4() -> None:
    row = {"return_60_pips": 12.0, "return_240_pips": 5.0, "efficiency_60": 0.2}
    assert MODULE._family(row, "H1_EXTREME_RANGE_FADE") == -1
    assert MODULE._family({**row, "return_240_pips": 20.0}, "H1_EXTREME_RANGE_FADE") is None
