import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid slow/flake-prone network calls to Secret Manager during imports.
os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

from workers.common import air_state


def test_strategy_style_wickreversalpro_is_range() -> None:
    assert air_state._strategy_style("WickReversalPro") == "range"


def test_strategy_style_wickreversal_is_range() -> None:
    assert air_state._strategy_style("WickReversal") == "range"
