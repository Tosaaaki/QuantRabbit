import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.exit_manager import ExitManager


def test_exit_manager_is_noop():
    manager = ExitManager()
    decisions = manager.plan_closures({}, [], {}, {}, event_soon=False, range_mode=False)
    assert decisions == []
