from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("run_mtf_long_window_v4.py")
SPEC = importlib.util.spec_from_file_location("mtf_long", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_h1_requires_all_twelve_m5():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"time": t + timedelta(minutes=5*i), "mid": {"o": 1, "h": 2, "l": .5, "c": 1.5}} for i in range(12)]
    assert len(MOD.completed_h1(rows)) == 1
    assert MOD.completed_h1(rows[:-1]) == []


def test_parent_opinion_uses_only_completed_hour():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for hour, close in enumerate((1.0, 2.0, 3.5)):
        for i in range(12):
            c = close if i == 11 else close - .1
            rows.append({"time": t + timedelta(hours=hour, minutes=5*i), "mid": {"o": close-.5, "h": close, "l": close-1, "c": c}})
    opinions = MOD.opinions_by_signal(rows, MOD.completed_h1(rows))
    assert (t + timedelta(hours=2, minutes=50)) not in opinions
    extra = {"time": t + timedelta(hours=3), "mid": {"o": 3.5, "h": 3.5, "l": 3.5, "c": 3.5}}
    opinions = MOD.opinions_by_signal(rows + [extra], MOD.completed_h1(rows))
    assert opinions[t + timedelta(hours=3)] == "LONG"
