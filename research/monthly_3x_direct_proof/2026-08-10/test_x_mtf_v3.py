from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("run_x_mtf_v3.py")
SPEC = importlib.util.spec_from_file_location("x_mtf", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _bar(start, o, h, l, c):
    return {"start": start, "end": start + timedelta(hours=1), "o": o, "h": h, "l": l, "c": c}


def test_parent_long_and_short_are_causal():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    long = [_bar(t, 1, 2, .5, 1), _bar(t + timedelta(hours=1), 1, 3, 1, 2), _bar(t + timedelta(hours=2), 2, 4, 2, 3.5)]
    short = [_bar(t, 4, 4.5, 3, 4), _bar(t + timedelta(hours=1), 4, 4, 2, 3), _bar(t + timedelta(hours=2), 3, 3, 1, 1.5)]
    assert MOD.parent_opinion(long, t + timedelta(hours=3)) == "LONG"
    assert MOD.parent_opinion(short, t + timedelta(hours=3)) == "SHORT"
    assert MOD.parent_opinion(long, t + timedelta(hours=2, minutes=30)) is None


def test_gap_breaks_parent_chain():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [_bar(t, 1, 2, .5, 1), _bar(t + timedelta(hours=2), 1, 3, 1, 2), _bar(t + timedelta(hours=3), 2, 4, 2, 3.5)]
    assert MOD.parent_opinion(bars, t + timedelta(hours=4)) is None
