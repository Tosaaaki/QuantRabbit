import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name("run_family_fusion_v5.py")
SPEC = importlib.util.spec_from_file_location("fusion", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_no_trade_without_two_non_dissenting_families(monkeypatch):
    votes = iter(["LONG", "SHORT", None])
    monkeypatch.setattr(MOD, "family_vote", lambda *args: next(votes))
    assert MOD.fused_side(60, {}, ["a", "b", "c"])[0] is None

def test_two_agree_no_dissent_trades(monkeypatch):
    votes = iter(["LONG", "LONG", None])
    monkeypatch.setattr(MOD, "family_vote", lambda *args: next(votes))
    assert MOD.fused_side(60, {}, ["a", "b", "c"]) == ("LONG", 2)
