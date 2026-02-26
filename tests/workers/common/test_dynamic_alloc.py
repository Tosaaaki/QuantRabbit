from __future__ import annotations

import json
from pathlib import Path

from workers.common.dynamic_alloc import load_strategy_profile


def test_load_strategy_profile_applies_soft_participation_policy(tmp_path: Path) -> None:
    payload = {
        "allocation_policy": {
            "soft_participation": True,
            "allow_loser_block": False,
            "allow_winner_only": False,
            "min_lot_multiplier": 0.55,
            "max_lot_multiplier": 1.35,
        },
        "strategies": {
            "M1Scalper-M1": {
                "pocket": "scalp",
                "score": 0.21,
                "lot_multiplier": 0.30,
                "trades": 24,
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile("M1Scalper-M1", "scalp", path=path)
    assert profile["found"] is True
    assert profile["soft_participation"] is True
    assert profile["allow_loser_block"] is False
    assert profile["allow_winner_only"] is False
    assert profile["lot_multiplier"] == 0.55


def test_load_strategy_profile_defaults_keep_block_behavior_without_policy(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "scalp_ping_5s_c_live": {
                "pocket": "scalp_fast",
                "score": 0.65,
                "lot_multiplier": 1.18,
                "trades": 64,
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile("scalp_ping_5s_c_live", "scalp_fast", path=path)
    assert profile["found"] is True
    assert profile["soft_participation"] is False
    assert profile["allow_loser_block"] is True
    assert profile["allow_winner_only"] is True


def test_load_strategy_profile_falls_back_to_case_insensitive_key(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "m1scalper-m1": {
                "pocket": "scalp",
                "score": 0.41,
                "lot_multiplier": 0.74,
                "trades": 48,
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile("M1Scalper-M1", "scalp", path=path)
    assert profile["found"] is True
    assert profile["strategy_key"] == "m1scalper-m1"
    assert profile["lot_multiplier"] == 0.74
