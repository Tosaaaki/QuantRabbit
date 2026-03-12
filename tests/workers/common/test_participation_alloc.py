from __future__ import annotations

import json
from pathlib import Path

from workers.common.participation_alloc import load_strategy_profile


def test_load_strategy_profile_prefers_setup_override(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "lot_multiplier": 0.92,
                "probability_offset": -0.01,
                "setup_overrides": [
                    {
                        "match_dimension": "setup_fingerprint",
                        "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
                        "flow_regime": "trend_long",
                        "microstructure_bucket": "tight_fast",
                        "action": "trim_units",
                        "lot_multiplier": 0.81,
                        "probability_offset": -0.03,
                        "attempts": 44,
                    }
                ],
            }
        }
    }
    path = tmp_path / "participation_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile(
        "RangeFader-sell-fade",
        "scalp",
        path=path,
        entry_thesis={
            "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
            "live_setup_context": {
                "flow_regime": "trend_long",
                "microstructure_bucket": "tight_fast",
            },
        },
    )

    assert profile["found"] is True
    assert profile["lot_multiplier"] == 0.81
    assert profile["probability_offset"] == -0.03
    assert profile["setup_override"]["match_dimension"] == "setup_fingerprint"


def test_load_strategy_profile_derives_setup_override_from_technical_context(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "lot_multiplier": 0.92,
                "probability_offset": -0.01,
                "setup_overrides": [
                    {
                        "match_dimension": "flow_micro",
                        "flow_regime": "trend_long",
                        "microstructure_bucket": "tight_fast",
                        "action": "trim_units",
                        "lot_multiplier": 0.84,
                        "probability_offset": -0.02,
                        "attempts": 31,
                    }
                ],
            }
        }
    }
    path = tmp_path / "participation_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile(
        "RangeFader-sell-fade",
        "scalp",
        path=path,
        entry_thesis={
            "strategy_tag": "RangeFader-sell-fade",
            "side": "short",
            "range_mode": "trend",
            "range_score": 0.18,
            "spread_pips": 0.8,
            "technical_context": {
                "ticks": {"spread_pips": 0.8, "tick_rate": 9.2},
                "indicators": {
                    "M1": {
                        "atr_pips": 2.4,
                        "rsi": 67.0,
                        "adx": 29.0,
                        "plus_di": 31.0,
                        "minus_di": 14.0,
                        "ma10": 158.110,
                        "ma20": 158.080,
                    }
                },
            },
        },
    )

    assert profile["found"] is True
    assert profile["lot_multiplier"] == 0.84
    assert profile["probability_offset"] == -0.02
    assert profile["setup_override"]["match_dimension"] == "flow_micro"


def test_load_strategy_profile_preserves_setup_override_runtime_caps(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "PrecisionLowVol": {
                "pocket": "scalp",
                "lot_multiplier": 1.0,
                "setup_overrides": [
                    {
                        "match_dimension": "setup_fingerprint",
                        "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
                        "action": "boost_participation",
                        "lot_multiplier": 1.12,
                        "units_multiplier": 1.12,
                        "probability_boost": 0.05,
                        "max_probability_cut": 0.08,
                        "max_units_boost": 0.12,
                        "max_probability_boost": 0.05,
                        "attempts": 15,
                    }
                ],
            }
        }
    }
    path = tmp_path / "participation_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile(
        "PrecisionLowVol",
        "scalp",
        path=path,
        entry_thesis={
            "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
            "live_setup_context": {
                "flow_regime": "range_fade",
                "microstructure_bucket": "unknown",
            },
        },
    )

    assert profile["found"] is True
    assert profile["action"] == "boost_participation"
    assert profile["lot_multiplier"] == 1.12
    assert profile["max_units_boost"] == 0.12
    assert profile["max_probability_cut"] == 0.08
    assert profile["max_probability_boost"] == 0.05
