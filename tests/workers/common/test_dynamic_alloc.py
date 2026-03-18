from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from workers.common.dynamic_alloc import load_strategy_profile


def test_load_strategy_profile_applies_soft_participation_policy(
    tmp_path: Path,
) -> None:
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
    assert profile["lot_multiplier"] == 0.30


def test_load_strategy_profile_defaults_keep_block_behavior_without_policy(
    tmp_path: Path,
) -> None:
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


def test_load_strategy_profile_falls_back_to_case_insensitive_key(
    tmp_path: Path,
) -> None:
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


def test_load_strategy_profile_prefers_setup_override(tmp_path: Path) -> None:
    payload = {
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "score": 0.31,
                "lot_multiplier": 0.82,
                "trades": 61,
                "setup_overrides": [
                    {
                        "match_dimension": "setup_fingerprint",
                        "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
                        "flow_regime": "trend_long",
                        "microstructure_bucket": "tight_fast",
                        "score": 0.09,
                        "lot_multiplier": 0.58,
                        "effective_min_lot_multiplier": 0.18,
                        "trades": 24,
                        "pf": 0.42,
                        "win_rate": 0.29,
                    }
                ],
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
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
    assert profile["lot_multiplier"] == 0.58
    assert profile["effective_min_lot_multiplier"] == 0.18
    assert profile["setup_override"]["match_dimension"] == "setup_fingerprint"


def test_load_strategy_profile_derives_setup_override_from_technical_context(
    tmp_path: Path,
) -> None:
    payload = {
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "score": 0.31,
                "lot_multiplier": 0.82,
                "trades": 61,
                "setup_overrides": [
                    {
                        "match_dimension": "flow_micro",
                        "flow_regime": "trend_long",
                        "microstructure_bucket": "tight_fast",
                        "score": 0.14,
                        "lot_multiplier": 0.64,
                        "effective_min_lot_multiplier": 0.20,
                        "trades": 18,
                        "pf": 0.51,
                        "win_rate": 0.33,
                    }
                ],
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
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
    assert profile["lot_multiplier"] == 0.64
    assert profile["setup_override"]["match_dimension"] == "flow_micro"


def test_load_strategy_profile_keeps_strategy_level_trim_without_matching_setup_override(
    tmp_path: Path,
) -> None:
    payload = {
        "strategies": {
            "VwapRevertS": {
                "pocket": "scalp",
                "score": 0.27,
                "lot_multiplier": 0.14,
                "trades": 18,
            }
        },
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile(
        "VwapRevertS",
        "scalp",
        path=path,
        entry_thesis={
            "setup_fingerprint": "VwapRevertS|short|range_fade|tight_fast|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
            "live_setup_context": {
                "flow_regime": "range_fade",
                "microstructure_bucket": "tight_fast",
            },
        },
    )

    assert profile["found"] is True
    assert profile["lot_multiplier"] == 0.14
    assert profile["setup_trim_fallback"] == "strategy_level_trim"
    assert profile["setup_identity"]["flow_regime"] == "range_fade"


def test_load_strategy_profile_returns_policy_default_when_unknown(
    tmp_path: Path,
) -> None:
    payload = {
        "allocation_policy": {
            "soft_participation": True,
            "allow_loser_block": False,
            "allow_winner_only": False,
            "min_lot_multiplier": 0.25,
            "max_lot_multiplier": 1.65,
        },
        "strategies": {},
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_strategy_profile("scalp_ping_5s_flow_live", "scalp_fast", path=path)
    assert profile["found"] is True
    assert profile["soft_participation"] is True
    assert profile["allow_loser_block"] is False
    assert profile["allow_winner_only"] is False
    assert profile["lot_multiplier"] == 0.25


def test_load_strategy_profile_skips_unknown_soft_fallback_when_payload_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {
        "as_of": (datetime.now(timezone.utc) - timedelta(minutes=45))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "allocation_policy": {
            "soft_participation": True,
            "allow_loser_block": False,
            "allow_winner_only": False,
            "min_lot_multiplier": 0.45,
            "max_lot_multiplier": 1.65,
        },
        "strategies": {},
    }
    path = tmp_path / "dynamic_alloc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("WORKER_DYNAMIC_ALLOC_UNKNOWN_FALLBACK_MAX_AGE_SEC", "600")

    profile = load_strategy_profile("RangeFader-buy-fade", "scalp", path=path)
    assert profile["found"] is False
    assert profile["soft_participation"] is True
    assert profile["payload_stale"] is True
    assert profile["lot_multiplier"] == 1.0
    assert profile["soft_participation_skip_reason"] == "stale_unknown_strategy"
