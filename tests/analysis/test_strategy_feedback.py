from __future__ import annotations

import json

from analysis import strategy_feedback


def _reset_feedback_caches(monkeypatch, *, feedback_path, counterfactual_path) -> None:
    monkeypatch.setattr(strategy_feedback, "_FEEDBACK_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_feedback, "_PATH", feedback_path, raising=False)
    monkeypatch.setattr(strategy_feedback, "_COUNTERFACTUAL_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_feedback, "_COUNTERFACTUAL_PATH", counterfactual_path, raising=False)
    monkeypatch.setattr(strategy_feedback, "_FEEDBACK_MAX_AGE_SEC", 864000.0, raising=False)
    monkeypatch.setattr(strategy_feedback, "_COUNTERFACTUAL_MAX_AGE_SEC", 864000.0, raising=False)
    monkeypatch.setattr(strategy_feedback, "_PARAM_SOURCE_PATHS", [], raising=False)
    monkeypatch.setattr(
        strategy_feedback,
        "_CACHE",
        {"loaded": 0.0, "mtime": None, "payload": None},
        raising=False,
    )
    monkeypatch.setattr(
        strategy_feedback,
        "_COUNTERFACTUAL_CACHE",
        {"loaded": 0.0, "mtime": None, "payload": None},
        raising=False,
    )
    monkeypatch.setattr(
        strategy_feedback,
        "_PARAM_CACHE",
        {"loaded": 0.0, "paths": None, "payloads": []},
        raising=False,
    )


def test_current_advice_applies_counterfactual_overlay_for_side(monkeypatch, tmp_path) -> None:
    feedback_path = tmp_path / "strategy_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-03-07T00:00:00Z",
                "strategies": {
                    "scalp_ping_5s_b_live": {
                        "entry_units_multiplier": 0.9,
                        "entry_probability_multiplier": 0.95,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    counterfactual_path = tmp_path / "trade_counterfactual_latest.json"
    counterfactual_path.write_text(
        json.dumps(
            {
                "strategy_like": "scalp_ping_5s_b_live%",
                "policy_hints": {
                    "side_actions": {"short": "block"},
                    "reentry_overrides": {
                        "mode": "tighten",
                        "confidence": 0.9,
                        "cooldown_loss_mult": 1.35,
                        "cooldown_win_mult": 1.10,
                        "same_dir_reentry_pips_mult": 1.20,
                        "lcb_uplift_pips": 1.4,
                        "return_wait_bias": "avoid",
                    },
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_feedback_caches(monkeypatch, feedback_path=feedback_path, counterfactual_path=counterfactual_path)

    advice = strategy_feedback.current_advice("scalp_ping_5s_b_live-labc12345", side="sell")

    assert advice is not None
    assert advice["entry_units_multiplier"] < 0.9
    assert advice["entry_probability_multiplier"] < 0.95
    assert advice["entry_probability_delta"] < 0.0
    assert advice["sl_distance_multiplier"] < 1.0
    assert advice["tp_distance_multiplier"] < 1.0
    assert advice["_meta"]["counterfactual"]["side_action"] == "block"
    assert advice["_meta"]["counterfactual"]["return_wait_bias"] == "avoid"
    assert advice["strategy_params"]["counterfactual_feedback"]["mode"] == "tighten"


def test_current_advice_returns_counterfactual_only_when_feedback_missing(monkeypatch, tmp_path) -> None:
    counterfactual_path = tmp_path / "trade_counterfactual_latest.json"
    counterfactual_path.write_text(
        json.dumps(
            {
                "strategy_like": "m1scalper_m1%",
                "policy_hints": {
                    "reentry_overrides": {
                        "mode": "loosen",
                        "confidence": 0.8,
                        "cooldown_loss_mult": 0.85,
                        "cooldown_win_mult": 0.90,
                        "same_dir_reentry_pips_mult": 0.90,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_feedback_caches(monkeypatch, feedback_path=None, counterfactual_path=counterfactual_path)

    advice = strategy_feedback.current_advice("M1Scalper-M1", side="long")

    assert advice is not None
    assert advice["entry_units_multiplier"] > 1.0
    assert advice["entry_probability_multiplier"] > 1.0
    assert advice["sl_distance_multiplier"] > 1.0
    assert advice["tp_distance_multiplier"] > 1.0
    assert advice["_meta"]["counterfactual"]["mode"] == "loosen"


def test_current_advice_falls_back_to_base_strategy_feedback_for_directional_tag(
    monkeypatch,
    tmp_path,
) -> None:
    feedback_path = tmp_path / "strategy_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-03-10T00:00:00Z",
                "strategies": {
                    "MicroTrendRetest": {
                        "entry_units_multiplier": 0.84,
                        "entry_probability_multiplier": 0.91,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_feedback_caches(
        monkeypatch,
        feedback_path=feedback_path,
        counterfactual_path=tmp_path / "missing_counterfactual.json",
    )

    advice = strategy_feedback.current_advice("MicroTrendRetest-long", side="long")

    assert advice is not None
    assert advice["entry_units_multiplier"] == 0.84
    assert advice["entry_probability_multiplier"] == 0.91
    assert advice["_meta"]["strategy_tag"] == "MicroTrendRetest-long"


def test_current_advice_ignores_stale_feedback_payload(monkeypatch, tmp_path) -> None:
    feedback_path = tmp_path / "strategy_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-03-01T00:00:00Z",
                "strategies": {
                    "MicroTrendRetest": {
                        "entry_units_multiplier": 0.5,
                        "entry_probability_multiplier": 0.5,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    counterfactual_path = tmp_path / "trade_counterfactual_latest.json"
    counterfactual_path.write_text(
        json.dumps(
            {
                "strategy_like": "microtrendretest%",
                "policy_hints": {
                    "reentry_overrides": {
                        "mode": "loosen",
                        "confidence": 0.9,
                        "cooldown_loss_mult": 0.85,
                        "cooldown_win_mult": 0.90,
                        "same_dir_reentry_pips_mult": 0.90,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_feedback_caches(monkeypatch, feedback_path=feedback_path, counterfactual_path=counterfactual_path)
    monkeypatch.setattr(strategy_feedback, "_FEEDBACK_MAX_AGE_SEC", 60.0, raising=False)
    monkeypatch.setattr(strategy_feedback, "_COUNTERFACTUAL_MAX_AGE_SEC", 1800.0, raising=False)

    advice = strategy_feedback.current_advice("MicroTrendRetest-long", side="long")

    assert advice is not None
    assert advice["entry_units_multiplier"] > 1.0
    assert advice["entry_probability_multiplier"] > 1.0
    assert advice["_meta"]["payload_stale"] is True
    assert advice["_meta"]["payload_age_sec"] is not None
