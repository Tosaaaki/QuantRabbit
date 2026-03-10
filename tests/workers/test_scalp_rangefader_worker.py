from __future__ import annotations

import os

import pytest

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

from workers.scalp_rangefader import worker as rf_worker


def test_entry_cooldown_is_scoped_by_signal_tag_and_side(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 24.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 16.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    last_entry_ts_by_key = {
        rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"): 100.0,
    }

    assert rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-buy-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"),
        now_mono=112.0,
    )
    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-sell-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-sell-fade", "short"),
        now_mono=112.0,
    )
    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-neutral-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-neutral-fade", "short"),
        now_mono=112.0,
    )


def test_entry_cooldown_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)

    assert not rf_worker._entry_cooldown_active(
        {
            rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"): 100.0,
        },
        signal_tag="RangeFader-buy-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"),
        now_mono=101.0,
    )


def test_entry_cooldown_uses_shorter_buy_fade_window(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 20.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 14.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", False, raising=False)
    last_entry_ts_by_key = {
        rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"): 100.0,
        rf_worker._entry_cooldown_key("RangeFader-buy-supportive", "long"): 100.0,
        rf_worker._entry_cooldown_key("RangeFader-sell-fade", "short"): 100.0,
    }

    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-buy-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"),
        now_mono=115.0,
    )
    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-buy-supportive",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-supportive", "long"),
        now_mono=115.0,
    )
    assert rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        signal_tag="RangeFader-sell-fade",
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-sell-fade", "short"),
        now_mono=115.0,
    )


def test_entry_cooldown_extends_with_fresh_trim_participation_profile(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 20.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 14.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(
        rf_worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "trim_units",
            "cadence_floor": 0.90,
        },
        raising=False,
    )

    assert rf_worker._entry_cooldown_sec("RangeFader-sell-fade") == pytest.approx(22.2222, rel=1e-3)
    assert rf_worker._entry_cooldown_sec("RangeFader-buy-fade") == pytest.approx(15.5556, rel=1e-3)


def test_entry_cooldown_shortens_with_fresh_boost_participation_profile(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 20.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 14.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(
        rf_worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "boost_participation",
            "cadence_floor": 1.10,
        },
        raising=False,
    )

    assert rf_worker._entry_cooldown_sec("RangeFader-sell-fade") == pytest.approx(20.0 / 1.10, rel=1e-3)
    assert rf_worker._entry_cooldown_sec("RangeFader-buy-fade") == pytest.approx(14.0 / 1.10, rel=1e-3)


def test_entry_cooldown_ignores_stale_or_non_trim_participation_profile(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 20.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 14.0)
    monkeypatch.setattr(rf_worker, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)

    monkeypatch.setattr(
        rf_worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": True,
            "protect_frequency": True,
            "action": "trim_units",
            "cadence_floor": 0.90,
        },
        raising=False,
    )
    assert rf_worker._entry_cooldown_sec("RangeFader-sell-fade") == 20.0

    monkeypatch.setattr(
        rf_worker,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "payload_stale": False,
            "protect_frequency": True,
            "action": "hold",
            "cadence_floor": 0.90,
        },
        raising=False,
    )
    assert rf_worker._entry_cooldown_sec("RangeFader-sell-fade") == 20.0
