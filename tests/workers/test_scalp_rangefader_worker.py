from __future__ import annotations

from workers.scalp_rangefader import worker as rf_worker


def test_entry_cooldown_is_scoped_by_signal_tag_and_side(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 24.0)
    last_entry_ts_by_key = {
        rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"): 100.0,
    }

    assert rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"),
        now_mono=112.0,
    )
    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-sell-fade", "short"),
        now_mono=112.0,
    )
    assert not rf_worker._entry_cooldown_active(
        last_entry_ts_by_key,
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-neutral-fade", "short"),
        now_mono=112.0,
    )


def test_entry_cooldown_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 0.0)

    assert not rf_worker._entry_cooldown_active(
        {
            rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"): 100.0,
        },
        cooldown_key=rf_worker._entry_cooldown_key("RangeFader-buy-fade", "long"),
        now_mono=101.0,
    )
