from __future__ import annotations

from workers.scalp_rangefader import worker as rf_worker


def test_entry_cooldown_is_scoped_by_signal_tag_and_side(monkeypatch) -> None:
    monkeypatch.setattr(rf_worker.config, "COOLDOWN_SEC", 24.0)
    monkeypatch.setattr(rf_worker.config, "BUY_COOLDOWN_SEC", 16.0)
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
