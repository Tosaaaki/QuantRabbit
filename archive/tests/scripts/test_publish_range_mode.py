from __future__ import annotations

from datetime import datetime, timedelta

from scripts import publish_range_mode


def test_macro_max_age_uses_tf_based_default_when_unset(monkeypatch) -> None:
    monkeypatch.setattr(publish_range_mode, "MACRO_TF", "H4")
    monkeypatch.setattr(publish_range_mode, "DEFAULT_MAX_DATA_AGE_SEC", 900)
    monkeypatch.delenv("RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC", raising=False)

    # H4 default = max(M1 limit, 4h * 1.35) = 19440 sec.
    assert publish_range_mode._macro_max_data_age_sec() == 19440


def test_freshness_state_allows_macro_intra_bar_age(monkeypatch) -> None:
    monkeypatch.setattr(publish_range_mode, "MACRO_TF", "H4")
    monkeypatch.setattr(publish_range_mode, "DEFAULT_MAX_DATA_AGE_SEC", 900)
    monkeypatch.delenv("RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC", raising=False)

    now = datetime(2026, 2, 26, 7, 0, 0)
    ts_m1 = now - timedelta(seconds=120)
    ts_macro = now - timedelta(seconds=12392)
    stale, m1_age, macro_age, macro_limit = publish_range_mode._freshness_state(
        now,
        ts_m1,
        ts_macro,
    )

    assert stale is False
    assert int(m1_age or 0) == 120
    assert int(macro_age or 0) == 12392
    assert macro_limit == 19440


def test_freshness_state_flags_macro_beyond_limit(monkeypatch) -> None:
    monkeypatch.setattr(publish_range_mode, "MACRO_TF", "H4")
    monkeypatch.setattr(publish_range_mode, "DEFAULT_MAX_DATA_AGE_SEC", 900)
    monkeypatch.setenv("RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC", "10000")

    now = datetime(2026, 2, 26, 7, 0, 0)
    ts_m1 = now - timedelta(seconds=60)
    ts_macro = now - timedelta(seconds=12000)
    stale, _m1_age, _macro_age, macro_limit = publish_range_mode._freshness_state(
        now,
        ts_m1,
        ts_macro,
    )

    assert stale is True
    assert macro_limit == 10000
