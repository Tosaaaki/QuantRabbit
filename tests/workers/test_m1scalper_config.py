from __future__ import annotations

import importlib


def _reload_config():
    from workers.scalp_m1scalper import config as config_mod

    return importlib.reload(config_mod)


def test_side_filter_normalization(monkeypatch):
    monkeypatch.setenv("M1SCALP_SIDE_FILTER", "buy")
    cfg = _reload_config()
    assert cfg.SIDE_FILTER == "long"

    monkeypatch.setenv("M1SCALP_SIDE_FILTER", "sell")
    cfg = _reload_config()
    assert cfg.SIDE_FILTER == "short"


def test_signal_tag_filter_and_override(monkeypatch):
    monkeypatch.setenv("M1SCALP_SIGNAL_TAG_CONTAINS", "trend-long, nwave")
    monkeypatch.setenv("M1SCALP_STRATEGY_TAG_OVERRIDE", "M1Scalper-trend-long-mirror")
    cfg = _reload_config()
    assert cfg.SIGNAL_TAG_CONTAINS == {"trend-long", "nwave"}
    assert cfg.STRATEGY_TAG_OVERRIDE == "M1Scalper-trend-long-mirror"


def test_confidence_floor_from_env(monkeypatch):
    monkeypatch.setenv("M1SCALP_CONFIDENCE_FLOOR", "62")
    monkeypatch.setenv("M1SCALP_CONFIDENCE_CEIL", "94")
    cfg = _reload_config()
    assert cfg.CONFIDENCE_FLOOR == 62
    assert cfg.CONFIDENCE_CEIL == 94


def test_usdjpy_quickshot_env(monkeypatch):
    monkeypatch.setenv("M1SCALP_USDJPY_QUICKSHOT_ENABLED", "1")
    monkeypatch.setenv("M1SCALP_USDJPY_QUICKSHOT_TARGET_JPY", "120")
    monkeypatch.setenv("M1SCALP_USDJPY_QUICKSHOT_BLOCK_JST_HOURS", "7,8")
    cfg = _reload_config()
    assert cfg.USDJPY_QUICKSHOT_ENABLED is True
    assert cfg.USDJPY_QUICKSHOT_TARGET_JPY == 120.0
    assert cfg.USDJPY_QUICKSHOT_BLOCK_JST_HOURS == {7, 8}
