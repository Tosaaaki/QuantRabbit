from __future__ import annotations

import os

import pytest

from workers.common import reentry_decider as rd


def _clear_prefix_env(monkeypatch, prefix: str) -> None:
    pref = prefix.strip().upper()
    for key in list(os.environ):
        if key.startswith(f"{pref}_"):
            monkeypatch.delenv(key, raising=False)


def test_legacy_threshold_envs_remain_constant_across_atr(monkeypatch) -> None:
    _clear_prefix_env(monkeypatch, "TEST")
    monkeypatch.setenv("TEST_REENTRY_ENABLE", "1")
    monkeypatch.setenv("TEST_REENTRY_REVERT_MIN", "0.72")
    monkeypatch.setenv("TEST_REENTRY_TREND_MIN", "0.61")
    monkeypatch.setenv("TEST_REENTRY_TREND_MAX", "0.43")
    monkeypatch.setenv("TEST_REENTRY_EDGE_MIN", "0.57")

    cfg = rd._load_config("TEST")
    assert cfg.revert_min == pytest.approx(0.72)
    assert cfg.revert_min_low_atr == pytest.approx(0.72)
    assert cfg.revert_min_high_atr == pytest.approx(0.72)
    assert cfg.trend_min_low_atr == pytest.approx(0.61)
    assert cfg.trend_min_high_atr == pytest.approx(0.61)
    assert cfg.trend_max_low_atr == pytest.approx(0.43)
    assert cfg.trend_max_high_atr == pytest.approx(0.43)
    assert cfg.edge_min_low_atr == pytest.approx(0.57)
    assert cfg.edge_min_high_atr == pytest.approx(0.57)

    low = rd._blend_by_atr(
        atr_pips=4.0,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    )
    high = rd._blend_by_atr(
        atr_pips=20.0,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    )
    assert low == pytest.approx(0.72)
    assert high == pytest.approx(0.72)


def test_global_reentry_profile_is_used_when_prefix_specific_is_missing(
    monkeypatch,
) -> None:
    _clear_prefix_env(monkeypatch, "TEST")
    monkeypatch.setenv("TEST_REENTRY_ENABLE", "1")
    monkeypatch.setenv("REENTRY_ATR_LOW_PIPS", "5.0")
    monkeypatch.setenv("REENTRY_ATR_HIGH_PIPS", "11.0")
    monkeypatch.setenv("REENTRY_REVERT_MIN_ATR_LOW", "0.58")
    monkeypatch.setenv("REENTRY_REVERT_MIN_ATR_HIGH", "0.72")
    monkeypatch.setenv("REENTRY_TREND_MIN_ATR_LOW", "0.63")
    monkeypatch.setenv("REENTRY_TREND_MIN_ATR_HIGH", "0.49")
    monkeypatch.setenv("REENTRY_TREND_MAX_ATR_LOW", "0.52")
    monkeypatch.setenv("REENTRY_TREND_MAX_ATR_HIGH", "0.35")
    monkeypatch.setenv("REENTRY_EDGE_MIN_ATR_LOW", "0.61")
    monkeypatch.setenv("REENTRY_EDGE_MIN_ATR_HIGH", "0.44")

    cfg = rd._load_config("TEST")
    assert cfg.atr_low_pips == pytest.approx(5.0)
    assert cfg.atr_high_pips == pytest.approx(11.0)
    assert cfg.revert_min_low_atr == pytest.approx(0.58)
    assert cfg.revert_min_high_atr == pytest.approx(0.72)
    assert cfg.trend_min_low_atr == pytest.approx(0.63)
    assert cfg.trend_min_high_atr == pytest.approx(0.49)
    assert cfg.trend_max_low_atr == pytest.approx(0.52)
    assert cfg.trend_max_high_atr == pytest.approx(0.35)
    assert cfg.edge_min_low_atr == pytest.approx(0.61)
    assert cfg.edge_min_high_atr == pytest.approx(0.44)


def test_thresholds_blend_continuously_when_atr_profile_is_set(monkeypatch) -> None:
    _clear_prefix_env(monkeypatch, "TEST")
    monkeypatch.setenv("TEST_REENTRY_ENABLE", "1")
    monkeypatch.setenv("TEST_REENTRY_ATR_LOW_PIPS", "6")
    monkeypatch.setenv("TEST_REENTRY_ATR_HIGH_PIPS", "14")
    monkeypatch.setenv("TEST_REENTRY_REVERT_MIN_ATR_LOW", "0.55")
    monkeypatch.setenv("TEST_REENTRY_REVERT_MIN_ATR_HIGH", "0.75")
    monkeypatch.setenv("TEST_REENTRY_TREND_MIN_ATR_LOW", "0.45")
    monkeypatch.setenv("TEST_REENTRY_TREND_MIN_ATR_HIGH", "0.65")
    monkeypatch.setenv("TEST_REENTRY_TREND_MAX_ATR_LOW", "0.55")
    monkeypatch.setenv("TEST_REENTRY_TREND_MAX_ATR_HIGH", "0.35")
    monkeypatch.setenv("TEST_REENTRY_EDGE_MIN_ATR_LOW", "0.40")
    monkeypatch.setenv("TEST_REENTRY_EDGE_MIN_ATR_HIGH", "0.60")

    cfg = rd._load_config("TEST")
    assert rd._blend_by_atr(
        atr_pips=6.0,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    ) == pytest.approx(0.55)
    assert rd._blend_by_atr(
        atr_pips=10.0,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    ) == pytest.approx(0.65)
    assert rd._blend_by_atr(
        atr_pips=14.0,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    ) == pytest.approx(0.75)


def test_revert_score_weights_change_with_atr_profile(monkeypatch) -> None:
    _clear_prefix_env(monkeypatch, "TEST")
    monkeypatch.setenv("TEST_REENTRY_ENABLE", "1")
    monkeypatch.setenv("TEST_REENTRY_ATR_LOW_PIPS", "6")
    monkeypatch.setenv("TEST_REENTRY_ATR_HIGH_PIPS", "14")
    monkeypatch.setenv("TEST_REENTRY_RANGE_REVERT_BONUS_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_REVERT_BONUS_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_TREND_PENALTY_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_TREND_PENALTY_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_RSI_LOW_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_ADX_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_BBW_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_VWAP_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_RSI_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_ADX_HIGH_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_BBW_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_VWAP_HIGH_ATR", "0.0")

    cfg = rd._load_config("TEST")
    low_revert, _ = rd._reentry_scores(
        side="short",
        rsi=70.0,
        adx=35.0,
        atr_pips=6.0,
        bbw=0.30,
        vwap_gap=0.0,
        ma_pair=(1.0, 2.0),
        range_active=False,
        cfg=cfg,
    )
    high_revert, _ = rd._reentry_scores(
        side="short",
        rsi=70.0,
        adx=35.0,
        atr_pips=14.0,
        bbw=0.30,
        vwap_gap=0.0,
        ma_pair=(1.0, 2.0),
        range_active=False,
        cfg=cfg,
    )
    assert low_revert == pytest.approx(1.0)
    assert high_revert == pytest.approx(0.0)


def test_range_adjustment_changes_with_atr_profile(monkeypatch) -> None:
    _clear_prefix_env(monkeypatch, "TEST")
    monkeypatch.setenv("TEST_REENTRY_ENABLE", "1")
    monkeypatch.setenv("TEST_REENTRY_ATR_LOW_PIPS", "6")
    monkeypatch.setenv("TEST_REENTRY_ATR_HIGH_PIPS", "14")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_RSI_LOW_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_ADX_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_BBW_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_VWAP_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_RSI_HIGH_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_ADX_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_BBW_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_REVERT_WEIGHT_VWAP_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_ADX_LOW_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_ATR_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_MA_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_VWAP_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_ADX_HIGH_ATR", "1.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_ATR_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_MA_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_TREND_WEIGHT_VWAP_HIGH_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_REVERT_BONUS_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_REVERT_BONUS_HIGH_ATR", "0.2")
    monkeypatch.setenv("TEST_REENTRY_RANGE_TREND_PENALTY_LOW_ATR", "0.0")
    monkeypatch.setenv("TEST_REENTRY_RANGE_TREND_PENALTY_HIGH_ATR", "0.2")

    cfg = rd._load_config("TEST")
    low_revert, low_trend = rd._reentry_scores(
        side="short",
        rsi=58.0,
        adx=22.0,
        atr_pips=6.0,
        bbw=0.20,
        vwap_gap=0.0,
        ma_pair=(1.0, 2.0),
        range_active=True,
        cfg=cfg,
    )
    high_revert, high_trend = rd._reentry_scores(
        side="short",
        rsi=58.0,
        adx=22.0,
        atr_pips=14.0,
        bbw=0.20,
        vwap_gap=0.0,
        ma_pair=(1.0, 2.0),
        range_active=True,
        cfg=cfg,
    )
    assert low_revert == pytest.approx(0.2)
    assert high_revert == pytest.approx(0.4)
    assert low_trend == pytest.approx((22.0 - 18.0) / (35.0 - 18.0))
    assert high_trend == pytest.approx(max(low_trend - 0.2, 0.0))
