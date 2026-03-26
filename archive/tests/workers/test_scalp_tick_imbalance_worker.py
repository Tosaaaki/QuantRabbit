from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

from workers.scalp_tick_imbalance import worker


def _range_ctx(*, active: bool, score: float, mode: str) -> SimpleNamespace:
    return SimpleNamespace(active=active, score=score, mode=mode)


def test_tick_imbalance_blocks_trend_exhausted_short(monkeypatch) -> None:
    monkeypatch.setattr(worker, "TICK_IMB_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "TICK_IMB_BLOCK_RANGE_MODE", False)
    monkeypatch.setattr(worker, "TICK_IMB_REQUIRE_MA_ALIGN", 0)
    monkeypatch.setattr(worker, "TICK_IMB_ENTRY_QUALITY_ENABLED", False)
    monkeypatch.setattr(
        worker, "tick_snapshot", lambda *_args, **_kwargs: ([159.31, 159.29], 4.5)
    )
    monkeypatch.setattr(
        worker,
        "tick_imbalance",
        lambda *_args, **_kwargs: SimpleNamespace(
            ratio=0.82, momentum_pips=-0.86, range_pips=1.14
        ),
    )

    signal = worker._signal_tick_imbalance(
        {
            "close": 159.29,
            "adx": 48.246,
            "atr_pips": 3.11,
            "rsi": 20.57,
            "bbw": 0.00188,
            "vwap_gap": -5.37,
            "ema_slope_10": -0.0913,
            "macd_hist": -0.1839,
        },
        range_ctx=_range_ctx(active=False, score=0.10, mode="TREND"),
        tag="TickImbalance",
    )

    assert signal is None


def test_tick_imbalance_allows_non_exhausted_reclaim_long_and_preserves_diag(
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker, "TICK_IMB_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "TICK_IMB_BLOCK_RANGE_MODE", False)
    monkeypatch.setattr(worker, "TICK_IMB_REQUIRE_MA_ALIGN", 0)
    monkeypatch.setattr(worker, "TICK_IMB_ENTRY_QUALITY_ENABLED", False)
    monkeypatch.setattr(
        worker, "tick_snapshot", lambda *_args, **_kwargs: ([159.41, 159.45], 4.5)
    )
    monkeypatch.setattr(
        worker,
        "tick_imbalance",
        lambda *_args, **_kwargs: SimpleNamespace(
            ratio=0.79, momentum_pips=0.74, range_pips=0.98
        ),
    )
    signal = worker._signal_tick_imbalance(
        {
            "close": 159.44,
            "adx": 25.651,
            "atr_pips": 3.038,
            "rsi": 62.458,
            "bbw": 0.00129,
            "vwap_gap": -16.38,
            "ema_slope_10": 0.0615,
            "macd_hist": 0.0366,
            "ma10": 159.45,
            "ma20": 159.44,
        },
        range_ctx=_range_ctx(active=False, score=0.18, mode="TREND"),
        tag="TickImbalance",
    )

    assert signal is not None
    tick_diag = signal["tick_imbalance"]
    assert tick_diag["mode"] == "tick_imbalance"
    assert tick_diag["direction"] == "long"
    assert tick_diag["exhaustion_guard"]["blocked"] == 0.0

    thesis = worker._build_entry_thesis(
        signal,
        {
            "adx": 25.651,
            "atr_pips": 3.038,
            "rsi": 62.458,
            "bbw": 0.00129,
            "vwap_gap": -16.38,
            "ema_slope_10": 0.0615,
            "macd_hist": 0.0366,
        },
        _range_ctx(active=False, score=0.18, mode="TREND"),
    )

    assert thesis["side"] == "long"
    assert thesis["tick_imbalance"]["mode"] == "tick_imbalance"
    assert thesis["tick_imbalance"]["direction"] == "long"
    assert thesis["tick_imbalance"]["exhaustion_guard"]["blocked"] == 0.0


def test_tick_imbalance_rrplus_blocks_trend_exhausted_long(monkeypatch) -> None:
    monkeypatch.setattr(worker, "TIRP_BLOCK_RANGE_MODE", False)
    monkeypatch.setattr(worker, "TIRP_REQUIRE_MA_ALIGN", False)
    monkeypatch.setattr(
        worker, "tick_snapshot", lambda *_args, **_kwargs: ([159.41, 159.45], 5.0)
    )
    monkeypatch.setattr(
        worker,
        "tick_imbalance",
        lambda *_args, **_kwargs: SimpleNamespace(
            ratio=0.84, momentum_pips=0.92, range_pips=1.20
        ),
    )
    monkeypatch.setattr(worker, "spread_ok", lambda **_kwargs: (True, {}))

    signal = worker._signal_tick_imbalance_rrplus(
        {
            "close": 159.45,
            "adx": 72.769,
            "atr_pips": 2.495,
            "rsi": 98.24,
            "bbw": 0.00285,
            "vwap_gap": 65.18,
            "ema_slope_10": 0.1488,
            "macd_hist": 0.2513,
            "ma10": 159.47,
            "ma20": 159.45,
        },
        range_ctx=_range_ctx(active=False, score=0.04, mode="TREND"),
        tag="TickImbalanceRRPlus",
    )

    assert signal is None
