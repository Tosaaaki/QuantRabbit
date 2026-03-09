from __future__ import annotations

import asyncio
from types import SimpleNamespace

from workers.scalp_extrema_reversal import worker


def _range_ctx(*, active: bool, score: float, mode: str) -> SimpleNamespace:
    return SimpleNamespace(active=active, score=score, mode=mode, reason=mode.lower())


def test_extrema_trend_gate_blocks_countertrend_continuation(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 24.0, "ma10": 158.452, "ma20": 158.434},
        range_ctx=_range_ctx(active=False, score=0.26, mode="TREND"),
    )

    assert ok is False
    assert diag["range_score"] == 0.26
    assert diag["against_gap_pips"] > 1.2


def test_extrema_trend_gate_blocks_trend_mode_without_range_ready(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 14.5, "ma10": 158.4402, "ma20": 158.4402},
        range_ctx=_range_ctx(active=False, score=0.2909, mode="TREND"),
    )

    assert ok is False
    assert diag["range_mode"] == 1.0
    assert diag["range_score"] == 0.2909


def test_extrema_trend_gate_allows_when_range_context_is_ready(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 24.0, "ma10": 158.452, "ma20": 158.434},
        range_ctx=_range_ctx(active=True, score=0.40, mode="RANGE"),
    )

    assert ok is True
    assert diag["range_active"] == 1.0


def test_place_order_uses_actual_free_ratio_for_cap(monkeypatch):
    called: dict[str, float] = {}

    def fake_compute_cap(**kwargs):
        called["free_ratio"] = kwargs["free_ratio"]
        return SimpleNamespace(cap=0.5, reasons={})

    async def fake_market_order(**kwargs):
        return kwargs

    monkeypatch.setattr(worker, "compute_cap", fake_compute_cap)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 3.0)
    monkeypatch.setattr(worker, "_adx", lambda *_args, **_kwargs: 18.0)
    monkeypatch.setattr(
        worker,
        "compute_units",
        lambda **_kwargs: SimpleNamespace(units=100, factors={"free_ratio": 0.07}),
    )
    monkeypatch.setattr(worker, "clamp_sl_tp", lambda **kwargs: (kwargs["sl"], kwargs["tp"]))
    monkeypatch.setattr(worker, "market_order", fake_market_order)
    monkeypatch.setattr(worker.spread_monitor, "get_state", lambda: {"spread_pips": 0.8})

    import utils.oanda_account as oanda_account

    monkeypatch.setattr(
        oanda_account,
        "get_account_snapshot",
        lambda: SimpleNamespace(free_margin_ratio=0.07),
    )

    result = asyncio.run(
        worker._place_order(
            {
                "action": "OPEN_SHORT",
                "sl_pips": 1.5,
                "tp_pips": 2.0,
                "confidence": 70,
                "tag": "scalp_extrema_reversal_live",
                "reason": "extrema_reversal",
                "extrema": {},
            },
            fac_m1={"close": 158.450, "adx": 18.0, "atr_pips": 3.0, "rsi": 58.0},
            fac_h4={},
            range_ctx=_range_ctx(active=False, score=0.32, mode="RANGE"),
        )
    )

    assert called["free_ratio"] == 0.07
    assert result["meta"]["cap"] == 0.5
