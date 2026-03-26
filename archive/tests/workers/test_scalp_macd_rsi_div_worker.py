from __future__ import annotations


def test_signal_side_long_on_cross_and_bull_divergence(monkeypatch) -> None:
    from workers.scalp_macd_rsi_div import worker

    monkeypatch.setattr(worker.config, "ALLOW_HIDDEN_DIVERGENCE", False)
    monkeypatch.setattr(worker.config, "MAX_DIV_AGE_BARS", 8)
    monkeypatch.setattr(worker.config, "MIN_DIV_SCORE", 0.15)
    monkeypatch.setattr(worker.config, "MIN_DIV_STRENGTH", 0.22)
    monkeypatch.setattr(worker.config, "RSI_LONG_ENTRY", 30.0)
    monkeypatch.setattr(worker.config, "RSI_SHORT_ENTRY", 70.0)

    side, meta = worker._signal_side(
        prev_rsi=29.4,
        rsi=31.0,
        long_armed=True,
        short_armed=False,
        div_kind=1,
        div_score=0.28,
        div_strength=0.41,
        div_age_bars=3.0,
    )
    assert side == "long"
    assert meta["div_kind"] == 1.0


def test_signal_side_blocks_stale_divergence(monkeypatch) -> None:
    from workers.scalp_macd_rsi_div import worker

    monkeypatch.setattr(worker.config, "ALLOW_HIDDEN_DIVERGENCE", False)
    monkeypatch.setattr(worker.config, "MAX_DIV_AGE_BARS", 4)
    monkeypatch.setattr(worker.config, "MIN_DIV_SCORE", 0.15)
    monkeypatch.setattr(worker.config, "MIN_DIV_STRENGTH", 0.22)
    monkeypatch.setattr(worker.config, "RSI_LONG_ENTRY", 30.0)
    monkeypatch.setattr(worker.config, "RSI_SHORT_ENTRY", 70.0)

    side, _ = worker._signal_side(
        prev_rsi=71.0,
        rsi=68.0,
        long_armed=False,
        short_armed=True,
        div_kind=-1,
        div_score=-0.42,
        div_strength=0.55,
        div_age_bars=9.0,
    )
    assert side is None


def test_compute_targets_keeps_min_rr(monkeypatch) -> None:
    from workers.scalp_macd_rsi_div import worker

    monkeypatch.setattr(worker.config, "SL_ATR_MULT", 1.0)
    monkeypatch.setattr(worker.config, "TP_ATR_MULT", 0.7)
    monkeypatch.setattr(worker.config, "MIN_SL_PIPS", 0.6)
    monkeypatch.setattr(worker.config, "MAX_SL_PIPS", 4.0)
    monkeypatch.setattr(worker.config, "MIN_TP_PIPS", 0.6)
    monkeypatch.setattr(worker.config, "MAX_TP_PIPS", 4.0)
    monkeypatch.setattr(worker.config, "MIN_TP_RR", 0.95)

    tp_pips, sl_pips = worker._compute_targets(atr_pips=2.0)
    assert sl_pips == 2.0
    assert tp_pips == 1.9
