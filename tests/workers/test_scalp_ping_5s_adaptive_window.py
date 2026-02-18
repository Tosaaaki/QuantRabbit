from __future__ import annotations

import os

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _signal(*, worker, window_sec: float, side: str = "long", mode: str = "momentum"):
    return worker.TickSignal(
        side=side,
        mode=mode,
        mode_score=1.0,
        momentum_score=1.0,
        revert_score=0.0,
        confidence=72,
        momentum_pips=0.8,
        trigger_pips=0.2,
        imbalance=0.7,
        tick_rate=8.0,
        span_sec=1.0,
        tick_age_ms=120.0,
        spread_pips=0.2,
        bid=150.0,
        ask=150.002,
        mid=150.001,
        range_pips=1.0,
        instant_range_pips=0.6,
        signal_window_sec=window_sec,
    )


def test_signal_window_shadow_only_keeps_live_signal(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_ENABLED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED", True)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC", (0.8, 1.5))
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES", 20)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS", 0.05)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SHADOW_LOG_INTERVAL_SEC", 99999.0)
    monkeypatch.setattr(worker, "_SIGNAL_WINDOW_SHADOW_LOG_MONO", 0.0)

    live = _signal(worker=worker, window_sec=1.2)
    alt_fast = _signal(worker=worker, window_sec=0.8)
    alt_slow = _signal(worker=worker, window_sec=1.5)

    def _fake_build(_ticks, _spread, *, signal_window_override_sec=None, allow_window_fallback=True):
        if abs(float(signal_window_override_sec or 0.0) - 0.8) < 1e-9:
            return alt_fast, "ok"
        if abs(float(signal_window_override_sec or 0.0) - 1.5) < 1e-9:
            return alt_slow, "ok"
        return None, "unsupported"

    monkeypatch.setattr(worker, "_build_tick_signal", _fake_build)
    monkeypatch.setattr(
        worker,
        "_load_signal_window_stats",
        lambda **_: [
            {
                "window_sec": 0.8,
                "window_bucket": worker._signal_window_bucket(0.8),
                "mode": "momentum",
                "side": "long",
                "sample": 60,
                "mean_pips": 0.45,
                "win_rate": 0.62,
            },
            {
                "window_sec": 1.2,
                "window_bucket": worker._signal_window_bucket(1.2),
                "mode": "momentum",
                "side": "long",
                "sample": 60,
                "mean_pips": 0.10,
                "win_rate": 0.53,
            },
        ],
    )

    selected, meta = worker._maybe_adapt_signal_window(
        ticks=[],
        spread_pips=0.2,
        base_signal=live,
    )

    assert selected.signal_window_sec == 1.2
    assert meta["applied"] is False
    assert meta["best_window_sec"] == 0.8


def test_signal_window_adaptive_applies_when_improvement_is_large(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_ENABLED", True)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC", (0.8,))
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES", 20)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS", 0.05)

    live = _signal(worker=worker, window_sec=1.2)
    alt_fast = _signal(worker=worker, window_sec=0.8)

    monkeypatch.setattr(
        worker,
        "_build_tick_signal",
        lambda *_args, **_kwargs: (alt_fast, "ok"),
    )
    monkeypatch.setattr(
        worker,
        "_load_signal_window_stats",
        lambda **_: [
            {
                "window_sec": 0.8,
                "window_bucket": worker._signal_window_bucket(0.8),
                "mode": "momentum",
                "side": "long",
                "sample": 40,
                "mean_pips": 0.52,
                "win_rate": 0.61,
            },
            {
                "window_sec": 1.2,
                "window_bucket": worker._signal_window_bucket(1.2),
                "mode": "momentum",
                "side": "long",
                "sample": 40,
                "mean_pips": 0.10,
                "win_rate": 0.51,
            },
        ],
    )

    selected, meta = worker._maybe_adapt_signal_window(
        ticks=[],
        spread_pips=0.2,
        base_signal=live,
    )

    assert selected.signal_window_sec == 0.8
    assert meta["applied"] is True
    assert meta["selected_window_sec"] == 0.8


def test_signal_window_adaptive_requires_min_sample(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_ENABLED", True)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC", (0.8,))
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED", False)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES", 50)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS", 0.0)

    live = _signal(worker=worker, window_sec=1.2)
    alt_fast = _signal(worker=worker, window_sec=0.8)

    monkeypatch.setattr(
        worker,
        "_build_tick_signal",
        lambda *_args, **_kwargs: (alt_fast, "ok"),
    )
    monkeypatch.setattr(
        worker,
        "_load_signal_window_stats",
        lambda **_: [
            {
                "window_sec": 0.8,
                "window_bucket": worker._signal_window_bucket(0.8),
                "mode": "momentum",
                "side": "long",
                "sample": 12,
                "mean_pips": 0.70,
                "win_rate": 0.70,
            },
            {
                "window_sec": 1.2,
                "window_bucket": worker._signal_window_bucket(1.2),
                "mode": "momentum",
                "side": "long",
                "sample": 120,
                "mean_pips": 0.20,
                "win_rate": 0.55,
            },
        ],
    )

    selected, meta = worker._maybe_adapt_signal_window(
        ticks=[],
        spread_pips=0.2,
        base_signal=live,
    )

    assert selected.signal_window_sec == 1.2
    assert meta["applied"] is False
    assert meta["best_window_sec"] == 0.8
