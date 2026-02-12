from __future__ import annotations

import json


def _cache_row(epoch: float, bid: float = 150.0) -> dict[str, float]:
    ask = bid + 0.01
    return {
        "epoch": float(epoch),
        "bid": float(bid),
        "ask": float(ask),
        "mid": float((bid + ask) * 0.5),
    }


def _prepare(monkeypatch, tmp_path):
    from market_data import tick_window

    cache_path = tmp_path / "tick_cache.json"
    monkeypatch.setattr(tick_window, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(tick_window, "_MIN_RELOAD_INTERVAL_SEC", 0.0)
    tick_window._TICKS.clear()
    tick_window._cache_mtime = 0.0
    tick_window._last_reload_ts = 0.0
    return tick_window, cache_path


def test_reload_keeps_fresher_in_memory_ticks(monkeypatch, tmp_path):
    tick_window, cache_path = _prepare(monkeypatch, tmp_path)

    tick_window._TICKS.append(tick_window._TickRow(**_cache_row(200.0, 151.0)))
    cache_path.write_text(json.dumps([_cache_row(120.0, 150.0)]), encoding="utf-8")

    tick_window._reload_cache_if_updated()

    assert len(tick_window._TICKS) == 1
    assert tick_window._TICKS[-1].epoch == 200.0
    assert tick_window._TICKS[-1].bid == 151.0


def test_reload_appends_newer_file_ticks(monkeypatch, tmp_path):
    tick_window, cache_path = _prepare(monkeypatch, tmp_path)

    tick_window._TICKS.append(tick_window._TickRow(**_cache_row(100.0, 150.0)))
    cache_path.write_text(
        json.dumps(
            [
                _cache_row(90.0, 149.0),
                _cache_row(120.0, 151.0),
                _cache_row(130.0, 152.0),
            ]
        ),
        encoding="utf-8",
    )

    tick_window._reload_cache_if_updated()

    epochs = [row.epoch for row in tick_window._TICKS]
    bids = [row.bid for row in tick_window._TICKS]
    assert epochs == [100.0, 120.0, 130.0]
    assert bids == [150.0, 151.0, 152.0]
