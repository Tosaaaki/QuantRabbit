from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_m1scalper import worker as m1_worker


class _DummyPositionManager:
    def __init__(self, positions: dict | None = None, *, err: Exception | None = None) -> None:
        self._positions = positions or {}
        self._err = err

    def get_open_positions(self, include_unknown: bool = False) -> dict:
        if self._err is not None:
            raise self._err
        return self._positions


def test_open_trades_guard_blocks_when_limit_reached(monkeypatch) -> None:
    monkeypatch.setattr(m1_worker.config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(m1_worker.config, "FAIL_CLOSED_ON_POSITIONS_ERROR", True)

    allowed, reason, count = m1_worker._passes_open_trades_guard(
        _DummyPositionManager(
            {
                "scalp": {
                    "open_trades": [
                        {"strategy_tag": "M1Scalper-M1"},
                    ]
                }
            }
        ),
        "M1Scalper-M1",
    )

    assert allowed is False
    assert reason == "max_open_trades"
    assert count == 1


def test_open_trades_guard_blocks_for_same_m1_family(monkeypatch) -> None:
    monkeypatch.setattr(m1_worker.config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(m1_worker.config, "FAIL_CLOSED_ON_POSITIONS_ERROR", True)
    monkeypatch.setattr(m1_worker.config, "ENV_PREFIX", "M1SCALP")

    allowed, reason, count = m1_worker._passes_open_trades_guard(
        _DummyPositionManager(
            {
                "scalp": {
                    "open_trades": [
                        {
                            "strategy_tag": "TrendBreakout",
                            "entry_thesis": {"env_prefix": "M1SCALP"},
                        },
                    ]
                }
            }
        ),
        "M1Scalper-M1",
    )

    assert allowed is False
    assert reason == "max_open_trades"
    assert count == 1


def test_open_trades_guard_fails_closed_on_positions_error(monkeypatch) -> None:
    monkeypatch.setattr(m1_worker.config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(m1_worker.config, "FAIL_CLOSED_ON_POSITIONS_ERROR", True)

    allowed, reason, count = m1_worker._passes_open_trades_guard(
        _DummyPositionManager(err=RuntimeError("position manager down")),
        "M1Scalper-M1",
    )

    assert allowed is False
    assert reason.startswith("open_positions_error:")
    assert count == -1


def test_resolve_account_snapshot_uses_cached_snapshot_on_error(monkeypatch) -> None:
    cached = object()

    def _raise(**_kwargs):
        raise RuntimeError("summary 503")

    monkeypatch.setattr(m1_worker, "get_account_snapshot", _raise)

    snap, unavailable, err = m1_worker._resolve_account_snapshot(cached, cache_ttl_sec=0.5)

    assert snap is cached
    assert unavailable is False
    assert isinstance(err, RuntimeError)


def test_resolve_account_snapshot_marks_unavailable_without_cache(monkeypatch) -> None:
    def _raise(**_kwargs):
        raise RuntimeError("summary 503")

    monkeypatch.setattr(m1_worker, "get_account_snapshot", _raise)

    snap, unavailable, err = m1_worker._resolve_account_snapshot(None, cache_ttl_sec=0.5)

    assert snap is None
    assert unavailable is True
    assert isinstance(err, RuntimeError)
