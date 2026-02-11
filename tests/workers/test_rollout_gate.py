from __future__ import annotations

from datetime import datetime, timezone

import pytest

from workers.common.rollout_gate import (
    parse_rollout_start_ts,
    parse_trade_open_ts,
    trade_passes_rollout,
)


def test_parse_rollout_start_ts_supports_seconds_ms_and_iso() -> None:
    assert parse_rollout_start_ts("1739203200") == pytest.approx(1739203200.0)
    assert parse_rollout_start_ts("1739203200000") == pytest.approx(1739203200.0)
    assert parse_rollout_start_ts("2025-02-10T00:00:00Z") == pytest.approx(1739145600.0)


def test_parse_rollout_start_ts_fallback() -> None:
    assert parse_rollout_start_ts("", default=12.0) == pytest.approx(12.0)
    assert parse_rollout_start_ts("bad-ts", default=5.0) == pytest.approx(5.0)
    assert parse_rollout_start_ts(-1.0) == pytest.approx(0.0)


def test_parse_trade_open_ts_handles_datetime_and_iso() -> None:
    dt = datetime(2025, 2, 11, 0, 0, 0, tzinfo=timezone.utc)
    assert parse_trade_open_ts(dt) == pytest.approx(1739232000.0)
    assert parse_trade_open_ts("2025-02-11T00:00:00Z") == pytest.approx(1739232000.0)


def test_trade_passes_rollout() -> None:
    start_ts = 1739232000.0
    assert trade_passes_rollout("2025-02-11T00:00:00Z", start_ts) is True
    assert trade_passes_rollout("2025-02-10T23:59:59Z", start_ts) is False
    assert trade_passes_rollout(None, start_ts, unknown_is_new=False) is False
    assert trade_passes_rollout(None, start_ts, unknown_is_new=True) is True
