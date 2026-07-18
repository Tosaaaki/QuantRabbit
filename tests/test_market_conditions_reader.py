from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.market_conditions_reader import (
    MarketConditionsError,
    _canonical_sha,
    read_market_conditions,
)

UTC = timezone.utc
START = datetime(2026, 7, 15, 6, 0, tzinfo=UTC)
AS_OF = START + timedelta(minutes=200)


def _series(drift: float, base: float = 1.2) -> list[dict]:
    # Monotone drift with small additive wobble: a genuine trend to the
    # efficiency-ratio classifier, not an amplitude-growing zigzag.
    closes = [
        base + i * drift + (abs(drift) * 0.3 if i % 2 else 0.0) for i in range(140)
    ]
    return [
        {"time": START + timedelta(minutes=i), "close": c}
        for i, c in enumerate(closes)
    ]


def test_usd_theme_board_reads_dominant_currency() -> None:
    # USD strengthens against everything: USD_JPY up, others down.
    panel = {
        "EUR_USD": _series(-0.0005),
        "GBP_USD": _series(-0.0005),
        "AUD_USD": _series(-0.0004),
        "USD_JPY": _series(0.0005, base=150.0),
    }
    snapshot = read_market_conditions(panel, as_of_utc=AS_OF)

    assert snapshot["dominant_theme"] is not None
    assert snapshot["dominant_theme"]["currency"] == "USD"
    assert snapshot["dominant_theme"]["direction"] == "STRONG"
    assert snapshot["trend_breadth"] > 0.5
    assert snapshot["board_reading"] in {"THEMED_TREND_BOARD", "BROAD_TREND_BOARD"}
    body = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    assert snapshot["snapshot_sha256"] == _canonical_sha(body)


def test_compressed_board_reads_squeeze() -> None:
    def tight() -> list[dict]:
        wide = [1.2 + (0.003 if i % 2 else -0.003) for i in range(120)]
        narrow = [1.2 + (0.00005 if i % 2 else -0.00005) for i in range(20)]
        closes = wide + narrow
        return [
            {"time": START + timedelta(minutes=i), "close": c}
            for i, c in enumerate(closes)
        ]

    panel = {p: tight() for p in ("EUR_USD", "GBP_USD", "AUD_USD")}
    snapshot = read_market_conditions(panel, as_of_utc=AS_OF)
    assert snapshot["board_reading"] == "COMPRESSED_BOARD"
    assert snapshot["regime_counts"].get("SQUEEZE", 0) == 3


def test_event_flag_and_empty_panel_fail_closed() -> None:
    panel = {"EUR_USD": _series(0.0005)}
    snapshot = read_market_conditions(
        panel, as_of_utc=AS_OF, high_impact_event_active=True
    )
    assert snapshot["board_reading"] == "EVENT_BOARD"

    with pytest.raises(MarketConditionsError, match="non-empty"):
        read_market_conditions({}, as_of_utc=AS_OF)
