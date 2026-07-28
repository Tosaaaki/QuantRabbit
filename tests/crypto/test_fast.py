from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_rabbit.crypto.cli import _select_fast_pairs
from quant_rabbit.crypto.fast import (
    FastMarketState,
    FastMicrostructureRouter,
    FastPaperConfig,
    FastPaperRunner,
    LocalOrderBook,
)
from quant_rabbit.crypto.ledger import CryptoLedger
from quant_rabbit.crypto.paper import PaperEngine


def test_order_book_buffers_diff_then_applies_above_whole_sequence() -> None:
    book = LocalOrderBook()
    assert (
        book.apply_diff(
            {"b": [["100", "3"]], "a": [], "t": 1002, "s": "11"}
        )
        is False
    )
    assert book.apply_whole(
        {
            "bids": [["100", "1"], ["99", "1"]],
            "asks": [["101", "1"], ["102", "1"]],
            "timestamp": 1000,
            "sequenceId": "10",
        }
    )
    assert book.sequence == 11
    assert book.bids[Decimal("100")] == Decimal("3")
    assert book.apply_diff(
        {"b": [["99", "0"]], "a": [["101", "2"]], "t": 1003, "s": "12"}
    )
    assert Decimal("99") not in book.bids
    assert book.asks[Decimal("101")] == Decimal("2")
    assert book.apply_whole(
        {
            "bids": [["100", "1"], ["99", "1"]],
            "asks": [["101", "1"], ["102", "1"]],
            "timestamp": 1001,
            "sequenceId": "11",
        }
    )
    assert book.sequence == 12
    assert Decimal("99") not in book.bids
    assert book.asks[Decimal("101")] == Decimal("2")


def test_order_book_accepts_monotonic_nonconsecutive_sequence() -> None:
    book = LocalOrderBook()
    assert book.apply_whole(
        {
            "bids": [["100", "1"]],
            "asks": [["101", "1"]],
            "timestamp": 1_000,
            "sequenceId": "10",
        }
    )
    assert book.apply_diff(
        {"b": [["100", "2"]], "a": [], "t": 1_001, "s": "12"}
    )
    assert book.ready
    assert book.sequence == 12
    assert book.bids[Decimal("100")] == Decimal("2")


def _ready_state() -> FastMarketState:
    state = FastMarketState("btc_jpy", 8)
    assert state.book.apply_whole(
        {
            "bids": [["100", "20"], ["99", "10"]],
            "asks": [["100.01", "1"], ["100.02", "1"]],
            "timestamp": 1_000,
            "sequenceId": "1",
        }
    )
    state.event_count = 20
    state.ticker_at_ms = 1_000
    state.observe_price(Decimal("100"))
    state.observe_price(Decimal("100.01"))
    state.observe_price(Decimal("100.02"))
    return state


def test_router_enters_then_exits_without_live_authority() -> None:
    config = FastPaperConfig(
        warmup_events=3,
        min_momentum_bps=Decimal("0.01"),
        min_imbalance=Decimal("0.01"),
        safety_buffer_bps=Decimal("0"),
        adverse_selection_bps=Decimal("0"),
        max_spread_bps=Decimal("200"),
        cooldown_ms=0,
        max_data_age_ms=5_000,
        max_hold_ms=1,
    )
    router = FastMicrostructureRouter(config)
    state = _ready_state()
    entry = router.decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0"),
        allow_short=False,
        now_ns=1_000_000,
        wall_time_ms=1_001,
    )
    assert entry["action"] == "ENTER"
    assert entry["authority"] == "NONE"
    assert entry["live_permission"] is False
    exit_decision = router.decide(
        state,
        position=Decimal("1"),
        average_cost=Decimal("100"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0"),
        allow_short=False,
        now_ns=3_000_000,
        wall_time_ms=1_001,
    )
    assert exit_decision["action"] == "EXIT"
    assert exit_decision["reason"] in {"TAKE_PROFIT", "MAX_HOLD"}


def test_router_fails_closed_on_stale_book() -> None:
    router = FastMicrostructureRouter(
        FastPaperConfig(warmup_events=1, max_data_age_ms=10)
    )
    decision = router.decide(
        _ready_state(),
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
        allow_short=False,
        now_ns=1,
        wall_time_ms=2_000,
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "STALE_STREAM_DATA"


def test_router_emits_short_only_for_margin_paper() -> None:
    state = FastMarketState("btc_jpy", 8)
    assert state.book.apply_whole(
        {
            "bids": [["100", "1"]],
            "asks": [["100.01", "100"]],
            "timestamp": 1_000,
            "sequenceId": "1",
        }
    )
    state.event_count = 20
    state.observe_price(Decimal("100.02"))
    state.observe_price(Decimal("100.01"))
    state.observe_price(Decimal("100"))
    router = FastMicrostructureRouter(
        FastPaperConfig(
            warmup_events=1,
            min_imbalance=Decimal("0.01"),
            imbalance_edge_scale_bps=Decimal("2"),
            adverse_selection_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            max_spread_bps=Decimal("5"),
            max_data_age_ms=5_000,
            cooldown_ms=0,
        )
    )
    spot = router.decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0"),
        allow_short=False,
        now_ns=1,
        wall_time_ms=1_001,
    )
    assert spot["reason"] == "SHORT_DISABLED"
    margin = router.decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0"),
        allow_short=True,
        now_ns=2,
        wall_time_ms=1_001,
    )
    assert margin["action"] == "ENTER"
    assert margin["position_side"] == "SHORT"


def test_router_blocks_entry_when_round_trip_cost_exceeds_edge() -> None:
    state = _ready_state()
    router = FastMicrostructureRouter(
        FastPaperConfig(
            warmup_events=1,
            min_imbalance=Decimal("0.01"),
            imbalance_edge_scale_bps=Decimal("1"),
            adverse_selection_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            max_spread_bps=Decimal("200"),
            cooldown_ms=0,
        )
    )
    decision = router.decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.0012"),
        allow_short=False,
        now_ns=1,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "NET_EDGE_BELOW_BUFFER"
    assert Decimal(decision["taker_cost_bps"]) == Decimal("12")


class FakeStream:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    async def messages(
        self, rooms: Iterable[str], *, max_messages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        del rooms
        for message in self._messages[:max_messages]:
            yield message


def test_fast_runner_executes_event_driven_paper_round(
    tmp_path: Path,
) -> None:
    timestamp = int(time.time() * 1000)
    messages = [
        {
            "room_name": "depth_whole_btc_jpy",
            "message": {
                "data": {
                    "bids": [["100", "100"]],
                    "asks": [["100.01", "1"]],
                    "timestamp": timestamp,
                    "sequenceId": "1",
                }
            },
        },
        {
            "room_name": "ticker_btc_jpy",
            "message": {
                "data": {
                    "buy": "100",
                    "sell": "100.01",
                    "timestamp": timestamp + 1,
                }
            },
        },
        {
            "room_name": "depth_diff_btc_jpy",
            "message": {
                "data": {
                    "b": [["100", "1"]],
                    "a": [["100.01", "100"]],
                    "t": timestamp + 2,
                    "s": "2",
                }
            },
        },
    ]
    ledger = CryptoLedger(tmp_path / "fast.db")
    paper = PaperEngine(ledger)
    runner = FastPaperRunner(
        ledger,
        paper,
        stream=FakeStream(messages),  # type: ignore[arg-type]
        config=FastPaperConfig(
            warmup_events=1,
            min_momentum_bps=Decimal("0.02"),
            min_imbalance=Decimal("0.01"),
            imbalance_edge_scale_bps=Decimal("2"),
            adverse_selection_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            cooldown_ms=0,
            max_spread_bps=Decimal("5"),
            telemetry_every_events=1,
        ),
    )
    result = asyncio.run(
        runner.run(
            ["btc_jpy"],
            {"btc_jpy": (Decimal("-0.0002"), Decimal("0"))},
            duration_sec=1,
            max_events=3,
        )
    )
    assert result["runtime"]["events_processed"] == 3
    assert result["decisions"]["actions"]["ENTER"] == 1
    assert result["decisions"]["actions"]["EXIT"] == 1
    assert result["metrics"]["trade_count"] == 2
    assert result["safety"]["broker_mutation_allowed"] is False
    assert result["ledger_integrity"]["valid"] is True


def test_margin_runner_checks_loss_cut_on_market_update_without_trade(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "margin-fast.db")
    paper = PaperEngine(
        ledger,
        allow_short=True,
        max_leverage=Decimal("2"),
    )
    paper.process_intent(
        {
            "intent_id": "open-short",
            "pair": "btc_jpy",
            "side": "SELL",
            "position_effect": "OPEN",
            "position_side": "SHORT",
            "amount": "200",
            "order_style": "PAPER_TAKER",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"bids": [["100", "200"]], "asks": [["101", "200"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    timestamp = int(time.time() * 1000)
    messages = [
        {
            "room_name": "depth_whole_btc_jpy",
            "message": {
                "data": {
                    "bids": [["149", "200"]],
                    "asks": [["150", "200"]],
                    "timestamp": timestamp,
                    "sequenceId": "1",
                }
            },
        }
    ]
    runner = FastPaperRunner(
        ledger,
        paper,
        stream=FakeStream(messages),  # type: ignore[arg-type]
        config=FastPaperConfig(
            warmup_events=100,
            telemetry_every_events=1,
        ),
    )
    result = asyncio.run(
        runner.run(
            ["btc_jpy"],
            {"btc_jpy": (Decimal("0"), Decimal("0"))},
            duration_sec=1,
            max_events=1,
        )
    )
    assert result["decisions"]["actions"]["WAIT"] == 1
    assert result["metrics"]["forced_liquidation_count"] == 1
    assert result["metrics"]["short_position_count"] == 0
    assert result["metrics"]["initial_cash_jpy"] == "10000"
    assert result["ledger_integrity"]["valid"] is True


def test_margin_pair_selection_uses_conservative_margin_fees() -> None:
    class FakeClient:
        @staticmethod
        def fetch_pair_settings() -> list[dict[str, Any]]:
            return [
                {
                    "name": "btc_jpy",
                    "quote_asset": "jpy",
                    "is_enabled": True,
                    "stop_order": False,
                    "stop_order_and_cancel": False,
                    "stop_margin_long_order": False,
                    "stop_margin_short_order": False,
                    "margin_current_individual_ratio": "0.5",
                    "maker_fee_rate_quote": "-0.0002",
                    "taker_fee_rate_quote": "0.0012",
                    "margin_open_maker_fee_rate_quote": "-0.0001",
                    "margin_close_maker_fee_rate_quote": "0",
                    "margin_open_taker_fee_rate_quote": "0.001",
                    "margin_close_taker_fee_rate_quote": "0.0015",
                    "margin_long_interest": "0.0004",
                    "margin_short_interest": "0.0005",
                }
            ]

        @staticmethod
        def fetch_tickers_jpy() -> list[dict[str, Any]]:
            return [{"pair": "btc_jpy", "last": "100", "vol": "10"}]

    pairs, fees, interest = _select_fast_pairs(
        FakeClient(),  # type: ignore[arg-type]
        [],
        1,
        margin_paper=True,
    )
    assert pairs == ["btc_jpy"]
    assert fees == {"btc_jpy": (Decimal("0"), Decimal("0.0015"))}
    assert interest == {
        "btc_jpy": (Decimal("0.0004"), Decimal("0.0005"))
    }
