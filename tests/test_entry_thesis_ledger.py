"""Unit tests for strategy/entry_thesis_ledger.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.entry_thesis_ledger import (
    EntryThesis,
    REQUIRE_THESIS_REPAIR_VERDICT,
    UNVERIFIABLE_STATUS,
    evaluate_all_open_positions,
    evaluate_thesis_evolution,
    load_entry_thesis,
    load_latest_forecast,
    load_pending_entry_thesis,
    record_entry_thesis,
    record_entry_thesis_from_order_fill,
    record_entry_thesis_from_response,
    record_entry_thesis_from_response_result,
    write_thesis_evolution_report,
)


class _Forecast:
    """Lightweight stand-in for `DirectionalForecast`."""

    def __init__(self, direction: str, confidence: float) -> None:
        self.direction = direction
        self.confidence = confidence


class _SideEnum:
    def __init__(self, value: str) -> None:
        self.value = value


class _Position:
    def __init__(
        self,
        *,
        trade_id: str,
        pair: str,
        side: str,
        owner: str = "trader",
        open_time_utc: str | None = None,
    ) -> None:
        self.trade_id = trade_id
        self.pair = pair
        self.side = _SideEnum(side)
        self.owner = _SideEnum(owner)
        self.open_time_utc = open_time_utc


class _MarketContext:
    regime = "TREND_CONTINUATION current"
    method = "TREND_CONTINUATION"
    session = "LONDON"
    event_risk = "calendar:US_NFP"
    chart_story = (
        "matrix matrix:EUR_USD:LONG supports=2 rejects=1; "
        "GOLD_CONTEXT_TECHNICAL_DIRECTION context_asset:XAU_USD maps to EUR_USD LONG; "
        "OIL_CONTEXT_TECHNICAL_DIRECTION context_asset:WTICO_USD rejects"
    )


def _tech_chart(move: str) -> dict:
    direction = move.upper()
    if direction == "UP":
        regime = "TREND_UP"
        rsi = 70.0
        macd_hist = 0.0002
        supertrend = 1
        cloud = 1
        plus_di = 35.0
        minus_di = 10.0
        event = "CHOCH_UP"
    else:
        regime = "TREND_DOWN"
        rsi = 30.0
        macd_hist = -0.0002
        supertrend = -1
        cloud = -1
        plus_di = 10.0
        minus_di = 35.0
        event = "CHOCH_DOWN"
    return {
        "views": [
            {
                "granularity": tf,
                "regime": regime,
                "indicators": {
                    "rsi_14": rsi,
                    "macd_hist": macd_hist,
                    "supertrend_dir": supertrend,
                    "ichimoku_cloud_pos": cloud,
                    "plus_di_14": plus_di,
                    "minus_di_14": minus_di,
                },
                "structure": {"last_event": {"kind": event, "close_confirmed": True}},
            }
            for tf in ("M5", "M15")
        ]
    }


class EntryThesisLedgerTest(unittest.TestCase):
    def test_record_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            thesis = EntryThesis(
                timestamp_utc="2026-05-15T10:00:00Z",
                trade_id="42",
                pair="EUR_JPY",
                side="LONG",
                entry_price=163.12,
                forecast_direction="UP",
                forecast_confidence=0.72,
                regime="TREND",
                invalidation_price=162.50,
                target_price=164.50,
                key_drivers=["pattern", "projection"],
            )
            record_entry_thesis(thesis, root)
            loaded = load_entry_thesis("42", root)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.pair, "EUR_JPY")
            self.assertEqual(loaded.forecast_direction, "UP")
            self.assertAlmostEqual(loaded.forecast_confidence, 0.72)

    def test_load_latest_forecast_returns_most_recent_per_pair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fh = root / "forecast_history.jsonl"
            entries = [
                {"pair": "EUR_JPY", "direction": "UP", "confidence": 0.4},
                {"pair": "USD_JPY", "direction": "DOWN", "confidence": 0.6},
                {"pair": "EUR_JPY", "direction": "DOWN", "confidence": 0.7},
            ]
            fh.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
            latest = load_latest_forecast("EUR_JPY", root)
            self.assertEqual(latest, entries[2])
            self.assertEqual(load_latest_forecast("USD_JPY", root), entries[1])
            self.assertIsNone(load_latest_forecast("GBP_USD", root))

    def test_record_from_send_response_extracts_trade_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "forecast_history.jsonl").write_text(json.dumps({
                "pair": "EUR_JPY", "direction": "UP", "confidence": 0.7,
                "invalidation_price": 162.5, "target_price": 164.5,
            }) + "\n")

            class FakeIntent:
                pair = "EUR_JPY"
                side = _SideEnum("LONG")
                thesis = "EUR_JPY LONG breakout retest"
                entry = 163.10
                metadata = {"desk": "spec", "regime_state": "TREND"}

            response = {
                "orderFillTransaction": {
                    "tradeOpened": {"tradeID": "999999"},
                    "price": 163.12,
                }
            }
            written = record_entry_thesis_from_response(
                response=response, intent=FakeIntent(), data_root=root,
            )
            self.assertIsNotNone(written)
            assert written is not None
            self.assertEqual(written.trade_id, "999999")
            self.assertEqual(written.pair, "EUR_JPY")
            self.assertEqual(written.side, "LONG")
            self.assertEqual(written.forecast_direction, "UP")
            self.assertAlmostEqual(written.forecast_confidence, 0.7)
            # Verify it was actually appended to the ledger
            loaded = load_entry_thesis("999999", root)
            self.assertIsNotNone(loaded)

    def test_record_from_send_response_persists_market_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "forecast_history.jsonl").write_text(json.dumps({
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.68,
                "invalidation_price": 1.091,
                "target_price": 1.104,
                "drivers_for": ["news_theme_followthrough USD soft"],
            }) + "\n")

            class FakeIntent:
                pair = "EUR_USD"
                side = _SideEnum("LONG")
                thesis = "EUR_USD LONG with matrix support"
                entry = 1.1
                market_context = _MarketContext()
                metadata = {
                    "market_context_matrix_ref": "matrix:EUR_USD:LONG",
                    "matrix_support_count": 2,
                    "matrix_reject_count": 1,
                    "matrix_support_layers": ["context_asset_chart"],
                    "matrix_reject_layers": ["cross_asset"],
                    "matrix_support_context": [
                        "GOLD_CONTEXT_TECHNICAL_DIRECTION context_asset:XAU_USD maps to EUR_USD LONG",
                    ],
                    "matrix_reject_context": [
                        "OIL_CONTEXT_TECHNICAL_DIRECTION context_asset:WTICO_USD rejects EUR_USD LONG",
                    ],
                    "matrix_context_refs": ["context_asset:XAU_USD", "context_asset:WTICO_USD", "cot:EUR"],
                    "forecast_drivers_for": ["news_theme_followthrough USD soft"],
                    "news_refs": ["news:digest", "news:items"],
                    "news_digest_ref": "news:digest",
                    "news_signal_names": ["news_theme_followthrough"],
                }

            response = {
                "orderFillTransaction": {
                    "tradeOpened": {"tradeID": "ctx-1"},
                    "price": 1.1002,
                }
            }

            result = record_entry_thesis_from_response_result(
                response=response,
                intent=FakeIntent(),
                data_root=root,
            )

            self.assertEqual(result.status, "RECORDED")
            assert result.thesis is not None
            context = result.thesis.context_evidence
            self.assertEqual(context["market_context_matrix_ref"], "matrix:EUR_USD:LONG")
            self.assertEqual(context["matrix_support_count"], 2)
            self.assertEqual(context["matrix_support_layers"], ["context_asset_chart"])
            self.assertEqual(context["matrix_context_refs"], ["context_asset:XAU_USD", "context_asset:WTICO_USD", "cot:EUR"])
            self.assertIn("context_asset:XAU_USD", context["context_asset_refs"])
            self.assertIn("context_asset:WTICO_USD", context["context_asset_refs"])
            self.assertIn("cot:EUR", context["evidence_refs"])
            self.assertIn("XAU_USD", context["context_asset_symbols"])
            self.assertNotIn("EUR_USD", context["context_asset_symbols"])
            self.assertIn("news_context", context)
            self.assertIn("news:digest", context["evidence_refs"])
            self.assertIn("news:items", context["evidence_refs"])
            loaded = load_entry_thesis("ctx-1", root)
            assert loaded is not None
            self.assertEqual(loaded.context_evidence["market_context_matrix_ref"], "matrix:EUR_USD:LONG")

    def test_record_from_response_no_trade_id_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class FakeIntent:
                pair = "EUR_JPY"
                side = _SideEnum("LONG")
                thesis = "x"
                entry = 163.10
                metadata: dict = {}

            # Response with no tradeOpened block
            response = {"orderRejectTransaction": {"reason": "x"}}
            written = record_entry_thesis_from_response(
                response=response, intent=FakeIntent(), data_root=root,
            )
            self.assertIsNone(written)

    def test_pending_stop_response_promotes_to_entry_thesis_on_fill(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "forecast_history.jsonl").write_text(json.dumps({
                "pair": "EUR_USD", "direction": "DOWN", "confidence": 0.63,
                "invalidation_price": 1.1641, "target_price": 1.1583,
            }) + "\n")

            class FakeIntent:
                pair = "EUR_USD"
                side = _SideEnum("SHORT")
                thesis = "EUR_USD SHORT failed-break stop trigger"
                entry = 1.16013
                tp = 1.1583
                sl = 1.1641
                market_context = _MarketContext()
                metadata = {
                    "desk": "failure_trader",
                    "campaign_role": "FORECAST_FIRST",
                    "regime_state": "FAILURE_RISK",
                    "parent_lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                    "market_context_matrix_ref": "matrix:EUR_USD:SHORT",
                    "matrix_support_layers": ["context_asset_chart"],
                    "matrix_support_context": [
                        "GOLD_CONTEXT_TECHNICAL_DIRECTION context_asset:XAU_USD maps to EUR_USD SHORT",
                    ],
                }

            response = {
                "orderCreateTransaction": {
                    "id": "471491",
                    "type": "STOP_ORDER",
                    "instrument": "EUR_USD",
                }
            }
            written = record_entry_thesis_from_response(
                response=response, intent=FakeIntent(), data_root=root,
                now=datetime(2026, 5, 22, 15, 30, tzinfo=timezone.utc),
            )

            self.assertIsNone(written)
            pending = load_pending_entry_thesis("471491", root)
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertEqual(pending.forecast_direction, "DOWN")
            self.assertEqual(pending.context_evidence["market_context_matrix_ref"], "matrix:EUR_USD:SHORT")
            self.assertIn("context_asset:XAU_USD", pending.context_evidence["context_asset_refs"])

            promoted = record_entry_thesis_from_order_fill(
                transaction={
                    "id": "471492",
                    "type": "ORDER_FILL",
                    "time": "2026-05-22T15:30:43.566015322Z",
                    "orderID": "471491",
                    "instrument": "EUR_USD",
                    "units": "-2700",
                    "price": "1.16013",
                    "tradeOpened": {
                        "tradeID": "471492",
                        "units": "-2700",
                        "price": "1.16013",
                    },
                },
                data_root=root,
            )

            self.assertIsNotNone(promoted)
            assert promoted is not None
            self.assertEqual(promoted.trade_id, "471492")
            self.assertEqual(promoted.side, "SHORT")
            self.assertEqual(promoted.forecast_direction, "DOWN")
            self.assertAlmostEqual(promoted.entry_price, 1.16013)
            self.assertEqual(promoted.context_evidence["market_context_matrix_ref"], "matrix:EUR_USD:SHORT")
            loaded = load_entry_thesis("471492", root)
            self.assertIsNotNone(loaded)

    def test_record_from_response_result_verifies_pending_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class FakeIntent:
                pair = "EUR_USD"
                side = _SideEnum("SHORT")
                thesis = "pending write verification"
                entry = 1.16013
                tp = 1.1583
                sl = 1.1641
                metadata: dict = {}

            result = record_entry_thesis_from_response_result(
                response={
                    "orderCreateTransaction": {
                        "id": "471491",
                        "type": "STOP_ORDER",
                        "instrument": "EUR_USD",
                    }
                },
                intent=FakeIntent(),
                data_root=root,
            )

            self.assertEqual(result.status, "PENDING_RECORDED")
            self.assertEqual(result.order_id, "471491")
            self.assertIsNotNone(result.pending)
            self.assertEqual(result.to_dict()["pending"]["order_id"], "471491")

    def _seed_thesis(self, root: Path) -> None:
        record_entry_thesis(
            EntryThesis(
                timestamp_utc="2026-05-15T10:00:00Z",
                trade_id="T1",
                pair="EUR_JPY",
                side="LONG",
                entry_price=163.0,
                forecast_direction="UP",
                forecast_confidence=0.7,
                regime="TREND",
                invalidation_price=162.0,
                target_price=164.5,
                key_drivers=[],
            ),
            root,
        )

    def test_evolution_broken_when_forecast_flips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("DOWN", 0.65),
                current_regime="TREND",
                data_root=root,
                now=datetime(2026, 5, 15, 13, 0, tzinfo=timezone.utc),
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "BROKEN")
            self.assertEqual(ev.verdict, "RECOMMEND_CLOSE")
            self.assertIn("FORECAST FLIPPED", ev.rationale)

    def test_evolution_age_parses_oanda_nanosecond_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31.164316691Z",
                    trade_id="S1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )

            ev = evaluate_thesis_evolution(
                trade_id="S1",
                pair="EUR_USD",
                side="SHORT",
                open_time_utc=None,
                current_forecast=_Forecast("DOWN", 0.70),
                current_regime="RANGE",
                data_root=root,
                now=datetime(2026, 5, 28, 9, 55, 31, tzinfo=timezone.utc),
            )

            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertAlmostEqual(ev.age_hours, 3.0, places=3)

    def test_low_confidence_forecast_flip_is_not_close_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("DOWN", 0.21),
                current_regime="TREND",
                data_root=root,
                now=datetime(2026, 5, 15, 13, 0, tzinfo=timezone.utc),
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "WEAKENED")
            self.assertEqual(ev.verdict, "HOLD")
            self.assertIn("do not convert a weak forecast into Gate A", ev.rationale)

    def test_evolution_broken_when_long_invalidation_price_is_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("UNCLEAR", 0.1),
                current_regime="RANGE",
                data_root=root,
                current_price=161.97,
                current_price_label="bid",
                invalidation_buffer_pips=2.0,
                pair_chart=_tech_chart("DOWN"),
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "BROKEN")
            self.assertEqual(ev.verdict, "RECOMMEND_CLOSE")
            self.assertIn("invalidation hit", ev.rationale)

    def test_evolution_broken_when_short_invalidation_price_is_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31Z",
                    trade_id="S1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )
            ev = evaluate_thesis_evolution(
                trade_id="S1", pair="EUR_USD", side="SHORT",
                open_time_utc="2026-05-28T06:55:31Z",
                current_forecast=_Forecast("UNCLEAR", 0.1),
                current_regime="RANGE",
                data_root=root,
                current_price=1.16325,
                current_price_label="ask",
                invalidation_buffer_pips=2.0,
                pair_chart=_tech_chart("UP"),
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "BROKEN")
            self.assertEqual(ev.verdict, "RECOMMEND_CLOSE")
            self.assertIn("current ask 1.16325 >= buffered invalidation 1.16117", ev.rationale)

    def test_evolution_not_broken_on_small_invalidation_wick_inside_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31Z",
                    trade_id="S1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )
            ev = evaluate_thesis_evolution(
                trade_id="S1", pair="EUR_USD", side="SHORT",
                open_time_utc="2026-05-28T06:55:31Z",
                current_forecast=_Forecast("UNCLEAR", 0.1),
                current_regime="RANGE",
                data_root=root,
                current_price=1.16108,
                current_price_label="ask",
                invalidation_buffer_pips=2.0,
                pair_chart=_tech_chart("UP"),
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "WEAKENED")
            self.assertEqual(ev.verdict, "HOLD")

    def test_evolution_not_broken_without_chart_technical_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31Z",
                    trade_id="S1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )
            ev = evaluate_thesis_evolution(
                trade_id="S1", pair="EUR_USD", side="SHORT",
                open_time_utc="2026-05-28T06:55:31Z",
                current_forecast=_Forecast("UNCLEAR", 0.1),
                current_regime="RANGE",
                data_root=root,
                current_price=1.16325,
                current_price_label="ask",
                invalidation_buffer_pips=2.0,
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "WEAKENED")
            self.assertEqual(ev.verdict, "HOLD")
            self.assertIn("waiting for chart/technical confirmation", ev.rationale)

    def test_evolution_still_valid_when_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("UP", 0.85),
                current_regime="TREND",
                data_root=root,
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "STILL_VALID")
            # higher confidence than entry → EXTEND
            self.assertEqual(ev.verdict, "EXTEND")

    def test_evolution_weakened_on_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("RANGE", 0.4),
                current_regime="RANGE",
                data_root=root,
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "WEAKENED")
            self.assertEqual(ev.verdict, "HOLD")

    def test_evolution_returns_none_for_missing_thesis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # No thesis recorded
            ev = evaluate_thesis_evolution(
                trade_id="UNKNOWN", pair="EUR_JPY", side="LONG",
                open_time_utc=None,
                current_forecast=_Forecast("UP", 0.7),
                current_regime="TREND",
                data_root=root,
            )
            self.assertIsNone(ev)

    def test_evaluate_all_positions_surfaces_missing_thesis_coverage_gap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evs = evaluate_all_open_positions(
                [
                    _Position(
                        trade_id="LEGACY1",
                        pair="EUR_USD",
                        side="SHORT",
                        owner="trader",
                        open_time_utc="2026-05-28T06:55:31.164316691Z",
                    )
                ],
                current_forecasts_by_pair={"EUR_USD": _Forecast("UNCLEAR", 0.06)},
                current_regimes_by_pair={"EUR_USD": "RANGE"},
                data_root=root,
                now=datetime(2026, 5, 28, 9, 55, 31, tzinfo=timezone.utc),
            )

            self.assertEqual(len(evs), 1)
            self.assertEqual(evs[0].trade_id, "LEGACY1")
            self.assertEqual(evs[0].entry_forecast, "MISSING_ENTRY_THESIS")
            self.assertEqual(evs[0].status, UNVERIFIABLE_STATUS)
            self.assertEqual(evs[0].verdict, REQUIRE_THESIS_REPAIR_VERDICT)
            self.assertIn("missing entry_thesis_ledger row", evs[0].rationale)
            self.assertIn("do not expand TP", evs[0].rationale)
            self.assertAlmostEqual(evs[0].age_hours, 3.0, places=3)

    def test_regime_shift_demotes_still_valid_to_weakened(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)  # entry regime=TREND
            ev = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc="2026-05-15T10:00:00Z",
                current_forecast=_Forecast("UP", 0.75),
                current_regime="RANGE",  # SHIFTED
                data_root=root,
            )
            self.assertIsNotNone(ev)
            assert ev is not None
            self.assertEqual(ev.status, "WEAKENED")
            self.assertIn("regime SHIFTED", ev.rationale)

    def test_write_report_marks_missing_thesis_as_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evs = evaluate_all_open_positions(
                [_Position(trade_id="LEGACY1", pair="EUR_USD", side="SHORT", owner="trader")],
                current_forecasts_by_pair={"EUR_USD": _Forecast("UNCLEAR", 0.06)},
                current_regimes_by_pair={"EUR_USD": "RANGE"},
                data_root=root,
            )

            path = write_thesis_evolution_report(evs, data_root=root)
            payload = json.loads(path.read_text())

            self.assertEqual(payload["by_status"]["UNVERIFIABLE"], 1)
            self.assertEqual(payload["entry_thesis_coverage"]["missing"], 1)
            self.assertTrue(payload["entry_thesis_coverage"]["blocking"])
            self.assertEqual(payload["entry_thesis_coverage"]["blocking_trade_ids"], ["LEGACY1"])

    def test_evaluate_all_positions_filters_non_trader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            positions = [
                _Position(trade_id="T1", pair="EUR_JPY", side="LONG", owner="trader",
                          open_time_utc="2026-05-15T10:00:00Z"),
                _Position(trade_id="T2", pair="USD_JPY", side="SHORT", owner="user",
                          open_time_utc="2026-05-15T10:00:00Z"),
            ]
            forecasts = {
                "EUR_JPY": _Forecast("DOWN", 0.7),
                "USD_JPY": _Forecast("UP", 0.7),
            }
            regimes = {"EUR_JPY": "TREND", "USD_JPY": "TREND"}
            evs = evaluate_all_open_positions(
                positions,
                current_forecasts_by_pair=forecasts,
                current_regimes_by_pair=regimes,
                data_root=root,
            )
            # Only T1 (trader-owned + has thesis) returned
            self.assertEqual(len(evs), 1)
            self.assertEqual(evs[0].trade_id, "T1")

    def test_evaluate_all_positions_uses_quotes_for_short_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31Z",
                    trade_id="S1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )
            evs = evaluate_all_open_positions(
                [_Position(trade_id="S1", pair="EUR_USD", side="SHORT", owner="trader")],
                current_forecasts_by_pair={"EUR_USD": _Forecast("UNCLEAR", 0.1)},
                current_regimes_by_pair={"EUR_USD": "RANGE"},
                quotes_by_pair={"EUR_USD": {"bid": 1.16317, "ask": 1.16325}},
                pair_charts_by_pair={"EUR_USD": _tech_chart("UP")},
                data_root=root,
            )
            self.assertEqual(len(evs), 1)
            self.assertEqual(evs[0].status, "BROKEN")
            self.assertEqual(evs[0].verdict, "RECOMMEND_CLOSE")

    def test_write_report_summarizes_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_thesis(root)
            ev_broken = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc=None,
                current_forecast=_Forecast("DOWN", 0.7),
                current_regime="TREND",
                data_root=root,
            )
            ev_valid = evaluate_thesis_evolution(
                trade_id="T1", pair="EUR_JPY", side="LONG",
                open_time_utc=None,
                current_forecast=_Forecast("UP", 0.8),
                current_regime="TREND",
                data_root=root,
            )
            assert ev_broken is not None and ev_valid is not None
            path = write_thesis_evolution_report(
                [ev_broken, ev_valid], data_root=root,
            )
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["by_status"]["BROKEN"], 1)
            self.assertEqual(payload["by_status"]["STILL_VALID"], 1)
            self.assertEqual(payload["by_status"]["UNVERIFIABLE"], 0)
            self.assertEqual(payload["entry_thesis_coverage"]["missing"], 0)
            self.assertFalse(payload["entry_thesis_coverage"]["blocking"])


if __name__ == "__main__":
    unittest.main()
