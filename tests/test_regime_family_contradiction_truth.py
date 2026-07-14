from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.regime_family_contradiction_truth import (
    PRODUCTION_OANDA_BASE_URL,
    _FetchTask,
    _parse_exact_m1_bid_ask_payload,
    resolve_due_regime_family_contradiction_from_oanda,
)
from quant_rabbit.strategy.regime_family_contradiction_shadow import (
    build_regime_family_contradiction_shadow,
    load_regime_family_contradiction_ledger,
    persist_regime_family_contradiction_emission,
)
from tests.test_regime_family_contradiction_shadow import _context


def _trial(
    trial_id: str,
    *,
    pair: str = "EUR_USD",
    due: str = "2026-07-15T01:00:00Z",
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "pair": pair,
        "evaluation_due_at_utc": due,
    }


def _loaded(
    trials: list[dict[str, object]],
    *,
    results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "status": "VALID",
        "trials": trials,
        "results": results or [],
        "ledger_recorded_at_by_trial_id": {
            str(trial["trial_id"]): "2026-07-15T00:00:01Z"
            for trial in trials
        },
    }


def _selection(trials: list[dict[str, object]]) -> dict[str, object]:
    return {
        "selected_trials": trials,
        "invalid_trials": [],
        "recording_anchor_missing_count": 0,
        "recording_anchor_invalid_count": 0,
        "recording_anchor_after_due_count": 0,
    }


def _payload(
    *,
    pair: str = "EUR_USD",
    time: str = "2026-07-15T00:59:00.000000000Z",
    bid_close: str = "1.10010",
    ask_close: str = "1.10020",
) -> dict[str, object]:
    if pair == "GBP_USD":
        bid_ohlc = {
            "o": "1.30000",
            "h": "1.30030",
            "l": "1.29990",
            "c": bid_close,
        }
        ask_ohlc = {
            "o": "1.30010",
            "h": "1.30040",
            "l": "1.30000",
            "c": ask_close,
        }
    else:
        bid_ohlc = {
            "o": "1.10000",
            "h": "1.10030",
            "l": "1.09990",
            "c": bid_close,
        }
        ask_ohlc = {
            "o": "1.10010",
            "h": "1.10040",
            "l": "1.10000",
            "c": ask_close,
        }
    return {
        "instrument": pair,
        "granularity": "M1",
        "candles": [
            {
                "time": time,
                "complete": True,
                "volume": 12,
                "bid": bid_ohlc,
                "ask": ask_ohlc,
            }
        ],
    }


def _resolver_from_candles(
    trials: list[dict[str, object]],
    candles: list[object],
    **_kwargs: object,
) -> dict[str, object]:
    candle_pairs = {getattr(candle, "pair") for candle in candles}
    resolved = [
        {"trial_id": trial["trial_id"]}
        for trial in trials
        if trial["pair"] in candle_pairs
    ]
    pending = [
        str(trial["trial_id"])
        for trial in trials
        if trial["pair"] not in candle_pairs
    ]
    return {
        "status": "OK",
        "resolved_results": resolved,
        "pending_due_without_truth": pending,
    }


class _FakeClient:
    base_url = PRODUCTION_OANDA_BASE_URL
    http_timeout_seconds = 0.001

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, object]:
        assert query is not None
        self.calls.append((path, dict(query)))
        pair = path.split("/")[-2]
        response = self.responses[pair]
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, dict)
        return response


class RegimeFamilyContradictionTruthTest(unittest.TestCase):
    def test_unmocked_core_adapter_persists_and_reruns_idempotently(self) -> None:
        emitted_at = datetime(2026, 7, 15, 0, 0, 30, tzinfo=timezone.utc)
        terminal_start = emitted_at + timedelta(minutes=60, seconds=-30)
        first_as_of = terminal_start + timedelta(minutes=2)
        payload = _payload(time="2026-07-15T01:00:00.000000000Z")
        context = _context()
        shadow = build_regime_family_contradiction_shadow(
            pair="EUR_USD",
            current_price=1.1,
            detector_direction="UP",
            detector_scores={
                "UP": 100.0,
                "DOWN": 10.0,
                "RANGE": 5.0,
                "EITHER": 2.0,
            },
            technical_context_v1=context,
            entry_bid=1.09999,
            entry_ask=1.10001,
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            with mock.patch(
                "quant_rabbit.strategy.regime_family_contradiction_shadow."
                "_ledger_recorded_at_utc",
                return_value=emitted_at + timedelta(seconds=1),
            ):
                appended = persist_regime_family_contradiction_emission(
                    shadow,
                    technical_context_v1=context,
                    emitted_at_utc=emitted_at,
                    cycle_id="truth-adapter-e2e",
                    data_root=data_root,
                )
            self.assertEqual(appended, 1)
            client = _FakeClient({"EUR_USD": payload})
            factory = mock.Mock(return_value=client)
            with mock.patch(
                "quant_rabbit.strategy.regime_family_contradiction_shadow."
                "_ledger_recorded_at_utc",
                return_value=first_as_of + timedelta(seconds=1),
            ):
                first = resolve_due_regime_family_contradiction_from_oanda(
                    data_root=data_root,
                    client_factory=factory,
                    clock=lambda: first_as_of,
                )

            loaded = load_regime_family_contradiction_ledger(data_root)
            self.assertEqual(first["status"], "OK")
            self.assertEqual(first["resolved_count"], 1)
            self.assertEqual(first["persisted_count"], 1)
            self.assertEqual(loaded["status"], "VALID")
            self.assertEqual(len(loaded["results"]), 1)
            self.assertEqual(
                loaded["results"][0]["terminal_interval_start_utc"],
                "2026-07-15T01:00:00Z",
            )
            self.assertFalse(loaded["results"][0]["proof_eligible"])
            self.assertFalse(
                loaded["results"][0][
                    "source_artifact_authenticated_by_evaluator"
                ]
            )

            second = resolve_due_regime_family_contradiction_from_oanda(
                data_root=data_root,
                client_factory=factory,
                clock=lambda: first_as_of + timedelta(minutes=1),
            )
            self.assertEqual(second["status"], "NO_DUE")
            factory.assert_called_once()
            self.assertEqual(len(client.calls), 1)

    def test_no_ledger_no_due_and_corrupt_ledger_never_build_client(self) -> None:
        factory = mock.Mock()

        def now() -> datetime:
            return datetime(2026, 7, 15, 2, tzinfo=timezone.utc)

        cases = (
            (mock.Mock(return_value={"status": "MISSING"}), "NO_LEDGER"),
            (
                mock.Mock(side_effect=ValueError("hash chain mismatch")),
                "LEDGER_INVALID",
            ),
        )
        for loader, expected in cases:
            with self.subTest(expected=expected), mock.patch(
                "quant_rabbit.regime_family_contradiction_truth."
                "load_regime_family_contradiction_ledger",
                loader,
            ):
                result = resolve_due_regime_family_contradiction_from_oanda(
                    data_root=Path("unused"),
                    client_factory=factory,
                    clock=now,
                )
            self.assertEqual(result["status"], expected)

        future = _trial("future", due="2026-07-15T03:00:00Z")
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([future]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([future]),
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=now,
            )
        self.assertEqual(result["status"], "NO_DUE")
        factory.assert_not_called()

    def test_exact_and_nanosecond_due_query_only_the_frozen_m1_window(self) -> None:
        exact = _trial("exact", due="2026-07-15T01:00:00Z")
        fractional = _trial(
            "fractional",
            pair="GBP_USD",
            due="2026-07-15T02:00:00.000000001Z",
        )
        client = _FakeClient(
            {
                "EUR_USD": _payload(),
                "GBP_USD": _payload(
                    pair="GBP_USD",
                    time="2026-07-15T02:00:00Z",
                    bid_close="1.30010",
                    ask_close="1.30020",
                ),
            }
        )
        clock_calls = 0

        def clock() -> datetime:
            nonlocal clock_calls
            clock_calls += 1
            return datetime(2026, 7, 15, 2, 2, tzinfo=timezone.utc)

        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([exact, fractional]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([exact, fractional]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "persist_regime_family_contradiction_results",
            return_value=1,
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=lambda: client,
                clock=clock,
            )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_count"], 2)
        self.assertEqual(clock_calls, 1)
        calls = {path.split("/")[-2]: query for path, query in client.calls}
        self.assertEqual(calls["EUR_USD"]["from"], "2026-07-15T00:59:00Z")
        self.assertEqual(calls["EUR_USD"]["to"], "2026-07-15T01:00:00Z")
        self.assertEqual(calls["GBP_USD"]["from"], "2026-07-15T02:00:00Z")
        self.assertEqual(calls["GBP_USD"]["to"], "2026-07-15T02:01:00Z")
        for query in calls.values():
            self.assertEqual(
                query,
                {
                    "from": query["from"],
                    "to": query["to"],
                    "granularity": "M1",
                    "price": "BA",
                    "includeFirst": "true",
                    "smooth": "false",
                },
            )

    def test_due_before_next_m1_close_waits_without_client(self) -> None:
        trial = _trial(
            "wait",
            due="2026-07-15T01:00:00.000000001Z",
        )
        factory = mock.Mock()
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([trial]),
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=lambda: datetime(
                    2026,
                    7,
                    15,
                    1,
                    0,
                    30,
                    tzinfo=timezone.utc,
                ),
            )

        self.assertEqual(result["status"], "WAITING_FOR_COMPLETE_M1")
        self.assertEqual(result["pending_due_without_truth"], ["wait"])
        factory.assert_not_called()

    def test_later_historical_fetch_uses_the_same_predeclared_window(self) -> None:
        trial = _trial(
            "delayed",
            due="2026-07-15T01:00:00.000000001Z",
        )
        client = _FakeClient(
            {
                "EUR_USD": _payload(
                    time="2026-07-15T01:00:00.000000000Z",
                )
            }
        )
        factory = mock.Mock(return_value=client)
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "persist_regime_family_contradiction_results",
            return_value=1,
        ):
            early = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=lambda: datetime(
                    2026,
                    7,
                    15,
                    1,
                    0,
                    30,
                    tzinfo=timezone.utc,
                ),
            )
            later = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=lambda: datetime(
                    2026,
                    7,
                    15,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            )

        self.assertEqual(early["status"], "WAITING_FOR_COMPLETE_M1")
        self.assertEqual(later["status"], "OK")
        factory.assert_called_once()
        self.assertEqual(client.calls[0][1]["from"], "2026-07-15T01:00:00Z")
        self.assertEqual(client.calls[0][1]["to"], "2026-07-15T01:01:00Z")

    def test_one_pair_failure_is_partial_and_success_is_persisted(self) -> None:
        eur = _trial("eur")
        gbp = _trial("gbp", pair="GBP_USD")
        client = _FakeClient(
            {
                "EUR_USD": _payload(),
                "GBP_USD": TimeoutError("bounded read timeout"),
            }
        )
        persisted: list[str] = []

        def persist(rows: list[dict[str, object]], **_kwargs: object) -> int:
            persisted.append(str(rows[0]["trial_id"]))
            return 1

        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([eur, gbp]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([eur, gbp]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "persist_regime_family_contradiction_results",
            side_effect=persist,
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=lambda: client,
                clock=lambda: datetime(2026, 7, 15, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["fetch_request_count"], 2)
        self.assertEqual(result["fetched_candle_count"], 1)
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["persisted_count"], 1)
        self.assertEqual(result["pending_due_without_truth"], ["gbp"])
        self.assertEqual(persisted, ["eur"])
        self.assertFalse(result["broker_write_attempted"])
        self.assertFalse(result["live_permission"])

    def test_budget_defers_outcome_blind_tasks_without_later_price(self) -> None:
        trial = _trial("deferred")
        client = _FakeClient({"EUR_USD": _payload()})
        client.http_timeout_seconds = 15.0
        ticks = iter((0.0, 44.5))
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=lambda: client,
                clock=lambda: datetime(2026, 7, 15, 1, 2, tzinfo=timezone.utc),
                monotonic_clock=lambda: next(ticks),
                budget_seconds=45.0,
            )

        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["fetch_request_count"], 0)
        self.assertEqual(result["deferred_due_count"], 1)
        self.assertEqual(result["pending_due_without_truth"], ["deferred"])
        self.assertEqual(client.calls, [])

    def test_rerun_with_result_is_idempotent_and_does_not_refetch(self) -> None:
        trial = _trial("once")
        client = _FakeClient({"EUR_USD": _payload()})
        factory = mock.Mock(return_value=client)
        loader = mock.Mock(
            side_effect=(
                _loaded([trial]),
                _loaded([trial], results=[{"trial_id": "once"}]),
            )
        )
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            loader,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "persist_regime_family_contradiction_results",
            return_value=1,
        ):
            first = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=lambda: datetime(2026, 7, 15, 1, 2, tzinfo=timezone.utc),
            )
            second = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=factory,
                clock=lambda: datetime(2026, 7, 15, 1, 3, tzinfo=timezone.utc),
            )

        self.assertEqual(first["status"], "OK")
        self.assertEqual(second["status"], "NO_DUE")
        factory.assert_called_once()
        self.assertEqual(len(client.calls), 1)

    def test_persistence_failure_reloads_actual_idempotent_state(self) -> None:
        eur = _trial("eur")
        gbp = _trial("gbp", pair="GBP_USD")
        client = _FakeClient(
            {
                "EUR_USD": _payload(),
                "GBP_USD": _payload(
                    pair="GBP_USD",
                    bid_close="1.30010",
                    ask_close="1.30020",
                ),
            }
        )
        loader = mock.Mock(
            side_effect=(
                _loaded([eur, gbp]),
                _loaded(
                    [eur, gbp],
                    results=[{"trial_id": "eur"}, {"trial_id": "gbp"}],
                ),
            )
        )
        persist = mock.Mock(side_effect=(1, RuntimeError("concurrent append")))
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            loader,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([eur, gbp]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "resolve_due_regime_family_contradiction_trials",
            side_effect=_resolver_from_candles,
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "persist_regime_family_contradiction_results",
            persist,
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=lambda: client,
                clock=lambda: datetime(2026, 7, 15, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["resolved_count"], 2)
        self.assertEqual(result["persisted_count"], 2)
        self.assertTrue(
            any(error["phase"] == "RESULT_PERSISTENCE" for error in result["errors"])
        )

    def test_non_production_endpoint_is_rejected_before_get(self) -> None:
        trial = _trial("practice")
        client = _FakeClient({"EUR_USD": _payload()})
        client.base_url = "https://api-fxpractice.oanda.com"
        with mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "load_regime_family_contradiction_ledger",
            return_value=_loaded([trial]),
        ), mock.patch(
            "quant_rabbit.regime_family_contradiction_truth."
            "select_independent_regime_family_contradiction_trials",
            return_value=_selection([trial]),
        ):
            result = resolve_due_regime_family_contradiction_from_oanda(
                data_root=Path("unused"),
                client_factory=lambda: client,
                clock=lambda: datetime(2026, 7, 15, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(result["status"], "CLIENT_UNAVAILABLE")
        self.assertEqual(client.calls, [])

    def test_strict_payload_accepts_only_exact_complete_bid_ask_truth(self) -> None:
        task = _FetchTask(
            interval_start_epoch=1784077140,
            pair="EUR_USD",
            interval_end_epoch=1784077200,
            trial_ids=("trial",),
        )

        def parse(payload: object):
            return _parse_exact_m1_bid_ask_payload(
                payload,
                task=task,
                base_url=PRODUCTION_OANDA_BASE_URL,
                path="/v3/instruments/EUR_USD/candles",
                query={"granularity": "M1", "price": "BA"},
            )

        valid = _payload()
        candle = parse(valid)
        assert candle is not None
        self.assertEqual(candle.timestamp_utc, "2026-07-15T00:59:00Z")
        self.assertEqual(len(candle.source_sha256 or ""), 64)

        empty = copy.deepcopy(valid)
        empty["candles"] = []
        self.assertIsNone(parse(empty))

        duplicate = copy.deepcopy(valid)
        duplicate["candles"].append(copy.deepcopy(duplicate["candles"][0]))
        self.assertEqual(parse(duplicate).bid_close, candle.bid_close)

        mutations: list[tuple[str, callable]] = [
            ("INSTRUMENT_MISMATCH", lambda p: p.update(instrument="GBP_USD")),
            ("GRANULARITY_MISMATCH", lambda p: p.update(granularity="M5")),
            ("CANDLES_NOT_ARRAY", lambda p: p.update(candles={})),
            ("CANDLE_NOT_OBJECT", lambda p: p.update(candles=["bad"])),
            (
                "CANDLE_OUTSIDE_FROZEN_WINDOW",
                lambda p: p["candles"][0].update(time="2025-07-15T01:00:00Z"),
            ),
            (
                "candle.time invalid",
                lambda p: p["candles"][0].update(
                    time="2026-07-15T00:59:00+00:00"
                ),
            ),
            ("CANDLE_INCOMPLETE", lambda p: p["candles"][0].update(complete=False)),
            ("OHLC_NOT_OBJECT", lambda p: p["candles"][0].update(bid=None)),
            (
                "OHLC_DECIMAL_INVALID",
                lambda p: p["candles"][0]["bid"].update(c="1.1001"),
            ),
            (
                "OHLC_GEOMETRY_INVALID",
                lambda p: p["candles"][0]["bid"].update(h="1.09900"),
            ),
            (
                "BID_ASK_ENVELOPE_INVALID",
                lambda p: p["candles"][0]["ask"].update(l="1.09980"),
            ),
            (
                "TERMINAL_SPREAD_INVALID",
                lambda p: p["candles"][0]["ask"].update(c="1.10010"),
            ),
            ("VOLUME_INVALID", lambda p: p["candles"][0].update(volume=True)),
        ]
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                malformed = copy.deepcopy(valid)
                mutate(malformed)
                with self.assertRaisesRegex(ValueError, expected):
                    parse(malformed)

        conflict = copy.deepcopy(valid)
        second = copy.deepcopy(conflict["candles"][0])
        second["bid"]["c"] = "1.10011"
        conflict["candles"].append(second)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_INTERVAL_CONFLICT"):
            parse(conflict)

        jpy_task = _FetchTask(
            interval_start_epoch=1784077140,
            pair="USD_JPY",
            interval_end_epoch=1784077200,
            trial_ids=("jpy",),
        )
        jpy_payload = {
            "instrument": "USD_JPY",
            "granularity": "M1",
            "candles": [
                {
                    "time": "2026-07-15T00:59:00.000000000Z",
                    "complete": True,
                    "volume": 1,
                    "bid": {
                        "o": "150.000",
                        "h": "150.020",
                        "l": "149.990",
                        "c": "150.010",
                    },
                    "ask": {
                        "o": "150.010",
                        "h": "150.030",
                        "l": "150.000",
                        "c": "150.020",
                    },
                }
            ],
        }
        jpy = _parse_exact_m1_bid_ask_payload(
            jpy_payload,
            task=jpy_task,
            base_url=PRODUCTION_OANDA_BASE_URL,
            path="/v3/instruments/USD_JPY/candles",
            query={"granularity": "M1", "price": "BA"},
        )
        assert jpy is not None
        self.assertEqual(jpy.bid_close, 150.01)
        bad_jpy = copy.deepcopy(jpy_payload)
        bad_jpy["candles"][0]["bid"]["c"] = "150.01000"
        with self.assertRaisesRegex(ValueError, "OHLC_DECIMAL_INVALID"):
            _parse_exact_m1_bid_ask_payload(
                bad_jpy,
                task=jpy_task,
                base_url=PRODUCTION_OANDA_BASE_URL,
                path="/v3/instruments/USD_JPY/candles",
                query={"granularity": "M1", "price": "BA"},
            )


if __name__ == "__main__":
    unittest.main()
