from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import (
    TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
    TECHNICAL_CANDLE_PROVENANCE_INVALID,
    TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
    TechnicalCandleBatch,
    _candles_from_payload,
    _technical_candles_from_payload,
    fetch_technical_candles_via_client,
)
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS
from quant_rabbit.risk import RiskPolicy


def _mba_candle(index: int, *, ask_widening: float = 0.0) -> dict[str, object]:
    timestamp = datetime(2026, 7, 14, tzinfo=timezone.utc) + timedelta(minutes=index)
    base = 1.1000 + index * 0.00001
    reference_mid = {"o": base, "h": base + 0.0002, "l": base - 0.0002, "c": base + 0.00005}
    bid = {key: value - 0.00003 for key, value in reference_mid.items()}
    ask = {key: value + 0.00003 + ask_widening for key, value in reference_mid.items()}
    mid = {key: (bid[key] + ask[key]) / 2.0 for key in reference_mid}
    return {
        "time": timestamp.isoformat().replace("+00:00", "Z"),
        "complete": True,
        "mid": {key: f"{value:.5f}" for key, value in mid.items()},
        "bid": {key: f"{value:.5f}" for key, value in bid.items()},
        "ask": {key: f"{value:.5f}" for key, value in ask.items()},
        "volume": 100,
    }


def _parse_technical(
    entries: list[dict[str, object]],
    *,
    granularity: str = "M1",
) -> TechnicalCandleBatch:
    return _parse_technical_payload({
        "instrument": "EUR_USD",
        "granularity": granularity,
        "candles": entries,
    }, granularity=granularity)


def _parse_technical_payload(
    payload: object,
    *,
    granularity: str = "M1",
    requested_count: int | None = None,
) -> TechnicalCandleBatch:
    normal_spread_pips = NORMAL_SPREAD_PIPS["EUR_USD"]
    max_spread_multiple = RiskPolicy().max_spread_multiple
    entries = payload.get("candles") if payload.__class__ is dict else None
    if requested_count is None:
        requested_count = len(entries) if isinstance(entries, list) and entries else 1
    return _technical_candles_from_payload(
        payload,
        pair="EUR_USD",
        granularity=granularity,
        requested_count=requested_count,
        pip_factor=10000,
        normal_spread_pips=normal_spread_pips,
        max_spread_multiple=max_spread_multiple,
    )


def _eur_usd_spread_cap_pips() -> float:
    return NORMAL_SPREAD_PIPS["EUR_USD"] * RiskPolicy().max_spread_multiple


def _above_cap_ask_widening(*, extra_pips: float = 1.0) -> float:
    # `_mba_candle` starts with a representable 0.6-pip BID/ASK width.
    return (_eur_usd_spread_cap_pips() + extra_pips - 0.6) / 10000.0


class CandleParsingTest(unittest.TestCase):
    def test_requested_count_and_granularity_cadence_fail_closed(self) -> None:
        normal = [_mba_candle(index) for index in range(3)]
        short_packet = _parse_technical_payload(
            {
                "instrument": "EUR_USD",
                "granularity": "M1",
                "candles": normal,
            },
            requested_count=200,
        )
        compressed = [_mba_candle(index) for index in range(3)]
        for index, candle in enumerate(compressed):
            candle["time"] = f"2026-07-14T00:00:0{index}Z"
        compressed_packet = _parse_technical(compressed)

        for batch in (short_packet, compressed_packet):
            with self.subTest(integrity=batch.integrity):
                self.assertTrue(batch.integrity["forecast_blocking"])
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    batch.integrity["blocking_codes"],
                )
        self.assertFalse(short_packet.integrity["coverage_complete"])
        self.assertTrue(compressed_packet.integrity["coverage_complete"])

    def test_forming_oanda_tail_candle_is_excluded_from_strategy_series(self) -> None:
        payload = {
            "candles": [
                {
                    "time": "2026-07-10T10:00:00Z",
                    "complete": True,
                    "mid": {"o": "1.1400", "h": "1.1410", "l": "1.1390", "c": "1.1405"},
                    "volume": 100,
                },
                {
                    "time": "2026-07-10T10:05:00Z",
                    "complete": False,
                    "mid": {"o": "1.1405", "h": "1.1450", "l": "1.1400", "c": "1.1448"},
                    "volume": 2,
                },
            ]
        }

        candles = _candles_from_payload(payload)

        self.assertEqual(len(candles), 1)
        self.assertTrue(candles[0].complete)
        self.assertEqual(candles[0].close, 1.1405)

    def test_missing_complete_flag_remains_backward_compatible_as_complete(self) -> None:
        payload = {
            "candles": [
                {
                    "time": "2026-07-10T10:00:00Z",
                    "mid": {"o": "1.1400", "h": "1.1410", "l": "1.1390", "c": "1.1405"},
                }
            ]
        }

        candles = _candles_from_payload(payload)

        self.assertEqual(len(candles), 1)
        self.assertTrue(candles[0].complete)

    def test_ask_only_rollover_widening_is_quarantined_from_mid_indicators(self) -> None:
        entries = [_mba_candle(index) for index in range(40)]
        entries.append(_mba_candle(40, ask_widening=_above_cap_ask_widening()))

        batch = _parse_technical(entries)

        self.assertEqual(len(batch.candles), 40)
        self.assertEqual(batch.integrity["contaminated_count"], 1)
        self.assertEqual(
            batch.integrity["spread_cap_pips"],
            round(_eur_usd_spread_cap_pips(), 6),
        )
        self.assertEqual(batch.integrity["recent_tail_state"], "SPREAD_CONTAMINATED")
        self.assertTrue(batch.integrity["forecast_blocking"])
        self.assertEqual(batch.integrity["blocking_codes"], [TECHNICAL_CANDLE_SPREAD_CONTAMINATED])
        self.assertNotIn(40, [candle.timestamp_utc.minute for candle in batch.candles])

    def test_shared_spread_cap_boundary_separates_representable_below_and_above(self) -> None:
        cap_tenth_pips = math.floor(_eur_usd_spread_cap_pips() * 10 + 1e-9)
        below_spread_pips = cap_tenth_pips / 10.0
        above_spread_pips = (cap_tenth_pips + 1) / 10.0
        clean_prefix = [_mba_candle(index) for index in range(29)]
        below_cap = _parse_technical(clean_prefix + [_mba_candle(
            29,
            ask_widening=(below_spread_pips - 0.6) / 10000.0,
        )])
        above_cap = _parse_technical(clean_prefix + [_mba_candle(
            29,
            ask_widening=(above_spread_pips - 0.6) / 10000.0,
        )])

        self.assertEqual(len(below_cap.candles), 30)
        self.assertEqual(below_cap.integrity["evaluation_status"], "PASS")
        self.assertEqual(len(above_cap.candles), 29)
        self.assertIn(TECHNICAL_CANDLE_SPREAD_CONTAMINATED, above_cap.integrity["blocking_codes"])

    def test_clean_tail_recovers_after_older_spread_contamination(self) -> None:
        entries = [_mba_candle(index) for index in range(40)]
        entries.append(_mba_candle(40, ask_widening=_above_cap_ask_widening()))
        entries.extend(_mba_candle(index) for index in range(41, 71))

        batch = _parse_technical(entries)

        self.assertEqual(len(batch.candles), 70)
        self.assertEqual(
            batch.integrity["recent_clean_tail_count"],
            TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
        )
        self.assertEqual(batch.integrity["evaluation_status"], "DEGRADED")
        self.assertEqual(batch.integrity["recent_tail_state"], "CLEAN")
        self.assertFalse(batch.integrity["forecast_blocking"])
        self.assertEqual(batch.integrity["blocking_codes"], [])
        self.assertIn(TECHNICAL_CANDLE_SPREAD_CONTAMINATED, batch.integrity["codes"])
        self.assertEqual(
            batch.candles[-1].timestamp_utc,
            datetime(2026, 7, 14, tzinfo=timezone.utc) + timedelta(minutes=70),
        )
        self.assertNotIn(
            datetime(2026, 7, 14, tzinfo=timezone.utc) + timedelta(minutes=40),
            [candle.timestamp_utc for candle in batch.candles],
        )

    def test_one_clean_tail_after_199_quarantined_rows_cannot_warm_indicators(self) -> None:
        entries = [
            _mba_candle(index, ask_widening=_above_cap_ask_widening())
            for index in range(199)
        ]
        entries.append(_mba_candle(199))

        batch = _parse_technical(entries)

        self.assertEqual(batch.integrity["raw_entry_count"], 200)
        self.assertEqual(batch.integrity["contaminated_count"], 199)
        self.assertEqual(batch.integrity["clean_count"], 1)
        self.assertEqual(batch.integrity["recent_clean_tail_count"], 1)
        self.assertFalse(batch.integrity["indicator_warmup_complete"])
        self.assertFalse(batch.integrity["recent_clean_coverage_complete"])
        self.assertTrue(batch.integrity["forecast_blocking"])
        self.assertIn(
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
            batch.integrity["blocking_codes"],
        )

    def test_total_warmup_does_not_replace_latest_clean_tail_coverage(self) -> None:
        entries = [_mba_candle(index) for index in range(40)]
        entries.append(_mba_candle(40, ask_widening=_above_cap_ask_widening()))
        entries.append(_mba_candle(41))

        batch = _parse_technical(entries)

        self.assertEqual(batch.integrity["clean_count"], 41)
        self.assertTrue(batch.integrity["indicator_warmup_complete"])
        self.assertEqual(batch.integrity["recent_clean_tail_count"], 1)
        self.assertFalse(batch.integrity["recent_clean_coverage_complete"])
        self.assertTrue(batch.integrity["forecast_blocking"])
        self.assertIn(
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
            batch.integrity["blocking_codes"],
        )

    def test_whole_cadence_no_tick_gap_preserves_clean_tail_coverage(self) -> None:
        entries = [_mba_candle(index) for index in range(30)]
        for entry in entries[15:]:
            timestamp = datetime.fromisoformat(str(entry["time"]).replace("Z", "+00:00"))
            entry["time"] = (timestamp + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

        batch = _parse_technical(entries)

        self.assertEqual(batch.integrity["recent_clean_tail_count"], 30)
        self.assertTrue(batch.integrity["recent_clean_coverage_complete"])
        self.assertFalse(batch.integrity["forecast_blocking"])

    def test_missing_bid_ask_provenance_fails_closed(self) -> None:
        entry = _mba_candle(0)
        entry.pop("ask")

        batch = _parse_technical([entry])

        self.assertEqual(batch.candles, ())
        self.assertTrue(batch.integrity["forecast_blocking"])
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, batch.integrity["blocking_codes"])

    def test_incomplete_candle_before_packet_tail_fails_closed(self) -> None:
        entries = [_mba_candle(0), _mba_candle(1)]
        entries[0]["complete"] = False

        batch = _parse_technical(entries)

        self.assertEqual(len(batch.candles), 1)
        self.assertTrue(batch.integrity["forecast_blocking"])
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, batch.integrity["blocking_codes"])

    def test_naive_oanda_timestamp_fails_closed(self) -> None:
        entry = _mba_candle(0)
        entry["time"] = "2026-07-14T00:00:00"

        batch = _parse_technical([entry])

        self.assertEqual(batch.candles, ())
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, batch.integrity["blocking_codes"])

    def test_mba_timestamp_requires_canonical_rfc3339_utc_z(self) -> None:
        for value in (
            "2026-07-14T00:00:00+00:00",
            "2026-07-14T09:00:00+09:00",
            "20260714T000000Z",
            "2026-07-14 00:00:00Z",
            "2026-07-14T00:00:00z",
            "2026-07-14T00:00:00.123456789Z",
            "2026-07-14T00:00:00.1234567890Z",
        ):
            with self.subTest(value=value):
                entry = _mba_candle(0)
                entry["time"] = value

                batch = _parse_technical([entry])

                self.assertEqual(batch.candles, ())
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    batch.integrity["blocking_codes"],
                )

    def test_mba_timestamp_accepts_canonical_nanosecond_utc_z(self) -> None:
        entries = [_mba_candle(index) for index in range(30)]
        entries[0]["time"] = "2026-07-14T00:00:00.000000000Z"

        batch = _parse_technical(entries)

        self.assertEqual(len(batch.candles), 30)
        self.assertEqual(batch.integrity["evaluation_status"], "PASS")

    def test_mba_payload_identity_must_match_requested_pair_and_timeframe(self) -> None:
        base = {
            "instrument": "EUR_USD",
            "granularity": "M1",
            "candles": [_mba_candle(0)],
        }
        variants = (
            ("missing_instrument", {key: value for key, value in base.items() if key != "instrument"}),
            ("pair_mismatch", {**base, "instrument": "GBP_USD"}),
            ("pair_nonstring", {**base, "instrument": True}),
            ("pair_repeated_list", {**base, "instrument": ["EUR_USD", "EUR_USD"]}),
            ("missing_granularity", {key: value for key, value in base.items() if key != "granularity"}),
            ("timeframe_mismatch", {**base, "granularity": "H1"}),
            ("timeframe_nonstring", {**base, "granularity": 1}),
            ("timeframe_repeated_list", {**base, "granularity": ["M1", "M1"]}),
        )

        for name, payload in variants:
            with self.subTest(name=name):
                batch = _parse_technical_payload(payload)

                self.assertEqual(batch.candles, ())
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    batch.integrity["blocking_codes"],
                )

    def test_mba_receipt_binds_verified_payload_identity(self) -> None:
        batch = _parse_technical([_mba_candle(0)], granularity="M5")

        self.assertEqual(batch.integrity["payload_instrument"], "EUR_USD")
        self.assertEqual(batch.integrity["payload_granularity"], "M5")

    def test_mid_outside_narrow_bid_ask_provenance_fails_closed(self) -> None:
        entry = _mba_candle(0)
        entry["mid"] = {
            "o": "2.00000",
            "h": "2.00020",
            "l": "1.99980",
            "c": "2.00000",
        }

        batch = _parse_technical([entry])

        self.assertEqual(batch.candles, ())
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, batch.integrity["blocking_codes"])

    def test_oanda_extrema_midpoint_divergence_remains_valid_mba_provenance(self) -> None:
        entry = {
            "time": "2026-07-14T00:00:00Z",
            "complete": True,
            "bid": {
                "o": "1.10000",
                "h": "1.10020",
                "l": "1.09980",
                "c": "1.10005",
            },
            "ask": {
                "o": "1.10012",
                "h": "1.10032",
                "l": "1.09992",
                "c": "1.10017",
            },
            # Observed OANDA MBA extrema are calculated independently: MID
            # H/L can sit away from the arithmetic BID/ASK extrema midpoint.
            "mid": {
                "o": "1.10006",
                "h": "1.10032",
                "l": "1.09980",
                "c": "1.10011",
            },
            "volume": 100,
        }

        batch = _parse_technical([entry] + [_mba_candle(index) for index in range(1, 30)])

        self.assertEqual(len(batch.candles), 30)
        self.assertEqual(batch.integrity["malformed_count"], 0)
        self.assertEqual(batch.integrity["contaminated_count"], 0)
        self.assertEqual(batch.integrity["evaluation_status"], "PASS")

    def test_oanda_extrema_width_alone_does_not_trigger_spread_quarantine(self) -> None:
        entry = _mba_candle(0)
        bid = entry["bid"]
        ask = entry["ask"]
        mid = entry["mid"]
        assert isinstance(bid, dict) and isinstance(ask, dict) and isinstance(mid, dict)
        bid["h"] = "1.10010"
        ask["h"] = "1.10110"
        mid["h"] = "1.10080"
        bid["l"] = "1.09890"
        ask["l"] = "1.09990"
        mid["l"] = "1.09920"

        batch = _parse_technical([entry] + [_mba_candle(index) for index in range(1, 30)])

        self.assertEqual(len(batch.candles), 30)
        self.assertEqual(batch.integrity["malformed_count"], 0)
        self.assertEqual(batch.integrity["contaminated_count"], 0)
        self.assertEqual(batch.integrity["evaluation_status"], "PASS")

    def test_oanda_close_endpoint_width_still_triggers_spread_quarantine(self) -> None:
        entry = _mba_candle(0, ask_widening=_above_cap_ask_widening())
        clean_entry = _mba_candle(0)
        ask = entry["ask"]
        mid = entry["mid"]
        clean_ask = clean_entry["ask"]
        clean_mid = clean_entry["mid"]
        assert all(isinstance(block, dict) for block in (ask, mid, clean_ask, clean_mid))
        for key in ("o", "l"):
            ask[key] = clean_ask[key]
            mid[key] = clean_mid[key]

        batch = _parse_technical([entry])

        self.assertEqual(batch.candles, ())
        self.assertEqual(batch.integrity["malformed_count"], 0)
        self.assertEqual(batch.integrity["contaminated_count"], 1)
        self.assertIn(
            TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
            batch.integrity["blocking_codes"],
        )

    def test_daily_wide_endpoint_is_provenance_only_not_spread_quarantined(self) -> None:
        batch = _parse_technical(
            [_mba_candle(0, ask_widening=0.0010)],
            granularity="D",
        )

        self.assertEqual(len(batch.candles), 1)
        self.assertEqual(batch.integrity["malformed_count"], 0)
        self.assertEqual(batch.integrity["contaminated_count"], 0)
        self.assertEqual(
            batch.integrity["spread_evaluation_mode"],
            "PROVENANCE_ONLY_HIGHER_TIMEFRAME",
        )
        self.assertEqual(batch.integrity["evaluation_status"], "PASS")

    def test_daily_malformed_mba_still_blocks_provenance(self) -> None:
        entry = _mba_candle(0, ask_widening=_above_cap_ask_widening())
        entry.pop("ask")

        batch = _parse_technical([entry], granularity="D")

        self.assertEqual(batch.candles, ())
        self.assertEqual(batch.integrity["malformed_count"], 1)
        self.assertIn(
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
            batch.integrity["blocking_codes"],
        )

    def test_oanda_price_fields_reject_noncanonical_number_grammar(self) -> None:
        for value in (1.1, " 1.10000", "1.1e0"):
            with self.subTest(value=value):
                entry = _mba_candle(0)
                mid = entry["mid"]
                assert isinstance(mid, dict)
                mid["o"] = value

                batch = _parse_technical([entry])

                self.assertEqual(batch.candles, ())
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    batch.integrity["blocking_codes"],
                )

    def test_quarantine_details_are_bounded_while_counts_cover_full_packet(self) -> None:
        entries = [
            _mba_candle(index, ask_widening=_above_cap_ask_widening())
            for index in range(20)
        ]

        batch = _parse_technical(entries)

        self.assertEqual(batch.integrity["contaminated_count"], 20)
        self.assertEqual(len(batch.integrity["quarantine_details"]), 8)
        self.assertEqual(batch.integrity["quarantine_details_truncated"], 12)
        window = batch.integrity["quarantine_details_window"]
        self.assertEqual(window["selection"], "LATEST_BOUNDED_WINDOW")
        self.assertEqual(window["order"], "CHRONOLOGICAL_ASC_NULLS_FIRST")
        self.assertEqual(window["start_index"], 12)
        self.assertEqual(window["end_index_exclusive"], 20)
        self.assertEqual(window["total_count"], 20)
        self.assertEqual(
            window["total_code_counts"],
            {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 20,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
        )
        self.assertEqual(
            window["published_code_counts"],
            {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 8,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
        )
        self.assertEqual(
            window["omitted_code_counts"],
            {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 12,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
        )
        self.assertEqual(window["total_timestamped_count"], 20)
        self.assertEqual(window["published_timestamped_count"], 8)
        self.assertEqual(window["omitted_timestamped_count"], 12)
        self.assertEqual(
            window["latest_timestamp_utc"],
            batch.integrity["quarantine_details"][-1]["timestamp_utc"],
        )

    def test_fetch_uses_one_mba_request(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, str]]] = []

            def get_json(self, path: str, params: dict[str, str]) -> dict[str, object]:
                self.calls.append((path, params))
                return {
                    "instrument": "EUR_USD",
                    "granularity": params["granularity"],
                    "candles": [_mba_candle(0)],
                }

        client = _Client()
        batch = fetch_technical_candles_via_client(  # type: ignore[arg-type]
            client, "EUR_USD", "M1", count=1
        )

        self.assertEqual(len(batch.candles), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["price"], "MBA")

    def test_fetch_rejects_noncanonical_count_before_network(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls = 0

            def get_json(self, _path: str, _params: dict[str, str]) -> dict[str, object]:
                self.calls += 1
                return {}

        for count in (True, 0, 5001):
            with self.subTest(count=count):
                client = _Client()
                with self.assertRaisesRegex(ValueError, "count must be an exact integer"):
                    fetch_technical_candles_via_client(  # type: ignore[arg-type]
                        client,
                        "EUR_USD",
                        "M1",
                        count=count,
                    )
                self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
