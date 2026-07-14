from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.instruments import (
    DEFAULT_TRADER_PAIRS,
    NORMAL_SPREAD_PIPS,
    NORMAL_SPREAD_PIPS_CALIBRATION_SHA256,
    NORMAL_SPREAD_PIPS_SOURCE_EVIDENCE_SHA256,
    OANDA_SPREAD_CALIBRATION_V1,
    OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
    OANDA_SPREAD_CALIBRATION_V1_PATH,
    OANDA_SPREAD_CALIBRATION_V1_SOURCE_EVIDENCE_SHA256,
    SpreadCalibrationError,
    load_oanda_spread_calibration_v1,
)
from quant_rabbit.models import AccountSummary, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
from quant_rabbit.models import BrokerSnapshot
from quant_rabbit.risk import DEFAULT_SPECS, RiskEngine, RiskPolicy


EXPECTED_SPREAD_CALIBRATION = {
    "EUR_USD": (216, 1.6, 1.7, 1.8, 1.9, 0.7),
    "GBP_USD": (216, 1.9, 2.1, 2.6, 2.8, 0.9),
    "AUD_USD": (216, 1.4, 1.5, 1.6, 1.8, 0.6),
    "NZD_USD": (216, 1.5, 1.6, 1.8, 2.1, 0.7),
    "USD_JPY": (216, 1.6, 1.8, 1.9, 2.5, 0.8),
    "USD_CAD": (216, 1.8, 1.9, 2.3, 3.3, 0.8),
    "USD_CHF": (216, 1.5, 1.7, 2.0, 2.5, 0.7),
    "EUR_GBP": (216, 1.4, 1.5, 2.1, 2.6, 0.6),
    "EUR_JPY": (216, 2.8, 3.1, 5.0, 5.7, 1.3),
    "EUR_AUD": (216, 3.6, 4.0, 6.6, 7.4, 1.6),
    "EUR_CAD": (216, 2.8, 3.1, 4.8, 6.6, 1.3),
    "EUR_CHF": (216, 1.6, 1.7, 2.6, 3.1, 0.7),
    "EUR_NZD": (216, 6.4, 7.2, 11.1, 12.3, 2.9),
    "GBP_JPY": (216, 3.4, 3.7, 7.2, 7.8, 1.5),
    "GBP_AUD": (216, 5.1, 5.7, 8.2, 9.2, 2.3),
    "GBP_CAD": (216, 3.4, 4.0, 6.5, 9.8, 1.6),
    "GBP_CHF": (216, 2.4, 2.6, 4.1, 6.8, 1.1),
    "GBP_NZD": (216, 8.4, 9.5, 13.4, 14.6, 3.8),
    "AUD_JPY": (216, 2.3, 2.5, 4.4, 4.8, 1.0),
    "AUD_CAD": (216, 2.3, 3.0, 3.6, 5.7, 1.2),
    "AUD_CHF": (216, 1.5, 1.8, 2.5, 3.8, 0.8),
    "AUD_NZD": (216, 2.7, 2.9, 5.1, 7.5, 1.2),
    "CAD_JPY": (216, 2.0, 2.6, 3.1, 4.9, 1.1),
    "CAD_CHF": (216, 1.4, 1.8, 2.9, 7.8, 0.8),
    "CHF_JPY": (216, 3.9, 4.3, 6.8, 10.7, 1.8),
    "NZD_JPY": (216, 2.6, 3.2, 5.5, 7.9, 1.3),
    "NZD_CAD": (216, 2.5, 3.3, 4.7, 6.4, 1.4),
    "NZD_CHF": (216, 1.5, 1.9, 3.1, 4.8, 0.8),
}
CALIBRATION_EVALUATION_TIME = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _fresh_calibration_payload() -> dict[str, object]:
    return json.loads(OANDA_SPREAD_CALIBRATION_V1_PATH.read_text(encoding="utf-8"))


def _sealed_calibration_bytes(payload: dict[str, object]) -> bytes:
    material = dict(payload)
    material.pop("calibration_sha256", None)
    payload["calibration_sha256"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _load_resealed_payload(payload: dict[str, object], path: Path) -> None:
    raw = _sealed_calibration_bytes(payload)
    path.write_bytes(raw)
    load_oanda_spread_calibration_v1(
        path,
        expected_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        evaluated_at_utc=CALIBRATION_EVALUATION_TIME,
    )


class InstrumentUniverseTest(unittest.TestCase):
    def test_default_trader_universe_covers_g8_crosses_with_spread_specs(self) -> None:
        self.assertEqual(len(DEFAULT_TRADER_PAIRS), 28)
        for pair in ("NZD_USD", "USD_CAD", "USD_CHF", "CAD_JPY", "CHF_JPY", "EUR_NZD"):
            self.assertIn(pair, DEFAULT_TRADER_PAIRS)
        self.assertEqual(set(DEFAULT_TRADER_PAIRS), set(NORMAL_SPREAD_PIPS))

    def test_pinned_calibration_contains_all_28_exact_evidence_values(self) -> None:
        self.assertEqual(tuple(EXPECTED_SPREAD_CALIBRATION), DEFAULT_TRADER_PAIRS)
        self.assertEqual(
            hashlib.sha256(OANDA_SPREAD_CALIBRATION_V1_PATH.read_bytes()).hexdigest(),
            OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
        )
        self.assertEqual(
            OANDA_SPREAD_CALIBRATION_V1.source_evidence_sha256,
            OANDA_SPREAD_CALIBRATION_V1_SOURCE_EVIDENCE_SHA256,
        )
        self.assertEqual(
            NORMAL_SPREAD_PIPS_SOURCE_EVIDENCE_SHA256,
            OANDA_SPREAD_CALIBRATION_V1_SOURCE_EVIDENCE_SHA256,
        )
        self.assertEqual(
            OANDA_SPREAD_CALIBRATION_V1.evidence_policy_version,
            "OANDA_M5_MBA_SESSION_SPREAD_MONTHLY_V1",
        )
        self.assertEqual(OANDA_SPREAD_CALIBRATION_V1.max_age_days_after_window, 31)
        self.assertEqual(
            OANDA_SPREAD_CALIBRATION_V1.valid_until_utc,
            datetime(2026, 8, 13, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(len(OANDA_SPREAD_CALIBRATION_V1.business_days_utc), 6)
        for pair, expected in EXPECTED_SPREAD_CALIBRATION.items():
            row = OANDA_SPREAD_CALIBRATION_V1.pairs[pair]
            self.assertEqual(
                (
                    row.sample_count,
                    row.p50_pips,
                    row.p95_pips,
                    row.p99_pips,
                    row.max_pips,
                    row.recommended_baseline_pips,
                ),
                expected,
                pair,
            )
            self.assertEqual(NORMAL_SPREAD_PIPS[pair], expected[-1], pair)

    def test_risk_and_technical_modules_share_digest_verified_spread_dict(self) -> None:
        from quant_rabbit import risk as risk_module
        from quant_rabbit.analysis import chart_reader

        self.assertIs(risk_module.NORMAL_SPREAD_PIPS, NORMAL_SPREAD_PIPS)
        self.assertIs(chart_reader.NORMAL_SPREAD_PIPS, NORMAL_SPREAD_PIPS)
        self.assertEqual(
            NORMAL_SPREAD_PIPS_CALIBRATION_SHA256,
            OANDA_SPREAD_CALIBRATION_V1.calibration_sha256,
        )
        self.assertEqual(RiskPolicy().max_spread_multiple, 2.5)
        self.assertEqual(
            {pair: spec.normal_spread_pips for pair, spec in DEFAULT_SPECS.items()},
            NORMAL_SPREAD_PIPS,
        )

    def test_runtime_spread_mapping_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            NORMAL_SPREAD_PIPS["EUR_USD"] = 99.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            DEFAULT_SPECS["EUR_USD"] = DEFAULT_SPECS["GBP_USD"]  # type: ignore[index]
        self.assertEqual(DEFAULT_SPECS["EUR_USD"].normal_spread_pips, 0.7)

    def test_risk_engine_snapshots_explicit_spec_mapping(self) -> None:
        injected_specs = {"EUR_USD": DEFAULT_SPECS["EUR_USD"]}
        engine = RiskEngine(specs=injected_specs)

        injected_specs["EUR_USD"] = DEFAULT_SPECS["GBP_USD"]

        self.assertEqual(engine.specs["EUR_USD"].pair, "EUR_USD")
        self.assertEqual(engine.specs["EUR_USD"].normal_spread_pips, 0.7)
        with self.assertRaises(TypeError):
            engine.specs["EUR_USD"] = DEFAULT_SPECS["GBP_USD"]  # type: ignore[index]

    def test_calibration_is_valid_only_from_window_end_through_monthly_expiry(self) -> None:
        for label, evaluated_at, expected_message in (
            (
                "before_window_end",
                datetime(2026, 7, 13, 14, 59, 59, tzinfo=timezone.utc),
                "not yet valid",
            ),
            (
                "after_expiry",
                datetime(2026, 8, 13, 15, 0, 1, tzinfo=timezone.utc),
                "expired",
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                SpreadCalibrationError,
                expected_message,
            ):
                load_oanda_spread_calibration_v1(
                    OANDA_SPREAD_CALIBRATION_V1_PATH,
                    expected_bytes_sha256=OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
                    evaluated_at_utc=evaluated_at,
                )

    def test_calibration_requires_aware_exact_datetime(self) -> None:
        with self.assertRaisesRegex(SpreadCalibrationError, "aware datetime"):
            load_oanda_spread_calibration_v1(
                OANDA_SPREAD_CALIBRATION_V1_PATH,
                expected_bytes_sha256=OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
                evaluated_at_utc=datetime(2026, 7, 14),
            )

    def test_loader_rejects_forged_unbounded_valid_until(self) -> None:
        payload = _fresh_calibration_payload()
        payload["valid_until_utc"] = "2030-01-01T00:00:00Z"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                SpreadCalibrationError,
                "plus the 31-day policy",
            ):
                _load_resealed_payload(
                    payload,
                    Path(temp_dir) / "calibration.json",
                )

    def test_missing_calibration_fails_closed_without_static_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"
            with self.assertRaisesRegex(SpreadCalibrationError, "unavailable"):
                load_oanda_spread_calibration_v1(
                    missing,
                    expected_bytes_sha256=OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
                    evaluated_at_utc=CALIBRATION_EVALUATION_TIME,
                )

    def test_any_unpinned_byte_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_bytes(OANDA_SPREAD_CALIBRATION_V1_PATH.read_bytes() + b" ")
            with self.assertRaisesRegex(SpreadCalibrationError, "bytes SHA-256 mismatch"):
                load_oanda_spread_calibration_v1(
                    path,
                    expected_bytes_sha256=OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
                    evaluated_at_utc=CALIBRATION_EVALUATION_TIME,
                )

    def test_canonical_content_digest_rejects_resealed_bytes_with_stale_digest(self) -> None:
        payload = _fresh_calibration_payload()
        pairs = payload["pairs"]
        self.assertIsInstance(pairs, list)
        pairs[0]["p50_pips"] = 1.7
        raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(SpreadCalibrationError, "calibration_sha256 mismatch"):
                load_oanda_spread_calibration_v1(
                    path,
                    expected_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                    evaluated_at_utc=CALIBRATION_EVALUATION_TIME,
                )

    def test_strict_loader_rejects_bool_string_subset_and_noncanonical_pair_set(self) -> None:
        cases = {
            "bool_sample": lambda payload: payload["pairs"][0].__setitem__(
                "sample_count", True
            ),
            "string_percentile": lambda payload: payload["pairs"][0].__setitem__(
                "p95_pips", "1.7"
            ),
            "subset": lambda payload: payload["pairs"].pop(),
            "duplicate_pair": lambda payload: payload["pairs"][1].__setitem__(
                "pair", "EUR_USD"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                payload = _fresh_calibration_payload()
                mutate(payload)
                with self.assertRaises(SpreadCalibrationError):
                    _load_resealed_payload(payload, Path(temp_dir) / "calibration.json")

    def test_strict_loader_rejects_both_under_and_over_derived_baselines(self) -> None:
        for label, baseline in (("under", 0.6), ("over", 0.8)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                payload = _fresh_calibration_payload()
                payload["pairs"][0]["recommended_baseline_pips"] = baseline
                with self.assertRaisesRegex(
                    SpreadCalibrationError,
                    "recommended baseline must equal",
                ):
                    _load_resealed_payload(
                        payload,
                        Path(temp_dir) / "calibration.json",
                    )

    def test_strict_loader_rejects_both_under_and_over_216_samples(self) -> None:
        for sample_count in (215, 217):
            with (
                self.subTest(sample_count=sample_count),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                payload = _fresh_calibration_payload()
                payload["pairs"][0]["sample_count"] = sample_count
                with self.assertRaisesRegex(
                    SpreadCalibrationError,
                    "sample_count must equal 216",
                ):
                    _load_resealed_payload(
                        payload,
                        Path(temp_dir) / "calibration.json",
                    )

    def test_strict_loader_rejects_post_or_claimed_broker_write(self) -> None:
        mutations = {
            "post": lambda payload: payload.__setitem__(
                "broker_http_methods_used", ["GET", "POST"]
            ),
            "write": lambda payload: payload.__setitem__(
                "broker_write_performed", True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                payload = _fresh_calibration_payload()
                mutate(payload)
                with self.assertRaises(SpreadCalibrationError):
                    _load_resealed_payload(payload, Path(temp_dir) / "calibration.json")

    def test_strict_loader_rejects_duplicate_json_keys(self) -> None:
        raw = OANDA_SPREAD_CALIBRATION_V1_PATH.read_bytes().replace(
            b'{\n  "schema":',
            b'{\n  "schema": "QR_OANDA_SPREAD_CALIBRATION_V1",\n  "schema":',
            1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(SpreadCalibrationError, "JSON is invalid"):
                load_oanda_spread_calibration_v1(
                    path,
                    expected_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                    evaluated_at_utc=CALIBRATION_EVALUATION_TIME,
                )

    def test_risk_engine_supports_expanded_g8_pair(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "NZD_USD": Quote("NZD_USD", 0.60000, 0.60008, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", 157.00, 157.01, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
        )
        intent = OrderIntent(
            pair="NZD_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=0.60200,
            sl=0.59900,
            thesis="expanded G8 universe smoke test",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="NZD strength versus USD weakness",
                chart_story="trend continuation after shallow pullback",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
            metadata={"max_loss_jpy": 10_000.0},
        )

        decision = RiskEngine().validate(intent, snapshot)

        self.assertTrue(decision.allowed, decision.block_reasons)


if __name__ == "__main__":
    unittest.main()
