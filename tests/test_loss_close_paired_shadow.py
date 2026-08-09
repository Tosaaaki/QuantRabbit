from __future__ import annotations

import unittest
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone

from quant_rabbit.loss_close_paired_shadow import (
    MAX_PAIRED_SHADOW_FILL_DELAY_SECONDS,
    MAX_PAIRED_SHADOW_QUOTE_AGE_SECONDS,
    MAX_PAIRED_SHADOW_UNITS,
    PAIRED_SHADOW_STATE_CONTRACT,
    S5BidAskCandle,
    S5Ohlc,
    paired_shadow_state_sha256,
    score_loss_close_paired_shadow,
    seal_paired_shadow_state,
    validate_paired_shadow_state,
)


def _dt(second: int = 0) -> datetime:
    return datetime(2026, 7, 14, 0, 0, second, tzinfo=timezone.utc)


def _state(*, side: str = "LONG", **updates: object) -> dict[str, object]:
    if side == "LONG":
        bid, ask, executable, tp, sl = 100.0, 100.1, 100.0, 102.0, 99.0
    else:
        bid, ask, executable, tp, sl = 99.9, 100.0, 100.0, 98.0, 101.0
    body: dict[str, object] = {
        "contract": PAIRED_SHADOW_STATE_CONTRACT,
        "trade_id": "test-trade-1",
        "close_decision_event_uid": "gpt-close:test-trade-1:decision:1",
        "pair": "USD_JPY",
        "side": side,
        "units": 10,
        "decision_timestamp_utc": "2026-07-14T00:00:00Z",
        "quote_timestamp_utc": "2026-07-14T00:00:00Z",
        "decision_bid": bid,
        "decision_ask": ask,
        "executable_close_price": executable,
        "take_profit": tp,
        "stop_loss": sl,
        "quote_to_jpy": 2.0,
        "broker_snapshot_sha256": "b" * 64,
        "decision_unrealized_pnl_jpy": -1.0,
        "close_verifier_receipt_sha256": "c" * 64,
        "close_verifier_verdict": "PASS",
        "technical_context_sha256": "d" * 64,
        "cost_surface_sha256": "a" * 64,
        "take_profit_exit_non_spread_cost_jpy": 3.0,
        "stop_loss_exit_non_spread_cost_jpy": 5.0,
        "control_financing_stress_jpy": 7.0,
        "read_only": True,
        "live_permission_allowed": False,
    }
    body.update(updates)
    return seal_paired_shadow_state(body)


def _ohlc(
    open_price: float,
    high: float,
    low: float,
    close: float | None = None,
) -> S5Ohlc:
    return S5Ohlc(
        open=open_price,
        high=high,
        low=low,
        close=open_price if close is None else close,
    )


def _candle(
    second: int,
    *,
    bid: S5Ohlc,
    ask: S5Ohlc,
    complete: bool = True,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=_dt(second),
        pair="USD_JPY",
        bid=bid,
        ask=ask,
        complete=complete,
    )


def _score(
    state: object,
    candles: tuple[S5BidAskCandle, ...],
    **updates: object,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "gpt_exit_price": 100.0,
        "gpt_exit_non_spread_cost_jpy": 0.0,
        "gpt_financing_cost_jpy": 0.0,
        "gpt_execution_evidence_sha256": "e" * 64,
        "gpt_fill_timestamp_utc": "2026-07-14T00:00:00Z",
    }
    arguments.update(updates)
    return score_loss_close_paired_shadow(state, candles, **arguments)  # type: ignore[arg-type]


class _ChangingCandleSequence(Sequence[S5BidAskCandle]):
    """Expose a different candle on a second iteration (TOCTOU regression)."""

    def __init__(self, first: S5BidAskCandle, later: S5BidAskCandle) -> None:
        self._first = first
        self._later = later
        self.iterations = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> S5BidAskCandle:
        if index in {0, -1}:
            return self._first if self.iterations <= 1 else self._later
        raise IndexError(index)

    def __iter__(self) -> Iterator[S5BidAskCandle]:
        self.iterations += 1
        return iter((self._first if self.iterations == 1 else self._later,))


class _SplitReadState(Mapping[str, object]):
    """Return sealed values to get/items but a different calculation value."""

    def __init__(self, sealed: Mapping[str, object]) -> None:
        self._sealed = dict(sealed)

    def __len__(self) -> int:
        return len(self._sealed)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sealed)

    def __getitem__(self, key: str) -> object:
        if key == "executable_close_price":
            return 50.0
        return self._sealed[key]

    def get(self, key: str, default: object = None) -> object:
        return self._sealed.get(key, default)

    def items(self):  # type: ignore[no-untyped-def]
        return self._sealed.items()


class _AliasKey:
    def __str__(self) -> str:
        return "pair"


class _RedirectFloat(float):
    def __new__(cls, stored: float, converted: float):
        instance = super().__new__(cls, stored)
        instance.converted = converted
        return instance

    def __float__(self) -> float:
        return self.converted


class _RedirectInt(int):
    def __new__(cls, stored: int, converted: int):
        instance = super().__new__(cls, stored)
        instance.converted = converted
        return instance

    def __int__(self) -> int:
        return self.converted


class LossClosePairedShadowTest(unittest.TestCase):
    def assert_audit_only(self, payload: dict[str, object]) -> None:
        self.assertIs(payload["read_only"], True)
        self.assertIs(payload["live_permission_allowed"], False)
        self.assertIs(payload["proof_eligible"], False)
        self.assertIs(payload["artifact_bindings_verified_by_evaluator"], False)
        self.assertIs(payload["external_artifact_verification_required"], True)
        self.assertIs(payload["state_sha_is_content_hash_not_event_identity"], True)

    def test_state_hash_detects_tamper_and_missing_protection_fails_closed(
        self,
    ) -> None:
        state = _state()
        valid = validate_paired_shadow_state(state)
        self.assertTrue(valid["valid"])
        self.assertEqual(state["state_sha256"], paired_shadow_state_sha256(state))
        self.assertEqual(
            paired_shadow_state_sha256(state),
            paired_shadow_state_sha256(dict(reversed(tuple(state.items())))),
        )
        self.assert_audit_only(valid)

        tampered = dict(state)
        tampered["take_profit"] = 103.0
        tamper_validation = validate_paired_shadow_state(tampered)
        self.assertFalse(tamper_validation["valid"])
        self.assertIn("STATE_SHA256_MISMATCH", tamper_validation["issues"])
        self.assert_audit_only(tamper_validation)

        missing = dict(state)
        missing.pop("stop_loss")
        missing["state_sha256"] = paired_shadow_state_sha256(missing)
        missing_validation = validate_paired_shadow_state(missing)
        self.assertFalse(missing_validation["valid"])
        self.assertIn("MISSING_FIELD:stop_loss", missing_validation["issues"])
        blocked = score_loss_close_paired_shadow(missing, ())
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIsNone(blocked["delta_R"])
        self.assert_audit_only(blocked)

    def test_seal_canonicalizes_state_representation_and_requires_event_uid(
        self,
    ) -> None:
        canonical = _state()
        equivalent = dict(canonical)
        equivalent.pop("state_sha256")
        equivalent.update(
            {
                "decision_timestamp_utc": "2026-07-14T00:00:00.0Z",
                "quote_timestamp_utc": "2026-07-14T00:00:00.000Z",
                "decision_bid": 100,
                "executable_close_price": 100,
                "take_profit": 102,
                "stop_loss": 99,
                "quote_to_jpy": 2,
                "decision_unrealized_pnl_jpy": -1,
                "take_profit_exit_non_spread_cost_jpy": 3,
                "stop_loss_exit_non_spread_cost_jpy": 5,
                "control_financing_stress_jpy": 7,
            }
        )

        resealed = seal_paired_shadow_state(equivalent)

        self.assertEqual(resealed, canonical)
        self.assertEqual(resealed["state_sha256"], canonical["state_sha256"])

        direct_noncanonical = dict(canonical)
        direct_noncanonical["decision_bid"] = 100
        direct_noncanonical["state_sha256"] = paired_shadow_state_sha256(
            direct_noncanonical
        )
        validation = validate_paired_shadow_state(direct_noncanonical)
        self.assertFalse(validation["valid"])
        self.assertIn("NON_CANONICAL_NUMBER:decision_bid", validation["issues"])

        for updates in (
            {"trade_id": " test-trade-1 "},
            {"close_decision_event_uid": "bad event uid"},
            {"close_decision_event_uid": None},
            {"decision_bid": 10**400},
        ):
            with self.subTest(updates=updates):
                invalid = dict(canonical)
                invalid.pop("state_sha256")
                invalid.update(updates)
                with self.assertRaises(ValueError):
                    seal_paired_shadow_state(invalid)

    def test_state_snapshot_prevents_mapping_toctou_and_key_aliases(self) -> None:
        state = _state()
        split_read = _SplitReadState(state)
        tp = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )

        blocked = _score(split_read, (tp,))

        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(
            any("STATE_SHA256_MISMATCH" in issue for issue in blocked["blockers"])
        )
        self.assertIsNone(blocked["delta_R"])
        self.assert_audit_only(blocked)

        aliased: dict[object, object] = dict(state)
        aliased[_AliasKey()] = "EUR_USD"
        with self.assertRaises(ValueError):
            paired_shadow_state_sha256(aliased)  # type: ignore[arg-type]
        alias_validation = validate_paired_shadow_state(aliased)
        self.assertFalse(alias_validation["valid"])
        self.assertIn("NON_STRING_STATE_KEY", alias_validation["issues"])
        self.assert_audit_only(alias_validation)

    def test_state_and_s5_require_exact_builtin_scalar_types(self) -> None:
        state = _state()
        for field, redirected in (
            ("take_profit", _RedirectFloat(102.0, 150.0)),
            ("units", _RedirectInt(10, 1)),
        ):
            with self.subTest(field=field):
                altered = dict(state)
                altered[field] = redirected
                altered["state_sha256"] = paired_shadow_state_sha256(altered)
                validation = validate_paired_shadow_state(altered)
                self.assertFalse(validation["valid"])
                expected_issue = (
                    "INVALID_UNITS"
                    if field == "units"
                    else "NON_CANONICAL_NUMBER:take_profit"
                )
                self.assertIn(expected_issue, validation["issues"])
                blocked = _score(
                    altered,
                    (
                        _candle(
                            0,
                            bid=_ohlc(100.0, 102.1, 99.5),
                            ask=_ohlc(100.1, 102.2, 99.6),
                        ),
                    ),
                )
                self.assertEqual(blocked["status"], "BLOCKED")

        redirected_high = _RedirectFloat(101.0, 200.0)
        forged_candle = _candle(
            0,
            bid=S5Ohlc(100.0, redirected_high, 99.5, 100.0),
            ask=S5Ohlc(100.1, _RedirectFloat(101.1, 200.1), 99.6, 100.1),
        )
        forged = _score(_state(), (forged_candle,), gpt_exit_price=150.0)
        self.assertEqual(forged["status"], "BLOCKED")
        self.assertTrue(
            any("INVALID_BID_OHLC" in issue for issue in forged["blockers"])
        )
        self.assertIsNone(forged["delta_R"])

    def test_quote_freshness_uses_exact_twenty_second_boundary(self) -> None:
        at_limit = dict(_state())
        at_limit["decision_timestamp_utc"] = "2026-07-14T00:00:20Z"
        at_limit["quote_timestamp_utc"] = "2026-07-14T00:00:00Z"
        at_limit["state_sha256"] = paired_shadow_state_sha256(at_limit)

        limit_validation = validate_paired_shadow_state(at_limit)

        self.assertTrue(limit_validation["valid"])
        self.assertEqual(limit_validation["quote_age_nanoseconds"], 20_000_000_000)
        self.assertEqual(limit_validation["quote_age_seconds"], 20.0)
        self.assertEqual(
            limit_validation["max_quote_age_seconds"],
            MAX_PAIRED_SHADOW_QUOTE_AGE_SECONDS,
        )
        self.assert_audit_only(limit_validation)

        over_limit = dict(at_limit)
        over_limit["decision_timestamp_utc"] = "2026-07-14T00:00:20.000000001Z"
        over_limit["state_sha256"] = paired_shadow_state_sha256(over_limit)

        stale_validation = validate_paired_shadow_state(over_limit)

        self.assertFalse(stale_validation["valid"])
        self.assertIn("QUOTE_STALE_AT_DECISION", stale_validation["issues"])
        self.assertEqual(stale_validation["quote_age_nanoseconds"], 20_000_000_001)
        self.assertGreater(stale_validation["quote_age_seconds"], 20.0)
        self.assert_audit_only(stale_validation)

        future_quote = dict(_state())
        future_quote["decision_timestamp_utc"] = "2026-07-14T00:00:00.123456788Z"
        future_quote["quote_timestamp_utc"] = "2026-07-14T00:00:00.123456789Z"
        future_quote["state_sha256"] = paired_shadow_state_sha256(future_quote)

        future_validation = validate_paired_shadow_state(future_quote)

        self.assertFalse(future_validation["valid"])
        self.assertIn("QUOTE_AFTER_DECISION", future_validation["issues"])
        self.assertEqual(future_validation["quote_age_nanoseconds"], -1)
        self.assert_audit_only(future_validation)

    def test_units_are_bounded_before_economic_float_conversion(self) -> None:
        oversized = dict(_state())
        oversized["units"] = MAX_PAIRED_SHADOW_UNITS + 1
        oversized["state_sha256"] = paired_shadow_state_sha256(oversized)

        validation = validate_paired_shadow_state(oversized)

        self.assertFalse(validation["valid"])
        self.assertIn("INVALID_UNITS", validation["issues"])
        blocked = _score(
            oversized,
            (
                _candle(
                    0,
                    bid=_ohlc(100.0, 102.1, 99.5),
                    ask=_ohlc(100.1, 102.2, 99.6),
                ),
            ),
        )
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIsNone(blocked["delta_R"])
        self.assert_audit_only(blocked)

    def test_state_requires_loss_side_snapshot_pass_and_technical_proofs(self) -> None:
        for unrealized in (0.0, 1.0):
            with self.subTest(unrealized=unrealized):
                not_loss_side = dict(_state())
                not_loss_side["decision_unrealized_pnl_jpy"] = unrealized
                not_loss_side["state_sha256"] = paired_shadow_state_sha256(
                    not_loss_side
                )
                validation = validate_paired_shadow_state(not_loss_side)
                self.assertFalse(validation["valid"])
                self.assertIn("DECISION_NOT_LOSS_SIDE", validation["issues"])
                self.assert_audit_only(validation)

        for digest_field in (
            "broker_snapshot_sha256",
            "close_verifier_receipt_sha256",
            "technical_context_sha256",
        ):
            with self.subTest(digest_field=digest_field):
                missing_proof = dict(_state())
                missing_proof.pop(digest_field)
                missing_proof["state_sha256"] = paired_shadow_state_sha256(
                    missing_proof
                )
                validation = validate_paired_shadow_state(missing_proof)
                self.assertFalse(validation["valid"])
                self.assertIn(f"MISSING_FIELD:{digest_field}", validation["issues"])
                self.assertIn(f"INVALID_SHA256:{digest_field}", validation["issues"])

        non_pass = dict(_state())
        non_pass["close_verifier_verdict"] = "BLOCK"
        non_pass["state_sha256"] = paired_shadow_state_sha256(non_pass)
        validation = validate_paired_shadow_state(non_pass)
        self.assertFalse(validation["valid"])
        self.assertIn("CLOSE_VERIFIER_VERDICT_NOT_PASS", validation["issues"])

    def test_long_uses_bid_barrier_not_ask(self) -> None:
        state = _state()
        candles = (
            # Ask crosses the long TP, but the executable bid does not.
            _candle(
                0,
                bid=_ohlc(100.0, 101.9, 99.5),
                ask=_ohlc(100.1, 102.2, 99.6),
            ),
            _candle(
                5,
                bid=_ohlc(101.9, 102.1, 101.8),
                ask=_ohlc(102.0, 102.2, 101.9),
            ),
        )

        scored = _score(state, candles)

        self.assertEqual(scored["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(scored["first_touch"]["reason"], "TP")
        self.assertEqual(scored["first_touch"]["quote_side"], "BID")
        self.assertEqual(
            scored["first_touch"]["candle_timestamp_utc"], "2026-07-14T00:00:05Z"
        )
        self.assert_audit_only(scored)

    def test_short_uses_ask_barrier_not_bid(self) -> None:
        state = _state(side="SHORT")
        candles = (
            # Bid crosses the short TP, but ask remains above it.
            _candle(
                0,
                bid=_ohlc(99.9, 100.2, 97.8),
                ask=_ohlc(100.0, 100.3, 98.1),
            ),
            _candle(
                5,
                bid=_ohlc(98.0, 98.2, 97.7),
                ask=_ohlc(98.1, 98.3, 97.9),
            ),
        )

        scored = _score(state, candles)

        self.assertEqual(scored["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(scored["first_touch"]["reason"], "TP")
        self.assertEqual(scored["first_touch"]["quote_side"], "ASK")
        self.assertEqual(
            scored["first_touch"]["candle_timestamp_utc"], "2026-07-14T00:00:05Z"
        )

    def test_partial_decision_candle_must_be_present_and_unambiguous(self) -> None:
        decision_timestamp = "2026-07-14T00:00:00.000000001Z"
        state = _state(
            decision_timestamp_utc=decision_timestamp,
            quote_timestamp_utc="2026-07-14T00:00:00Z",
        )
        no_touch_containing = _candle(
            0,
            bid=_ohlc(100.0, 101.0, 99.5),
            ask=_ohlc(100.1, 101.1, 99.6),
        )
        next_tp = _candle(
            5,
            bid=_ohlc(101.9, 102.1, 101.8),
            ask=_ohlc(102.0, 102.2, 101.9),
        )

        scored = _score(
            state,
            (no_touch_containing, next_tp),
            gpt_fill_timestamp_utc=decision_timestamp,
        )

        self.assertEqual(scored["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(
            scored["first_touch"]["candle_timestamp_utc"], "2026-07-14T00:00:05Z"
        )
        self.assert_audit_only(scored)

        touching_containing = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        ambiguous = _score(
            state,
            (touching_containing, next_tp),
            gpt_fill_timestamp_utc=decision_timestamp,
        )
        self.assertEqual(ambiguous["status"], "BLOCKED")
        self.assertIn("S5_DECISION_CANDLE_TOUCH_AMBIGUOUS", ambiguous["blockers"])
        self.assertIsNone(ambiguous["delta_R"])
        self.assert_audit_only(ambiguous)

        missing_containing = _score(
            state,
            (next_tp,),
            gpt_fill_timestamp_utc=decision_timestamp,
        )
        self.assertEqual(missing_containing["status"], "BLOCKED")
        self.assertIn("S5_TRUTH_LEADING_GAP", missing_containing["blockers"])
        self.assertIsNone(missing_containing["delta_R"])

        aligned = _score(
            _state(),
            (touching_containing,),
        )
        self.assertEqual(aligned["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(aligned["first_touch"]["reason"], "TP")

    def test_same_s5_candle_tp_sl_order_is_blocked(self) -> None:
        state = _state()
        candle = _candle(
            0,
            bid=_ohlc(100.0, 102.2, 98.8),
            ask=_ohlc(100.1, 102.3, 98.9),
        )

        scored = _score(state, (candle,))

        self.assertEqual(scored["status"], "BLOCKED")
        self.assertIn("S5_CONTROL_SAME_CANDLE_TOUCH_AMBIGUOUS", scored["blockers"])
        self.assertEqual(scored["first_touch"]["reason"], "AMBIGUOUS")
        self.assertTrue(scored["first_touch"]["same_candle_tp_and_sl"])
        self.assertEqual(
            scored["first_touch"]["same_candle_policy"],
            "BLOCK_UNORDERED_S5_TOUCH",
        )
        self.assertIsNone(scored["delta_R"])

    def test_incomplete_and_conflicting_candles_fail_closed(self) -> None:
        state = _state()
        incomplete = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
            complete=False,
        )
        blocked_incomplete = _score(state, (incomplete,))
        self.assertEqual(blocked_incomplete["status"], "BLOCKED")
        self.assertTrue(
            any(
                "INCOMPLETE_S5_CANDLE" in item
                for item in blocked_incomplete["blockers"]
            )
        )
        self.assertIsNone(blocked_incomplete["delta_R"])

        first = _candle(
            0,
            bid=_ohlc(100.0, 101.0, 99.5),
            ask=_ohlc(100.1, 101.1, 99.6),
        )
        conflicting = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        blocked_conflict = _score(state, (first, conflicting))
        self.assertEqual(blocked_conflict["status"], "BLOCKED")
        self.assertTrue(
            any(
                "CONFLICTING_S5_CANDLE" in item for item in blocked_conflict["blockers"]
            )
        )
        self.assertIsNone(blocked_conflict["delta_R"])
        self.assert_audit_only(blocked_conflict)

        malformed = S5BidAskCandle(
            timestamp_utc=_dt(0),
            pair="USD_JPY",
            bid=None,  # type: ignore[arg-type]
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        blocked_malformed = _score(state, (malformed,))
        self.assertEqual(blocked_malformed["status"], "BLOCKED")
        self.assertTrue(
            any("INVALID_BID_OHLC" in item for item in blocked_malformed["blockers"])
        )

    def test_delta_r_uses_positive_risk_and_applies_costs_and_financing_once(
        self,
    ) -> None:
        state = _state()
        tp = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        scored = _score(
            state,
            (tp,),
            gpt_exit_price=100.0,
            gpt_exit_non_spread_cost_jpy=2.0,
            gpt_financing_cost_jpy=1.0,
        )

        self.assertEqual(scored["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        # Bid-to-bid LONG comparison: ask spread is not subtracted again.
        self.assertEqual(scored["gpt_arm"]["gross_incremental_jpy"], 0.0)
        self.assertEqual(scored["gpt_arm"]["net_incremental_jpy"], -3.0)
        self.assertEqual(scored["frozen_control_arm"]["gross_incremental_jpy"], 40.0)
        self.assertEqual(scored["frozen_control_arm"]["net_incremental_jpy"], 30.0)
        self.assertEqual(scored["frozen_control_arm"]["financing_application_count"], 1)
        self.assertEqual(scored["decision_risk"]["total_jpy"], 32.0)
        self.assertGreater(scored["decision_risk"]["total_jpy"], 0.0)
        self.assertEqual(scored["delta_jpy"], -33.0)
        self.assertAlmostEqual(scored["delta_R"], -33.0 / 32.0)
        self.assertEqual(
            scored["spread_handling"],
            "INTRINSIC_BID_ASK_PRICES_NO_EXTRA_SPREAD_CHARGE",
        )
        self.assert_audit_only(scored)

    def test_gpt_execution_evidence_is_explicit_and_fill_window_is_exact(self) -> None:
        state = _state()
        tp = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        fill_window = (tp,) + tuple(
            _candle(
                second,
                bid=_ohlc(100.0, 101.0, 99.5),
                ask=_ohlc(100.1, 101.1, 99.6),
            )
            for second in (5, 10, 15, 20)
        )

        implicit = score_loss_close_paired_shadow(state, (tp,))

        self.assertEqual(implicit["status"], "BLOCKED")
        self.assertIn("MISSING_OR_INVALID_GPT_EXIT_PRICE", implicit["blockers"])
        self.assertIn("MISSING_OR_INVALID_GPT_NON_SPREAD_COST", implicit["blockers"])
        self.assertIn("MISSING_OR_INVALID_GPT_FINANCING_COST", implicit["blockers"])
        self.assertIn(
            "MISSING_OR_INVALID_GPT_EXECUTION_EVIDENCE_SHA256",
            implicit["blockers"],
        )
        self.assertIn("MISSING_OR_INVALID_GPT_FILL_TIMESTAMP", implicit["blockers"])
        self.assertIsNone(implicit["delta_R"])
        self.assert_audit_only(implicit)

        at_limit = _score(
            state,
            fill_window,
            gpt_fill_timestamp_utc="2026-07-14T00:00:20Z",
        )
        self.assertEqual(at_limit["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(at_limit["gpt_arm"]["fill_delay_nanoseconds"], 20_000_000_000)
        self.assertEqual(
            at_limit["gpt_arm"]["max_fill_delay_seconds"],
            MAX_PAIRED_SHADOW_FILL_DELAY_SECONDS,
        )

        after_limit = _score(
            state,
            fill_window,
            gpt_fill_timestamp_utc="2026-07-14T00:00:20.000000001Z",
        )
        self.assertEqual(after_limit["status"], "BLOCKED")
        self.assertIn("GPT_FILL_AFTER_MAX_DELAY", after_limit["blockers"])
        self.assertIsNone(after_limit["delta_R"])

        before_decision = _score(
            state,
            fill_window,
            gpt_fill_timestamp_utc="2026-07-13T23:59:59.999999999Z",
        )
        self.assertEqual(before_decision["status"], "BLOCKED")
        self.assertIn("GPT_FILL_BEFORE_DECISION", before_decision["blockers"])

        noncanonical_fill = _score(
            state,
            (tp,),
            gpt_fill_timestamp_utc="2026-07-14T00:00:00.0Z",
        )
        self.assertEqual(noncanonical_fill["status"], "BLOCKED")
        self.assertIn("NON_CANONICAL_GPT_FILL_TIMESTAMP", noncanonical_fill["blockers"])

    def test_fill_price_must_match_executable_s5_and_remains_unverified(self) -> None:
        state = _state()
        tp = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )

        outside = _score(state, (tp,), gpt_exit_price=110.0)

        self.assertEqual(outside["status"], "BLOCKED")
        self.assertIn("GPT_FILL_PRICE_OUTSIDE_S5_EXECUTABLE_RANGE", outside["blockers"])
        self.assertIsNone(outside["delta_R"])
        self.assert_audit_only(outside)

        missing_fill_candle = _score(
            state,
            (tp,),
            gpt_fill_timestamp_utc="2026-07-14T00:00:05Z",
        )
        self.assertEqual(missing_fill_candle["status"], "BLOCKED")
        self.assertIn("S5_GPT_FILL_CANDLE_MISSING", missing_fill_candle["blockers"])

        calculated = _score(state, (tp,))
        self.assertEqual(
            calculated["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS"
        )
        self.assertEqual(
            calculated["gpt_arm"]["exit_price_source"],
            "CALLER_SUPPLIED_UNVERIFIED_EXECUTION_FIELDS",
        )
        self.assertFalse(
            calculated["gpt_arm"]["s5_executable_range_evidence"][
                "authoritative_truth_verified_by_evaluator"
            ]
        )
        self.assertFalse(
            calculated["gpt_execution_evidence_digest_verified_by_evaluator"]
        )
        self.assertFalse(calculated["s5_truth_verified_by_evaluator"])
        self.assert_audit_only(calculated)

    def test_fill_binding_reuses_the_frozen_validated_s5_sequence(self) -> None:
        safe = _candle(
            0,
            bid=_ohlc(100.0, 102.1, 99.5),
            ask=_ohlc(100.1, 102.2, 99.6),
        )
        substituted = _candle(
            0,
            bid=_ohlc(100.0, 200.0, 99.5),
            ask=_ohlc(100.1, 200.1, 99.6),
        )
        changing = _ChangingCandleSequence(safe, substituted)

        result = score_loss_close_paired_shadow(
            _state(),
            changing,
            gpt_exit_price=150.0,
            gpt_exit_non_spread_cost_jpy=0.0,
            gpt_financing_cost_jpy=0.0,
            gpt_execution_evidence_sha256="e" * 64,
            gpt_fill_timestamp_utc="2026-07-14T00:00:00Z",
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("GPT_FILL_PRICE_OUTSIDE_S5_EXECUTABLE_RANGE", result["blockers"])
        self.assertEqual(changing.iterations, 1)
        self.assertIsNone(result["delta_R"])
        self.assert_audit_only(result)

    def test_non_positive_or_non_finite_risk_cannot_be_scored(self) -> None:
        invalid = dict(_state())
        invalid["stop_loss"] = invalid["executable_close_price"]
        invalid["state_sha256"] = paired_shadow_state_sha256(invalid)

        validation = validate_paired_shadow_state(invalid)
        self.assertFalse(validation["valid"])
        self.assertIn("INVALID_LONG_PROTECTION_GEOMETRY", validation["issues"])
        scored = score_loss_close_paired_shadow(
            invalid,
            (
                _candle(
                    0,
                    bid=_ohlc(100.0, 102.1, 99.5),
                    ask=_ohlc(100.1, 102.2, 99.6),
                ),
            ),
        )
        self.assertEqual(scored["status"], "BLOCKED")
        self.assertIsNone(scored["delta_R"])

        non_finite = dict(_state())
        non_finite["quote_to_jpy"] = float("inf")
        with self.assertRaises(ValueError):
            paired_shadow_state_sha256(non_finite)


if __name__ == "__main__":
    unittest.main()
