from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.strategy.forecast_technical_context import (
    build_forecast_technical_context,
)
from quant_rabbit.strategy.regime_family_contradiction_shadow import (
    EVALUATION_HORIZON_MINUTES,
    LEDGER_FILENAME,
    MAX_EMISSION_AGE_SECONDS,
    MAX_EMISSION_FUTURE_SKEW_SECONDS,
    M1BidAskCandle,
    SHADOW_CONTRACT,
    TERMINAL_EVALUATION_CONTRACT,
    bind_regime_family_contradiction_emission,
    build_regime_family_contradiction_shadow,
    load_regime_family_contradiction_ledger,
    persist_regime_family_contradiction_emission as _persist_emission,
    persist_regime_family_contradiction_results,
    resolve_due_regime_family_contradiction_ledger,
    resolve_due_regime_family_contradiction_trials,
    resolve_regime_family_contradiction_trial,
    seal_regime_family_contradiction_holdout_lock,
    select_independent_regime_family_contradiction_trials,
    validate_regime_family_contradiction_result,
    validate_regime_family_contradiction_result_binding,
    validate_regime_family_contradiction_shadow,
    validate_regime_family_contradiction_trial,
    verify_regime_family_contradiction_source_binding,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


_LEDGER_CLOCK = (
    "quant_rabbit.strategy.regime_family_contradiction_shadow."
    "_ledger_recorded_at_utc"
)


def _rewrite_ledger(root: Path, events: list[dict]) -> None:
    (root / LEDGER_FILENAME).write_text(
        "".join(
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _reseal_event(event: dict) -> None:
    event["event_sha256"] = _sha(
        {key: item for key, item in event.items() if key != "event_sha256"}
    )


def _trial_pair_material(trial: dict) -> dict:
    return {
        "contract": SHADOW_CONTRACT,
        "pair": trial["pair"],
        "entry_reference_price": trial["entry_reference_price"],
        "entry_bid": trial["entry_bid"],
        "entry_ask": trial["entry_ask"],
        "detector_arm": trial["detector_arm"],
        "family_arm": trial["family_arm"],
        "detector_scores": trial["detector_scores"],
        "technical_context_sha256": trial["technical_context_sha256"],
        "regime_family_weighting_receipt_sha256": trial[
            "regime_family_weighting_receipt_sha256"
        ],
        "weighted_directional_score": trial["weighted_directional_score"],
        "directional_coverage_weight": trial["directional_coverage_weight"],
        "selected_family_coverage_weight": trial["selected_family_coverage_weight"],
        "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
        "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
    }


def _reseal_trial(trial: dict, *, pair_identity: bool = False) -> None:
    if pair_identity:
        trial["pair_id"] = _sha(_trial_pair_material(trial))
        trial["trial_id"] = _sha(
            {
                "contract": trial["contract"],
                "pair_id": trial["pair_id"],
                "pair": trial["pair"],
                "emitted_at_utc": trial["emitted_at_utc"],
                "evaluation_due_at_utc": trial["evaluation_due_at_utc"],
            }
        )
    trial["trial_sha256"] = _sha(
        {key: item for key, item in trial.items() if key != "trial_sha256"}
    )


def _reseal_shadow(shadow: dict, *, pair_identity: bool = False) -> None:
    if pair_identity:
        shadow["pair_id"] = _sha(_trial_pair_material(shadow))
    shadow["shadow_sha256"] = _sha(
        {key: item for key, item in shadow.items() if key != "shadow_sha256"}
    )


def _chart(*, pair: str = "EUR_USD", transition: bool = False) -> dict:
    del pair
    return {
        "session": {"current_tag": "LONDON"},
        "confluence": {
            "dominant_regime": "TREND_UP",
            "atr_percentile_24h": 0.82,
            "price_percentile_24h": 0.88,
            "price_percentile_7d": 0.42,
        },
        "views": [
            {
                "granularity": "M5",
                "regime_reading": {
                    "state": "TREND_STRONG",
                    "atr_percentile": 80.0,
                },
                "indicators": {"atr_pips": 2.0},
                "family_scores": {
                    "trend_score": -1.0,
                    "mean_rev_score": 0.0,
                    "breakout_score": 0.0,
                    "disagreement": 0.0,
                },
                "structure": {
                    "structure_events": [
                        {"kind": "BOS_UP", "index": 10, "close_confirmed": True}
                    ]
                },
            },
            {
                "granularity": "M15",
                "regime_reading": {
                    "state": "TRANSITION" if transition else "TREND_WEAK",
                    "atr_percentile": 60.0,
                },
                "indicators": {"atr_pips": 5.0},
                "family_scores": {
                    "trend_score": -1.0,
                    "mean_rev_score": 0.0,
                    "breakout_score": 0.0,
                    "disagreement": 0.0,
                },
                "structure": {
                    "structure_events": [
                        {
                            "kind": "CHOCH_DOWN",
                            "index": 15,
                            "close_confirmed": True,
                            "timestamp": "2026-07-13T01:00:00Z",
                        }
                    ]
                },
            },
            {
                "granularity": "H1",
                "regime_reading": {"state": "RANGE", "atr_percentile": 20.0},
                "indicators": {"atr_pips": 14.0},
                "family_scores": {
                    "trend_score": 0.0,
                    "mean_rev_score": -1.0,
                    "breakout_score": 0.0,
                    "disagreement": 0.0,
                },
                "structure": {
                    "structure_events": [
                        {
                            "kind": "BOS_DOWN",
                            "index": 18,
                            "close_confirmed": True,
                        }
                    ]
                },
            },
        ],
    }


def _context(
    *,
    pair: str = "EUR_USD",
    price: float = 1.1,
    transition: bool = False,
) -> dict:
    return build_forecast_technical_context(
        _chart(pair=pair, transition=transition),
        pair=pair,
        current_price=price,
        spread_pips=0.2,
    )


_TEST_CONTEXT_BY_SHA: dict[str, dict] = {}


def _shadow(
    *,
    pair: str = "EUR_USD",
    price: float = 1.1,
    up_score: float = 100.0,
    transition: bool = False,
) -> dict:
    pip_factor = 100.0 if pair.endswith("_JPY") else 10_000.0
    half_spread = 0.2 / pip_factor / 2.0
    context = _context(
        pair=pair,
        price=price,
        transition=transition,
    )
    shadow = build_regime_family_contradiction_shadow(
        pair=pair,
        current_price=price,
        detector_direction="UP",
        detector_scores={
            "UP": up_score,
            "DOWN": 10.0,
            "RANGE": 5.0,
            "EITHER": 2.0,
        },
        technical_context_v1=context,
        entry_bid=price - half_spread,
        entry_ask=price + half_spread,
    )
    _TEST_CONTEXT_BY_SHA[str(context["context_sha256"])] = context
    return shadow


def persist_regime_family_contradiction_emission(
    shadow: dict | None,
    **kwargs: object,
) -> int:
    """Keep test call sites explicit about the now-mandatory source binding."""

    context = kwargs.pop("technical_context_v1", None)
    if context is None:
        context = (
            _TEST_CONTEXT_BY_SHA.get(str(shadow.get("technical_context_sha256")))
            if shadow is not None
            else _context()
        )
    if context is None:
        raise AssertionError("test shadow was not paired with its source context")
    return _persist_emission(
        shadow,
        technical_context_v1=context,
        **kwargs,
    )


class RegimeFamilyContradictionShadowTest(unittest.TestCase):
    def test_reconstructed_quote_is_diagnostic_only_and_not_persisted(self) -> None:
        context = _context()
        diagnostic = build_regime_family_contradiction_shadow(
            pair="EUR_USD",
            current_price=1.1,
            detector_direction="UP",
            detector_scores={"UP": 100.0, "DOWN": 10.0, "RANGE": 5.0, "EITHER": 2.0},
            technical_context_v1=context,
        )
        emitted = datetime.now(timezone.utc) + timedelta(minutes=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                persist_regime_family_contradiction_emission(
                    diagnostic,
                    technical_context_v1=context,
                    emitted_at_utc=emitted,
                    cycle_id="diagnostic-only",
                    data_root=root,
                ),
                0,
            )
            self.assertFalse((root / LEDGER_FILENAME).exists())
            malformed = copy.deepcopy(diagnostic)
            malformed["entry_bid"] -= 0.0001
            with self.assertRaisesRegex(ValueError, "INVALID_CONTRADICTION_SHADOW"):
                persist_regime_family_contradiction_emission(
                    malformed,
                    technical_context_v1=context,
                    emitted_at_utc=emitted,
                    cycle_id="malformed",
                    data_root=root,
                )

    def test_builder_freezes_paired_arms_pre_veto_scores_and_permissions(self) -> None:
        context = _context()
        shadow = build_regime_family_contradiction_shadow(
            pair="EUR_USD",
            current_price=1.1,
            detector_direction="UP",
            detector_scores={"UP": 100.0, "DOWN": 10.0, "RANGE": 5.0, "EITHER": 2.0},
            technical_context_v1=context,
            entry_bid=1.09999,
            entry_ask=1.10001,
        )

        self.assertEqual(shadow["contract"], SHADOW_CONTRACT)
        self.assertEqual(shadow["detector_arm"]["direction"], "UP")
        self.assertEqual(shadow["family_arm"]["direction"], "DOWN")
        self.assertEqual(shadow["detector_scores"]["UP"], 100.0)
        self.assertEqual(shadow["evaluation_contract"], TERMINAL_EVALUATION_CONTRACT)
        self.assertEqual(shadow["evaluation_horizon_minutes"], 60)
        self.assertEqual(shadow["entry_spread_pips"], 0.2)
        self.assertTrue(shadow["read_only"])
        self.assertTrue(shadow["shadow_only"])
        self.assertFalse(shadow["live_permission"])
        self.assertFalse(shadow["sizing_permission"])
        self.assertFalse(shadow["gate_relaxation_allowed"])
        self.assertFalse(shadow["automatic_promotion_allowed"])
        self.assertEqual(validate_regime_family_contradiction_shadow(shadow), ())
        self.assertEqual(
            shadow,
            build_regime_family_contradiction_shadow(
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
            ),
        )
        self.assertEqual(
            verify_regime_family_contradiction_source_binding(
                shadow,
                technical_context_v1=context,
            ),
            (True, None),
        )

        tampered = copy.deepcopy(shadow)
        tampered["detector_scores"]["UP"] = 101.0
        self.assertIn(
            "SHADOW_SHA256_MISMATCH",
            validate_regime_family_contradiction_shadow(tampered),
        )

    def test_builder_rejects_non_contradiction_and_non_winning_score(self) -> None:
        context = _context()
        aligned = copy.deepcopy(context)
        receipt = aligned["regime_family_weighting"]
        for row in receipt["by_timeframe"].values():
            if row["selected_direction"] == "DOWN":
                row["selected_direction"] = "UP"
                row["selected_score"] = abs(row["selected_score"])
                row["weighted_directional_score"] = abs(
                    row["weighted_directional_score"]
                )
        # A hand-edited receipt cannot be made valid merely for this builder.
        with self.assertRaises(ValueError):
            build_regime_family_contradiction_shadow(
                pair="EUR_USD",
                current_price=1.1,
                detector_direction="UP",
                detector_scores={"UP": 100.0, "DOWN": 0.0, "RANGE": 0.0, "EITHER": 0.0},
                technical_context_v1=aligned,
            )
        with self.assertRaisesRegex(ValueError, "DETECTOR_DIRECTION_NOT_SCORE_WINNER"):
            build_regime_family_contradiction_shadow(
                pair="EUR_USD",
                current_price=1.1,
                detector_direction="UP",
                detector_scores={"UP": 5.0, "DOWN": 10.0, "RANGE": 0.0, "EITHER": 0.0},
                technical_context_v1=context,
            )

    def test_source_binding_rederives_context_fields_and_entry_economics(self) -> None:
        shadow = _shadow()
        context = _TEST_CONTEXT_BY_SHA[shadow["technical_context_sha256"]]
        self.assertEqual(
            verify_regime_family_contradiction_source_binding(
                shadow,
                technical_context_v1=context,
            ),
            (True, None),
        )

        for field, forged_value in (
            ("dominant_regime", "FORGED_REGIME"),
            ("situation_label", "FORGED_SITUATION"),
            ("selected_method", "FORGED_METHOD"),
        ):
            forged = copy.deepcopy(shadow)
            forged[field] = forged_value
            _reseal_shadow(forged)
            self.assertEqual(validate_regime_family_contradiction_shadow(forged), ())
            valid, error = verify_regime_family_contradiction_source_binding(
                forged,
                technical_context_v1=context,
            )
            self.assertFalse(valid)
            self.assertIn(field.upper(), str(error))

        forged_score = copy.deepcopy(shadow)
        forged_score["weighted_directional_score"] += 0.1
        _reseal_shadow(forged_score, pair_identity=True)
        self.assertEqual(
            validate_regime_family_contradiction_shadow(forged_score),
            (),
        )
        score_valid, score_error = verify_regime_family_contradiction_source_binding(
            forged_score,
            technical_context_v1=context,
        )
        self.assertFalse(score_valid)
        self.assertIn("WEIGHTED_DIRECTIONAL_SCORE", str(score_error))

        forged_spread = copy.deepcopy(shadow)
        forged_spread["entry_bid"] = 1.09995
        forged_spread["entry_ask"] = 1.10005
        forged_spread["entry_spread_pips"] = 1.0
        _reseal_shadow(forged_spread, pair_identity=True)
        self.assertEqual(
            validate_regime_family_contradiction_shadow(forged_spread),
            (),
        )
        spread_valid, spread_error = verify_regime_family_contradiction_source_binding(
            forged_spread,
            technical_context_v1=context,
        )
        self.assertFalse(spread_valid)
        self.assertIn("ENTRY_SPREAD_PIPS", str(spread_error))

        forged_mid = copy.deepcopy(shadow)
        forged_mid["entry_reference_price"] = 1.100005
        forged_mid["entry_bid"] = 1.099995
        forged_mid["entry_ask"] = 1.100015
        _reseal_shadow(forged_mid, pair_identity=True)
        self.assertEqual(validate_regime_family_contradiction_shadow(forged_mid), ())
        mid_valid, mid_error = verify_regime_family_contradiction_source_binding(
            forged_mid,
            technical_context_v1=context,
        )
        self.assertFalse(mid_valid)
        self.assertIn("ENTRY_REFERENCE_PRICE", str(mid_error))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T00:00:31Z")):
                with self.assertRaisesRegex(
                    ValueError,
                    "INVALID_CONTRADICTION_SHADOW_SOURCE_BINDING",
                ):
                    persist_regime_family_contradiction_emission(
                        forged_spread,
                        technical_context_v1=context,
                        emitted_at_utc="2026-07-15T00:00:30Z",
                        cycle_id="forged-source",
                        data_root=root,
                    )
            self.assertFalse((root / LEDGER_FILENAME).exists())

            with self.assertRaisesRegex(TypeError, "technical_context_v1"):
                _persist_emission(
                    shadow,
                    emitted_at_utc="2026-07-15T00:00:30Z",
                    cycle_id="missing-source",
                    data_root=root,
                )

    def test_arm_sources_are_fixed_in_self_rehashed_shadow_and_trial(self) -> None:
        shadow = _shadow()
        forged_shadow = copy.deepcopy(shadow)
        forged_shadow["detector_arm"]["source"] = "FORGED"
        forged_shadow["family_arm"]["source"] = "FORGED"
        _reseal_shadow(forged_shadow, pair_identity=True)
        self.assertIn(
            "SHADOW_FIELD_INVALID",
            validate_regime_family_contradiction_shadow(forged_shadow),
        )

        trial = bind_regime_family_contradiction_emission(
            shadow,
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="fixed-arm-source",
        )
        forged_trial = copy.deepcopy(trial)
        forged_trial["detector_arm"]["source"] = "FORGED"
        forged_trial["family_arm"]["source"] = "FORGED"
        _reseal_trial(forged_trial, pair_identity=True)
        self.assertIn(
            "TRIAL_FIELD_INVALID",
            validate_regime_family_contradiction_trial(forged_trial),
        )

    def test_transition_or_method_none_remains_shadow_only_and_excluded(self) -> None:
        shadow = _shadow(transition=True)

        self.assertTrue(shadow["transition_or_method_none"])
        self.assertIsNone(shadow["selected_method"])
        self.assertEqual(
            shadow["promotion_exclusion_reason"],
            "PRIMARY_REGIME_TRANSITION_OR_METHOD_NONE",
        )
        self.assertFalse(shadow["live_permission"])
        self.assertFalse(shadow["automatic_promotion_allowed"])

    def test_holdout_binding_and_non_overlap_are_pre_outcome_and_pair_scoped(
        self,
    ) -> None:
        lock = seal_regime_family_contradiction_holdout_lock(
            holdout_id="july-forward-v1",
            locked_at_utc="2026-07-14T23:00:00Z",
            holdout_start_utc="2026-07-15T00:00:00Z",
            holdout_end_utc="2026-07-20T00:00:00Z",
            source_prefix_sha256="1" * 64,
            source_row_count=100,
            selection_policy_sha256="2" * 64,
        )
        base = datetime(2026, 7, 15, tzinfo=timezone.utc)
        trials = [
            bind_regime_family_contradiction_emission(
                _shadow(up_score=100.0 + index),
                emitted_at_utc=base + timedelta(minutes=minute),
                cycle_id=f"cycle-{index}",
                holdout_lock=lock,
            )
            for index, minute in enumerate((0, 30, 60))
        ]
        trials.append(
            bind_regime_family_contradiction_emission(
                _shadow(pair="GBP_USD", price=1.3),
                emitted_at_utc=base + timedelta(minutes=30),
                cycle_id="gbp-cycle",
                holdout_lock=lock,
            )
        )

        selection = select_independent_regime_family_contradiction_trials(
            trials,
            as_of_utc="2026-07-15T01:00:00Z",
            require_locked_holdout=True,
        )

        self.assertEqual(selection["selected_count"], 3)
        self.assertEqual(selection["skipped_overlapping_count"], 1)
        self.assertFalse(selection["outcome_fields_read_by_selector"])
        self.assertEqual(
            [trial["cycle_id"] for trial in selection["selected_trials"]],
            ["cycle-0", "gbp-cycle", "cycle-2"],
        )

    def test_fixed_terminal_result_includes_round_trip_spread_and_cost(self) -> None:
        trial = bind_regime_family_contradiction_emission(
            _shadow(),
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="terminal-math",
        )

        result = resolve_regime_family_contradiction_trial(
            trial,
            terminal_bid=1.10095,
            terminal_ask=1.10105,
            terminal_interval_start_utc="2026-07-15T01:00:00Z",
            terminal_observed_at_utc="2026-07-15T01:01:00Z",
            resolved_as_of_utc="2026-07-15T01:02:00Z",
            terminal_source_sha256="3" * 64,
            non_spread_cost_pips=0.1,
        )

        self.assertEqual(result["terminal_interval_start_utc"], "2026-07-15T01:00:00Z")
        self.assertEqual(result["terminal_interval_end_utc"], "2026-07-15T01:01:00Z")
        self.assertAlmostEqual(result["terminal_observation_lag_seconds"], 30.0)
        self.assertAlmostEqual(
            result["detector_result"]["round_trip_spread_cost_pips"], 0.6
        )
        self.assertAlmostEqual(result["detector_result"]["post_cost_pips"], 9.3)
        self.assertAlmostEqual(result["family_result"]["post_cost_pips"], -10.7)
        self.assertEqual(result["post_cost_winner"], "DETECTOR")
        self.assertTrue(result["round_trip_spread_included"])
        self.assertFalse(result["live_permission"])

        with self.assertRaisesRegex(
            ValueError, "TERMINAL_INTERVAL_MUST_BE_EXACT_COMPLETE_M1"
        ):
            resolve_regime_family_contradiction_trial(
                trial,
                terminal_bid=1.10095,
                terminal_ask=1.10105,
                terminal_interval_start_utc="2026-07-15T01:00:30Z",
                terminal_observed_at_utc="2026-07-15T01:01:00Z",
                resolved_as_of_utc="2026-07-15T01:02:00Z",
                terminal_source_sha256="3" * 64,
            )

    def test_self_consistent_swapped_arms_cannot_bind_to_current_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted_at = _dt("2026-07-15T00:00:30Z")
            due_at = emitted_at + timedelta(minutes=60)
            terminal_start = due_at.replace(second=0, microsecond=0)
            terminal_end = terminal_start + timedelta(minutes=1)
            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T00:00:31Z")):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=_utc(emitted_at),
                    cycle_id="binding-cycle",
                    data_root=root,
                )
            trial = load_regime_family_contradiction_ledger(root)["trials"][0]
            result = resolve_regime_family_contradiction_trial(
                trial,
                terminal_bid=1.10095,
                terminal_ask=1.10105,
                terminal_interval_start_utc=_utc(terminal_start),
                terminal_observed_at_utc=_utc(terminal_end),
                resolved_as_of_utc=_utc(terminal_end + timedelta(minutes=1)),
                terminal_source_sha256="8" * 64,
            )
            forged = copy.deepcopy(result)
            original_detector = copy.deepcopy(result["detector_result"])
            original_family = copy.deepcopy(result["family_result"])
            forged["detector_result"] = {
                **original_family,
                "arm_id": "DETECTOR",
            }
            forged["family_result"] = {
                **original_detector,
                "arm_id": "REGIME_FAMILY",
            }
            forged["post_cost_winner"] = "REGIME_FAMILY"
            forged["result_sha256"] = _sha(
                {key: item for key, item in forged.items() if key != "result_sha256"}
            )

            # The forgery is arithmetically and cryptographically consistent
            # with itself; only the immutable trial binding exposes the swap.
            self.assertEqual(validate_regime_family_contradiction_result(forged), ())
            binding_issues = validate_regime_family_contradiction_result_binding(
                forged,
                trial,
            )
            self.assertIn("RESULT_TRIAL_DETECTOR_ARM_MISMATCH", binding_issues)
            self.assertIn("RESULT_TRIAL_FAMILY_ARM_MISMATCH", binding_issues)
            with self.assertRaisesRegex(
                ValueError,
                "INVALID_CONTRADICTION_RESULT_TRIAL_BINDING",
            ):
                persist_regime_family_contradiction_results([forged], data_root=root)
            loaded = load_regime_family_contradiction_ledger(root)
            self.assertEqual(len(loaded["events"]), 1)
            self.assertEqual(loaded["results"], [])

    def test_duplicate_m1_interval_is_order_invariant_and_conflict_fails(self) -> None:
        trial = bind_regime_family_contradiction_emission(
            _shadow(),
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="duplicate-m1",
        )
        first = {
            "instrument": "EUR_USD",
            "granularity": "M1",
            "time": "2026-07-15T01:00:00.000000000Z",
            "complete": True,
            "bid": {"c": "1.10095"},
            "ask": {"c": "1.10105"},
            "source_page": "A",
        }
        economically_identical = {
            **first,
            "source_page": "B",
        }
        forward = resolve_due_regime_family_contradiction_trials(
            [trial],
            [first, economically_identical],
            as_of_utc="2026-07-15T01:02:00Z",
        )
        reversed_order = resolve_due_regime_family_contradiction_trials(
            [trial],
            [economically_identical, first],
            as_of_utc="2026-07-15T01:02:00Z",
        )
        self.assertEqual(
            forward["resolved_results"], reversed_order["resolved_results"]
        )
        self.assertEqual(forward["resolved_count"], 1)

        conflict = copy.deepcopy(economically_identical)
        conflict["bid"] = {"c": "1.09895"}
        with self.assertRaisesRegex(ValueError, "DUPLICATE_INTERVAL_CONFLICT"):
            resolve_due_regime_family_contradiction_trials(
                [trial],
                [first, conflict],
                as_of_utc="2026-07-15T01:02:00Z",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted_at = _dt("2026-07-15T00:00:30Z")
            due_at = emitted_at + timedelta(minutes=60)
            terminal_start = due_at.replace(second=0, microsecond=0)
            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T00:00:31Z")):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=_utc(emitted_at),
                    cycle_id="duplicate-ledger",
                    data_root=root,
                )
            dynamic_first = {
                **first,
                "time": _utc(terminal_start),
            }
            dynamic_conflict = {
                **conflict,
                "time": _utc(terminal_start),
            }
            with self.assertRaisesRegex(ValueError, "DUPLICATE_INTERVAL_CONFLICT"):
                resolve_due_regime_family_contradiction_ledger(
                    data_root=root,
                    candles=[dynamic_first, dynamic_conflict],
                    as_of_utc=_utc(terminal_start + timedelta(minutes=2)),
                )
            loaded = load_regime_family_contradiction_ledger(root)
            self.assertEqual(len(loaded["events"]), 1)
            self.assertEqual(loaded["results"], [])

    def test_due_resolver_uses_first_complete_m1_close_not_later_price(self) -> None:
        trial = bind_regime_family_contradiction_emission(
            _shadow(),
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="first-terminal",
        )
        first = {
            "instrument": "EUR_USD",
            "granularity": "M1",
            "time": "2026-07-15T01:00:00Z",
            "complete": True,
            "bid": {"c": "1.10095"},
            "ask": {"c": "1.10105"},
        }
        later = {
            "instrument": "EUR_USD",
            "granularity": "M1",
            "time": "2026-07-15T01:01:00Z",
            "complete": True,
            "bid": {"c": "1.09895"},
            "ask": {"c": "1.09905"},
        }
        incomplete = {
            **first,
            "complete": False,
            "bid": {"c": "1.2"},
            "ask": {"c": "1.2001"},
        }

        resolved = resolve_due_regime_family_contradiction_trials(
            [trial],
            [later, incomplete, first],
            as_of_utc="2026-07-15T01:02:00Z",
        )

        self.assertEqual(resolved["resolved_count"], 1)
        result = resolved["resolved_results"][0]
        self.assertEqual(result["terminal_interval_start_utc"], first["time"])
        self.assertEqual(result["terminal_bid"], 1.10095)
        self.assertEqual(result["terminal_source_sha256"], _sha(first))

    def test_hash_chained_ledger_is_idempotent_replaceable_then_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted_at = _dt("2026-07-15T00:00:30Z")
            emitted = _utc(emitted_at)
            due_at = emitted_at + timedelta(minutes=60)
            terminal_start = due_at.replace(second=0, microsecond=0)
            terminal_end = terminal_start + timedelta(minutes=1)
            resolved_as_of = terminal_end + timedelta(minutes=1)
            first = _shadow(up_score=100.0)
            replacement = _shadow(up_score=101.0)

            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T00:00:31Z")):
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        None,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle",
                        data_root=root,
                    ),
                    0,
                )
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        first,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle",
                        data_root=root,
                    ),
                    1,
                )
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        first,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle-retry",
                        data_root=root,
                    ),
                    0,
                )
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        first,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle",
                        data_root=root,
                    ),
                    0,
                )
                with self.assertRaisesRegex(ValueError, "IDENTITY_COLLISION"):
                    persist_regime_family_contradiction_emission(
                        replacement,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle",
                        data_root=root,
                    )
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        replacement,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle",
                        data_root=root,
                        replace_existing=True,
                    ),
                    1,
                )
                # A→B replacement permanently tombstones A.  A retry from a new
                # cycle must not resurrect the old observation as another sample.
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        first,
                        emitted_at_utc=emitted,
                        cycle_id="ledger-cycle-after-replace",
                        data_root=root,
                    ),
                    0,
                )
            loaded = load_regime_family_contradiction_ledger(root)
            self.assertEqual(loaded["status"], "VALID")
            self.assertEqual(len(loaded["events"]), 2)
            self.assertEqual(len(loaded["trials"]), 1)
            self.assertEqual(len(loaded["historical_trial_ids"]), 2)
            self.assertEqual(loaded["trials"][0]["detector_scores"]["UP"], 101.0)
            selected = select_independent_regime_family_contradiction_trials(
                loaded["trials"],
                ledger_recorded_at_by_trial_id=loaded["ledger_recorded_at_by_trial_id"],
            )
            self.assertEqual(selected["recording_anchor_missing_count"], 0)
            self.assertFalse(selected["proof_eligible"])
            result = resolve_regime_family_contradiction_trial(
                loaded["trials"][0],
                terminal_bid=1.10095,
                terminal_ask=1.10105,
                terminal_interval_start_utc=_utc(terminal_start),
                terminal_observed_at_utc=_utc(terminal_end),
                resolved_as_of_utc=_utc(resolved_as_of),
                terminal_source_sha256="4" * 64,
            )
            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T01:02:01Z")):
                self.assertEqual(
                    persist_regime_family_contradiction_results(
                        [result], data_root=root
                    ),
                    1,
                )
            self.assertEqual(
                persist_regime_family_contradiction_results([result], data_root=root),
                0,
            )
            self.assertEqual(
                persist_regime_family_contradiction_emission(
                    first,
                    emitted_at_utc=emitted,
                    cycle_id="ledger-cycle",
                    data_root=root,
                    replace_existing=True,
                ),
                0,
            )
            with self.assertRaisesRegex(ValueError, "CANNOT_BE_REPLACED"):
                persist_regime_family_contradiction_emission(
                    _shadow(up_score=102.0),
                    emitted_at_utc=emitted,
                    cycle_id="ledger-cycle",
                    data_root=root,
                    replace_existing=True,
                )
            final = load_regime_family_contradiction_ledger(root)
            self.assertEqual(len(final["events"]), 3)
            self.assertEqual(len(final["results"]), 1)
            self.assertTrue((root / LEDGER_FILENAME).exists())

            rehashed_tamper = copy.deepcopy(result)
            rehashed_tamper["detector_result"]["post_cost_pips"] += 1.0
            rehashed_tamper["result_sha256"] = _sha(
                {
                    key: item
                    for key, item in rehashed_tamper.items()
                    if key != "result_sha256"
                }
            )
            self.assertIn(
                "RESULT_DETECTOR_POST_COST_PIPS_MISMATCH",
                validate_regime_family_contradiction_result(rehashed_tamper),
            )

    def test_ledger_due_resolver_closes_learning_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted_at = _dt("2026-07-15T00:00:30Z")
            due_at = emitted_at + timedelta(minutes=60)
            terminal_start = due_at.replace(second=0, microsecond=0)
            terminal_end = terminal_start + timedelta(minutes=1)
            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T00:00:31Z")):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=_utc(emitted_at),
                    cycle_id="due-ledger",
                    data_root=root,
                )
            candle = M1BidAskCandle(
                pair="EUR_USD",
                timestamp_utc=_utc(terminal_start),
                bid_close=1.10095,
                ask_close=1.10105,
                complete=True,
            )

            with patch(_LEDGER_CLOCK, return_value=_dt("2026-07-15T01:03:01Z")):
                first = resolve_due_regime_family_contradiction_ledger(
                    data_root=root,
                    candles=[candle],
                    as_of_utc=_utc(terminal_end + timedelta(minutes=1)),
                )
                second = resolve_due_regime_family_contradiction_ledger(
                    data_root=root,
                    candles=[candle],
                    as_of_utc=_utc(terminal_end + timedelta(minutes=2)),
                )

            self.assertEqual(first["resolved_count"], 1)
            self.assertEqual(first["persisted_count"], 1)
            self.assertEqual(second["resolved_count"], 0)
            self.assertEqual(second["persisted_count"], 0)

    def test_persistence_clock_rejects_future_stale_naive_and_regression(self) -> None:
        emitted = _dt("2026-07-15T00:00:30Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            too_early_wall = emitted - timedelta(
                seconds=MAX_EMISSION_FUTURE_SKEW_SECONDS + 1
            )
            with patch(_LEDGER_CLOCK, return_value=too_early_wall):
                with self.assertRaisesRegex(
                    ValueError,
                    "EMISSION_TIMESTAMP_EXCEEDS_LEDGER_FUTURE_SKEW",
                ):
                    persist_regime_family_contradiction_emission(
                        _shadow(),
                        emitted_at_utc=emitted,
                        cycle_id="future-emission",
                        data_root=root,
                    )
            self.assertFalse((root / LEDGER_FILENAME).exists())

            stale_wall = emitted + timedelta(seconds=MAX_EMISSION_AGE_SECONDS + 1)
            with patch(_LEDGER_CLOCK, return_value=stale_wall):
                with self.assertRaisesRegex(
                    ValueError,
                    "EMISSION_TIMESTAMP_EXCEEDS_LEDGER_MAX_AGE",
                ):
                    persist_regime_family_contradiction_emission(
                        _shadow(),
                        emitted_at_utc=emitted,
                        cycle_id="stale-emission",
                        data_root=root,
                    )
            self.assertFalse((root / LEDGER_FILENAME).exists())

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            bind_regime_family_contradiction_emission(
                _shadow(),
                emitted_at_utc=datetime(2026, 7, 15, 0, 0, 30),
                cycle_id="naive-datetime",
            )
        with self.assertRaisesRegex(ValueError, "emitted_at_utc invalid"):
            bind_regime_family_contradiction_emission(
                _shadow(),
                emitted_at_utc="2026-07-15T00:00:30",
                cycle_id="naive-string",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="clock-regression",
                    data_root=root,
                )
            with patch(_LEDGER_CLOCK, return_value=emitted):
                with self.assertRaisesRegex(
                    ValueError,
                    "LEDGER_RECORDED_AT_REGRESSION",
                ):
                    persist_regime_family_contradiction_emission(
                        _shadow(up_score=101.0),
                        emitted_at_utc=emitted,
                        cycle_id="clock-regression",
                        data_root=root,
                        replace_existing=True,
                    )
            self.assertEqual(
                len(load_regime_family_contradiction_ledger(root)["events"]),
                1,
            )

    def test_result_event_requires_terminal_and_resolution_clocks(self) -> None:
        emitted = _dt("2026-07-15T00:00:30Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="result-clock",
                    data_root=root,
                )
            trial = load_regime_family_contradiction_ledger(root)["trials"][0]
            result = resolve_regime_family_contradiction_trial(
                trial,
                terminal_bid=1.10095,
                terminal_ask=1.10105,
                terminal_interval_start_utc="2026-07-15T01:00:00Z",
                terminal_observed_at_utc="2026-07-15T01:01:00Z",
                resolved_as_of_utc="2026-07-15T01:02:00Z",
                terminal_source_sha256="9" * 64,
            )
            for premature_wall in (
                _dt("2026-07-15T01:00:59Z"),
                _dt("2026-07-15T01:01:30Z"),
            ):
                with self.subTest(premature_wall=premature_wall):
                    with patch(_LEDGER_CLOCK, return_value=premature_wall):
                        with self.assertRaisesRegex(
                            ValueError,
                            "RESULT_RECORDED_BEFORE_RESOLUTION_CLOCK",
                        ):
                            persist_regime_family_contradiction_results(
                                [result],
                                data_root=root,
                            )
            self.assertEqual(
                len(load_regime_family_contradiction_ledger(root)["events"]),
                1,
            )
            with patch(
                _LEDGER_CLOCK,
                return_value=_dt("2026-07-15T01:02:00Z"),
            ):
                self.assertEqual(
                    persist_regime_family_contradiction_results(
                        [result],
                        data_root=root,
                    ),
                    1,
                )

    def test_replay_rejects_rehashed_clock_tampering(self) -> None:
        emitted = _dt("2026-07-15T00:00:30Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="replay-future",
                    data_root=root,
                )
            events = load_regime_family_contradiction_ledger(root)["events"]
            events[0]["recorded_at_utc"] = _utc(
                emitted - timedelta(seconds=MAX_EMISSION_FUTURE_SKEW_SECONDS + 1)
            )
            _reseal_event(events[0])
            _rewrite_ledger(root, events)
            with self.assertRaisesRegex(ValueError, "EMISSION_FUTURE_SKEW"):
                load_regime_family_contradiction_ledger(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="replay-stale",
                    data_root=root,
                )
            events = load_regime_family_contradiction_ledger(root)["events"]
            events[0]["recorded_at_utc"] = _utc(
                emitted + timedelta(seconds=MAX_EMISSION_AGE_SECONDS + 1)
            )
            _reseal_event(events[0])
            _rewrite_ledger(root, events)
            with self.assertRaisesRegex(ValueError, "EMISSION_STALE"):
                load_regime_family_contradiction_ledger(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="replay-result",
                    data_root=root,
                )
            trial = load_regime_family_contradiction_ledger(root)["trials"][0]
            result = resolve_regime_family_contradiction_trial(
                trial,
                terminal_bid=1.10095,
                terminal_ask=1.10105,
                terminal_interval_start_utc="2026-07-15T01:00:00Z",
                terminal_observed_at_utc="2026-07-15T01:01:00Z",
                resolved_as_of_utc="2026-07-15T01:02:00Z",
                terminal_source_sha256="a" * 64,
            )
            with patch(
                _LEDGER_CLOCK,
                return_value=_dt("2026-07-15T01:02:01Z"),
            ):
                persist_regime_family_contradiction_results([result], data_root=root)
            events = load_regime_family_contradiction_ledger(root)["events"]
            events[1]["recorded_at_utc"] = "2026-07-15T01:01:30Z"
            _reseal_event(events[1])
            _rewrite_ledger(root, events)
            with self.assertRaisesRegex(ValueError, "RESULT_PREMATURE"):
                load_regime_family_contradiction_ledger(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=emitted,
                    cycle_id="replay-regression",
                    data_root=root,
                )
            with patch(
                _LEDGER_CLOCK,
                return_value=emitted + timedelta(seconds=2),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(up_score=101.0),
                    emitted_at_utc=emitted,
                    cycle_id="replay-regression",
                    data_root=root,
                    replace_existing=True,
                )
            events = load_regime_family_contradiction_ledger(root)["events"]
            events[1]["recorded_at_utc"] = _utc(emitted)
            _reseal_event(events[1])
            _rewrite_ledger(root, events)
            with self.assertRaisesRegex(ValueError, "RECORDED_AT_REGRESSION"):
                load_regime_family_contradiction_ledger(root)

    def test_replacement_must_precede_superseded_due_on_append_and_replay(self) -> None:
        first_emitted = _dt("2026-07-15T00:00:30Z")
        replacement_emitted = _dt("2026-07-15T01:00:15Z")
        superseded_due = _dt("2026-07-15T01:00:30Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                _LEDGER_CLOCK,
                return_value=first_emitted + timedelta(seconds=1),
            ):
                persist_regime_family_contradiction_emission(
                    _shadow(),
                    emitted_at_utc=first_emitted,
                    cycle_id="replace-before-old-due",
                    data_root=root,
                )

            with patch(_LEDGER_CLOCK, return_value=superseded_due):
                with self.assertRaisesRegex(
                    ValueError,
                    "REPLACEMENT_RECORDED_AT_OR_AFTER_SUPERSEDED_OUTCOME_WINDOW",
                ):
                    persist_regime_family_contradiction_emission(
                        _shadow(up_score=101.0),
                        emitted_at_utc=replacement_emitted,
                        cycle_id="replace-before-old-due",
                        data_root=root,
                        replace_existing=True,
                    )

            with patch(
                _LEDGER_CLOCK,
                return_value=_dt("2026-07-15T01:00:20Z"),
            ):
                self.assertEqual(
                    persist_regime_family_contradiction_emission(
                        _shadow(up_score=101.0),
                        emitted_at_utc=replacement_emitted,
                        cycle_id="replace-before-old-due",
                        data_root=root,
                        replace_existing=True,
                    ),
                    1,
                )
            events = load_regime_family_contradiction_ledger(root)["events"]
            events[1]["recorded_at_utc"] = _utc(superseded_due)
            _reseal_event(events[1])
            _rewrite_ledger(root, events)
            with self.assertRaisesRegex(
                ValueError,
                "OUTCOME_AWARE_REPLACEMENT",
            ):
                load_regime_family_contradiction_ledger(root)

    def test_pip_midpoint_and_pair_identity_survive_self_rehash_tampering(self) -> None:
        jpy_shadow = _shadow(pair="USD_JPY", price=150.0)
        self.assertEqual(validate_regime_family_contradiction_shadow(jpy_shadow), ())

        wrong_shadow_factor = copy.deepcopy(jpy_shadow)
        wrong_shadow_factor["pip_factor"] = 10_000.0
        wrong_shadow_factor["entry_spread_pips"] = round(
            (wrong_shadow_factor["entry_ask"] - wrong_shadow_factor["entry_bid"])
            * wrong_shadow_factor["pip_factor"],
            9,
        )
        wrong_shadow_factor["shadow_sha256"] = _sha(
            {
                key: item
                for key, item in wrong_shadow_factor.items()
                if key != "shadow_sha256"
            }
        )
        self.assertIn(
            "SHADOW_PIP_FACTOR_MISMATCH",
            validate_regime_family_contradiction_shadow(wrong_shadow_factor),
        )

        jpy_trial = bind_regime_family_contradiction_emission(
            jpy_shadow,
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="jpy-pip-factor",
        )
        wrong_trial_factor = copy.deepcopy(jpy_trial)
        wrong_trial_factor["pip_factor"] = 10_000.0
        wrong_trial_factor["entry_spread_pips"] = round(
            (wrong_trial_factor["entry_ask"] - wrong_trial_factor["entry_bid"])
            * wrong_trial_factor["pip_factor"],
            9,
        )
        _reseal_trial(wrong_trial_factor)
        self.assertIn(
            "TRIAL_PIP_FACTOR_MISMATCH",
            validate_regime_family_contradiction_trial(wrong_trial_factor),
        )

        trial = bind_regime_family_contradiction_emission(
            _shadow(),
            emitted_at_utc="2026-07-15T00:00:30Z",
            cycle_id="pair-material",
        )
        changed_material = copy.deepcopy(trial)
        changed_material["weighted_directional_score"] += 0.1
        _reseal_trial(changed_material)
        changed_material_issues = validate_regime_family_contradiction_trial(
            changed_material
        )
        self.assertIn("TRIAL_PAIR_ID_MISMATCH", changed_material_issues)
        self.assertNotIn("TRIAL_SHA256_MISMATCH", changed_material_issues)

        wrong_reference = copy.deepcopy(trial)
        wrong_reference["entry_reference_price"] += 0.00001
        _reseal_trial(wrong_reference, pair_identity=True)
        reference_issues = validate_regime_family_contradiction_trial(wrong_reference)
        self.assertIn("TRIAL_ENTRY_REFERENCE_MISMATCH", reference_issues)
        self.assertNotIn("TRIAL_PAIR_ID_MISMATCH", reference_issues)
        self.assertNotIn("TRIAL_ID_MISMATCH", reference_issues)
        self.assertNotIn("TRIAL_SHA256_MISMATCH", reference_issues)

    def test_trial_digest_and_due_clock_fail_closed_on_tamper(self) -> None:
        trial = bind_regime_family_contradiction_emission(
            _shadow(),
            emitted_at_utc="2026-07-15T00:00:00Z",
            cycle_id="tamper",
        )
        self.assertEqual(
            trial["evaluation_horizon_minutes"], EVALUATION_HORIZON_MINUTES
        )
        self.assertEqual(validate_regime_family_contradiction_trial(trial), ())

        tampered = copy.deepcopy(trial)
        tampered["evaluation_due_at_utc"] = "2026-07-15T02:00:00Z"
        self.assertIn(
            "TRIAL_DUE_AT_MISMATCH",
            validate_regime_family_contradiction_trial(tampered),
        )


if __name__ == "__main__":
    unittest.main()
