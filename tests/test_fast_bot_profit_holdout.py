from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot import HORIZON_LANE, SHADOW_CONTRACT, SIGNAL_CONTRACT
from quant_rabbit.fast_bot_profit_holdout import (
    POLICY_CONTRACT,
    POLICY_CONTRACT_V2,
    POLICY_CONTRACT_V3,
    SELECTION_POLICY,
    SIGNAL_FILTER_POLICY_V2,
    V2_MINIMUM_M5_ATR_PIPS,
    append_decision_once,
    build_holdout_decision,
    build_holdout_scorecard,
    canonical_sha,
    load_policy,
    run_selection,
    seal,
)
from quant_rabbit.fast_bot_profitability_gate import DEFAULT_THRESHOLDS
from quant_rabbit.fast_bot_truth import (
    build_fast_bot_scorecard,
    resolve_fast_bot_signal,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _policy() -> tuple[dict, str]:
    lanes = [
        {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "RANGE_ROTATION",
            "horizon_lane": HORIZON_LANE,
            "priority": 100,
            "candidate_status": "UNPROVEN_PROSPECTIVE_CANDIDATE",
        }
    ]
    policy = {
        "contract": POLICY_CONTRACT,
        "schema_version": 1,
        "policy_id": "test-prospective-lane-v1",
        "holdout": {
            "frozen_at_utc": CUTOFF.isoformat(),
            "eligible_after_utc": CUTOFF.isoformat(),
            "cohort_policy": "STRICTLY_AFTER_ELIGIBLE_AFTER_UTC",
            "retroactive_signal_admission_allowed": False,
        },
        "selection": {
            "selection_policy": SELECTION_POLICY,
            "maximum_selected_per_cycle": 1,
            "maximum_concurrent_per_pair_horizon": 1,
            "reservation_seconds": 990,
            "maximum_selection_delay_seconds": 45,
            "unknown_lane_policy": "REJECT",
            "equal_priority_policy": "REJECT_ALL_TIED_TOP_PRIORITY",
            "opposite_side_policy": "REJECT_CYCLE_ON_SAME_PAIR_HORIZON_OPPOSITE_GO",
            "post_outcome_reranking_allowed": False,
            "allowed_lanes": lanes,
        },
        "training_evidence": {
            "generated_at_utc": CUTOFF.isoformat(),
            "source_scorecard_contract_sha256": "a" * 64,
            "source_scorecard_file_sha256": "b" * 64,
            "forward_evidence_passed": False,
            "profitability_claim": "UNPROVEN",
        },
        "acceptance_thresholds": dict(DEFAULT_THRESHOLDS),
        "authority": {
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation_allowed": False,
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "manual_tagless_policy": "NO_TOUCH",
        },
    }
    return policy, canonical_sha(policy)


def _v2_policy() -> tuple[dict, str]:
    policy, _ = _policy()
    policy.update(
        contract=POLICY_CONTRACT_V2,
        schema_version=2,
        policy_id="test-prospective-lane-v2",
        supersession={
            "supersedes_policy_id": "usdjpy-short-range-rotation-prospective-v1",
            "prior_policy_status": "RETIRED_ZERO_ELIGIBLE_SELECTIONS",
            "prior_rows_admitted": False,
            "replacement_reason": "ZERO_ELIGIBLE_SELECTIONS_AND_NEGATIVE_POST_CUTOFF_DIAGNOSTIC_LANE",
            "single_factor_changed": "M5_ATR_PIPS_MINIMUM",
            "reward_risk_changed": False,
            "prior_policy_decisions_screened": 1,
            "prior_policy_selected_signals": 0,
            "prior_policy_resolved_signals": 0,
            "prior_policy_last_decision_sha256": "d" * 64,
            "prior_policy_observed_at_utc": CUTOFF.isoformat(),
            "prior_policy_source_bundle_sha256": "e" * 64,
            "prior_policy_source_commit": "f" * 40,
            "post_cutoff_diagnostic_lane_filled_signals": 1,
            "post_cutoff_diagnostic_lane_net_pips": -1.0,
            "post_cutoff_diagnostic_lane_profit_factor": 0.5,
            "post_cutoff_diagnostic_rows_admitted": False,
        },
    )
    policy["selection"]["signal_filter"] = {
        "filter_policy": SIGNAL_FILTER_POLICY_V2,
        "entry_reference": "PASSIVE_NEAR_SIDE",
        "m5_atr_pips_operator": "GREATER_THAN_OR_EQUAL",
        "m5_atr_pips_minimum": V2_MINIMUM_M5_ATR_PIPS,
        "units": "PIPS",
        "missing_or_invalid_policy": "REJECT",
        "threshold_role": "POST_HOC_HYPOTHESIS_ONLY",
        "historical_rows_admitted": False,
    }
    return policy, canonical_sha(policy)


def _signal(
    name: str,
    *,
    generated: datetime,
    side: str = "LONG",
    method: str = "RANGE_ROTATION",
    m5_atr_pips: float | None = None,
) -> dict:
    if side == "LONG":
        entry, target, stop = 1.1000, 1.1003, 1.0997
    else:
        entry, target, stop = 1.1001, 1.0998, 1.1004
    body = {
        "contract": SIGNAL_CONTRACT,
        "schema_version": 1,
        "signal_id": hashlib.sha256(name.encode()).hexdigest()[:24],
        "pair": "EUR_USD",
        "side": side,
        "method": method,
        "horizon_lane": HORIZON_LANE,
        "m1_closed_candle_utc": (generated - timedelta(minutes=1)).isoformat(),
        "regime_contract_sha256": "c" * 64,
        "generated_at_utc": generated.isoformat(),
        "quote_timestamp_utc": generated.isoformat(),
        "order_type": "LIMIT",
        "entry_reference": "PASSIVE_NEAR_SIDE",
        "entry": entry,
        "take_profit": target,
        "stop_loss": stop,
        "take_profit_pips": 3.0,
        "stop_loss_pips": 3.0,
        "reward_risk": 1.0,
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "attached_take_profit_required": True,
        "attached_stop_loss_required": True,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    if m5_atr_pips is not None:
        body["m5_atr_pips"] = m5_atr_pips
    return {**body, "signal_sha256": canonical_sha(body)}


def _shadow(*signals: dict) -> dict:
    generated = max(
        (signal["generated_at_utc"] for signal in signals),
        default=NOW.isoformat(),
    )
    return seal(
        {
            "contract": SHADOW_CONTRACT,
            "schema_version": 1,
            "generated_at_utc": generated,
            "status": "EMITTED" if signals else "NO_GO_SIGNAL",
            "signals": list(signals),
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
    )


def _candle(
    at: datetime,
    *,
    bid_h: float,
    bid_l: float,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=at,
        bid_o=1.0999,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=1.0999,
        ask_o=1.1000,
        ask_h=max(1.1001, bid_h + 0.0002),
        ask_l=min(1.1000, bid_l + 0.0002),
        ask_c=1.1000,
    )


def _outcome(signal: dict, *, win: bool) -> dict:
    generated = datetime.fromisoformat(signal["generated_at_utc"])
    candles = [
        _candle(
            generated + timedelta(seconds=5),
            bid_h=1.0999,
            bid_l=1.0998,
        ),
        _candle(
            generated + timedelta(seconds=10),
            bid_h=1.1004 if win else 1.1000,
            bid_l=1.0998 if win else 1.0996,
        ),
    ]
    return resolve_fast_bot_signal(
        signal,
        candles,
        resolved_at_utc=generated + timedelta(minutes=20),
        truth_chunk_sha256=["d" * 64],
    )


class FastBotProfitHoldoutTest(unittest.TestCase):
    def test_repository_policy_is_valid_and_explicitly_unproven(self) -> None:
        policy, policy_sha = load_policy(
            Path(__file__).resolve().parents[1]
            / "config"
            / "fast_bot_profit_holdout_v1.json"
        )

        self.assertEqual(len(policy_sha), 64)
        self.assertEqual(policy["selection"]["maximum_selected_per_cycle"], 1)
        self.assertEqual(
            policy["selection"]["allowed_lanes"][0]["candidate_status"],
            "UNPROVEN_PROSPECTIVE_CANDIDATE",
        )
        self.assertFalse(policy["training_evidence"]["forward_evidence_passed"])
        self.assertEqual(policy["authority"]["execution_authority"], "NONE")
        self.assertFalse(policy["authority"]["live_permission"])

    def test_repository_v2_policy_retires_v1_and_starts_unproven(self) -> None:
        policy, policy_sha = load_policy(
            Path(__file__).resolve().parents[1]
            / "config"
            / "fast_bot_profit_holdout_v2.json"
        )

        self.assertEqual(len(policy_sha), 64)
        self.assertEqual(policy["contract"], POLICY_CONTRACT_V2)
        self.assertEqual(
            policy["supersession"]["prior_policy_status"],
            "RETIRED_ZERO_ELIGIBLE_SELECTIONS",
        )
        self.assertFalse(policy["supersession"]["prior_rows_admitted"])
        self.assertEqual(policy["supersession"]["prior_policy_selected_signals"], 0)
        self.assertEqual(
            policy["supersession"]["post_cutoff_diagnostic_lane_net_pips"],
            -51.4,
        )
        self.assertEqual(
            policy["selection"]["signal_filter"]["m5_atr_pips_minimum"],
            5.0,
        )
        self.assertEqual(
            policy["training_evidence"]["selection_use"],
            "POST_HOC_HYPOTHESIS_ONLY_FUTURE_ROWS_REQUIRED",
        )
        self.assertFalse(policy["training_evidence"]["forward_evidence_passed"])

    def test_repository_v3_policy_retires_v2_without_inventing_a_candidate(self) -> None:
        policy, policy_sha = load_policy(
            Path(__file__).resolve().parents[1]
            / "config"
            / "fast_bot_profit_holdout_v3.json"
        )

        self.assertEqual(len(policy_sha), 64)
        self.assertEqual(policy["contract"], POLICY_CONTRACT_V3)
        self.assertEqual(policy["selection"]["allowed_lanes"], [])
        self.assertEqual(policy["selection"]["maximum_selected_per_cycle"], 0)
        self.assertEqual(policy["candidate_admission"]["research_lead_count"], 0)
        self.assertEqual(
            policy["candidate_admission"]["v2_candidate_reassessment"][
                "filled_signals"
            ],
            3,
        )
        self.assertFalse(
            policy["candidate_admission"]["v2_candidate_reassessment"][
                "admission_passed"
            ]
        )
        self.assertEqual(
            policy["training_evidence"]["profitability_claim"],
            "NO_ADMISSIBLE_CANDIDATE",
        )
        self.assertEqual(policy["authority"]["execution_authority"], "NONE")

    def test_v3_rejects_every_row_and_scorecard_states_no_candidate(self) -> None:
        policy, policy_sha = load_policy(
            Path(__file__).resolve().parents[1]
            / "config"
            / "fast_bot_profit_holdout_v3.json"
        )
        cutoff = datetime.fromisoformat(policy["holdout"]["eligible_after_utc"])
        generated = cutoff + timedelta(days=1)
        signal = _signal("v3-no-candidate", generated=generated)
        decision = build_holdout_decision(
            _shadow(signal),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        truth = build_fast_bot_scorecard(
            [],
            [],
            as_of_utc=generated + timedelta(seconds=1),
        )

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[decision],
            raw_signals=[signal],
            selected_signals=[],
            outcomes=[],
            truth_scorecard=truth,
            now_utc=generated + timedelta(seconds=1),
        )

        self.assertEqual(decision["status"], "NO_ACTIVE_PROFIT_CANDIDATE")
        self.assertEqual(decision["selected_signal_count"], 0)
        self.assertIn(
            "NO_ADMISSIBLE_CANDIDATE",
            decision["selection_rows"][0]["reasons"],
        )
        self.assertEqual(
            decision["contract"],
            "QR_FAST_BOT_PROFIT_HOLDOUT_DECISION_V3",
        )
        self.assertTrue(scorecard["cohort_integrity_passed"])
        self.assertEqual(scorecard["status"], "NO_ADMISSIBLE_PROFIT_CANDIDATE")
        self.assertEqual(scorecard["candidate_status"], "NO_ADMISSIBLE_CANDIDATE")
        self.assertIsNone(scorecard["profitability_evidence"])
        self.assertIsNone(scorecard["profitability_gate"])
        self.assertEqual(scorecard["execution_authority"], "NONE")
        self.assertFalse(scorecard["live_permission"])

    def test_v2_selects_only_at_or_above_precommitted_atr_floor(self) -> None:
        policy, policy_sha = _v2_policy()
        generated = CUTOFF + timedelta(days=1)
        below = build_holdout_decision(
            _shadow(_signal("below", generated=generated, m5_atr_pips=4.999999)),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        at_floor = build_holdout_decision(
            _shadow(_signal("at-floor", generated=generated, m5_atr_pips=5.0)),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )

        self.assertEqual(below["selected_signal_count"], 0)
        self.assertIn(
            "M5_ATR_BELOW_PRECOMMITTED_MINIMUM",
            below["selection_rows"][0]["reasons"],
        )
        self.assertEqual(at_floor["status"], "SELECTED_PROSPECTIVE_HOLDOUT")
        self.assertEqual(at_floor["contract"], "QR_FAST_BOT_PROFIT_HOLDOUT_DECISION_V2")
        self.assertEqual(
            at_floor["signal_filter"],
            policy["selection"]["signal_filter"],
        )

    def test_v2_rejects_missing_atr_while_v1_remains_readable(self) -> None:
        generated = CUTOFF + timedelta(days=1)
        signal = _signal("missing-atr", generated=generated)
        v1, v1_sha = _policy()
        v2, v2_sha = _v2_policy()

        v1_decision = build_holdout_decision(
            _shadow(signal),
            policy=v1,
            policy_sha256=v1_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        v2_decision = build_holdout_decision(
            _shadow(signal),
            policy=v2,
            policy_sha256=v2_sha,
            now_utc=generated + timedelta(seconds=1),
        )

        self.assertEqual(v1_decision["status"], "SELECTED_PROSPECTIVE_HOLDOUT")
        self.assertEqual(v2_decision["selected_signal_count"], 0)
        self.assertIn(
            "M5_ATR_INVALID_FOR_PRECOMMITTED_FILTER",
            v2_decision["selection_rows"][0]["reasons"],
        )

    def test_selects_only_precommitted_future_lane_without_mutating_raw_shadow(self) -> None:
        policy, policy_sha = _policy()
        selected = _signal("selected", generated=CUTOFF + timedelta(days=1))
        excluded = _signal(
            "excluded",
            generated=CUTOFF + timedelta(days=1),
            method="TREND_CONTINUATION",
        )
        raw = _shadow(selected, excluded)
        before = copy.deepcopy(raw)

        decision = build_holdout_decision(
            raw,
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=CUTOFF + timedelta(days=1, seconds=1),
        )

        self.assertEqual(decision["status"], "SELECTED_PROSPECTIVE_HOLDOUT")
        self.assertEqual(decision["selected_signal_sha256s"], [selected["signal_sha256"]])
        self.assertEqual(raw, before)
        self.assertEqual(decision["execution_authority"], "NONE")
        self.assertFalse(decision["broker_mutation_allowed"])
        self.assertFalse(decision["live_permission"])
        self.assertEqual(decision["external_orders"], 0)

    def test_simultaneous_opposite_side_go_blocks_the_whole_pair_horizon(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        decision = build_holdout_decision(
            _shadow(
                _signal("allowed-long", generated=generated, side="LONG"),
                _signal("opposite-short", generated=generated, side="SHORT"),
            ),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )

        self.assertEqual(decision["status"], "BLOCKED_OPPOSITE_SIDE_AMBIGUITY")
        self.assertEqual(decision["selected_signal_count"], 0)
        self.assertTrue(
            all(
                "OPPOSITE_SIDE_GO_AMBIGUITY" in row["reasons"]
                for row in decision["selection_rows"]
            )
        )

    def test_signal_selected_after_forty_five_seconds_is_rejected(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        decision = build_holdout_decision(
            _shadow(_signal("late", generated=generated)),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=46),
        )

        self.assertEqual(decision["selected_signal_count"], 0)
        self.assertIn(
            "SIGNAL_SELECTION_WINDOW_EXPIRED",
            decision["selection_rows"][0]["reasons"],
        )

    def test_pre_policy_and_overlapping_signals_are_rejected(self) -> None:
        policy, policy_sha = _policy()
        prior = _signal("prior", generated=CUTOFF + timedelta(days=1))
        overlap = _signal(
            "overlap",
            generated=CUTOFF + timedelta(days=1, minutes=5),
        )
        old = _signal("old", generated=CUTOFF)

        overlap_decision = build_holdout_decision(
            _shadow(overlap),
            policy=policy,
            policy_sha256=policy_sha,
            selected_history=[prior],
            now_utc=CUTOFF + timedelta(days=1, minutes=5, seconds=1),
        )
        old_decision = build_holdout_decision(
            _shadow(old),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=CUTOFF + timedelta(seconds=1),
        )

        self.assertEqual(overlap_decision["status"], "NO_SELECTION_LANE_RESERVED")
        self.assertIn(
            "PAIR_HORIZON_RESERVED_BY_PRIOR_SELECTION",
            overlap_decision["selection_rows"][0]["reasons"],
        )
        self.assertEqual(old_decision["selected_signal_count"], 0)
        self.assertIn(
            "IN_SAMPLE_OR_PRE_POLICY_SIGNAL",
            old_decision["selection_rows"][0]["reasons"],
        )

    def test_duplicate_top_lane_candidates_fail_closed(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        decision = build_holdout_decision(
            _shadow(
                _signal("first", generated=generated),
                _signal("second", generated=generated),
            ),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )

        self.assertEqual(decision["status"], "BLOCKED_AMBIGUOUS_TOP_PRIORITY")
        self.assertEqual(decision["selected_signal_count"], 0)
        self.assertTrue(
            all(
                row["reasons"] == ["AMBIGUOUS_TOP_PRIORITY"]
                for row in decision["selection_rows"]
            )
        )

    def test_tampered_source_shadow_fails_closed(self) -> None:
        policy, policy_sha = _policy()
        raw = _shadow(_signal("one", generated=CUTOFF + timedelta(days=1)))
        raw["signals"][0]["take_profit_pips"] = 99.0

        decision = build_holdout_decision(
            raw,
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=CUTOFF + timedelta(days=1, seconds=1),
        )

        self.assertEqual(decision["status"], "BLOCKED_SOURCE_INTEGRITY")
        self.assertEqual(decision["selected_signal_count"], 0)

    def test_run_selection_writes_only_separate_holdout_ledgers(self) -> None:
        policy, _ = _policy()
        signal = _signal("persist", generated=CUTOFF + timedelta(days=1))
        raw = _shadow(signal)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "raw.json"
            raw_ledger_path = root / "raw.jsonl"
            policy_path = root / "policy.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_ledger_path.write_text(
                json.dumps(signal, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            raw_before = raw_path.read_bytes()

            result = run_selection(
                raw_shadow_path=raw_path,
                raw_signal_ledger_path=raw_ledger_path,
                policy_path=policy_path,
                selected_ledger_path=root / "selected.jsonl",
                decision_ledger_path=root / "decisions.jsonl",
                output_path=root / "selection.json",
                report_path=root / "selection.md",
                now_utc=CUTOFF + timedelta(days=1, seconds=1),
            )
            second = run_selection(
                raw_shadow_path=raw_path,
                raw_signal_ledger_path=raw_ledger_path,
                policy_path=policy_path,
                selected_ledger_path=root / "selected.jsonl",
                decision_ledger_path=root / "decisions.jsonl",
                output_path=root / "selection.json",
                report_path=root / "selection.md",
                now_utc=CUTOFF + timedelta(days=1, seconds=1),
            )

            self.assertEqual(result["selected_signals_appended"], 1)
            self.assertEqual(result["raw_cycles_screened"], 1)
            self.assertEqual(second["selected_signals_appended"], 0)
            self.assertEqual(len((root / "selected.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((root / "decisions.jsonl").read_text().splitlines()), 1)
            self.assertEqual(raw_path.read_bytes(), raw_before)

    def test_append_only_raw_backlog_is_screened_before_current_cycle(self) -> None:
        policy, _ = _policy()
        old = _signal("old-cycle", generated=CUTOFF + timedelta(days=1))
        current = _signal("current-cycle", generated=CUTOFF + timedelta(days=2))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "raw.json"
            raw_ledger_path = root / "raw.jsonl"
            policy_path = root / "policy.json"
            raw_path.write_text(json.dumps(_shadow(current)), encoding="utf-8")
            raw_ledger_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in (old, current))
                + "\n",
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(policy), encoding="utf-8")

            result = run_selection(
                raw_shadow_path=raw_path,
                raw_signal_ledger_path=raw_ledger_path,
                policy_path=policy_path,
                selected_ledger_path=root / "selected.jsonl",
                decision_ledger_path=root / "decisions.jsonl",
                output_path=root / "selection.json",
                report_path=root / "selection.md",
                now_utc=CUTOFF + timedelta(days=2, seconds=1),
            )
            decisions = [
                json.loads(line)
                for line in (root / "decisions.jsonl").read_text().splitlines()
            ]

            self.assertEqual(result["raw_cycles_screened"], 2)
            self.assertEqual(result["raw_cycle_backlog_remaining"], 0)
            self.assertEqual(len(decisions), 2)
            self.assertIn(
                "SIGNAL_SELECTION_WINDOW_EXPIRED",
                decisions[0]["selection_rows"][0]["reasons"],
            )
            self.assertEqual(decisions[1]["status"], "SELECTED_PROSPECTIVE_HOLDOUT")
            self.assertEqual(len((root / "selected.jsonl").read_text().splitlines()), 1)

    def test_restart_repairs_decision_first_crash_without_duplicate_binding(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        signal = _signal("crash-recovery", generated=generated)
        raw = _shadow(signal)
        first_decision = build_holdout_decision(
            raw,
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "raw.json"
            raw_ledger_path = root / "raw.jsonl"
            policy_path = root / "policy.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_ledger_path.write_text(
                json.dumps(signal, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            append_decision_once(root / "decisions.jsonl", first_decision)

            recovered = run_selection(
                raw_shadow_path=raw_path,
                raw_signal_ledger_path=raw_ledger_path,
                policy_path=policy_path,
                selected_ledger_path=root / "selected.jsonl",
                decision_ledger_path=root / "decisions.jsonl",
                output_path=root / "selection.json",
                report_path=root / "selection.md",
                now_utc=generated + timedelta(seconds=2),
            )

            self.assertEqual(recovered["decision_appended"], 0)
            self.assertEqual(recovered["selected_signals_appended"], 1)
            self.assertEqual(len((root / "decisions.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((root / "selected.jsonl").read_text().splitlines()), 1)

    def test_same_timestamp_cycles_use_one_canonical_replay_order(self) -> None:
        policy, _ = _policy()
        generated = CUTOFF + timedelta(days=1)
        first_regime = _signal("same-time-first", generated=generated)
        second_regime = _signal("same-time-second", generated=generated)
        second_body = {
            key: value
            for key, value in second_regime.items()
            if key != "signal_sha256"
        }
        second_body["regime_contract_sha256"] = "e" * 64
        second_regime = {
            **second_body,
            "signal_sha256": canonical_sha(second_body),
        }
        raw_signals = [first_regime, second_regime]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "raw.json"
            raw_ledger_path = root / "raw.jsonl"
            policy_path = root / "policy.json"
            raw_path.write_text(json.dumps(_shadow(second_regime)), encoding="utf-8")
            raw_ledger_path.write_text(
                "\n".join(
                    json.dumps(row, sort_keys=True) for row in raw_signals
                )
                + "\n",
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(policy), encoding="utf-8")

            run_selection(
                raw_shadow_path=raw_path,
                raw_signal_ledger_path=raw_ledger_path,
                policy_path=policy_path,
                selected_ledger_path=root / "selected.jsonl",
                decision_ledger_path=root / "decisions.jsonl",
                output_path=root / "selection.json",
                report_path=root / "selection.md",
                now_utc=generated + timedelta(seconds=1),
            )
            selected = [
                json.loads(line)
                for line in (root / "selected.jsonl").read_text().splitlines()
            ]
            decisions = [
                json.loads(line)
                for line in (root / "decisions.jsonl").read_text().splitlines()
            ]
            truth = build_fast_bot_scorecard(selected, [], as_of_utc=NOW)
            scorecard = build_holdout_scorecard(
                policy=policy,
                policy_sha256=canonical_sha(policy),
                decisions=decisions,
                raw_signals=raw_signals,
                selected_signals=selected,
                outcomes=[],
                truth_scorecard=truth,
                now_utc=NOW,
            )

        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(selected), 1)
        self.assertTrue(scorecard["cohort_integrity_passed"])
        self.assertNotIn(
            "DECISION_SELECTION_SEMANTICS_MISMATCH",
            scorecard["cohort_integrity_errors"],
        )

    def test_positive_100_fill_holdout_passes_shadow_gate_but_never_live(self) -> None:
        policy, policy_sha = _policy()
        signals: list[dict] = []
        outcomes: list[dict] = []
        decisions: list[dict] = []
        for day in range(10):
            for seat in range(10):
                generated = CUTOFF + timedelta(
                    days=day + 1,
                    minutes=seat * 20,
                )
                signal = _signal(
                    f"forward-{day}-{seat}",
                    generated=generated,
                )
                decision = build_holdout_decision(
                    _shadow(signal),
                    policy=policy,
                    policy_sha256=policy_sha,
                    selected_history=signals,
                    now_utc=generated + timedelta(seconds=1),
                )
                self.assertEqual(
                    decision["status"],
                    "SELECTED_PROSPECTIVE_HOLDOUT",
                )
                signals.append(signal)
                decisions.append(decision)
                outcomes.append(_outcome(signal, win=seat < 8))
        truth = build_fast_bot_scorecard(signals, outcomes, as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=decisions,
            raw_signals=signals,
            selected_signals=signals,
            outcomes=outcomes,
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertTrue(truth["forward_evidence_passed"])
        self.assertEqual(scorecard["status"], "SHADOW_PROFITABILITY_EVIDENCE_PASSED")
        self.assertTrue(scorecard["cohort_integrity_passed"])
        self.assertEqual(scorecard["truth_metrics"]["filled_signals"], 100)
        self.assertEqual(scorecard["truth_metrics"]["active_days"], 10)
        self.assertEqual(scorecard["truth_metrics"]["profit_factor"], 4.0)
        self.assertEqual(scorecard["execution_authority"], "NONE")
        self.assertFalse(scorecard["automatic_adoption_allowed"])
        self.assertFalse(scorecard["promotion_allowed"])
        self.assertFalse(scorecard["live_permission"])
        self.assertEqual(scorecard["external_order_attempts"], 0)
        self.assertEqual(scorecard["external_orders"], 0)
        self.assertEqual(scorecard["gateway_invocations"], 0)

        thin_truth = build_fast_bot_scorecard(
            signals[:99],
            outcomes[:99],
            as_of_utc=NOW,
        )
        thin = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=decisions[:99],
            raw_signals=signals[:99],
            selected_signals=signals[:99],
            outcomes=outcomes[:99],
            truth_scorecard=thin_truth,
            now_utc=NOW,
        )
        self.assertEqual(thin["status"], "COLLECT_MORE_INDEPENDENT_DAYS")
        self.assertIn("INSUFFICIENT_SAMPLES", thin["blockers"])
        self.assertFalse(thin["live_permission"])

    def test_tampered_selected_cohort_is_rejected_not_scored_as_zero(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        signal = _signal("tamper", generated=generated)
        decision = build_holdout_decision(
            _shadow(signal),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        outcome = _outcome(signal, win=True)
        truth = build_fast_bot_scorecard([signal], [outcome], as_of_utc=NOW)
        tampered = {**signal, "stop_loss_pips": 99.0}

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[decision],
            raw_signals=[signal],
            selected_signals=[tampered],
            outcomes=[outcome],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn("SELECTED_SIGNAL_INVALID", scorecard["cohort_integrity_errors"])
        self.assertIsNone(scorecard["profitability_gate"])
        self.assertFalse(scorecard["live_permission"])

    def test_v2_scorecard_rejects_a_below_floor_selected_ledger_row(self) -> None:
        policy, policy_sha = _v2_policy()
        generated = CUTOFF + timedelta(days=1)
        below = _signal("below-ledger-floor", generated=generated, m5_atr_pips=4.9)
        truth = build_fast_bot_scorecard([below], [], as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[],
            raw_signals=[below],
            selected_signals=[below],
            outcomes=[],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn(
            "SELECTED_SIGNAL_M5_ATR_BELOW_PRECOMMITTED_MINIMUM",
            scorecard["cohort_integrity_errors"],
        )
        self.assertIsNone(scorecard["profitability_evidence"])
        self.assertEqual(
            scorecard["contract"],
            "QR_FAST_BOT_PROFIT_HOLDOUT_SCORECARD_V2",
        )

    def test_unscreened_raw_cycle_invalidates_holdout_completeness(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        selected = _signal("covered", generated=generated)
        unseen = _signal("unseen", generated=generated + timedelta(days=1))
        decision = build_holdout_decision(
            _shadow(selected),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        outcome = _outcome(selected, win=True)
        truth = build_fast_bot_scorecard([selected], [outcome], as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[decision],
            raw_signals=[selected, unseen],
            selected_signals=[selected],
            outcomes=[outcome],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn(
            "RAW_SIGNAL_CYCLE_DECISION_COVERAGE_INVALID",
            scorecard["cohort_integrity_errors"],
        )

    def test_resealed_late_selection_decision_is_rejected_by_evaluator(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        signal = _signal("late-decision", generated=generated)
        decision = build_holdout_decision(
            _shadow(signal),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        decision["generated_at_utc"] = (generated + timedelta(seconds=46)).isoformat()
        decision = seal(decision)
        outcome = _outcome(signal, win=True)
        truth = build_fast_bot_scorecard([signal], [outcome], as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[decision],
            raw_signals=[signal],
            selected_signals=[signal],
            outcomes=[outcome],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn(
            "DECISION_SELECTION_WINDOW_EXPIRED",
            scorecard["cohort_integrity_errors"],
        )

    def test_resealed_decision_cannot_bypass_opposite_side_rejection(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        allowed = _signal("allowed-opposite", generated=generated)
        opposite = _signal(
            "blocked-opposite",
            generated=generated,
            side="SHORT",
        )
        decision = build_holdout_decision(
            _shadow(allowed, opposite),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        self.assertEqual(decision["status"], "BLOCKED_OPPOSITE_SIDE_AMBIGUITY")
        tampered = copy.deepcopy(decision)
        tampered["status"] = "SELECTED_PROSPECTIVE_HOLDOUT"
        tampered["selected_signal_count"] = 1
        tampered["selected_signal_sha256s"] = [allowed["signal_sha256"]]
        tampered["selected_signals"] = [allowed]
        for row in tampered["selection_rows"]:
            if row["signal_sha256"] == allowed["signal_sha256"]:
                row["status"] = "SELECTED_PROSPECTIVE_HOLDOUT"
                row["reasons"] = []
        tampered = seal(tampered)
        outcome = _outcome(allowed, win=True)
        truth = build_fast_bot_scorecard([allowed], [outcome], as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[tampered],
            raw_signals=[allowed, opposite],
            selected_signals=[allowed],
            outcomes=[outcome],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn(
            "DECISION_SELECTION_SEMANTICS_MISMATCH",
            scorecard["cohort_integrity_errors"],
        )

    def test_resealed_decision_cannot_bypass_equal_priority_tie(self) -> None:
        policy, policy_sha = _policy()
        generated = CUTOFF + timedelta(days=1)
        first = _signal("first-tied", generated=generated)
        second = _signal("second-tied", generated=generated)
        decision = build_holdout_decision(
            _shadow(first, second),
            policy=policy,
            policy_sha256=policy_sha,
            now_utc=generated + timedelta(seconds=1),
        )
        self.assertEqual(decision["status"], "BLOCKED_AMBIGUOUS_TOP_PRIORITY")
        tampered = copy.deepcopy(decision)
        tampered["status"] = "SELECTED_PROSPECTIVE_HOLDOUT"
        tampered["selected_signal_count"] = 1
        tampered["selected_signal_sha256s"] = [first["signal_sha256"]]
        tampered["selected_signals"] = [first]
        for row in tampered["selection_rows"]:
            if row["signal_sha256"] == first["signal_sha256"]:
                row["status"] = "SELECTED_PROSPECTIVE_HOLDOUT"
                row["reasons"] = []
        tampered = seal(tampered)
        outcome = _outcome(first, win=True)
        truth = build_fast_bot_scorecard([first], [outcome], as_of_utc=NOW)

        scorecard = build_holdout_scorecard(
            policy=policy,
            policy_sha256=policy_sha,
            decisions=[tampered],
            raw_signals=[first, second],
            selected_signals=[first],
            outcomes=[outcome],
            truth_scorecard=truth,
            now_utc=NOW,
        )

        self.assertEqual(scorecard["status"], "REJECT_INVALID_HOLDOUT_COHORT")
        self.assertIn(
            "DECISION_SELECTION_SEMANTICS_MISMATCH",
            scorecard["cohort_integrity_errors"],
        )


if __name__ == "__main__":
    unittest.main()
