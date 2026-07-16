from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot import (
    _entry_experiment_arms as _primary_entry_experiment_arms,
)
from quant_rabbit.fast_bot import _shadow_geometry_pips as _primary_geometry_pips
from quant_rabbit.fast_bot_learning import (
    LEARNING_ARM_SPECS,
    LEARNING_SEAT_CONTRACT,
    _append_learning_seats_once,
    _learning_arms,
    build_fast_bot_learning_shadow,
    run_fast_bot_learning_shadow,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
METHODS = ("BREAKOUT_FAILURE", "RANGE_ROTATION", "TREND_CONTINUATION")
SIDES = ("LONG", "SHORT")


def _canonical_sha(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(body: dict) -> dict:
    return {**body, "contract_sha256": _canonical_sha(body)}


def _snapshot(
    pairs: dict[str, tuple[float, float]],
    *,
    now: datetime = NOW,
) -> dict:
    return {
        "fetched_at_utc": now.isoformat(),
        "quotes": {
            pair: {
                "bid": bid,
                "ask": ask,
                "timestamp_utc": now.isoformat(),
            }
            for pair, (bid, ask) in pairs.items()
        },
        "positions": [],
        "orders": [],
    }


def _row(
    pair: str,
    side: str,
    method: str,
    *,
    state: str,
    spread_pips: float,
    m5_atr_pips: float = 5.0,
    hard_blockers: tuple[str, ...] = (),
    caution_reasons: tuple[str, ...] = (),
    score: float = 1.0,
    m1_closed: datetime = NOW,
) -> dict:
    return {
        "pair": pair,
        "side": side,
        "method": method,
        "state": state,
        "score": score,
        "execution_enabled": state == "GO",
        "hard_blockers": list(hard_blockers),
        "caution_reasons": list(caution_reasons),
        "m1_closed_candle_utc": m1_closed.isoformat(),
        "m5_atr_pips": m5_atr_pips,
        "spread_pips": spread_pips,
        "spread_to_m5_atr": round(spread_pips / m5_atr_pips, 6),
        "failed_break_direction": "NONE",
    }


def _regime(
    snapshot: dict,
    rows: list[dict],
    *,
    now: datetime = NOW,
) -> dict:
    return _seal(
        {
            "contract": "QR_HIERARCHICAL_BOT_REGIME_V1",
            "schema_version": 1,
            "generated_at_utc": now.isoformat(),
            "rows": rows,
            "sources": {"broker_snapshot_sha256": _canonical_sha(snapshot)},
        }
    )


def _spread(pair: str, bid: float, ask: float) -> float:
    return round((ask - bid) * (100 if pair.endswith("_JPY") else 10000), 6)


class FastBotLearningTest(unittest.TestCase):
    def test_separate_seats_collect_cost_caution_and_go_without_authority(self) -> None:
        prices = {
            "EUR_USD": (1.10000, 1.10040),
            "GBP_USD": (1.25000, 1.25008),
        }
        snapshot = _snapshot(prices)
        eur_spread = _spread("EUR_USD", *prices["EUR_USD"])
        gbp_spread = _spread("GBP_USD", *prices["GBP_USD"])
        rows = [
            _row(
                "EUR_USD",
                "LONG",
                "BREAKOUT_FAILURE",
                state="STOP",
                spread_pips=eur_spread,
                m5_atr_pips=8.0,
                hard_blockers=("SPREAD_ANOMALY",),
            ),
            _row(
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
                state="STOP",
                spread_pips=eur_spread,
                m5_atr_pips=8.0,
                hard_blockers=("SPREAD_ANOMALY", "TECHNICAL_INPUT_STALE"),
            ),
            _row(
                "EUR_USD",
                "LONG",
                "RANGE_ROTATION",
                state="STOP",
                spread_pips=eur_spread,
                m5_atr_pips=8.0,
                hard_blockers=("AI_REGIME_SUPERVISOR_STOP",),
            ),
            _row(
                "GBP_USD",
                "SHORT",
                "RANGE_ROTATION",
                state="CAUTION",
                spread_pips=gbp_spread,
                caution_reasons=("RANGE_ROTATION_TRIGGER_NOT_READY",),
            ),
            _row(
                "GBP_USD",
                "LONG",
                "TREND_CONTINUATION",
                state="GO",
                spread_pips=gbp_spread,
                score=9.0,
            ),
        ]

        shadow = build_fast_bot_learning_shadow(
            _regime(snapshot, rows),
            snapshot,
            now_utc=NOW,
        )

        self.assertEqual(shadow["status"], "EMITTED")
        self.assertEqual(shadow["seat_count"], 2)
        self.assertEqual(shadow["candidate_count"], 3)
        classes = {
            candidate["candidate_class"]
            for seat in shadow["seats"]
            for candidate in seat["candidates"]
        }
        self.assertEqual(
            classes,
            {"COST_BLOCKED", "CAUTION_TECHNICAL", "GO_CONTROL"},
        )
        cost_seat = next(seat for seat in shadow["seats"] if seat["pair"] == "EUR_USD")
        self.assertEqual(cost_seat["eligible_counts"]["COST_BLOCKED"], 1)
        self.assertEqual(cost_seat["cost_context"]["cost_pressure_bucket"], "1_25_1_50")
        self.assertFalse(cost_seat["cost_context"]["current_cost_gate_pass"])
        for item in [shadow, *shadow["seats"]]:
            self.assertTrue(item["diagnostic_only"])
            self.assertFalse(item["primary_promotion_eligible"])
            self.assertTrue(item["shadow_only"])
            self.assertFalse(item["live_permission"])
            self.assertFalse(item["broker_mutation_allowed"])
            self.assertEqual(item["order_authority"], "NONE")
            self.assertEqual(
                item["lifecycle"],
                "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW",
            )
        for seat in shadow["seats"]:
            self.assertTrue(seat["frozen_bid_ask_truth_path_required"])
            self.assertEqual(
                seat["future_metrics_required"],
                ["FILL", "POST_COST_PNL", "MFE", "MAE"],
            )
            self.assertTrue(
                all(
                    candidate["counterfactual_comparison_group_id"] == seat["seat_id"]
                    and candidate["frozen_bid_ask_truth_path_required"] is True
                    for candidate in seat["candidates"]
                )
            )
        self.assertFalse(shadow["top_one_selection_assumption"])
        self.assertEqual(
            shadow["future_parallel_go_policy"],
            "ALL_ELIGIBLE_GO_CONCURRENT_SUBJECT_TO_PORTFOLIO_GATES",
        )
        self.assertEqual(
            shadow["scorecard_aggregation_policy"],
            "EXACT_PAIR_SIDE_METHOD_HORIZON_LANE_NO_NETTING",
        )
        self.assertFalse(shadow["static_pair_correlation_exclusion_allowed"])
        self.assertFalse(shadow["pair_direction_netting_allowed"])
        self.assertFalse(shadow["future_parallel_go_policy_is_live_permission"])
        self.assertFalse(shadow["primary_artifacts_mutated"])
        self.assertFalse(shadow["current_risk_gate_changed"])
        self.assertEqual(shadow["pair_universe_size"], 28)
        self.assertEqual(shadow["maximum_seats_per_utc_day"], 4032)
        self.assertEqual(shadow["maximum_candidates_per_seat"], 6)
        self.assertEqual(shadow["maximum_candidates_per_utc_day"], 24192)
        self.assertEqual(
            shadow["hot_ledger_retention_policy"],
            "PENDING_NOT_IMPLEMENTED",
        )
        self.assertEqual(
            shadow["learning_outcome_aggregation_status"],
            "PENDING_NOT_IMPLEMENTED",
        )
        self.assertIn("UNBOUNDED", shadow["storage_growth_disclosure"])
        self.assertFalse(shadow["promotion_allowed"])
        self.assertEqual(
            shadow["promotion_blockers"],
            [
                "HOT_LEDGER_RETENTION_POLICY_NOT_IMPLEMENTED",
                "LEARNING_OUTCOME_AGGREGATION_NOT_IMPLEMENTED",
                "SEPARATE_FRESH_FORWARD_PROMOTION_CONTRACT_REQUIRED",
            ],
        )

    def test_all_six_cells_share_one_frozen_seat_without_unselected_candidates(
        self,
    ) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        snapshot = _snapshot(prices)
        spread = _spread(pair, *prices[pair])
        rows = [
            _row(
                pair,
                side,
                method,
                state="CAUTION",
                spread_pips=spread,
                caution_reasons=("TRIGGER_NOT_READY",),
            )
            for method in METHODS
            for side in SIDES
        ]

        regime = _regime(snapshot, rows)
        first = build_fast_bot_learning_shadow(regime, snapshot, now_utc=NOW)
        second = build_fast_bot_learning_shadow(regime, snapshot, now_utc=NOW)
        seat = first["seats"][0]
        cells = [
            (candidate["side"], candidate["method"]) for candidate in seat["candidates"]
        ]

        self.assertEqual(seat["selected_candidate_count"], 6)
        self.assertEqual(seat["eligible_counts"]["CAUTION_TECHNICAL"], 6)
        self.assertEqual(
            seat["eligible_but_unselected_counts"]["CAUTION_TECHNICAL"],
            0,
        )
        self.assertEqual(len(seat["eligible_cells"]), 6)
        self.assertEqual(seat["eligible_but_unselected_cells"], [])
        self.assertEqual(
            set(cells),
            {(side, method) for method in METHODS for side in SIDES},
        )
        for method in METHODS:
            self.assertEqual(
                {
                    side
                    for side, candidate_method in cells
                    if candidate_method == method
                },
                set(SIDES),
            )
        self.assertEqual(
            {
                candidate["counterfactual_comparison_group_id"]
                for candidate in seat["candidates"]
            },
            {seat["seat_id"]},
        )
        self.assertTrue(seat["frozen_bid_ask_truth_path_required"])
        self.assertEqual(
            seat["future_truth_fetch_unit"],
            "ONE_BID_ASK_PATH_PER_PAIR_10M_SEAT",
        )
        self.assertEqual(
            cells,
            [
                (candidate["side"], candidate["method"])
                for candidate in second["seats"][0]["candidates"]
            ],
        )

    def test_go_controls_do_not_encode_top_one_and_preserve_parallel_summary(
        self,
    ) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        snapshot = _snapshot(prices)
        spread = _spread(pair, *prices[pair])
        rows = [
            _row(
                pair,
                side,
                method,
                state="GO",
                spread_pips=spread,
                score=float(index),
            )
            for index, (method, side) in enumerate(
                ((method, side) for method in METHODS for side in SIDES),
                start=1,
            )
        ]

        seat = build_fast_bot_learning_shadow(
            _regime(snapshot, rows), snapshot, now_utc=NOW
        )["seats"][0]

        self.assertEqual(seat["eligible_counts"]["GO_CONTROL"], 6)
        self.assertEqual(seat["selected_counts"]["GO_CONTROL"], 6)
        self.assertEqual(seat["eligible_but_unselected_counts"]["GO_CONTROL"], 0)
        self.assertFalse(seat["top_one_selection_assumption"])
        self.assertEqual(
            seat["future_parallel_go_portfolio_gates"],
            ["CURRENCY_EXPOSURE", "CORRELATION", "BROKER_MARGIN"],
        )
        self.assertTrue(
            all(
                candidate["comparison_role"] == "ELIGIBLE_GO_CONTROL"
                and candidate["top_one_selection_assumption"] is False
                for candidate in seat["candidates"]
            )
        )

    def test_eight_arms_are_one_factor_at_a_time_and_strictly_passive(self) -> None:
        expected_ids = [spec[0] for spec in LEARNING_ARM_SPECS]
        for side in SIDES:
            with self.subTest(side=side):
                arms = _learning_arms(
                    pair="EUR_USD",
                    side=side,
                    method="RANGE_ROTATION",
                    bid=1.10000,
                    ask=1.10008,
                    spread_pips=0.8,
                    m5_atr_pips=5.0,
                )
                self.assertEqual([arm["arm_id"] for arm in arms], expected_ids)
                self.assertEqual(len(arms), 8)
                self.assertEqual(
                    {arm["horizon_lane"] for arm in arms},
                    {"M1_EXECUTION_HOLD_900S", "M1_EXECUTION_HOLD_1800S"},
                )
                base = arms[0]
                if side == "LONG":
                    self.assertTrue(
                        all(1.10000 <= arm["entry"] < 1.10008 for arm in arms)
                    )
                else:
                    self.assertTrue(
                        all(1.10000 < arm["entry"] <= 1.10008 for arm in arms)
                    )
                for arm in arms[1:4]:
                    self.assertEqual(
                        arm["entry_ttl_seconds"], base["entry_ttl_seconds"]
                    )
                    self.assertEqual(arm["max_hold_seconds"], base["max_hold_seconds"])
                    self.assertEqual(arm["tp_multiplier"], 1.0)
                    self.assertEqual(arm["sl_multiplier"], 1.0)
                self.assertEqual(arms[4]["entry"], base["entry"])
                self.assertEqual(arms[4]["max_hold_seconds"], base["max_hold_seconds"])
                self.assertEqual(arms[4]["entry_ttl_seconds"], 180)
                self.assertEqual(arms[5]["entry"], base["entry"])
                self.assertEqual(
                    arms[5]["entry_ttl_seconds"], base["entry_ttl_seconds"]
                )
                self.assertEqual(arms[5]["max_hold_seconds"], 1800)
                self.assertEqual(arms[6]["entry"], base["entry"])
                self.assertEqual(arms[6]["stop_loss"], base["stop_loss"])
                self.assertEqual(arms[7]["entry"], base["entry"])
                self.assertEqual(arms[7]["take_profit"], base["take_profit"])

        narrow_long = _learning_arms(
            pair="EUR_USD",
            side="LONG",
            method="TREND_CONTINUATION",
            bid=1.10000,
            ask=1.10001,
            spread_pips=0.1,
            m5_atr_pips=5.0,
        )
        self.assertTrue(all(arm["entry"] == 1.10000 for arm in narrow_long))

    def test_duplicated_base_geometry_matches_primary_contract(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                tp_pips, sl_pips = _primary_geometry_pips(
                    method,
                    spread=0.8,
                    m5_atr=5.0,
                )
                primary = _primary_entry_experiment_arms(
                    pair="EUR_USD",
                    side="LONG",
                    bid=1.10000,
                    ask=1.10008,
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                )
                learning = _learning_arms(
                    pair="EUR_USD",
                    side="LONG",
                    method=method,
                    bid=1.10000,
                    ask=1.10008,
                    spread_pips=0.8,
                    m5_atr_pips=5.0,
                )[:4]
                for primary_arm, learning_arm in zip(primary, learning):
                    for field in (
                        "entry",
                        "take_profit",
                        "stop_loss",
                        "take_profit_pips",
                        "stop_loss_pips",
                    ):
                        self.assertEqual(learning_arm[field], primary_arm[field])

    def test_pair_bucket_append_is_idempotent_even_when_m1_and_seat_id_change(
        self,
    ) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regime_path = root / "regime.json"
            snapshot_path = root / "snapshot.json"
            output_path = root / "latest.json"
            ledger_path = root / "ledger.jsonl"

            def write_inputs(now: datetime) -> None:
                snapshot = _snapshot(prices, now=now)
                spread = _spread(pair, *prices[pair])
                rows = [
                    _row(
                        pair,
                        "LONG",
                        "TREND_CONTINUATION",
                        state="GO",
                        spread_pips=spread,
                        m1_closed=now,
                    )
                ]
                snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
                regime_path.write_text(
                    json.dumps(_regime(snapshot, rows, now=now)),
                    encoding="utf-8",
                )

            write_inputs(NOW)
            first = run_fast_bot_learning_shadow(
                regime_contract_path=regime_path,
                broker_snapshot_path=snapshot_path,
                output_path=output_path,
                ledger_path=ledger_path,
                now_utc=NOW,
            )
            write_inputs(NOW + timedelta(minutes=5))
            second = run_fast_bot_learning_shadow(
                regime_contract_path=regime_path,
                broker_snapshot_path=snapshot_path,
                output_path=output_path,
                ledger_path=ledger_path,
                now_utc=NOW + timedelta(minutes=5),
            )
            ledger_rows = [
                json.loads(line) for line in ledger_path.read_text().splitlines()
            ]
            latest = json.loads(output_path.read_text())

        self.assertEqual(first["ledger_appended"], 1)
        self.assertEqual(second["ledger_appended"], 0)
        self.assertEqual(second["bucket_duplicates_suppressed"], 1)
        self.assertEqual(len(ledger_rows), 1)
        self.assertEqual(ledger_rows[0]["contract"], LEARNING_SEAT_CONTRACT)
        self.assertEqual(latest["ledger_status"], "APPENDED")
        self.assertTrue(latest["always_on_counterfactual_shadow"])

    def test_invalid_or_mismatched_inputs_fail_closed_without_candidates(self) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        snapshot = _snapshot(prices)
        spread = _spread(pair, *prices[pair])
        row = _row(
            pair,
            "LONG",
            "TREND_CONTINUATION",
            state="GO",
            spread_pips=spread,
        )
        regime = _regime(snapshot, [row])

        tampered_snapshot = json.loads(json.dumps(snapshot))
        tampered_snapshot["quotes"][pair]["bid"] = 1.10001
        mismatch = build_fast_bot_learning_shadow(
            regime,
            tampered_snapshot,
            now_utc=NOW,
        )
        self.assertEqual(mismatch["status"], "REGIME_SNAPSHOT_BINDING_INVALID")
        self.assertEqual(mismatch["seats"], [])

        invalid_atr_row = {**row, "m5_atr_pips": None, "spread_to_m5_atr": None}
        invalid_atr = build_fast_bot_learning_shadow(
            _regime(snapshot, [invalid_atr_row]),
            snapshot,
            now_utc=NOW,
        )
        self.assertEqual(invalid_atr["status"], "NO_ELIGIBLE_LEARNING_SEAT")
        self.assertEqual(invalid_atr["seat_count"], 0)

    def test_corrupt_existing_ledger_fails_closed_and_latest_records_error(
        self,
    ) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        snapshot = _snapshot(prices)
        spread = _spread(pair, *prices[pair])
        regime = _regime(
            snapshot,
            [
                _row(
                    pair,
                    "LONG",
                    "TREND_CONTINUATION",
                    state="GO",
                    spread_pips=spread,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regime_path = root / "regime.json"
            snapshot_path = root / "snapshot.json"
            output_path = root / "latest.json"
            ledger_path = root / "ledger.jsonl"
            regime_path.write_text(json.dumps(regime), encoding="utf-8")
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            ledger_path.write_text("not-json\n", encoding="utf-8")

            result = run_fast_bot_learning_shadow(
                regime_contract_path=regime_path,
                broker_snapshot_path=snapshot_path,
                output_path=output_path,
                ledger_path=ledger_path,
                now_utc=NOW,
            )
            latest = json.loads(output_path.read_text())

        self.assertEqual(result["status"], "LEARNING_LEDGER_INVALID")
        self.assertEqual(result["ledger_appended"], 0)
        self.assertEqual(latest["ledger_status"], "INVALID_FAIL_CLOSED")

    def test_resealed_but_inexact_seat_identity_is_rejected(self) -> None:
        pair = "EUR_USD"
        prices = {pair: (1.10000, 1.10008)}
        snapshot = _snapshot(prices)
        spread = _spread(pair, *prices[pair])
        seat = build_fast_bot_learning_shadow(
            _regime(
                snapshot,
                [
                    _row(
                        pair,
                        "LONG",
                        "TREND_CONTINUATION",
                        state="GO",
                        spread_pips=spread,
                    )
                ],
            ),
            snapshot,
            now_utc=NOW,
        )["seats"][0]
        tampered_body = {
            **{key: value for key, value in seat.items() if key != "contract_sha256"},
            "seat_id": "0" * 24,
        }
        tampered = _seal(tampered_body)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "invalid learning seat"):
                _append_learning_seats_once(
                    Path(tmp) / "ledger.jsonl",
                    [tampered],
                )


if __name__ == "__main__":
    unittest.main()
