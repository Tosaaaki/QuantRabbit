from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.fast_bot import build_hierarchical_regime_contract
from quant_rabbit.fast_bot_episode import (
    run_fast_bot_episode_shadow,
    verify_episode_ledger,
)
from quant_rabbit.fast_bot_episode_truth import (
    INVERSE_DIRECTION,
    INVERSE_SIDE_OTHER_METHOD,
    ROUTE_ALIGNED,
    SAME_SIDE_OTHER_METHOD,
    _episode_outcome_valid,
    _write_jsonl_atomic,
    build_fast_bot_episode_scorecard,
    resolve_fast_bot_episode_vehicle,
    run_fast_bot_episode_truth_cycle,
)
from quant_rabbit.fast_bot_learning import build_fast_bot_learning_shadow
from quant_rabbit.fast_bot_learning_truth import _seat_maturity
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
CONFIRMED_AT = NOW + timedelta(minutes=1)
HASH = "a" * 64
TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D": 1440,
}


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(body: dict) -> dict:
    return {**body, "contract_sha256": _sha(body)}


def _candles(timeframe: str, *, failed_break: bool = False) -> list[dict]:
    minutes = TIMEFRAME_MINUTES[timeframe]
    rows = [
        {
            "t": (NOW - timedelta(minutes=(21 - index) * minutes)).isoformat(),
            "o": 1.105,
            "h": 1.110,
            "l": 1.100,
            "c": 1.105,
            "complete": True,
        }
        for index in range(21)
    ]
    if failed_break:
        rows[-1] = {
            "t": (NOW - timedelta(minutes=minutes)).isoformat(),
            "o": 1.109,
            "h": 1.111,
            "l": 1.104,
            "c": 1.109,
            "complete": True,
        }
    return rows


def _view(timeframe: str, *, failed_break: bool = False) -> dict:
    direction = "DOWN"
    phase = "PRE_RANGE" if timeframe in {"M1", "M5", "M15", "M30"} else "TREND"
    return {
        "granularity": timeframe,
        "recent_candles": _candles(timeframe, failed_break=failed_break),
        "candle_integrity": {"forecast_blocking": False},
        "indicators": {
            "atr_pips": 5.0,
            "adx_14": 22.0,
            "plus_di_14": 25.0,
            "minus_di_14": 20.0,
            "rsi_14": 55.0,
            "macd_hist": 0.1,
            "z_score_20": 0.2,
            "vortex_plus_14": 1.1,
            "vortex_minus_14": 0.9,
            "linreg_slope_20": 0.05,
            "linreg_r2_20": 0.8,
            "bb_squeeze": 0,
            "bb_width_percentile_100": 0.5,
            "aroon_osc_14": 10.0,
            "realized_vol_20": 0.1,
            "roc_10": 0.2,
            "close": 1.109,
            "vwap": 1.106,
            "supertrend_dir": -1,
            "psar_dir": -1,
            "ichimoku_cloud_pos": -1,
            "regime_quantile": "NORMAL",
            "unknown_indicator_must_not_leak": 123.0,
        },
        "indicator_series": {
            "rsi_14": [50.0, 52.0, 55.0],
            "macd_hist": [-0.1, 0.0, 0.1],
            "adx_14": [18.0, 20.0, 22.0],
            "atr_pips": [4.0, 4.5, 5.0],
            "ema_12_minus_50_pips": [-0.1, 0.0, 0.2],
            "unknown_series_must_not_leak": [999.0],
        },
        "market_state": {
            "direction": direction,
            "phase": phase,
            "readiness": "TRIGGERED",
            "trigger": "BREAKOUT_CLOSE",
            "structure": "BREAKOUT_ACTIVE",
            "location": "MIDDLE_THIRD",
            "value_zone": "FAIR_VALUE",
            "extension": "BALANCED",
            "evidence_complete": True,
        },
    }


def _inputs() -> tuple[dict, dict, dict]:
    fast = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    _view("M1"),
                    _view("M5", failed_break=True),
                    _view("M15"),
                ],
            }
        ],
    }
    slow = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [_view(tf) for tf in ("M30", "H1", "H4", "D")],
            }
        ],
    }
    snapshot = {
        "fetched_at_utc": NOW.isoformat(),
        "quotes": {
            "EUR_USD": {
                "bid": 1.10000,
                "ask": 1.10008,
                "timestamp_utc": NOW.isoformat(),
            }
        },
        "positions": [],
        "orders": [],
    }
    return fast, slow, snapshot


def _handoff(
    *,
    regime: dict,
    fast: dict,
    slow: dict,
    snapshot: dict,
    shadow: dict,
    cycle: datetime,
) -> dict:
    return _seal(
        {
            "contract": "QR_FAST_BOT_EPISODE_HANDOFF_V2",
            "schema_version": 2,
            "cycle_generated_at_utc": cycle.isoformat(),
            "regime_contract_sha256": regime["contract_sha256"],
            "fast_pair_charts_sha256": _sha(fast),
            "slow_pair_charts_sha256": _sha(slow),
            "broker_snapshot_sha256": _sha(snapshot),
            "prospective_vehicle_shadow_sha256": shadow["contract_sha256"],
            "regime_contract": regime,
            "fast_pair_charts": copy.deepcopy(fast),
            "slow_pair_charts": copy.deepcopy(slow),
            "broker_snapshot": copy.deepcopy(snapshot),
            "prospective_vehicle_shadow": copy.deepcopy(shadow),
            "diagnostic_only": True,
            "shadow_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
    )


def _build_fixture(root: Path, *, late_confirmation: bool = False) -> dict:
    fast, slow, snapshot = _inputs()
    ledger = root / "episode_ledger.jsonl"
    sources = root / "episode_sources"
    state = root / "episode_state.json"
    regime = build_hierarchical_regime_contract(
        fast_pair_charts=fast,
        slow_pair_charts=slow,
        broker_snapshot=snapshot,
        now_utc=NOW,
    )
    first = run_fast_bot_episode_shadow(
        regime_contract=regime,
        fast_pair_charts=fast,
        slow_pair_charts=slow,
        output_path=state,
        ledger_path=ledger,
        source_archive_dir=sources,
        now_utc=NOW,
    )
    if first["latest_episodes"][0]["state"] != "REJECTED":
        raise AssertionError("fixture must start from a rejected M5 attempt")
    rejected_shadow = build_fast_bot_learning_shadow(
        regime,
        snapshot,
        now_utc=NOW,
    )
    rejected_handoff = _handoff(
        regime=regime,
        fast=fast,
        slow=slow,
        snapshot=snapshot,
        shadow=rejected_shadow,
        cycle=NOW,
    )

    m1 = fast["charts"][0]["views"][0]
    m1["recent_candles"] = [
        *m1["recent_candles"],
        {
            "t": NOW.isoformat(),
            "o": 1.105,
            "h": 1.106,
            "l": 1.103,
            "c": 1.104,
            "complete": True,
        },
    ][-21:]
    for packet in (fast, slow):
        packet["generated_at_utc"] = CONFIRMED_AT.isoformat()
    snapshot["fetched_at_utc"] = CONFIRMED_AT.isoformat()
    snapshot["quotes"]["EUR_USD"]["timestamp_utc"] = CONFIRMED_AT.isoformat()
    regime = build_hierarchical_regime_contract(
        fast_pair_charts=fast,
        slow_pair_charts=slow,
        broker_snapshot=snapshot,
        now_utc=CONFIRMED_AT,
    )
    second = run_fast_bot_episode_shadow(
        regime_contract=regime,
        fast_pair_charts=fast,
        slow_pair_charts=slow,
        output_path=state,
        ledger_path=ledger,
        source_archive_dir=sources,
        now_utc=CONFIRMED_AT,
        processed_at_utc=(
            CONFIRMED_AT + timedelta(minutes=2)
            if late_confirmation
            else CONFIRMED_AT
        ),
    )
    if second["latest_episodes"][0]["state"] != "CONFIRMED":
        raise AssertionError("fixture must end in a confirmed episode")
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    if verify_episode_ledger(
        events,
        as_of_utc=CONFIRMED_AT,
        source_archive_dir=sources,
    ) != (True, None):
        raise AssertionError("fixture episode ledger must pass full verification")

    shadow = build_fast_bot_learning_shadow(
        regime,
        snapshot,
        now_utc=CONFIRMED_AT,
    )
    handoff = _handoff(
        regime=regime,
        fast=fast,
        slow=slow,
        snapshot=snapshot,
        shadow=shadow,
        cycle=CONFIRMED_AT,
    )
    return {
        "ledger": ledger,
        "sources": sources,
        "handoff": handoff,
        "rejected_handoff": rejected_handoff,
        "shadow": shadow,
        "vehicle_ledger": root / "vehicle_ledger.jsonl",
        "outcome_ledger": root / "outcome_ledger.jsonl",
        "scorecard": root / "scorecard.json",
        "lock": root / "truth.lock",
    }


def _run(paths: dict, *, handoffs: list[dict], now: datetime, client_factory) -> dict:
    return run_fast_bot_episode_truth_cycle(
        handoffs=handoffs,
        episode_ledger_path=paths["ledger"],
        source_archive_dir=paths["sources"],
        vehicle_ledger_path=paths["vehicle_ledger"],
        outcome_ledger_path=paths["outcome_ledger"],
        scorecard_path=paths["scorecard"],
        lock_path=paths["lock"],
        client_factory=client_factory,
        clock=lambda: now,
    )


def _load_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1:
        raise AssertionError(f"expected one row, got {len(rows)}")
    return rows[0]


def _flat_path(start: datetime, end: datetime) -> list[S5BidAskCandle]:
    rows: list[S5BidAskCandle] = []
    current = start
    index = 0
    while current < end:
        bid_close = 1.100035 + (index % 3) * 0.000005
        rows.append(
            S5BidAskCandle(
                timestamp_utc=current,
                bid_o=1.10004,
                bid_h=1.10010,
                bid_l=1.09990,
                bid_c=bid_close,
                ask_o=1.10012,
                ask_h=1.10018,
                ask_l=1.09998,
                ask_c=bid_close + 0.00008,
            )
        )
        current += timedelta(seconds=5)
        index += 1
    return rows


class FastBotEpisodeTruthTest(unittest.TestCase):
    def test_jsonl_writer_rejects_row_cap_before_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bounded.jsonl"
            with self.assertRaisesRegex(ValueError, "row cap"):
                _write_jsonl_atomic(
                    path,
                    [{"row": 1}, {"row": 2}],
                    max_bytes=1024,
                    max_rows=1,
                )
            self.assertFalse(path.exists())

    def test_projects_all_six_cells_generic_arms_features_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))

            def no_client():
                raise AssertionError("immature projection must not create a client")

            first = _run(
                paths,
                handoffs=[paths["handoff"]],
                now=CONFIRMED_AT,
                client_factory=no_client,
            )
            vehicle = _load_one(paths["vehicle_ledger"])

            self.assertEqual(first["status"], "PROJECTED_NO_DUE")
            self.assertEqual(first["vehicle_projection_status"], "VERIFIED")
            self.assertEqual(first["vehicle_ledger_appended"], 1)
            self.assertEqual(vehicle["candidate_count"], 6)
            self.assertEqual(
                vehicle["arm_count"],
                sum(len(row["arms"]) for row in vehicle["learning_seat"]["candidates"]),
            )
            self.assertGreater(vehicle["arm_count"], 6)
            self.assertTrue(vehicle["causal_input_proof_eligible"])
            self.assertTrue(vehicle["scorecard_eligible"])
            self.assertEqual(vehicle["scorecard_ineligibility_reasons"], [])
            roles = {
                (row["side"], row["method"]): row["episode_role"]
                for row in vehicle["candidate_roles"]
            }
            self.assertEqual(roles[("SHORT", "BREAKOUT_FAILURE")], ROUTE_ALIGNED)
            self.assertEqual(roles[("LONG", "BREAKOUT_FAILURE")], INVERSE_DIRECTION)
            self.assertEqual(
                roles[("SHORT", "TREND_CONTINUATION")],
                SAME_SIDE_OTHER_METHOD,
            )
            self.assertEqual(
                roles[("LONG", "TREND_CONTINUATION")],
                INVERSE_SIDE_OTHER_METHOD,
            )
            features = vehicle["technical_feature_snapshot"]["timeframes"]
            self.assertEqual([row["timeframe"] for row in features], list(TIMEFRAME_MINUTES))
            for row in features:
                self.assertEqual(
                    set(row["indicator_series"]),
                    {
                        "rsi_14",
                        "macd_hist",
                        "adx_14",
                        "atr_pips",
                        "ema_12_minus_50_pips",
                    },
                )
                self.assertEqual(row["indicators"]["supertrend_dir"], -1)
                self.assertEqual(row["indicators"]["psar_dir"], -1)
                self.assertEqual(row["indicators"]["ichimoku_cloud_pos"], -1)
                self.assertIn("z_score_20", row["indicators"])
                self.assertNotIn("unknown_indicator_must_not_leak", row["indicators"])
                self.assertNotIn("unknown_series_must_not_leak", row["indicator_series"])
            technical_shadow = vehicle["technical_hypothesis_shadow"]
            self.assertEqual(technical_shadow["status"], "EMITTED")
            self.assertEqual(len(technical_shadow["hypotheses"]), 8)
            self.assertEqual(
                vehicle["technical_hypothesis_shadow_sha256"],
                technical_shadow["contract_sha256"],
            )
            self.assertEqual(
                vehicle["source_binding"]["technical_cost_state_sha256"],
                technical_shadow["cost_state_sha256"],
            )
            self.assertEqual(
                vehicle["technical_hypothesis_evaluator_policy"],
                technical_shadow["evaluator_policy"],
            )
            hypothesis_rows = {
                row["hypothesis_id"]: row
                for row in technical_shadow["hypotheses"]
            }
            self.assertEqual(hypothesis_rows["H05"]["status"], "ACTIVE_SHADOW")
            self.assertEqual(hypothesis_rows["H07"]["status"], "INACTIVE_SHADOW")
            for key in (
                "diagnostic_only",
                "shadow_only",
            ):
                self.assertTrue(vehicle[key])
            self.assertEqual(vehicle["order_authority"], "NONE")
            self.assertFalse(vehicle["live_permission"])
            self.assertFalse(vehicle["broker_mutation_allowed"])

            second = _run(
                paths,
                handoffs=[paths["handoff"]],
                now=CONFIRMED_AT,
                client_factory=no_client,
            )
            self.assertEqual(second["vehicle_ledger_appended"], 0)
            self.assertEqual(second["vehicle_ledger_idempotent"], 1)
            self.assertEqual(len(paths["vehicle_ledger"].read_text().splitlines()), 1)

    def test_v1_or_no_same_cycle_confirmed_event_projects_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            v1 = copy.deepcopy(paths["handoff"])
            v1["contract"] = "QR_FAST_BOT_EPISODE_HANDOFF_V1"
            v1["schema_version"] = 1
            v1 = _seal({key: value for key, value in v1.items() if key != "contract_sha256"})
            failed = _run(
                paths,
                handoffs=[v1],
                now=CONFIRMED_AT,
                client_factory=lambda: None,
            )
            self.assertEqual(failed["vehicle_projection_status"], "FAILED")
            self.assertFalse(paths["vehicle_ledger"].exists())

            no_match = _run(
                paths,
                handoffs=[paths["rejected_handoff"]],
                now=CONFIRMED_AT,
                client_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("no due vehicle may not create a client")
                ),
            )
            self.assertEqual(no_match["vehicle_projection_status"], "VERIFIED")
            self.assertEqual(no_match["handoff_confirmed_vehicle_count"], 0)
            self.assertEqual(no_match["vehicle_ledger_appended"], 0)
            self.assertFalse(paths["vehicle_ledger"].exists())

    def test_confirmed_without_frozen_seat_is_explicitly_unscored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            shadow_body = {
                key: value
                for key, value in paths["shadow"].items()
                if key != "contract_sha256"
            }
            shadow_body.update(
                {
                    "status": "NO_ELIGIBLE_LEARNING_SEAT",
                    "seat_count": 0,
                    "candidate_count": 0,
                    "complete_six_cell_seat_count": 0,
                    "partial_valid_input_seat_count": 0,
                    "excluded_pair_counts": {"QUOTE_OR_ATR_INVALID": 1},
                    "seats": [],
                }
            )
            empty_shadow = _seal(shadow_body)
            handoff_body = {
                key: value
                for key, value in paths["handoff"].items()
                if key != "contract_sha256"
            }
            handoff_body["prospective_vehicle_shadow"] = empty_shadow
            handoff_body["prospective_vehicle_shadow_sha256"] = empty_shadow[
                "contract_sha256"
            ]
            no_seat_handoff = _seal(handoff_body)

            result = _run(
                paths,
                handoffs=[no_seat_handoff],
                now=CONFIRMED_AT,
                client_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("an unscored event has no due truth fetch")
                ),
            )

            self.assertEqual(result["status"], "NO_DUE_VEHICLES")
            self.assertEqual(result["vehicle_projection_status"], "VERIFIED")
            self.assertEqual(result["handoff_confirmed_event_count"], 1)
            self.assertEqual(result["handoff_confirmed_vehicle_count"], 0)
            self.assertEqual(result["handoff_confirmed_unscored_count"], 1)
            self.assertEqual(
                result["handoff_confirmed_unscored_reason_counts"],
                {"CONFIRMED_EVENT_NO_FROZEN_VEHICLE_SEAT": 1},
            )
            self.assertEqual(
                result["handoff_confirmed_unscored_examples"][0]["reason"],
                "CONFIRMED_EVENT_NO_FROZEN_VEHICLE_SEAT",
            )
            self.assertEqual(
                result["handoff_confirmed_unscored_examples"][0][
                    "prospective_shadow_status"
                ],
                "NO_ELIGIBLE_LEARNING_SEAT",
            )
            self.assertEqual(result["vehicle_ledger_appended"], 0)
            self.assertFalse(paths["vehicle_ledger"].exists())

    def test_mature_cycle_uses_one_path_and_separates_proof_from_bid_ask_mark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            seat = paths["shadow"]["seats"][0]
            maturity = _seat_maturity(seat)
            candles = _flat_path(CONFIRMED_AT, maturity)
            client = object()
            with patch(
                "quant_rabbit.fast_bot_episode_truth.fetch_frozen_s5_truth",
                return_value=(candles, [HASH]),
            ) as fetch:
                result = _run(
                    paths,
                    handoffs=[paths["handoff"]],
                    now=maturity + timedelta(seconds=1),
                    client_factory=lambda: client,
                )
            fetch.assert_called_once()
            self.assertIs(fetch.call_args.args[0], client)
            self.assertEqual(fetch.call_args.kwargs["time_from"], CONFIRMED_AT)
            self.assertEqual(fetch.call_args.kwargs["time_to"], maturity)
            self.assertEqual(result["status"], "RESOLVED")
            self.assertEqual(result["outcome_ledger_appended"], 1)

            outcome = _load_one(paths["outcome_ledger"])
            vehicle = _load_one(paths["vehicle_ledger"])
            self.assertTrue(outcome["truth_path_candles"])
            self.assertTrue(
                outcome[
                    "truth_path_candles_persisted_for_membership_validation"
                ]
            )
            by_clock = {row.timestamp_utc.isoformat(): row for row in candles}
            long_seen = short_seen = False
            for row in outcome["arm_observations"]:
                self.assertEqual(row["proof_exit_reason"], "HORIZON_FULL_STOP_LOSS")
                self.assertLess(row["proof_post_cost_realized_pips"], 0.0)
                self.assertEqual(row["observed_position_state"], "OPEN_AT_HORIZON")
                candle = by_clock[row["observed_horizon_mark_s5_interval_utc"]["from_utc"]]
                if row["side"] == "LONG":
                    long_seen = True
                    self.assertEqual(row["observed_horizon_mark_price_side"], "BID")
                    self.assertAlmostEqual(row["observed_horizon_mark_price"], candle.bid_c)
                else:
                    short_seen = True
                    self.assertEqual(row["observed_horizon_mark_price_side"], "ASK")
                    self.assertAlmostEqual(row["observed_horizon_mark_price"], candle.ask_c)
                self.assertEqual(
                    row["exit_s5_interval_semantics"],
                    "LAST_EXECUTABLE_MARK_ONLY_NOT_PROOF_EXIT",
                )
            self.assertTrue(long_seen and short_seen)
            scorecard = json.loads(paths["scorecard"].read_text())
            self.assertTrue(scorecard["proof_and_observed_mark_must_not_be_mixed"])
            self.assertTrue(scorecard["clusters"])
            self.assertTrue(
                all(
                    cluster["proof_score"]["paired_episode_count"] == 1
                    and cluster["observed_open_horizon_mark"]["paired_episode_count"] == 1
                    for cluster in scorecard["clusters"]
                )
            )
            self.assertTrue(scorecard["technical_hypothesis_clusters"])
            self.assertTrue(
                any(
                    cluster["hypothesis_id"] == "H05"
                    and cluster["hypothesis_status"] == "ACTIVE_SHADOW"
                    and cluster["observed_forward_statistics"]["sample_count"] == 1
                    for cluster in scorecard["technical_hypothesis_clusters"]
                )
            )
            self.assertFalse(
                scorecard[
                    "technical_cluster_statistics_are_forecast_probabilities"
                ]
            )
            self.assertEqual(
                scorecard["no_trade_control"]["basis"],
                "ZERO_PNL_NO_TRADE_CONTROL",
            )
            self.assertTrue(
                any(
                    row["hypothesis_id"] == "H07"
                    and row["unscored_reason"]
                    == "HYPOTHESIS_SPECIFIC_ENTRY_VEHICLE_NOT_IMPLEMENTED"
                    and row["pnl_joined"] is False
                    for row in scorecard["technical_hypothesis_unscored"]
                )
            )

            tampered = copy.deepcopy(outcome)
            mark_row = next(
                row
                for row in tampered["arm_observations"]
                if row["proof_exit_reason"] == "HORIZON_FULL_STOP_LOSS"
            )
            arm = next(
                arm
                for candidate in tampered["learning_outcome"]["candidates"]
                if candidate["candidate_sha256"] == mark_row["candidate_sha256"]
                for arm in candidate["arms"]
                if arm["arm_id"] == mark_row["arm_id"]
            )
            fake_price = float(mark_row["observed_horizon_mark_price"]) + 0.01
            mark_row["observed_horizon_mark_price"] = fake_price
            mark_row["observed_horizon_mark_post_cost_pips"] = round(
                (
                    fake_price - float(arm["entry"])
                    if mark_row["side"] == "LONG"
                    else float(arm["entry"]) - fake_price
                )
                * 10_000.0,
                6,
            )
            mark_row["observation_sha256"] = _sha(
                {
                    key: value
                    for key, value in mark_row.items()
                    if key != "observation_sha256"
                }
            )
            tampered = _seal(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "contract_sha256"
                }
            )
            self.assertFalse(_episode_outcome_valid(tampered, vehicle))

            stale_member = copy.deepcopy(outcome)
            mark_row = next(
                row
                for row in stale_member["arm_observations"]
                if row["proof_exit_reason"] == "HORIZON_FULL_STOP_LOSS"
            )
            arm = next(
                arm
                for candidate in stale_member["learning_outcome"]["candidates"]
                if candidate["candidate_sha256"] == mark_row["candidate_sha256"]
                for arm in candidate["arms"]
                if arm["arm_id"] == mark_row["arm_id"]
            )
            fill_clock = datetime.fromisoformat(arm["fill_at_utc"])
            exit_clock = datetime.fromisoformat(arm["exit_at_utc"])
            eligible_receipts = [
                row
                for row in stale_member["truth_path_candles"]
                if fill_clock
                <= datetime.fromisoformat(row["timestamp_utc"])
                < exit_clock
            ]
            self.assertGreater(len(eligible_receipts), 1)
            earlier = eligible_receipts[0]
            earlier_clock = datetime.fromisoformat(earlier["timestamp_utc"])
            earlier_interval = {
                "from_utc": earlier_clock.isoformat(),
                "to_utc": (earlier_clock + timedelta(seconds=5)).isoformat(),
            }
            earlier_price = float(
                earlier["bid"][3]
                if mark_row["side"] == "LONG"
                else earlier["ask"][3]
            )
            mark_row["observed_horizon_mark_price"] = earlier_price
            mark_row["observed_horizon_mark_post_cost_pips"] = round(
                (
                    earlier_price - float(arm["entry"])
                    if mark_row["side"] == "LONG"
                    else float(arm["entry"]) - earlier_price
                )
                * 10_000.0,
                6,
            )
            mark_row["observed_horizon_mark_s5_interval_utc"] = earlier_interval
            mark_row["exit_s5_interval_utc"] = earlier_interval
            mark_row["observation_sha256"] = _sha(
                {
                    key: value
                    for key, value in mark_row.items()
                    if key != "observation_sha256"
                }
            )
            stale_member = _seal(
                {
                    key: value
                    for key, value in stale_member.items()
                    if key != "contract_sha256"
                }
            )
            self.assertFalse(_episode_outcome_valid(stale_member, vehicle))

    def test_same_s5_dual_touch_is_conservative_and_has_no_mark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            _run(
                paths,
                handoffs=[paths["handoff"]],
                now=CONFIRMED_AT,
                client_factory=lambda: (_ for _ in ()).throw(AssertionError()),
            )
            vehicle = _load_one(paths["vehicle_ledger"])
            maturity = _seat_maturity(vehicle["learning_seat"])
            fill = _flat_path(CONFIRMED_AT, CONFIRMED_AT + timedelta(seconds=5))[0]
            dual = S5BidAskCandle(
                timestamp_utc=CONFIRMED_AT + timedelta(seconds=5),
                bid_o=1.10004,
                bid_h=1.10150,
                bid_l=1.09850,
                bid_c=1.10004,
                ask_o=1.10012,
                ask_h=1.10158,
                ask_l=1.09858,
                ask_c=1.10012,
            )
            outcome = resolve_fast_bot_episode_vehicle(
                vehicle,
                [fill, dual],
                resolved_at_utc=maturity + timedelta(seconds=1),
                truth_chunk_sha256=[HASH],
            )
            self.assertTrue(outcome["arm_observations"])
            for row in outcome["arm_observations"]:
                self.assertIn("AMBIGUOUS", row["proof_exit_reason"])
                self.assertLess(row["proof_post_cost_realized_pips"], 0.0)
                self.assertIsNone(row["observed_horizon_mark_price"])
                self.assertEqual(
                    row["exit_s5_interval_semantics"],
                    "EXECUTABLE_ATTACHED_EXIT_TOUCH",
                )

    def test_client_factory_failure_after_durable_projection_stays_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            maturity = _seat_maturity(paths["shadow"]["seats"][0])
            result = _run(
                paths,
                handoffs=[paths["handoff"]],
                now=maturity + timedelta(seconds=1),
                client_factory=lambda: (_ for _ in ()).throw(
                    RuntimeError("missing credentials")
                ),
            )
            self.assertEqual(result["status"], "RESOLVED_WITH_ERRORS")
            self.assertEqual(result["vehicle_projection_status"], "VERIFIED")
            self.assertEqual(result["vehicle_ledger_appended"], 1)
            self.assertFalse(result["broker_read"])
            self.assertIn("missing credentials", result["errors"][0]["error"])
            self.assertTrue(paths["vehicle_ledger"].exists())

    def test_late_episode_is_diagnostic_and_duplicate_outcome_cannot_inflate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir), late_confirmation=True)
            _run(
                paths,
                handoffs=[paths["handoff"]],
                now=CONFIRMED_AT,
                client_factory=lambda: (_ for _ in ()).throw(AssertionError()),
            )
            vehicle = _load_one(paths["vehicle_ledger"])
            maturity = _seat_maturity(vehicle["learning_seat"])
            outcome = resolve_fast_bot_episode_vehicle(
                vehicle,
                _flat_path(CONFIRMED_AT, maturity),
                resolved_at_utc=maturity + timedelta(seconds=1),
                truth_chunk_sha256=[HASH],
            )
            scorecard = build_fast_bot_episode_scorecard(
                [vehicle],
                [outcome],
                as_of_utc=maturity + timedelta(seconds=1),
            )
            self.assertEqual(scorecard["scorecard_eligible_episode_count"], 0)
            self.assertEqual(scorecard["diagnostic_late_episode_count"], 1)
            self.assertTrue(
                all(
                    cluster["scorecard_eligible_episode_count"] == 0
                    and cluster["diagnostic_late_episode_count"] == 1
                    and cluster["proof_score"]["paired_episode_count"] == 0
                    for cluster in scorecard["clusters"]
                )
            )
            with self.assertRaisesRegex(ValueError, "multiple current outcomes"):
                build_fast_bot_episode_scorecard(
                    [vehicle],
                    [outcome, outcome],
                    as_of_utc=maturity + timedelta(seconds=1),
                )
            encoded = json.dumps(
                outcome,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            paths["outcome_ledger"].write_text(
                f"{encoded}\n{encoded}\n",
                encoding="utf-8",
            )
            conflict = _run(
                paths,
                handoffs=[],
                now=maturity + timedelta(seconds=2),
                client_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("identity conflict must precede client creation")
                ),
            )
            self.assertEqual(conflict["status"], "OUTCOME_IDENTITY_CONFLICT")
            self.assertEqual(conflict["vehicle_projection_status"], "VERIFIED")
            self.assertFalse(conflict["broker_read"])

    def test_tamper_and_lock_conflict_fail_before_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _build_fixture(Path(temp_dir))
            paths["lock"].touch()
            with paths["lock"].open("a+b") as descriptor:
                fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = _run(
                    paths,
                    handoffs=[paths["handoff"]],
                    now=CONFIRMED_AT,
                    client_factory=lambda: (_ for _ in ()).throw(AssertionError()),
                )
            self.assertEqual(locked["status"], "LOCK_BUSY")
            self.assertFalse(paths["vehicle_ledger"].exists())

            tampered = copy.deepcopy(paths["handoff"])
            tampered["broker_snapshot"]["quotes"]["EUR_USD"]["bid"] = 1.2
            # Preserve the outer seal to prove that nested SHA binding, not
            # merely JSON readability, rejects the changed snapshot.
            failed = _run(
                paths,
                handoffs=[tampered],
                now=CONFIRMED_AT,
                client_factory=lambda: (_ for _ in ()).throw(AssertionError()),
            )
            self.assertEqual(failed["vehicle_projection_status"], "FAILED")
            self.assertEqual(failed["vehicle_ledger_appended"], 0)


if __name__ == "__main__":
    unittest.main()
