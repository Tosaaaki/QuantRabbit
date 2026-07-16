from __future__ import annotations

import json
import tempfile
import unittest
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot import (
    AI_SUPERVISION_CONTRACT,
    ENTRY_ARM_SPREAD_FRACTIONS,
    ENTRY_EXPERIMENT_CONTRACT,
    HORIZON_LANE,
    REGIME_CONTRACT,
    _append_signals_once,
    _entry_experiment_arms,
    build_fast_bot_shadow,
    build_hierarchical_regime_contract,
    run_fast_bot_shadow,
)
from quant_rabbit.guardian_observation import CURRENT_M1_CONTRACT
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}


def _candles(timeframe: str, *, failed_break_short: bool = False) -> list[dict]:
    minutes = TF_MINUTES[timeframe]
    rows = []
    for index in range(21):
        started = NOW - timedelta(minutes=(21 - index) * minutes)
        rows.append(
            {
                "t": started.isoformat(),
                "o": 1.105,
                "h": 1.110,
                "l": 1.100,
                "c": 1.105,
                "complete": True,
            }
        )
    if failed_break_short:
        rows[-1] = {
            "t": (NOW - timedelta(minutes=minutes)).isoformat(),
            "o": 1.109,
            "h": 1.111,
            "l": 1.104,
            "c": 1.109,
            "complete": True,
        }
    return rows


def _view(
    timeframe: str,
    *,
    direction: str,
    phase: str,
    readiness: str = "TRIGGERED",
    location: str = "MIDDLE_THIRD",
    value_zone: str = "FAIR_VALUE",
    extension: str = "BALANCED",
    failed_break_short: bool = False,
) -> dict:
    return {
        "granularity": timeframe,
        "recent_candles": _candles(timeframe, failed_break_short=failed_break_short),
        "candle_integrity": {"forecast_blocking": False},
        "indicators": {"atr_pips": 5.0},
        "market_state": {
            "direction": direction,
            "phase": phase,
            "readiness": readiness,
            "trigger": "BREAKOUT_CLOSE",
            "structure": "BREAKOUT_ACTIVE",
            "location": location,
            "value_zone": value_zone,
            "extension": extension,
            "evidence_complete": True,
        },
    }


def _inputs(*, failed_break_short: bool = False) -> tuple[dict, dict, dict]:
    fast = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    _view("M1", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND"),
                    _view("M5", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND", failed_break_short=failed_break_short),
                    _view("M15", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND"),
                ],
            }
        ],
    }
    slow = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    _view(tf, direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short and tf == "M30" else "TREND")
                    for tf in ("M30", "H1", "H4", "D")
                ],
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


def _row(contract: dict, *, side: str, method: str) -> dict:
    return next(
        item
        for item in contract["rows"]
        if item["pair"] == "EUR_USD" and item["side"] == side and item["method"] == method
    )


def _seal_contract(body: dict) -> dict:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**body, "contract_sha256": hashlib.sha256(raw).hexdigest()}


class FastBotTest(unittest.TestCase):
    def test_passive_entry_arms_never_round_onto_opposite_quote(self) -> None:
        long_arms = _entry_experiment_arms(
            pair="EUR_USD",
            side="LONG",
            bid=1.10000,
            ask=1.10001,
            tp_pips=3.0,
            sl_pips=3.0,
        )
        short_arms = _entry_experiment_arms(
            pair="EUR_USD",
            side="SHORT",
            bid=1.10000,
            ask=1.10001,
            tp_pips=3.0,
            sl_pips=3.0,
        )

        self.assertTrue(all(arm["entry"] < 1.10001 for arm in long_arms))
        self.assertTrue(all(arm["entry"] > 1.10000 for arm in short_arms))
        self.assertTrue(all(arm["entry"] == 1.10000 for arm in long_arms))
        self.assertTrue(all(arm["entry"] == 1.10001 for arm in short_arms))

    def test_hierarchical_trend_gate_is_bot_owned_and_go(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "GO")
        self.assertTrue(trend["execution_enabled"])
        self.assertEqual(contract["entry_decision_authority"], "DETERMINISTIC_BOT")
        self.assertFalse(contract["ai_per_trade_approval_required"])
        self.assertEqual(
            contract["timeframe_roles"],
            {
                "execution": ["M1"],
                "operating": ["M5", "M15", "M30"],
                "structure": ["H1", "H4"],
                "anchor": ["D"],
            },
        )

    def test_future_packet_snapshot_and_pair_quote_fail_closed(self) -> None:
        cases = (
            ("fast", "FAST_CHART_PACKET_STALE"),
            ("snapshot", "BROKER_SNAPSHOT_OR_QUOTES_STALE"),
            ("quote", "PAIR_QUOTE_STALE_OR_FUTURE"),
        )
        for target, blocker in cases:
            with self.subTest(target=target):
                fast, slow, snapshot = _inputs()
                if target == "fast":
                    fast["generated_at_utc"] = (NOW + timedelta(seconds=1)).isoformat()
                elif target == "snapshot":
                    snapshot["fetched_at_utc"] = (NOW + timedelta(seconds=1)).isoformat()
                else:
                    snapshot["quotes"]["EUR_USD"]["timestamp_utc"] = (
                        NOW + timedelta(seconds=1)
                    ).isoformat()
                contract = build_hierarchical_regime_contract(
                    fast_pair_charts=fast,
                    slow_pair_charts=slow,
                    broker_snapshot=snapshot,
                    guardian_events={"events": []},
                    now_utc=NOW,
                )
                trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
                self.assertEqual(trend["state"], "STOP")
                self.assertIn(blocker, trend["hard_blockers"])

    def test_blocked_all_pair_current_keeps_exact_28_stop_surface(self) -> None:
        _, slow, snapshot = _inputs()
        blocked = _seal_contract(
            {
                "contract": CURRENT_M1_CONTRACT,
                "schema_version": 1,
                "status": "BLOCKED",
                "configured_pairs": [],
                "charts": [],
            }
        )

        contract = build_hierarchical_regime_contract(
            fast_pair_charts=blocked,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        self.assertEqual(len(contract["rows"]), 28 * 2 * 3)
        self.assertEqual({row["pair"] for row in contract["rows"]}, set(DEFAULT_TRADER_PAIRS))
        self.assertTrue(all(row["state"] == "STOP" for row in contract["rows"]))
        eur_trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertIn("FAST_CHART_PACKET_STALE", eur_trend["hard_blockers"])
        self.assertIn("FAST_TIMEFRAME_EVIDENCE_MISSING:M1,M5,M15", eur_trend["hard_blockers"])

    def test_arm_pips_are_recomputed_after_broker_tick_rounding(self) -> None:
        for pair, bid, ask in (
            ("EUR_USD", 1.10000, 1.10008),
            ("USD_JPY", 150.000, 150.008),
        ):
            with self.subTest(pair=pair):
                arms = _entry_experiment_arms(
                    pair=pair,
                    side="LONG",
                    bid=bid,
                    ask=ask,
                    tp_pips=6.05,
                    sl_pips=3.05,
                )
                pip_factor = 100 if pair.endswith("_JPY") else 10000
                for arm in arms:
                    self.assertAlmostEqual(
                        arm["take_profit_pips"],
                        abs(arm["take_profit"] - arm["entry"]) * pip_factor,
                        places=6,
                    )
                    self.assertAlmostEqual(
                        arm["stop_loss_pips"],
                        abs(arm["entry"] - arm["stop_loss"]) * pip_factor,
                        places=6,
                    )

    def test_technical_stale_event_stops_fast_entry_and_wakes_ai_only_for_change(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={
                "events": [
                    {"event_id": "stale", "event_type": "TECHNICAL_INPUT_STALE", "pair": "EUR_USD"},
                    {"event_id": "change", "event_type": "TECHNICAL_STATE_CHANGE", "pair": "EUR_USD"},
                ]
            },
            ai_supervision={"last_tuned_at_utc": (NOW - timedelta(hours=1)).isoformat()},
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "STOP")
        self.assertIn("TECHNICAL_INPUT_STALE", trend["hard_blockers"])
        self.assertTrue(contract["ai_wake_required"])
        self.assertIn("GUARDIAN_EVENT:TECHNICAL_STATE_CHANGE:EUR_USD", contract["ai_wake_reasons"])

    def test_breakout_failure_binds_exact_m5_side(self) -> None:
        fast, slow, snapshot = _inputs(failed_break_short=True)
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        short = _row(contract, side="SHORT", method="BREAKOUT_FAILURE")
        long = _row(contract, side="LONG", method="BREAKOUT_FAILURE")
        self.assertEqual(short["failed_break_direction"], "SHORT")
        self.assertEqual(short["state"], "GO")
        self.assertEqual(long["state"], "STOP")
        self.assertIn("M5_FAILED_BREAK_DIRECTION_NOT_BOUND_TO_SIDE", long["hard_blockers"])

    def test_breakout_failure_uses_retained_m5_after_fast_m1_split(self) -> None:
        fast, slow, snapshot = _inputs(failed_break_short=True)
        fast_views = fast["charts"][0]["views"]
        retained_m5 = next(view for view in fast_views if view["granularity"] == "M5")
        fast["charts"][0]["views"] = [
            view for view in fast_views if view["granularity"] != "M5"
        ]
        slow["charts"][0]["views"].insert(0, retained_m5)

        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        short = _row(contract, side="SHORT", method="BREAKOUT_FAILURE")
        self.assertEqual(short["failed_break_direction"], "SHORT")
        self.assertEqual(short["state"], "GO")

    def test_ai_regime_stop_is_pair_level_not_trade_approval(self) -> None:
        fast, slow, snapshot = _inputs()
        supervision = _seal_contract({
            "contract": AI_SUPERVISION_CONTRACT,
            "schema_version": 1,
            "last_tuned_at_utc": NOW.isoformat(),
            "ai_role": "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
            "ai_order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
            "pairs": {
                "EUR_USD": {
                    "mode": "STOP",
                    "reason": "material volatility transition",
                    "expires_at_utc": (NOW + timedelta(minutes=30)).isoformat(),
                }
            },
        })
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision=supervision,
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "STOP")
        self.assertIn("AI_REGIME_SUPERVISOR_STOP", trend["hard_blockers"])
        self.assertFalse(contract["ai_per_trade_approval_required"])

    def test_sealed_ai_supervision_with_order_authority_is_ignored(self) -> None:
        fast, slow, snapshot = _inputs()
        supervision = _seal_contract({
            "contract": AI_SUPERVISION_CONTRACT,
            "schema_version": 1,
            "last_tuned_at_utc": NOW.isoformat(),
            "ai_role": "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
            "ai_order_authority": "LIVE",
            "live_permission": True,
            "broker_mutation_allowed": True,
            "pairs": {
                "EUR_USD": {
                    "mode": "STOP",
                    "reason": "must not be accepted",
                    "expires_at_utc": (NOW + timedelta(minutes=30)).isoformat(),
                }
            },
        })
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision=supervision,
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "GO")
        self.assertEqual(trend["ai_supervision"]["mode"], "UNSUPERVISED")
        self.assertTrue(contract["tuning_due"])

    def test_expired_stop_survives_only_the_scheduled_handoff_grace(self) -> None:
        fast, slow, snapshot = _inputs()

        def supervision(expires_at: datetime) -> dict:
            return _seal_contract({
                "contract": AI_SUPERVISION_CONTRACT,
                "schema_version": 1,
                "last_tuned_at_utc": (NOW - timedelta(hours=6)).isoformat(),
                "ai_role": "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
                "ai_order_authority": "NONE",
                "live_permission": False,
                "broker_mutation_allowed": False,
                "pairs": {
                    "EUR_USD": {
                        "mode": "STOP",
                        "reason": "material volatility transition",
                        "expires_at_utc": expires_at.isoformat(),
                    }
                },
            })

        in_grace = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision=supervision(NOW - timedelta(minutes=5)),
            now_utc=NOW,
        )
        after_grace = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision=supervision(NOW - timedelta(minutes=16)),
            now_utc=NOW,
        )

        grace_row = _row(in_grace, side="LONG", method="TREND_CONTINUATION")
        expired_row = _row(after_grace, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(grace_row["ai_supervision"]["mode"], "STOP")
        self.assertIn("SCHEDULED_SUPERVISOR_HANDOFF_GRACE", grace_row["ai_supervision"]["reason"])
        self.assertIn("AI_REGIME_SUPERVISOR_STOP", grace_row["hard_blockers"])
        self.assertEqual(expired_row["ai_supervision"]["mode"], "UNSUPERVISED")
        self.assertNotIn("AI_REGIME_SUPERVISOR_STOP", expired_row["hard_blockers"])

    def test_unsealed_ai_supervision_cannot_stop_or_reset_tuning_clock(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision={
                "contract": AI_SUPERVISION_CONTRACT,
                "last_tuned_at_utc": NOW.isoformat(),
                "pairs": {
                    "EUR_USD": {
                        "mode": "STOP",
                        "expires_at_utc": (NOW + timedelta(minutes=30)).isoformat(),
                    }
                },
            },
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "GO")
        self.assertEqual(trend["ai_supervision"]["mode"], "UNSUPERVISED")
        self.assertTrue(contract["tuning_due"])

    def test_shadow_signal_has_no_live_permission_and_ledger_dedupes(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "fast": root / "fast.json",
                "slow": root / "slow.json",
                "snapshot": root / "snapshot.json",
                "events": root / "events.json",
                "regime": root / "regime.json",
                "shadow": root / "shadow.json",
                "ledger": root / "shadow.jsonl",
                "report": root / "report.md",
            }
            for key, value in (("fast", fast), ("slow", slow), ("snapshot", snapshot), ("events", {"events": []})):
                paths[key].write_text(json.dumps(value), encoding="utf-8")
            kwargs = dict(
                fast_pair_charts_path=paths["fast"],
                slow_pair_charts_path=paths["slow"],
                broker_snapshot_path=paths["snapshot"],
                guardian_events_path=paths["events"],
                ai_supervision_path=root / "missing-ai.json",
                regime_output_path=paths["regime"],
                shadow_output_path=paths["shadow"],
                shadow_ledger_path=paths["ledger"],
                report_path=paths["report"],
                now_utc=NOW,
            )
            first = run_fast_bot_shadow(**kwargs)
            fast_later = {**fast, "generated_at_utc": (NOW + timedelta(seconds=30)).isoformat()}
            snapshot_later = {
                **snapshot,
                "fetched_at_utc": (NOW + timedelta(seconds=30)).isoformat(),
                "quotes": {
                    "EUR_USD": {
                        **snapshot["quotes"]["EUR_USD"],
                        "bid": 1.10001,
                        "ask": 1.10009,
                        "timestamp_utc": (NOW + timedelta(seconds=30)).isoformat(),
                    }
                },
            }
            paths["fast"].write_text(json.dumps(fast_later), encoding="utf-8")
            paths["snapshot"].write_text(json.dumps(snapshot_later), encoding="utf-8")
            second = run_fast_bot_shadow(
                **{**kwargs, "now_utc": NOW + timedelta(seconds=30)}
            )
            shadow = json.loads(paths["shadow"].read_text())
            ledger_rows = [json.loads(line) for line in paths["ledger"].read_text().splitlines()]

        self.assertEqual(first["ledger_appended"], 1)
        self.assertEqual(second["ledger_appended"], 0)
        self.assertEqual(len(ledger_rows), 1)
        self.assertEqual(shadow["signals"][0]["signal_id"], ledger_rows[0]["signal_id"])
        self.assertFalse(shadow["live_permission"])
        self.assertFalse(shadow["broker_mutation_allowed"])
        self.assertFalse(shadow["ai_per_trade_approval_required"])
        signal = shadow["signals"][0]
        arms = signal["entry_experiment_arms"]
        self.assertEqual(signal["schema_version"], 3)
        self.assertEqual(signal["horizon_lane"], HORIZON_LANE)
        self.assertEqual(signal["entry_experiment_contract"], ENTRY_EXPERIMENT_CONTRACT)
        self.assertEqual(
            [(arm["arm_id"], arm["spread_fraction_toward_market"]) for arm in arms],
            list(ENTRY_ARM_SPREAD_FRACTIONS),
        )
        self.assertEqual(signal["entry"], arms[0]["entry"])
        self.assertEqual(signal["take_profit"], arms[0]["take_profit"])
        self.assertEqual(signal["stop_loss"], arms[0]["stop_loss"])
        self.assertEqual(signal["quote_bid"], arms[0]["entry"])
        self.assertTrue(
            all(
                signal["quote_bid"] <= arm["entry"] < signal["quote_ask"]
                for arm in arms
            )
        )
        self.assertEqual(len(signal["signal_sha256"]), 64)
        self.assertFalse(signal["broker_mutation_allowed"])
        self.assertGreater(shadow["signals"][0]["take_profit"], shadow["signals"][0]["entry"])

    def test_shadow_preserves_every_go_side_method_pair_and_horizon_identity(self) -> None:
        rows = [
            {
                "pair": pair,
                "side": side,
                "method": method,
                "state": "GO",
                "execution_enabled": True,
                "score": score,
                "m1_closed_candle_utc": NOW.isoformat(),
                "m5_atr_pips": 5.0,
            }
            for pair, side, method, score in (
                ("EUR_USD", "LONG", "TREND_CONTINUATION", 6.0),
                ("EUR_USD", "SHORT", "RANGE_ROTATION", 5.0),
                ("EUR_USD", "LONG", "BREAKOUT_FAILURE", 7.0),
                ("GBP_USD", "LONG", "TREND_CONTINUATION", 4.0),
            )
        ]
        regime = _seal_contract(
            {
                "contract": REGIME_CONTRACT,
                "schema_version": 1,
                "generated_at_utc": NOW.isoformat(),
                "rows": rows,
            }
        )
        snapshot = {
            "fetched_at_utc": NOW.isoformat(),
            "quotes": {
                pair: {
                    "bid": bid,
                    "ask": ask,
                    "timestamp_utc": NOW.isoformat(),
                }
                for pair, bid, ask in (
                    ("EUR_USD", 1.10000, 1.10008),
                    ("GBP_USD", 1.30000, 1.30008),
                )
            },
        }

        shadow = build_fast_bot_shadow(regime, broker_snapshot=snapshot, now_utc=NOW)

        self.assertEqual(len(shadow["signals"]), 4)
        identities = {
            (
                signal["pair"],
                signal["side"],
                signal["method"],
                signal["horizon_lane"],
            )
            for signal in shadow["signals"]
        }
        self.assertEqual(len(identities), 4)
        self.assertEqual(len({signal["signal_id"] for signal in shadow["signals"]}), 4)
        self.assertEqual(shadow["candidate_projection"], "ALL_GO_ROWS_NO_PAIR_OR_SIDE_NETTING")
        self.assertEqual(shadow["candidate_count_by_horizon_lane"], {HORIZON_LANE: 4})
        self.assertTrue(all(signal["schema_version"] == 3 for signal in shadow["signals"]))
        self.assertTrue(all(signal["live_permission"] is False for signal in shadow["signals"]))
        self.assertIn(
            "HORIZON_AWARE_MULTI_GO_PORTFOLIO_FORWARD_PROOF_REQUIRED",
            shadow["promotion_contract"]["blockers"],
        )
        self.assertNotIn(
            "OVERLAPPING_AI_TRADER_ENTRY_AUTHORITY_RETIREMENT_REQUIRED",
            shadow["promotion_contract"]["blockers"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "shadow.jsonl"
            self.assertEqual(_append_signals_once(ledger, shadow), 4)
            self.assertEqual(_append_signals_once(ledger, shadow), 0)
            self.assertEqual(len(ledger.read_text().splitlines()), 4)

        legacy_source = shadow["signals"][0]
        legacy_body = {
            key: value
            for key, value in legacy_source.items()
            if key not in {"signal_sha256", "identity_contract", "horizon_lane"}
        }
        legacy_body["schema_version"] = 2
        legacy_identity = {
            "pair": legacy_body["pair"],
            "m1_closed_candle_utc": legacy_body["m1_closed_candle_utc"],
        }
        legacy_body["signal_id"] = hashlib.sha256(
            json.dumps(
                legacy_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        legacy = {
            **legacy_body,
            "signal_sha256": hashlib.sha256(
                json.dumps(
                    legacy_body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "shadow.jsonl"
            self.assertEqual(_append_signals_once(ledger, {"signals": [legacy]}), 1)
            self.assertEqual(
                _append_signals_once(ledger, {"signals": [legacy_source]}),
                1,
            )
            self.assertEqual(len(ledger.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
