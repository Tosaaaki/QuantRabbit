from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.forecast_precision import (
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
)
from quant_rabbit.models import (
    AccountSummary,
    BrokerOrder,
    BrokerPosition,
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Side,
    TradeMethod,
)
from quant_rabbit.predictive_scout import (
    PREDICTIVE_SCOUT_LIVE_ENV,
    predictive_scout_broker_signal_ids,
    predictive_scout_broker_vehicle_counts,
    predictive_scout_forward_proof,
    predictive_scout_intent_issues,
    predictive_scout_vehicle_outcome_stats,
    predictive_scout_vehicle_id,
)


class PredictiveScoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)

    def _policy(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "enabled": True,
            "mode": "FORWARD_EVIDENCE_ONLY",
            "allowed_sources": ["BIDASK_REPLAY_PRECISION"],
            "order_types": ["LIMIT"],
            "units": 1000,
            "max_concurrent": 2,
            "max_sent_per_campaign_day": 8,
            "max_ttl_minutes": 90,
            "minimum_replay_samples": 30,
            "minimum_active_days": 5,
            "minimum_profit_factor": 1.2,
            "minimum_positive_day_rate": 0.6667,
            "loss_cooldown_hours": 6,
            "quarantine_after_resolved_losses": 3,
            "quarantine_requires_negative_net": True,
            "promotion_min_resolved_exits": 30,
            "promotion_min_active_days": 5,
            "promotion_min_profit_factor": 1.2,
            "promotion_min_positive_day_rate": 0.6667,
            "promotion_one_sided_confidence": 0.95,
            "promotion_requires_all_resolved_exit_expectancy_lower_bound_positive": True,
        }

    def _metadata(self) -> dict[str, object]:
        rule_name = (
            "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_S5_"
            "BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
        )
        rule = canonical_bidask_replay_precision_rule(rule_name)
        self.assertIsNotNone(rule)
        assert rule is not None
        return {
            "predictive_scout": True,
            "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
            "predictive_scout_mode": "FORWARD_EVIDENCE_ONLY",
            "predictive_scout_hypothesis": "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN",
            "predictive_scout_vehicle_proof_status": "UNPROVEN_PASSIVE_LIMIT",
            "predictive_scout_rule_is_vehicle_proof": False,
            "predictive_scout_rule_name": rule_name,
            "predictive_scout_rule_digest": bidask_replay_precision_rule_digest(rule),
            "predictive_scout_generated_at_utc": self.now.isoformat(),
            "predictive_scout_expires_at_utc": (self.now + timedelta(minutes=45)).isoformat(),
            "predictive_scout_ttl_minutes": 45,
            "predictive_scout_promotion_allowed": False,
            "attach_take_profit_on_fill": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "broker_stop_loss_mode": "INTENT_SL",
            "desk": "failure_trader",
            "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
            "forecast_cycle_id": "test-usdcad-down-c050-065-h31-60",
            "forecast_direction": "DOWN",
            "forecast_confidence": 0.60,
            "forecast_horizon_min": 45,
            "bidask_replay_precision_seed_rule": rule,
        }

    def _intent(self, **overrides: object) -> OrderIntent:
        values: dict[str, object] = {
            "pair": "USD_CAD",
            "side": Side.LONG,
            "order_type": OrderType.LIMIT,
            "units": 1000,
            "entry": 1.3500,
            "tp": 1.3510,
            "sl": 1.3493,
            "thesis": "bounded forward evidence",
            "owner": Owner.TRADER,
            "market_context": MarketContext(
                regime="range",
                narrative="reproducible forecast failure bucket",
                chart_story="passive retest",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="broker stop",
            ),
            "metadata": self._metadata(),
        }
        values.update(overrides)
        return OrderIntent(**values)  # type: ignore[arg-type]

    @staticmethod
    def _init_ledger(path: Path) -> None:
        with sqlite3.connect(path) as con:
            con.execute(
                """
                CREATE TABLE sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            con.execute(
                "INSERT INTO sync_state(key, value, updated_at_utc) VALUES ('last_oanda_transaction_id', '100', '2026-07-10T03:00:00Z')"
            )
            con.execute(
                """
                CREATE TABLE gateway_receipts (
                    ts_utc TEXT NOT NULL,
                    sent INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE execution_events (
                    event_type TEXT NOT NULL,
                    order_id TEXT,
                    trade_id TEXT,
                    client_order_id TEXT,
                    raw_json TEXT,
                    realized_pl_jpy REAL,
                    financing_jpy REAL,
                    ts_utc TEXT,
                    exit_reason TEXT
                )
                """
            )

    def _issues(
        self,
        root: Path,
        *,
        intent: OrderIntent | None = None,
        snapshot: BrokerSnapshot | None = None,
        policy: dict[str, object] | None = None,
        validation_time: datetime | None = None,
    ) -> list[dict[str, str]]:
        data_root = root / "data"
        data_root.mkdir(exist_ok=True)
        ledger = data_root / "execution_ledger.db"
        if not ledger.exists():
            self._init_ledger(ledger)
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy or self._policy()), encoding="utf-8")
        broker = snapshot or BrokerSnapshot(fetched_at_utc=self.now)
        if broker.account is None:
            broker = replace(
                broker,
                account=AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    margin_available_jpy=200_000.0,
                    last_transaction_id="100",
                    fetched_at_utc=self.now,
                ),
            )
        with patch.dict(os.environ, {PREDICTIVE_SCOUT_LIVE_ENV: "1"}, clear=False):
            return predictive_scout_intent_issues(
                intent or self._intent(),
                snapshot=broker,
                data_root=data_root,
                validation_time_utc=validation_time or self.now,
                policy_path=policy_path,
            )

    def test_accepts_only_bounded_forward_evidence_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._issues(Path(tmp)), [])

    def test_manual_owned_same_pair_position_blocks_regardless_of_side_or_override(self) -> None:
        for owner in (Owner.MANUAL, Owner.OPERATOR_MANUAL, Owner.UNKNOWN):
            for position_side in (Side.LONG, Side.SHORT):
                with (
                    self.subTest(owner=owner, position_side=position_side),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    metadata = self._metadata()
                    metadata["operator_authorized_manual_overlap"] = True
                    snapshot = BrokerSnapshot(
                        fetched_at_utc=self.now,
                        positions=(
                            BrokerPosition(
                                trade_id=f"manual-{owner.value}-{position_side.value}",
                                pair="USD_CAD",
                                side=position_side,
                                units=10_000,
                                entry_price=1.35,
                                owner=owner,
                            ),
                        ),
                    )

                    codes = {
                        item["code"]
                        for item in self._issues(
                            Path(tmp),
                            intent=self._intent(metadata=metadata),
                            snapshot=snapshot,
                        )
                    }

                self.assertIn("PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED", codes)

    def test_manual_owned_different_pair_does_not_block_scout(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=self.now,
            positions=(
                BrokerPosition(
                    trade_id="manual-eurusd",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=10_000,
                    entry_price=1.15,
                    owner=Owner.OPERATOR_MANUAL,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), snapshot=snapshot)
            }

        self.assertNotIn("PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED", codes)

    def test_manual_owned_same_pair_pending_order_blocks_scout(self) -> None:
        for owner in (Owner.OPERATOR_MANUAL, Owner.UNKNOWN):
            with self.subTest(owner=owner), tempfile.TemporaryDirectory() as tmp:
                snapshot = BrokerSnapshot(
                    fetched_at_utc=self.now,
                    orders=(
                        BrokerOrder(
                            order_id=f"manual-pending-usdcad-{owner.value}",
                            pair="USD_CAD",
                            order_type="LIMIT",
                            owner=owner,
                        ),
                    ),
                )
                codes = {
                    item["code"]
                    for item in self._issues(Path(tmp), snapshot=snapshot)
                }

            self.assertIn("PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED", codes)

    def test_tagless_pending_order_on_different_pair_does_not_block_scout(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=self.now,
            orders=(
                BrokerOrder(
                    order_id="tagless-pending-eurusd",
                    pair="EUR_USD",
                    order_type="LIMIT",
                    owner=Owner.UNKNOWN,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), snapshot=snapshot)
            }

        self.assertNotIn("PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED", codes)

    def test_pairless_dependent_order_does_not_crash_or_block_scout(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=self.now,
            orders=(
                BrokerOrder(
                    order_id="manual-position-dependent-tp",
                    pair=None,
                    order_type="TAKE_PROFIT",
                    owner=Owner.UNKNOWN,
                    trade_id="manual-position-trade",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), snapshot=snapshot)
            }

        self.assertNotIn("PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED", codes)

    def test_environment_and_policy_are_independent_live_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            self._init_ledger(data_root / "execution_ledger.db")
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self._policy()), encoding="utf-8")
            with patch.dict(os.environ, {PREDICTIVE_SCOUT_LIVE_ENV: "0"}, clear=False):
                issues = predictive_scout_intent_issues(
                    self._intent(),
                    snapshot=BrokerSnapshot(fetched_at_utc=self.now),
                    data_root=data_root,
                    validation_time_utc=self.now,
                    policy_path=policy_path,
                )
            self.assertIn("PREDICTIVE_SCOUT_LIVE_DISABLED", {item["code"] for item in issues})

    def test_expired_or_auto_promotable_intent_is_blocked(self) -> None:
        metadata = self._metadata()
        metadata["predictive_scout_expires_at_utc"] = (self.now - timedelta(seconds=1)).isoformat()
        metadata["predictive_scout_promotion_allowed"] = True
        with tempfile.TemporaryDirectory() as tmp:
            codes = {item["code"] for item in self._issues(Path(tmp), intent=self._intent(metadata=metadata))}
        self.assertIn("PREDICTIVE_SCOUT_EXPIRED", codes)
        self.assertIn("PREDICTIVE_SCOUT_AUTO_PROMOTION_FORBIDDEN", codes)

    def test_unreadable_ledger_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "execution_ledger.db").write_text("not sqlite", encoding="utf-8")
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self._policy()), encoding="utf-8")
            with patch.dict(os.environ, {PREDICTIVE_SCOUT_LIVE_ENV: "1"}, clear=False):
                issues = predictive_scout_intent_issues(
                    self._intent(),
                    snapshot=BrokerSnapshot(fetched_at_utc=self.now),
                    data_root=data_root,
                    validation_time_utc=self.now,
                    policy_path=policy_path,
                )
        self.assertIn("PREDICTIVE_SCOUT_LEDGER_UNAVAILABLE", {item["code"] for item in issues})

    def test_broker_and_execution_ledger_transaction_ids_must_match(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=self.now,
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_available_jpy=200_000.0,
                last_transaction_id="999",
                fetched_at_utc=self.now,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), snapshot=snapshot)
            }
        self.assertIn("PREDICTIVE_SCOUT_LEDGER_NOT_CURRENT", codes)

    def test_daily_send_and_concurrent_caps_are_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            ledger = data_root / "execution_ledger.db"
            self._init_ledger(ledger)
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    "INSERT INTO gateway_receipts(ts_utc, sent, payload_json) VALUES (?, 1, ?)",
                    [
                        (self.now.isoformat(), json.dumps({"predictive_scout": True, "n": n}))
                        for n in range(8)
                    ],
                )
            active = BrokerOrder(
                order_id="scout-1",
                pair="USD_CAD",
                order_type="LIMIT",
                owner=Owner.TRADER,
                raw={
                    "clientExtensions": {
                        "comment": "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT"
                    }
                },
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=self.now,
                orders=(active, replace(active, order_id="scout-2")),
            )
            codes = {item["code"] for item in self._issues(root, snapshot=snapshot)}
        self.assertIn("PREDICTIVE_SCOUT_DAILY_CAP_REACHED", codes)
        self.assertIn("PREDICTIVE_SCOUT_CONCURRENT_CAP_REACHED", codes)

    def test_broker_vehicle_counts_reconcile_claims_without_hiding_unknown_scout(self) -> None:
        known = BrokerOrder(
            order_id="known-scout",
            pair="USD_CAD",
            order_type="LIMIT",
            owner=Owner.TRADER,
            raw={
                "clientExtensions": {
                    "id": "qrv1-USDCAD-L-test-pss-known",
                    "comment": (
                        "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT "
                        "vehicle=psv-known lane=scout"
                    )
                }
            },
        )
        unknown = replace(
            known,
            order_id="unknown-scout",
            raw={
                "clientExtensions": {
                    "comment": "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT"
                }
            },
        )
        filled_order_shadow = replace(known, order_id="filled-shadow", trade_id="trade-1")
        snapshot = BrokerSnapshot(
            fetched_at_utc=self.now,
            orders=(known, unknown, filled_order_shadow),
        )

        self.assertEqual(
            predictive_scout_broker_vehicle_counts(snapshot),
            {"psv-known": 1},
        )
        self.assertEqual(
            predictive_scout_broker_signal_ids(snapshot),
            {"pss-known"},
        )

    def test_policy_cannot_raise_hard_caps(self) -> None:
        policy = self._policy()
        policy["max_concurrent"] = 3
        policy["max_sent_per_campaign_day"] = 10
        with tempfile.TemporaryDirectory() as tmp:
            codes = {item["code"] for item in self._issues(Path(tmp), policy=policy)}
        self.assertIn("PREDICTIVE_SCOUT_POLICY_INVALID", codes)

    def test_resolved_net_loss_starts_exact_vehicle_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            ledger = data_root / "execution_ledger.db"
            self._init_ledger(ledger)
            intent = self._intent()
            vehicle_id = predictive_scout_vehicle_id(intent)
            gateway_payload = {
                "predictive_scout": True,
                "predictive_scout_receipt": {
                    "predictive_scout": True,
                    "predictive_scout_vehicle_id": vehicle_id,
                },
            }
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    "INSERT INTO execution_events(event_type, order_id, trade_id, raw_json, realized_pl_jpy, financing_jpy, ts_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("GATEWAY_ORDER_SENT", "o-1", None, json.dumps(gateway_payload), None, None, (self.now - timedelta(hours=2)).isoformat()),
                        ("ORDER_FILLED", "o-1", "t-1", "{}", None, None, (self.now - timedelta(hours=2)).isoformat()),
                        ("TRADE_REDUCED", None, "t-1", "{}", 100.0, -120.0, (self.now - timedelta(hours=1, minutes=5)).isoformat()),
                        ("TRADE_CLOSED", None, "t-1", "{}", 40.0, -30.0, (self.now - timedelta(hours=1)).isoformat()),
                    ],
                )
            codes = {item["code"] for item in self._issues(root, intent=intent)}
        self.assertIn("PREDICTIVE_SCOUT_VEHICLE_LOSS_COOLDOWN", codes)
        self.assertNotIn("PREDICTIVE_SCOUT_VEHICLE_QUARANTINED_NEGATIVE_NET", codes)

    def test_single_loss_does_not_permanently_stop_forward_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            ledger = data_root / "execution_ledger.db"
            self._init_ledger(ledger)
            intent = self._intent()
            receipt = {
                "predictive_scout": True,
                "predictive_scout_vehicle_id": predictive_scout_vehicle_id(intent),
            }
            closed_at = self.now - timedelta(hours=7)
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    "INSERT INTO execution_events(event_type, order_id, trade_id, raw_json, realized_pl_jpy, financing_jpy, ts_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("GATEWAY_ORDER_SENT", "o-1", None, json.dumps(receipt), None, None, closed_at.isoformat()),
                        ("ORDER_FILLED", "o-1", "t-1", "{}", None, None, closed_at.isoformat()),
                        ("TRADE_CLOSED", None, "t-1", "{}", -100.0, 0.0, closed_at.isoformat()),
                    ],
                )
            codes = {item["code"] for item in self._issues(root, intent=intent)}
        self.assertNotIn("PREDICTIVE_SCOUT_VEHICLE_LOSS_COOLDOWN", codes)
        self.assertNotIn("PREDICTIVE_SCOUT_VEHICLE_QUARANTINED_NEGATIVE_NET", codes)

    def test_three_losses_with_negative_net_quarantine_vehicle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            ledger = data_root / "execution_ledger.db"
            self._init_ledger(ledger)
            intent = self._intent()
            receipt = {
                "predictive_scout": True,
                "predictive_scout_vehicle_id": predictive_scout_vehicle_id(intent),
            }
            rows: list[tuple[object, ...]] = []
            for index in range(3):
                ts = (self.now - timedelta(hours=10 - index)).isoformat()
                rows.extend(
                    [
                        ("GATEWAY_ORDER_SENT", f"o-{index}", None, json.dumps(receipt), None, None, ts),
                        ("ORDER_FILLED", f"o-{index}", f"t-{index}", "{}", None, None, ts),
                        ("TRADE_CLOSED", None, f"t-{index}", "{}", -100.0, 0.0, ts),
                    ]
                )
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    "INSERT INTO execution_events(event_type, order_id, trade_id, raw_json, realized_pl_jpy, financing_jpy, ts_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            codes = {item["code"] for item in self._issues(root, intent=intent)}
        self.assertIn("PREDICTIVE_SCOUT_VEHICLE_QUARANTINED_NEGATIVE_NET", codes)

    def test_post_reservation_reconciles_fill_by_client_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "execution_ledger.db"
            self._init_ledger(ledger)
            intent = self._intent()
            reservation = {
                "predictive_scout": True,
                "predictive_scout_receipt": {
                    "predictive_scout": True,
                    "predictive_scout_vehicle_id": predictive_scout_vehicle_id(intent),
                    "predictive_scout_experiment_id": "psx-reserved",
                },
            }
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    """
                    INSERT INTO execution_events(
                        event_type, order_id, trade_id, client_order_id, raw_json,
                        realized_pl_jpy, financing_jpy, ts_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "GATEWAY_ORDER_STAGED",
                            None,
                            None,
                            "qrv1-reserved",
                            json.dumps(reservation),
                            None,
                            None,
                            (self.now - timedelta(hours=2)).isoformat(),
                        ),
                        (
                            "ORDER_FILLED",
                            "o-reserved",
                            "t-reserved",
                            "qrv1-reserved",
                            "{}",
                            None,
                            None,
                            (self.now - timedelta(hours=2)).isoformat(),
                        ),
                        (
                            "TRADE_CLOSED",
                            None,
                            "t-reserved",
                            None,
                            "{}",
                            -50.0,
                            -1.0,
                            (self.now - timedelta(hours=1)).isoformat(),
                        ),
                    ],
                )

            stats = predictive_scout_vehicle_outcome_stats(ledger, intent=intent)

        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats["loss_count"], 1)
        self.assertEqual(stats["net_jpy"], -51.0)

    def test_policy_cannot_lower_statistical_floors_or_raise_bad_types(self) -> None:
        for key, value in (
            ("minimum_replay_samples", 1),
            ("minimum_active_days", 1),
            ("minimum_profit_factor", 1.0),
            ("minimum_positive_day_rate", 0.1),
            ("units", "bad"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                policy = self._policy()
                policy[key] = value
                codes = {item["code"] for item in self._issues(Path(tmp), policy=policy)}
                self.assertIn("PREDICTIVE_SCOUT_POLICY_INVALID", codes)

    def test_ttl_cannot_exceed_current_forecast_horizon(self) -> None:
        metadata = self._metadata()
        metadata["forecast_horizon_min"] = 31
        metadata["predictive_scout_expires_at_utc"] = (self.now + timedelta(minutes=45)).isoformat()
        metadata["predictive_scout_ttl_minutes"] = 45
        with tempfile.TemporaryDirectory() as tmp:
            codes = {item["code"] for item in self._issues(Path(tmp), intent=self._intent(metadata=metadata))}
        self.assertIn("PREDICTIVE_SCOUT_TTL_TOO_LONG", codes)

    def test_exit_shape_cannot_mint_a_new_vehicle_after_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = self._intent(tp=1.35095, sl=1.34929)
            codes = {item["code"] for item in self._issues(root, intent=intent)}
        self.assertIn("PREDICTIVE_SCOUT_CANONICAL_GEOMETRY_MISMATCH", codes)

    def test_updated_evidence_statistics_do_not_reset_failure_memory_vehicle(self) -> None:
        original = self._intent()
        metadata = dict(original.metadata)
        rule = dict(metadata["bidask_replay_precision_seed_rule"])
        rule["samples"] = int(rule["samples"]) + 1
        rule["optimized_profit_factor"] = float(rule["optimized_profit_factor"]) + 0.25
        metadata["bidask_replay_precision_seed_rule"] = rule
        metadata["predictive_scout_rule_digest"] = bidask_replay_precision_rule_digest(rule)
        updated = replace(original, metadata=metadata)

        self.assertNotEqual(
            original.metadata["predictive_scout_rule_digest"],
            updated.metadata["predictive_scout_rule_digest"],
        )
        self.assertEqual(
            predictive_scout_vehicle_id(original),
            predictive_scout_vehicle_id(updated),
        )

    def test_material_selector_change_creates_new_failure_memory_vehicle(self) -> None:
        original = self._intent()
        metadata = dict(original.metadata)
        rule = dict(metadata["bidask_replay_precision_seed_rule"])
        rule["confidence_bucket"] = "0.65-0.80"
        metadata["bidask_replay_precision_seed_rule"] = rule
        changed = replace(original, metadata=metadata)

        self.assertNotEqual(
            predictive_scout_vehicle_id(original),
            predictive_scout_vehicle_id(changed),
        )

    def test_broker_visible_scout_role_is_required(self) -> None:
        metadata = self._metadata()
        metadata.pop("campaign_role")
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), intent=self._intent(metadata=metadata))
            }
        self.assertIn("PREDICTIVE_SCOUT_ROLE_REQUIRED", codes)

    def test_method_or_desk_relabel_cannot_reset_failure_vehicle(self) -> None:
        metadata = self._metadata()
        metadata["desk"] = "range_trader"
        relabeled = self._intent(
            metadata=metadata,
            market_context=replace(
                self._intent().market_context,
                method=TradeMethod.RANGE_ROTATION,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            codes = {
                item["code"]
                for item in self._issues(Path(tmp), intent=relabeled)
            }

        self.assertIn("PREDICTIVE_SCOUT_METHOD_REQUIRED", codes)
        self.assertIn("PREDICTIVE_SCOUT_DESK_REQUIRED", codes)

    def test_scout_marker_cannot_be_stripped_to_downgrade_the_lane(self) -> None:
        for strip_mode in ("marker_only", "reserved", "reserved_and_rule"):
            with self.subTest(strip_mode=strip_mode):
                metadata = self._metadata()
                if strip_mode != "marker_only":
                    metadata = {
                        key: value
                        for key, value in metadata.items()
                        if not key.startswith("predictive_scout")
                    }
                    metadata["campaign_role"] = "NOW"
                else:
                    metadata.pop("predictive_scout")
                if strip_mode == "reserved_and_rule":
                    metadata.pop("bidask_replay_precision_seed_rule")
                with tempfile.TemporaryDirectory() as tmp:
                    codes = {
                        item["code"]
                        for item in self._issues(
                            Path(tmp),
                            intent=self._intent(metadata=metadata),
                        )
                    }
                self.assertIn("PREDICTIVE_SCOUT_MARKER_REQUIRED", codes)

    def test_duplicate_broker_orders_for_one_signal_are_proof_anomaly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            self._init_ledger(ledger)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self._policy()), encoding="utf-8")
            intent = self._intent()
            receipt = {
                "predictive_scout": True,
                "predictive_scout_vehicle_id": predictive_scout_vehicle_id(intent),
                "predictive_scout_signal_id": "pss-same-signal",
                "predictive_scout_experiment_id": "psx-same-experiment",
                "pair": intent.pair,
                "side": intent.side.value,
            }
            rows: list[tuple[object, ...]] = []
            for index in range(2):
                order_id = f"duplicate-order-{index}"
                trade_id = f"duplicate-trade-{index}"
                raw = json.dumps(
                    {"predictive_scout": True, "predictive_scout_receipt": receipt}
                )
                rows.extend(
                    [
                        ("GATEWAY_ORDER_SENT", order_id, None, raw, None, None, self.now.isoformat(), None),
                        ("ORDER_FILLED", order_id, trade_id, "{}", None, None, self.now.isoformat(), None),
                        ("TRADE_CLOSED", None, trade_id, "{}", 100.0, 0.0, self.now.isoformat(), "TAKE_PROFIT_ORDER"),
                    ]
                )
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    "INSERT INTO execution_events(event_type, order_id, trade_id, raw_json, realized_pl_jpy, financing_jpy, ts_utc, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

            proof = predictive_scout_forward_proof(ledger, policy_path=policy_path)

        vehicle = proof["vehicles"][0]
        self.assertEqual(vehicle["sent_count"], 1)
        self.assertEqual(vehicle["duplicate_signal_count"], 1)
        self.assertFalse(vehicle["statistically_eligible_for_operator_review"])

    def test_forward_proof_requires_all_resolved_exits_and_positive_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            self._init_ledger(ledger)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self._policy()), encoding="utf-8")
            intent = self._intent()
            vehicle_id = predictive_scout_vehicle_id(intent)
            receipt = {
                "predictive_scout": True,
                "predictive_scout_vehicle_id": vehicle_id,
                "predictive_scout_rule_digest": intent.metadata["predictive_scout_rule_digest"],
                "predictive_scout_rule_name": intent.metadata["predictive_scout_rule_name"],
                "pair": intent.pair,
                "side": intent.side.value,
            }
            rows: list[tuple[object, ...]] = []
            for index in range(30):
                order_id = f"o-{index}"
                trade_id = f"t-{index}"
                day = f"2026-07-{1 + index % 5:02d}T03:00:00+00:00"
                raw = json.dumps(
                    {
                        "predictive_scout": True,
                        "predictive_scout_receipt": {
                            **receipt,
                            "predictive_scout_signal_id": f"pss-independent-{index}",
                            "predictive_scout_experiment_id": f"psx-independent-{index}",
                        },
                    }
                )
                rows.extend(
                    [
                        ("GATEWAY_ORDER_SENT", order_id, None, raw, None, None, day, None),
                        ("ORDER_FILLED", order_id, trade_id, "{}", None, None, day, None),
                        ("TRADE_CLOSED", None, trade_id, "{}", 100.0, -1.0, day, "TAKE_PROFIT_ORDER"),
                    ]
                )
            with sqlite3.connect(ledger) as con:
                con.executemany(
                    """
                    INSERT INTO execution_events(
                        event_type, order_id, trade_id, raw_json, realized_pl_jpy,
                        financing_jpy, ts_utc, exit_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            proof = predictive_scout_forward_proof(ledger, policy_path=policy_path)

        self.assertEqual(proof["status"], "PROOF_ELIGIBLE_FOR_OPERATOR_REVIEW")
        self.assertFalse(proof["promotion_allowed"])
        vehicle = proof["vehicles"][0]
        self.assertEqual(vehicle["resolved_count"], 30)
        self.assertEqual(vehicle["unresolved_filled_count"], 0)
        self.assertEqual(vehicle["independent_signal_count"], 30)
        self.assertTrue(vehicle["complete_signal_attribution"])
        self.assertEqual(vehicle["net_jpy"], 2970.0)
        self.assertGreater(vehicle["one_sided_95_mean_lower_jpy"], 0.0)
        self.assertTrue(vehicle["statistically_eligible_for_operator_review"])
        self.assertEqual(vehicle["exit_reason_counts"], {"TAKE_PROFIT_ORDER": 30})


if __name__ == "__main__":
    unittest.main()
