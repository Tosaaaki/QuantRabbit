from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import quant_rabbit.broker.execution as execution_module
import quant_rabbit.market_read_overlay as market_read_overlay_module
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.gpt_trader import GPTTraderBrain, StaticTraderProvider
from quant_rabbit.guardian_events import _margin_pressure_event
from quant_rabbit.guardian_receipt_consumption import (
    P1_MARGIN_WARNING_CONTRACT,
    P1_MARGIN_WARNING_OBSERVED_CODE,
    build_guardian_receipt_consumption,
)
from quant_rabbit.market_read_overlay import (
    CODEX_MARKET_READ_AUTHOR,
    apply_codex_market_read_overlay,
    baseline_core_payload,
    canonical_json_sha256,
    prepare_market_read_baseline,
)
from quant_rabbit.models import (
    AccountSummary,
    BrokerPosition,
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    Side,
    TradeMethod,
)
from quant_rabbit.strategy.directional_forecaster import synthesize_forecast
from quant_rabbit.strategy.intent_generator import (
    IntentGenerator,
    _capture_loss_asymmetry_guard,
    _exact_vehicle_net_metrics,
    _exact_vehicle_take_profit_metrics,
    _forecast_seed_lane,
    _m15_recovery_tp_proof_bootstrap_evidence,
)
from quant_rabbit.strategy.trader_brain import ACTION_SEND_ENTRY, TraderBrain
import tests.test_directional_forecaster as directional_forecaster_fixtures
from tests.test_gpt_trader import (
    _fixtures as gpt_fixtures,
    _synthetic_execution_cost_surface,
    _trade_decision,
)
import tests.test_intent_generator as intent_generator_fixtures
from tests.test_live_order_gateway import (
    FixedReconciliationTargetLedger,
    ReconciliationExecutionClient,
    _insert_m15_recovery_exact_tp_outcomes,
    _synthetic_execution_cost_floor,
)


LANE_ID = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
RECOVERY_CONTEXT_TIMEFRAMES = ["M15", "M30", "H1", "H4", "D"]


@dataclass(frozen=True)
class _Signal:
    direction: str
    bonus_magnitude: float
    confidence: float
    rationale: str
    timeframe: str


class _NoTouchExecutionClient(ReconciliationExecutionClient):
    """Return an S5 path that touches neither this SHORT entry nor its bounds."""

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = super().get_json(path, query)
        for candle in payload.get("candles", []):
            candle["mid"] = {
                "o": "1.10004",
                "h": "1.10006",
                "l": "1.10000",
                "c": "1.10004",
            }
        return payload


def _shift_chart_to_now(chart: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    def shifted(value: Any, *, delta: timedelta) -> Any:
        if isinstance(value, dict):
            return {
                key: shifted(item, delta=delta)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [shifted(item, delta=delta) for item in value]
        if isinstance(value, str) and "T" in value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return value
            shifted_time = (parsed + delta).isoformat()
            return (
                shifted_time.replace("+00:00", "Z")
                if value.endswith("Z")
                else shifted_time
            )
        return value

    source = deepcopy(chart)
    integrity = source["technical_candle_integrity"]
    timeframes = integrity["timeframes"]
    views = source["views"]
    view_by_tf = {
        str(view["granularity"]): view
        for view in views
        if isinstance(view, dict)
    }
    cadence_seconds = {"M1": 60, "M5": 5 * 60, "M15": 15 * 60}
    for timeframe, seconds in cadence_seconds.items():
        item = timeframes[timeframe]
        source_latest = datetime.fromisoformat(
            str(item["latest_complete_timestamp_utc"]).replace("Z", "+00:00")
        )
        boundary_epoch = int(now.timestamp()) // seconds * seconds
        target_latest = datetime.fromtimestamp(
            boundary_epoch - seconds,
            tz=timezone.utc,
        )
        delta = target_latest - source_latest
        transformed_item = shifted(item, delta=delta)
        source_view = view_by_tf[timeframe]
        transformed_view = shifted(
            {
                key: value
                for key, value in source_view.items()
                if key != "candle_integrity"
            },
            delta=delta,
        )
        assert isinstance(transformed_item, dict)
        assert isinstance(transformed_view, dict)
        transformed_view["candle_integrity"] = transformed_item
        timeframes[timeframe] = transformed_item
        view_by_tf[timeframe].clear()
        view_by_tf[timeframe].update(transformed_view)

    source["generated_at_utc"] = now.isoformat().replace("+00:00", "Z")
    result = source
    return result


def _manual_positions() -> tuple[BrokerPosition, ...]:
    return tuple(
        BrokerPosition(
            trade_id=trade_id,
            pair="EUR_USD",
            side=Side.SHORT,
            units=units,
            entry_price=entry_price,
            take_profit=1.13610,
            stop_loss=None,
            owner=Owner.OPERATOR_MANUAL,
            raw={
                "operator_manual_position": {
                    "packet_type": "OPERATOR_MANUAL_POSITION",
                    "classification": "OPERATOR_MANUAL",
                    "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                    "management_intent": "NO_TOUCH_OBSERVE_ONLY",
                    "no_live_side_effects": True,
                    "system_pl_counted": False,
                    "same_theme_auto_add_allowed": True,
                    "loss_side_auto_close_allowed": False,
                    "auto_sl_attach_allowed": False,
                    "auto_tp_modify_allowed": False,
                    "trade_id": trade_id,
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": units,
                }
            },
        )
        for trade_id, units, entry_price in (
            ("473055", 3_000, 1.13951),
            ("473032", 8_000, 1.13832),
            ("472987", 22_500, 1.14048),
        )
    )


def _broker_snapshot(now: datetime) -> BrokerSnapshot:
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=_manual_positions(),
        orders=(),
        quotes={
            "EUR_USD": Quote(
                "EUR_USD",
                bid=1.10000,
                ask=1.10008,
                timestamp_utc=now,
            ),
            "USD_JPY": Quote(
                "USD_JPY",
                bid=157.00,
                ask=157.01,
                timestamp_utc=now,
            ),
        },
        account=AccountSummary(
            balance_jpy=294_442.8959,
            nav_jpy=267_573.5836,
            margin_used_jpy=248_430.64,
            margin_available_jpy=19_428.1634,
            last_transaction_id="100",
            fetched_at_utc=now,
        ),
        home_conversions={"USD": 157.0, "JPY": 1.0},
    )


def _position_payload(position: BrokerPosition) -> dict[str, Any]:
    return {
        "trade_id": position.trade_id,
        "pair": position.pair,
        "side": position.side.value,
        "units": position.units,
        "entry_price": position.entry_price,
        "take_profit": position.take_profit,
        "stop_loss": position.stop_loss,
        "owner": position.owner.value,
        "raw": position.raw,
    }


def _snapshot_payload(snapshot: BrokerSnapshot) -> dict[str, Any]:
    assert snapshot.account is not None
    return {
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "positions": [_position_payload(item) for item in snapshot.positions],
        "orders": [],
        "quotes": {
            pair: {
                "pair": quote.pair,
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in snapshot.quotes.items()
        },
        "account": {
            "balance_jpy": snapshot.account.balance_jpy,
            "nav_jpy": snapshot.account.nav_jpy,
            "margin_used_jpy": snapshot.account.margin_used_jpy,
            "margin_available_jpy": snapshot.account.margin_available_jpy,
            "last_transaction_id": snapshot.account.last_transaction_id,
            "hedging_enabled": True,
            "fetched_at_utc": snapshot.account.fetched_at_utc.isoformat(),
        },
        "home_conversions": dict(snapshot.home_conversions),
    }


def _manual_position_bytes(snapshot: BrokerSnapshot) -> bytes:
    return json.dumps(
        [_position_payload(item) for item in snapshot.positions],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_p1_margin_guardian_artifacts(
    files: dict[str, Path],
    *,
    snapshot_payload: dict[str, Any],
    now: datetime,
) -> None:
    account = snapshot_payload["account"]
    event = _margin_pressure_event(
        account,
        positions=list(snapshot_payload["positions"]),
        now=now,
    )
    assert event is not None
    event_payload = event.to_payload()
    assert event_payload["severity"] == "P1"
    assert event_payload["details"]["fresh_entry_risk_block_active"] is False
    assert (
        event_payload["details"]["fresh_entry_margin_contract"]
        == P1_MARGIN_WARNING_CONTRACT
    )

    expires_at = now + timedelta(hours=1)
    receipt = {
        "status": "ACCEPTED",
        "source": "guardian_wake_dispatcher",
        "model": "gpt-5.5",
        "receipt_status": "ACCEPTED",
        "receipt_lifecycle": "ACTIVE",
        "consumed_by_trader": False,
        "dispatcher_status": "RECEIPT_WRITTEN",
        "generated_at_utc": now.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "selected_event_id": event.event_id,
        "selected_event_dedupe_key": event.dedupe_key,
        "gateway_required": True,
        "no_direct_oanda": True,
        "selected_event": event_payload,
        "event": dict(event_payload),
        "receipt": {
            "action": "HOLD",
            "event_id": event.event_id,
            "dedupe_key": event.dedupe_key,
            "pair": event.pair,
            "side": "NONE",
            "thesis_state": event.thesis_state,
            "new_information": True,
            "ownership": "OPERATOR_MANUAL",
            "reason": "current portfolio margin is below but near the hard cap",
            "invalidation": "margin utilization leaves the P1 warning range",
            "harvest_trigger": "not applicable to observation-only warning",
            "margin_state": "P1_MARGIN_PRESSURE_WARNING_ONLY",
            "gateway_required": True,
            "no_direct_oanda": True,
        },
        "execution_boundary": {
            "gpt_wake_never_calls_oanda_directly": True,
            "guardian_never_trades": True,
            "only_live_order_gateway_may_send_cancel_close": True,
        },
        "issues": [],
    }
    files["guardian_action_receipt"].write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    issue = {
        "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        "severity": "P1",
        "message": "current Guardian P1 margin warning",
        "receipt_event_id": event.event_id,
        "receipt_action": "HOLD",
        "receipt_lifecycle": "ACTIVE",
        "emergency_or_margin_risk": True,
        "consumed_by_trader": False,
        "normal_routing_allowed": False,
        "event_type": event.event_type,
        "event_severity": event.severity,
        "event_action_hint": event.action_hint,
        "event_details": event.details,
    }
    watchdog = {
        "generated_at_utc": now.isoformat(),
        "status": "BLOCKED",
        "runtime_status": "OK",
        "issue_status": "P1",
        "overall_status": "P1",
        "severity": "P1",
        "issues": [issue],
        "guardian_receipt_issues": [issue],
        "guardian_receipt": {
            "issues": [issue],
            "emergency_or_margin_risk": True,
        },
    }
    files["qr_trader_run_watchdog"].write_text(
        json.dumps(watchdog),
        encoding="utf-8",
    )
    consumption = build_guardian_receipt_consumption(
        watchdog,
        now_utc=now,
        broker_snapshot=snapshot_payload,
    )
    assert consumption["normal_routing_allowed"] is False
    assert consumption["current_p0_p1_blocks_routing"] is True
    files["guardian_receipt_consumption"].write_text(
        json.dumps(consumption),
        encoding="utf-8",
    )
    files["guardian_receipt_operator_review"].write_text(
        json.dumps(
            {
                "status": "OK",
                "normal_routing_allowed": True,
                "reviews": [],
            }
        ),
        encoding="utf-8",
    )
    files["snapshot"].write_text(
        json.dumps(snapshot_payload),
        encoding="utf-8",
    )


def _initialize_execution_ledger(path: Path, report_path: Path) -> None:
    if path.exists():
        path.unlink()
    ExecutionLedger(db_path=path, report_path=report_path)._init_db()
    now = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(path)) as connection, connection:
        for key, value in (
            ("last_oanda_transaction_id", "100"),
            (
                "oanda_transaction_coverage_start_utc",
                (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            ),
        ):
            connection.execute(
                """
                INSERT INTO sync_state(key, value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (key, value, now),
            )


def _capture_economics_payload(now: datetime) -> dict[str, Any]:
    return {
        "status": "NEGATIVE_EXPECTANCY",
        "generated_at_utc": now.isoformat(),
        "overall": {
            "trades": 30,
            "avg_win_jpy": 500.0,
            "avg_loss_jpy": 1_042.0,
            "payoff_ratio": 0.48,
            "breakeven_payoff_at_win_rate": 0.70,
        },
        "by_exit_reason": {
            "TAKE_PROFIT_ORDER": {
                "trades": 7,
                "wins": 7,
                "losses": 0,
                "net_jpy": 6_427.5932,
                "expectancy_jpy_per_trade": 918.2276,
                "avg_win_jpy": 918.2276,
                "avg_loss_jpy": 0.0,
            },
            "MARKET_ORDER_TRADE_CLOSE": {
                "trades": 23,
                "wins": 0,
                "losses": 23,
                "expectancy_jpy_per_trade": -1_042.0,
            },
        },
        "by_pair_side_exit_reason": {},
        "by_pair_side_method_exit_reason": {},
    }


def _write_strategy_inputs(files: dict[str, Path], lane: dict[str, Any]) -> None:
    files["campaign"].write_text(
        json.dumps({"lanes": [lane]}),
        encoding="utf-8",
    )
    files["strategy"].write_text(
        json.dumps(
            {
                "system_contract": {"loss_cap_jpy": 500.0},
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "status": "CANDIDATE",
                        "required_fix": "eligible",
                        "pretrade_net_jpy": 1_000.0,
                        "live_net_jpy": 1_000.0,
                        "live_worst_jpy": -100.0,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 100.0,
                        "positive_best_jpy": 200.0,
                        "seat_discovered": 10,
                        "seat_orderable": 10,
                        "seat_captured": 7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    files["story"].write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"BREAKOUT_FAILURE": 35},
                        "themes": {"breakout_failure": 5},
                        "examples": ["M15 upper-rail rejection"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _market_read_for_intent(decision: dict[str, Any], intent: dict[str, Any]) -> None:
    market_read = decision["market_read_first"]
    for horizon in ("next_30m_prediction", "next_2h_prediction"):
        market_read[horizon].update(
            {
                "pair": "EUR_USD",
                "direction": "SHORT",
                "expected_path": "M15 rejection extends toward the bound target.",
                "target_zone": str(intent["tp"]),
                "invalidation": str(intent["sl"]),
            }
        )
    market_read["best_trade_if_forced"].update(
        {
            "pair": "EUR_USD",
            "direction": "SHORT",
            "vehicle": "STOP",
            "entry": str(intent["entry"]),
            "tp": str(intent["tp"]),
            "sl": str(intent["sl"]),
            "why_this_pays": (
                "The content-addressed M15 target is farther than the bound "
                "invalidation risk."
            ),
        }
    )


class M15RecoveryAuthorityE2ETest(unittest.TestCase):
    def _producer_bundle(
        self,
        root: Path,
        *,
        exact_tp_trades: int = 7,
    ) -> tuple[dict[str, Path], datetime, BrokerSnapshot, dict[str, Any], dict[str, Any]]:
        files = gpt_fixtures(root)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        snapshot = _broker_snapshot(now)
        snapshot_payload = _snapshot_payload(snapshot)
        _write_p1_margin_guardian_artifacts(
            files,
            snapshot_payload=snapshot_payload,
            now=now,
        )

        chart = _shift_chart_to_now(
            directional_forecaster_fixtures.TechnicalCandleIntegrityForecastGateTest._m15_recovery_chart(),
            now=now,
        )
        m15_view = next(
            view
            for view in chart["views"]
            if view.get("granularity") == "M15"
        )
        # Market input, not an authorization patch: the current upper swing is
        # closer than the downside target, yielding positive bound RR.
        m15_view["structure"]["swings"][0]["price"] = 1.10080
        chart_row = {key: value for key, value in chart.items() if key != "generated_at_utc"}
        files["pair_charts"].write_text(
            json.dumps(
                {
                    "generated_at_utc": chart["generated_at_utc"],
                    "charts": [chart_row],
                }
            ),
            encoding="utf-8",
        )

        _initialize_execution_ledger(
            files["execution_ledger"],
            root / "execution_ledger_report.md",
        )
        _insert_m15_recovery_exact_tp_outcomes(
            files["execution_ledger"],
            trades=exact_tp_trades,
        )
        files["capture_economics"].write_text(
            json.dumps(_capture_economics_payload(datetime.now(timezone.utc))),
            encoding="utf-8",
        )

        quote = snapshot.quotes["EUR_USD"]
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=quote.mid,
            pattern_signals=[],
            projection_signals=[
                _Signal(
                    "DOWN",
                    100.0,
                    1.0,
                    "M15 downside failed-acceptance detector",
                    "M15",
                )
            ],
            correlation_signals=[],
            paths=[],
            spread_pips=abs(quote.ask - quote.bid) * 10_000,
            now_utc=now,
            require_technical_candle_integrity=True,
        )
        self.assertEqual(forecast.direction, "DOWN")
        self.assertTrue(forecast.m15_recovery_receipt)
        cycle_id = f"m15-recovery-e2e-{now.isoformat()}"
        lane = _forecast_seed_lane(
            {"reason": "actual forecaster M15 recovery authority E2E"},
            pair="EUR_USD",
            side="SHORT",
            method="BREAKOUT_FAILURE",
            forecast=forecast,
            cycle_id=cycle_id,
        )
        binding = lane["forecast_m15_recovery_binding"]
        history_identity = dict(binding["forecast_history_identity"])
        history_identity.pop("identity_sha256")
        (root / "forecast_history.jsonl").write_text(
            json.dumps(
                {
                    "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                    **history_identity,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        indexed_charts = intent_generator_fixtures.M15RecoveryMicroIntentTest._indexed_chart(
            chart,
            m15_view,
        )
        generator = IntentGenerator(
            campaign_plan=files["campaign"],
            strategy_profile=files["strategy"],
            output_path=files["intents"],
            report_path=root / "intents.md",
            pair_charts_path=files["pair_charts"],
            data_root=root,
            max_loss_jpy=500.0,
        )
        result = generator._build_for_lane(
            lane,
            snapshot,
            None,
            max_loss_jpy=500.0,
            pair_charts=indexed_charts,
            validation_time_utc=now,
            data_root=root,
            loss_asymmetry_guard=_capture_loss_asymmetry_guard(root),
            exact_vehicle_tp_metrics=_exact_vehicle_take_profit_metrics(
                files["execution_ledger"]
            ),
            exact_vehicle_net_metrics=_exact_vehicle_net_metrics(
                files["execution_ledger"]
            ),
        )
        self.assertEqual(result.status, "LIVE_READY", result.live_blocker_codes)
        self.assertTrue(result.risk_allowed)
        self.assertIsNotNone(result.intent)
        assert result.intent is not None
        self.assertEqual(result.intent["order_type"], "STOP-ENTRY")
        self.assertLessEqual(result.intent["units"], 999)
        self.assertGreater(result.intent["units"], 0)
        self.assertLessEqual(
            float(result.risk_metrics["margin_utilization_after_pct"]),
            95.0,
        )
        self.assertTrue(
            result.intent["metadata"]["m15_recovery_micro_risk_revalidated"]
        )

        files["intents"].write_text(
            json.dumps(
                {
                    "generated_at_utc": now.isoformat(),
                    "campaign_plan": str(files["campaign"]),
                    "strategy_profile": str(files["strategy"]),
                    "snapshot_path": str(files["snapshot"]),
                    "results": [asdict(result)],
                }
            ),
            encoding="utf-8",
        )
        _write_strategy_inputs(files, lane)
        return files, now, snapshot, lane, result.intent

    def _run_overlay_and_gpt(
        self,
        root: Path,
        files: dict[str, Path],
        *,
        intent: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        baseline = _trade_decision(
            lane_id=LANE_ID,
            method="BREAKOUT_FAILURE",
            pair="EUR_USD",
            direction="SHORT",
        )
        _market_read_for_intent(baseline, intent)
        baseline.pop("decision_provenance", None)
        paths = {
            "baseline": root / "trader_decision_baseline.json",
            "packet": root / "market_read_evidence_packet.json",
            "overlay": root / "codex_market_read_overlay.json",
            "response": root / "codex_trader_decision_response.json",
            "verified": root / "gpt_decision.json",
        }
        paths["baseline"].write_text(json.dumps(baseline), encoding="utf-8")

        source_brain = GPTTraderBrain(
            provider=StaticTraderProvider(baseline),
            intents_path=files["intents"],
            campaign_plan_path=files["campaign"],
            strategy_profile_path=files["strategy"],
            market_story_profile_path=files["story"],
            market_status_path=files["market_status"],
            target_state_path=files["target"],
            pair_charts_path=files["pair_charts"],
            context_asset_charts_path=files["context_asset_charts"],
            broker_instruments_path=files["broker_instruments"],
            cross_asset_path=files["cross_asset"],
            flow_path=files["flow"],
            currency_strength_path=files["currency_strength"],
            levels_path=files["levels"],
            market_context_matrix_path=files["market_context_matrix"],
            calendar_path=files["calendar"],
            cot_path=files["cot"],
            option_skew_path=files["option_skew"],
            attack_advice_path=files["attack_advice"],
            capture_economics_path=files["capture_economics"],
            profitability_acceptance_path=files["profitability_acceptance"],
            execution_timing_audit_path=files["execution_timing_audit"],
            coverage_optimization_path=files["coverage_optimization"],
            learning_audit_path=files["learning_audit"],
            verification_ledger_path=files["verification_ledger"],
            self_improvement_audit_path=files["self_improvement_audit"],
            projection_ledger_path=files["projection_ledger"],
            operator_precedent_path=files["operator_precedent"],
            manual_market_context_path=files["manual_market_context"],
            trader_overrides_path=files["trader_overrides"],
            predictive_limits_path=files["predictive_limits"],
            news_items_path=files["news_items"],
            news_health_path=files["news_health"],
            qr_trader_run_watchdog_path=files["qr_trader_run_watchdog"],
            guardian_action_receipt_path=files["guardian_action_receipt"],
            guardian_receipt_consumption_path=files["guardian_receipt_consumption"],
            guardian_receipt_operator_review_path=files[
                "guardian_receipt_operator_review"
            ],
            active_trader_contract_path=files["active_trader_contract"],
            active_opportunity_board_path=files["active_opportunity_board"],
            non_eurusd_live_grade_frontier_path=files["non_eurusd_frontier"],
            range_rail_geometry_repair_path=files["range_rail_geometry_repair"],
            output_path=paths["verified"],
            report_path=root / "gpt_decision.md",
            execution_ledger_path=files["execution_ledger"],
            market_read_artifact_validation_required=True,
            market_read_baseline_path=paths["baseline"],
            market_read_evidence_packet_path=paths["packet"],
            market_read_overlay_path=paths["overlay"],
        )
        sources = source_brain._market_read_evidence_sources(files["snapshot"])
        applied_at = datetime.now(timezone.utc)
        prepare_market_read_baseline(
            baseline_path=paths["baseline"],
            packet_path=paths["packet"],
            evidence_sources=sources,
            now=applied_at,
        )
        stamped = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        packet = json.loads(paths["packet"].read_text(encoding="utf-8"))
        selected = packet["capital_allocation_board"]["selected_lane"]
        self.assertTrue(selected["allocation_eligible"], selected)
        self.assertEqual(
            selected["m15_recovery"]["edge_basis"],
            "M15_RECOVERY_EDGE_COLLECTION",
        )
        overlay = {
            "schema_version": packet["schema_version"],
            "author_kind": CODEX_MARKET_READ_AUTHOR,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "authored_at_utc": applied_at.isoformat(),
            "baseline_sha256": canonical_json_sha256(
                baseline_core_payload(stamped)
            ),
            "evidence_packet_sha256": packet["evidence_packet_sha256"],
            "baseline_disposition": "ACCEPT_BASELINE",
            "market_read_first": baseline["market_read_first"],
            "market_read_review": {
                "prior_prediction_ids": [],
                "what_failed": "NO_RESOLVED_PRIOR",
                "adjustment": "Keep the content-addressed M15 geometry.",
                "no_change_reason": "",
            },
            "market_read_counterargument": (
                "The M15 rejection can fail before the stop entry triggers."
            ),
            "market_read_change_summary": (
                "Accepted the same bounded recovery geometry and micro size."
            ),
            "market_read_veto_reason": "",
            "capital_allocation": {
                "decision": "ALLOCATE",
                "lane_id": LANE_ID,
                "size_multiple": 1.0,
                "selected_units": int(selected["base_units"]),
                "allocation_board_sha256": packet[
                    "capital_allocation_board_sha256"
                ],
                "rationale": (
                    "The M15 recovery edge-collection contract authorizes only "
                    "the producer-bounded units."
                ),
            },
        }
        paths["overlay"].write_text(json.dumps(overlay), encoding="utf-8")
        apply_codex_market_read_overlay(
            baseline_path=paths["baseline"],
            packet_path=paths["packet"],
            overlay_path=paths["overlay"],
            output_path=paths["response"],
            evidence_sources=sources,
            now=applied_at,
        )
        response = json.loads(paths["response"].read_text(encoding="utf-8"))
        source_brain.provider = StaticTraderProvider(response)
        summary = source_brain.run(snapshot_path=files["snapshot"])
        self.assertEqual(summary.status, "ACCEPTED")
        verified = json.loads(paths["verified"].read_text(encoding="utf-8"))
        recovery = verified["input_packet"]["lanes"][0]["forecast"][
            "m15_recovery"
        ]
        self.assertEqual(recovery["status"], "VERIFIED")
        self.assertEqual(
            recovery["source_receipt_sha256"],
            intent["metadata"]["m15_recovery_micro_receipt_sha256"],
        )
        return verified, paths

    def test_zero_history_m15_recovery_can_bootstrap_one_bounded_tp_sample(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "quant_rabbit.capture_economics.read_execution_cost_surface",
                    return_value=_synthetic_execution_cost_surface(),
                ),
                patch.object(
                    market_read_overlay_module,
                    "execution_cost_floor_from_surface",
                    side_effect=lambda _surface, *, exact_key, as_of: (
                        _synthetic_execution_cost_floor(exact_key)
                    ),
                ),
                patch.object(
                    execution_module,
                    "execution_cost_floor_from_surface",
                    side_effect=lambda _surface, *, exact_key, as_of: (
                        _synthetic_execution_cost_floor(exact_key)
                    ),
                ),
            ):
                _files, _now, _snapshot, _lane, intent = self._producer_bundle(
                    root,
                    exact_tp_trades=0,
                )

        metadata = intent["metadata"]
        self.assertEqual(
            metadata["positive_rotation_mode"],
            "TP_PROOF_COLLECTION_HARVEST",
        )
        self.assertEqual(
            metadata["positive_rotation_proof_collection_bootstrap_contract"],
            "QR_M15_RECOVERY_TP_PROOF_BOOTSTRAP_V1",
        )
        self.assertEqual(
            metadata["positive_rotation_proof_collection_existing_net_trades"],
            0,
        )
        self.assertEqual(
            metadata["positive_rotation_proof_collection_existing_net_jpy"],
            0.0,
        )
        self.assertTrue(metadata["m15_recovery_micro_shape_eligible"])
        self.assertTrue(metadata["m15_recovery_micro_risk_revalidated"])

    def test_negative_exact_vehicle_net_stops_m15_bootstrap_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "quant_rabbit.capture_economics.read_execution_cost_surface",
                    return_value=_synthetic_execution_cost_surface(),
                ),
                patch.object(
                    market_read_overlay_module,
                    "execution_cost_floor_from_surface",
                    side_effect=lambda _surface, *, exact_key, as_of: (
                        _synthetic_execution_cost_floor(exact_key)
                    ),
                ),
                patch.object(
                    execution_module,
                    "execution_cost_floor_from_surface",
                    side_effect=lambda _surface, *, exact_key, as_of: (
                        _synthetic_execution_cost_floor(exact_key)
                    ),
                ),
            ):
                _files, _now, _snapshot, _lane, intent = self._producer_bundle(
                    root,
                    exact_tp_trades=0,
                )

        metadata = deepcopy(intent["metadata"])
        probe = OrderIntent(
            pair=intent["pair"],
            side=Side(intent["side"]),
            order_type=OrderType(intent["order_type"]),
            units=int(intent["units"]),
            entry=float(intent["entry"]),
            tp=float(intent["tp"]),
            sl=float(intent["sl"]),
            thesis=str(intent["thesis"]),
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_DOWN",
                narrative="M15 recovery",
                chart_story="M15 only",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="M15 structure",
            ),
            metadata=metadata,
        )
        self.assertIsNotNone(_m15_recovery_tp_proof_bootstrap_evidence(probe))
        metadata.update(
            {
                "capture_exact_vehicle_net_trades": 1,
                "capture_exact_vehicle_net_wins": 0,
                "capture_exact_vehicle_net_losses": 1,
                "capture_exact_vehicle_net_jpy": -100.0,
                "capture_exact_vehicle_net_expectancy_jpy": -100.0,
            }
        )
        self.assertIsNone(_m15_recovery_tp_proof_bootstrap_evidence(probe))

    def test_actual_m15_recovery_authority_chain_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with (
                    patch(
                        "quant_rabbit.capture_economics.read_execution_cost_surface",
                        return_value=_synthetic_execution_cost_surface(),
                    ),
                    patch.object(
                        market_read_overlay_module,
                        "execution_cost_floor_from_surface",
                        side_effect=lambda _surface, *, exact_key, as_of: (
                            _synthetic_execution_cost_floor(exact_key)
                        ),
                    ),
                    patch.object(
                        execution_module,
                        "execution_cost_floor_from_surface",
                        side_effect=lambda _surface, *, exact_key, as_of: (
                            _synthetic_execution_cost_floor(exact_key)
                        ),
                    ),
                ):
                    files, now, snapshot, lane, intent = self._producer_bundle(root)
                    producer_digests = {
                        "receipt": intent["metadata"][
                            "m15_recovery_micro_receipt_sha256"
                        ],
                        "forecast": intent["metadata"][
                            "forecast_m15_recovery_binding_sha256"
                        ],
                        "lane": intent["metadata"][
                            "m15_recovery_lane_binding_sha256"
                        ],
                    }

                    trader = TraderBrain(
                        intents_path=files["intents"],
                        campaign_plan_path=files["campaign"],
                        strategy_profile_path=files["strategy"],
                        market_story_profile_path=files["story"],
                        target_state_path=files["target"],
                        pair_charts_path=files["pair_charts"],
                        output_path=root / "trader_decision.json",
                        report_path=root / "trader_decision.md",
                    ).run(snapshot)
                    self.assertEqual(trader.action, ACTION_SEND_ENTRY)
                    self.assertEqual(trader.selected_lane_id, LANE_ID)
                    selected_score = next(
                        score for score in trader.scores if score.lane_id == LANE_ID
                    )
                    self.assertTrue(selected_score.m15_recovery_verified)

                    verified, paths = self._run_overlay_and_gpt(
                        root,
                        files,
                        intent=intent,
                    )
                    self.assertEqual(verified["status"], "ACCEPTED")
                    self.assertEqual(
                        verified["decision"]["selected_lane_id"],
                        trader.selected_lane_id,
                    )

                    client = _NoTouchExecutionClient()
                    client.snapshot_value = snapshot
                    before_manual = _manual_position_bytes(snapshot)
                    gateway = LiveOrderGateway(
                        client=client,
                        strategy_profile=files["strategy"],
                        output_path=root / "live_order_request.json",
                        report_path=root / "live_order_report.md",
                        target_state_path=files["target"],
                        target_report_path=root / "daily_target_report.md",
                        self_improvement_audit=files["self_improvement_audit"],
                        verified_decision_path=paths["verified"],
                        guardian_action_receipt_path=files[
                            "guardian_action_receipt"
                        ],
                        qr_trader_run_watchdog_path=files[
                            "qr_trader_run_watchdog"
                        ],
                        guardian_receipt_consumption_path=files[
                            "guardian_receipt_consumption"
                        ],
                        guardian_receipt_operator_review_path=files[
                            "guardian_receipt_operator_review"
                        ],
                        broker_snapshot_path=files["snapshot"],
                        pair_charts_path=files["pair_charts"],
                        forecast_history_path=root / "forecast_history.jsonl",
                        execution_ledger_db_path=files["execution_ledger"],
                        execution_ledger_report_path=root
                        / "execution_ledger_report.md",
                        live_enabled=True,
                        max_loss_jpy=500.0,
                    )
                    with (
                        patch.object(
                            execution_module,
                            "DailyTargetLedger",
                            FixedReconciliationTargetLedger,
                        ),
                        patch.object(
                            execution_module.RiskEngine,
                            "_now",
                            return_value=now,
                        ),
                    ):
                        summary = (
                            gateway.run_batch(
                                intents_path=files["intents"],
                                lane_ids=(LANE_ID,),
                                size_multiples={LANE_ID: 1.0},
                                send=True,
                                confirm_live=True,
                            )
                            if batch
                            else gateway.run(
                                intents_path=files["intents"],
                                lane_id=LANE_ID,
                                size_multiple=1.0,
                                send=True,
                                confirm_live=True,
                            )
                        )

                    payload = json.loads(
                        (root / "live_order_request.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    order = payload["orders"][0] if batch else payload
                    self.assertTrue(summary.sent, order)
                    self.assertEqual(len(client.orders), 1)
                    request = client.orders[0]
                    self.assertEqual(request["type"], "STOP")
                    self.assertEqual(request["units"], f"-{intent['units']}")
                    self.assertEqual(request["price"], f"{intent['entry']:.5f}")
                    self.assertEqual(
                        request["takeProfitOnFill"]["price"],
                        f"{intent['tp']:.5f}",
                    )
                    self.assertNotIn("priceBound", request)
                    self.assertLessEqual(abs(int(request["units"])), 999)
                    self.assertLessEqual(
                        float(order["risk_metrics"]["margin_utilization_after_pct"]),
                        95.0,
                    )
                    self.assertIn(
                        P1_MARGIN_WARNING_OBSERVED_CODE,
                        {
                            issue.get("code")
                            for issue in order["risk_issues"]
                            if isinstance(issue, dict)
                        },
                    )
                    self.assertEqual(
                        _manual_position_bytes(client.snapshot_value),
                        before_manual,
                    )
                    final_boundary = order["pre_post_reconciliation"][
                        "final_post_reservation_boundary"
                    ]
                    final_recheck = final_boundary[
                        "m15_recovery_micro_final_recheck"
                    ]
                    self.assertEqual(
                        final_recheck["receipt_sha256"],
                        producer_digests["receipt"],
                    )
                    self.assertEqual(
                        final_recheck["forecast_binding_sha256"],
                        producer_digests["forecast"],
                    )
                    self.assertEqual(
                        final_recheck["lane_binding_sha256"],
                        producer_digests["lane"],
                    )
                    guardian_recheck = final_boundary[
                        "guardian_action_receipt_scope_recheck"
                    ]
                    self.assertEqual(guardian_recheck["status"], "PASSED")
                    self.assertEqual(
                        guardian_recheck["scope"],
                        "GLOBAL_MARGIN_OBSERVATION",
                    )
                    self.assertFalse(guardian_recheck["global_safety"])
                    self.assertTrue(guardian_recheck["digest_matches"])


if __name__ == "__main__":
    unittest.main()
