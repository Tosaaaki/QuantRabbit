from __future__ import annotations

import json
import os
import hashlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

import quant_rabbit.market_read_overlay as market_read_overlay_module
from quant_rabbit.forecast_learning import (
    build_forecast_learning_execution_geometry,
)
from quant_rabbit.forecast_precision import (
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
    hit_rate_wilson_lower,
)
from quant_rabbit.market_read_overlay import (
    CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES,
    CODEX_MARKET_READ_AUTHOR,
    MARKET_READ_OVERLAY_SCHEMA_VERSION,
    MarketReadOverlayError,
    apply_codex_market_read_overlay,
    baseline_core_payload,
    canonical_json_sha256,
    execution_envelope_payload,
    prepare_market_read_baseline,
    projection_calibration_evidence,
    revalidate_codex_market_read_artifacts,
    validate_codex_market_read_provenance,
)
from quant_rabbit.strategy.forecast_technical_context import (
    CONFIDENCE_SEMANTICS,
    MAX_EVIDENCE_BYTES,
    build_forecast_technical_context,
    build_forecast_technical_context_evidence,
    technical_context_sha256,
)
from quant_rabbit.strategy.m15_recovery_contract import (
    FORECAST_CONTRACT as M15_RECOVERY_FORECAST_CONTRACT,
    GEOMETRY_CONTRACT as M15_RECOVERY_GEOMETRY_CONTRACT,
    LANE_CONTRACT as M15_RECOVERY_LANE_CONTRACT,
    build_forecast_binding as build_m15_recovery_forecast_binding,
    build_lane_binding as build_m15_recovery_lane_binding,
)
from quant_rabbit.strategy.projection_ledger import LedgerEntry, write_ledger


NOW = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
M15_RECOVERY_LANE_ID = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"


def _projection_trial(
    base: datetime,
    index: int,
    *,
    status: str,
    direction: str = "UP",
    regime: str = "TREND",
    pair: str = "EUR_USD",
) -> LedgerEntry:
    up = direction == "UP"
    emitted_at = base + timedelta(hours=index)
    evidence = (
        "target touched before invalidation"
        if status == "HIT"
        else "invalidation touched before target"
        if status == "MISS"
        else "neither target nor invalidation touched"
    )
    return LedgerEntry(
        timestamp_emitted_utc=emitted_at.isoformat(),
        pair=pair,
        signal_name="directional_forecast",
        direction=direction,
        lead_time_min=60,
        confidence=0.24,
        raw_confidence=0.80,
        calibration_multiplier=0.30,
        entry_price=1.1000,
        predicted_target_price=1.1020 if up else 1.0980,
        predicted_invalidation_price=1.0990 if up else 1.1010,
        resolution_window_min=60,
        resolution_status=status,
        resolved_at_utc=(emitted_at + timedelta(hours=1)).isoformat(),
        resolution_evidence=evidence,
        regime_at_emission=regime,
        cycle_id=f"{direction.lower()}-{index}-{status.lower()}",
    )


def _forecast_context_evidence(
    pair: str,
    current_price: float,
    *,
    direction: str = "UP",
    family_score: float | None = None,
    failed_break_long: bool = False,
    now_utc: datetime = NOW,
    pair_charts_path: Path | None = None,
    session_tag: str | None = None,
    chart_story: str = "",
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
) -> dict:
    direction_text = str(direction).upper()
    range_regime = direction_text == "RANGE"
    up = direction_text != "DOWN"
    selected_family_score = (
        family_score if family_score is not None else (1.0 if up else -1.0)
    )
    chart = {
            "pair": pair,
            "confluence": {
                "dominant_regime": (
                    "RANGE"
                    if range_regime
                    else "TREND_UP"
                    if up
                    else "TREND_DOWN"
                ),
                "price_percentile_24h": 0.7 if up else 0.3,
                "price_percentile_7d": 0.6 if up else 0.4,
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime_reading": {
                        "state": "RANGE" if range_regime else "TREND_STRONG",
                        "atr_percentile": 60.0,
                    },
                    "indicators": {"atr_pips": 2.0},
                    "family_scores": {
                        "trend_score": (
                            0.0 if range_regime else selected_family_score
                        ),
                        "mean_rev_score": (
                            selected_family_score if range_regime else 0.0
                        ),
                        "breakout_score": 0.0,
                        "disagreement": 0.0,
                    },
                    "structure": {
                        "structure_events": [
                            {
                                "kind": "BOS_UP" if up else "BOS_DOWN",
                                "index": 1,
                                "close_confirmed": True,
                            }
                        ]
                    },
                }
            ],
        }
    if session_tag is not None:
        chart["session"] = {"current_tag": session_tag}
    if chart_story:
        chart["chart_story"] = chart_story
    if failed_break_long:
        start = datetime(2026, 7, 13, tzinfo=timezone.utc)
        prior = [
            {
                "t": (start + timedelta(minutes=5 * index)).isoformat(),
                "o": 1.1000,
                "h": 1.1010,
                "l": 1.0990,
                "c": 1.1000,
                "complete": True,
            }
            for index in range(20)
        ]
        chart["views"][0]["recent_candles"] = [
            *prior,
            {
                "t": (start + timedelta(minutes=100)).isoformat(),
                "o": 1.0995,
                "h": 1.1005,
                "l": 1.0985,
                "c": 1.0992,
                "complete": True,
            },
        ]
    if pair_charts_path is not None:
        pair_charts_path.write_text(
            json.dumps({"charts": [chart]})
        )
    context = build_forecast_technical_context(
        chart,
        pair=pair,
        current_price=current_price,
        spread_pips=2.0,
        now_utc=now_utc,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
    )
    return build_forecast_technical_context_evidence(
        context,
        pair=pair,
        current_price=current_price,
    )


def _forecast_weighting_metadata(evidence: dict) -> dict:
    body = evidence.get("technical_context_v1") or {}
    receipt = body.get("regime_family_weighting") or {}
    source = receipt.get("source_identity") or {}
    aggregate = receipt.get("aggregate") or {}
    return {
        "forecast_regime_family_weighting_sha256": receipt.get("receipt_sha256"),
        "forecast_regime_family_selected_method": source.get("selected_method"),
        "forecast_regime_family_direction": aggregate.get("direction"),
    }


def _forecast_policy_source_descriptors(evidence: dict) -> dict:
    body = evidence.get("technical_context_v1") or {}
    source_context = body.get("dynamic_tf_policy_source_context") or {}

    def packet_source(source: dict) -> dict:
        status = source.get("status")
        exists = status not in {"MISSING", "READ_ERROR"}
        return {
            "path": source.get("path"),
            "exists": exists,
            "sha256": source.get("sha256"),
            "size_bytes": source.get("byte_length"),
        }

    return {
        "calendar": packet_source(source_context.get("news_source") or {}),
        "strategy_profile": packet_source(
            source_context.get("strategy_profile_source") or {}
        ),
    }


class MarketReadOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        cost_patch = patch(
            "quant_rabbit.capture_economics.read_execution_cost_surface",
            side_effect=lambda _path: _synthetic_execution_cost_surface(),
        )
        cost_patch.start()
        self.addCleanup(cost_patch.stop)

    def test_projection_calibration_uses_independent_numeric_outcomes_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "projection_ledger.jsonl"
            base = NOW - timedelta(hours=12)
            statuses = ["HIT"] * 8 + ["MISS"] * 3 + ["TIMEOUT"]
            write_ledger(
                [
                    _projection_trial(base, index, status=status)
                    for index, status in enumerate(statuses)
                ],
                root,
            )
            os.utime(ledger, (NOW.timestamp(), NOW.timestamp()))

            evidence = projection_calibration_evidence(
                ledger,
                scopes=[{"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}],
                now=NOW,
            )

            self.assertEqual(evidence["status"], "VALID")
            self.assertEqual(len(evidence["rows"]), 1)
            row = evidence["rows"][0]
            bucket = row["specific_bucket"]
            self.assertEqual(row["calibration_name"], "directional_forecast_up")
            self.assertEqual(bucket["bucket"], "EUR_USD:TREND")
            self.assertEqual(bucket["samples"], 11)
            self.assertEqual(bucket["economic_samples"], 12)
            self.assertAlmostEqual(bucket["hit_rate"], 8 / 11, places=3)
            self.assertAlmostEqual(bucket["economic_hit_rate"], 8 / 12, places=3)
            self.assertEqual(bucket["timeout_count"], 1)
            self.assertEqual(bucket["invalidation_first_count"], 3)
            self.assertEqual(row["edge_status"], "PRECISION_FLOOR_FAILED_NO_EDGE")
            self.assertFalse(row["precision_floor_eligible"])
            self.assertFalse(evidence["selection_contract"]["raw_ledger_rows_exposed"])
            self.assertNotIn("timestamp_emitted_utc", json.dumps(evidence))
            material = dict(evidence)
            claimed_sha = material.pop("evidence_sha256")
            self.assertEqual(claimed_sha, canonical_json_sha256(material))

            multi_regime = projection_calibration_evidence(
                ledger,
                scopes=[
                    {"pair": "EUR_USD", "direction": "UP", "regime": "TREND"},
                    {"pair": "EUR_USD", "direction": "UP", "regime": "RANGE"},
                ],
                now=NOW,
            )
            refs = [item["evidence_ref"] for item in multi_regime["rows"]]
            self.assertEqual(len(refs), len(set(refs)))
            self.assertIn(
                "projection:calibration:EUR_USD:TREND:directional_forecast_up",
                refs,
            )

    def test_projection_calibration_missing_empty_and_regimeless_never_fake_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "projection_ledger.jsonl"
            scope = [{"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}]

            missing = projection_calibration_evidence(ledger, scopes=scope, now=NOW)
            self.assertEqual(missing["status"], "MISSING")
            self.assertEqual(missing["rows"], [])

            ledger.write_text("")
            os.utime(ledger, (NOW.timestamp(), NOW.timestamp()))
            empty = projection_calibration_evidence(ledger, scopes=scope, now=NOW)
            self.assertEqual(empty["status"], "VALID")
            self.assertIsNone(empty["rows"][0]["specific_bucket"])
            self.assertIsNone(empty["rows"][0]["selected_bucket"])
            self.assertEqual(
                empty["rows"][0]["edge_status"],
                "INSUFFICIENT_SAMPLES_NO_EDGE",
            )

            base = NOW - timedelta(hours=40)
            write_ledger(
                [_projection_trial(base, index, status="HIT") for index in range(40)],
                root,
            )
            os.utime(ledger, (NOW.timestamp(), NOW.timestamp()))
            regimeless = projection_calibration_evidence(
                ledger,
                scopes=[{"pair": "EUR_USD", "direction": "UP", "regime": None}],
                now=NOW,
            )["rows"][0]
            self.assertTrue(regimeless["precision_floor_eligible"])
            self.assertEqual(regimeless["specific_scope"], "PAIR_ALL_REGIMES")
            self.assertFalse(regimeless["exact_pair_regime_edge"])
            self.assertEqual(
                regimeless["edge_status"],
                "PAIR_ALL_REGIMES_CONTEXT_ONLY_NO_PAIR_REGIME_EDGE",
            )

    def test_projection_calibration_malformed_would_be_loss_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "projection_ledger.jsonl"
            base = NOW - timedelta(hours=40)
            write_ledger(
                [_projection_trial(base, index, status="HIT") for index in range(40)],
                root,
            )
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    '{"signal_name":"directional_forecast",'
                    '"direction":"UP","resolution_status":"MISS"\n'
                )

            evidence = projection_calibration_evidence(
                ledger,
                scopes=[
                    {"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}
                ],
                now=NOW,
            )

            self.assertEqual(evidence["status"], "MALFORMED")
            self.assertEqual(evidence["rows"], [])
            self.assertEqual(evidence["parse_integrity"]["status"], "INVALID")
            self.assertEqual(
                evidence["parse_integrity"]["malformed_json_rows"],
                1,
            )
            self.assertEqual(
                evidence["parse_integrity"]["invalid_nonblank_rows"],
                1,
            )
            material = dict(evidence)
            claimed_sha = material.pop("evidence_sha256")
            self.assertEqual(claimed_sha, canonical_json_sha256(material))

    def test_projection_calibration_core_invalid_object_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "projection_ledger.jsonl"
            base = NOW - timedelta(hours=40)
            write_ledger(
                [_projection_trial(base, index, status="HIT") for index in range(40)],
                root,
            )
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "resolution_status": "MISS",
                        }
                    )
                    + "\n"
                )

            evidence = projection_calibration_evidence(
                ledger,
                scopes=[
                    {"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}
                ],
                now=NOW,
            )

            self.assertEqual(evidence["status"], "MALFORMED")
            self.assertEqual(evidence["rows"], [])
            self.assertEqual(evidence["parse_integrity"]["status"], "INVALID")
            self.assertEqual(
                evidence["parse_integrity"]["unloadable_object_rows"],
                1,
            )

    def test_projection_calibration_labels_pair_and_global_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "projection_ledger.jsonl"
            base = NOW - timedelta(hours=30)
            write_ledger(
                [
                    *[
                        _projection_trial(base, index, status="HIT", regime="TREND")
                        for index in range(5)
                    ],
                    *[
                        _projection_trial(
                            base,
                            index + 5,
                            status="HIT",
                            regime="RANGE",
                        )
                        for index in range(5)
                    ],
                ],
                root,
            )
            pair_fallback = projection_calibration_evidence(
                ledger,
                scopes=[{"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}],
                now=NOW,
            )["rows"][0]
            self.assertEqual(pair_fallback["specific_bucket"]["samples"], 5)
            self.assertEqual(
                pair_fallback["selected_bucket"]["bucket"],
                "EUR_USD:_all_regimes",
            )
            self.assertEqual(pair_fallback["selected_bucket"]["samples"], 10)
            self.assertTrue(pair_fallback["fallback_used"])
            self.assertFalse(pair_fallback["exact_pair_regime_edge"])
            self.assertEqual(
                pair_fallback["edge_status"],
                "PRECISION_FLOOR_FAILED_NO_EDGE",
            )

            write_ledger(
                [
                    *[
                        _projection_trial(base, index, status="HIT")
                        for index in range(5)
                    ],
                    *[
                        _projection_trial(
                            base,
                            index,
                            status="HIT",
                            pair="GBP_USD",
                        )
                        for index in range(30)
                    ],
                ],
                root,
            )
            global_fallback = projection_calibration_evidence(
                ledger,
                scopes=[{"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}],
                now=NOW,
            )["rows"][0]
            self.assertEqual(
                global_fallback["selected_bucket"]["bucket"],
                "_all_pairs:TREND",
            )
            self.assertTrue(global_fallback["precision_floor_eligible"])
            self.assertTrue(global_fallback["fallback_used"])
            self.assertFalse(global_fallback["exact_pair_regime_edge"])
            self.assertEqual(
                global_fallback["edge_status"],
                "FALLBACK_CONTEXT_ONLY_NO_PAIR_REGIME_EDGE",
            )

    def test_projection_calibration_is_semantic_tamper_evident_and_cannot_upgrade_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(
                root,
                baseline=_baseline(action="WAIT", lane_ids=[]),
            )
            projection = root / "projection_ledger.jsonl"
            base = NOW - timedelta(hours=40)
            up_rows = [
                _projection_trial(base, index, status="HIT")
                for index in range(40)
            ]
            write_ledger(up_rows, root)
            old_mtime = (NOW - timedelta(days=1)).timestamp()
            os.utime(projection, (old_mtime, old_mtime))
            sources = {**_sources(paths), "projection_ledger": projection}
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=sources,
                now=NOW,
            )
            packet = json.loads(paths["packet"].read_text())
            original_sha = packet["projection_calibration_evidence_sha256"]
            self.assertEqual(
                packet["projection_calibration_evidence"]["rows"][0]["pair"],
                "EUR_USD",
            )
            selected_score = packet["projection_calibration_evidence"]["rows"][0]
            self.assertTrue(selected_score["precision_floor_eligible"])
            self.assertTrue(selected_score["exact_pair_regime_precision_context"])
            self.assertFalse(selected_score["exact_pair_regime_edge"])
            self.assertEqual(
                selected_score["edge_status"],
                "HISTORICAL_PAIR_REGIME_PRECISION_CONTEXT_ONLY_NO_EDGE",
            )
            self.assertEqual(
                packet["projection_calibration_evidence"]["freshness_status"],
                "CURRENT_RECOMPUTATION",
            )
            self.assertFalse(
                packet["projection_calibration_evidence"][
                    "selected_scope_outcome_recency_proven"
                ]
            )
            self.assertFalse(
                packet["projection_calibration_evidence"]["usage_policy"][
                    "may_upgrade_non_trade_baseline_to_trade"
                ]
            )
            _write_overlay(paths)

            # A fresh different-direction append neither invalidates nor
            # falsely refreshes the selected old UP outcome lineage.
            down = _projection_trial(
                base,
                40,
                status="HIT",
                direction="DOWN",
            )
            write_ledger([*up_rows, down], root)
            os.utime(projection, (NOW.timestamp(), NOW.timestamp()))
            summary = apply_codex_market_read_overlay(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                overlay_path=paths["overlay"],
                output_path=paths["output"],
                evidence_sources=sources,
                now=NOW,
            )
            self.assertEqual(summary.action, "WAIT")
            rebuilt = projection_calibration_evidence(
                projection,
                scopes=[
                    {"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}
                ],
                now=NOW,
            )
            self.assertEqual(
                rebuilt["evidence_sha256"],
                original_sha,
            )

            packet["projection_calibration_evidence"]["rows"][0][
                "edge_status"
            ] = "INSUFFICIENT_SAMPLES_NO_EDGE"
            paths["packet"].write_text(json.dumps(packet))
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_PROJECTION_CALIBRATION_INVALID",
            ):
                apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=sources,
                    now=NOW,
                )

    def test_projection_scope_falls_back_to_forced_canonical_direction(self) -> None:
        scopes = market_read_overlay_module._projection_calibration_scopes_for_market_read(
            baseline={"action": "WAIT", "market_read_first": _market_read()},
            capital_allocation_board={},
        )
        self.assertEqual(
            scopes,
            [{"pair": "EUR_USD", "direction": "UP", "regime": None}],
        )

        technical = {
            "status": "VALID",
            "technical_context_v1": {"regime": {"dominant": "TREND_UP"}},
        }
        selected_board = {
            "forecast_context_scope": "SELECTED_LANE",
            "forecast_context": {
                "pair": "EUR_USD",
                "direction": "UP",
                "technical_context": technical,
            },
            "selected_lane": {
                "pair": "EUR_USD",
                "forecast": {
                    "direction": "UP",
                    "calibration_name": "directional_forecast_up",
                },
            },
        }
        self.assertEqual(
            market_read_overlay_module._projection_calibration_scopes_for_market_read(
                baseline={"action": "TRADE", "market_read_first": _market_read()},
                capital_allocation_board=selected_board,
            ),
            [{"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}],
        )
        selected_board["selected_lane"]["forecast"][
            "calibration_name"
        ] = "directional_forecast_down"
        self.assertEqual(
            market_read_overlay_module._projection_calibration_scopes_for_market_read(
                baseline={"action": "TRADE", "market_read_first": _market_read()},
                capital_allocation_board=selected_board,
            ),
            [],
        )
        self.assertEqual(
            market_read_overlay_module._projection_calibration_scopes_for_market_read(
                baseline={"action": "TRADE", "market_read_first": _market_read()},
                capital_allocation_board={
                    "forecast_context_scope": "FORCED_PREDICTION",
                    "forecast_context": {"pair": "EUR_USD", "direction": "UP"},
                },
            ),
            [],
        )

    def test_missing_ledger_source_cannot_be_replaced_by_cwd_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(root)
            sentinel = root / "__missing_execution_ledger_for_market_read__"
            _write_exact_vehicle_ledger(sentinel, [320.0] * 8)
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            sources = _sources(paths)
            sources.pop("execution_ledger")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                prepare_market_read_baseline(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    evidence_sources=sources,
                    now=NOW,
                )
            finally:
                os.chdir(original_cwd)

            packet = json.loads(paths["packet"].read_text())
            self.assertEqual(
                packet["execution_ledger_allocation_surface"]["parse_status"],
                "MISSING",
            )
            self.assertFalse(
                packet["capital_allocation_board"]["selected_lane"][
                    "allocation_eligible"
                ]
            )

    def test_nonsemantic_source_metadata_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            packet = json.loads(paths["packet"].read_text())
            packet["source_metadata"]["broker_snapshot"][
                "generated_at_utc"
            ] = "2099-01-01T00:00:00+00:00"
            paths["packet"].write_text(json.dumps(packet))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_BODY_STALE",
            ):
                _apply(paths)

    def test_uncheckpointed_selected_loss_stales_but_checkpoint_only_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            with sqlite3.connect(paths["execution_ledger"]) as writer:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower(), "wal")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                main_sha_before = hashlib.sha256(
                    paths["execution_ledger"].read_bytes()
                ).hexdigest()
                _append_vehicle_trade(
                    paths["execution_ledger"],
                    pair="EUR_USD",
                    side="LONG",
                    method="TREND_CONTINUATION",
                    vehicle="MARKET",
                    realized=-5000.0,
                    index=200,
                    connection=writer,
                )
                writer.commit()
                wal_path = Path(str(paths["execution_ledger"]) + "-wal")
                self.assertTrue(wal_path.exists())
                self.assertGreater(wal_path.stat().st_size, 0)
                self.assertEqual(
                    hashlib.sha256(paths["execution_ledger"].read_bytes()).hexdigest(),
                    main_sha_before,
                )
                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_EVIDENCE_PACKET_STALE",
                ):
                    _apply(paths)

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            with sqlite3.connect(paths["execution_ledger"]) as writer:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower(), "wal")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                _append_vehicle_trade(
                    paths["execution_ledger"],
                    pair="GBP_USD",
                    side="SHORT",
                    method="RANGE_ROTATION",
                    vehicle="LIMIT",
                    realized=-5000.0,
                    index=201,
                    connection=writer,
                )
                writer.commit()
                _reprepare(paths)
                _write_overlay(paths)
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            self.assertEqual(_apply(paths).action, "TRADE")

    def test_mature_negative_all_exit_surface_suppresses_positive_tp_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "capture_exact_vehicle_net_trades": 20,
                    "capture_exact_vehicle_net_wins": 8,
                    "capture_exact_vehicle_net_losses": 12,
                    "capture_exact_vehicle_net_jpy": -9440.0,
                    "capture_exact_vehicle_net_expectancy_jpy": -472.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 320.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 1000.0,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _write_exact_vehicle_ledger(
                paths["execution_ledger"],
                [320.0] * 8 + [-1000.0] * 12,
            )
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]

            self.assertFalse(lane["allocation_eligible"])
            self.assertFalse(lane["positive_edge_proven"])
            self.assertEqual(
                lane["edge_basis"],
                "EXACT_VEHICLE_ALL_EXIT_CONTRADICTS_TP",
            )
            self.assertTrue(
                lane["capture"]["exact_vehicle_all_exit"][
                    "blocks_tp_exception"
                ]
            )

    def test_tp_proof_binds_explicit_nondivisible_net_before_rounded_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            outcomes = [100.0] * 5 + [100.0001]
            _write_exact_vehicle_ledger(paths["execution_ledger"], outcomes)
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "capture_take_profit_expectancy_jpy": 100.0,
                    "capture_take_profit_net_jpy": 600.0001,
                    "capture_take_profit_trades": 6,
                    "capture_take_profit_wins": 6,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_avg_win_jpy": 100.0,
                    "capture_take_profit_avg_loss_jpy": 0.0,
                    "capture_exact_vehicle_net_trades": 6,
                    "capture_exact_vehicle_net_wins": 6,
                    "capture_exact_vehicle_net_losses": 0,
                    "capture_exact_vehicle_net_jpy": 600.0001,
                    "capture_exact_vehicle_net_expectancy_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]

            self.assertTrue(
                lane["allocation_eligible"],
                json.dumps(lane, sort_keys=True),
            )
            self.assertEqual(lane["edge_basis"], "EXACT_VEHICLE_TAKE_PROFIT")
            self.assertEqual(
                lane["capture"]["exact_vehicle_all_exit"]["net_jpy"],
                600.0001,
            )
            _write_overlay(paths)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_selected_vehicle_loss_stales_packet_but_unrelated_vehicle_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="GBP_USD",
                side="SHORT",
                method="RANGE_ROTATION",
                vehicle="LIMIT",
                realized=-5000.0,
                index=100,
            )

            self.assertEqual(_apply(paths).action, "TRADE")

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="EUR_USD",
                side="LONG",
                method="TREND_CONTINUATION",
                vehicle="MARKET",
                realized=-5000.0,
                index=100,
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

    def test_global_cost_change_is_material_when_unrelated_edge_row_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            baseline = json.loads(paths["baseline"].read_text())
            intents = json.loads(paths["intents"].read_text())
        selected = {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "TREND_CONTINUATION",
            "vehicle": "MARKET",
            "trades": 20,
        }
        surface = {
            "contract": "QR_EXACT_VEHICLE_ALLOCATION_SURFACE_V2",
            "parse_status": "VALID",
            "coverage_start_utc": "2026-01-01T00:00:00Z",
            "exact_vehicle_net": [selected],
            "exact_vehicle_take_profit": [],
            "execution_cost": _synthetic_execution_cost_surface(),
        }
        project = (
            market_read_overlay_module
            ._selected_execution_ledger_allocation_surface
        )
        original = project(
            surface,
            baseline=baseline,
            order_intents=intents,
        )

        unrelated = json.loads(json.dumps(surface))
        unrelated["exact_vehicle_net"].append(
            {
                "pair": "GBP_USD",
                "side": "SHORT",
                "method": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "trades": 999,
            }
        )
        unrelated_projection = project(
            unrelated,
            baseline=baseline,
            order_intents=intents,
        )
        self.assertEqual(
            original["allocation_surface_sha256"],
            unrelated_projection["allocation_surface_sha256"],
        )

        cost_changed = json.loads(json.dumps(surface))
        cost = cost_changed["execution_cost"]
        cost["market_entry"]["adverse_p95_pips"] = 0.2
        cost_material = dict(cost)
        cost_material.pop("execution_cost_surface_sha256", None)
        cost["execution_cost_surface_sha256"] = canonical_json_sha256(
            cost_material
        )
        changed_projection = project(
            cost_changed,
            baseline=baseline,
            order_intents=intents,
        )
        self.assertNotEqual(
            original["allocation_surface_sha256"],
            changed_projection["allocation_surface_sha256"],
        )

    def test_selected_unresolved_reduction_stales_and_blocks_tp_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="EUR_USD",
                side="LONG",
                method="TREND_CONTINUATION",
                vehicle="MARKET",
                realized=50.0,
                index=300,
                terminal=False,
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            all_exit = lane["capture"]["exact_vehicle_all_exit"]
            self.assertFalse(lane["allocation_eligible"])
            self.assertEqual(
                lane["edge_basis"],
                "EXACT_VEHICLE_ALL_EXIT_CONTRADICTS_TP",
            )
            self.assertEqual(all_exit["unresolved_realized_trades"], 1)
            self.assertEqual(all_exit["unresolved_realized_net_jpy"], 50.0)
            self.assertTrue(all_exit["blocks_tp_exception"])

    def test_stored_allocation_board_and_prediction_body_tampering_is_rejected(self) -> None:
        for mutation, expected_code in (
            ("board", "MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE"),
            ("board_context", "MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE"),
            ("packet_context", "MARKET_READ_EVIDENCE_PACKET_BODY_STALE"),
            ("resolved_predictions", "MARKET_READ_EVIDENCE_PACKET_BODY_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                packet = json.loads(paths["packet"].read_text())
                if mutation == "board":
                    packet["capital_allocation_board"]["selected_lane"]["capture"][
                        "take_profit_expectancy_jpy"
                    ] = 999_999.0
                elif mutation == "board_context":
                    packet["capital_allocation_board"]["forecast_context"][
                        "technical_context"
                    ]["technical_context_v1"]["regime"]["dominant"] = "RANGE"
                elif mutation == "packet_context":
                    packet["forecast_context"]["technical_context"][
                        "technical_context_v1"
                    ]["regime"]["dominant"] = "RANGE"
                else:
                    packet["recent_resolved_predictions"].append(
                        {"prediction_id": "mr2:" + "f" * 64, "verdict": "FORGED"}
                    )
                paths["packet"].write_text(json.dumps(packet))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_unknown_or_broad_capture_edge_cannot_receive_allocation(self) -> None:
        for mutation, expected_basis in (
            ("unknown", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("positive_broad", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("broad", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("invalid_scout", "INVALID_PREDICTIVE_SCOUT_CLAIM"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                for key in list(metadata):
                    if key.startswith("capture_take_profit") or key in {
                        "attach_take_profit_on_fill",
                        "tp_execution_mode",
                    }:
                        metadata.pop(key, None)
                metadata.pop("capture_economics_status", None)
                if mutation == "positive_broad":
                    metadata.update(
                        {
                            "capture_economics_status": "POSITIVE_EXPECTANCY",
                            "capture_expectancy_jpy": 100.0,
                        }
                    )
                elif mutation == "broad":
                    metadata.update(
                        {
                            "attach_take_profit_on_fill": True,
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                            "capture_take_profit_scope_key": (
                                "EUR_USD|LONG|TREND_CONTINUATION|TAKE_PROFIT_ORDER"
                            ),
                            "capture_take_profit_expectancy_jpy": 900.0,
                            "capture_take_profit_trades": 100,
                            "capture_take_profit_losses": 0,
                            "capture_take_profit_avg_win_jpy": 900.0,
                        }
                    )
                elif mutation == "invalid_scout":
                    metadata["predictive_scout"] = True
                    metadata["predictive_scout_source"] = "UNVERIFIED_FIXTURE"
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)
                packet = json.loads(paths["packet"].read_text())
                lane = packet["capital_allocation_board"]["selected_lane"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertFalse(lane["positive_edge_proven"])
                self.assertEqual(lane["edge_basis"], expected_basis)
                _write_overlay(paths)

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_CAPITAL_ALLOCATION_EDGE_NOT_PROVEN",
                ):
                    _apply(paths)

    def test_exact_vehicle_all_exit_net_edge_can_receive_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            for key in list(metadata):
                if key.startswith("capture_take_profit"):
                    metadata.pop(key, None)
            metadata.update(
                {
                    "capture_economics_status": "POSITIVE_EXPECTANCY",
                    "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_exact_vehicle_net_scope_key": (
                        "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                    ),
                    "capture_exact_vehicle_net_vehicle": "MARKET",
                    "capture_exact_vehicle_net_metrics_source": (
                        "data/execution_ledger.db:exact_vehicle_net"
                    ),
                    "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                    "capture_exact_vehicle_net_trades": 20,
                    "capture_exact_vehicle_net_wins": 18,
                    "capture_exact_vehicle_net_losses": 2,
                    "capture_exact_vehicle_net_jpy": 1780.0,
                    "capture_exact_vehicle_net_expectancy_jpy": 89.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 10.0,
                    "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                    "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                    "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                        hashlib.sha256(b"[]").hexdigest()
                    ),
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _write_exact_vehicle_ledger(
                paths["execution_ledger"],
                [100.0] * 18 + [-10.0] * 2,
            )
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())["capital_allocation_board"][
                "selected_lane"
            ]
            self.assertTrue(lane["allocation_eligible"])
            self.assertTrue(lane["positive_edge_proven"])
            self.assertEqual(lane["edge_basis"], "EXACT_VEHICLE_ALL_EXIT_NET")
            self.assertGreater(
                lane["capture"]["exact_vehicle_all_exit"][
                    "wilson_stressed_expectancy_jpy"
                ],
                0.0,
            )

            _write_overlay(paths)
            summary = _apply(paths)
            self.assertEqual(summary.action, "TRADE")

    def test_metadata_method_cannot_redirect_typed_market_context_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intent = intents["results"][0]["intent"]
            self.assertEqual(
                intent["market_context"]["method"],
                "TREND_CONTINUATION",
            )
            intent["metadata"]["method"] = "BREAKOUT_FAILURE"
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            lane = packet["capital_allocation_board"]["selected_lane"]
            self.assertFalse(lane["method_scope_consistent"])
            self.assertFalse(lane["positive_edge_proven"])
            self.assertFalse(lane["allocation_eligible"])
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertEqual(lane["edge_basis"], "METHOD_SCOPE_MISMATCH")
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "METHOD_SCOPE_MISMATCH",
            )
            self.assertEqual(lane["numeric_ceiling"]["max_multiple"], 0.0)
            self.assertIsNone(
                packet["execution_ledger_allocation_surface"][
                    "selected_scope_key"
                ]
            )

    def test_exact_vehicle_all_exit_net_edge_rejects_weak_or_mismatched_proof(self) -> None:
        for mutation in (
            "thin_sample",
            "wrong_source",
            "negative_stress",
            "zero_avg_loss",
            "missing_avg_loss",
            "negative_avg_loss",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                for key in list(metadata):
                    if key.startswith("capture_take_profit"):
                        metadata.pop(key, None)
                metadata.update(
                    {
                        "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
                        "capture_exact_vehicle_net_scope_key": (
                            "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                        ),
                        "capture_exact_vehicle_net_vehicle": "MARKET",
                        "capture_exact_vehicle_net_metrics_source": (
                            "data/execution_ledger.db:exact_vehicle_net"
                        ),
                        "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                        "capture_exact_vehicle_net_trades": 20,
                        "capture_exact_vehicle_net_wins": 18,
                        "capture_exact_vehicle_net_losses": 2,
                        "capture_exact_vehicle_net_jpy": 1780.0,
                        "capture_exact_vehicle_net_expectancy_jpy": 89.0,
                        "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                        "capture_exact_vehicle_net_avg_loss_jpy": 10.0,
                    }
                )
                if mutation == "thin_sample":
                    metadata["capture_exact_vehicle_net_trades"] = 19
                    metadata["capture_exact_vehicle_net_wins"] = 17
                elif mutation == "wrong_source":
                    metadata["capture_exact_vehicle_net_metrics_source"] = "handwritten"
                elif mutation == "negative_stress":
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = 1000.0
                elif mutation == "zero_avg_loss":
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = 0.0
                elif mutation == "missing_avg_loss":
                    metadata.pop("capture_exact_vehicle_net_avg_loss_jpy")
                else:
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = -10.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertFalse(
                    lane["capture"]["exact_vehicle_all_exit"]["proven"]
                )

    def test_allocation_schema_rejects_non_string_rationale_and_near_enum_multiple(self) -> None:
        for mutation in ("bool_rationale", "nan_rationale", "near_multiple"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                overlay = _overlay(paths)
                if mutation == "bool_rationale":
                    overlay["capital_allocation"]["rationale"] = True
                elif mutation == "nan_rationale":
                    overlay["capital_allocation"]["rationale"] = float("nan")
                else:
                    overlay["capital_allocation"]["size_multiple"] = 0.5000000000005
                    overlay["capital_allocation"]["selected_units"] = 600
                paths["overlay"].write_text(json.dumps(overlay))

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_CAPITAL_ALLOCATION_INVALID|MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
                ):
                    _apply(paths)

    def test_one_unit_lane_advertises_only_executable_allocation_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intents["results"][0]["intent"]["units"] = 1
            intents["results"][0]["risk_metrics"].update(
                {
                    "jpy_per_pip": 0.01,
                    "risk_jpy": 0.22,
                    "reward_jpy": 0.38,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)
            packet = json.loads(paths["packet"].read_text())

            self.assertEqual(
                packet["capital_allocation_board"]["selected_lane"][
                    "allowed_size_multiples"
                ],
                [1.0],
            )
            _write_overlay(paths, size_multiple=1.0)
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            self.assertEqual(final["capital_allocation"]["selected_units"], 1)

    def test_veto_requires_schema_v2_and_exact_artifact_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, disposition="VETO_WAIT")
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            final["decision_provenance"]["schema_version"] = 1
            issues = validate_codex_market_read_provenance(
                action="WAIT",
                market_read=final["market_read_first"],
                provenance=final["decision_provenance"],
                review=final["market_read_review"],
                counterargument=final["market_read_counterargument"],
                change_summary=final["market_read_change_summary"],
                disposition=final["market_read_disposition"],
                veto_reason=final["market_read_veto_reason"],
                vetoed_lane_ids=tuple(final["market_read_vetoed_lane_ids"]),
                capital_allocation=final["capital_allocation"],
                execution_envelope_sha256=canonical_json_sha256(
                    execution_envelope_payload(final)
                ),
                now=NOW,
            )
            self.assertIn("AI_MARKET_READ_PROVENANCE_INVALID", {code for code, _ in issues})
            artifact_issues = revalidate_codex_market_read_artifacts(
                final_payload=final,
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                overlay_path=paths["overlay"],
                evidence_sources=_sources(paths),
            )
            self.assertIn(
                "AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH",
                {code for code, _ in artifact_issues},
            )

    def test_breakout_failure_allocation_requires_proof_and_blocks_family_conflict(self) -> None:
        aligned = _forecast_context_evidence(
            "EUR_USD",
            1.1001,
            direction="UP",
            family_score=1.0,
            failed_break_long=True,
        )
        aligned_metadata = _forecast_weighting_metadata(aligned)
        aligned_binding = market_read_overlay_module._regime_family_allocation_binding(
            technical_context=aligned,
            pair="EUR_USD",
            side="LONG",
            method="BREAKOUT_FAILURE",
            forecast_direction="UP",
            explicit_receipt_sha256=aligned_metadata[
                "forecast_regime_family_weighting_sha256"
            ],
            explicit_selected_method=aligned_metadata[
                "forecast_regime_family_selected_method"
            ],
            explicit_family_direction=aligned_metadata[
                "forecast_regime_family_direction"
            ],
            canonical_policy_sources=_forecast_policy_source_descriptors(
                aligned
            ),
            canonical_forecast_context_sha256=aligned["context_sha256"],
        )
        self.assertTrue(aligned_binding["passed"])
        self.assertTrue(aligned_binding["failed_break_direction_bound"])
        self.assertEqual(aligned_binding["failed_break_side"], "LONG")

        intermediate_body = json.loads(
            json.dumps(aligned["technical_context_v1"])
        )
        intermediate_body.pop("dynamic_tf_policy_evidence")
        intermediate_body.pop("dynamic_tf_policy_source_context")
        intermediate_body["context_sha256"] = technical_context_sha256(
            intermediate_body
        )
        intermediate = build_forecast_technical_context_evidence(
            intermediate_body,
            pair="EUR_USD",
            current_price=1.1001,
        )
        self.assertEqual(intermediate["status"], "VALID")
        intermediate_metadata = _forecast_weighting_metadata(intermediate)
        intermediate_binding = (
            market_read_overlay_module._regime_family_allocation_binding(
                technical_context=intermediate,
                pair="EUR_USD",
                side="LONG",
                method="BREAKOUT_FAILURE",
                forecast_direction="UP",
                explicit_receipt_sha256=intermediate_metadata[
                    "forecast_regime_family_weighting_sha256"
                ],
                explicit_selected_method=intermediate_metadata[
                    "forecast_regime_family_selected_method"
                ],
                explicit_family_direction=intermediate_metadata[
                    "forecast_regime_family_direction"
                ],
                canonical_policy_sources=_forecast_policy_source_descriptors(
                    aligned
                ),
                canonical_forecast_context_sha256=aligned["context_sha256"],
            )
        )
        self.assertTrue(intermediate_binding["receipt_valid"])
        self.assertFalse(intermediate_binding["current_context_bound"])
        self.assertFalse(intermediate_binding["passed"])

        conflicting = _forecast_context_evidence(
            "EUR_USD",
            1.1001,
            direction="UP",
            family_score=-1.0,
            failed_break_long=True,
        )
        conflicting_metadata = _forecast_weighting_metadata(conflicting)
        conflict_binding = market_read_overlay_module._regime_family_allocation_binding(
            technical_context=conflicting,
            pair="EUR_USD",
            side="LONG",
            method="BREAKOUT_FAILURE",
            forecast_direction="UP",
            explicit_receipt_sha256=conflicting_metadata[
                "forecast_regime_family_weighting_sha256"
            ],
            explicit_selected_method=conflicting_metadata[
                "forecast_regime_family_selected_method"
            ],
            explicit_family_direction=conflicting_metadata[
                "forecast_regime_family_direction"
            ],
            canonical_policy_sources=_forecast_policy_source_descriptors(
                conflicting
            ),
            canonical_forecast_context_sha256=conflicting["context_sha256"],
        )
        self.assertFalse(conflict_binding["passed"])
        self.assertFalse(conflict_binding["direction_consistent"])
        self.assertFalse(conflict_binding["actionable_direction_bound"])

        missing = _forecast_context_evidence("EUR_USD", 1.1001, direction="UP")
        missing_metadata = _forecast_weighting_metadata(missing)
        missing_binding = market_read_overlay_module._regime_family_allocation_binding(
            technical_context=missing,
            pair="EUR_USD",
            side="LONG",
            method="BREAKOUT_FAILURE",
            forecast_direction="UP",
            explicit_receipt_sha256=missing_metadata[
                "forecast_regime_family_weighting_sha256"
            ],
            explicit_selected_method=missing_metadata[
                "forecast_regime_family_selected_method"
            ],
            explicit_family_direction=missing_metadata[
                "forecast_regime_family_direction"
            ],
            canonical_policy_sources=_forecast_policy_source_descriptors(
                missing
            ),
            canonical_forecast_context_sha256=missing["context_sha256"],
        )
        self.assertFalse(missing_binding["passed"])
        self.assertFalse(missing_binding["executable_method_bound"])

    def test_packet_canonical_rebuild_rejects_fully_rehashed_alternate_contexts(self) -> None:
        for mutation in (
            "session_classifier",
            "calendar_profile",
            "evaluated_at",
            "failed_break_proof",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = _prepared_paths(root)
                kwargs: dict = {
                    "now_utc": NOW,
                    "session_tag": "ASIA",
                    "chart_story": (
                        "M5(TREND_STRONG ADX=35) M15(TREND_STRONG ADX=35) "
                        "H1(TREND_STRONG ADX=35) H4(TREND_STRONG ADX=35)"
                    ),
                }
                if mutation == "calendar_profile":
                    calendar = root / "alternate_calendar.json"
                    profile = root / "alternate_profile.json"
                    calendar.write_text(
                        json.dumps(
                            {
                                "events": [
                                    {
                                        "timestamp_utc": (
                                            NOW + timedelta(minutes=10)
                                        ).isoformat(),
                                        "currency": "USD",
                                        "impact": "HIGH",
                                        "title": "Alternate high event",
                                    }
                                ]
                            }
                        )
                    )
                    profile.write_text(
                        json.dumps(
                            {
                                "profiles": [
                                    {
                                        "pair": "EUR_USD",
                                        "method": "TREND_CONTINUATION",
                                        "positive_evidence_n": 1,
                                        "live_net_jpy": -1.0,
                                    }
                                ]
                            }
                        )
                    )
                    kwargs.update(
                        calendar_path=calendar,
                        strategy_profile_path=profile,
                    )
                if mutation == "failed_break_proof":
                    kwargs["failed_break_long"] = True
                if mutation == "evaluated_at":
                    kwargs.update(
                        now_utc=NOW - timedelta(days=1),
                        session_tag="ROLLOVER",
                        chart_story=(
                            "M5(TREND_STRONG ADX=31) M15(TREND_WEAK ADX=24) "
                            "H1(TREND_STRONG ADX=30) H4(TREND_STRONG ADX=30)"
                        ),
                    )

                forged = _forecast_context_evidence(
                    "EUR_USD",
                    1.1001,
                    **kwargs,
                )
                self.assertEqual(forged["status"], "VALID")
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                metadata = result["intent"]["metadata"]
                metadata["forecast_technical_context"] = forged
                metadata.update(_forecast_weighting_metadata(forged))
                if mutation == "failed_break_proof":
                    result["intent"]["market_context"][
                        "method"
                    ] = "BREAKOUT_FAILURE"
                paths["intents"].write_text(json.dumps(intents))

                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                binding = lane["forecast"]["regime_family_binding"]
                self.assertTrue(binding["current_context_bound"])
                self.assertFalse(binding["canonical_forecast_context_bound"])
                self.assertFalse(binding["passed"])
                self.assertFalse(lane["allocation_eligible"])
                self.assertIn(
                    "FORECAST_TECHNICAL_CONTEXT_CANONICAL_REBUILD_MISMATCH",
                    lane["live_blocker_codes"],
                )

    def test_predictive_scout_rebuild_keeps_emission_quote_when_broker_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
                    "predictive_scout_generated_at_utc": NOW.isoformat(),
                    "forecast_cycle_id": (
                        f"pre-entry-forecast-refresh:{NOW.isoformat()}:chart"
                    ),
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot["fetched_at_utc"] = (NOW + timedelta(minutes=2)).isoformat()
            snapshot["quotes"]["EUR_USD"].update(
                {
                    "bid": 1.1010,
                    "ask": 1.1012,
                    "timestamp_utc": (NOW + timedelta(minutes=2)).isoformat(),
                }
            )
            paths["snapshot"].write_text(json.dumps(snapshot))

            _reprepare(paths)

            board = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]
            lane = board["selected_lane"]
            binding = lane["forecast"]["regime_family_binding"]
            self.assertEqual(
                board["canonical_forecast_context_sha256"],
                metadata["forecast_technical_context"]["context_sha256"],
            )
            self.assertTrue(binding["canonical_forecast_context_bound"])
            self.assertTrue(binding["passed"])
            self.assertNotIn(
                "FORECAST_TECHNICAL_CONTEXT_CANONICAL_REBUILD_MISMATCH",
                lane["live_blocker_codes"],
            )

    def test_predictive_scout_point_in_time_inputs_must_all_bind(self) -> None:
        for mutation in ("price", "spread", "clock", "cycle"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                metadata = result["intent"]["metadata"]
                metadata.update(
                    {
                        "predictive_scout": True,
                        "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
                        "predictive_scout_generated_at_utc": NOW.isoformat(),
                        "forecast_cycle_id": (
                            f"pre-entry-forecast-refresh:{NOW.isoformat()}:chart"
                        ),
                    }
                )
                if mutation == "price":
                    metadata["forecast_current_price"] = 1.1002
                elif mutation == "spread":
                    result["risk_metrics"]["spread_pips"] = 3.0
                elif mutation == "clock":
                    metadata["predictive_scout_generated_at_utc"] = (
                        NOW + timedelta(seconds=1)
                    ).isoformat()
                else:
                    metadata["forecast_cycle_id"] = "unbound-cycle"
                paths["intents"].write_text(json.dumps(intents))

                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                binding = lane["forecast"]["regime_family_binding"]
                self.assertFalse(binding["canonical_forecast_context_bound"])
                self.assertFalse(lane["allocation_eligible"])
                self.assertIn(
                    "FORECAST_TECHNICAL_CONTEXT_CANONICAL_REBUILD_MISMATCH",
                    lane["live_blocker_codes"],
                )

    def test_forecast_learning_scout_is_an_allocatable_predictive_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intent = intents["results"][0]["intent"]
            intent["order_type"] = "LIMIT"
            metadata = intent["metadata"]
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
                    "predictive_scout_generated_at_utc": NOW.isoformat(),
                    "forecast_cycle_id": (
                        f"pre-entry-forecast-refresh:{NOW.isoformat()}:chart"
                    ),
                    "campaign_role": "FORECAST_LEARNING_SCOUT",
                    "desk": "trend_trader",
                    "forecast_learning_v1": {
                        "decision_sha256": "direct-learning-decision",
                        "original_direction": "UP",
                        "rank_direction": "UP",
                        "orientation": "DIRECT",
                        "features": {
                            "technical_selected_method": "TREND_CONTINUATION"
                        }
                    },
                }
            )
            metadata["forecast_learning_execution_geometry_v1"] = (
                build_forecast_learning_execution_geometry(
                    pair="EUR_USD",
                    side="LONG",
                    method="TREND_CONTINUATION",
                    entry=1.1002,
                    take_profit=1.1040,
                    stop_loss=1.0980,
                    source_decision_sha256="direct-learning-decision",
                    forecast_current_price=1.1001,
                    forecast_target_price=1.1050,
                    forecast_invalidation_price=1.0985,
                )
            )
            paths["intents"].write_text(json.dumps(intents))

            with patch(
                "quant_rabbit.market_read_overlay.predictive_scout_metadata_supported",
                return_value=True,
            ):
                _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertTrue(lane["allocation_eligible"])
            self.assertTrue(lane["positive_edge_proven"])
            self.assertEqual(
                lane["edge_basis"],
                "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
            )
            self.assertEqual(lane["allowed_size_multiples"], [1.0])

    def test_inverse_forecast_learning_scout_allocates_ranked_side_not_original_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            result = intents["results"][0]
            intent = result["intent"]
            intent.update(
                {
                    "side": "SHORT",
                    "order_type": "LIMIT",
                    "entry": 1.1002,
                    "tp": 1.0964,
                    "sl": 1.1024,
                }
            )
            metadata = intent["metadata"]
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
                    "predictive_scout_generated_at_utc": NOW.isoformat(),
                    "forecast_cycle_id": (
                        f"pre-entry-forecast-refresh:{NOW.isoformat()}:chart"
                    ),
                    "campaign_role": "FORECAST_LEARNING_SCOUT",
                    "desk": "trend_trader",
                    "forecast_learning_v1": {
                        "decision_sha256": "inverse-learning-decision",
                        "original_direction": "UP",
                        "rank_direction": "DOWN",
                        "orientation": "INVERSE",
                        "features": {
                            "technical_selected_method": "TREND_CONTINUATION"
                        },
                    },
                }
            )
            metadata["forecast_learning_execution_geometry_v1"] = (
                build_forecast_learning_execution_geometry(
                    pair="EUR_USD",
                    side="SHORT",
                    method="TREND_CONTINUATION",
                    entry=1.1002,
                    take_profit=1.0964,
                    stop_loss=1.1024,
                    source_decision_sha256="inverse-learning-decision",
                    forecast_current_price=1.1001,
                    forecast_target_price=1.1050,
                    forecast_invalidation_price=1.0985,
                )
            )
            paths["intents"].write_text(json.dumps(intents))

            with patch(
                "quant_rabbit.market_read_overlay.predictive_scout_metadata_supported",
                return_value=True,
            ):
                _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]
            lane = board["selected_lane"]
            self.assertTrue(lane["allocation_eligible"])
            self.assertTrue(lane["forecast_learning"]["valid"])
            self.assertEqual(lane["allowed_size_multiples"], [1.0])
            self.assertEqual(
                lane["edge_basis"],
                "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
            )
            self.assertEqual(
                board["forecast_context_scope"],
                "SELECTED_LANE_FORECAST_ORIENTATION_LEARNING",
            )
            self.assertEqual(board["forecast_context"]["direction"], "DOWN")
            self.assertEqual(
                board["forecast_context"]["original_direction"],
                "UP",
            )

            short_read = _market_read()
            short_read["naked_read"].update(
                {
                    "currency_bought": "USD",
                    "currency_sold": "EUR",
                    "what_price_is_trying_to_do_now": (
                        "EUR_USD is trying to reject 1.1002 and rotate lower."
                    ),
                }
            )
            for key, target, invalidation in (
                ("next_30m_prediction", "1.0985 to 1.0990", "1.1024"),
                ("next_2h_prediction", "1.0964 to 1.0980", "1.1024"),
            ):
                short_read[key].update(
                    {
                        "direction": "SHORT",
                        "expected_path": "Reject the passive entry and rotate lower.",
                        "target_zone": target,
                        "invalidation": invalidation,
                    }
                )
            short_read["best_trade_if_forced"].update(
                {
                    "direction": "SHORT",
                    "vehicle": "LIMIT",
                    "entry": "1.1002",
                    "tp": "1.0964",
                    "sl": "1.1024",
                }
            )
            _write_overlay(
                paths,
                size_multiple=1.0,
                market_read=short_read,
            )
            with patch(
                "quant_rabbit.market_read_overlay.predictive_scout_metadata_supported",
                return_value=True,
            ):
                _apply(paths)
            final = json.loads(paths["output"].read_text())
            self.assertEqual(final["action"], "TRADE")
            self.assertEqual(final["capital_allocation"]["selected_units"], 1200)

    def test_trade_allocation_is_bound_to_numeric_lane_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]

            self.assertEqual(board["selected_lane"]["base_units"], 1200)
            self.assertEqual(
                board["selected_lane"]["forecast"]["calibration_name"],
                "directional_forecast_up",
            )
            self.assertEqual(
                board["selected_lane"]["capture"]["take_profit_expectancy_jpy"],
                320.0,
            )
            self.assertTrue(board["selected_lane"]["allocation_eligible"])
            _write_overlay(paths, size_multiple=0.75)

            _apply(paths)

            final = json.loads(paths["output"].read_text())
            allocation = final["capital_allocation"]
            self.assertEqual(allocation["decision"], "ALLOCATE")
            self.assertEqual(allocation["lane_id"], LANE_ID)
            self.assertEqual(allocation["size_multiple"], 0.75)
            self.assertEqual(allocation["selected_units"], 900)
            self.assertEqual(
                final["decision_provenance"]["authorized_units"],
                900,
            )

    def test_selected_lane_context_is_content_addressed_into_gpt_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]
            forecast_context = board["forecast_context"]
            lane_context = board["selected_lane"]["forecast"]["technical_context"]

            self.assertEqual(board["forecast_context_scope"], "SELECTED_LANE")
            self.assertEqual(forecast_context["technical_context"], lane_context)
            self.assertEqual(lane_context["status"], "VALID")
            self.assertEqual(
                lane_context["confidence_semantics"],
                CONFIDENCE_SEMANTICS,
            )
            self.assertEqual(forecast_context["candidate_count"], 1)
            self.assertEqual(forecast_context["valid_candidate_count"], 1)
            self.assertEqual(forecast_context["invalid_candidate_count"], 0)
            self.assertEqual(packet["forecast_context_scope"], "SELECTED_LANE")
            self.assertEqual(packet["forecast_context"], forecast_context)
            self.assertEqual(
                packet["capital_allocation_board_sha256"],
                canonical_json_sha256(board),
            )
            receipt = lane_context["technical_context_v1"][
                "regime_family_weighting"
            ]
            self.assertEqual(
                board["selected_lane"]["forecast"][
                    "regime_family_weighting_sha256"
                ],
                receipt["receipt_sha256"],
            )

    def test_fresh_allocation_requires_bound_actionable_family_receipt(self) -> None:
        for mutation, blocker in (
            ("missing_sha", "REGIME_FAMILY_WEIGHTING_SHA_UNBOUND"),
            (
                "reverse_direction",
                "REGIME_FAMILY_WEIGHTING_FORECAST_CONTRADICTION",
            ),
            (
                "non_directional",
                "REGIME_FAMILY_WEIGHTING_DIRECTION_NOT_ACTIONABLE",
            ),
            (
                "legacy_display",
                "REGIME_FAMILY_WEIGHTING_MISSING_OR_INVALID",
            ),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                if mutation == "missing_sha":
                    metadata.pop("forecast_regime_family_weighting_sha256")
                elif mutation in {"reverse_direction", "non_directional"}:
                    evidence = _forecast_context_evidence(
                        "EUR_USD",
                        1.1001,
                        direction=("DOWN" if mutation == "reverse_direction" else "UP"),
                        family_score=(0.0 if mutation == "non_directional" else None),
                    )
                    metadata["forecast_technical_context"] = evidence
                    metadata.update(_forecast_weighting_metadata(evidence))
                else:
                    evidence = metadata["forecast_technical_context"]
                    legacy_body = evidence["technical_context_v1"]
                    legacy_body.pop("regime_family_weighting")
                    legacy_body["context_sha256"] = technical_context_sha256(
                        legacy_body
                    )
                    metadata["forecast_technical_context"] = (
                        build_forecast_technical_context_evidence(
                            legacy_body,
                            pair="EUR_USD",
                            current_price=1.1001,
                        )
                    )
                    metadata.pop("forecast_regime_family_weighting_sha256")
                    metadata.pop("forecast_regime_family_selected_method")
                    metadata.pop("forecast_regime_family_direction")
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(
                    lane["forecast"]["technical_context"]["status"],
                    "VALID",
                )
                self.assertFalse(lane["allocation_eligible"])
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertIn(blocker, lane["live_blocker_codes"])

    def test_duplicate_selected_lane_id_is_ambiguous_and_cannot_allocate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            duplicate = json.loads(json.dumps(intents["results"][0]))
            duplicate["intent"]["metadata"].pop("forecast_technical_context")
            intents["results"].append(duplicate)
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]
            context = board["forecast_context"]["technical_context"]
            self.assertEqual(board["selected_lane_match_count"], 2)
            self.assertFalse(board["selected_lane_unique"])
            self.assertIsNone(board["selected_lane"])
            self.assertEqual(board["forecast_context_scope"], "SELECTED_LANE_INVALID")
            self.assertEqual(context["status"], "UNKNOWN")
            self.assertEqual(
                context["reason"],
                "FORECAST_TECHNICAL_CONTEXT_SELECTED_LANE_DUPLICATE",
            )

            _write_overlay(paths)
            with self.assertRaises(MarketReadOverlayError) as caught:
                _apply(paths)
            self.assertEqual(
                caught.exception.code,
                "MARKET_READ_CAPITAL_ALLOCATION_LANE_MISSING",
            )

    def test_missing_invalid_or_mismatched_selected_context_blocks_allocation(self) -> None:
        for mutation, expected_reason in (
            ("missing", "TECHNICAL_CONTEXT_EVIDENCE_MISSING"),
            ("missing_price", "TECHNICAL_CONTEXT_EVIDENCE_PRICE_MISSING"),
            ("body_tamper", "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH"),
            ("pair_mismatch", "TECHNICAL_CONTEXT_PAIR_MISMATCH"),
            ("price_mismatch", "TECHNICAL_CONTEXT_PRICE_MISMATCH"),
            ("oversized_unknown", "TECHNICAL_CONTEXT_EVIDENCE_TOO_LARGE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                if mutation == "missing":
                    metadata.pop("forecast_technical_context")
                elif mutation == "missing_price":
                    metadata.pop("forecast_current_price")
                elif mutation == "body_tamper":
                    metadata["forecast_technical_context"]["technical_context_v1"][
                        "regime"
                    ]["dominant"] = "RANGE"
                elif mutation == "pair_mismatch":
                    metadata["forecast_technical_context"] = (
                        _forecast_context_evidence("GBP_USD", 1.1001)
                    )
                elif mutation == "oversized_unknown":
                    marker = "DO_NOT_FORWARD_TO_BOARD"
                    oversized = metadata["forecast_technical_context"]
                    oversized.update(
                        {
                            "status": "UNKNOWN",
                            "reason": marker * MAX_EVIDENCE_BYTES,
                            "technical_context_v1": None,
                            "context_sha256": None,
                        }
                    )
                    oversized["evidence_sha256"] = canonical_json_sha256(
                        {
                            key: item
                            for key, item in oversized.items()
                            if key != "evidence_sha256"
                        }
                    )
                else:
                    metadata["forecast_technical_context"] = (
                        _forecast_context_evidence("EUR_USD", 1.2001)
                    )
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                board = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]
                lane = board["selected_lane"]
                context = lane["forecast"]["technical_context"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertFalse(lane["positive_edge_proven"])
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertEqual(
                    lane["edge_basis"],
                    "FORECAST_TECHNICAL_CONTEXT_UNKNOWN",
                )
                self.assertIn(
                    "FORECAST_TECHNICAL_CONTEXT_UNKNOWN_FOR_ALLOCATION",
                    lane["live_blocker_codes"],
                )
                self.assertEqual(context["status"], "UNKNOWN")
                self.assertEqual(context["reason"], expected_reason)
                self.assertIsNone(context["technical_context_v1"])
                self.assertFalse(context["live_permission"])
                self.assertEqual(board["forecast_context_scope"], "SELECTED_LANE")
                self.assertEqual(
                    board["forecast_context"]["invalid_candidate_count"],
                    1,
                )
                if mutation == "oversized_unknown":
                    self.assertNotIn(marker, json.dumps(board))

    def test_forced_prediction_context_requires_valid_unambiguous_candidates(self) -> None:
        for mutation, expected_status, expected_reason in (
            ("none", "VALID", None),
            (
                "one_invalid",
                "UNKNOWN",
                "FORECAST_TECHNICAL_CONTEXT_CANDIDATE_INVALID",
            ),
            (
                "ambiguous",
                "UNKNOWN",
                "FORECAST_TECHNICAL_CONTEXT_AMBIGUOUS",
            ),
            (
                "side_mismatch",
                "UNKNOWN",
                "FORECAST_TECHNICAL_CONTEXT_CANDIDATE_INVALID",
            ),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(
                    Path(tmp),
                    baseline=_baseline(action="REQUEST_EVIDENCE", lane_ids=[]),
                )
                if mutation != "none":
                    intents = json.loads(paths["intents"].read_text())
                    duplicate = json.loads(json.dumps(intents["results"][0]))
                    duplicate["lane_id"] = f"{LANE_ID}:{mutation}"
                    metadata = duplicate["intent"]["metadata"]
                    if mutation == "one_invalid":
                        metadata["forecast_technical_context"] = (
                            _forecast_context_evidence("GBP_USD", 1.1001)
                        )
                    elif mutation == "side_mismatch":
                        duplicate["intent"]["side"] = "SHORT"
                    else:
                        metadata["forecast_technical_context"] = (
                            _forecast_context_evidence(
                                "EUR_USD",
                                1.1001,
                                direction="DOWN",
                            )
                        )
                    intents["results"].append(duplicate)
                    paths["intents"].write_text(json.dumps(intents))
                    _reprepare(paths)

                board = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]
                context = board["forecast_context"]
                technical_context = context["technical_context"]
                self.assertEqual(board["forecast_context_scope"], "FORCED_PREDICTION")
                self.assertEqual(technical_context["status"], expected_status)
                self.assertEqual(technical_context["reason"], expected_reason)
                if mutation == "none":
                    self.assertEqual(context["candidate_count"], 1)
                    self.assertEqual(context["valid_candidate_count"], 1)
                    self.assertEqual(context["invalid_candidate_count"], 0)
                elif mutation in {"one_invalid", "side_mismatch"}:
                    self.assertEqual(context["candidate_count"], 2)
                    self.assertEqual(context["valid_candidate_count"], 1)
                    self.assertEqual(context["invalid_candidate_count"], 1)
                else:
                    self.assertEqual(context["candidate_count"], 2)
                    self.assertEqual(context["valid_candidate_count"], 2)
                    self.assertEqual(context["invalid_candidate_count"], 0)
                    self.assertEqual(context["distinct_context_sha256_count"], 2)

    def test_numeric_ceiling_fails_closed_on_missing_or_invalid_inputs(self) -> None:
        cases = (
            ("missing_economic_rate", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("bool_economic_rate", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("missing_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("bool_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("thin_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("missing_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("bool_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("nan_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("economic_rate_over_one", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("wrong_calibration", "FORECAST_CALIBRATION_IDENTITY_MISMATCH"),
            ("reward_risk_mismatch", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                metadata = result["intent"]["metadata"]
                if mutation == "missing_economic_rate":
                    metadata.pop("forecast_directional_economic_hit_rate")
                elif mutation == "bool_economic_rate":
                    metadata["forecast_directional_economic_hit_rate"] = True
                elif mutation == "missing_economic_samples":
                    metadata.pop("forecast_directional_economic_samples")
                elif mutation == "bool_economic_samples":
                    metadata["forecast_directional_economic_samples"] = True
                elif mutation == "thin_economic_samples":
                    metadata["forecast_directional_economic_samples"] = (
                        CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES - 1
                    )
                elif mutation in {"missing_nav", "bool_nav", "nan_nav"}:
                    snapshot = json.loads(paths["snapshot"].read_text())
                    if mutation == "missing_nav":
                        snapshot.pop("account")
                    elif mutation == "bool_nav":
                        snapshot["account"]["nav_jpy"] = True
                    else:
                        snapshot["account"]["nav_jpy"] = float("nan")
                    paths["snapshot"].write_text(json.dumps(snapshot))
                elif mutation == "economic_rate_over_one":
                    metadata["forecast_directional_economic_hit_rate"] = 1.01
                elif mutation == "wrong_calibration":
                    metadata["forecast_directional_calibration_name"] = (
                        "directional_forecast_down"
                    )
                else:
                    result["risk_metrics"]["reward_risk"] = 2.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(
                    lane["positive_edge_proven"],
                    mutation != "wrong_calibration",
                )
                self.assertFalse(lane["allocation_eligible"])
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)

    def test_headline_precision_cannot_hide_negative_economic_ev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "forecast_directional_hit_rate": 1.0,
                    "forecast_directional_samples": 100,
                    "forecast_directional_economic_hit_rate": 0.30,
                    "forecast_directional_economic_samples": 100,
                    "forecast_directional_timeout_rate": 0.70,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                ceiling["inputs"]["headline_hit_rate_context_only"],
                1.0,
            )
            self.assertLess(
                ceiling["probability"]["economic_wilson95_lower"],
                0.30,
            )
            self.assertFalse(ceiling["ev_lower"]["positive"])
            self.assertEqual(ceiling["reason"], "CONSERVATIVE_EV_NOT_POSITIVE")
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertFalse(lane["allocation_eligible"])

    def test_numeric_ceiling_rejects_direction_or_forecast_rail_mismatch(self) -> None:
        cases = (
            ("direction", "FORECAST_DIRECTION_SIDE_MISMATCH"),
            ("target", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
            ("invalidation", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                if mutation == "direction":
                    metadata["forecast_direction"] = "DOWN"
                elif mutation == "target":
                    metadata["forecast_target_price"] = 1.1030
                else:
                    metadata["forecast_invalidation_price"] = 1.0970
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)

    def test_numeric_ceiling_rejects_forecast_current_outside_rails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            # Current broker mid remains 1.1001, but the half-spread-adjusted
            # invalidation is now above it, so the forecast rail is already
            # invalid before this order can be sized.
            metadata["forecast_invalidation_price"] = 1.1003
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertTrue(
                ceiling["geometry"]["forecast_current_matches_broker_mid"]
            )
            self.assertFalse(
                ceiling["geometry"][
                    "forecast_outcome_conservatively_contains_order"
                ]
            )
            self.assertEqual(
                ceiling["reason"],
                "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_numeric_ceiling_rejects_half_spread_and_broker_metric_tampering(self) -> None:
        cases = (
            ("half_spread_boundary", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
            ("current_mid", "FORECAST_CURRENT_BROKER_MID_MISMATCH"),
            ("spread", "BROKER_SPREAD_RISK_METRICS_MISMATCH"),
            ("reward_pips", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
            ("scaled_jpy_metrics", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                intent = result["intent"]
                metadata = intent["metadata"]
                risk_metrics = result["risk_metrics"]
                if mutation == "half_spread_boundary":
                    # Raw SL<=invalidation passes, but the bid-side
                    # invalidation after half-spread does not protect the SL.
                    metadata["forecast_invalidation_price"] = (
                        intent["sl"] + 0.00005
                    )
                elif mutation == "current_mid":
                    metadata["forecast_current_price"] = 1.10011
                elif mutation == "spread":
                    risk_metrics["spread_pips"] = 2.1
                elif mutation == "reward_pips":
                    risk_metrics["reward_pips"] = 39.0
                else:
                    risk_metrics["jpy_per_pip"] *= 2.0
                    risk_metrics["risk_jpy"] *= 2.0
                    risk_metrics["reward_jpy"] *= 2.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)
                self.assertEqual(lane["allowed_size_multiples"], [])

    def test_pending_entry_is_unbound_and_mismatch_is_explicit(self) -> None:
        for mutation, expected_reason in (
            ("matched", "FORECAST_ECONOMIC_PROBABILITY_ENTRY_VEHICLE_UNBOUND"),
            ("mismatched", "ORDER_ENTRY_RISK_METRICS_BINDING_INVALID"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                intent = intents["results"][0]["intent"]
                intent["order_type"] = "STOP"
                intent["entry"] = (
                    intents["results"][0]["risk_metrics"]["entry_price"]
                    if mutation == "matched"
                    else 1.1003
                )
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                ceiling = lane["numeric_ceiling"]
                self.assertEqual(ceiling["reason"], expected_reason)
                self.assertEqual(lane["allowed_size_multiples"], [])
                if mutation == "matched":
                    self.assertTrue(ceiling["geometry"]["entry_binding_passed"])
                else:
                    self.assertFalse(ceiling["geometry"]["entry_binding_passed"])

    def test_market_entry_must_equal_fresh_broker_executable_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                ceiling["geometry"]["entry_binding_basis"],
                "MARKET_RISK_ENTRY_EQUALS_FRESH_BROKER_ASK_OR_BID",
            )
            self.assertTrue(ceiling["geometry"]["entry_binding_passed"])

            intents = json.loads(paths["intents"].read_text())
            intents["results"][0]["risk_metrics"]["entry_price"] = 1.1003
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "ORDER_ENTRY_RISK_METRICS_BINDING_INVALID",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_gross_break_even_ev_is_net_negative_after_cost_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            result = intents["results"][0]
            intent = result["intent"]
            risk_metrics = result["risk_metrics"]
            metadata = intent["metadata"]
            p_lower = hit_rate_wilson_lower(
                metadata["forecast_directional_economic_hit_rate"],
                metadata["forecast_directional_economic_samples"],
            )
            self.assertIsNotNone(p_lower)
            assert p_lower is not None
            break_even_rr = (1.0 - p_lower) / p_lower
            loss_pips = risk_metrics["loss_pips"]
            reward_pips = loss_pips * break_even_rr
            intent["tp"] = risk_metrics["entry_price"] + reward_pips / 10_000.0
            risk_metrics["reward_pips"] = reward_pips
            risk_metrics["reward_jpy"] = reward_pips * risk_metrics["jpy_per_pip"]
            risk_metrics["reward_risk"] = break_even_rr
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            gross_ev = (
                p_lower * risk_metrics["reward_jpy"]
                - (1.0 - p_lower) * risk_metrics["risk_jpy"]
            )
            self.assertAlmostEqual(
                gross_ev,
                0.0,
                places=8,
            )
            self.assertGreater(
                ceiling["ev_lower"]["additional_cost_jpy"],
                0.0,
            )
            self.assertAlmostEqual(
                ceiling["ev_lower"]["value_jpy_snapshot"],
                -ceiling["ev_lower"]["additional_cost_jpy"],
                places=8,
            )
            self.assertFalse(ceiling["ev_lower"]["positive"])
            self.assertEqual(ceiling["reason"], "CONSERVATIVE_EV_NOT_POSITIVE")
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_short_numeric_ceiling_uses_symmetric_forecast_rail(self) -> None:
        intent = {
            "pair": "EUR_USD",
            "side": "SHORT",
            "order_type": "MARKET",
            "units": 1000,
            "tp": 1.0960,
            "sl": 1.1030,
            "market_context": {"method": "TREND_CONTINUATION"},
        }
        metadata = {
            "forecast_direction": "DOWN",
            "forecast_directional_calibration_name": "directional_forecast_down",
            "forecast_current_price": 1.1001,
            "forecast_target_price": 1.0950,
            "forecast_invalidation_price": 1.1020,
            "forecast_directional_economic_hit_rate": 0.70,
            "forecast_directional_economic_samples": 100,
            "forecast_directional_hit_rate": 0.75,
            "forecast_directional_samples": 100,
            "forecast_directional_timeout_rate": 0.10,
        }
        risk_metrics = {
            "entry_price": 1.1000,
            "risk_jpy": 300.0,
            "reward_jpy": 400.0,
            "loss_pips": 30.0,
            "reward_pips": 40.0,
            "jpy_per_pip": 10.0,
            "reward_risk": 4.0 / 3.0,
            "spread_pips": 2.0,
        }
        evidence, max_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|SHORT|TREND_CONTINUATION|MARKET"
                ),
            )
        )
        self.assertTrue(evidence["geometry"]["passed"])
        self.assertEqual(max_multiple, 1.0)
        self.assertEqual(
            evidence["ev_lower"]["additional_cost_jpy"],
            20.0,
        )
        self.assertEqual(evidence["ev_lower"]["net_risk_jpy_snapshot"], 320.0)
        self.assertEqual(evidence["ev_lower"]["net_reward_jpy_snapshot"], 380.0)
        p_lower = evidence["probability"]["economic_wilson95_lower"]
        self.assertAlmostEqual(
            evidence["ev_lower"]["value_jpy_snapshot"],
            p_lower * 400.0 - (1.0 - p_lower) * 300.0 - 20.0,
            places=8,
        )

        wrong_scope_evidence, wrong_scope_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|LONG|UNKNOWN|MARKET"
                ),
            )
        )
        self.assertEqual(wrong_scope_multiple, 0.0)
        self.assertEqual(
            wrong_scope_evidence["reason"],
            "NET_EXECUTION_COST_FLOOR_MISSING_INVALID_OR_STALE",
        )

        metadata["forecast_target_price"] = 1.0970
        evidence, max_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|SHORT|TREND_CONTINUATION|MARKET"
                ),
            )
        )
        self.assertFalse(evidence["geometry"]["passed"])
        self.assertEqual(max_multiple, 0.0)

    def test_range_tp_prebounded_numeric_contract_is_rail_cost_and_side_bound(self) -> None:
        def fixture(
            side: str,
            *,
            entry_override: float | None = None,
            financing_per_unit: float = 0.0,
        ) -> tuple[dict, dict, dict, dict, dict]:
            low = 1.0990
            high = 1.1012
            long = side == "LONG"
            entry = entry_override if entry_override is not None else (
                1.0991 if long else 1.1011
            )
            tp = 1.1009 if long else 1.0993
            sl = 1.0988 if long else 1.1014
            loss_pips = abs(entry - sl) * 10_000.0
            reward_pips = abs(tp - entry) * 10_000.0
            jpy_per_pip = 12.0
            trades = 20
            wins = 20
            wilson = hit_rate_wilson_lower(wins / trades, trades)
            self.assertIsNotNone(wilson)
            assert wilson is not None
            basis = {
                "basis": "EXACT_TP_PROVEN_HARVEST",
                "conditional_tp_exit_win_rate": wins / trades,
                "tp_exit_samples": trades,
                "minimum_tp_exit_samples": 20,
                "conditional_tp_exit_wilson95_lower": wilson,
                "stressed_harvest_expectancy_jpy": (
                    wilson * 320.0 - (1.0 - wilson) * 300.0
                ),
                "execution_ledger_surface_sha256": "a" * 64,
            }
            intent = {
                "pair": "EUR_USD",
                "side": side,
                "order_type": "LIMIT",
                "units": 1200,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "market_context": {"method": "RANGE_ROTATION"},
            }
            metadata = {
                "method": "RANGE_ROTATION",
                "forecast_direction": "RANGE",
                "forecast_directional_calibration_name": (
                    "directional_forecast_range"
                ),
                "forecast_current_price": 1.1001,
                "forecast_range_low_price": low,
                "forecast_range_high_price": high,
                "range_support": low,
                "range_resistance": high,
                "range_entry_side": "support" if long else "resistance",
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                # These describe box integrity only and must never become the
                # side probability/Kelly input for this exception.
                "forecast_directional_economic_hit_rate": 1.0,
                "forecast_directional_economic_samples": 1000,
                "max_loss_jpy": 400.0,
            }
            risk = {
                "entry_price": entry,
                "risk_jpy": loss_pips * jpy_per_pip,
                "reward_jpy": reward_pips * jpy_per_pip,
                "loss_pips": loss_pips,
                "reward_pips": reward_pips,
                "jpy_per_pip": jpy_per_pip,
                "reward_risk": reward_pips / loss_pips,
                "spread_pips": 2.0,
            }
            cost_material = {
                "contract": "QR_NET_EXECUTION_COST_FLOOR_V1",
                "status": "PASSED",
                "market_entry_adverse_p95_pips": 0.0,
                "audited_protected_exit_adverse_p95_pips": 0.0,
                "financing_adverse_stress_jpy_per_unit": financing_per_unit,
                "scope_key": f"EUR_USD|{side}|RANGE_ROTATION|LIMIT",
                "spread_double_count_forbidden": True,
            }
            cost = {
                **cost_material,
                "proof_sha256": canonical_json_sha256(cost_material),
            }
            return intent, metadata, risk, basis, cost

        for side in ("LONG", "SHORT"):
            with self.subTest(side=side):
                intent, metadata, risk, basis, cost = fixture(side)
                evidence, maximum = (
                    market_read_overlay_module
                    ._capital_allocation_numeric_ceiling(
                        intent=intent,
                        metadata=metadata,
                        risk_metrics=risk,
                        account_nav_jpy=100_000.0,
                        broker_bid=1.1000,
                        broker_ask=1.1002,
                        broker_quote_to_jpy=100.0,
                        predictive_scout=False,
                        hedge=False,
                        range_economic_basis=basis,
                        execution_cost_floor=cost,
                    )
                )
                self.assertEqual(maximum, 1.0)
                self.assertEqual(
                    evidence["reason"],
                    "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT",
                )
                self.assertTrue(evidence["geometry"]["range_rails_bound"])
                self.assertTrue(
                    evidence["geometry"]["range_entry_exact_rail_bound"]
                )
                self.assertIsNone(evidence["inputs"]["economic_hit_rate"])
                self.assertIsNone(
                    evidence["probability"]["economic_wilson95_lower"]
                )
                self.assertIsNone(evidence["ev_lower"]["formula"])
                self.assertIsNone(evidence["kelly"]["formula"])

        intent, metadata, risk, basis, cost = fixture(
            "LONG", entry_override=1.1000
        )
        evidence, maximum = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                range_economic_basis=basis,
                execution_cost_floor=cost,
            )
        )
        self.assertEqual(maximum, 0.0)
        self.assertFalse(evidence["geometry"]["range_entry_exact_rail_bound"])

        for mutation in ("side_label", "stop_vehicle", "legacy_basis"):
            intent, metadata, risk, basis, cost = fixture("LONG")
            if mutation == "side_label":
                metadata["range_entry_side"] = "resistance"
            elif mutation == "stop_vehicle":
                intent["order_type"] = "STOP"
                cost = _synthetic_execution_cost_floor(
                    scope_key="EUR_USD|LONG|RANGE_ROTATION|STOP"
                )
            else:
                basis = {
                    "basis": "EXACT_TP_PROVEN_HARVEST",
                    "hit_rate": 1.0,
                    "samples": 20,
                    "wilson95_lower": 0.8,
                    "stressed_expectancy_jpy": 100.0,
                    "execution_ledger_surface_sha256": "a" * 64,
                }
            evidence, maximum = (
                market_read_overlay_module
                ._capital_allocation_numeric_ceiling(
                    intent=intent,
                    metadata=metadata,
                    risk_metrics=risk,
                    account_nav_jpy=100_000.0,
                    broker_bid=1.1000,
                    broker_ask=1.1002,
                    broker_quote_to_jpy=100.0,
                    predictive_scout=False,
                    hedge=False,
                    range_economic_basis=basis,
                    execution_cost_floor=cost,
                )
            )
            with self.subTest(mutation=mutation):
                self.assertEqual(maximum, 0.0)

        intent, metadata, risk, basis, cost = fixture(
            "LONG", financing_per_unit=0.5
        )
        evidence, maximum = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                range_economic_basis=basis,
                execution_cost_floor=cost,
            )
        )
        self.assertEqual(maximum, 0.0)
        self.assertEqual(
            evidence["reason"],
            "TP_PROVEN_RANGE_PREBOUNDED_RISK_CAP_INVALID",
        )

    def test_range_rotation_generated_board_uses_tp_basis_not_box_precision(self) -> None:
        range_lane_id = (
            "range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(
                Path(tmp),
                baseline=_baseline(lane_ids=[range_lane_id]),
            )
            baseline = json.loads(paths["baseline"].read_text())
            baseline["method"] = "RANGE_ROTATION"
            baseline["evidence_refs"] = [
                f"intent:{range_lane_id}",
                "broker:snapshot",
            ]
            paths["baseline"].write_text(json.dumps(baseline))

            context = _forecast_context_evidence(
                "EUR_USD",
                1.1001,
                direction="RANGE",
                family_score=1.0,
                now_utc=NOW,
                pair_charts_path=paths["pair_charts"],
                session_tag="LONDON",
                chart_story=(
                    "M5(RANGE ADX=16) M15(RANGE ADX=17) "
                    "H1(RANGE ADX=18) H4(RANGE ADX=19)"
                ),
            )
            wilson = hit_rate_wilson_lower(1.0, 20)
            self.assertIsNotNone(wilson)
            assert wilson is not None
            metadata = {
                "method": "RANGE_ROTATION",
                "position_intent": "NEW",
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "positive_rotation_mode": "TP_PROVEN_HARVEST",
                "positive_rotation_live_ready": True,
                "positive_rotation_tp_trades": 20,
                "positive_rotation_tp_wins": 20,
                "positive_rotation_loss_proxy_jpy": 300.0,
                "positive_rotation_tp_win_rate_lower": round(wilson, 6),
                "positive_rotation_pessimistic_expectancy_jpy": round(
                    wilson * 320.0 - (1.0 - wilson) * 300.0,
                    4,
                ),
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_expectancy_jpy": -50.0,
                "capture_avg_win_jpy": 320.0,
                "capture_avg_loss_jpy": 300.0,
                "capture_take_profit_exact_vehicle_required": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                "capture_take_profit_scope_key": (
                    "EUR_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER"
                ),
                "capture_take_profit_vehicle": "LIMIT",
                "capture_take_profit_metrics_source": (
                    "data/execution_ledger.db:exact_vehicle_take_profit"
                ),
                "capture_take_profit_expectancy_jpy": 320.0,
                "capture_take_profit_net_jpy": 6400.0,
                "capture_take_profit_trades": 20,
                "capture_take_profit_wins": 20,
                "capture_take_profit_losses": 0,
                "capture_take_profit_avg_win_jpy": 320.0,
                "capture_take_profit_avg_loss_jpy": 0.0,
                "capture_exact_vehicle_net_scope": (
                    "PAIR_SIDE_METHOD_VEHICLE"
                ),
                "capture_exact_vehicle_net_scope_key": (
                    "EUR_USD|LONG|RANGE_ROTATION|LIMIT|ALL_AUDITED_EXITS"
                ),
                "capture_exact_vehicle_net_vehicle": "LIMIT",
                "capture_exact_vehicle_net_metrics_source": (
                    "data/execution_ledger.db:exact_vehicle_net"
                ),
                "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                "capture_exact_vehicle_net_trades": 20,
                "capture_exact_vehicle_net_wins": 20,
                "capture_exact_vehicle_net_losses": 0,
                "capture_exact_vehicle_net_jpy": 6400.0,
                "capture_exact_vehicle_net_expectancy_jpy": 320.0,
                "capture_exact_vehicle_net_avg_win_jpy": 320.0,
                "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
                "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                    hashlib.sha256(b"[]").hexdigest()
                ),
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.80,
                "forecast_current_price": 1.1001,
                "forecast_range_low_price": 1.0990,
                "forecast_range_high_price": 1.1012,
                "forecast_target_price": 1.1012,
                "forecast_invalidation_price": 1.0990,
                "forecast_directional_calibration_name": (
                    "directional_forecast_range"
                ),
                "forecast_directional_economic_hit_rate": 1.0,
                "forecast_directional_economic_samples": 1000,
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 1000,
                "forecast_directional_timeout_rate": 0.0,
                "forecast_technical_context": context,
                **_forecast_weighting_metadata(context),
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_support": 1.0990,
                "range_resistance": 1.1012,
                "range_entry_side": "support",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "max_loss_jpy": 400.0,
            }
            paths["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": range_lane_id,
                                "status": "LIVE_READY",
                                "risk_allowed": True,
                                "live_blocker_codes": [],
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "units": 1200,
                                    "entry": 1.0991,
                                    "tp": 1.1009,
                                    "sl": 1.0988,
                                    "market_context": {
                                        "method": "RANGE_ROTATION"
                                    },
                                    "metadata": metadata,
                                },
                                "risk_metrics": {
                                    "entry_price": 1.0991,
                                    "loss_pips": 3.0,
                                    "reward_pips": 18.0,
                                    "risk_jpy": 36.0,
                                    "reward_jpy": 216.0,
                                    "reward_risk": 6.0,
                                    "spread_pips": 2.0,
                                    "jpy_per_pip": 12.0,
                                    "estimated_margin_jpy": 1200.0,
                                },
                            }
                        ],
                    }
                )
            )
            _write_exact_vehicle_ledger(paths["execution_ledger"], [])
            for index in range(20):
                _append_vehicle_trade(
                    paths["execution_ledger"],
                    pair="EUR_USD",
                    side="LONG",
                    method="RANGE_ROTATION",
                    vehicle="LIMIT",
                    realized=320.0,
                    index=500 + index,
                )
            _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            lane = packet["capital_allocation_board"]["selected_lane"]
            self.assertTrue(
                lane["allocation_eligible"],
                json.dumps(lane, sort_keys=True),
            )
            self.assertEqual(
                lane["edge_basis"], "EXACT_VEHICLE_TAKE_PROFIT"
            )
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT",
            )
            self.assertIsNone(
                lane["numeric_ceiling"]["inputs"]["economic_hit_rate"]
            )
            self.assertEqual(
                lane["forecast"]["range_trade_economic_basis"]["basis"],
                "EXACT_TP_PROVEN_HARVEST",
            )
            _write_overlay(paths)
            self.assertEqual(_apply(paths).action, "TRADE")
            final = json.loads(paths["output"].read_text())
            self.assertEqual(
                final["decision_provenance"][
                    "capital_allocation_edge_basis"
                ],
                "EXACT_VEHICLE_TAKE_PROFIT",
            )

    def test_quarter_kelly_nav_ratio_can_cap_lane_at_half_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot["account"]["nav_jpy"] = 4_000.0
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(lane["allowed_size_multiples"], [0.5])
            self.assertEqual(ceiling["kelly"]["decision_basis"], "NAV_PERCENT_RATIO")
            self.assertGreater(ceiling["max_multiple"], 0.5)
            self.assertLess(ceiling["max_multiple"], 0.75)
            self.assertGreater(
                ceiling["kelly"]["base_risk_nav_pct"],
                ceiling["kelly"]["quarter_kelly_risk_nav_pct"],
            )

            _write_overlay(paths, size_multiple=0.75)
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
            ):
                _apply(paths)
            _write_overlay(paths, size_multiple=0.5)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_quarter_kelly_cap_below_half_advertises_no_trade_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot["account"]["nav_jpy"] = 2_000.0
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertGreater(ceiling["max_multiple"], 0.0)
            self.assertLess(ceiling["max_multiple"], 0.5)
            self.assertEqual(
                ceiling["reason"],
                "QUARTER_KELLY_CAP_BELOW_MINIMUM_MULTIPLE",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertFalse(lane["allocation_eligible"])

    def test_strong_numeric_ceiling_allows_full_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["schema_version"],
                2,
            )
            self.assertEqual(lane["allowed_size_multiples"], [0.5, 0.75, 1.0])
            self.assertEqual(ceiling["max_multiple"], 1.0)
            self.assertTrue(ceiling["ev_lower"]["positive"])
            self.assertGreater(
                ceiling["kelly"]["quarter_kelly_risk_nav_pct"],
                ceiling["kelly"]["base_risk_nav_pct"],
            )
            _write_overlay(paths, size_multiple=1.0)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_hedge_keeps_prebounded_size_but_scout_cannot_bypass_method_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata["position_intent"] = "HEDGE"
            for key in (
                "forecast_directional_economic_hit_rate",
                "forecast_directional_economic_samples",
                "forecast_target_price",
                "forecast_invalidation_price",
            ):
                metadata.pop(key, None)
            paths["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot.pop("account")
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertEqual(lane["allowed_size_multiples"], [1.0])
            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT",
            )

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intent = intents["results"][0]["intent"]
            metadata = intent["metadata"]
            rule_name = (
                "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_S5_"
                "BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
            )
            rule = canonical_bidask_replay_precision_rule(rule_name)
            self.assertIsNotNone(rule)
            assert rule is not None
            intent["order_type"] = "LIMIT"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
                    "predictive_scout_hypothesis": (
                        "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN"
                    ),
                    "predictive_scout_vehicle_proof_status": (
                        "UNPROVEN_PASSIVE_LIMIT"
                    ),
                    "predictive_scout_rule_is_vehicle_proof": False,
                    "predictive_scout_rule_digest": (
                        bidask_replay_precision_rule_digest(rule)
                    ),
                    "bidask_replay_precision_seed_rule": rule,
                    "desk": "failure_trader",
                    "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
                    "attach_take_profit_on_fill": True,
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                }
            )
            for key in (
                "forecast_directional_economic_hit_rate",
                "forecast_directional_economic_samples",
                "forecast_target_price",
                "forecast_invalidation_price",
            ):
                metadata.pop(key, None)
            paths["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot.pop("account")
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertTrue(lane["predictive_scout"])
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertFalse(lane["allocation_eligible"])
            self.assertIn(
                "REGIME_FAMILY_WEIGHTING_METHOD_MISMATCH",
                lane["live_blocker_codes"],
            )
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT",
            )

    def test_hedge_and_predictive_scout_cannot_bypass_invalid_forecast_binding(self) -> None:
        for mode in ("HEDGE", "PREDICTIVE_SCOUT"):
            for failure in ("DIRECTION", "CALIBRATION"):
                with self.subTest(mode=mode, failure=failure), tempfile.TemporaryDirectory() as tmp:
                    paths = _prepared_paths(Path(tmp))
                    intents = json.loads(paths["intents"].read_text())
                    intent = intents["results"][0]["intent"]
                    metadata = intent["metadata"]
                    if mode == "HEDGE":
                        metadata["position_intent"] = "HEDGE"
                    else:
                        rule_name = (
                            "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_S5_"
                            "BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
                        )
                        rule = canonical_bidask_replay_precision_rule(rule_name)
                        self.assertIsNotNone(rule)
                        assert rule is not None
                        intent["order_type"] = "LIMIT"
                        intent["market_context"]["method"] = "BREAKOUT_FAILURE"
                        metadata.update(
                            {
                                "predictive_scout": True,
                                "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
                                "predictive_scout_hypothesis": (
                                    "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN"
                                ),
                                "predictive_scout_vehicle_proof_status": (
                                    "UNPROVEN_PASSIVE_LIMIT"
                                ),
                                "predictive_scout_rule_is_vehicle_proof": False,
                                "predictive_scout_rule_digest": (
                                    bidask_replay_precision_rule_digest(rule)
                                ),
                                "bidask_replay_precision_seed_rule": rule,
                                "desk": "failure_trader",
                                "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
                                "attach_take_profit_on_fill": True,
                                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            }
                        )

                    # The context envelope is individually valid, but the live
                    # forecast binding either says DOWN for a LONG lane or names
                    # the DOWN calibration for an UP forecast. Previously both
                    # pre-bounded paths returned max_multiple=1 before these checks.
                    metadata["forecast_direction"] = (
                        "DOWN" if failure == "DIRECTION" else "UP"
                    )
                    metadata["forecast_directional_calibration_name"] = (
                        "directional_forecast_down"
                    )
                    for key in (
                        "forecast_directional_economic_hit_rate",
                        "forecast_directional_economic_samples",
                        "forecast_target_price",
                        "forecast_invalidation_price",
                    ):
                        metadata.pop(key, None)
                    paths["intents"].write_text(json.dumps(intents))
                    snapshot = json.loads(paths["snapshot"].read_text())
                    snapshot.pop("account")
                    paths["snapshot"].write_text(json.dumps(snapshot))
                    _reprepare(paths)

                    board = json.loads(paths["packet"].read_text())[
                        "capital_allocation_board"
                    ]
                    lane = board["selected_lane"]
                    self.assertEqual(
                        lane["forecast"]["technical_context"]["status"],
                        "VALID",
                    )
                    self.assertEqual(
                        board["forecast_context"]["technical_context"]["status"],
                        "UNKNOWN",
                    )
                    self.assertEqual(
                        board["forecast_context"]["technical_context"]["reason"],
                        (
                            "FORECAST_TECHNICAL_CONTEXT_DIRECTION_MISMATCH"
                            if failure == "DIRECTION"
                            else "FORECAST_TECHNICAL_CONTEXT_CALIBRATION_MISMATCH"
                        ),
                    )
                    self.assertEqual(
                        lane["numeric_ceiling"]["geometry"]["direction_side_aligned"],
                        failure != "DIRECTION",
                    )
                    self.assertEqual(
                        lane["numeric_ceiling"]["reason"],
                        (
                            "FORECAST_DIRECTION_SIDE_MISMATCH"
                            if failure == "DIRECTION"
                            else "FORECAST_CALIBRATION_IDENTITY_MISMATCH"
                        ),
                    )
                    self.assertEqual(lane["allowed_size_multiples"], [])
                    self.assertFalse(lane["positive_edge_proven"])
                    self.assertFalse(lane["allocation_eligible"])

                    _write_overlay(paths, size_multiple=1.0)
                    with self.assertRaisesRegex(
                        MarketReadOverlayError,
                        "MARKET_READ_CAPITAL_ALLOCATION_FORECAST_CONTEXT_INVALID",
                    ):
                        _apply(paths)

    def test_trade_allocation_rejects_expansion_units_and_stale_board(self) -> None:
        for mutation, expected_code in (
            ("expansion", "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID"),
            ("units", "MARKET_READ_CAPITAL_ALLOCATION_UNITS_MISMATCH"),
            ("board", "MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                overlay = _overlay(paths)
                allocation = overlay["capital_allocation"]
                if mutation == "expansion":
                    allocation["size_multiple"] = 1.25
                    allocation["selected_units"] = 1500
                elif mutation == "units":
                    allocation["selected_units"] += 1
                else:
                    allocation["allocation_board_sha256"] = "0" * 64
                paths["overlay"].write_text(json.dumps(overlay))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_nontrade_allocation_cannot_smuggle_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths, disposition="VETO_WAIT")
            overlay["capital_allocation"].update(
                {
                    "decision": "ALLOCATE",
                    "lane_id": LANE_ID,
                    "size_multiple": 0.5,
                    "selected_units": 600,
                }
            )
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_NONTRADE_REQUIRED",
            ):
                _apply(paths)

    def test_explicit_negative_lane_edge_requires_zero_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata["capture_take_profit_expectancy_jpy"] = -25.0
            metadata["capture_take_profit_wins"] = 2
            metadata["capture_take_profit_losses"] = 2
            paths["intents"].write_text(json.dumps(intents))
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=_sources(paths),
                now=NOW,
            )
            _write_overlay(paths)

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_EDGE_NOT_PROVEN",
            ):
                _apply(paths)

    def test_accept_preserves_the_deterministic_execution_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            baseline = json.loads(paths["baseline"].read_text())
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(final["action"], baseline["action"])
            self.assertEqual(final["selected_lane_id"], baseline["selected_lane_id"])
            self.assertEqual(final["selected_lane_ids"], baseline["selected_lane_ids"])
            self.assertEqual(final["cancel_order_ids"], baseline["cancel_order_ids"])
            self.assertEqual(final["risk_notes"], baseline["risk_notes"])
            provenance = final["decision_provenance"]
            self.assertEqual(provenance["author_kind"], CODEX_MARKET_READ_AUTHOR)
            self.assertEqual(provenance["baseline_action"], "TRADE")
            self.assertEqual(provenance["final_action"], "TRADE")
            self.assertFalse(provenance["action_downgrade_only"])
            self.assertTrue(provenance["execution_fields_preserved"])
            self.assertTrue(provenance["risk_envelope_not_expanded"])
            self.assertFalse(provenance["live_permission_granted"])

    def test_veto_can_only_downgrade_trade_and_clears_selected_lanes(self) -> None:
        for disposition, expected_action in (
            ("VETO_WAIT", "WAIT"),
            ("VETO_REQUEST_EVIDENCE", "REQUEST_EVIDENCE"),
        ):
            with self.subTest(disposition=disposition), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                baseline = json.loads(paths["baseline"].read_text())
                _write_overlay(paths, disposition=disposition)

                _apply(paths)

                final = json.loads(paths["output"].read_text())
                self.assertEqual(final["action"], expected_action)
                self.assertIsNone(final["selected_lane_id"])
                self.assertEqual(final["selected_lane_ids"], [])
                self.assertEqual(final["market_read_vetoed_lane_ids"], [LANE_ID])
                self.assertEqual(final["cancel_order_ids"], baseline["cancel_order_ids"])
                self.assertEqual(final["risk_notes"], baseline["risk_notes"])
                provenance = final["decision_provenance"]
                self.assertEqual(provenance["baseline_action"], "TRADE")
                self.assertEqual(provenance["final_action"], expected_action)
                self.assertTrue(provenance["action_downgrade_only"])

    def test_accept_rejects_crafted_multi_lane_trade_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
            paths = _prepared_paths(
                Path(tmp),
                baseline=_baseline(lane_ids=[LANE_ID, second_lane]),
            )
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_BASELINE_SINGLE_LANE_REQUIRED",
            ):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_multi_lane_trade_baseline_can_still_publish_nontrade_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
            paths = _prepared_paths(
                Path(tmp),
                baseline=_baseline(lane_ids=[LANE_ID, second_lane]),
            )
            _write_overlay(paths, disposition="VETO_WAIT")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "WAIT")
            self.assertEqual(
                final["market_read_vetoed_lane_ids"],
                [LANE_ID, second_lane],
            )

    def test_nontrade_baseline_cannot_be_changed_to_any_other_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="WAIT", lane_ids=[])
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="VETO_WAIT")

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_NONTRADE_UPGRADE_FORBIDDEN"):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_accept_close_baseline_preserves_exact_close_trade_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "CLOSE")
            self.assertEqual(final["action"], "CLOSE")
            self.assertEqual(final["close_trade_ids"], ["555"])
            self.assertEqual(final["selected_lane_ids"], [])
            self.assertEqual(final["cancel_order_ids"], [])
            self.assertEqual(
                final["capital_allocation"],
                {
                    "decision": "NO_TRADE",
                    "lane_id": None,
                    "size_multiple": 0.0,
                    "selected_units": 0,
                    "allocation_board_sha256": final["capital_allocation"][
                        "allocation_board_sha256"
                    ],
                    "rationale": "No fresh entry capital is authorized for this receipt.",
                },
            )
            self.assertEqual(
                final["decision_provenance"]["baseline_action"],
                "CLOSE",
            )
            self.assertTrue(
                final["decision_provenance"]["execution_fields_preserved"]
            )

    def test_close_baseline_rejects_allocate_capital_smuggling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            overlay = _overlay(paths, disposition="ACCEPT_BASELINE")
            overlay["capital_allocation"].update(
                {
                    "decision": "ALLOCATE",
                    "lane_id": LANE_ID,
                    "size_multiple": 1.0,
                    "selected_units": 1,
                }
            )
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_NONTRADE_REQUIRED",
            ):
                _apply(paths)

    def test_accept_rejects_multi_target_close_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555", "556"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_BASELINE_SINGLE_CLOSE_REQUIRED",
            ):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_accept_rejects_close_baseline_with_entry_or_cancel_scope(self) -> None:
        for field, value in (
            ("selected_lane_id", LANE_ID),
            ("selected_lane_ids", [LANE_ID]),
            ("cancel_order_ids", ["123"]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = _baseline(action="CLOSE", lane_ids=[])
                baseline["cancel_order_ids"] = []
                baseline["close_trade_ids"] = ["555"]
                baseline["method"] = "POSITION_MANAGEMENT"
                baseline[field] = value
                paths = _prepared_paths(root, baseline=baseline)
                _write_overlay(paths, disposition="ACCEPT_BASELINE")

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_BASELINE_CLOSE_SCOPE_INVALID",
                ):
                    _apply(paths)

                self.assertFalse(paths["output"].exists())

    def test_overlay_rejects_any_execution_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            overlay["action"] = "TRADE"
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_OVERLAY_SCHEMA_INVALID"):
                _apply(paths)

    def test_overlay_rejects_missing_canonical_market_read_field_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            naked = overlay["market_read_first"]["naked_read"]
            naked["known_winning_setup_state"] = naked.pop(
                "known_winning_trade_shape_match"
            )
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_OVERLAY_SCHEMA_INVALID.*known_winning_trade_shape_match",
            ):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_overlay_rejects_noncanonical_market_read_enum_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            overlay["market_read_first"]["naked_read"]["tape_state"] = "TREND_DOWN"
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_OVERLAY_SCHEMA_INVALID.*naked_read.tape_state",
            ):
                _apply(paths)

    def test_overlay_rejects_unknown_market_read_alias_even_with_canonical_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            overlay["market_read_first"]["naked_read"][
                "known_winning_setup_state"
            ] = "alias must not be accepted"
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_OVERLAY_SCHEMA_INVALID.*known_winning_setup_state",
            ):
                _apply(paths)

    def test_evidence_packet_exposes_exact_market_read_first_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            schema = json.loads(paths["packet"].read_text())["contract"][
                "market_read_first"
            ]

            self.assertIn(
                "known_winning_trade_shape_match",
                schema["required_fields"]["naked_read"],
            )
            self.assertEqual(
                schema["enums"]["naked_read.tape_state"],
                ["FADE", "RANGE", "ROTATION", "SQUEEZE", "TREND"],
            )

    def test_evidence_packet_embeds_bounded_numeric_forecast_replay_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(root)
            replay = root / "oanda_history_replay_validate_latest.json"
            replay.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-07-10T00:00:00Z",
                        "truth_source": "BID_ASK_S5",
                        "granularity": "S5",
                        "pair_filter": ["EUR_USD", "GBP_JPY"],
                        "history_pairs": 2,
                        "evaluated_rows": 9420,
                        "summary": {
                            "n": 9420,
                            "hit_rate": 0.3297,
                            "avg_final_pips": -2.8666,
                            "median_final_pips": -2.3,
                            "avg_mfe_pips": 5.1498,
                            "avg_mae_pips": 10.6671,
                            "target_touch_rate": 0.3021,
                            "invalidation_touch_rate": 0.5825,
                            "target_before_invalidation_rate": 0.2322,
                        },
                        "segments": {
                            "by_pair_direction": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "UP",
                                    "n": 120,
                                    "hit_rate": 0.45,
                                    "avg_final_pips": 0.6,
                                },
                                {
                                    "pair": "GBP_JPY",
                                    "direction": "DOWN",
                                    "n": 80,
                                    "hit_rate": 0.38,
                                    "avg_final_pips": -2.8,
                                },
                            ],
                            "by_confidence": [
                                {
                                    "confidence_bucket": ">=0.90",
                                    "n": 188,
                                    "hit_rate": 0.367,
                                    "avg_final_pips": -0.09,
                                }
                            ],
                            "by_horizon": [
                                {
                                    "horizon_bucket": "31-60m",
                                    "n": 7883,
                                    "hit_rate": 0.3166,
                                    "avg_final_pips": -2.9177,
                                }
                            ],
                            "by_primary_driver_family": [
                                {
                                    "primary_driver_family": "RANGE_BREAKOUT_CONFIRMED",
                                    "n": 51,
                                    "hit_rate": 0.4902,
                                    "avg_final_pips": 0.6039,
                                    "avg_realized_r": -0.0788,
                                }
                            ],
                            "by_driver_family_presence": [
                                {
                                    "driver_family": "MARKET_LOCATION",
                                    "n": 171,
                                    "hit_rate": 0.2982,
                                    "avg_final_pips": -4.4877,
                                    "avg_realized_r": -0.1613,
                                }
                            ],
                            "by_primary_driver_family_direction": [
                                {
                                    "primary_driver_family": "RANGE_BREAKOUT_CONFIRMED",
                                    "direction": "DOWN",
                                    "n": 20,
                                    "hit_rate": 0.55,
                                    "avg_final_pips": 6.32,
                                    "avg_realized_r": -0.0406,
                                }
                            ],
                            "by_raw_confidence": [
                                {
                                    "raw_confidence_bucket": ">=0.90",
                                    "n": 47,
                                    "hit_rate": 0.34,
                                    "avg_final_pips": -3.326,
                                }
                            ],
                            "by_session": [
                                {
                                    "utc_session_bucket": "UTC_17_22",
                                    "n": 68,
                                    "hit_rate": 0.25,
                                    "avg_final_pips": -4.376,
                                }
                            ],
                            "by_technical_regime": [
                                {
                                    "technical_regime": "TREND_WEAK",
                                    "n": 30,
                                    "hit_rate": 0.6,
                                    "avg_final_pips": 1.2,
                                }
                            ],
                            "by_technical_context_completeness": [
                                {
                                    "technical_context_complete": False,
                                    "n": 30,
                                    "hit_rate": 0.6,
                                    "avg_final_pips": 1.2,
                                }
                            ],
                            "by_technical_structure_alignment": [
                                {
                                    "technical_structure_alignment": "ALIGNED",
                                    "n": 24,
                                    "hit_rate": 0.625,
                                    "avg_final_pips": 1.5,
                                }
                            ],
                        },
                        "train_validation_exit_selection": {
                            "status": "OK",
                            "train_n": 354,
                            "validation_n": 240,
                            "validation_start_utc": "2026-07-07T11:12:49Z",
                            "selected_by_train": {
                                "n": 354,
                                "take_profit_pips": 10.0,
                                "stop_loss_pips": 2.0,
                                "avg_realized_pips": -1.73,
                                "profit_factor": 0.095,
                                "win_rate": 0.034,
                                "tp_rate": 0.011,
                                "sl_rate": 0.952,
                                "timeout_rate": 0.037,
                            },
                            "validation": {
                                "n": 240,
                                "take_profit_pips": 10.0,
                                "stop_loss_pips": 2.0,
                                "avg_realized_pips": -1.8,
                                "profit_factor": 0.074,
                                "win_rate": 0.025,
                                "tp_rate": 0.008,
                                "sl_rate": 0.971,
                                "timeout_rate": 0.021,
                            },
                        },
                    }
                )
            )
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            sources = _sources(paths)
            sources["bidask_replay_validation"] = replay
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=sources,
                now=NOW,
            )

            scorecard = json.loads(paths["packet"].read_text())[
                "forecast_replay_scorecard"
            ]
            self.assertEqual(scorecard["status"], "VALID")
            self.assertEqual(scorecard["contract"], "QR_FORECAST_REPLAY_SCORECARD_V3")
            self.assertEqual(scorecard["global"]["n"], 9420)
            self.assertEqual(scorecard["global"]["avg_final_pips"], -2.8666)
            self.assertEqual(scorecard["selected_pair"], "EUR_USD")
            self.assertEqual(scorecard["selected_direction"], "UP")
            self.assertEqual(scorecard["selected_coverage_status"], "COVERED")
            self.assertEqual(
                scorecard["selected_pair_direction"]["avg_final_pips"],
                0.6,
            )
            self.assertGreater(
                scorecard["selected_pair_direction"]["hit_wilson95_lower"],
                0.36,
            )
            self.assertEqual(scorecard["proof_status"], "UNVERIFIED_LEGACY")
            self.assertFalse(scorecard["proof_eligible"])
            self.assertEqual(scorecard["scope"]["pair_filter"], ["EUR_USD", "GBP_JPY"])
            self.assertFalse(scorecard["scope"]["pair_direction_rows_truncated"])
            self.assertEqual(
                scorecard["scope"]["confidence_segment_rows_accounted"],
                188,
            )
            self.assertEqual(
                scorecard["scope"]["confidence_segment_rows_unreported"],
                9232,
            )
            self.assertFalse(scorecard["scope"]["confidence_segment_complete"])
            self.assertEqual(
                scorecard["by_primary_driver_family"][0]["primary_driver_family"],
                "RANGE_BREAKOUT_CONFIRMED",
            )
            self.assertEqual(
                scorecard["by_primary_driver_family"][0]["avg_final_pips"],
                0.6039,
            )
            self.assertEqual(
                scorecard["by_primary_driver_family"][0]["avg_realized_r"],
                -0.0788,
            )
            self.assertEqual(
                scorecard["by_driver_family_presence"][0]["driver_family"],
                "MARKET_LOCATION",
            )
            self.assertEqual(
                scorecard["by_primary_driver_family_direction"][0]["direction"],
                "DOWN",
            )
            self.assertEqual(
                scorecard["by_primary_driver_family_direction"][0]["avg_realized_r"],
                -0.0406,
            )
            self.assertEqual(
                scorecard["by_raw_confidence"][0]["raw_confidence_bucket"],
                ">=0.90",
            )
            self.assertEqual(
                scorecard["by_session"][0]["utc_session_bucket"],
                "UTC_17_22",
            )
            self.assertEqual(
                scorecard["by_technical_regime"][0]["technical_regime"],
                "TREND_WEAK",
            )
            self.assertIs(
                scorecard["by_technical_context_completeness"][0][
                    "technical_context_complete"
                ],
                False,
            )
            self.assertEqual(
                scorecard["by_technical_structure_alignment"][0]["technical_structure_alignment"],
                "ALIGNED",
            )
            self.assertEqual(
                scorecard["exit_policy_validation"]["validation"]["profit_factor"],
                0.074,
            )
            self.assertEqual(
                scorecard["exit_policy_validation"]["validation"]["sl_rate"],
                0.971,
            )
            self.assertTrue(scorecard["read_only"])
            self.assertFalse(scorecard["live_permission"])

    def test_forecast_replay_change_stales_market_read_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(root)
            replay = root / "oanda_history_replay_validate_latest.json"
            replay.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-07-10T00:00:00Z",
                        "summary": {"n": 100, "avg_final_pips": -1.0},
                        "segments": {"by_pair_direction": []},
                    }
                )
            )
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            sources = _sources(paths)
            sources["bidask_replay_validation"] = replay
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=sources,
                now=NOW,
            )
            _write_overlay(paths)
            replay.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-07-10T00:01:00Z",
                        "summary": {"n": 101, "avg_final_pips": -2.0},
                        "segments": {"by_pair_direction": []},
                    }
                )
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=sources,
                    now=NOW,
                )

    def test_missing_or_malformed_forecast_replay_never_fabricates_zero_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(root)
            missing_scorecard = json.loads(paths["packet"].read_text())[
                "forecast_replay_scorecard"
            ]
            self.assertEqual(missing_scorecard["status"], "MISSING")
            self.assertNotIn("global", missing_scorecard)

            malformed = root / "malformed_replay.json"
            malformed.write_text("{not-json")
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            sources = _sources(paths)
            sources["bidask_replay_validation"] = malformed
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=sources,
                now=NOW,
            )
            malformed_scorecard = json.loads(paths["packet"].read_text())[
                "forecast_replay_scorecard"
            ]
            self.assertEqual(malformed_scorecard["status"], "MALFORMED")
            self.assertNotIn("global", malformed_scorecard)

    def test_baseline_or_evidence_mutation_rejects_stale_overlay(self) -> None:
        for mutation, expected_code in (
            ("baseline", "MARKET_READ_BASELINE_SHA_STALE"),
            ("evidence", "MARKET_READ_EVIDENCE_PACKET_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                if mutation == "baseline":
                    baseline = json.loads(paths["baseline"].read_text())
                    baseline["risk_notes"] = ["mutated after AI review"]
                    paths["baseline"].write_text(json.dumps(baseline))
                else:
                    snapshot = json.loads(paths["snapshot"].read_text())
                    snapshot["quotes"]["EUR_USD"]["ask"] = 1.1010
                    paths["snapshot"].write_text(json.dumps(snapshot))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_equivalent_relative_and_absolute_source_paths_share_one_evidence_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            relative_snapshot = Path(os.path.relpath(paths["snapshot"], Path.cwd()))
            relative_sources = {**_sources(paths), "broker_snapshot": relative_snapshot}
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=relative_sources,
                now=NOW,
            )
            _write_overlay(paths)

            summary = _apply(paths)

            packet = json.loads(paths["packet"].read_text())
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(
                packet["source_paths"]["broker_snapshot"],
                str(paths["snapshot"].resolve()),
            )

    def test_identical_bytes_at_a_different_source_path_do_not_share_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            copied_snapshot = paths["snapshot"].with_name("copied_snapshot.json")
            copied_snapshot.write_bytes(paths["snapshot"].read_bytes())
            relocated_sources = {**_sources(paths), "broker_snapshot": copied_snapshot}

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=relocated_sources,
                    now=NOW,
                )

    def test_watchdog_observation_clock_rewrite_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog["generated_at_utc"] = (NOW + timedelta(minutes=1)).isoformat()
            watchdog["minutes_since_last_run"] = 61.0
            watchdog["last_trader_run_at"] = NOW.isoformat()
            watchdog["last_trader_run_source"] = "decision_response.generated_at_utc"
            watchdog["last_decision_artifact_at"] = NOW.isoformat()
            watchdog["guardian_receipt"].update(
                {
                    "action": "NO_ACTION",
                    "expired_before_trader_run": True,
                    "next_run_window_missed": False,
                    "receipt_after_last_trader_run": True,
                    "receipt_lifecycle": "EXPIRED",
                    "receipt_status": "ACCEPTED",
                    "terminal_lifecycle": True,
                    "will_expire_before_next_run": False,
                    "receipt_summaries": [
                        {
                            "action": "REDUCE",
                            "active": False,
                            "canonical_present": False,
                            "emergency_or_margin_risk": False,
                            "event_id": "historical-event",
                            "high_urgency_action": True,
                            "identity": "event|historical-event|REDUCE",
                            "receipt_lifecycle": "SUPERSEDED",
                        }
                    ],
                }
            )
            watchdog["weekend_pause"]["now_jst"] = "2026-07-11T12:01:00+09:00"
            watchdog["automation_config"]["weekend_pause"]["now_jst"] = (
                "2026-07-11T12:01:00+09:00"
            )
            watchdog["codex_logs"]["queried_at_utc"] = (
                NOW + timedelta(minutes=1)
            ).isoformat()
            watchdog["guardian_receipt"]["review_excerpt"] = (
                "Generated at a later observation clock"
            )
            paths["watchdog"].write_text(json.dumps(watchdog))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_watchdog_stale_age_message_rewrite_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog.update(
                {
                    "status": "STALE",
                    "runtime_status": "STALE",
                    "issue_status": "P0",
                    "overall_status": "BLOCKED",
                    "severity": "P0",
                    "missed_expected_window": True,
                    "issues": [
                        {
                            "code": "QR_TRADER_RUN_STALE",
                            "message": (
                                "Latest trader run evidence is 1861.4 minutes old; "
                                "expected <= 75 minutes."
                            ),
                            "severity": "P0",
                        }
                    ],
                }
            )
            paths["watchdog"].write_text(json.dumps(watchdog))
            _reprepare(paths)
            _write_overlay(paths)

            watchdog["generated_at_utc"] = (NOW + timedelta(minutes=5)).isoformat()
            watchdog["minutes_since_last_run"] = 1866.4
            watchdog["issues"][0]["message"] = (
                "Latest trader run evidence is 1866.4 minutes old; "
                "expected <= 75 minutes."
            )
            paths["watchdog"].write_text(json.dumps(watchdog))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_watchdog_issue_code_or_severity_change_stales_ai_review(self) -> None:
        for mutation in ("code", "severity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                watchdog = json.loads(paths["watchdog"].read_text())
                watchdog["issues"] = [
                    {
                        "code": "QR_TRADER_RUN_STALE",
                        "message": "Rendered diagnostic prose",
                        "severity": "P0",
                    }
                ]
                paths["watchdog"].write_text(json.dumps(watchdog))
                _reprepare(paths)
                _write_overlay(paths)

                watchdog["issues"][0][mutation] = (
                    "QR_TRADER_RUN_EVIDENCE_MISSING" if mutation == "code" else "P1"
                )
                paths["watchdog"].write_text(json.dumps(watchdog))

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_EVIDENCE_PACKET_STALE",
                ):
                    _apply(paths)

    def test_unknown_watchdog_issue_message_change_stales_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog["issues"] = [
                {
                    "code": "FUTURE_SAFETY_ISSUE",
                    "message": "first safety meaning",
                    "severity": "P0",
                }
            ]
            paths["watchdog"].write_text(json.dumps(watchdog))
            _reprepare(paths)
            _write_overlay(paths)
            watchdog["issues"][0]["message"] = "changed safety meaning"
            paths["watchdog"].write_text(json.dumps(watchdog))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

    def test_watchdog_material_health_change_stales_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog["status"] = "BROKEN"
            watchdog["runtime_status"] = "BROKEN"
            watchdog["issues"] = ["MISSED_TRADER_RUN"]
            paths["watchdog"].write_text(json.dumps(watchdog))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

    def test_guardian_action_receipt_named_source_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            sources = _sources(paths)
            sources.pop("guardian_action_receipt")
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_SOURCE_CONTRACT_INVALID",
            ):
                prepare_market_read_baseline(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    evidence_sources=sources,
                    now=NOW,
                )

            _reprepare(paths)
            _write_overlay(paths)
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_SOURCE_CONTRACT_INVALID",
            ):
                apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=sources,
                    now=NOW,
                )

    def test_watchdog_receipt_identity_catchup_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog["guardian_receipt"].update(
                {
                    "active": True,
                    "dependency_before_next_run": True,
                    "exists": True,
                    "action": "REDUCE",
                    "receipt_lifecycle": "ACTIVE",
                    "receipt_status": "ACCEPTED",
                }
            )
            watchdog["guardian_receipt"]["receipt_summaries"] = [
                {
                    "action": "REDUCE",
                    "active": True,
                    "canonical_present": True,
                    "event_id": "new-event",
                    "identity": "event|new-event|REDUCE",
                    "high_urgency_action": True,
                    "receipt_lifecycle": "ACTIVE",
                    "receipt_status": "ACCEPTED",
                }
            ]
            paths["watchdog"].write_text(json.dumps(watchdog))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_unrelated_routine_guardian_receipt_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            paths["guardian_action_receipt"].write_text(
                json.dumps(
                    _guardian_action_receipt(
                        pair="EUR_CAD",
                        action="HOLD",
                        event_type="SPREAD_ANOMALY",
                        severity="P1",
                        event_id="other-pair-routine",
                    )
                )
            )

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_guardian_scope_requires_selected_event_object(self) -> None:
        for replacement in (None, "scalar"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "guardian_action_receipt.json"
                receipt = _guardian_action_receipt(pair="EUR_USD")
                if replacement is None:
                    receipt.pop("selected_event")
                    expected_reason = "SELECTED_EVENT_MISSING"
                else:
                    receipt["selected_event"] = replacement
                    expected_reason = "SELECTED_EVENT_INVALID"
                path.write_text(json.dumps(receipt))

                material = (
                    market_read_overlay_module.guardian_action_receipt_scope_material(
                        path,
                        baseline_pairs=["EUR_USD"],
                        as_of=NOW,
                    )
                )

                self.assertEqual(material["parse_status"], "VALID")
                self.assertTrue(material["global_safety"])
                self.assertIn(expected_reason, material["global_reasons"])

    def test_exact_p1_margin_receipt_is_global_observation_but_every_contradiction_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_action_receipt.json"
            canonical_receipt = _p1_margin_action_receipt()
            path.write_text(json.dumps(canonical_receipt))

            canonical = (
                market_read_overlay_module.guardian_action_receipt_scope_material(
                    path,
                    baseline_pairs=["EUR_USD"],
                    as_of=NOW,
                )
            )

            self.assertEqual(canonical["parse_status"], "VALID")
            self.assertFalse(canonical["global_safety"])
            self.assertEqual(canonical["scope"], "GLOBAL_MARGIN_OBSERVATION")
            self.assertTrue(canonical["p1_margin_warning_observed"])
            self.assertEqual(
                canonical["margin_contract"],
                "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
            )

            variants: dict[str, dict] = {}
            missing_severity = json.loads(json.dumps(canonical_receipt))
            missing_severity["selected_event"].pop("severity")
            missing_severity["event"].pop("severity")
            variants["missing_severity"] = missing_severity
            contradictory = json.loads(json.dumps(canonical_receipt))
            for key in ("selected_event", "event"):
                contradictory[key]["details"][
                    "fresh_entry_risk_block_active"
                ] = True
            variants["contradictory_flags"] = contradictory
            hard_cap = json.loads(json.dumps(canonical_receipt))
            for key in ("selected_event", "event"):
                hard_cap[key]["details"]["margin_used_jpy"] = 95_000.0
            variants["hard_cap_reached"] = hard_cap
            legacy_cap = json.loads(json.dumps(canonical_receipt))
            for key in ("selected_event", "event"):
                legacy_cap[key]["details"].update(
                    {
                        "margin_used_jpy": 86_000.0,
                        "margin_available_jpy": 14_000.0,
                        "max_margin_utilization_pct": 92.0,
                    }
                )
            variants["legacy_cap_mismatch"] = legacy_cap
            compound = json.loads(json.dumps(canonical_receipt))
            compound["issues"] = [
                {"code": "UNRELATED_P1", "severity": "P1"}
            ]
            variants["compound_issue"] = compound
            p0 = json.loads(json.dumps(canonical_receipt))
            for key in ("selected_event", "event"):
                p0[key]["severity"] = "P0"
                p0[key]["thesis_state"] = "EMERGENCY"
                p0[key]["details"].update(
                    {
                        "margin_used_jpy": 95_000.0,
                        "fresh_entry_risk_block_active": True,
                        "fresh_entry_risk_observation_only": False,
                        "fresh_entry_margin_contract": (
                            "QR_GUARDIAN_P0_MARGIN_HARD_CAP_V1"
                        ),
                    }
                )
            p0["receipt"]["thesis_state"] = "EMERGENCY"
            variants["p0"] = p0

            for name, receipt in variants.items():
                with self.subTest(name=name):
                    path.write_text(json.dumps(receipt))
                    material = market_read_overlay_module.guardian_action_receipt_scope_material(
                        path,
                        baseline_pairs=["EUR_USD"],
                        as_of=NOW,
                    )
                    self.assertTrue(material["global_safety"])
                    self.assertFalse(
                        material.get("p1_margin_warning_observed", False)
                    )

    def test_guardian_scope_enforces_runtime_clock_states_without_hashing_clocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_action_receipt.json"
            receipt = _guardian_action_receipt(pair="EUR_USD")
            path.write_text(json.dumps(receipt))
            original = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            receipt["generated_at_utc"] = (NOW + timedelta(seconds=5)).isoformat()
            receipt["expires_at_utc"] = (NOW + timedelta(hours=2)).isoformat()
            path.write_text(json.dumps(receipt))
            clock_only = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            self.assertEqual(canonical_json_sha256(original), canonical_json_sha256(clock_only))

            receipt["generated_at_utc"] = (NOW + timedelta(seconds=6)).isoformat()
            path.write_text(json.dumps(receipt))
            future = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            self.assertTrue(future["global_safety"])
            self.assertIn("GENERATED_AT_FUTURE", future["global_reasons"])

            receipt["generated_at_utc"] = (NOW - timedelta(hours=2)).isoformat()
            receipt["expires_at_utc"] = (NOW - timedelta(hours=1)).isoformat()
            receipt["receipt_lifecycle"] = "CONSUMED"
            receipt["consumed_by_trader"] = True
            path.write_text(json.dumps(receipt))
            consumed = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            self.assertFalse(consumed["global_safety"])
            self.assertEqual(
                consumed["time_state"]["expires_at_utc"],
                "NOT_APPLICABLE_CONSUMED",
            )

    def test_production_packet_uses_guardian_read_clock_after_mid_build_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))

            original_projection = (
                market_read_overlay_module.projection_calibration_evidence
            )
            rotated = False

            def rotate_receipt_at_t_plus_10(*args, **kwargs):
                nonlocal rotated
                if not rotated:
                    receipt = _guardian_action_receipt(
                        pair="EUR_USD",
                        action="NO_ACTION",
                        event_id="mid-build-clock-rotation",
                    )
                    receipt["generated_at_utc"] = (
                        NOW + timedelta(seconds=10)
                    ).isoformat()
                    receipt["expires_at_utc"] = (
                        NOW + timedelta(hours=1)
                    ).isoformat()
                    paths["guardian_action_receipt"].write_text(
                        json.dumps(receipt)
                    )
                    rotated = True
                return original_projection(*args, **kwargs)

            original_utc_now = market_read_overlay_module._utc_now
            runtime_instants = iter(
                (
                    NOW,
                    NOW + timedelta(seconds=10),
                    NOW + timedelta(seconds=20),
                    NOW + timedelta(seconds=30),
                )
            )

            def observed_clock(value):
                if value is not None:
                    return original_utc_now(value)
                return next(runtime_instants)

            with patch.object(
                market_read_overlay_module,
                "_utc_now",
                side_effect=observed_clock,
            ), patch.object(
                market_read_overlay_module,
                "projection_calibration_evidence",
                side_effect=rotate_receipt_at_t_plus_10,
            ):
                prepared = prepare_market_read_baseline(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    evidence_sources=_sources(paths),
                    now=None,
                )
                packet = json.loads(paths["packet"].read_text())
                guardian = packet["guardian_action_receipt_material"]
                self.assertEqual(
                    guardian["time_state"],
                    {
                        "generated_at_utc": "VALID",
                        "expires_at_utc": "FRESH",
                    },
                )
                self.assertFalse(guardian["global_safety"])

                _write_overlay(paths, authored_at=NOW + timedelta(seconds=20))
                applied = apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=_sources(paths),
                    now=None,
                )

            self.assertEqual(applied.action, "TRADE")
            self.assertEqual(
                applied.evidence_packet_sha256,
                prepared.evidence_packet_sha256,
            )

    def test_guardian_technical_observation_cannot_carry_entry_action(self) -> None:
        for event_type in ("TECHNICAL_STATE_CHANGE", "TECHNICAL_INPUT_STALE"):
            with self.subTest(event_type=event_type), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "guardian_action_receipt.json"
                receipt = _guardian_action_receipt(
                    pair="EUR_USD",
                    action="TRADE",
                    event_type=event_type,
                )
                receipt["selected_event"]["direction"] = "LONG"
                receipt["event"]["direction"] = "LONG"
                receipt["receipt"]["side"] = "LONG"
                path.write_text(json.dumps(receipt))

                material = (
                    market_read_overlay_module.guardian_action_receipt_scope_material(
                        path,
                        baseline_pairs=["EUR_USD"],
                        as_of=NOW,
                    )
                )

                self.assertTrue(material["global_safety"])
                self.assertIn(
                    "TECHNICAL_OBSERVATION_ACTION_FORBIDDEN:TRADE",
                    material["global_reasons"],
                )
                self.assertIn(
                    "TECHNICAL_OBSERVATION_ACTION_HINT_FORBIDDEN:TRADE",
                    material["global_reasons"],
                )

    def test_guardian_entry_requires_canonical_mirrored_direction(self) -> None:
        for direction in (None, "BUY", "SELL", "UP", "DOWN"):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "guardian_action_receipt.json"
                receipt = _guardian_action_receipt(
                    pair="EUR_USD",
                    action="TRADE",
                    event_type="FAILED_ACCEPTANCE",
                )
                receipt["selected_event"]["direction"] = direction
                receipt["event"]["direction"] = direction
                receipt["receipt"]["side"] = (
                    "LONG" if direction in {"BUY", "UP"} else "SHORT"
                )
                path.write_text(json.dumps(receipt))

                material = (
                    market_read_overlay_module.guardian_action_receipt_scope_material(
                        path,
                        baseline_pairs=["EUR_USD"],
                        as_of=NOW,
                    )
                )

                self.assertTrue(material["global_safety"])
                self.assertIn(
                    "ENTRY_DIRECTION_CONTRACT_BROKEN",
                    material["global_reasons"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_action_receipt.json"
            receipt = _guardian_action_receipt(
                pair="EUR_USD",
                action="TRADE",
                event_type="FAILED_ACCEPTANCE",
            )
            receipt["selected_event"]["direction"] = "LONG"
            receipt["event"]["direction"] = "LONG"
            receipt["receipt"]["side"] = "LONG"
            path.write_text(json.dumps(receipt))

            canonical = (
                market_read_overlay_module.guardian_action_receipt_scope_material(
                    path,
                    baseline_pairs=["EUR_USD"],
                    as_of=NOW,
                )
            )

            self.assertFalse(canonical["global_safety"])

    def test_guardian_scope_rejects_overflow_float_and_lone_surrogate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_action_receipt.json"
            encoded = json.dumps(_guardian_action_receipt(pair="EUR_USD"))
            path.write_text(encoded[:-1] + ',"overflow":1e9999}')
            overflow = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            self.assertEqual(overflow["parse_status"], "INVALID")

            receipt = _guardian_action_receipt(pair="EUR_USD")
            receipt["untrusted_text"] = "\ud800"
            path.write_text(json.dumps(receipt))
            surrogate = market_read_overlay_module.guardian_action_receipt_scope_material(
                path,
                baseline_pairs=["EUR_USD"],
                as_of=NOW,
            )
            self.assertEqual(surrogate["parse_status"], "INVALID")

    def test_unrelated_guardian_contract_or_issue_change_stales_ai_review(self) -> None:
        for mutation in (
            "status",
            "receipt_status",
            "receipt_lifecycle",
            "dispatcher_status",
            "block_issue",
            "unknown_event_type",
            "unknown_action",
            "nested_gateway",
            "nested_no_direct",
            "mirror_identity_missing",
            "all_identity_missing",
            "event_live_permission",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                receipt = _guardian_action_receipt()
                if mutation == "status":
                    receipt["status"] = "REJECTED"
                elif mutation == "receipt_status":
                    receipt["receipt_status"] = "REJECTED"
                elif mutation == "receipt_lifecycle":
                    receipt["receipt_lifecycle"] = "EXPIRED"
                elif mutation == "dispatcher_status":
                    receipt["dispatcher_status"] = "FAILED"
                elif mutation == "block_issue":
                    receipt["issues"] = [
                        {"code": "GUARDIAN_CONTRACT_BROKEN", "severity": "BLOCK"}
                    ]
                elif mutation == "unknown_event_type":
                    receipt["selected_event"]["event_type"] = "FUTURE_UNKNOWN"
                    receipt["event"]["event_type"] = "FUTURE_UNKNOWN"
                elif mutation == "unknown_action":
                    receipt["selected_event"]["action_hint"] = "UNKNOWN_ACTION"
                    receipt["event"]["action_hint"] = "UNKNOWN_ACTION"
                    receipt["receipt"]["action"] = "UNKNOWN_ACTION"
                elif mutation == "nested_gateway":
                    receipt["receipt"]["gateway_required"] = False
                elif mutation == "nested_no_direct":
                    receipt["receipt"]["no_direct_oanda"] = False
                elif mutation == "mirror_identity_missing":
                    receipt["event"].pop("event_type")
                elif mutation == "all_identity_missing":
                    for source in (
                        receipt["selected_event"],
                        receipt["event"],
                        receipt["receipt"],
                    ):
                        source.pop("event_id", None)
                        source.pop("dedupe_key", None)
                    receipt.pop("selected_event_id")
                    receipt.pop("selected_event_dedupe_key")
                else:
                    receipt["selected_event"]["details"][
                        "live_permission_allowed"
                    ] = True
                paths["guardian_action_receipt"].write_text(json.dumps(receipt))

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "changed_material=source:guardian_action_receipt",
                ):
                    _apply(paths)

    def test_unrelated_consumed_guardian_receipt_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            receipt = _guardian_action_receipt()
            receipt["receipt_lifecycle"] = "CONSUMED"
            receipt["consumed_by_trader"] = True
            receipt["consumed_at_utc"] = (NOW + timedelta(seconds=30)).isoformat()
            paths["guardian_action_receipt"].write_text(json.dumps(receipt))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_selected_pair_consumed_guardian_receipt_stales_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            receipt = _guardian_action_receipt(
                pair="EUR_USD",
                event_id="selected-consumption",
            )
            paths["guardian_action_receipt"].write_text(json.dumps(receipt))
            _reprepare(paths)
            _write_overlay(paths)
            receipt["receipt_lifecycle"] = "CONSUMED"
            receipt["consumed_by_trader"] = True
            receipt["consumed_at_utc"] = (NOW + timedelta(seconds=30)).isoformat()
            paths["guardian_action_receipt"].write_text(json.dumps(receipt))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "changed_material=source:guardian_action_receipt",
            ):
                _apply(paths)

    def test_selected_pair_guardian_receipt_stales_with_named_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            paths["guardian_action_receipt"].write_text(
                json.dumps(
                    _guardian_action_receipt(
                        pair="EUR_USD",
                        action="NO_ACTION",
                        event_id="selected-pair-change",
                    )
                )
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "changed_material=source:guardian_action_receipt",
            ):
                _apply(paths)

    def test_selected_pair_guardian_publication_clock_only_does_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            selected = _guardian_action_receipt(
                pair="EUR_USD",
                action="NO_ACTION",
                event_id="selected-pair-clock",
            )
            paths["guardian_action_receipt"].write_text(json.dumps(selected))
            _reprepare(paths)
            _write_overlay(paths)
            selected["generated_at_utc"] = (NOW + timedelta(seconds=5)).isoformat()
            selected["expires_at_utc"] = (NOW + timedelta(hours=2)).isoformat()
            paths["guardian_action_receipt"].write_text(json.dumps(selected))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_selected_pair_guardian_semantic_change_stales_same_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            selected = _guardian_action_receipt(
                pair="EUR_USD",
                action="NO_ACTION",
                event_id="selected-pair-same-identity",
            )
            paths["guardian_action_receipt"].write_text(json.dumps(selected))
            _reprepare(paths)
            _write_overlay(paths)
            selected["selected_event"]["details"]["material_fingerprint"][
                "dominant_regime"
            ] = "TREND"
            paths["guardian_action_receipt"].write_text(json.dumps(selected))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "changed_material=source:guardian_action_receipt",
            ):
                _apply(paths)

    def test_global_guardian_margin_receipt_stales_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            paths["guardian_action_receipt"].write_text(
                json.dumps(
                    _guardian_action_receipt(
                        pair="AUD_CAD",
                        action="HOLD",
                        event_type="MARGIN_PRESSURE",
                        severity="P0",
                        event_id="global-margin-change",
                        thesis_state="EMERGENCY",
                    )
                )
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "changed_material=source:guardian_action_receipt",
            ):
                _apply(paths)

    def test_guardian_rotation_during_packet_rebuild_is_not_missed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            original_projection = (
                market_read_overlay_module.projection_calibration_evidence
            )

            def rotate_before_projection_returns(*args, **kwargs):
                paths["guardian_action_receipt"].write_text(
                    json.dumps(
                        _guardian_action_receipt(
                            pair="EUR_USD",
                            action="NO_ACTION",
                            event_id="mid-build-selected-change",
                        )
                    )
                )
                return original_projection(*args, **kwargs)

            with patch.object(
                market_read_overlay_module,
                "projection_calibration_evidence",
                side_effect=rotate_before_projection_returns,
            ), self.assertRaisesRegex(
                MarketReadOverlayError,
                "changed_material=source:guardian_action_receipt",
            ):
                _apply(paths)

    def test_stale_overlay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, authored_at=NOW - timedelta(minutes=16))

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_OVERLAY_STALE"):
                _apply(paths)

    def test_latest_truly_resolved_v2_prediction_must_be_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_id = "mr2:" + "a" * 64
            predictions = root / "market_read_predictions.jsonl"
            rows = [
                _v2_prediction("mr2:" + "b" * 64, resolution_status="UNRESOLVED"),
                {
                    **_v2_prediction("mr2:" + "c" * 64),
                    "source_snapshot_conflict": True,
                    "score_eligible": False,
                },
                _v2_prediction(prediction_id),
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths = _prepared_paths(root, predictions_path=predictions)
            _write_overlay(paths)

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_PRIOR_PREDICTION_NOT_REVIEWED",
            ):
                _apply(paths)

            _write_overlay(paths, prior_prediction_ids=[prediction_id])
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            self.assertEqual(
                final["market_read_review"]["prior_prediction_ids"],
                [prediction_id],
            )

    def test_directional_read_requires_numeric_geometry_around_current_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            read = _market_read()
            read["next_30m_prediction"]["target_zone"] = "higher after 30m"
            _write_overlay(paths, market_read=read)

            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_GEOMETRY_INCOMPLETE"):
                _apply(paths)

    def test_directional_read_rejects_any_target_or_invalidation_rail_on_wrong_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            read = _market_read()
            read["next_30m_prediction"]["target_zone"] = "1.0990 to 1.1030"
            _write_overlay(paths, market_read=read)

            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_GEOMETRY_CONFLICT"):
                _apply(paths)

            read = _market_read()
            read["best_trade_if_forced"]["tp"] = "1.0990 to 1.1040"
            _write_overlay(paths, market_read=read)
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "AI_MARKET_READ_FORCED_GEOMETRY_CONFLICT",
            ):
                _apply(paths)

    def test_trade_source_uses_five_minute_ai_read_window_not_post_quote_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(
                Path(tmp),
                snapshot_at=NOW - timedelta(seconds=30),
            )
            _write_overlay(paths)

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_trade_rejects_source_older_than_read_only_snapshot_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(
                Path(tmp),
                snapshot_at=NOW - timedelta(minutes=5, seconds=1),
            )
            _write_overlay(paths)

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_SOURCE_STALE"):
                _apply(paths)

    def test_stale_trade_baseline_can_still_publish_a_nontrade_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = _baseline()
            baseline["generated_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths = _prepared_paths(
                Path(tmp),
                baseline=baseline,
                snapshot_at=NOW - timedelta(minutes=10),
            )
            _write_overlay(paths, disposition="VETO_WAIT")

            summary = _apply(paths)

            self.assertEqual(summary.action, "WAIT")

    def test_range_read_requires_bracketed_targets_and_outer_invalidations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, market_read=_range_market_read())
            _apply(paths)

            read = _range_market_read()
            read["next_2h_prediction"]["invalidation"] = "1.1000 to 1.1030"
            _write_overlay(paths, market_read=read)
            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_RANGE_GEOMETRY_CONFLICT"):
                _apply(paths)

    def test_m15_recovery_allocation_uses_dedicated_context_for_both_proof_modes(
        self,
    ) -> None:
        cases = (
            (
                "TP_PROOF_COLLECTION_HARVEST",
                7,
                "M15_RECOVERY_EDGE_COLLECTION",
                False,
                True,
            ),
            (
                "TP_PROVEN_HARVEST",
                20,
                "EXACT_VEHICLE_TAKE_PROFIT",
                True,
                False,
            ),
        )
        for mode, trades, edge_basis, positive_proven, collection_proven in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _configure_m15_recovery_lane(
                    paths,
                    mode=mode,
                    trades=trades,
                )
                packet = json.loads(paths["packet"].read_text())
                board = packet["capital_allocation_board"]
                lane = board["selected_lane"]

                self.assertTrue(lane["allocation_eligible"])
                self.assertEqual(lane["edge_basis"], edge_basis)
                self.assertIs(lane["positive_edge_proven"], positive_proven)
                self.assertIs(
                    lane["edge_collection_proven"],
                    collection_proven,
                )
                self.assertEqual(
                    lane["m15_recovery"]["status"],
                    "M15_RECOVERY_VERIFIED",
                )
                self.assertEqual(lane["allowed_size_multiples"], [1.0])
                self.assertNotEqual(
                    lane["forecast"]["technical_context"]["status"],
                    "VALID",
                )
                self.assertEqual(
                    board["forecast_context_scope"],
                    "SELECTED_LANE_M15_RECOVERY",
                )
                self.assertEqual(
                    board["forecast_context"]["status"],
                    "M15_RECOVERY_VERIFIED",
                )
                self.assertNotIn(
                    "technical_context",
                    board["forecast_context"],
                )
                self.assertEqual(
                    board["forecast_context"]["edge_basis"],
                    edge_basis,
                )
                self.assertIn(
                    "M15_RECOVERY_EDGE_COLLECTION is evidence collection, not positive edge proof",
                    board["allocation_rule"],
                )

                _write_overlay(paths, size_multiple=0.75)
                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
                ):
                    _apply(paths)

                _write_overlay(paths, size_multiple=1.0)
                summary = _apply(paths)
                self.assertEqual(summary.action, "TRADE")

    def test_m15_recovery_missing_or_tampered_chain_fails_closed(self) -> None:
        cases = (
            ("TP_PROOF_COLLECTION_HARVEST", 7, "source_receipt"),
            ("TP_PROOF_COLLECTION_HARVEST", 7, "risk_marker"),
            ("TP_PROOF_COLLECTION_HARVEST", 7, "promote_collection"),
            ("TP_PROOF_COLLECTION_HARVEST", 7, "context_timeframes"),
            ("TP_PROVEN_HARVEST", 20, "missing_lane_binding"),
            ("TP_PROVEN_HARVEST", 20, "method"),
            ("TP_PROVEN_HARVEST", 20, "market_order"),
            ("TP_PROVEN_HARVEST", 20, "limit_order"),
            ("TP_PROVEN_HARVEST", 20, "direction"),
            ("TP_PROVEN_HARVEST", 20, "units"),
            ("TP_PROVEN_HARVEST", 20, "manual_mutation"),
            ("TP_PROVEN_HARVEST", 20, "geometry_only_residue"),
            ("TP_PROVEN_HARVEST", 20, "forecast_mode_only_residue"),
            ("TP_PROVEN_HARVEST", 20, "lane_contract_only_residue"),
            ("TP_PROVEN_HARVEST", 20, "atr_only_residue"),
            ("TP_PROVEN_HARVEST", 20, "tp_source_only_residue"),
        )
        for mode, trades, mutation in cases:
            with self.subTest(mode=mode, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _configure_m15_recovery_lane(
                    paths,
                    mode=mode,
                    trades=trades,
                )
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                intent = result["intent"]
                metadata = intent["metadata"]
                if mutation == "source_receipt":
                    metadata["m15_recovery_micro_receipt"][
                        "spread_cap_pips"
                    ] = 9.0
                elif mutation == "risk_marker":
                    result["risk_issues"] = []
                elif mutation == "promote_collection":
                    metadata["positive_rotation_live_ready"] = True
                elif mutation == "context_timeframes":
                    metadata["m15_recovery_context_timeframes"] = [
                        "M5",
                        "M15",
                        "M30",
                        "H1",
                        "H4",
                        "D",
                    ]
                elif mutation == "missing_lane_binding":
                    metadata.pop("m15_recovery_lane_binding")
                elif mutation == "method":
                    intent["market_context"]["method"] = "TREND_CONTINUATION"
                elif mutation == "market_order":
                    intent["order_type"] = "MARKET"
                elif mutation == "limit_order":
                    intent["order_type"] = "LIMIT"
                elif mutation == "direction":
                    metadata["forecast_direction"] = "DOWN"
                elif mutation == "units":
                    intent["units"] = 1000
                elif mutation == "manual_mutation":
                    metadata[
                        "m15_recovery_micro_manual_position_mutation_allowed"
                    ] = True
                elif mutation in {
                    "geometry_only_residue",
                    "forecast_mode_only_residue",
                    "lane_contract_only_residue",
                    "atr_only_residue",
                    "tp_source_only_residue",
                }:
                    for key in tuple(metadata):
                        if key.startswith("m15_recovery_") or key.startswith(
                            "forecast_m15_recovery_"
                        ):
                            metadata.pop(key)
                    geometry_keys = {
                        "geometry_source_recovery_receipt_sha256",
                        "geometry_forecast_binding_sha256",
                        "geometry_forecast_target_price",
                        "geometry_forecast_invalidation_price",
                        "geometry_tp_within_forecast_target",
                        "geometry_sl_at_or_beyond_forecast_invalidation",
                        "geometry_generic_overwrite_forbidden",
                    }
                    if mutation == "geometry_only_residue":
                        for key in tuple(metadata):
                            if key.startswith("geometry_") and key not in {
                                "geometry_model",
                            } | geometry_keys:
                                metadata.pop(key)
                    else:
                        for key in tuple(metadata):
                            if key.startswith("geometry_"):
                                metadata.pop(key)
                        metadata["geometry_model"] = "ORDINARY_GEOMETRY"
                        if mutation == "forecast_mode_only_residue":
                            metadata["forecast_m15_recovery_mode"] = (
                                "M15_RECOVERY_MICRO"
                            )
                        elif mutation == "lane_contract_only_residue":
                            metadata["m15_recovery_lane_contract"] = (
                                "QR_M15_RECOVERY_LANE_V1"
                            )
                        elif mutation == "atr_only_residue":
                            metadata["geometry_atr_source_timeframe"] = "M15"
                        else:
                            metadata["tp_target_source"] = (
                                "M15_RECOVERY_FORECAST_BOUND"
                            )
                else:
                    raise AssertionError(f"unknown mutation {mutation}")
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                packet = json.loads(paths["packet"].read_text())
                lane = packet["capital_allocation_board"]["selected_lane"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertTrue(lane["m15_recovery"]["claimed"])
                self.assertFalse(lane["m15_recovery"]["valid"])
                self.assertTrue(lane["m15_recovery"]["error_codes"])

                _write_overlay(paths)
                with self.assertRaises(MarketReadOverlayError):
                    _apply(paths)

    def test_m15_recovery_acceptance_revalidates_self_hashed_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _configure_m15_recovery_lane(
                paths,
                mode="TP_PROOF_COLLECTION_HARVEST",
                trades=7,
            )
            packet = json.loads(paths["packet"].read_text())
            original_board = packet["capital_allocation_board"]

            mutations = {
                "side": "SHORT",
                "method": "TREND_CONTINUATION",
                "forecast_cycle_id": "attacker-cycle",
                "source_receipt_sha256": "0" * 64,
                "forecast_binding_sha256": "1" * 64,
                "lane_binding_sha256": "2" * 64,
                "edge_basis": "EXACT_VEHICLE_TAKE_PROFIT",
            }
            for field, value in mutations.items():
                with self.subTest(field=field):
                    board = json.loads(json.dumps(original_board))
                    scoped = board["forecast_context"]
                    scoped[field] = value
                    body = dict(scoped)
                    body.pop("recovery_context_sha256")
                    scoped["recovery_context_sha256"] = canonical_json_sha256(
                        body
                    )
                    error = market_read_overlay_module._selected_lane_forecast_context_binding_error(
                        board=board,
                        selected_lane=board["selected_lane"],
                        lane_id=M15_RECOVERY_LANE_ID,
                    )
                    self.assertIsNotNone(error)


def _synthetic_execution_cost_surface() -> dict:
    """Stable, valid global cost material for tests outside cost calibration.

    The allocation tests use small semantic ledgers to exercise exact-lane
    binding and WAL behavior. They intentionally do not manufacture the 20
    independently audited entry/TP/SL transport receipts required by the live
    cost contract; this fixture keeps that orthogonal concern constant.
    """

    observed_at = (NOW - timedelta(days=1)).isoformat()

    def transport_section(label: str) -> dict:
        rows_sha256 = hashlib.sha256(
            f"synthetic-{label}-cost-cohort".encode()
        ).hexdigest()
        return {
            "samples": 20,
            "adverse_p95_pips": 0.0,
            "adverse_max_pips": 0.0,
            "oldest_fill_utc": observed_at,
            "latest_fill_utc": observed_at,
            "rows_sha256": rows_sha256,
        }

    material = {
        "contract": "QR_NET_EXECUTION_COST_FLOOR_V1",
        "parse_status": "VALID",
        "scope": "SYSTEM_GATEWAY_ATTRIBUTED_ALL_PAIRS_SIDES_METHODS",
        "minimum_samples": 20,
        "maximum_sample_age_seconds": 90 * 24 * 60 * 60,
        "market_entry": transport_section("market-entry"),
        "take_profit_exit": transport_section("take-profit-exit"),
        "stop_loss_exit": transport_section("stop-loss-exit"),
        "global_financing": {
            "observation_trades": 20,
            "adverse_trades": 1,
            "entry_units_total": 20_000.0,
            "adverse_total_jpy": 0.002,
            "adverse_mean_jpy_per_unit": 0.000002,
            "adverse_occurrence_wilson95_upper": 0.236131193,
            "adverse_stress_jpy_per_unit": 0.000000472262,
            "oldest_observation_utc": observed_at,
            "latest_observation_utc": observed_at,
        },
    }
    return {
        **material,
        "execution_cost_surface_sha256": canonical_json_sha256(material),
    }


def _synthetic_execution_cost_floor(
    *,
    scope_key: str = "EUR_USD|LONG|TREND_CONTINUATION|MARKET",
) -> dict:
    """Minimal content-addressed proof accepted by the pure numeric helper."""

    material = {
        "contract": "QR_NET_EXECUTION_COST_FLOOR_V1",
        "status": "PASSED",
        "market_entry_adverse_p95_pips": 0.0,
        "audited_protected_exit_adverse_p95_pips": 0.0,
        "financing_adverse_stress_jpy_per_unit": 0.0,
        "scope_key": scope_key,
        "spread_double_count_forbidden": True,
    }
    return {**material, "proof_sha256": canonical_json_sha256(material)}


    def test_target_path_live_learning_is_bounded_edge_collection_not_positive_edge(self) -> None:
        lane_id = "target-path:NZD_USD:LONG:RELOAD"
        metadata = {
            "target_path_live_mode": "LIVE_LEARNING",
            "valid_as_target_path": "YES",
            "daily_target_mode": "PACE_5",
            "remaining_to_5pct_yen": 14_725.79,
            "target_path_role": "RELOAD",
            "path_board_available": True,
            "five_pct_path_available": True,
            "attack_stack_available": True,
            "maps_to_attack_stack": True,
            "attack_stack_slot": "RELOAD",
            "conviction_grade": "B+",
            "suggested_units": 1_000,
            "risk_yen": 300.61,
            "risk_pct": 0.1021,
            "target_yen": 617.47,
            "contribution_to_5pct": 617.47,
            "exact_pretrade_passed": True,
            "spread_guard_passed": True,
            "pricing_probe_passed": True,
            "fill_guard_passed": True,
            "same_thesis_lost_recently": False,
            "vehicle_unchanged_after_loss": False,
        }
        result = {
            "lane_id": lane_id,
            "status": "LIVE_READY",
            "risk_allowed": True,
            "live_blocker_codes": [],
            "risk_metrics": {
                "entry_price": 0.58070,
                "loss_pips": 18.5,
                "reward_pips": 38.0,
                "risk_jpy": 300.61,
                "reward_jpy": 617.47,
                "reward_risk": 38.0 / 18.5,
                "spread_pips": 1.5,
                "jpy_per_pip": 16.2492,
                "estimated_margin_jpy": 3_776.58,
            },
            "intent": {
                "pair": "NZD_USD",
                "side": "LONG",
                "order_type": "LIMIT",
                "units": 1_000,
                "entry": 0.58070,
                "tp": 0.58450,
                "sl": 0.57885,
                "market_context": {"method": "TREND_CONTINUATION"},
                "metadata": metadata,
            },
        }
        snapshot = {
            "account": {"nav_jpy": 274_904.69},
            "quotes": {"NZD_USD": {"bid": 0.58097, "ask": 0.58112}},
            "home_conversions": {"USD": 162.492},
        }

        lane = market_read_overlay_module._capital_allocation_lane(
            result,
            account_nav_jpy=274_904.69,
            broker_snapshot=snapshot,
        )

        assert lane is not None
        self.assertTrue(lane["allocation_eligible"], lane)
        self.assertFalse(lane["positive_edge_proven"])
        self.assertTrue(lane["edge_collection_proven"])
        self.assertEqual(
            lane["edge_basis"],
            "TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION",
        )
        self.assertEqual(lane["allowed_size_multiples"], [1.0])
        self.assertEqual(
            lane["numeric_ceiling"]["reason"],
            "TARGET_PATH_LIVE_LEARNING_PREBOUNDED_CONTRACT",
        )

        metadata["risk_pct"] = 0.16
        blocked = market_read_overlay_module._capital_allocation_lane(
            result,
            account_nav_jpy=274_904.69,
            broker_snapshot=snapshot,
        )
        assert blocked is not None
        self.assertFalse(blocked["allocation_eligible"])
        self.assertIn(
            "TARGET_PATH_LIVE_LEARNING_RISK_CAP_EXCEEDED",
            blocked["live_blocker_codes"],
        )

    def test_target_path_live_learning_context_survives_overlay_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            result = intents["results"][0]
            intent = result["intent"]
            intent.update(
                {
                    "order_type": "LIMIT",
                    "units": 500,
                    "entry": 1.0995,
                    "tp": 1.1030,
                    "sl": 1.0975,
                }
            )
            intent["metadata"] = {
                "target_path_live_mode": "LIVE_LEARNING",
                "valid_as_target_path": "YES",
                "daily_target_mode": "PACE_5",
                "remaining_to_5pct_yen": 4_000.0,
                "target_path_role": "RELOAD",
                "path_board_available": True,
                "five_pct_path_available": True,
                "attack_stack_available": True,
                "maps_to_attack_stack": True,
                "attack_stack_slot": "RELOAD",
                "conviction_grade": "B+",
                "suggested_units": 500,
                "risk_yen": 100.0,
                "risk_pct": 0.1,
                "target_yen": 175.0,
                "contribution_to_5pct": 175.0,
                "exact_pretrade_passed": True,
                "spread_guard_passed": True,
                "pricing_probe_passed": True,
                "fill_guard_passed": True,
                "same_thesis_lost_recently": False,
                "vehicle_unchanged_after_loss": False,
            }
            result["risk_metrics"] = {
                "entry_price": 1.0995,
                "loss_pips": 20.0,
                "reward_pips": 35.0,
                "risk_jpy": 100.0,
                "reward_jpy": 175.0,
                "reward_risk": 1.75,
                "spread_pips": 2.0,
                "jpy_per_pip": 5.0,
            }
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]
            lane = board["selected_lane"]
            self.assertEqual(
                board["forecast_context_scope"],
                "SELECTED_LANE_TARGET_PATH_LIVE_LEARNING",
            )
            self.assertTrue(lane["allocation_eligible"], lane)
            self.assertEqual(lane["allowed_size_multiples"], [1.0])
            self.assertFalse(lane["positive_edge_proven"])
            self.assertTrue(lane["edge_collection_proven"])

            _write_overlay(paths, size_multiple=1.0)
            summary = _apply(paths)
            self.assertEqual(summary.action, "TRADE")
            output = json.loads(paths["output"].read_text())
            self.assertEqual(
                output["decision_provenance"]["capital_allocation_edge_basis"],
                "TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION",
            )
            self.assertEqual(output["capital_allocation"]["selected_units"], 500)


def _prepared_paths(
    root: Path,
    *,
    baseline: dict | None = None,
    predictions_path: Path | None = None,
    snapshot_at: datetime = NOW,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": root / "baseline.json",
        "packet": root / "packet.json",
        "overlay": root / "overlay.json",
        "output": root / "output.json",
        "snapshot": root / "broker_snapshot.json",
        "intents": root / "order_intents.json",
        "predictions": predictions_path or root / "market_read_predictions.jsonl",
        "watchdog": root / "qr_trader_run_watchdog.json",
        "guardian_action_receipt": root / "guardian_action_receipt.json",
        "execution_ledger": root / "execution_ledger.db",
        "pair_charts": root / "pair_charts.json",
    }
    _write_exact_vehicle_ledger(
        paths["execution_ledger"],
        [320.0] * 8,
    )
    paths["baseline"].write_text(json.dumps(baseline or _baseline()))
    paths["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": snapshot_at.isoformat(),
                "account": {
                    "nav_jpy": 100_000.0,
                    "fetched_at_utc": snapshot_at.isoformat(),
                },
                "home_conversions": {"USD": 100.0, "JPY": 1.0},
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "timestamp_utc": snapshot_at.isoformat(),
                    },
                },
            }
        )
    )
    forecast_context_evidence = _forecast_context_evidence(
        "EUR_USD",
        1.1001,
        now_utc=snapshot_at,
        pair_charts_path=paths["pair_charts"],
        session_tag="ROLLOVER",
        chart_story=(
            "M5(TREND_STRONG ADX=31) M15(TREND_WEAK ADX=24) "
            "H1(TREND_STRONG ADX=30) H4(TREND_STRONG ADX=30)"
        ),
    )
    paths["intents"].write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "results": [
                    {
                        "lane_id": LANE_ID,
                        "status": "LIVE_READY",
                        "risk_allowed": True,
                        "live_blocker_codes": [],
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "order_type": "MARKET",
                            "units": 1200,
                            "entry": 1.1002,
                            "tp": 1.1040,
                            "sl": 1.0980,
                            "market_context": {"method": "TREND_CONTINUATION"},
                            "metadata": {
                                "attach_take_profit_on_fill": True,
                                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                                "capture_take_profit_exact_vehicle_required": True,
                                "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                                "capture_take_profit_scope_key": (
                                    "EUR_USD|LONG|TREND_CONTINUATION|MARKET|TAKE_PROFIT_ORDER"
                                ),
                                "capture_take_profit_vehicle": "MARKET",
                                "capture_take_profit_metrics_source": (
                                    "data/execution_ledger.db:exact_vehicle_take_profit"
                                ),
                                "capture_take_profit_expectancy_jpy": 320.0,
                                "capture_take_profit_net_jpy": 2560.0,
                                "capture_take_profit_trades": 8,
                                "capture_take_profit_wins": 8,
                                "capture_take_profit_losses": 0,
                                "capture_take_profit_avg_win_jpy": 320.0,
                                "capture_take_profit_avg_loss_jpy": 0.0,
                                "capture_exact_vehicle_net_scope": (
                                    "PAIR_SIDE_METHOD_VEHICLE"
                                ),
                                "capture_exact_vehicle_net_scope_key": (
                                    "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                                ),
                                "capture_exact_vehicle_net_vehicle": "MARKET",
                                "capture_exact_vehicle_net_metrics_source": (
                                    "data/execution_ledger.db:exact_vehicle_net"
                                ),
                                "capture_exact_vehicle_net_exit_scope": (
                                    "ALL_AUDITED_EXITS"
                                ),
                                "capture_exact_vehicle_net_trades": 8,
                                "capture_exact_vehicle_net_wins": 8,
                                "capture_exact_vehicle_net_losses": 0,
                                "capture_exact_vehicle_net_jpy": 2560.0,
                                "capture_exact_vehicle_net_expectancy_jpy": 320.0,
                                "capture_exact_vehicle_net_avg_win_jpy": 320.0,
                                "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
                                "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                                "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                                "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                                    hashlib.sha256(b"[]").hexdigest()
                                ),
                                "capture_market_close_expectancy_jpy": -120.0,
                                "capture_avg_win_jpy": 500.0,
                                "capture_avg_loss_jpy": 300.0,
                                "forecast_direction": "UP",
                                "forecast_confidence": 0.72,
                                "forecast_raw_confidence": 0.80,
                                "forecast_current_price": 1.1001,
                                "forecast_technical_context": forecast_context_evidence,
                                **_forecast_weighting_metadata(
                                    forecast_context_evidence
                                ),
                                "forecast_target_price": 1.1050,
                                "forecast_invalidation_price": 1.0985,
                                "forecast_directional_calibration_name": "directional_forecast_up",
                                "forecast_calibration_multiplier": 0.90,
                                "forecast_directional_economic_hit_rate": 0.61,
                                "forecast_directional_economic_samples": 100,
                                "forecast_directional_hit_rate": 0.67,
                                "forecast_directional_samples": 72,
                                "forecast_directional_timeout_rate": 0.28,
                                "max_loss_jpy": 400.0,
                            },
                        },
                        "risk_metrics": {
                            "entry_price": 1.1002,
                            "loss_pips": 22.0,
                            "reward_pips": 38.0,
                            "risk_jpy": 264.0,
                            "reward_jpy": 456.0,
                            "reward_risk": 38.0 / 22.0,
                            "spread_pips": 2.0,
                            "jpy_per_pip": 12.0,
                            "estimated_margin_jpy": 1200.0,
                        },
                    }
                ],
            }
        )
    )
    if not paths["predictions"].exists():
        paths["predictions"].write_text("")
    paths["watchdog"].write_text(json.dumps(_watchdog()))
    paths["guardian_action_receipt"].write_text(
        json.dumps(_guardian_action_receipt())
    )
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=_sources(paths),
        now=NOW,
    )
    return paths


def _sources(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "calendar": Path("data/economic_calendar.json"),
        "strategy_profile": Path("data/strategy_profile.json"),
        "pair_charts": paths["pair_charts"],
        "broker_snapshot": paths["snapshot"],
        "order_intents": paths["intents"],
        "market_read_predictions": paths["predictions"],
        "qr_trader_run_watchdog": paths["watchdog"],
        "guardian_action_receipt": paths["guardian_action_receipt"],
        "execution_ledger": paths["execution_ledger"],
    }


def _write_exact_vehicle_ledger(path: Path, outcomes: list[float]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT,
                event_type TEXT,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                financing_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT, updated_at_utc TEXT)"
        )
        conn.execute(
            "INSERT INTO sync_state VALUES (?, ?, ?)",
            (
                "oanda_transaction_coverage_start_utc",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        lane = LANE_ID
        for index, realized in enumerate(outcomes):
            trade_id = f"trade-{index}"
            order_id = f"entry-{index}"
            entry_ts = f"2026-07-01T00:{index:02d}:00Z"
            close_ts = f"2026-07-01T01:{index:02d}:00Z"
            entry_raw = {
                "id": f"fill-{index}",
                "time": entry_ts,
                "type": "ORDER_FILL",
                "orderID": order_id,
                "instrument": "EUR_USD",
                "units": "1000",
                "reason": "MARKET_ORDER",
                "tradeOpened": {
                    "tradeID": trade_id,
                    "units": "1000",
                },
            }
            exit_reason = (
                "TAKE_PROFIT_ORDER"
                if realized > 0
                else "MARKET_ORDER_TRADE_CLOSE"
            )
            close_raw = {
                "id": f"close-{index}",
                "time": close_ts,
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "orderID": f"close-{index}",
                "reason": exit_reason,
                "commission": "0.0",
                "guaranteedExecutionFee": "0.0",
                "tradesClosed": [
                    {
                        "tradeID": trade_id,
                        "realizedPL": str(realized),
                        "financing": "0.0",
                    }
                ],
            }
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"gateway-{index}", entry_ts, "GATEWAY_ORDER_SENT", lane,
                    order_id, trade_id, "EUR_USD", "LONG", 1000, None, 0.0,
                    "MARKET_ORDER", json.dumps({"type": "MARKET_ORDER"}),
                ),
            )
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"fill-{index}", entry_ts, "ORDER_FILLED", lane,
                    order_id, trade_id, "EUR_USD", "LONG", 1000, None, 0.0,
                    "MARKET_ORDER", json.dumps(entry_raw),
                ),
            )
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"close-{index}", close_ts, "TRADE_CLOSED", None,
                    f"close-{index}", trade_id, "EUR_USD", "SHORT", 1000,
                    realized, 0.0, exit_reason, json.dumps(close_raw),
                ),
            )


def _append_vehicle_trade(
    path: Path,
    *,
    pair: str,
    side: str,
    method: str,
    vehicle: str,
    realized: float,
    index: int,
    connection: sqlite3.Connection | None = None,
    terminal: bool = True,
) -> None:
    trade_id = f"appended-trade-{index}"
    order_id = f"appended-entry-{index}"
    entry_ts = "2026-07-02T00:00:00Z"
    close_ts = "2026-07-02T01:00:00Z"
    signed_units = 1000 if side == "LONG" else -1000
    entry_reason = {
        "LIMIT": "LIMIT_ORDER",
        "STOP": "STOP_ORDER",
        "MARKET": "MARKET_ORDER",
    }[vehicle]
    lane = f"test_trader:{pair}:{side}:{method}:{vehicle}"
    entry_raw = {
        "id": f"appended-fill-{index}",
        "time": entry_ts,
        "type": "ORDER_FILL",
        "orderID": order_id,
        "instrument": pair,
        "units": str(signed_units),
        "reason": entry_reason,
        "tradeOpened": {"tradeID": trade_id, "units": str(signed_units)},
    }
    exit_reason = (
        "TAKE_PROFIT_ORDER" if realized > 0 else "MARKET_ORDER_TRADE_CLOSE"
    )
    close_raw = {
        "id": f"appended-close-{index}",
        "time": close_ts,
        "type": "ORDER_FILL",
        "instrument": pair,
        "orderID": f"appended-close-{index}",
        "reason": exit_reason,
        "commission": "0.0",
        "guaranteedExecutionFee": "0.0",
        "tradesClosed": [
            {
                "tradeID": trade_id,
                "realizedPL": str(realized),
                "financing": "0.0",
            }
        ],
    }
    if not terminal:
        close_raw.pop("tradesClosed")
        close_raw["tradeReduced"] = {
            "tradeID": trade_id,
            "realizedPL": str(realized),
            "financing": "0.0",
        }
    rows = [
        (
            f"appended-gateway-{index}", entry_ts, "GATEWAY_ORDER_SENT", lane,
            order_id, trade_id, pair, side, signed_units, None, 0.0,
            entry_reason, json.dumps({"type": entry_reason}),
        ),
        (
            f"appended-fill-{index}", entry_ts, "ORDER_FILLED", lane,
            order_id, trade_id, pair, side, signed_units, None, 0.0,
            entry_reason, json.dumps(entry_raw),
        ),
        (
            f"appended-close-{index}", close_ts,
            "TRADE_CLOSED" if terminal else "TRADE_REDUCED", None,
            f"appended-close-{index}", trade_id, pair,
            "SHORT" if side == "LONG" else "LONG", abs(signed_units),
            realized, 0.0, exit_reason, json.dumps(close_raw),
        ),
    ]
    if connection is not None:
        connection.executemany(
            "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    else:
        with sqlite3.connect(path) as conn:
            conn.executemany(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


def _configure_m15_recovery_lane(
    paths: dict[str, Path],
    *,
    mode: str,
    trades: int,
) -> None:
    for index in range(trades):
        _append_vehicle_trade(
            paths["execution_ledger"],
            pair="EUR_USD",
            side="LONG",
            method="BREAKOUT_FAILURE",
            vehicle="STOP",
            realized=320.0,
            index=10_000 + index,
        )

    baseline = json.loads(paths["baseline"].read_text())
    baseline["selected_lane_id"] = M15_RECOVERY_LANE_ID
    baseline["selected_lane_ids"] = [M15_RECOVERY_LANE_ID]
    baseline["method"] = "BREAKOUT_FAILURE"
    baseline["evidence_refs"] = [
        f"intent:{M15_RECOVERY_LANE_ID}",
        "broker:snapshot",
    ]
    paths["baseline"].write_text(json.dumps(baseline))

    receipt_body = {
        "contract": "QR_M15_RECOVERY_MICRO_V1",
        "mode": "M15_RECOVERY_MICRO",
        "status": "ELIGIBLE_FOR_MICRO_REVALIDATION",
        "pair": "EUR_USD",
        "chart_generated_at_utc": NOW.isoformat(),
        "m15_plus_scoring_input_sha256": hashlib.sha256(
            b"m15-plus-input"
        ).hexdigest(),
        "validated_at_utc": NOW.isoformat(),
        "fast_timeframe_receipts": {
            "M1": {"recovery_state": "RECOVERING"},
            "M5": {"recovery_state": "RECOVERED"},
        },
        "geometry_source": {
            "timeframe": "M15",
            "latest_clean_timestamp_utc": NOW.isoformat(),
            "atr_pips": 8.0,
            "candles_count": 120,
        },
        "current_spread_pips": 2.0,
        "spread_cap_pips": 2.5,
        "sizing": {
            "max_units": 999,
            "full_size_allowed": False,
            "minimum_units_override": False,
        },
        "live_permission": False,
        "requires_risk_gateway_revalidation": True,
        "manual_position_mutation_allowed": False,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_json_sha256(receipt_body),
    }
    scores = {"UP": 1.2, "DOWN": -0.2, "RANGE": 0.1}
    evidence_body = {
        "contract": M15_RECOVERY_FORECAST_CONTRACT,
        "source_recovery_receipt_sha256": receipt["receipt_sha256"],
        "pair": "EUR_USD",
        "chart_generated_at_utc": receipt["chart_generated_at_utc"],
        "forecast_current_price": 1.1001,
        "forecast_spread_pips": 2.0,
        "filtered_input_sha256": hashlib.sha256(
            b"m15-filtered-input"
        ).hexdigest(),
        "raw_winner": "UP",
        "component_scores": scores,
        "final_direction": "UP",
        "raw_confidence": 0.8,
        "calibration_multiplier": 0.9,
        "calibration_scope": "M15_RECOVERY_CONSERVATIVE_DIRECTIONAL_PRIOR",
        "confidence": 0.72,
        "target_price": 1.1025,
        "invalidation_price": 1.0980,
        "horizon_min": 60,
        "geometry_source_timeframe": "M15",
        "live_permission": False,
    }
    forecast_evidence = {
        **evidence_body,
        "evidence_sha256": canonical_json_sha256(evidence_body),
    }
    cycle_id = f"m15-recovery-{mode.lower()}-{trades}"
    forecast_binding = build_m15_recovery_forecast_binding(
        forecast_evidence,
        cycle_id=cycle_id,
    )
    if forecast_binding is None:
        raise AssertionError("M15 recovery forecast fixture did not bind")

    lower = hit_rate_wilson_lower(1.0, trades)
    if lower is None:
        raise AssertionError("M15 recovery Wilson fixture did not calculate")
    pessimistic = lower * 320.0 - (1.0 - lower) * 100.0
    proof_collection = mode == "TP_PROOF_COLLECTION_HARVEST"
    effective_max_loss = 320.0 if proof_collection else 400.0
    net_jpy = float(trades * 320)
    scope_key = (
        "EUR_USD|LONG|BREAKOUT_FAILURE|STOP|TAKE_PROFIT_ORDER"
    )
    exact_net_scope_key = (
        "EUR_USD|LONG|BREAKOUT_FAILURE|STOP|ALL_AUDITED_EXITS"
    )
    metadata = {
        "method": "BREAKOUT_FAILURE",
        "position_intent": "NEW",
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "capture_expectancy_jpy": -120.0,
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": scope_key,
        "capture_take_profit_vehicle": "STOP",
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_take_profit"
        ),
        "capture_take_profit_expectancy_jpy": 320.0,
        "capture_take_profit_net_jpy": net_jpy,
        "capture_take_profit_trades": trades,
        "capture_take_profit_wins": trades,
        "capture_take_profit_losses": 0,
        "capture_take_profit_avg_win_jpy": 320.0,
        "capture_take_profit_avg_loss_jpy": 0.0,
        "capture_market_close_expectancy_jpy": -120.0,
        "capture_avg_win_jpy": 320.0,
        "capture_avg_loss_jpy": 100.0,
        "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_exact_vehicle_net_scope_key": exact_net_scope_key,
        "capture_exact_vehicle_net_vehicle": "STOP",
        "capture_exact_vehicle_net_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_net"
        ),
        "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
        "capture_exact_vehicle_net_trades": trades,
        "capture_exact_vehicle_net_wins": trades,
        "capture_exact_vehicle_net_losses": 0,
        "capture_exact_vehicle_net_jpy": net_jpy,
        "capture_exact_vehicle_net_expectancy_jpy": 320.0,
        "capture_exact_vehicle_net_avg_win_jpy": 320.0,
        "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
        "capture_exact_vehicle_net_unresolved_realized_trades": 0,
        "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
        "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
            hashlib.sha256(b"[]").hexdigest()
        ),
        "positive_rotation_mode": mode,
        "positive_rotation_confidence_method": (
            "WILSON_LOWER_BOUND_STRESS_EXPECTANCY"
        ),
        "positive_rotation_confidence_z": 1.96,
        "positive_rotation_tp_wins": trades,
        "positive_rotation_tp_trades": trades,
        "positive_rotation_tp_win_rate_lower": round(lower, 6),
        "positive_rotation_loss_proxy_jpy": 100.0,
        "positive_rotation_pessimistic_expectancy_jpy": round(
            pessimistic,
            4,
        ),
        "positive_rotation_proof_collection_ready": proof_collection,
        "positive_rotation_proof_collection_mode": (
            "TP_PROOF_COLLECTION_HARVEST" if proof_collection else None
        ),
        "positive_rotation_proof_collection_min_trades": (
            5 if proof_collection else None
        ),
        "positive_rotation_proof_collection_target_trades": (
            20 if proof_collection else None
        ),
        "positive_rotation_proof_collection_gap_trades": (
            20 - trades if proof_collection else None
        ),
        "positive_rotation_live_ready": not proof_collection,
        "loss_asymmetry_guard_active": True,
        "loss_asymmetry_guard_mode": (
            "CAP_AVG_WIN" if proof_collection else "TP_PROVEN_RELAXED"
        ),
        "loss_asymmetry_guard_relaxed": not proof_collection,
        "loss_asymmetry_guard_loss_cap_jpy": 320.0,
        "loss_asymmetry_guard_base_max_loss_jpy": 400.0,
        "loss_asymmetry_guard_effective_max_loss_jpy": effective_max_loss,
        "max_loss_jpy": effective_max_loss,
        "m15_recovery_micro_contract": "QR_M15_RECOVERY_MICRO_V1",
        "m15_recovery_micro_mode": "M15_RECOVERY_MICRO",
        "m15_recovery_micro_receipt": receipt,
        "m15_recovery_micro_receipt_sha256": receipt["receipt_sha256"],
        "m15_recovery_micro_units": 999,
        "m15_recovery_micro_max_units": 999,
        "m15_recovery_micro_full_size_allowed": False,
        "m15_recovery_micro_live_permission": False,
        "m15_recovery_micro_requires_risk_gateway_revalidation": True,
        "m15_recovery_micro_manual_position_mutation_allowed": False,
        "m15_recovery_micro_shape_eligible": True,
        "m15_recovery_micro_positive_rotation_mode": mode,
        "m15_recovery_micro_proof_contract": {
            "positive_rotation_mode": mode,
            "reachable": True,
            "failed_checks": [],
        },
        "m15_recovery_micro_risk_revalidated": True,
        "m15_recovery_micro_gateway_revalidated": False,
        "forecast_m15_recovery_receipt": receipt,
        "forecast_m15_recovery_mode": "M15_RECOVERY_MICRO",
        "forecast_m15_recovery_live_permission": False,
        "forecast_m15_recovery_evidence": forecast_evidence,
        "forecast_m15_recovery_binding": forecast_binding,
        "forecast_m15_recovery_binding_sha256": forecast_binding[
            "binding_sha256"
        ],
        "forecast_cycle_id": cycle_id,
        "forecast_direction": "UP",
        "forecast_confidence": 0.72,
        "forecast_raw_confidence": 0.8,
        "forecast_calibration_multiplier": 0.9,
        "forecast_current_price": 1.1001,
        "forecast_target_price": 1.1025,
        "forecast_invalidation_price": 1.0980,
        "forecast_horizon_min": 60,
        "forecast_component_scores": scores,
        "forecast_technical_context": {},
        "m15_recovery_context_contract": M15_RECOVERY_FORECAST_CONTRACT,
        "m15_recovery_context_timeframes": ["M15", "M30", "H1", "H4", "D"],
        "geometry_model": M15_RECOVERY_GEOMETRY_CONTRACT,
        "geometry_atr_source_timeframe": "M15",
        "geometry_atr_pips": receipt["geometry_source"]["atr_pips"],
        "geometry_source_recovery_receipt_sha256": receipt[
            "receipt_sha256"
        ],
        "geometry_forecast_binding_sha256": forecast_binding[
            "binding_sha256"
        ],
        "geometry_forecast_target_price": forecast_binding["target_price"],
        "geometry_forecast_invalidation_price": forecast_binding[
            "invalidation_price"
        ],
        "geometry_tp_within_forecast_target": True,
        "geometry_sl_at_or_beyond_forecast_invalidation": True,
        "geometry_generic_overwrite_forbidden": True,
    }
    lane_binding = build_m15_recovery_lane_binding(
        forecast_binding=forecast_binding,
        pair="EUR_USD",
        side="LONG",
        method="BREAKOUT_FAILURE",
        order_type="STOP-ENTRY",
        entry=1.1003,
        tp=1.1025,
        sl=1.0980,
        producer_units=999,
        metadata=metadata,
    )
    if lane_binding is None:
        raise AssertionError("M15 recovery lane fixture did not bind")
    metadata.update(
        {
            "m15_recovery_lane_contract": M15_RECOVERY_LANE_CONTRACT,
            "m15_recovery_lane_binding": lane_binding,
            "m15_recovery_lane_binding_sha256": lane_binding[
                "binding_sha256"
            ],
        }
    )

    intents = json.loads(paths["intents"].read_text())
    intents["results"] = [
        {
            "lane_id": M15_RECOVERY_LANE_ID,
            "status": "LIVE_READY",
            "risk_allowed": True,
            "risk_issues": [
                {
                    "code": "M15_RECOVERY_RISK_REVALIDATED",
                    "message": "independent current-source validation passed",
                    "severity": "WARN",
                }
            ],
            "live_blocker_codes": [],
            "intent": {
                "pair": "EUR_USD",
                "side": "LONG",
                "order_type": "STOP-ENTRY",
                "units": 999,
                "entry": 1.1003,
                "tp": 1.1025,
                "sl": 1.0980,
                "market_context": {"method": "BREAKOUT_FAILURE"},
                "metadata": metadata,
            },
            "risk_metrics": {
                "entry_price": 1.1003,
                "loss_pips": 23.0,
                "reward_pips": 22.0,
                "risk_jpy": 229.77,
                "reward_jpy": 219.78,
                "reward_risk": 22.0 / 23.0,
                "spread_pips": 2.0,
                "jpy_per_pip": 9.99,
                "estimated_margin_jpy": 999.0,
            },
        }
    ]
    paths["intents"].write_text(json.dumps(intents))
    _reprepare(paths)


def _reprepare(paths: dict[str, Path]) -> None:
    baseline = json.loads(paths["baseline"].read_text())
    baseline.pop("decision_provenance", None)
    paths["baseline"].write_text(json.dumps(baseline))
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=_sources(paths),
        now=NOW,
    )


def _guardian_action_receipt(
    *,
    pair: str = "AUD_CAD",
    action: str = "NO_ACTION",
    event_type: str = "TECHNICAL_STATE_CHANGE",
    severity: str = "P2",
    event_id: str = "guardian-event-1",
    thesis_state: str = "WOUNDED",
) -> dict:
    dedupe_key = f"{pair}|fixture|{event_type}|{action}"
    selected_event = {
        "event_id": event_id,
        "dedupe_key": dedupe_key,
        "pair": pair,
        "event_type": event_type,
        "severity": severity,
        "action_hint": action,
        "direction": None,
        "thesis_state": thesis_state,
        "recommended_review_type": "TUNING_REVIEW",
        "details": {
            "material_fingerprint": {
                "dominant_regime": "UNCLEAR",
                "volatility_bucket": "NORMAL",
            }
        },
    }
    return {
        "status": "ACCEPTED",
        "source": "guardian_wake_dispatcher",
        "model": "gpt-5.5",
        "receipt_status": "ACCEPTED",
        "receipt_lifecycle": "ACTIVE",
        "consumed_by_trader": False,
        "dispatcher_status": "RECEIPT_WRITTEN",
        "generated_at_utc": NOW.isoformat(),
        "expires_at_utc": (NOW + timedelta(hours=1)).isoformat(),
        "selected_event_id": event_id,
        "selected_event_dedupe_key": dedupe_key,
        "gateway_required": True,
        "no_direct_oanda": True,
        "selected_event": selected_event,
        "event": dict(selected_event),
        "receipt": {
            "action": action,
            "event_id": event_id,
            "dedupe_key": dedupe_key,
            "pair": pair,
            "side": "NONE",
            "thesis_state": thesis_state,
            "new_information": True,
            "ownership": "SYSTEM",
            "reason": "synthetic technical review",
            "invalidation": "technical invalidation",
            "harvest_trigger": "technical target",
            "margin_state": "NORMAL",
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


def _p1_margin_action_receipt() -> dict:
    receipt = _guardian_action_receipt(
        pair="PORTFOLIO",
        action="HOLD",
        event_type="MARGIN_PRESSURE",
        severity="P1",
        event_id="guardian-margin-p1",
        thesis_state="WOUNDED",
    )
    details = {
        "nav_jpy": 100_000.0,
        "margin_used_jpy": 92_000.0,
        "margin_available_jpy": 8_000.0,
        "max_margin_utilization_pct": 95.0,
        "fresh_entry_risk_block_active": False,
        "fresh_entry_risk_block_reason": "MARGIN_PRESSURE",
        "fresh_entry_risk_observation_only": True,
        "fresh_entry_margin_contract": "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
    }
    receipt["selected_event"]["details"] = dict(details)
    receipt["event"] = json.loads(json.dumps(receipt["selected_event"]))
    receipt["receipt"]["margin_state"] = (
        "margin_used/nav=0.920; available/nav=0.080; cap=0.950"
    )
    return receipt


def _watchdog() -> dict:
    weekend = {
        "active": True,
        "automation_status": "PAUSED",
        "exists": True,
        "in_weekend_pause_window": True,
        "mode": "paused",
        "now_jst": "2026-07-11T12:00:00+09:00",
        "qr_trader_managed": True,
        "reason": "weekend guard",
    }
    return {
        "generated_at_utc": NOW.isoformat(),
        "status": "OK",
        "runtime_status": "OK",
        "issue_status": "OK",
        "overall_status": "OK",
        "severity": "OK",
        "issues": [],
        "missed_expected_window": False,
        "minutes_since_last_run": 60.0,
        "last_trader_run_at": (NOW - timedelta(hours=1)).isoformat(),
        "last_trader_run_source": "trader_journal.ts",
        "automation_config": {
            "exists": True,
            "issues": [],
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "cadence_minutes": 60,
            "cwd": "/runtime",
            "cwds": ["/runtime"],
            "status": "PAUSED",
            "weekend_pause": dict(weekend),
        },
        "weekend_pause": dict(weekend),
        "guardian_receipt": {
            "action": None,
            "active": False,
            "exists": False,
            "issues": [],
            "receipt_summaries": [],
            "review_excerpt": "Generated at the original observation clock",
        },
        "execution_boundary": {
            "broker_writes_enabled": False,
            "no_live_side_effects": True,
            "read_only": True,
        },
        "environment": {"QR_TRADER_WATCHDOG_CAN_WAKE": "0"},
        "codex_logs": {
            "available": True,
            "queried_at_utc": NOW.isoformat(),
            "entries": [],
        },
    }


def _baseline(*, action: str = "TRADE", lane_ids: list[str] | None = None) -> dict:
    selected = [LANE_ID] if lane_ids is None else lane_ids
    return {
        "generated_at_utc": NOW.isoformat(),
        "market_read_first": _market_read(),
        "action": action,
        "selected_lane_id": selected[0] if selected else None,
        "selected_lane_ids": selected,
        "cancel_order_ids": ["old-pending-1"],
        "confidence": "HIGH",
        "method": "TREND_CONTINUATION",
        "risk_notes": ["deterministic units and stops are immutable"],
        "evidence_refs": [f"intent:{LANE_ID}", "broker:snapshot"],
        "operator_summary": "deterministic baseline",
    }


def _market_read() -> dict:
    return {
        "naked_read": {
            "currency_bought": "EUR",
            "currency_sold": "USD",
            "cleanest_pair_expression": "EUR_USD",
            "is_cleanest_currency_theme": "YES",
            "location_24h": "MIDDLE",
            "h1_h4_alignment": "H1/H4 aligned long",
            "tape_state": "TREND",
            "known_winning_trade_shape_match": "partial match",
            "proposed_building_style_allowed": "single entry only",
            "thesis_state": "ALIVE",
            "what_price_is_trying_to_do_now": "break the 1.1015 shelf",
        },
        "next_30m_prediction": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "expected_path": "hold 1.1000 then test 1.1020",
            "target_zone": "1.1020 to 1.1030",
            "invalidation": "1.0985",
        },
        "next_2h_prediction": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "expected_path": "extend through 1.1030",
            "target_zone": "1.1040 to 1.1050",
            "invalidation": "1.0975",
        },
        "best_trade_if_forced": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "vehicle": "MARKET",
            "entry": "1.1002",
            "tp": "1.1040",
            "sl": "1.0980",
            "why_this_pays": "target is above entry and invalidation is below it",
        },
    }


def _range_market_read() -> dict:
    read = _market_read()
    for key in ("next_30m_prediction", "next_2h_prediction"):
        read[key]["direction"] = "RANGE"
        read[key]["target_zone"] = "1.0995 to 1.1010"
        read[key]["invalidation"] = "1.0980 to 1.1025"
    return read


def _overlay(
    paths: dict[str, Path],
    *,
    disposition: str = "ACCEPT_BASELINE",
    authored_at: datetime = NOW,
    market_read: dict | None = None,
    prior_prediction_ids: list[str] | None = None,
    size_multiple: float = 0.75,
) -> dict:
    baseline = json.loads(paths["baseline"].read_text())
    packet = json.loads(paths["packet"].read_text())
    trade_allocation = (
        str(baseline.get("action") or "").upper() == "TRADE"
        and disposition == "ACCEPT_BASELINE"
    )
    base_units = int(
        ((packet.get("capital_allocation_board") or {}).get("selected_lane") or {}).get(
            "base_units"
        )
        or 0
    )
    return {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "author_kind": CODEX_MARKET_READ_AUTHOR,
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "authored_at_utc": authored_at.isoformat(),
        "baseline_sha256": canonical_json_sha256(baseline_core_payload(baseline)),
        "evidence_packet_sha256": packet["evidence_packet_sha256"],
        "baseline_disposition": disposition,
        "market_read_first": market_read or _market_read(),
        "market_read_review": {
            "prior_prediction_ids": prior_prediction_ids or [],
            "what_failed": "Reviewed the latest resolved path" if prior_prediction_ids else "NO_RESOLVED_PRIOR",
            "adjustment": "Use numeric quote-relative geometry and veto if the counterargument dominates.",
            "no_change_reason": "",
        },
        "market_read_counterargument": "The apparent breakout can fail back into the prior range.",
        "market_read_change_summary": "Rebuilt the directional path from the current broker quote.",
        "market_read_veto_reason": (
            "Current numeric forecast contradicts the deterministic entry trigger."
            if disposition.startswith("VETO_")
            else ""
        ),
        "capital_allocation": {
            "decision": "ALLOCATE" if trade_allocation else "NO_TRADE",
            "lane_id": baseline.get("selected_lane_id") if trade_allocation else None,
            "size_multiple": size_multiple if trade_allocation else 0.0,
            "selected_units": int(base_units * size_multiple) if trade_allocation else 0,
            "allocation_board_sha256": packet["capital_allocation_board_sha256"],
            "rationale": (
                "Direction-specific economic precision and exact-vehicle TP expectancy support bounded exposure."
                if trade_allocation
                else "No fresh entry capital is authorized for this receipt."
            ),
        },
    }


def _write_overlay(paths: dict[str, Path], **kwargs: object) -> None:
    paths["overlay"].write_text(json.dumps(_overlay(paths, **kwargs)))


def _apply(paths: dict[str, Path]):
    return apply_codex_market_read_overlay(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        overlay_path=paths["overlay"],
        output_path=paths["output"],
        evidence_sources=_sources(paths),
        now=NOW,
    )


def _v2_prediction(prediction_id: str, *, resolution_status: str = "RESOLVED_MID_CANDLE_DIAGNOSTIC") -> dict:
    result = {
        "resolution_status": resolution_status,
        "direction_status": "WRONG" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "target_completion_status": "NOT_TOUCHED" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "invalidation_status": "TOUCHED" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "first_touch_status": "INVALIDATION_FIRST" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "full_read_status": "INVALIDATION_FIRST" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
    }
    return {
        "schema_version": 2,
        "prediction_id": prediction_id,
        "generated_at_utc": (NOW - timedelta(hours=3)).isoformat(),
        "pair": "EUR_USD",
        "direction": "LONG",
        "action": "TRADE",
        "score_eligible": True,
        "source_snapshot_conflict": False,
        "verdict": "FULL_READ_INCOMPLETE" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "horizon_results": {"30m": result, "2h": {**result, "resolution_status": "UNRESOLVED"}},
    }


if __name__ == "__main__":
    unittest.main()
