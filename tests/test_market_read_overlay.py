from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.market_read_overlay import (
    CODEX_MARKET_READ_AUTHOR,
    MarketReadOverlayError,
    apply_codex_market_read_overlay,
    baseline_core_payload,
    canonical_json_sha256,
    prepare_market_read_baseline,
)


NOW = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"


class MarketReadOverlayTest(unittest.TestCase):
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
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
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
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
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

    def test_overlay_rejects_any_execution_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            overlay["action"] = "TRADE"
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_OVERLAY_SCHEMA_INVALID"):
                _apply(paths)

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
    }
    paths["baseline"].write_text(json.dumps(baseline or _baseline()))
    paths["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": snapshot_at.isoformat(),
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
    paths["intents"].write_text(json.dumps({"results": [{"lane_id": LANE_ID, "units": 1200}]}))
    if not paths["predictions"].exists():
        paths["predictions"].write_text("")
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=_sources(paths),
        now=NOW,
    )
    return paths


def _sources(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "broker_snapshot": paths["snapshot"],
        "order_intents": paths["intents"],
        "market_read_predictions": paths["predictions"],
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
            "vehicle": "STOP",
            "entry": "1.1005",
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
) -> dict:
    baseline = json.loads(paths["baseline"].read_text())
    packet = json.loads(paths["packet"].read_text())
    return {
        "schema_version": 1,
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
