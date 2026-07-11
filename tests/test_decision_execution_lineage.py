from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quant_rabbit.decision_execution_lineage as lineage_module
from quant_rabbit.broker.execution import _client_order_id
from quant_rabbit.decision_execution_lineage import (
    DecisionExecutionLineage,
    DecisionExecutionLineageError,
    append_execution_link,
    broker_identifiers_from_gateway_response,
    build_execution_link,
    decision_lineage_from_verified_payload,
    read_execution_links,
)
from quant_rabbit.models import OrderIntent, OrderType, Side
from tests.test_gpt_trader import LANE_ID, _brain, _fixtures, _trade_decision


def _lineage(seed: str = "a") -> DecisionExecutionLineage:
    decision_id = "gptd:" + seed * 64
    prediction_id = "mr2:" + chr(ord(seed) + 1) * 64
    token = "mdl-" + hashlib.sha256(
        f"{decision_id}\0{prediction_id}".encode("utf-8")
    ).hexdigest()[: lineage_module.BROKER_LINEAGE_TOKEN_HEX_LENGTH]
    return DecisionExecutionLineage(
        decision_receipt_id=decision_id,
        market_read_prediction_id=prediction_id,
        lineage_token=token,
        decision_generated_at_utc="2026-07-11T00:00:00Z",
    )


def _response(*, order_id: str = "101", fill_id: str = "102", trade_id: str = "200") -> dict:
    return {
        "orderCreateTransaction": {"id": order_id},
        "orderFillTransaction": {
            "id": fill_id,
            "orderID": order_id,
            "tradeOpened": {"tradeID": trade_id},
        },
        "relatedTransactionIDs": [order_id, fill_id],
        "lastTransactionID": fill_id,
    }


class DecisionExecutionLineageTest(unittest.TestCase):
    def test_recomputes_gptd_from_actual_gpt_trader_brain_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            lineage = decision_lineage_from_verified_payload(
                payload,
                selected_lane_id=LANE_ID,
            )
            self.assertIsNotNone(lineage)
            assert lineage is not None
            recorded = payload["market_read_prediction"]
            self.assertEqual(lineage.decision_receipt_id, recorded["decision_receipt_id"])
            self.assertEqual(lineage.market_read_prediction_id, recorded["prediction_id"])

            tampered = json.loads(json.dumps(payload))
            tampered["decision"]["confidence"] = "LOW"
            with self.assertRaisesRegex(
                DecisionExecutionLineageError,
                "no longer matches",
            ):
                decision_lineage_from_verified_payload(
                    tampered,
                    selected_lane_id=LANE_ID,
                )

    def test_actual_gateway_ids_append_and_exact_duplicate_coalesces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market_read_execution_links.jsonl"
            lineage = _lineage()
            response = _response()
            link = build_execution_link(
                lineage=lineage,
                gateway_response=response,
                lane_id=LANE_ID,
                parent_lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="c" * 64,
                client_extension_id=f"qrv1-EURUSD-L-test-{lineage.lineage_token}",
                recorded_at_utc="2026-07-11T00:01:00Z",
            )

            first = append_execution_link(path, link)
            duplicate = append_execution_link(path, link)
            rows = read_execution_links(path)

            self.assertEqual(first["status"], "RECORDED")
            self.assertEqual(duplicate["status"], "COALESCED_EXACT_DUPLICATE")
            self.assertEqual(len(rows), 1)
            broker_ids = rows[0]["broker_ids"]
            self.assertEqual(broker_ids["order_ids"], ["101"])
            self.assertEqual(broker_ids["fill_transaction_ids"], ["102"])
            self.assertEqual(broker_ids["trade_ids"], ["200"])
            self.assertFalse(rows[0]["pair_or_time_inference_used"])
            self.assertNotIn("pair", broker_identifiers_from_gateway_response(response))

    def test_rejects_response_without_explicit_broker_ids(self) -> None:
        with self.assertRaisesRegex(DecisionExecutionLineageError, "no explicit broker"):
            build_execution_link(
                lineage=_lineage(),
                gateway_response={"instrument": "EUR_USD", "time": "2026-07-11T00:00:00Z"},
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id=None,
                claim_id=None,
                order_request_sha256=None,
                client_extension_id=None,
            )

    def test_malformed_or_mutated_append_only_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionExecutionLineageError, "malformed"):
                read_execution_links(path)

            lineage = _lineage()
            valid = build_execution_link(
                lineage=lineage,
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id=None,
                claim_id=None,
                order_request_sha256="d" * 64,
                client_extension_id=f"qrv1-test-{lineage.lineage_token}",
            )
            mutated = json.loads(json.dumps(valid))
            mutated["broker_ids"]["trade_ids"] = ["different-trade"]
            path.write_text(json.dumps(mutated) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionExecutionLineageError, "content digest"):
                read_execution_links(path)

    def test_append_checks_current_size_plus_new_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.jsonl"
            first = build_execution_link(
                lineage=_lineage("a"),
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="d" * 64,
                client_extension_id="qrv1-first",
            )
            append_execution_link(path, first)
            second = build_execution_link(
                lineage=_lineage("c"),
                gateway_response=_response(order_id="201", fill_id="202", trade_id="300"),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-2",
                claim_id="claim-2",
                order_request_sha256="e" * 64,
                client_extension_id="qrv1-second",
            )
            second_bytes = len(
                (json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            limit = path.stat().st_size + second_bytes - 1
            with patch.object(lineage_module, "MAX_EXECUTION_LINK_FILE_BYTES", limit):
                with self.assertRaisesRegex(DecisionExecutionLineageError, "bounded file size"):
                    append_execution_link(path, second)
            self.assertEqual(len(read_execution_links(path)), 1)

    def test_same_broker_id_cannot_belong_to_different_lineage_owners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.jsonl"
            first = build_execution_link(
                lineage=_lineage("a"),
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="d" * 64,
                client_extension_id="qrv1-first",
            )
            conflicting = build_execution_link(
                lineage=_lineage("c"),
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-2",
                claim_id="claim-2",
                order_request_sha256="e" * 64,
                client_extension_id="qrv1-second",
            )
            append_execution_link(path, first)

            with self.assertRaisesRegex(
                DecisionExecutionLineageError,
                "conflicting lineage owners",
            ):
                append_execution_link(path, conflicting)
            self.assertEqual(read_execution_links(path), [first])

            # Read-time validation also fails closed if conflicting bytes were
            # introduced outside the append helper.
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(conflicting) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DecisionExecutionLineageError,
                "conflicting lineage owners",
            ):
                read_execution_links(path)

    def test_same_owner_may_repeat_broker_id_for_later_exact_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.jsonl"
            lineage = _lineage("a")
            first = build_execution_link(
                lineage=lineage,
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="d" * 64,
                client_extension_id="qrv1-first",
            )
            repeated = build_execution_link(
                lineage=lineage,
                gateway_response=_response(),
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-1",
                claim_id="claim-enriched",
                order_request_sha256="d" * 64,
                client_extension_id="qrv1-first",
            )

            append_execution_link(path, first)
            append_execution_link(path, repeated)

            self.assertEqual(len(read_execution_links(path)), 2)

    def test_different_owners_may_share_account_last_transaction_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.jsonl"
            first_response = _response(order_id="101", fill_id="102", trade_id="200")
            second_response = _response(order_id="201", fill_id="202", trade_id="300")
            first_response["lastTransactionID"] = "account-watermark-999"
            second_response["lastTransactionID"] = "account-watermark-999"
            first = build_execution_link(
                lineage=_lineage("a"),
                gateway_response=first_response,
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="d" * 64,
                client_extension_id="qrv1-first",
            )
            second = build_execution_link(
                lineage=_lineage("c"),
                gateway_response=second_response,
                lane_id=LANE_ID,
                parent_lane_id=None,
                forecast_cycle_id="forecast-2",
                claim_id="claim-2",
                order_request_sha256="e" * 64,
                client_extension_id="qrv1-second",
            )

            append_execution_link(path, first)
            append_execution_link(path, second)

            self.assertEqual(len(read_execution_links(path)), 2)

    def test_client_extension_reserves_lineage_suffix_inside_128_chars(self) -> None:
        lineage = _lineage()
        intent = OrderIntent(
            pair="X" * 110,
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1,
            entry=None,
            tp=2.0,
            sl=1.0,
            thesis="lineage suffix bound",
            metadata={
                "lane_id": "lane:" + "x" * 200,
                "gpt_decision_receipt_id": lineage.decision_receipt_id,
                "market_read_prediction_id": lineage.market_read_prediction_id,
                "gpt_decision_lineage_token": lineage.lineage_token,
            },
        )

        client_id = _client_order_id(intent)

        self.assertEqual(len(client_id), 128)
        self.assertTrue(client_id.endswith("-" + lineage.lineage_token))


if __name__ == "__main__":
    unittest.main()
