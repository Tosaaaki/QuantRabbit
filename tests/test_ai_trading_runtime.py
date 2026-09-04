from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.ai_trading_runtime import AIRuntimeError, accept_run, prepare_run
from quant_rabbit.entry_decision import compute_dynamic_units


NOW = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)


class FrozenGatewayDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = NOW + timedelta(seconds=2)
        return value if tz is None else value.astimezone(tz)


class AITradingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        for name in ("broker.json", "market.json", "news.json", "performance.json"):
            (self.root / "data" / name).write_text(json.dumps({"name": name}))
        self.evidence = self.root / "data" / "evidence.json"
        self.evidence.write_text(json.dumps(_evidence_packet()))
        self.state = self.root / "state"
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps({
            "version": 2,
            "state_root": str(self.state),
            "profiles": {
                "intraday": {
                    "kind": "trade",
                    "decision_max_age_seconds": 900,
                    "sink": "paper_ledger",
                    "allowed_actions": ["ENTER", "WAIT", "REQUEST_EVIDENCE"],
                    "workers": {
                        "evidence": [
                            {"path": "data/evidence.json", "required": True, "max_age_seconds": 10**9},
                        ]
                    },
                },
                "strategic": {
                    "kind": "review",
                    "decision_max_age_seconds": 3600,
                    "sink": "review_overlay",
                    "workers": {
                        "performance": [
                            {"path": "data/performance.json", "required": True, "max_age_seconds": 10**9}
                        ]
                    },
                },
            },
        }))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_prepare_and_accept_full_ai_trade(self) -> None:
        prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        self.assertTrue(prepared.ready)
        manifest = json.loads(prepared.manifest_path.read_text())
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update({
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "decided_at_utc": (NOW + timedelta(seconds=5)).isoformat(),
            "thesis": "EUR strength and a clean invalidation support a bounded paper entry.",
            "evidence_refs": ["evidence:data/evidence.json"],
            "action": "ENTER",
            "confidence": 0.72,
            "orders": [{
                "decision_id": "order-1",
                "pair": "EUR_USD",
                "side": "LONG",
                "method": "TREND_CONTINUATION",
                "vehicle": "LIMIT",
                "order_type": "LIMIT",
                "entry": 1.10,
                "take_profit": 1.12,
                "stop_loss": 1.09,
                "units": 600,
                "sizing_receipt": _sizing_receipt(),
                "evidence_binding": _evidence_binding(),
                "net_edge_proof": {"expected_net_after_costs": 1.0, "source": "sealed-evidence"},
                "cost_proof": {"spread": 0.0002, "source": "sealed-evidence"},
                "rationale": "Defined reward and invalidation.",
                "extensions": {},
            }],
            "position_actions": [],
            "requested_evidence": [],
        })
        prepared.candidate_path.write_text(json.dumps(candidate))
        result = accept_run(
            config_path=self.config,
            manifest_path=prepared.manifest_path,
            candidate_path=prepared.candidate_path,
            repo_root=self.root,
            now=NOW + timedelta(seconds=10),
        )
        self.assertEqual(result.status, "ACCEPTED_PAPER")
        receipt = json.loads(result.receipt_path.read_text())
        self.assertFalse(receipt["execution"]["broker_mutation_allowed"])
        self.assertEqual(receipt["decision"]["orders"][0]["units"], 600)
        self.assertTrue(receipt["entry_decision"]["decision_id"].startswith("qre_"))
        self.assertTrue(receipt["adjudication"]["adjudication_id"].startswith("qra_"))
        self.assertTrue((self.state / "decisions.jsonl").exists())

    def test_trade_geometry_is_rejected(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update({
            "model": "any-model", "reasoning_effort": "high",
            "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
            "thesis": "test", "evidence_refs": ["evidence:data/evidence.json"], "action": "ENTER",
            "confidence": 0.5, "position_actions": [], "requested_evidence": [],
            "orders": [{
                "decision_id": "bad", "pair": "EUR_USD", "side": "LONG",
                "method": "TREND_CONTINUATION", "vehicle": "MARKET",
                "order_type": "MARKET", "entry": 1.1, "take_profit": 1.09,
                "stop_loss": 1.08, "units": 600, "sizing_receipt": _sizing_receipt(),
                "evidence_binding": _evidence_binding(),
                "net_edge_proof": {"expected_net_after_costs": 1.0},
                "cost_proof": {"spread": 0.0002},
                "rationale": "bad", "extensions": {},
            }],
        })
        prepared.candidate_path.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(AIRuntimeError, "LONG requires"):
            accept_run(
                config_path=self.config, manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path, repo_root=self.root,
                now=NOW + timedelta(seconds=2),
            )

    def test_live_sink_wait_never_calls_broker_gateway(self) -> None:
        config = json.loads(self.config.read_text())
        config["profiles"]["intraday"]["sink"] = "live_gateway"
        self.config.write_text(json.dumps(config))
        prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update(
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
                "thesis": "No current edge warrants new exposure.",
                "evidence_refs": ["evidence:data/evidence.json"],
                "action": "WAIT",
                "confidence": 0.6,
                "orders": [],
                "position_actions": [],
                "requested_evidence": [],
            }
        )
        prepared.candidate_path.write_text(json.dumps(candidate))
        with patch("quant_rabbit.ai_live_gateway.datetime", FrozenGatewayDateTime):
            result = accept_run(
                config_path=self.config,
                manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path,
                repo_root=self.root,
                now=NOW + timedelta(seconds=2),
            )
        receipt = json.loads(result.receipt_path.read_text())
        self.assertEqual(result.status, "ACCEPTED_NO_BROKER_ACTION")
        self.assertEqual(receipt["execution"]["status"], "NO_BROKER_ACTION")
        self.assertEqual(receipt["execution"]["broker_order_posts"], 0)

    def test_long_ai_thesis_is_bound_by_digest_not_duplicated(self) -> None:
        prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        candidate = json.loads(prepared.candidate_path.read_text())
        thesis = "Detailed evidence supports waiting. " * 40
        candidate.update(
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
                "thesis": thesis,
                "evidence_refs": ["evidence:data/evidence.json"],
                "action": "WAIT",
                "confidence": 0.6,
                "orders": [],
                "position_actions": [],
                "requested_evidence": [],
            }
        )
        candidate_sha256 = hashlib.sha256(
            json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        prepared.candidate_path.write_text(json.dumps(candidate))

        result = accept_run(
            config_path=self.config,
            manifest_path=prepared.manifest_path,
            candidate_path=prepared.candidate_path,
            repo_root=self.root,
            now=NOW + timedelta(seconds=2),
        )

        receipt = json.loads(result.receipt_path.read_text())
        self.assertEqual(receipt["decision"]["thesis"], thesis)
        self.assertEqual(
            receipt["entry_decision"]["reasons"],
            [f"ai_candidate_sha256:{candidate_sha256}"],
        )

    def test_profile_action_allowlist_rejects_exit_before_live_sink(self) -> None:
        config = json.loads(self.config.read_text())
        config["profiles"]["intraday"]["sink"] = "live_gateway"
        self.config.write_text(json.dumps(config))
        prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        manifest = json.loads(prepared.manifest_path.read_text())
        self.assertEqual(
            manifest["candidate_schema"]["actions"],
            ["ENTER", "WAIT", "REQUEST_EVIDENCE"],
        )
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update(
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
                "thesis": "exit attempt must be rejected before sink dispatch",
                "evidence_refs": ["evidence:data/evidence.json"],
                "action": "EXIT",
                "confidence": 0.8,
                "orders": [],
                "position_actions": [
                    {
                        "action": "CLOSE_ALL",
                        "trade_id": "system-trade-1",
                        "instrument": "EUR_USD",
                        "position_revision": "rev-1",
                        "owner_binding": {"owner_kind": "AI_SYSTEM"},
                        "reason": "test",
                    }
                ],
                "requested_evidence": [],
            }
        )
        prepared.candidate_path.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(AIRuntimeError, "not enabled"):
            accept_run(
                config_path=self.config,
                manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path,
                repo_root=self.root,
                now=NOW + timedelta(seconds=2),
            )
        self.assertFalse((self.state / "hotpath_lease.json").exists())

    def test_evidence_change_rejects_candidate(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        self.evidence.write_text('{"changed": true}')
        with self.assertRaisesRegex(AIRuntimeError, "changed after"):
            accept_run(
                config_path=self.config, manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path, repo_root=self.root,
                now=NOW + timedelta(seconds=1),
            )

    def test_expiring_optional_source_is_ignored_for_entire_decision_window(self) -> None:
        optional = self.root / "data" / "optional.json"
        optional.write_text(json.dumps({"review": "near expiry"}))
        os.utime(optional, (NOW.timestamp() - 95, NOW.timestamp() - 95))
        config = json.loads(self.config.read_text())
        config["profiles"]["intraday"]["decision_max_age_seconds"] = 10
        config["profiles"]["intraday"]["workers"]["strategy"] = [
            {"path": "data/optional.json", "required": False, "max_age_seconds": 100}
        ]
        self.config.write_text(json.dumps(config))

        prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        manifest = json.loads(prepared.manifest_path.read_text())
        optional_source = next(row for row in manifest["sources"] if row["worker"] == "strategy")
        self.assertEqual(optional_source["status"], "IGNORED")
        self.assertIsNone(optional_source["sha256"])

        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update(
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
                "thesis": "The optional review is excluded because it cannot stay fresh.",
                "evidence_refs": ["evidence:data/evidence.json"],
                "action": "WAIT",
                "confidence": 0.6,
                "orders": [],
                "position_actions": [],
                "requested_evidence": [],
            }
        )
        prepared.candidate_path.write_text(json.dumps(candidate))

        result = accept_run(
            config_path=self.config,
            manifest_path=prepared.manifest_path,
            candidate_path=prepared.candidate_path,
            repo_root=self.root,
            now=NOW + timedelta(seconds=10),
        )
        self.assertEqual(result.status, "ACCEPTED_PAPER")

    def test_missing_required_source_blocks_prepare(self) -> None:
        self.evidence.unlink()
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        self.assertFalse(prepared.ready)
        self.assertIn("MISSING", prepared.blockers[0])

    def test_manifest_cannot_replace_configured_sink(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        manifest = json.loads(prepared.manifest_path.read_text())
        manifest["sink"] = "review_overlay"
        prepared.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AIRuntimeError, "kind or sink"):
            accept_run(
                config_path=self.config, manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path, repo_root=self.root,
                now=NOW + timedelta(seconds=1),
            )

    def test_strategic_review_updates_overlay(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="strategic", repo_root=self.root, now=NOW)
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update({
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "decided_at_utc": (NOW + timedelta(seconds=2)).isoformat(),
            "thesis": "Range regime warrants smaller exposure and stricter confirmation.",
            "evidence_refs": ["performance:data/performance.json"],
            "regime": "RANGE",
            "risk_posture": "CAUTIOUS",
            "valid_until_utc": (NOW + timedelta(hours=3)).isoformat(),
            "themes": ["mean reversion"],
            "instructions": ["prefer limit entries"],
        })
        prepared.candidate_path.write_text(json.dumps(candidate))
        result = accept_run(
            config_path=self.config, manifest_path=prepared.manifest_path,
            candidate_path=prepared.candidate_path, repo_root=self.root,
            now=NOW + timedelta(seconds=3),
        )
        self.assertEqual(result.status, "ACCEPTED_REVIEW")
        overlay = json.loads((self.state / "strategic_review.json").read_text())
        self.assertEqual(overlay["decision"]["risk_posture"], "CAUTIOUS")


def _sizing_receipt() -> dict[str, object]:
    return compute_dynamic_units(
        daily_remaining=1000,
        portfolio_allowance=900,
        nav_risk_ceiling=800,
        calibration_factor=0.75,
        drawdown_factor=1.0,
        correlation_factor=1.0,
        net_edge_factor=1.0,
        loss_per_unit_at_stop=1.0,
        margin_max_units=10_000,
        correlation_max_units=10_000,
        broker_max_units=10_000,
    )


def _evidence_binding() -> dict[str, str]:
    packet = _evidence_packet()
    return {
        "packet_sha256": str(packet["packet_sha256"]),
        "source_set_sha256": str(packet["source_set_sha256"]),
        "broker_epoch": "9001",
    }


def _evidence_packet() -> dict[str, object]:
    body: dict[str, object] = {
        "contract": "QR_AI_EVIDENCE_PACKET_V1",
        "schema_version": 1,
        "status": "READY",
        "evidence_as_of_utc": NOW.isoformat(),
        "source_set_sha256": "sources-1",
        "broker_epoch": {"as_of_utc": NOW.isoformat(), "last_transaction_id": "9001"},
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "packet_sha256": hashlib.sha256(raw.encode()).hexdigest()}


if __name__ == "__main__":
    unittest.main()
