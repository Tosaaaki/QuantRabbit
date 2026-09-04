from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.ai_trading_runtime import AIRuntimeError, accept_run, prepare_run


NOW = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)


class AITradingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        for name in ("broker.json", "market.json", "news.json", "performance.json"):
            (self.root / "data" / name).write_text(json.dumps({"name": name}))
        self.state = self.root / "state"
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps({
            "version": 1,
            "state_root": str(self.state),
            "profiles": {
                "intraday": {
                    "kind": "trade",
                    "decision_max_age_seconds": 900,
                    "sink": "paper_ledger",
                    "workers": {
                        "market": [
                            {"path": "data/broker.json", "required": True, "max_age_seconds": 10**9},
                            {"path": "data/market.json", "required": True, "max_age_seconds": 10**9},
                            {"path": "data/news.json", "required": True, "max_age_seconds": 10**9},
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
            "evidence_refs": ["market:data/market.json", "market:data/broker.json"],
            "action": "TRADE",
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
                "units": 1200,
                "allocation_multiplier": 0.75,
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
        self.assertEqual(receipt["decision"]["orders"][0]["units"], 1200)
        self.assertTrue((self.state / "decisions.jsonl").exists())

    def test_trade_geometry_is_rejected(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        candidate = json.loads(prepared.candidate_path.read_text())
        candidate.update({
            "model": "any-model", "reasoning_effort": "high",
            "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
            "thesis": "test", "evidence_refs": ["market:data/market.json"], "action": "TRADE",
            "confidence": 0.5, "position_actions": [], "requested_evidence": [],
            "orders": [{
                "decision_id": "bad", "pair": "EUR_USD", "side": "LONG",
                "method": "TREND_CONTINUATION", "vehicle": "MARKET",
                "order_type": "MARKET", "entry": 1.1, "take_profit": 1.09,
                "stop_loss": 1.08, "units": 1, "allocation_multiplier": 1.0,
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
                "evidence_refs": ["market:data/market.json"],
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
            now=NOW + timedelta(seconds=2),
        )
        receipt = json.loads(result.receipt_path.read_text())
        self.assertEqual(result.status, "ACCEPTED_NO_BROKER_ACTION")
        self.assertEqual(receipt["execution"]["status"], "NO_BROKER_ACTION")
        self.assertEqual(receipt["execution"]["broker_order_posts"], 0)

    def test_evidence_change_rejects_candidate(self) -> None:
        prepared = prepare_run(config_path=self.config, profile="intraday", repo_root=self.root, now=NOW)
        (self.root / "data" / "market.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(AIRuntimeError, "changed after"):
            accept_run(
                config_path=self.config, manifest_path=prepared.manifest_path,
                candidate_path=prepared.candidate_path, repo_root=self.root,
                now=NOW + timedelta(seconds=1),
            )

    def test_missing_required_source_blocks_prepare(self) -> None:
        (self.root / "data" / "news.json").unlink()
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


if __name__ == "__main__":
    unittest.main()
