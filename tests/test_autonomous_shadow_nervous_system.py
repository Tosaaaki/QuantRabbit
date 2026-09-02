from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.autonomous_shadow_nervous_system import (
    AutonomousShadowNervousSystem,
)
from quant_rabbit.cli import main


NOW = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
WORKERS = (
    "perception",
    "hypothesis",
    "critic",
    "admission",
    "fill_truth",
    "lifecycle",
    "exit",
    "learning",
)


def _decision(worker: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "decision_id": f"decision-{worker}",
        "worker": worker,
        "verdict": "ADVANCE",
        "reason": f"{worker} evidence passed its bounded contract",
        "observed_at_utc": NOW.isoformat(),
        "expires_at_utc": (NOW + timedelta(minutes=5)).isoformat(),
        "supporting_evidence": [f"sha256:{worker}"],
        "contradicting_evidence": [],
        "counterevidence_reviewed": True,
        "confidence": 0.90,
        "uncertainty": 0.10,
    }
    value.update(overrides)
    return value


def _packet(cycle_id: str = "cycle-001") -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "decisions": [_decision(worker) for worker in WORKERS],
        "human_assist": [],
        "kill_switch": False,
    }


class AutonomousShadowNervousSystemTest(unittest.TestCase):
    def _system(self, root: Path) -> AutonomousShadowNervousSystem:
        return AutonomousShadowNervousSystem(
            ledger_path=root / "synapses.jsonl",
            output_path=root / "state.json",
            report_path=root / "report.md",
        )

    def test_full_cycle_runs_without_human_approval_or_broker_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._system(root).run(_packet(), now_utc=NOW)
            events = [json.loads(line) for line in (root / "synapses.jsonl").read_text().splitlines()]
            state = json.loads((root / "state.json").read_text())

        self.assertEqual(summary.status, "COMPLETE")
        self.assertEqual(summary.state, "LEARNED")
        self.assertEqual(summary.events_appended, 8)
        self.assertIsNone(summary.expected_worker)
        self.assertFalse(summary.human_approval_required)
        self.assertFalse(summary.live_permission_allowed)
        self.assertEqual(
            [event["to_state"] for event in events],
            [
                "SIGNAL",
                "HYPOTHESIS",
                "CHALLENGED",
                "ADMITTED",
                "FILLED",
                "OPEN",
                "EXITED",
                "LEARNED",
            ],
        )
        self.assertTrue(all(event["execution_authority"] == "NONE" for event in events))
        self.assertTrue(all(event["manual_tagless_policy"] == "NO_TOUCH" for event in events))
        self.assertTrue(all(event["broker_mutation_allowed"] is False for event in events))
        self.assertTrue(all(event["external_order_attempts"] == 0 for event in events))
        self.assertFalse(state["human_approval_required"])
        self.assertTrue(state["human_assist_is_evidence_only"])

    def test_critic_must_review_counterevidence_before_admission(self) -> None:
        packet = _packet()
        packet["decisions"][2] = _decision("critic", counterevidence_reviewed=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._system(root).run(packet, now_utc=NOW)
            events = [json.loads(line) for line in (root / "synapses.jsonl").read_text().splitlines()]

        self.assertEqual(summary.status, "WAITING_FOR_CRITIC")
        self.assertEqual(summary.state, "HYPOTHESIS")
        self.assertEqual(summary.events_appended, 3)
        self.assertEqual(events[-1]["system_outcome"], "WAIT")
        self.assertIn("counterevidence review", events[-1]["reason"])

    def test_low_net_confidence_waits_instead_of_advancing(self) -> None:
        packet = _packet()
        packet["decisions"][0] = _decision("perception", confidence=0.60, uncertainty=0.10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._system(root).run(packet, now_utc=NOW)
            event = json.loads((root / "synapses.jsonl").read_text().splitlines()[0])

        self.assertEqual(summary.state, "IDLE")
        self.assertEqual(summary.status, "WAITING_FOR_PERCEPTION")
        self.assertEqual(event["system_outcome"], "WAIT")

    def test_repeated_wait_receipt_is_exactly_once(self) -> None:
        packet = _packet()
        packet["decisions"][0] = _decision("perception", verdict="WAIT")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = self._system(root)
            first = system.run(packet, now_utc=NOW)
            second = system.run(packet, now_utc=NOW + timedelta(seconds=1))
            events = (root / "synapses.jsonl").read_text().splitlines()

        self.assertEqual(first.events_appended, 1)
        self.assertEqual(second.events_appended, 0)
        self.assertEqual(len(events), 1)

    def test_reused_decision_identity_with_changed_content_fails_before_append(self) -> None:
        packet = _packet()
        packet["decisions"][0] = _decision("perception", verdict="WAIT")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = self._system(root)
            system.run(packet, now_utc=NOW)
            changed = _packet()
            changed["decisions"][0] = _decision(
                "perception",
                verdict="WAIT",
                reason="same identity but different material",
            )
            with self.assertRaisesRegex(ValueError, "decision identity conflict"):
                system.run(changed, now_utc=NOW + timedelta(seconds=1))
            events = (root / "synapses.jsonl").read_text().splitlines()

        self.assertEqual(len(events), 1)

    def test_order_shaped_fields_are_rejected(self) -> None:
        packet = _packet()
        packet["decisions"][0]["side"] = "LONG"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported fields: side"):
                self._system(Path(tmp)).run(packet, now_utc=NOW)

    def test_output_paths_cannot_alias_the_append_only_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared.json"
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                AutonomousShadowNervousSystem(
                    ledger_path=shared,
                    output_path=shared,
                    report_path=Path(tmp) / "report.md",
                )

    def test_human_assist_is_recorded_as_evidence_not_transition_authority(self) -> None:
        packet = _packet()
        packet["human_assist"] = [
            {
                "note": "Check the data-source outage annotation.",
                "evidence_refs": ["incident:feed-7"],
                "observed_at_utc": NOW.isoformat(),
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._system(root).run(packet, now_utc=NOW)
            first = json.loads((root / "synapses.jsonl").read_text().splitlines()[0])

        assist = first["human_assist"][0]
        self.assertEqual(assist["role"], "ASSIST")
        self.assertFalse(assist["can_approve_transition"])
        self.assertFalse(assist["can_grant_live_permission"])

    def test_completed_cycle_replay_is_idempotent_and_next_cycle_can_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = self._system(root)
            first = system.run(_packet(), now_utc=NOW)
            replay = system.run(_packet(), now_utc=NOW + timedelta(seconds=1))
            second = system.run(_packet("cycle-002"), now_utc=NOW + timedelta(seconds=2))
            events = (root / "synapses.jsonl").read_text().splitlines()

        self.assertEqual(first.events_appended, 8)
        self.assertEqual(replay.events_appended, 0)
        self.assertEqual(replay.status, "COMPLETE")
        self.assertEqual(second.events_appended, 8)
        self.assertEqual(second.status, "COMPLETE")
        self.assertEqual(len(events), 16)

    def test_kill_switch_halts_without_worker_or_broker_action(self) -> None:
        packet = _packet()
        packet["kill_switch"] = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._system(root).run(packet, now_utc=NOW)
            event = json.loads((root / "synapses.jsonl").read_text().splitlines()[0])

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(summary.events_appended, 1)
        self.assertEqual(event["worker"], "kill_switch")
        self.assertEqual(event["external_orders"], 0)

    def test_no_fill_routes_to_learning_without_fabricating_a_fill(self) -> None:
        packet = _packet()
        packet["decisions"] = [
            _decision(worker, verdict="NO_FILL" if worker == "fill_truth" else "ADVANCE")
            for worker in ("perception", "hypothesis", "critic", "admission", "fill_truth", "learning")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._system(root).run(packet, now_utc=NOW)
            states = [
                json.loads(line)["to_state"]
                for line in (root / "synapses.jsonl").read_text().splitlines()
            ]

        self.assertEqual(summary.status, "COMPLETE")
        self.assertEqual(states[-2:], ["UNFILLED", "LEARNED"])
        self.assertNotIn("FILLED", states)

    def test_tampered_ledger_fails_closed_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = self._system(root)
            system.run(_packet(), now_utc=NOW)
            ledger = root / "synapses.jsonl"
            rows = ledger.read_text().splitlines()
            event = json.loads(rows[0])
            event["live_permission"] = True
            rows[0] = json.dumps(event)
            ledger.write_text("\n".join(rows) + "\n")

            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                system.run(_packet("cycle-002"), now_utc=NOW + timedelta(seconds=1))
            self.assertEqual(len(ledger.read_text().splitlines()), 8)

    def test_cli_runs_cycle_from_json_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.json"
            cli_now = datetime.now(timezone.utc)
            packet = _packet()
            for decision in packet["decisions"]:
                decision["observed_at_utc"] = cli_now.isoformat()
                decision["expires_at_utc"] = (cli_now + timedelta(minutes=5)).isoformat()
            input_path.write_text(json.dumps(packet))
            code = main(
                [
                    "autonomous-shadow-cycle",
                    "--input",
                    str(input_path),
                    "--ledger",
                    str(root / "ledger.jsonl"),
                    "--output",
                    str(root / "state.json"),
                    "--report",
                    str(root / "report.md"),
                ]
            )
            state = json.loads((root / "state.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "COMPLETE")
        self.assertFalse(state["live_permission_allowed"])


if __name__ == "__main__":
    unittest.main()
