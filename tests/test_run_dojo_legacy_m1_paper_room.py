from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_dojo_legacy_m1_paper_room",
    ROOT / "scripts/run-dojo-legacy-m1-paper-room.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M1PaperLauncherTests(unittest.TestCase):
    def test_registry_has_unique_rooms_owners_and_operations(self) -> None:
        path = (
            ROOT
            / "research/training/legacy-m1-signal-paper-20260729/paper_rooms.json"
        )
        registry = json.loads(path.read_text(encoding="utf-8"))
        for room in registry["rooms"]:
            loaded, selected = MODULE._load(path, room["room_id"])
            self.assertEqual(loaded["authority"]["order_authority"], "NONE")
            self.assertEqual(selected["bot_config"]["fixed_units"], 1000)

    def test_duplicate_operation_id_fails_closed(self) -> None:
        source = json.loads(
            (
                ROOT
                / "research/training/legacy-m1-signal-paper-20260729/paper_rooms.json"
            ).read_text(encoding="utf-8")
        )
        source["rooms"][1]["bot_config"]["operation_id"] = source["rooms"][0][
            "bot_config"
        ]["operation_id"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(SystemExit):
                MODULE._load(path, source["rooms"][0]["room_id"])


if __name__ == "__main__":
    unittest.main()
