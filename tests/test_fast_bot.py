from __future__ import annotations

import json
import os
import fcntl
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.fast_bot import (
    AI_SUPERVISION_CONTRACT,
    ENTRY_ARM_SPREAD_FRACTIONS,
    ENTRY_EXPERIMENT_CONTRACT,
    EPISODE_HANDOFF_CONTRACT,
    HORIZON_LANE,
    LEGACY_EPISODE_HANDOFF_CONTRACT,
    REGIME_CONTRACT,
    _append_signals_once,
    _entry_experiment_arms,
    _write_text_atomic,
    build_fast_bot_shadow,
    build_hierarchical_regime_contract,
    load_fast_bot_episode_handoff,
    run_fast_bot_shadow,
)
from quant_rabbit.guardian_observation import CURRENT_M1_CONTRACT
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}
EPISODE_WORKER = Path(__file__).resolve().parents[1] / "scripts" / "launch-fast-bot-episode-worker.py"
EPISODE_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run-fast-bot-episode-shadow.py"
EPISODE_OUTCOME_RESOLVER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "resolve-fast-bot-episode-outcomes.py"
)


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


def _episode_worker_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env["QR_LIVE_ENABLED"] = "0"
    env["QR_AUTOTRADE_LOCK_HELD"] = "0"
    env.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
    env.update(overrides)
    return env


def _run_episode_worker_cli(*args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EPISODE_WORKER), *(str(arg) for arg in args)],
        cwd=EPISODE_WORKER.parents[1],
        env=env or _episode_worker_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )


def _load_episode_worker_module():
    spec = importlib.util.spec_from_file_location("quant_rabbit_episode_worker_test", EPISODE_WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("episode worker module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_episode_runner_module():
    spec = importlib.util.spec_from_file_location(
        "quant_rabbit_episode_runner_test",
        EPISODE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("episode runner module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_episode_outcome_resolver_module():
    spec = importlib.util.spec_from_file_location(
        "quant_rabbit_episode_outcome_resolver_test",
        EPISODE_OUTCOME_RESOLVER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("episode outcome resolver module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode_destinations(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "episode.json",
        root / "episode.jsonl",
        root / "sources",
    )


def _episode_destination_args(root: Path) -> tuple[object, ...]:
    output, ledger, source_archive = _episode_destinations(root)
    return (
        "--output",
        output,
        "--ledger",
        ledger,
        "--source-archive",
        source_archive,
    )


def _ensure_test_spool_owner(root: Path, spool: Path) -> str:
    worker_module = _load_episode_worker_module()
    output, ledger, source_archive = _episode_destinations(root)
    return worker_module._ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )


def _write_episode_handoff(
    root: Path,
    spool: Path,
    *,
    now: datetime,
    suffix: str,
) -> Path:
    fast, slow, snapshot = _inputs()
    inputs = {
        "fast": fast,
        "slow": slow,
        "snapshot": snapshot,
        "events": {"events": []},
    }
    paths: dict[str, Path] = {}
    for name, payload in inputs.items():
        path = root / f"{suffix}-{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    spool.mkdir(parents=True, exist_ok=True)
    owner_id = _ensure_test_spool_owner(root, spool)
    handoff = spool / (
        f".handoff-{os.getpid()}-{owner_id}-{suffix}.tmp"
    )
    run_fast_bot_shadow(
        fast_pair_charts_path=paths["fast"],
        slow_pair_charts_path=paths["slow"],
        broker_snapshot_path=paths["snapshot"],
        guardian_events_path=paths["events"],
        ai_supervision_path=None,
        regime_output_path=root / f"{suffix}-regime.json",
        shadow_output_path=root / f"{suffix}-shadow.json",
        shadow_ledger_path=root / f"{suffix}-shadow.jsonl",
        report_path=root / f"{suffix}-report.md",
        now_utc=now,
        episode_handoff_path=handoff,
    )
    return handoff


class FastBotTest(unittest.TestCase):
    def test_atomic_text_failure_removes_private_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "state.json"
            atomic_temp = root / f".state.json.{os.getpid()}.tmp"
            with patch("quant_rabbit.fast_bot.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    _write_text_atomic(target, "sealed\n")
            self.assertFalse(target.exists())
            self.assertFalse(atomic_temp.exists())

    def test_atomic_text_fsyncs_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "state.json"
            with patch("quant_rabbit.fast_bot.os.fsync", wraps=os.fsync) as fsync:
                _write_text_atomic(target, "sealed\n")
            self.assertEqual(target.read_text(), "sealed\n")
            self.assertGreaterEqual(fsync.call_count, 2)

    def test_episode_handoff_loader_rejects_unbounded_or_nonregular_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "size is invalid"):
                load_fast_bot_episode_handoff(oversized)

            target = root / "target.json"
            target.write_text("{}\n")
            symlink = root / "handoff-link.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "size is invalid"):
                load_fast_bot_episode_handoff(symlink)

            fifo = root / "handoff.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "size is invalid"):
                load_fast_bot_episode_handoff(fifo)

    def test_episode_spool_cleans_only_dead_pid_temps_and_enforces_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            dead_outer = spool / f".handoff-999999-{owner_id}-dead.tmp"
            live_outer = (
                spool / f".handoff-{os.getpid()}-{owner_id}-live.tmp"
            )
            dead_atomic = spool / ".state.json.999999.tmp"
            for path in (dead_outer, live_outer, dead_atomic):
                path.write_text("partial\n")

            available = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )

            self.assertEqual(available.returncode, 0, available.stderr)
            self.assertTrue(dead_outer.exists())
            self.assertFalse(dead_atomic.exists())
            self.assertTrue(live_outer.exists())

            recovered = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                "--output",
                Path(temp_dir) / "episode.json",
                "--ledger",
                Path(temp_dir) / "episode.jsonl",
                "--source-archive",
                root / "sources",
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertFalse(dead_outer.exists())
            self.assertTrue(live_outer.exists())

            for index in range(64):
                (
                    spool
                    / f"handoff-{index}-{owner_id}-999999-{index:016x}.json"
                ).write_text("{}\n")
            count_full = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(count_full.returncode, 75, count_full.stderr)

            for path in spool.glob("handoff-*.json"):
                path.unlink()
            for index in range(8):
                large = (
                    spool
                    / f"handoff-{index}-{owner_id}-999999-{index:016x}.json"
                )
                with large.open("wb") as handle:
                    handle.truncate(64 * 1024 * 1024)
            bytes_full = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(bytes_full.returncode, 75, bytes_full.stderr)

    def test_episode_spool_reservation_is_bounded_and_mismatch_is_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            _ensure_test_spool_owner(root, spool)
            foreign_atomic = spool / ".state.json.999999.tmp"
            foreign_atomic.write_text("foreign\n")

            wrong_root = root / "wrong-destinations"
            mismatch = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                spool,
                *_episode_destination_args(wrong_root),
            )
            self.assertEqual(mismatch.returncode, 1, mismatch.stderr)
            self.assertTrue(foreign_atomic.exists())

            foreign_atomic.unlink()
            reservations: list[Path] = []
            for _ in range(8):
                reserved = _run_episode_worker_cli(
                    "--reserve",
                    "--spool",
                    spool,
                    *_episode_destination_args(root),
                )
                self.assertEqual(reserved.returncode, 0, reserved.stderr)
                reservations.append(Path(reserved.stdout.strip()))
            self.assertEqual(len(set(reservations)), 8)
            self.assertTrue(all(path.is_file() for path in reservations))

            byte_cap = _run_episode_worker_cli(
                "--reserve",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(byte_cap.returncode, 75, byte_cap.stderr)

            for path in reservations:
                path.unlink()
            owner_id = _ensure_test_spool_owner(root, spool)
            for index in range(63):
                (
                    spool
                    / f"handoff-{index}-{owner_id}-999999-{index:016x}.json"
                ).write_text("{}\n")
            count_boundary = _run_episode_worker_cli(
                "--reserve",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(count_boundary.returncode, 0, count_boundary.stderr)
            count_temp = Path(count_boundary.stdout.strip())
            count_temp.write_text("{}\n")
            count_publish = _run_episode_worker_cli(
                "--publish",
                count_temp,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(count_publish.returncode, 0, count_publish.stderr)
            self.assertEqual(len(list(spool.glob("handoff-*.json"))), 64)

            for path in spool.glob("handoff-*.json"):
                path.unlink()
            for index in range(7):
                large = (
                    spool
                    / f"handoff-{index}-{owner_id}-999999-{index:016x}.json"
                )
                with large.open("wb") as handle:
                    handle.truncate(64 * 1024 * 1024)
            byte_boundary = _run_episode_worker_cli(
                "--reserve",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(byte_boundary.returncode, 0, byte_boundary.stderr)
            byte_temp = Path(byte_boundary.stdout.strip())
            with byte_temp.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024)
            byte_publish = _run_episode_worker_cli(
                "--publish",
                byte_temp,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(byte_publish.returncode, 0, byte_publish.stderr)
            self.assertEqual(
                sum(path.stat().st_size for path in spool.glob("handoff-*.json")),
                512 * 1024 * 1024,
            )

    def test_episode_spool_recovers_over_cap_outer_and_surfaces_nonregular(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            for index in range(9):
                (
                    spool
                    / f".handoff-999999-{owner_id}-{index:016x}.tmp"
                ).write_text("{}\n")

            with (
                patch.object(
                    worker_module,
                    "load_fast_bot_episode_handoff",
                    return_value={"sealed": True},
                ),
                patch("builtins.print"),
            ):
                promoted, discarded = worker_module.recover_stale_outer_handoffs(
                    spool,
                    owner_id=owner_id,
                )
            self.assertEqual((promoted, discarded), (9, 0))
            self.assertEqual(len(list(spool.glob("handoff-*.json"))), 9)

            for path in spool.glob("handoff-*.json"):
                path.unlink()
            fifo = spool / f".handoff-999999-{owner_id}-fifo.tmp"
            os.mkfifo(fifo)
            surfaced = _run_episode_worker_cli(
                "--launch",
                "--spool",
                spool,
                "--log",
                root / "worker.log",
                *_episode_destination_args(root),
            )
            self.assertEqual(surfaced.returncode, 1, surfaced.stderr)
            self.assertTrue(fifo.exists())
            self.assertIn("spool operation failed:", surfaced.stderr)

    def test_episode_spool_recovers_only_exact_owner_crash_temp(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "recoverable-spool"
            spool.mkdir()
            output, ledger, source_archive = _episode_destinations(root)
            expected = worker_module._spool_owner_contract(
                spool=spool,
                output=output,
                ledger=ledger,
                source_archive=source_archive,
            )
            owner_id = expected["owner_id"]
            crash_temp = (
                spool
                / f"..owner.json-{owner_id}-0000000000000001.999999.tmp"
            )
            crash_temp.write_bytes(
                worker_module._canonical_json_bytes(expected) + b"\n"
            )

            recovered = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertFalse(crash_temp.exists())
            self.assertEqual(
                worker_module._read_spool_owner(spool / ".owner.json"),
                expected,
            )

            foreign_spool = root / "foreign-spool"
            foreign_spool.mkdir()
            foreign_expected = worker_module._spool_owner_contract(
                spool=foreign_spool,
                output=output,
                ledger=ledger,
                source_archive=source_archive,
            )
            foreign_temp = foreign_spool / (
                f"..owner.json-{foreign_expected['owner_id']}-"
                "0000000000000001.999999.tmp"
            )
            foreign_temp.write_bytes(
                worker_module._canonical_json_bytes(foreign_expected) + b"\n"
            )
            foreign = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                foreign_spool,
                *_episode_destination_args(root / "wrong"),
            )
            self.assertEqual(foreign.returncode, 1, foreign.stderr)
            self.assertTrue(foreign_temp.exists())
            self.assertFalse((foreign_spool / ".owner.json").exists())

            torn_spool = root / "torn-spool"
            torn_spool.mkdir()
            torn_expected = worker_module._spool_owner_contract(
                spool=torn_spool,
                output=output,
                ledger=ledger,
                source_archive=source_archive,
            )
            torn_owner_id = torn_expected["owner_id"]
            empty_temp = torn_spool / (
                f"..owner.json-{torn_owner_id}-"
                "0000000000000001.999999.tmp"
            )
            partial_temp = torn_spool / (
                f"..owner.json-{torn_owner_id}-"
                "0000000000000002.999999.tmp"
            )
            empty_temp.touch()
            partial_temp.write_text("{partial")
            torn_recovered = _run_episode_worker_cli(
                "--check-capacity",
                "--spool",
                torn_spool,
                "--output",
                output,
                "--ledger",
                ledger,
                "--source-archive",
                source_archive,
            )
            self.assertEqual(torn_recovered.returncode, 0, torn_recovered.stderr)
            self.assertFalse(empty_temp.exists())
            self.assertFalse(partial_temp.exists())
            self.assertEqual(
                worker_module._read_spool_owner(torn_spool / ".owner.json"),
                torn_expected,
            )

    def test_episode_worker_lock_busy_and_invalid_inputs_remain_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            temp_handoff = _write_episode_handoff(
                root,
                spool,
                now=NOW,
                suffix="lockbusy",
            )
            published = _run_episode_worker_cli(
                "--publish",
                temp_handoff,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(published.returncode, 0, published.stderr)
            final_handoff = Path(published.stdout.strip())

            lock_path = spool / ".worker.lock"
            lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = _run_episode_worker_cli(
                    "--worker",
                    "--spool",
                    spool,
                    "--output",
                    root / "episode.json",
                    "--ledger",
                    root / "episode.jsonl",
                    "--source-archive",
                    root / "sources",
                )
            finally:
                os.close(lock_descriptor)
            self.assertEqual(busy.returncode, 75, busy.stderr)
            self.assertTrue(final_handoff.exists())

            final_handoff.write_text("{invalid\n")
            invalid = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                "--output",
                root / "episode.json",
                "--ledger",
                root / "episode.jsonl",
                "--source-archive",
                root / "sources",
            )
            self.assertEqual(invalid.returncode, 1, invalid.stderr)
            self.assertTrue(final_handoff.exists())

    def test_episode_spool_owner_mismatch_retains_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            temp_handoff = _write_episode_handoff(
                root,
                spool,
                now=NOW,
                suffix="owner",
            )
            published = _run_episode_worker_cli(
                "--publish",
                temp_handoff,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(published.returncode, 0, published.stderr)
            final_handoff = Path(published.stdout.strip())

            wrong_root = root / "wrong-destinations"
            mismatch = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                *_episode_destination_args(wrong_root),
            )

            self.assertEqual(mismatch.returncode, 1, mismatch.stderr)
            self.assertIn("operation failed: ValueError", mismatch.stderr)
            self.assertTrue(final_handoff.exists())
            self.assertFalse((wrong_root / "episode.json").exists())
            self.assertFalse((wrong_root / "episode.jsonl").exists())

    def test_episode_spool_metadata_busy_is_nonblocking_and_outer_temp_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            temp_handoff = _write_episode_handoff(
                root,
                spool,
                now=NOW,
                suffix="recover",
            )
            owner_id = _ensure_test_spool_owner(root, spool)
            metadata_lock = spool / ".metadata.lock"
            lock_descriptor = os.open(metadata_lock, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = _run_episode_worker_cli(
                    "--publish",
                    temp_handoff,
                    "--spool",
                    spool,
                    *_episode_destination_args(root),
                )
            finally:
                os.close(lock_descriptor)
            self.assertEqual(busy.returncode, 75, busy.stderr)
            self.assertTrue(temp_handoff.exists())

            stale_handoff = (
                spool / f".handoff-999999-{owner_id}-recover.tmp"
            )
            os.replace(temp_handoff, stale_handoff)
            worker = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                "--output",
                root / "episode.json",
                "--ledger",
                root / "episode.jsonl",
                "--source-archive",
                root / "sources",
            )

            self.assertEqual(worker.returncode, 0, worker.stderr)
            self.assertFalse(stale_handoff.exists())
            self.assertFalse(list(spool.glob("handoff-*.json")))
            self.assertIn(f"spool recovered {stale_handoff.name}", worker.stderr)

    def test_episode_worker_refuses_live_lock_environment_and_keeps_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            handoff = spool / "handoff-queued.json"
            handoff.write_text("queued\n")

            result = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                env=_episode_worker_env(QR_AUTOTRADE_LOCK_HELD="1"),
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertTrue(handoff.exists())
            self.assertIn("refuses the shared live lock", result.stderr)

    def test_episode_launcher_rejects_fifo_log_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            handoff = (
                spool
                / f"handoff-1-{owner_id}-999999-0000000000000001.json"
            )
            handoff.write_text("queued\n")
            log_path = root / "worker.fifo"
            os.mkfifo(log_path)

            result = _run_episode_worker_cli(
                "--launch",
                "--spool",
                spool,
                "--log",
                log_path,
                *_episode_destination_args(root),
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(handoff.exists())
            self.assertIn("spool operation failed:", result.stderr)

    def test_episode_launcher_rotates_bounded_regular_log(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            (
                spool
                / f"handoff-1-{owner_id}-999999-0000000000000001.json"
            ).write_text("queued\n")
            log_path = root / "worker.log"
            with log_path.open("wb") as handle:
                handle.truncate(worker_module.MAX_WORKER_LOG_BYTES)

            with (
                patch.object(worker_module.subprocess, "Popen") as popen,
                patch.dict(
                    os.environ,
                    {
                        "QR_LIVE_ENABLED": "0",
                        "QR_AUTOTRADE_LOCK_HELD": "0",
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                result = worker_module.launch_worker(
                    spool=spool,
                    output=root / "episode.json",
                    ledger=root / "episode.jsonl",
                    source_archive=root / "sources",
                    log_path=log_path,
                )

            self.assertEqual(result, 0)
            popen.assert_called_once()
            self.assertTrue(log_path.is_file())
            self.assertEqual(log_path.stat().st_size, 0)
            self.assertEqual(
                log_path.with_name("worker.log.1").stat().st_size,
                worker_module.MAX_WORKER_LOG_BYTES,
            )

    def test_episode_launcher_runs_empty_spool_for_mature_outcome_catchup(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            log_path = root / "worker.log"
            with (
                patch.object(worker_module.subprocess, "Popen") as popen,
                patch.dict(
                    os.environ,
                    {
                        "QR_LIVE_ENABLED": "0",
                        "QR_AUTOTRADE_LOCK_HELD": "0",
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                result = worker_module.launch_worker(
                    spool=spool,
                    output=root / "episode.json",
                    ledger=root / "fast_bot_episode_ledger.jsonl",
                    source_archive=root / "sources",
                    log_path=log_path,
                    outcome_enabled=True,
                )

            self.assertEqual(result, 0)
            popen.assert_called_once()
            command = popen.call_args.args[0]
            self.assertIn("--worker", command)
            self.assertIn("--outcome-enabled", command)
            self.assertFalse(list(spool.glob("handoff-*.json")))

    def test_episode_v2_waits_for_vehicle_projection_but_outcome_error_does_not_restore_it(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            handoff_path = (
                spool
                / f"handoff-1-{owner_id}-999999-0000000000000001.json"
            )
            handoff_path.write_text("{}\n")
            sealed = {
                "schema_version": 2,
                "cycle_generated_at_utc": NOW.isoformat(),
                "contract_sha256": "a" * 64,
                "regime_contract": {},
                "fast_pair_charts": {},
                "slow_pair_charts": {},
            }
            environment = {
                "QR_LIVE_ENABLED": "0",
                "QR_AUTOTRADE_LOCK_HELD": "0",
            }
            with (
                patch.object(
                    worker_module,
                    "load_fast_bot_episode_handoff",
                    return_value=sealed,
                ),
                patch.object(
                    worker_module,
                    "run_fast_bot_episode_shadow",
                    return_value={"status": "UPDATED"},
                ),
                patch.object(
                    worker_module,
                    "_run_episode_truth_cycle",
                    return_value={
                        "status": "LOCK_BUSY",
                        "vehicle_projection_status": "FAILED",
                    },
                ),
                patch.dict(os.environ, environment, clear=False),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                busy = worker_module.run_worker(
                    spool=spool,
                    output=root / "episode.json",
                    ledger=root / "episode.jsonl",
                    source_archive=root / "sources",
                )

            self.assertEqual(busy, 75)
            self.assertTrue(handoff_path.exists())

            with (
                patch.object(
                    worker_module,
                    "load_fast_bot_episode_handoff",
                    return_value=sealed,
                ),
                patch.object(
                    worker_module,
                    "run_fast_bot_episode_shadow",
                    return_value={"status": "NO_NEW_EVENT"},
                ),
                patch.object(
                    worker_module,
                    "_run_episode_truth_cycle",
                    return_value={
                        "status": "OUTCOME_IDENTITY_CONFLICT",
                        "vehicle_projection_status": "VERIFIED",
                    },
                ),
                patch.dict(os.environ, environment, clear=False),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                conflicted = worker_module.run_worker(
                    spool=spool,
                    output=root / "episode.json",
                    ledger=root / "episode.jsonl",
                    source_archive=root / "sources",
                )

            self.assertEqual(conflicted, 1)
            self.assertFalse(handoff_path.exists())

    def test_episode_outcome_resolver_derives_siblings_and_rejects_live_before_import(self) -> None:
        resolver = _load_episode_outcome_resolver_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "fast_bot_episode_ledger.jsonl"
            paths = resolver.derive_episode_truth_paths(ledger)
            resolved_root = ledger.resolve().parent
            self.assertEqual(
                paths,
                {
                    "vehicle_ledger_path": resolved_root / "fast_bot_episode_vehicle_ledger.jsonl",
                    "outcome_ledger_path": resolved_root / "fast_bot_episode_outcome_ledger.jsonl",
                    "scorecard_path": resolved_root / "fast_bot_episode_scorecard.json",
                    "lock_path": resolved_root / "fast_bot_episode_truth.lock",
                },
            )
            with (
                patch.object(
                    resolver,
                    "_load_truth_dependencies",
                    side_effect=AssertionError("unsafe runtime imported truth core"),
                ),
                patch.dict(
                    os.environ,
                    {
                        "QR_LIVE_ENABLED": "1",
                        "QR_AUTOTRADE_LOCK_HELD": "0",
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                rejected = resolver.run_episode_outcome_resolution(
                    episode_ledger_path=ledger,
                    source_archive_dir=root / "sources",
                )
            self.assertEqual(rejected["status"], "RUNTIME_SAFETY_REJECTED")
            self.assertFalse(rejected["broker_read"])
            self.assertFalse(ledger.exists())

            captured: dict = {}

            class ReadOnlyClient:
                def get_json(self, path: str, query: dict | None = None) -> dict:
                    return {}

            def truth_cycle(**kwargs):
                captured.update(kwargs)
                return {
                    "status": "NO_DUE_VEHICLES",
                    "vehicle_projection_status": "VERIFIED",
                }

            with (
                patch.object(
                    resolver,
                    "_load_truth_dependencies",
                    return_value=(truth_cycle, ReadOnlyClient),
                ),
                patch.dict(
                    os.environ,
                    {
                        "QR_LIVE_ENABLED": "0",
                        "QR_AUTOTRADE_LOCK_HELD": "0",
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                resolved = resolver.run_episode_outcome_resolution(
                    handoffs=({"schema_version": 2},),
                    episode_ledger_path=ledger,
                    source_archive_dir=root / "sources",
                )
            self.assertEqual(resolved["status"], "NO_DUE_VEHICLES")
            self.assertEqual(captured["handoffs"], ({"schema_version": 2},))
            self.assertIs(captured["client_factory"], ReadOnlyClient)
            self.assertEqual(captured["vehicle_ledger_path"], paths["vehicle_ledger_path"])
            self.assertEqual(captured["outcome_ledger_path"], paths["outcome_ledger_path"])
            self.assertEqual(captured["scorecard_path"], paths["scorecard_path"])
            self.assertEqual(captured["lock_path"], paths["lock_path"])

    def test_recovered_pending_batch_replays_same_handoff_before_delete(self) -> None:
        worker_module = _load_episode_worker_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            spool.mkdir()
            owner_id = _ensure_test_spool_owner(root, spool)
            handoff_path = (
                spool
                / f"handoff-1-{owner_id}-999999-0000000000000001.json"
            )
            handoff_path.write_text("{}\n")
            sealed = {
                "cycle_generated_at_utc": NOW.isoformat(),
                "contract_sha256": "a" * 64,
                "regime_contract": {},
                "fast_pair_charts": {},
                "slow_pair_charts": {},
            }
            with (
                patch.object(
                    worker_module,
                    "load_fast_bot_episode_handoff",
                    return_value=sealed,
                ),
                patch.object(
                    worker_module,
                    "run_fast_bot_episode_shadow",
                    side_effect=[
                        {"status": "RECOVERED_PENDING_BATCH"},
                        {"status": "NO_NEW_EVENT"},
                    ],
                ) as run_episode,
                patch.object(worker_module, "_run_episode_truth_cycle") as run_truth,
                patch.dict(
                    os.environ,
                    {
                        "QR_LIVE_ENABLED": "0",
                        "QR_AUTOTRADE_LOCK_HELD": "0",
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
                result = worker_module.run_worker(
                    spool=spool,
                    output=root / "episode.json",
                    ledger=root / "episode.jsonl",
                    source_archive=root / "sources",
                )

            self.assertEqual(result, 0)
            self.assertEqual(run_episode.call_count, 2)
            run_truth.assert_not_called()
            self.assertFalse(handoff_path.exists())

    def test_episode_worker_consumes_sealed_handoffs_in_cycle_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spool = root / "spool"
            first_temp = _write_episode_handoff(
                root,
                spool,
                now=NOW,
                suffix="first",
            )
            second_temp = _write_episode_handoff(
                root,
                spool,
                now=NOW + timedelta(seconds=30),
                suffix="second",
            )
            first_publish = _run_episode_worker_cli(
                "--publish",
                first_temp,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            second_publish = _run_episode_worker_cli(
                "--publish",
                second_temp,
                "--spool",
                spool,
                *_episode_destination_args(root),
            )
            self.assertEqual(first_publish.returncode, 0, first_publish.stderr)
            self.assertEqual(second_publish.returncode, 0, second_publish.stderr)
            first_name = Path(first_publish.stdout.strip()).name
            second_name = Path(second_publish.stdout.strip()).name

            worker = _run_episode_worker_cli(
                "--worker",
                "--spool",
                spool,
                "--output",
                root / "episode.json",
                "--ledger",
                root / "episode.jsonl",
                "--source-archive",
                root / "sources",
            )

            self.assertEqual(worker.returncode, 0, worker.stderr)
            self.assertFalse(list(spool.glob("handoff-*.json")))
            self.assertLess(worker.stderr.index(first_name), worker.stderr.index(second_name))
            state = json.loads((root / "episode.json").read_text())
            self.assertIn(state["status"], {"NO_NEW_EVENT", "UPDATED"})
            self.assertEqual(
                datetime.fromisoformat(state["generated_at_utc"]),
                NOW + timedelta(seconds=30),
            )
            scorecard = json.loads(
                (root / "episode_scorecard.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                scorecard["contract"],
                "QR_FAST_BOT_EPISODE_CLUSTER_SCORECARD_V1",
            )
            self.assertFalse(scorecard["live_permission"])
            self.assertEqual(scorecard["order_authority"], "NONE")
            self.assertIn("spool_delay_seconds=", worker.stderr)

    def test_primary_seals_same_validated_packets_for_post_lock_episode(self) -> None:
        fast, slow, snapshot = _inputs()
        invalid_fast = _seal_contract(
            {
                "contract": CURRENT_M1_CONTRACT,
                "schema_version": 1,
                "status": "CURRENT",
                "configured_pairs": ["EUR_USD"],
                "charts": fast["charts"],
            }
        )
        invalid_fast["contract_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "fast": root / "fast.json",
                "slow": root / "slow.json",
                "snapshot": root / "snapshot.json",
                "events": root / "events.json",
            }
            paths["fast"].write_text(json.dumps(invalid_fast), encoding="utf-8")
            paths["slow"].write_text(json.dumps(slow), encoding="utf-8")
            paths["snapshot"].write_text(json.dumps(snapshot), encoding="utf-8")
            paths["events"].write_text(json.dumps({"events": []}), encoding="utf-8")
            result = run_fast_bot_shadow(
                fast_pair_charts_path=paths["fast"],
                slow_pair_charts_path=paths["slow"],
                broker_snapshot_path=paths["snapshot"],
                guardian_events_path=paths["events"],
                ai_supervision_path=None,
                regime_output_path=root / "regime.json",
                shadow_output_path=root / "shadow.json",
                shadow_ledger_path=root / "shadow.jsonl",
                report_path=root / "report.md",
                now_utc=NOW,
                episode_handoff_path=root / "episode-handoff.json",
            )

            handoff = load_fast_bot_episode_handoff(root / "episode-handoff.json")
            episode_fast = handoff["fast_pair_charts"]
            regime = json.loads((root / "regime.json").read_text())
            self.assertEqual(handoff["contract"], EPISODE_HANDOFF_CONTRACT)
            self.assertEqual(handoff["cycle_generated_at_utc"], NOW.isoformat())
            self.assertEqual(handoff["regime_contract"], regime)
            self.assertEqual(handoff["regime_contract_sha256"], regime["contract_sha256"])
            self.assertEqual(handoff["schema_version"], 2)
            self.assertEqual(handoff["broker_snapshot"], snapshot)
            self.assertEqual(
                handoff["prospective_vehicle_shadow_sha256"],
                handoff["prospective_vehicle_shadow"]["contract_sha256"],
            )
            self.assertEqual(
                handoff["broker_snapshot_sha256"],
                regime["sources"]["broker_snapshot_sha256"],
            )
            self.assertEqual(result["episode_handoff_sha256"], handoff["contract_sha256"])
            self.assertEqual(len(episode_fast["charts"]), 28)
            self.assertTrue(all(not chart["views"] for chart in episode_fast["charts"]))
            self.assertTrue(handoff["diagnostic_only"])
            self.assertEqual(handoff["order_authority"], "NONE")
            self.assertFalse(handoff["live_permission"])
            self.assertFalse(handoff["broker_mutation_allowed"])

    def test_primary_has_no_synchronous_episode_side_effect(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, payload in (
                ("fast", fast),
                ("slow", slow),
                ("snapshot", snapshot),
                ("events", {"events": []}),
            ):
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            result = run_fast_bot_shadow(
                fast_pair_charts_path=root / "fast.json",
                slow_pair_charts_path=root / "slow.json",
                broker_snapshot_path=root / "snapshot.json",
                guardian_events_path=root / "events.json",
                ai_supervision_path=None,
                regime_output_path=root / "regime.json",
                shadow_output_path=root / "shadow.json",
                shadow_ledger_path=root / "shadow.jsonl",
                report_path=root / "report.md",
                now_utc=NOW,
                episode_handoff_path=root / "episode-handoff.json",
            )

            self.assertNotIn("episode_status", result)
            self.assertNotIn("episode_events_appended", result)
            self.assertTrue((root / "shadow.json").exists())
            self.assertTrue((root / "shadow.jsonl").exists())
            self.assertTrue((root / "episode-handoff.json").exists())
            self.assertFalse((root / "episode.json").exists())
            self.assertFalse((root / "episode.jsonl").exists())
            shadow = json.loads((root / "shadow.json").read_text())
            self.assertTrue(shadow["shadow_only"])
            self.assertFalse(shadow["live_permission"])

    def test_primary_never_publishes_an_unreadable_over_cap_handoff(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, payload in (
                ("fast", fast),
                ("slow", slow),
                ("snapshot", snapshot),
                ("events", {"events": []}),
            ):
                (root / f"{name}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            handoff_path = root / "episode-handoff.json"
            with (
                patch("quant_rabbit.fast_bot.MAX_EPISODE_HANDOFF_BYTES", 1024),
                self.assertRaisesRegex(ValueError, "byte cap"),
            ):
                run_fast_bot_shadow(
                    fast_pair_charts_path=root / "fast.json",
                    slow_pair_charts_path=root / "slow.json",
                    broker_snapshot_path=root / "snapshot.json",
                    guardian_events_path=root / "events.json",
                    ai_supervision_path=None,
                    regime_output_path=root / "regime.json",
                    shadow_output_path=root / "shadow.json",
                    shadow_ledger_path=root / "shadow.jsonl",
                    report_path=root / "report.md",
                    now_utc=NOW,
                    episode_handoff_path=handoff_path,
                )
            self.assertFalse(handoff_path.exists())

    def test_dedicated_episode_runner_uses_sealed_cycle_after_sources_change(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, payload in (
                ("fast", fast),
                ("slow", slow),
                ("snapshot", snapshot),
                ("events", {"events": []}),
            ):
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            handoff_path = root / "episode-handoff.json"
            regime_path = root / "regime.json"
            run_fast_bot_shadow(
                fast_pair_charts_path=root / "fast.json",
                slow_pair_charts_path=root / "slow.json",
                broker_snapshot_path=root / "snapshot.json",
                guardian_events_path=root / "events.json",
                ai_supervision_path=None,
                regime_output_path=regime_path,
                shadow_output_path=root / "shadow.json",
                shadow_ledger_path=root / "shadow.jsonl",
                report_path=root / "report.md",
                now_utc=NOW,
                episode_handoff_path=handoff_path,
            )
            (root / "fast.json").write_text('{"next_cycle":true}\n')
            (root / "slow.json").write_text('{"next_cycle":true}\n')
            regime_path.write_text('{"next_cycle":true}\n')
            env = os.environ.copy()
            env.update({"QR_LIVE_ENABLED": "0", "QR_AUTOTRADE_LOCK_HELD": "0"})
            env.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)

            result = subprocess.run(
                [
                    str(Path(__file__).resolve().parents[1] / "scripts" / "run-fast-bot-episode-shadow.py"),
                    "--handoff",
                    str(handoff_path),
                    "--output",
                    str(root / "episode.json"),
                    "--ledger",
                    str(root / "episode.jsonl"),
                    "--source-archive",
                    str(root / "episode-sources"),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((root / "episode.json").read_text())
            self.assertEqual(state["generated_at_utc"], NOW.isoformat())
            self.assertTrue(state["diagnostic_only"])
            self.assertFalse(state["live_permission"])

    def test_dedicated_episode_runner_rejects_any_live_lock_and_reports_busy(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, payload in (
                ("fast", fast),
                ("slow", slow),
                ("snapshot", snapshot),
                ("events", {"events": []}),
            ):
                (root / f"{name}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            handoff_path = root / "episode-handoff.json"
            run_fast_bot_shadow(
                fast_pair_charts_path=root / "fast.json",
                slow_pair_charts_path=root / "slow.json",
                broker_snapshot_path=root / "snapshot.json",
                guardian_events_path=root / "events.json",
                ai_supervision_path=None,
                regime_output_path=root / "regime.json",
                shadow_output_path=root / "shadow.json",
                shadow_ledger_path=root / "shadow.jsonl",
                report_path=root / "report.md",
                now_utc=NOW,
                episode_handoff_path=handoff_path,
            )
            command = [
                sys.executable,
                str(EPISODE_RUNNER),
                "--handoff",
                str(handoff_path),
                "--output",
                str(root / "episode.json"),
                "--ledger",
                str(root / "episode.jsonl"),
                "--source-archive",
                str(root / "episode-sources"),
            ]

            held = subprocess.run(
                command,
                env=_episode_worker_env(QR_AUTOTRADE_LOCK_HELD="2"),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(held.returncode, 2, held.stderr)

            ledger_descriptor = os.open(
                root / "episode.jsonl",
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            try:
                fcntl.flock(ledger_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = subprocess.run(
                    command,
                    env=_episode_worker_env(),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            finally:
                os.close(ledger_descriptor)
            self.assertEqual(busy.returncode, 75, busy.stderr)
            self.assertEqual(json.loads(busy.stdout)["status"], "LOCK_BUSY")

    def test_dedicated_episode_runner_replays_recovery_and_fails_closed(self) -> None:
        runner_module = _load_episode_runner_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sealed_handoff = {
                "cycle_generated_at_utc": NOW.isoformat(),
                "regime_contract": {"sealed": True},
                "fast_pair_charts": {"fast": True},
                "slow_pair_charts": {"slow": True},
            }
            argv = [
                str(EPISODE_RUNNER),
                "--handoff",
                str(root / "handoff.json"),
                "--output",
                str(root / "episode.json"),
                "--ledger",
                str(root / "episode.jsonl"),
                "--source-archive",
                str(root / "sources"),
            ]
            environment = {
                "QR_LIVE_ENABLED": "0",
                "QR_AUTOTRADE_LOCK_HELD": "0",
                "QR_AUTOTRADE_LOCK_OWNER_TOKEN": "",
            }

            with (
                patch.object(
                    runner_module,
                    "load_fast_bot_episode_handoff",
                    return_value=sealed_handoff,
                ),
                patch.object(
                    runner_module,
                    "run_fast_bot_episode_shadow",
                    side_effect=[
                        {"status": "RECOVERED_PENDING_BATCH"},
                        {"status": "NO_NEW_EVENT"},
                    ],
                ) as run_episode,
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, environment, clear=False),
                patch("builtins.print"),
            ):
                recovered = runner_module.main()

            self.assertEqual(recovered, 0)
            self.assertEqual(run_episode.call_count, 2)
            first_call, second_call = run_episode.call_args_list
            self.assertEqual(first_call.kwargs, second_call.kwargs)

            for statuses in (
                [
                    {"status": "RECOVERED_PENDING_BATCH"},
                    {"status": "RECOVERED_PENDING_BATCH"},
                ],
                [{"status": "LEDGER_INVALID"}],
                [{}],
            ):
                with self.subTest(statuses=statuses):
                    with (
                        patch.object(
                            runner_module,
                            "load_fast_bot_episode_handoff",
                            return_value=sealed_handoff,
                        ),
                        patch.object(
                            runner_module,
                            "run_fast_bot_episode_shadow",
                            side_effect=statuses,
                        ),
                        patch.object(sys, "argv", argv),
                        patch.dict(os.environ, environment, clear=False),
                        patch("builtins.print"),
                    ):
                        self.assertEqual(runner_module.main(), 1)

    def test_episode_handoff_rejects_nested_cycle_tamper(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, payload in (
                ("fast", fast),
                ("slow", slow),
                ("snapshot", snapshot),
                ("events", {"events": []}),
            ):
                (root / f"{name}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            handoff_path = root / "episode-handoff.json"
            run_fast_bot_shadow(
                fast_pair_charts_path=root / "fast.json",
                slow_pair_charts_path=root / "slow.json",
                broker_snapshot_path=root / "snapshot.json",
                guardian_events_path=root / "events.json",
                ai_supervision_path=None,
                regime_output_path=root / "regime.json",
                shadow_output_path=root / "shadow.json",
                shadow_ledger_path=root / "shadow.jsonl",
                report_path=root / "report.md",
                now_utc=NOW,
                episode_handoff_path=handoff_path,
            )
            tampered = json.loads(handoff_path.read_text())
            tampered["fast_pair_charts"]["charts"][0]["pair"] = "GBP_USD"
            handoff_path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "seal|binding"):
                load_fast_bot_episode_handoff(handoff_path)

    def test_episode_handoff_v2_freezes_snapshot_and_still_drains_legacy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            handoff_path = _write_episode_handoff(
                root,
                root / "spool",
                now=NOW,
                suffix="feedfacefeedface",
            )
            current = json.loads(handoff_path.read_text())
            self.assertEqual(current["schema_version"], 2)
            self.assertIn("broker_snapshot", current)

            tampered = json.loads(json.dumps(current))
            tampered["broker_snapshot"]["quotes"]["EUR_USD"]["bid"] = 1.0
            tampered = _seal_contract(
                {key: value for key, value in tampered.items() if key != "contract_sha256"}
            )
            handoff_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding"):
                load_fast_bot_episode_handoff(handoff_path)

            legacy_body = {
                key: value
                for key, value in current.items()
                if key
                not in {
                    "contract_sha256",
                    "broker_snapshot",
                    "broker_snapshot_sha256",
                    "prospective_vehicle_shadow",
                    "prospective_vehicle_shadow_sha256",
                }
            }
            legacy_body["contract"] = LEGACY_EPISODE_HANDOFF_CONTRACT
            legacy_body["schema_version"] = 1
            handoff_path.write_text(
                json.dumps(_seal_contract(legacy_body)),
                encoding="utf-8",
            )
            loaded = load_fast_bot_episode_handoff(handoff_path)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertNotIn("broker_snapshot", loaded)

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
