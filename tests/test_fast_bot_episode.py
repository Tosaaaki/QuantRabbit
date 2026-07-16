from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import quant_rabbit.fast_bot_episode as episode_module
from quant_rabbit.fast_bot_episode import (
    EPISODE_EVENT_CONTRACT,
    verify_episode_event,
    verify_episode_ledger,
    run_fast_bot_episode_shadow,
)


START = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
ATTEMPT_START = START + timedelta(minutes=100)
ATTEMPT_CLOSE = ATTEMPT_START + timedelta(minutes=5)
NOW = ATTEMPT_CLOSE + timedelta(seconds=30)
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
}


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _sha(body)}


def _ledger_line(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _m5_window(current: dict[str, float]) -> list[dict[str, object]]:
    prior = [
        {
            "t": (START + timedelta(minutes=5 * index)).isoformat(),
            "o": 150.0,
            "h": 200.0,
            "l": 100.0,
            "c": 150.0,
            "complete": True,
        }
        for index in range(20)
    ]
    return [
        *prior,
        {
            "t": ATTEMPT_START.isoformat(),
            **current,
            "complete": True,
        },
    ]


def _m1_window(*, final_close: float = 150.0) -> list[dict[str, object]]:
    first = ATTEMPT_CLOSE - timedelta(minutes=21)
    rows = []
    for index in range(21):
        close = final_close if index == 20 else 150.0
        rows.append(
            {
                "t": (first + timedelta(minutes=index)).isoformat(),
                "o": close,
                "h": close + 1.0,
                "l": close - 1.0,
                "c": close,
                "complete": True,
                "v": 10,
            }
        )
    return rows


def _charts(current: dict[str, float], *, m1_close: float = 150.0) -> tuple[dict, dict]:
    fast = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    {
                        "granularity": "M1",
                        "recent_candles": _m1_window(final_close=m1_close),
                        "market_state": {"evidence_complete": True},
                    }
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
                    {
                        "granularity": "M5",
                        "recent_candles": _m5_window(current),
                        "market_state": {"evidence_complete": True},
                    }
                ],
            }
        ],
    }
    return fast, slow


def _votes() -> dict[str, dict[str, object]]:
    return {
        timeframe: {
            "observed_direction": "UP",
            "direction_score": 1,
            "phase": "RANGE" if timeframe in {"M5", "M15", "M30"} else "TREND",
            "readiness": "ARMED",
            "trigger": "NONE",
            "structure": "RANGE_BOUND",
            "location": "MIDDLE_THIRD",
            "value_zone": "FAIR_VALUE",
            "extension": "BALANCED",
            "evidence_complete": True,
        }
        for timeframe in TIMEFRAMES
    }


def _chart_clocks(fast: dict, slow: dict) -> dict[str, str | None]:
    clocks: dict[str, str | None] = {timeframe: None for timeframe in TIMEFRAMES}
    for payload in (slow, fast):
        for chart in payload.get("charts", []):
            for view in chart.get("views", []):
                timeframe = str(view.get("granularity") or "").upper()
                if timeframe not in TIMEFRAME_SECONDS:
                    continue
                closes = [
                    datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00"))
                    + timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
                    for row in view.get("recent_candles", [])
                    if row.get("complete") is True
                ]
                if closes:
                    clocks[timeframe] = max(closes).isoformat()
    return clocks


def _regime(
    now: datetime = NOW,
    *,
    source_clocks: dict[str, str | None] | None = None,
) -> dict:
    clocks = source_clocks or {
        timeframe: ATTEMPT_CLOSE.isoformat()
        if timeframe in {"M1", "M5"}
        else None
        for timeframe in TIMEFRAMES
    }
    return _seal(
        {
            "contract": "QR_HIERARCHICAL_BOT_REGIME_V1",
            "schema_version": 1,
            "generated_at_utc": now.isoformat(),
            "rows": [
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "RANGE_ROTATION",
                    "state": "CAUTION",
                    "execution_enabled": False,
                    "timeframe_votes": _votes(),
                    "source_timeframe_clocks": clocks,
                    "source_timeframe_clocks_sha256": _sha(clocks),
                    "m1_closed_candle_utc": clocks["M1"],
                }
            ],
        }
    )


def _next_m1(fast: dict, *, close: float) -> None:
    view = fast["charts"][0]["views"][0]
    rows = list(view["recent_candles"])
    rows.append(
        {
            "t": ATTEMPT_CLOSE.isoformat(),
            "o": close,
            "h": close + 1.0,
            "l": close - 1.0,
            "c": close,
            "complete": True,
            "v": 11,
        }
    )
    view["recent_candles"] = rows[-21:]


class FastBotEpisodeTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        current: dict[str, float],
        now: datetime = NOW,
        processed_at: datetime | None = None,
        fast: dict | None = None,
        slow: dict | None = None,
    ) -> dict:
        if fast is None or slow is None:
            fast, slow = _charts(current)
        return run_fast_bot_episode_shadow(
            regime_contract=_regime(
                now,
                source_clocks=_chart_clocks(fast, slow),
            ),
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            output_path=root / "episode_state.json",
            ledger_path=root / "episode_ledger.jsonl",
            source_archive_dir=root / "episode_sources",
            now_utc=now,
            processed_at_utc=processed_at,
        )

    def test_all_archive_directory_locks_fail_closed_without_waiting(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        real_flock = episode_module.fcntl.flock

        def reject_directory_lock(descriptor: int, operation: int) -> None:
            if (
                operation & episode_module.fcntl.LOCK_EX
                and episode_module.stat.S_ISDIR(
                    episode_module.os.fstat(descriptor).st_mode
                )
            ):
                raise BlockingIOError("injected directory lock contention")
            real_flock(descriptor, operation)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "quant_rabbit.fast_bot_episode.fcntl.flock",
                side_effect=reject_directory_lock,
            ):
                result = self._run(root, current=current)
            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_SOURCE_ARCHIVE_LOCK_BUSY",
                result["blockers"],
            )

            separate_archive = root / "separate_sources"
            separate_archive.mkdir()
            with patch(
                "quant_rabbit.fast_bot_episode.fcntl.flock",
                side_effect=reject_directory_lock,
            ):
                with self.assertRaisesRegex(ValueError, "archive lock busy"):
                    episode_module._archive_regime_contract(
                        _regime(),
                        separate_archive,
                    )
                cleanup_error = episode_module._cleanup_unreferenced_archives(
                    separate_archive,
                    referenced_digests=set(),
                )
            self.assertEqual(
                cleanup_error,
                "EPISODE_SOURCE_ARCHIVE_LOCK_BUSY",
            )

    def test_metadata_fifos_are_rejected_before_any_text_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_dir = root / "episode_sources"
            archive_dir.mkdir()
            owner = archive_dir / ".owner.json"
            state = root / "episode_state.json"
            pending = root / "episode_state.json.pending"
            for path in (owner, state, pending):
                episode_module.os.mkfifo(path)

            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("FIFO must not reach read_text"),
            ):
                owner_error = episode_module._ensure_archive_owner(
                    archive_dir,
                    root / "episode_ledger.jsonl",
                )
                state_value, state_error = episode_module._read_state_checkpoint(
                    state
                )
                pending_value, pending_error = (
                    episode_module._read_pending_checkpoint(pending)
                )

            self.assertEqual(
                owner_error,
                "EPISODE_SOURCE_ARCHIVE_OWNER_INVALID",
            )
            self.assertIsNone(state_value)
            self.assertEqual(state_error, "EPISODE_STATE_CHECKPOINT_INVALID")
            self.assertIsNone(pending_value)
            self.assertEqual(
                pending_error,
                "EPISODE_PENDING_CHECKPOINT_INVALID",
            )

    def test_maximum_episode_count_cannot_make_state_self_unreadable(self) -> None:
        events = [
            {
                "episode_id": f"episode-{index}",
                "event_seq": 1,
                "event_sha256": f"{index:064x}",
                "pair": "EUR_USD",
                "state": "RESOLVED",
                "generated_at_utc": (
                    NOW - timedelta(microseconds=index)
                ).isoformat(),
                "anchor": {"attempt_direction": "UP"},
                "route": {
                    "branch_outcome": "ACCEPTED",
                    "trade_side": "LONG",
                    "candidate_methods": ["TREND_CONTINUATION"],
                },
                "observation": {"candle_close_utc": NOW.isoformat()},
                "late_detected": False,
            }
            for index in range(episode_module.MAX_LEDGER_EVENTS)
        ]
        state = episode_module._build_state(
            events=events,
            now=NOW,
            processed_at=NOW,
            status="VERIFIED",
            appended=0,
            blockers=[],
            ledger_path=Path("episode.jsonl"),
            source_archive_dir=Path("episode_sources"),
            ledger_event_count=len(events),
            ledger_size_bytes=episode_module.MAX_LEDGER_BYTES,
            ledger_bytes_sha256="a" * 64,
            ledger_tail_sha256=events[-1]["event_sha256"],
            ledger_head_verified=True,
        )
        encoded = (
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        self.assertLessEqual(len(encoded), episode_module.MAX_STATE_BYTES)
        self.assertEqual(
            state["latest_episode_summaries"],
            episode_module.MAX_STATE_EPISODE_SUMMARIES,
        )
        self.assertTrue(state["latest_episodes_truncated"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            episode_module._write_json_atomic(
                path,
                state,
                max_bytes=episode_module.MAX_STATE_BYTES,
            )
            loaded, error = episode_module._read_state_checkpoint(path)
            self.assertIsNone(error)
            self.assertEqual(loaded, state)

    def test_metadata_writer_rejects_oversize_before_creating_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            with self.assertRaisesRegex(ValueError, "byte cap"):
                episode_module._write_json_atomic(
                    path,
                    {"payload": "too large"},
                    max_bytes=1,
                )
            self.assertFalse(path.exists())
            self.assertFalse(
                path.with_name(f".{path.name}.{episode_module.os.getpid()}.tmp").exists()
            )

    def test_four_attempt_branch_mappings_are_not_reversed(self) -> None:
        cases = (
            ({"o": 190.0, "h": 210.0, "l": 140.0, "c": 206.0}, "UP", "ACCEPTED", "LONG"),
            ({"o": 190.0, "h": 201.0, "l": 140.0, "c": 194.0}, "UP", "REJECTED", "SHORT"),
            ({"o": 110.0, "h": 160.0, "l": 90.0, "c": 94.0}, "DOWN", "ACCEPTED", "SHORT"),
            ({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}, "DOWN", "REJECTED", "LONG"),
        )
        for current, attempt_direction, branch, trade_side in cases:
            with self.subTest(attempt_direction=attempt_direction, branch=branch):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    result = self._run(root, current=current)
                    event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

                    self.assertEqual(result["status"], "UPDATED")
                    self.assertEqual(event["contract"], EPISODE_EVENT_CONTRACT)
                    self.assertEqual(event["anchor"]["attempt_direction"], attempt_direction)
                    self.assertEqual(event["route"]["branch_outcome"], branch)
                    self.assertEqual(event["route"]["trade_side"], trade_side)
                    self.assertEqual(event["transition_path"], ["SETUP", "ATTEMPT", branch])
                    self.assertEqual(verify_episode_event(event), (True, None))
                    self.assertFalse(event["live_permission"])
                    self.assertEqual(event["order_authority"], "NONE")

    def test_delayed_processing_preserves_cycle_identity_and_marks_late(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        processed_at = NOW + timedelta(minutes=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            result = self._run(
                root,
                current=current,
                processed_at=processed_at,
            )

            event = json.loads(
                (root / "episode_ledger.jsonl").read_text().strip()
            )
            state = json.loads((root / "episode_state.json").read_text())
            self.assertEqual(result["status"], "UPDATED")
            self.assertEqual(result["appended_events"], 1)
            self.assertEqual(event["generated_at_utc"], NOW.isoformat())
            self.assertTrue(event["late_detected"])
            self.assertEqual(state["processed_at_utc"], processed_at.isoformat())
            self.assertEqual(state["processing_delay_seconds"], 120.0)
            self.assertTrue(state["operationally_late"])

    def test_processing_clock_cannot_precede_sealed_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "processing clock precedes"):
                self._run(
                    Path(temp_dir),
                    current={"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0},
                    processed_at=NOW - timedelta(microseconds=1),
                )

    def test_accepted_breakout_never_routes_to_range_fade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(
                root,
                current={"o": 190.0, "h": 210.0, "l": 140.0, "c": 206.0},
            )
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            self.assertEqual(event["route"]["candidate_methods"], ["TREND_CONTINUATION"])
            self.assertEqual(event["route"]["route_family"], "BREAKOUT_CONTINUATION")
            self.assertNotIn("RANGE_ROTATION", event["route"]["candidate_methods"])

    def test_exact_buffer_boundary_remains_pending(self) -> None:
        for current in (
            {"o": 190.0, "h": 201.0, "l": 140.0, "c": 195.0},
            {"o": 110.0, "h": 160.0, "l": 99.0, "c": 105.0},
        ):
            with self.subTest(current=current):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    self._run(root, current=current)
                    event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
                    self.assertEqual(event["state"], "ATTEMPT")
                    self.assertEqual(event["route"]["branch_outcome"], "PENDING")
                    self.assertEqual(event["transition_path"], ["SETUP", "ATTEMPT"])

    def test_both_rails_pierced_is_atomic_invalidated_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(
                root,
                current={"o": 150.0, "h": 201.0, "l": 99.0, "c": 150.0},
            )
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            self.assertEqual(event["state"], "INVALIDATED")
            self.assertEqual(event["route"]["branch_outcome"], "AMBIGUOUS")
            self.assertEqual(
                event["transition_path"],
                ["SETUP", "ATTEMPT", "INVALIDATED"],
            )

    def test_confirmation_requires_a_strictly_later_closed_m1(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            first = self._run(root, current=current, fast=fast, slow=slow)
            self.assertEqual(first["latest_episodes"][0]["state"], "REJECTED")

            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            fast["generated_at_utc"] = later.isoformat()
            slow["generated_at_utc"] = later.isoformat()
            second = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )
            events = [json.loads(line) for line in (root / "episode_ledger.jsonl").read_text().splitlines()]

            self.assertEqual(second["latest_episodes"][0]["state"], "CONFIRMED")
            self.assertEqual(events[1]["transition_path"], ["REJECTED", "CONFIRMED"])
            self.assertGreater(
                datetime.fromisoformat(events[1]["observation"]["candle_close_utc"]),
                datetime.fromisoformat(events[0]["observation"]["candle_close_utc"]),
            )
            self.assertEqual(
                verify_episode_ledger(
                    events,
                    source_archive_dir=root / "episode_sources",
                ),
                (True, None),
            )

    def test_same_source_is_idempotent(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self._run(root, current=current)
            second = self._run(root, current=current)

            self.assertEqual(first["appended_events"], 1)
            self.assertEqual(second["appended_events"], 0)
            self.assertEqual(len((root / "episode_ledger.jsonl").read_text().splitlines()), 1)

    def test_cached_prefix_rechecks_gzip_sha_without_decompressing_history(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)

            with patch(
                "quant_rabbit.fast_bot_episode._read_archived_regime_contract",
                side_effect=AssertionError("trusted prefix must not be expanded"),
            ):
                result = self._run(root, current=current)

            self.assertEqual(result["status"], "NO_NEW_EVENT")
            self.assertEqual(result["appended_events"], 0)

    def test_cached_prefix_rejects_changed_gzip_bytes(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            archive = next((root / "episode_sources").glob("*.json.gz"))
            compressed = bytearray(archive.read_bytes())
            compressed[-1] ^= 1
            archive.write_bytes(compressed)

            result = self._run(root, current=current, now=NOW + timedelta(minutes=1))

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_SOURCE_ARCHIVE_GZIP_SHA_MISMATCH",
                result["blockers"],
            )

    def test_rehashed_event_and_alternate_gzip_are_still_rejected(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            archive = next((root / "episode_sources").glob("*.json.gz"))
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
            alternate = gzip.compress(gzip.decompress(archive.read_bytes()), mtime=1)
            archive.write_bytes(alternate)
            event["observation"]["regime_archive_gzip_sha256"] = hashlib.sha256(
                alternate
            ).hexdigest()
            body = {key: item for key, item in event.items() if key != "event_sha256"}
            event["event_sha256"] = _sha(body)

            result = verify_episode_ledger(
                [event],
                source_archive_dir=root / "episode_sources",
            )

            self.assertEqual(
                result,
                (False, "EPISODE_SOURCE_ARCHIVE_COMPRESSION_NONCANONICAL"),
            )

    def test_decompressed_source_budget_is_enforced(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            with (
                patch(
                    "quant_rabbit.fast_bot_episode.MAX_SOURCE_VERIFY_DECOMPRESSED_BYTES",
                    1,
                ),
                patch(
                    "quant_rabbit.fast_bot_episode._compressed_regime_contract",
                    side_effect=AssertionError(
                        "budget rejection must precede parse/recompression"
                    ),
                ),
            ):
                result = verify_episode_ledger(
                    [event],
                    source_archive_dir=root / "episode_sources",
                )

            self.assertEqual(
                result,
                (False, "EPISODE_SOURCE_VERIFY_DECOMPRESSED_CAP_EXCEEDED"),
            )

    def test_archive_cap_is_durable_and_does_not_append(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "quant_rabbit.fast_bot_episode.MAX_SOURCE_ARCHIVE_TOTAL_BYTES",
                1,
            ):
                result = self._run(root, current=current)

            state = json.loads((root / "episode_state.json").read_text())
            self.assertEqual(result["status"], "SOURCE_ARCHIVE_CAP_REACHED")
            self.assertEqual(state["status"], "SOURCE_ARCHIVE_CAP_REACHED")
            self.assertEqual(result["appended_events"], 0)
            self.assertEqual((root / "episode_ledger.jsonl").read_bytes(), b"")
            self.assertFalse((root / "episode_state.json.pending").exists())

    def test_existing_archive_inventory_cap_is_checked_before_cached_read(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)

            with patch(
                "quant_rabbit.fast_bot_episode.MAX_SOURCE_ARCHIVE_TOTAL_BYTES",
                1,
            ):
                result = self._run(
                    root,
                    current=current,
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_SOURCE_ARCHIVE_AGGREGATE_CAP_EXCEEDED",
                result["blockers"],
            )

    def test_owned_orphan_archives_and_temps_are_cleaned(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            archive_dir = root / "episode_sources"
            orphan = archive_dir / f"{'f' * 64}.json.gz"
            state_temp = root / ".episode_state.json.999.tmp"
            pending_temp = root / ".episode_state.json.pending.999.tmp"
            archive_temp = archive_dir / f".{('e' * 64)}.999.tmp"
            owner_temp = archive_dir / "..owner.json.999.tmp"
            orphan.write_bytes(gzip.compress(b"orphan", mtime=0))
            for path in (state_temp, pending_temp, archive_temp, owner_temp):
                path.write_bytes(b"stale")

            result = self._run(root, current=current)

            self.assertEqual(result["status"], "NO_NEW_EVENT")
            for path in (
                orphan,
                state_temp,
                pending_temp,
                archive_temp,
                owner_temp,
            ):
                self.assertFalse(path.exists(), path)

    def test_wal_recovers_ledger_fsync_before_state_checkpoint(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            original_write = episode_module._write_json_atomic

            def crash_before_state(
                path: Path,
                value: dict,
                **kwargs: object,
            ) -> None:
                if path.name == "episode_state.json":
                    raise OSError("injected state checkpoint crash")
                original_write(path, value, **kwargs)

            with patch(
                "quant_rabbit.fast_bot_episode._write_json_atomic",
                side_effect=crash_before_state,
            ):
                with self.assertRaises(OSError):
                    self._run(
                        root,
                        current=current,
                        now=later,
                        fast=fast,
                        slow=slow,
                    )

            self.assertTrue((root / "episode_state.json.pending").exists())
            self.assertEqual(
                len((root / "episode_ledger.jsonl").read_text().splitlines()),
                2,
            )
            self.assertEqual(
                json.loads((root / "episode_state.json").read_text())[
                    "ledger_event_count"
                ],
                1,
            )

            recovered = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )

            self.assertEqual(recovered["status"], "RECOVERED_PENDING_BATCH")
            self.assertIn(
                "EPISODE_PENDING_BATCH_RECONCILED",
                recovered["blockers"],
            )
            self.assertEqual(recovered["ledger_event_count"], 2)
            self.assertFalse((root / "episode_state.json.pending").exists())

    def test_wal_completes_only_its_exact_partial_suffix(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            original_pending_write = episode_module._write_pending_checkpoint

            def stop_after_wal(path: Path, value: dict) -> None:
                original_pending_write(path, value)
                raise OSError("injected stop after pending fsync")

            with patch(
                "quant_rabbit.fast_bot_episode._write_pending_checkpoint",
                side_effect=stop_after_wal,
            ):
                with self.assertRaises(OSError):
                    self._run(
                        root,
                        current=current,
                        now=later,
                        fast=fast,
                        slow=slow,
                    )

            pending = json.loads((root / "episode_state.json.pending").read_text())
            batch = base64.b64decode(pending["batch_jsonl_base64"], validate=True)
            with (root / "episode_ledger.jsonl").open("ab") as handle:
                handle.write(batch[: len(batch) // 2])

            recovered = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )

            self.assertEqual(recovered["status"], "RECOVERED_PENDING_BATCH")
            self.assertEqual(recovered["ledger_event_count"], 2)
            self.assertFalse((root / "episode_state.json.pending").exists())

    def test_invalid_wal_target_is_rejected_before_ledger_mutation(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            original_pending_write = episode_module._write_pending_checkpoint

            def stop_after_wal(path: Path, value: dict) -> None:
                original_pending_write(path, value)
                raise OSError("injected stop after pending fsync")

            with patch(
                "quant_rabbit.fast_bot_episode._write_pending_checkpoint",
                side_effect=stop_after_wal,
            ):
                with self.assertRaises(OSError):
                    self._run(
                        root,
                        current=current,
                        now=later,
                        fast=fast,
                        slow=slow,
                    )

            ledger = root / "episode_ledger.jsonl"
            base_raw = ledger.read_bytes()
            pending_path = root / "episode_state.json.pending"
            pending = json.loads(pending_path.read_text())
            batch_event = json.loads(
                base64.b64decode(
                    pending["batch_jsonl_base64"],
                    validate=True,
                ).decode("utf-8")
            )
            batch_event["observation"]["source_timeframe_votes"]["H4"][
                "phase"
            ] = "RANGE"
            batch_event["observation"]["source_timeframe_votes_sha256"] = _sha(
                batch_event["observation"]["source_timeframe_votes"]
            )
            event_body = {
                key: item
                for key, item in batch_event.items()
                if key != "event_sha256"
            }
            batch_event["event_sha256"] = _sha(event_body)
            poisoned_batch = _ledger_line(batch_event)
            pending["batch_size_bytes"] = len(poisoned_batch)
            pending["batch_bytes_sha256"] = hashlib.sha256(
                poisoned_batch
            ).hexdigest()
            pending["batch_jsonl_base64"] = base64.b64encode(
                poisoned_batch
            ).decode("ascii")
            pending["target_head"]["size_bytes"] = len(base_raw) + len(
                poisoned_batch
            )
            pending["target_head"]["bytes_sha256"] = hashlib.sha256(
                base_raw + poisoned_batch
            ).hexdigest()
            pending["target_head"]["tail_sha256"] = batch_event["event_sha256"]
            pending_path.write_text(
                json.dumps(_seal(pending), sort_keys=True),
                encoding="utf-8",
            )

            result = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_SOURCE_ROW_MEMBERSHIP_INVALID",
                result["blockers"],
            )
            self.assertEqual(ledger.read_bytes(), base_raw)
            self.assertTrue(pending_path.exists())
            self.assertEqual(result["appended_events"], 0)
            self.assertNotIn(
                "EPISODE_PENDING_BATCH_RECONCILED",
                result["blockers"],
            )

    def test_valid_extension_without_matching_wal_is_rejected(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            original_write = episode_module._write_json_atomic

            def crash_before_state(
                path: Path,
                value: dict,
                **kwargs: object,
            ) -> None:
                if path.name == "episode_state.json":
                    raise OSError("injected state checkpoint crash")
                original_write(path, value, **kwargs)

            with patch(
                "quant_rabbit.fast_bot_episode._write_json_atomic",
                side_effect=crash_before_state,
            ):
                with self.assertRaises(OSError):
                    self._run(
                        root,
                        current=current,
                        now=later,
                        fast=fast,
                        slow=slow,
                    )
            (root / "episode_state.json.pending").unlink()

            result = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_LEDGER_HEAD_CHECKPOINT_MISMATCH",
                result["blockers"],
            )

    def test_archive_owner_prevents_cross_ledger_cleanup(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            regime = _regime(NOW, source_clocks=_chart_clocks(fast, slow))
            first = run_fast_bot_episode_shadow(
                regime_contract=regime,
                fast_pair_charts=fast,
                slow_pair_charts=slow,
                output_path=root / "first_state.json",
                ledger_path=root / "first_ledger.jsonl",
                source_archive_dir=root / "shared_sources",
                now_utc=NOW,
            )
            original_archive = next((root / "shared_sources").glob("*.json.gz"))

            second = run_fast_bot_episode_shadow(
                regime_contract=regime,
                fast_pair_charts=fast,
                slow_pair_charts=slow,
                output_path=root / "second_state.json",
                ledger_path=root / "second_ledger.jsonl",
                source_archive_dir=root / "shared_sources",
                now_utc=NOW,
            )

            self.assertEqual(first["status"], "UPDATED")
            self.assertEqual(second["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_SOURCE_ARCHIVE_OWNER_MISMATCH",
                second["blockers"],
            )
            self.assertTrue(original_archive.exists())

    def test_tampered_ledger_fails_closed_without_append(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            ledger = root / "episode_ledger.jsonl"
            event = json.loads(ledger.read_text().strip())
            event["route"]["trade_side"] = "SHORT"
            ledger.write_text(json.dumps(event) + "\n", encoding="utf-8")

            result = self._run(root, current=current, now=NOW + timedelta(minutes=1))

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertEqual(result["appended_events"], 0)
            self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_source_evidence_rehash_cannot_change_frozen_rail(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
            tampered = copy.deepcopy(event)
            tampered["source_m5_evidence"]["prior_low"] = 101.0
            evidence_body = {
                key: item
                for key, item in tampered["source_m5_evidence"].items()
                if key != "evidence_sha256"
            }
            tampered["source_m5_evidence"]["evidence_sha256"] = _sha(evidence_body)
            tampered["anchor"]["source_evidence_sha256"] = tampered["source_m5_evidence"]["evidence_sha256"]
            event_body = {key: item for key, item in tampered.items() if key != "event_sha256"}
            tampered["event_sha256"] = _sha(event_body)

            valid, error = verify_episode_event(tampered)

            self.assertFalse(valid)
            self.assertEqual(error, "EPISODE_SOURCE_EVIDENCE_INVALID")

    def test_rehashed_votes_cannot_escape_archived_regime_membership(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
            event["observation"]["source_timeframe_votes"]["H4"]["phase"] = "RANGE"
            event["observation"]["source_timeframe_votes_sha256"] = _sha(
                event["observation"]["source_timeframe_votes"]
            )
            body = {key: item for key, item in event.items() if key != "event_sha256"}
            event["event_sha256"] = _sha(body)

            self.assertEqual(
                verify_episode_ledger(
                    [event],
                    source_archive_dir=root / "episode_sources",
                ),
                (False, "EPISODE_SOURCE_ROW_MEMBERSHIP_INVALID"),
            )

    def test_missing_archived_regime_fails_closed_before_append(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            for path in (root / "episode_sources").iterdir():
                path.unlink()

            result = self._run(
                root,
                current=current,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn("EPISODE_SOURCE_ARCHIVE_MISSING", result["blockers"])
            self.assertEqual(result["appended_events"], 0)

    def test_genesis_branch_and_state_clocks_are_bound_to_evidence(self) -> None:
        current = {"o": 190.0, "h": 210.0, "l": 140.0, "c": 206.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            branch_tamper = copy.deepcopy(event)
            branch_tamper["route"]["branch_close"] = 205.0
            body = {
                key: item
                for key, item in branch_tamper.items()
                if key != "event_sha256"
            }
            branch_tamper["event_sha256"] = _sha(body)
            self.assertEqual(
                verify_episode_event(branch_tamper),
                (False, "EPISODE_SOURCE_EVIDENCE_INVALID"),
            )

            clock_tamper = copy.deepcopy(event)
            clock_tamper["state_entered_at_utc"] = (
                NOW + timedelta(minutes=1)
            ).isoformat()
            body = {
                key: item
                for key, item in clock_tamper.items()
                if key != "event_sha256"
            }
            clock_tamper["event_sha256"] = _sha(body)
            self.assertEqual(
                verify_episode_event(clock_tamper),
                (False, "EPISODE_EVENT_CLOCK_INVALID"),
            )

    def test_future_source_clock_is_rejected_even_after_rehash(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
            event["observation"]["source_timeframe_clocks"]["H4"] = (
                NOW + timedelta(days=1)
            ).isoformat()
            event["observation"]["source_timeframe_clocks_sha256"] = _sha(
                event["observation"]["source_timeframe_clocks"]
            )
            body = {key: item for key, item in event.items() if key != "event_sha256"}
            event["event_sha256"] = _sha(body)

            self.assertEqual(
                verify_episode_event(event),
                (False, "EPISODE_EVENT_CLOCK_INVALID"),
            )

    def test_context_newer_than_observation_is_quarantined_as_late(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)

            result = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            self.assertEqual(result["status"], "UPDATED")
            self.assertTrue(event["late_detected"])
            self.assertGreater(
                datetime.fromisoformat(
                    event["observation"]["source_timeframe_clocks"]["M1"]
                ),
                datetime.fromisoformat(event["observation"]["candle_close_utc"]),
            )

    def test_boolean_schema_version_is_not_integer_one(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())
            event["schema_version"] = True
            body = {key: item for key, item in event.items() if key != "event_sha256"}
            event["event_sha256"] = _sha(body)

            self.assertEqual(
                verify_episode_event(event),
                (False, "EPISODE_EVENT_SCHEMA_INVALID"),
            )

    def test_duplicate_json_key_fails_closed(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            ledger = root / "episode_ledger.jsonl"
            raw = ledger.read_text(encoding="utf-8")
            ledger.write_text(
                raw.replace("{", '{"contract":"DUPLICATE",', 1),
                encoding="utf-8",
            )

            result = self._run(root, current=current, now=NOW + timedelta(minutes=1))

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertEqual(result["appended_events"], 0)

    def test_missing_next_m1_cannot_be_skipped_for_later_confirmation(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            view = fast["charts"][0]["views"][0]
            rows = list(view["recent_candles"])
            rows.append(
                {
                    "t": (ATTEMPT_CLOSE + timedelta(minutes=1)).isoformat(),
                    "o": 107.0,
                    "h": 108.0,
                    "l": 106.0,
                    "c": 107.0,
                    "complete": True,
                    "v": 12,
                }
            )
            view["recent_candles"] = rows[-21:]
            later = NOW + timedelta(minutes=2)
            fast["generated_at_utc"] = later.isoformat()
            slow["generated_at_utc"] = later.isoformat()

            result = self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )

            self.assertEqual(result["appended_events"], 1)
            events = [
                json.loads(line)
                for line in (root / "episode_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[-1]["state"], "INVALIDATED")
            self.assertEqual(
                events[-1]["transition_reason"],
                "M1_SEQUENCE_GAP_UNOBSERVABLE",
            )

            m5_view = slow["charts"][0]["views"][0]
            m5_rows = list(m5_view["recent_candles"])
            m5_rows.append(
                {
                    "t": ATTEMPT_CLOSE.isoformat(),
                    "o": 190.0,
                    "h": 210.0,
                    "l": 140.0,
                    "c": 207.0,
                    "complete": True,
                }
            )
            m5_view["recent_candles"] = m5_rows[-21:]
            recovery_now = NOW + timedelta(minutes=5)
            fast["generated_at_utc"] = recovery_now.isoformat()
            slow["generated_at_utc"] = recovery_now.isoformat()
            recovered = self._run(
                root,
                current=current,
                now=recovery_now,
                fast=fast,
                slow=slow,
            )
            recovered_events = [
                json.loads(line)
                for line in (root / "episode_ledger.jsonl").read_text().splitlines()
            ]

            self.assertEqual(recovered["appended_events"], 1)
            self.assertEqual(len(recovered_events), 3)
            self.assertNotEqual(
                recovered_events[0]["episode_id"],
                recovered_events[-1]["episode_id"],
            )

    def test_global_chain_and_head_checkpoint_detect_deletion_and_reorder(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )
            ledger = root / "episode_ledger.jsonl"
            events = [json.loads(line) for line in ledger.read_text().splitlines()]

            self.assertEqual(events[0]["ledger_seq"], 1)
            self.assertEqual(events[1]["ledger_seq"], 2)
            self.assertEqual(
                events[1]["previous_ledger_event_sha256"],
                events[0]["event_sha256"],
            )
            self.assertEqual(
                verify_episode_ledger(
                    list(reversed(events)),
                    source_archive_dir=root / "episode_sources",
                ),
                (False, "EPISODE_GLOBAL_LEDGER_CHAIN_INVALID"),
            )

            ledger.write_bytes(_ledger_line(events[0]))
            result = self._run(
                root,
                current=current,
                now=later + timedelta(minutes=1),
                fast=fast,
                slow=slow,
            )

            self.assertEqual(result["status"], "LEDGER_INVALID")
            self.assertIn(
                "EPISODE_LEDGER_HEAD_CHECKPOINT_MISMATCH",
                result["blockers"],
            )

    def test_ledger_byte_cap_rejects_whole_batch_without_stopping_history(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            ledger = root / "episode_ledger.jsonl"
            original = ledger.read_bytes()
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)

            with patch(
                "quant_rabbit.fast_bot_episode.MAX_LEDGER_BYTES",
                len(original) + 10,
            ):
                result = self._run(
                    root,
                    current=current,
                    now=later,
                    fast=fast,
                    slow=slow,
                )

            self.assertEqual(result["status"], "LEDGER_CAP_REACHED")
            self.assertEqual(result["appended_events"], 0)
            self.assertIn("EPISODE_LEDGER_BYTE_CAP_REACHED", result["blockers"])
            self.assertEqual(ledger.read_bytes(), original)

    def test_one_invalid_pair_does_not_discard_another_valid_pair(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            gbp_fast = copy.deepcopy(fast["charts"][0])
            gbp_slow = copy.deepcopy(slow["charts"][0])
            gbp_fast["pair"] = "GBP_USD"
            gbp_slow["pair"] = "GBP_USD"
            fast["charts"].append(gbp_fast)
            slow["charts"].append(gbp_slow)
            fast["charts"][0]["views"][0]["recent_candles"] = []
            clocks = _chart_clocks(fast, slow)
            regime = _regime(NOW, source_clocks=clocks)
            body = {
                key: copy.deepcopy(item)
                for key, item in regime.items()
                if key != "contract_sha256"
            }
            gbp_row = copy.deepcopy(body["rows"][0])
            gbp_row["pair"] = "GBP_USD"
            body["rows"].append(gbp_row)
            regime = _seal(body)

            result = run_fast_bot_episode_shadow(
                regime_contract=regime,
                fast_pair_charts=fast,
                slow_pair_charts=slow,
                output_path=root / "episode_state.json",
                ledger_path=root / "episode_ledger.jsonl",
                source_archive_dir=root / "episode_sources",
                now_utc=NOW,
            )
            event = json.loads((root / "episode_ledger.jsonl").read_text().strip())

            self.assertEqual(result["appended_events"], 1)
            self.assertEqual(event["pair"], "GBP_USD")
            self.assertIn(
                "EUR_USD:PAIR_INPUT_INVALID:VALUEERROR",
                result["blockers"],
            )

    def test_future_ledger_clock_fails_closed(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root, current=current)
            ledger = root / "episode_ledger.jsonl"
            event = json.loads(ledger.read_text().strip())
            event["generated_at_utc"] = (NOW + timedelta(hours=1)).isoformat()
            body = {key: item for key, item in event.items() if key != "event_sha256"}
            event["event_sha256"] = _sha(body)
            self.assertEqual(
                verify_episode_ledger(
                    [event],
                    as_of_utc=NOW + timedelta(minutes=1),
                    source_archive_dir=root / "episode_sources",
                ),
                (False, "EPISODE_LEDGER_FUTURE_CLOCK"),
            )

    def test_rehashed_confirmation_semantic_tamper_is_rejected(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            fast["generated_at_utc"] = later.isoformat()
            slow["generated_at_utc"] = later.isoformat()
            self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )
            events = [
                json.loads(line)
                for line in (root / "episode_ledger.jsonl").read_text().splitlines()
            ]
            tampered = copy.deepcopy(events[1])
            tampered["state"] = "REJECTED"
            tampered["state_entered_at_utc"] = events[0]["state_entered_at_utc"]
            tampered["transition_path"] = ["REJECTED"]
            tampered["transition_reason"] = "M1_CONFIRMATION_PENDING"
            identity = {
                "episode_id": tampered["episode_id"],
                "event_seq": tampered["event_seq"],
                "state": tampered["state"],
                "observation_candle_utc": tampered["observation"]["candle_close_utc"],
                "previous_event_sha256": tampered["previous_event_sha256"],
            }
            tampered["event_id"] = _sha(identity)[:24]
            body = {key: item for key, item in tampered.items() if key != "event_sha256"}
            tampered["event_sha256"] = _sha(body)

            self.assertEqual(
                verify_episode_ledger(
                    [events[0], tampered],
                    source_archive_dir=root / "episode_sources",
                ),
                (False, "EPISODE_TRANSITION_INVALID"),
            )

    def test_rehashed_ledger_gap_is_rejected(self) -> None:
        current = {"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast, slow = _charts(current)
            self._run(root, current=current, fast=fast, slow=slow)
            _next_m1(fast, close=107.0)
            later = NOW + timedelta(minutes=1)
            fast["generated_at_utc"] = later.isoformat()
            slow["generated_at_utc"] = later.isoformat()
            self._run(
                root,
                current=current,
                now=later,
                fast=fast,
                slow=slow,
            )
            events = [
                json.loads(line)
                for line in (root / "episode_ledger.jsonl").read_text().splitlines()
            ]
            tampered = copy.deepcopy(events[1])
            shifted_start = datetime.fromisoformat(
                tampered["observation"]["candle"]["t"].replace("Z", "+00:00")
            ) + timedelta(minutes=1)
            tampered["observation"]["candle"]["t"] = shifted_start.isoformat().replace(
                "+00:00", "Z"
            )
            shifted_close = datetime.fromisoformat(
                tampered["observation"]["candle_close_utc"]
            ) + timedelta(minutes=1)
            tampered["observation"]["candle_close_utc"] = shifted_close.isoformat()
            tampered["observation"]["candle_sha256"] = _sha(
                tampered["observation"]["candle"]
            )
            tampered["state_entered_at_utc"] = shifted_close.isoformat()
            tampered["generated_at_utc"] = (
                shifted_close + timedelta(seconds=30)
            ).isoformat()
            identity = {
                "episode_id": tampered["episode_id"],
                "event_seq": tampered["event_seq"],
                "state": tampered["state"],
                "observation_candle_utc": tampered["observation"]["candle_close_utc"],
                "previous_event_sha256": tampered["previous_event_sha256"],
            }
            tampered["event_id"] = _sha(identity)[:24]
            body = {key: item for key, item in tampered.items() if key != "event_sha256"}
            tampered["event_sha256"] = _sha(body)

            self.assertEqual(
                verify_episode_ledger(
                    [events[0], tampered],
                    as_of_utc=later + timedelta(minutes=2),
                    source_archive_dir=root / "episode_sources",
                ),
                (False, "EPISODE_SOURCE_CYCLE_CLOCK_MISMATCH"),
            )


if __name__ == "__main__":
    unittest.main()
