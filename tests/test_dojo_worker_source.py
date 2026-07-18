from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import quant_rabbit.dojo_worker_source as worker_source

from quant_rabbit.dojo_worker_source import (
    DojoWorkerSourceError,
    collect_and_seal_day,
    expected_open_slots,
    normalize_oanda_payload,
    verify_collected_day,
)


REPO = Path(__file__).resolve().parents[1]
FORWARD_RUN = REPO / "research/forward/dojo-worker-forward-smoke-v1"


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def candle(stamp: str, *, bid: str = "150.000", ask: str = "150.002") -> dict:
    return {
        "complete": True,
        "volume": 7,
        "time": stamp,
        "bid": {"o": bid, "h": bid, "l": bid, "c": bid},
        "ask": {"o": ask, "h": ask, "l": ask, "c": ask},
    }


def payload(*candles: dict) -> dict:
    return {
        "instrument": "USD_JPY",
        "granularity": "M1",
        "candles": list(candles),
    }


def complete_payload(day_start: str, day_end: str) -> dict:
    return payload(
        *[candle(stamp) for stamp in expected_open_slots(utc(day_start), utc(day_end))]
    )


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.base_url = "https://api-fxtrade.oanda.com"
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_json(self, path: str, query: dict[str, str]) -> dict:
        self.calls.append((path, query))
        return copy.deepcopy(self.response)


class FailingClient(FakeClient):
    def get_json(self, path: str, query: dict[str, str]) -> dict:
        self.calls.append((path, query))
        raise RuntimeError("simulated transport failure")


def make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy2(FORWARD_RUN / "precommit.json", run_dir / "precommit.json")
    shutil.copy2(FORWARD_RUN / "start.json", run_dir / "start.json")
    return run_dir


def test_expected_open_slots_are_dst_aware_for_registered_window() -> None:
    start = utc("2026-07-20T00:00:00Z")
    counts = [
        len(
            expected_open_slots(
                start + timedelta(days=day),
                start + timedelta(days=day + 1),
            )
        )
        for day in range(14)
    ]
    assert counts == [
        1436,
        1436,
        1436,
        1436,
        1260,
        0,
        176,
        1436,
        1436,
        1436,
        1436,
        1260,
        0,
        176,
    ]
    day1 = set(
        expected_open_slots(utc("2026-07-20T00:00:00Z"), utc("2026-07-21T00:00:00Z"))
    )
    assert "2026-07-20T20:58:00Z" in day1
    assert "2026-07-20T20:59:00Z" in day1
    assert "2026-07-20T21:00:00Z" not in day1
    assert "2026-07-20T21:04:00Z" in day1
    assert "2026-07-20T21:05:00Z" in day1


def test_collect_day_derives_exact_query_bytes_manifest_and_seal(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path)
    client = FakeClient(
        complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    )
    result = collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=client,
        now_utc=utc("2026-07-21T00:02:00Z"),
        repo_root=REPO,
        collector_paths=[REPO / "scripts/collect-dojo-worker-day.py"],
    )
    assert client.calls == [
        (
            "/v3/instruments/USD_JPY/candles",
            {
                "from": "2026-07-20T00:00:00Z",
                "granularity": "M1",
                "includeFirst": "true",
                "price": "BA",
                "to": "2026-07-21T00:00:00Z",
            },
        )
    ]
    assert result["state"] == "COLLECTING"
    assert result["source_row_count"] == 1436
    assert result["expected_open_slot_count"] == 1436
    assert result["missing_slot_count"] == 0
    assert result["promotion_eligible"] is False
    assert result["live_permission"] is False
    manifest = json.loads(
        (run_dir / "source-evidence/day-001/source-manifest.json").read_text()
    )
    source = manifest["sources"][0]
    corpus = list((run_dir / "corpus/day-001/USD_JPY").glob("*.jsonl.gz"))
    assert len(corpus) == 1
    assert corpus[0].stat().st_size == source["size_bytes"]
    assert verify_collected_day(run_dir, ordinal=1)["day_seal_sha256"]

    before = len(client.calls)
    repeated = collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=client,
        now_utc=utc("2026-07-21T00:03:00Z"),
        repo_root=REPO,
        collector_paths=[REPO / "scripts/collect-dojo-worker-day.py"],
    )
    assert len(client.calls) == before
    assert repeated["day_seal_sha256"] == result["day_seal_sha256"]


@pytest.mark.parametrize(
    "now",
    ["2026-07-21T00:01:59Z", "2026-07-21T12:00:01Z"],
)
def test_collect_rejects_outside_maturity_window_without_network(
    tmp_path: Path, now: str
) -> None:
    client = FakeClient(payload(candle("2026-07-20T00:00:00Z")))
    with pytest.raises(DojoWorkerSourceError):
        collect_and_seal_day(
            make_run(tmp_path),
            ordinal=1,
            client=client,
            now_utc=utc(now),
            repo_root=REPO,
        )
    assert client.calls == []


def test_response_crossing_deadline_is_not_backdated(tmp_path: Path) -> None:
    client = FakeClient(payload(candle("2026-07-20T00:00:00Z")))
    clocks = iter(
        [
            utc("2026-07-21T00:02:00Z"),
            utc("2026-07-21T12:00:01Z"),
        ]
    )
    with pytest.raises(DojoWorkerSourceError, match="response arrived after"):
        collect_and_seal_day(
            make_run(tmp_path),
            ordinal=1,
            client=client,
            repo_root=REPO,
            clock=lambda: next(clocks),
        )
    assert len(client.calls) == 1


def test_tampered_source_bytes_are_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=FakeClient(
            complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
        ),
        now_utc=utc("2026-07-21T00:02:00Z"),
        repo_root=REPO,
    )
    corpus = next((run_dir / "corpus/day-001/USD_JPY").glob("*.jsonl.gz"))
    data = bytearray(corpus.read_bytes())
    data[-1] ^= 1
    corpus.write_bytes(bytes(data))
    with pytest.raises(DojoWorkerSourceError, match="source bytes mismatch"):
        verify_collected_day(run_dir, ordinal=1)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda value: value["candles"][0].update(complete=False), "incomplete"),
        (lambda value: value["candles"][0].pop("ask"), "exact BA"),
        (lambda value: value["candles"][0]["bid"].update(h="149.000"), "high geometry"),
        (
            lambda value: value["candles"][0].update(
                ask={"o": "149.999", "h": "149.999", "l": "149.999", "c": "149.999"}
            ),
            "ask is below bid",
        ),
        (lambda value: value["candles"][0]["bid"].update(o="NaN"), "plain decimal"),
        (
            lambda value: value["candles"][0].update(time="2026-07-20T00:00:01Z"),
            "M1-aligned",
        ),
    ],
)
def test_payload_validation_rejects_malformed_rows(mutator, match: str) -> None:
    value = payload(candle("2026-07-20T00:00:00Z"))
    mutator(value)
    with pytest.raises(DojoWorkerSourceError, match=match):
        normalize_oanda_payload(
            value,
            day_start=utc("2026-07-20T00:00:00Z"),
            day_end=utc("2026-07-21T00:00:00Z"),
        )


def test_payload_validation_rejects_duplicate_and_unsorted_rows() -> None:
    duplicate = candle("2026-07-20T00:00:00Z")
    with pytest.raises(DojoWorkerSourceError, match="duplicate"):
        normalize_oanda_payload(
            payload(duplicate, copy.deepcopy(duplicate)),
            day_start=utc("2026-07-20T00:00:00Z"),
            day_end=utc("2026-07-21T00:00:00Z"),
        )


def test_open_day_subset_must_pass_fixed_coverage_gap_and_boundaries() -> None:
    with pytest.raises(DojoWorkerSourceError, match="coverage floor"):
        normalize_oanda_payload(
            payload(candle("2026-07-20T00:00:00Z")),
            day_start=utc("2026-07-20T00:00:00Z"),
            day_end=utc("2026-07-21T00:00:00Z"),
        )
    slots = expected_open_slots(
        utc("2026-07-20T00:00:00Z"), utc("2026-07-21T00:00:00Z")
    )
    sparse = [stamp for index, stamp in enumerate(slots) if index % 100]
    rows, coverage = normalize_oanda_payload(
        payload(*[candle(stamp) for stamp in sparse]),
        day_start=utc("2026-07-20T00:00:00Z"),
        day_end=utc("2026-07-21T00:00:00Z"),
    )
    assert len(rows) == len(sparse)
    assert coverage["missing_slot_count"] == len(slots) - len(sparse)
    assert coverage["coverage_passed"] is True
    gap = [stamp for stamp in slots if not "T12:00:" <= stamp[10:] <= "T12:20:"]
    with pytest.raises(DojoWorkerSourceError, match="contiguous-gap"):
        normalize_oanda_payload(
            payload(*[candle(stamp) for stamp in gap]),
            day_start=utc("2026-07-20T00:00:00Z"),
            day_end=utc("2026-07-21T00:00:00Z"),
        )
    with pytest.raises(DojoWorkerSourceError, match="chronological"):
        normalize_oanda_payload(
            payload(
                candle("2026-07-20T00:01:00Z"),
                candle("2026-07-20T00:00:00Z"),
            ),
            day_start=utc("2026-07-20T00:00:00Z"),
            day_end=utc("2026-07-21T00:00:00Z"),
        )


def test_only_full_closure_accepts_zero_rows() -> None:
    rows, coverage = normalize_oanda_payload(
        payload(),
        day_start=utc("2026-07-25T00:00:00Z"),
        day_end=utc("2026-07-26T00:00:00Z"),
    )
    assert rows == []
    assert coverage["expected_open_slot_count"] == 0
    with pytest.raises(DojoWorkerSourceError, match="coverage floor"):
        normalize_oanda_payload(
            payload(),
            day_start=utc("2026-07-26T00:00:00Z"),
            day_end=utc("2026-07-27T00:00:00Z"),
        )
    sunday_rows, sunday_coverage = normalize_oanda_payload(
        complete_payload("2026-07-26T00:00:00Z", "2026-07-27T00:00:00Z"),
        day_start=utc("2026-07-26T00:00:00Z"),
        day_end=utc("2026-07-27T00:00:00Z"),
    )
    assert len(sunday_rows) == 176
    assert sunday_coverage["expected_open_slot_count"] == 176


def test_request_and_receipt_do_not_persist_credentials(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = FakeClient(
        complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    )
    client.token = "SHOULD_NOT_APPEAR"
    client.account_id = "SHOULD_NOT_APPEAR_EITHER"
    collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=client,
        now_utc=utc("2026-07-21T00:02:00Z"),
        repo_root=REPO,
    )
    evidence = b"".join(
        path.read_bytes()
        for path in (run_dir / "source-evidence/day-001").iterdir()
        if path.is_file()
    )
    assert b"SHOULD_NOT_APPEAR" not in evidence


def test_source_directory_symlink_is_rejected_before_network(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "source-evidence").symlink_to(outside, target_is_directory=True)
    client = FakeClient(payload(candle("2026-07-20T00:00:00Z")))
    with pytest.raises(DojoWorkerSourceError, match="safe directory"):
        collect_and_seal_day(
            run_dir,
            ordinal=1,
            client=client,
            now_utc=utc("2026-07-21T00:02:00Z"),
            repo_root=REPO,
        )
    assert client.calls == []


def test_non_official_oanda_host_is_rejected_before_network(tmp_path: Path) -> None:
    client = FakeClient(
        complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    )
    client.base_url = "https://attacker.invalid"
    with pytest.raises(DojoWorkerSourceError, match="official OANDA"):
        collect_and_seal_day(
            make_run(tmp_path),
            ordinal=1,
            client=client,
            now_utc=utc("2026-07-21T00:02:00Z"),
            repo_root=REPO,
        )
    assert client.calls == []


def test_retry_cannot_drift_collector_source_bindings(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    first = FailingClient(
        complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    )
    with pytest.raises(RuntimeError, match="transport failure"):
        collect_and_seal_day(
            run_dir,
            ordinal=1,
            client=first,
            now_utc=utc("2026-07-21T00:02:00Z"),
            repo_root=REPO,
            collector_paths=[REPO / "scripts/collect-dojo-worker-day.py"],
        )
    second = FakeClient(first.response)
    with pytest.raises(DojoWorkerSourceError, match="source bindings differ"):
        collect_and_seal_day(
            run_dir,
            ordinal=1,
            client=second,
            now_utc=utc("2026-07-21T00:03:00Z"),
            repo_root=REPO,
            collector_paths=[
                REPO / "scripts/collect-dojo-worker-day.py",
                REPO / "tests/test_dojo_worker_source.py",
            ],
        )
    assert second.calls == []


def test_crash_before_receipt_resumes_first_captured_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = make_run(tmp_path)
    first_payload = complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    real_write = worker_source._write_pretty_json_new_or_same

    def fail_receipt(path: Path, value: dict) -> None:
        if path.name == "acquisition-receipt.json":
            raise RuntimeError("simulated receipt crash")
        real_write(path, value)

    monkeypatch.setattr(worker_source, "_write_pretty_json_new_or_same", fail_receipt)
    with pytest.raises(RuntimeError, match="receipt crash"):
        collect_and_seal_day(
            run_dir,
            ordinal=1,
            client=FakeClient(first_payload),
            now_utc=utc("2026-07-21T00:02:00Z"),
            repo_root=REPO,
        )
    assert (run_dir / "source-evidence/day-001/capture.json").is_file()

    monkeypatch.setattr(worker_source, "_write_pretty_json_new_or_same", real_write)
    revised = copy.deepcopy(first_payload)
    revised["candles"][0]["bid"] = {
        "o": "149.000",
        "h": "149.000",
        "l": "149.000",
        "c": "149.000",
    }
    revised["candles"][0]["ask"] = {
        "o": "149.002",
        "h": "149.002",
        "l": "149.002",
        "c": "149.002",
    }
    second = FakeClient(revised)
    result = collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=second,
        now_utc=utc("2026-07-21T00:03:00Z"),
        repo_root=REPO,
    )
    assert second.calls == []
    assert result["source_row_count"] == 1436


def test_day_two_cannot_change_day_one_collector_or_coverage_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = make_run(tmp_path)
    collect_and_seal_day(
        run_dir,
        ordinal=1,
        client=FakeClient(
            complete_payload("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
        ),
        now_utc=utc("2026-07-21T00:02:00Z"),
        repo_root=REPO,
    )
    real_checks = worker_source._collector_source_checks

    def changed_checks(repo_root: Path, collector_paths: tuple[Path, ...]) -> list[dict]:
        rows = real_checks(repo_root, collector_paths)
        rows[0] = {**rows[0], "sha256": "f" * 64}
        return rows

    monkeypatch.setattr(worker_source, "_collector_source_checks", changed_checks)
    with pytest.raises(DojoWorkerSourceError, match="immutable day-1 lock"):
        collect_and_seal_day(
            run_dir,
            ordinal=2,
            client=FakeClient(
                complete_payload(
                    "2026-07-21T00:00:00Z", "2026-07-22T00:00:00Z"
                )
            ),
            now_utc=utc("2026-07-22T00:02:00Z"),
            repo_root=REPO,
        )
    capture = json.loads((run_dir / "source-evidence/day-001/capture.json").read_text())
    assert capture["response"]["candles"][0]["bid"]["o"] == "150.000"
