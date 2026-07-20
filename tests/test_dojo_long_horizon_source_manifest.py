from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit import dojo_long_horizon_source_manifest as source_manifest
from quant_rabbit import fast_bot_historical_s5 as _history
from quant_rabbit.dojo_long_horizon_plan import (
    M1_CORE5_BINDING_ID,
    M1_FULL28_BINDING_ID,
    M5_BINDING_ID,
)
from quant_rabbit.dojo_long_horizon_source_manifest import (
    DojoLongHorizonSourceManifestError,
    _BindingSpec,
    _build_from_specs,
    _fixed_specs,
    _inventory,
    _resolve_equivalent_candidates,
    _spec_shards,
    _validate_structure,
    _verify_seal_for_specs,
    _write_canonical_manifest,
)


PAIR = "AUD_USD"
FETCH_SCRIPTS = {
    "M1": Path(__file__).resolve().parents[1] / "scripts/oanda_history_fetch_m1.py",
    "M5": Path(__file__).resolve().parents[1] / "scripts/oanda_history_fetch.py",
}


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def _row(
    *,
    timestamp: datetime,
    granularity: str,
    synthetic: bool = False,
    price_offset: float = 0.0,
) -> dict:
    row = {
        "ask": {
            "c": 1.1002 + price_offset,
            "h": 1.1003 + price_offset,
            "l": 1.1001 + price_offset,
            "o": 1.1002 + price_offset,
        },
        "bid": {
            "c": 1.1000 + price_offset,
            "h": 1.1001 + price_offset,
            "l": 1.0999 + price_offset,
            "o": 1.1000 + price_offset,
        },
        "complete": True,
        "granularity": granularity,
        "pair": PAIR,
        "price": "BA",
        "time": timestamp.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "volume": 1,
    }
    if synthetic:
        row["synthetic"] = True
    return row


def _write_shard(
    root: Path,
    *,
    granularity: str,
    start: datetime,
    end: datetime,
    duplicate_timestamp: bool = False,
    synthetic: bool = False,
    omit_index: int | None = None,
    fetch_script: Path | None = None,
    run_id: str = "20260719T000000Z",
    recorded_at_utc: str = "2026-07-19T00:00:00+00:00",
    price_offset: float = 0.0,
) -> Path:
    cadence = 60 if granularity == "M1" else 300
    run = root / run_id / PAIR
    run.mkdir(parents=True, exist_ok=True)
    path = run / (f"{PAIR}_{granularity}_BA_{_stamp(start)}_{_stamp(end)}.jsonl.gz")
    timestamps = []
    cursor = start
    while cursor < end:
        timestamps.append(cursor)
        cursor += timedelta(seconds=cadence)
    if duplicate_timestamp:
        timestamps.insert(2, timestamps[1])
    if omit_index is not None:
        timestamps.pop(omit_index)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for index, timestamp in enumerate(timestamps):
            handle.write(
                json.dumps(
                    _row(
                        timestamp=timestamp,
                        granularity=granularity,
                        synthetic=synthetic and index == 0,
                        price_offset=price_offset if index == 0 else 0.0,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    receipt = _write_receipt(
        root,
        path=path,
        start=start,
        end=end,
        rows=len(timestamps),
        fetch_script=fetch_script,
        recorded_at_utc=recorded_at_utc,
    )
    _write_summary(
        root,
        path=path,
        start=start,
        end=end,
        rows=len(timestamps),
        receipt_sha256=receipt["receipt_sha256"],
    )
    return path


def _write_receipt(
    root: Path,
    *,
    path: Path,
    start: datetime,
    end: datetime,
    rows: int,
    fetch_script: Path | None = None,
    recorded_at_utc: str = "2026-07-19T00:00:00+00:00",
) -> dict:
    raw = path.read_bytes()
    granularity = path.name.split("_")[2]
    selected_fetch_script = fetch_script or FETCH_SCRIPTS[granularity]
    fetch_sha = hashlib.sha256(selected_fetch_script.read_bytes()).hexdigest()
    ledger_path = root / _history.TRUTH_RECEIPT_FILE
    existing = (
        [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
        if ledger_path.exists()
        else []
    )
    body = {
        "schema_version": _history.TRUTH_RECEIPT_SCHEMA,
        "sequence": len(existing) + 1,
        "recorded_at_utc": recorded_at_utc,
        "output_root": str(root),
        "candle_path": str(path),
        "candle_sha256": hashlib.sha256(raw).hexdigest(),
        "pair": PAIR,
        "granularity": granularity,
        "price_component": "BA",
        "window": {
            "from_utc": start.isoformat(),
            "to_utc": end.isoformat(),
        },
        "rows": rows,
        "fetch_script_path": str(selected_fetch_script),
        "fetch_script_sha256": fetch_sha,
        "previous_receipt_sha256": (
            existing[-1]["receipt_sha256"] if existing else None
        ),
    }
    receipt = {**body, "receipt_sha256": _history._canonical_sha(body)}
    ledger_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in [*existing, receipt]
        ),
        encoding="utf-8",
    )
    return receipt


def _write_summary(
    root: Path,
    *,
    path: Path,
    start: datetime,
    end: datetime,
    rows: int,
    receipt_sha256: str,
) -> None:
    granularity = path.name.split("_")[2]
    cadence = 60 if granularity == "M1" else 300
    max_candles = 5_000
    chunk_seconds = cadence * (max_candles - 1)
    duration = int((end - start).total_seconds())
    requests = (duration + chunk_seconds - 1) // chunk_seconds
    task = {
        "compressed": True,
        "dry_run": False,
        "errors": [],
        "from": start.isoformat(),
        "granularity": granularity,
        "pair": PAIR,
        "partial_path": None,
        "path": str(path),
        "price": "BA",
        "published": True,
        "requests": requests,
        "rows": rows,
        "to": end.isoformat(),
        "truth_acquisition_receipt_sha256": receipt_sha256,
        "windows": requests,
    }
    summary = {
        "dry_run": False,
        "errors": [],
        "generated_at_utc": "2026-07-19T00:00:00+00:00",
        "granularities": [granularity],
        "max_candles_per_request": max_candles,
        "output_dir": str(path.parents[1]),
        "pairs": [PAIR],
        "price": "BA",
        "tasks": [task],
        "total_requests": requests,
        "total_rows": rows,
        "window": {"from": start.isoformat(), "to": end.isoformat()},
    }
    (path.parents[1] / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    m5 = tmp_path / "m5"
    m1 = tmp_path / "m1"
    m5.mkdir()
    m1.mkdir()
    return m5.resolve(), m1.resolve()


def _small_specs() -> tuple[_BindingSpec, ...]:
    return (
        _BindingSpec(M5_BINDING_ID, "M5", (PAIR,), ("2020-01",)),
        _BindingSpec(M1_CORE5_BINDING_ID, "M1", (PAIR,), ("2020-01",)),
    )


def test_fixed_binding_geometry_is_exact_and_keeps_contexts_separate() -> None:
    specs = _fixed_specs()
    assert [row.binding_id for row in specs] == [
        M5_BINDING_ID,
        M1_CORE5_BINDING_ID,
        M1_FULL28_BINDING_ID,
    ]
    assert len(_spec_shards(specs[0])) == 28 * 7
    assert len(_spec_shards(specs[1])) == 5 * 7
    assert len(_spec_shards(specs[2])) == 28 * 2
    unique = {shard.cell_key for spec in specs for shard in _spec_shards(spec)}
    assert len(unique) == 28 * 7 + 5 * 7 + (28 - 5) * 2 == 277


def test_deep_scan_seals_file_and_slice_evidence_without_trusting_summary(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(m1, granularity="M1", start=start, end=end)

    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())

    assert manifest["binding_count"] == 2
    assert manifest["physical_shard_count"] == 2
    assert manifest["summary_only_admission_allowed"] is False
    assert manifest["raw_rows_embedded"] is False
    for shard in manifest["physical_shards"]:
        assert shard["file_size_bytes"] > 0
        assert len(shard["file_sha256"]) == 64
        assert shard["full_file_row_count"] == shard["slice_row_count"]
        assert shard["slice_first_observed_utc"] == start.isoformat()
        assert shard["synthetic_row_count"] == 0
        assert shard["duplicate_timestamp_count"] == 0
        assert shard["observed_slot_coverage"] == 1.0
        assert shard["acquisition_receipt_sha256"]
        assert shard["request_window_completion_report_proved"] is True
        assert (
            shard["pair_month_coverage"][0]["missing_slot_legitimacy_proved"] is False
        )
        assert (
            shard["pair_month_coverage"][0]["calendar_open_quote_coverage_proved"]
            is False
        )
        assert shard["pair_month_coverage"][0]["coverage_cell_sha256"]
    inputs = manifest["plan_digest_inputs"]
    assert set(inputs["source_digests"]) == {M5_BINDING_ID, M1_CORE5_BINDING_ID}
    assert set(inputs["corpus_digests"]) == {M5_BINDING_ID, M1_CORE5_BINDING_ID}
    assert all(
        inputs["source_digests"][key] != inputs["corpus_digests"][key]
        for key in inputs["source_digests"]
    )
    for binding in manifest["bindings"]:
        assert binding["physical_shard_ids_sha256"]
        assert binding["month_pair_coverage_count"] == 1
        assert binding["month_pair_coverage_sha256"]
    _validate_structure(manifest)
    assert _verify_seal_for_specs(manifest, specs=_small_specs()) == manifest


def test_missing_duplicate_or_orphan_shard_fails_before_admission(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    with pytest.raises(DojoLongHorizonSourceManifestError, match="missing"):
        _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())

    original = _write_shard(m1, granularity="M1", start=start, end=end)
    duplicate = m1 / "20260719T000001Z" / PAIR / original.name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(DojoLongHorizonSourceManifestError, match="lacks.*receipt"):
        _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())

    duplicate.unlink()
    (m1 / "orphan.jsonl.gz").write_bytes(b"not-a-source")
    with pytest.raises(DojoLongHorizonSourceManifestError, match="orphan"):
        _inventory(m1, root_kind="M1")


@pytest.mark.parametrize(
    ("duplicate_timestamp", "synthetic", "message"),
    [
        (True, False, "duplicated or not increasing"),
        (False, True, "synthetic metadata"),
    ],
)
def test_duplicate_timestamp_and_explicit_synthetic_row_are_rejected(
    roots: tuple[Path, Path],
    duplicate_timestamp: bool,
    synthetic: bool,
    message: str,
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(
        m1,
        granularity="M1",
        start=start,
        end=end,
        duplicate_timestamp=duplicate_timestamp,
        synthetic=synthetic,
    )
    with pytest.raises(DojoLongHorizonSourceManifestError, match=message):
        _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())


def test_symlink_anywhere_in_source_root_is_rejected(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    _, m1 = roots
    outside = tmp_path / "outside"
    outside.mkdir()
    (m1 / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DojoLongHorizonSourceManifestError, match="symlink"):
        _inventory(m1, root_kind="M1")


def test_manifest_write_is_canonical_and_never_overwrites(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(m1, granularity="M1", start=start, end=end)
    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())
    output = (tmp_path / "source-manifest.json").resolve()

    _write_canonical_manifest(output, manifest)

    assert json.loads(output.read_text()) == manifest
    with pytest.raises(DojoLongHorizonSourceManifestError, match="already exists"):
        _write_canonical_manifest(output, manifest)


def test_self_rehashed_summary_tampering_is_not_a_valid_manifest(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(m1, granularity="M1", start=start, end=end)
    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())
    manifest["summary_only_admission_allowed"] = True
    body = {
        key: value for key, value in manifest.items() if key != "source_manifest_sha256"
    }
    manifest["source_manifest_sha256"] = _history._canonical_sha(body)

    with pytest.raises(DojoLongHorizonSourceManifestError, match="authority"):
        _validate_structure(manifest)


def test_missing_candle_slots_are_observed_but_never_declared_legitimate(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(
        m1,
        granularity="M1",
        start=start,
        end=end,
        omit_index=100,
    )

    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())
    m1_shard = next(
        row for row in manifest["physical_shards"] if row["granularity"] == "M1"
    )
    month = m1_shard["pair_month_coverage"][0]

    assert month["no_candle_slot_count"] == 1
    assert month["internal_gap_interval_count"] == 1
    assert month["missing_slot_legitimacy_proved"] is False
    assert month["calendar_open_quote_coverage_proved"] is False
    assert _verify_seal_for_specs(manifest, specs=_small_specs()) == manifest


def test_receipted_byte_equivalent_duplicate_selects_unique_latest_and_is_sealed(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    older = _write_shard(m1, granularity="M1", start=start, end=end)
    latest = _write_shard(
        m1,
        granularity="M1",
        start=start,
        end=end,
        run_id="20260719T000001Z",
        recorded_at_utc="2026-07-19T00:00:01+00:00",
    )

    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())

    assert manifest["duplicate_equivalence_record_count"] == 1
    record = manifest["duplicate_equivalence_records"][0]
    assert record["candidate_count"] == 2
    assert record["equivalence_proved"] is True
    assert record["selected_relative_path"] == latest.relative_to(m1).as_posix()
    assert {row["relative_path"] for row in record["candidates"]} == {
        older.relative_to(m1).as_posix(),
        latest.relative_to(m1).as_posix(),
    }
    assert (
        len({row["full_uncompressed_bytes_sha256"] for row in record["candidates"]})
        == 1
    )
    assert record["selected_physical_shard_id"] == next(
        row["physical_shard_id"]
        for row in manifest["physical_shards"]
        if row["granularity"] == "M1"
    )
    assert _verify_seal_for_specs(manifest, specs=_small_specs()) == manifest


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    [
        ("full_file_row_count", 99, "not byte/time/row equivalent"),
        (
            "full_uncompressed_bytes_sha256",
            "c" * 64,
            "not byte/time/row equivalent",
        ),
        (
            "acquisition_recorded_at_utc",
            "2026-07-19T00:00:00+00:00",
            "tie at the latest receipt clock",
        ),
    ],
)
def test_duplicate_difference_or_latest_clock_tie_fails_closed(
    changed_field: str,
    changed_value: object,
    message: str,
) -> None:
    shard = _spec_shards(_small_specs()[1])[0]
    base = {
        "relative_path": "20260719T000000Z/AUD_USD/source.jsonl.gz",
        "file_sha256": "a" * 64,
        "full_uncompressed_bytes_sha256": "b" * 64,
        "full_file_row_count": 100,
        "full_first_observed_utc": "2020-01-01T00:00:00+00:00",
        "full_last_observed_utc": "2020-01-31T23:59:00+00:00",
        "declared_from_utc": "2020-01-01T00:00:00+00:00",
        "declared_to_utc": "2020-02-01T00:00:00+00:00",
        "acquisition_receipt_sha256": "d" * 64,
        "acquisition_recorded_at_utc": "2026-07-19T00:00:00+00:00",
        "physical_shard_id": "e" * 64,
    }
    second = {
        **base,
        "relative_path": "20260719T000001Z/AUD_USD/source.jsonl.gz",
        "file_sha256": "f" * 64,
        "acquisition_receipt_sha256": "1" * 64,
        "acquisition_recorded_at_utc": "2026-07-19T00:00:01+00:00",
        "physical_shard_id": "2" * 64,
        changed_field: changed_value,
    }

    with pytest.raises(DojoLongHorizonSourceManifestError, match=message):
        _resolve_equivalent_candidates((base, second), shard=shard)


def test_terminal_acquisition_superset_is_parented_but_slice_stops_at_july(
    roots: tuple[Path, Path],
) -> None:
    m5, m1 = roots
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    slice_to = datetime(2026, 7, 1, tzinfo=timezone.utc)
    acquired_to = datetime(2026, 7, 10, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=acquired_to)
    _write_shard(m1, granularity="M1", start=start, end=acquired_to)
    specs = (
        _BindingSpec(
            M5_BINDING_ID,
            "M5",
            (PAIR,),
            ("2026-06",),
        ),
        _BindingSpec(
            M1_CORE5_BINDING_ID,
            "M1",
            (PAIR,),
            ("2026-06",),
        ),
    )

    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=specs)

    for shard in manifest["physical_shards"]:
        assert shard["declared_to_utc"] == acquired_to.isoformat()
        assert shard["slice_to_utc"] == slice_to.isoformat()
        assert shard["terminal_file_superset_seconds"] == 9 * 24 * 60 * 60
    assert _verify_seal_for_specs(manifest, specs=specs) == manifest


def test_self_receipted_alternate_fetcher_is_rejected(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    alternate = tmp_path / "oanda_history_fetch_m1.py"
    alternate.write_text("# alternate acquisition implementation\n", encoding="utf-8")
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(
        m1,
        granularity="M1",
        start=start,
        end=end,
        fetch_script=alternate,
    )

    with pytest.raises(
        DojoLongHorizonSourceManifestError,
        match="receipt provenance",
    ):
        _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())


def test_pure_seal_verifier_rejects_nested_tamper_without_source_io(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(m1, granularity="M1", start=start, end=end)
    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())
    monkeypatch.setattr(
        source_manifest,
        "_approved_fetcher",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected source I/O")),
    )

    assert _verify_seal_for_specs(manifest, specs=_small_specs()) == manifest
    manifest["physical_shards"][0]["pair_month_coverage"][0][
        "missing_slot_legitimacy_proved"
    ] = True
    unsigned = {
        key: value for key, value in manifest.items() if key != "source_manifest_sha256"
    }
    manifest["source_manifest_sha256"] = _history._canonical_sha(unsigned)

    with pytest.raises(
        DojoLongHorizonSourceManifestError,
        match="pair-month coverage relation",
    ):
        _verify_seal_for_specs(manifest, specs=_small_specs())


def test_failed_manifest_write_leaves_no_final_or_temp_artifact(
    roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m5, m1 = roots
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 2, 1, tzinfo=timezone.utc)
    _write_shard(m5, granularity="M5", start=start, end=end)
    _write_shard(m1, granularity="M1", start=start, end=end)
    manifest = _build_from_specs(m5_root=m5, m1_root=m1, specs=_small_specs())
    destination = (tmp_path / "failed-source-manifest.json").resolve()

    def _no_space(_fd: int, _payload: bytes) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", _no_space)
    with pytest.raises(OSError, match="No space"):
        _write_canonical_manifest(destination, manifest)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-source-manifest.json.tmp-*"))
