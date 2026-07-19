#!/usr/bin/env python3
"""Seal and execute a fail-closed DOJO bot TRAIN study.

The command never accepts replay economics on ``run``: the pair universe,
window, balance, bot configurations, cost arms, and corpus identity all come
from the sealed study.  Every main cell is paired with true leave-one-pair-out
replays.  Historical TRAIN output remains diagnostic only and grants no live
permission or order authority.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_bot_trainer import (  # noqa: E402
    CELL_CONTRACT,
    REQUIRED_COST_ARMS,
    REQUIRED_INTRABAR_PATHS,
    DojoBotTrainerError,
    evaluate_training,
    score_ledger_metrics,
    seal_cell_result,
    seal_study,
    verify_sealed_study,
)
from quant_rabbit.dojo_bot_catalog import (  # noqa: E402
    DojoBotCatalogError,
    validate_bot_config,
)
from quant_rabbit.dojo_market_calendar import (  # noqa: E402
    OANDA_FX_HOURS_POLICY,
    expected_oanda_fx_slots,
)


RUN_CONTRACT = "QR_DOJO_BOT_TRAINER_RUN_V1"
FAILURE_CONTRACT = "QR_DOJO_BOT_TRAINER_RUN_FAILURE_V1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MANDATORY_SOURCE_PATHS = frozenset(
    {
        "bots/lab_bot.py",
        "scripts/run-dojo-bot-trainer.py",
        "scripts/run-virtual-market-session.py",
        "src/quant_rabbit/dojo_bot_catalog.py",
        "src/quant_rabbit/dojo_bot_trainer.py",
        "src/quant_rabbit/dojo_lab_provenance.py",
        "src/quant_rabbit/dojo_market_calendar.py",
        "src/quant_rabbit/virtual_broker.py",
    }
)
_ZERO_SHA256 = "0" * 64
# A healthy multi-pair monthly M1 replay is expected to finish well inside two
# hours on the research host.  The fixed bound prevents a corrupt feed/worker
# from hanging the denominator forever; replace it with a measured high-percentile
# runtime bound if the corpus or host class changes materially.
REPLAY_TIMEOUT_SECONDS = 2 * 60 * 60
FULL_DAY_COVERAGE_FLOOR = 0.98
PARTIAL_DAY_COVERAGE_FLOOR = 0.80
FULL_DAY_MINIMUM_EXPECTED_SLOTS = 1000
MAX_OPEN_SLOT_GAP_SECONDS = 900


class TrainerRunnerError(ValueError):
    """Raised when orchestration cannot preserve the sealed TRAIN contract."""


def _reject_constant(value: str) -> None:
    raise TrainerRunnerError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainerRunnerError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _validate_json(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainerRunnerError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrainerRunnerError(f"{field} contains a non-string key")
            _validate_json(item, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json(item, field=f"{field}[{index}]")
        return
    raise TrainerRunnerError(f"{field} contains a non-JSON value")


def _strict_json_bytes(raw: bytes, *, field: str) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise TrainerRunnerError(f"{field} exceeds the JSON size limit")
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except TrainerRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrainerRunnerError(f"{field} is not strict JSON") from exc
    _validate_json(value, field=field)
    return value


def _load_json(path: Path, *, field: str) -> Any:
    if not path.is_file():
        raise TrainerRunnerError(f"{field} file is missing")
    return _strict_json_bytes(path.read_bytes(), field=field)


def _canonical_bytes(value: Any) -> bytes:
    _validate_json(value, field="output")
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainerRunnerError("output is not canonical JSON") from exc


def _exclusive_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_source_path(raw_path: str) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path:
        raise TrainerRunnerError("source path must be non-empty")
    posix = PurePosixPath(raw_path)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or posix.as_posix() != raw_path
        or raw_path.startswith("./")
    ):
        raise TrainerRunnerError("source path must be canonical and repo-relative")
    resolved = (REPO_ROOT / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise TrainerRunnerError("source path escapes repository") from exc
    if not resolved.is_file():
        raise TrainerRunnerError(f"source file is missing: {raw_path}")
    return raw_path, resolved


def _source_digests(raw_paths: Sequence[str]) -> dict[str, str]:
    if not raw_paths:
        raise TrainerRunnerError("at least one --source-path is required")
    normalized: dict[str, str] = {}
    for raw_path in raw_paths:
        relative, resolved = _repo_source_path(raw_path)
        if relative in normalized:
            raise TrainerRunnerError(f"duplicate source path: {relative}")
        normalized[relative] = _file_sha256(resolved)
    missing = sorted(MANDATORY_SOURCE_PATHS - set(normalized))
    if missing:
        raise TrainerRunnerError(
            "source closure is missing mandatory paths: " + ",".join(missing)
        )
    return dict(sorted(normalized.items()))


def _current_source_digests(sealed: Mapping[str, Any]) -> dict[str, str]:
    raw = sealed.get("source_digests")
    if not isinstance(raw, Mapping) or not raw:
        raise TrainerRunnerError("sealed source digest map is missing")
    return _source_digests(list(raw))


def _selected_shards(
    corpus_root: Path,
    pairs: Sequence[str],
    start_utc: str,
    end_utc: str,
) -> list[Path]:
    def parse(value: str) -> datetime:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TrainerRunnerError("sealed window timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise TrainerRunnerError("sealed window timestamp lacks an offset")
        return parsed.astimezone(timezone.utc)

    start = parse(start_utc)
    end = parse(end_utc)
    if end <= start:
        raise TrainerRunnerError("sealed replay window is empty")
    root = corpus_root.resolve()
    if not root.is_dir():
        raise TrainerRunnerError("corpus root is missing")
    shards: set[Path] = set()
    for year in range(start.year, end.year + 1):
        for pair in pairs:
            for candidate in root.glob(f"*/{pair}/{pair}_M1_BA_{year}*.jsonl.gz"):
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise TrainerRunnerError(
                        "corpus shard escapes corpus root"
                    ) from exc
                if not resolved.is_file():
                    raise TrainerRunnerError("selected corpus shard is not a file")
                shards.add(resolved)
    if not shards:
        raise TrainerRunnerError("corpus contains no selected M1 shards")
    for pair in pairs:
        if not any(path.parent.name == pair for path in shards):
            raise TrainerRunnerError(f"corpus is missing selected shards for {pair}")
    return sorted(shards)


def _window_bounds(start_utc: str, end_utc: str) -> tuple[datetime, datetime]:
    def parse(value: str) -> datetime:
        text_value = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text_value)
        except (TypeError, ValueError) as exc:
            raise TrainerRunnerError("sealed window timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise TrainerRunnerError("sealed window timestamp lacks an offset")
        return parsed.astimezone(timezone.utc)

    start = parse(start_utc)
    end = parse(end_utc)
    if end <= start:
        raise TrainerRunnerError("sealed replay window is empty")
    return start, end


def _m1_epoch(value: Any, *, pair: str, shard: Path, line_number: int) -> int:
    if not isinstance(value, str) or not value.strip():
        raise TrainerRunnerError(
            f"corpus timestamp is missing for {pair} at {shard.name}:{line_number}"
        )
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?" r"(Z|[+-]\d{2}:\d{2})",
        value,
    )
    if match is None:
        raise TrainerRunnerError(
            f"corpus timestamp is invalid for {pair} at {shard.name}:{line_number}"
        )
    head, fraction, zone = match.groups()
    text_value = head
    if fraction:
        text_value += "." + fraction[:6]
    text_value += "+00:00" if zone == "Z" else zone
    try:
        stamp = datetime.fromisoformat(text_value)
    except (TypeError, ValueError) as exc:
        raise TrainerRunnerError(
            f"corpus timestamp is invalid for {pair} at {shard.name}:{line_number}"
        ) from exc
    if stamp.tzinfo is None:
        raise TrainerRunnerError(
            f"corpus timestamp lacks an offset for {pair} at "
            f"{shard.name}:{line_number}"
        )
    stamp = stamp.astimezone(timezone.utc)
    if stamp.second != 0 or stamp.microsecond != 0:
        raise TrainerRunnerError(
            f"corpus timestamp is not M1-aligned for {pair} at "
            f"{shard.name}:{line_number}"
        )
    return int(stamp.timestamp())


def _validate_sparse_m1_corpus(
    shards: Sequence[Path],
    feed_pairs: Sequence[str],
    start_utc: str,
    end_utc: str,
) -> dict[str, Any]:
    """Seal bounded OANDA sparse coverage without inventing carry quotes."""

    start, end = _window_bounds(start_utc, end_utc)
    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())
    by_pair: dict[str, set[int]] = {pair: set() for pair in feed_pairs}
    pair_shards: dict[str, list[Path]] = {pair: [] for pair in feed_pairs}
    for shard in shards:
        pair = shard.parent.name
        if pair in pair_shards:
            pair_shards[pair].append(shard)
    for pair in feed_pairs:
        for shard in pair_shards[pair]:
            try:
                handle = gzip.open(shard, "rb")
            except OSError as exc:
                raise TrainerRunnerError(
                    f"selected corpus shard is not readable gzip: {shard.name}"
                ) from exc
            try:
                with handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        try:
                            row = _strict_json_bytes(
                                raw_line, field=f"corpus {shard.name}:{line_number}"
                            )
                        except (OSError, EOFError) as exc:
                            raise TrainerRunnerError(
                                f"selected corpus shard is corrupt: {shard.name}"
                            ) from exc
                        if not isinstance(row, Mapping):
                            raise TrainerRunnerError(
                                f"corpus row must be an object: "
                                f"{shard.name}:{line_number}"
                            )
                        epoch = _m1_epoch(
                            row.get("time"),
                            pair=pair,
                            shard=shard,
                            line_number=line_number,
                        )
                        if not start_epoch <= epoch < end_epoch:
                            continue
                        if epoch in by_pair[pair]:
                            raise TrainerRunnerError(
                                f"duplicate corpus M1 epoch for {pair}: {epoch}"
                            )
                        by_pair[pair].add(epoch)
            except (OSError, EOFError) as exc:
                raise TrainerRunnerError(
                    f"selected corpus shard is corrupt: {shard.name}"
                ) from exc

    expected_slots = expected_oanda_fx_slots(start, end, step=timedelta(minutes=1))
    expected_epochs = [int(stamp.timestamp()) for stamp in expected_slots]
    if not expected_epochs:
        raise TrainerRunnerError("selected M1 period has no OANDA open-market slots")
    expected_set = set(expected_epochs)
    unexpected = {
        pair: sorted(epochs - expected_set) for pair, epochs in by_pair.items()
    }
    unexpected = {pair: rows for pair, rows in unexpected.items() if rows}
    if unexpected:
        summary = ",".join(f"{pair}:{len(rows)}" for pair, rows in unexpected.items())
        raise TrainerRunnerError(
            "selected M1 corpus contains rows outside the canonical OANDA calendar: "
            + summary
        )

    expected_by_day: dict[str, list[int]] = {}
    for epoch in expected_epochs:
        day = datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()
        expected_by_day.setdefault(day, []).append(epoch)
    window_start_day = start.date()
    window_end_day = (end - timedelta(microseconds=1)).date()
    day_receipts: list[dict[str, Any]] = []
    expected_index = {epoch: index for index, epoch in enumerate(expected_epochs)}
    for day, day_epochs in sorted(expected_by_day.items()):
        day_value = datetime.fromisoformat(day).date()
        day_start = datetime.combine(day_value, time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        partial_window_day = (day_value == window_start_day and start > day_start) or (
            day_value == window_end_day and end < day_end
        )
        partial_day = (
            partial_window_day or len(day_epochs) < FULL_DAY_MINIMUM_EXPECTED_SLOTS
        )
        floor = PARTIAL_DAY_COVERAGE_FLOOR if partial_day else FULL_DAY_COVERAGE_FLOOR
        counts: dict[str, int] = {}
        ratios: dict[str, float] = {}
        for pair in feed_pairs:
            count = len(by_pair[pair].intersection(day_epochs))
            ratio = count / len(day_epochs)
            counts[pair] = count
            ratios[pair] = ratio
            if ratio < floor:
                raise TrainerRunnerError(
                    f"OANDA sparse M1 coverage below {floor:.0%} for "
                    f"{pair} on {day}: {count}/{len(day_epochs)}"
                )
        day_receipts.append(
            {
                "utc_date": day,
                "partial_day": partial_day,
                "partial_window_day": partial_window_day,
                "expected_slot_count": len(day_epochs),
                "coverage_floor": floor,
                "pair_row_counts": counts,
                "pair_coverage_ratios": ratios,
            }
        )

    for pair, observed in by_pair.items():
        indices = sorted(expected_index[epoch] for epoch in observed)
        if len(indices) < 2:
            raise TrainerRunnerError(
                f"selected M1 period requires at least two rows for {pair}"
            )
        boundary_missing = max(indices[0], len(expected_epochs) - indices[-1] - 1)
        interior_open_slot_age = max(
            (right - left) for left, right in zip(indices, indices[1:], strict=False)
        )
        if (
            max(boundary_missing * 60, interior_open_slot_age * 60)
            > MAX_OPEN_SLOT_GAP_SECONDS
        ):
            raise TrainerRunnerError(
                f"OANDA sparse M1 boundary/gap exceeds "
                f"{MAX_OPEN_SLOT_GAP_SECONDS}s for {pair}"
            )

    union_epochs = sorted(set().union(*by_pair.values()))
    availability_rows = [
        {
            "epoch": epoch,
            "batch_pairs": [pair for pair in feed_pairs if epoch in by_pair[pair]],
        }
        for epoch in union_epochs
    ]
    partial_epoch_count = sum(
        row["batch_pairs"] != list(feed_pairs) for row in availability_rows
    )
    pair_row_counts = {pair: len(by_pair[pair]) for pair in feed_pairs}
    return {
        "contract": "QR_DOJO_OANDA_SPARSE_M1_COVERAGE_V2",
        "calendar_policy": OANDA_FX_HOURS_POLICY,
        "coordinate_schedule": "SEALED_CORPUS_EPOCH_UNION",
        "quote_policy": "OBSERVED_ONLY_NO_SYNTHETIC_CARRY_QUOTES",
        "feed_pairs": list(feed_pairs),
        "expected_calendar_slot_count": len(expected_epochs),
        "expected_union_epoch_count": len(union_epochs),
        "expected_full_epoch_count": len(union_epochs) - partial_epoch_count,
        "expected_partial_epoch_count": partial_epoch_count,
        "expected_partial_phase_count": partial_epoch_count * 4,
        "pair_row_counts": pair_row_counts,
        "availability_mask_sha256": _canonical_sha256(availability_rows),
        "expected_quote_count": sum(pair_row_counts.values()) * 4,
        "max_carried_quote_age_seconds": MAX_OPEN_SLOT_GAP_SECONDS,
        "synthetic_quote_count": 0,
        "full_day_coverage_floor": FULL_DAY_COVERAGE_FLOOR,
        "partial_day_coverage_floor": PARTIAL_DAY_COVERAGE_FLOOR,
        "full_day_minimum_expected_slots": FULL_DAY_MINIMUM_EXPECTED_SLOTS,
        "daily_coverage": day_receipts,
        "first_epoch": union_epochs[0],
        "last_epoch": union_epochs[-1],
    }


def _corpus_manifest(
    corpus_root: Path,
    pairs: Sequence[str],
    start_utc: str,
    end_utc: str,
) -> dict[str, Any]:
    """Reproduce the virtual-session corpus identity from exact shard bytes."""

    root = corpus_root.resolve()
    feed_pairs = sorted(pairs)
    if not feed_pairs or len(feed_pairs) != len(set(feed_pairs)):
        raise TrainerRunnerError("feed pairs must be non-empty and unique")
    shards = _selected_shards(root, feed_pairs, start_utc, end_utc)
    coverage = _validate_sparse_m1_corpus(shards, feed_pairs, start_utc, end_utc)
    body = {
        "root": str(root),
        "shards": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in shards
        ],
    }
    return {
        **body,
        "corpus_sha256": _canonical_sha256(body),
        "sparse_m1_coverage": coverage,
    }


def _expected_coordinates(sealed: Mapping[str, Any]) -> list[dict[str, str]]:
    study = sealed["study"]
    return [
        {
            "candidate_id": candidate["candidate_id"],
            "intrabar": intrabar,
            "cost_arm": cost_arm,
        }
        for candidate in study["candidates"]
        for intrabar in REQUIRED_INTRABAR_PATHS
        for cost_arm in REQUIRED_COST_ARMS
    ]


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:1000]}"


def _failure_artifact(
    *,
    status: str,
    error: str,
    expected_coordinates: Sequence[Mapping[str, str]],
    study_sha256: str | None,
) -> dict[str, Any]:
    return {
        "contract": FAILURE_CONTRACT,
        "schema_version": 1,
        "status": status,
        "error": error,
        "study_sha256": study_sha256,
        "fixed_denominator": {
            "expected_cell_count": len(expected_coordinates),
            "coordinates": list(expected_coordinates),
            "dropped_cell_count": len(expected_coordinates),
            "coordinate_receipts_complete": False,
            "execution_success_complete": False,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }


def _preflight_run(
    sealed_path: Path, corpus_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _load_json(sealed_path, field="sealed study")
    if not isinstance(raw, Mapping):
        raise TrainerRunnerError("sealed study must be a JSON object")
    declared_sources = raw.get("source_digests")
    if not isinstance(declared_sources, Mapping):
        raise TrainerRunnerError("sealed study source digests are missing")
    try:
        normalized = verify_sealed_study(raw, declared_sources)
        current = _current_source_digests(normalized)
        normalized = verify_sealed_study(normalized, current)
    except DojoBotTrainerError as exc:
        raise TrainerRunnerError(str(exc)) from exc
    study = normalized["study"]
    for arm in REQUIRED_COST_ARMS:
        multiplier = study["cost_arms"][arm]["recorded_spread_multiplier"]
        if multiplier != 1.0:
            raise TrainerRunnerError(
                f"{arm} recorded_spread_multiplier must equal 1 because the "
                "virtual runner cannot widen recorded bid/ask spreads"
            )
    window = study["window"]
    corpus = _corpus_manifest(
        corpus_root,
        study["feed_pairs"],
        window["start_utc"],
        window["end_utc"],
    )
    if corpus["corpus_sha256"] != window["corpus_sha256"]:
        raise TrainerRunnerError("selected M1 corpus identity drift detected")
    return normalized, corpus


def _owner_id(
    study_sha256: str,
    candidate_id: str,
    intrabar: str,
    cost_arm: str,
    held_out_pair: str | None,
) -> str:
    material = "|".join(
        [study_sha256, candidate_id, intrabar, cost_arm, held_out_pair or "MAIN"]
    )
    return "dojo-trainer:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _runtime_config(
    config: Mapping[str, Any], *, pairs: Sequence[str], owner_id: str
) -> tuple[dict[str, Any], str]:
    counterfactual = json.loads(json.dumps(config, allow_nan=False))
    counterfactual["pairs"] = list(pairs)
    pair_cap = int(counterfactual["max_concurrent_per_pair"])
    counterfactual["global_max_concurrent"] = min(
        int(counterfactual["global_max_concurrent"]), pair_cap * len(pairs)
    )
    try:
        runtime = validate_bot_config(counterfactual)
    except DojoBotCatalogError as exc:
        raise TrainerRunnerError("derived replay config violates bot catalog") from exc
    runtime["strategy_owner_id"] = owner_id
    encoded = json.dumps(
        runtime,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return runtime, encoded


def _ledger_start(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise TrainerRunnerError("virtual replay did not create a ledger")
    with path.open("rb") as handle:
        raw = handle.readline()
    row = _strict_json_bytes(raw, field="SESSION_START ledger row")
    if not isinstance(row, Mapping) or row.get("event") != "SESSION_START":
        raise TrainerRunnerError("ledger does not begin with SESSION_START")
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        raise TrainerRunnerError("SESSION_START payload is missing")
    return payload


def _verify_replay_manifest(
    ledger_path: Path,
    *,
    pairs: Sequence[str],
    intrabar: str,
    cost: Mapping[str, Any],
    owner_id: str,
    config_json: str,
    expected_corpus_sha256: str,
    expected_sparse_coverage: Mapping[str, Any],
    source_digests: Mapping[str, str],
) -> None:
    payload = _ledger_start(ledger_path)
    if payload.get("order_authority") != "NONE":
        raise TrainerRunnerError("ledger order authority is not NONE")
    manifest = payload.get("reproducibility_manifest")
    if not isinstance(manifest, Mapping):
        raise TrainerRunnerError("ledger reproducibility manifest is missing")
    replay = manifest.get("replay")
    recorded_costs = manifest.get("costs")
    corpus = manifest.get("corpus")
    bot = manifest.get("bot")
    if not all(
        isinstance(item, Mapping) for item in (replay, recorded_costs, corpus, bot)
    ):
        raise TrainerRunnerError("ledger replay provenance is incomplete")
    assert isinstance(replay, Mapping)
    assert isinstance(recorded_costs, Mapping)
    assert isinstance(corpus, Mapping)
    assert isinstance(bot, Mapping)
    expected_replay = {
        "feed": "replay",
        "pairs": list(pairs),
        "granularity": "M1",
        "intrabar": intrabar,
        "bot_bar": "feed",
        "period_end_settlement": True,
        "continuous_mtm": True,
    }
    if any(replay.get(key) != value for key, value in expected_replay.items()):
        raise TrainerRunnerError("ledger replay coordinate drift detected")
    expected_mtm = {
        "mtm_coordinate_contract": "QR_REPLAY_ASYNC_MTM_COORDINATES_V2",
        "coordinate_schedule": "SEALED_CORPUS_EPOCH_UNION",
        "quote_policy": "OBSERVED_ONLY_NO_SYNTHETIC_CARRY_QUOTES",
        "feed_pairs": list(pairs),
        "expected_union_epoch_count": expected_sparse_coverage[
            "expected_union_epoch_count"
        ],
        "expected_full_epoch_count": expected_sparse_coverage[
            "expected_full_epoch_count"
        ],
        "expected_partial_epoch_count": expected_sparse_coverage[
            "expected_partial_epoch_count"
        ],
        "expected_phase_mark_count": expected_sparse_coverage[
            "expected_union_epoch_count"
        ]
        * 4,
        "expected_partial_phase_count": expected_sparse_coverage[
            "expected_partial_phase_count"
        ],
        "pair_row_counts": expected_sparse_coverage["pair_row_counts"],
        "availability_mask_sha256": expected_sparse_coverage[
            "availability_mask_sha256"
        ],
        "expected_quote_count": expected_sparse_coverage["expected_quote_count"],
        "max_carried_quote_age_seconds": expected_sparse_coverage[
            "max_carried_quote_age_seconds"
        ],
        "synthetic_quote_count": 0,
    }
    if any(replay.get(key) != value for key, value in expected_mtm.items()):
        raise TrainerRunnerError("ledger sparse MTM commitment drift detected")
    if float(recorded_costs.get("slippage_pips_per_fill", -1)) != float(
        cost["slippage_pips_per_fill"]
    ) or float(recorded_costs.get("financing_pips_per_day", -1)) != float(
        cost["financing_pips_per_day"]
    ):
        raise TrainerRunnerError("ledger cost arm drift detected")
    if float(recorded_costs.get("leverage", -1)) != 25.0:
        raise TrainerRunnerError("ledger leverage must equal the sealed 25.0")
    if corpus.get("corpus_sha256") != expected_corpus_sha256:
        raise TrainerRunnerError("ledger corpus identity drift detected")
    if (
        bot.get("kind") != "custom_module"
        or bot.get("class") != "Bot"
        or bot.get("strategy_owner_id") != owner_id
        or bot.get("module_sha256") != source_digests["bots/lab_bot.py"]
    ):
        raise TrainerRunnerError("ledger custom bot identity drift detected")
    bindings = bot.get("configuration_bindings")
    if not isinstance(bindings, Mapping):
        raise TrainerRunnerError("ledger bot configuration binding is missing")
    config_binding = bindings.get("DOJO_BOT_CONFIG")
    if (
        not isinstance(config_binding, Mapping)
        or set(config_binding) != {"sha256", "length"}
        or config_binding.get("sha256")
        != hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        or config_binding.get("length") != len(config_json)
    ):
        raise TrainerRunnerError("ledger bot configuration drift detected")
    dependency_sha = bot.get("dependency_sha256")
    if dependency_sha != dict(source_digests):
        raise TrainerRunnerError("ledger source dependency closure drift detected")


def _run_replay(
    *,
    sealed: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trade_pairs: Sequence[str],
    feed_pairs: Sequence[str],
    intrabar: str,
    cost_arm: str,
    held_out_pair: str | None,
    corpus_root: Path,
    session_dir: Path,
    expected_corpus_sha256: str,
    expected_sparse_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    session_dir.mkdir(parents=True, exist_ok=False)
    study = sealed["study"]
    cost = study["cost_arms"][cost_arm]
    owner = _owner_id(
        sealed["study_sha256"],
        candidate["candidate_id"],
        intrabar,
        cost_arm,
        held_out_pair,
    )
    runtime_config, config_json = _runtime_config(
        candidate["config"], pairs=trade_pairs, owner_id=owner
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run-virtual-market-session.py"),
        "--feed",
        "replay",
        "--session-dir",
        str(session_dir),
        "--pairs",
        ",".join(feed_pairs),
        "--balance",
        str(study["initial_balance_jpy"]),
        "--corpus-root",
        str(corpus_root.resolve()),
        "--from",
        study["window"]["start_utc"],
        "--to",
        study["window"]["end_utc"],
        "--bars-per-second",
        "1000000",
        "--bot-module",
        str(REPO_ROOT / "bots" / "lab_bot.py") + ":Bot",
        "--granularity",
        "M1",
        "--bot-bar",
        "feed",
        "--state-every",
        "1000000",
        "--fast-ledger",
        "--continuous-mtm",
        "--slippage-pips",
        str(cost["slippage_pips_per_fill"]),
        "--financing-pips-day",
        str(cost["financing_pips_per_day"]),
        "--intrabar",
        intrabar,
        "--strategy-owner-id",
        owner,
        "--settle-at-end",
    ]
    for relative in sealed["source_digests"]:
        command.extend(["--bot-dependency", relative])
    environment = os.environ.copy()
    environment.pop("DOJO_BOT_COMBO", None)
    environment["DOJO_BOT_CONFIG"] = config_json
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=REPLAY_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise TrainerRunnerError(
            "virtual replay failed "
            f"(exit={completed.returncode}, stderr={completed.stderr[-2000:]})"
        )
    ledger = session_dir / "ledger.jsonl"
    _verify_replay_manifest(
        ledger,
        pairs=feed_pairs,
        intrabar=intrabar,
        cost=cost,
        owner_id=owner,
        config_json=config_json,
        expected_corpus_sha256=expected_corpus_sha256,
        expected_sparse_coverage=expected_sparse_coverage,
        source_digests=sealed["source_digests"],
    )
    metrics = score_ledger_metrics(
        ledger,
        study["initial_balance_jpy"],
        trade_pairs,
        study["window"]["start_utc"],
        study["window"]["end_utc"],
        expected_intrabar=intrabar,
        expected_slippage_pips_per_fill=cost["slippage_pips_per_fill"],
        expected_financing_pips_per_day=cost["financing_pips_per_day"],
        expected_corpus_sha256=expected_corpus_sha256,
        expected_bot_config_sha256=hashlib.sha256(
            config_json.encode("utf-8")
        ).hexdigest(),
        expected_strategy_owner_id=owner,
        expected_bot_module_sha256=sealed["source_digests"]["bots/lab_bot.py"],
        expected_bot_dependency_sha256=sealed["source_digests"],
        expected_feed_pairs=feed_pairs,
        expected_max_concurrent_per_pair=runtime_config["max_concurrent_per_pair"],
        expected_global_max_concurrent=runtime_config["global_max_concurrent"],
    )
    return {
        "owner_id": owner,
        "session_dir": str(session_dir),
        "ledger_path": str(ledger),
        "metrics": metrics,
        "stdout_tail": completed.stdout[-2000:],
    }


def _failure_metrics(expected_pairs: Sequence[str]) -> dict[str, Any]:
    """A non-evidence denominator sentinel that deterministically fails gates."""

    zeros = {pair: 0.0 for pair in expected_pairs}
    return {
        "terminal_net_jpy": 0.0,
        "terminal_flat": False,
        "margin_closeouts": 0,
        "realized_max_drawdown_fraction": 0.0,
        "mtm_complete": False,
        "mtm_max_drawdown_fraction": None,
        "peak_margin_usage_fraction": 0.0,
        "fill_count": 0,
        "margin_reject_count": 0,
        "capital_lock_margin_jpy_hours": 0.0,
        "pair_pnl_jpy": dict(zeros),
        "leave_one_pair_out_net_jpy": dict(zeros),
        "lopo_replay_complete": False,
    }


def _empty_ledger_evidence(expected_pairs: Sequence[str]) -> dict[str, Any]:
    empty = {
        "artifact_relpath": None,
        "artifact_size_bytes": 0,
        "artifact_sha256": _ZERO_SHA256,
        "ledger_terminal_sha256": _ZERO_SHA256,
        "metrics_sha256": _ZERO_SHA256,
        "corpus_sha256": _ZERO_SHA256,
    }
    return {
        "main": dict(empty),
        "lopo_by_pair": {pair: dict(empty) for pair in expected_pairs},
    }


def _metric_evidence(
    metrics: Mapping[str, Any], *, ledger_path: str, artifact_root: Path
) -> dict[str, Any]:
    try:
        relative = (
            Path(ledger_path)
            .resolve(strict=True)
            .relative_to(artifact_root.resolve(strict=True))
        )
    except (OSError, ValueError) as exc:
        raise TrainerRunnerError("scored ledger is outside the artifact root") from exc
    if relative.name != "ledger.jsonl":
        raise TrainerRunnerError("scored ledger artifact name is invalid")
    terminal = metrics.get("ledger_terminal_sha256")
    score = metrics.get("metrics_sha256")
    artifact_sha = metrics.get("ledger_file_sha256")
    corpus_sha = metrics.get("corpus_sha256")
    for field, value in (
        ("artifact_sha256", artifact_sha),
        ("ledger_terminal_sha256", terminal),
        ("metrics_sha256", score),
        ("corpus_sha256", corpus_sha),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TrainerRunnerError(f"scored ledger has invalid {field}")
    return {
        "artifact_relpath": relative.as_posix(),
        "artifact_size_bytes": int(metrics["ledger_size_bytes"]),
        "artifact_sha256": str(artifact_sha),
        "ledger_terminal_sha256": str(terminal),
        "metrics_sha256": str(score),
        "corpus_sha256": str(corpus_sha),
    }


def _cell_metrics(
    main: Mapping[str, Any],
    *,
    expected_pairs: Sequence[str],
    lopo_net: Mapping[str, float],
    lopo_complete: bool,
) -> dict[str, Any]:
    return {
        "terminal_net_jpy": main["terminal_net_jpy"],
        "terminal_flat": main["terminal_flat"],
        "margin_closeouts": main["margin_closeouts"],
        "realized_max_drawdown_fraction": main["realized_max_drawdown_fraction"],
        "mtm_complete": main["mtm_complete"],
        "mtm_max_drawdown_fraction": main["mtm_max_drawdown_fraction"],
        "peak_margin_usage_fraction": main["peak_entry_margin_estimate_fraction"],
        "fill_count": main["fill_count"],
        "margin_reject_count": main["margin_reject_count"],
        "capital_lock_margin_jpy_hours": main["capital_lock_margin_jpy_hours"],
        "pair_pnl_jpy": {pair: main["pair_pnl_jpy"][pair] for pair in expected_pairs},
        "leave_one_pair_out_net_jpy": {
            pair: float(lopo_net.get(pair, 0.0)) for pair in expected_pairs
        },
        "lopo_replay_complete": lopo_complete,
    }


def _session_name(index: int, suffix: str) -> str:
    digest = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:12]
    return f"session-{index:04d}-{digest}"


def _execute_study(
    sealed: Mapping[str, Any],
    *,
    corpus_root: Path,
    output_dir: Path,
    full_corpus: Mapping[str, Any],
) -> dict[str, Any]:
    study = sealed["study"]
    expected_pairs = study["trade_pairs"]
    feed_pairs = study["feed_pairs"]
    coordinates = _expected_coordinates(sealed)
    candidate_map = {
        candidate["candidate_id"]: candidate for candidate in study["candidates"]
    }
    sealed_cells: list[dict[str, Any]] = []
    coordinate_receipts: list[dict[str, Any]] = []
    replay_ordinal = 0
    for coordinate in coordinates:
        candidate = candidate_map[coordinate["candidate_id"]]
        intrabar = coordinate["intrabar"]
        cost_arm = coordinate["cost_arm"]
        main_result: dict[str, Any] | None = None
        main_error: str | None = None
        replay_ordinal += 1
        main_dir = (
            output_dir
            / "sessions"
            / _session_name(
                replay_ordinal,
                f"{candidate['candidate_id']}|{intrabar}|{cost_arm}|MAIN",
            )
        )
        try:
            main_result = _run_replay(
                sealed=sealed,
                candidate=candidate,
                trade_pairs=expected_pairs,
                feed_pairs=feed_pairs,
                intrabar=intrabar,
                cost_arm=cost_arm,
                held_out_pair=None,
                corpus_root=corpus_root,
                session_dir=main_dir,
                expected_corpus_sha256=full_corpus["corpus_sha256"],
                expected_sparse_coverage=full_corpus["sparse_m1_coverage"],
            )
        except Exception as exc:  # each denominator coordinate must survive
            main_error = _safe_error(exc)
            _exclusive_write_json(
                main_dir / "failure.json",
                {
                    "contract": FAILURE_CONTRACT,
                    "schema_version": 1,
                    "status": "MAIN_REPLAY_FAILED",
                    "coordinate": coordinate,
                    "error": main_error,
                    "order_authority": "NONE",
                    "live_permission": False,
                },
            )

        lopo_net = {pair: 0.0 for pair in expected_pairs}
        ledger_evidence = _empty_ledger_evidence(expected_pairs)
        if main_result is not None:
            ledger_evidence["main"] = _metric_evidence(
                main_result["metrics"],
                ledger_path=main_result["ledger_path"],
                artifact_root=output_dir,
            )
        lopo_receipts: list[dict[str, Any]] = []
        lopo_complete = main_result is not None
        if main_result is not None:
            for held_out_pair in expected_pairs:
                replay_ordinal += 1
                subset = [pair for pair in expected_pairs if pair != held_out_pair]
                lopo_dir = (
                    output_dir
                    / "sessions"
                    / _session_name(
                        replay_ordinal,
                        f"{candidate['candidate_id']}|{intrabar}|{cost_arm}|{held_out_pair}",
                    )
                )
                try:
                    result = _run_replay(
                        sealed=sealed,
                        candidate=candidate,
                        trade_pairs=subset,
                        feed_pairs=feed_pairs,
                        intrabar=intrabar,
                        cost_arm=cost_arm,
                        held_out_pair=held_out_pair,
                        corpus_root=corpus_root,
                        session_dir=lopo_dir,
                        expected_corpus_sha256=full_corpus["corpus_sha256"],
                        expected_sparse_coverage=full_corpus["sparse_m1_coverage"],
                    )
                    metrics = result["metrics"]
                    if metrics.get("terminal_flat") is not True:
                        raise TrainerRunnerError("LOPO replay did not settle flat")
                    lopo_net[held_out_pair] = float(metrics["terminal_net_jpy"])
                    ledger_evidence["lopo_by_pair"][held_out_pair] = _metric_evidence(
                        metrics,
                        ledger_path=result["ledger_path"],
                        artifact_root=output_dir,
                    )
                    lopo_receipts.append(
                        {
                            "held_out_pair": held_out_pair,
                            "status": "VALID_COUNTERFACTUAL_REPLAY",
                            "terminal_net_jpy": lopo_net[held_out_pair],
                            "session_dir": result["session_dir"],
                            "ledger_path": result["ledger_path"],
                            "corpus_sha256": full_corpus["corpus_sha256"],
                        }
                    )
                except Exception as exc:
                    lopo_complete = False
                    error = _safe_error(exc)
                    _exclusive_write_json(
                        lopo_dir / "failure.json",
                        {
                            "contract": FAILURE_CONTRACT,
                            "schema_version": 1,
                            "status": "LOPO_REPLAY_FAILED",
                            "coordinate": coordinate,
                            "held_out_pair": held_out_pair,
                            "error": error,
                            "additive_substitute_used": False,
                            "order_authority": "NONE",
                            "live_permission": False,
                        },
                    )
                    lopo_receipts.append(
                        {
                            "held_out_pair": held_out_pair,
                            "status": "FAILED_NO_ADDITIVE_SUBSTITUTE",
                            "terminal_net_jpy": None,
                            "session_dir": str(lopo_dir),
                            "error": error,
                        }
                    )

        execution_failed = main_result is None or not lopo_complete
        if execution_failed:
            metrics = _failure_metrics(expected_pairs)
            ledger_evidence = _empty_ledger_evidence(expected_pairs)
        else:
            metrics = _cell_metrics(
                main_result["metrics"],
                expected_pairs=expected_pairs,
                lopo_net=lopo_net,
                lopo_complete=lopo_complete,
            )
        cell_body = {
            "contract": CELL_CONTRACT,
            "schema_version": 1,
            "study_sha256": sealed["study_sha256"],
            "candidate_id": candidate["candidate_id"],
            "proposal_sha256": candidate["proposal_sha256"],
            "intrabar": intrabar,
            "cost_arm": cost_arm,
            "execution_status": "FAILED" if execution_failed else "SUCCESS",
            "failure_code": (
                "MAIN_REPLAY_FAILED"
                if main_result is None
                else ("COUNTERFACTUAL_LOPO_INCOMPLETE" if not lopo_complete else None)
            ),
            "metrics": metrics,
            "ledger_evidence": ledger_evidence,
        }
        sealed_cell = seal_cell_result(cell_body, sealed, artifact_root=output_dir)
        sealed_cells.append(sealed_cell)
        coordinate_receipts.append(
            {
                **coordinate,
                "status": (
                    "MAIN_REPLAY_FAILED_SENTINEL"
                    if main_result is None
                    else (
                        "COMPLETE"
                        if lopo_complete
                        else "LOPO_INCOMPLETE_NO_ADDITIVE_SUBSTITUTE"
                    )
                ),
                "main_session_dir": str(main_dir),
                "main_error": main_error,
                "lopo_replay_complete": lopo_complete,
                "lopo": lopo_receipts,
                "cell_sha256": sealed_cell["cell_sha256"],
            }
        )

    current = _current_source_digests(sealed)
    try:
        verify_sealed_study(sealed, current)
    except DojoBotTrainerError as exc:
        raise TrainerRunnerError("source drift detected during study run") from exc
    evaluation = evaluate_training(sealed, sealed_cells, artifact_root=output_dir)
    _exclusive_write_json(output_dir / "cells.json", sealed_cells)
    _exclusive_write_json(output_dir / "evaluation.json", evaluation)
    failures = sum(row["status"] != "COMPLETE" for row in coordinate_receipts)
    body = {
        "contract": RUN_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "status": "COMPLETE" if failures == 0 else "COMPLETE_WITH_FAILED_CELLS",
        "corpus": dict(full_corpus),
        "fixed_denominator": {
            "expected_cell_count": len(coordinates),
            "observed_cell_count": len(sealed_cells),
            "failed_cell_count": failures,
            "dropped_cell_count": 0,
            "coordinate_receipts_complete": True,
            "execution_success_complete": failures == 0,
        },
        "coordinates": coordinate_receipts,
        "cells_path": str(output_dir / "cells.json"),
        "evaluation_path": str(output_dir / "evaluation.json"),
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    receipt = {**body, "run_sha256": _canonical_sha256(body)}
    _exclusive_write_json(output_dir / "run.json", receipt)
    return receipt


def _seal_command(args: argparse.Namespace) -> int:
    study = _load_json(args.study, field="study")
    if not isinstance(study, Mapping):
        raise TrainerRunnerError("study must be a JSON object")
    sources = _source_digests(args.source_path)
    try:
        sealed = seal_study(study, sources)
    except DojoBotTrainerError as exc:
        raise TrainerRunnerError(str(exc)) from exc
    normalized_study = sealed["study"]
    window = normalized_study["window"]
    corpus = _corpus_manifest(
        args.corpus_root,
        normalized_study["feed_pairs"],
        window["start_utc"],
        window["end_utc"],
    )
    if corpus["corpus_sha256"] != window["corpus_sha256"]:
        raise TrainerRunnerError("selected M1 corpus identity drift detected at seal")
    _exclusive_write_json(args.output, sealed)
    print(json.dumps({"status": "SEALED", "study_sha256": sealed["study_sha256"]}))
    return 0


def _run_command(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    sealed: dict[str, Any] | None = None
    coordinates: list[dict[str, str]] = []
    try:
        sealed, corpus = _preflight_run(args.sealed_study, args.corpus_root)
        coordinates = _expected_coordinates(sealed)
        receipt = _execute_study(
            sealed,
            corpus_root=args.corpus_root,
            output_dir=output_dir,
            full_corpus=corpus,
        )
    except Exception as exc:
        if sealed is None:
            try:
                raw = _load_json(args.sealed_study, field="sealed study")
                if isinstance(raw, Mapping):
                    declared = raw.get("source_digests")
                    if isinstance(declared, Mapping):
                        normalized = verify_sealed_study(raw, declared)
                        coordinates = _expected_coordinates(normalized)
                        sealed = normalized
            except Exception:
                pass
        failure = _failure_artifact(
            status="RUN_ABORTED_FAIL_CLOSED",
            error=_safe_error(exc),
            expected_coordinates=coordinates,
            study_sha256=sealed.get("study_sha256") if sealed else None,
        )
        _exclusive_write_json(output_dir / "run_failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    seal_parser = subcommands.add_parser("seal", help="seal a strict TRAIN study")
    seal_parser.add_argument("--study", type=Path, required=True)
    seal_parser.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="validate synchronized sealed M1 feed coverage before sealing",
    )
    seal_parser.add_argument(
        "--source-path",
        action="append",
        required=True,
        help="canonical repo-relative source path; repeat for the full closure",
    )
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.set_defaults(handler=_seal_command)

    run_parser = subcommands.add_parser("run", help="execute a sealed TRAIN study")
    run_parser.add_argument("--sealed-study", type=Path, required=True)
    run_parser.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="location only; selected shard identity must match the sealed study",
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.set_defaults(handler=_run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, TrainerRunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
