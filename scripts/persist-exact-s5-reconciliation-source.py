#!/usr/bin/env python3
"""Persist a compact reviewed source from the independent TRAIN audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT = "QR_EXACT_S5_TRAIN_RECONCILIATION_SOURCE_V1"
EXPECTED_SCENARIOS = frozenset(
    {
        "prior_exact_anchor",
        "legacy_three_changes",
        "execution_gap300_only",
        "stale_signal_only",
        "ready4_only",
        "decision_due_only",
        "fixed_continuous_hold",
    }
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _load_object(args.independent_summary)
    manifest = _load_object(args.manifest)
    _verify_digest(summary, "summary_sha256")
    _verify_digest(manifest, "manifest_sha256")
    _validate_sources(summary, manifest)

    body: dict[str, Any] = {
        "artifact": ARTIFACT,
        "source_summary_file_sha256": _file_sha(args.independent_summary),
        "source_summary_sha256": summary["summary_sha256"],
        "source_summary_contract": summary["contract"],
        "source_manifest_path": str(args.manifest.resolve()),
        "source_manifest_file_sha256": _file_sha(args.manifest),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "independent_of_adaptive_engine": True,
        "train_from_utc": summary["train_from_utc"],
        "train_to_utc": summary["train_to_utc"],
        "pair_count": summary["pair_count"],
        "anchor_contract": summary["anchor_contract"],
        "scenarios": summary["scenarios"],
        "source_checks": summary["source_checks"],
        "prior_anchor_ledger_sha256": summary["ledger_sha256"],
        "legacy_three_changes_ledger_sha256": summary[
            "legacy_three_changes_ledger_sha256"
        ],
        "raw_trade_rows_persisted": False,
        "accepted_positive_result": False,
        "invalidated_provisional_lock": {
            "label": "DIRECT_TOP_BOTTOM_1_RETURN_8H_OVER_ABS_1H_CADENCE_4H_HOLD_12H_DISPERSION_5",
            "trade_count": 242,
            "raw_net_pips": 386.7,
            "profit_factor": 1.156103665,
            "stressed_net_pips": 265.7,
            "leave_best_day_stressed_net_pips": 19.1,
            "leave_best_pair_stressed_net_pips": 83.9,
            "long_stressed_net_pips": 193.4,
            "short_stressed_net_pips": 72.3,
            "status": "INVALID_EXECUTION_GAP_SURVIVORSHIP_CONTRACT",
            "stale_lock_hash_accepted": False,
        },
        "holdout_state": {
            "july_10_17_manifest_integrity_bytes_read": True,
            "july_10_17_strategy_evaluated": False,
            "july_20_august_3_source_available": False,
            "july_20_august_3_opened": False,
        },
        "historical_only": True,
        "diagnostic_only": True,
        "forward_proof_eligible": False,
        "promotion_allowed": False,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    artifact = {**body, "artifact_sha256": _canonical_sha(body)}
    _atomic_json(args.output, artifact)
    return artifact


def _validate_sources(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if summary.get("adaptive_engine_imported") is not False:
        raise ValueError("independent summary imported the disputed engine")
    if summary.get("independent_of_adaptive_engine") is not True:
        raise ValueError("independent summary boundary is missing")
    if summary.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("independent summary manifest identity mismatch")
    if summary.get("pair_count") != 28 or len(summary.get("source_checks", [])) != 28:
        raise ValueError("independent summary exact-28 coverage mismatch")
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != EXPECTED_SCENARIOS:
        raise ValueError("independent scenario set mismatch")
    prior = scenarios["prior_exact_anchor"]["metrics"]
    legacy = scenarios["legacy_three_changes"]["metrics"]
    gap = scenarios["execution_gap300_only"]["metrics"]
    if prior.get("trade_count") != 524 or prior.get("net_pips") != -747.6:
        raise ValueError("prior exact anchor metrics drifted")
    if legacy.get("trade_count") != 484 or legacy.get("net_pips") != 689.4:
        raise ValueError("legacy reproduction metrics drifted")
    if gap.get("trade_count") != 464 or gap.get("net_pips") != 691.2:
        raise ValueError("execution-gap isolation metrics drifted")


def _verify_digest(value: Mapping[str, Any], digest_key: str) -> None:
    body = {key: item for key, item in value.items() if key != digest_key}
    if value.get(digest_key) != _canonical_sha(body):
        raise ValueError(f"source digest mismatch: {digest_key}")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON source must be an object")
    return value


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = run(args)
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "output": str(args.output.resolve()),
                "artifact_sha256": artifact["artifact_sha256"],
                "accepted_positive_result": False,
                "raw_trade_rows_persisted": False,
                "validation_accessed": False,
                "order_authority": "NONE",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
