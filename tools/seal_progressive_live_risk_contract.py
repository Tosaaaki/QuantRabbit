#!/usr/bin/env python3
"""Seal an explicitly approved progressive-live release without broker access.

The tool cannot place, stage, or route an order.  It converts one immutable
approval candidate plus a healthy zero-authority resident-shadow readback into
content-addressed admission and risk contracts.  Account sizing and the single
Gateway remain separate, later gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.fast_bot_promotion import (
    EXTERNAL_MUTATION_GATEWAY,
    FORWARD_ADMISSION_CONTRACT,
    RISK_CONTRACT,
    seal_forward_admission,
    seal_risk_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_CONTRACT = "QR_PROGRESSIVE_LIVE_RISK_APPROVAL_PACKET_V1"
RELEASE_CONTRACT = "QR_PROGRESSIVE_LIVE_RELEASE_RECEIPT_V1"
LIVE_SOURCE_PATHS = (
    Path("src/quant_rabbit/fast_bot.py"),
    Path("src/quant_rabbit/fast_bot_promotion.py"),
    Path("src/quant_rabbit/inventory_controller.py"),
    Path("src/quant_rabbit/trade_readiness.py"),
    Path("src/quant_rabbit/broker/execution.py"),
    Path("src/quant_rabbit/broker/oanda.py"),
    Path("src/quant_rabbit/broker/position_execution.py"),
    Path("tools/run_progressive_live_preflight.py"),
    Path("tools/run_progressive_live_owner_cycle.py"),
    Path("tools/seal_progressive_live_supervision_receipt.py"),
    Path("config/fast_bot_progressive_strategy_profile_v1.json"),
    Path("config/oanda_spread_calibration_v1.json"),
    Path("config/oanda_spread_calibration_source_v1.json.gz"),
    Path("config/qr_progressive_live_risk_approval_packet_v1.json"),
)


class SealBlocked(RuntimeError):
    """The approval, release, or resident evidence is not exact enough to seal."""


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealBlocked(f"JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise SealBlocked(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def verify_approval_packet(
    packet: Mapping[str, Any], *, expected_packet_sha256: str
) -> dict[str, Any]:
    if packet.get("contract") != PACKET_CONTRACT:
        raise SealBlocked("APPROVAL_PACKET_CONTRACT_INVALID")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    actual = canonical_sha(body)
    if actual != packet.get("packet_sha256") or actual != expected_packet_sha256:
        raise SealBlocked("APPROVAL_PACKET_SHA256_MISMATCH")
    if (
        packet.get("status") != "NEEDS_USER_DECISION"
        or packet.get("live_permission") is not False
        or packet.get("broker_mutation_allowed") is not False
        or not isinstance(packet.get("candidate_limits"), Mapping)
    ):
        raise SealBlocked("APPROVAL_PACKET_STATE_INVALID")
    pairs = packet.get("initial_pairs")
    strategies = packet.get("allowed_strategy_ids")
    if pairs != ["EUR_USD", "USD_JPY"] or not isinstance(strategies, list) or not strategies:
        raise SealBlocked("APPROVAL_UNIVERSE_INVALID")
    return dict(packet)


def verify_resident_shadow(status: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_state": "RUNNING",
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "promotion_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise SealBlocked("RESIDENT_SHADOW_AUTHORITY_OR_STATE_INVALID")
    if status.get("last_error") not in (None, ""):
        raise SealBlocked("RESIDENT_SHADOW_ERROR_PRESENT")
    source_commit = str(status.get("source_commit") or "")
    source_sha = str(status.get("source_bundle_sha256") or "")
    if len(source_commit) != 40 or len(source_sha) != 64:
        raise SealBlocked("RESIDENT_SHADOW_SEAL_INVALID")
    return {
        "source_commit": source_commit,
        "source_bundle_sha256": source_sha,
        "pid": status.get("pid"),
        "started_at_utc": status.get("started_at_utc"),
        "heartbeat_at_utc": status.get("heartbeat_at_utc"),
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def software_manifest(repo_root: Path = REPO_ROOT, *, require_clean: bool = True) -> dict[str, Any]:
    try:
        top = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
        commit = _git(repo_root, "rev-parse", "HEAD")
        tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
        dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    except (subprocess.CalledProcessError, OSError) as exc:
        raise SealBlocked("SOFTWARE_GIT_READBACK_FAILED") from exc
    if top != repo_root.resolve():
        raise SealBlocked("SOFTWARE_REPO_TOP_MISMATCH")
    if require_clean and dirty:
        raise SealBlocked("SOFTWARE_WORKTREE_DIRTY")
    files: dict[str, str] = {}
    for relative in LIVE_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise SealBlocked(f"SOFTWARE_SOURCE_MISSING:{relative}")
        files[str(relative)] = sha256_file(path)
    binding = {"git_tree": tree, "files": files}
    return {
        "commit": commit,
        "git_tree": tree,
        "files": files,
        "software_version_sha256": canonical_sha(binding),
    }


def _positive_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SealBlocked(f"LIMIT_INVALID:{name}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise SealBlocked(f"LIMIT_INVALID:{name}")
    return parsed


def build_release_receipt(
    *,
    packet: Mapping[str, Any],
    expected_packet_sha256: str,
    resident_status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    acceptance_id: str,
    accepted_at_utc: datetime,
    live_campaign_id: str,
) -> dict[str, Any]:
    approved = verify_approval_packet(
        packet,
        expected_packet_sha256=expected_packet_sha256,
    )
    resident = verify_resident_shadow(resident_status)
    acceptance = str(acceptance_id or "").strip()
    if not acceptance:
        raise SealBlocked("EXPLICIT_ACCEPTANCE_ID_REQUIRED")
    if accepted_at_utc.tzinfo is None or accepted_at_utc.utcoffset() is None:
        raise SealBlocked("ACCEPTED_AT_MUST_BE_TIMEZONE_AWARE")
    accepted_at = accepted_at_utc.astimezone(timezone.utc)
    campaign = str(live_campaign_id or "")
    if not campaign.startswith("live-fb-"):
        raise SealBlocked("LIVE_CAMPAIGN_ID_INVALID")
    software_sha = str(manifest.get("software_version_sha256") or "")
    if len(software_sha) != 64:
        raise SealBlocked("SOFTWARE_VERSION_SHA256_INVALID")

    admission = seal_forward_admission(
        {
            "contract": FORWARD_ADMISSION_CONTRACT,
            "status": "ADMITTED",
            "promotion_allowed": True,
            "live_permission": True,
            "external_mutation_gateway": EXTERNAL_MUTATION_GATEWAY,
            "software_version_sha256": software_sha,
            "allowed_strategy_ids": list(approved["allowed_strategy_ids"]),
            "allowed_pairs": list(approved["initial_pairs"]),
            "admission_mode": "PROGRESSIVE_MICRO_LIVE",
            "progressive_live_user_authorized": True,
            "authorization_source": "EXPLICIT_USER_DECISION",
            "authorization_id": acceptance,
            "resident_shadow_required": True,
            "resident_shadow_status": "RUNNING",
            "resident_shadow_execution_authority": "NONE",
            "resident_shadow_broker_mutation_count": 0,
            "resident_shadow_external_order_attempts": 0,
            "resident_shadow_external_orders": 0,
            "scorecard_monitoring_active": True,
            "scorecard_can_force_demotion": True,
            "fixed_sample_wait_required_for_micro_live": False,
            "micro_live_only": True,
            "independent_readback_verified": True,
            "resolved_fills": 0,
            "active_days": 0,
        }
    )
    limits = approved["candidate_limits"]
    current_mcp = _positive_float(
        limits.get("max_post_entry_current_mcp"), "max_post_entry_current_mcp"
    )
    stress_mcp = _positive_float(
        limits.get("max_post_entry_stress_mcp"), "max_post_entry_stress_mcp"
    )
    hysteresis = _positive_float(
        limits.get("mode_hysteresis_mcp"), "mode_hysteresis_mcp"
    )
    max_positions = limits.get("max_bot_positions")
    if (
        not isinstance(max_positions, int)
        or isinstance(max_positions, bool)
        or max_positions <= 0
        or not 0.0 < current_mcp < stress_mcp < 1.0
        or not 0.0 < hysteresis < current_mcp
    ):
        raise SealBlocked("LIMIT_RELATION_INVALID")
    risk = seal_risk_contract(
        {
            "contract": RISK_CONTRACT,
            "status": "ACCEPTED",
            "accepted_by_user": True,
            "acceptance_source": "EXPLICIT_USER_DECISION",
            "acceptance_id": acceptance,
            "accepted_at_utc": accepted_at.isoformat(),
            "software_version_sha256": software_sha,
            "forward_admission_sha256": admission["admission_sha256"],
            "live_campaign_id": campaign,
            "max_loss_per_order_jpy": _positive_float(
                limits.get("max_loss_per_order_jpy"), "max_loss_per_order_jpy"
            ),
            "stop_drawdown_jpy": _positive_float(
                limits.get("stop_drawdown_jpy"), "stop_drawdown_jpy"
            ),
            "minimum_margin_buffer_jpy": _positive_float(
                limits.get("minimum_margin_buffer_jpy"), "minimum_margin_buffer_jpy"
            ),
            "max_post_entry_current_mcp": current_mcp,
            "max_post_entry_stress_mcp": stress_mcp,
            "max_currency_factor_nav_multiple": _positive_float(
                limits.get("max_currency_factor_nav_multiple"),
                "max_currency_factor_nav_multiple",
            ),
            "max_bot_positions": max_positions,
            "mode_hysteresis_mcp": hysteresis,
            "stress_pips": _positive_float(limits.get("stress_pips"), "stress_pips"),
            "max_account_snapshot_age_seconds": _positive_float(
                limits.get("max_account_snapshot_age_seconds"),
                "max_account_snapshot_age_seconds",
            ),
        }
    )
    body = {
        "contract": RELEASE_CONTRACT,
        "status": "SEALED_AWAITING_FRESH_ACCOUNT_GATE",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_packet_sha256": expected_packet_sha256,
        "acceptance_id": acceptance,
        "software_manifest": dict(manifest),
        "resident_shadow_readback": resident,
        "forward_admission": admission,
        "risk_contract": risk,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "next_required_gate": "FRESH_ACCOUNT_WIDE_SIZING_AND_PROMOTION",
    }
    return {**body, "release_receipt_sha256": canonical_sha(body)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SealBlocked("ACCEPTED_AT_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SealBlocked("ACCEPTED_AT_MUST_BE_TIMEZONE_AWARE")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--resident-status", type=Path)
    parser.add_argument("--acceptance-id")
    parser.add_argument("--accepted-at-utc")
    parser.add_argument("--live-campaign-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    packet = verify_approval_packet(
        read_json(args.approval_packet),
        expected_packet_sha256=args.expected_packet_sha256,
    )
    manifest = software_manifest()
    if args.inspect_only:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_EXPLICIT_USER_ACCEPTANCE",
                    "approval_packet_sha256": packet["packet_sha256"],
                    "software_manifest": manifest,
                    "live_permission": False,
                    "broker_mutation_allowed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if not all(
        (
            args.resident_status,
            args.acceptance_id,
            args.accepted_at_utc,
            args.live_campaign_id,
            args.output,
        )
    ):
        raise SealBlocked("SEAL_ARGUMENTS_INCOMPLETE")
    receipt = build_release_receipt(
        packet=packet,
        expected_packet_sha256=args.expected_packet_sha256,
        resident_status=read_json(args.resident_status),
        manifest=manifest,
        acceptance_id=args.acceptance_id,
        accepted_at_utc=parse_utc(args.accepted_at_utc),
        live_campaign_id=args.live_campaign_id,
    )
    atomic_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "release_receipt_sha256": receipt["release_receipt_sha256"],
                "output": str(args.output),
                "live_permission": False,
                "broker_mutation_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
