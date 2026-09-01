"""Prospective pair-side quarantine for the zero-authority fast-bot shadow.

The raw ledger remains an untouched control.  Only signals generated after a
frozen cutoff are copied into a separate candidate ledger, and precommitted
pair/side lanes are rejected before outcome resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.contextual_technical_forward import (
    append_jsonl_once,
    write_json_atomic,
)
from quant_rabbit.fast_bot_profitability_gate import DEFAULT_THRESHOLDS
from quant_rabbit.fast_bot_truth import _fast_bot_signal_valid


POLICY_CONTRACT = "QR_FAST_BOT_PAIR_SIDE_QUARANTINE_POLICY_V1"
DECISION_CONTRACT = "QR_FAST_BOT_PAIR_SIDE_QUARANTINE_DECISION_V1"
MAX_PENDING_SIGNALS_PER_RUN = 128


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and value.get("contract_sha256") == canonical_sha(body)


def parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.astimezone(timezone.utc)


def load_policy(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pair-side quarantine policy must be an object")
    validate_policy(value)
    return value, canonical_sha(value)


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("contract") != POLICY_CONTRACT or policy.get("schema_version") != 1:
        raise ValueError("pair-side quarantine policy contract mismatch")
    frozen = parse_utc(policy.get("frozen_at_utc"))
    cutoff = parse_utc(policy.get("forward_evaluation_not_before_utc"))
    if cutoff < frozen:
        raise ValueError("forward cutoff predates policy freeze")
    selection = policy.get("selection")
    blocked = selection.get("blocked_pair_sides") if isinstance(selection, Mapping) else None
    if (
        not isinstance(selection, Mapping)
        or selection.get("maximum_selection_delay_seconds") != 45
        or selection.get("historical_rows_admitted") is not False
        or selection.get("unknown_pair_side_policy") != "ADMIT_TO_SEPARATE_SHADOW_COHORT"
        or not isinstance(blocked, list)
        or len(blocked) != 1
        or blocked[0].get("pair") != "EUR_USD"
        or blocked[0].get("side") != "SHORT"
    ):
        raise ValueError("pair-side quarantine selection mismatch")
    authority = policy.get("authority")
    if not isinstance(authority, Mapping) or (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("automatic_adoption_allowed") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("external_order_attempts") != 0
        or authority.get("external_orders") != 0
        or authority.get("manual_tagless_policy") != "NO_TOUCH"
    ):
        raise ValueError("pair-side quarantine authority boundary mismatch")
    thresholds = policy.get("forward_evaluation")
    if not isinstance(thresholds, Mapping) or any(
        thresholds.get(key) != expected for key, expected in DEFAULT_THRESHOLDS.items()
    ):
        raise ValueError("pair-side quarantine forward thresholds mismatch")
    evidence = policy.get("training_evidence")
    numeric = (
        "baseline_filled_signals",
        "baseline_wins",
        "baseline_losses",
        "baseline_net_pips",
        "blocked_pair_side_filled_signals",
        "blocked_pair_side_losses",
        "blocked_pair_side_net_pips",
        "blocked_pair_side_share_of_loss_pips",
    )
    if not isinstance(evidence, Mapping) or evidence.get("profitability_claim") != "UNPROVEN_POST_HOC_HYPOTHESIS_ONLY":
        raise ValueError("pair-side quarantine evidence disclosure mismatch")
    if any(isinstance(evidence.get(key), bool) or not math.isfinite(float(evidence.get(key))) for key in numeric):
        raise ValueError("pair-side quarantine evidence is not finite")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(value)
    return rows


def build_decision(
    signal: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    now_utc: datetime,
) -> dict[str, Any]:
    validate_policy(policy)
    if policy_sha256 != canonical_sha(policy) or not _fast_bot_signal_valid(signal):
        raise ValueError("signal or policy integrity failure")
    if now_utc.tzinfo is None:
        raise ValueError("decision timestamp must be timezone-aware")
    now = now_utc.astimezone(timezone.utc)
    generated = parse_utc(signal.get("generated_at_utc"))
    cutoff = parse_utc(policy["forward_evaluation_not_before_utc"])
    age_seconds = (now - generated).total_seconds()
    blocked = {
        (str(row["pair"]), str(row["side"]))
        for row in policy["selection"]["blocked_pair_sides"]
    }
    reasons: list[str] = []
    if generated < cutoff:
        reasons.append("PRE_POLICY_SIGNAL")
    if age_seconds < 0:
        reasons.append("SIGNAL_GENERATED_IN_FUTURE")
    elif age_seconds > int(policy["selection"]["maximum_selection_delay_seconds"]):
        reasons.append("SELECTION_WINDOW_EXPIRED")
    if (str(signal.get("pair")), str(signal.get("side"))) in blocked:
        reasons.append("PRECOMMITTED_PAIR_SIDE_QUARANTINE")
    admitted = not reasons
    body = {
        "contract": DECISION_CONTRACT,
        "schema_version": 1,
        "decision_id": canonical_sha([policy_sha256, signal["signal_sha256"]]),
        "decided_at_utc": now.isoformat(),
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256,
        "forward_evaluation_not_before_utc": cutoff.isoformat(),
        "source_signal_id": signal["signal_id"],
        "source_signal_sha256": signal["signal_sha256"],
        "pair": signal["pair"],
        "side": signal["side"],
        "method": signal["method"],
        "status": "ADMITTED_PROSPECTIVE_CANDIDATE" if admitted else "REJECTED",
        "reasons": sorted(reasons),
        "selected_signals": [dict(signal)] if admitted else [],
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    return seal(body)


def run_selection(
    *,
    raw_signal_ledger_path: Path,
    policy_path: Path,
    selected_ledger_path: Path,
    decision_ledger_path: Path,
    output_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    policy, policy_sha = load_policy(policy_path)
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    raw_signals = load_jsonl(raw_signal_ledger_path)
    raw_shas: set[str] = set()
    for signal in raw_signals:
        sha = str(signal.get("signal_sha256") or "")
        if not _fast_bot_signal_valid(signal) or not sha or sha in raw_shas:
            raise ValueError("raw signal ledger integrity failure")
        raw_shas.add(sha)
    decisions = load_jsonl(decision_ledger_path)
    processed: set[str] = set()
    admitted_shas: set[str] = set()
    for decision in decisions:
        source_sha = str(decision.get("source_signal_sha256") or "")
        selected_rows = decision.get("selected_signals")
        if (
            not sealed_valid(decision, DECISION_CONTRACT)
            or decision.get("policy_sha256") != policy_sha
            or decision.get("decision_id")
            != canonical_sha([policy_sha, source_sha])
            or source_sha not in raw_shas
            or not isinstance(selected_rows, list)
            or len(selected_rows) > 1
            or (decision.get("status") == "ADMITTED_PROSPECTIVE_CANDIDATE")
            != (len(selected_rows) == 1)
            or decision.get("execution_authority") != "NONE"
            or decision.get("broker_mutation_allowed") is not False
            or decision.get("live_permission") is not False
            or decision.get("external_order_attempts") != 0
            or decision.get("external_orders") != 0
        ):
            raise ValueError("pair-side quarantine decision ledger integrity failure")
        processed.add(source_sha)
        for selected in selected_rows:
            if (
                not isinstance(selected, Mapping)
                or not _fast_bot_signal_valid(selected)
                or selected.get("signal_sha256") != source_sha
            ):
                raise ValueError("pair-side quarantine selected payload integrity failure")
            admitted_shas.add(source_sha)
            append_jsonl_once(
                selected_ledger_path,
                selected,
                identity_key="signal_sha256",
                expected_identity=str(selected["signal_sha256"]),
            )
    selected_history = load_jsonl(selected_ledger_path)
    selected_history_shas: set[str] = set()
    for selected in selected_history:
        selected_sha = str(selected.get("signal_sha256") or "")
        if (
            not _fast_bot_signal_valid(selected)
            or selected_sha in selected_history_shas
            or selected_sha not in admitted_shas
        ):
            raise ValueError("pair-side quarantine selected ledger integrity failure")
        selected_history_shas.add(selected_sha)
    cutoff = parse_utc(policy["forward_evaluation_not_before_utc"])
    pending = [
        signal
        for signal in raw_signals
        if parse_utc(signal["generated_at_utc"]) >= cutoff
        and str(signal["signal_sha256"]) not in processed
    ]
    pending.sort(key=lambda row: (parse_utc(row["generated_at_utc"]), str(row["signal_sha256"])))
    appended_decisions = 0
    appended_signals = 0
    quarantined = 0
    latest: Mapping[str, Any] | None = decisions[-1] if decisions else None
    for signal in pending[:MAX_PENDING_SIGNALS_PER_RUN]:
        decision = build_decision(signal, policy=policy, policy_sha256=policy_sha, now_utc=now)
        appended_decisions += int(
            append_jsonl_once(
                decision_ledger_path,
                decision,
                identity_key="decision_id",
                expected_identity=str(decision["decision_id"]),
            )
        )
        if "PRECOMMITTED_PAIR_SIDE_QUARANTINE" in decision["reasons"]:
            quarantined += 1
        for selected in decision["selected_signals"]:
            appended_signals += int(
                append_jsonl_once(
                    selected_ledger_path,
                    selected,
                    identity_key="signal_sha256",
                    expected_identity=str(selected["signal_sha256"]),
                )
            )
        latest = decision
    result = {
        "status": "WAITING_FOR_FORWARD_SIGNALS" if latest is None else str(latest["status"]),
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha,
        "forward_evaluation_not_before_utc": cutoff.isoformat(),
        "decisions_appended": appended_decisions,
        "selected_signals_appended": appended_signals,
        "quarantined_signals_this_run": quarantined,
        "backlog_remaining": max(0, len(pending) - MAX_PENDING_SIGNALS_PER_RUN),
        "latest_decision": dict(latest) if latest is not None else None,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    write_json_atomic(output_path, result)
    return result
