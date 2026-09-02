"""Bind the resident fast-bot shadow ledgers into autonomous worker episodes.

This is an evidence adapter, not a trading adapter. It reads sealed shadow,
shock-guard, exact-S5 outcome, and learning ledgers and advances one isolated
nervous-system ledger per immutable signal. It has no broker client and cannot
grant execution authority.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.autonomous_shadow_nervous_system import (
    AutonomousShadowNervousSystem,
)
from quant_rabbit.fast_bot_knowledge import EPISODE_CONTRACT
from quant_rabbit.fast_bot_shock_guard import (
    DECISION_CONTRACT as SHOCK_GUARD_DECISION_CONTRACT,
    sealed_valid,
)
from quant_rabbit.fast_bot_truth import OUTCOME_CONTRACT, _fast_bot_signal_valid


CONTRACT = "QR_AUTONOMOUS_SHADOW_RESIDENT_INTEGRATION_V1"
EXECUTION_AUTHORITY = "NONE"
MANUAL_TAGLESS_POLICY = "NO_TOUCH"
TERMINAL_STATES = {"LEARNED", "EXPIRED", "BLOCKED"}
RECEIPT_VALIDITY_DAYS = 36500


def run_autonomous_shadow_integration(
    *,
    shadow_ledger_path: Path,
    shock_guard_decision_ledger_path: Path,
    outcome_ledger_path: Path,
    learning_episode_ledger_path: Path,
    state_root: Path,
    output_path: Path,
    report_path: Path,
    max_signals: int = 128,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Advance bounded, per-signal episodes from already-persisted evidence."""

    if isinstance(max_signals, bool) or not isinstance(max_signals, int) or max_signals <= 0:
        raise ValueError("max_signals must be a positive integer")
    now = _aware(now_utc or datetime.now(timezone.utc))
    signals = _indexed_signals(_load_jsonl(shadow_ledger_path))
    guards = _indexed_guards(_load_jsonl(shock_guard_decision_ledger_path), signals)
    outcomes = _indexed_outcomes(_load_jsonl(outcome_ledger_path), signals)
    learning = _indexed_learning(_load_jsonl(learning_episode_ledger_path), signals, outcomes)

    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for signal_id, signal in signals.items():
        state = _read_episode_state(state_root, signal_id)
        rank = 0 if state and state.get("state") not in TERMINAL_STATES else 1
        if not state or state.get("state") not in TERMINAL_STATES:
            candidates.append((rank, str(signal.get("generated_at_utc") or ""), signal))
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2]["signal_id"])))

    processed = 0
    events_appended = 0
    for _, _, signal in candidates[:max_signals]:
        signal_id = str(signal["signal_id"])
        episode_root = state_root / "episodes" / signal_id
        decisions = _decisions_for(
            signal=signal,
            guard=guards.get(signal_id),
            outcome=outcomes.get(signal_id),
            learning=learning.get(signal_id),
        )
        summary = AutonomousShadowNervousSystem(
            ledger_path=episode_root / "synapses.jsonl",
            output_path=episode_root / "state.json",
            report_path=episode_root / "report.md",
        ).run(
            {
                "cycle_id": f"signal-{signal_id}",
                "decisions": decisions,
                "human_assist": [],
                "kill_switch": False,
            },
            now_utc=now,
        )
        processed += 1
        events_appended += summary.events_appended

    state_counts: Counter[str] = Counter()
    for signal_id in signals:
        episode_state = _read_episode_state(state_root, signal_id)
        state_counts[str(episode_state.get("state") or "UNSEEN") if episode_state else "UNSEEN"] += 1
    unseen = state_counts["UNSEEN"]
    waiting = sum(count for state, count in state_counts.items() if state not in TERMINAL_STATES | {"UNSEEN"})
    if not signals:
        status = "NO_SIGNALS"
    elif unseen:
        status = "DRAINING_BACKLOG"
    elif waiting:
        status = "WAITING_FOR_EVIDENCE"
    else:
        status = "COMPLETE_THROUGH_SOURCE"
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "status": status,
        "source_counts": {
            "signals": len(signals),
            "shock_guard_decisions": len(guards),
            "outcomes": len(outcomes),
            "learning_episodes": len(learning),
        },
        "episode_state_counts": dict(sorted(state_counts.items())),
        "processed_this_run": processed,
        "events_appended_this_run": events_appended,
        "remaining_unseen": unseen,
        "waiting_for_evidence": waiting,
        "human_role": "ASSIST",
        "human_approval_required": False,
        "execution_authority": EXECUTION_AUTHORITY,
        "manual_tagless_positions_policy": MANUAL_TAGLESS_POLICY,
        "shadow_only": True,
        "broker_http_methods_used": [],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "state_root": str(state_root),
    }
    _atomic_write(output_path, json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(report_path, _render_report(body))
    return body


def _decisions_for(
    *,
    signal: Mapping[str, Any],
    guard: Mapping[str, Any] | None,
    outcome: Mapping[str, Any] | None,
    learning: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    signal_sha = str(signal["signal_sha256"])
    observed = _parse_utc(signal["generated_at_utc"])
    decisions = [
        _decision("perception", signal_sha, observed, "Sealed resident shadow signal observed."),
        _decision("hypothesis", signal_sha, observed, "Immutable shadow hypothesis identity and geometry validated."),
    ]
    if guard is None:
        return decisions
    guard_sha = str(guard["contract_sha256"])
    contradiction = str(guard.get("rejection_reason") or "")
    decisions.append(
        _decision(
            "critic",
            guard_sha,
            observed,
            "Deterministic shock guard reviewed counterevidence.",
            contradicting=(f"guard-rejection:{contradiction}",) if contradiction else (),
        )
    )
    admission_verdict = "ADVANCE" if guard.get("entry_allowed") is True else "EXPIRE"
    decisions.append(
        _decision(
            "admission",
            guard_sha,
            observed,
            "Shock-guard admission passed." if admission_verdict == "ADVANCE" else "Shock guard rejected this shadow candidate.",
            verdict=admission_verdict,
            contradicting=(f"guard-rejection:{contradiction or 'ENTRY_NOT_ALLOWED'}",) if admission_verdict == "EXPIRE" else (),
        )
    )
    if admission_verdict != "ADVANCE" or outcome is None:
        return decisions
    outcome_sha = str(outcome["contract_sha256"])
    resolved = _parse_utc(outcome["resolved_at_utc"])
    if outcome.get("filled") is False:
        decisions.append(
            _decision("fill_truth", outcome_sha, resolved, "Exact OANDA S5 truth proved no passive fill.", verdict="NO_FILL")
        )
        if learning is not None:
            decisions.append(_learning_decision(learning))
        return decisions
    decisions.extend(
        (
            _decision("fill_truth", outcome_sha, resolved, "Exact OANDA S5 bid/ask truth proved a virtual fill."),
            _decision("lifecycle", outcome_sha, resolved, "Resolved shadow lifecycle contains immutable fill and horizon evidence."),
            _decision("exit", outcome_sha, resolved, "Resolved shadow outcome contains an immutable exit and realized result."),
        )
    )
    if learning is not None:
        decisions.append(_learning_decision(learning))
    return decisions


def _learning_decision(learning: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(
        "learning",
        str(learning["contract_sha256"]),
        _parse_utc(learning["resolved_at_utc"]),
        "Versioned learning episode bound outcome and precommitted counterfactuals.",
    )


def _decision(
    worker: str,
    evidence_sha: str,
    observed: datetime,
    reason: str,
    *,
    verdict: str = "ADVANCE",
    contradicting: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "decision_id": f"resident:{worker}:{evidence_sha[:32]}:{verdict.lower()}",
        "worker": worker,
        "verdict": verdict,
        "reason": reason,
        "observed_at_utc": observed.isoformat(),
        "expires_at_utc": (observed + timedelta(days=RECEIPT_VALIDITY_DAYS)).isoformat(),
        "supporting_evidence": [f"sha256:{evidence_sha}"],
        "contradicting_evidence": list(contradicting),
        "counterevidence_reviewed": True,
        "confidence": 1.0,
        "uncertainty": 0.0,
    }


def _indexed_signals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not _fast_bot_signal_valid(row):
            raise ValueError("invalid fast-bot shadow signal")
        _authority(row, broker_key="broker_mutation_allowed")
        _insert_unique(result, str(row["signal_id"]), row, "shadow signal")
    return result


def _indexed_guards(rows: Sequence[Mapping[str, Any]], signals: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not sealed_valid(row, SHOCK_GUARD_DECISION_CONTRACT):
            raise ValueError("invalid shock-guard decision seal")
        if (
            row.get("execution_authority") != EXECUTION_AUTHORITY
            or row.get("broker_mutation_allowed") is not False
            or row.get("external_order_attempts") != 0
            or row.get("external_orders") != 0
            or row.get("llm_order_fields_allowed") is not False
        ):
            raise ValueError("shock-guard authority invariant mismatch")
        signal_id = str(row.get("signal_id") or "")
        signal = signals.get(signal_id)
        if signal is None or any(
            row.get(key) != signal.get(key)
            for key in ("pair", "side", "method", "strategy_id")
        ):
            raise ValueError("shock-guard decision does not match a shadow signal")
        if not isinstance(row.get("entry_allowed"), bool):
            raise ValueError("shock-guard entry_allowed must be boolean")
        _insert_unique(result, signal_id, row, "shock-guard decision")
    return result


def _indexed_outcomes(rows: Sequence[Mapping[str, Any]], signals: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not sealed_valid(row, OUTCOME_CONTRACT):
            raise ValueError("invalid fast-bot outcome seal")
        _authority(row, broker_key="broker_mutation")
        signal_id = str(row.get("signal_id") or "")
        signal = signals.get(signal_id)
        if signal is None or any(row.get(key) != signal.get(key) for key in ("signal_sha256", "pair", "side", "method")):
            raise ValueError("fast-bot outcome does not match a shadow signal")
        filled = row.get("filled")
        if filled is True:
            if not row.get("fill_at_utc") or not row.get("exit_at_utc") or row.get("exit_reason") == "UNFILLED":
                raise ValueError("filled outcome lacks lifecycle or exit evidence")
        elif filled is False:
            if row.get("fill_at_utc") is not None or row.get("exit_at_utc") is not None or row.get("exit_reason") != "UNFILLED":
                raise ValueError("unfilled outcome is internally inconsistent")
        else:
            raise ValueError("outcome filled must be boolean")
        if row.get("truth_request_coverage_proved") is not True:
            raise ValueError("outcome truth coverage is not proved")
        _insert_unique(result, signal_id, row, "shadow outcome")
    return result


def _indexed_learning(
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not sealed_valid(row, EPISODE_CONTRACT):
            raise ValueError("invalid learning episode seal")
        _authority(row, broker_key="broker_mutation")
        signal_id = str(row.get("trade_id") or "")
        signal = signals.get(signal_id)
        outcome = outcomes.get(signal_id)
        refs = row.get("raw_source_refs") or {}
        if (
            signal is None
            or outcome is None
            or refs.get("signal_sha256") != signal.get("signal_sha256")
            or refs.get("outcome_sha256") != outcome.get("contract_sha256")
            or (row.get("outcome") or {}).get("filled") is not outcome.get("filled")
        ):
            raise ValueError("learning episode does not match signal and outcome")
        _insert_unique(result, signal_id, row, "learning episode")
    return result


def _authority(value: Mapping[str, Any], *, broker_key: str) -> None:
    if (
        value.get("shadow_only") is not True
        or value.get("live_permission") is not False
        or value.get(broker_key) is not False
        or value.get("execution_authority", EXECUTION_AUTHORITY) != EXECUTION_AUTHORITY
        or value.get("external_order_attempts", 0) != 0
        or value.get("external_orders", 0) != 0
    ):
        raise ValueError("source authority invariant mismatch")


def _insert_unique(target: dict[str, Mapping[str, Any]], identity: str, row: Mapping[str, Any], label: str) -> None:
    if not identity or identity in target:
        raise ValueError(f"duplicate or missing {label} identity")
    target[identity] = row


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return rows


def _read_episode_state(state_root: Path, signal_id: str) -> dict[str, Any]:
    path = state_root / "episodes" / signal_id / "state.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid autonomous episode state: {signal_id}") from exc
    if not isinstance(value, dict) or value.get("execution_authority") != EXECUTION_AUTHORITY:
        raise ValueError(f"invalid autonomous episode authority: {signal_id}")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _render_report(value: Mapping[str, Any]) -> str:
    counts = value["source_counts"]
    states = value["episode_state_counts"]
    return "\n".join(
        (
            "# Autonomous Shadow Resident Integration",
            "",
            f"- Status: `{value['status']}`",
            f"- Signals: `{counts['signals']}`",
            f"- Outcomes: `{counts['outcomes']}`",
            f"- Learning episodes: `{counts['learning_episodes']}`",
            f"- Episode states: `{json.dumps(states, sort_keys=True)}`",
            f"- Waiting for evidence: `{value['waiting_for_evidence']}`",
            "- Human role: `ASSIST` (no routine approval)",
            "- Execution authority: `NONE`",
            "- Manual/tagless positions: `NO_TOUCH`",
            "- Broker mutation: `False`",
            "",
        )
    )


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.astimezone(timezone.utc)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = ["CONTRACT", "run_autonomous_shadow_integration"]
