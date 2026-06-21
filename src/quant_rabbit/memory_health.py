from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ENTRY_THESIS_LEDGER,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_MEMORY_HEALTH,
    DEFAULT_MEMORY_HEALTH_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_STRATEGY_PROFILE,
)


STATUS_PASS = "MEMORY_HEALTH_PASS"
STATUS_WARN = "MEMORY_HEALTH_WARN"
STATUS_BLOCKED = "MEMORY_HEALTH_BLOCKED"

LAYER_SHORT = "short_term"
LAYER_MEDIUM = "medium_term"
LAYER_LONG = "long_term"
LAYER_POSITION = "position_memory"

_MEMORY_BLOCKER_TOKENS = (
    "STRATEGY_PROFILE",
    "TELEMETRY_",
    "FORECAST_HISTORY",
    "PROJECTION_LEDGER",
    "EXECUTION_LEDGER",
    "ENTRY_THESIS",
    "LEARNING_AUDIT",
    "CAPTURE_ECONOMICS",
)

_STRATEGY_PROFILE_GAP_CODES = (
    "STRATEGY_PROFILE_MISSING",
    "STRATEGY_METHOD_PROFILE_MISSING",
)

_LEARNING_AUDIT_LANE_QUARANTINE_MESSAGES = (
    "risk-increasing learning influence is active while recent effect window is negative",
)

# Memory-health audits a market packet assembled over several network and
# forecast calls. The live gateway may refresh broker truth immediately after
# forecast generation for send safety; this engineering grace treats that
# short same-cycle drift as current without allowing stale multi-cycle memory.
FORECAST_SNAPSHOT_GRACE = timedelta(
    seconds=int(os.environ.get("QR_MEMORY_FORECAST_SNAPSHOT_GRACE_SECONDS", "90"))
)
# The trader runtime resolves expired projections before intent generation, but
# intent generation plus sidecar refresh can take long enough for a projection
# to cross its resolution boundary before memory-health runs. This grace
# represents that same-cycle orchestration latency, not extra market confidence:
# materially stale PENDING rows still block routing, while boundary rows wait
# for the next verify-projections pass instead of starving a current entry.
# Kept in lockstep with the live-entry gate and self-improvement audit
# (QR_PROJECTION_PENDING_EXPIRY_GRACE_SECONDS = 1200): the measured live
# refresh-to-gateway latency is ~10 minutes, so one 20-minute scheduler
# cadence is the boundary between same-cycle latency and a real defect.
PROJECTION_PENDING_EXPIRY_GRACE = timedelta(
    seconds=int(os.environ.get("QR_MEMORY_PROJECTION_EXPIRY_GRACE_SECONDS", "1200"))
)


@dataclass(frozen=True)
class MemoryHealthSummary:
    output_path: Path
    report_path: Path
    status: str
    issues: int
    blockers: int
    warnings: int
    layers: dict[str, str]
    metrics: dict[str, Any]


class MemoryHealthAuditor:
    """Audit short, medium, long, and position memory before trader routing."""

    def __init__(
        self,
        *,
        output_path: Path = DEFAULT_MEMORY_HEALTH,
        report_path: Path = DEFAULT_MEMORY_HEALTH_REPORT,
    ) -> None:
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
        entry_thesis_ledger_path: Path = DEFAULT_ENTRY_THESIS_LEDGER,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        now: datetime | None = None,
    ) -> MemoryHealthSummary:
        clock = _to_utc(now or datetime.now(timezone.utc))
        issues: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        snapshot_loaded = _read_json(snapshot_path)
        target_loaded = _read_json(target_state_path)
        intents_loaded = _read_json(order_intents_path)
        capture_loaded = _read_json(capture_economics_path)
        strategy_loaded = _read_json(strategy_profile_path)
        learning_loaded = _read_json(learning_audit_path)

        snapshot = snapshot_loaded.payload or {}
        target_state = target_loaded.payload or {}
        intents = intents_loaded.payload or {}
        capture = capture_loaded.payload or {}
        target_open = _target_open(target_state)
        snapshot_ts = _parse_utc(snapshot.get("fetched_at_utc") or (snapshot.get("account") or {}).get("fetched_at_utc"))
        quote_timestamps = _quote_timestamps(snapshot)
        latest_quote_ts = max(quote_timestamps.values()) if quote_timestamps else None
        intents_ts = _parse_utc(intents.get("generated_at_utc"))
        capture_ts = _parse_utc(capture.get("generated_at_utc"))
        active_positions = _active_trader_positions(snapshot)
        live_ready_pairs = _live_ready_pairs(intents)
        intent_pairs = _intent_pairs(intents)
        required_pairs = tuple(dict.fromkeys([*live_ready_pairs, *(p["pair"] for p in active_positions)]))

        metrics["runtime"] = {
            "target_open": target_open,
            "snapshot_fetched_at_utc": snapshot_ts.isoformat() if snapshot_ts else None,
            "order_intents_generated_at_utc": intents_ts.isoformat() if intents_ts else None,
            "capture_economics_generated_at_utc": capture_ts.isoformat() if capture_ts else None,
            "active_trader_positions": len(active_positions),
            "live_ready_pairs": list(live_ready_pairs),
            "intent_pairs": list(intent_pairs),
            "required_pairs": list(required_pairs),
        }

        _check_required_json(
            issues,
            loaded=snapshot_loaded,
            path=snapshot_path,
            layer=LAYER_SHORT,
            code="SHORT_BROKER_SNAPSHOT_UNREADABLE",
            label="broker_snapshot",
        )
        _check_required_json(
            issues,
            loaded=target_loaded,
            path=target_state_path,
            layer=LAYER_SHORT,
            code="SHORT_DAILY_TARGET_UNREADABLE",
            label="daily_target_state",
        )
        _check_required_json(
            issues,
            loaded=intents_loaded,
            path=order_intents_path,
            layer=LAYER_SHORT,
            code="SHORT_ORDER_INTENTS_UNREADABLE",
            label="order_intents",
        )
        _audit_capture_economics(
            issues,
            metrics,
            loaded=capture_loaded,
            path=capture_economics_path,
            intents=intents,
            intents_ts=intents_ts,
            target_open=target_open,
        )

        forecast_rows, forecast_malformed, forecast_error = _read_jsonl(forecast_history_path)
        _audit_forecast_history(
            issues,
            metrics,
            rows=forecast_rows,
            malformed=forecast_malformed,
            error=forecast_error,
            path=forecast_history_path,
            snapshot_ts=snapshot_ts,
            latest_quote_ts=latest_quote_ts,
            quote_timestamps=quote_timestamps,
            required_pairs=required_pairs,
            target_open=target_open,
            active_positions=active_positions,
        )

        projection_rows, projection_malformed, projection_error = _read_jsonl(projection_ledger_path)
        _audit_projection_ledger(
            issues,
            metrics,
            rows=projection_rows,
            malformed=projection_malformed,
            error=projection_error,
            path=projection_ledger_path,
            forecast_rows=forecast_rows,
            snapshot_ts=snapshot_ts,
            required_pairs=required_pairs,
            target_open=target_open,
            active_positions=active_positions,
            now=clock,
        )

        _audit_execution_ledger(
            issues,
            metrics,
            db_path=execution_ledger_db_path,
            snapshot=snapshot,
        )
        _audit_learning_audit(
            issues,
            metrics,
            loaded=learning_loaded,
            path=learning_audit_path,
        )
        _audit_strategy_profile(
            issues,
            metrics,
            loaded=strategy_loaded,
            path=strategy_profile_path,
        )

        _maybe_backfill_active_entry_theses(
            metrics,
            active_positions=active_positions,
            entry_thesis_ledger_path=entry_thesis_ledger_path,
            execution_ledger_db_path=execution_ledger_db_path,
        )
        entry_rows, entry_malformed, entry_error = _read_jsonl(entry_thesis_ledger_path)
        _audit_entry_thesis(
            issues,
            metrics,
            rows=entry_rows,
            malformed=entry_malformed,
            error=entry_error,
            path=entry_thesis_ledger_path,
            active_positions=active_positions,
        )
        _audit_intent_memory_blockers(issues, metrics, intents=intents)

        layer_statuses = _layer_statuses(issues)
        blockers = [item for item in issues if item["severity"] == "BLOCK"]
        warnings = [item for item in issues if item["severity"] == "WARN"]
        status = STATUS_BLOCKED if blockers else (STATUS_WARN if warnings else STATUS_PASS)
        paths = {
            "broker_snapshot": str(snapshot_path),
            "daily_target_state": str(target_state_path),
            "order_intents": str(order_intents_path),
            "capture_economics": str(capture_economics_path),
            "strategy_profile": str(strategy_profile_path),
            "forecast_history": str(forecast_history_path),
            "projection_ledger": str(projection_ledger_path),
            "learning_audit": str(learning_audit_path),
            "entry_thesis_ledger": str(entry_thesis_ledger_path),
            "execution_ledger_db": str(execution_ledger_db_path),
        }
        payload = {
            "generated_at_utc": clock.isoformat(),
            "status": status,
            "layers": layer_statuses,
            "issues": issues,
            "blockers": [item["message"] for item in blockers],
            "warnings": [item["message"] for item in warnings],
            "metrics": metrics,
            "artifact_paths": paths,
            "contract": {
                "short_term": "broker snapshot, capture economics, order intents, and current forecast history must describe the same executable cycle",
                "medium_term": "projection and execution ledgers must be reconcilable before new live exposure",
                "long_term": "strategy_profile must contain mined evidence before target-open entry routing",
                "position_memory": "open trader-owned positions must retain an entry_thesis row for machine-checkable management",
                "advisory_only": True,
                "hard_gates_remain": ["RiskEngine", "IntentGenerator telemetry validation", "LiveOrderGateway"],
            },
        }
        self._write_output(payload)
        self._write_report(payload)
        return MemoryHealthSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            issues=len(issues),
            blockers=len(blockers),
            warnings=len(warnings),
            layers=layer_statuses,
            metrics=metrics,
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Memory Health Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            "",
            "## Layers",
            "",
        ]
        for layer, status in payload["layers"].items():
            lines.append(f"- `{layer}`: `{status}`")
        lines.extend(["", "## Issues", ""])
        if payload["issues"]:
            for issue in payload["issues"]:
                lines.append(
                    f"- `{issue['severity']}` `{issue['layer']}` `{issue['code']}`: {issue['message']}"
                )
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- This audit does not grant permission to trade.",
                "- BLOCK means a memory artifact is missing, stale, unreconciled, or internally inconsistent before routing.",
                "- Final broker send remains governed by RiskEngine, IntentGenerator telemetry validation, and LiveOrderGateway.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _LoadedJson:
    payload: dict[str, Any] | None
    error: str | None


def _read_json(path: Path) -> _LoadedJson:
    if not path.exists():
        return _LoadedJson(None, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _LoadedJson(None, str(exc))
    if not isinstance(payload, dict):
        return _LoadedJson(None, "not a JSON object")
    return _LoadedJson(payload, None)


def _read_jsonl(path: Path) -> tuple[tuple[dict[str, Any], ...], int, str | None]:
    if not path.exists():
        return (), 0, "missing"
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    malformed += 1
    except OSError as exc:
        return (), malformed, str(exc)
    return tuple(rows), malformed, None


def _check_required_json(
    issues: list[dict[str, Any]],
    *,
    loaded: _LoadedJson,
    path: Path,
    layer: str,
    code: str,
    label: str,
) -> None:
    if loaded.error is None:
        return
    issues.append(
        _issue(
            layer=layer,
            severity="BLOCK",
            code=code,
            message=f"{label} is unreadable: {path}: {loaded.error}",
        )
    )


def _audit_capture_economics(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    loaded: _LoadedJson,
    path: Path,
    intents: dict[str, Any],
    intents_ts: datetime | None,
    target_open: bool,
) -> None:
    payload = loaded.payload or {}
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    capture_trades = int(_float_value(overall.get("trades") or payload.get("trades")) or 0)
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    embedded_values, embedded_examples = _capture_trade_counts_in_intents(intents, capture_trades)
    timestamp_stale = bool(
        intents_ts is not None
        and generated_at is not None
        and generated_at > intents_ts
    )
    metadata_mismatch = bool(
        capture_trades > 0
        and embedded_values
        and any(value != capture_trades for value in embedded_values)
    )
    metrics["capture_economics"] = {
        "path": str(path),
        "generated_at_utc": generated_at.isoformat() if generated_at else None,
        "status": payload.get("status"),
        "trades": capture_trades,
        "error": loaded.error,
        "capture_generated_after_order_intents": timestamp_stale,
        "intent_capture_economics_trades": sorted(set(embedded_values)),
        "metadata_trade_count_mismatch": metadata_mismatch,
        "mismatch_examples": embedded_examples,
    }
    if not target_open:
        return
    if loaded.error is not None:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="BLOCK",
                code="SHORT_CAPTURE_ECONOMICS_UNREADABLE",
                message=f"capture_economics is unreadable while target is open: {path}: {loaded.error}",
            )
        )
        return
    if generated_at is None:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="BLOCK",
                code="SHORT_CAPTURE_ECONOMICS_TIMESTAMP_MISSING",
                message=f"capture_economics lacks generated_at_utc while target is open: {path}",
            )
        )
    if not timestamp_stale and not metadata_mismatch:
        return
    reasons: list[str] = []
    if timestamp_stale:
        reasons.append("capture_economics generated after order_intents")
    if metadata_mismatch:
        reasons.append("intent metadata embeds stale capture_economics trade count")
    issues.append(
        _issue(
            layer=LAYER_SHORT,
            severity="BLOCK",
            code="SHORT_ORDER_INTENTS_CAPTURE_ECONOMICS_STALE",
            message=(
                "order_intents were priced from stale capture_economics evidence; "
                "run capture-economics before generate-intents"
            ),
            evidence={
                "order_intents_generated_at_utc": intents_ts.isoformat() if intents_ts else None,
                "capture_economics_generated_at_utc": generated_at.isoformat() if generated_at else None,
                "capture_trades": capture_trades,
                "intent_capture_economics_trades": sorted(set(embedded_values)),
                "reasons": reasons,
                "mismatch_examples": embedded_examples,
            },
        )
    )


def _capture_trade_counts_in_intents(
    intents: dict[str, Any],
    capture_trades: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    values: list[int] = []
    examples: list[dict[str, Any]] = []
    results = intents.get("results") if isinstance(intents.get("results"), list) else []
    for item in results:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        embedded = _float_value(metadata.get("capture_economics_trades"))
        if embedded is None:
            continue
        embedded_int = int(embedded)
        values.append(embedded_int)
        if capture_trades > 0 and embedded_int != capture_trades and len(examples) < 5:
            examples.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "intent_capture_economics_trades": embedded_int,
                }
            )
    return values, examples


def _audit_forecast_history(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    rows: tuple[dict[str, Any], ...],
    malformed: int,
    error: str | None,
    path: Path,
    snapshot_ts: datetime | None,
    latest_quote_ts: datetime | None,
    quote_timestamps: dict[str, datetime],
    required_pairs: tuple[str, ...],
    target_open: bool,
    active_positions: list[dict[str, str]],
) -> None:
    latest_by_pair: dict[str, dict[str, Any]] = {}
    latest_ts: datetime | None = None
    duplicate_cycle_pairs = 0
    cycle_pair_latest_ts: dict[tuple[str, str], datetime | None] = {}
    cycle_pair_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        pair = str(row.get("pair") or "")
        ts = _parse_utc(row.get("timestamp_utc"))
        if pair:
            previous = latest_by_pair.get(pair)
            previous_ts = _parse_utc(previous.get("timestamp_utc")) if previous else None
            if previous is None or _row_is_newer(ts, previous_ts):
                latest_by_pair[pair] = row
        if ts is not None and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
        cycle_id = str(row.get("cycle_id") or "")
        if pair and cycle_id:
            key = (cycle_id, pair)
            cycle_pair_counts[key] = cycle_pair_counts.get(key, 0) + 1
            current_latest = cycle_pair_latest_ts.get(key)
            if ts is not None and (current_latest is None or ts > current_latest):
                cycle_pair_latest_ts[key] = ts
            else:
                cycle_pair_latest_ts.setdefault(key, current_latest)
    duplicate_keys = {key for key, count in cycle_pair_counts.items() if count > 1}
    duplicate_cycle_pairs = len(duplicate_keys)
    current_duplicate_cycle_pairs = _current_duplicate_count(
        duplicate_keys,
        latest_by_key=cycle_pair_latest_ts,
        freshness_cutoff=snapshot_ts,
    )
    metrics["forecast_history"] = {
        "path": str(path),
        "rows": len(rows),
        "malformed_rows": malformed,
        "latest_timestamp_utc": latest_ts.isoformat() if latest_ts else None,
        "pairs": sorted(latest_by_pair),
        "duplicate_cycle_pairs": duplicate_cycle_pairs,
        "current_duplicate_cycle_pairs": current_duplicate_cycle_pairs,
        "snapshot_grace_seconds": FORECAST_SNAPSHOT_GRACE.total_seconds(),
        "latest_quote_timestamp_utc": latest_quote_ts.isoformat() if latest_quote_ts else None,
        "quotes_predate_snapshot": _timestamp_predates_snapshot_beyond_grace(
            latest_quote_ts,
            snapshot_ts,
            FORECAST_SNAPSHOT_GRACE,
        ),
    }
    if error is not None:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="BLOCK",
                code="SHORT_FORECAST_HISTORY_UNREADABLE",
                message=f"forecast_history is unreadable: {path}: {error}",
            )
        )
        return
    if not rows:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="BLOCK",
                code="SHORT_FORECAST_HISTORY_EMPTY",
                message=f"forecast_history has no rows: {path}",
            )
        )
    if malformed:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="WARN",
                code="SHORT_FORECAST_HISTORY_MALFORMED_ROWS",
                message=f"forecast_history skipped {malformed} malformed row(s): {path}",
            )
        )
    if current_duplicate_cycle_pairs:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="WARN",
                code="SHORT_FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR",
                message=(
                    "forecast_history has "
                    f"{current_duplicate_cycle_pairs} current duplicate cycle_id/pair key(s)"
                ),
            )
        )
    if (
        snapshot_ts is not None
        and latest_ts is not None
        and _timestamp_predates_snapshot_beyond_grace(latest_ts, snapshot_ts, FORECAST_SNAPSHOT_GRACE)
        and (target_open or active_positions)
    ):
        quote_stale = _timestamp_predates_snapshot_beyond_grace(
            latest_quote_ts,
            snapshot_ts,
            FORECAST_SNAPSHOT_GRACE,
        )
        # When broker quotes themselves predate the snapshot, the market packet
        # has no newer executable price to forecast from. Keep the gap visible
        # but do not escalate it as a forecast-memory defect; stale quotes and
        # telemetry gates already prevent live entries.
        severity = "WARN" if quote_stale else "BLOCK"
        code = (
            "SHORT_FORECAST_HISTORY_STALE_WHILE_QUOTES_STALE"
            if quote_stale
            else "SHORT_FORECAST_HISTORY_STALE"
        )
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity=severity,
                code=code,
                message=(
                    "forecast_history latest row predates broker snapshot "
                    f"(forecast={latest_ts.isoformat()}, snapshot={snapshot_ts.isoformat()})"
                    + (
                        f" while latest quote is also stale (quote={latest_quote_ts.isoformat()})"
                        if quote_stale and latest_quote_ts is not None
                        else ""
                    )
                ),
            )
        )
    for pair in required_pairs:
        row = latest_by_pair.get(pair)
        if row is None:
            issues.append(
                _issue(
                    layer=LAYER_SHORT,
                    severity="BLOCK",
                    code="SHORT_FORECAST_PAIR_MISSING",
                    message=f"forecast_history has no row for required pair {pair}",
                    evidence={"pair": pair},
                )
            )
            continue
        row_ts = _parse_utc(row.get("timestamp_utc"))
        if (
            snapshot_ts is not None
            and row_ts is not None
            and _timestamp_predates_snapshot_beyond_grace(row_ts, snapshot_ts, FORECAST_SNAPSHOT_GRACE)
            and (target_open or active_positions)
        ):
            pair_quote_ts = quote_timestamps.get(pair) or latest_quote_ts
            pair_quote_stale = _timestamp_predates_snapshot_beyond_grace(
                pair_quote_ts,
                snapshot_ts,
                FORECAST_SNAPSHOT_GRACE,
            )
            severity = "WARN" if pair_quote_stale else "BLOCK"
            code = (
                "SHORT_FORECAST_PAIR_STALE_WHILE_QUOTE_STALE"
                if pair_quote_stale
                else "SHORT_FORECAST_PAIR_STALE"
            )
            issues.append(
                _issue(
                    layer=LAYER_SHORT,
                    severity=severity,
                    code=code,
                    message=(
                        f"forecast_history row for {pair} predates broker snapshot "
                        f"(forecast={row_ts.isoformat()}, snapshot={snapshot_ts.isoformat()})"
                        + (
                            f" while pair quote is also stale (quote={pair_quote_ts.isoformat()})"
                            if pair_quote_stale and pair_quote_ts is not None
                            else ""
                        )
                    ),
                    evidence={"pair": pair},
                )
            )


def _audit_projection_ledger(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    rows: tuple[dict[str, Any], ...],
    malformed: int,
    error: str | None,
    path: Path,
    forecast_rows: tuple[dict[str, Any], ...],
    snapshot_ts: datetime | None,
    required_pairs: tuple[str, ...],
    target_open: bool,
    active_positions: list[dict[str, str]],
    now: datetime,
) -> None:
    status_counts: dict[str, int] = {}
    expired_pending = 0
    directional_keys: set[tuple[str, str]] = set()
    projection_key_counts: dict[tuple[Any, ...], int] = {}
    projection_key_latest_ts: dict[tuple[Any, ...], datetime | None] = {}
    for row in rows:
        status = str(row.get("resolution_status") or "PENDING").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "PENDING" and _projection_expired(row, now=now):
            expired_pending += 1
        pair = str(row.get("pair") or "")
        cycle_id = str(row.get("cycle_id") or "")
        if pair and cycle_id and str(row.get("signal_name") or "") == "directional_forecast":
            directional_keys.add((pair, cycle_id))
        key = _projection_key(row)
        if key is not None:
            projection_key_counts[key] = projection_key_counts.get(key, 0) + 1
            ts = _parse_utc(row.get("timestamp_emitted_utc"))
            current_latest = projection_key_latest_ts.get(key)
            if ts is not None and (current_latest is None or ts > current_latest):
                projection_key_latest_ts[key] = ts
            else:
                projection_key_latest_ts.setdefault(key, current_latest)
    duplicate_projection_keys = {key for key, count in projection_key_counts.items() if count > 1}
    current_duplicate_projection_keys = _current_duplicate_count(
        duplicate_projection_keys,
        latest_by_key=projection_key_latest_ts,
        freshness_cutoff=snapshot_ts,
    )
    metrics["projection_ledger"] = {
        "path": str(path),
        "rows": len(rows),
        "malformed_rows": malformed,
        "status_counts": status_counts,
        "expired_pending": expired_pending,
        "directional_forecast_keys": len(directional_keys),
        "duplicate_projection_keys": len(duplicate_projection_keys),
        "current_duplicate_projection_keys": current_duplicate_projection_keys,
    }
    if error is not None:
        severity = "BLOCK" if target_open or active_positions else "WARN"
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity=severity,
                code="MEDIUM_PROJECTION_LEDGER_UNREADABLE",
                message=f"projection_ledger is unreadable: {path}: {error}",
            )
        )
        return
    if not rows and (target_open or active_positions):
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_PROJECTION_LEDGER_EMPTY",
                message=f"projection_ledger has no rows while live memory is needed: {path}",
            )
        )
    if malformed:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="WARN",
                code="MEDIUM_PROJECTION_LEDGER_MALFORMED_ROWS",
                message=f"projection_ledger skipped {malformed} malformed row(s): {path}",
            )
        )
    if current_duplicate_projection_keys:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="WARN",
                code="MEDIUM_PROJECTION_LEDGER_DUPLICATE_KEY",
                message=(
                    "projection_ledger has "
                    f"{current_duplicate_projection_keys} current duplicate cycle projection key(s)"
                ),
            )
        )
    if expired_pending:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_PROJECTION_LEDGER_EXPIRED_PENDING",
                message=f"projection_ledger has {expired_pending} expired PENDING projection(s); run verify-projections",
            )
        )
    latest_forecasts = _latest_forecasts_by_pair(forecast_rows)
    for pair in required_pairs:
        forecast = latest_forecasts.get(pair)
        if not forecast:
            continue
        direction = str(forecast.get("direction") or "").upper()
        cycle_id = str(forecast.get("cycle_id") or "")
        forecast_ts = _parse_utc(forecast.get("timestamp_utc"))
        if (
            snapshot_ts is not None
            and forecast_ts is not None
            and _forecast_predates_snapshot_beyond_grace(forecast_ts, snapshot_ts)
        ):
            continue
        if direction in {"UP", "DOWN"} and cycle_id and (pair, cycle_id) not in directional_keys:
            issues.append(
                _issue(
                    layer=LAYER_MEDIUM,
                    severity="BLOCK",
                    code="MEDIUM_DIRECTIONAL_PROJECTION_MISSING",
                    message=(
                        f"{pair} directional forecast cycle_id={cycle_id} has no matching "
                        "directional_forecast row in projection_ledger"
                    ),
                    evidence={"pair": pair, "cycle_id": cycle_id},
                )
            )


def _audit_execution_ledger(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    db_path: Path,
    snapshot: dict[str, Any],
) -> None:
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    expected = str((account or {}).get("last_transaction_id") or "").strip()
    actual: str | None = None
    events = 0
    transactions = 0
    error: str | None = None
    if not db_path.exists():
        error = "missing"
    else:
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "select value from sync_state where key = ?",
                    ("last_oanda_transaction_id",),
                ).fetchone()
                actual = str(row[0] or "").strip() if row else None
                events = int(conn.execute("select count(*) from execution_events").fetchone()[0])
                transactions = int(conn.execute("select count(*) from oanda_transactions").fetchone()[0])
        except sqlite3.Error as exc:
            error = str(exc)
    metrics["execution_ledger"] = {
        "path": str(db_path),
        "last_oanda_transaction_id": actual,
        "snapshot_last_transaction_id": expected or None,
        "events": events,
        "transactions": transactions,
        "error": error,
    }
    if error is not None:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_EXECUTION_LEDGER_UNREADABLE",
                message=f"execution_ledger DB is unreadable: {db_path}: {error}",
            )
        )
        return
    if expected and not actual:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_EXECUTION_LEDGER_SYNC_MISSING",
                message=f"execution_ledger lacks last_oanda_transaction_id while snapshot is at {expected}",
            )
        )
    if expected and actual and _transaction_id_is_behind(actual, expected):
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_EXECUTION_LEDGER_STALE",
                message=f"execution_ledger last_oanda_transaction_id={actual} is behind broker snapshot {expected}",
            )
        )


def _audit_learning_audit(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    loaded: _LoadedJson,
    path: Path,
) -> None:
    payload = loaded.payload or {}
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    influenced_lanes = _int_value((influence or {}).get("influenced_lanes"))
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    status = str(payload.get("status") or "")
    metrics["learning_audit"] = {
        "path": str(path),
        "status": status or None,
        "influenced_lanes": influenced_lanes,
        "blockers": len(blockers),
        "warnings": len(warnings),
        "error": loaded.error,
    }
    if loaded.error is not None:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="WARN",
                code="MEDIUM_LEARNING_AUDIT_UNREADABLE",
                message=f"learning_audit is unreadable: {path}: {loaded.error}",
            )
        )
        return
    if ("BLOCK" in status.upper() or blockers) and influenced_lanes > 0:
        if _learning_audit_block_only_quarantines_influenced_lanes(payload):
            issues.append(
                _issue(
                    layer=LAYER_MEDIUM,
                    severity="WARN",
                    code="MEDIUM_LEARNING_AUDIT_INFLUENCED_LANES_QUARANTINED",
                    message=(
                        "learning_audit blocks only risk-increasing learning influence; "
                        "quarantine influenced lanes without blocking clean live-ready lanes"
                    ),
                    evidence={
                        "influenced_lanes": influenced_lanes,
                        "risk_increasing_lanes": _int_value((influence or {}).get("risk_increasing_lanes")),
                        "blockers": blockers[:8],
                    },
                )
            )
            return
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="BLOCK",
                code="MEDIUM_LEARNING_AUDIT_BLOCKING_INFLUENCE",
                message=(
                    f"learning_audit is blocked while {influenced_lanes} lane(s) receive learning influence"
                ),
            )
        )
    elif "BLOCK" in status.upper() or blockers:
        issues.append(
            _issue(
                layer=LAYER_MEDIUM,
                severity="WARN",
                code="MEDIUM_LEARNING_AUDIT_BLOCKED_ADVISORY",
                message="learning_audit is blocked, but no live lane receives learning influence",
            )
        )


def _learning_audit_block_only_quarantines_influenced_lanes(payload: dict[str, Any]) -> bool:
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    if _int_value((influence or {}).get("influenced_lanes")) <= 0:
        return False
    blockers = [str(item) for item in (payload.get("blockers") or []) if str(item).strip()]
    if not blockers:
        return False
    if not all(_is_learning_lane_quarantine_message(message) for message in blockers):
        return False
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    blocking_checks = [
        item
        for item in checks
        if isinstance(item, dict)
        and (
            str(item.get("status") or "").upper() == "BLOCK"
            or str(item.get("severity") or "").upper() == "BLOCK"
        )
    ]
    if not blocking_checks:
        return True
    return all(
        str(item.get("check_name") or "") == "learning_influence_recent_outcome"
        or _is_learning_lane_quarantine_message(str(item.get("message") or ""))
        for item in blocking_checks
    )


def _is_learning_lane_quarantine_message(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in _LEARNING_AUDIT_LANE_QUARANTINE_MESSAGES)


def _audit_strategy_profile(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    loaded: _LoadedJson,
    path: Path,
) -> None:
    payload = loaded.payload or {}
    profiles = payload.get("profiles")
    status_counts: dict[str, int] = {}
    if isinstance(profiles, list):
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            status = str(profile.get("status") or "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    history_db_raw = str(payload.get("history_db") or "").strip()
    history_db = Path(history_db_raw) if history_db_raw else None
    if history_db is not None and not history_db.is_absolute():
        history_db = path.parent.parent / history_db
    metrics["strategy_profile"] = {
        "path": str(path),
        "generated_at_utc": generated_at.isoformat() if generated_at else None,
        "profiles": len(profiles) if isinstance(profiles, list) else None,
        "status_counts": status_counts,
        "history_db": str(history_db) if history_db is not None else None,
        "error": loaded.error,
    }
    if loaded.error is not None:
        issues.append(
            _issue(
                layer=LAYER_LONG,
                severity="BLOCK",
                code="LONG_STRATEGY_PROFILE_UNREADABLE",
                message=f"strategy_profile is unreadable: {path}: {loaded.error}",
            )
        )
        return
    if not isinstance(profiles, list):
        issues.append(
            _issue(
                layer=LAYER_LONG,
                severity="BLOCK",
                code="LONG_STRATEGY_PROFILE_MALFORMED",
                message=f"strategy_profile profiles must be a list: {path}",
            )
        )
        return
    if not profiles:
        issues.append(
            _issue(
                layer=LAYER_LONG,
                severity="BLOCK",
                code="LONG_STRATEGY_PROFILE_EMPTY",
                message=f"strategy_profile has zero mined profiles: {path}; run import-legacy and mine-strategy",
            )
        )
    if generated_at is None:
        issues.append(
            _issue(
                layer=LAYER_LONG,
                severity="BLOCK",
                code="LONG_STRATEGY_PROFILE_GENERATED_AT_MISSING",
                message=f"strategy_profile lacks generated_at_utc: {path}",
            )
        )
    if history_db is not None and history_db.exists() and path.exists() and history_db.stat().st_mtime_ns > path.stat().st_mtime_ns:
        issues.append(
            _issue(
                layer=LAYER_LONG,
                severity="BLOCK",
                code="LONG_STRATEGY_PROFILE_STALE",
                message=f"strategy_profile is older than history DB: {path}",
            )
        )


def _audit_entry_thesis(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    rows: tuple[dict[str, Any], ...],
    malformed: int,
    error: str | None,
    path: Path,
    active_positions: list[dict[str, str]],
) -> None:
    thesis_trade_ids = {
        str(row.get("trade_id") or "")
        for row in rows
        if str(row.get("trade_id") or "")
    }
    active_trade_ids = [position["trade_id"] for position in active_positions if position.get("trade_id")]
    missing_trade_ids = [trade_id for trade_id in active_trade_ids if trade_id not in thesis_trade_ids]
    metrics["entry_thesis_ledger"] = {
        "path": str(path),
        "rows": len(rows),
        "malformed_rows": malformed,
        "active_trade_ids": active_trade_ids,
        "missing_active_trade_ids": missing_trade_ids,
        "error": error,
    }
    if error is not None:
        if not active_trade_ids:
            return
        issues.append(
            _issue(
                layer=LAYER_POSITION,
                severity="BLOCK",
                code="POSITION_ENTRY_THESIS_UNREADABLE",
                message=f"entry_thesis_ledger is unreadable: {path}: {error}",
            )
        )
        return
    if malformed:
        issues.append(
            _issue(
                layer=LAYER_POSITION,
                severity="WARN",
                code="POSITION_ENTRY_THESIS_MALFORMED_ROWS",
                message=f"entry_thesis_ledger skipped {malformed} malformed row(s): {path}",
            )
        )
    if missing_trade_ids:
        issues.append(
            _issue(
                layer=LAYER_POSITION,
                severity="BLOCK",
                code="POSITION_ENTRY_THESIS_MISSING_FOR_OPEN_TRADE",
                message=(
                    "entry_thesis_ledger is missing active trader-owned trade id(s): "
                    + ", ".join(missing_trade_ids)
                ),
                evidence={"trade_ids": missing_trade_ids},
            )
        )


def _maybe_backfill_active_entry_theses(
    metrics: dict[str, Any],
    *,
    active_positions: list[dict[str, str]],
    entry_thesis_ledger_path: Path,
    execution_ledger_db_path: Path,
) -> None:
    active_trade_ids = [position["trade_id"] for position in active_positions if position.get("trade_id")]
    if not active_trade_ids:
        metrics["entry_thesis_backfill"] = {"status": "NO_ACTIVE_TRADES"}
        return
    try:
        from quant_rabbit.strategy.entry_thesis_ledger import backfill_entry_theses_from_execution_ledger

        result = backfill_entry_theses_from_execution_ledger(
            db_path=execution_ledger_db_path,
            data_root=entry_thesis_ledger_path.parent,
            trade_ids=active_trade_ids,
        )
        metrics["entry_thesis_backfill"] = result.to_dict()
    except Exception as exc:
        metrics["entry_thesis_backfill"] = {
            "status": "ERROR",
            "requested_trade_ids": active_trade_ids,
            "issue": str(exc),
        }


def _audit_intent_memory_blockers(
    issues: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    intents: dict[str, Any],
) -> None:
    memory_blockers: list[str] = []
    advisory_memory_blockers = 0
    live_ready_pairs = _live_ready_pairs(intents)
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        result_memory_blockers: list[str] = []
        structured_blockers: set[str] = set()
        structured_advisory: set[str] = set()
        for container_name in ("risk_issues", "strategy_issues", "live_strategy_issues"):
            for item in result.get(container_name) or []:
                if _is_self_improvement_intent_blocker(item):
                    structured_advisory.add(_issue_text(item))
                    advisory_memory_blockers += 1
                    continue
                text = _issue_text(item).upper()
                if not any(token in text for token in _MEMORY_BLOCKER_TOKENS):
                    continue
                issue_text = _issue_text(item)
                issue_variants = {issue_text}
                if isinstance(item, dict):
                    message = str(item.get("message") or "")
                    if message:
                        issue_variants.add(message)
                if _is_quote_freshness_blocker(item):
                    structured_advisory.update(issue_variants)
                    advisory_memory_blockers += 1
                    continue
                if _is_capture_economics_profitability_blocker(item):
                    structured_advisory.update(issue_variants)
                    advisory_memory_blockers += 1
                    continue
                if _is_strategy_profile_gap(item):
                    structured_advisory.update(issue_variants)
                    advisory_memory_blockers += 1
                    continue
                if isinstance(item, dict) and str(item.get("severity") or "").upper() != "BLOCK":
                    structured_advisory.update(issue_variants)
                    advisory_memory_blockers += 1
                    continue
                structured_blockers.update(issue_variants)
                result_memory_blockers.append(issue_text)
        for item in result.get("live_blockers") or []:
            issue_text = _issue_text(item)
            text = issue_text.upper()
            if _is_self_improvement_intent_blocker(item):
                advisory_memory_blockers += 1
                continue
            if any(token in text for token in _MEMORY_BLOCKER_TOKENS):
                if issue_text in structured_blockers:
                    continue
                if issue_text in structured_advisory:
                    continue
                if _is_quote_freshness_blocker(item):
                    advisory_memory_blockers += 1
                    continue
                if _is_capture_economics_profitability_blocker(item):
                    advisory_memory_blockers += 1
                    continue
                if _is_strategy_profile_gap(item):
                    advisory_memory_blockers += 1
                    continue
                result_memory_blockers.append(issue_text)
        if result_memory_blockers:
            if live_ready_pairs or str(result.get("status") or "") == "LIVE_READY":
                advisory_memory_blockers += len(result_memory_blockers)
            else:
                memory_blockers.extend(result_memory_blockers)
    metrics["order_intents"] = {
        "results": len([item for item in intents.get("results", []) or [] if isinstance(item, dict)]),
        "live_ready": len(live_ready_pairs),
        "memory_blockers": len(memory_blockers),
        "advisory_memory_blockers": advisory_memory_blockers,
    }
    if memory_blockers:
        issues.append(
            _issue(
                layer=LAYER_SHORT,
                severity="BLOCK",
                code="SHORT_ORDER_INTENTS_MEMORY_BLOCKERS",
                message=f"order_intents contains {len(memory_blockers)} memory/telemetry blocker(s)",
                evidence={"examples": memory_blockers[:5]},
            )
        )


def _issue(
    *,
    layer: str,
    severity: str,
    code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "layer": layer,
        "severity": severity,
        "code": code,
        "message": message,
    }
    if evidence:
        payload["evidence"] = evidence
    return payload


def _layer_statuses(issues: list[dict[str, Any]]) -> dict[str, str]:
    statuses = {layer: "PASS" for layer in (LAYER_SHORT, LAYER_MEDIUM, LAYER_LONG, LAYER_POSITION)}
    for issue in issues:
        layer = str(issue.get("layer") or "")
        severity = str(issue.get("severity") or "")
        if layer not in statuses:
            continue
        if severity == "BLOCK":
            statuses[layer] = "BLOCK"
        elif severity == "WARN" and statuses[layer] != "BLOCK":
            statuses[layer] = "WARN"
    return statuses


def _target_open(target_state: dict[str, Any]) -> bool:
    try:
        remaining = float(target_state.get("remaining_target_jpy") or 0.0)
    except (TypeError, ValueError):
        return False
    return remaining > 0.0 and target_state.get("status") != "TARGET_REACHED_PROTECT"


def _active_trader_positions(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    positions: list[dict[str, str]] = []
    for raw in snapshot.get("positions", []) or []:
        if not isinstance(raw, dict):
            continue
        owner = str(raw.get("owner") or "").lower()
        if owner != "trader":
            continue
        trade_id = str(raw.get("trade_id") or "")
        pair = str(raw.get("pair") or "")
        side = str(raw.get("side") or "").upper()
        if trade_id and pair and side in {"LONG", "SHORT"}:
            positions.append({"trade_id": trade_id, "pair": pair, "side": side})
    return positions


def _live_ready_pairs(intents: dict[str, Any]) -> tuple[str, ...]:
    pairs: list[str] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict) or result.get("status") != "LIVE_READY":
            continue
        pair = _intent_pair(result)
        if pair:
            pairs.append(pair)
    return tuple(dict.fromkeys(pairs))


def _intent_pairs(intents: dict[str, Any]) -> tuple[str, ...]:
    pairs: list[str] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        pair = _intent_pair(result)
        if pair:
            pairs.append(pair)
    return tuple(dict.fromkeys(pairs))


def _intent_pair(result: dict[str, Any]) -> str | None:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    pair = str((intent or {}).get("pair") or result.get("pair") or "").strip()
    if pair:
        return pair
    lane_id = str(result.get("lane_id") or "")
    parts = lane_id.split(":")
    if len(parts) >= 2 and "_" in parts[1]:
        return parts[1]
    return None


def _latest_forecasts_by_pair(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair = str(row.get("pair") or "")
        if pair:
            latest[pair] = row
    return latest


def _projection_expired(row: dict[str, Any], *, now: datetime) -> bool:
    emitted = _parse_utc(row.get("timestamp_emitted_utc"))
    window_min = _float_value(row.get("resolution_window_min"))
    if emitted is None or window_min is None or window_min <= 0:
        return True
    return now >= emitted + timedelta(minutes=window_min) + PROJECTION_PENDING_EXPIRY_GRACE


def _projection_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    cycle_id = str(row.get("cycle_id") or "")
    if not cycle_id:
        return None
    return (
        cycle_id,
        str(row.get("pair") or ""),
        str(row.get("signal_name") or ""),
        str(row.get("direction") or ""),
        row.get("entry_price"),
        row.get("predicted_target_price"),
    )


def _current_duplicate_count(
    duplicate_keys: set[tuple[Any, ...]],
    *,
    latest_by_key: dict[tuple[Any, ...], datetime | None],
    freshness_cutoff: datetime | None,
) -> int:
    if not duplicate_keys:
        return 0
    if freshness_cutoff is None:
        return len(duplicate_keys)
    cutoff = _to_utc(freshness_cutoff)
    current = 0
    for key in duplicate_keys:
        latest = latest_by_key.get(key)
        if latest is None or _to_utc(latest) >= cutoff:
            current += 1
    return current


def _row_is_newer(candidate_ts: datetime | None, current_ts: datetime | None) -> bool:
    if candidate_ts is None:
        return current_ts is None
    if current_ts is None:
        return True
    return _to_utc(candidate_ts) >= _to_utc(current_ts)


def _quote_timestamps(snapshot: dict[str, Any]) -> dict[str, datetime]:
    quotes = snapshot.get("quotes")
    if not isinstance(quotes, dict):
        return {}
    timestamps: dict[str, datetime] = {}
    for pair, quote in quotes.items():
        if not isinstance(quote, dict):
            continue
        ts = _parse_utc(quote.get("timestamp_utc"))
        if ts is not None:
            timestamps[str(pair)] = ts
    return timestamps


def _timestamp_predates_snapshot_beyond_grace(
    timestamp: datetime | None,
    snapshot_ts: datetime | None,
    grace: timedelta,
) -> bool:
    if timestamp is None or snapshot_ts is None:
        return False
    return _to_utc(timestamp) + grace < _to_utc(snapshot_ts)


def _forecast_predates_snapshot_beyond_grace(forecast_ts: datetime, snapshot_ts: datetime) -> bool:
    return _timestamp_predates_snapshot_beyond_grace(forecast_ts, snapshot_ts, FORECAST_SNAPSHOT_GRACE)


def _issue_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("code") or item.get("message") or item.get("rationale") or item)
    return str(item)


def _is_quote_freshness_blocker(item: Any) -> bool:
    text = _issue_text(item).upper()
    if isinstance(item, dict):
        code = str(item.get("code") or "").upper()
        if code in {"STALE_QUOTE", "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE"}:
            return True
    return "QUOTE IS" in text and "LIVE FRESHNESS CONTRACT" in text


def _is_capture_economics_profitability_blocker(item: Any) -> bool:
    """Profitability gates may cite capture_economics without being memory defects."""

    text = _issue_text(item).upper()
    if isinstance(item, dict):
        code = str(item.get("code") or "").upper()
        if code == "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION":
            return True
        message = str(item.get("message") or "").upper()
        text = f"{text} {message}"
    return "CAPTURE_ECONOMICS IS NEGATIVE_EXPECTANCY" in text


def _is_strategy_profile_gap(item: Any) -> bool:
    text = _issue_text(item).upper()
    if isinstance(item, dict):
        code = str(item.get("code") or "").upper()
        if code in _STRATEGY_PROFILE_GAP_CODES:
            return True
    return text in _STRATEGY_PROFILE_GAP_CODES


def _is_self_improvement_intent_blocker(item: Any) -> bool:
    """Self-improvement gates may cite ledgers without being memory defects."""

    if isinstance(item, dict):
        code = str(item.get("code") or "").upper()
        message = str(item.get("message") or "").upper()
        return code.startswith("SELF_IMPROVEMENT") or "SELF-IMPROVEMENT" in message
    return "SELF-IMPROVEMENT" in str(item).upper()


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _to_utc(parsed)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _transaction_id_is_behind(actual: str, expected: str) -> bool:
    try:
        return int(actual) < int(expected)
    except (TypeError, ValueError):
        return actual != expected
