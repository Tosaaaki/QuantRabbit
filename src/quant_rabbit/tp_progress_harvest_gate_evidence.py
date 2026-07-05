from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .market_close_leak_gate import (
    MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS,
    MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS,
)
from .paths import (
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE,
    DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE_REPORT,
)


@dataclass(frozen=True)
class GateEvidenceSummary:
    output_path: Path
    report_path: Path
    payload: dict[str, Any]


def build_tp_progress_harvest_gate_evidence(
    timing_audit: dict[str, Any],
    *,
    generated_at_utc: str | None = None,
    source_path: Path | str = DEFAULT_EXECUTION_TIMING_AUDIT,
) -> dict[str, Any]:
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    summary = timing_audit.get("summary") if isinstance(timing_audit.get("summary"), dict) else {}
    rows = timing_audit.get("loss_close_regrets") if isinstance(timing_audit.get("loss_close_regrets"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    missed_rows = [row for row in rows if bool(row.get("profit_capture_missed_before_loss_close"))]

    evidence_rows = [_row_evidence(row, summary=summary) for row in missed_rows]
    all_loss_rows = [_row_evidence(row, summary=summary) for row in rows]

    current_replay = _month_scale_replay(all_loss_rows, exclude_family=False, exclude_manual=False)
    proposed_replay = _month_scale_replay(all_loss_rows, exclude_family=True, exclude_manual=True)
    classifications = _classification_counts(evidence_rows)
    attribution_counts = _attribution_counts(evidence_rows)

    executable_rows = [row for row in evidence_rows if row["executable_profit_capture_before_loss_close"]]
    below_noise_rows = [row for row in evidence_rows if row["trigger_status"] == "BELOW_NOISE_FLOOR"]

    return {
        "generated_at_utc": generated_at,
        "source": {
            "execution_timing_audit_generated_at_utc": timing_audit.get("generated_at_utc"),
            "execution_timing_audit_path": str(source_path),
            "lookback_hours": timing_audit.get("lookback_hours"),
        },
        "contract": {
            "historical_repair_evidence_allowed": True,
            "historical_evidence_grants_live_permission": False,
            "live_permission_result": "NO_A_S_LIVE_PERMISSION_FROM_HISTORICAL_REPLAY_ONLY",
            "future_leak_reduction_gate": (
                "TP-progress harvest may fire only when the production replay trigger is above "
                "noise floor, reaches the progress gate, and occurs before the loss close."
            ),
            "manual_trade_ids_excluded_from_system_pl": list(MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS),
            "market_close_leak_family_excluded_trade_ids": list(MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS),
        },
        "metrics": {
            "loss_closes_audited": len(rows),
            "historical_missed_capture_count": len(missed_rows),
            "current_rule_trigger_count": classifications.get("CURRENT_RULE_TRIGGER", 0),
            "executable_profit_capture_before_loss_close_count": len(executable_rows),
            "below_noise_floor_count": len(below_noise_rows),
            "manual_rows": attribution_counts.get("OPERATOR_MANUAL", 0),
            "system_gateway_rows": attribution_counts.get("SYSTEM_GATEWAY", 0),
            "unknown_gateway_lane_rows": attribution_counts.get("SYSTEM_GATEWAY_UNKNOWN_LANE", 0),
            "execution_timing_summary": {
                "loss_close_repair_replay_actual_pl_jpy": _round(summary.get("loss_close_repair_replay_actual_pl_jpy")),
                "loss_close_repair_replay_counterfactual_pl_jpy": _round(
                    summary.get("loss_close_repair_replay_counterfactual_pl_jpy")
                ),
                "loss_close_repair_replay_delta_jpy": _round(summary.get("loss_close_repair_replay_delta_jpy")),
                "loss_closes_profit_capture_missed": summary.get("loss_closes_profit_capture_missed"),
                "loss_closes_repair_replay_triggered": summary.get("loss_closes_repair_replay_triggered"),
                "tp_progress_repair_live_evidence_status": summary.get(
                    "tp_progress_repair_live_evidence_status"
                ),
            },
        },
        "trigger_classifications": classifications,
        "attribution_counts": attribution_counts,
        "missed_capture_evidence": evidence_rows,
        "month_scale_replay": {
            "current_inputs": current_replay,
            "after_proposed_gates": proposed_replay,
            "month_scale_tp_progress_replay_still_negative_clears": proposed_replay["residual_pl_jpy"] >= 0,
            "basis": (
                "Replay uses all audited loss closes. Triggered rows use executable TP-progress "
                "counterfactual P/L; non-triggered rows keep actual P/L. Proposed gates remove "
                "the exact market-close leak family and operator-manual trade ids from future "
                "system-edge accounting."
            ),
        },
    }


def write_tp_progress_harvest_gate_evidence(
    *,
    timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
    output_path: Path = DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE,
    report_path: Path = DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE_REPORT,
) -> GateEvidenceSummary:
    timing_audit = json.loads(timing_audit_path.read_text(encoding="utf-8"))
    payload = build_tp_progress_harvest_gate_evidence(timing_audit, source_path=timing_audit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(payload), encoding="utf-8")
    return GateEvidenceSummary(output_path=output_path, report_path=report_path, payload=payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TP-progress harvest gate evidence.")
    parser.add_argument("--execution-timing-audit", type=Path, default=DEFAULT_EXECUTION_TIMING_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE_REPORT)
    args = parser.parse_args(argv)
    write_tp_progress_harvest_gate_evidence(
        timing_audit_path=args.execution_timing_audit,
        output_path=args.output,
        report_path=args.report,
    )
    return 0


def _row_evidence(row: dict[str, Any], *, summary: dict[str, Any]) -> dict[str, Any]:
    trigger_at = _parse_utc(row.get("repair_replay_trigger_at_utc"))
    close_at = _parse_utc(row.get("close_at_utc"))
    triggered = bool(row.get("repair_replay_triggered_before_loss_close"))
    before_loss_close = bool(trigger_at and close_at and trigger_at < close_at)
    profit_pips = _float_or_none(row.get("repair_replay_profit_pips"))
    noise_floor_pips = _float_or_none(row.get("repair_replay_noise_floor_pips"))
    above_noise = (
        profit_pips is not None
        and noise_floor_pips is not None
        and profit_pips > noise_floor_pips
    )
    counterfactual_pl = _float_or_none(row.get("repair_replay_counterfactual_pl_jpy"))
    actual_pl = _float_or_none(row.get("realized_pl_jpy")) or 0.0
    executable = (
        triggered
        and before_loss_close
        and above_noise
        and counterfactual_pl is not None
        and counterfactual_pl > actual_pl
        and str(row.get("repair_replay_exit") or "") == "TP_PROGRESS_PRODUCTION_GATE_REPLAY"
    )
    block_reason = str(row.get("repair_replay_block_reason") or "")
    if triggered:
        trigger_status = "CURRENT_RULE_TRIGGER"
    elif block_reason == "BELOW_NOISE_FLOOR":
        trigger_status = "BELOW_NOISE_FLOOR"
    elif block_reason:
        trigger_status = block_reason
    else:
        trigger_status = "NO_CURRENT_RULE_TRIGGER"

    return {
        "trade_id": str(row.get("trade_id") or ""),
        "pair": row.get("pair"),
        "side": row.get("side"),
        "method": _method(row),
        "lane_id": row.get("lane_id"),
        "exit_reason": row.get("exit_reason"),
        "attribution": _attribution(row),
        "historical_vs_live": _historical_vs_live(row, summary),
        "profit_capture_missed_before_loss_close": bool(row.get("profit_capture_missed_before_loss_close")),
        "trigger_status": trigger_status,
        "repair_replay_block_reason": block_reason or None,
        "trigger_before_loss_close": before_loss_close,
        "above_noise_floor": above_noise,
        "executable_profit_capture_before_loss_close": executable,
        "realized_pl_jpy": _round(actual_pl),
        "repair_replay_pl_jpy": _round(counterfactual_pl if executable else actual_pl),
        "repair_replay_counterfactual_pl_jpy": _round(counterfactual_pl),
        "repair_replay_delta_jpy": _round((counterfactual_pl - actual_pl) if executable and counterfactual_pl is not None else 0.0),
        "repair_replay_profit_pips": _round(profit_pips),
        "repair_replay_noise_floor_pips": _round(noise_floor_pips),
        "repair_replay_tp_progress": _round(row.get("repair_replay_tp_progress")),
        "repair_replay_progress_gate": _round(row.get("repair_replay_progress_gate")),
        "repair_replay_trigger_at_utc": row.get("repair_replay_trigger_at_utc"),
        "close_at_utc": row.get("close_at_utc"),
        "excluded_by_market_close_leak_family_gate": str(row.get("trade_id") or "") in MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS,
        "excluded_as_operator_manual": str(row.get("trade_id") or "") in MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS,
    }


def _month_scale_replay(
    rows: list[dict[str, Any]],
    *,
    exclude_family: bool,
    exclude_manual: bool,
) -> dict[str, Any]:
    included = []
    excluded_ids = []
    for row in rows:
        if exclude_family and row["excluded_by_market_close_leak_family_gate"]:
            excluded_ids.append(row["trade_id"])
            continue
        if exclude_manual and row["excluded_as_operator_manual"]:
            excluded_ids.append(row["trade_id"])
            continue
        included.append(row)
    baseline = sum(float(row["realized_pl_jpy"] or 0.0) for row in included)
    improved = sum(float(row["repair_replay_pl_jpy"] or 0.0) for row in included)
    return {
        "rows": len(included),
        "excluded_trade_ids": excluded_ids,
        "baseline_pl_jpy": _round(baseline),
        "improved_pl_jpy": _round(improved),
        "delta_jpy": _round(improved - baseline),
        "residual_pl_jpy": _round(improved),
        "residual_groups": _residual_groups(included),
    }


def _residual_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        residual = float(row["repair_replay_pl_jpy"] or 0.0)
        if residual >= 0:
            continue
        key = (str(row.get("pair") or ""), str(row.get("side") or ""), str(row.get("method") or "UNKNOWN"))
        group = groups.setdefault(
            key,
            {
                "pair": key[0],
                "side": key[1],
                "method": key[2],
                "rows": 0,
                "baseline_pl_jpy": 0.0,
                "residual_pl_jpy": 0.0,
                "triggered_rows": 0,
                "block_reasons": defaultdict(int),
                "examples": [],
            },
        )
        group["rows"] += 1
        group["baseline_pl_jpy"] += float(row["realized_pl_jpy"] or 0.0)
        group["residual_pl_jpy"] += residual
        if row.get("executable_profit_capture_before_loss_close"):
            group["triggered_rows"] += 1
        reason = str(row.get("repair_replay_block_reason") or row.get("trigger_status") or "")
        if reason:
            group["block_reasons"][reason] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(
                {
                    "trade_id": row.get("trade_id"),
                    "lane_id": row.get("lane_id"),
                    "realized_pl_jpy": row.get("realized_pl_jpy"),
                    "repair_replay_pl_jpy": row.get("repair_replay_pl_jpy"),
                    "trigger_status": row.get("trigger_status"),
                }
            )
    result = []
    for group in groups.values():
        result.append(
            {
                "pair": group["pair"],
                "side": group["side"],
                "method": group["method"],
                "rows": group["rows"],
                "triggered_rows": group["triggered_rows"],
                "baseline_pl_jpy": _round(group["baseline_pl_jpy"]),
                "residual_pl_jpy": _round(group["residual_pl_jpy"]),
                "block_reasons": dict(sorted(group["block_reasons"].items())),
                "examples": group["examples"],
            }
        )
    return sorted(result, key=lambda item: float(item["residual_pl_jpy"]))[:12]


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("trigger_status") or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def _attribution_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("attribution") or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def _method(row: dict[str, Any]) -> str:
    lane_id = str(row.get("lane_id") or "")
    parts = [part for part in lane_id.split(":") if part]
    if len(parts) >= 4:
        return parts[3]
    return str(row.get("method") or "UNKNOWN")


def _attribution(row: dict[str, Any]) -> str:
    trade_id = str(row.get("trade_id") or "")
    if trade_id in MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS:
        return "OPERATOR_MANUAL"
    owner = str(row.get("owner") or row.get("source_owner") or "").strip().lower()
    if owner in {"operator_manual", "manual"}:
        return "OPERATOR_MANUAL"
    if row.get("lane_id"):
        return "SYSTEM_GATEWAY"
    return "SYSTEM_GATEWAY_UNKNOWN_LANE"


def _historical_vs_live(row: dict[str, Any], summary: dict[str, Any]) -> str:
    boundary = _parse_utc(summary.get("tp_progress_repair_live_evidence_boundary_utc"))
    fill_at = _parse_utc(row.get("fill_at_utc"))
    if boundary is None or fill_at is None:
        return "UNKNOWN"
    return "POST_REPAIR_LIVE_EVIDENCE" if fill_at >= boundary else "HISTORICAL_REPAIR_EVIDENCE"


def _markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    current = payload["month_scale_replay"]["current_inputs"]
    proposed = payload["month_scale_replay"]["after_proposed_gates"]
    lines = [
        "# TP Progress Harvest Gate Evidence",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- missed captures: `{metrics['historical_missed_capture_count']}`",
        f"- current-rule triggers: `{metrics['current_rule_trigger_count']}`",
        f"- executable before loss close: `{metrics['executable_profit_capture_before_loss_close_count']}`",
        f"- below noise floor: `{metrics['below_noise_floor_count']}`",
        f"- attribution: system_gateway=`{metrics['system_gateway_rows']}`, unknown_lane=`{metrics['unknown_gateway_lane_rows']}`, manual=`{metrics['manual_rows']}`",
        "",
        "## Contract",
        "",
        "- Historical repair evidence is allowed as evidence.",
        "- Historical repair evidence does not by itself create live A/S permission.",
        "- Future TP-progress harvest requires trigger-before-loss-close, above-noise-floor proof, and production replay gate status.",
        "",
        "## Month-Scale Replay",
        "",
        f"- current baseline P/L: `{current['baseline_pl_jpy']}` JPY",
        f"- current improved P/L: `{current['improved_pl_jpy']}` JPY",
        f"- proposed-gate baseline P/L: `{proposed['baseline_pl_jpy']}` JPY",
        f"- proposed-gate improved P/L: `{proposed['improved_pl_jpy']}` JPY",
        f"- proposed-gate residual P/L: `{proposed['residual_pl_jpy']}` JPY",
        f"- MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE clears: `{payload['month_scale_replay']['month_scale_tp_progress_replay_still_negative_clears']}`",
        "",
        "## Residual Groups",
        "",
        "| pair | side | method | rows | residual_jpy | blockers |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for group in proposed["residual_groups"][:10]:
        blockers = ", ".join(f"{key}:{value}" for key, value in group["block_reasons"].items())
        lines.append(
            f"| {group['pair']} | {group['side']} | {group['method']} | {group['rows']} | "
            f"{group['residual_pl_jpy']} | {blockers or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Trigger Evidence",
            "",
            "| trade_id | pair | side | method | attribution | status | executable | actual_jpy | replay_jpy |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in payload["missed_capture_evidence"]:
        lines.append(
            f"| {row['trade_id']} | {row['pair']} | {row['side']} | {row['method']} | "
            f"{row['attribution']} | {row['trigger_status']} | "
            f"{row['executable_profit_capture_before_loss_close']} | {row['realized_pl_jpy']} | "
            f"{row['repair_replay_pl_jpy']} |"
        )
    return "\n".join(lines) + "\n"


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _round(value: Any) -> float | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return round(numeric, 4)


if __name__ == "__main__":
    raise SystemExit(main())
