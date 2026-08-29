"""Bounded, outcome-only EUR/USD learning for zero-authority shadow routing.

This module does not fit an unconstrained model.  It seals a deterministic
quarantine and routing policy from resolved exact-S5 outcomes before a fixed
cutoff.  Historical diagnostics and post-activation prospective evidence are
different contracts and must never share a ledger.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quant_rabbit.fast_bot_corrective_challenger import (
    ROW_CONTRACT as CORRECTIVE_ROW_CONTRACT,
    sealed_valid as corrective_sealed_valid,
)
from quant_rabbit.fast_bot_shock_follow import market_is_closed
from quant_rabbit.fast_bot_truth import OUTCOME_CONTRACT


CONFIG_CONTRACT = "QR_EURUSD_LEARNED_POLICY_CONFIG_V1"
TRAINING_ROW_CONTRACT = "QR_EURUSD_LEARNING_TRAINING_ROW_V1"
TRAINING_RECEIPT_CONTRACT = "QR_EURUSD_LEARNING_TRAINING_RECEIPT_V1"
POLICY_CONTRACT = "QR_EURUSD_LEARNED_POLICY_V1"
MANIFEST_CONTRACT = "QR_EURUSD_LEARNED_POLICY_MANIFEST_V1"
DIAGNOSTIC_ROW_CONTRACT = "QR_EURUSD_LEARNED_POLICY_DIAGNOSTIC_ROW_V1"
DIAGNOSTIC_SCORECARD_CONTRACT = "QR_EURUSD_LEARNED_POLICY_DIAGNOSTIC_SCORECARD_V1"
PROSPECTIVE_DECISION_CONTRACT = "QR_EURUSD_LEARNED_POLICY_PROSPECTIVE_DECISION_V1"
PROSPECTIVE_OUTCOME_CONTRACT = "QR_EURUSD_LEARNED_POLICY_PROSPECTIVE_OUTCOME_V1"
PROSPECTIVE_SCORECARD_CONTRACT = "QR_EURUSD_LEARNED_POLICY_PROSPECTIVE_SCORECARD_V1"
CURRENT_POINTER_CONTRACT = "QR_EURUSD_LEARNED_POLICY_CURRENT_POINTER_V1"

PAIR = "EUR_USD"
CHOICES = (
    "NO_TRADE",
    "SHOCK_BREAKOUT_FOLLOW",
    "SHOCK_PULLBACK_CONTINUATION",
    "TREND_CONTINUATION",
)
UNKNOWN = "UNKNOWN_NOT_CAPTURED"
DECISION_FEATURES = (
    "pair",
    "side",
    "strategy",
    "regime",
    "atr_ratio",
    "atr_bucket",
    "impulse_direction",
    "impulse_magnitude_atr",
    "spread_to_atr",
    "session",
    "higher_timeframe_alignment",
)
LABEL_FIELDS = (
    "filled",
    "fill_at_utc",
    "exit_reason",
    "exit_at_utc",
    "realized_pips",
    "mfe_pips",
    "mae_pips",
    "time_to_stop_seconds",
)
def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and value.get("contract_sha256") == canonical_sha(body)


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aware timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("aware datetime is required")
    return value.astimezone(timezone.utc)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(row)
    return rows


def _finite(value: Any, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite number is required") from exc
    if not math.isfinite(result):
        raise ValueError("finite number is required")
    return result


def _config_without_seal(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in config.items() if key != "contract_sha256"}


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != CONFIG_CONTRACT:
        raise ValueError("EURUSD learned policy config contract mismatch")
    if config.get("pair") != PAIR:
        raise ValueError("EURUSD learned policy pair mismatch")
    authority = config.get("authority") or {}
    if (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("automatic_adoption_allowed") is not False
        or authority.get("automatic_parameter_change_allowed") is not False
    ):
        raise ValueError("EURUSD learned policy authority boundary mismatch")
    evidence = config.get("evidence") or {}
    if (
        evidence.get("lookahead_policy") != "SIGNAL_TIME_FEATURES_ONLY"
        or evidence.get("retrospective_reinterpretation_allowed") is not False
        or evidence.get("historical_rows_are_diagnostic_only") is not True
        or evidence.get("pre_activation_rows_count_as_forward_evidence") is not False
        or evidence.get("post_activation_rows_are_prospective_only") is not True
    ):
        raise ValueError("EURUSD learned policy evidence boundary mismatch")
    feature = config.get("feature_contract") or {}
    if tuple(feature.get("allowed_decision_features") or ()) != DECISION_FEATURES:
        raise ValueError("decision feature allowlist mismatch")
    if tuple(feature.get("diagnostic_only_labels") or ()) != LABEL_FIELDS:
        raise ValueError("diagnostic label allowlist mismatch")
    router = config.get("router") or {}
    if tuple(router.get("choices") or ()) != CHOICES:
        raise ValueError("bounded router choices mismatch")
    selected = router.get("selected_thresholds") or {}
    allowlist = router.get("threshold_allowlist") or {}
    if set(selected) != set(allowlist):
        raise ValueError("threshold selection/allowlist mismatch")
    for key, value in selected.items():
        if value not in (allowlist.get(key) or []):
            raise ValueError(f"threshold outside allowlist: {key}")
    profiles = config.get("training", {}).get("quarantine_profile_allowlist") or []
    selected_profile = config.get("training", {}).get("selected_quarantine_profile_id")
    if len([row for row in profiles if row.get("profile_id") == selected_profile]) != 1:
        raise ValueError("selected quarantine profile is not uniquely allowlisted")
    parse_utc(config.get("training", {}).get("cutoff_at_utc"))
    return config, canonical_sha(_config_without_seal(config))


def _signal_seal_valid(row: Mapping[str, Any]) -> bool:
    body = {key: item for key, item in row.items() if key != "signal_sha256"}
    return (
        row.get("contract") == "QR_FAST_BOT_SHADOW_SIGNAL_V1"
        and row.get("signal_sha256") == canonical_sha(body)
        and row.get("shadow_only") is True
        and row.get("live_permission") is False
        and row.get("broker_mutation_allowed") is False
    )


def _outcome_seal_valid(row: Mapping[str, Any]) -> bool:
    body = {key: item for key, item in row.items() if key != "contract_sha256"}
    return (
        row.get("contract") == OUTCOME_CONTRACT
        and row.get("contract_sha256") == canonical_sha(body)
        and row.get("truth_source") == "OANDA_S5_BID_ASK"
        and row.get("truth_request_coverage_proved") is True
        and row.get("broker_mutation") is False
        and row.get("live_permission") is False
    )


def _session(at: datetime) -> str:
    hour = aware_utc(at).hour
    if hour < 7:
        return "ASIA_UTC"
    if hour < 13:
        return "LONDON_UTC"
    if hour < 17:
        return "LONDON_NEW_YORK_OVERLAP_UTC"
    if hour < 22:
        return "NEW_YORK_UTC"
    return "ROLLOVER_UTC"


def _training_row(
    signal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    generated = parse_utc(signal.get("generated_at_utc"))
    resolved = parse_utc(outcome.get("resolved_at_utc"))
    evaluated = parse_utc(baseline.get("evaluated_at_utc"))
    maturity = parse_utc(outcome.get("maturity_at_utc"))
    if not generated < maturity <= resolved <= evaluated:
        raise ValueError(f"outcome chronology invalid for {signal.get('signal_id')}")
    atr = _finite(signal.get("m5_atr_pips"))
    spread = _finite(signal.get("spread_pips"))
    if atr is None or atr <= 0.0 or spread is None or spread < 0.0:
        raise ValueError("signal-time ATR/spread invalid")
    features = {
        "pair": PAIR,
        "side": str(signal.get("side")),
        "strategy": str(signal.get("method")),
        "regime": str(baseline.get("regime_bucket")),
        "atr_ratio": baseline.get("causal_atr_ratio"),
        "atr_bucket": str(baseline.get("atr_bucket")),
        "impulse_direction": UNKNOWN,
        "impulse_magnitude_atr": None,
        "spread_to_atr": round(spread / atr, 6),
        "session": _session(generated),
        "higher_timeframe_alignment": UNKNOWN,
    }
    if set(features) != set(DECISION_FEATURES):
        raise ValueError("decision feature contract mismatch")
    labels = {key: baseline.get(key) for key in LABEL_FIELDS}
    body = {
        "contract": TRAINING_ROW_CONTRACT,
        "schema_version": 1,
        "signal_id": str(signal.get("signal_id")),
        "signal_sha256": str(signal.get("signal_sha256")),
        "outcome_sha256": str(outcome.get("contract_sha256")),
        "corrective_baseline_sha256": str(baseline.get("contract_sha256")),
        "signal_generated_at_utc": generated.isoformat(),
        "maturity_at_utc": maturity.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "entered_training_at_utc": evaluated.isoformat(),
        "features": features,
        "labels": labels,
        "diagnostic_context": {
            "existing_vol_shock_veto": baseline.get("vol_shock") is True,
            "existing_vol_shock_reasons": list(baseline.get("vol_shock_reasons") or []),
            "rapid_time_bucket_utc": baseline.get("rapid_time_bucket_utc"),
        },
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_chunk_sha256": list(outcome.get("truth_chunk_sha256") or []),
        "outcome_fields_used_as_decision_features": [],
        "lookahead_used": False,
        "execution_authority": "NONE",
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
        "automatic_adoption_allowed": False,
    }
    return seal(body)


def build_training_rows(
    *,
    signal_ledger_path: Path,
    outcome_ledger_path: Path,
    corrective_ledger_path: Path,
    cutoff_at_utc: datetime,
    now_utc: datetime,
    maximum_resolution_lag_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cutoff = aware_utc(cutoff_at_utc)
    now = aware_utc(now_utc)
    if cutoff > now:
        raise ValueError("training cutoff is in the future")
    signals = load_jsonl(signal_ledger_path)
    outcomes = load_jsonl(outcome_ledger_path)
    corrective = load_jsonl(corrective_ledger_path)
    signal_by_id: dict[str, Mapping[str, Any]] = {}
    for row in signals:
        if row.get("pair") != PAIR or parse_utc(row.get("generated_at_utc")) > cutoff:
            continue
        if not _signal_seal_valid(row):
            raise ValueError("signal ledger seal mismatch")
        signal_id = str(row.get("signal_id") or "")
        if not signal_id or signal_id in signal_by_id:
            raise ValueError("duplicate or missing signal identity")
        signal_by_id[signal_id] = row
    outcome_by_id: dict[str, Mapping[str, Any]] = {}
    last_resolved: datetime | None = None
    for row in outcomes:
        if row.get("pair") != PAIR:
            continue
        signal_id = str(row.get("signal_id") or "")
        if signal_id not in signal_by_id:
            continue
        if not _outcome_seal_valid(row):
            raise ValueError("outcome ledger seal or S5 truth mismatch")
        resolved = parse_utc(row.get("resolved_at_utc"))
        if resolved > cutoff:
            continue
        if resolved > now:
            raise ValueError("future resolved outcome")
        if last_resolved is not None and resolved < last_resolved:
            raise ValueError("out-of-order resolved outcome")
        last_resolved = resolved
        generated = parse_utc(signal_by_id[signal_id].get("generated_at_utc"))
        if not 0 < (resolved - generated).total_seconds() <= maximum_resolution_lag_seconds:
            raise ValueError("stale or non-causal outcome resolution")
        if signal_id in outcome_by_id:
            raise ValueError("duplicate resolved outcome")
        outcome_by_id[signal_id] = row
    baseline_by_id: dict[str, Mapping[str, Any]] = {}
    last_evaluated: datetime | None = None
    for row in corrective:
        if row.get("pair") != PAIR or row.get("arm_id") != "BASELINE":
            continue
        signal_id = str(row.get("signal_id") or "")
        if signal_id not in outcome_by_id:
            continue
        if not corrective_sealed_valid(row, CORRECTIVE_ROW_CONTRACT):
            raise ValueError("corrective baseline seal mismatch")
        evaluated = parse_utc(row.get("evaluated_at_utc"))
        if evaluated > now:
            raise ValueError("future corrective evaluation")
        if last_evaluated is not None and evaluated < last_evaluated:
            raise ValueError("out-of-order corrective evaluation")
        last_evaluated = evaluated
        if row.get("outcome_sha256") != outcome_by_id[signal_id].get("contract_sha256"):
            raise ValueError("corrective/outcome hash binding mismatch")
        if row.get("truth_hash_match") is not True:
            raise ValueError("corrective truth hash mismatch")
        if signal_id in baseline_by_id:
            raise ValueError("duplicate corrective baseline")
        baseline_by_id[signal_id] = row
    unresolved = sorted(set(signal_by_id) - set(outcome_by_id))
    if unresolved:
        raise ValueError("unresolved signal at or before training cutoff")
    missing_baseline = sorted(set(outcome_by_id) - set(baseline_by_id))
    if missing_baseline:
        raise ValueError("resolved outcome missing corrective baseline")
    rows = [
        _training_row(signal_by_id[signal_id], outcome_by_id[signal_id], baseline_by_id[signal_id])
        for signal_id in signal_by_id
    ]
    rows.sort(key=lambda row: (row["resolved_at_utc"], row["signal_id"]))
    if not rows:
        raise ValueError("no resolved EURUSD rows at cutoff")
    for row in rows:
        if not sealed_valid(row, TRAINING_ROW_CONTRACT):
            raise ValueError("training row seal mismatch")
        if set(row["features"]) != set(DECISION_FEATURES):
            raise ValueError("training decision feature mismatch")
        if set(row["labels"]) != set(LABEL_FIELDS):
            raise ValueError("training label mismatch")
    hashes = {
        "signal_ledger_sha256": file_sha(signal_ledger_path),
        "outcome_ledger_sha256": file_sha(outcome_ledger_path),
        "corrective_ledger_sha256": file_sha(corrective_ledger_path),
    }
    hashes["input_ledger_sha256"] = canonical_sha(hashes)
    return rows, hashes


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row.get("signal_generated_at_utc") or ""), str(row.get("signal_id") or "")))
    eligible = [row for row in ordered if row.get("eligible") is not False]
    filled = [row for row in eligible if row.get("filled") is True]
    values = [float(row.get("realized_pips") or 0.0) for row in filled]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    streak = maximum = 0
    for value in values:
        streak = streak + 1 if value < 0.0 else 0
        maximum = max(maximum, streak)
    tail_count = max(1, math.ceil(len(values) * 0.05)) if values else 0
    gross_loss = abs(sum(losses))
    return {
        "signal_count": len(ordered),
        "eligible_count": len(eligible),
        "filled_count": len(filled),
        "fill_rate": round(len(filled) / len(eligible), 6) if eligible else None,
        "win_rate": round(len(wins) / len(filled), 6) if filled else None,
        "net_pips": round(sum(values), 6),
        "profit_factor": round(sum(wins) / gross_loss, 6) if gross_loss else "INF" if wins else None,
        "max_consecutive_losses": maximum,
        "tail_5pct_loss_pips": round(sum(sorted(values)[:tail_count]), 6),
        "mean_mfe_pips": _mean(float(row.get("mfe_pips") or 0.0) for row in filled),
        "mean_mae_pips": _mean(float(row.get("mae_pips") or 0.0) for row in filled),
    }


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _training_cell_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        feature = row["features"]
        label = row["labels"]
        grouped[(feature["regime"], feature["strategy"], feature["side"])].append(
            {
                "signal_id": row["signal_id"],
                "signal_generated_at_utc": row["signal_generated_at_utc"],
                "eligible": True,
                **label,
            }
        )
    return [
        {
            "regime": key[0],
            "strategy": key[1],
            "side": key[2],
            "resolved_count": len(group),
            **_metrics(group),
        }
        for key, group in sorted(grouped.items())
    ]


def _selected_quarantine_profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    profile_id = config["training"]["selected_quarantine_profile_id"]
    return next(row for row in config["training"]["quarantine_profile_allowlist"] if row["profile_id"] == profile_id)


def _quarantine_cells(cells: Sequence[Mapping[str, Any]], profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    quarantined: list[dict[str, Any]] = []
    for cell in cells:
        pf = cell["profit_factor"]
        numeric_pf = float(pf) if isinstance(pf, (int, float)) else float("inf")
        if (
            int(cell["resolved_count"]) >= int(profile["min_resolved"])
            and int(cell["filled_count"]) >= int(profile["min_filled"])
            and float(cell["net_pips"]) < float(profile["max_net_pips"])
            and numeric_pf < float(profile["max_profit_factor"])
        ):
            quarantined.append(
                {
                    "regime": cell["regime"],
                    "strategy": cell["strategy"],
                    "side": cell["side"],
                    "reason": "BOUNDED_NEGATIVE_CELL",
                    "resolved_count": cell["resolved_count"],
                    "filled_count": cell["filled_count"],
                    "net_pips": cell["net_pips"],
                    "profit_factor": cell["profit_factor"],
                }
            )
    return quarantined


def train_policy(
    *,
    training_rows: Sequence[Mapping[str, Any]],
    input_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    config_sha256: str,
    activation_at_utc: datetime,
) -> dict[str, Any]:
    if not training_rows:
        raise ValueError("training rows required")
    for row in training_rows:
        if not sealed_valid(row, TRAINING_ROW_CONTRACT):
            raise ValueError("training row seal mismatch")
    activation = aware_utc(activation_at_utc)
    cutoff = parse_utc(config["training"]["cutoff_at_utc"])
    max_training_time = max(
        max(parse_utc(row["resolved_at_utc"]), parse_utc(row["entered_training_at_utc"]))
        for row in training_rows
    )
    if activation <= max(cutoff, max_training_time):
        raise ValueError("activation must be strictly after every training row and cutoff")
    days = sorted({parse_utc(row["resolved_at_utc"]).date().isoformat() for row in training_rows})
    if len(training_rows) < int(config["training"]["minimum_initial_resolved_samples"]) or len(days) < int(config["training"]["minimum_initial_resolved_days"]):
        raise ValueError("initial training evidence floor not met")
    cells = _training_cell_metrics(training_rows)
    profile = _selected_quarantine_profile(config)
    quarantined = _quarantine_cells(cells, profile)
    training_rows_sha256 = canonical_sha([row["contract_sha256"] for row in training_rows])
    receipt = seal(
        {
            "contract": TRAINING_RECEIPT_CONTRACT,
            "schema_version": 1,
            "pair": PAIR,
            "training_cutoff_at_utc": cutoff.isoformat(),
            "activation_at_utc": activation.isoformat(),
            "training_row_count": len(training_rows),
            "resolved_day_count": len(days),
            "resolved_days": days,
            "first_signal_at_utc": min(row["signal_generated_at_utc"] for row in training_rows),
            "last_signal_at_utc": max(row["signal_generated_at_utc"] for row in training_rows),
            "last_resolved_at_utc": max(row["resolved_at_utc"] for row in training_rows),
            "last_entered_training_at_utc": max(row["entered_training_at_utc"] for row in training_rows),
            "training_rows_sha256": training_rows_sha256,
            **dict(input_hashes),
            "config_sha256": config_sha256,
            "decision_features": list(DECISION_FEATURES),
            "diagnostic_only_labels": list(LABEL_FIELDS),
            "outcome_fields_used_as_decision_features": [],
            "unresolved_rows_used": 0,
            "future_rows_used": 0,
            "lookahead_used": False,
            "execution_authority": "NONE",
            "external_order_attempts": 0,
            "external_orders": 0,
            "automatic_adoption_allowed": False,
            "live_permission": False,
        }
    )
    policy = seal(
        {
            "contract": POLICY_CONTRACT,
            "schema_version": 1,
            "policy_name": "EURUSD learned policy v1",
            "pair": PAIR,
            "trained_at_utc": activation.isoformat(),
            "activation_at_utc": activation.isoformat(),
            "training_cutoff_at_utc": cutoff.isoformat(),
            "training_receipt_sha256": receipt["contract_sha256"],
            "input_ledger_sha256": input_hashes["input_ledger_sha256"],
            "config_sha256": config_sha256,
            "router_choices": list(CHOICES),
            "router_thresholds": dict(config["router"]["selected_thresholds"]),
            "quarantine_profile_id": profile["profile_id"],
            "quarantined_cells": quarantined,
            "mandatory_rules": [
                "SHOCK_OR_NEGATIVE_REGIME_RANGE_ROTATION_LONG_NO_TRADE",
                "MISSING_EX_ANTE_FEATURE_NO_TRADE",
                "STALE_FUTURE_OR_OUT_OF_ORDER_NO_TRADE",
                "SPREAD_ATR_LIMIT_NO_TRADE",
            ],
            "historical_rows_are_diagnostic_only": True,
            "pre_activation_rows_count_as_forward_evidence": False,
            "post_activation_rows_are_prospective_only": True,
            "status": "TEST_REQUIRED",
            "profitability_claim_allowed": False,
            "automatic_adoption_allowed": False,
            "automatic_parameter_change_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "manual_tagless_positions_policy": "NO_TOUCH",
            "existing_tp_sl_policy": "NO_TOUCH",
        }
    )
    manifest = seal(
        {
            "contract": MANIFEST_CONTRACT,
            "schema_version": 1,
            "policy": policy,
            "training_receipt": receipt,
            "policy_sha256": policy["contract_sha256"],
            "training_receipt_sha256": receipt["contract_sha256"],
            "config_sha256": config_sha256,
            "input_ledger_sha256": input_hashes["input_ledger_sha256"],
            "activation_at_utc": activation.isoformat(),
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
            "execution_authority": "NONE",
        }
    )
    verify_manifest(manifest, config=config, config_sha256=config_sha256, training_rows=training_rows)
    return {"training_receipt": receipt, "policy": policy, "manifest": manifest, "cell_metrics": cells}


def verify_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    training_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if not sealed_valid(manifest, MANIFEST_CONTRACT):
        raise ValueError("learned policy manifest seal mismatch")
    policy = manifest.get("policy")
    receipt = manifest.get("training_receipt")
    if not isinstance(policy, Mapping) or not sealed_valid(policy, POLICY_CONTRACT):
        raise ValueError("learned policy seal mismatch")
    if not isinstance(receipt, Mapping) or not sealed_valid(receipt, TRAINING_RECEIPT_CONTRACT):
        raise ValueError("training receipt seal mismatch")
    if manifest.get("policy_sha256") != policy.get("contract_sha256") or manifest.get("training_receipt_sha256") != receipt.get("contract_sha256"):
        raise ValueError("manifest content address mismatch")
    if manifest.get("config_sha256") != config_sha256 or policy.get("config_sha256") != config_sha256 or receipt.get("config_sha256") != config_sha256:
        raise ValueError("config hash mismatch")
    if tuple(policy.get("router_choices") or ()) != CHOICES:
        raise ValueError("policy router choices mismatch")
    if policy.get("router_thresholds") != config["router"]["selected_thresholds"]:
        raise ValueError("policy thresholds are not the sealed allowlisted selection")
    activation = parse_utc(policy.get("activation_at_utc"))
    cutoff = parse_utc(receipt.get("training_cutoff_at_utc"))
    last_resolved = parse_utc(receipt.get("last_resolved_at_utc"))
    last_training = parse_utc(receipt.get("last_entered_training_at_utc"))
    if activation <= max(cutoff, last_resolved, last_training):
        raise ValueError("policy activation chronology mismatch")
    if training_rows is not None:
        if int(receipt.get("training_row_count") or -1) != len(training_rows):
            raise ValueError("training row count mismatch")
        expected = canonical_sha([row.get("contract_sha256") for row in training_rows])
        if receipt.get("training_rows_sha256") != expected:
            raise ValueError("training rows hash mismatch")
    if any(
        value is not False
        for value in (
            policy.get("automatic_adoption_allowed"),
            policy.get("promotion_allowed"),
            policy.get("live_permission"),
        )
    ) or policy.get("execution_authority") != "NONE":
        raise ValueError("learned policy authority mismatch")


def _is_quarantined(features: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    identity = (features.get("regime"), features.get("strategy"), features.get("side"))
    return any(
        identity == (row.get("regime"), row.get("strategy"), row.get("side"))
        for row in policy.get("quarantined_cells") or []
    )


def route_state(
    features: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    observed_at_utc: datetime,
    now_utc: datetime,
    maximum_age_seconds: int = 120,
) -> dict[str, Any]:
    observed = aware_utc(observed_at_utc)
    now = aware_utc(now_utc)
    activation = parse_utc(policy.get("activation_at_utc"))
    if observed < activation:
        return {"choice": "NO_TRADE", "reason": "PRE_ACTIVATION_DIAGNOSTIC_ONLY"}
    age = (now - observed).total_seconds()
    if age < 0 or age > maximum_age_seconds:
        return {"choice": "NO_TRADE", "reason": "STALE_OR_FUTURE_STATE"}
    if features.get("pair") != PAIR or set(features) != set(DECISION_FEATURES):
        return {"choice": "NO_TRADE", "reason": "FEATURE_CONTRACT_MISMATCH"}
    if _is_quarantined(features, policy):
        return {"choice": "NO_TRADE", "reason": "QUARANTINED_CELL"}
    regime = str(features.get("regime"))
    strategy = str(features.get("strategy"))
    side = str(features.get("side"))
    atr_ratio = _finite(features.get("atr_ratio"), allow_none=True)
    impulse = _finite(features.get("impulse_magnitude_atr"), allow_none=True)
    spread_ratio = _finite(features.get("spread_to_atr"), allow_none=True)
    alignment = str(features.get("higher_timeframe_alignment"))
    impulse_direction = str(features.get("impulse_direction"))
    thresholds = policy["router_thresholds"]
    shock = atr_ratio is not None and atr_ratio >= float(thresholds["shock_atr_ratio_min"])
    if (shock or regime == "REGIME_NEGATIVE") and strategy == "RANGE_ROTATION" and side == "LONG":
        return {"choice": "NO_TRADE", "reason": "RANGE_ROTATION_LONG_FORBIDDEN"}
    if spread_ratio is None or spread_ratio > float(thresholds["max_spread_to_atr"]):
        return {"choice": "NO_TRADE", "reason": "SPREAD_ATR_LIMIT"}
    if shock:
        if impulse is None or impulse < float(thresholds["shock_impulse_atr_min"]):
            return {"choice": "NO_TRADE", "reason": "SHOCK_IMPULSE_NOT_CONFIRMED"}
        if impulse_direction != side or alignment not in {"ALIGNED", side}:
            return {"choice": "NO_TRADE", "reason": "SHOCK_DIRECTION_NOT_ALIGNED"}
        if strategy in {"SHOCK_BREAKOUT_FOLLOW", "SHOCK_PULLBACK_CONTINUATION"}:
            return {"choice": strategy, "reason": "BOUNDED_SHOCK_FOLLOW"}
        return {"choice": "SHOCK_BREAKOUT_FOLLOW", "reason": "BOUNDED_SHOCK_BREAKOUT_ROUTE"}
    if regime == "REGIME_POSITIVE" and strategy == "TREND_CONTINUATION" and alignment in {"ALIGNED", side, UNKNOWN}:
        return {"choice": "TREND_CONTINUATION", "reason": "NON_SHOCK_TREND_CELL"}
    return {"choice": "NO_TRADE", "reason": "NO_ALLOWLISTED_ROUTE"}


def _diagnostic_choice(features: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    if _is_quarantined(features, policy):
        return {"choice": "NO_TRADE", "reason": "QUARANTINED_CELL"}
    if features.get("strategy") == "RANGE_ROTATION" and features.get("side") == "LONG" and (
        features.get("regime") == "REGIME_NEGATIVE" or (features.get("atr_ratio") or 0.0) >= policy["router_thresholds"]["shock_atr_ratio_min"]
    ):
        return {"choice": "NO_TRADE", "reason": "RANGE_ROTATION_LONG_FORBIDDEN"}
    if features.get("strategy") == "TREND_CONTINUATION" and features.get("regime") == "REGIME_POSITIVE":
        return {"choice": "TREND_CONTINUATION", "reason": "DIAGNOSTIC_TREND_ROUTE"}
    return {"choice": "NO_TRADE", "reason": "NO_ALLOWLISTED_ROUTE"}


def build_diagnostic_rows(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in training_rows:
        features = row["features"]
        labels = row["labels"]
        vol_shock = row.get("diagnostic_context", {}).get("existing_vol_shock_veto") is True
        learned = _diagnostic_choice(features, policy)
        arms = {
            "BASELINE": (True, "BASELINE"),
            "EXISTING_VETO": (not vol_shock, "VOL_SHOCK_VETO" if vol_shock else "PASS"),
            "EXISTING_SHOCK_FOLLOW": (False, "PROSPECTIVE_STRATEGY_NOT_AVAILABLE_PRE_ACTIVATION"),
            "LEARNED_ROUTER": (learned["choice"] != "NO_TRADE", learned["reason"]),
        }
        for arm, (eligible, reason) in arms.items():
            body = {
                "contract": DIAGNOSTIC_ROW_CONTRACT,
                "schema_version": 1,
                "diagnostic_identity": canonical_sha([policy["contract_sha256"], row["signal_id"], arm]),
                "policy_sha256": policy["contract_sha256"],
                "training_row_sha256": row["contract_sha256"],
                "arm_id": arm,
                "signal_id": row["signal_id"],
                "signal_generated_at_utc": row["signal_generated_at_utc"],
                **features,
                "choice": learned["choice"] if arm == "LEARNED_ROUTER" else arm,
                "eligible": eligible,
                "veto_reason": None if eligible else reason,
                **labels,
                "evidence_mode": "PRE_ACTIVATION_DIAGNOSTIC_ONLY",
                "counts_as_forward_evidence": False,
                "profitability_claim_allowed": False,
                "execution_authority": "NONE",
                "external_order_attempts": 0,
                "external_orders": 0,
                "live_permission": False,
                "automatic_adoption_allowed": False,
            }
            result.append(seal(body))
    return result


def _groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return [
        {**dict(zip(keys, identity)), **_metrics(group)}
        for identity, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def build_diagnostic_scorecard(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    generated_at_utc: datetime,
    leave_block_out: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for row in rows:
        if not sealed_valid(row, DIAGNOSTIC_ROW_CONTRACT) or row.get("policy_sha256") != policy.get("contract_sha256"):
            raise ValueError("diagnostic row seal or policy binding mismatch")
        if row.get("counts_as_forward_evidence") is not False:
            raise ValueError("diagnostic row leaked into forward evidence")
    arms = ("BASELINE", "EXISTING_VETO", "EXISTING_SHOCK_FOLLOW", "LEARNED_ROUTER")
    comparison = [
        {"arm_id": arm, **_metrics([row for row in rows if row.get("arm_id") == arm])}
        for arm in arms
    ]
    body = {
        "contract": DIAGNOSTIC_SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": aware_utc(generated_at_utc).isoformat(),
        "policy_sha256": policy["contract_sha256"],
        "evidence_mode": "PRE_ACTIVATION_DIAGNOSTIC_ONLY",
        "comparison": comparison,
        "by_regime_strategy_side_session": _groups(rows, ["arm_id", "regime", "strategy", "side", "session"]),
        "leave_time_block_out": dict(leave_block_out or {"status": "NOT_RUN"}),
        "pre_activation_row_count": len(rows),
        "post_activation_prospective_row_count": 0,
        "counts_as_forward_evidence": False,
        "profitability_claim_allowed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
        "external_order_attempts": 0,
        "external_orders": 0,
        "limitations": [
            "SINGLE_RESOLVED_DAY_DIAGNOSTIC_IS_NOT_PROFITABILITY_PROOF",
            "HISTORICAL_IMPULSE_AND_HIGHER_TIMEFRAME_FEATURES_WERE_NOT_CAPTURED",
            "EXISTING_SHOCK_FOLLOW_IS_PROSPECTIVE_ONLY_AND_HAS_ZERO_PRE_ACTIVATION_ELIGIBLE_ROWS",
            "DIAGNOSTIC_ROWS_NEVER_COUNT_AS_FORWARD_EVIDENCE",
        ],
    }
    return seal(body)


def build_leave_block_out_diagnostic(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in training_rows:
        at = parse_utc(row["signal_generated_at_utc"])
        blocks[at.strftime("%Y-%m-%dT%H:00Z")].append(row)
    profile = _selected_quarantine_profile(config)
    all_holdout: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for block, holdout in sorted(blocks.items()):
        fit = [row for key, rows in blocks.items() if key != block for row in rows]
        quarantined = _quarantine_cells(_training_cell_metrics(fit), profile) if fit else []
        fold_policy = {**dict(policy), "quarantined_cells": quarantined}
        fold_rows: list[dict[str, Any]] = []
        for row in holdout:
            decision = _diagnostic_choice(row["features"], fold_policy)
            fold_rows.append(
                {
                    "signal_id": row["signal_id"],
                    "signal_generated_at_utc": row["signal_generated_at_utc"],
                    "eligible": decision["choice"] != "NO_TRADE",
                    **row["labels"],
                }
            )
        all_holdout.extend(fold_rows)
        summaries.append(
            {
                "holdout_block": block,
                "fit_row_count": len(fit),
                "holdout_row_count": len(holdout),
                "quarantined_cell_count": len(quarantined),
                **_metrics(fold_rows),
            }
        )
    return {
        "status": "DIAGNOSTIC_ONLY",
        "method": "LEAVE_UTC_HOUR_BLOCK_OUT_FIXED_THRESHOLDS",
        "block_count": len(blocks),
        "overall_holdout": _metrics(all_holdout),
        "blocks": summaries,
        "counts_as_forward_evidence": False,
        "profitability_claim_allowed": False,
    }


def _shock_signal_features(signal: Mapping[str, Any]) -> dict[str, Any]:
    generated = parse_utc(signal.get("generated_at_utc"))
    side = str(signal.get("side") or "")
    m5_direction = str(signal.get("m5_direction") or "")
    aligned = (side == "LONG" and m5_direction == "UP") or (side == "SHORT" and m5_direction == "DOWN")
    return {
        "pair": str(signal.get("pair") or ""),
        "side": side,
        "strategy": str(signal.get("strategy_id") or ""),
        "regime": "REGIME_SHOCK",
        "atr_ratio": signal.get("m1_atr_expansion_ratio"),
        "atr_bucket": str(signal.get("shock_bucket") or ""),
        "impulse_direction": str(signal.get("direction") or ""),
        "impulse_magnitude_atr": signal.get("m1_impulse_body_atr_ratio"),
        "spread_to_atr": signal.get("spread_to_m1_atr"),
        "session": _session(generated),
        "higher_timeframe_alignment": "ALIGNED" if aligned else "MISALIGNED",
    }


def build_prospective_decisions(
    signals: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    now_utc: datetime,
) -> list[dict[str, Any]]:
    now = aware_utc(now_utc)
    activation = parse_utc(policy["activation_at_utc"])
    result: list[dict[str, Any]] = []
    last_generated: datetime | None = None
    for signal in signals:
        if signal.get("pair") != PAIR:
            continue
        if not sealed_valid(signal, "QR_FAST_BOT_SHOCK_FOLLOW_SIGNAL_V1"):
            raise ValueError("prospective shock signal seal mismatch")
        generated = parse_utc(signal.get("generated_at_utc"))
        if generated < activation:
            continue
        if generated > now:
            raise ValueError("future prospective signal")
        if last_generated is not None and generated < last_generated:
            raise ValueError("out-of-order prospective signal")
        last_generated = generated
        features = _shock_signal_features(signal)
        decision = route_state(
            features,
            policy=policy,
            observed_at_utc=generated,
            now_utc=now,
        )
        body = {
            "contract": PROSPECTIVE_DECISION_CONTRACT,
            "schema_version": 1,
            "decision_identity": canonical_sha([policy["contract_sha256"], signal["contract_sha256"]]),
            "policy_sha256": policy["contract_sha256"],
            "signal_id": signal["signal_id"],
            "signal_sha256": signal["contract_sha256"],
            "signal_generated_at_utc": generated.isoformat(),
            "features": features,
            **decision,
            "evidence_mode": "POST_ACTIVATION_PROSPECTIVE_ONLY",
            "counts_as_forward_evidence": True,
            "order_fields_authored": [],
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
        }
        result.append(seal(body))
    return result


def build_prospective_outcomes(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_signal = {str(row.get("signal_id")): row for row in decisions}
    result: list[dict[str, Any]] = []
    activation = parse_utc(policy["activation_at_utc"])
    last_resolved: datetime | None = None
    for outcome in outcomes:
        signal_id = str(outcome.get("signal_id") or "")
        decision = by_signal.get(signal_id)
        if decision is None:
            continue
        if not sealed_valid(outcome, "QR_FAST_BOT_SHOCK_FOLLOW_S5_OUTCOME_V1"):
            raise ValueError("prospective shock outcome seal mismatch")
        resolved = parse_utc(outcome.get("resolved_at_utc"))
        generated = parse_utc(outcome.get("signal_generated_at_utc"))
        if generated < activation or resolved <= generated or resolved < activation:
            raise ValueError("prospective outcome chronology mismatch")
        if last_resolved is not None and resolved < last_resolved:
            raise ValueError("out-of-order prospective outcome")
        last_resolved = resolved
        eligible = decision.get("choice") != "NO_TRADE"
        body = {
            "contract": PROSPECTIVE_OUTCOME_CONTRACT,
            "schema_version": 1,
            "prospective_identity": canonical_sha([decision["contract_sha256"], outcome["contract_sha256"]]),
            "policy_sha256": policy["contract_sha256"],
            "decision_sha256": decision["contract_sha256"],
            "signal_id": signal_id,
            "signal_generated_at_utc": generated.isoformat(),
            "resolved_at_utc": resolved.isoformat(),
            **decision["features"],
            "choice": decision["choice"],
            "eligible": eligible,
            "filled": bool(outcome.get("filled")) if eligible else False,
            "fill_at_utc": outcome.get("fill_at_utc") if eligible else None,
            "exit_reason": outcome.get("exit_reason") if eligible else "ROUTER_NO_TRADE",
            "exit_at_utc": outcome.get("exit_at_utc") if eligible else None,
            "realized_pips": float(outcome.get("realized_pips") or 0.0) if eligible else 0.0,
            "mfe_pips": float(outcome.get("mfe_pips") or 0.0) if eligible else 0.0,
            "mae_pips": float(outcome.get("mae_pips") or 0.0) if eligible else 0.0,
            "entry_slippage_pips": float(outcome.get("entry_slippage_pips") or 0.0) if eligible else 0.0,
            "truth_source": "OANDA_S5_BID_ASK",
            "truth_chunk_sha256": list(outcome.get("truth_chunk_sha256") or []),
            "evidence_mode": "POST_ACTIVATION_PROSPECTIVE_ONLY",
            "counts_as_forward_evidence": True,
            "profitability_claim_allowed": False,
            "execution_authority": "NONE",
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
        }
        result.append(seal(body))
    return result


def build_prospective_scorecard(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    generated_at_utc: datetime,
) -> dict[str, Any]:
    for row in decisions:
        if not sealed_valid(row, PROSPECTIVE_DECISION_CONTRACT) or row.get("counts_as_forward_evidence") is not True:
            raise ValueError("prospective decision contract mismatch")
    for row in outcomes:
        if not sealed_valid(row, PROSPECTIVE_OUTCOME_CONTRACT) or row.get("counts_as_forward_evidence") is not True:
            raise ValueError("prospective outcome contract mismatch")
    body = {
        "contract": PROSPECTIVE_SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": aware_utc(generated_at_utc).isoformat(),
        "policy_sha256": policy["contract_sha256"],
        "evidence_mode": "POST_ACTIVATION_PROSPECTIVE_ONLY",
        "decision_count": len(decisions),
        "resolved_count": len(outcomes),
        "overall": _metrics(outcomes),
        "by_regime_strategy_side_session": _groups(outcomes, ["regime", "strategy", "side", "session"]),
        "pre_activation_diagnostic_rows_included": 0,
        "counts_as_forward_evidence": True,
        "forward_evidence_passed": False,
        "profitability_claim_allowed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
    }
    return seal(body)


def observe_prospective(
    *,
    manifest_path: Path,
    config_path: Path,
    shock_signal_ledger_path: Path,
    shock_outcome_ledger_path: Path,
    decision_ledger_path: Path,
    prospective_outcome_ledger_path: Path,
    prospective_scorecard_path: Path,
    now_utc: datetime,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(manifest, config=config, config_sha256=config_sha256)
    policy = manifest["policy"]
    decision_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    prospective_outcome_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    decision_ledger_path.touch(exist_ok=True)
    prospective_outcome_ledger_path.touch(exist_ok=True)
    if market_is_closed(now_utc):
        decisions = load_jsonl(decision_ledger_path)
        outcomes = load_jsonl(prospective_outcome_ledger_path)
        scorecard = build_prospective_scorecard(decisions, outcomes, policy=policy, generated_at_utc=now_utc)
        write_json_atomic(prospective_scorecard_path, scorecard)
        return {
            "status": "MARKET_CLOSED_NO_OBSERVATION",
            "policy_sha256": policy["contract_sha256"],
            "prospective_decision_count": len(decisions),
            "prospective_sample_count": len(outcomes),
            "launchagent_restart_requested": False,
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
        }
    new_decisions = build_prospective_decisions(load_jsonl(shock_signal_ledger_path), policy=policy, now_utc=now_utc)
    append_sealed_rows(decision_ledger_path, new_decisions, contract=PROSPECTIVE_DECISION_CONTRACT, identity_key="decision_identity")
    all_decisions = load_jsonl(decision_ledger_path)
    new_outcomes = build_prospective_outcomes(all_decisions, load_jsonl(shock_outcome_ledger_path), policy=policy)
    append_sealed_rows(prospective_outcome_ledger_path, new_outcomes, contract=PROSPECTIVE_OUTCOME_CONTRACT, identity_key="prospective_identity")
    all_outcomes = load_jsonl(prospective_outcome_ledger_path)
    scorecard = build_prospective_scorecard(all_decisions, all_outcomes, policy=policy, generated_at_utc=now_utc)
    write_json_atomic(prospective_scorecard_path, scorecard)
    return {
        "status": "COLLECTING_POST_ACTIVATION_PROSPECTIVE_EVIDENCE",
        "policy_sha256": policy["contract_sha256"],
        "prospective_decision_count": len(all_decisions),
        "prospective_sample_count": len(all_outcomes),
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
    }


def retraining_status(
    *,
    previous_receipt: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not sealed_valid(previous_receipt, TRAINING_RECEIPT_CONTRACT):
        raise ValueError("previous training receipt seal mismatch")
    previous_cutoff = parse_utc(previous_receipt["training_cutoff_at_utc"])
    new_rows = [row for row in candidate_rows if parse_utc(row["resolved_at_utc"]) > previous_cutoff]
    days = {parse_utc(row["resolved_at_utc"]).date().isoformat() for row in new_rows}
    governance = config["retraining"]
    qualified = len(new_rows) >= int(governance["minimum_new_resolved_samples"]) and len(days) >= int(governance["minimum_new_resolved_days"])
    return {
        "status": governance["qualified_candidate_status"] if qualified else governance["no_change_status"],
        "new_resolved_samples": len(new_rows),
        "new_resolved_days": len(days),
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
    }


def append_sealed_rows(path: Path, rows: Sequence[Mapping[str, Any]], *, contract: str, identity_key: str) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not sealed_valid(row, contract):
                raise ValueError(f"invalid {contract} row at line {number}")
            identity = str(row.get(identity_key) or "")
            if not identity or identity in seen:
                raise ValueError(f"duplicate {contract} identity at line {number}")
            seen.add(identity)
        handle.seek(0, os.SEEK_END)
        appended = 0
        for row in rows:
            identity = str(row.get(identity_key) or "")
            if not identity or identity in seen:
                continue
            if not sealed_valid(row, contract):
                raise ValueError(f"invalid {contract} append row")
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(identity)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_current_manifest(pointer_path: Path) -> Path:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, Mapping) or not sealed_valid(pointer, CURRENT_POINTER_CONTRACT):
        raise ValueError("learned policy current pointer seal mismatch")
    if (
        pointer.get("automatic_adoption_allowed") is not False
        or pointer.get("promotion_allowed") is not False
        or pointer.get("live_permission") is not False
        or pointer.get("execution_authority") != "NONE"
    ):
        raise ValueError("learned policy current pointer authority mismatch")
    manifest_path = Path(str(pointer.get("manifest_path") or ""))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("learned policy manifest path is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("policy_sha256") != pointer.get("policy_sha256"):
        raise ValueError("learned policy pointer/content mismatch")
    return manifest_path


def freeze_training_artifacts(
    *,
    signal_ledger_path: Path,
    outcome_ledger_path: Path,
    corrective_ledger_path: Path,
    config_path: Path,
    output_root: Path,
    activation_at_utc: datetime,
    now_utc: datetime,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    rows, hashes = build_training_rows(
        signal_ledger_path=signal_ledger_path,
        outcome_ledger_path=outcome_ledger_path,
        corrective_ledger_path=corrective_ledger_path,
        cutoff_at_utc=parse_utc(config["training"]["cutoff_at_utc"]),
        now_utc=now_utc,
        maximum_resolution_lag_seconds=int(config["training"]["maximum_resolution_lag_seconds"]),
    )
    trained = train_policy(
        training_rows=rows,
        input_hashes=hashes,
        config=config,
        config_sha256=config_sha256,
        activation_at_utc=activation_at_utc,
    )
    diagnostics = build_diagnostic_rows(rows, policy=trained["policy"])
    leave_block_out = build_leave_block_out_diagnostic(rows, config=config, policy=trained["policy"])
    scorecard = build_diagnostic_scorecard(
        diagnostics,
        policy=trained["policy"],
        generated_at_utc=now_utc,
        leave_block_out=leave_block_out,
    )
    content_root = output_root / trained["policy"]["contract_sha256"]
    append_sealed_rows(content_root / "training_rows.jsonl", rows, contract=TRAINING_ROW_CONTRACT, identity_key="signal_id")
    append_sealed_rows(content_root / "pre_activation_diagnostic_ledger.jsonl", diagnostics, contract=DIAGNOSTIC_ROW_CONTRACT, identity_key="diagnostic_identity")
    write_json_atomic(content_root / "training_receipt.json", trained["training_receipt"])
    write_json_atomic(content_root / "policy.json", trained["policy"])
    write_json_atomic(content_root / "manifest.json", trained["manifest"])
    write_json_atomic(content_root / "pre_activation_diagnostic_scorecard.json", scorecard)
    current = {
        "contract": CURRENT_POINTER_CONTRACT,
        "policy_sha256": trained["policy"]["contract_sha256"],
        "manifest_path": str(content_root / "manifest.json"),
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
    }
    write_json_atomic(output_root / "current.json", seal(current))
    return {
        "contract": "QR_EURUSD_OUTCOME_LEARNING_TRAIN_RUN_V1",
        "status": "TEST_REQUIRED",
        "training_row_count": len(rows),
        "training_cutoff_at_utc": config["training"]["cutoff_at_utc"],
        "input_ledger_sha256": hashes["input_ledger_sha256"],
        "config_sha256": config_sha256,
        "policy_sha256": trained["policy"]["contract_sha256"],
        "training_receipt_sha256": trained["training_receipt"]["contract_sha256"],
        "activation_at_utc": trained["policy"]["activation_at_utc"],
        "quarantined_cells": trained["policy"]["quarantined_cells"],
        "diagnostic_scorecard_path": str(content_root / "pre_activation_diagnostic_scorecard.json"),
        "manifest_path": str(content_root / "manifest.json"),
        "prospective_sample_count": 0,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
    }


def closed_market_observation(*, manifest_path: Path, config_path: Path, now_utc: datetime) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(manifest, config=config, config_sha256=config_sha256)
    if not market_is_closed(now_utc):
        return {
            "status": "OBSERVATION_REQUIRED",
            "policy_sha256": manifest["policy_sha256"],
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "external_order_attempts": 0,
            "external_orders": 0,
            "live_permission": False,
        }
    return {
        "status": "MARKET_CLOSED_NO_OBSERVATION",
        "policy_sha256": manifest["policy_sha256"],
        "prospective_sample_count": 0,
        "launchagent_restart_requested": False,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
    }
