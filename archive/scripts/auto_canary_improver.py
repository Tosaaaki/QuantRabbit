#!/usr/bin/env python3
"""Generate lightweight live canary overrides from analysis artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.strategy_tags import resolve_strategy_tag


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _setup_match_dimension(
    setup_fingerprint: str, flow_regime: str, microstructure_bucket: str
) -> str:
    if setup_fingerprint:
        return "setup_fingerprint"
    if flow_regime and microstructure_bucket:
        return "flow_micro"
    if flow_regime:
        return "flow_regime"
    if microstructure_bucket:
        return "microstructure_bucket"
    return ""


def _cluster_setup_context(cluster: dict[str, Any]) -> dict[str, str]:
    features = cluster.get("features")
    feature_map = features if isinstance(features, dict) else {}
    setup = cluster.get("setup")
    setup_map = setup if isinstance(setup, dict) else {}
    setup_context = cluster.get("setup_context")
    context_map = setup_context if isinstance(setup_context, dict) else {}
    setup_fingerprint = (
        _safe_text(cluster.get("setup_fingerprint"))
        or _safe_text(context_map.get("setup_fingerprint"))
        or _safe_text(setup_map.get("setup_fingerprint"))
        or _safe_text(feature_map.get("setup_fingerprint"))
    )
    flow_regime = (
        _safe_text(cluster.get("flow_regime"))
        or _safe_text(context_map.get("flow_regime"))
        or _safe_text(setup_map.get("flow_regime"))
        or _safe_text(feature_map.get("flow_regime"))
    )
    microstructure_bucket = (
        _safe_text(cluster.get("microstructure_bucket"))
        or _safe_text(context_map.get("microstructure_bucket"))
        or _safe_text(setup_map.get("microstructure_bucket"))
        or _safe_text(feature_map.get("microstructure_bucket"))
    )
    match_dimension = _setup_match_dimension(
        setup_fingerprint,
        flow_regime,
        microstructure_bucket,
    )
    if not match_dimension:
        return {}
    return {
        "match_dimension": match_dimension,
        "setup_fingerprint": setup_fingerprint,
        "flow_regime": flow_regime,
        "microstructure_bucket": microstructure_bucket,
    }


def _apply_validation_adjustments(
    *,
    strategy_key: str,
    confidence: float,
    units_multiplier: float,
    probability_offset: float,
    replay_status: str,
    counterfactual_strategy: str,
    top_action: str,
    top_uplift: float,
) -> tuple[float, float, float, list[str]]:
    reasons: list[str] = []
    if (
        strategy_key
        and counterfactual_strategy
        and strategy_key == counterfactual_strategy
    ):
        if top_action in {"block", "reduce"}:
            confidence = min(1.0, confidence + 0.15)
            units_multiplier -= 0.05
            probability_offset -= 0.005
            reasons.append(f"counterfactual_{top_action}")
        elif top_action == "boost":
            confidence = min(1.0, confidence + 0.05)
            units_multiplier = max(units_multiplier, 1.02)
            probability_offset = max(probability_offset, 0.01)
            reasons.append("counterfactual_boost")
        if top_uplift > 0.0:
            confidence = min(1.0, confidence + min(0.10, top_uplift / 100.0))

    if replay_status in {"error", "timeout", "report_missing"}:
        units_multiplier = min(1.0, units_multiplier + 0.05)
        probability_offset = min(0.0, probability_offset + 0.01)
        confidence *= 0.75
        reasons.append("replay_validation_degraded")

    confidence = max(0.0, min(1.0, confidence))
    units_multiplier = max(0.80, min(1.08, units_multiplier))
    probability_offset = max(-0.04, min(0.02, probability_offset))
    return confidence, units_multiplier, probability_offset, reasons


def _build_setup_overrides(
    strategy_key: str,
    summary: dict[str, Any],
    *,
    replay_status: str,
    counterfactual_strategy: str,
    top_action: str,
    top_uplift: float,
    min_confidence: float,
) -> list[dict[str, Any]]:
    clusters = summary.get("clusters")
    if not isinstance(clusters, list):
        return []
    overrides: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        setup_context = _cluster_setup_context(cluster)
        if not setup_context:
            continue
        suggestion = (
            cluster.get("suggestion")
            if isinstance(cluster.get("suggestion"), dict)
            else {}
        )
        severity = _safe_float(cluster.get("severity"), 0.0)
        confidence = severity * 0.75
        units_multiplier = _safe_float(suggestion.get("units_multiplier"), 1.0)
        probability_offset = _safe_float(suggestion.get("probability_offset"), 0.0)
        confidence, units_multiplier, probability_offset, reasons = (
            _apply_validation_adjustments(
                strategy_key=strategy_key,
                confidence=confidence,
                units_multiplier=units_multiplier,
                probability_offset=probability_offset,
                replay_status=replay_status,
                counterfactual_strategy=counterfactual_strategy,
                top_action=top_action,
                top_uplift=top_uplift,
            )
        )
        if confidence < min_confidence:
            continue
        if not reasons:
            reasons = ["loser_cluster_canary"]
        overrides.append(
            {
                **setup_context,
                "enabled": True,
                "mode": "canary",
                "confidence": round(confidence, 6),
                "units_multiplier": round(units_multiplier, 4),
                "probability_offset": round(probability_offset, 4),
                "reason": reasons[0],
                "reasons": reasons,
                "samples": int(cluster.get("samples") or 0),
                "severity": round(severity, 6),
                "source": {
                    "cluster_key": cluster.get("cluster_key"),
                    "counterfactual_strategy": counterfactual_strategy or None,
                    "counterfactual_action": top_action or None,
                    "replay_status": replay_status,
                },
            }
        )
    specificity_rank = {
        "setup_fingerprint": 4,
        "flow_micro": 3,
        "flow_regime": 2,
        "microstructure_bucket": 1,
    }
    overrides.sort(
        key=lambda item: (
            specificity_rank.get(str(item.get("match_dimension") or ""), 0),
            int(item.get("samples") or 0),
        ),
        reverse=True,
    )
    return overrides[:8]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def build_auto_canary(
    loser_cluster: dict[str, Any],
    replay_quality_gate: dict[str, Any],
    trade_counterfactual: dict[str, Any],
    *,
    min_confidence: float,
) -> dict[str, Any]:
    replay_status = (
        str(
            replay_quality_gate.get("gate_status")
            or replay_quality_gate.get("status")
            or ""
        ).strip()
        or "unknown"
    )
    counterfactual_like = (
        str(trade_counterfactual.get("strategy_like") or "").strip().rstrip("%")
    )
    counterfactual_strategy = (
        resolve_strategy_tag(counterfactual_like) or counterfactual_like
    )
    recommendations = trade_counterfactual.get("recommendations")
    top_recommendation = (
        recommendations[0]
        if isinstance(recommendations, list) and recommendations
        else {}
    )
    top_action = (
        str(top_recommendation.get("action") or "").strip().lower()
        if isinstance(top_recommendation, dict)
        else ""
    )
    top_uplift = _safe_float(
        (
            (top_recommendation or {}).get("noise_lcb_uplift_pips")
            if isinstance(top_recommendation, dict)
            else 0.0
        ),
        0.0,
    )

    strategies = loser_cluster.get("strategies")
    if not isinstance(strategies, dict):
        strategies = {}

    out_strategies: dict[str, Any] = {}
    for raw_key, summary in sorted(strategies.items()):
        if not isinstance(summary, dict):
            continue
        strategy_key = (
            resolve_strategy_tag(str(raw_key or "").strip())
            or str(raw_key or "").strip()
        )
        suggestion = (
            summary.get("suggestion")
            if isinstance(summary.get("suggestion"), dict)
            else {}
        )
        severity = _safe_float(summary.get("worst_severity"), 0.0)
        confidence = severity * 0.75
        units_multiplier = _safe_float(suggestion.get("units_multiplier"), 1.0)
        probability_offset = _safe_float(suggestion.get("probability_offset"), 0.0)
        confidence, units_multiplier, probability_offset, reasons = (
            _apply_validation_adjustments(
                strategy_key=strategy_key,
                confidence=confidence,
                units_multiplier=units_multiplier,
                probability_offset=probability_offset,
                replay_status=replay_status,
                counterfactual_strategy=counterfactual_strategy,
                top_action=top_action,
                top_uplift=top_uplift,
            )
        )

        if confidence < min_confidence:
            continue

        record = {
            "enabled": True,
            "mode": "canary",
            "confidence": round(confidence, 6),
            "units_multiplier": round(units_multiplier, 4),
            "probability_offset": round(probability_offset, 4),
            "reason": reasons[0] if reasons else "loser_cluster_canary",
            "reasons": reasons or ["loser_cluster_canary"],
            "source": {
                "worst_severity": round(severity, 6),
                "replay_status": replay_status,
                "counterfactual_strategy": counterfactual_strategy or None,
                "counterfactual_action": top_action or None,
            },
        }
        setup_overrides = _build_setup_overrides(
            strategy_key,
            summary,
            replay_status=replay_status,
            counterfactual_strategy=counterfactual_strategy,
            top_action=top_action,
            top_uplift=top_uplift,
            min_confidence=min_confidence,
        )
        if setup_overrides:
            record["setup_overrides"] = setup_overrides
        out_strategies[strategy_key] = record

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "replay_status": replay_status,
            "counterfactual_strategy": counterfactual_strategy or None,
            "counterfactual_action": top_action or None,
            "counterfactual_uplift_lcb_pips": round(top_uplift, 6),
        },
        "strategies": out_strategies,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build auto-canary override artifact")
    ap.add_argument("--loser-cluster-path", default="logs/loser_cluster_latest.json")
    ap.add_argument(
        "--replay-quality-gate-path", default="logs/replay_quality_gate_latest.json"
    )
    ap.add_argument(
        "--trade-counterfactual-path", default="logs/trade_counterfactual_latest.json"
    )
    ap.add_argument("--output", default="config/auto_canary_overrides.json")
    ap.add_argument("--latest-output", default="logs/auto_canary_latest.json")
    ap.add_argument("--history", default="logs/auto_canary_history.jsonl")
    ap.add_argument("--min-confidence", type=float, default=0.55)
    args = ap.parse_args()

    payload = build_auto_canary(
        _read_json(Path(args.loser_cluster_path).resolve()),
        _read_json(Path(args.replay_quality_gate_path).resolve()),
        _read_json(Path(args.trade_counterfactual_path).resolve()),
        min_confidence=max(0.0, min(1.0, float(args.min_confidence))),
    )
    _write_json_atomic(Path(args.output).resolve(), payload)
    _write_json_atomic(Path(args.latest_output).resolve(), payload)
    _append_jsonl(Path(args.history).resolve(), payload)
    print(
        f"[auto-canary-improver] wrote {Path(args.output).resolve()} "
        f"strategies={len(payload['strategies'])}"
    )


if __name__ == "__main__":
    main()
